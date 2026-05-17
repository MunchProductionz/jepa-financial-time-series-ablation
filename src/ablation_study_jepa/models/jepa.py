"""JEPA auxiliary modules for Transformer hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ablation_study_jepa.config.schemas import JEPAConfig, JEPAMode, NegativeStrategy, SIGRegApplyTo


class ResidualMLPPredictor(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 2
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class JEPAHead(nn.Module):
    """Project one Transformer block output and predict a future latent."""

    def __init__(
        self,
        hidden_dim: int,
        projection_dim: int,
        predictor_type: str = "mlp",
        horizons: list[int] | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.horizons = list(horizons or [1])
        self.horizon_to_index = {int(h): idx for idx, h in enumerate(self.horizons)}
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.horizon_embedding = (
            nn.Embedding(len(self.horizons), projection_dim) if len(self.horizons) > 1 else None
        )
        if predictor_type == "linear":
            self.predictor = nn.Linear(projection_dim, projection_dim)
        elif predictor_type == "residual_mlp":
            self.predictor = ResidualMLPPredictor(projection_dim, dropout=dropout)
        elif predictor_type == "mlp":
            self.predictor = nn.Sequential(
                nn.LayerNorm(projection_dim),
                nn.Linear(projection_dim, projection_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(projection_dim * 2, projection_dim),
            )
        else:
            raise ValueError(f"Unknown predictor_type: {predictor_type}")

    def encode_context(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projector(hidden_state)

    def predict_from_latent(self, z_context: torch.Tensor, horizon: int) -> torch.Tensor:
        if self.horizon_embedding is not None:
            index = self.horizon_to_index[int(horizon)]
            horizon_index = torch.full(
                (z_context.size(0),),
                index,
                dtype=torch.long,
                device=z_context.device,
            )
            z_context = z_context + self.horizon_embedding(horizon_index)
        return self.predictor(z_context)

    def predict_from_context(self, hidden_state: torch.Tensor, horizon: int) -> torch.Tensor:
        return self.predict_from_latent(self.encode_context(hidden_state), horizon)

    def encode_target(self, hidden_state: torch.Tensor, detach: bool = True) -> torch.Tensor:
        z_target = self.projector(hidden_state)
        return z_target.detach() if detach else z_target


@dataclass
class ContrastiveDiagnostics:
    mean_positive_similarity: torch.Tensor
    mean_negative_similarity: torch.Tensor
    contrastive_accuracy: torch.Tensor
    embedding_norm: torch.Tensor


class ContrastiveLoss(nn.Module):
    """InfoNCE loss with diagonal positives and masked in-batch negatives."""

    def __init__(self, temperature: float = 0.1, similarity: str = "cosine") -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if similarity not in {"cosine", "dot"}:
            raise ValueError("similarity must be 'cosine' or 'dot'")
        self.temperature = temperature
        self.similarity = similarity

    def forward(
        self,
        query: torch.Tensor,
        positive_targets: torch.Tensor,
        negative_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ContrastiveDiagnostics]:
        if query.shape != positive_targets.shape:
            raise ValueError("query and positive_targets must have the same shape")
        batch_size = query.size(0)
        if self.similarity == "cosine":
            query_for_sim = F.normalize(query, dim=-1)
            target_for_sim = F.normalize(positive_targets, dim=-1)
        else:
            query_for_sim = query
            target_for_sim = positive_targets
        logits = query_for_sim @ target_for_sim.T
        raw_similarities = logits.detach()
        logits = logits / self.temperature

        valid = torch.eye(batch_size, dtype=torch.bool, device=query.device)
        if negative_mask is None:
            negative_mask = ~valid
        else:
            negative_mask = negative_mask.to(device=query.device, dtype=torch.bool)
        valid = valid | negative_mask
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        labels = torch.arange(batch_size, device=query.device)
        loss = F.cross_entropy(logits, labels)

        pos_sim = raw_similarities.diag().mean()
        if negative_mask.any():
            neg_sim = raw_similarities[negative_mask].mean()
        else:
            neg_sim = torch.zeros((), dtype=query.dtype, device=query.device)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()
        embedding_norm = positive_targets.norm(dim=-1).mean()
        return loss, ContrastiveDiagnostics(pos_sim, neg_sim, accuracy, embedding_norm)


class SIGRegLoss(nn.Module):
    """Sliced characteristic-function regularizer toward an isotropic Gaussian."""

    def __init__(
        self,
        embedding_dim: int,
        num_slices: int = 256,
        num_t: int = 17,
        t_max: float = 5.0,
        resample_directions_each_step: bool = True,
        min_batch_size: int = 8,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if num_slices <= 0:
            raise ValueError("num_slices must be positive")
        if num_t <= 0:
            raise ValueError("num_t must be positive")
        if t_max <= 0:
            raise ValueError("t_max must be positive")
        if min_batch_size <= 0:
            raise ValueError("min_batch_size must be positive")
        self.embedding_dim = embedding_dim
        self.num_slices = num_slices
        self.num_t = num_t
        self.t_max = t_max
        self.resample_directions_each_step = resample_directions_each_step
        self.min_batch_size = min_batch_size
        self.eps = eps

        if resample_directions_each_step:
            self.register_buffer("_fixed_directions", torch.empty(0), persistent=False)
        else:
            fixed = torch.randn(embedding_dim, num_slices)
            self.register_buffer("_fixed_directions", self._normalize_directions(fixed), persistent=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("SIGReg expects embeddings with shape [batch, dim]")
        if z.size(1) != self.embedding_dim:
            raise ValueError(f"SIGReg expected embedding dim {self.embedding_dim}, got {z.size(1)}")
        if z.size(0) < self.min_batch_size:
            return z.sum() * 0.0

        z_float = z.float()
        directions = self._directions(device=z.device, dtype=torch.float32)
        projected = z_float @ directions
        t_grid = torch.linspace(
            -self.t_max,
            self.t_max,
            self.num_t,
            device=z.device,
            dtype=torch.float32,
        )
        phase = projected.unsqueeze(-1) * t_grid.view(1, 1, -1)
        real_phi = torch.cos(phase).mean(dim=0)
        imag_phi = torch.sin(phase).mean(dim=0)
        normal_phi = torch.exp(-0.5 * t_grid.square()).view(1, -1)
        loss = (real_phi - normal_phi).square() + imag_phi.square()
        return loss.mean().to(dtype=z.dtype)

    def _directions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.resample_directions_each_step:
            raw = torch.randn(self.embedding_dim, self.num_slices, device=device, dtype=dtype)
            return self._normalize_directions(raw)
        return self._fixed_directions.to(device=device, dtype=dtype)

    def _normalize_directions(self, directions: torch.Tensor) -> torch.Tensor:
        return directions / directions.norm(dim=0, keepdim=True).clamp_min(self.eps)


class NegativeSampler:
    """Market-aware in-batch negative mask builder."""

    def __init__(
        self,
        strategy: NegativeStrategy | str = NegativeStrategy.MIXED,
        num_negatives: int | None = None,
        exclusion_window: int = 5,
        allow_same_date_cross_asset_negatives: bool = False,
        allow_same_sector_negatives: bool = True,
        sector_filtering: bool = False,
        same_asset_negative_min_gap: int = 20,
    ) -> None:
        self.strategy = NegativeStrategy(strategy)
        self.num_negatives = num_negatives
        self.exclusion_window = exclusion_window
        self.allow_same_date_cross_asset_negatives = allow_same_date_cross_asset_negatives
        self.allow_same_sector_negatives = allow_same_sector_negatives
        self.sector_filtering = sector_filtering
        self.same_asset_negative_min_gap = same_asset_negative_min_gap

    def build_mask(
        self,
        metadata: dict[str, Any],
        horizon: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        asset_ids = _as_list(metadata.get("asset_id", [""] * batch_size))
        sectors = _as_list(metadata.get("sector", [""] * batch_size))
        target_ordinals = _as_int_list(
            metadata.get(
                f"target_date_ordinal_horizon_{horizon}",
                metadata.get("target_date_ordinal", metadata.get("anchor_date_ordinal", [0] * batch_size)),
            )
        )
        target_positions = _as_int_list(
            metadata.get(
                f"target_position_horizon_{horizon}",
                metadata.get("target_position", metadata.get("anchor_position", [0] * batch_size)),
            )
        )

        mask = torch.zeros(batch_size, batch_size, dtype=torch.bool)
        for i in range(batch_size):
            for j in range(batch_size):
                if i == j:
                    continue
                if self._is_valid_pair(
                    i,
                    j,
                    asset_ids=asset_ids,
                    sectors=sectors,
                    target_ordinals=target_ordinals,
                    target_positions=target_positions,
                    horizon=horizon,
                ):
                    mask[i, j] = True

        if self.num_negatives is not None and self.num_negatives > 0:
            mask = self._subsample(mask, self.num_negatives)

        if batch_size > 1:
            no_negative_rows = ~mask.any(dim=1)
            fallback = ~torch.eye(batch_size, dtype=torch.bool)
            mask[no_negative_rows] = fallback[no_negative_rows]

        return mask.to(device=device)

    def _is_valid_pair(
        self,
        i: int,
        j: int,
        asset_ids: list[str],
        sectors: list[str],
        target_ordinals: list[int],
        target_positions: list[int],
        horizon: int,
    ) -> bool:
        same_asset = asset_ids[i] == asset_ids[j]
        same_date = target_ordinals[i] == target_ordinals[j]
        same_sector = sectors[i] and sectors[i] == sectors[j]
        date_gap = abs(target_ordinals[i] - target_ordinals[j])
        position_gap = abs(target_positions[i] - target_positions[j])
        exclusion = max(self.exclusion_window, int(horizon))

        if same_date and not self.allow_same_date_cross_asset_negatives and not same_asset:
            return False
        if self.sector_filtering and same_sector and not self.allow_same_sector_negatives:
            return False

        if self.strategy == NegativeStrategy.IN_BATCH_ALL:
            return True
        if self.strategy == NegativeStrategy.IN_BATCH_FILTERED:
            return date_gap > self.exclusion_window
        if self.strategy == NegativeStrategy.SAME_ASSET_FAR_TIME:
            return same_asset and position_gap >= max(self.same_asset_negative_min_gap, exclusion)
        if self.strategy == NegativeStrategy.DIFFERENT_ASSET_DIFFERENT_TIME:
            return (not same_asset) and date_gap > exclusion
        if self.strategy == NegativeStrategy.MIXED:
            same_asset_far = same_asset and position_gap >= max(self.same_asset_negative_min_gap, exclusion)
            different_asset_time = (not same_asset) and date_gap > exclusion
            return same_asset_far or different_asset_time
        raise ValueError(f"Unknown negative strategy: {self.strategy}")

    @staticmethod
    def _subsample(mask: torch.Tensor, num_negatives: int) -> torch.Tensor:
        sampled = torch.zeros_like(mask)
        for row in range(mask.size(0)):
            candidates = torch.nonzero(mask[row], as_tuple=False).flatten()
            if candidates.numel() <= num_negatives:
                sampled[row, candidates] = True
            else:
                selected = candidates[torch.randperm(candidates.numel())[:num_negatives]]
                sampled[row, selected] = True
        return sampled


class MultiLayerJEPAModule(nn.Module):
    """Own JEPA heads, layer selection, weighted loss aggregation, and diagnostics."""

    def __init__(
        self,
        hidden_dim: int,
        num_transformer_blocks: int,
        config: JEPAConfig,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = config
        self.mode = JEPAMode(config.mode)
        self.selected_layers = config.resolve_selected_layers(num_transformer_blocks)
        self.layer_weights = config.normalized_layer_weights(self.selected_layers)
        self.horizon_weights = config.normalized_horizon_weights()
        self.horizons = list(config.horizons)
        self.heads = nn.ModuleDict(
            {
                str(layer): JEPAHead(
                    hidden_dim=hidden_dim,
                    projection_dim=config.projection_dim,
                    predictor_type=config.predictor_type.value,
                    horizons=self.horizons,
                    dropout=dropout,
                )
                for layer in self.selected_layers
            }
        )
        self.contrastive_loss = ContrastiveLoss(temperature=config.contrastive.temperature)
        self.negative_sampler = NegativeSampler(
            strategy=config.contrastive.negative_strategy,
            num_negatives=config.contrastive.num_negatives,
            exclusion_window=config.contrastive.exclusion_window,
            allow_same_date_cross_asset_negatives=config.contrastive.allow_same_date_cross_asset_negatives,
            allow_same_sector_negatives=config.contrastive.allow_same_sector_negatives,
            sector_filtering=config.contrastive.sector_filtering,
            same_asset_negative_min_gap=config.contrastive.same_asset_negative_min_gap,
        )
        self.sigreg_loss = SIGRegLoss(
            embedding_dim=config.projection_dim,
            num_slices=config.lejepa.sigreg.num_slices,
            num_t=config.lejepa.sigreg.num_t,
            t_max=config.lejepa.sigreg.t_max,
            resample_directions_each_step=config.lejepa.sigreg.resample_directions_each_step,
            min_batch_size=config.lejepa.sigreg.min_batch_size,
        )

    def forward(
        self,
        context_hidden_states: list[torch.Tensor],
        target_hidden_states_by_horizon: dict[int, list[torch.Tensor]],
        metadata: dict[str, Any],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if not self.selected_layers:
            return self._zero_output(context_hidden_states[-1])
        if self.mode == JEPAMode.LEJEPA:
            return self._forward_lejepa(
                context_hidden_states=context_hidden_states,
                target_hidden_states_by_horizon=target_hidden_states_by_horizon,
                metadata=metadata,
            )
        return self._forward_contrastive(
            context_hidden_states=context_hidden_states,
            target_hidden_states_by_horizon=target_hidden_states_by_horizon,
            metadata=metadata,
        )

    def _forward_contrastive(
        self,
        context_hidden_states: list[torch.Tensor],
        target_hidden_states_by_horizon: dict[int, list[torch.Tensor]],
        metadata: dict[str, Any],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        total = torch.zeros((), device=context_hidden_states[-1].device)
        logs: dict[str, torch.Tensor] = {}
        for layer_weight, layer in zip(self.layer_weights, self.selected_layers, strict=True):
            head = self.heads[str(layer)]
            layer_total = torch.zeros((), device=context_hidden_states[layer].device)
            context_state = context_hidden_states[layer][:, -1, :]
            for horizon_weight, horizon in zip(self.horizon_weights, self.horizons, strict=True):
                if horizon not in target_hidden_states_by_horizon:
                    raise ValueError(f"Missing target hidden states for JEPA horizon {horizon}")
                target_state = target_hidden_states_by_horizon[horizon][layer][:, -1, :]
                query = head.predict_from_context(context_state, horizon)
                positive = head.encode_target(target_state, detach=True)
                negative_mask = self.negative_sampler.build_mask(
                    metadata=metadata,
                    horizon=horizon,
                    batch_size=query.size(0),
                    device=query.device,
                )
                loss, diagnostics = self.contrastive_loss(query, positive, negative_mask)
                layer_total = layer_total + float(horizon_weight) * loss
                prefix = f"jepa_loss_layer_{layer}_horizon_{horizon}"
                logs[prefix] = loss.detach()
                logs[f"{prefix}_mean_positive_similarity"] = diagnostics.mean_positive_similarity.detach()
                logs[f"{prefix}_mean_negative_similarity"] = diagnostics.mean_negative_similarity.detach()
                logs[f"{prefix}_contrastive_accuracy"] = diagnostics.contrastive_accuracy.detach()
                logs[f"{prefix}_embedding_norm"] = diagnostics.embedding_norm.detach()
            total = total + float(layer_weight) * layer_total
            logs[f"jepa_loss_layer_{layer}"] = layer_total.detach()
            logs[f"jepa_layer_{layer}_loss"] = layer_total.detach()
        logs["total_jepa_loss_unweighted"] = total.detach()
        return {"loss": total, "logs": logs}

    def _forward_lejepa(
        self,
        context_hidden_states: list[torch.Tensor],
        target_hidden_states_by_horizon: dict[int, list[torch.Tensor]],
        metadata: dict[str, Any],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        total = torch.zeros((), device=context_hidden_states[-1].device)
        total_prediction = torch.zeros_like(total)
        total_sigreg = torch.zeros_like(total)
        logs: dict[str, torch.Tensor] = {}
        context_embeddings_for_diagnostics: list[torch.Tensor] = []
        sigreg_apply_to = SIGRegApplyTo(self.config.lejepa.sigreg.apply_to)
        sigreg_uses_targets = sigreg_apply_to in {
            SIGRegApplyTo.TARGETS_ONLY,
            SIGRegApplyTo.CONTEXT_AND_TARGETS,
        }

        for layer_weight, layer in zip(self.layer_weights, self.selected_layers, strict=True):
            head = self.heads[str(layer)]
            context_state = context_hidden_states[layer][:, -1, :]
            z_context_all = head.encode_context(context_state)
            context_embeddings_for_diagnostics.append(z_context_all.detach())
            layer_prediction = torch.zeros((), device=context_state.device, dtype=context_state.dtype)
            horizon_weight_sum = 0.0
            context_valid_any = torch.zeros(
                context_state.size(0),
                dtype=torch.bool,
                device=context_state.device,
            )
            target_sigreg_embeddings: list[torch.Tensor] = []

            for horizon_weight, horizon in zip(self.horizon_weights, self.horizons, strict=True):
                if horizon not in target_hidden_states_by_horizon:
                    continue
                target_state_all = target_hidden_states_by_horizon[horizon][layer][:, -1, :]
                valid_mask = self._valid_horizon_mask(
                    metadata=metadata,
                    horizon=horizon,
                    batch_size=context_state.size(0),
                    device=context_state.device,
                )
                valid_count = int(valid_mask.sum().item())
                logs[f"jepa_layer_{layer}_horizon_{horizon}_valid_count"] = torch.as_tensor(
                    valid_count,
                    device=context_state.device,
                    dtype=context_state.dtype,
                )
                if valid_count == 0:
                    continue

                context_valid_any = context_valid_any | valid_mask
                z_context = z_context_all[valid_mask]
                z_pred = head.predict_from_latent(z_context, horizon)
                target_state = target_state_all[valid_mask]
                z_target_prediction = head.encode_target(
                    target_state,
                    detach=self.config.lejepa.detach_target,
                )
                prediction_loss = F.mse_loss(z_pred, z_target_prediction)
                if sigreg_uses_targets:
                    target_sigreg_embeddings.append(
                        head.encode_target(
                            target_state,
                            detach=False,
                        )
                    )
                layer_prediction = layer_prediction + float(horizon_weight) * prediction_loss
                horizon_weight_sum += float(horizon_weight)
                logs[f"jepa_layer_{layer}_horizon_{horizon}_prediction_loss"] = (
                    prediction_loss.detach()
                )

            if horizon_weight_sum > 0.0:
                layer_prediction = layer_prediction / horizon_weight_sum

            layer_sigreg = self._compute_layer_sigreg(
                z_context_all=z_context_all,
                context_valid_any=context_valid_any,
                target_embeddings=target_sigreg_embeddings,
                apply_to=sigreg_apply_to,
            )
            if self.config.lejepa.sigreg.enabled:
                lambda_sigreg = self.config.lejepa.loss_mix.lambda_sigreg
                layer_loss = (1.0 - lambda_sigreg) * layer_prediction + lambda_sigreg * layer_sigreg
            else:
                layer_loss = layer_prediction

            total = total + float(layer_weight) * layer_loss
            total_prediction = total_prediction + float(layer_weight) * layer_prediction
            total_sigreg = total_sigreg + float(layer_weight) * layer_sigreg
            logs[f"jepa_layer_{layer}_loss"] = layer_loss.detach()
            logs[f"jepa_layer_{layer}_prediction_loss"] = layer_prediction.detach()
            logs[f"jepa_layer_{layer}_sigreg_loss"] = layer_sigreg.detach()

        logs["jepa_loss"] = total.detach()
        logs["jepa_prediction_loss"] = total_prediction.detach()
        logs["jepa_sigreg_loss"] = total_sigreg.detach()
        logs["total_jepa_loss_unweighted"] = total.detach()
        logs.update(self._context_embedding_diagnostics(context_embeddings_for_diagnostics))
        return {"loss": total, "logs": logs}

    def _compute_layer_sigreg(
        self,
        z_context_all: torch.Tensor,
        context_valid_any: torch.Tensor,
        target_embeddings: list[torch.Tensor],
        apply_to: SIGRegApplyTo | None = None,
    ) -> torch.Tensor:
        zero = torch.zeros((), device=z_context_all.device, dtype=z_context_all.dtype)
        if not self.config.lejepa.sigreg.enabled:
            return zero

        apply_to = apply_to or SIGRegApplyTo(self.config.lejepa.sigreg.apply_to)
        embeddings: list[torch.Tensor] = []
        if apply_to in {SIGRegApplyTo.CONTEXT_ONLY, SIGRegApplyTo.CONTEXT_AND_TARGETS}:
            if bool(context_valid_any.any().item()):
                embeddings.append(z_context_all[context_valid_any])
        if apply_to in {SIGRegApplyTo.TARGETS_ONLY, SIGRegApplyTo.CONTEXT_AND_TARGETS}:
            embeddings.extend(target_embeddings)
        if not embeddings:
            return zero

        losses = [self.sigreg_loss(embedding) for embedding in embeddings if embedding.numel() > 0]
        if not losses:
            return zero
        return torch.stack(losses).mean()

    @staticmethod
    def _valid_horizon_mask(
        metadata: dict[str, Any],
        horizon: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = metadata.get(
            f"jepa_valid_horizon_{horizon}",
            metadata.get(f"valid_horizon_{horizon}", None),
        )
        if value is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if isinstance(value, torch.Tensor):
            mask = value.to(device=device, dtype=torch.bool).flatten()
        elif isinstance(value, (list, tuple)):
            mask = torch.as_tensor(value, dtype=torch.bool, device=device).flatten()
        else:
            mask = torch.full((batch_size,), bool(value), dtype=torch.bool, device=device)
        if mask.numel() == 1 and batch_size != 1:
            mask = mask.expand(batch_size)
        if mask.numel() != batch_size:
            raise ValueError(
                f"valid mask for JEPA horizon {horizon} has length {mask.numel()}, "
                f"expected {batch_size}"
            )
        return mask

    @staticmethod
    def _context_embedding_diagnostics(
        context_embeddings: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not context_embeddings:
            return {}
        z_context = torch.cat(context_embeddings, dim=0)
        std = z_context.float().std(dim=0, unbiased=False)
        return {
            "jepa_z_context_mean_abs": z_context.float().abs().mean().detach(),
            "jepa_z_context_std_mean": std.mean().detach(),
            "jepa_z_context_std_min": std.min().detach(),
            "jepa_z_context_std_max": std.max().detach(),
        }

    @staticmethod
    def _zero_output(reference: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
        return {"loss": zero, "logs": {"total_jepa_loss_unweighted": zero}}


class LossAggregator(nn.Module):
    """Combine supervised and normalized weighted JEPA losses."""

    def __init__(self, lambda_jepa: float = 0.05) -> None:
        super().__init__()
        self.lambda_jepa = float(lambda_jepa)

    def forward(
        self,
        supervised_loss: torch.Tensor,
        jepa_loss: torch.Tensor | None = None,
        jepa_scale: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if jepa_loss is None:
            jepa_loss = torch.zeros((), dtype=supervised_loss.dtype, device=supervised_loss.device)
        effective_lambda = self.lambda_jepa * float(jepa_scale)
        weighted_jepa_loss = effective_lambda * jepa_loss
        total = supervised_loss + weighted_jepa_loss
        return {
            "supervised_loss": supervised_loss,
            "total_jepa_loss": jepa_loss,
            "weighted_jepa_loss": weighted_jepa_loss,
            "effective_lambda_jepa": torch.as_tensor(
                effective_lambda,
                dtype=supervised_loss.dtype,
                device=supervised_loss.device,
            ),
            "total_loss": total,
        }


def _as_list(value: Any) -> list[str]:
    if isinstance(value, torch.Tensor):
        return [str(v.item()) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _as_int_list(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(v.item()) for v in value.flatten()]
    if isinstance(value, tuple):
        return [int(v) for v in value]
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(value)]
