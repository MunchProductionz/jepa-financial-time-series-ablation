"""Reusable LightningModule wrapping return forecasters and optional JEPA heads."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any
import warnings

import torch
from torch import nn

from ablation_study_jepa.models.jepa import LossAggregator, MultiLayerJEPAModule

warnings.filterwarnings(
    "ignore",
    message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
    category=UserWarning,
    module=r"lightning\.pytorch\.utilities\._pytree",
)

try:  # pragma: no cover - exercised when Lightning is installed.
    import lightning.pytorch as pl
except ModuleNotFoundError:  # pragma: no cover
    pl = None


BaseLightningModule = pl.LightningModule if pl is not None else nn.Module


class ReturnPredictionLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        jepa_module: MultiLayerJEPAModule | None = None,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-5,
        lambda_jepa: float = 0.05,
    ) -> None:
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.jepa_module = jepa_module
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_aggregator = LossAggregator(lambda_jepa=lambda_jepa)
        if pl is not None:
            self.save_hyperparameters(ignore=["model", "criterion", "jepa_module"])

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(batch, return_hidden_states=False)["y_pred"]

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        outputs = self.model(batch, return_hidden_states=self.jepa_module is not None)
        supervised_loss = self.criterion(outputs["y_pred"], batch["y"])
        jepa_loss = None
        jepa_logs: dict[str, torch.Tensor] = {}

        if self.jepa_module is not None:
            if outputs["hidden_states"] is None:
                raise RuntimeError("JEPA training requires hidden states from the base model")
            target_hidden_by_horizon = self._compute_target_hidden_states(batch)
            if hasattr(self.model, "compute_jepa_loss"):
                jepa_output = self.model.compute_jepa_loss(
                    context_hidden_states=outputs["hidden_states"],
                    target_hidden_states_by_horizon=target_hidden_by_horizon,
                    metadata=batch["metadata"],
                    context_transformer_input=outputs.get("transformer_input"),
                    attention_mask=outputs.get("attention_mask"),
                    domain_context=self._jepa_domain_context(batch),
                )
            else:
                jepa_output = self.jepa_module(
                    context_hidden_states=outputs["hidden_states"],
                    target_hidden_states_by_horizon=target_hidden_by_horizon,
                    metadata=batch["metadata"],
                    domain_context=self._jepa_domain_context(batch),
                )
            jepa_loss = jepa_output["loss"]
            jepa_logs = jepa_output["logs"]

        warmup_scale = self._jepa_warmup_scale()
        losses = self.loss_aggregator(supervised_loss, jepa_loss, jepa_scale=warmup_scale)
        jepa_diagnostic_logs = self._jepa_diagnostic_logs(
            supervised_loss=supervised_loss,
            losses=losses,
            jepa_logs=jepa_logs,
            warmup_scale=warmup_scale,
        )
        gradient_logs = self._gradient_norm_logs(
            supervised_loss=supervised_loss,
            weighted_jepa_loss=losses["weighted_jepa_loss"],
        )
        self._log_dict(
            {
                "train/supervised_loss": losses["supervised_loss"],
                "train/total_jepa_loss": losses["total_jepa_loss"],
                "train/weighted_jepa_loss": losses["weighted_jepa_loss"],
                "train/total_loss": losses["total_loss"],
                **{
                    f"train/{key}": value
                    for key, value in {
                        **jepa_logs,
                        **jepa_diagnostic_logs,
                        **gradient_logs,
                    }.items()
                },
            },
            batch_size=self._batch_size(batch),
            on_step=True,
            on_epoch=True,
        )
        return losses["total_loss"]

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._eval_step(batch, "test")

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:
        y_pred = self.model(batch, return_hidden_states=False)["y_pred"]
        return {"y_pred": y_pred.detach(), "y_true": batch["y"].detach(), "metadata": batch["metadata"]}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _eval_step(self, batch: dict[str, Any], prefix: str) -> torch.Tensor:
        y_pred = self.model(batch, return_hidden_states=False)["y_pred"]
        loss = self.criterion(y_pred, batch["y"])
        self._log_dict(
            {f"{prefix}/prediction_loss": loss},
            batch_size=self._batch_size(batch),
            on_step=False,
            on_epoch=True,
        )
        return loss

    def _compute_target_hidden_states(self, batch: dict[str, Any]) -> dict[int, list[torch.Tensor]]:
        if "future_x" not in batch or "future_horizons" not in batch:
            raise RuntimeError(
                "JEPA is enabled but the batch does not contain future windows. "
                "Set dataset.include_future_window=true."
            )
        future_x = batch["future_x"]
        future_horizons = batch["future_horizons"]
        target_hidden_by_horizon: dict[int, list[torch.Tensor]] = {}
        horizon_count = future_x.size(1)
        requires_grad = self._jepa_target_hidden_requires_grad()
        grad_context = nullcontext() if requires_grad else torch.no_grad()
        with grad_context:
            for horizon_index in range(horizon_count):
                horizon = int(future_horizons[0, horizon_index].item())
                future_batch = {"x": future_x[:, horizon_index, :, :]}
                if "static" in batch:
                    future_batch["static"] = batch["static"]
                future_outputs = self.model(future_batch, return_hidden_states=True)
                hidden_states = future_outputs["hidden_states"]
                if hidden_states is None:
                    raise RuntimeError("Base model did not return target hidden states")
                if requires_grad:
                    target_hidden_by_horizon[horizon] = hidden_states
                else:
                    target_hidden_by_horizon[horizon] = [state.detach() for state in hidden_states]
        return target_hidden_by_horizon

    def _jepa_target_hidden_requires_grad(self) -> bool:
        if self.jepa_module is None:
            return False
        config = self.jepa_module.config
        if config.mode != "lejepa":
            return False
        if config.lejepa.auxiliary.gradient_strategy.compute_target_with_no_grad:
            return False
        return not config.lejepa.detach_target

    def _jepa_domain_context(self, batch: dict[str, Any]) -> torch.Tensor | None:
        if self.jepa_module is None:
            return None
        config = self.jepa_module.config
        if config.mode != "lejepa":
            return None
        if not config.lejepa.representation.domain_context.enabled:
            return None
        if "static" not in batch:
            raise RuntimeError(
                "LeJEPA domain_context.enabled=true requires static features in the batch"
            )
        return batch["static"]

    def _jepa_warmup_scale(self) -> float:
        auxiliary_config = self._active_auxiliary_config()
        if auxiliary_config is None:
            return 1.0
        epoch = int(getattr(self, "current_epoch", 0))
        return float(auxiliary_config.warmup.scale_for_epoch(epoch))

    def _jepa_diagnostic_logs(
        self,
        supervised_loss: torch.Tensor,
        losses: dict[str, torch.Tensor],
        jepa_logs: dict[str, torch.Tensor],
        warmup_scale: float,
    ) -> dict[str, torch.Tensor]:
        auxiliary_config = self._active_auxiliary_config()
        if self.jepa_module is None or auxiliary_config is None:
            return {}
        device = supervised_loss.device
        dtype = supervised_loss.dtype
        logs: dict[str, torch.Tensor] = {
            "jepa_warmup_scale": torch.as_tensor(warmup_scale, device=device, dtype=dtype),
            "jepa_effective_global_weight": losses["effective_lambda_jepa"].detach(),
        }
        for layer, layer_weight in zip(
            self.jepa_module.selected_layers,
            self.jepa_module.layer_weights,
            strict=True,
        ):
            layer_loss = jepa_logs.get(f"jepa_layer_{layer}_loss")
            if layer_loss is None:
                layer_loss = jepa_logs.get(f"jepa_loss_layer_{layer}")
            if layer_loss is None:
                continue
            effective_weight = float(layer_weight) * float(
                losses["effective_lambda_jepa"].detach().item()
            )
            effective_weight_tensor = torch.as_tensor(effective_weight, device=device, dtype=dtype)
            logs[f"jepa_layer_{layer}_effective_weight"] = effective_weight_tensor
            logs[f"jepa_layer_{layer}_weighted_loss"] = layer_loss.detach() * effective_weight_tensor

        if auxiliary_config.diagnostics.log_aux_loss_ratios:
            denominator = supervised_loss.detach().abs().clamp_min(1e-12)
            logs["jepa_weighted_to_supervised_loss_ratio"] = (
                losses["weighted_jepa_loss"].detach().abs() / denominator
            )
        return logs

    def _gradient_norm_logs(
        self,
        supervised_loss: torch.Tensor,
        weighted_jepa_loss: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        auxiliary_config = self._active_auxiliary_config()
        if self.jepa_module is None or auxiliary_config is None:
            return {}
        if not auxiliary_config.diagnostics.log_gradient_norms:
            return {}

        groups = self._transformer_block_parameter_groups()
        if not groups:
            return {}
        supervised_vectors = self._loss_grad_vectors_by_group(supervised_loss, groups)
        auxiliary_vectors = self._loss_grad_vectors_by_group(weighted_jepa_loss, groups)

        logs: dict[str, torch.Tensor] = {}
        zero = torch.zeros((), dtype=supervised_loss.dtype, device=supervised_loss.device)
        for layer, _ in groups:
            supervised_vector = supervised_vectors.get(layer)
            auxiliary_vector = auxiliary_vectors.get(layer)
            if supervised_vector is None or supervised_vector.numel() == 0:
                supervised_norm = zero
            else:
                supervised_norm = supervised_vector.norm().to(dtype=supervised_loss.dtype)
            if auxiliary_vector is None or auxiliary_vector.numel() == 0:
                auxiliary_norm = zero
            else:
                auxiliary_norm = auxiliary_vector.norm().to(dtype=supervised_loss.dtype)
            logs[f"grad_norm_supervised_block_{layer}"] = supervised_norm
            logs[f"grad_norm_aux_block_{layer}"] = auxiliary_norm
            logs[f"grad_norm_aux_to_supervised_ratio_block_{layer}"] = (
                auxiliary_norm / supervised_norm.clamp_min(1e-12)
            )
            if (
                supervised_vector is not None
                and auxiliary_vector is not None
                and supervised_vector.numel() > 0
                and auxiliary_vector.numel() > 0
            ):
                denominator = supervised_vector.norm() * auxiliary_vector.norm()
                if bool((denominator > 0).item()):
                    cosine = torch.dot(supervised_vector, auxiliary_vector) / denominator
                else:
                    cosine = zero
            else:
                cosine = zero
            logs[f"grad_cosine_aux_supervised_block_{layer}"] = cosine.to(
                dtype=supervised_loss.dtype
            )
        return logs

    def _active_auxiliary_config(self) -> Any | None:
        if self.jepa_module is None:
            return None
        config = self.jepa_module.config
        if config.mode == "contrastive":
            return config.contrastive.auxiliary
        if config.mode == "lejepa":
            return config.lejepa.auxiliary
        return None

    def _transformer_block_parameter_groups(self) -> list[tuple[int, list[nn.Parameter]]]:
        transformer_stack = getattr(self.model, "transformer_stack", None)
        blocks = getattr(transformer_stack, "blocks", None)
        if blocks is None:
            return []
        return [
            (idx, [param for param in block.parameters() if param.requires_grad])
            for idx, block in enumerate(blocks)
        ]

    @staticmethod
    def _loss_grad_vectors_by_group(
        loss: torch.Tensor,
        groups: list[tuple[int, list[nn.Parameter]]],
    ) -> dict[int, torch.Tensor]:
        filtered_groups: list[tuple[int, list[nn.Parameter]]] = [
            (idx, params) for idx, params in groups if params
        ]
        if not filtered_groups or not loss.requires_grad:
            return {
                idx: torch.zeros(0, dtype=torch.float32, device=loss.device)
                for idx, _ in groups
            }

        flat_params = [param for _, params in filtered_groups for param in params]
        grads = torch.autograd.grad(
            loss,
            flat_params,
            retain_graph=True,
            allow_unused=True,
        )

        vectors: dict[int, torch.Tensor] = {}
        offset = 0
        for idx, params in filtered_groups:
            group_grads = grads[offset : offset + len(params)]
            offset += len(params)
            flat_grads = [
                (
                    grad.detach().float().flatten()
                    if grad is not None
                    else torch.zeros(param.numel(), dtype=torch.float32, device=loss.device)
                )
                for param, grad in zip(params, group_grads, strict=True)
            ]
            vectors[idx] = torch.cat(flat_grads) if flat_grads else torch.zeros(
                0,
                dtype=torch.float32,
                device=loss.device,
            )
        for idx, _ in groups:
            vectors.setdefault(idx, torch.zeros(0, dtype=torch.float32, device=loss.device))
        return vectors

    def _log_dict(self, values: dict[str, torch.Tensor], **kwargs: Any) -> None:
        if pl is not None:
            self.log_dict(values, prog_bar=False, logger=True, **kwargs)

    @staticmethod
    def _batch_size(batch: dict[str, Any]) -> int:
        return int(batch["y"].shape[0])
