"""TFT-inspired PyTorch model exposing Transformer block hidden states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


class GLU(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value(x) * torch.sigmoid(self.gate(x))


class GatedResidualNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_dim = output_dim or input_dim
        self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False) if context_dim else None
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.gate = GLU(output_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.fc1(x)
        if context is not None and self.context_proj is not None:
            if context.dim() == 2 and x.dim() == 3:
                context = context.unsqueeze(1)
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.dropout(self.fc2(hidden))
        return self.norm(residual + self.gate(hidden))


class VariableSelectionNetwork(nn.Module):
    """Minimal temporal variable selection over raw scalar features."""

    def __init__(self, input_dim: int, hidden_dim: int, context_dim: int | None = None) -> None:
        super().__init__()
        self.feature_projections = nn.ModuleList([nn.Linear(1, hidden_dim) for _ in range(input_dim)])
        self.weight_net = nn.Linear(input_dim + (context_dim or 0), input_dim)
        self.context_dim = context_dim or 0

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weight_input = x
        if context is not None and self.context_dim:
            if context.dim() == 2:
                context = context.unsqueeze(1).expand(-1, x.size(1), -1)
            weight_input = torch.cat([x, context], dim=-1)
        weights = torch.softmax(self.weight_net(weight_input), dim=-1)
        projected = torch.stack(
            [proj(x[..., idx : idx + 1]) for idx, proj in enumerate(self.feature_projections)],
            dim=-2,
        )
        selected = (weights.unsqueeze(-1) * projected).sum(dim=-2)
        return selected, weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, weights = self.attention(
            query,
            key,
            value,
            attn_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )
        return output, weights


class TFTTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = InterpretableMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.attn_gate = GLU(hidden_dim, hidden_dim)
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ff = GatedResidualNetwork(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim * 4,
            output_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out, attn_weights = self.attention(x, x, x, mask=mask)
        x = self.attn_norm(x + self.attn_gate(attn_out))
        x = self.ff(x)
        return x, attn_weights


class TransformerBlockStack(nn.Module):
    """Apply Transformer blocks in a loop and optionally retain each output."""

    def __init__(
        self,
        hidden_dim: int,
        num_attention_heads: int,
        num_transformer_blocks: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TFTTransformerBlock(hidden_dim, num_attention_heads, dropout)
                for _ in range(num_transformer_blocks)
            ]
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, return_hidden_states: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x, _ = block(x, mask=mask)
            if return_hidden_states:
                hidden_states.append(x)
        return x, hidden_states


@dataclass
class TFTOutput:
    y_pred: torch.Tensor
    hidden_states: list[torch.Tensor] | None = None
    transformer_input: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    feature_weights: torch.Tensor | None = None


class TFT(nn.Module):
    """A compact TFT-inspired forecaster for stock-return prediction."""

    def __init__(
        self,
        input_dim: int,
        static_input_dim: int = 0,
        hidden_dim: int = 128,
        num_attention_heads: int = 4,
        num_lstm_layers: int = 1,
        num_transformer_blocks: int = 4,
        dropout: float = 0.1,
        output_dim: int = 1,
        use_causal_mask: bool = True,
        use_variable_selection: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_attention_heads")
        self.input_dim = input_dim
        self.static_input_dim = static_input_dim
        self.hidden_dim = hidden_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.use_causal_mask = use_causal_mask
        self.use_variable_selection = use_variable_selection

        self.static_embedding = (
            nn.Sequential(nn.Linear(static_input_dim, hidden_dim), nn.ELU(), nn.LayerNorm(hidden_dim))
            if static_input_dim > 0
            else None
        )
        if use_variable_selection:
            self.temporal_vsn = VariableSelectionNetwork(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                context_dim=hidden_dim if static_input_dim > 0 else None,
            )
            self.temporal_embedding = None
        else:
            self.temporal_vsn = None
            self.temporal_embedding = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

        self.lstm_encoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.post_lstm_gate = GLU(hidden_dim, hidden_dim)
        self.post_lstm_norm = nn.LayerNorm(hidden_dim)
        self.enrichment_grn = (
            GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, hidden_dim, dropout)
            if static_input_dim > 0
            else None
        )
        self.transformer_stack = TransformerBlockStack(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout,
        )
        self.output_projection = nn.Sequential(
            GatedResidualNetwork(hidden_dim, hidden_dim * 2, hidden_dim, dropout=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        batch_or_x: dict[str, torch.Tensor] | torch.Tensor,
        static_features: torch.Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        if isinstance(batch_or_x, dict):
            x = batch_or_x["x"]
            static_features = batch_or_x.get("static", static_features)
        else:
            x = batch_or_x

        static_context = self.static_embedding(static_features) if self.static_embedding and static_features is not None else None
        if self.temporal_vsn is not None:
            temporal, feature_weights = self.temporal_vsn(x, context=static_context)
        else:
            temporal = self.temporal_embedding(x)
            feature_weights = None

        lstm_out, _ = self.lstm_encoder(temporal)
        encoded = self.post_lstm_norm(temporal + self.post_lstm_gate(lstm_out))
        if self.enrichment_grn is not None and static_context is not None:
            encoded = self.enrichment_grn(encoded, context=static_context)

        mask = causal_attention_mask(x.size(1), x.device) if self.use_causal_mask else None
        transformed, hidden_states = self.transformer_stack(
            encoded, mask=mask, return_hidden_states=return_hidden_states
        )
        last_state = transformed[:, -1, :]
        y_pred = self.output_projection(last_state)
        return {
            "y_pred": y_pred,
            "hidden_states": hidden_states if return_hidden_states else None,
            "transformer_input": encoded if return_hidden_states else None,
            "attention_mask": mask if return_hidden_states else None,
            "feature_weights": feature_weights,
        }

    def recompute_transformer_layers(
        self,
        transformer_input: torch.Tensor,
        hidden_states: list[torch.Tensor],
        layers: list[int],
        mask: torch.Tensor | None = None,
        detach_lower_inputs: bool = True,
    ) -> list[torch.Tensor]:
        """Recompute selected Transformer blocks from detached lower-layer inputs."""

        if len(hidden_states) != len(self.transformer_stack.blocks):
            raise ValueError("hidden_states length must match the number of Transformer blocks")

        recomputed = list(hidden_states)
        for layer in layers:
            if layer < 0 or layer >= len(self.transformer_stack.blocks):
                raise ValueError(f"Transformer layer index out of range: {layer}")
            block_input = transformer_input if layer == 0 else hidden_states[layer - 1]
            if detach_lower_inputs:
                block_input = block_input.detach()
            block_output, _ = self.transformer_stack.blocks[layer](block_input, mask=mask)
            recomputed[layer] = block_output
        return recomputed


def causal_attention_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    """Return a boolean mask where True entries are not attendable."""

    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1,
    )
