"""Simple LSTM baseline for stock-return prediction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LSTMReturnForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_size: int = 1,
        **_: Any,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(
        self,
        batch_or_x: dict[str, torch.Tensor] | torch.Tensor,
        return_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        x = batch_or_x["x"] if isinstance(batch_or_x, dict) else batch_or_x
        encoded, _ = self.encoder(x)
        y_pred = self.head(encoded[:, -1, :])
        return {"y_pred": y_pred, "hidden_states": [encoded] if return_hidden_states else None}

