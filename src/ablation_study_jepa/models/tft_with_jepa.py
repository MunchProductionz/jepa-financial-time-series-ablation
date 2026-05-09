"""TFT model variant with JEPA heads attached after Transformer blocks."""

from __future__ import annotations

from typing import Any

import torch

from ablation_study_jepa.config.schemas import JEPAConfig
from ablation_study_jepa.models.jepa import MultiLayerJEPAModule
from ablation_study_jepa.models.tft import TFT


class TFTWithJEPA(TFT):
    """TFT forecaster with embedded training-only JEPA auxiliary heads.

    The supervised forward path is identical to ``TFT``. When hidden states are
    requested during training, ``compute_jepa_loss`` applies the attached JEPA
    heads to selected Transformer block outputs and detached future-window
    target representations.
    """

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
        jepa_config: JEPAConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            static_input_dim=static_input_dim,
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            num_lstm_layers=num_lstm_layers,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout,
            output_dim=output_dim,
            use_causal_mask=use_causal_mask,
            use_variable_selection=use_variable_selection,
            **kwargs,
        )
        self.jepa_module: MultiLayerJEPAModule | None = None
        if jepa_config is not None and jepa_config.enabled and jepa_config.num_jepa_layers > 0:
            self.jepa_module = MultiLayerJEPAModule(
                hidden_dim=hidden_dim,
                num_transformer_blocks=num_transformer_blocks,
                config=jepa_config,
                dropout=dropout,
            )

    @property
    def has_jepa(self) -> bool:
        return self.jepa_module is not None

    def compute_jepa_loss(
        self,
        context_hidden_states: list[torch.Tensor],
        target_hidden_states_by_horizon: dict[int, list[torch.Tensor]],
        metadata: dict[str, Any],
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.jepa_module is None:
            device = context_hidden_states[-1].device
            zero = torch.zeros((), device=device)
            return {"loss": zero, "logs": {"total_jepa_loss_unweighted": zero}}
        return self.jepa_module(
            context_hidden_states=context_hidden_states,
            target_hidden_states_by_horizon=target_hidden_states_by_horizon,
            metadata=metadata,
        )

