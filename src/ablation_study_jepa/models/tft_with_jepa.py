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
    heads to selected Transformer block outputs and future-window target
    representations. Target detachment is controlled by the active JEPA mode.
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
        context_transformer_input: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.jepa_module is None:
            device = context_hidden_states[-1].device
            zero = torch.zeros((), device=device)
            return {"loss": zero, "logs": {"total_jepa_loss_unweighted": zero}}
        if self._use_local_recompute():
            if context_transformer_input is None:
                raise RuntimeError("local_recompute JEPA strategy requires transformer_input")
            gradient_config = self._active_auxiliary_gradient_config()
            context_hidden_states = self.recompute_transformer_layers(
                transformer_input=context_transformer_input,
                hidden_states=context_hidden_states,
                layers=self.jepa_module.selected_layers,
                mask=attention_mask,
                detach_lower_inputs=gradient_config.detach_lower_inputs,
            )
        return self.jepa_module(
            context_hidden_states=context_hidden_states,
            target_hidden_states_by_horizon=target_hidden_states_by_horizon,
            metadata=metadata,
        )

    def _use_local_recompute(self) -> bool:
        gradient_config = self._active_auxiliary_gradient_config()
        if gradient_config is None:
            return False
        return gradient_config.name == "local_recompute"

    def _active_auxiliary_gradient_config(self) -> Any | None:
        if self.jepa_module is None:
            return None
        config = self.jepa_module.config
        if config.mode == "contrastive":
            return config.contrastive.auxiliary.gradient_strategy
        if config.mode == "lejepa":
            return config.lejepa.auxiliary.gradient_strategy
        return None
