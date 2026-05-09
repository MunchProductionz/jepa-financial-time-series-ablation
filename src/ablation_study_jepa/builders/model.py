"""Model, criterion, and JEPA module factories."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.models.jepa import MultiLayerJEPAModule
from ablation_study_jepa.utils.instantiate import locate


@dataclass
class ModelBundle:
    model: nn.Module
    criterion: nn.Module
    jepa: MultiLayerJEPAModule | None


def build_model_bundle(
    config: ExperimentConfig,
    input_dim: int,
    static_input_dim: int = 0,
) -> ModelBundle:
    model_cls = locate(config.model.target)
    model_kwargs = {
        "input_dim": input_dim,
        "static_input_dim": static_input_dim,
        "hidden_dim": config.model.hidden_dim,
        "num_attention_heads": config.model.num_attention_heads,
        "num_lstm_layers": config.model.num_lstm_layers,
        "num_transformer_blocks": config.model.num_transformer_blocks,
        "dropout": config.model.dropout,
        "output_dim": config.model.output_dim,
        "use_causal_mask": config.model.use_causal_mask,
        "use_variable_selection": config.model.use_variable_selection,
        "jepa_config": config.jepa,
    }
    model_kwargs.update(config.model.params)
    model = model_cls(**model_kwargs)
    criterion = locate(config.model.criterion)()

    jepa_module = getattr(model, "jepa_module", None)
    if config.jepa.enabled and config.jepa.num_jepa_layers > 0:
        if jepa_module is None:
            jepa_module = MultiLayerJEPAModule(
                hidden_dim=config.model.hidden_dim,
                num_transformer_blocks=config.model.num_transformer_blocks,
                config=config.jepa,
                dropout=config.model.dropout,
            )
    return ModelBundle(model=model, criterion=criterion, jepa=jepa_module)
