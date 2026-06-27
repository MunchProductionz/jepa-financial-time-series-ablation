"""Model size and architecture summaries for experiment analysis."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from torch import nn

from ablation_study_jepa.config.schemas import ExperimentConfig


def summarize_model(
    model: nn.Module,
    *,
    config: ExperimentConfig | None = None,
    jepa_module: nn.Module | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable model-size summary.

    The summary mixes actual instantiated-module counts with config dimensions
    so downstream analysis can group by both parameter count and architectural
    settings across TFT, JEPA, and other baselines.
    """

    model_parameter_count = count_parameters(model)
    model_trainable_parameter_count = count_parameters(model, trainable_only=True)
    jepa_parameter_count = count_parameters(jepa_module) if jepa_module is not None else 0
    jepa_trainable_parameter_count = (
        count_parameters(jepa_module, trainable_only=True) if jepa_module is not None else 0
    )
    jepa_attached = _contains_module(model, jepa_module)
    parameter_count = (
        model_parameter_count if jepa_attached else model_parameter_count + jepa_parameter_count
    )
    trainable_parameter_count = (
        model_trainable_parameter_count
        if jepa_attached
        else model_trainable_parameter_count + jepa_trainable_parameter_count
    )
    base_parameter_count = (
        model_parameter_count - jepa_parameter_count if jepa_attached else model_parameter_count
    )
    module_counts = _module_counts(model)
    architecture = _architecture_summary(model, config)
    return {
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "non_trainable_parameter_count": parameter_count - trainable_parameter_count,
        "base_parameter_count": base_parameter_count,
        "jepa_parameter_count": jepa_parameter_count,
        "top_level_parameter_counts": _top_level_parameter_counts(model),
        "module_counts": module_counts,
        "architecture": architecture,
    }


def model_summary_flat_fields(
    model_summary: dict[str, Any] | None,
    config: ExperimentConfig,
) -> dict[str, Any]:
    """Flatten common model-size dimensions into stable CSV-friendly columns."""

    summary = model_summary or {}
    modules = summary.get("module_counts") or {}
    architecture = summary.get("architecture") or _architecture_summary(None, config)
    model_params = config.model.params
    return {
        "model_parameter_count": summary.get("parameter_count"),
        "model_trainable_parameter_count": summary.get("trainable_parameter_count"),
        "model_non_trainable_parameter_count": summary.get("non_trainable_parameter_count"),
        "model_base_parameter_count": summary.get("base_parameter_count"),
        "model_jepa_parameter_count": summary.get("jepa_parameter_count"),
        "model_top_level_parameter_counts": _json_string(
            summary.get("top_level_parameter_counts") or {}
        ),
        "model_linear_layer_count": modules.get("linear"),
        "model_lstm_module_count": modules.get("lstm_modules"),
        "model_lstm_layer_count": modules.get("lstm_layers"),
        "model_attention_layer_count": modules.get("multihead_attention"),
        "model_transformer_block_count": architecture.get("transformer_block_count"),
        "model_mlp_linear_layer_count": architecture.get("mlp_linear_layer_count"),
        "model_predictor_linear_layer_count": architecture.get("jepa_predictor_linear_layer_count"),
        "model_input_dim": architecture.get("input_dim"),
        "model_static_input_dim": architecture.get("static_input_dim"),
        "model_hidden_dim_actual": architecture.get("hidden_dim"),
        "model_params_json": _json_string(model_params),
        "model_params_num_layers": model_params.get("num_layers"),
        "model_params_hidden_size": model_params.get("hidden_size"),
        "model_params_num_mlp_layers": model_params.get("num_mlp_layers"),
        "model_params_mlp_layers": model_params.get("mlp_layers"),
    }


def count_parameters(module: nn.Module | None, *, trainable_only: bool = False) -> int:
    if module is None:
        return 0
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def _contains_module(model: nn.Module, candidate: nn.Module | None) -> bool:
    if candidate is None:
        return True
    return any(module is candidate for module in model.modules())


def _module_counts(model: nn.Module) -> dict[str, int]:
    linear_count = 0
    lstm_modules = 0
    lstm_layers = 0
    multihead_attention = 0
    layer_norm = 0
    dropout = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            linear_count += 1
        elif isinstance(module, nn.LSTM):
            lstm_modules += 1
            lstm_layers += int(module.num_layers)
        elif isinstance(module, nn.MultiheadAttention):
            multihead_attention += 1
        elif isinstance(module, nn.LayerNorm):
            layer_norm += 1
        elif isinstance(module, nn.Dropout):
            dropout += 1
    return {
        "linear": linear_count,
        "lstm_modules": lstm_modules,
        "lstm_layers": lstm_layers,
        "multihead_attention": multihead_attention,
        "layer_norm": layer_norm,
        "dropout": dropout,
    }


def _architecture_summary(
    model: nn.Module | None,
    config: ExperimentConfig | None,
) -> dict[str, Any]:
    transformer_stack = getattr(model, "transformer_stack", None)
    transformer_blocks = getattr(transformer_stack, "blocks", None)
    jepa_module = getattr(model, "jepa_module", None)
    selected_jepa_layers = getattr(jepa_module, "selected_layers", None)
    architecture: dict[str, Any] = {
        "model_class": None if model is None else model.__class__.__name__,
        "input_dim": getattr(model, "input_dim", None),
        "static_input_dim": getattr(model, "static_input_dim", None),
        "hidden_dim": getattr(model, "hidden_dim", None),
        "transformer_block_count": (
            len(transformer_blocks)
            if transformer_blocks is not None
            else _config_value(config, "model", "num_transformer_blocks")
        ),
        "mlp_linear_layer_count": _mlp_linear_layer_count(model),
        "jepa_selected_layers": list(selected_jepa_layers) if selected_jepa_layers is not None else None,
        "jepa_predictor_linear_layer_count": _module_linear_count(jepa_module, "predictor"),
        "jepa_projector_linear_layer_count": _module_linear_count(jepa_module, "projector"),
    }
    if config is not None:
        architecture.update(
            {
                "model_target": config.model.target,
                "config_hidden_dim": config.model.hidden_dim,
                "config_num_transformer_blocks": config.model.num_transformer_blocks,
                "config_num_attention_heads": config.model.num_attention_heads,
                "config_num_lstm_layers": config.model.num_lstm_layers,
                "config_use_variable_selection": config.model.use_variable_selection,
                "config_model_params": config.model.params,
                "jepa_enabled": config.jepa.enabled,
                "jepa_mode": _value(config.jepa.mode),
                "jepa_num_layers": config.jepa.num_jepa_layers,
                "jepa_projection_dim": config.jepa.projection_dim,
                "jepa_predictor_type": _value(config.jepa.predictor_type),
                "jepa_horizons": list(config.jepa.horizons),
                "lejepa_representation_mode": _value(config.jepa.lejepa.representation.mode),
                "lejepa_adapter_dim": config.jepa.lejepa.representation.adapter_dim,
            }
        )
        if architecture["jepa_selected_layers"] is None:
            architecture["jepa_selected_layers"] = _resolved_jepa_layers(config)
    return architecture


def _top_level_parameter_counts(model: nn.Module) -> dict[str, int]:
    return {name: count_parameters(child) for name, child in model.named_children()}


def _mlp_linear_layer_count(model: nn.Module | None) -> int | None:
    if model is None:
        return None
    count = 0
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        lowered = module_name.lower()
        if any(token in lowered for token in ("ff", "grn", "head", "projection", "predictor", "projector")):
            count += 1
    return count


def _module_linear_count(module: nn.Module | None, name_fragment: str) -> int:
    if module is None:
        return 0
    return sum(
        1
        for module_name, child in module.named_modules()
        if name_fragment in module_name and isinstance(child, nn.Linear)
    )


def _config_value(config: ExperimentConfig | None, *keys: str) -> Any:
    current: Any = config
    for key in keys:
        if current is None:
            return None
        current = getattr(current, key, None)
    return current


def _resolved_jepa_layers(config: ExperimentConfig) -> list[int]:
    try:
        return config.jepa.resolve_selected_layers(config.model.num_transformer_blocks)
    except ValueError:
        return []


def _value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _json_string(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)
