from torch import nn

from ablation_study_jepa.config.schemas import ExperimentConfig, SplitsConfig
from ablation_study_jepa.evaluation.model_summary import (
    model_summary_flat_fields,
    summarize_model,
)


def test_model_summary_counts_attached_and_separate_auxiliary_modules() -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 1))
    attached_summary = summarize_model(model, jepa_module=model[1])

    assert attached_summary["parameter_count"] == 21
    assert attached_summary["base_parameter_count"] == 16
    assert attached_summary["jepa_parameter_count"] == 5

    separate_aux = nn.Linear(1, 2)
    separate_summary = summarize_model(model, jepa_module=separate_aux)

    assert separate_summary["parameter_count"] == 25
    assert separate_summary["base_parameter_count"] == 21
    assert separate_summary["jepa_parameter_count"] == 4


def test_model_summary_flat_fields_are_csv_friendly() -> None:
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
    )
    fields = model_summary_flat_fields(
        {
            "parameter_count": 21,
            "trainable_parameter_count": 21,
            "base_parameter_count": 16,
            "jepa_parameter_count": 5,
            "module_counts": {"linear": 2, "lstm_layers": 0},
            "architecture": {"transformer_block_count": 4, "mlp_linear_layer_count": 2},
        },
        config,
    )

    assert fields["model_parameter_count"] == 21
    assert fields["model_base_parameter_count"] == 16
    assert fields["model_jepa_parameter_count"] == 5
    assert fields["model_transformer_block_count"] == 4
    assert fields["model_linear_layer_count"] == 2
