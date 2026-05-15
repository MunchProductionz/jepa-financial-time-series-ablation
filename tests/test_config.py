import pytest

from ablation_study_jepa.cli import parse_dotted_overrides, prepare_sweep_config
from ablation_study_jepa.config.loader import load_config
from ablation_study_jepa.config.schemas import (
    ExperimentConfig,
    JEPAConfig,
    SlidingWindowConfig,
    SplitsConfig,
    normalize_weights,
)


def test_last_l_layer_selection_and_linear_weights() -> None:
    config = JEPAConfig(
        enabled=True,
        num_jepa_layers=3,
        layer_selection_mode="last_L",
        layer_weight_scheme="linear",
    )

    selected = config.resolve_selected_layers(num_transformer_blocks=6)

    assert selected == [3, 4, 5]
    assert config.normalized_layer_weights(selected) == [1 / 6, 2 / 6, 3 / 6]


def test_manual_weights_are_normalized() -> None:
    assert normalize_weights(3, "manual", manual_weights=[2, 2, 4]) == [0.25, 0.25, 0.5]


def test_yaml_defaults_loader_merges_base_config() -> None:
    config = load_config("configs/exp/jepa_ablation.yaml")

    assert config.jepa.enabled is True
    assert config.jepa.mode == "contrastive"
    assert config.jepa.resolve_selected_layers(config.model.num_transformer_blocks) == [3]
    assert config.jepa.contrastive.temperature == pytest.approx(0.1)
    assert config.jepa.contrastive.negative_strategy == "mixed"
    assert config.data.fit_scaler_on_train_only is True
    assert config.model.use_causal_mask is True


def test_dotted_overrides_are_applied_to_loaded_config() -> None:
    config = load_config(
        "configs/exp/jepa_ablation.yaml",
        overrides={
            "jepa.num_jepa_layers": 2,
            "jepa.horizons": [1, 5],
            "logging.wandb.enabled": True,
        },
    )

    assert config.jepa.num_jepa_layers == 2
    assert config.jepa.horizons == [1, 5]
    assert config.logging.wandb.enabled is True


def test_cli_parses_wandb_style_dotted_overrides() -> None:
    overrides = parse_dotted_overrides(
        [
            "--jepa.num_jepa_layers=2",
            "--jepa.horizons=[1, 5, 20]",
            "--logging.wandb.enabled=true",
        ]
    )

    assert overrides == {
        "jepa.num_jepa_layers": 2,
        "jepa.horizons": [1, 5, 20],
        "logging.wandb.enabled": True,
    }


def test_sweep_config_gets_repo_safe_command_and_wandb_defaults() -> None:
    sweep_config = {
        "program": "ablation-study-jepa",
        "method": "grid",
        "parameters": {"config": {"value": "configs/exp/jepa_ablation.yaml"}},
    }

    prepare_sweep_config(sweep_config, project="ablation-study-jepa")

    assert sweep_config["command"] == [
        "${env}",
        "uv",
        "run",
        "ablation-study-jepa",
        "run",
        "${args}",
    ]
    assert sweep_config["parameters"]["logging.wandb.enabled"] == {"value": True}
    assert sweep_config["parameters"]["logging.wandb.mode"] == {"value": "online"}


def test_legacy_flat_contrastive_config_is_migrated() -> None:
    config = JEPAConfig(enabled=True, temperature=0.2, negative_strategy="in_batch_all")

    assert config.contrastive.temperature == pytest.approx(0.2)
    assert config.contrastive.negative_strategy == "in_batch_all"


def test_lejepa_config_parses_separate_loss_settings() -> None:
    config = load_config("configs/exp/lejepa_ablation.yaml")

    assert config.jepa.mode == "lejepa"
    assert config.features.max_missing_fraction == pytest.approx(0.3)
    assert config.jepa.resolve_selected_layers(config.model.num_transformer_blocks) == [2, 3]
    assert config.jepa.normalized_layer_weights([2, 3]) == [1 / 3, 2 / 3]
    assert config.jepa.lejepa.detach_target is False
    assert config.jepa.lejepa.loss_mix.lambda_sigreg == pytest.approx(0.05)
    assert config.jepa.lejepa.sigreg.apply_to == "context_and_targets"


def test_lambda_jepa_alias_sets_global_weight() -> None:
    config = JEPAConfig(enabled=True, mode="lejepa", lambda_jepa=0.01)

    assert config.global_weight == pytest.approx(0.01)


def test_manual_lejepa_layer_selection_and_weights() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        selected_layers=[0, 2],
        layer_selection_mode="manual",
        layer_weight_scheme="manual",
        manual_layer_weights=[1.0, 3.0],
    )

    selected = config.resolve_selected_layers(num_transformer_blocks=4)

    assert selected == [0, 2]
    assert config.normalized_layer_weights(selected) == [0.25, 0.75]


def test_fraction_split_config_requires_fractions_sum_to_one() -> None:
    config = SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1)

    assert config.method == "fraction"


def test_sliding_window_requires_fraction_splits() -> None:
    with pytest.raises(ValueError, match="splits.method='fraction'"):
        ExperimentConfig(
            splits=SplitsConfig(
                method="date",
                train_end="2020-12-31",
                val_end="2021-12-31",
                test_end="2022-12-31",
            ),
            sliding_window=SlidingWindowConfig(
                enabled=True,
                window_size_days=252,
                step_days=20,
            ),
        )
