from ablation_study_jepa.config.loader import load_config
from ablation_study_jepa.config.schemas import JEPAConfig, SplitsConfig, normalize_weights


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
    assert config.jepa.resolve_selected_layers(config.model.num_transformer_blocks) == [3]
    assert config.data.fit_scaler_on_train_only is True
    assert config.model.use_causal_mask is True


def test_fraction_split_config_requires_fractions_sum_to_one() -> None:
    config = SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1)

    assert config.method == "fraction"
