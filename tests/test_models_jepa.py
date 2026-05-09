import pytest

torch = pytest.importorskip("torch")

from ablation_study_jepa.config.schemas import JEPAConfig
from ablation_study_jepa.models.jepa import MultiLayerJEPAModule
from ablation_study_jepa.models.tft import TFT
from ablation_study_jepa.models.tft_with_jepa import TFTWithJEPA


def test_tft_returns_all_transformer_hidden_states() -> None:
    model = TFT(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=3,
        dropout=0.0,
    )
    batch = {"x": torch.randn(2, 12, 5)}

    outputs = model(batch, return_hidden_states=True)

    assert outputs["y_pred"].shape == (2, 1)
    assert len(outputs["hidden_states"]) == 3
    assert outputs["hidden_states"][0].shape == (2, 12, 16)


def test_tft_with_jepa_embeds_jepa_heads_after_transformer_blocks() -> None:
    config = JEPAConfig(
        enabled=True,
        num_jepa_layers=2,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        negative_strategy="in_batch_all",
    )
    model = TFTWithJEPA(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=3,
        jepa_config=config,
    )

    assert model.jepa_module is not None
    assert model.jepa_module.selected_layers == [1, 2]


def test_jepa_loss_detaches_target_hidden_states() -> None:
    config = JEPAConfig(
        enabled=True,
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        negative_strategy="in_batch_all",
    )
    module = MultiLayerJEPAModule(hidden_dim=12, num_transformer_blocks=2, config=config)
    context_hidden = [
        torch.randn(4, 5, 12, requires_grad=True),
        torch.randn(4, 5, 12, requires_grad=True),
    ]
    target_hidden = [
        torch.randn(4, 5, 12, requires_grad=True),
        torch.randn(4, 5, 12, requires_grad=True),
    ]
    metadata = {
        "asset_id": ["A", "B", "C", "D"],
        "anchor_date_ordinal": torch.tensor([100, 101, 102, 103]),
        "target_date_ordinal_horizon_1": torch.tensor([101, 102, 103, 104]),
        "anchor_position": torch.tensor([10, 10, 10, 10]),
        "target_position_horizon_1": torch.tensor([11, 11, 11, 11]),
        "sector": ["x", "y", "z", "w"],
    }

    output = module(context_hidden, {1: target_hidden}, metadata)
    output["loss"].backward()

    assert context_hidden[1].grad is not None
    assert target_hidden[1].grad is None
