# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from ablation_study_jepa.config.schemas import JEPAConfig
from ablation_study_jepa.models.jepa import MultiLayerJEPAModule, SIGRegLoss
from ablation_study_jepa.models.tft import TFT
from ablation_study_jepa.models.tft_with_jepa import TFTWithJEPA
from ablation_study_jepa.training.lightning_module import ReturnPredictionLightningModule


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
        contrastive={"negative_strategy": "in_batch_all"},
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
        contrastive={"negative_strategy": "in_batch_all"},
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


def test_sigreg_returns_finite_differentiable_scalar() -> None:
    loss_fn = SIGRegLoss(
        embedding_dim=6,
        num_slices=8,
        num_t=5,
        t_max=2.0,
        min_batch_size=2,
    )
    z = torch.randn(4, 6, requires_grad=True)

    loss = loss_fn(z)
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert z.grad is not None


def test_sigreg_skips_small_batches_without_crashing() -> None:
    loss_fn = SIGRegLoss(embedding_dim=6, min_batch_size=8)
    z = torch.randn(2, 6, requires_grad=True)

    loss = loss_fn(z)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert z.grad is not None


def test_lejepa_loss_combines_prediction_and_sigreg_terms() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": False,
            "loss_mix": {"lambda_sigreg": 0.25},
            "sigreg": {
                "enabled": True,
                "num_slices": 8,
                "num_t": 5,
                "t_max": 2.0,
                "min_batch_size": 2,
            },
        },
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

    output = module(context_hidden, {1: target_hidden}, metadata={})
    output["loss"].backward()
    logs = output["logs"]
    expected = 0.75 * logs["jepa_prediction_loss"] + 0.25 * logs["jepa_sigreg_loss"]

    assert torch.allclose(logs["jepa_loss"], expected)
    assert context_hidden[1].grad is not None
    assert target_hidden[1].grad is not None


def test_lejepa_can_detach_only_target_latents() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": True,
            "sigreg": {"enabled": False},
        },
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

    output = module(context_hidden, {1: target_hidden}, metadata={})
    output["loss"].backward()

    assert context_hidden[1].grad is not None
    assert target_hidden[1].grad is None


def test_lejepa_target_sigreg_is_independent_of_prediction_target_detach() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": True,
            "loss_mix": {"lambda_sigreg": 1.0},
            "sigreg": {
                "enabled": True,
                "apply_to": "targets_only",
                "num_slices": 8,
                "num_t": 5,
                "t_max": 2.0,
                "min_batch_size": 2,
            },
        },
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

    output = module(context_hidden, {1: target_hidden}, metadata={})
    output["loss"].backward()

    assert _tensor_grad_norm(context_hidden[1]) == pytest.approx(0.0)
    assert target_hidden[1].grad is not None
    assert _tensor_grad_norm(target_hidden[1]) > 0.0


def test_lejepa_horizon_mask_excludes_invalid_pairs() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1, 5],
        lejepa={"sigreg": {"enabled": False}},
    )
    module = MultiLayerJEPAModule(
        hidden_dim=12,
        num_transformer_blocks=1,
        config=config,
        dropout=0.0,
    )
    context_hidden = [torch.randn(3, 5, 12)]
    target_hidden_h1 = [torch.randn(3, 5, 12)]
    target_hidden_h5 = [torch.randn(3, 5, 12)]
    metadata = {
        "valid_horizon_1": torch.tensor([True, False, True]),
        "valid_horizon_5": torch.tensor([False, False, False]),
    }

    output = module(
        context_hidden,
        {1: target_hidden_h1, 5: target_hidden_h5},
        metadata=metadata,
    )
    head = module.heads["0"]
    valid = metadata["valid_horizon_1"]
    z_context = head.encode_context(context_hidden[0][:, -1, :])[valid]
    z_pred = head.predict_from_latent(z_context, horizon=1)
    z_target = head.encode_target(target_hidden_h1[0][:, -1, :][valid], detach=False)
    expected = torch.nn.functional.mse_loss(z_pred, z_target)

    assert torch.allclose(output["loss"], expected)
    assert output["logs"]["jepa_layer_0_horizon_1_valid_count"].item() == pytest.approx(2.0)
    assert output["logs"]["jepa_layer_0_horizon_5_valid_count"].item() == pytest.approx(0.0)


def test_lejepa_inference_forward_does_not_require_future_windows() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
    )
    model = TFTWithJEPA(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=2,
        jepa_config=config,
    )
    batch = {"x": torch.randn(2, 12, 5)}

    output = model(batch, return_hidden_states=False)

    assert output["y_pred"].shape == (2, 1)
    assert output["hidden_states"] is None


def test_local_recompute_auxiliary_loss_does_not_update_lower_blocks() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": True,
            "sigreg": {"enabled": False},
            "auxiliary": {"gradient_strategy": {"name": "local_recompute"}},
        },
    )
    model = TFTWithJEPA(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=2,
        dropout=0.0,
        jepa_config=config,
    )
    context_batch = {"x": torch.randn(3, 12, 5)}
    target_batch = {"x": torch.randn(3, 12, 5)}
    context_outputs = model(context_batch, return_hidden_states=True)
    with torch.no_grad():
        target_outputs = model(target_batch, return_hidden_states=True)

    output = model.compute_jepa_loss(
        context_hidden_states=context_outputs["hidden_states"],
        target_hidden_states_by_horizon={1: target_outputs["hidden_states"]},
        metadata={},
        context_transformer_input=context_outputs["transformer_input"],
        attention_mask=context_outputs["attention_mask"],
    )
    output["loss"].backward()

    lower_block_grad = _module_grad_norm(model.transformer_stack.blocks[0])
    selected_block_grad = _module_grad_norm(model.transformer_stack.blocks[1])

    assert lower_block_grad == pytest.approx(0.0)
    assert selected_block_grad > 0.0


def test_global_weighted_auxiliary_loss_updates_lower_blocks() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": True,
            "sigreg": {"enabled": False},
            "auxiliary": {"gradient_strategy": {"name": "global_weighted"}},
        },
    )
    model = TFTWithJEPA(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=2,
        dropout=0.0,
        jepa_config=config,
    )
    context_batch = {"x": torch.randn(3, 12, 5)}
    target_batch = {"x": torch.randn(3, 12, 5)}
    context_outputs = model(context_batch, return_hidden_states=True)
    with torch.no_grad():
        target_outputs = model(target_batch, return_hidden_states=True)

    output = model.compute_jepa_loss(
        context_hidden_states=context_outputs["hidden_states"],
        target_hidden_states_by_horizon={1: target_outputs["hidden_states"]},
        metadata={},
        context_transformer_input=context_outputs["transformer_input"],
        attention_mask=context_outputs["attention_mask"],
    )
    output["loss"].backward()

    assert _module_grad_norm(model.transformer_stack.blocks[0]) > 0.0


def test_gradient_norm_diagnostics_report_transformer_block_ratios() -> None:
    config = JEPAConfig(
        enabled=True,
        mode="lejepa",
        num_jepa_layers=1,
        layer_selection_mode="last_L",
        projection_dim=8,
        horizons=[1],
        lejepa={
            "detach_target": True,
            "sigreg": {"enabled": False},
            "auxiliary": {
                "diagnostics": {"log_gradient_norms": True},
            },
        },
    )
    model = TFTWithJEPA(
        input_dim=5,
        hidden_dim=16,
        num_attention_heads=4,
        num_transformer_blocks=2,
        dropout=0.0,
        jepa_config=config,
    )
    lightning = ReturnPredictionLightningModule(
        model=model,
        criterion=torch.nn.MSELoss(),
        jepa_module=model.jepa_module,
        lambda_jepa=0.05,
    )
    context_batch = {"x": torch.randn(3, 12, 5)}
    target_batch = {"x": torch.randn(3, 12, 5)}
    context_outputs = model(context_batch, return_hidden_states=True)
    supervised_loss = torch.nn.functional.mse_loss(context_outputs["y_pred"], torch.randn(3, 1))
    with torch.no_grad():
        target_outputs = model(target_batch, return_hidden_states=True)
    jepa_output = model.compute_jepa_loss(
        context_hidden_states=context_outputs["hidden_states"],
        target_hidden_states_by_horizon={1: target_outputs["hidden_states"]},
        metadata={},
        context_transformer_input=context_outputs["transformer_input"],
        attention_mask=context_outputs["attention_mask"],
    )
    losses = lightning.loss_aggregator(supervised_loss, jepa_output["loss"])

    logs = lightning._gradient_norm_logs(
        supervised_loss=supervised_loss,
        weighted_jepa_loss=losses["weighted_jepa_loss"],
    )

    assert "grad_norm_supervised_block_0" in logs
    assert "grad_norm_aux_block_1" in logs
    assert "grad_norm_aux_to_supervised_ratio_block_1" in logs
    assert torch.isfinite(logs["grad_norm_aux_to_supervised_ratio_block_1"])


def _module_grad_norm(module: torch.nn.Module) -> float:
    squared = [
        param.grad.detach().float().square().sum()
        for param in module.parameters()
        if param.grad is not None
    ]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().item())


def _tensor_grad_norm(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().float().square().sum().sqrt().item())
