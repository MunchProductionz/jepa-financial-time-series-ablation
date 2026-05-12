import json

import pandas as pd
import pytest

from ablation_study_jepa.api.experiment import (
    ExperimentRunner,
    _compute_prediction_metrics,
    _window_metrics_by_index,
)
from ablation_study_jepa.config.schemas import ExperimentConfig, SplitsConfig
from ablation_study_jepa.evaluation.predictions import (
    _prediction_frame,
    make_prediction_run_dir,
    save_predictions,
)


def test_empty_prediction_frame_keeps_target_columns() -> None:
    frame = _prediction_frame([])

    assert list(frame.columns) == ["y_true", "y_pred", "y_true_return", "y_pred_return"]
    assert frame.empty


def test_optional_empty_test_metrics_return_nan() -> None:
    frame = pd.DataFrame({"y_true": [], "y_pred": []})

    metrics = _compute_prediction_metrics(
        frame,
        metric_names=["mse", "mae"],
        split="test",
        require_nonempty=False,
    )

    assert set(metrics) == {"mse", "mae"}
    assert all(pd.isna(value) for value in metrics.values())


def test_prediction_artifacts_are_saved_inside_run_directory(tmp_path) -> None:
    run_dir = make_prediction_run_dir(
        tmp_path,
        run_name="tft_jepa_last1_h1_lambda005",
        tags=["jepa", "last_L", "horizon_1"],
        timestamp="20260509T214304Z",
        config_dict={"seed": 42},
    )
    path = save_predictions(
        pd.DataFrame({"y_true": [0.1], "y_pred": [0.2]}),
        run_dir,
        "test",
    )

    assert run_dir.parent == tmp_path
    assert run_dir.name.startswith("tft_jepa_last1_h1_lambda005_jepa_last_L_horizon_1_")
    assert path == run_dir / "test.csv"
    assert path.exists()


def test_prediction_artifacts_convert_log_returns_to_simple_returns(tmp_path) -> None:
    path = save_predictions(
        pd.DataFrame({"y_true": [0.09531018], "y_pred": [0.18232156]}),
        tmp_path,
        "val",
    )

    saved = pd.read_csv(path)
    assert saved.loc[0, "y_true"] == pytest.approx(0.09531018)
    assert saved.loc[0, "y_pred"] == pytest.approx(0.18232156)
    assert saved.loc[0, "y_true_return"] == pytest.approx(0.10)
    assert saved.loc[0, "y_pred_return"] == pytest.approx(0.20)


def test_metrics_json_groups_total_and_window_metrics(tmp_path) -> None:
    runner = ExperimentRunner(
        ExperimentConfig(
            splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
        )
    )
    window_metrics = [
        {"index": 0, "val": {"mse": 0.1}, "test": {"mse": 0.2}},
        {"index": 1, "val": {"mse": 0.3}, "test": {"mse": 0.4}},
    ]

    path = runner._save_metrics(
        val_metrics={"mse": 0.2},
        test_metrics={"mse": 0.3},
        window_metrics=window_metrics,
        config_dict={"seed": 42},
        output_dir=tmp_path,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"config", "metrics", "run_name"}
    assert payload["metrics"] == {
        "total": {
            "val": {"mse": 0.2},
            "test": {"mse": 0.3},
        },
        "windows": {
            "0": {"val": {"mse": 0.1}, "test": {"mse": 0.2}},
            "1": {"val": {"mse": 0.3}, "test": {"mse": 0.4}},
        },
    }


def test_window_metrics_are_keyed_by_window_index() -> None:
    metrics = _window_metrics_by_index(
        [
            {"index": 3, "label": "window_003", "val": {"mae": 0.1}, "test": {"mae": 0.2}},
        ]
    )

    assert metrics == {"3": {"val": {"mae": 0.1}, "test": {"mae": 0.2}}}
