import pandas as pd

from ablation_study_jepa.api.experiment import _compute_prediction_metrics
from ablation_study_jepa.evaluation.predictions import (
    _prediction_frame,
    make_prediction_run_dir,
    save_predictions,
)


def test_empty_prediction_frame_keeps_target_columns() -> None:
    frame = _prediction_frame([])

    assert list(frame.columns) == ["y_true", "y_pred"]
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
