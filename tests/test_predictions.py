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
from ablation_study_jepa.training.history import combine_history_files


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


def test_prediction_metrics_include_requested_economic_metrics() -> None:
    frame = pd.DataFrame(
        {
            "anchor_date": ["2024-01-02"] * 4 + ["2024-01-03"] * 4,
            "asset_id": ["A", "B", "C", "D", "A", "B", "C", "D"],
            "sector": ["tech", "tech", "energy", "energy"] * 2,
            "y_true": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.01, -0.03],
            "y_pred": [0.04, -0.03, 0.02, -0.02, 0.03, -0.02, 0.01, -0.04],
            "y_true_return": [0.01, -0.02, 0.03, -0.01, 0.02, -0.01, 0.01, -0.03],
        }
    )

    metrics = _compute_prediction_metrics(
        frame,
        metric_names=[
            "rmse",
            "positive_prediction_share",
            "long_short_decile_return",
            "long_short_decile_sharpe",
            "long_short_decile_turnover",
            "long_short_decile_transaction_cost_adjusted_return",
            "sector_neutral_spearman_rank_ic",
        ],
        split="val",
        require_nonempty=True,
        transaction_cost_bps=10.0,
    )

    assert metrics["rmse"] >= 0.0
    assert metrics["positive_prediction_share"] == pytest.approx(0.5)
    assert metrics["long_short_decile_return"] > 0.0
    assert "long_short_decile_sharpe" in metrics
    assert "long_short_decile_turnover" in metrics
    assert metrics["long_short_decile_transaction_cost_adjusted_return"] <= metrics[
        "long_short_decile_return"
    ]
    assert pd.notna(metrics["sector_neutral_spearman_rank_ic"])


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
        config_path=tmp_path / "configs.json",
        training_history_paths={"combined": tmp_path / "training_history" / "combined.csv"},
        training_plot_paths={"losses": tmp_path / "training_history" / "plots" / "losses.svg"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"artifacts", "config", "metrics", "run_name"}
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
    assert payload["artifacts"]["training_history"] == {
        "combined": str(tmp_path / "training_history" / "combined.csv")
    }
    assert payload["artifacts"]["training_plots"] == {
        "losses": str(tmp_path / "training_history" / "plots" / "losses.svg")
    }
    assert payload["artifacts"]["config"] == str(tmp_path / "configs.json")


def test_config_json_is_saved_as_separate_run_artifact(tmp_path) -> None:
    runner = ExperimentRunner(
        ExperimentConfig(
            splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
        )
    )

    path = runner._save_config(config_dict={"seed": 42}, output_dir=tmp_path)

    assert path == tmp_path / "configs.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"seed": 42}


def test_window_metrics_are_keyed_by_window_index() -> None:
    metrics = _window_metrics_by_index(
        [
            {"index": 3, "label": "window_003", "val": {"mae": 0.1}, "test": {"mae": 0.2}},
        ]
    )

    assert metrics == {"3": {"val": {"mae": 0.1}, "test": {"mae": 0.2}}}


def test_training_history_files_are_combined_for_plotting(tmp_path) -> None:
    first = tmp_path / "window_000.csv"
    second = tmp_path / "window_001.csv"
    output = tmp_path / "combined_epoch_history.csv"
    pd.DataFrame(
        {
            "window_label": ["window_000"],
            "epoch": [0],
            "event": ["validation_epoch_end"],
            "val/prediction_loss": [0.2],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "window_label": ["window_001"],
            "epoch": [0],
            "event": ["validation_epoch_end"],
            "val/prediction_loss": [0.1],
        }
    ).to_csv(second, index=False)

    path = combine_history_files([first, second], output)

    assert path == output
    combined = pd.read_csv(output)
    assert combined["window_label"].tolist() == ["window_000", "window_001"]
    assert combined["val/prediction_loss"].tolist() == [0.2, 0.1]
    assert output.with_suffix(".json").exists()


def test_training_history_plots_are_saved_next_to_combined_history(tmp_path) -> None:
    runner = ExperimentRunner(
        ExperimentConfig(
            splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
        )
    )
    history_path = tmp_path / "training_history" / "combined_epoch_history.csv"
    history_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "window_label": ["window_000", "window_000"],
            "event": ["train_epoch_end", "validation_epoch_end"],
            "epoch": [0, 0],
            "train/total_loss": [0.5, 0.5],
            "val/prediction_loss": [None, 0.4],
        }
    ).to_csv(history_path, index=False)

    paths = runner._save_training_history_plots(history_path)

    assert paths["losses"] == tmp_path / "training_history" / "plots" / "loss_history.svg"
    assert paths["losses"].exists()
