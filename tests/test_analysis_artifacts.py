import json

import pandas as pd
import pytest

from ablation_study_jepa.builders.windows import ExperimentWindow, WindowPlan
from ablation_study_jepa.config.schemas import (
    EvaluationConfig,
    ExperimentConfig,
    LoggingConfig,
    SplitsConfig,
    WandbConfig,
)
from ablation_study_jepa.data.preprocessing import DateSplit
from ablation_study_jepa.evaluation.analysis_artifacts import (
    collect_data_provenance,
    prediction_diagnostics_frame,
    save_analysis_artifacts,
    update_run_manifests,
)


def test_analysis_artifacts_write_reusable_tables(tmp_path) -> None:
    config = ExperimentConfig(
        run_name="analysis_test",
        splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
    )
    history_path = tmp_path / "training_history" / "combined_epoch_history.csv"
    history_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "window_label": ["window_000", "window_000"],
            "event": ["validation_epoch_end", "train_epoch_end"],
            "epoch": [0, 0],
            "global_step": [4, 4],
            "val/prediction_loss": [0.2, None],
            "train/total_loss_epoch": [None, 0.4],
        }
    ).to_csv(history_path, index=False)
    checkpoint_dir = tmp_path / "checkpoints" / "window_000"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "epoch=000.ckpt").write_text("checkpoint", encoding="utf-8")

    paths = save_analysis_artifacts(
        output_dir=tmp_path,
        config=config,
        config_dict=config.model_dump(mode="json"),
        val_predictions=_prediction_rows(),
        test_predictions=_prediction_rows(split_offset=0.01),
        val_metrics={"mse": 0.1},
        test_metrics={"mse": 0.2},
        window_metrics=[
            {
                "index": 0,
                "label": "window_000",
                "start": "2020-01-01",
                "end": "2020-01-03",
                "splits": {
                    "train": {"start": None, "end": "2020-01-01"},
                    "val": {"start": "2020-01-01", "end": "2020-01-02"},
                    "test": {"start": "2020-01-02", "end": "2020-01-03"},
                },
                "val": {"mse": 0.1},
                "test": {"mse": 0.2},
                "started_at": "2026-06-16T00:00:00Z",
                "finished_at": "2026-06-16T00:00:05Z",
                "elapsed_seconds": 5.0,
            }
        ],
        data_provenance={"source_hash": "abc", "feature_panel": {"rows": 6}},
        code_provenance={"git_commit": "commit", "dirty": False},
        run_started_at="2026-06-16T00:00:00Z",
        run_finished_at="2026-06-16T00:00:10Z",
        elapsed_seconds=10.0,
        training_history_path=history_path,
        model_summary={
            "parameter_count": 1234,
            "trainable_parameter_count": 1200,
            "non_trainable_parameter_count": 34,
            "base_parameter_count": 1000,
            "jepa_parameter_count": 234,
            "module_counts": {
                "linear": 12,
                "lstm_modules": 1,
                "lstm_layers": 2,
                "multihead_attention": 4,
            },
            "architecture": {
                "transformer_block_count": 4,
                "mlp_linear_layer_count": 8,
                "jepa_predictor_linear_layer_count": 2,
                "input_dim": 16,
                "static_input_dim": 4,
                "hidden_dim": 32,
            },
        },
    )

    assert {
        "config_summary",
        "window_metrics",
        "per_date_metrics",
        "per_asset_metrics",
        "prediction_diagnostics",
        "portfolio_returns",
        "portfolio_memberships",
        "training_summary",
        "provenance",
    }.issubset(paths)
    diagnostics = pd.read_csv(paths["prediction_diagnostics"])
    assert {"residual", "abs_error", "prediction_rank", "prediction_quantile"}.issubset(
        diagnostics.columns
    )
    assert diagnostics["split"].unique().tolist() == ["val", "test"]

    per_date = pd.read_csv(paths["per_date_metrics"])
    assert {"spearman_rank_ic", "top_bottom_quantile_spread_return"}.issubset(
        per_date.columns
    )
    portfolio = pd.read_csv(paths["portfolio_returns"])
    assert {"long_short_return", "cumulative_long_short_return", "top_turnover"}.issubset(
        portfolio.columns
    )
    training = pd.read_csv(paths["training_summary"])
    assert training.loc[0, "best_epoch"] == pytest.approx(0)
    assert training.loc[0, "checkpoint_path"].endswith("epoch=000.ckpt")

    provenance = json.loads(paths["provenance"].read_text(encoding="utf-8"))
    assert provenance["run"]["metrics"]["val"] == {"mse": 0.1}
    assert provenance["run"]["model_summary"]["parameter_count"] == 1234
    config_summary = pd.read_csv(paths["config_summary"])
    assert config_summary.loc[0, "model_parameter_count"] == 1234
    assert config_summary.loc[0, "model_transformer_block_count"] == 4
    assert config_summary.loc[0, "model_lstm_layer_count"] == 2


def test_prediction_diagnostics_adds_cross_sectional_ranks() -> None:
    diagnostics = prediction_diagnostics_frame({"val": _prediction_rows(), "test": pd.DataFrame()})

    first_date = diagnostics.loc[diagnostics["anchor_date"] == "2020-01-01"]
    assert first_date["prediction_rank"].tolist() == [1.0, 2.0, 3.0]
    assert first_date["prediction_quantile"].tolist() == [2.0, 4.0, 5.0]


def test_run_manifests_are_upserted_and_grouped_by_study(tmp_path) -> None:
    config = ExperimentConfig(
        run_name="manifest_test",
        splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
        evaluation=EvaluationConfig(predictions_dir=tmp_path),
        logging=LoggingConfig(wandb=WandbConfig(group="study-a", tags=["baseline"])),
    )
    output_dir = tmp_path / "run-a"
    output_dir.mkdir()

    update_run_manifests(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        config=config,
        config_dict=config.model_dump(mode="json"),
        status="running",
        started_at="2026-06-16T00:00:00Z",
    )
    paths = update_run_manifests(
        predictions_dir=tmp_path,
        output_dir=output_dir,
        config=config,
        config_dict=config.model_dump(mode="json"),
        status="completed",
        started_at="2026-06-16T00:00:00Z",
        finished_at="2026-06-16T00:00:10Z",
        elapsed_seconds=10.0,
        val_metrics={"mse": 0.1},
        test_metrics={"mse": 0.2},
        artifact_paths={"metrics": output_dir / "metrics.json"},
        data_provenance={
            "source_hash": "abc",
            "feature_panel": {
                "rows": 10,
                "asset_count": 2,
                "date_start": "2020-01-01",
                "date_end": "2020-01-10",
            },
            "splits": {"window_count": 1},
        },
        code_provenance={"git_commit": "commit", "dirty": False},
        model_summary={
            "parameter_count": 4321,
            "trainable_parameter_count": 4300,
            "non_trainable_parameter_count": 21,
            "base_parameter_count": 4000,
            "jepa_parameter_count": 321,
            "module_counts": {"linear": 10, "lstm_modules": 1, "lstm_layers": 2},
            "architecture": {"transformer_block_count": 4},
        },
    )

    root = pd.read_csv(paths["runs_manifest"])
    assert len(root) == 1
    assert root.loc[0, "status"] == "completed"
    assert root.loc[0, "val_mse"] == pytest.approx(0.1)
    assert root.loc[0, "model_parameter_count"] == 4321
    assert root.loc[0, "artifact_metrics"] == str(output_dir / "metrics.json")

    grouped = pd.read_csv(paths["study_runs_manifest"])
    assert grouped["study_id"].tolist() == ["study-a"]
    assert (tmp_path / "studies" / "study_a" / "study_manifest.json").exists()


def test_data_provenance_hashes_configured_sources(tmp_path) -> None:
    price_path = tmp_path / "panel.csv"
    macro_path = tmp_path / "macro.csv"
    price_path.write_text("ticker,date,close\nAAA,2020-01-01,1.0\n", encoding="utf-8")
    macro_path.write_text("date,macro\n2020-01-01,2.0\n", encoding="utf-8")
    config = ExperimentConfig(
        splits=SplitsConfig(method="fraction", train=0.7, validation=0.2, test=0.1),
    )
    config.data.data_dir = price_path
    config.data.macro_data_path = macro_path
    panel = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        }
    )

    provenance = collect_data_provenance(
        config=config,
        raw_panel=panel,
        feature_panel=panel,
        window_plan=WindowPlan(
            windows=[
                ExperimentWindow(
                    index=0,
                    start=pd.Timestamp("2020-01-01"),
                    end=pd.Timestamp("2020-01-02"),
                    date_count=2,
                    train_date_count=1,
                    val_date_count=1,
                    test_date_count=0,
                    splits={
                        "train": DateSplit(
                            "train",
                            None,
                            pd.Timestamp("2020-01-01"),
                            pd.Timestamp("2020-01-01"),
                        ),
                        "val": DateSplit(
                            "val",
                            pd.Timestamp("2020-01-01"),
                            pd.Timestamp("2020-01-02"),
                            pd.Timestamp("2020-01-02"),
                        ),
                        "test": DateSplit(
                            "test",
                            pd.Timestamp("2020-01-02"),
                            pd.Timestamp("2020-01-02"),
                            pd.Timestamp("2020-01-02"),
                        ),
                    },
                )
            ],
            sliding_enabled=False,
        ),
    )

    assert provenance["source_hash"]
    assert [record["role"] for record in provenance["input_files"]] == ["price", "macro"]
    assert provenance["feature_panel"]["asset_count"] == 2


def _prediction_rows(split_offset: float = 0.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_index": [0, 0, 0, 0, 0, 0],
            "window_start": ["2020-01-01"] * 6,
            "window_end": ["2020-01-03"] * 6,
            "asset_id": ["AAA", "BBB", "CCC", "AAA", "BBB", "CCC"],
            "anchor_date": [
                "2020-01-01",
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
            ],
            "target_date": [
                "2020-01-02",
                "2020-01-02",
                "2020-01-02",
                "2020-01-03",
                "2020-01-03",
                "2020-01-03",
            ],
            "y_true": [0.01, 0.02, -0.01, 0.03, -0.02, 0.01],
            "y_pred": [
                -0.02 + split_offset,
                0.01 + split_offset,
                0.03 + split_offset,
                0.02 + split_offset,
                -0.01 + split_offset,
                0.04 + split_offset,
            ],
        }
    )
