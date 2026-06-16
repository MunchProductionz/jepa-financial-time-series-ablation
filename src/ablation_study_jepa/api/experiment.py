"""Top-level experiment orchestration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from ablation_study_jepa.builders.data import build_feature_panel, scale_panel_for_splits
from ablation_study_jepa.builders.datasets import build_datasets
from ablation_study_jepa.builders.model import build_model_bundle
from ablation_study_jepa.builders.trainer import build_data_module, build_trainer
from ablation_study_jepa.builders.windows import (
    ExperimentWindow,
    build_experiment_windows,
    filter_panel_to_window,
)
from ablation_study_jepa.config.loader import load_config
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.evaluation.analysis_artifacts import (
    collect_code_provenance,
    collect_data_provenance,
    save_analysis_artifacts,
    save_run_status,
    update_run_manifests,
    utc_now,
)
from ablation_study_jepa.evaluation.metrics import compute_metrics
from ablation_study_jepa.evaluation.predictions import (
    PREDICTION_COLUMNS,
    collect_predictions,
    make_prediction_run_dir,
    save_predictions,
)
from ablation_study_jepa.evaluation.training_plots import plot_training_history
from ablation_study_jepa.training.history import (
    combined_history_file_path,
    combine_history_files,
)
from ablation_study_jepa.training.lightning_module import ReturnPredictionLightningModule


@dataclass
class ExperimentResult:
    run_name: str | None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    prediction_paths: dict[str, Path]
    metrics_path: Path
    window_metrics: list[dict[str, Any]] = field(default_factory=list)
    training_history_paths: dict[str, Path] = field(default_factory=dict)
    analysis_paths: dict[str, Path] = field(default_factory=dict)
    manifest_paths: dict[str, Path] = field(default_factory=dict)
    status_path: Path | None = None


def run_experiment(config_path: str | Path) -> ExperimentResult:
    config = load_config(config_path)
    return ExperimentRunner(config).run()


class ExperimentRunner:
    """Run the minimum reproducible stock-return experiment pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        seed_everything(self.config.seed)
        run_started_at = utc_now()
        run_timer = perf_counter()
        prediction_paths: dict[str, Path] = {}
        config_dict = self.config.model_dump(mode="json")
        artifact_dir = make_prediction_run_dir(
            self.config.evaluation.predictions_dir,
            self.config.run_name,
            config_dict,
            tags=self.config.logging.wandb.tags,
        )
        config_path = self._save_config(config_dict=config_dict, output_dir=artifact_dir)
        code_provenance = collect_code_provenance()
        data_provenance: dict[str, Any] = {}
        training_history_paths: dict[str, Path] = {}
        training_plot_paths: dict[str, Path] = {}
        manifest_paths: dict[str, Path] = {}
        status_path = save_run_status(
            output_dir=artifact_dir,
            status="running",
            config=self.config,
            config_dict=config_dict,
            started_at=run_started_at,
            artifacts={"config": str(config_path)},
        )
        manifest_paths = update_run_manifests(
            predictions_dir=self.config.evaluation.predictions_dir,
            output_dir=artifact_dir,
            config=self.config,
            config_dict=config_dict,
            status="running",
            started_at=run_started_at,
            artifact_paths={"config": config_path, "run_status": status_path},
            code_provenance=code_provenance,
        )

        try:
            return self._run_with_artifacts(
                artifact_dir=artifact_dir,
                config_dict=config_dict,
                code_provenance=code_provenance,
                data_provenance=data_provenance,
                prediction_paths=prediction_paths,
                training_history_paths=training_history_paths,
                training_plot_paths=training_plot_paths,
                manifest_paths=manifest_paths,
                status_path=status_path,
                run_started_at=run_started_at,
                run_timer=run_timer,
            )
        except Exception as exc:
            run_finished_at = utc_now()
            elapsed_seconds = perf_counter() - run_timer
            error = f"{type(exc).__name__}: {exc}"
            status_path = save_run_status(
                output_dir=artifact_dir,
                status="failed",
                config=self.config,
                config_dict=self.config.model_dump(mode="json"),
                started_at=run_started_at,
                finished_at=run_finished_at,
                elapsed_seconds=elapsed_seconds,
                error=error,
                artifacts={"config": str(config_path)},
            )
            update_run_manifests(
                predictions_dir=self.config.evaluation.predictions_dir,
                output_dir=artifact_dir,
                config=self.config,
                config_dict=self.config.model_dump(mode="json"),
                status="failed",
                started_at=run_started_at,
                finished_at=run_finished_at,
                elapsed_seconds=elapsed_seconds,
                artifact_paths={"config": config_path, "run_status": status_path},
                data_provenance=data_provenance,
                code_provenance=code_provenance,
                error=error,
            )
            raise

    def _run_with_artifacts(
        self,
        *,
        artifact_dir: Path,
        config_dict: dict[str, Any],
        code_provenance: dict[str, Any],
        data_provenance: dict[str, Any],
        prediction_paths: dict[str, Path],
        training_history_paths: dict[str, Path],
        training_plot_paths: dict[str, Path],
        manifest_paths: dict[str, Path],
        status_path: Path,
        run_started_at: str,
        run_timer: float,
    ) -> ExperimentResult:
        prepared = build_feature_panel(self.config)
        window_plan = build_experiment_windows(self.config, prepared.feature_panel)
        config_dict = self.config.model_dump(mode="json")
        config_path = self._save_config(config_dict=config_dict, output_dir=artifact_dir)
        data_provenance.update(
            collect_data_provenance(
                config=self.config,
                raw_panel=prepared.raw_panel,
                feature_panel=prepared.feature_panel,
                window_plan=window_plan,
            )
        )
        print(
            json.dumps(
                {
                    "sliding_window": window_plan.sliding_enabled,
                    "windows": len(window_plan.windows),
                    "dropped_incomplete_windows": window_plan.dropped_incomplete_windows,
                },
                sort_keys=True,
            )
        )
        _log_experiment_progress(
            "window plan",
            sliding_window=window_plan.sliding_enabled,
            windows=len(window_plan.windows),
            dropped_incomplete_windows=window_plan.dropped_incomplete_windows,
            output_dir=str(artifact_dir),
        )

        window_results: list[dict[str, Any]] = []
        all_val_predictions: list[pd.DataFrame] = []
        all_test_predictions: list[pd.DataFrame] = []
        for window in window_plan.windows:
            result = self._run_window(
                window,
                prepared.feature_panel,
                total_windows=len(window_plan.windows),
                artifact_dir=artifact_dir,
            )
            window_results.append(result["metrics"])
            all_val_predictions.append(result["val_predictions"])
            all_test_predictions.append(result["test_predictions"])
            if result["training_history_path"] is not None:
                training_history_paths[window.label] = result["training_history_path"]

        val_metrics = _average_metric_dicts(
            [result["val"] for result in window_results],
            self.config.evaluation.metrics,
        )
        test_metrics = _average_metric_dicts(
            [result["test"] for result in window_results],
            self.config.evaluation.metrics,
        )
        combined_val_predictions = _concat_predictions(all_val_predictions)
        combined_test_predictions = _concat_predictions(all_test_predictions)
        if self.config.evaluation.save_predictions:
            _log_experiment_progress(
                "saving combined predictions",
                val_rows=len(combined_val_predictions),
                test_rows=len(combined_test_predictions),
                output_dir=str(artifact_dir),
            )
            prediction_paths["val"] = save_predictions(
                combined_val_predictions,
                artifact_dir,
                "val",
            )
            prediction_paths["test"] = save_predictions(
                combined_test_predictions,
                artifact_dir,
                "test",
            )
        combined_training_history_path = self._save_combined_training_history(
            output_dir=artifact_dir,
            history_paths=list(training_history_paths.values()),
        )
        if combined_training_history_path is not None:
            training_history_paths["combined"] = combined_training_history_path
            training_plot_paths = self._save_training_history_plots(combined_training_history_path)

        run_finished_at = utc_now()
        elapsed_seconds = perf_counter() - run_timer
        analysis_paths = save_analysis_artifacts(
            output_dir=artifact_dir,
            config=self.config,
            config_dict=config_dict,
            val_predictions=combined_val_predictions,
            test_predictions=combined_test_predictions,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            window_metrics=window_results,
            data_provenance=data_provenance,
            code_provenance=code_provenance,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            elapsed_seconds=elapsed_seconds,
            training_history_path=training_history_paths.get("combined"),
        )
        metrics_path = self._save_metrics(
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            window_metrics=window_results,
            config_dict=config_dict,
            output_dir=artifact_dir,
            config_path=config_path,
            training_history_paths=training_history_paths,
            training_plot_paths=training_plot_paths,
            analysis_paths=analysis_paths,
            manifest_paths=manifest_paths,
            status_path=status_path,
        )
        artifact_paths = {
            "config": config_path,
            "metrics": metrics_path,
            "run_status": status_path,
            **{f"prediction_{name}": path for name, path in prediction_paths.items()},
            **{f"training_history_{name}": path for name, path in training_history_paths.items()},
            **{f"training_plot_{name}": path for name, path in training_plot_paths.items()},
            **{f"analysis_{name}": path for name, path in analysis_paths.items()},
        }
        status_path = save_run_status(
            output_dir=artifact_dir,
            status="completed",
            config=self.config,
            config_dict=config_dict,
            started_at=run_started_at,
            finished_at=run_finished_at,
            elapsed_seconds=elapsed_seconds,
            metrics={"val": val_metrics, "test": test_metrics},
            artifacts={name: str(path) for name, path in artifact_paths.items()},
        )
        artifact_paths["run_status"] = status_path
        manifest_paths = update_run_manifests(
            predictions_dir=self.config.evaluation.predictions_dir,
            output_dir=artifact_dir,
            config=self.config,
            config_dict=config_dict,
            status="completed",
            started_at=run_started_at,
            finished_at=run_finished_at,
            elapsed_seconds=elapsed_seconds,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            artifact_paths=artifact_paths,
            data_provenance=data_provenance,
            code_provenance=code_provenance,
        )
        print(
            json.dumps(
                {
                    "metrics": {
                        "total": {"val": val_metrics, "test": test_metrics},
                        "windows": _window_metrics_by_index(window_results),
                    }
                },
                indent=2,
                sort_keys=True,
            )
        )
        _log_experiment_progress(
            "completed run",
            metrics_path=str(metrics_path),
            val_predictions=str(prediction_paths.get("val", "")),
            test_predictions=str(prediction_paths.get("test", "")),
            training_history=str(training_history_paths.get("combined", "")),
            training_plots=str(training_plot_paths),
            analysis=str(analysis_paths),
            manifests=str(manifest_paths),
        )
        _log_final_metrics_to_wandb(
            enabled=self.config.logging.wandb.enabled,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
        )
        return ExperimentResult(
            run_name=self.config.run_name,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            prediction_paths=prediction_paths,
            metrics_path=metrics_path,
            window_metrics=window_results,
            training_history_paths=training_history_paths,
            analysis_paths=analysis_paths,
            manifest_paths=manifest_paths,
            status_path=status_path,
        )

    def _run_window(
        self,
        window: ExperimentWindow,
        feature_panel: pd.DataFrame,
        total_windows: int,
        artifact_dir: Path,
    ) -> dict[str, Any]:
        window_started_at = utc_now()
        window_timer = perf_counter()
        seed_everything(self.config.seed)
        _log_window_progress(
            window,
            total_windows,
            "start",
            window_start=window.start.strftime("%Y-%m-%d"),
            window_end=window.end.strftime("%Y-%m-%d"),
            train_end=window.splits["train"].end.strftime("%Y-%m-%d"),
            val_end=window.splits["val"].end.strftime("%Y-%m-%d"),
            test_end=window.splits["test"].end.strftime("%Y-%m-%d"),
        )
        window_panel = filter_panel_to_window(
            feature_panel,
            date_column=self.config.data.date_column,
            window=window,
        )
        scaled_panel, _ = scale_panel_for_splits(self.config, window_panel, window.splits)
        datasets = build_datasets(self.config, scaled_panel, window.splits)
        if len(datasets.train) == 0:
            raise RuntimeError(
                f"{window.label} training dataset is empty after windowing and leakage filters"
            )
        if len(datasets.val) == 0:
            raise RuntimeError(
                f"{window.label} validation dataset is empty after windowing and leakage filters"
            )
        _log_window_progress(
            window,
            total_windows,
            "datasets ready",
            train_samples=len(datasets.train),
            val_samples=len(datasets.val),
            test_samples=len(datasets.test),
            panel_rows=len(window_panel),
        )

        model_bundle = build_model_bundle(
            self.config,
            input_dim=len(self.config.data.feature_columns),
            static_input_dim=len(self.config.data.static_feature_columns),
        )
        data_module = build_data_module(
            self.config,
            train_dataset=datasets.train,
            val_dataset=datasets.val,
            test_dataset=datasets.test,
        )
        lightning_module = ReturnPredictionLightningModule(
            model=model_bundle.model,
            criterion=model_bundle.criterion,
            jepa_module=model_bundle.jepa,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            lambda_jepa=self.config.jepa.global_weight,
        )
        trainer = build_trainer(
            self.config,
            output_dir=artifact_dir,
            window_label=window.label,
        )
        _log_window_progress(
            window,
            total_windows,
            "training start",
            max_epochs=self.config.training.max_epochs,
            batch_size=self.config.dataset.batch_size,
        )
        trainer.fit(lightning_module, datamodule=data_module)
        _log_window_progress(window, total_windows, "training complete")
        if len(datasets.test) > 0:
            _log_window_progress(window, total_windows, "testing start")
            trainer.test(lightning_module, datamodule=data_module)
            _log_window_progress(window, total_windows, "testing complete")

        _log_window_progress(window, total_windows, "collecting predictions")
        val_predictions = _annotate_predictions(
            collect_predictions(model_bundle.model, data_module.val_dataloader()),
            window,
        )
        test_predictions = _annotate_predictions(
            collect_predictions(model_bundle.model, data_module.test_dataloader()),
            window,
        )
        val_metrics = _compute_prediction_metrics(
            val_predictions,
            self.config.evaluation.metrics,
            split="val",
            require_nonempty=True,
        )
        test_metrics = _compute_prediction_metrics(
            test_predictions,
            self.config.evaluation.metrics,
            split="test",
            require_nonempty=False,
        )

        metrics = {
            **window.to_dict(),
            "val": val_metrics,
            "test": test_metrics,
            "started_at": window_started_at,
            "finished_at": utc_now(),
            "elapsed_seconds": perf_counter() - window_timer,
        }
        _log_window_progress(
            window,
            total_windows,
            "completed",
            val_metrics=_format_metric_summary(val_metrics),
            test_metrics=_format_metric_summary(test_metrics),
        )
        print(
            json.dumps(
                {window.label: {"val": val_metrics, "test": test_metrics}},
                sort_keys=True,
            )
        )
        return {
            "metrics": metrics,
            "val_predictions": val_predictions,
            "test_predictions": test_predictions,
            "training_history_path": self._window_training_history_path(artifact_dir, window),
        }

    def _save_metrics(
        self,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        window_metrics: list[dict[str, Any]],
        config_dict: dict[str, Any],
        output_dir: Path,
        config_path: Path | None = None,
        training_history_paths: dict[str, Path] | None = None,
        training_plot_paths: dict[str, Path] | None = None,
        analysis_paths: dict[str, Path] | None = None,
        manifest_paths: dict[str, Path] | None = None,
        status_path: Path | None = None,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.json"
        payload = {
            "run_name": self.config.run_name,
            "metrics": {
                "total": {
                    "val": val_metrics,
                    "test": test_metrics,
                },
                "windows": _window_metrics_by_index(window_metrics),
            },
            "config": config_dict,
            "artifacts": {
                "config": str(config_path) if config_path is not None else None,
                "training_history": {
                    name: str(path)
                    for name, path in (training_history_paths or {}).items()
                },
                "training_plots": {
                    name: str(path)
                    for name, path in (training_plot_paths or {}).items()
                },
                "analysis": {
                    name: str(path)
                    for name, path in (analysis_paths or {}).items()
                },
                "manifests": {
                    name: str(path)
                    for name, path in (manifest_paths or {}).items()
                },
                "run_status": str(status_path) if status_path is not None else None,
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def _save_training_history_plots(self, history_path: Path) -> dict[str, Path]:
        if not self.config.logging.training_history.enabled:
            return {}
        try:
            return plot_training_history(history_path)
        except ValueError as exc:
            _log_experiment_progress(
                "skipping training-history plots",
                reason=str(exc),
                history=str(history_path),
            )
            return {}

    def _save_config(self, config_dict: dict[str, Any], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "configs.json"
        path.write_text(
            json.dumps(config_dict, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        return path

    def _save_combined_training_history(
        self,
        output_dir: Path,
        history_paths: list[Path],
    ) -> Path | None:
        if not self.config.logging.training_history.enabled:
            return None
        if not self.config.logging.training_history.save_epoch_metrics:
            return None
        return combine_history_files(
            history_paths,
            combined_history_file_path(
                output_dir,
                self.config.logging.training_history.directory_name,
            ),
        )

    def _window_training_history_path(
        self,
        output_dir: Path,
        window: ExperimentWindow,
    ) -> Path | None:
        if not self.config.logging.training_history.enabled:
            return None
        if not self.config.logging.training_history.save_epoch_metrics:
            return None
        path = output_dir / self.config.logging.training_history.directory_name / f"{window.label}.csv"
        return path if path.exists() else None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _log_experiment_progress(stage: str, **fields: Any) -> None:
    _log_progress("experiment", stage, **fields)


def _log_window_progress(
    window: ExperimentWindow,
    total_windows: int,
    stage: str,
    **fields: Any,
) -> None:
    label = f"window {window.index + 1:03d}/{total_windows:03d}"
    _log_progress(label, stage, **fields)


def _log_progress(scope: str, stage: str, **fields: Any) -> None:
    details = " ".join(f"{key}={value}" for key, value in fields.items())
    suffix = f" {details}" if details else ""
    print(f"[{scope}] {stage}{suffix}", flush=True)


def _format_metric_summary(metrics: dict[str, float]) -> str:
    parts = []
    for name, value in metrics.items():
        if pd.isna(value):
            parts.append(f"{name}=nan")
        else:
            parts.append(f"{name}={value:.6g}")
    return ",".join(parts)


def _compute_prediction_metrics(
    predictions: pd.DataFrame,
    metric_names: list[str],
    split: str,
    require_nonempty: bool,
) -> dict[str, float]:
    required_columns = {"y_true", "y_pred"}
    missing = sorted(required_columns.difference(predictions.columns))
    if missing:
        raise RuntimeError(f"{split} predictions are missing required columns: {missing}")
    if predictions.empty:
        if require_nonempty:
            raise RuntimeError(f"{split} predictions are empty")
        return {name: float("nan") for name in metric_names}
    return compute_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_pred"].to_numpy(),
        metric_names,
    )


def _annotate_predictions(predictions: pd.DataFrame, window: ExperimentWindow) -> pd.DataFrame:
    result = predictions.copy()
    result.insert(0, "window_end", window.end.strftime("%Y-%m-%d"))
    result.insert(0, "window_start", window.start.strftime("%Y-%m-%d"))
    result.insert(0, "window_index", window.index)
    return result


def _concat_predictions(predictions: list[pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        return pd.DataFrame({column: pd.Series(dtype=float) for column in PREDICTION_COLUMNS})
    return pd.concat(predictions, ignore_index=True, sort=False)


def _average_metric_dicts(
    metric_dicts: list[dict[str, float]],
    metric_names: list[str],
) -> dict[str, float]:
    averages: dict[str, float] = {}
    for name in metric_names:
        values = [
            float(metrics[name])
            for metrics in metric_dicts
            if name in metrics and pd.notna(metrics[name])
        ]
        averages[name] = float(np.mean(values)) if values else float("nan")
    return averages


def _window_metrics_by_index(window_metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(window["index"]): {
            "val": window["val"],
            "test": window["test"],
        }
        for window in window_metrics
    }


def _log_final_metrics_to_wandb(
    enabled: bool,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ModuleNotFoundError:
        return
    if wandb.run is None:
        return
    wandb.log(
        {
            **{f"val/{name}": value for name, value in val_metrics.items()},
            **{f"test/{name}": value for name, value in test_metrics.items()},
        }
    )
