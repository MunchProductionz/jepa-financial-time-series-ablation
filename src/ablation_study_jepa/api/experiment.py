"""Top-level experiment orchestration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ablation_study_jepa.builders.data import build_data
from ablation_study_jepa.builders.datasets import build_datasets
from ablation_study_jepa.builders.model import build_model_bundle
from ablation_study_jepa.builders.trainer import build_data_module, build_trainer
from ablation_study_jepa.config.loader import load_config
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.evaluation.metrics import compute_metrics
from ablation_study_jepa.evaluation.predictions import (
    collect_predictions,
    make_prediction_run_dir,
    save_predictions,
)
from ablation_study_jepa.training.lightning_module import ReturnPredictionLightningModule


@dataclass
class ExperimentResult:
    run_name: str | None
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    prediction_paths: dict[str, Path]
    metrics_path: Path


def run_experiment(config_path: str | Path) -> ExperimentResult:
    config = load_config(config_path)
    return ExperimentRunner(config).run()


class ExperimentRunner:
    """Run the minimum reproducible stock-return experiment pipeline."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> ExperimentResult:
        seed_everything(self.config.seed)
        prepared = build_data(self.config)
        datasets = build_datasets(self.config, prepared.scaled_panel, prepared.splits)
        if len(datasets.train) == 0:
            raise RuntimeError("Training dataset is empty after windowing and leakage filters")
        if len(datasets.val) == 0:
            raise RuntimeError("Validation dataset is empty after windowing and leakage filters")

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
        trainer = build_trainer(self.config)
        trainer.fit(lightning_module, datamodule=data_module)
        if len(datasets.test) > 0:
            trainer.test(lightning_module, datamodule=data_module)

        val_predictions = collect_predictions(model_bundle.model, data_module.val_dataloader())
        test_predictions = collect_predictions(model_bundle.model, data_module.test_dataloader())
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

        prediction_paths: dict[str, Path] = {}
        config_dict = self.config.model_dump(mode="json")
        artifact_dir = make_prediction_run_dir(
            self.config.evaluation.predictions_dir,
            self.config.run_name,
            config_dict,
            tags=self.config.logging.wandb.tags,
        )
        if self.config.evaluation.save_predictions:
            prediction_paths["val"] = save_predictions(
                val_predictions,
                artifact_dir,
                "val",
            )
            prediction_paths["test"] = save_predictions(
                test_predictions,
                artifact_dir,
                "test",
            )
        metrics_path = self._save_metrics(val_metrics, test_metrics, config_dict, artifact_dir)
        print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2, sort_keys=True))
        return ExperimentResult(
            run_name=self.config.run_name,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            prediction_paths=prediction_paths,
            metrics_path=metrics_path,
        )

    def _save_metrics(
        self,
        val_metrics: dict[str, float],
        test_metrics: dict[str, float],
        config_dict: dict[str, Any],
        output_dir: Path,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "metrics.json"
        payload = {
            "run_name": self.config.run_name,
            "val": val_metrics,
            "test": test_metrics,
            "config": config_dict,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        return path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
