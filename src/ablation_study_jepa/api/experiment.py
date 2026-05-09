"""Top-level experiment orchestration."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ablation_study_jepa.builders.data import build_data
from ablation_study_jepa.builders.datasets import build_datasets
from ablation_study_jepa.builders.model import build_model_bundle
from ablation_study_jepa.builders.trainer import build_data_module, build_trainer
from ablation_study_jepa.config.loader import load_config
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.evaluation.metrics import compute_metrics
from ablation_study_jepa.evaluation.predictions import collect_predictions, save_predictions
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
        val_metrics = compute_metrics(
            val_predictions["y_true"].to_numpy(),
            val_predictions["y_pred"].to_numpy(),
            self.config.evaluation.metrics,
        )
        test_metrics = compute_metrics(
            test_predictions["y_true"].to_numpy(),
            test_predictions["y_pred"].to_numpy(),
            self.config.evaluation.metrics,
        )

        prediction_paths: dict[str, Path] = {}
        config_dict = self.config.model_dump(mode="json")
        if self.config.evaluation.save_predictions:
            prediction_paths["val"] = save_predictions(
                val_predictions,
                self.config.evaluation.predictions_dir,
                self.config.run_name,
                "val",
                config_dict,
            )
            prediction_paths["test"] = save_predictions(
                test_predictions,
                self.config.evaluation.predictions_dir,
                self.config.run_name,
                "test",
                config_dict,
            )
        metrics_path = self._save_metrics(val_metrics, test_metrics, config_dict)
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
    ) -> Path:
        output_dir = self.config.evaluation.predictions_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.config.run_name or 'run'}_metrics.json"
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

