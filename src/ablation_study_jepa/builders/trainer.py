"""Training object factories."""

from __future__ import annotations

from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.datasets.windowed import WindowedStockDataset
from ablation_study_jepa.training.data_module import StockDataModule
from ablation_study_jepa.training.trainer_factory import build_trainer


def build_data_module(
    config: ExperimentConfig,
    train_dataset: WindowedStockDataset,
    val_dataset: WindowedStockDataset,
    test_dataset: WindowedStockDataset,
) -> StockDataModule:
    return StockDataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
        pin_memory=config.dataset.pin_memory,
        drop_last=config.dataset.drop_last,
    )


__all__ = ["build_data_module", "build_trainer"]

