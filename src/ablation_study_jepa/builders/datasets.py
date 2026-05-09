"""Dataset factory layer."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.data.preprocessing import DateSplit
from ablation_study_jepa.datasets.windowed import WindowedStockDataset


@dataclass
class DatasetBundle:
    train: WindowedStockDataset
    val: WindowedStockDataset
    test: WindowedStockDataset


def build_datasets(
    config: ExperimentConfig,
    panel: pd.DataFrame,
    splits: dict[str, DateSplit],
) -> DatasetBundle:
    horizons = config.jepa.horizons if config.jepa.enabled else []
    include_future = config.dataset.include_future_window and config.jepa.enabled
    kwargs = dict(
        frame=panel,
        feature_columns=config.data.feature_columns,
        target_column=config.data.target_column,
        asset_id_column=config.data.asset_id_column,
        date_column=config.data.date_column,
        lookback=config.dataset.lookback,
        jepa_horizons=horizons,
        include_future_window=include_future,
        static_feature_columns=config.data.static_feature_columns,
        sector_column=config.data.sector_column,
    )
    return DatasetBundle(
        train=WindowedStockDataset(
            **kwargs,
            split_start=splits["train"].start,
            split_end=splits["train"].end,
            max_target_date=splits["train"].target_end,
            include_start=True,
        ),
        val=WindowedStockDataset(
            **kwargs,
            split_start=splits["val"].start,
            split_end=splits["val"].end,
            max_target_date=splits["val"].target_end,
            include_start=False,
        ),
        test=WindowedStockDataset(
            **kwargs,
            split_start=splits["test"].start,
            split_end=splits["test"].end,
            max_target_date=splits["test"].target_end,
            include_start=False,
        ),
    )

