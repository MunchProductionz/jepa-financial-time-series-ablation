"""Config-driven data loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ablation_study_jepa.builders.features import build_features
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.data.cleaning import clean_price_panel
from ablation_study_jepa.data.preprocessing import PanelScaler, filter_anchor_rows, make_date_splits
from ablation_study_jepa.utils.instantiate import locate


@dataclass
class PreparedData:
    raw_panel: pd.DataFrame
    feature_panel: pd.DataFrame
    scaled_panel: pd.DataFrame
    scaler: PanelScaler
    splits: dict


def build_data(config: ExperimentConfig) -> PreparedData:
    loader = locate(config.data.loader)
    raw = loader(
        data_dir=config.data.data_dir,
        tickers=config.data.tickers,
        start_date=config.data.start_date,
        end_date=config.data.end_date,
        date_column=config.data.date_column,
        asset_id_column=config.data.asset_id_column,
        price_columns=config.data.fast_feature_columns,
        macro_data_path=config.data.macro_data_path,
        macro_date_column=config.data.macro_date_column,
        macro_feature_columns=config.data.macro_feature_columns,
        macro_missing=config.data.macro_missing,
    )
    cleaned = clean_price_panel(
        raw,
        asset_id_column=config.data.asset_id_column,
        date_column=config.data.date_column,
        required_columns=[config.data.price_column],
    )
    missing_fast = [
        column for column in config.data.fast_feature_columns if column not in cleaned.columns
    ]
    missing_slow = [
        column for column in config.data.slow_feature_columns if column not in cleaned.columns
    ]
    if missing_fast:
        raise ValueError(f"Configured fast feature columns were not loaded: {missing_fast}")
    if missing_slow:
        raise ValueError(f"Configured slow feature columns were not loaded: {missing_slow}")

    featured = build_features(cleaned, config)
    missing_features = [column for column in config.data.feature_columns if column not in featured.columns]
    if missing_features:
        raise ValueError(f"Configured feature columns were not created/found: {missing_features}")

    splits = make_date_splits(
        train_end=config.splits.train_end,
        val_end=config.splits.val_end,
        test_end=config.splits.test_end,
        train_start=config.splits.train_start,
        val_start=config.splits.val_start,
        test_start=config.splits.test_start,
    )
    train_rows = filter_anchor_rows(
        featured,
        date_column=config.data.date_column,
        split=splits["train"],
        include_start=True,
    )
    scaler = PanelScaler(config.data.scaler_type)
    if config.data.fit_scaler_on_train_only:
        scaler.fit(train_rows, config.data.feature_columns)
        scaled = scaler.transform(featured, config.data.feature_columns)
    else:
        scaled = scaler.fit_transform(featured, config.data.feature_columns)
    return PreparedData(
        raw_panel=cleaned,
        feature_panel=featured,
        scaled_panel=scaled,
        scaler=scaler,
        splits=splits,
    )
