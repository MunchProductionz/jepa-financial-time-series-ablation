"""Config-driven data loading and preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ablation_study_jepa.builders.features import build_features
from ablation_study_jepa.config.schemas import ExperimentConfig
from ablation_study_jepa.data.cleaning import clean_price_panel
from ablation_study_jepa.data.preprocessing import (
    PanelScaler,
    filter_anchor_rows,
    make_date_splits,
    make_fraction_splits,
)
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
    cleaned = add_sector_one_hot_features(
        cleaned,
        sector_column=config.data.sector_column,
        static_feature_columns=config.data.static_feature_columns,
        prefix=config.data.sector_one_hot_prefix,
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
    missing_static = [
        column for column in config.data.static_feature_columns if column not in cleaned.columns
    ]
    if missing_static:
        raise ValueError(f"Configured static feature columns were not created/found: {missing_static}")

    featured = build_features(cleaned, config)
    missing_features = [column for column in config.data.feature_columns if column not in featured.columns]
    if missing_features:
        raise ValueError(f"Configured feature columns were not created/found: {missing_features}")

    if config.splits.method == "fraction":
        splits = make_fraction_splits(
            featured,
            date_column=config.data.date_column,
            train_fraction=config.splits.train,
            validation_fraction=config.splits.validation,
            test_fraction=config.splits.test,
        )
    else:
        if (
            config.splits.train_end is None
            or config.splits.val_end is None
            or config.splits.test_end is None
        ):
            raise ValueError("date splits require train_end, val_end, and test_end")
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


def add_sector_one_hot_features(
    frame: pd.DataFrame,
    sector_column: str | None,
    static_feature_columns: list[str],
    prefix: str = "sector_",
) -> pd.DataFrame:
    """Create configured static sector one-hot columns.

    The expected one-hot columns are driven by config rather than inferred so the
    model input shape is stable across train/validation/test and across runs.
    """

    sector_columns = [column for column in static_feature_columns if column.startswith(prefix)]
    if not sector_columns:
        return frame

    result = frame.copy()
    if sector_column is not None and sector_column in result.columns:
        sectors = result[sector_column].fillna("unknown").astype(str).str.strip().str.lower()
    else:
        sectors = pd.Series("unknown", index=result.index)

    known_categories = {column.removeprefix(prefix) for column in sector_columns}
    unknown_column = f"{prefix}unknown" if "unknown" in known_categories else None
    matched = pd.Series(False, index=result.index)
    for column in sector_columns:
        category = column.removeprefix(prefix)
        if category == "unknown":
            continue
        values = sectors == category
        result[column] = values.astype(float)
        matched |= values
    if unknown_column is not None:
        result[unknown_column] = (~matched).astype(float)
    return result
