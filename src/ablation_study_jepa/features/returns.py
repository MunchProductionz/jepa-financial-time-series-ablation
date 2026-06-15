"""Feature and forward log-return target creation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_return_features(
    frame: pd.DataFrame,
    asset_id_column: str = "ticker",
    date_column: str = "date",
    price_column: str = "close",
    target_column: str = "target_return",
    target_horizon: int = 1,
    volume_column: str = "volume",
) -> pd.DataFrame:
    """Add lagged return features and a trading-day forward log-return target."""

    if target_horizon <= 0:
        raise ValueError("target_horizon must be a positive trading-day offset")
    if price_column not in frame.columns:
        raise ValueError(f"Missing price column: {price_column}")

    result = frame.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    result = result.sort_values([asset_id_column, date_column]).reset_index(drop=True)

    grouped = result.groupby(asset_id_column, group_keys=False, sort=False)
    close = grouped[price_column]

    result["return_1d"] = close.pct_change(1)
    result["return_5d"] = close.pct_change(5)
    result["return_20d"] = close.pct_change(20)
    result["volatility_20d"] = grouped["return_1d"].transform(
        lambda values: values.rolling(20, min_periods=5).std()
    )

    if volume_column in result.columns:
        rolling_mean = grouped[volume_column].transform(lambda s: s.rolling(20, min_periods=5).mean())
        rolling_std = grouped[volume_column].transform(lambda s: s.rolling(20, min_periods=5).std())
        zscore = (result[volume_column] - rolling_mean) / rolling_std.replace(0.0, np.nan)
        result["volume_zscore"] = zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif "volume_zscore" not in result.columns:
        result["volume_zscore"] = 0.0

    future_close = close.shift(-target_horizon)
    result[target_column] = np.log(future_close / result[price_column])
    result[f"{target_column}_date"] = grouped[date_column].shift(-target_horizon)
    result[f"{target_column}_position"] = grouped.cumcount() + target_horizon
    return result
