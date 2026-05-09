"""Basic panel cleaning helpers."""

from __future__ import annotations

import pandas as pd


def clean_price_panel(
    frame: pd.DataFrame,
    asset_id_column: str = "ticker",
    date_column: str = "date",
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    required = [asset_id_column, date_column] + list(required_columns or [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = frame.copy()
    cleaned[date_column] = pd.to_datetime(cleaned[date_column])
    cleaned = cleaned.dropna(subset=[asset_id_column, date_column])
    cleaned = cleaned.sort_values([asset_id_column, date_column])
    cleaned = cleaned.drop_duplicates([asset_id_column, date_column], keep="last")
    return cleaned.reset_index(drop=True)

