"""Panel data loaders for stock OHLCV experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

import pandas as pd

PRICE_COLUMN_RENAMES = {
    "open": "Price Open",
    "high": "Price High",
    "low": "Price Low",
    "close": "Price Close",
    "adj_close": "Price Adj_Close",
    "volume": "Volume",
}

DEFAULT_PRICE_COLUMNS = [
    "Price Open",
    "Price High",
    "Price Low",
    "Price Close",
    "Price Adj_Close",
    "Volume",
]

DEFAULT_MACRO_FEATURE_COLUMNS = [
    "S&P 500",
    "FEDFUNDS",
    "GS1",
    "GS5",
    "GS10",
    "OILPRICEx",
    "S&P: indust",
    "S&P div yield",
    "S&P PE ratio",
]


def load_price_panel(
    data_dir: str | Path,
    tickers: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    date_column: str = "date",
    asset_id_column: str = "ticker",
    price_columns: Iterable[str] | None = None,
    price_column_renames: dict[str, str] | None = None,
    macro_data_path: str | Path | None = None,
    macro_date_column: str = "date",
    macro_feature_columns: Iterable[str] | None = None,
    macro_missing: Literal["error", "ignore"] = "ignore",
    **_: object,
) -> pd.DataFrame:
    """Load a price panel and optionally as-of merge monthly macro features.

    Price files can be stored as ``panel.csv`` or one CSV per ticker. Yahoo-style
    OHLCV columns are renamed to the configured experiment names by default.
    If macro data is provided, monthly values are backward as-of joined onto
    each daily price row by date. The loader intentionally does not infer
    prediction horizons; downstream datasets use row positions within each
    asset's sorted trading-day history.
    """

    data_dir = Path(data_dir)
    if data_dir.is_file():
        frame = pd.read_csv(data_dir)
    elif (data_dir / "panel.csv").exists():
        frame = pd.read_csv(data_dir / "panel.csv")
    else:
        selected = list(tickers or [])
        if not selected:
            selected = sorted(path.stem for path in data_dir.glob("*.csv"))
        frames = []
        for ticker in selected:
            path = data_dir / f"{ticker}.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing ticker file: {path}")
            ticker_frame = pd.read_csv(path)
            if asset_id_column not in ticker_frame.columns:
                ticker_frame[asset_id_column] = ticker
            frames.append(ticker_frame)
        if not frames:
            raise FileNotFoundError(f"No CSV data found in {data_dir}")
        frame = pd.concat(frames, ignore_index=True)

    frame = _normalize_price_columns(
        frame=frame,
        price_columns=list(price_columns or DEFAULT_PRICE_COLUMNS),
        price_column_renames=price_column_renames or PRICE_COLUMN_RENAMES,
        date_column=date_column,
        asset_id_column=asset_id_column,
    )
    frame[date_column] = pd.to_datetime(frame[date_column])
    if tickers is not None:
        frame = frame[frame[asset_id_column].isin(list(tickers))]
    if start_date is not None:
        frame = frame[frame[date_column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame[date_column] <= pd.Timestamp(end_date)]
    frame = frame.sort_values([asset_id_column, date_column]).reset_index(drop=True)

    if macro_data_path is not None:
        frame = merge_macro_features(
            price_panel=frame,
            macro_data_path=macro_data_path,
            macro_date_column=macro_date_column,
            macro_feature_columns=list(macro_feature_columns or DEFAULT_MACRO_FEATURE_COLUMNS),
            date_column=date_column,
            asset_id_column=asset_id_column,
            missing=macro_missing,
        )
    return frame


def _normalize_price_columns(
    frame: pd.DataFrame,
    price_columns: list[str],
    price_column_renames: dict[str, str],
    date_column: str,
    asset_id_column: str,
) -> pd.DataFrame:
    normalized = frame.rename(columns=price_column_renames).copy()
    if "Price Adj_Close" in price_columns and "Price Adj_Close" not in normalized.columns:
        if "Price Close" not in normalized.columns:
            raise ValueError("Cannot synthesize Price Adj_Close because Price Close is missing")
        normalized["Price Adj_Close"] = normalized["Price Close"]

    required = [asset_id_column, date_column, *price_columns]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Missing required price columns after renaming: {missing}")

    for column in price_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    ordered = _deduplicate([asset_id_column, date_column, *price_columns, *normalized.columns])
    return normalized[ordered]


def merge_macro_features(
    price_panel: pd.DataFrame,
    macro_data_path: str | Path,
    macro_feature_columns: Iterable[str],
    date_column: str = "date",
    asset_id_column: str = "ticker",
    macro_date_column: str = "date",
    missing: Literal["error", "ignore"] = "ignore",
) -> pd.DataFrame:
    """Backward as-of merge monthly macro columns into a daily price panel."""

    requested = _deduplicate([str(column) for column in macro_feature_columns])
    macro = pd.read_csv(macro_data_path)
    if macro_date_column not in macro.columns:
        raise ValueError(f"Missing macro date column: {macro_date_column}")

    missing_columns = [column for column in requested if column not in macro.columns]
    if missing_columns and missing == "error":
        raise ValueError(f"Missing macro columns in {macro_data_path}: {missing_columns}")
    available = [column for column in requested if column in macro.columns]
    if not available:
        return price_panel.sort_values([asset_id_column, date_column]).reset_index(drop=True)

    macro = macro[[macro_date_column, *available]].copy()
    macro[macro_date_column] = pd.to_datetime(macro[macro_date_column])
    for column in available:
        macro[column] = pd.to_numeric(macro[column], errors="coerce")
    macro = macro.sort_values(macro_date_column).drop_duplicates(macro_date_column, keep="last")

    panel = price_panel.copy()
    panel[date_column] = pd.to_datetime(panel[date_column])
    original_columns = list(panel.columns)
    merged = pd.merge_asof(
        panel.sort_values(date_column),
        macro,
        left_on=date_column,
        right_on=macro_date_column,
        direction="backward",
    )
    if macro_date_column != date_column and macro_date_column in merged.columns:
        merged = merged.drop(columns=[macro_date_column])

    return merged[[*original_columns, *available]].sort_values(
        [asset_id_column, date_column]
    ).reset_index(drop=True)


def _deduplicate(values: Iterable[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result
