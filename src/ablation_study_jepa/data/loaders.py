"""Panel data loaders for stock OHLCV experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_price_panel(
    data_dir: str | Path,
    tickers: Iterable[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    date_column: str = "date",
    asset_id_column: str = "ticker",
    **_: object,
) -> pd.DataFrame:
    """Load a panel from ``panel.csv`` or one CSV per ticker.

    The loader intentionally does not infer calendar horizons. Downstream datasets
    use row positions within each asset's sorted trading-day history.
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

    frame[date_column] = pd.to_datetime(frame[date_column])
    if tickers is not None:
        frame = frame[frame[asset_id_column].isin(list(tickers))]
    if start_date is not None:
        frame = frame[frame[date_column] >= pd.Timestamp(start_date)]
    if end_date is not None:
        frame = frame[frame[date_column] <= pd.Timestamp(end_date)]
    return frame.sort_values([asset_id_column, date_column]).reset_index(drop=True)

