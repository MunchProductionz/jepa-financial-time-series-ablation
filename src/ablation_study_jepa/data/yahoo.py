"""Yahoo Finance OHLCV download helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN")
DEFAULT_START_DATE = "1960-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_DIR = Path("data/prices/sp500/data")

YAHOO_COLUMN_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "adj_close",
    "adj_close": "adj_close",
    "volume": "volume",
}
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")


@dataclass(frozen=True)
class YahooPriceDownloadResult:
    """Summary for one ticker download."""

    ticker: str
    path: Path
    rows: int
    start_date: str | None
    end_date: str | None
    status: str
    error: str | None = None


def parse_tickers(tickers: str | Iterable[str]) -> list[str]:
    """Normalize comma-separated or iterable ticker input."""

    if isinstance(tickers, str):
        raw_tickers = tickers.split(",")
    else:
        raw_tickers = list(tickers)

    normalized = []
    seen = set()
    for ticker in raw_tickers:
        value = str(ticker).strip().upper()
        if value and value not in seen:
            normalized.append(value)
            seen.add(value)
    if not normalized:
        raise ValueError("At least one ticker is required.")
    return normalized


def _exclusive_end_date(end_date: str | pd.Timestamp) -> str:
    # yfinance treats end as exclusive, so add one day for an inclusive CLI range.
    return (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")


def _canonical_column_name(column: object) -> str:
    name = str(column).strip().replace("_", " ").lower()
    return YAHOO_COLUMN_MAP.get(name, name.replace(" ", "_"))


def _drop_single_value_multiindex_levels(frame: pd.DataFrame) -> pd.DataFrame:
    while isinstance(frame.columns, pd.MultiIndex) and frame.columns.nlevels > 1:
        for level in range(frame.columns.nlevels):
            if frame.columns.get_level_values(level).nunique() == 1:
                frame = frame.droplevel(level, axis=1)
                break
        else:
            break
    return frame


def normalize_yahoo_price_frame(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Convert a yfinance OHLCV frame to the repository's price panel schema."""

    if frame is None or frame.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    ticker = ticker.upper()
    frame = frame.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        for level in range(frame.columns.nlevels):
            level_values = [str(value).upper() for value in frame.columns.get_level_values(level)]
            if ticker in level_values:
                frame = frame.xs(ticker, axis=1, level=level, drop_level=True)
                break
        frame = _drop_single_value_multiindex_levels(frame)
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                "_".join(str(part) for part in column if str(part))
                for column in frame.columns.to_flat_index()
            ]

    frame = frame.reset_index()
    date_column = next(
        (column for column in frame.columns if str(column).strip().lower() in {"date", "datetime"}),
        frame.columns[0],
    )
    frame = frame.rename(columns={date_column: "date"})
    frame.columns = ["date" if column == "date" else _canonical_column_name(column) for column in frame.columns]

    missing_columns = [column for column in PRICE_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Yahoo data for {ticker} is missing columns: {missing_columns}")

    dates = pd.to_datetime(frame["date"], errors="coerce")
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(None)
    frame["date"] = dates.dt.strftime("%Y-%m-%d")
    frame.insert(0, "ticker", ticker)

    for column in PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["date", "open", "high", "low", "close", "volume"], how="any")
    output_columns = ["ticker", "date", *PRICE_COLUMNS]
    return frame[output_columns].sort_values("date").reset_index(drop=True)


def _coverage(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty or "date" not in frame.columns:
        return None, None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _import_yfinance() -> object:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is required for Yahoo downloads. Run `uv sync --dev` first."
        ) from exc
    return yf


def download_yahoo_prices(
    tickers: str | Iterable[str] = DEFAULT_TICKERS,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    skip_existing: bool = True,
    overwrite: bool = False,
    progress: bool = False,
    timeout: float | None = 30,
    continue_on_error: bool = False,
) -> list[YahooPriceDownloadResult]:
    """Download daily Yahoo Finance OHLCV files, one CSV per ticker.

    Files are written as ``{output_dir}/{TICKER}.csv`` with columns compatible
    with ``load_price_panel``. Existing files are skipped by default so a later
    run can add tickers without disturbing already-downloaded assets.
    """

    yf = _import_yfinance()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for ticker in parse_tickers(tickers):
        path = output_dir / f"{ticker}.csv"
        if path.exists() and skip_existing and not overwrite:
            existing = pd.read_csv(path)
            first_date, last_date = _coverage(existing)
            results.append(
                YahooPriceDownloadResult(
                    ticker=ticker,
                    path=path,
                    rows=len(existing),
                    start_date=first_date,
                    end_date=last_date,
                    status="skipped",
                )
            )
            continue

        try:
            raw = yf.download(
                ticker,
                start=start_date,
                end=_exclusive_end_date(end_date),
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=progress,
                threads=False,
                timeout=timeout,
                multi_level_index=False,
            )
            normalized = normalize_yahoo_price_frame(raw, ticker)
        except Exception as exc:
            if not continue_on_error:
                raise
            results.append(
                YahooPriceDownloadResult(
                    ticker=ticker,
                    path=path,
                    rows=0,
                    start_date=None,
                    end_date=None,
                    status="failed",
                    error=str(exc),
                )
            )
            continue

        if path.exists() and not overwrite:
            existing = pd.read_csv(path)
            normalized = pd.concat([existing, normalized], ignore_index=True)
            normalized = normalized.drop_duplicates(subset=["ticker", "date"], keep="last")
            status = "updated"
        else:
            status = "downloaded"

        normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
        normalized.to_csv(path, index=False)
        first_date, last_date = _coverage(normalized)
        results.append(
            YahooPriceDownloadResult(
                ticker=ticker,
                path=path,
                rows=len(normalized),
                start_date=first_date,
                end_date=last_date,
                status=status,
            )
        )

    return results
