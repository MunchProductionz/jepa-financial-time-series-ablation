"""S&P 500 historical universe helpers.

The universe CSV is a retrieval manifest, not a claim that Yahoo Finance has
complete delisted security coverage. It preserves source membership dates and
candidate Yahoo tickers so failed downloads can be audited.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from io import StringIO
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable
from urllib.request import Request, urlopen

import pandas as pd

from ablation_study_jepa.data.yahoo import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    YahooPriceDownloadResult,
    download_yahoo_prices,
)

WRDS_SP500_CHANGES_URL = (
    "https://wrds-www.wharton.upenn.edu/classroom/sp500-introduction/over-time/"
)
WIKIPEDIA_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
DEFAULT_UNIVERSE_PATH = Path("data/universe/sp500_since_1960.csv")
DEFAULT_UNIVERSE_JSON_PATH = Path("data/universe/sp500_since_1960.json")
DEFAULT_DOWNLOAD_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "download_manifest.csv"
DEFAULT_UNAVAILABLE_TICKERS_PATH = DEFAULT_OUTPUT_DIR / "unavailable_tickers.csv"
DOWNLOAD_COMPLETE_STATUSES = {"downloaded", "updated", "skipped", "json_skipped"}
DOWNLOAD_TERMINAL_STATUSES = {*DOWNLOAD_COMPLETE_STATUSES, "failed", "json_failed_skipped"}

UNIVERSE_COLUMNS = [
    "ticker",
    "ticker_raw",
    "yahoo_ticker_candidates",
    "company",
    "permno",
    "sp500_start_date",
    "sp500_end_date",
    "first_year_available",
    "entered_sp500_year",
    "left_sp500_year",
    "delisted_or_shutdown_year",
    "is_current_sp500",
    "membership_status",
    "source",
    "notes",
]


@dataclass(frozen=True)
class Sp500UniverseBuildResult:
    """Summary of an S&P 500 universe CSV build."""

    path: Path
    json_path: Path
    rows: int
    unique_tickers: int
    current_tickers: int
    missing_ticker_rows: int


def clean_yahoo_ticker(ticker: object) -> str:
    """Convert common source ticker syntax to Yahoo Finance syntax."""

    value = str(ticker).strip().upper()
    if not value or value in {"NAN", "NONE", "NULL", "--", "-"}:
        return ""
    value = re.sub(r"\[[^\]]*\]", "", value)
    value = re.sub(r"\s+", "", value)
    value = value.replace(".", "-")
    value = re.sub(r"[^A-Z0-9^-]", "", value)
    return value


def ticker_candidates(ticker_raw: object) -> list[str]:
    """Return unique Yahoo-compatible candidates from a raw source ticker field."""

    candidates = []
    seen = set()
    for ticker in re.split(r"[,;/]", str(ticker_raw)):
        cleaned = clean_yahoo_ticker(ticker)
        if cleaned and cleaned not in seen:
            candidates.append(cleaned)
            seen.add(cleaned)
    return candidates


def build_sp500_universe(
    output_path: str | Path = DEFAULT_UNIVERSE_PATH,
    json_path: str | Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    wrds_source: str | Path = WRDS_SP500_CHANGES_URL,
    wikipedia_source: str | Path = WIKIPEDIA_SP500_URL,
) -> Sp500UniverseBuildResult:
    """Build a CSV of candidate tickers that have touched the S&P 500 since 1960."""

    output_path = Path(output_path)
    json_output_path = Path(json_path) if json_path is not None else output_path.with_suffix(".json")
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    wrds_intervals = wrds_sp500_intervals(wrds_source, start_date=start_date, end_date=end_date)
    recent_events = wikipedia_sp500_change_events(wikipedia_source)
    current = wikipedia_current_sp500_constituents(wikipedia_source)

    if not wrds_intervals.empty:
        source_cutoff = wrds_intervals["sp500_end_date"].dropna().max()
        active_at_cutoff = wrds_intervals["sp500_end_date"].eq(source_cutoff)
        wrds_intervals.loc[active_at_cutoff, "sp500_end_date"] = pd.NaT
        wrds_intervals.loc[active_at_cutoff, "membership_status"] = "active_at_wrds_cutoff"
    else:
        source_cutoff = start

    recent_events = recent_events[
        recent_events["effective_date"].notna() & (recent_events["effective_date"] > source_cutoff)
    ].copy()
    recent_intervals = _recent_change_intervals(recent_events)

    universe = pd.concat([wrds_intervals, recent_intervals], ignore_index=True)
    universe = _apply_recent_removals(universe, recent_events)
    universe = _merge_current_constituents(universe, current, source_cutoff=source_cutoff)

    current_tickers = _current_ticker_set(current)
    universe = _expand_candidate_rows(universe)
    universe = _filter_overlapping_intervals(universe, start=start, end=end)
    universe = _finalize_universe_columns(universe, current_tickers=current_tickers)
    universe = universe.sort_values(["ticker", "sp500_start_date", "sp500_end_date"]).reset_index(
        drop=True
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    write_sp500_universe_json(
        universe=universe,
        output_path=json_output_path,
        end_date=end_date,
    )

    return Sp500UniverseBuildResult(
        path=output_path,
        json_path=json_output_path,
        rows=len(universe),
        unique_tickers=len(_unique_nonempty(universe["ticker"])),
        current_tickers=universe.loc[universe["is_current_sp500"], "ticker"].nunique(),
        missing_ticker_rows=int((universe["ticker"] == "").sum()),
    )


def wrds_sp500_intervals(
    source: str | Path = WRDS_SP500_CHANGES_URL,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """Parse WRDS classroom S&P 500 added/removed tables into membership intervals."""

    frames = []
    for table in _read_html_tables(source):
        normalized = _normalize_table_columns(table)
        columns = {_normalize_name(column): column for column in normalized.columns}
        required = {
            "added_removed",
            "permno",
            "company",
            "ticker",
            "sp500_start",
            "sp500_end",
        }
        if not required.issubset(columns):
            continue
        frames.append(
            pd.DataFrame(
                {
                    "event_type": normalized[columns["added_removed"]],
                    "permno": normalized[columns["permno"]],
                    "company": normalized[columns["company"]],
                    "ticker_raw": normalized[columns["ticker"]],
                    "sp500_start_date": pd.to_datetime(
                        normalized[columns["sp500_start"]], errors="coerce"
                    ),
                    "sp500_end_date": pd.to_datetime(
                        normalized[columns["sp500_end"]], errors="coerce"
                    ),
                    "is_current_sp500": False,
                    "membership_status": "closed",
                    "source": "wrds_classroom_sp500_changes",
                    "notes": "",
                }
            )
        )

    if not frames:
        return _empty_universe_frame()

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["sp500_start_date", "sp500_end_date"], how="any")
    frame = frame[_interval_overlaps(frame, start=pd.Timestamp(start_date), end=pd.Timestamp(end_date))]
    frame = frame.drop_duplicates(
        subset=["permno", "ticker_raw", "company", "sp500_start_date", "sp500_end_date"]
    )
    return frame.reset_index(drop=True)


def wikipedia_current_sp500_constituents(
    source: str | Path = WIKIPEDIA_SP500_URL,
) -> pd.DataFrame:
    """Parse the current S&P 500 constituents table from Wikipedia."""

    for table in _read_html_tables(source):
        normalized = _normalize_table_columns(table)
        columns = {_normalize_name(column): column for column in normalized.columns}
        if {"symbol", "security"}.issubset(columns):
            date_column = columns.get("date_added")
            if date_column is None:
                start_dates = pd.NaT
            else:
                start_dates = pd.to_datetime(normalized[date_column], errors="coerce")
            return pd.DataFrame(
                {
                    "event_type": "Current",
                    "permno": "",
                    "company": normalized[columns["security"]],
                    "ticker_raw": normalized[columns["symbol"]],
                    "current_ticker": normalized[columns["symbol"]].map(
                        lambda value: ticker_candidates(value)[0] if ticker_candidates(value) else ""
                    ),
                    "sp500_start_date": start_dates,
                    "sp500_end_date": pd.NaT,
                    "is_current_sp500": True,
                    "membership_status": "current",
                    "source": "wikipedia_current_sp500",
                    "notes": "",
                }
            )
    return _empty_universe_frame()


def wikipedia_sp500_change_events(source: str | Path = WIKIPEDIA_SP500_URL) -> pd.DataFrame:
    """Parse selected S&P 500 change events from Wikipedia."""

    events = []
    for table in _read_html_tables(source):
        normalized = _normalize_table_columns(table)
        columns = {_normalize_name(column): column for column in normalized.columns}
        if "effective_date" not in columns or "reason" not in columns:
            continue
        added_ticker = _find_column(columns, ["added_ticker", "ticker_added"])
        added_security = _find_column(columns, ["added_security", "security_added"])
        removed_ticker = _find_column(columns, ["removed_ticker", "ticker_removed"])
        removed_security = _find_column(columns, ["removed_security", "security_removed"])
        if added_ticker is None and removed_ticker is None:
            continue

        for _, row in normalized.iterrows():
            effective_date = pd.to_datetime(row[columns["effective_date"]], errors="coerce")
            reason = row.get(columns["reason"], "")
            if added_ticker is not None and str(row.get(added_ticker, "")).strip():
                events.append(
                    {
                        "effective_date": effective_date,
                        "event_type": "added",
                        "ticker_raw": row.get(added_ticker, ""),
                        "company": row.get(added_security, ""),
                        "reason": reason,
                        "source": "wikipedia_selected_changes",
                    }
                )
            if removed_ticker is not None and str(row.get(removed_ticker, "")).strip():
                events.append(
                    {
                        "effective_date": effective_date,
                        "event_type": "removed",
                        "ticker_raw": row.get(removed_ticker, ""),
                        "company": row.get(removed_security, ""),
                        "reason": reason,
                        "source": "wikipedia_selected_changes",
                    }
                )

    if not events:
        return pd.DataFrame(
            columns=["effective_date", "event_type", "ticker_raw", "company", "reason", "source"]
        )
    frame = pd.DataFrame(events)
    frame["ticker"] = frame["ticker_raw"].map(lambda value: ticker_candidates(value)[0] if ticker_candidates(value) else "")
    return frame


def download_sp500_universe_prices(
    universe_path: str | Path = DEFAULT_UNIVERSE_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    ticker_column: str = "ticker",
    skip_existing: bool = True,
    overwrite: bool = False,
    progress: bool = False,
    continue_on_error: bool = True,
    manifest_path: str | Path | None = DEFAULT_DOWNLOAD_MANIFEST_PATH,
    lookup_json_path: str | Path | None = DEFAULT_UNIVERSE_JSON_PATH,
    unavailable_path: str | Path | None = DEFAULT_UNAVAILABLE_TICKERS_PATH,
    max_tickers: int | None = None,
    eta_window: int = 10,
    retry_failed: bool = False,
    progress_printer: Callable[[str], None] | None = None,
) -> list[YahooPriceDownloadResult]:
    """Download Yahoo prices for every unique ticker in a universe CSV."""

    output_dir = Path(output_dir)
    universe = pd.read_csv(universe_path)
    if ticker_column not in universe.columns:
        raise ValueError(f"Universe file is missing ticker column: {ticker_column}")
    lookup_path = _resolve_lookup_json_path(universe_path, lookup_json_path)
    lookup = ensure_sp500_universe_json(
        universe=universe,
        json_path=lookup_path,
        price_dir=output_dir,
        end_date=end_date,
    )
    tickers = [
        ticker
        for ticker in _unique_nonempty(lookup.get("ticker_order", lookup["tickers"].keys()))
        if ticker in lookup["tickers"]
    ]
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    manifest_records = _read_manifest_records(manifest_path)
    durations: deque[float] = deque(maxlen=max(1, eta_window))

    results = []
    for index, ticker in enumerate(tickers, start=1):
        if skip_existing and not overwrite and _lookup_has_terminal_ticker(
            lookup=lookup,
            ticker=ticker,
            output_dir=output_dir,
            retry_failed=retry_failed,
        ):
            result = _json_skipped_result(lookup, ticker, output_dir)
            results.append(result)
            _upsert_manifest_result(manifest_records, result)
            _persist_download_state(
                lookup=lookup,
                lookup_path=lookup_path,
                manifest_records=manifest_records,
                manifest_path=manifest_path,
                unavailable_path=unavailable_path,
            )
            _print_progress(
                progress_printer=progress_printer,
                result=result,
                index=index,
                total=len(tickers),
                elapsed=None,
                remaining_downloads=_remaining_download_count(
                    lookup=lookup,
                    tickers=tickers[index:],
                    output_dir=output_dir,
                    skip_existing=skip_existing,
                    overwrite=overwrite,
                    retry_failed=retry_failed,
                ),
                durations=durations,
            )
            continue

        started_at = time.monotonic()
        ticker_results = download_yahoo_prices(
            tickers=[ticker],
            output_dir=output_dir,
            start_date=start_date,
            end_date=end_date,
            skip_existing=skip_existing,
            overwrite=overwrite,
            progress=progress,
            continue_on_error=continue_on_error,
        )
        elapsed = time.monotonic() - started_at
        result = ticker_results[0]
        results.append(result)
        if result.status != "skipped":
            durations.append(elapsed)
        _update_lookup_download_status(lookup, result)
        _upsert_manifest_result(manifest_records, result)
        _persist_download_state(
            lookup=lookup,
            lookup_path=lookup_path,
            manifest_records=manifest_records,
            manifest_path=manifest_path,
            unavailable_path=unavailable_path,
        )
        _print_progress(
            progress_printer=progress_printer,
            result=result,
            index=index,
            total=len(tickers),
            elapsed=elapsed,
            remaining_downloads=_remaining_download_count(
                lookup=lookup,
                tickers=tickers[index:],
                output_dir=output_dir,
                skip_existing=skip_existing,
                overwrite=overwrite,
                retry_failed=retry_failed,
            ),
            durations=durations,
        )

    annotate_universe_with_price_coverage(
        universe_path=universe_path,
        price_dir=output_dir,
        end_date=end_date,
    )
    lookup = ensure_sp500_universe_json(
        universe=pd.read_csv(universe_path),
        json_path=lookup_path,
        price_dir=output_dir,
        end_date=end_date,
    )
    _persist_download_state(
        lookup=lookup,
        lookup_path=lookup_path,
        manifest_records=manifest_records,
        manifest_path=manifest_path,
        unavailable_path=unavailable_path,
    )
    return results


def ensure_sp500_universe_json(
    universe: pd.DataFrame | str | Path,
    json_path: str | Path,
    price_dir: str | Path | None = None,
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, Any]:
    """Create or refresh the durable S&P 500 ticker lookup JSON."""

    json_path = Path(json_path)
    frame = pd.read_csv(universe) if isinstance(universe, str | Path) else universe.copy()
    existing = _read_lookup_json(json_path)
    lookup = build_sp500_universe_lookup(frame, end_date=end_date)
    _merge_existing_download_state(lookup, existing)
    if price_dir is not None:
        _annotate_lookup_with_price_files(lookup, price_dir=price_dir, end_date=end_date)
    _refresh_unavailable_tickers(lookup)
    _write_lookup_json(lookup, json_path)
    return lookup


def write_sp500_universe_json(
    universe: pd.DataFrame | str | Path,
    output_path: str | Path,
    price_dir: str | Path | None = None,
    end_date: str = DEFAULT_END_DATE,
) -> Path:
    """Write the S&P 500 ticker lookup JSON and return its path."""

    return Path(
        ensure_sp500_universe_json(
            universe=universe,
            json_path=output_path,
            price_dir=price_dir,
            end_date=end_date,
        )["_path"]
    )


def build_sp500_universe_lookup(
    universe: pd.DataFrame,
    end_date: str = DEFAULT_END_DATE,
) -> dict[str, Any]:
    """Build a JSON-serializable lookup keyed by Yahoo ticker."""

    frame = universe.copy()
    end = pd.Timestamp(end_date)
    tickers: dict[str, dict[str, Any]] = {}
    source_rows_without_ticker = []
    for _, row in frame.iterrows():
        record = row.to_dict()
        ticker = clean_yahoo_ticker(record.get("ticker", ""))
        if not ticker:
            source_rows_without_ticker.append(_source_row_without_ticker_record(record))
            continue

        entry = tickers.setdefault(
            ticker,
            {
                "ticker": ticker,
                "companies": [],
                "raw_tickers": [],
                "yahoo_ticker_candidates": [],
                "sp500_periods": [],
                "is_current_sp500": False,
                "first_year_available": None,
                "delisted_or_shutdown_year": None,
                "download": {"status": "not_attempted", "path": None},
            },
        )
        entry["companies"] = _unique_text([*entry["companies"], record.get("company", "")])
        entry["raw_tickers"] = _unique_text([*entry["raw_tickers"], record.get("ticker_raw", "")])
        entry["yahoo_ticker_candidates"] = _unique_text(
            [
                *entry["yahoo_ticker_candidates"],
                *str(record.get("yahoo_ticker_candidates", "")).split("|"),
            ]
        )
        entry["is_current_sp500"] = bool(entry["is_current_sp500"]) or _json_bool(
            record.get("is_current_sp500")
        )
        entry["first_year_available"] = _min_optional_int(
            entry.get("first_year_available"),
            _json_int(record.get("first_year_available")),
        )
        entry["delisted_or_shutdown_year"] = _max_optional_int(
            entry.get("delisted_or_shutdown_year"),
            _json_int(record.get("delisted_or_shutdown_year")),
        )
        entry["sp500_periods"].append(_membership_period(record, end=end))

    for entry in tickers.values():
        entry["sp500_periods"] = _deduplicate_periods(entry["sp500_periods"])

    lookup = {
        "schema_version": 1,
        "default_end_date": end_date,
        "ticker_order": list(tickers),
        "tickers": tickers,
        "source_rows_without_ticker": source_rows_without_ticker,
        "unavailable_tickers": [],
        "updated_at_utc": _utc_now(),
        "_path": "",
    }
    _refresh_unavailable_tickers(lookup)
    return lookup


def write_download_manifest(
    results: Iterable[YahooPriceDownloadResult],
    output_path: str | Path = DEFAULT_DOWNLOAD_MANIFEST_PATH,
) -> Path:
    """Write one row per attempted ticker download."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**asdict(result), "path": str(result.path)} for result in results])
    frame.to_csv(output_path, index=False)
    return output_path


def annotate_universe_with_price_coverage(
    universe_path: str | Path,
    price_dir: str | Path = DEFAULT_OUTPUT_DIR,
    end_date: str = DEFAULT_END_DATE,
) -> Path:
    """Fill first/last available price years in an existing universe CSV."""

    universe_path = Path(universe_path)
    price_dir = Path(price_dir)
    universe = pd.read_csv(universe_path)
    end = pd.Timestamp(end_date)
    first_years = {}
    last_years = {}
    for ticker in _unique_nonempty(universe["ticker"]):
        path = price_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce").dropna()
        if dates.empty:
            continue
        first_years[ticker] = int(dates.min().year)
        last_date = dates.max()
        if last_date < end:
            last_years[ticker] = int(last_date.year)

    universe["first_year_available"] = universe["ticker"].map(first_years).combine_first(
        universe.get("first_year_available")
    )
    universe["delisted_or_shutdown_year"] = universe["ticker"].map(last_years).combine_first(
        universe.get("delisted_or_shutdown_year")
    )
    universe.to_csv(universe_path, index=False)
    return universe_path


def write_unavailable_tickers(
    lookup: dict[str, Any],
    output_path: str | Path = DEFAULT_UNAVAILABLE_TICKERS_PATH,
) -> Path:
    """Write source rows and ticker attempts that Yahoo could not retrieve."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = lookup.get("unavailable_tickers", [])
    columns = [
        "ticker",
        "company",
        "From",
        "To",
        "is_current_sp500",
        "reason",
        "download_status",
        "yahoo_error",
        "membership_periods",
        "notes",
        "source",
    ]
    pd.DataFrame(records, columns=columns).to_csv(output_path, index=False)
    return output_path


def _source_row_without_ticker_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker": "",
        "company": _clean_json_scalar(record.get("company")),
        "From": _json_int(record.get("entered_sp500_year")),
        "To": _period_to_value(record),
        "from_date": _clean_json_scalar(record.get("sp500_start_date")),
        "to_date": _clean_json_scalar(record.get("sp500_end_date")),
        "is_current_sp500": _json_bool(record.get("is_current_sp500")),
        "reason": "source_row_without_ticker",
        "download_status": "unretrievable",
        "yahoo_error": "No ticker symbol in source membership row.",
        "membership_status": _clean_json_scalar(record.get("membership_status")),
        "notes": _clean_json_scalar(record.get("notes")),
        "source": _clean_json_scalar(record.get("source")),
    }


def _membership_period(record: dict[str, Any], end: pd.Timestamp) -> dict[str, Any]:
    return {
        "From": _json_int(record.get("entered_sp500_year")),
        "To": _period_to_value(record),
        "from_date": _clean_json_scalar(record.get("sp500_start_date")),
        "to_date": _clean_json_scalar(record.get("sp500_end_date")),
        "active_at_default_end_date": _active_at_default_end_date(record, end=end),
        "is_current_sp500": _json_bool(record.get("is_current_sp500")),
        "membership_status": _clean_json_scalar(record.get("membership_status")),
        "company": _clean_json_scalar(record.get("company")),
        "source": _clean_json_scalar(record.get("source")),
        "notes": _clean_json_scalar(record.get("notes")),
    }


def _period_to_value(record: dict[str, Any]) -> int | str | None:
    if _json_bool(record.get("is_current_sp500")):
        return None
    left_year = _json_int(record.get("left_sp500_year"))
    if left_year is not None:
        return left_year
    if _clean_json_scalar(record.get("sp500_end_date")) is None:
        return "UNKNOWN"
    return None


def _active_at_default_end_date(record: dict[str, Any], end: pd.Timestamp) -> bool:
    start_date = pd.to_datetime(record.get("sp500_start_date"), errors="coerce")
    end_date = pd.to_datetime(record.get("sp500_end_date"), errors="coerce")
    if pd.isna(start_date) or start_date > end:
        return False
    if pd.notna(end_date):
        return end_date >= end
    return _json_bool(record.get("is_current_sp500"))


def _deduplicate_periods(periods: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped = []
    seen = set()
    for period in periods:
        key = (
            period.get("From"),
            period.get("To"),
            period.get("from_date"),
            period.get("to_date"),
            period.get("company"),
        )
        if key in seen:
            continue
        deduped.append(period)
        seen.add(key)
    return deduped


def _merge_existing_download_state(lookup: dict[str, Any], existing: dict[str, Any]) -> None:
    existing_tickers = existing.get("tickers", {}) if isinstance(existing, dict) else {}
    for ticker, entry in lookup["tickers"].items():
        previous = existing_tickers.get(ticker, {})
        download = previous.get("download")
        if isinstance(download, dict) and download.get("status"):
            entry["download"] = download
        for key in ["first_year_available", "delisted_or_shutdown_year"]:
            value = previous.get(key)
            if value is not None and not pd.isna(value):
                entry[key] = value


def _annotate_lookup_with_price_files(
    lookup: dict[str, Any],
    price_dir: str | Path,
    end_date: str = DEFAULT_END_DATE,
) -> None:
    price_dir = Path(price_dir)
    end = pd.Timestamp(end_date)
    for ticker, entry in lookup["tickers"].items():
        path = price_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        coverage = _price_file_coverage(path)
        if coverage is None:
            continue
        rows, first_date, last_date = coverage
        entry["first_year_available"] = int(pd.Timestamp(first_date).year)
        if pd.Timestamp(last_date) < end:
            entry["delisted_or_shutdown_year"] = int(pd.Timestamp(last_date).year)
        entry["download"] = {
            "status": "downloaded",
            "path": str(path),
            "rows": rows,
            "start_date": first_date,
            "end_date": last_date,
            "error": None,
            "last_attempted_at_utc": entry.get("download", {}).get("last_attempted_at_utc"),
        }


def _price_file_coverage(path: str | Path) -> tuple[int, str, str] | None:
    try:
        dates = pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"], errors="coerce").dropna()
    except (FileNotFoundError, ValueError, pd.errors.EmptyDataError):
        return None
    if dates.empty:
        return None
    return len(dates), dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def _lookup_has_terminal_ticker(
    lookup: dict[str, Any],
    ticker: str,
    output_dir: Path,
    retry_failed: bool,
) -> bool:
    download = lookup.get("tickers", {}).get(ticker, {}).get("download", {})
    status = download.get("status")
    if status == "failed" and not retry_failed:
        return True
    if status not in DOWNLOAD_COMPLETE_STATUSES:
        return False
    path = Path(download.get("path") or output_dir / f"{ticker}.csv")
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.exists() and int(download.get("rows") or 0) > 0


def _json_skipped_result(
    lookup: dict[str, Any],
    ticker: str,
    output_dir: Path,
) -> YahooPriceDownloadResult:
    download = lookup["tickers"][ticker].get("download", {})
    previous_status = str(download.get("status") or "")
    status = "json_failed_skipped" if previous_status == "failed" else "json_skipped"
    return YahooPriceDownloadResult(
        ticker=ticker,
        path=Path(download.get("path") or output_dir / f"{ticker}.csv"),
        rows=int(download.get("rows") or 0),
        start_date=download.get("start_date"),
        end_date=download.get("end_date"),
        status=status,
        error=download.get("error"),
    )


def _update_lookup_download_status(
    lookup: dict[str, Any],
    result: YahooPriceDownloadResult,
) -> None:
    entry = lookup["tickers"].setdefault(
        result.ticker,
        {
            "ticker": result.ticker,
            "companies": [],
            "raw_tickers": [],
            "yahoo_ticker_candidates": [result.ticker],
            "sp500_periods": [],
            "is_current_sp500": False,
            "first_year_available": None,
            "delisted_or_shutdown_year": None,
        },
    )
    entry["download"] = {
        "status": result.status,
        "path": str(result.path),
        "rows": int(result.rows),
        "start_date": result.start_date,
        "end_date": result.end_date,
        "error": result.error,
        "last_attempted_at_utc": _utc_now(),
    }
    if result.start_date:
        entry["first_year_available"] = int(pd.Timestamp(result.start_date).year)
    if result.end_date and pd.Timestamp(result.end_date) < pd.Timestamp(lookup["default_end_date"]):
        entry["delisted_or_shutdown_year"] = int(pd.Timestamp(result.end_date).year)
    _refresh_unavailable_tickers(lookup)


def _refresh_unavailable_tickers(lookup: dict[str, Any]) -> None:
    records = list(lookup.get("source_rows_without_ticker", []))
    for ticker, entry in lookup.get("tickers", {}).items():
        download = entry.get("download", {})
        if download.get("status") != "failed":
            continue
        periods = entry.get("sp500_periods", [])
        records.append(
            {
                "ticker": ticker,
                "company": "; ".join(entry.get("companies", [])),
                "From": _first_period_value(periods, "From"),
                "To": _last_period_value(periods, "To"),
                "is_current_sp500": bool(entry.get("is_current_sp500")),
                "reason": (
                    "yahoo_download_failed_current_constituent"
                    if entry.get("is_current_sp500")
                    else "yahoo_download_failed_delisted_or_historical"
                ),
                "download_status": download.get("status"),
                "yahoo_error": download.get("error"),
                "membership_periods": json.dumps(periods, sort_keys=True),
                "notes": "; ".join(
                    _unique_text(period.get("notes", "") for period in periods)
                ),
                "source": "; ".join(
                    _unique_text(period.get("source", "") for period in periods)
                ),
            }
        )
    lookup["unavailable_tickers"] = records


def _first_period_value(periods: list[dict[str, Any]], key: str) -> Any:
    values = [period.get(key) for period in periods if period.get(key) not in {None, ""}]
    return values[0] if values else None


def _last_period_value(periods: list[dict[str, Any]], key: str) -> Any:
    values = [period.get(key) for period in periods if period.get(key) not in {None, ""}]
    return values[-1] if values else None


def _read_lookup_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _write_lookup_json(lookup: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lookup["updated_at_utc"] = _utc_now()
    lookup["_path"] = str(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_without_private_keys(lookup), handle, indent=2, sort_keys=True)
        handle.write("\n")
    lookup["_path"] = str(path)
    return path


def _without_private_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_private_keys(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, list):
        return [_without_private_keys(item) for item in value]
    return value


def _read_manifest_records(manifest_path: str | Path | None) -> dict[str, dict[str, Any]]:
    if manifest_path is None:
        return {}
    path = Path(manifest_path)
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return {}
    if "ticker" not in frame.columns:
        return {}
    return {
        str(row["ticker"]): {key: _clean_json_scalar(value) for key, value in row.items()}
        for _, row in frame.iterrows()
    }


def _upsert_manifest_result(
    manifest_records: dict[str, dict[str, Any]],
    result: YahooPriceDownloadResult,
) -> None:
    manifest_records[result.ticker] = {**asdict(result), "path": str(result.path)}


def _write_manifest_records(
    manifest_records: dict[str, dict[str, Any]],
    manifest_path: str | Path | None,
) -> None:
    if manifest_path is None:
        return
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(manifest_records.values()).to_csv(path, index=False)


def _persist_download_state(
    lookup: dict[str, Any],
    lookup_path: Path,
    manifest_records: dict[str, dict[str, Any]],
    manifest_path: str | Path | None,
    unavailable_path: str | Path | None,
) -> None:
    _refresh_unavailable_tickers(lookup)
    _write_lookup_json(lookup, lookup_path)
    _write_manifest_records(manifest_records, manifest_path)
    if unavailable_path is not None:
        write_unavailable_tickers(lookup, unavailable_path)


def _remaining_download_count(
    lookup: dict[str, Any],
    tickers: list[str],
    output_dir: Path,
    skip_existing: bool,
    overwrite: bool,
    retry_failed: bool,
) -> int:
    if not skip_existing or overwrite:
        return len(tickers)
    return sum(
        not _lookup_has_terminal_ticker(
            lookup=lookup,
            ticker=ticker,
            output_dir=output_dir,
            retry_failed=retry_failed,
        )
        for ticker in tickers
    )


def _print_progress(
    progress_printer: Callable[[str], None] | None,
    result: YahooPriceDownloadResult,
    index: int,
    total: int,
    elapsed: float | None,
    remaining_downloads: int,
    durations: deque[float],
) -> None:
    if progress_printer is None:
        return
    elapsed_text = "from JSON" if elapsed is None else f"{elapsed:.1f}s"
    if durations:
        average = sum(durations) / len(durations)
        eta_text = _format_duration(average * remaining_downloads)
        average_text = f"{average:.1f}s over last {len(durations)} retrieved"
    else:
        eta_text = "unknown"
        average_text = "unknown"
    message = (
        f"[{index:,}/{total:,}] {result.ticker}: {result.status} "
        f"({result.rows:,} rows, {result.start_date} to {result.end_date}); "
        f"elapsed={elapsed_text}; remaining_downloads={remaining_downloads:,}; "
        f"avg={average_text}; ETA={eta_text}"
    )
    if result.error:
        message = f"{message}; error={result.error}"
    progress_printer(message)


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes:d}m {seconds:02d}s"
    return f"{seconds:d}s"


def _resolve_lookup_json_path(
    universe_path: str | Path,
    lookup_json_path: str | Path | None,
) -> Path:
    if lookup_json_path is None:
        return Path(universe_path).with_suffix(".json")
    return Path(lookup_json_path)


def _json_int(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _json_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _clean_json_scalar(value: object) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return None
    return text


def _min_optional_int(left: object, right: object) -> int | None:
    values = [value for value in [_json_int(left), _json_int(right)] if value is not None]
    return min(values) if values else None


def _max_optional_int(left: object, right: object) -> int | None:
    values = [value for value in [_json_int(left), _json_int(right)] if value is not None]
    return max(values) if values else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _recent_change_intervals(events: pd.DataFrame) -> pd.DataFrame:
    additions = events[events["event_type"].eq("added")].copy()
    removals = events[events["event_type"].eq("removed")].copy()
    rows = []
    for _, addition in additions.iterrows():
        ticker = addition["ticker"]
        if not ticker:
            continue
        later_removals = removals[
            removals["ticker"].eq(ticker) & (removals["effective_date"] >= addition["effective_date"])
        ]
        end_date = later_removals["effective_date"].min() if not later_removals.empty else pd.NaT
        rows.append(
            {
                "event_type": "Added",
                "permno": "",
                "company": addition["company"],
                "ticker_raw": addition["ticker_raw"],
                "sp500_start_date": addition["effective_date"],
                "sp500_end_date": end_date,
                "is_current_sp500": False,
                "membership_status": "closed" if pd.notna(end_date) else "open_recent_addition",
                "source": "wikipedia_selected_changes",
                "notes": addition.get("reason", ""),
            }
        )
    return pd.DataFrame(rows) if rows else _empty_universe_frame()


def _apply_recent_removals(universe: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if universe.empty or events.empty:
        return universe
    removals = events[events["event_type"].eq("removed") & events["ticker"].ne("")]
    for _, removal in removals.sort_values("effective_date").iterrows():
        candidates = universe["ticker_raw"].map(lambda raw: removal["ticker"] in ticker_candidates(raw))
        open_interval = universe["sp500_end_date"].isna()
        match = candidates & open_interval & (universe["sp500_start_date"] <= removal["effective_date"])
        universe.loc[match, "sp500_end_date"] = removal["effective_date"]
        universe.loc[match, "membership_status"] = "closed"
        universe.loc[match, "is_current_sp500"] = False
    return universe


def _merge_current_constituents(
    universe: pd.DataFrame,
    current: pd.DataFrame,
    source_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    rows = universe.to_dict("records")
    for _, current_row in current.iterrows():
        candidates = ticker_candidates(current_row["ticker_raw"])
        if not candidates:
            continue
        ticker = candidates[0]
        matched = False
        for row in rows:
            if ticker not in ticker_candidates(row["ticker_raw"]):
                continue
            if not _looks_like_same_company(row.get("company", ""), current_row["company"]):
                continue
            if pd.notna(row.get("sp500_end_date")):
                continue
            current_start = current_row["sp500_start_date"]
            row_start = row.get("sp500_start_date")
            if pd.notna(current_start) and (pd.isna(row_start) or current_start < row_start):
                row["sp500_start_date"] = current_start
            row["company"] = current_row["company"] or row.get("company", "")
            row["is_current_sp500"] = True
            row["membership_status"] = "current"
            row["current_ticker"] = "|".join(_unique_text([row.get("current_ticker", ""), ticker]))
            row["source"] = _join_sources(row.get("source", ""), "wikipedia_current_sp500")
            row["notes"] = _join_notes(row.get("notes", ""), f"current as of source; WRDS cutoff {source_cutoff.date()}")
            matched = True
        if not matched:
            rows.append(current_row.to_dict())
    return pd.DataFrame(rows) if rows else _empty_universe_frame()


def _expand_candidate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in frame.iterrows():
        candidates = ticker_candidates(row.get("ticker_raw", ""))
        if not candidates:
            record = row.to_dict()
            record["ticker"] = ""
            record["yahoo_ticker_candidates"] = ""
            record["notes"] = _join_notes(record.get("notes", ""), "source row has no ticker")
            rows.append(record)
            continue
        for ticker in candidates:
            record = row.to_dict()
            record["ticker"] = ticker
            record["yahoo_ticker_candidates"] = "|".join(candidates)
            rows.append(record)
    return pd.DataFrame(rows) if rows else _empty_universe_frame()


def _filter_overlapping_intervals(
    frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    current_rows = frame["is_current_sp500"].fillna(False)
    return frame[_interval_overlaps(frame, start=start, end=end) | current_rows].copy()


def _interval_overlaps(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    interval_start = frame["sp500_start_date"].fillna(start)
    interval_end = frame["sp500_end_date"].fillna(end)
    return (interval_start <= end) & (interval_end >= start)


def _finalize_universe_columns(frame: pd.DataFrame, current_tickers: set[str]) -> pd.DataFrame:
    for column in UNIVERSE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""

    frame["sp500_start_date"] = pd.to_datetime(frame["sp500_start_date"], errors="coerce")
    frame["sp500_end_date"] = pd.to_datetime(frame["sp500_end_date"], errors="coerce")
    current_marker = frame.get("current_ticker", pd.Series("", index=frame.index)).fillna("")
    frame["is_current_sp500"] = frame["is_current_sp500"].fillna(False).astype(bool)
    frame.loc[~frame["ticker"].isin(current_tickers), "is_current_sp500"] = False
    marker_mask = current_marker.ne("")
    frame.loc[marker_mask, "is_current_sp500"] = [
        ticker in str(marker).split("|")
        for ticker, marker in zip(
            frame.loc[marker_mask, "ticker"],
            current_marker[marker_mask],
            strict=False,
        )
    ]
    current_alias = frame["membership_status"].eq("current") & ~frame["is_current_sp500"]
    frame.loc[current_alias, "membership_status"] = "open_unknown_after_source_cutoff"
    frame.loc[current_alias, "notes"] = [
        _join_notes(note, f"historical alias; current ticker match is {marker}")
        for note, marker in zip(
            frame.loc[current_alias, "notes"],
            current_marker[current_alias],
            strict=False,
        )
    ]
    frame["entered_sp500_year"] = frame["sp500_start_date"].dt.year
    frame["left_sp500_year"] = frame["sp500_end_date"].dt.year

    for column in ["first_year_available", "delisted_or_shutdown_year"]:
        if column not in frame or frame[column].isna().all():
            frame[column] = ""

    stale_open = frame["sp500_end_date"].isna() & ~frame["is_current_sp500"].fillna(False)
    frame.loc[stale_open, "notes"] = frame.loc[stale_open, "notes"].map(
        lambda note: _join_notes(note, "left date unknown after membership source cutoff")
    )
    frame.loc[stale_open, "membership_status"] = frame.loc[stale_open, "membership_status"].replace(
        {"active_at_wrds_cutoff": "open_unknown_after_source_cutoff"}
    )

    for column in ["sp500_start_date", "sp500_end_date"]:
        frame[column] = frame[column].dt.strftime("%Y-%m-%d").fillna("")
    for column in ["entered_sp500_year", "left_sp500_year"]:
        frame[column] = frame[column].astype("Int64").astype(str).replace("<NA>", "")

    frame = frame.drop_duplicates(
        subset=["ticker", "company", "sp500_start_date", "sp500_end_date", "source"]
    )
    return frame[UNIVERSE_COLUMNS]


def _current_ticker_set(current: pd.DataFrame) -> set[str]:
    tickers: set[str] = set()
    for value in current.get("ticker_raw", []):
        tickers.update(ticker_candidates(value))
    return tickers


def _looks_like_same_company(left: object, right: object) -> bool:
    left_normalized = _normalize_company_name(left)
    right_normalized = _normalize_company_name(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return True
    return bool(_company_tokens(left_normalized) & _company_tokens(right_normalized))


def _normalize_company_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    stopwords = {
        "class",
        "co",
        "company",
        "corp",
        "corporation",
        "inc",
        "incorporated",
        "limited",
        "ltd",
        "plc",
        "the",
    }
    return " ".join(token for token in text.split() if token not in stopwords)


def _company_tokens(value: str) -> set[str]:
    generic = {
        "american",
        "financial",
        "general",
        "global",
        "group",
        "holdings",
        "international",
        "national",
        "systems",
        "technologies",
        "technology",
        "united",
    }
    return {token for token in value.split() if len(token) >= 4 and token not in generic}


def _read_html_tables(source: str | Path) -> list[pd.DataFrame]:
    source_value = str(source)
    if "<table" in source_value.lower():
        html_source = StringIO(source_value)
    else:
        html_source = source_value
    try:
        return pd.read_html(html_source)
    except (ImportError, ValueError):
        if "<table" in source_value.lower():
            html = source_value
        elif Path(source_value).exists():
            html = Path(source_value).read_text(encoding="utf-8")
        else:
            request = Request(source_value, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8", errors="replace")
        return _parse_html_tables_with_stdlib(html)


@dataclass(frozen=True)
class _HtmlCell:
    text: str
    is_header: bool
    colspan: int = 1
    rowspan: int = 1


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[_HtmlCell]]] = []
        self._table_depth = 0
        self._current_table: list[list[_HtmlCell]] | None = None
        self._current_row: list[_HtmlCell] | None = None
        self._current_cell_tag: str | None = None
        self._current_cell_attrs: dict[str, str] = {}
        self._current_cell_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._table_depth == 0:
                self._current_table = []
            self._table_depth += 1
            return
        if self._table_depth != 1:
            return
        if tag == "tr":
            self._current_row = []
        elif tag in {"th", "td"}:
            self._current_cell_tag = tag
            self._current_cell_attrs = {key: value or "" for key, value in attrs}
            self._current_cell_text = []

    def handle_data(self, data: str) -> None:
        if self._current_cell_tag is not None:
            self._current_cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._current_cell_tag == tag and self._current_row is not None:
            text = re.sub(r"\s+", " ", "".join(self._current_cell_text)).strip()
            self._current_row.append(
                _HtmlCell(
                    text=text,
                    is_header=tag == "th",
                    colspan=max(1, _safe_int(self._current_cell_attrs.get("colspan"), default=1)),
                    rowspan=max(1, _safe_int(self._current_cell_attrs.get("rowspan"), default=1)),
                )
            )
            self._current_cell_tag = None
            self._current_cell_attrs = {}
            self._current_cell_text = []
        elif tag == "tr" and self._table_depth == 1 and self._current_table is not None:
            if self._current_row:
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table":
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_table is not None:
                self.tables.append(self._current_table)
                self._current_table = None


def _parse_html_tables_with_stdlib(html: str) -> list[pd.DataFrame]:
    parser = _TableParser()
    parser.feed(html)
    frames = []
    for table in parser.tables:
        frame = _html_table_to_frame(table)
        if not frame.empty:
            frames.append(frame)
    return frames


def _html_table_to_frame(rows: list[list[_HtmlCell]]) -> pd.DataFrame:
    grid = _expand_cell_grid(rows)
    if not grid:
        return pd.DataFrame()
    header_rows = 0
    for row in grid:
        if any(cell.is_header for cell in row):
            header_rows += 1
        else:
            break
    if header_rows == 0:
        header_rows = 1
    max_columns = max(len(row) for row in grid)
    columns = []
    for column_index in range(max_columns):
        parts = []
        for row in grid[:header_rows]:
            if column_index >= len(row):
                continue
            text = row[column_index].text
            if text and text not in parts:
                parts.append(text)
        columns.append(" ".join(parts) or f"column_{column_index}")
    data = [
        [row[column_index].text if column_index < len(row) else "" for column_index in range(max_columns)]
        for row in grid[header_rows:]
    ]
    return pd.DataFrame(data, columns=columns)


def _expand_cell_grid(rows: list[list[_HtmlCell]]) -> list[list[_HtmlCell]]:
    grid: list[list[_HtmlCell]] = []
    occupied: dict[tuple[int, int], _HtmlCell] = {}
    for row_index, row in enumerate(rows):
        expanded_row: list[_HtmlCell] = []
        column_index = 0
        for cell in row:
            while (row_index, column_index) in occupied:
                expanded_row.append(occupied[(row_index, column_index)])
                column_index += 1
            for span_index in range(cell.colspan):
                expanded_row.append(cell)
                for rowspan_index in range(1, cell.rowspan):
                    occupied[(row_index + rowspan_index, column_index + span_index)] = cell
            column_index += cell.colspan
        while (row_index, column_index) in occupied:
            expanded_row.append(occupied[(row_index, column_index)])
            column_index += 1
        grid.append(expanded_row)
    return grid


def _safe_int(value: object, default: int) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def _normalize_table_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [_flatten_column(column) for column in frame.columns]
    if not any(_normalize_name(column) for column in frame.columns):
        return frame
    header_rows = frame.apply(
        lambda row: any(str(value).strip().lower() == "added/removed" for value in row),
        axis=1,
    )
    if header_rows.any():
        header_index = header_rows.idxmax()
        frame.columns = [_flatten_column(column) for column in frame.loc[header_index]]
        frame = frame.loc[header_index + 1 :].reset_index(drop=True)
    return frame


def _flatten_column(column: object) -> str:
    if isinstance(column, tuple):
        parts = [
            str(part).strip()
            for part in column
            if str(part).strip() and not str(part).startswith("Unnamed")
        ]
    else:
        parts = [str(column).strip()]
    deduped = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    return " ".join(deduped)


def _normalize_name(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _find_column(columns: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _unique_nonempty(values: Iterable[object]) -> list[str]:
    tickers = []
    seen = set()
    for value in values:
        ticker = clean_yahoo_ticker(value)
        if ticker and ticker not in seen:
            tickers.append(ticker)
            seen.add(ticker)
    return tickers


def _join_sources(left: str, right: str) -> str:
    return ";".join(part for part in _unique_text([left, right]) if part)


def _join_notes(left: object, right: str) -> str:
    return "; ".join(part for part in _unique_text([left, right]) if part)


def _unique_text(values: Iterable[object]) -> list[str]:
    output = []
    seen = set()
    for value in values:
        text = str(value).strip()
        if text and text.lower() != "nan" and text not in seen:
            output.append(text)
            seen.add(text)
    return output


def _empty_universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_type",
            "permno",
            "company",
            "ticker_raw",
            "current_ticker",
            "sp500_start_date",
            "sp500_end_date",
            "is_current_sp500",
            "membership_status",
            "source",
            "notes",
        ]
    )
