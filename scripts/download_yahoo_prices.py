"""Download Yahoo Finance OHLCV data for configured tickers."""

from __future__ import annotations

import argparse

from ablation_study_jepa.data.yahoo import (
    DEFAULT_END_DATE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    DEFAULT_TICKERS,
    download_yahoo_prices,
)

SP500_PRICE_DOWNLOAD_COMMAND = """
uv run ablation-study-jepa download-sp500-prices \
  --universe data/universe/sp500_since_1960.csv \
  --lookup-json data/universe/sp500_since_1960.json \
  --start-date 1960-01-01 \
  --end-date 2025-12-31 \
  --output-dir data/prices/sp500 \
  --manifest data/prices/sp500/download_manifest.csv \
  --unavailable data/prices/sp500/unavailable_tickers.csv
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    results = download_yahoo_prices(
        tickers=args.tickers,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        skip_existing=not args.overwrite,
        overwrite=args.overwrite,
        progress=args.progress,
    )
    for result in results:
        print(
            f"{result.status}: {result.ticker} -> {result.path} "
            f"({result.rows:,} rows, {result.start_date} to {result.end_date})"
        )


if __name__ == "__main__":
    main()
