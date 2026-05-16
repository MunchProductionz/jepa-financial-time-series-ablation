"""Create deterministic synthetic OHLCV data for quick pipeline runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from ablation_study_jepa.data.synthetic import build_sample_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/prices/sp500/panel.csv")
    parser.add_argument("--tickers", default="AAPL,MSFT,NVDA,AMZN")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--periods", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    panel = build_sample_panel(
        tickers=args.tickers.split(","),
        start=args.start,
        periods=args.periods,
        seed=args.seed,
    )
    panel.to_csv(output, index=False)
    print(f"Wrote {len(panel):,} rows to {output}")


if __name__ == "__main__":
    main()
