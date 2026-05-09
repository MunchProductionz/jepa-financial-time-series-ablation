"""Download the FRED-MD macro database."""

from __future__ import annotations

import argparse

from ablation_study_jepa.data.fred_md import (
    DEFAULT_END_DATE,
    DEFAULT_FRED_MD_VINTAGE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_START_DATE,
    download_fred_md,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--vintage", default=DEFAULT_FRED_MD_VINTAGE)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = download_fred_md(
        output_dir=args.output_dir,
        vintage=args.vintage,
        start_date=args.start_date,
        end_date=args.end_date,
        overwrite=args.overwrite,
    )
    print(
        f"{result.status}: FRED-MD {result.vintage} -> {result.raw_path}; "
        f"filtered -> {result.data_path} "
        f"({result.rows:,} rows, {result.series_count:,} series, "
        f"{result.start_date} to {result.end_date})"
    )
    if result.transformations_path is not None:
        print(f"transformations -> {result.transformations_path}")


if __name__ == "__main__":
    main()
