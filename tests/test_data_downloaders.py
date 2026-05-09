from pathlib import Path

import pandas as pd

from ablation_study_jepa.data.fred_md import (
    fred_md_candidate_urls,
    parse_fred_md_file,
)
from ablation_study_jepa.data.loaders import load_price_panel
from ablation_study_jepa.data.yahoo import (
    parse_tickers,
    normalize_yahoo_price_frame,
)


def test_parse_tickers_deduplicates_and_normalizes() -> None:
    assert parse_tickers("aapl, MSFT, aapl") == ["AAPL", "MSFT"]


def test_normalize_yahoo_price_frame_uses_panel_schema() -> None:
    raw = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.0],
            "Close": [101.0, 102.0],
            "Adj Close": [100.5, 101.5],
            "Volume": [1_000, 2_000],
        },
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    raw.index.name = "Date"

    panel = normalize_yahoo_price_frame(raw, ticker="aapl")

    assert list(panel.columns) == [
        "ticker",
        "date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert panel["ticker"].tolist() == ["AAPL", "AAPL"]
    assert panel["date"].tolist() == ["2025-01-02", "2025-01-03"]


def test_fred_md_candidate_urls_include_vintage_csv() -> None:
    assert fred_md_candidate_urls("2025-12")[0].endswith("/2025-12.csv")


def test_parse_fred_md_file_keeps_transformations_and_filters_dates(tmp_path: Path) -> None:
    raw_path = tmp_path / "fred_md.csv"
    raw_path.write_text(
        "sasdate,RPI,UNRATE\n"
        "Transform:,5,2\n"
        "1959-12-01,1.0,2.0\n"
        "1960-01-01,3.0,4.0\n"
        "2025-12-01,5.0,6.0\n"
        "2026-01-01,7.0,8.0\n"
    )

    data, transformations = parse_fred_md_file(
        raw_path,
        start_date="1960-01-01",
        end_date="2025-12-31",
    )

    assert data["date"].tolist() == ["1960-01-01", "2025-12-01"]
    assert data["RPI"].tolist() == [3.0, 5.0]
    assert transformations is not None
    assert transformations.to_dict("records") == [
        {"series": "RPI", "transform_code": 5},
        {"series": "UNRATE", "transform_code": 2},
    ]


def test_price_loader_renames_prices_and_asof_merges_macro(tmp_path: Path) -> None:
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    (price_dir / "AAPL.csv").write_text(
        "ticker,date,open,high,low,close,adj_close,volume\n"
        "AAPL,2025-01-31,1,2,0.5,1.5,1.4,100\n"
        "AAPL,2025-02-03,2,3,1.5,2.5,2.4,200\n"
    )
    macro_path = tmp_path / "fred_md.csv"
    macro_path.write_text(
        "date,S&P 500,FEDFUNDS,GS1,GS5,GS10,OILPRICEx,S&P div yield,S&P PE ratio\n"
        "2025-01-01,5000,4.5,4.0,4.1,4.2,75,1.3,25\n"
        "2025-02-01,5100,4.4,3.9,4.0,4.1,76,1.2,26\n"
    )

    panel = load_price_panel(
        price_dir,
        tickers=["AAPL"],
        macro_data_path=macro_path,
        macro_feature_columns=[
            "S&P 500",
            "FEDFUNDS",
            "GS1",
            "GS5",
            "GS10",
            "OILPRICEx",
            "S&P: indust",
            "S&P div yield",
            "S&P PE ratio",
        ],
    )

    assert list(panel.columns) == [
        "ticker",
        "date",
        "Price Open",
        "Price High",
        "Price Low",
        "Price Close",
        "Price Adj_Close",
        "Volume",
        "S&P 500",
        "FEDFUNDS",
        "GS1",
        "GS5",
        "GS10",
        "OILPRICEx",
        "S&P div yield",
        "S&P PE ratio",
    ]
    assert panel["Price Close"].tolist() == [1.5, 2.5]
    assert panel["Volume"].tolist() == [100, 200]
    assert panel["S&P 500"].tolist() == [5000, 5100]
