import json
from pathlib import Path

import pandas as pd

from ablation_study_jepa.data import sp500_universe
from ablation_study_jepa.data import yahoo
from ablation_study_jepa.builders.data import add_sector_one_hot_features
from ablation_study_jepa.data.fred_md import (
    fred_md_candidate_urls,
    parse_fred_md_file,
)
from ablation_study_jepa.data.loaders import load_price_panel
from ablation_study_jepa.data.yahoo import (
    parse_tickers,
    normalize_yahoo_price_frame,
)
from ablation_study_jepa.data.sp500_universe import (
    build_sp500_universe,
    download_sp500_universe_prices,
    ensure_sp500_universe_json,
    ticker_candidates,
    validate_yahoo_price_file,
    wikipedia_sp500_change_events,
    wrds_sp500_intervals,
)


def test_parse_tickers_deduplicates_and_normalizes() -> None:
    assert parse_tickers("aapl, MSFT, aapl") == ["AAPL", "MSFT"]


def test_ticker_candidates_convert_share_class_syntax() -> None:
    assert ticker_candidates("brk.b, BF.B, old") == ["BRK-B", "BF-B", "OLD"]


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


def test_load_price_panel_limit_uses_all_files_when_null(
    tmp_path: Path,
    capsys,
) -> None:
    _write_price_file(tmp_path, "AAA")
    _write_price_file(tmp_path, "BBB")
    _write_price_file(tmp_path, "CCC")

    panel = load_price_panel(tmp_path, limit=None)
    captured = capsys.readouterr()

    assert panel["ticker"].nunique() == 3
    assert "[data] using 3 ticker files from 3 available (limit=all)" in captured.out


def test_load_price_panel_limit_caps_selected_files(
    tmp_path: Path,
    capsys,
) -> None:
    _write_price_file(tmp_path, "AAA")
    _write_price_file(tmp_path, "BBB")
    _write_price_file(tmp_path, "CCC")

    panel = load_price_panel(tmp_path, limit=2)
    captured = capsys.readouterr()

    assert panel["ticker"].unique().tolist() == ["AAA", "BBB"]
    assert "[data] using 2 ticker files from 3 available (limit=2)" in captured.out


def test_load_price_panel_limit_above_file_count_uses_available_count(
    tmp_path: Path,
    capsys,
) -> None:
    _write_price_file(tmp_path, "AAA")
    _write_price_file(tmp_path, "BBB")

    panel = load_price_panel(tmp_path, limit=10)
    captured = capsys.readouterr()

    assert panel["ticker"].nunique() == 2
    assert "[data] using 2 ticker files from 2 available (limit=10)" in captured.out


def test_download_yahoo_prices_can_continue_after_empty_response(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeYfinance:
        @staticmethod
        def download(*args, **kwargs) -> pd.DataFrame:
            return pd.DataFrame()

    monkeypatch.setattr(yahoo, "_import_yfinance", lambda: FakeYfinance)

    results = yahoo.download_yahoo_prices(
        tickers=["MISSING"],
        output_dir=tmp_path,
        continue_on_error=True,
    )

    assert results[0].status == "failed"
    assert results[0].ticker == "MISSING"
    assert "No price data returned" in str(results[0].error)


def _write_price_file(path: Path, ticker: str) -> None:
    frame = pd.DataFrame(
        {
            "ticker": [ticker, ticker],
            "date": ["2025-01-02", "2025-01-03"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "adj_close": [100.5, 101.5],
            "volume": [1_000, 2_000],
        }
    )
    frame.to_csv(path / f"{ticker}.csv", index=False)


def test_wrds_sp500_intervals_parse_membership_rows() -> None:
    html = """
    <table>
      <tr>
        <th>Added/Removed</th><th>PERMNO</th><th>Company</th>
        <th>Ticker</th><th>SP500 Start</th><th>SP500 End</th>
      </tr>
      <tr>
        <td>Added</td><td>12345</td><td>Legacy Co</td>
        <td>OLD, NEW</td><td>1959-04-01</td><td>2022-12-30</td>
      </tr>
      <tr>
        <td>Removed</td><td>22222</td><td>Exit Co</td>
        <td>EXT</td><td>1957-03-01</td><td>1959-12-31</td>
      </tr>
    </table>
    """

    intervals = wrds_sp500_intervals(html)

    assert intervals["ticker_raw"].tolist() == ["OLD, NEW"]
    assert intervals["permno"].astype(str).tolist() == ["12345"]


def test_wikipedia_change_events_parse_added_and_removed_tickers() -> None:
    html = """
    <table>
      <tr><th>Symbol</th><th>Security</th><th>Date added</th></tr>
      <tr><td>ABC</td><td>ABC Corp</td><td>2023-01-01</td></tr>
    </table>
    <table>
      <tr>
        <th>Effective Date</th><th>Added Ticker</th><th>Added Security</th>
        <th>Removed Ticker</th><th>Removed Security</th><th>Reason</th>
      </tr>
      <tr>
        <td>January 2, 2024</td><td>ADD</td><td>Add Co</td>
        <td>REM</td><td>Remove Co</td><td>Market capitalization change.</td>
      </tr>
    </table>
    """

    events = wikipedia_sp500_change_events(html)

    assert events[["event_type", "ticker"]].to_dict("records") == [
        {"event_type": "added", "ticker": "ADD"},
        {"event_type": "removed", "ticker": "REM"},
    ]


def test_build_sp500_universe_writes_candidate_csv(tmp_path: Path) -> None:
    wrds_html = """
    <table>
      <tr>
        <th>Added/Removed</th><th>PERMNO</th><th>Company</th>
        <th>Ticker</th><th>SP500 Start</th><th>SP500 End</th>
      </tr>
      <tr>
        <td>Added</td><td>12345</td><td>Legacy Co</td>
        <td>OLD</td><td>1959-04-01</td><td>2022-12-30</td>
      </tr>
    </table>
    """
    wiki_html = """
    <table>
      <tr><th>Symbol</th><th>Security</th><th>Date added</th></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>2010-02-16</td></tr>
    </table>
    <table>
      <tr>
        <th>Effective Date</th><th>Added Ticker</th><th>Added Security</th>
        <th>Removed Ticker</th><th>Removed Security</th><th>Reason</th>
      </tr>
      <tr>
        <td>January 2, 2024</td><td>ADD</td><td>Add Co</td>
        <td>OLD</td><td>Legacy Co</td><td>Market capitalization change.</td>
      </tr>
    </table>
    """
    output = tmp_path / "sp500.csv"

    result = build_sp500_universe(
        output_path=output,
        wrds_source=wrds_html,
        wikipedia_source=wiki_html,
    )
    universe = pd.read_csv(output)
    lookup = json.loads(output.with_suffix(".json").read_text())

    assert result.unique_tickers == 3
    assert result.json_path == output.with_suffix(".json")
    assert set(universe["ticker"]) == {"ADD", "BRK-B", "OLD"}
    assert universe.loc[universe["ticker"].eq("OLD"), "left_sp500_year"].tolist() == [2024]
    assert lookup["tickers"]["OLD"]["sp500_periods"][0]["From"] == 1959
    assert lookup["tickers"]["OLD"]["sp500_periods"][0]["To"] == 2024
    assert lookup["tickers"]["BRK-B"]["sp500_periods"][0]["To"] is None


def test_download_sp500_prices_resumes_from_json_and_notes_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "A",
                "ticker_raw": "A",
                "yahoo_ticker_candidates": "A",
                "company": "Downloaded Co",
                "permno": "",
                "sp500_start_date": "2000-01-01",
                "sp500_end_date": "",
                "first_year_available": "",
                "entered_sp500_year": 2000,
                "left_sp500_year": "",
                "delisted_or_shutdown_year": "",
                "is_current_sp500": True,
                "membership_status": "current",
                "source": "test",
                "notes": "",
            },
            {
                "ticker": "ZZZ",
                "ticker_raw": "ZZZ",
                "yahoo_ticker_candidates": "ZZZ",
                "company": "Unavailable Co",
                "permno": "",
                "sp500_start_date": "1960-01-01",
                "sp500_end_date": "1970-01-01",
                "first_year_available": "",
                "entered_sp500_year": 1960,
                "left_sp500_year": 1970,
                "delisted_or_shutdown_year": "",
                "is_current_sp500": False,
                "membership_status": "closed",
                "source": "test",
                "notes": "",
            },
        ]
    )
    universe_path = tmp_path / "sp500.csv"
    lookup_path = tmp_path / "sp500.json"
    price_dir = tmp_path / "prices"
    manifest_path = tmp_path / "manifest.csv"
    unavailable_path = tmp_path / "unavailable.csv"
    validation_report_path = tmp_path / "validation.csv"
    price_dir.mkdir()
    universe.to_csv(universe_path, index=False)
    (price_dir / "A.csv").write_text(
        "ticker,date,open,high,low,close,adj_close,volume\n"
        "A,2025-01-02,1,2,0.5,1.5,1.4,100\n"
    )
    calls = []

    def fake_download_yahoo_prices(**kwargs):
        calls.extend(kwargs["tickers"])
        return [
            yahoo.YahooPriceDownloadResult(
                ticker="ZZZ",
                path=price_dir / "ZZZ.csv",
                rows=0,
                start_date=None,
                end_date=None,
                status="failed",
                error="No price data returned for ZZZ.",
            )
        ]

    monkeypatch.setattr(sp500_universe, "download_yahoo_prices", fake_download_yahoo_prices)
    progress_messages = []

    results = download_sp500_universe_prices(
        universe_path=universe_path,
        output_dir=price_dir,
        manifest_path=manifest_path,
        lookup_json_path=lookup_path,
        unavailable_path=unavailable_path,
        validation_report_path=validation_report_path,
        metadata_validation="none",
        progress_printer=progress_messages.append,
    )

    assert calls == ["ZZZ"]
    assert [result.status for result in results] == ["json_skipped", "failed"]
    assert "[1/2] A: json_skipped" in progress_messages[0]
    assert "ETA=" in progress_messages[1]
    lookup = json.loads(lookup_path.read_text())
    assert lookup["tickers"]["A"]["download"]["status"] == "downloaded"
    assert lookup["tickers"]["ZZZ"]["download"]["status"] == "failed"
    unavailable = pd.read_csv(unavailable_path)
    assert unavailable["ticker"].tolist() == ["ZZZ"]
    assert unavailable["reason"].tolist() == ["yahoo_download_failed_delisted_or_historical"]

    calls.clear()
    resumed = download_sp500_universe_prices(
        universe_path=universe_path,
        output_dir=price_dir,
        manifest_path=manifest_path,
        lookup_json_path=lookup_path,
        unavailable_path=unavailable_path,
        validation_report_path=validation_report_path,
        metadata_validation="none",
        progress_printer=None,
    )

    assert calls == []
    assert [result.status for result in resumed] == ["json_skipped", "json_failed_skipped"]


def test_validate_yahoo_price_file_rejects_reused_delisted_symbol(tmp_path: Path) -> None:
    price_path = tmp_path / "POM.csv"
    price_path.write_text(
        "ticker,date,open,high,low,close,adj_close,volume\n"
        "POM,2025-01-02,1,2,0.5,1.5,1.4,1000\n"
        "POM,2025-01-03,1,2,0.5,1.5,1.4,1000\n"
    )
    lookup_entry = {
        "companies": ["PEPCO HOLDINGS INC"],
        "sp500_periods": [
            {
                "From": 2007,
                "To": 2016,
                "from_date": "2007-01-01",
                "to_date": "2016-03-23",
            }
        ],
    }

    validation = validate_yahoo_price_file(
        ticker="POM",
        path=price_path,
        lookup_entry=lookup_entry,
        end_date="2025-12-31",
        metadata_fetcher=lambda ticker: {
            "quoteType": "EQUITY",
            "longName": "Pomdoctor Limited",
            "exchange": "NGM",
            "currency": "USD",
        },
    )

    assert validation["status"] == "invalid"
    assert "price_dates_do_not_overlap_sp500_membership" in validation["reasons"]
    assert "yahoo_name_mismatch" in validation["reasons"]
    assert validation["quote_type"] == "EQUITY"


def test_download_sp500_prices_quarantines_invalid_yahoo_symbol(
    tmp_path: Path,
    monkeypatch,
) -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "BAD",
                "ticker_raw": "BAD",
                "yahoo_ticker_candidates": "BAD",
                "company": "Old Index Co",
                "permno": "",
                "sp500_start_date": "1960-01-01",
                "sp500_end_date": "1970-01-01",
                "first_year_available": "",
                "entered_sp500_year": 1960,
                "left_sp500_year": 1970,
                "delisted_or_shutdown_year": "",
                "is_current_sp500": False,
                "membership_status": "closed",
                "source": "test",
                "notes": "",
            }
        ]
    )
    universe_path = tmp_path / "sp500.csv"
    lookup_path = tmp_path / "sp500.json"
    price_dir = tmp_path / "prices"
    manifest_path = tmp_path / "manifest.csv"
    unavailable_path = tmp_path / "unavailable.csv"
    validation_report_path = tmp_path / "validation.csv"
    quarantine_dir = price_dir / "_quarantine"
    price_dir.mkdir()
    universe.to_csv(universe_path, index=False)

    def fake_download_yahoo_prices(**kwargs):
        price_path = Path(kwargs["output_dir"]) / "BAD.csv"
        price_path.write_text(
            "ticker,date,open,high,low,close,adj_close,volume\n"
            "BAD,2025-01-02,1,2,0.5,1.5,1.4,1000\n"
            "BAD,2025-01-03,1,2,0.5,1.5,1.4,1000\n"
        )
        return [
            yahoo.YahooPriceDownloadResult(
                ticker="BAD",
                path=price_path,
                rows=2,
                start_date="2025-01-02",
                end_date="2025-01-03",
                status="downloaded",
                error=None,
            )
        ]

    monkeypatch.setattr(sp500_universe, "download_yahoo_prices", fake_download_yahoo_prices)

    results = download_sp500_universe_prices(
        universe_path=universe_path,
        output_dir=price_dir,
        manifest_path=manifest_path,
        lookup_json_path=lookup_path,
        unavailable_path=unavailable_path,
        validation_report_path=validation_report_path,
        quarantine_dir=quarantine_dir,
        progress_printer=None,
        metadata_fetcher=lambda ticker: {
            "quoteType": "EQUITY",
            "longName": "Unrelated Public Company",
            "exchange": "NMS",
            "currency": "USD",
        },
    )

    assert [result.status for result in results] == ["invalid"]
    assert not (price_dir / "BAD.csv").exists()
    assert (quarantine_dir / "BAD.csv").exists()
    lookup = json.loads(lookup_path.read_text())
    assert lookup["tickers"]["BAD"]["download"]["status"] == "invalid"
    assert lookup["tickers"]["BAD"]["validation"]["status"] == "invalid"
    unavailable = pd.read_csv(unavailable_path)
    assert unavailable["ticker"].tolist() == ["BAD"]
    assert unavailable["reason"].tolist() == ["yahoo_validation_failed"]
    validation = pd.read_csv(validation_report_path)
    assert validation["ticker"].tolist() == ["BAD"]
    assert validation["status"].tolist() == ["invalid"]


def test_universe_json_recovers_from_empty_file(tmp_path: Path) -> None:
    universe = pd.DataFrame(
        [
            {
                "ticker": "A",
                "ticker_raw": "A",
                "yahoo_ticker_candidates": "A",
                "company": "Example Co",
                "permno": "",
                "sp500_start_date": "2000-01-01",
                "sp500_end_date": "",
                "first_year_available": "",
                "entered_sp500_year": 2000,
                "left_sp500_year": "",
                "delisted_or_shutdown_year": "",
                "is_current_sp500": True,
                "membership_status": "current",
                "source": "test",
                "notes": "",
            }
        ]
    )
    json_path = tmp_path / "sp500.json"
    json_path.write_text("")

    lookup = ensure_sp500_universe_json(universe, json_path=json_path)

    assert lookup["tickers"]["A"]["sp500_periods"][0]["From"] == 2000
    assert json.loads(json_path.read_text())["tickers"]["A"]["is_current_sp500"] is True


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


def test_price_loader_synthesizes_adj_close_and_preserves_panel_extras(tmp_path: Path) -> None:
    price_dir = tmp_path / "prices"
    price_dir.mkdir()
    (price_dir / "panel.csv").write_text(
        "ticker,date,open,high,low,close,volume,sector,beta,pe_ratio,debt_to_equity\n"
        "AAPL,2025-01-02,1,2,0.5,1.5,100,technology,1.2,24,0.4\n"
    )

    panel = load_price_panel(price_dir, macro_data_path=None)

    assert panel["Price Adj_Close"].tolist() == [1.5]
    assert panel["sector"].tolist() == ["technology"]
    assert panel["beta"].tolist() == [1.2]


def test_sector_one_hot_features_use_configured_static_columns() -> None:
    frame = pd.DataFrame({"sector": ["technology", "consumer", "other", None]})

    encoded = add_sector_one_hot_features(
        frame,
        sector_column="sector",
        static_feature_columns=[
            "sector_consumer",
            "sector_technology",
            "sector_unknown",
        ],
    )

    assert encoded[["sector_consumer", "sector_technology", "sector_unknown"]].to_dict(
        "records"
    ) == [
        {"sector_consumer": 0.0, "sector_technology": 1.0, "sector_unknown": 0.0},
        {"sector_consumer": 1.0, "sector_technology": 0.0, "sector_unknown": 0.0},
        {"sector_consumer": 0.0, "sector_technology": 0.0, "sector_unknown": 1.0},
        {"sector_consumer": 0.0, "sector_technology": 0.0, "sector_unknown": 1.0},
    ]
