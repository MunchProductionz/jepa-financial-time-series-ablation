"""FRED-MD download and parsing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

DEFAULT_FRED_MD_VINTAGE = "current"
DEFAULT_START_DATE = "1960-01-01"
DEFAULT_END_DATE = "2025-12-31"
DEFAULT_OUTPUT_DIR = Path("data/macro/fred_md")

FRED_MD_FILE_BASE_URL = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly"
FRED_MD_MEDIA_BASE_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/"
    "research/fred-md/monthly"
)
FRED_DATABASES_PAGE_URL = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"


@dataclass(frozen=True)
class FredMDDownloadResult:
    """Summary for a FRED-MD download."""

    vintage: str
    source_url: str
    raw_path: Path
    data_path: Path
    transformations_path: Path | None
    rows: int
    series_count: int
    start_date: str | None
    end_date: str | None
    status: str


def _normalize_vintage(vintage: str) -> str:
    value = vintage.strip().lower().removesuffix(".csv")
    if not value:
        raise ValueError("FRED-MD vintage must be `current` or YYYY-MM.")
    return value


def fred_md_candidate_urls(vintage: str) -> list[str]:
    """Return official FRED-MD monthly CSV URLs to try for a vintage."""

    normalized = _normalize_vintage(vintage)
    filename = "current.csv" if normalized == "current" else f"{normalized}.csv"
    urls = [f"{FRED_MD_FILE_BASE_URL}/{filename}"]
    if normalized != "current":
        urls.append(f"{FRED_MD_MEDIA_BASE_URL}/{normalized}-md.csv")
    return urls


class _FredMDLinkParser(HTMLParser):
    def __init__(self, label: str) -> None:
        super().__init__()
        self.label = label
        self._current_href: str | None = None
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        self._current_href = dict(attrs).get("href")

    def handle_data(self, data: str) -> None:
        if data.strip() == self.label and self._current_href is not None:
            self.hrefs.append(self._current_href)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._current_href = None


def discover_fred_md_page_url(vintage: str) -> str | None:
    """Discover the current FRED-MD CSV URL from the St. Louis Fed page."""

    normalized = _normalize_vintage(vintage)
    label = "current.csv" if normalized == "current" else f"{normalized}.csv"
    content, _ = _download_bytes([FRED_DATABASES_PAGE_URL])
    parser = _FredMDLinkParser(label)
    parser.feed(content.decode("utf-8", errors="replace"))
    for href in parser.hrefs:
        absolute_url = urljoin(FRED_DATABASES_PAGE_URL, href)
        if "/fred-md/monthly/" in absolute_url and absolute_url.endswith(".csv"):
            return absolute_url
        if "/fred-md/monthly/" in absolute_url and ".csv?" in absolute_url:
            return absolute_url
    return None


def _download_with_curl_cffi(url: str) -> bytes:
    try:
        from curl_cffi import requests
    except ImportError as exc:
        raise RuntimeError("curl-cffi is not installed") from exc

    response = requests.get(url, impersonate="chrome", timeout=60)
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code}: {response.reason}")
    return response.content


def _download_with_urllib(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ablation-study-jepa/0.1"})
    with urlopen(request, timeout=60) as response:
        return response.read()


def _download_bytes(urls: list[str]) -> tuple[bytes, str]:
    errors = []
    for url in urls:
        for downloader in (_download_with_curl_cffi, _download_with_urllib):
            try:
                return downloader(url), url
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                errors.append(f"{url} via {downloader.__name__}: {exc}")
    joined_errors = "\n".join(errors)
    raise RuntimeError(f"Could not download FRED-MD from any candidate URL:\n{joined_errors}")


def parse_fred_md_file(
    raw_path: str | Path,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Parse a raw FRED-MD CSV and return filtered data plus transformation codes."""

    raw = pd.read_csv(raw_path)
    if raw.empty:
        raise ValueError(f"FRED-MD file is empty: {raw_path}")

    date_column = raw.columns[0]
    raw = raw.rename(columns={date_column: "date"})
    parsed_dates = pd.to_datetime(raw["date"], errors="coerce", format="mixed")

    metadata_rows = raw.loc[parsed_dates.isna()]
    transformations = None
    if not metadata_rows.empty:
        transform_row = metadata_rows.iloc[0]
        transform_codes = pd.to_numeric(transform_row.iloc[1:], errors="coerce").astype("Int64")
        transformations = pd.DataFrame(
            {
                "series": raw.columns[1:],
                "transform_code": transform_codes.to_numpy(),
            }
        )
        transformations = transformations.dropna(subset=["transform_code"], how="all")

    data = raw.loc[parsed_dates.notna()].copy()
    data_dates = pd.to_datetime(data["date"], errors="raise", format="mixed")
    mask = (data_dates >= pd.Timestamp(start_date)) & (data_dates <= pd.Timestamp(end_date))
    data = data.loc[mask].copy()
    data_dates = data_dates.loc[mask]
    data["date"] = data_dates.dt.strftime("%Y-%m-%d")

    for column in data.columns[1:]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    return data.reset_index(drop=True), transformations


def _date_bounds(frame: pd.DataFrame) -> tuple[str | None, str | None]:
    if frame.empty:
        return None, None
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None, None
    return dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")


def download_fred_md(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    vintage: str = DEFAULT_FRED_MD_VINTAGE,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    overwrite: bool = False,
) -> FredMDDownloadResult:
    """Download a raw FRED-MD CSV and write a date-filtered parsed CSV."""

    normalized_vintage = _normalize_vintage(vintage)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_filename = "current.csv" if normalized_vintage == "current" else f"{normalized_vintage}.csv"
    raw_path = output_dir / raw_filename
    source_url = fred_md_candidate_urls(normalized_vintage)[0]
    status = "reused"
    if overwrite or not raw_path.exists():
        candidate_urls = fred_md_candidate_urls(normalized_vintage)
        try:
            discovered_url = discover_fred_md_page_url(normalized_vintage)
        except RuntimeError:
            discovered_url = None
        if discovered_url is not None and discovered_url not in candidate_urls:
            candidate_urls.append(discovered_url)
        content, source_url = _download_bytes(candidate_urls)
        raw_path.write_bytes(content)
        status = "downloaded"

    data, transformations = parse_fred_md_file(raw_path, start_date=start_date, end_date=end_date)

    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year
    data_path = output_dir / f"fred_md_{start_year}_{end_year}.csv"
    data.to_csv(data_path, index=False)

    transformations_path = None
    if transformations is not None:
        transformations_path = output_dir / f"fred_md_{normalized_vintage}_transformations.csv"
        transformations.to_csv(transformations_path, index=False)

    first_date, last_date = _date_bounds(data)
    return FredMDDownloadResult(
        vintage=normalized_vintage,
        source_url=source_url,
        raw_path=raw_path,
        data_path=data_path,
        transformations_path=transformations_path,
        rows=len(data),
        series_count=max(len(data.columns) - 1, 0),
        start_date=first_date,
        end_date=last_date,
        status=status,
    )
