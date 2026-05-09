"""Deterministic synthetic OHLCV panel for smoke tests and CI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_panel(
    tickers: list[str] | tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMZN"),
    start: str = "2015-01-01",
    periods: int = 900,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=periods)
    sectors = ["technology", "technology", "semiconductors", "consumer"]
    frames = []
    market = rng.normal(0.0003, 0.009, size=periods)

    for idx, ticker in enumerate(tickers):
        beta = 0.8 + 0.2 * idx
        idio = rng.normal(0.0001 * (idx + 1), 0.012 + 0.002 * idx, size=periods)
        returns = beta * market + idio
        close = 100.0 * np.exp(np.cumsum(returns))
        open_ = close * (1.0 + rng.normal(0.0, 0.003, size=periods))
        high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=periods))
        low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=periods))
        volume = rng.lognormal(mean=15.0 + idx * 0.05, sigma=0.2, size=periods).astype(int)
        pe_ratio = 18.0 + idx + rng.normal(0.0, 1.0, size=periods)
        debt_to_equity = 0.4 + idx * 0.05 + rng.normal(0.0, 0.02, size=periods)
        interest_rate = 0.02 + np.linspace(0, 0.01, periods) + rng.normal(0.0, 0.001, periods)
        vix = 18 + 3 * np.sin(np.arange(periods) / 60) + rng.normal(0.0, 1.0, periods)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                    "sector": sectors[idx % len(sectors)],
                    "beta": beta,
                    "pe_ratio": pe_ratio,
                    "debt_to_equity": debt_to_equity,
                    "interest_rate": interest_rate,
                    "vix": vix,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
