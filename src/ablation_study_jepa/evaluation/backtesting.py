"""Small research backtest helpers."""

from __future__ import annotations

import pandas as pd


def daily_long_short_spread(
    predictions: pd.DataFrame,
    date_column: str = "anchor_date",
    realized_column: str = "y_true",
    score_column: str = "y_pred",
    quantile: float = 0.2,
) -> pd.Series:
    """Compute an equal-weight top-minus-bottom spread by date.

    This is a diagnostic research metric, not a production backtest.
    """

    spreads = {}
    for date, group in predictions.groupby(date_column):
        if len(group) < 2:
            continue
        lower = group[score_column].quantile(quantile)
        upper = group[score_column].quantile(1.0 - quantile)
        top = group.loc[group[score_column] >= upper, realized_column].mean()
        bottom = group.loc[group[score_column] <= lower, realized_column].mean()
        spreads[date] = top - bottom
    return pd.Series(spreads).sort_index()

