"""Supervised return-prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def positive_prediction_share(_: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_pred > 0.0))


def spearman_rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def top_bottom_quantile_spread(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float = 0.2,
) -> float:
    if len(y_true) == 0:
        return float("nan")
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).dropna()
    if len(frame) < 2:
        return float("nan")
    lower = frame["y_pred"].quantile(quantile)
    upper = frame["y_pred"].quantile(1.0 - quantile)
    top = frame.loc[frame["y_pred"] >= upper, "y_true"].mean()
    bottom = frame.loc[frame["y_pred"] <= lower, "y_true"].mean()
    return float(top - bottom)


METRIC_REGISTRY = {
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
    "correlation": correlation,
    "directional_accuracy": directional_accuracy,
    "mda": directional_accuracy,
    "positive_prediction_share": positive_prediction_share,
    "spearman_rank_ic": spearman_rank_ic,
    "rank_correlation": spearman_rank_ic,
    "top_bottom_quantile_spread": top_bottom_quantile_spread,
}

PREDICTION_FRAME_METRICS = {
    "long_short_decile_return",
    "long_short_decile_cagr",
    "long_short_decile_annualized_volatility",
    "long_short_decile_sharpe",
    "long_short_decile_sortino",
    "long_short_decile_max_drawdown",
    "long_short_decile_hit_rate",
    "long_short_decile_turnover",
    "long_short_decile_transaction_cost_adjusted_return",
    "long_short_decile_transaction_cost_adjusted_cagr",
    "long_only_decile_return",
    "equal_weight_benchmark_return",
    "sector_neutral_spearman_rank_ic",
}


def compute_metrics(
    y_true: np.ndarray | list[float],
    y_pred: np.ndarray | list[float],
    metric_names: list[str],
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
    y_true_arr = y_true_arr[mask]
    y_pred_arr = y_pred_arr[mask]
    results = {}
    for name in metric_names:
        if name not in METRIC_REGISTRY:
            raise ValueError(f"Unknown evaluation metric: {name}")
        results[name] = METRIC_REGISTRY[name](y_true_arr, y_pred_arr)
    return results


def compute_prediction_frame_metrics(
    predictions: pd.DataFrame,
    metric_names: list[str],
    portfolio_quantile: float = 0.1,
    annualization_factor: int = 252,
    transaction_cost_bps: float = 0.0,
) -> dict[str, float]:
    """Compute array metrics plus optional portfolio metrics from prediction rows."""

    known_metrics = set(METRIC_REGISTRY) | PREDICTION_FRAME_METRICS
    missing = [name for name in metric_names if name not in known_metrics]
    if missing:
        raise ValueError(f"Unknown evaluation metric: {missing[0]}")

    results: dict[str, float] = {}
    array_names = [name for name in metric_names if name in METRIC_REGISTRY]
    if array_names:
        results.update(
            compute_metrics(
                predictions["y_true"].to_numpy(),
                predictions["y_pred"].to_numpy(),
                array_names,
            )
        )

    requested_frame_metrics = [name for name in metric_names if name in PREDICTION_FRAME_METRICS]
    if not requested_frame_metrics:
        return results

    decile_returns = long_short_returns(predictions, quantile=0.1)
    long_only_decile = long_only_returns(predictions, quantile=0.1)
    benchmark_returns = equal_weight_benchmark_returns(predictions)
    net_decile_returns = transaction_cost_adjusted_returns(
        decile_returns,
        transaction_cost_bps=transaction_cost_bps,
    )

    frame_values = {
        "long_short_decile_return": _mean_return(decile_returns, "long_short_return"),
        "long_short_decile_cagr": _cagr(
            decile_returns.get("long_short_return"),
            annualization_factor,
        ),
        "long_short_decile_annualized_volatility": _annualized_volatility(
            decile_returns.get("long_short_return"),
            annualization_factor,
        ),
        "long_short_decile_sharpe": _sharpe(
            decile_returns.get("long_short_return"),
            annualization_factor,
        ),
        "long_short_decile_sortino": _sortino(
            decile_returns.get("long_short_return"),
            annualization_factor,
        ),
        "long_short_decile_max_drawdown": _max_drawdown(
            decile_returns.get("long_short_return")
        ),
        "long_short_decile_hit_rate": _hit_rate(decile_returns.get("long_short_return")),
        "long_short_decile_turnover": _mean_return(decile_returns, "long_short_turnover"),
        "long_short_decile_transaction_cost_adjusted_return": _mean_return(
            net_decile_returns,
            "net_long_short_return",
        ),
        "long_short_decile_transaction_cost_adjusted_cagr": _cagr(
            net_decile_returns.get("net_long_short_return"),
            annualization_factor,
        ),
        "long_only_decile_return": _mean_return(long_only_decile, "long_only_return"),
        "equal_weight_benchmark_return": _series_mean(benchmark_returns),
        "sector_neutral_spearman_rank_ic": sector_neutral_spearman_rank_ic(predictions),
    }
    for name in requested_frame_metrics:
        results[name] = frame_values[name]
    return results


def long_short_returns(
    predictions: pd.DataFrame,
    quantile: float = 0.1,
) -> pd.DataFrame:
    prepared = _prepare_prediction_frame(predictions)
    date_column = _date_column(prepared)
    if prepared.empty or date_column is None:
        return pd.DataFrame(
            columns=["date", "long_short_return", "long_short_turnover"]
        )

    rows = []
    memberships = []
    for date_value, group in prepared.groupby(date_column, dropna=False, sort=True):
        valid = group.dropna(subset=["y_pred", "y_true_return"])
        if len(valid) < 2:
            continue
        lower = valid["y_pred"].quantile(quantile)
        upper = valid["y_pred"].quantile(1.0 - quantile)
        top = valid.loc[valid["y_pred"] >= upper]
        bottom = valid.loc[valid["y_pred"] <= lower]
        if top.empty or bottom.empty:
            continue
        rows.append(
            {
                "date": date_value,
                "long_short_return": float(
                    top["y_true_return"].mean() - bottom["y_true_return"].mean()
                ),
            }
        )
        if "asset_id" in valid:
            memberships.append(
                {
                    "date": date_value,
                    "top": set(top["asset_id"].dropna().astype(str)),
                    "bottom": set(bottom["asset_id"].dropna().astype(str)),
                }
            )
    returns = pd.DataFrame(rows)
    if returns.empty:
        return pd.DataFrame(
            columns=["date", "long_short_return", "long_short_turnover"]
        )
    returns = returns.sort_values("date").reset_index(drop=True)
    returns["long_short_turnover"] = _turnover_by_date(memberships, returns["date"])
    return returns


def long_only_returns(predictions: pd.DataFrame, quantile: float = 0.1) -> pd.DataFrame:
    prepared = _prepare_prediction_frame(predictions)
    date_column = _date_column(prepared)
    if prepared.empty or date_column is None:
        return pd.DataFrame(columns=["date", "long_only_return"])
    rows = []
    for date_value, group in prepared.groupby(date_column, dropna=False, sort=True):
        valid = group.dropna(subset=["y_pred", "y_true_return"])
        if len(valid) < 2:
            continue
        upper = valid["y_pred"].quantile(1.0 - quantile)
        top = valid.loc[valid["y_pred"] >= upper]
        if not top.empty:
            rows.append({"date": date_value, "long_only_return": float(top["y_true_return"].mean())})
    return pd.DataFrame(rows)


def equal_weight_benchmark_returns(predictions: pd.DataFrame) -> pd.Series:
    prepared = _prepare_prediction_frame(predictions)
    date_column = _date_column(prepared)
    if prepared.empty or date_column is None:
        return pd.Series(dtype=float)
    return (
        prepared.dropna(subset=["y_true_return"])
        .groupby(date_column, sort=True)["y_true_return"]
        .mean()
    )


def transaction_cost_adjusted_returns(
    returns: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.DataFrame:
    result = returns.copy()
    if result.empty:
        result["net_long_short_return"] = pd.Series(dtype=float)
        return result
    cost_rate = float(transaction_cost_bps) / 10_000.0
    turnover = pd.to_numeric(result.get("long_short_turnover", 0.0), errors="coerce").fillna(0.0)
    result["net_long_short_return"] = result["long_short_return"] - turnover * cost_rate
    return result


def sector_neutral_spearman_rank_ic(predictions: pd.DataFrame) -> float:
    prepared = _prepare_prediction_frame(predictions)
    date_column = _date_column(prepared)
    if prepared.empty or date_column is None or "sector" not in prepared:
        return float("nan")
    values = []
    for _, date_group in prepared.groupby(date_column, dropna=False, sort=True):
        residualized = date_group.dropna(subset=["y_true", "y_pred", "sector"]).copy()
        if len(residualized) < 2:
            continue
        for column in ("y_true", "y_pred"):
            residualized[column] = residualized[column] - residualized.groupby("sector")[
                column
            ].transform("mean")
        corr = residualized["y_true"].corr(residualized["y_pred"], method="spearman")
        if pd.notna(corr):
            values.append(float(corr))
    return float(np.mean(values)) if values else float("nan")


def _prepare_prediction_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    prepared = predictions.copy()
    for column in ("y_true", "y_pred"):
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if "y_true" in prepared and "y_true_return" not in prepared:
        prepared["y_true_return"] = np.expm1(prepared["y_true"])
    if "y_pred" in prepared and "y_pred_return" not in prepared:
        prepared["y_pred_return"] = np.expm1(prepared["y_pred"])
    return prepared


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in ("anchor_date", "target_date", "date"):
        if column in frame:
            return column
    return None


def _turnover_by_date(memberships: list[dict[str, object]], dates: pd.Series) -> list[float]:
    by_date = {item["date"]: item for item in memberships}
    previous_top: set[str] | None = None
    previous_bottom: set[str] | None = None
    turnovers = []
    for date_value in dates:
        membership = by_date.get(date_value)
        if membership is None:
            turnovers.append(float("nan"))
            continue
        top = membership["top"]
        bottom = membership["bottom"]
        turnover = np.nan
        if previous_top is not None and previous_bottom is not None:
            top_turnover = _set_turnover(previous_top, top)
            bottom_turnover = _set_turnover(previous_bottom, bottom)
            turnover = float(np.nanmean([top_turnover, bottom_turnover]))
        turnovers.append(turnover)
        previous_top = top
        previous_bottom = bottom
    return turnovers


def _set_turnover(previous: set[str], current: set[str]) -> float:
    if not current:
        return float("nan")
    return 1.0 - len(current.intersection(previous)) / len(current)


def _mean_return(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return float("nan")
    return _series_mean(frame[column])


def _series_mean(values: pd.Series | None) -> float:
    if values is None:
        return float("nan")
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else float("nan")


def _cagr(values: pd.Series | None, annualization_factor: int) -> float:
    returns = _clean_return_series(values)
    if returns.empty:
        return float("nan")
    terminal = float((1.0 + returns).prod())
    if terminal <= 0.0:
        return float("nan")
    return float(terminal ** (annualization_factor / len(returns)) - 1.0)


def _annualized_volatility(values: pd.Series | None, annualization_factor: int) -> float:
    returns = _clean_return_series(values)
    if len(returns) < 2:
        return float("nan")
    return float(returns.std(ddof=1) * np.sqrt(annualization_factor))


def _sharpe(values: pd.Series | None, annualization_factor: int) -> float:
    returns = _clean_return_series(values)
    volatility = _annualized_volatility(returns, annualization_factor)
    if not np.isfinite(volatility) or volatility == 0.0:
        return float("nan")
    return float(returns.mean() * annualization_factor / volatility)


def _sortino(values: pd.Series | None, annualization_factor: int) -> float:
    returns = _clean_return_series(values)
    downside = returns.loc[returns < 0.0]
    if len(downside) < 2:
        return float("nan")
    downside_vol = downside.std(ddof=1) * np.sqrt(annualization_factor)
    if not np.isfinite(downside_vol) or downside_vol == 0.0:
        return float("nan")
    return float(returns.mean() * annualization_factor / downside_vol)


def _max_drawdown(values: pd.Series | None) -> float:
    returns = _clean_return_series(values)
    if returns.empty:
        return float("nan")
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _hit_rate(values: pd.Series | None) -> float:
    returns = _clean_return_series(values)
    if returns.empty:
        return float("nan")
    return float((returns > 0.0).mean())


def _clean_return_series(values: pd.Series | None) -> pd.Series:
    if values is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
