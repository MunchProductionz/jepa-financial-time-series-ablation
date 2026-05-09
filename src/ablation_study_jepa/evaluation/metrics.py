"""Supervised return-prediction metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred - y_true) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="pearson"))


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


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
    "mae": mae,
    "correlation": correlation,
    "directional_accuracy": directional_accuracy,
    "spearman_rank_ic": spearman_rank_ic,
    "rank_correlation": spearman_rank_ic,
    "top_bottom_quantile_spread": top_bottom_quantile_spread,
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

