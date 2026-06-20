"""LaTeX table exports for experiment comparison artifacts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import pandas as pd

from ablation_study_jepa.evaluation.compare_runs import (
    comparison_metrics_frame,
    discover_run_dirs,
    window_metrics_wide_frame,
)

MetricDirection = Literal["max", "min", "neutral"]


DEFAULT_METRIC_DIRECTIONS: dict[str, MetricDirection] = {
    "mse": "min",
    "rmse": "min",
    "mae": "min",
    "prediction_loss": "min",
    "supervised_loss": "min",
    "total_loss": "min",
    "correlation": "max",
    "spearman_rank_ic": "max",
    "rank_correlation": "max",
    "sector_neutral_spearman_rank_ic": "max",
    "directional_accuracy": "max",
    "mda": "max",
    "top_bottom_quantile_spread": "max",
    "long_short_decile_return": "max",
    "long_short_decile_cagr": "max",
    "long_short_decile_annualized_volatility": "min",
    "long_short_decile_sharpe": "max",
    "long_short_decile_sortino": "max",
    # The implementation stores drawdown as a negative return, so closer to zero is better.
    "long_short_decile_max_drawdown": "max",
    "long_short_decile_hit_rate": "max",
    "long_short_decile_turnover": "min",
    "long_short_decile_transaction_cost_adjusted_return": "max",
    "long_short_decile_transaction_cost_adjusted_cagr": "max",
    "long_only_decile_return": "max",
    "equal_weight_benchmark_return": "max",
}

DEFAULT_METRIC_LABELS = {
    "mse": "MSE",
    "rmse": "RMSE",
    "mae": "MAE",
    "spearman_rank_ic": "Spearman IC",
    "rank_correlation": "Rank correlation",
    "sector_neutral_spearman_rank_ic": "Sector-neutral IC",
    "directional_accuracy": "Directional accuracy",
    "mda": "MDA",
    "positive_prediction_share": "Positive prediction share",
    "top_bottom_quantile_spread": "Top-bottom spread",
    "long_short_decile_return": "Long-short return",
    "long_short_decile_cagr": "Long-short CAGR",
    "long_short_decile_annualized_volatility": "Long-short volatility",
    "long_short_decile_sharpe": "Long-short Sharpe",
    "long_short_decile_sortino": "Long-short Sortino",
    "long_short_decile_max_drawdown": "Long-short max drawdown",
    "long_short_decile_hit_rate": "Long-short hit rate",
    "long_short_decile_turnover": "Long-short turnover",
    "long_short_decile_transaction_cost_adjusted_return": "TC-adjusted return",
    "long_short_decile_transaction_cost_adjusted_cagr": "TC-adjusted CAGR",
    "long_only_decile_return": "Long-only return",
    "equal_weight_benchmark_return": "Benchmark return",
}


def comparison_latex_table(
    frame: pd.DataFrame,
    metric_columns: list[str] | None = None,
    metric_directions: dict[str, MetricDirection] | None = None,
    index_columns: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "{:.4g}",
) -> str:
    """Render a comparison table with metrics as columns and best values in bold."""

    if frame.empty:
        return _empty_table(caption=caption, label=label)

    directions = {**DEFAULT_METRIC_DIRECTIONS, **(metric_directions or {})}
    metric_columns = metric_columns or [column for column in frame.columns if column in directions]
    index_columns = index_columns or ["run_name"]
    columns = [column for column in [*index_columns, *metric_columns] if column in frame.columns]
    table = frame.loc[:, columns].copy()
    bold_cells = _best_value_cells(table, metric_columns, directions)
    return _render_latex_tabular(
        table,
        metric_columns=metric_columns,
        metric_directions=directions,
        bold_cells=bold_cells,
        caption=caption,
        label=label,
        float_format=float_format,
    )


def per_model_latex_table(
    frame: pd.DataFrame,
    metric_columns: list[str] | None = None,
    metric_directions: dict[str, MetricDirection] | None = None,
    index_columns: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
    float_format: str = "{:.4g}",
) -> str:
    """Render one model/config table with metrics as columns."""

    if frame.empty:
        return _empty_table(caption=caption, label=label)
    directions = {**DEFAULT_METRIC_DIRECTIONS, **(metric_directions or {})}
    metric_columns = metric_columns or [column for column in frame.columns if column in directions]
    index_columns = index_columns or ["window_index"]
    columns = [column for column in [*index_columns, *metric_columns] if column in frame.columns]
    return _render_latex_tabular(
        frame.loc[:, columns].copy(),
        metric_columns=metric_columns,
        metric_directions=directions,
        bold_cells=set(),
        caption=caption,
        label=label,
        float_format=float_format,
    )


def export_comparison_latex(
    predictions_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    metric_columns: list[str] | None = None,
    metric_directions: dict[str, MetricDirection] | None = None,
    index_columns: list[str] | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> Path:
    """Write a cross-run comparison table as LaTeX."""

    frame = comparison_metrics_frame(
        predictions_dir,
        split=split,
        metric_names=metric_columns,
    )
    latex = comparison_latex_table(
        frame,
        metric_columns=metric_columns,
        metric_directions=metric_directions,
        index_columns=index_columns,
        caption=caption or f"{split.title()} metric comparison across model configs.",
        label=label or f"tab:{split}_metric_comparison",
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"comparison_{split}_metrics.tex"
    path.write_text(latex, encoding="utf-8")
    return path


def export_per_model_window_latex(
    predictions_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    metric_columns: list[str] | None = None,
    metric_directions: dict[str, MetricDirection] | None = None,
) -> list[Path]:
    """Write per-run window metric tables as LaTeX."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for run_dir in discover_run_dirs(predictions_dir):
        frame = window_metrics_wide_frame(run_dir, split=split)
        run_name = _safe_name(run_dir.name)
        latex = per_model_latex_table(
            frame,
            metric_columns=metric_columns,
            metric_directions=metric_directions,
            caption=f"{split.title()} per-window metrics for {run_dir.name}.",
            label=f"tab:{run_name}_{split}_window_metrics",
        )
        path = output_path / f"per_model_window_metrics_{run_name}_{split}.tex"
        path.write_text(latex, encoding="utf-8")
        paths.append(path)
    return paths


def export_study_latex_tables(
    predictions_dir: str | Path,
    output_dir: str | Path,
    metric_columns: list[str] | None = None,
    metric_directions: dict[str, MetricDirection] | None = None,
    splits: tuple[str, ...] = ("val", "test"),
) -> dict[str, Path | list[Path]]:
    """Write comparison and per-model LaTeX tables for a study."""

    paths: dict[str, Path | list[Path]] = {}
    for split in splits:
        paths[f"comparison_{split}"] = export_comparison_latex(
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            split=split,
            metric_columns=metric_columns,
            metric_directions=metric_directions,
        )
        paths[f"per_model_windows_{split}"] = export_per_model_window_latex(
            predictions_dir=predictions_dir,
            output_dir=output_dir,
            split=split,
            metric_columns=metric_columns,
            metric_directions=metric_directions,
        )
    return paths


def metric_header(
    metric: str,
    metric_directions: dict[str, MetricDirection] | None = None,
) -> str:
    """Return an escaped metric label with an up/down arrow when directional."""

    directions = {**DEFAULT_METRIC_DIRECTIONS, **(metric_directions or {})}
    label = _latex_escape(DEFAULT_METRIC_LABELS.get(metric, metric.replace("_", " ")))
    direction = directions.get(metric)
    if direction == "max":
        return f"{label} ($\\uparrow$)"
    if direction == "min":
        return f"{label} ($\\downarrow$)"
    return label


def _render_latex_tabular(
    frame: pd.DataFrame,
    metric_columns: list[str],
    metric_directions: dict[str, MetricDirection],
    bold_cells: set[tuple[int, str]],
    caption: str | None,
    label: str | None,
    float_format: str,
) -> str:
    column_spec = "l" * len(frame.columns)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
    ]
    if caption:
        lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.extend(
        [
            f"\\begin{{tabular}}{{{column_spec}}}",
            "\\toprule",
            " & ".join(
                metric_header(column, metric_directions)
                if column in metric_columns
                else _latex_escape(_column_label(column))
                for column in frame.columns
            )
            + " \\\\",
            "\\midrule",
        ]
    )
    for row_idx, row in frame.reset_index(drop=True).iterrows():
        values = []
        for column in frame.columns:
            value = _format_cell(row[column], float_format=float_format)
            if (int(row_idx), column) in bold_cells:
                value = f"\\textbf{{{value}}}"
            values.append(value)
        lines.append(" & ".join(values) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _best_value_cells(
    frame: pd.DataFrame,
    metric_columns: list[str],
    metric_directions: dict[str, MetricDirection],
    tolerance: float = 1e-12,
) -> set[tuple[int, str]]:
    bold_cells: set[tuple[int, str]] = set()
    for column in metric_columns:
        if column not in frame:
            continue
        direction = metric_directions.get(column)
        if direction not in {"max", "min"}:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        finite = values.dropna()
        if finite.empty:
            continue
        best = finite.max() if direction == "max" else finite.min()
        for idx, value in values.items():
            if pd.notna(value) and math.isclose(float(value), float(best), abs_tol=tolerance):
                bold_cells.add((int(idx), column))
    return bold_cells


def _format_cell(value: object, float_format: str) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "--"
        return float_format.format(value)
    if isinstance(value, int):
        return str(value)
    return _latex_escape(str(value))


def _column_label(column: str) -> str:
    if column == "run_name":
        return "Run"
    if column == "window_index":
        return "Window"
    return DEFAULT_METRIC_LABELS.get(column, column.replace("_", " "))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() else "_" for char in value)
    return "_".join(part for part in safe.split("_") if part)


def _empty_table(caption: str | None = None, label: str | None = None) -> str:
    frame = pd.DataFrame({"message": ["No runs found"]})
    return _render_latex_tabular(
        frame,
        metric_columns=[],
        metric_directions={},
        bold_cells=set(),
        caption=caption,
        label=label,
        float_format="{:.4g}",
    )
