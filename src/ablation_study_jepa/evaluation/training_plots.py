"""Plot training-history CSV artifacts as lightweight SVG figures."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


DEFAULT_LOSS_COLUMNS = [
    "train/supervised_loss",
    "train/total_loss",
    "train/total_jepa_loss",
    "train/weighted_jepa_loss",
    "val/prediction_loss",
]

COLORS = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#d97706",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
]


def plot_training_history(
    history_csv: str | Path,
    output_dir: str | Path | None = None,
    loss_columns: list[str] | None = None,
    gradient_columns: list[str] | None = None,
) -> dict[str, Path]:
    """Render available loss and gradient-history columns from a training-history CSV."""

    history_path = Path(history_csv)
    frame = pd.read_csv(history_path)
    if frame.empty:
        raise ValueError(f"Training history is empty: {history_path}")

    output_root = Path(output_dir) if output_dir is not None else history_path.parent / "plots"
    output_root.mkdir(parents=True, exist_ok=True)

    losses = [column for column in (loss_columns or DEFAULT_LOSS_COLUMNS) if column in frame]
    gradients = gradient_columns or [
        column
        for column in frame.columns
        if column.startswith("grad_norm_") or column.startswith("lr-")
    ]
    gradients = [column for column in gradients if column in frame]

    outputs: dict[str, Path] = {}
    if losses:
        path = output_root / "loss_history.svg"
        path.write_text(_line_chart_svg(frame, losses, title="Training and Validation Loss"), "utf-8")
        outputs["losses"] = path
    if gradients:
        path = output_root / "gradient_history.svg"
        path.write_text(_line_chart_svg(frame, gradients, title="Gradient Diagnostics"), "utf-8")
        outputs["gradients"] = path
    if not outputs:
        raise ValueError(f"No loss or gradient columns found in {history_path}")
    return outputs


def _line_chart_svg(frame: pd.DataFrame, columns: list[str], title: str) -> str:
    series = _collect_series(frame, columns)
    if not series:
        raise ValueError(f"No numeric series available for {title}")

    width = 960
    height = 560
    margin_left = 76
    margin_right = 32
    margin_top = 54
    margin_bottom = 76
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_values = [x for _, points in series for x, _ in points]
    y_values = [y for _, points in series for _, y in points]
    x_min, x_max = _range_with_padding(min(x_values), max(x_values), pad_equal=1.0)
    y_min, y_max = _range_with_padding(min(y_values), max(y_values), pad_equal=0.05)

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    axis = [
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" '
        f'x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" '
        f'x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827"/>',
    ]
    for tick in _ticks(x_min, x_max):
        x = sx(tick)
        axis.append(
            f'<line x1="{x:.2f}" y1="{margin_top + plot_height}" '
            f'x2="{x:.2f}" y2="{margin_top + plot_height + 5}" stroke="#111827"/>'
        )
        axis.append(
            f'<text x="{x:.2f}" y="{margin_top + plot_height + 22}" '
            f'text-anchor="middle" font-size="12">{tick:g}</text>'
        )
    for tick in _ticks(y_min, y_max):
        y = sy(tick)
        axis.append(
            f'<line x1="{margin_left - 5}" y1="{y:.2f}" '
            f'x2="{margin_left}" y2="{y:.2f}" stroke="#111827"/>'
        )
        axis.append(
            f'<text x="{margin_left - 10}" y="{y + 4:.2f}" '
            f'text-anchor="end" font-size="12">{tick:.4g}</text>'
        )

    lines: list[str] = []
    legend: list[str] = []
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        polyline_points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        lines.append(
            f'<polyline points="{polyline_points}" fill="none" stroke="{color}" '
            'stroke-width="2.25" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = margin_top + 22 * index
        legend.append(
            f'<line x1="{margin_left + plot_width - 210}" y1="{legend_y}" '
            f'x2="{margin_left + plot_width - 190}" y2="{legend_y}" stroke="{color}" '
            'stroke-width="2.25"/>'
        )
        legend.append(
            f'<text x="{margin_left + plot_width - 184}" y="{legend_y + 4}" '
            f'font-size="12">{escape(label)}</text>'
        )

    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{margin_left}" y="30" font-size="22" font-weight="700">{escape(title)}</text>',
            *axis,
            *lines,
            f'<text x="{margin_left + plot_width / 2}" y="{height - 24}" '
            'text-anchor="middle" font-size="13">epoch</text>',
            *legend,
            "</svg>",
        ]
    )


def _collect_series(frame: pd.DataFrame, columns: list[str]) -> list[tuple[str, list[tuple[float, float]]]]:
    collected: list[tuple[str, list[tuple[float, float]]]] = []
    group_columns = ["window_label"] if "window_label" in frame.columns else []
    for column in columns:
        event = _preferred_event(column)
        subset = frame
        if "event" in frame.columns and event in set(frame["event"].dropna()):
            subset = frame.loc[frame["event"] == event]
        for group_key, group in _iter_groups(subset, group_columns):
            if "epoch" not in group.columns:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            epochs = pd.to_numeric(group["epoch"], errors="coerce")
            points = [
                (float(x), float(y))
                for x, y in zip(epochs, values, strict=True)
                if pd.notna(x) and pd.notna(y)
            ]
            if points:
                label = f"{group_key} {column}" if group_key else column
                collected.append((label, points))
    return collected


def _iter_groups(
    frame: pd.DataFrame,
    group_columns: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    if not group_columns:
        return [("", frame)]
    return [(str(key), group) for key, group in frame.groupby(group_columns[0], sort=True)]


def _preferred_event(column: str) -> str:
    if column.startswith("val/"):
        return "validation_epoch_end"
    if column.startswith("test/"):
        return "test_epoch_end"
    return "train_epoch_end"


def _range_with_padding(min_value: float, max_value: float, pad_equal: float) -> tuple[float, float]:
    if min_value == max_value:
        return min_value - pad_equal, max_value + pad_equal
    pad = (max_value - min_value) * 0.05
    return min_value - pad, max_value + pad


def _ticks(min_value: float, max_value: float, count: int = 5) -> list[float]:
    if count <= 1:
        return [min_value]
    step = (max_value - min_value) / (count - 1)
    return [min_value + i * step for i in range(count)]
