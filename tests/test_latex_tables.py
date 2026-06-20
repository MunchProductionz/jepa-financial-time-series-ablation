import json

import pandas as pd

from ablation_study_jepa.evaluation.compare_runs import comparison_metrics_frame
from ablation_study_jepa.evaluation.latex_tables import (
    comparison_latex_table,
    export_comparison_latex,
    metric_header,
)


def test_metric_headers_include_direction_arrows() -> None:
    assert metric_header("spearman_rank_ic") == "Spearman IC ($\\uparrow$)"
    assert metric_header("rmse") == "RMSE ($\\downarrow$)"


def test_comparison_latex_bolds_best_values_by_metric_direction() -> None:
    frame = pd.DataFrame(
        {
            "run_name": ["baseline", "lejepa"],
            "rmse": [0.25, 0.20],
            "spearman_rank_ic": [0.05, 0.12],
            "long_short_decile_turnover": [0.40, 0.60],
        }
    )

    latex = comparison_latex_table(
        frame,
        metric_columns=["rmse", "spearman_rank_ic", "long_short_decile_turnover"],
    )

    assert "RMSE ($\\downarrow$)" in latex
    assert "Spearman IC ($\\uparrow$)" in latex
    assert "Long-short turnover ($\\downarrow$)" in latex
    assert "lejepa & \\textbf{0.2} & \\textbf{0.12} & 0.6" in latex
    assert "baseline & 0.25 & 0.05 & \\textbf{0.4}" in latex


def test_comparison_metrics_frame_and_latex_export_from_saved_runs(tmp_path) -> None:
    run_a = tmp_path / "run-a"
    run_b = tmp_path / "run-b"
    run_a.mkdir()
    run_b.mkdir()
    _write_metrics(run_a, "baseline", rmse=0.30, spearman_rank_ic=0.05)
    _write_metrics(run_b, "lejepa", rmse=0.20, spearman_rank_ic=0.10)

    frame = comparison_metrics_frame(
        tmp_path,
        split="test",
        metric_names=["rmse", "spearman_rank_ic"],
    )
    path = export_comparison_latex(
        tmp_path,
        tmp_path / "latex",
        split="test",
        metric_columns=["rmse", "spearman_rank_ic"],
    )

    assert frame["run_name"].tolist() == ["baseline", "lejepa"]
    assert path.exists()
    latex = path.read_text(encoding="utf-8")
    assert "\\textbf{0.2}" in latex
    assert "\\textbf{0.1}" in latex


def _write_metrics(run_dir, run_name: str, rmse: float, spearman_rank_ic: float) -> None:
    payload = {
        "run_name": run_name,
        "metrics": {
            "total": {
                "val": {"rmse": rmse + 0.01, "spearman_rank_ic": spearman_rank_ic - 0.01},
                "test": {"rmse": rmse, "spearman_rank_ic": spearman_rank_ic},
            },
            "windows": {
                "0": {"test": {"rmse": rmse, "spearman_rank_ic": spearman_rank_ic}},
            },
        },
        "config": {
            "model": {"target": "ablation_study_jepa.models.tft:TFT"},
            "jepa": {"enabled": False, "mode": "contrastive", "num_jepa_layers": 0},
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
