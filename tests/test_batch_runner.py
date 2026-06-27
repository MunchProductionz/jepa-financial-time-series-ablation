import json
from pathlib import Path

import pandas as pd

from ablation_study_jepa.api.batch_runner import (
    BatchRunOptions,
    build_trial_specs,
    run_experiment_batch,
)


def test_batch_runner_dry_run_writes_expected_files(tmp_path) -> None:
    result = run_experiment_batch(
        BatchRunOptions(
            config_path=Path("configs/exp/smoke_short_tft.yaml"),
            experiment_name="dry_run_test",
            output_dir=tmp_path,
            models=["tft"],
            max_trials=1,
            seed=123,
            dry_run=True,
            resource_log_interval=60,
            command=["python", "scripts/run_experiment.py", "--dry-run"],
        )
    )

    run_dir = result.run_dir
    assert (run_dir / "resolved_config.yaml").exists()
    assert (run_dir / "command.txt").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "system_info.json").exists()
    assert (run_dir / "resource_usage.jsonl").exists()
    assert (run_dir / "trial_results.jsonl").exists()
    assert (run_dir / "trial_results.csv").exists()
    assert (run_dir / "trial_configs" / "trial_0000_tft.yaml").exists()

    records = [
        json.loads(line)
        for line in (run_dir / "trial_results.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["status"] == "completed"
    assert records[-1]["dry_run"] is True
    assert records[-1]["seed"] == 123
    assert records[-1]["dry_run_summary"]["dataset_samples"]["train"] > 0
    assert records[-1]["model_summary"]["parameter_count"] > 0
    assert records[-1]["model_parameter_count"] == records[-1]["model_summary"]["parameter_count"]
    assert records[-1]["model_transformer_block_count"] == 2

    summary = pd.read_csv(run_dir / "trial_results.csv")
    assert summary.loc[0, "status"] == "completed"
    assert summary.loc[0, "model_parameter_count"] > 0
    assert summary.loc[0, "model_transformer_block_count"] == 2


def test_batch_runner_materializes_sweep_trials(tmp_path) -> None:
    specs = build_trial_specs(
        BatchRunOptions(
            config_path=Path("configs/exp/smoke_short_tft.yaml"),
            sweep_config_path=Path("configs/sweeps/jepa_contrastive.yaml"),
            experiment_name="sweep_test",
            output_dir=tmp_path,
            models=["contrastive"],
            max_trials=2,
            seed=42,
        ),
        run_dir=tmp_path / "run",
    )

    assert len(specs) == 2
    assert specs[0].trial_id == "trial_0000_contrastive_sweep_0000"
    assert specs[0].overrides["model.target"].endswith(":TFTWithJEPA")
    assert specs[0].overrides["jepa.mode"] == "contrastive"
    assert "jepa.num_jepa_layers" in specs[0].sweep_overrides
