# Running Long Experiments

This repository still supports the original entrypoint:

```bash
uv run ablation-study-jepa run --config configs/exp/smoke_short_tft.yaml
```

The script runner is an additive wrapper around the same `ExperimentRunner`. It
does not change model internals, dataset construction, training logic, or the
notebook workflow.

## Current Structure

- Training launch: `ablation-study-jepa run --config <yaml>` loads an
  `ExperimentConfig` and calls `ExperimentRunner(config).run()`.
- Model selection: config controls `model.target`, `jepa.enabled`, and
  `jepa.mode`. The named variants are `tft`, `contrastive`, and `lejepa`.
- Hyperparameters: normal runs use YAML configs in `configs/exp/`; sweeps use
  W&B-style YAML grids in `configs/sweeps/`.
- Training: `training/trainer_factory.py` builds a Lightning trainer with
  checkpoints, early stopping, CSV/W&B loggers, and training-history callbacks.
- Existing artifacts: each run writes `metrics.json`, predictions, checkpoints,
  `run_status.json`, `training_history/`, `analysis/`, provenance, and manifests
  under `evaluation.predictions_dir`.

The new script puts those existing per-run artifacts under a unique batch
directory and adds batch-level resource and trial logs.

## Smoke Test

Validate setup without fitting:

```bash
python scripts/run_experiment.py \
  --config configs/exp/smoke_short_tft.yaml \
  --experiment-name smoke_dry_run \
  --output-dir runs \
  --models tft \
  --seed 42 \
  --dry-run
```

Run a short training smoke test:

```bash
python scripts/run_experiment.py \
  --config configs/exp/smoke_short_tft.yaml \
  --experiment-name smoke_train \
  --output-dir runs \
  --models tft contrastive lejepa \
  --seed 42
```

## Full Experiment

Use an existing full config and select model variants:

```bash
python scripts/run_experiment.py \
  --config configs/exp/contrastive_jepa_ablation.yaml \
  --experiment-name contrastive_ablation \
  --output-dir runs \
  --models tft contrastive lejepa \
  --seed 42 \
  --resume \
  --resource-log-interval 60
```

Run a bounded local grid from an existing sweep YAML:

```bash
python scripts/run_experiment.py \
  --sweep-config configs/sweeps/jepa_contrastive.yaml \
  --experiment-name contrastive_grid \
  --output-dir runs \
  --max-trials 20 \
  --seed 42 \
  --resume
```

You can pass config overrides either as repeated flags:

```bash
python scripts/run_experiment.py \
  --config configs/exp/smoke_short_tft.yaml \
  --experiment-name smoke_override \
  --override training.max_epochs=2 \
  --override dataset.batch_size=32
```

or in the same dotted style as the package CLI:

```bash
python scripts/run_experiment.py \
  --config configs/exp/smoke_short_tft.yaml \
  --experiment-name smoke_override \
  --training.max_epochs=2 \
  --dataset.batch_size=32
```

## Notebook Usage

Existing notebooks can keep using:

```python
from ablation_study_jepa.api.experiment import ExperimentRunner
```

For the script runner from a notebook:

```python
from pathlib import Path
from ablation_study_jepa.api.batch_runner import BatchRunOptions, run_experiment_batch

result = run_experiment_batch(
    BatchRunOptions(
        config_path=Path("configs/exp/smoke_short_tft.yaml"),
        experiment_name="notebook_smoke",
        output_dir=Path("runs"),
        models=["tft"],
        dry_run=True,
    )
)
result.run_dir
```

## Cloud VM

Use the helper script to run in `tmux` when available, with `nohup` as fallback:

```bash
CONFIG=configs/exp/contrastive_jepa_ablation.yaml \
EXPERIMENT_NAME=contrastive_cloud \
MODELS="tft contrastive lejepa" \
MAX_TRIALS=20 \
bash scripts/run_cloud_tmux_example.sh
```

Attach to the session:

```bash
tmux attach -t jepa-exp
```

## Slurm

Submit the template:

```bash
sbatch scripts/slurm/run_experiment.sbatch configs/exp/smoke_short_tft.yaml smoke_slurm
```

Useful environment overrides:

```bash
MODELS="tft contrastive lejepa" MAX_TRIALS=20 OUTPUT_DIR=runs \
sbatch scripts/slurm/run_experiment.sbatch configs/exp/contrastive_jepa_ablation.yaml full_ablation
```

Edit the environment setup block in `scripts/slurm/run_experiment.sbatch` for
your cluster modules, virtualenv, or conda environment.

## Outputs

Each script invocation creates or resumes a directory like:

```text
runs/
  20260628T120000Z_ablation_test/
    resolved_config.yaml
    command.txt
    environment.json
    system_info.json
    resource_usage.jsonl
    trial_results.jsonl
    trial_results.csv
    metrics.jsonl
    trial_configs/
    logs/
      stdout.log
      stderr.log
    predictions/
      runs_manifest.csv
      <run_name>_<timestamp>_<config_hash>/
        metrics.json
        run_status.json
        checkpoints/
        training_history/
        analysis/
        val.csv
        test.csv
```

`resource_usage.jsonl` contains periodic CPU, RAM, disk, GPU, and PyTorch CUDA
memory samples. GPU utilization is recorded with `pynvml` when installed, then
`nvidia-smi` when available, then PyTorch CUDA memory-only fallback. CPU-only and
Apple MPS runs log unavailable GPU metrics instead of failing.

`trial_results.jsonl` is append-safe and records trial starts, completions,
failures, and skips. Completed records include a nested `model_summary` object
with actual instantiated parameter counts and module counts. `trial_results.csv`
is a latest-row summary by trial ID and flattens common size fields into columns
such as `model_parameter_count`, `model_trainable_parameter_count`,
`model_base_parameter_count`, `model_jepa_parameter_count`,
`model_transformer_block_count`, `model_lstm_layer_count`,
`model_linear_layer_count`, `model_mlp_linear_layer_count`, and
`model_predictor_linear_layer_count`. These columns are intended for easy
slice/group operations across model variants and configs.

Each per-run `metrics.json`, `run_status.json`, `analysis/provenance.json`, and
`analysis/config_summary.csv` also stores the model-size summary when training
reaches model construction. The root `predictions/runs_manifest.csv` receives
the same flat model-size columns for completed runs. `metrics.jsonl` mirrors
epoch-level training-history rows when the existing training-history callback
writes them.

## Resume Behavior

`--resume` reuses the latest matching batch directory for the experiment name
unless `--run-dir` is supplied. Completed trials with the same trial ID and
resolved config hash are skipped. Partial or failed trials are run again from the
start.

Lightning checkpoints are still written by the existing trainer under each
per-run artifact directory. The script does not yet pass `ckpt_path` back into
`trainer.fit`, so checkpoint-level continuation of an interrupted trial is not
implemented.

## Dependencies

The resource logger uses `psutil` for CPU and RAM metrics. GPU metrics are
optional and use whichever of `pynvml`, `nvidia-smi`, or PyTorch CUDA is
available.
