# Experiment Setup

This repo should be evaluated as an ablation study, not as a search for a single best trading model. Keep the supervised TFT setup, chronological splits, feature set, target horizon, scaler policy, and data universe fixed while varying one JEPA dimension at a time.

## Smoke Runs

Use smoke runs before full experiments to verify that data loading, training, metrics, predictions, training-history CSVs, SVG plots, and checkpoints are all produced.

Build the deterministic smoke panels:

```bash
uv run ablation-study-jepa build-sample-data --output data/prices/smoke/short/panel.csv --periods 240 --tickers AAPL,MSFT,NVDA,AMZN
uv run ablation-study-jepa build-sample-data --output data/prices/smoke/long/panel.csv --periods 720 --tickers AAPL,MSFT,NVDA,AMZN,GOOGL,META,JPM,XOM
```

Short smoke matrix:

```bash
uv run ablation-study-jepa run --config configs/exp/smoke_short_tft.yaml
uv run ablation-study-jepa run --config configs/exp/smoke_short_contrastive_jepa.yaml
uv run ablation-study-jepa run --config configs/exp/smoke_short_lejepa.yaml
```

Longer smoke matrix:

```bash
uv run ablation-study-jepa run --config configs/exp/smoke_long_tft.yaml
uv run ablation-study-jepa run --config configs/exp/smoke_long_contrastive_jepa.yaml
uv run ablation-study-jepa run --config configs/exp/smoke_long_lejepa.yaml
```

The short configs are fast sanity checks. The longer configs use more assets, more dates, larger context windows, and sliding windows, so they are closer to the real pipeline without requiring downloaded market data.

## Core Experiment Matrix

Run these first on the real panel:

- Baseline TFT with JEPA disabled: `configs/exp/tft.yaml`.
- Contrastive JEPA: `configs/exp/contrastive_jepa_ablation.yaml`.
- LeJEPA: `configs/exp/lejepa_ablation.yaml`.

For each JEPA mode, compare against the same baseline seed, split plan, target horizon, feature set, model size, and training budget.

## Primary Ablations

Prioritize this sequence:

- JEPA enabled vs disabled.
- JEPA mode: `contrastive` vs `lejepa`.
- Attached layers: final block only, last 2 blocks, last 3 blocks, all blocks.
- Layer weights: `uniform`, `linear`, `exponential`.
- Global auxiliary weight: `jepa.global_weight` in `[0.001, 0.01, 0.05, 0.1]`.
- JEPA horizons: `[1]`, `[60]`, `[1, 5, 20, 60]`.
- Auxiliary gradient strategy: `global_weighted` vs `local_recompute`.
- Warmup: disabled vs enabled with `start_scale: 0.0`, `end_scale: 1.0`, and 5-10 epochs.

Then run smaller follow-up sweeps:

- Projection dimension: `[64, 128, 256]`.
- Predictor type: `linear`, `mlp`, `residual_mlp`.
- Horizon weights: `uniform`, `linear`, `exponential`.

## Mode-Specific Ablations

For contrastive JEPA:

- Temperature: `[0.05, 0.1, 0.2]`.
- Negative strategy: `in_batch_all`, `in_batch_filtered`, `same_asset_far_time`, `different_asset_different_time`, `mixed`.
- Exclusion window and same-date cross-asset negative policy.

For LeJEPA:

- `jepa.lejepa.loss_mix.lambda_sigreg`: `[0.01, 0.05, 0.1, 0.5]`.
- `jepa.lejepa.detach_target`: keep `true` as the conservative default; test `false` only after the default is stable.
- `jepa.lejepa.sigreg.apply_to`: `context_only` first, then `context_and_targets`.
- SIGReg slices and t-grid only after the main loss-weighting behavior is understood.

## Acceptance Checks Before Comparing Metrics

For every run, inspect:

- `metrics.json` for validation and test metrics.
- `predictions/val_predictions.csv` and `predictions/test_predictions.csv`.
- `training_history/combined_epoch_history.csv`.
- `training_history/plots/loss_history.svg`.
- `train/jepa_weighted_to_supervised_loss_ratio` when JEPA is enabled.
- Gradient diagnostics only when `log_gradient_norms: true`, because they are more expensive.
