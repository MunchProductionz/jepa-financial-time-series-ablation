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
- Legacy LeJEPA ratio mix: `configs/exp/lejepa_ablation.yaml`.
- Fixed-coefficient LeJEPA combined predictive plus SIGReg: `configs/exp/lejepa_fixed_base.yaml`.

For each JEPA mode, compare against the same baseline seed, split plan, target horizon, feature set, model size, and training budget.

The config selection is:

- `tft`: `TFT` model target, `jepa.enabled: false`.
- `contrastive`: `TFTWithJEPA`, `jepa.enabled: true`, `jepa.mode: contrastive`.
- `lejepa`: `TFTWithJEPA`, `jepa.enabled: true`, `jepa.mode: lejepa`.

Contrastive JEPA and LeJEPA/SIGReg are separate branches. Do not mix their losses unless you add an explicit combined config.

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

- Prefer fixed coefficients for the main ablation:
  - `jepa.global_weight: 1.0`
  - `jepa.lejepa.loss_mix.mode: fixed`
  - `jepa.lejepa.loss_mix.lambda_pred`: `[0.0, 0.01, 0.05, 0.1]`
  - `jepa.lejepa.loss_mix.lambda_sigreg`: `[0.0, 0.01, 0.05, 0.1]`
- The legacy ratio config remains available through `jepa.lejepa.loss_mix.mode: lambda_sigreg`, where `lambda_sigreg` must be in `[0, 1]`.
- `jepa.lejepa.detach_target`: keep `true` as the conservative default; test `false` only after the default is stable.
- `jepa.lejepa.sigreg.apply_to`: `context_only` first, then `context_and_targets`.
- `jepa.lejepa.representation.mode`:
  - `projected`: original JEPA projector path.
  - `direct_h`: directly regularize/predict the native block output `h_l`.
  - `adapter_whitened`: use `u_l=A_l(norm(h_l), c)`, `z_l=W_l(u_l)`, and apply SIGReg to `z_l`.
- `jepa.lejepa.representation.domain_context.enabled`: compare `false` vs `true`; the current source is existing static features such as sector one-hots.
- SIGReg slices and t-grid only after the main loss-weighting behavior is understood.

Fixed-coefficient LeJEPA config entry points:

- Predictive only: `configs/exp/lejepa_fixed_predictive_only.yaml`.
- SIGReg only on native hidden state `h_l`: `configs/exp/lejepa_fixed_sigreg_only_direct_h.yaml`.
- SIGReg only on adapter/whitened `z_l`: `configs/exp/lejepa_fixed_sigreg_only_adapter_whitened.yaml`.
- Predictive plus SIGReg on adapter/whitened `z_l`: `configs/exp/lejepa_fixed_predictive_sigreg_adapter_whitened.yaml`.
- Domain-conditioned adapter: `configs/exp/lejepa_fixed_domain_adapter.yaml`.
- No-stop-gradient target: `configs/exp/lejepa_fixed_no_stop_gradient.yaml`.
- Multiple taps: `configs/exp/lejepa_fixed_multi_tap.yaml`.

These cover the first ablation set:

- supervised TFT only,
- contrastive JEPA only,
- LeJEPA/SIGReg only,
- predictive auxiliary only,
- SIGReg only,
- predictive plus SIGReg,
- direct SIGReg on `h_l`,
- structured adapter plus whitening with SIGReg on `z_l`,
- with and without domain-conditioned adapters,
- one tap and multiple taps,
- stop-gradient and no-stop-gradient targets,
- different `lambda_pred` and `lambda_sigreg` values.

## Evaluation Metrics

The required comparison should include predictive and economic metrics. The fixed LeJEPA configs request:

- Predictive: `rmse`, `mae`, `spearman_rank_ic`, `directional_accuracy`, `positive_prediction_share`.
- Portfolio/economic: `long_short_decile_return`, `long_short_decile_cagr`, `long_short_decile_annualized_volatility`, `long_short_decile_sharpe`, `long_short_decile_sortino`, `long_short_decile_max_drawdown`, `long_short_decile_hit_rate`, `long_short_decile_turnover`, `long_short_decile_transaction_cost_adjusted_return`, `long_only_decile_return`, `equal_weight_benchmark_return`.
- Robustness where available: `sector_neutral_spearman_rank_ic`.

Set `evaluation.portfolio_quantile: 0.1` for decile portfolios and `evaluation.transaction_cost_bps` for transaction-cost-adjusted metrics. Country, size, value, momentum, volatility, liquidity, and regime-neutral diagnostics require those columns to be present or added as future as-of features; do not derive them from future realized returns.

## Acceptance Checks Before Comparing Metrics

For every run, inspect:

- `metrics.json` for validation and test metrics.
- `predictions/val_predictions.csv` and `predictions/test_predictions.csv`.
- `training_history/combined_epoch_history.csv`.
- `training_history/plots/loss_history.svg`.
- `train/jepa_weighted_to_supervised_loss_ratio` when JEPA is enabled.
- `train/jepa_prediction_loss`, `train/jepa_sigreg_loss`, `train/jepa_lambda_pred`, and `train/jepa_lambda_sigreg` for LeJEPA fixed-coefficient runs.
- Optional representation diagnostics for `h`, `u`, and `z` only when `log_representation_stats: true`.
- Gradient diagnostics only when `log_gradient_norms: true`, because they are more expensive.

Auxiliary diagnostics are not acceptance criteria by themselves. A run with more isotropic `z_l` is only useful if it improves held-out prediction and portfolio metrics.
