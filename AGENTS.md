# AGENTS.md

## Repository Purpose

This repo is a research codebase for testing whether JEPA-style predictive latent-space regularization improves stock-return prediction. It is not a production trading system. The main experiment compares a TFT-style return forecaster against the same forecaster augmented with contrastive JEPA auxiliary heads attached after selected Transformer blocks.

The supervised task is future return prediction over trading-day horizons, for example:

```text
return_{i,t+k} = Close_{i,t+k} / Close_{i,t} - 1
```

where `i` is an asset, `t` is an anchor trading day, and `k` is measured in row steps within each asset's trading-day sequence. Do not use calendar-day offsets for horizons.

## Research Concept

The base model is a Temporal Fusion Transformer-style model with input projection, optional static embeddings, an LSTM temporal encoder, and a configurable stack of causal Transformer blocks. The final block output predicts future stock returns.

Optional JEPA heads attach to intermediate Transformer block outputs. A head projects a context latent into a JEPA space, predicts a future latent representation, and optimizes an InfoNCE-style contrastive loss against a detached target latent. JEPA is used only during training. Validation, test, and inference use only the supervised prediction path.

The main ablation dimensions are:

- JEPA disabled vs enabled
- number of JEPA heads: 0, 1, 2, 3, or all Transformer blocks
- selected layers: final block, last `L` blocks, or manual layer indices
- layer weights: uniform, linear, exponential, manual, always normalized to sum to 1
- global JEPA weight `lambda_jepa`
- JEPA projection dimension and predictor type
- JEPA horizons such as `[1]`, `[60]`, or `[1, 5, 20, 60]`
- contrastive temperature
- negative sampling strategy and exclusion window

## Data Assumptions

Expected panel columns are configurable, but defaults are:

- asset id: `ticker`
- date: `date`
- prices: `open`, `high`, `low`, `close`
- optional volume: `volume`
- optional sector/static fields

Fast OHLCV features must be lagged or interpreted according to the configured prediction timing. The default assumes prediction after market close, so `Close_t` can be used for an anchor at `t`. Slow fundamentals and macro variables must be as-of features using actual publication/availability dates, not period labels or revised future values.

Splits must be chronological. Scalers are fitted on the training split only unless an explicitly as-of rolling scaler is implemented.

## Package Layout

```text
src/ablation_study_jepa/
  api/experiment.py          experiment orchestration
  builders/                  config-driven factories
  config/                    Pydantic schemas and YAML loader
  data/                      loaders, cleaning, preprocessing
  datasets/windowed.py       trading-day window dataset
  features/returns.py        return and volatility features
  models/tft.py              base TFT model
  models/tft_with_jepa.py    TFT model with embedded JEPA heads
  models/jepa.py             JEPA heads, sampler, contrastive loss
  training/                  Lightning wrappers and trainer factory
  evaluation/                metrics and prediction export
  utils/instantiate.py       dotted-path import helper
```

Keep orchestration in `api/experiment.py`, object construction in `builders/`, and pure model code in `models/`.

## Reproducible Entry Points

Set up the local environment with `uv`:

```bash
uv sync --dev
```

Run an experiment from YAML:

```bash
uv run ablation-study-jepa run --config configs/exp/jepa_ablation.yaml
```

Build deterministic synthetic sample data:

```bash
uv run ablation-study-jepa build-sample-data --output data/prices/panel.csv
```

Run the test suite:

```bash
uv run pytest
```

## Correctness Rules

- The supervised prediction path must only use information available as of anchor trading day `t`.
- JEPA target windows ending at `t+k` are auxiliary targets only and must be detached.
- If a single sequence contains both context and future positions, the model must use causal attention so anchor states cannot attend to future states.
- The default implementation uses separate as-of context and JEPA target windows.
- Memory banks, if enabled later, must only contain training-split examples.
- Negatives must be sampled within the same JEPA projection space and should avoid obvious false negatives such as same-date market-regime examples unless explicitly allowed.
