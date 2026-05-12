# JEPA Financial Time-Series Ablation

Config-driven PyTorch research code for testing whether contrastive JEPA-style auxiliary heads improve stock-return prediction with a TFT-style model.

The repo is built for ablations, not production trading. It focuses on clean experiment configuration, chronological stock-panel datasets, causal/as-of representations, detachable JEPA heads, and supervised evaluation metrics.

## Quick Start

```bash
uv sync --dev
uv run ablation-study-jepa download-yahoo-prices \
  --tickers AAPL,MSFT,NVDA,AMZN \
  --start-date 1960-01-01 \
  --end-date 2025-12-31 \
  --output-dir data/prices

uv run ablation-study-jepa download-fred-md \
  --vintage current \
  --start-date 1960-01-01 \
  --end-date 2025-12-31 \
  --output-dir data/macro/fred_md

uv run ablation-study-jepa run --config configs/exp/jepa_ablation.yaml
```

Yahoo Finance files are written one ticker per CSV, for example
`data/prices/AAPL.csv`, and existing ticker files are skipped unless
`--overwrite` is passed. The FRED-MD command stores the raw downloaded vintage
and a filtered parsed CSV such as `data/macro/fred_md/fred_md_1960_2025.csv`.

Run tests with the same `uv` environment:

```bash
uv run pytest
```

The project is intentionally configured around `uv`; use `.python-version` and
`pyproject.toml` as the source of truth for the Python version and dependencies.

## What Is Being Tested

The supervised task predicts future log-returns over trading-day horizons:

```text
log_return_{i,t+k} = log(Close_{i,t+k} / Close_{i,t})
```

The base model is a TFT-inspired forecaster with:

- feature projection or variable selection
- optional static feature embeddings
- LSTM temporal encoding
- a configurable stack of causal Transformer blocks
- a supervised log-return prediction head

When JEPA is enabled, selected Transformer block outputs feed auxiliary heads that:

1. project context states into a JEPA latent space,
2. predict future latent states at configurable trading-day horizons,
3. compare against detached target latents with InfoNCE,
4. aggregate normalized per-layer and per-horizon losses.

JEPA modules are training-only. Validation, testing, and inference use only the base forecasting path.

## Leakage Prevention

The default dataset creates separate windows:

- context window ending at anchor day `t`
- JEPA target window ending at `t+k`

The supervised model only sees the context window. The future JEPA target is detached and contributes no supervised information. Splits are chronological, and scalers are fitted on the training period only.

All horizons are row offsets within each asset's trading-day sequence. Calendar-day arithmetic is intentionally not used.

## Project Layout

```text
configs/exp/                    YAML experiments
scripts/build_sample_data.py     deterministic synthetic panel generator
src/ablation_study_jepa/
  api/experiment.py              experiment runner
  builders/                      factories from config
  config/                        schemas and YAML defaults loader
  data/                          loading, cleaning, scaling, splits
  datasets/windowed.py           trading-day window dataset
  features/returns.py            return target and feature creation
  models/tft.py                  base TFT model
  models/tft_with_jepa.py        TFT model with embedded JEPA heads
  models/jepa.py                 JEPA heads, InfoNCE, negative sampler
  training/                      Lightning module/datamodule/trainer
  evaluation/                    metrics and prediction exports
tests/                           focused unit tests
```

See `AGENTS.md` for contributor guidance and the research intent.
