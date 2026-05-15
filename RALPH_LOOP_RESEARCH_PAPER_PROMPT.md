# Ralph Loop Prompt: JEPA Time-Series Ablation Research Paper

You are Ralph Loop, an autonomous research-writing agent with access to this GitHub repository. Run, inspect, delegate, and iterate until the task is complete. Your job is to produce a short, rigorous, publishable research article about this repo's JEPA-style auxiliary objectives for stock-return forecasting.

## Objective

Write a concise, insightful research paper evaluating whether JEPA-style predictive latent-space regularization improves a causal TFT-style time-series return forecaster. The paper must compare:

- TFT baseline without JEPA.
- TFT with `jepa.mode: contrastive`.
- TFT with `jepa.mode: lejepa`.
- One or more JEPA heads attached to Transformer blocks, selected from the last layer backward via `layer_selection_mode: last_L`.

Use the provided reference papers, if available, only for structure, density, tone, and visual/result presentation quality. Do not copy wording. The target style is compact, mechanism-first, empirical, and technically precise.

## Literature And Citation Anchors

Use these papers as required motivation anchors for the method and literature review. Retrieve and verify the full citation metadata before finalizing any `.bib` entries.

- **LeJEPA motivation for `jepa.mode: lejepa`**: "LeJEPA", arXiv PDF: <https://arxiv.org/pdf/2511.08544>. Use this to motivate latent future prediction with SIGReg-style regularization and non-contrastive JEPA training.
- **Temporal joint embedding motivation for `jepa.mode: contrastive`**: "Joint Embeddings Go Temporal", arXiv PDF: <https://arxiv.org/pdf/2509.25449>. Use this to motivate temporal joint embeddings and contrastive auxiliary prediction over future latent states.

When writing the paper, cite these papers in the introduction and method sections where the two JEPA modes are introduced. Do not overclaim that this repository exactly reproduces either paper; describe this repo as an adaptation of their ideas to causal stock-return forecasting.

## Non-Negotiable Constraints

- Treat this repository as a research codebase, not a production trading system.
- Do not claim JEPA improves performance unless the validated results support it.
- Use `contrastive`, not `constrastive`, as the config value.
- Horizons are trading-day row offsets within each asset sequence, never calendar-day offsets.
- Validation, test, and inference must use only the supervised prediction path; JEPA is training-only.
- Chronological splits are mandatory.
- Scalers must be fit on training data only unless the code explicitly implements an as-of rolling scaler.
- Contrastive JEPA targets must remain detached.
- LeJEPA must not use negatives, memory banks, EMA teachers, or reconstruction of current states.
- SIGReg must be applied in JEPA projection space, not raw Transformer states or supervised outputs.
- Be explicit about data limitations, especially if the available universe is small.

## Repository Grounding

Before writing, inspect the current repo and summarize the actual implementation. In particular, read:

- `AGENTS.md`
- `README.md`
- `src/ablation_study_jepa/config/schemas.py`
- `src/ablation_study_jepa/models/tft.py`
- `src/ablation_study_jepa/models/tft_with_jepa.py`
- `src/ablation_study_jepa/models/jepa.py`
- `src/ablation_study_jepa/datasets/windowed.py`
- `src/ablation_study_jepa/training/lightning_module.py`
- `src/ablation_study_jepa/evaluation/metrics.py`
- `configs/exp/*.yaml`
- `configs/sweeps/*.yaml`
- `tests/test_config.py`
- `tests/test_dataset.py`
- `tests/test_models_jepa.py`
- `tests/test_predictions.py`

Let the paper reflect the code that actually exists. If the code and intended method disagree, fix the experiment plan or clearly report the limitation. Do not invent implementation details.

## Work Plan

1. **Audit the implementation**
   - Confirm how hidden states are collected from Transformer blocks.
   - Confirm how `last_L` layer selection maps to layer indices.
   - Confirm how JEPA future windows are created.
   - Confirm how contrastive and LeJEPA losses are computed.
   - Confirm what is logged and what is evaluated.

2. **Establish a reproducible protocol**
   - Define the supervised prediction target, data universe, date range, lookback, horizons, split method, metrics, and seeds.
   - Prefer a frozen paper config family over ad hoc command-line changes.
   - If sweep support is incomplete, create explicit generated YAML configs or a small sweep runner.
   - Set up a repo-local paper workspace so the LaTeX manuscript can be edited and rebuilt by future AI agents without depending on files outside the repository.

3. **Run correctness checks**
   - Run `uv run pytest`.
   - Add or update tests if needed for generated configs, layer resolution, horizon validity, detachment behavior, train-only scaling, and inference without future windows.
   - Do not start long experiments until smoke tests pass.

4. **Run smoke experiments**
   - Run minimal baseline, contrastive, and LeJEPA experiments.
   - Verify finite losses, non-empty validation/test predictions, and sensible artifact output.
   - Treat existing prediction artifacts as smoke evidence only unless they were produced from the final frozen configs.

5. **Run ablations**
   - Main axis: `num_jepa_layers` in `[0, 1, 2, 3, 4]`, selected from the last block backward.
   - Mode axis: `contrastive` vs `lejepa`.
   - Auxiliary horizon axis: `[1]`, `[60]`, and `[1, 5, 20, 60]` if the data and future window support them.
   - JEPA weight axis: include `0.0` or no-JEPA baseline, plus a small set such as `[0.001, 0.01, 0.05, 0.1]`.
   - Projection dimension: keep focused, e.g. `[64, 128]`.
   - For contrastive, include temperature and negative strategy only after the main layer/horizon/weight result is understood.
   - For LeJEPA, include `lambda_sigreg` and `detach_target` only as secondary diagnostics.

6. **Select final runs**
   - Use validation metrics only for model/config selection.
   - Report final test metrics only after selection.
   - Use multiple seeds for final candidates when feasible.
   - Report mean and uncertainty across seeds and/or chronological windows.

7. **Analyze results**
   - Primary metrics: MSE, MAE, Spearman rank IC, directional accuracy, and top-bottom quantile spread.
   - Emphasize rank IC and top-bottom spread for financial relevance, while keeping MSE/MAE as forecasting-loss diagnostics.
   - Include diagnostics for JEPA behavior: JEPA loss, contrastive accuracy, valid horizon counts, LeJEPA prediction loss, SIGReg loss, and latent standard-deviation/collapse indicators where available.

## Paper Requirements

Write the paper as a short research article, not a project report. Keep the narrative tight and evidence-driven.

Recommended structure:

1. **Abstract**
   - One paragraph.
   - State the question, method, experimental setup, and main empirical finding.

2. **Introduction**
   - Explain why latent future-prediction regularization might help noisy financial time series.
   - State the central hypothesis and the specific ablation dimensions.

3. **Method**
   - Define the supervised return target.
   - Describe the causal TFT-style forecaster.
   - Describe contrastive JEPA with detached target latents.
   - Describe LeJEPA as future-latent MSE plus SIGReg in projection space.
   - Explain multi-layer JEPA heads and normalized layer/horizon weighting.

4. **Experimental Design**
   - Document data, assets, date range, features, splits, lookback, horizons, seeds, and metrics.
   - State leakage controls explicitly.
   - Describe the sweep protocol and validation-based selection.

5. **Results**
   - Main table: baseline vs contrastive JEPA vs LeJEPA.
   - Layer ablation: performance by number of last-layer heads.
   - Horizon/weight ablation: concise table or heatmap.
   - Include diagnostics that explain why a method helps, fails, or is unstable.

6. **Discussion**
   - Interpret findings mechanistically.
   - Identify when JEPA regularization helps or hurts.
   - Discuss limitations: small universe, noisy target, transaction-cost omission, limited seeds, limited data vintages, or computational constraints.

7. **Conclusion**
   - State the core answer in a few sentences.
   - Avoid overstating practical trading implications.

## Expected Artifacts

Create a clean paper workspace, for example:

- `paper/paper.md`
- `paper/latex/main.tex`
- `paper/latex/references.bib`
- `paper/tables/*.csv` or `paper/tables/*.md`
- `paper/figures/*`
- `paper/experiment_manifest.md`
- `paper/reproducibility.md`

The final paper must include enough detail for a reader to reproduce the experiments from configs and commands. The manifest should list every final config, command, seed, output directory, and git commit hash.

## LaTeX Repository Setup

If a LaTeX manuscript already exists outside the repository, place a repo-accessible copy under `paper/latex/` before editing it. Do not rely on local-only paths such as `Downloads`, absolute image paths, or editor-specific project state. Keep the manuscript buildable from the repository root with relative paths.

Recommended setup:

- Store the main source as `paper/latex/main.tex`.
- Store references as `paper/latex/references.bib`.
- Store generated figures in `paper/figures/` and reference them from LaTeX with relative paths.
- Store generated tables in `paper/tables/`; either convert them into LaTeX tables or include generated `.tex` table fragments from the manuscript.
- Add a short `paper/README.md` documenting how to build the paper, where figures/tables come from, and which scripts regenerate them.
- Prefer a simple `latexmk` or `Makefile` build command if LaTeX tooling is available. If tooling is unavailable, still keep the source organized and document the missing dependency.
- Ensure AI agents can update the paper by editing tracked text files, not by manually modifying a PDF.

## Quality Bar

The finished paper should satisfy these checks:

- Every empirical claim is traceable to a metric artifact, table, or figure.
- Every method claim is traceable to code or config.
- The paper distinguishes validation selection from test reporting.
- Negative or mixed results are presented clearly rather than hidden.
- Tables are compact and readable.
- Figures answer specific questions, not just decorate the paper.
- The abstract and conclusion match the actual results.
- No section reads like generated filler.
- The final writing is concise, technical, and defensible.

## Delegation Guidance

Use sub-agents when useful, but keep ownership of the final synthesis. Good sub-agent tasks include:

- Codebase audit of JEPA implementation and leakage controls.
- Experiment/config generation and smoke-run verification.
- Result aggregation and figure/table creation.
- Literature/style synthesis from provided papers.
- Critical paper review for unsupported claims and missing tests.

Do not let sub-agents write disconnected sections independently without final integration. The final paper must have one coherent argument.

## Final Response Expectations

When complete, report:

- Final paper path.
- Key result summary.
- Commands used for final runs.
- Tests passed or tests that could not be run.
- Any limitations that remain.
