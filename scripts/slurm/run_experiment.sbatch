#!/bin/bash
#SBATCH --job-name=jepa-exp
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

# Usage:
#   sbatch scripts/slurm/run_experiment.sbatch configs/exp/smoke_short_tft.yaml ablation_test
#
# Edit the environment setup block for your cluster. Keep paths relative to the
# repository where possible so generated logs stay with the run artifacts.

CONFIG=${1:-configs/exp/smoke_short_tft.yaml}
EXPERIMENT_NAME=${2:-jepa_ablation}
OUTPUT_DIR=${OUTPUT_DIR:-runs}
MODELS=${MODELS:-"config"}
MAX_TRIALS=${MAX_TRIALS:-}
SEED=${SEED:-42}
RESOURCE_LOG_INTERVAL=${RESOURCE_LOG_INTERVAL:-60}

mkdir -p slurm_logs

# Environment setup examples. Use exactly one that matches your cluster.
# module load python/3.11 cuda/12.1
# source .venv/bin/activate
# conda activate jepa

if command -v uv >/dev/null 2>&1; then
  PYTHON_CMD="uv run python"
else
  PYTHON_CMD="python"
fi

EXTRA_ARGS=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA_ARGS+=(--max-trials "${MAX_TRIALS}")
fi

${PYTHON_CMD} scripts/run_experiment.py \
  --config "${CONFIG}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --models ${MODELS} \
  --seed "${SEED}" \
  --resume \
  --resource-log-interval "${RESOURCE_LOG_INTERVAL}" \
  "${EXTRA_ARGS[@]}"
