#!/bin/bash
set -euo pipefail

# Example long-running VM launch. Run from the repository root.
#
#   bash scripts/run_cloud_tmux_example.sh
#
# Override settings without editing the file:
#
#   CONFIG=configs/exp/contrastive_jepa_ablation.yaml \
#   EXPERIMENT_NAME=contrastive_full \
#   MODELS="tft contrastive lejepa" \
#   MAX_TRIALS=20 \
#   bash scripts/run_cloud_tmux_example.sh

CONFIG=${CONFIG:-configs/exp/smoke_short_tft.yaml}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-jepa_cloud_run}
OUTPUT_DIR=${OUTPUT_DIR:-runs}
MODELS=${MODELS:-config}
MAX_TRIALS=${MAX_TRIALS:-}
SEED=${SEED:-42}
RESOURCE_LOG_INTERVAL=${RESOURCE_LOG_INTERVAL:-60}
SESSION=${SESSION:-jepa-exp}

EXTRA_ARGS=()
if [[ -n "${MAX_TRIALS}" ]]; then
  EXTRA_ARGS+=(--max-trials "${MAX_TRIALS}")
fi

COMMAND=(
  python scripts/run_experiment.py
  --config "${CONFIG}"
  --experiment-name "${EXPERIMENT_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --models ${MODELS}
  --seed "${SEED}"
  --resume
  --resource-log-interval "${RESOURCE_LOG_INTERVAL}"
  "${EXTRA_ARGS[@]}"
)

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "${SESSION}" "${COMMAND[*]}"
  echo "Started tmux session: ${SESSION}"
  echo "Attach with: tmux attach -t ${SESSION}"
else
  mkdir -p "${OUTPUT_DIR}/nohup_logs"
  nohup "${COMMAND[@]}" > "${OUTPUT_DIR}/nohup_logs/${EXPERIMENT_NAME}.out" 2> "${OUTPUT_DIR}/nohup_logs/${EXPERIMENT_NAME}.err" &
  echo "Started nohup process: $!"
  echo "Logs: ${OUTPUT_DIR}/nohup_logs/${EXPERIMENT_NAME}.out"
fi
