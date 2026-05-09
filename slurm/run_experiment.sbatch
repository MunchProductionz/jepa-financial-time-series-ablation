#!/bin/bash
#SBATCH --job-name=jepa-ablation
#SBATCH --output=slurm-%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

CONFIG=${1:-configs/exp/jepa_ablation.yaml}
uv run ablation-study-jepa run --config "${CONFIG}"

