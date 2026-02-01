#!/usr/bin/env bash
set -euo pipefail

# ---- Config ----
ENV_NAME="boltz-env-torchfix"
SCRIPT="/home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py"
LOG_ROOT="/home/natasha/multimodal_model/outputs"
STAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/boltz_master_${STAMP}.log"

mkdir -p "$LOG_ROOT"

echo "=== Boltz long run started at $(date) ==="
echo "Environment: $ENV_NAME"
echo "Script: $SCRIPT"
echo "Master log: $MASTER_LOG"
echo

# ---- Activate env ONCE ----
source ~/miniconda3/etc/profile.d/conda.sh

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}"

conda activate "$ENV_NAME"

echo "Python: $(which python)"
echo "Boltz:  $(which boltz)"
echo "LD_PRELOAD: $LD_PRELOAD"
echo "LD_LIBRARY_PATH (head): $(echo "$LD_LIBRARY_PATH" | cut -d: -f1-5)"
echo
echo "=== Starting boltz_runs.py ==="
echo

# ---- Run ----
python "$SCRIPT"

echo
echo "=== Boltz long run finished at $(date) ==="
