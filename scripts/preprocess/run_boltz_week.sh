#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="boltz-env-torchfix"
SCRIPT="/home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py"
LOG_ROOT="/home/natasha/multimodal_model/outputs"
STAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="${LOG_ROOT}/boltz_master_${STAMP}.log"

mkdir -p "$LOG_ROOT"

echo "=== Boltz long run started at $(date) ===" | tee -a "$MASTER_LOG"
echo "Environment: $ENV_NAME" | tee -a "$MASTER_LOG"
echo "Script: $SCRIPT" | tee -a "$MASTER_LOG"
echo "Master log: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Python: $(which python)" | tee -a "$MASTER_LOG"
echo "Boltz:  $(which boltz)" | tee -a "$MASTER_LOG"
echo "LD_PRELOAD: ${LD_PRELOAD:-}" | tee -a "$MASTER_LOG"
echo "LD_LIBRARY_PATH (head): $(echo "${LD_LIBRARY_PATH:-}" | cut -d: -f1-5)" | tee -a "$MASTER_LOG"
echo | tee -a "$MASTER_LOG"

# IMPORTANT: avoid surprises with relative paths
cd /home/natasha/multimodal_model

echo "=== Starting boltz_runs.py ===" | tee -a "$MASTER_LOG"

python "$SCRIPT" \
  --skip_train \
  --recycling_steps 3 \
  --sampling_steps 100 \
  --diffusion_samples 1 \
  --max_parallel_samples 5 \
  --max_msa_seqs 64 \
  --num_subsampled_msa 64 \
  --write_embeddings \
  --progress_every 1000 \
  2>&1 | tee -a "$MASTER_LOG"

echo | tee -a "$MASTER_LOG"
echo "=== Boltz long run finished at $(date) ===" | tee -a "$MASTER_LOG"
