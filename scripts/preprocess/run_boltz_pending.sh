#!/usr/bin/env bash
# Manifest-aware Boltz runs: val + test -> outputs/, immrep_test -> outputs_data/
# Plan only (no GPU):
#   bash scripts/preprocess/run_boltz_pending.sh plan
# Single smoke test (one tulip decoy):
#   bash scripts/preprocess/run_boltz_pending.sh smoke val_tulip_epitope_uniform_decoy_000000
set -euo pipefail

REPO="/home/natasha/multimodal_model"
ENV_NAME="${ENV_NAME:-boltz-env-torchfix}"
SCRIPT="${REPO}/scripts/preprocess/boltz_runs_v2.py"
LOG_ROOT="${REPO}/outputs"
STAMP=$(date +"%Y%m%d_%H%M%S")
MODE="${1:-plan}"
PAIR_ID="${2:-}"

mkdir -p "$LOG_ROOT"
LOG="${LOG_ROOT}/boltz_v2_${MODE}_${STAMP}.log"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
# Match run_boltz_week.sh: repo root so YAML MSA paths (data/raw/MSA/...) resolve.
cd "$REPO"

if [[ "$MODE" != "plan" ]]; then
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "[FATAL] nvidia-smi failed — Boltz needs a working GPU driver on this host." >&2
    echo "  (Cursor remote/sandbox agents often have no GPU; run smoke/run in a local tmux shell.)" >&2
    exit 1
  fi
  echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
fi

SPLITS=(val test immrep_test)
EXTRA=(--base_dir "$REPO")
case "$MODE" in
  plan)
    EXTRA+=(--plan-only)
    ;;
  smoke)
    if [[ -z "$PAIR_ID" ]]; then
      echo "Usage: $0 smoke <pair_id>" >&2
      exit 1
    fi
    if [[ "$PAIR_ID" == val_* ]]; then SPLITS=(val)
    elif [[ "$PAIR_ID" == test_* ]]; then SPLITS=(test)
    elif [[ "$PAIR_ID" == immrep_* ]]; then SPLITS=(immrep_test)
    fi
    EXTRA+=(--pair-id "$PAIR_ID" --max-runs 1 --write_embeddings --debug --override)
    ;;
  run)
    EXTRA+=(--write_embeddings)
    ;;
  *)
    echo "Unknown mode: $MODE (use plan|smoke|run)" >&2
    exit 1
    ;;
esac

set +e
python "$SCRIPT" \
  --splits "${SPLITS[@]}" \
  --recycling_steps 3 \
  --sampling_steps 100 \
  --diffusion_samples 1 \
  --max_parallel_samples 5 \
  --max_msa_seqs 64 \
  --num_subsampled_msa 64 \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

echo "Log: $LOG"
if [[ "$RC" -ne 0 ]]; then
  echo "[FATAL] boltz_runs_v2.py exited with code $RC" >&2
  exit "$RC"
fi
