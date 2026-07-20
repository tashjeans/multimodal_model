#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PYTHON_BIN="${PYTHON_BIN:-python}"

BASE="/home/natasha/multimodal_model/models"
TRAIN_DIR="/home/natasha/multimodal_model/scripts/train/workshop"

ONEHOT_SCRIPT="$TRAIN_DIR/train_onehot_vicreg_workshop.py"
ESM_SCRIPT="$TRAIN_DIR/train_esm_vicreg_workshop.py"
ESM_RAW_SCRIPT="$TRAIN_DIR/train_esm_vicreg_raw_workshop.py"
COLLECT_SCRIPT="$TRAIN_DIR/collect_workshop_metrics.py"

LOG_ROOT="$BASE/log_files/workshop"
mkdir -p "$LOG_ROOT/onehot_vicreg_complete" "$LOG_ROOT/esm_vicreg_finetuned_complete" "$LOG_ROOT/esm_vicreg_raw_complete"

TRAIN_CSV="/home/natasha/multimodal_model/data/train/train_multiview.csv"
VAL_CSV="/home/natasha/multimodal_model/data/val/val_multiview.csv"
TEST_CSV="/home/natasha/multimodal_model/data/test/test_multiview.csv"
IMMREP_CSV="/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"

FINETUNED_ROOT="/home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids"
FINETUNED_IMMREP="/home/natasha/multimodal_model/models/embeddings/immrep_test_set/test"
PRETRAINED_ROOT="/home/natasha/multimodal_model/models/embeddings/raw_esmc_300m_multiview_ids"
PRETRAINED_IMMREP="/home/natasha/multimodal_model/models/embeddings/raw_esmc_300m_multiview_ids/immrep_test"

ONEHOT_CHECKPOINT_ROOT="$BASE/checkpoints/workshop/onehot_vicreg_complete"
ONEHOT_OUTPUT_ROOT="$BASE/outputs/workshop/onehot_vicreg_complete"
ONEHOT_FIGURE_ROOT="$BASE/figures/workshop/onehot_vicreg_complete"

ESM_CHECKPOINT_ROOT="$BASE/checkpoints/workshop/esm_vicreg_finetuned_complete"
ESM_OUTPUT_ROOT="$BASE/outputs/workshop/esm_vicreg_finetuned_complete"
ESM_FIGURE_ROOT="$BASE/figures/workshop/esm_vicreg_finetuned_complete"

ESM_RAW_CHECKPOINT_ROOT="$BASE/checkpoints/workshop/esm_vicreg_raw_complete"
ESM_RAW_OUTPUT_ROOT="$BASE/outputs/workshop/esm_vicreg_raw_complete"
ESM_RAW_FIGURE_ROOT="$BASE/figures/workshop/esm_vicreg_raw_complete"

run_and_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "Logging to: $log_file"
  set +e
  "$@" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    echo "Command failed with status $status. See log: $log_file" >&2
    exit $status
  fi
}

run_onehot() {
  local seed="$1"
  local epochs="$2"
  local min_epochs="$3"
  local patience="$4"
  local save_latents="$5"
  local log_file="$LOG_ROOT/onehot_vicreg_complete/seed_${seed}.log"
  local latent_args=()
  if [[ "$save_latents" == "1" ]]; then
    latent_args+=(--save-latents)
  fi

  run_and_log "$log_file" \
    "$PYTHON_BIN" "$ONEHOT_SCRIPT" \
      --seed "$seed" \
      --epochs "$epochs" \
      --min-epochs "$min_epochs" \
      --patience "$patience" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --immrep-csv "$IMMREP_CSV" \
      --checkpoint-root "$ONEHOT_CHECKPOINT_ROOT" \
      --output-root "$ONEHOT_OUTPUT_ROOT" \
      --figure-root "$ONEHOT_FIGURE_ROOT" \
      --missing-chain-policy complete_only \
      --overwrite \
      "${latent_args[@]}"
}

run_esm() {
  local seed="$1"
  local epochs="$2"
  local min_epochs="$3"
  local patience="$4"
  local save_latents="$5"
  local log_file="$LOG_ROOT/esm_vicreg_finetuned_complete/seed_${seed}.log"
  local latent_args=()
  if [[ "$save_latents" == "1" ]]; then
    latent_args+=(--save-latents)
  fi

  run_and_log "$log_file" \
    "$PYTHON_BIN" "$ESM_SCRIPT" \
      --seed "$seed" \
      --epochs "$epochs" \
      --min-epochs "$min_epochs" \
      --patience "$patience" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --immrep-csv "$IMMREP_CSV" \
      --finetuned-embed-root "$FINETUNED_ROOT" \
      --finetuned-immrep-shard-dir "$FINETUNED_IMMREP" \
      --pretrained-embed-root "$PRETRAINED_ROOT" \
      --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP" \
      --checkpoint-root "$ESM_CHECKPOINT_ROOT" \
      --output-root "$ESM_OUTPUT_ROOT" \
      --figure-root "$ESM_FIGURE_ROOT" \
      --overwrite \
      "${latent_args[@]}"
}

run_esm_raw() {
  local seed="$1"
  local epochs="$2"
  local min_epochs="$3"
  local patience="$4"
  local save_latents="$5"
  local log_file="$LOG_ROOT/esm_vicreg_raw_complete/seed_${seed}.log"
  local latent_args=(--save-latents)
  if [[ "$save_latents" != "1" ]]; then
    latent_args=(--no-save-latents)
  fi

  run_and_log "$log_file" \
    "$PYTHON_BIN" "$ESM_RAW_SCRIPT" \
      --seed "$seed" \
      --epochs "$epochs" \
      --min-epochs "$min_epochs" \
      --patience "$patience" \
      --train-csv "$TRAIN_CSV" \
      --val-csv "$VAL_CSV" \
      --test-csv "$TEST_CSV" \
      --immrep-csv "$IMMREP_CSV" \
      --pretrained-embed-root "$PRETRAINED_ROOT" \
      --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP" \
      --checkpoint-root "$ESM_RAW_CHECKPOINT_ROOT" \
      --output-root "$ESM_RAW_OUTPUT_ROOT" \
      --figure-root "$ESM_RAW_FIGURE_ROOT" \
      --overwrite \
      "${latent_args[@]}"
}

case "$MODE" in
  smoke)
    echo "Running smoke test. This writes to seed_31 folders and is intended to be overwritten by the full run."
    run_onehot 31 1 1 1 0
    run_esm 31 1 1 1 0
    ;;
  raw-smoke)
    echo "Run one-epoch raw ESMC VICReg smoke test on seed 31."
    run_esm_raw 31 1 1 1 1
    ;;
  raw)
    echo "Running full three-seed raw ESMC VICReg experiments."
    SAVE_LATENTS="${SAVE_LATENTS:-1}"
    for seed in 31 37 43; do
      run_esm_raw "$seed" 30 10 10 "$SAVE_LATENTS"
    done
    "$PYTHON_BIN" "$COLLECT_SCRIPT" \
      --outputs-root "$BASE/outputs/workshop" \
      --out-dir "$BASE/outputs/workshop/consolidated"
    ;;
  full)
    echo "Running full three-seed workshop experiments. Existing seed folders will be overwritten."
    SAVE_LATENTS="${SAVE_LATENTS:-1}"
    for seed in 31 37 43; do
      run_onehot "$seed" 30 10 10 "$SAVE_LATENTS"
    done
    for seed in 31 37 43; do
      run_esm "$seed" 30 10 10 "$SAVE_LATENTS"
    done
    "$PYTHON_BIN" "$COLLECT_SCRIPT" \
      --outputs-root "$BASE/outputs/workshop" \
      --out-dir "$BASE/outputs/workshop/consolidated"
    ;;
  collect)
    echo "Collecting existing workshop metrics."
    "$PYTHON_BIN" "$COLLECT_SCRIPT" \
      --outputs-root "$BASE/outputs/workshop" \
      --out-dir "$BASE/outputs/workshop/consolidated"
    ;;
  *)
    echo "Usage: $0 {smoke|raw-smoke|raw|full|collect}" >&2
    exit 2
    ;;
esac
