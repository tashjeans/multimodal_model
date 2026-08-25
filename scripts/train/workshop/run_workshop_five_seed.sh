#!/usr/bin/env bash
# Five-seed paper rerun on current occurrence-matched data.
# Shared VICReg HPs: alpha=beta=25, delta=1, d=256 (val-best cell on seed 31).
# Writes to models/{checkpoints,outputs,figures}/workshop/*_complete/seed_*.
#
#   bash scripts/train/workshop/run_workshop_five_seed.sh
set -euo pipefail

REPO="/home/natasha/multimodal_model"
PYTHON_BIN="${PYTHON_BIN:-/home/natasha/miniconda3/envs/tcr-multimodal/bin/python}"
TRAIN_DIR="$REPO/scripts/train/workshop"

ONEHOT_SCRIPT="$TRAIN_DIR/train_onehot_vicreg_workshop.py"
ESM_SCRIPT="$TRAIN_DIR/train_esm_vicreg_workshop.py"
ESM_RAW_SCRIPT="$TRAIN_DIR/train_esm_vicreg_raw_workshop.py"
COLLECT_SCRIPT="$TRAIN_DIR/collect_workshop_metrics.py"

TRAIN_CSV="$REPO/data/train/train_multiview.csv"
VAL_CSV="$REPO/data/val/val_multiview.csv"
TEST_CSV="$REPO/data/test/test_multiview.csv"
IMMREP_CSV="$REPO/data/immrep_test/immrep_test_multiview.csv"

FINETUNED_ROOT="$REPO/models/embeddings/no_boltz_multiview_ids"
FINETUNED_IMMREP="$REPO/models/embeddings/immrep_test_set/test"
PRETRAINED_ROOT="$REPO/models/embeddings/raw_esmc_300m_multiview_ids"
PRETRAINED_IMMREP="$REPO/models/embeddings/raw_esmc_300m_multiview_ids/immrep_test"

BASE="$REPO/models"
CKPT_ROOT="$BASE/checkpoints/workshop"
OUT_ROOT="$BASE/outputs/workshop"
FIG_ROOT="$BASE/figures/workshop"
LOG_ROOT="$BASE/log_files/workshop_five_seed"
STATUS_CSV="$LOG_ROOT/five_seed_status.csv"

ALPHA=25
BETA=25
DELTA=1
D=256
SEEDS=(31 37 43 49 55)
MODELS=(onehot_vicreg raw_esm_vicreg lora_esm_vicreg)

paper_run_name() {
  case "$1" in
    onehot_vicreg) echo "onehot_vicreg_complete" ;;
    raw_esm_vicreg) echo "esm_vicreg_raw_complete" ;;
    lora_esm_vicreg) echo "esm_vicreg_finetuned_complete" ;;
    *) echo "unknown"; return 1 ;;
  esac
}

append_status() {
  mkdir -p "$LOG_ROOT"
  if [[ ! -f "$STATUS_CSV" ]]; then
    echo "timestamp,model,alpha,beta,delta,d,seed,status,reason,log" > "$STATUS_CSV"
  fi
  echo "$(date -Is),$1,$ALPHA,$BETA,$DELTA,$D,$2,$3,$4,$5" >> "$STATUS_CSV"
}

run_one_job() {
  local model="$1" seed="$2"
  local run_name
  run_name="$(paper_run_name "$model")"
  local log_file="$LOG_ROOT/${run_name}/seed_${seed}.log"
  mkdir -p "$(dirname "$log_file")"
  local ckpt="$CKPT_ROOT/${run_name}"
  local out="$OUT_ROOT/${run_name}"
  local fig="$FIG_ROOT/${run_name}"

  echo "==== $(date -Is)  $model  seed=$seed  alpha=beta=$ALPHA  d=$D ===="
  echo "Logging to: $log_file"

  local cmd=()
  case "$model" in
    onehot_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ONEHOT_SCRIPT"
        --seed "$seed" --epochs 30 --min-epochs 10 --patience 10
        --alpha "$ALPHA" --beta "$BETA" --delta "$DELTA" --d "$D"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "$run_name"
        --missing-chain-policy complete_only
        --save-latents
        --overwrite
      )
      ;;
    raw_esm_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ESM_RAW_SCRIPT"
        --seed "$seed" --epochs 30 --min-epochs 10 --patience 10
        --alpha "$ALPHA" --beta "$BETA" --delta "$DELTA" --d "$D"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --pretrained-embed-root "$PRETRAINED_ROOT"
        --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "$run_name"
        --save-latents
        --overwrite
      )
      ;;
    lora_esm_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ESM_SCRIPT"
        --seed "$seed" --epochs 30 --min-epochs 10 --patience 10
        --alpha "$ALPHA" --beta "$BETA" --delta "$DELTA" --d "$D"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --finetuned-embed-root "$FINETUNED_ROOT"
        --finetuned-immrep-shard-dir "$FINETUNED_IMMREP"
        --pretrained-embed-root "$PRETRAINED_ROOT"
        --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "$run_name"
        --save-latents
        --overwrite
      )
      ;;
    *)
      echo "Unknown model $model" >&2
      return 1
      ;;
  esac

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  return "$status"
}

mkdir -p "$LOG_ROOT"
echo "Five-seed paper rerun  alpha=beta=$ALPHA  delta=$DELTA  d=$D"
echo "Seeds: ${SEEDS[*]}"
echo "Writes under $OUT_ROOT/*_complete/seed_* (overwrites July folders for these seeds)."

for model in "${MODELS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    append_status "$model" "$seed" "running" "train" "$LOG_ROOT/$(paper_run_name "$model")/seed_${seed}.log"
    t0=$(date +%s)
    if run_one_job "$model" "$seed"; then
      t1=$(date +%s)
      append_status "$model" "$seed" "done" "ok_sec=$((t1 - t0))" "$LOG_ROOT/$(paper_run_name "$model")/seed_${seed}.log"
    else
      t1=$(date +%s)
      append_status "$model" "$seed" "failed" "fail_sec=$((t1 - t0))" "$LOG_ROOT/$(paper_run_name "$model")/seed_${seed}.log"
      echo "FAILED $model seed=$seed  (continuing)" >&2
    fi
  done
done

echo "Collecting workshop metrics..."
"$PYTHON_BIN" "$COLLECT_SCRIPT" \
  --outputs-root "$OUT_ROOT" \
  --out-dir "$OUT_ROOT/consolidated"
echo "FIVE-SEED FINISHED  $(date -Is)"
