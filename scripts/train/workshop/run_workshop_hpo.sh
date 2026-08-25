#!/usr/bin/env bash
# Workshop VICReg hyperparameter sweep (seed 31 only, no latents).
#
#   bash scripts/train/workshop/run_workshop_hpo.sh smoke
#   bash scripts/train/workshop/run_workshop_hpo.sh grid
#
# Isolated under models/{checkpoints,outputs,figures,log_files}/workshop_hpo/
# Does not write to paper *_complete folders.
set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "smoke" && "$MODE" != "grid" ]]; then
  echo "Usage: $0 {smoke|grid}" >&2
  exit 2
fi

REPO="/home/natasha/multimodal_model"
PYTHON_BIN="${PYTHON_BIN:-/home/natasha/miniconda3/envs/tcr-multimodal/bin/python}"
TRAIN_DIR="$REPO/scripts/train/workshop"

ONEHOT_SCRIPT="$TRAIN_DIR/train_onehot_vicreg_workshop.py"
ESM_SCRIPT="$TRAIN_DIR/train_esm_vicreg_workshop.py"
ESM_RAW_SCRIPT="$TRAIN_DIR/train_esm_vicreg_raw_workshop.py"

TRAIN_CSV="$REPO/data/train/train_multiview.csv"
VAL_CSV="$REPO/data/val/val_multiview.csv"
TEST_CSV="$REPO/data/test/test_multiview.csv"
IMMREP_CSV="$REPO/data/immrep_test/immrep_test_multiview.csv"

FINETUNED_ROOT="$REPO/models/embeddings/no_boltz_multiview_ids"
FINETUNED_IMMREP="$REPO/models/embeddings/immrep_test_set/test"
PRETRAINED_ROOT="$REPO/models/embeddings/raw_esmc_300m_multiview_ids"
PRETRAINED_IMMREP="$REPO/models/embeddings/raw_esmc_300m_multiview_ids/immrep_test"

HPO_ROOT="$REPO/models"
CKPT_ROOT="$HPO_ROOT/checkpoints/workshop_hpo"
OUT_ROOT="$HPO_ROOT/outputs/workshop_hpo"
FIG_ROOT="$HPO_ROOT/figures/workshop_hpo"
LOG_ROOT="$HPO_ROOT/log_files/workshop_hpo"
STATUS_CSV="$LOG_ROOT/hpo_status.csv"
PAPER_OUT="$HPO_ROOT/outputs/workshop"

SEED=31
ALPHAS=(1 25 50)
DS=(64 128 256)
DELTA=1

MODELS=(onehot_vicreg raw_esm_vicreg lora_esm_vicreg)

cell_tag() {
  local alpha="$1" d="$2"
  echo "a${alpha}_d${d}"
}

paper_run_name() {
  case "$1" in
    onehot_vicreg) echo "onehot_vicreg_complete" ;;
    raw_esm_vicreg) echo "esm_vicreg_raw_complete" ;;
    lora_esm_vicreg) echo "esm_vicreg_finetuned_complete" ;;
    *) echo "unknown"; return 1 ;;
  esac
}

hpo_rel_dir() {
  local model="$1" tag="$2"
  echo "$(paper_run_name "$model")/${tag}"
}

summary_matches() {
  local summary="$1" alpha="$2" d="$3"
  [[ -f "$summary" ]] || return 1
  "$PYTHON_BIN" - "$summary" "$alpha" "$d" "$DELTA" "$SEED" <<'PY'
import json, sys
path, alpha, d, delta, seed = sys.argv[1:]
with open(path) as f:
    s = json.load(f)
cfg = s.get("config", {})
ok = (
    float(cfg.get("alpha", s.get("alpha", float("nan")))) == float(alpha)
    and float(cfg.get("beta", s.get("beta", float("nan")))) == float(alpha)
    and float(cfg.get("delta", s.get("delta", float("nan")))) == float(delta)
    and int(cfg.get("d", s.get("d", -1))) == int(d)
    and int(s.get("seed", cfg.get("seed", -1))) == int(seed)
)
sys.exit(0 if ok else 1)
PY
}

should_skip_grid_cell() {
  local model="$1" alpha="$2" d="$3" tag="$4"
  # Do not skip July paper *_complete runs: those used a different val/test
  # negative protocol. Only skip a cell already completed in workshop_hpo/.
  local hpo="$OUT_ROOT/$(hpo_rel_dir "$model" "$tag")/seed_${SEED}/summary.json"
  if summary_matches "$hpo" "$alpha" "$d"; then
    echo "hpo"
    return 0
  fi
  return 1
}

append_status() {
  mkdir -p "$LOG_ROOT"
  if [[ ! -f "$STATUS_CSV" ]]; then
    echo "timestamp,mode,model,alpha,beta,delta,d,seed,status,reason,log" > "$STATUS_CSV"
  fi
  echo "$(date -Is),$MODE,$1,$2,$2,$DELTA,$3,$SEED,$4,$5,$6" >> "$STATUS_CSV"
}

run_one_job() {
  local model="$1" alpha="$2" d="$3" epochs="$4" min_epochs="$5" patience="$6"
  local dest_root="$7"   # relative dir under ckpt/out/fig, e.g. esm_vicreg_raw_complete/a1_d64
  local log_file="$8"
  local tag
  tag="$(cell_tag "$alpha" "$d")"
  mkdir -p "$(dirname "$log_file")"
  local ckpt="$CKPT_ROOT/${dest_root}"
  local out="$OUT_ROOT/${dest_root}"
  local fig="$FIG_ROOT/${dest_root}"

  echo "==== $(date -Is)  $model  alpha=beta=$alpha  d=$d  epochs=$epochs ===="
  echo "Logging to: $log_file"

  local cmd=()
  case "$model" in
    onehot_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ONEHOT_SCRIPT"
        --seed "$SEED" --epochs "$epochs" --min-epochs "$min_epochs" --patience "$patience"
        --alpha "$alpha" --beta "$alpha" --delta "$DELTA" --d "$d"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "hpo_$(paper_run_name "$model")_${tag}"
        --missing-chain-policy complete_only
        --overwrite
      )
      ;;
    raw_esm_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ESM_RAW_SCRIPT"
        --seed "$SEED" --epochs "$epochs" --min-epochs "$min_epochs" --patience "$patience"
        --alpha "$alpha" --beta "$alpha" --delta "$DELTA" --d "$d"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --pretrained-embed-root "$PRETRAINED_ROOT"
        --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "hpo_$(paper_run_name "$model")_${tag}"
        --no-save-latents
        --overwrite
      )
      ;;
    lora_esm_vicreg)
      cmd=(
        "$PYTHON_BIN" "$ESM_SCRIPT"
        --seed "$SEED" --epochs "$epochs" --min-epochs "$min_epochs" --patience "$patience"
        --alpha "$alpha" --beta "$alpha" --delta "$DELTA" --d "$d"
        --train-csv "$TRAIN_CSV" --val-csv "$VAL_CSV" --test-csv "$TEST_CSV" --immrep-csv "$IMMREP_CSV"
        --finetuned-embed-root "$FINETUNED_ROOT"
        --finetuned-immrep-shard-dir "$FINETUNED_IMMREP"
        --pretrained-embed-root "$PRETRAINED_ROOT"
        --pretrained-immrep-shard-dir "$PRETRAINED_IMMREP"
        --checkpoint-root "$ckpt" --output-root "$out" --figure-root "$fig"
        --run-tag "hpo_$(paper_run_name "$model")_${tag}"
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

run_smoke() {
  mkdir -p "$LOG_ROOT"
  local model="raw_esm_vicreg"
  local alpha=1 d=64
  local tag
  tag="$(cell_tag "$alpha" "$d")"
  local dest="_smoke/$(hpo_rel_dir "$model" "$tag")"
  local log="$LOG_ROOT/smoke_${model}_${tag}_seed${SEED}.log"
  echo "HPO smoke: 1 epoch raw ESMC  alpha=beta=$alpha  d=$d  seed=$SEED"
  echo "Writes under workshop_hpo/_smoke/ (not a grid cell)."
  append_status "$model" "$alpha" "$d" "running" "smoke" "$log"
  local t0 t1
  t0=$(date +%s)
  if run_one_job "$model" "$alpha" "$d" 1 1 1 "$dest" "$log"; then
    t1=$(date +%s)
    append_status "$model" "$alpha" "$d" "done" "smoke_ok_sec=$((t1 - t0))" "$log"
    echo "SMOKE OK  elapsed=$((t1 - t0))s  log=$log"
  else
    t1=$(date +%s)
    append_status "$model" "$alpha" "$d" "failed" "smoke_fail_sec=$((t1 - t0))" "$log"
    echo "SMOKE FAILED  elapsed=$((t1 - t0))s  log=$log" >&2
    exit 1
  fi
}

run_grid() {
  mkdir -p "$LOG_ROOT"
  echo "HPO grid: seed=$SEED  alpha=beta in ${ALPHAS[*]}  d in ${DS[*]}  delta=$DELTA"
  local model alpha d tag skip_from dest log t0 t1
  for model in "${MODELS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      for d in "${DS[@]}"; do
        tag="$(cell_tag "$alpha" "$d")"
        dest="$(hpo_rel_dir "$model" "$tag")"
        log="$LOG_ROOT/${model}/${tag}_seed${SEED}.log"
        if skip_from=$(should_skip_grid_cell "$model" "$alpha" "$d" "$tag"); then
          echo "SKIP $model $tag (already have matching summary: $skip_from)"
          append_status "$model" "$alpha" "$d" "skipped" "$skip_from" ""
          continue
        fi
        append_status "$model" "$alpha" "$d" "running" "grid" "$log"
        t0=$(date +%s)
        if run_one_job "$model" "$alpha" "$d" 30 10 10 "$dest" "$log"; then
          t1=$(date +%s)
          append_status "$model" "$alpha" "$d" "done" "grid_ok_sec=$((t1 - t0))" "$log"
        else
          t1=$(date +%s)
          append_status "$model" "$alpha" "$d" "failed" "grid_fail_sec=$((t1 - t0))" "$log"
          echo "FAILED $model $tag  (continuing)" >&2
        fi
      done
    done
  done
  echo "GRID FINISHED  $(date -Is)"
}

case "$MODE" in
  smoke) run_smoke ;;
  grid) run_grid ;;
esac
