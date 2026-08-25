#!/usr/bin/env bash
# Regenerate workshop paper tables/figures from the 5-seed d=256 runs.
# Does not retrain.
#
#   bash scripts/train/workshop/run_workshop_paper_analysis.sh
set -euo pipefail

REPO="/home/natasha/multimodal_model"
PYTHON_BIN="${PYTHON_BIN:-/home/natasha/miniconda3/envs/tcr-multimodal/bin/python}"
TRAIN_DIR="$REPO/scripts/train/workshop"
LOG_ROOT="$REPO/models/log_files/workshop_paper_analysis"
STATUS_CSV="$LOG_ROOT/analysis_status.csv"
SEEDS=(31 37 43 49 55)
RUN_NAMES=(onehot_vicreg_complete esm_vicreg_raw_complete esm_vicreg_finetuned_complete)

mkdir -p "$LOG_ROOT"
if [[ ! -f "$STATUS_CSV" ]]; then
  echo "timestamp,step,status,reason,log" > "$STATUS_CSV"
fi

append_status() {
  echo "$(date -Is),$1,$2,$3,$4" >> "$STATUS_CSV"
}

run_step() {
  local name="$1"
  shift
  local log="$LOG_ROOT/${name}.log"
  echo "==== $(date -Is)  $name ===="
  echo "Logging to: $log"
  append_status "$name" "running" "start" "$log"
  local t0 t1
  t0=$(date +%s)
  set +e
  "$@" 2>&1 | tee "$log"
  local status=${PIPESTATUS[0]}
  set -e
  t1=$(date +%s)
  if [[ $status -eq 0 ]]; then
    append_status "$name" "done" "ok_sec=$((t1 - t0))" "$log"
  else
    append_status "$name" "failed" "fail_sec=$((t1 - t0)) exit=$status" "$log"
    echo "FAILED $name  (continuing)" >&2
  fi
  return 0
}

echo "Paper analysis  seeds=${SEEDS[*]}"
echo "Writes under $REPO/models/{outputs,figures}/workshop/paper_analysis/"

run_step 01_collect_metrics \
  "$PYTHON_BIN" "$TRAIN_DIR/collect_workshop_metrics.py" \
    --outputs-root "$REPO/models/outputs/workshop" \
    --out-dir "$REPO/models/outputs/workshop/consolidated" \
    --run-names "${RUN_NAMES[@]}"

run_step 02_main_paper_results \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_workshop_paper_results.py" \
    --seeds "${SEEDS[@]}"

run_step 03_peptide_balanced_geometry \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_peptide_balanced_geometry.py" \
    --seeds "${SEEDS[@]}" --split test

run_step 04_geometry_multipanel \
  "$PYTHON_BIN" "$TRAIN_DIR/plot_geometry_multipanel.py" \
    --split test --seeds "${SEEDS[@]}"

run_step 05_category_breakdown \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_category_breakdown.py" \
    --seeds "${SEEDS[@]}" \
    --val-csv "$REPO/data/val/val_multiview.csv" \
    --test-csv "$REPO/data/test/test_multiview.csv"

run_step 06_auc_by_training_distance \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_auc_by_training_distance.py" \
    --seeds "${SEEDS[@]}"

run_step 07_negative_difficulty_matched \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_negative_difficulty_matched.py" \
    --seeds "${SEEDS[@]}"

run_step 08_negative_set_difficulty \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_negative_set_difficulty.py" \
    --seeds "${SEEDS[@]}"

run_step 09_immrep_transfer_stage \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_immrep_transfer_stage_diagnostic.py" \
    --seeds "${SEEDS[@]}" --device cuda

run_step 10_immrep_unnormalised_mse \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_immrep_stage_unnormalised_mse.py" \
    --seeds "${SEEDS[@]}" --device cuda

run_step 11_immrep_failure \
  "$PYTHON_BIN" "$TRAIN_DIR/analyse_immrep_failure_diagnostics.py" \
    --seeds "${SEEDS[@]}"

run_step 12_hpo_appendix_table \
  "$PYTHON_BIN" "$TRAIN_DIR/export_workshop_hpo_table.py"

echo "PAPER ANALYSIS FINISHED  $(date -Is)"
echo "Status: $STATUS_CSV"
