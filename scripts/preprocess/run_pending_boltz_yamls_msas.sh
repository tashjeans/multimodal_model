#!/usr/bin/env bash
# Full jackhmmer + YAML build, then _chunks symlinks. Intended for tmux/screen.
set -euo pipefail

REPO="/home/natasha/multimodal_model"
SESSION_LOG_DIR="${REPO}/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${SESSION_LOG_DIR}/pending_boltz_yamls_msas_${TIMESTAMP}.log"

mkdir -p "${SESSION_LOG_DIR}"

export PATH="/home/natasha/miniconda3/envs/tcr-multimodal/bin:${PATH}"
PYTHON="${REPO}/scripts/preprocess/build_pending_boltz_yamls_msas.py"

# 4 parallel pairs × 6 jackhmmer threads ≈ 24 cores on a 28-core host
MAX_WORKERS="${MAX_WORKERS:-4}"
JACK_CPU="${JACK_CPU:-6}"

exec > >(tee -a "${LOG}") 2>&1

echo "=== pending boltz YAML/MSA run ==="
echo "started: $(date -Is)"
echo "log: ${LOG}"
echo "host: $(hostname)"
echo "cpus: $(nproc)"
echo "max_workers=${MAX_WORKERS} jack_cpu=${JACK_CPU}"
echo "jackhmmer: $(which jackhmmer)"
echo

cd "${REPO}"

echo "--- phase 1: jackhmmer + YAML (val-decoys, test-decoys, immrep) ---"
python3 "${PYTHON}" \
  --targets val-decoys test-decoys immrep \
  --max-workers "${MAX_WORKERS}" \
  --jack-cpu "${JACK_CPU}"

echo
echo "--- phase 2: _chunks symlinks ---"
python3 "${PYTHON}" \
  --targets val-decoys test-decoys immrep \
  --link-chunks-only

echo
echo "finished: $(date -Is)"
echo "report: ${REPO}/data/build_pending_boltz_yamls_msas_report.json"
