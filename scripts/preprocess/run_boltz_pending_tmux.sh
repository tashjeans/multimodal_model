#!/usr/bin/env bash
# Start full manifest-aware Boltz campaign in tmux (val+test -> outputs/, immrep -> outputs_data/).
#
#   bash scripts/preprocess/run_boltz_pending_tmux.sh          # attach to new session
#   bash scripts/preprocess/run_boltz_pending_tmux.sh attach # attach existing
#
# Smoke test first (on host GPU, not in sandbox):
#   bash scripts/preprocess/run_boltz_pending.sh smoke val_tulip_epitope_uniform_decoy_000000
set -euo pipefail

SESSION="${BOLTZ_TMUX_SESSION:-boltz_pending}"
REPO="/home/natasha/multimodal_model"
RUN_SH="${REPO}/scripts/preprocess/run_boltz_pending.sh"
LOG="${REPO}/outputs/boltz_v2_full_$(date +%Y%m%d_%H%M%S).log"

if [[ "${1:-}" == "attach" ]]; then
  exec tmux attach -t "$SESSION"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Use: tmux attach -t $SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$SESSION" -c "$REPO" \
  "bash -lc 'set -o pipefail; bash \"$RUN_SH\" run 2>&1 | tee \"$LOG\"; echo; echo finished: \$(date -Is); echo log: $LOG; exec bash -l'"

echo "Started tmux session: $SESSION"
echo "  attach: tmux attach -t $SESSION"
echo "  log:    $LOG"
echo "  ~10126 predictions (resume-safe; skips existing embeddings)"
