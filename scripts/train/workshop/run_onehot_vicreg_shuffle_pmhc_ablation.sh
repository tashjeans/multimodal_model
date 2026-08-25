#!/usr/bin/env bash
# Negative control: one-hot VICReg trained on shuffled TCR–pMHC pairings.
# Val/test/IMMREP remain real. Seeds 31/37/43/49/55.
#
# Hyperparameters and splits are held identical to the observed-cognate arm in
# run_workshop_five_seed.sh so the two arms differ only in the training pairing.
set -euo pipefail

ROOT="/home/natasha/multimodal_model"
PYTHON="${PYTHON:-/home/natasha/miniconda3/envs/tcr-multimodal/bin/python}"
SCRIPT="$ROOT/scripts/train/workshop/train_onehot_vicreg_workshop.py"
LOG_DIR="$ROOT/models/log_files/workshop/onehot_vicreg_shuffle_pmhc"
OUT_ROOT="$ROOT/models/outputs/workshop/onehot_vicreg_shuffle_pmhc"
CKPT_ROOT="$ROOT/models/checkpoints/workshop/onehot_vicreg_shuffle_pmhc"
FIG_ROOT="$ROOT/models/figures/workshop/onehot_vicreg_shuffle_pmhc"
RESULT="$OUT_ROOT/ablation_auroc_summary.csv"

mkdir -p "$LOG_DIR" "$OUT_ROOT"

TRAIN_CSV="$ROOT/data/train/train_multiview.csv"
VAL_CSV="$ROOT/data/val/val_multiview.csv"
TEST_CSV="$ROOT/data/test/test_multiview.csv"
IMMREP_CSV="$ROOT/data/immrep_test/immrep_test_multiview.csv"

ALPHA=25
BETA=25
DELTA=1
D=256
SEEDS=(31 37 43 49 55)

echo "=== onehot VICReg shuffled-pMHC ablation ==="
echo "start: $(date -Is)"
echo "python: $PYTHON"

for SEED in "${SEEDS[@]}"; do
  echo "----- seed ${SEED} -----"
  "$PYTHON" "$SCRIPT" \
    --seed "$SEED" \
    --epochs 30 --min-epochs 10 --patience 10 \
    --alpha "$ALPHA" --beta "$BETA" --delta "$DELTA" --d "$D" \
    --run-tag onehot_vicreg_shuffle_pmhc \
    --train-csv "$TRAIN_CSV" \
    --val-csv "$VAL_CSV" \
    --test-csv "$TEST_CSV" \
    --immrep-csv "$IMMREP_CSV" \
    --checkpoint-root "$CKPT_ROOT" \
    --output-root "$OUT_ROOT" \
    --figure-root "$FIG_ROOT" \
    --missing-chain-policy complete_only \
    --shuffle-train-pmhc \
    --overwrite \
    2>&1 | tee "$LOG_DIR/seed_${SEED}.log"
done

"$PYTHON" - <<'PY'
import json
from pathlib import Path
import pandas as pd

root = Path("/home/natasha/multimodal_model/models/outputs/workshop/onehot_vicreg_shuffle_pmhc")
rows = []
for seed in [31, 37, 43, 49, 55]:
    p = root / f"seed_{seed}" / "summary.json"
    s = json.loads(p.read_text())
    for split in ["val", "test", "immrep_test"]:
        if split not in s["metrics"]:
            continue
        for model in ["onehot_vicreg", "onehot_composition"]:
            m = s["metrics"][split][model]
            rows.append({
                "seed": seed,
                "split": split,
                "model": model,
                "best_epoch": s["best_epoch"],
                "global_auroc": m["global_auroc"],
                "peptide_weighted_auroc": m["peptide_weighted_auroc"],
                "auprc": m["auprc"],
            })
df = pd.DataFrame(rows)
out = root / "ablation_auroc_summary.csv"
df.to_csv(out, index=False)
print("\n=== SHUFFLE ABLATION AUROC SUMMARY ===")
print(df.to_string(index=False))
print("\n=== TEST onehot_vicreg mean±std ===")
t = df[(df.split == "test") & (df.model == "onehot_vicreg")]
print(f"global AUROC: {t.global_auroc.mean():.4f} ± {t.global_auroc.std(ddof=1):.4f}")
print(f"pep-w AUROC:  {t.peptide_weighted_auroc.mean():.4f} ± {t.peptide_weighted_auroc.std(ddof=1):.4f}")
print(f"wrote {out}")
print("ABLATION_COMPLETE")
PY

echo "end: $(date -Is)"
