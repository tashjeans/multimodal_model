#!/usr/bin/env python3
"""Export the seed-31 HPO grid into a paper-appendix CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
HPO_ROOT = REPO / "models/outputs/workshop_hpo"
OUT_PATH = REPO / "models/outputs/workshop/paper_analysis/tables/hpo_grid_seed31.csv"

FAMILY_MODEL = {
    "onehot_vicreg_complete": "onehot_vicreg",
    "esm_vicreg_raw_complete": "esm_vicreg",
    "esm_vicreg_finetuned_complete": "esm_vicreg",
}
FAMILY_LABEL = {
    "onehot_vicreg_complete": "onehot_vicreg",
    "esm_vicreg_raw_complete": "raw_esmc_vicreg",
    "esm_vicreg_finetuned_complete": "lora_esmc_vicreg",
}
CORE = [
    "peptide_weighted_auroc",
    "peptide_macro_auroc",
    "global_auroc",
    "peptide_weighted_auc0.1_mcclish",
    "global_auc0.1_mcclish",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hpo-root", type=Path, default=HPO_ROOT)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    rows = []
    for folder, model_key in FAMILY_MODEL.items():
        for summary_path in sorted((args.hpo_root / folder).glob("a*_d*/seed_31/summary.json")):
            s = json.loads(summary_path.read_text())
            cfg = s.get("config", {})
            row = {
                "model": FAMILY_LABEL[folder],
                "alpha": cfg.get("alpha"),
                "beta": cfg.get("beta"),
                "delta": cfg.get("delta"),
                "d": cfg.get("d"),
                "seed": s.get("seed", cfg.get("seed")),
                "best_epoch": s.get("best_epoch"),
                "cell": summary_path.parent.parent.name,
            }
            for split, models in s.get("metrics", {}).items():
                metrics = models.get(model_key, {})
                for key in CORE:
                    if key in metrics:
                        row[f"{split}_{key}"] = metrics[key]
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model", "alpha", "d"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}  n={len(df)}", flush=True)


if __name__ == "__main__":
    main()
