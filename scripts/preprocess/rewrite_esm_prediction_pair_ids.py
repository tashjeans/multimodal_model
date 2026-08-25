#!/usr/bin/env python3
"""Rewrite val/test ESM prediction pair_ids from strict IDs to multiview (tulip) IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

CONTENT_COLS = ["Peptide", "HLA_sequence", "TCR_full", "binding_flag"]

SPLITS = {
    "val": {
        "strict": REPO / "data/val/val_df_clean_pos_tulip_decoys_strict.csv",
        "tulip": REPO / "data/val/val_df_clean_pos_tulip_decoys.csv",
    },
    "test": {
        "strict": REPO / "data/test/test_df_clean_pos_tulip_decoys_strict.csv",
        "tulip": REPO / "data/test/test_df_clean_pos_tulip_decoys.csv",
    },
}


def build_id_map(split: str, strict_path: Path, tulip_path: Path) -> Dict[str, str]:
    strict = pd.read_csv(strict_path)
    tulip = pd.read_csv(tulip_path)
    if len(strict) != len(tulip):
        raise ValueError(f"{split}: strict rows {len(strict)} != tulip rows {len(tulip)}")

    for col in CONTENT_COLS:
        if not strict[col].astype(str).equals(tulip[col].astype(str)):
            raise ValueError(f"{split}: content mismatch in column {col} between strict and tulip CSVs")

    id_map = dict(zip(strict["pair_id"].astype(str), tulip["pair_id"].astype(str)))
    if len(id_map) != len(strict):
        raise ValueError(f"{split}: id_map size {len(id_map)} != rows {len(strict)} (duplicate strict ids?)")

    pos_mask = strict["binding_flag"].astype(int) == 1
    pos_changed = int(
        (strict.loc[pos_mask, "pair_id"].astype(str) != tulip.loc[pos_mask, "pair_id"].astype(str)).sum()
    )
    dec_mask = ~pos_mask
    dec_unchanged = int(
        (strict.loc[dec_mask, "pair_id"].astype(str) == tulip.loc[dec_mask, "pair_id"].astype(str)).sum()
    )
    if pos_changed != int(pos_mask.sum()):
        raise ValueError(f"{split}: expected all positives to change id, got {pos_changed}/{int(pos_mask.sum())}")
    if dec_unchanged != int(dec_mask.sum()):
        raise ValueError(f"{split}: expected all decoy ids unchanged, got {dec_unchanged}/{int(dec_mask.sum())}")

    return id_map


def rewrite_predictions(path: Path, id_map: Dict[str, str], split: str) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if "pair_id" not in df.columns:
        raise ValueError(f"{path}: missing pair_id column")

    if "legacy_pair_id" in df.columns:
        unchanged = df["pair_id"].astype(str).equals(df["legacy_pair_id"].astype(str))
        if not unchanged:
            raise ValueError(f"{path}: legacy_pair_id already present and differs from pair_id; refusing to overwrite")

    legacy = df["pair_id"].astype(str)
    missing = sorted(set(legacy) - set(id_map))
    if missing:
        raise KeyError(f"{split} {path.name}: {len(missing)} pair_id values not in strict->tulip map (e.g. {missing[:5]})")

    out = df.copy()
    out["legacy_pair_id"] = legacy
    out["pair_id"] = legacy.map(id_map)

    cols = list(out.columns)
    cols.remove("legacy_pair_id")
    pair_idx = cols.index("pair_id")
    cols.insert(pair_idx + 1, "legacy_pair_id")
    out = out[cols]

    out.to_csv(path, index=False)

    renamed = int((out["legacy_pair_id"] != out["pair_id"]).sum())
    return {
        "path": str(path),
        "split": split,
        "rows": int(len(out)),
        "renamed_pair_ids": renamed,
        "identity_pair_ids": int(len(out) - renamed),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--val-pred",
        default=str(
            REPO
            / "models/checkpoints/hpo_training/plain_vicreg_tulip_simple/"
            "tulip_decoys_plain_vicreg__seed31__lr0.0003__a25.0__b25.0__dlt1.0__val_predictions.csv"
        ),
    )
    p.add_argument(
        "--test-pred",
        default=str(
            REPO
            / "models/checkpoints/hpo_training/plain_vicreg_tulip_simple/"
            "tulip_decoys_plain_vicreg__seed31__lr0.0003__a25.0__b25.0__dlt1.0__test_predictions.csv"
        ),
    )
    p.add_argument(
        "--report",
        default=str(REPO / "models/checkpoints/hpo_training/plain_vicreg_tulip_simple/rewrite_prediction_pair_ids_report.json"),
    )
    args = p.parse_args()

    reports = []
    for split, cfg in SPLITS.items():
        id_map = build_id_map(split, cfg["strict"], cfg["tulip"])
        pred_path = Path(args.val_pred if split == "val" else args.test_pred)
        reports.append(rewrite_predictions(pred_path, id_map, split))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
