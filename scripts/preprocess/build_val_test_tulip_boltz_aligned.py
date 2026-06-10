#!/usr/bin/env python3
"""Build val/test CSVs with legacy positive pair_ids and tulip decoy negatives only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

SPLITS = {
    "val": {
        "strict": REPO / "data/val/val_df_clean_pos_tulip_decoys_strict.csv",
        "pos_neg": REPO / "data/val/val_df_clean_pos_neg.csv",
        "out": REPO / "data/val/val_df_clean_pos_tulip_decoys.csv",
    },
    "test": {
        "strict": REPO / "data/test/test_df_clean_pos_tulip_decoys_strict.csv",
        "pos_neg": REPO / "data/test/test_df_clean_pos_neg.csv",
        "out": REPO / "data/test/test_df_clean_pos_tulip_decoys.csv",
    },
}

KEY = ["Peptide", "HLA_sequence", "TCR_full"]


def build_split(name: str, paths: dict) -> dict:
    strict = pd.read_csv(paths["strict"])
    pos_neg = pd.read_csv(paths["pos_neg"])
    pos_lookup = pos_neg.loc[pos_neg["binding_flag"] == 1].set_index("pair_id")

    positives = strict.loc[strict["binding_flag"] == 1].copy()
    decoys = strict.loc[strict["binding_flag"] == 0].copy()

    missing_source = positives["source_pair_id"].isna().sum()
    if missing_source:
        raise ValueError(f"{name}: {missing_source} positives missing source_pair_id")

    not_in_pos_neg = ~positives["source_pair_id"].isin(pos_lookup.index)
    if not_in_pos_neg.any():
        bad = positives.loc[not_in_pos_neg, ["pair_id", "source_pair_id"]].head(10)
        raise ValueError(f"{name}: source_pair_id not in pos_neg positives:\n{bad}")

    seq_mismatch = []
    for _, row in positives.iterrows():
        old = pos_lookup.loc[row["source_pair_id"]]
        if any(row[k] != old[k] for k in KEY):
            seq_mismatch.append(row["source_pair_id"])
    if seq_mismatch:
        raise ValueError(
            f"{name}: sequence mismatch vs pos_neg for {len(seq_mismatch)} positives "
            f"(e.g. {seq_mismatch[:5]})"
        )

    positives["pair_id"] = positives["source_pair_id"]

    out = pd.concat([positives, decoys], ignore_index=True)
    if out["pair_id"].duplicated().any():
        dupes = out.loc[out["pair_id"].duplicated(keep=False), "pair_id"].unique()[:10]
        raise ValueError(f"{name}: duplicate pair_id after merge: {dupes}")

    immrep_like = out["pair_id"].str.match(r"^(negative_pair_|immrep_pair_)")
    if immrep_like.any():
        raise ValueError(f"{name}: immrep/legacy negative ids in output: {immrep_like.sum()}")

    renamed_pos = (strict.loc[strict["binding_flag"] == 1, "pair_id"] != positives["pair_id"]).sum()
    out.to_csv(paths["out"], index=False)

    return {
        "split": name,
        "output": str(paths["out"]),
        "rows": int(len(out)),
        "positives": int(len(positives)),
        "decoys": int(len(decoys)),
        "unique_pair_id": int(out["pair_id"].nunique()),
        "positives_renamed_from_test_positive": int(renamed_pos),
        "decoy_pair_id_prefix": decoys["pair_id"].iloc[0].rsplit("_", 1)[0],
    }


def main() -> None:
    reports = [build_split(name, paths) for name, paths in SPLITS.items()]
    report_path = REPO / "data/val_test_tulip_boltz_aligned_report.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
