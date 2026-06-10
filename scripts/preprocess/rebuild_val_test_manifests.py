#!/usr/bin/env python3
"""
Rebuild val/test manifests: keep positives, drop legacy negative_pair_* rows,
append tulip decoy rows with archive YAML paths.

Renames existing manifests to *_with_immrep_negatives_old.csv before writing new ones.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import pandas as pd
import yaml

BASE_DIR = Path("/home/natasha/multimodal_model")
MANIFEST_DIR = BASE_DIR / "manifests"
REPORT_PATH = BASE_DIR / "data/rebuild_val_test_manifests_report.json"

SPLITS = {
    "val": {
        "old_manifest": MANIFEST_DIR / "val_manifest.csv",
        "archived_manifest": MANIFEST_DIR / "val_manifest_with_immrep_negatives_old.csv",
        "new_manifest": MANIFEST_DIR / "val_manifest.csv",
        "decoys_csv": BASE_DIR / "data/val/val_df_clean_pos_tulip_decoys.csv",
        "yaml_arch": "data/val/_archive_root_yamls",
    },
    "test": {
        "old_manifest": MANIFEST_DIR / "test_manifest.csv",
        "archived_manifest": MANIFEST_DIR / "test_manifest_with_immrep_negatives_old.csv",
        "new_manifest": MANIFEST_DIR / "test_manifest.csv",
        "decoys_csv": BASE_DIR / "data/test/test_df_clean_pos_tulip_decoys.csv",
        "yaml_arch": "data/test/_archive_root_yamls",
    },
}

MANIFEST_COLS = ["pair_id", "yaml_path", "pep_len", "tcra_len", "tcrb_len", "hla_len", "binding_flag"]


def normalise_seq(s) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip().upper()
    return s if s and s.lower() != "nan" else ""


def chain_lengths_from_yaml(yaml_path: Path) -> tuple[int, int, int, int]:
    """Return pep_len, tcra_len, tcrb_len, hla_len from Boltz YAML (ids A/B/C/D)."""
    doc = yaml.safe_load(yaml_path.read_text())
    lens = {"A": 0, "B": 0, "C": 0, "D": 0}
    for entry in doc.get("sequences", []):
        prot = entry.get("protein") or {}
        cid = str(prot.get("id", "")).strip()
        seq = normalise_seq(prot.get("sequence", ""))
        if cid in lens:
            lens[cid] = len(seq)
    return lens["C"], lens["A"], lens["B"], lens["D"]


def manifest_row_from_yaml(pair_id: str, yaml_rel: str, binding_flag: int) -> dict:
    yaml_path = BASE_DIR / yaml_rel
    if not yaml_path.is_file() or yaml_path.stat().st_size < 50:
        raise FileNotFoundError(f"Missing or empty YAML: {yaml_path}")
    pep_len, tcra_len, tcrb_len, hla_len = chain_lengths_from_yaml(yaml_path)
    return {
        "pair_id": pair_id,
        "yaml_path": yaml_rel,
        "pep_len": pep_len,
        "tcra_len": tcra_len,
        "tcrb_len": tcrb_len,
        "hla_len": hla_len,
        "binding_flag": binding_flag,
    }


def is_legacy_negative(pair_id: str) -> bool:
    return str(pair_id).startswith("negative_pair_")


def rebuild_split(split: str, cfg: dict) -> dict:
    old_path = cfg["old_manifest"]
    if not old_path.is_file():
        raise FileNotFoundError(f"Expected manifest at {old_path}")

    old = pd.read_csv(old_path)
    for col in MANIFEST_COLS:
        if col not in old.columns:
            raise ValueError(f"{split}: manifest missing column {col}")

    positives = old[~old["pair_id"].map(is_legacy_negative)].copy()
    n_legacy = int(old["pair_id"].map(is_legacy_negative).sum())
    if (positives["binding_flag"] != 1).any():
        bad = positives[positives["binding_flag"] != 1]["pair_id"].head(5).tolist()
        raise ValueError(f"{split}: non-negative rows with binding_flag!=1: {bad}")

    decoys_df = pd.read_csv(cfg["decoys_csv"])
    decoys = decoys_df[decoys_df["binding_flag"] == 0].copy()
    yaml_arch = cfg["yaml_arch"]

    decoy_rows = []
    for pid in decoys["pair_id"].astype(str):
        yaml_rel = f"{yaml_arch}/{pid}.yaml"
        decoy_rows.append(manifest_row_from_yaml(pid, yaml_rel, binding_flag=0))

    decoy_manifest = pd.DataFrame(decoy_rows).reindex(columns=MANIFEST_COLS)
    new_manifest = pd.concat([positives.reindex(columns=MANIFEST_COLS), decoy_manifest], ignore_index=True)

    # Archive old manifest then write new
    archived = cfg["archived_manifest"]
    if archived.exists():
        raise FileExistsError(f"Archive already exists (refusing to overwrite): {archived}")
    shutil.move(str(old_path), str(archived))

    new_manifest.to_csv(cfg["new_manifest"], index=False)

    pos_ids = set(positives["pair_id"].astype(str))
    dec_ids = set(decoys["pair_id"].astype(str))
    overlap = pos_ids & dec_ids
    if overlap:
        raise ValueError(f"{split}: positive/decoy pair_id overlap: {list(overlap)[:5]}")

    return {
        "split": split,
        "archived_to": str(archived.relative_to(BASE_DIR)),
        "wrote": str(cfg["new_manifest"].relative_to(BASE_DIR)),
        "old_rows": len(old),
        "legacy_negative_pair_removed": n_legacy,
        "positives_kept": len(positives),
        "decoys_added": len(decoy_manifest),
        "new_rows": len(new_manifest),
        "unique_pair_id": int(new_manifest["pair_id"].nunique() == len(new_manifest)),
    }


def main() -> None:
    report = {"splits": []}
    for split, cfg in SPLITS.items():
        print(f"Rebuilding {split}...", flush=True)
        entry = rebuild_split(split, cfg)
        report["splits"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
