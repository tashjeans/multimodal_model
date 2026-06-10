#!/usr/bin/env python3
"""Report ESM shard and Boltz embedding coverage vs *_multiview.csv files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Set

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
EMBED_ROOT = REPO / "models/embeddings/no_boltz_multiview_ids"
LENGTH_COLS = ["pep_len", "tcra_len", "tcrb_len", "hla_len"]

SPLITS = {
    "train": {
        "multiview": REPO / "data/train/train_multiview.csv",
        "boltz_root": REPO / "outputs/train",
        "esm_subdir": "train",
    },
    "val": {
        "multiview": REPO / "data/val/val_multiview.csv",
        "boltz_root": REPO / "outputs/val",
        "esm_subdir": "val",
    },
    "test": {
        "multiview": REPO / "data/test/test_multiview.csv",
        "boltz_root": REPO / "outputs/test",
        "esm_subdir": "test",
    },
    "immrep_test": {
        "multiview": REPO / "data/immrep_test/immrep_test_multiview.csv",
        "boltz_root": REPO / "outputs_data/immrep_test",
        "esm_subdir": None,
    },
}


def load_shard_pair_ids(shard_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    if not shard_dir.exists():
        return ids
    for sp in sorted(shard_dir.glob("shard_*.pt")):
        data = torch.load(sp, map_location="cpu")
        for batch in data:
            for pid in batch["pair_id"]:
                ids.add(pid.decode("utf-8") if isinstance(pid, bytes) else str(pid))
    return ids


def boltz_path_ok(repo: Path, rel_path: str) -> bool:
    if not rel_path or str(rel_path).lower() in {"nan", "none"}:
        return False
    return (repo / rel_path).is_file()


def analyze_split(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    mv = pd.read_csv(cfg["multiview"])
    mv_ids = set(mv["pair_id"].astype(str))
    complete = (mv[LENGTH_COLS] > 0).all(axis=1)
    complete_ids = set(mv.loc[complete, "pair_id"].astype(str))

    has_boltz_col = mv["boltz_embedding_npz"].fillna("").astype(str).str.len() > 0
    boltz_col_ids = set(mv.loc[has_boltz_col, "pair_id"].astype(str))
    boltz_exists_ids = set(
        mv.loc[has_boltz_col & mv["boltz_embedding_npz"].apply(lambda p: boltz_path_ok(REPO, str(p))), "pair_id"].astype(str)
    )

    esm_ids: Set[str] = set()
    if cfg["esm_subdir"]:
        esm_ids = load_shard_pair_ids(EMBED_ROOT / cfg["esm_subdir"])

    report: Dict[str, Any] = {
        "split": name,
        "multiview_rows": int(len(mv)),
        "multiview_unique_pair_id": int(len(mv_ids)),
        "complete_chains": int(complete.sum()),
        "incomplete_chains": int((~complete).sum()),
        "boltz_path_in_csv": int(has_boltz_col.sum()),
        "boltz_path_exists_on_disk": int(len(boltz_exists_ids)),
        "boltz_missing_in_csv": int((~has_boltz_col).sum()),
        "esm_shard_unique_ids": int(len(esm_ids)) if esm_ids else None,
        "esm_shard_dir": str(EMBED_ROOT / cfg["esm_subdir"]) if cfg["esm_subdir"] else None,
    }

    if esm_ids:
        report["multiview_in_esm"] = int(len(mv_ids & esm_ids))
        report["esm_in_multiview"] = int(len(esm_ids & mv_ids))
        report["multiview_not_in_esm"] = int(len(mv_ids - esm_ids))
        report["esm_not_in_multiview"] = int(len(esm_ids - mv_ids))
        report["esm_and_boltz_exists"] = int(len(esm_ids & boltz_exists_ids))
        report["esm_and_boltz_and_complete"] = int(len(esm_ids & boltz_exists_ids & complete_ids))
        report["trainable_esm_boltz_complete"] = report["esm_and_boltz_and_complete"]

    report["boltz_exists_not_in_multiview"] = int(len(boltz_exists_ids - mv_ids))
    report["complete_and_boltz_exists"] = int(len(complete_ids & boltz_exists_ids))
    report["complete_and_boltz_missing"] = int(len(complete_ids - boltz_exists_ids))

    if "binding_flag" in mv.columns:
        for label in sorted(mv["binding_flag"].astype(int).unique()):
            sub = mv[mv["binding_flag"].astype(int) == label]
            sub_ids = set(sub["pair_id"].astype(str))
            report[f"binding_{label}_rows"] = int(len(sub))
            report[f"binding_{label}_boltz_csv"] = int(sub["boltz_embedding_npz"].fillna("").astype(str).str.len().gt(0).sum())
            report[f"binding_{label}_boltz_disk"] = int(len(sub_ids & boltz_exists_ids))
            if esm_ids:
                report[f"binding_{label}_esm"] = int(len(sub_ids & esm_ids))
                report[f"binding_{label}_esm_boltz_complete"] = int(len(sub_ids & esm_ids & boltz_exists_ids & complete_ids))

    report["ok_esm_equals_multiview"] = (esm_ids == mv_ids) if esm_ids else None
    report["ok_boltz_csv_equals_disk"] = int(has_boltz_col.sum()) == len(boltz_exists_ids)

    return report


def main() -> None:
    reports = [analyze_split(name, cfg) for name, cfg in SPLITS.items()]
    out_path = REPO / "data/multiview_coverage_report.json"
    out_path.write_text(json.dumps(reports, indent=2) + "\n")

    print("Multiview / ESM / Boltz coverage report")
    print("=" * 72)
    for r in reports:
        print(f"\n[{r['split']}] rows={r['multiview_rows']} complete={r['complete_chains']}")
        print(f"  Boltz: csv={r['boltz_path_in_csv']} disk={r['boltz_path_exists_on_disk']} missing_csv={r['boltz_missing_in_csv']}")
        if r["esm_shard_unique_ids"] is not None:
            print(f"  ESM:   shards={r['esm_shard_unique_ids']} | mv∩esm={r['multiview_in_esm']} | esm∩mv={r['esm_in_multiview']}")
            print(f"         mv not in esm={r['multiview_not_in_esm']} | esm not in mv={r['esm_not_in_multiview']}")
            print(f"         esm∩boltz∩complete (trainable)={r['trainable_esm_boltz_complete']}")
            print(f"  ESM==multiview: {r['ok_esm_equals_multiview']}")
        else:
            print("  ESM:   (no shard directory for this split)")
            print(f"         boltz∩complete={r['complete_and_boltz_exists']}")
        print(f"  Boltz csv==disk: {r['ok_boltz_csv_equals_disk']}")

    print(f"\nFull report: {out_path}")


if __name__ == "__main__":
    main()
