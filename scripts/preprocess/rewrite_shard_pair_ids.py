#!/usr/bin/env python3
"""Rewrite val/test ESM shard pair_ids from strict IDs to multiview (tulip) IDs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]

SRC_ROOT = REPO / "models/embeddings/no_boltz_train_swapped_tulip_decoys"
OUT_ROOT = REPO / "models/embeddings/no_boltz_multiview_ids"

CONTENT_COLS = ["Peptide", "HLA_sequence", "TCR_full", "binding_flag"]

SPLITS = {
    "val": {
        "strict": REPO / "data/val/val_df_clean_pos_tulip_decoys_strict.csv",
        "tulip": REPO / "data/val/val_df_clean_pos_tulip_decoys.csv",
        "rewrite": True,
    },
    "test": {
        "strict": REPO / "data/test/test_df_clean_pos_tulip_decoys_strict.csv",
        "tulip": REPO / "data/test/test_df_clean_pos_tulip_decoys.csv",
        "rewrite": True,
    },
    "train": {
        "rewrite": False,
    },
}


def to_str_pid(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


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

    # positives should change; decoys should be identity
    pos_mask = strict["binding_flag"].astype(int) == 1
    pos_changed = int((strict.loc[pos_mask, "pair_id"].astype(str) != tulip.loc[pos_mask, "pair_id"].astype(str)).sum())
    dec_mask = ~pos_mask
    dec_unchanged = int((strict.loc[dec_mask, "pair_id"].astype(str) == tulip.loc[dec_mask, "pair_id"].astype(str)).sum())
    if pos_changed != int(pos_mask.sum()):
        raise ValueError(f"{split}: expected all positives to change id, got {pos_changed}/{int(pos_mask.sum())}")
    if dec_unchanged != int(dec_mask.sum()):
        raise ValueError(f"{split}: expected all decoy ids unchanged, got {dec_unchanged}/{int(dec_mask.sum())}")

    return id_map


def rewrite_shard(in_path: Path, out_path: Path, id_map: Dict[str, str], split: str) -> Dict[str, int]:
    data = torch.load(in_path, map_location="cpu")
    if not isinstance(data, list):
        raise ValueError(f"{in_path}: expected list of batches, got {type(data)}")

    n_rows = 0
    n_mapped = 0
    for batch in data:
        old_ids = [to_str_pid(x) for x in batch["pair_id"]]
        new_ids: List[str] = []
        for old in old_ids:
            if old not in id_map:
                raise KeyError(f"{split} {in_path.name}: pair_id {old!r} not in strict->tulip map")
            new_ids.append(id_map[old])
            n_mapped += 1
        batch["pair_id"] = new_ids
        n_rows += len(new_ids)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    return {"batches": len(data), "rows": n_rows}


def copy_train_shards() -> Dict[str, Any]:
    src = SRC_ROOT / "train"
    dst = OUT_ROOT / "train"
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(src.glob("shard_*.pt"))
    for sp in shards:
        shutil.copy2(sp, dst / sp.name)
    return {"split": "train", "action": "copy", "shards": len(shards), "output_dir": str(dst)}


def rewrite_split(split: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    id_map = build_id_map(split, cfg["strict"], cfg["tulip"])
    src_dir = SRC_ROOT / split
    dst_dir = OUT_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)

    shard_reports = []
    for in_path in sorted(src_dir.glob("shard_*.pt")):
        out_path = dst_dir / in_path.name
        stats = rewrite_shard(in_path, out_path, id_map, split)
        shard_reports.append({"shard": in_path.name, **stats})

    return {
        "split": split,
        "action": "rewrite",
        "id_map_size": len(id_map),
        "positives_renamed": int(sum(1 for a, b in id_map.items() if a != b)),
        "identity_mappings": int(sum(1 for a, b in id_map.items() if a == b)),
        "output_dir": str(dst_dir),
        "shards": shard_reports,
    }


def verify_rewritten_split(split: str) -> Dict[str, Any]:
    mv = pd.read_csv(REPO / f"data/{split}/{split}_multiview.csv")
    mv_ids = set(mv["pair_id"].astype(str))

    shard_ids: List[str] = []
    shard_dir = OUT_ROOT / split
    for sp in sorted(shard_dir.glob("shard_*.pt")):
        data = torch.load(sp, map_location="cpu")
        for batch in data:
            shard_ids.extend(to_str_pid(x) for x in batch["pair_id"])

    shard_set = set(shard_ids)
    strict_prefixes = ("val_positive_", "test_positive_")
    leftover_strict = [p for p in shard_set if p.startswith(strict_prefixes)]

    dupes = len(shard_ids) - len(shard_set)
    report = {
        "split": split,
        "multiview_rows": int(len(mv)),
        "shard_rows": len(shard_ids),
        "shard_unique": len(shard_set),
        "duplicate_shard_ids": dupes,
        "leftover_strict_style_ids": leftover_strict[:5],
        "n_leftover_strict_style_ids": len(leftover_strict),
        "in_multiview_not_in_shards": len(mv_ids - shard_set),
        "in_shards_not_in_multiview": len(shard_set - mv_ids),
        "intersection": len(mv_ids & shard_set),
        "ok": (
            dupes == 0
            and len(leftover_strict) == 0
            and mv_ids == shard_set
            and len(shard_ids) == len(mv)
        ),
    }
    if not report["ok"]:
        if report["in_multiview_not_in_shards"]:
            sample = sorted(mv_ids - shard_set)[:5]
            report["sample_missing_from_shards"] = sample
        if report["in_shards_not_in_multiview"]:
            report["sample_extra_in_shards"] = sorted(shard_set - mv_ids)[:5]
    return report


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source shard root not found: {SRC_ROOT}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    reports = []

    reports.append(copy_train_shards())
    for split, cfg in SPLITS.items():
        if cfg.get("rewrite"):
            reports.append(rewrite_split(split, cfg))

    verify_reports = [verify_rewritten_split(s) for s in ["train", "val", "test"]]

    out = {"build": reports, "verify_shards": verify_reports}
    report_path = OUT_ROOT / "rewrite_report.json"
    report_path.write_text(json.dumps(out, indent=2) + "\n")

    for r in verify_reports:
        status = "OK" if r["ok"] else "FAIL"
        print(f"{r['split']} verify: {status} | shard_rows={r['shard_rows']} | intersection={r['intersection']}")
        if not r["ok"]:
            print(json.dumps(r, indent=2))

    if not all(r["ok"] for r in verify_reports):
        raise SystemExit("Shard rewrite verification failed")

    print(f"Wrote shards to: {OUT_ROOT}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
