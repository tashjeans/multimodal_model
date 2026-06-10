#!/usr/bin/env python3
"""
Phase 2: Move 4k legacy MSA dirs -> jackhmmer_msas/immrep_test/{immrep_pair_id}/

Uses immrep_negative_reloc_map.csv. No Boltz moves.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/home/natasha/multimodal_model")
RELOC_MAP = BASE_DIR / "data/immrep_test/immrep_negative_reloc_map.csv"
REPORT_OUT = BASE_DIR / "data/immrep_test/phase2_validation_report.json"
MSA_STEMS = ("tcra", "tcrb", "mhc")


def msa_complete(d: Path) -> bool:
    return d.is_dir() and all((d / f"{s}.filt.a3m").is_file() for s in MSA_STEMS)


def move_one(row: pd.Series) -> str:
    old = BASE_DIR / str(row["old_msa_dir"])
    new = BASE_DIR / str(row["new_msa_dir"])

    if new.is_dir() and msa_complete(new):
        return "skipped_dest_exists"
    if not old.is_dir():
        if new.is_dir() and msa_complete(new):
            return "skipped_dest_exists"
        return "missing_source"
    if new.exists():
        return "dest_exists_incomplete"

    new.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(new))
    if not msa_complete(new):
        return "moved_incomplete"
    return "moved"


def main() -> None:
    reloc = pd.read_csv(RELOC_MAP)
    stats: dict[str, int] = {}
    failures: list[dict] = []

    print(f"Moving {len(reloc)} MSA directories...", flush=True)
    for i, row in reloc.iterrows():
        status = move_one(row)
        stats[status] = stats.get(status, 0) + 1
        if status in {"missing_source", "dest_exists_incomplete", "moved_incomplete"}:
            failures.append(
                {
                    "immrep_pair_id": row["immrep_pair_id"],
                    "old_msa_dir": row["old_msa_dir"],
                    "new_msa_dir": row["new_msa_dir"],
                    "status": status,
                }
            )
        if (i + 1) % 500 == 0:
            print(f"  processed {i + 1}/{len(reloc)}", flush=True)

    immrep_root = BASE_DIR / "data/raw/MSA/jackhmmer_msas/immrep_test"
    ready = sum(1 for p in immrep_root.iterdir() if p.is_dir() and msa_complete(p)) if immrep_root.is_dir() else 0

    report = {
        "reloc_rows": len(reloc),
        "status_counts": stats,
        "msa_dirs_ready_under_immrep_test": ready,
        "failure_count": len(failures),
        "failures_sample": failures[:20],
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("Phase 2 complete.", flush=True)


if __name__ == "__main__":
    main()
