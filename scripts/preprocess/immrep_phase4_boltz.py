#!/usr/bin/env python3
"""
Phase 4: Copy/rename 4k Boltz output trees to outputs_data/immrep_test.

Cross-disk copy (outputs -> outputs_data). Renames path components and
filenames: negative_pair_* -> immrep_pair_*.

Resume: skips destinations that already have embeddings. Removes source after
verified copy (--delete-source, default on).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/home/natasha/multimodal_model")
RELOC_MAP = BASE_DIR / "data/immrep_test/immrep_negative_reloc_map.csv"
REPORT_OUT = BASE_DIR / "data/immrep_test/phase4_validation_report.json"
PROGRESS_LOG = BASE_DIR / "data/immrep_test/phase4_progress.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def remap_name(name: str, legacy: str, immrep: str) -> str:
    return name.replace(legacy, immrep)


def has_embeddings(dest_root: Path, immrep: str) -> bool:
    patterns = [
        f"predictions/{immrep}/embeddings*.npz",
        f"predictions/{immrep}/embeddings_{immrep}.npz",
        "predictions/*/embeddings*.npz",
    ]
    for pat in patterns:
        if any(dest_root.glob(pat)):
            return True
    return any(dest_root.glob("**/embeddings*.npz"))


def copy_run_renamed(src_root: Path, dst_root: Path, legacy: str, immrep: str) -> None:
    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.parent.mkdir(parents=True, exist_ok=True)

    for src in sorted(src_root.rglob("*")):
        rel = src.relative_to(src_root)
        new_parts = [remap_name(p, legacy, immrep) for p in rel.parts]
        dst = dst_root / Path(*new_parts)
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def process_row(row: pd.Series, *, delete_source: bool) -> str:
    legacy = str(row["legacy_pair_id"])
    immrep = str(row["immrep_pair_id"])
    src = BASE_DIR / str(row["old_boltz_dir"])
    dst = BASE_DIR / str(row["new_boltz_dir"])

    if not src.is_dir():
        if dst.is_dir() and has_embeddings(dst, immrep):
            return "skipped_dest_exists"
        return "missing_source"

    if dst.is_dir() and has_embeddings(dst, immrep):
        if delete_source and src.is_dir():
            shutil.rmtree(src)
            return "skipped_dest_exists_removed_source"
        return "skipped_dest_exists"

    copy_run_renamed(src, dst, legacy, immrep)
    if not has_embeddings(dst, immrep):
        return "copied_missing_embeddings"

    if delete_source:
        shutil.rmtree(src)
        return "copied_and_removed_source"
    return "copied"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Process first N rows only (0=all)")
    parser.add_argument("--no-delete-source", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="Start row offset")
    args = parser.parse_args()

    delete_source = not args.no_delete_source
    reloc = pd.read_csv(RELOC_MAP)
    if args.start:
        reloc = reloc.iloc[args.start :]
    if args.limit:
        reloc = reloc.iloc[: args.limit]

    log(f"Phase 4 start | rows={len(reloc)} delete_source={delete_source} start={args.start}")

    stats: dict[str, int] = {}
    failures: list[dict] = []
    t0 = time.time()

    for i, row in reloc.iterrows():
        try:
            status = process_row(row, delete_source=delete_source)
        except Exception as e:
            status = "error"
            failures.append(
                {
                    "immrep_pair_id": row["immrep_pair_id"],
                    "old_boltz_dir": row["old_boltz_dir"],
                    "error": repr(e),
                }
            )
        stats[status] = stats.get(status, 0) + 1

        n = sum(stats.values())
        if n % 50 == 0 or status not in ("copied_and_removed_source", "skipped_dest_exists"):
            elapsed = time.time() - t0
            log(f"  [{n}/{len(reloc)}] {row['immrep_pair_id']} -> {status} ({elapsed:.0f}s)")

    immrep_out = BASE_DIR / "outputs_data/immrep_test"
    n_dest = sum(1 for _ in immrep_out.glob("chunk_*/boltz_results_*")) if immrep_out.is_dir() else 0

    report = {
        "rows_processed": len(reloc),
        "status_counts": stats,
        "boltz_dirs_under_outputs_data_immrep_test": n_dest,
        "failure_count": len(failures),
        "failures_sample": failures[:30],
        "elapsed_sec": round(time.time() - t0, 1),
        "delete_source": delete_source,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    log(f"Phase 4 done: {json.dumps(stats)}")
    log(f"Report: {REPORT_OUT}")


if __name__ == "__main__":
    main()
