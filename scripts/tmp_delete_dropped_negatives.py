#!/usr/bin/env python3
"""Delete dropped-negative artifacts (Boltz, archive YAML, MSA, chunk symlink).

Safety: refuse any pair_id present in current manifests.
Uses CSV paths (no per-id glob). Keep-set verified via one-pass boltz index.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pandas as pd

BASE = Path("/home/natasha/multimodal_model")
REPORT = BASE / "data" / "dropped_negative_deletion_report.json"
LOG = BASE / "logs" / f"dropped_negative_deletion_{time.strftime('%Y%m%d_%H%M%S')}.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def index_boltz_pids(split: str) -> set[str]:
    """One-pass index of existing boltz_results_* under outputs/{split}/chunk_*."""
    root = BASE / f"outputs/{split}"
    pids: set[str] = set()
    if not root.is_dir():
        return pids
    for chunk in root.iterdir():
        if not chunk.is_dir() or not chunk.name.startswith("chunk_"):
            continue
        for child in chunk.iterdir():
            if child.name.startswith("boltz_results_"):
                pids.add(child.name[len("boltz_results_") :])
    return pids


def rm_path(p: Path, kind: str, stats: dict) -> None:
    if not p.exists() and not p.is_symlink():
        stats[f"missing_{kind}"] += 1
        return
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
            stats[f"deleted_{kind}"] += 1
        elif p.is_dir():
            shutil.rmtree(p)
            stats[f"deleted_{kind}"] += 1
        else:
            stats[f"skipped_unknown_{kind}"] += 1
    except Exception as e:
        stats[f"error_{kind}"] += 1
        stats.setdefault("errors", []).append(f"{kind}: {p}: {e}")


def main() -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    all_stats: dict = {}
    log(f"START report={REPORT} log={LOG}")

    for split in ("val", "test"):
        log("=" * 64 + f" {split}")
        man = pd.read_csv(BASE / f"manifests/{split}_manifest.csv")
        decoys = pd.read_csv(BASE / f"data/{split}/{split}_df_clean_pos_tulip_decoys.csv")
        man_ids = set(man.pair_id.astype(str))
        decoys_ids = set(decoys.pair_id.astype(str))
        if man_ids != decoys_ids:
            raise SystemExit(f"ABORT {split}: manifest vs decoys id mismatch")

        df = pd.read_csv(
            BASE / f"data/{split}/_archive_csv/{split}_dropped_negative_artifacts_to_delete.csv"
        )
        pids = df.pair_id.astype(str).tolist()
        drop_set = set(pids)
        overlap = drop_set & man_ids
        if overlap:
            raise SystemExit(
                f"ABORT {split}: {len(overlap)} delete ids in current manifest "
                f"e.g. {list(overlap)[:5]}"
            )

        log(f"Indexing boltz under outputs/{split} ...")
        boltz_before = index_boltz_pids(split)
        retained = {p for p in man_ids if "epitope_uniform_decoy" in p}
        pos = {p for p in man_ids if p.startswith("pair_")}
        keep_before = {
            "retained_boltz": len(retained & boltz_before),
            "retained_n": len(retained),
            "pos_boltz": len(pos & boltz_before),
            "pos_n": len(pos),
            "boltz_total": len(boltz_before),
        }
        log(
            f"KEEP before: retained boltz {keep_before['retained_boltz']}/{keep_before['retained_n']} "
            f"pos {keep_before['pos_boltz']}/{keep_before['pos_n']} "
            f"(boltz_total={keep_before['boltz_total']})"
        )
        log(f"Deleting {len(pids)} dropped ids (4 artifact types)...")

        stats = {
            "deleted_boltz": 0,
            "deleted_yaml": 0,
            "deleted_msa": 0,
            "deleted_chunk": 0,
            "missing_boltz": 0,
            "missing_yaml": 0,
            "missing_msa": 0,
            "missing_chunk": 0,
            "error_boltz": 0,
            "error_yaml": 0,
            "error_msa": 0,
            "error_chunk": 0,
            "errors": [],
        }

        for i, row in enumerate(df.itertuples(index=False), 1):
            pid = str(row.pair_id)
            if pid in man_ids:
                raise SystemExit(f"ABORT mid-loop: {pid} in manifest")

            # 1 Boltz
            bdirs = [BASE / p for p in str(row.boltz_dirs).split(";") if p and p != "nan"]
            if not bdirs:
                stats["missing_boltz"] += 1
            for b in bdirs:
                rm_path(b, "boltz", stats)

            # 2 archive YAML
            y = BASE / str(row.yaml_path) if str(row.yaml_path) not in ("", "nan") else None
            if y is None:
                stats["missing_yaml"] += 1
            else:
                rm_path(y, "yaml", stats)

            # 3 MSA dir
            m = BASE / str(row.msa_dir) if str(row.msa_dir) not in ("", "nan") else None
            if m is None:
                stats["missing_msa"] += 1
            else:
                rm_path(m, "msa", stats)

            # 4 chunk YAML symlink(s)
            chunks = [BASE / p for p in str(row.chunk_yamls).split(";") if p and p != "nan"]
            if not chunks:
                stats["missing_chunk"] += 1
            for c in chunks:
                rm_path(c, "chunk", stats)

            if i % 100 == 0 or i == len(pids):
                elapsed = time.time() - t0
                log(
                    f"  {split}: {i}/{len(pids)}  "
                    f"boltz={stats['deleted_boltz']} yaml={stats['deleted_yaml']} "
                    f"msa={stats['deleted_msa']} chunk={stats['deleted_chunk']} "
                    f"({elapsed:.0f}s)"
                )

        log(f"Re-indexing boltz under outputs/{split} ...")
        boltz_after = index_boltz_pids(split)
        keep_after = {
            "retained_boltz": len(retained & boltz_after),
            "pos_boltz": len(pos & boltz_after),
            "boltz_total": len(boltz_after),
        }
        log(
            f"KEEP after: retained boltz {keep_after['retained_boltz']}/{len(retained)} "
            f"pos {keep_after['pos_boltz']}/{len(pos)} "
            f"(boltz_total={keep_after['boltz_total']})"
        )
        if (
            keep_after["retained_boltz"] != keep_before["retained_boltz"]
            or keep_after["pos_boltz"] != keep_before["pos_boltz"]
        ):
            raise SystemExit(f"KEEP SET DAMAGED on {split}")

        # residual: dropped ids should not appear in boltz index / yaml / msa / chunk
        residual_b = len(drop_set & boltz_after)
        residual_y = sum(
            1 for pid in pids if (BASE / f"data/{split}/_archive_root_yamls/{pid}.yaml").exists()
        )
        residual_m = sum(
            1
            for pid in pids
            if (BASE / f"data/raw/MSA/jackhmmer_msas/{split}/{pid}").is_dir()
        )
        residual_c = 0
        chunks_root = BASE / f"data/{split}/_chunks"
        if chunks_root.is_dir():
            for chunk in chunks_root.iterdir():
                if not chunk.is_dir():
                    continue
                for pid in drop_set:
                    if (chunk / f"{pid}.yaml").exists() or (chunk / f"{pid}.yaml").is_symlink():
                        residual_c += 1
        log(
            f"Residual dropped: boltz={residual_b} yaml={residual_y} "
            f"msa={residual_m} chunk={residual_c}"
        )
        if residual_b or residual_y or residual_m or residual_c:
            raise SystemExit(f"Residuals remain on {split}")

        all_stats[split] = {
            "n_ids": len(pids),
            **{k: v for k, v in stats.items() if k != "errors"},
            "n_errors": len(stats["errors"]),
            "errors_sample": stats["errors"][:5],
            "keep_before": keep_before,
            "keep_after": keep_after,
            "residual": {
                "boltz": residual_b,
                "yaml": residual_y,
                "msa": residual_m,
                "chunk": residual_c,
            },
        }

    payload = {"elapsed_sec": round(time.time() - t0, 1), "splits": all_stats, "log": str(LOG)}
    REPORT.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"DONE {payload['elapsed_sec']}s report={REPORT}")


if __name__ == "__main__":
    main()
