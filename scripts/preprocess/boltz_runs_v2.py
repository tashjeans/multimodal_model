#!/usr/bin/env python3
"""
boltz_runs_v2.py — manifest-aware Boltz runner for val / test / immrep_test.

Differences from boltz_runs.py:
  - Filters YAMLs to pair_ids listed in each split's manifest (skips stale
    negative_pair_* symlinks still present under data/*/ _chunks).
  - immrep_test writes to outputs_data/immrep_test (separate mount).
  - --plan-only: print what would run without calling boltz.
  - --pair-id: run a single pair (for smoke tests).
  - --max-runs N: stop after N predictions (useful with --pair-id or sampling).

Reuse boltz_runs.py for low-level helpers (env check, CLI, resume logic).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# Reuse battle-tested helpers from v1
from boltz_runs import (  # noqa: E402 — same directory on sys.path when invoked as script
    BASE_DIR_DEFAULT,
    assert_correct_runtime,
    build_boltz_args,
    embeddings_exist_for_yaml,
    list_chunk_dirs,
    list_yamls,
    mark_done,
    run_cli,
    yaml_to_pair_dirname,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SPLIT_CONFIGS: Dict[str, dict] = {
    "val": {
        "chunks_root": BASE_DIR_DEFAULT / "data/val/_chunks",
        "out_root": BASE_DIR_DEFAULT / "outputs/val",
        "manifest": BASE_DIR_DEFAULT / "manifests/val_manifest.csv",
    },
    "test": {
        "chunks_root": BASE_DIR_DEFAULT / "data/test/_chunks",
        "out_root": BASE_DIR_DEFAULT / "outputs/test",
        "manifest": BASE_DIR_DEFAULT / "manifests/test_manifest.csv",
    },
    "immrep_test": {
        "chunks_root": BASE_DIR_DEFAULT / "data/immrep_test/_chunks",
        "out_root": BASE_DIR_DEFAULT / "outputs_data/immrep_test",
        "manifest": BASE_DIR_DEFAULT / "manifests/immrep_test_manifest.csv",
    },
}


def load_manifest_pair_ids(manifest_path: Path) -> Set[str]:
    df = pd.read_csv(manifest_path)
    if "pair_id" not in df.columns:
        raise ValueError(f"manifest missing pair_id: {manifest_path}")
    return set(df["pair_id"].astype(str))


def yaml_index_by_pair(chunks_root: Path) -> Dict[str, tuple[str, Path]]:
    """Map pair_id -> (chunk_name, yaml path for boltz input)."""
    idx: Dict[str, tuple[str, Path]] = {}
    for chunk_dir in list_chunk_dirs(chunks_root):
        for y in list_yamls(chunk_dir):
            stem = yaml_to_pair_dirname(y)
            if stem not in idx:
                # Keep symlink under _chunks; do not resolve() or outdir chunk is wrong.
                idx[stem] = (chunk_dir.name, y)
    return idx


def plan_split(split: str, cfg: dict, *, pair_id: Optional[str] = None) -> dict:
    allowed = load_manifest_pair_ids(cfg["manifest"])
    yaml_idx = yaml_index_by_pair(cfg["chunks_root"])
    out_root = Path(cfg["out_root"])

    stale_in_chunks = sorted(set(yaml_idx) - allowed)
    missing_from_chunks = sorted(allowed - set(yaml_idx))

    rows = []
    for pid in sorted(allowed):
        if pair_id and pid != pair_id:
            continue
        hit = yaml_idx.get(pid)
        if not hit:
            rows.append({"pair_id": pid, "status": "missing_yaml_in_chunks"})
            continue
        chunk, ypath = hit
        outdir = out_root / chunk
        done = embeddings_exist_for_yaml(ypath, outdir)
        rows.append(
            {
                "pair_id": pid,
                "chunk": chunk,
                "yaml": str(ypath.relative_to(BASE_DIR_DEFAULT)),
                "outdir": str(outdir.relative_to(BASE_DIR_DEFAULT)),
                "status": "done" if done else "todo",
            }
        )

    todo = [r for r in rows if r["status"] == "todo"]
    done = [r for r in rows if r["status"] == "done"]
    return {
        "split": split,
        "manifest": str(cfg["manifest"].relative_to(BASE_DIR_DEFAULT)),
        "manifest_rows": len(allowed),
        "yaml_in_chunks_total": len(yaml_idx),
        "stale_symlinks_in_chunks_not_in_manifest": len(stale_in_chunks),
        "stale_examples": stale_in_chunks[:5],
        "manifest_missing_from_chunks": len(missing_from_chunks),
        "todo": len(todo),
        "done": len(done),
        "todo_examples": todo[:5],
        "rows": rows,
    }


def run_manifest_split(
    split: str,
    cfg: dict,
    boltz_cfg: dict,
    base_dir: Path,
    *,
    plan_only: bool,
    pair_id: Optional[str],
    max_runs: Optional[int],
    progress_every: int,
    quiet: bool,
) -> dict:
    report = plan_split(split, cfg, pair_id=pair_id)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)

    if plan_only:
        return report

    todo_rows = [r for r in report["rows"] if r["status"] == "todo"]
    if not todo_rows:
        print(f"[{split}] nothing to run", flush=True)
        return report

    yaml_idx = yaml_index_by_pair(cfg["chunks_root"])
    n_run = 0
    n_fail = 0
    for row in todo_rows:
        if max_runs is not None and n_run >= max_runs:
            break
        pid = row["pair_id"]
        _chunk, ypath = yaml_idx[pid]
        outdir = base_dir / row["outdir"]
        outdir.mkdir(parents=True, exist_ok=True)
        ypath_in = ypath.resolve()
        print(f"[RUN] {split} {pid} -> {outdir}", flush=True)
        print(f"       yaml: {ypath_in}", flush=True)
        rc = run_cli(ypath_in, outdir, boltz_cfg, base_dir, quiet=quiet)
        n_run += 1
        if rc != 0:
            n_fail += 1
            fail_log = outdir / "failures.log"
            print(f"[FAIL] rc={rc} {ypath_in}", flush=True)
            if fail_log.is_file():
                print(f"       see {fail_log}", flush=True)
            continue
        if not embeddings_exist_for_yaml(ypath_in, outdir):
            n_fail += 1
            print(
                f"[FAIL] rc=0 but no embeddings*.npz under {outdir}/boltz_results_{pid}/predictions/",
                flush=True,
            )
            if not boltz_cfg.get("write_embeddings"):
                print("       Hint: pass --write_embeddings (required for this pipeline).", flush=True)

    report["executed"] = n_run
    report["failed"] = n_fail
    return report


def parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Manifest-aware Boltz runner (val/test/immrep_test).")
    p.add_argument("--base_dir", type=Path, default=BASE_DIR_DEFAULT)
    p.add_argument("--splits", nargs="+", default=["val", "test", "immrep_test"], choices=list(SPLIT_CONFIGS))
    p.add_argument("--plan-only", action="store_true", help="Only report todo/done; do not invoke boltz.")
    p.add_argument("--pair-id", type=str, default=None, help="Restrict to one pair_id (smoke test).")
    p.add_argument("--max-runs", type=int, default=None, help="Stop after N boltz invocations.")

    p.add_argument("--expected_env", type=str, default="boltz-env-torchfix")
    p.add_argument("--no_require_ld_preload", action="store_true")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "tpu"])
    p.add_argument("--model", type=str, default="boltz2", choices=["boltz1", "boltz2"])
    p.add_argument("--recycling_steps", type=int, default=3)
    p.add_argument("--sampling_steps", type=int, default=100)
    p.add_argument("--diffusion_samples", type=int, default=1)
    p.add_argument("--max_parallel_samples", type=int, default=5)
    p.add_argument("--max_msa_seqs", type=int, default=64)
    p.add_argument("--num_subsampled_msa", type=int, default=64)
    p.add_argument("--write_embeddings", action="store_true", default=False)
    p.add_argument("--override", action="store_true", default=False)
    p.add_argument("--no_kernels", action="store_true", default=False)
    p.add_argument("--progress_every", type=int, default=100)
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--report", type=Path, default=BASE_DIR_DEFAULT / "data/boltz_runs_v2_plan.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_dir = args.base_dir.resolve()

    if not args.plan_only:
        assert_correct_runtime(
            expected_env_substr=args.expected_env,
            require_ld_preload=(not args.no_require_ld_preload),
        )

    boltz_cfg = dict(
        devices=args.devices,
        accelerator=args.accelerator,
        model=args.model,
        recycling_steps=args.recycling_steps,
        sampling_steps=args.sampling_steps,
        diffusion_samples=args.diffusion_samples,
        max_parallel_samples=args.max_parallel_samples,
        max_msa_seqs=args.max_msa_seqs,
        num_subsampled_msa=args.num_subsampled_msa,
        write_embeddings=args.write_embeddings,
        override=args.override,
        no_kernels=args.no_kernels,
    )

    quiet = (not args.debug) and args.quiet
    full_report = {"boltz_cfg": boltz_cfg, "plan_only": args.plan_only, "splits": []}

    print("\n=== boltz_runs_v2 ===", flush=True)
    print("boltz CLI preview:", " ".join(build_boltz_args(Path("INPUT.yaml"), Path("OUT"), **boltz_cfg)[:16]), "...", flush=True)

    for split in args.splits:
        cfg = {
            "chunks_root": base_dir / SPLIT_CONFIGS[split]["chunks_root"].relative_to(BASE_DIR_DEFAULT),
            "out_root": base_dir / SPLIT_CONFIGS[split]["out_root"].relative_to(BASE_DIR_DEFAULT),
            "manifest": base_dir / SPLIT_CONFIGS[split]["manifest"].relative_to(BASE_DIR_DEFAULT),
        }
        if args.pair_id:
            allowed = load_manifest_pair_ids(cfg["manifest"])
            if args.pair_id not in allowed:
                print(f"[SKIP] {split}: --pair-id {args.pair_id} not in manifest", flush=True)
                continue
        rep = run_manifest_split(
            split,
            cfg,
            boltz_cfg,
            base_dir,
            plan_only=args.plan_only,
            pair_id=args.pair_id,
            max_runs=args.max_runs,
            progress_every=args.progress_every,
            quiet=quiet,
        )
        rep.pop("rows", None)  # keep JSON small
        full_report["splits"].append(rep)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(full_report, indent=2) + "\n")
    print(f"\nWrote report: {args.report}", flush=True)
    total_todo = sum(s.get("todo", 0) for s in full_report["splits"])
    total_fail = sum(s.get("failed", 0) for s in full_report["splits"])
    print(f"TOTAL todo across splits: {total_todo}", flush=True)
    if not args.plan_only:
        print(f"TOTAL failed this invocation: {total_fail}", flush=True)
        if total_fail:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
