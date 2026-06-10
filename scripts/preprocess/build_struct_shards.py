#!/usr/bin/env python3
"""Precompute Boltz interface structure shards (combined-TCR, 4 directional blocks).

Why this script exists
----------------------
Training previously opened each raw Boltz pair-embedding ``.npz`` inside the
dataset ``__getitem__`` on every epoch, sliced ``z`` into interface blocks, and
subsampled tokens on the fly. Each ``.npz`` carries a ``z`` array of shape
``(1, L, L, 128)`` (~180 MB for L~600), so re-reading and re-slicing it every
epoch dominated wall-clock time and the large per-interface token budget (512)
blew up GPU memory during validation.

This script does that slicing exactly once and writes compact, padded shards
that the training script can stream cheaply.

Representation (combined-TCR)
-----------------------------
Alpha and beta are treated as a single contiguous TCR block (chain order is
``tcra,tcrb,pep,hla``, so ``tcr = [0 : tcra_len + tcrb_len]``). We extract four
directional interface blocks from ``z`` and tag each token with a type id:

    type 0: z[tcr, pep]   -> tcr_pep
    type 1: z[pep, tcr]   -> pep_tcr   (reverse)
    type 2: z[tcr, hla]   -> tcr_hla
    type 3: z[hla, tcr]   -> hla_tcr   (reverse)

This controls the saved token count and avoids inflating ``N`` through separate
alpha/beta block caps.

Token budget / headroom
-----------------------
Each directional block is capped at ``--cap-per-block`` tokens (default 128) by
deterministic even-spacing subsampling, so an example holds at most
``4 * cap = N_max`` tokens (default 512). We deliberately bake in the *larger*
128 cap so the same shards serve both 64-token and 128-token experiments: the
training script subsamples each type-id group from 128 down to 64 at load time.

Caveat (documented on purpose): the train-time 128->64 step is a
subsample-of-a-subsample. It is deterministic but not byte-identical to
subsampling the full ``z`` block straight to 64. This is acceptable for the
current modelling objective.

Storage format
--------------
Each ``shard_XXXXX.pt`` is a dict of stacked, padded tensors:

    {
        "pair_id":         list[str]                 # length R
        "struct_tokens":   FloatTensor[R, N_max, 128] (stored as float16)
        "struct_type_ids": LongTensor[R, N_max]
        "struct_mask":     BoolTensor[R, N_max]      # True = real token
    }

``struct_tokens`` is stored as ``float16`` to cut storage/I-O pressure; the
training script casts to float32 (or uses AMP) after loading.

Alongside the shards we write ``struct_shard_index.json`` (``pair_id ->
{shard, row}``), ``build_report.json``, and ``missing_struct_rows.csv`` (rows
that were complete-chain but had no resolvable Boltz ``.npz``).

Typical launch
--------------
    cd /home/natasha/multimodal_model
    conda activate tcr-multimodal
    python scripts/preprocess/build_struct_shards.py --split all --num-workers 8
    python scripts/preprocess/build_struct_shards.py --split all --verify --verify-sample 50
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]

# Reuse the already-tested slicing/IO helpers from the training script so that
# the offline tokens are identical to what training used to compute on the fly.
sys.path.insert(0, str(REPO / "scripts" / "train" / "models"))
from train_vicreg_sequence_structure import (  # noqa: E402
    chain_slices,
    complete_complex_pair_ids,
    deterministic_subsample,
    first_z_array,
    make_boltz_path_map,
)


CHAIN_ORDER = "tcra,tcrb,pep,hla"

# type_id -> (name, source_chain, target_chain) using combined-TCR slices.
INTERFACE_DEFINITIONS: Dict[int, Dict[str, str]] = {
    0: {"name": "tcr_pep", "block": "z[tcr, pep]"},
    1: {"name": "pep_tcr", "block": "z[pep, tcr]"},
    2: {"name": "tcr_hla", "block": "z[tcr, hla]"},
    3: {"name": "hla_tcr", "block": "z[hla, tcr]"},
}

LENGTH_KEYS = ["tcra_len", "tcrb_len", "pep_len", "hla_len"]


# ============================================================
# Splits
# ============================================================


@dataclass
class SplitSpec:
    name: str
    csv: Path
    boltz_root: Path
    out_dir: Path


def default_splits() -> Dict[str, SplitSpec]:
    return {
        "train": SplitSpec(
            "train",
            REPO / "data/train/train_multiview.csv",
            REPO / "outputs/train",
            REPO / "outputs_data/train_struct_shards",
        ),
        "val": SplitSpec(
            "val",
            REPO / "data/val/val_multiview.csv",
            REPO / "outputs/val",
            REPO / "outputs_data/val_struct_shards",
        ),
        "test": SplitSpec(
            "test",
            REPO / "data/test/test_multiview.csv",
            REPO / "outputs/test",
            REPO / "outputs_data/test_struct_shards",
        ),
        "immrep_test": SplitSpec(
            "immrep_test",
            REPO / "data/immrep_test/immrep_test_multiview.csv",
            REPO / "outputs_data/immrep_test",
            REPO / "outputs_data/immrep_test_struct_shards",
        ),
    }


# ============================================================
# Combined-TCR interface extraction
# ============================================================


def combined_tcr_slices(row: pd.Series) -> Dict[str, slice]:
    """Return tcr (alpha+beta combined), pep and hla slices for a row."""
    sl = chain_slices(row, CHAIN_ORDER)  # tcra, tcrb, pep, hla
    tcr = slice(sl["tcra"].start, sl["tcrb"].stop)
    return {"tcr": tcr, "pep": sl["pep"], "hla": sl["hla"]}


def extract_combined_tcr_blocks(
    z: np.ndarray,
    row: pd.Series,
    cap_per_block: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slice z into the four directional interface blocks and cap each block.

    Returns (tokens [n, Dz], type_ids [n]) with n <= 4 * cap_per_block.
    """
    sl = combined_tcr_slices(row)
    blocks = [
        (0, sl["tcr"], sl["pep"]),
        (1, sl["pep"], sl["tcr"]),
        (2, sl["tcr"], sl["hla"]),
        (3, sl["hla"], sl["tcr"]),
    ]

    tokens: List[np.ndarray] = []
    type_ids: List[np.ndarray] = []
    for tid, a, b in blocks:
        block = z[a, b, :]
        if block.size == 0:
            continue
        flat = block.reshape(-1, block.shape[-1]).astype(np.float32, copy=False)
        flat = deterministic_subsample(flat, cap_per_block)
        tokens.append(flat)
        type_ids.append(np.full((flat.shape[0],), tid, dtype=np.int64))

    if not tokens:
        raise ValueError("No interface tokens were extracted from Boltz z")
    return np.concatenate(tokens, axis=0), np.concatenate(type_ids, axis=0)


def _pad_example(
    tokens: np.ndarray,
    type_ids: np.ndarray,
    n_max: int,
    dz: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(tokens.shape[0])
    if n > n_max:
        # Should not happen when n_max >= 4 * cap, but guard defensively.
        tokens = tokens[:n_max]
        type_ids = type_ids[:n_max]
        n = n_max
    st = np.zeros((n_max, dz), dtype=np.float16)
    sti = np.zeros((n_max,), dtype=np.int64)
    sm = np.zeros((n_max,), dtype=bool)
    st[:n] = tokens.astype(np.float16, copy=False)
    sti[:n] = type_ids
    sm[:n] = True
    return st, sti, sm


# Top-level worker so it is picklable by multiprocessing.
def _worker(task: Tuple[str, str, Dict[str, int], int, int, int]) -> Dict[str, Any]:
    """Extract one example. Returns a success or failure record (never raises).

    One corrupt .npz, a missing ``z`` array, an unexpected shape, or a chain-length
    mismatch must not crash the whole preprocessing run, so all errors are caught and
    reported per-row instead.
    """
    pair_id, npz_path, lengths, cap_per_block, n_max, dz = task
    try:
        row = pd.Series(lengths)
        z = first_z_array(Path(npz_path))

        # Hard chain-length validation BEFORE slicing. If z is not L x L for the
        # expected complex length, the chain order/metadata is wrong and the
        # combined-TCR blocks would silently slice the wrong interface tokens.
        expected_L = sum(int(lengths[k]) for k in LENGTH_KEYS)
        if z.shape[0] != expected_L or z.shape[1] != expected_L:
            raise ValueError(
                f"{pair_id}: z shape {z.shape[:2]} does not match expected complex length "
                f"{expected_L} from lengths={lengths}. This indicates a chain-order or metadata mismatch."
            )

        tokens, type_ids = extract_combined_tcr_blocks(z, row, cap_per_block)
        st, sti, sm = _pad_example(tokens, type_ids, n_max, dz)
        return {
            "ok": True,
            "pair_id": pair_id,
            "struct_tokens": st,
            "struct_type_ids": sti,
            "struct_mask": sm,
        }
    except Exception as e:  # noqa: BLE001 - we intentionally capture any per-row failure
        return {
            "ok": False,
            "pair_id": pair_id,
            "npz_path": str(npz_path),
            "error": repr(e),
            **{f"len_{k}": int(v) for k, v in lengths.items()},
        }


# ============================================================
# Build
# ============================================================


@dataclass
class BuildConfig:
    project: str = str(REPO)
    cap_per_block: int = 128
    n_max: int = 512
    dz: int = 128
    examples_per_shard: int = 512
    num_workers: int = 8
    overwrite: bool = False
    # Fatal only if the fraction of npz-matched rows that fail extraction exceeds this.
    max_failure_fraction: float = 0.05


def _resolve_present_missing(
    meta: pd.DataFrame, path_map: Dict[str, Path]
) -> Tuple[List[str], List[str]]:
    present, missing = [], []
    for pid in meta["pair_id"].astype(str).tolist():
        (present if pid in path_map else missing).append(pid)
    return present, missing


def _clean_split_dir(out_dir: Path) -> None:
    """Remove stale shards/index/reports before an --overwrite rebuild."""
    for p in out_dir.glob("shard_*.pt"):
        p.unlink()
    for name in [
        "struct_shard_index.json",
        "build_report.json",
        "missing_struct_rows.csv",
        "failed_struct_rows.csv",
    ]:
        p = out_dir / name
        if p.exists():
            p.unlink()


def build_split(spec: SplitSpec, cfg: BuildConfig) -> Dict[str, Any]:
    import multiprocessing as mp

    print("=" * 80, flush=True)
    print(f"[build] split={spec.name} | csv={spec.csv}", flush=True)

    raw = pd.read_csv(spec.csv)
    rows_seen = int(len(raw))

    # positives_only=False: precompute structure for every complete row (pos + neg).
    # Label-based filtering stays in the training Dataset.
    meta, _allowed = complete_complex_pair_ids(raw, positives_only=False, source_name=spec.name)
    complete_rows = int(len(meta))

    path_map = make_boltz_path_map(
        meta,
        boltz_root=str(spec.boltz_root),
        required=False,
        source_name=spec.name,
        project_root=cfg.project,
    )
    present, missing = _resolve_present_missing(meta, path_map)
    print(
        f"[build] {spec.name}: complete_rows={complete_rows} | npz_matched={len(present)} | npz_missing={len(missing)}",
        flush=True,
    )

    spec.out_dir.mkdir(parents=True, exist_ok=True)

    # When rebuilding, clear stale shards/index/reports so the directory only ever
    # contains outputs from the current build.
    if cfg.overwrite:
        _clean_split_dir(spec.out_dir)

    # Persist the coverage gap; do not fail if Boltz only covered a subset.
    missing_csv = spec.out_dir / "missing_struct_rows.csv"
    if missing:
        meta[meta["pair_id"].astype(str).isin(set(missing))].to_csv(missing_csv, index=False)
    else:
        # Write an empty file with headers for consistency.
        meta.iloc[0:0].to_csv(missing_csv, index=False)

    meta_by_pid = {str(r["pair_id"]): r for _, r in meta.iterrows()}

    # Deterministic order by pair_id keeps shards/index reproducible.
    tasks: List[Tuple[str, str, Dict[str, int], int, int, int]] = []
    for pid in sorted(present):
        r = meta_by_pid[pid]
        lengths = {k: int(r[k]) for k in LENGTH_KEYS}
        tasks.append((pid, str(path_map[pid]), lengths, cfg.cap_per_block, cfg.n_max, cfg.dz))

    index: Dict[str, Dict[str, Any]] = {}
    shards_written = 0
    examples_written = 0
    failed_rows: List[Dict[str, Any]] = []

    buf_pid: List[str] = []
    buf_tok: List[np.ndarray] = []
    buf_ty: List[np.ndarray] = []
    buf_mask: List[np.ndarray] = []

    def flush_shard() -> None:
        nonlocal shards_written, examples_written
        if not buf_pid:
            return
        shard_name = f"shard_{shards_written:05d}.pt"
        shard = {
            "pair_id": list(buf_pid),
            "struct_tokens": torch.from_numpy(np.stack(buf_tok, axis=0)),  # float16
            "struct_type_ids": torch.from_numpy(np.stack(buf_ty, axis=0)),  # int64
            "struct_mask": torch.from_numpy(np.stack(buf_mask, axis=0)),  # bool
        }
        torch.save(shard, spec.out_dir / shard_name)
        for row_idx, pid in enumerate(buf_pid):
            index[pid] = {"shard": shard_name, "row": row_idx}
        examples_written += len(buf_pid)
        shards_written += 1
        print(f"[build] {spec.name}: wrote {shard_name} ({len(buf_pid)} examples)", flush=True)
        buf_pid.clear()
        buf_tok.clear()
        buf_ty.clear()
        buf_mask.clear()

    def consume(result: Dict[str, Any]) -> None:
        if not result.get("ok", False):
            failed_rows.append({k: v for k, v in result.items() if k != "ok"})
            return
        buf_pid.append(result["pair_id"])
        buf_tok.append(result["struct_tokens"])
        buf_ty.append(result["struct_type_ids"])
        buf_mask.append(result["struct_mask"])
        if len(buf_pid) >= cfg.examples_per_shard:
            flush_shard()

    if tasks:
        if cfg.num_workers and cfg.num_workers > 1:
            # fork (Linux default) avoids re-importing the heavy training module per
            # worker. Workers do CPU-only numpy work and never touch CUDA, so fork is safe.
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=cfg.num_workers) as pool:
                for i, result in enumerate(pool.imap(_worker, tasks, chunksize=8), start=1):
                    consume(result)
                    if i % 200 == 0:
                        print(f"[build] {spec.name}: processed {i}/{len(tasks)}", flush=True)
        else:
            for i, task in enumerate(tasks, start=1):
                consume(_worker(task))
                if i % 200 == 0:
                    print(f"[build] {spec.name}: processed {i}/{len(tasks)}", flush=True)
        flush_shard()

    # Persist rows whose Boltz path resolved but extraction failed (corrupt npz,
    # missing z, bad shape, chain-length mismatch). Distinct from missing_struct_rows.
    failed_csv = spec.out_dir / "failed_struct_rows.csv"
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(failed_csv, index=False)
    else:
        pd.DataFrame(columns=["pair_id", "npz_path", "error"]).to_csv(failed_csv, index=False)

    # struct_shard_index.json: pair_id -> {shard, row} plus useful metadata.
    index_payload = {
        "split": spec.name,
        "n_max": cfg.n_max,
        "cap_per_block": cfg.cap_per_block,
        "dz": cfg.dz,
        "chain_order": CHAIN_ORDER,
        "interface_definitions": INTERFACE_DEFINITIONS,
        "index": index,
    }
    with open(spec.out_dir / "struct_shard_index.json", "w") as f:
        json.dump(index_payload, f)

    report: Dict[str, Any] = {
        "split": spec.name,
        "csv": str(spec.csv),
        "boltz_root": str(spec.boltz_root),
        "out_dir": str(spec.out_dir),
        "rows_seen": rows_seen,
        "complete_chain_rows": complete_rows,
        "npz_matched": len(present),
        "npz_missing": len(missing),
        "examples_written": examples_written,
        "shards_written": shards_written,
        "failed_extraction": len(failed_rows),
        "failed_struct_rows_csv": str(failed_csv),
        "index_size": len(index),
        "coverage_written_over_complete": examples_written / max(complete_rows, 1),
        "coverage_written_over_npz_matched": examples_written / max(len(present), 1),
        "cap_per_block": cfg.cap_per_block,
        "n_max": cfg.n_max,
        "dtype": "float16",
        "storage_dtype": "float16",
        "expected_total_token_cap": 4 * cfg.cap_per_block,
        "combined_tcr": True,
        "directional_blocks": ["tcr_pep", "pep_tcr", "tcr_hla", "hla_tcr"],
        "examples_per_shard": cfg.examples_per_shard,
        "chain_order": CHAIN_ORDER,
        "interface_definitions": INTERFACE_DEFINITIONS,
        "missing_struct_rows_csv": str(missing_csv),
        "max_failure_fraction": cfg.max_failure_fraction,
    }
    with open(spec.out_dir / "build_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[build] {spec.name}: {json.dumps(report)}", flush=True)

    # Fatal only if nothing was written, or failures exceed the configured threshold.
    if present and examples_written == 0:
        raise RuntimeError(
            f"{spec.name}: 0 examples written despite {len(present)} npz-matched rows "
            f"({len(failed_rows)} failed). See {failed_csv}."
        )
    if present:
        fail_frac = len(failed_rows) / max(len(present), 1)
        if fail_frac > cfg.max_failure_fraction:
            raise RuntimeError(
                f"{spec.name}: extraction failure fraction {fail_frac:.3f} exceeds "
                f"max_failure_fraction={cfg.max_failure_fraction}. "
                f"{len(failed_rows)}/{len(present)} rows failed. See {failed_csv}."
            )
    return report


# ============================================================
# Verify
# ============================================================


def verify_split(spec: SplitSpec, cfg: BuildConfig, n_sample: int, seed: int = 0) -> Dict[str, Any]:
    print("=" * 80, flush=True)
    print(f"[verify] split={spec.name}", flush=True)

    index_path = spec.out_dir / "struct_shard_index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"{spec.name}: no struct_shard_index.json in {spec.out_dir}; build first")
    payload = json.loads(index_path.read_text())
    index: Dict[str, Dict[str, Any]] = payload["index"]
    cap = int(payload["cap_per_block"])
    n_max = int(payload["n_max"])
    dz = int(payload.get("dz", 128))

    # Cross-check the build report against the index, and confirm every referenced
    # shard file actually exists on disk.
    report_path = spec.out_dir / "build_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report_examples = int(report.get("examples_written", -1))
    examples_index_match = report_examples == len(index)
    referenced_shards = sorted({rec["shard"] for rec in index.values()})
    missing_shard_files = [s for s in referenced_shards if not (spec.out_dir / s).exists()]
    print(
        f"[verify] {spec.name}: report.examples_written={report_examples} | index_size={len(index)} "
        f"| examples==index={examples_index_match} | referenced_shards={len(referenced_shards)} "
        f"| missing_shard_files={len(missing_shard_files)}",
        flush=True,
    )

    # Recompute the present-set the same way the build did, to check coverage.
    raw = pd.read_csv(spec.csv)
    meta, _ = complete_complex_pair_ids(raw, positives_only=False, source_name=spec.name)
    path_map = make_boltz_path_map(
        meta, boltz_root=str(spec.boltz_root), required=False, source_name=spec.name, project_root=cfg.project
    )
    present, _missing = _resolve_present_missing(meta, path_map)
    present_set = set(present)
    index_set = set(index.keys())

    # Rows that resolved a Boltz path but failed extraction are legitimately absent
    # from the index, so exclude them from the expected-coverage set.
    failed_csv = spec.out_dir / "failed_struct_rows.csv"
    failed_set: set[str] = set()
    if failed_csv.exists():
        failed_df = pd.read_csv(failed_csv)
        if "pair_id" in failed_df.columns:
            failed_set = set(failed_df["pair_id"].astype(str).tolist())
    expected_set = present_set - failed_set

    missing_from_index = sorted(expected_set - index_set)
    extra_in_index = sorted(index_set - present_set)
    coverage_ok = (not missing_from_index) and (not extra_in_index)
    print(
        f"[verify] {spec.name}: index_covers_present={coverage_ok} "
        f"| present={len(present_set)} | indexed={len(index_set)} "
        f"| missing_from_index={len(missing_from_index)} | extra_in_index={len(extra_in_index)}",
        flush=True,
    )

    meta_by_pid = {str(r["pair_id"]): r for _, r in meta.iterrows()}
    rng = random.Random(seed)
    sample_pids = rng.sample(sorted(index_set), k=min(n_sample, len(index_set))) if index_set else []

    shard_cache: Dict[str, Dict[str, Any]] = {}
    checked = 0
    mismatches: List[str] = []
    shape_errors: List[str] = []

    for pid in sample_pids:
        rec = index[pid]
        shard_name = rec["shard"]
        row_idx = int(rec["row"])
        if shard_name not in shard_cache:
            shard_cache[shard_name] = torch.load(spec.out_dir / shard_name, map_location="cpu")
        shard = shard_cache[shard_name]

        assert shard["pair_id"][row_idx] == pid, f"{spec.name}: pair_id mismatch at {shard_name}[{row_idx}]"

        saved_tok = shard["struct_tokens"][row_idx].numpy()  # float16 [n_max, dz]
        saved_ty = shard["struct_type_ids"][row_idx].numpy()
        saved_mask = shard["struct_mask"][row_idx].numpy().astype(bool)

        # Stored shapes must be exactly the padded (n_max, dz) layout.
        if not (saved_tok.shape == (n_max, dz) and saved_ty.shape == (n_max,) and saved_mask.shape == (n_max,)):
            shape_errors.append(pid)
            print(
                f"[verify] {spec.name}: SHAPE ERROR pid={pid} | tok={saved_tok.shape} "
                f"ty={saved_ty.shape} mask={saved_mask.shape} expected=({n_max},{dz})",
                flush=True,
            )

        z = first_z_array(path_map[pid])
        rc_tok, rc_ty = extract_combined_tcr_blocks(z, meta_by_pid[pid], cap)
        exp_tok, exp_ty, exp_mask = _pad_example(rc_tok, rc_ty, n_max, saved_tok.shape[1])

        ok_mask = np.array_equal(saved_mask, exp_mask)
        ok_ty = np.array_equal(saved_ty, exp_ty)
        ok_tok = np.array_equal(saved_tok, exp_tok)
        if not (ok_mask and ok_ty and ok_tok):
            mismatches.append(pid)
            print(
                f"[verify] {spec.name}: MISMATCH pid={pid} | mask_ok={ok_mask} type_ok={ok_ty} tok_ok={ok_tok}",
                flush=True,
            )
        checked += 1

    result = {
        "split": spec.name,
        "coverage_ok": bool(coverage_ok),
        "examples_index_match": bool(examples_index_match),
        "report_examples_written": report_examples,
        "missing_shard_files": missing_shard_files[:20],
        "n_missing_shard_files": len(missing_shard_files),
        "present": len(present_set),
        "indexed": len(index_set),
        "failed": len(failed_set),
        "missing_from_index": missing_from_index[:20],
        "n_missing_from_index": len(missing_from_index),
        "extra_in_index": extra_in_index[:20],
        "n_extra_in_index": len(extra_in_index),
        "sampled": checked,
        "mismatches": mismatches[:20],
        "n_mismatches": len(mismatches),
        "shape_errors": shape_errors[:20],
        "n_shape_errors": len(shape_errors),
        "ok": bool(
            coverage_ok
            and examples_index_match
            and not missing_shard_files
            and not mismatches
            and not shape_errors
        ),
    }
    print(f"[verify] {spec.name}: {json.dumps(result)}", flush=True)
    return result


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build/verify Boltz interface structure shards")
    p.add_argument("--split", choices=["train", "val", "test", "immrep_test", "all"], default="all")
    p.add_argument("--cap-per-block", type=int, default=128)
    p.add_argument("--n-max", type=int, default=512)
    p.add_argument("--dz", type=int, default=128)
    p.add_argument("--examples-per-shard", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-failure-fraction", type=float, default=0.05,
                   help="Fatal if fraction of npz-matched rows that fail extraction exceeds this")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--verify", action="store_true", help="Run verification instead of building")
    p.add_argument("--verify-sample", type=int, default=50)
    p.add_argument("--verify-seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BuildConfig(
        cap_per_block=args.cap_per_block,
        n_max=args.n_max,
        dz=args.dz,
        examples_per_shard=args.examples_per_shard,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        max_failure_fraction=args.max_failure_fraction,
    )
    if cfg.n_max < 4 * cfg.cap_per_block:
        raise ValueError(f"n_max ({cfg.n_max}) must be >= 4 * cap_per_block ({4 * cfg.cap_per_block})")

    splits = default_splits()
    selected = list(splits.values()) if args.split == "all" else [splits[args.split]]

    if args.verify:
        results = [verify_split(s, cfg, args.verify_sample, args.verify_seed) for s in selected]
        all_ok = all(r["ok"] for r in results)
        print(f"[verify] ALL_OK={all_ok}", flush=True)
        if not all_ok:
            sys.exit(1)
        return

    reports = []
    for spec in selected:
        index_exists = (spec.out_dir / "struct_shard_index.json").exists()
        if index_exists and not cfg.overwrite:
            print(f"[build] {spec.name}: already built at {spec.out_dir} (use --overwrite to rebuild). Skipping.", flush=True)
            reports.append(json.loads((spec.out_dir / "build_report.json").read_text()))
            continue
        reports.append(build_split(spec, cfg))

    print(f"[build] DONE: {json.dumps([r['split'] for r in reports])}", flush=True)


if __name__ == "__main__":
    main()
