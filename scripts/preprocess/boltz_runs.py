#!/usr/bin/env python3
"""
boltz_runs.py

Long-running Boltz runner with safe resume.

IMPORTANT USAGE:
1) Activate ONCE before running (do NOT use conda run):
   conda activate boltz-env-torchfix
   python /home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py

This ensures your conda activate.d CUDA fix is applied (LD_LIBRARY_PATH / LD_PRELOAD),
and avoids per-call activation overhead.

This script:
- Runs TRAIN chunks (data/train/_chunks/chunk_*/), VAL folder (data/val/), TEST chunks (data/test/_chunks/)
- Uses directory run first (fast) and falls back to per-YAML (robust)
- Skips completed outputs via a .DONE marker and by checking embeddings files
- Logs failures and continues
"""

from __future__ import annotations

import os
import sys
import time
import shlex
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional


# ----------------------------
# Defaults (tuned to your setup)
# ----------------------------

BASE_DIR_DEFAULT = Path("/home/natasha/multimodal_model")

# BASE_DIR/outputs is a symlink -> /data_ssd/boltz_outputs (per your setup)
RUN_ROOT_DEFAULT = BASE_DIR_DEFAULT / "outputs"

DONE_MARKER = ".DONE"
FAIL_LOG = "failed_yamls.txt"  # append-only, per outdir


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


# ----------------------------
# Safety checks: refuse to run in the wrong environment
# ----------------------------

def assert_correct_runtime(expected_env_substr: str, require_ld_preload: bool = True) -> None:
    """
    Fail fast if the user forgot to activate boltz-env-torchfix (or equivalent),
    because conda run will NOT reliably apply the activate.d CUDA fix you set up.
    """
    py = sys.executable
    boltz = shutil.which("boltz")

    if expected_env_substr and expected_env_substr not in py:
        raise RuntimeError(
            f"[FATAL] Wrong Python interpreter:\n  {py}\n\n"
            f"Expected to see '{expected_env_substr}' in the path.\n"
            "Activate the env ONCE then run the script:\n"
            f"  conda activate {expected_env_substr}\n"
            "  python /home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py\n"
        )

    if not boltz or (expected_env_substr and expected_env_substr not in boltz):
        raise RuntimeError(
            f"[FATAL] Wrong 'boltz' on PATH:\n  {boltz}\n\n"
            "Activate the correct env first so boltz resolves inside it."
        )

    # This is the key part of your recent fix: LD_PRELOAD must include cublas.
    if require_ld_preload:
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libcublas.so.12" not in ld_preload:
            raise RuntimeError(
                "[FATAL] LD_PRELOAD does not include libcublas.so.12.\n"
                "That strongly suggests the CUDA fix hook did not run.\n\n"
                "Do NOT run via `conda run`. Instead:\n"
                f"  conda activate {expected_env_substr}\n"
                "  (confirm LD_PRELOAD is set)\n"
                "  python /home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py\n"
            )


# ----------------------------
# File discovery + resume helpers
# ----------------------------

def list_chunk_dirs(chunks_root: Path) -> List[Path]:
    chunks_root = Path(chunks_root).resolve()
    if not chunks_root.exists():
        raise FileNotFoundError(f"Chunks root not found: {chunks_root}")
    chunk_dirs = sorted([p for p in chunks_root.iterdir() if p.is_dir() and p.name.startswith("chunk_")])
    if not chunk_dirs:
        raise ValueError(f"No chunk directories found in: {chunks_root}")
    return chunk_dirs


def list_yamls(dir_path: Path) -> List[Path]:
    d = Path(dir_path)
    return sorted(list(d.glob("*.yml")) + list(d.glob("*.yaml")))


def yaml_to_pair_dirname(yaml_path: Path) -> str:
    """
    Assumption matches your structure: pair_000.yaml -> pair_000
    """
    return yaml_path.stem


def embeddings_exist_for_yaml(yaml_path: Path, outdir: Path) -> bool:
    pair_dir = Path(outdir) / "predictions" / yaml_to_pair_dirname(yaml_path)
    # Your embeddings naming: embeddings_pair_*.npz
    return any(pair_dir.glob("embeddings_pair_*.npz"))


def all_embeddings_exist_for_dir(input_dir: Path, outdir: Path) -> bool:
    yamls = list_yamls(input_dir)
    if not yamls:
        return False
    return all(embeddings_exist_for_yaml(y, outdir) for y in yamls)


def mark_done(outdir: Path) -> None:
    (Path(outdir) / DONE_MARKER).write_text(f"done_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def append_fail(outdir: Path, yaml_path: Path, rc: int, note: str = "") -> None:
    p = Path(outdir) / FAIL_LOG
    with open(p, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\trc={rc}\t{yaml_path}\t{note}\n")


# ----------------------------
# Boltz invocation
# ----------------------------

def build_boltz_args(
    input_path: Path,
    outdir: Path,
    *,
    devices: int,
    accelerator: str,
    model: str,
    recycling_steps: int,
    sampling_steps: int,
    diffusion_samples: int,
    max_parallel_samples: int,
    max_msa_seqs: int,
    num_subsampled_msa: int,
    write_embeddings: bool,
    override: bool,
    no_kernels: bool,
) -> List[str]:
    args: List[str] = ["boltz", "predict", str(input_path), "--out_dir", str(outdir)]

    args += ["--accelerator", accelerator]
    args += ["--devices", str(devices)]
    args += ["--model", model]
    args += ["--recycling_steps", str(recycling_steps)]
    args += ["--sampling_steps", str(sampling_steps)]
    args += ["--diffusion_samples", str(diffusion_samples)]
    args += ["--max_parallel_samples", str(max_parallel_samples)]
    args += ["--max_msa_seqs", str(max_msa_seqs)]
    args += ["--num_subsampled_msa", str(num_subsampled_msa)]

    if write_embeddings:
        args += ["--write_embeddings"]
    if override:
        args += ["--override"]
    if no_kernels:
        args += ["--no_kernels"]

    return args


def run_cli(
    input_path: Path,
    outdir: Path,
    boltz_cfg: dict,
    base_dir: Path,
) -> int:
    input_path = Path(input_path).resolve()
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = build_boltz_args(input_path, outdir, **boltz_cfg)

    so_path = outdir / f"stdout_{stamp()}.log"
    se_path = outdir / f"stderr_{stamp()}.log"

    print("\nCMD:", " ".join(map(shlex.quote, cmd)))
    print("STDOUT:", so_path)
    print("STDERR:", se_path)

    with open(so_path, "w") as so, open(se_path, "w") as se:
        proc = subprocess.run(
            cmd,
            stdout=so,
            stderr=se,
            text=True,
            cwd=str(base_dir),
            env=os.environ.copy(),  # inherits activated env + LD_* CUDA fix
        )

    print("Return code:", proc.returncode)
    return proc.returncode


# ----------------------------
# Execution logic (dir run -> fallback per-YAML)
# ----------------------------

def run_dir_with_safe_resume(input_dir: Path, outdir: Path, label: str, boltz_cfg: dict, base_dir: Path) -> None:
    """
    Runs a directory (chunk or val folder) with safe resume:
      - skip if DONE exists
      - skip if embeddings already exist for all YAMLs
      - try a fast directory run
      - if that fails or incomplete, run per-YAML with per-YAML skip
      - continue past failures, record them, do not overwrite unless override enabled
    """
    input_dir = Path(input_dir)
    outdir = Path(outdir)

    if (outdir / DONE_MARKER).exists():
        print(f"[SKIP] {label}: {DONE_MARKER} exists -> {outdir}")
        return

    if all_embeddings_exist_for_dir(input_dir, outdir):
        print(f"[SKIP] {label}: embeddings already complete -> {outdir}")
        mark_done(outdir)
        return

    # 1) Try the fast path: run boltz on the directory
    print(f"\n=== {label} (dir run) ===")
    rc = run_cli(input_dir, outdir, boltz_cfg, base_dir)

    if rc == 0 and all_embeddings_exist_for_dir(input_dir, outdir):
        mark_done(outdir)
        return

    # 2) Fallback: per-YAML safe resume
    print(f"[FALLBACK] {label}: switching to per-YAML runs (safe resume).")
    yamls = list_yamls(input_dir)

    if not yamls:
        print(f"[WARN] {label}: no YAMLs found in {input_dir}")
        append_fail(outdir, input_dir, 998, note="No YAML files found in directory")
        return

    for y in yamls:
        if embeddings_exist_for_yaml(y, outdir):
            print(f"[SKIP-YAML] embeddings exist: {y.name}")
            continue

        rc_y = run_cli(y, outdir, boltz_cfg, base_dir)
        if rc_y != 0:
            print(f"[FAIL-YAML] {y.name} (rc={rc_y}) — logged, continuing.")
            append_fail(outdir, y, rc_y)
            continue

        if not embeddings_exist_for_yaml(y, outdir):
            print(f"[WARN] {y.name} returned 0 but embeddings not found — logged.")
            append_fail(outdir, y, 999, note="rc=0 but embeddings missing")

    if all_embeddings_exist_for_dir(input_dir, outdir):
        mark_done(outdir)
    else:
        print(f"[INFO] {label} not fully complete; see {outdir}/{FAIL_LOG} and rerun later.")


def run_chunked_dataset(chunks_root: Path, out_root: Path, label: str, boltz_cfg: dict, base_dir: Path) -> None:
    for chunk_dir in list_chunk_dirs(chunks_root):
        outdir = Path(out_root) / chunk_dir.name
        run_dir_with_safe_resume(chunk_dir, outdir, f"{label} {chunk_dir.name}", boltz_cfg, base_dir)


def run_val_folder(val_yaml_dir: Path, out_root: Path, boltz_cfg: dict, base_dir: Path) -> None:
    outdir = Path(out_root) / "val_full"
    run_dir_with_safe_resume(Path(val_yaml_dir), outdir, "VAL val_full", boltz_cfg, base_dir)


# ----------------------------
# Main
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Boltz over train/val/test with safe resume.")
    p.add_argument("--base_dir", type=Path, default=BASE_DIR_DEFAULT)

    # Env safety
    p.add_argument("--expected_env", type=str, default="boltz-env-torchfix",
                   help="Substring expected in sys.executable and boltz path. Set empty to disable.")
    p.add_argument("--no_require_ld_preload", action="store_true",
                   help="Disable LD_PRELOAD check (NOT recommended for your setup).")

    # Boltz options
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--accelerator", type=str, default="gpu", choices=["gpu", "cpu", "tpu"])
    p.add_argument("--model", type=str, default="boltz2", choices=["boltz1", "boltz2"])
    p.add_argument("--recycling_steps", type=int, default=1)
    p.add_argument("--sampling_steps", type=int, default=10)
    p.add_argument("--diffusion_samples", type=int, default=1)
    p.add_argument("--max_parallel_samples", type=int, default=1)
    p.add_argument("--max_msa_seqs", type=int, default=64)
    p.add_argument("--num_subsampled_msa", type=int, default=34)
    p.add_argument("--write_embeddings", action="store_true", default=True)
    p.add_argument("--override", action="store_true", default=False)
    p.add_argument("--no_kernels", action="store_true", default=False,
                   help="Force disable kernels. Leave OFF for your restored-quality runs.")

    # Output locations
    p.add_argument("--run_root", type=Path, default=RUN_ROOT_DEFAULT,
                   help="Root output dir (BASE_DIR/outputs; can be symlink).")

    # Which splits to run
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_val", action="store_true")
    p.add_argument("--skip_test", action="store_true")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Safety: ensure correct env is active (single activation mode)
    assert_correct_runtime(
        expected_env_substr=args.expected_env,
        require_ld_preload=(not args.no_require_ld_preload),
    )

    base_dir: Path = args.base_dir.resolve()

    # Prepare output structure
    run_root: Path = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    out_train = run_root / "train"
    out_val = run_root / "val"
    out_test = run_root / "test"
    for p in (out_train, out_val, out_test):
        p.mkdir(parents=True, exist_ok=True)

    # Input structure
    yaml_dir_train = base_dir / "data" / "train"
    yaml_dir_val = base_dir / "data" / "val"
    yaml_dir_test = base_dir / "data" / "test"

    train_chunks_root = yaml_dir_train / "_chunks"
    test_chunks_root = yaml_dir_test / "_chunks"

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

    print("\n==============================")
    print("Boltz runner starting")
    print("BASE_DIR:", base_dir)
    print("RUN_ROOT:", run_root)
    print("boltz:", shutil.which("boltz"))
    print("python:", sys.executable)
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("LD_LIBRARY_PATH (first 5):", ":".join(os.environ.get("LD_LIBRARY_PATH", "").split(":")[:5]))
    print("BOLTZ CONFIG:", boltz_cfg)
    print("==============================\n")

    # Execute
    try:
        if not args.skip_train:
            run_chunked_dataset(train_chunks_root, out_train, "TRAIN", boltz_cfg, base_dir)
        else:
            print("[SKIP] TRAIN")

        if not args.skip_val:
            run_val_folder(yaml_dir_val, out_val, boltz_cfg, base_dir)
        else:
            print("[SKIP] VAL")

        if not args.skip_test:
            run_chunked_dataset(test_chunks_root, out_test, "TEST", boltz_cfg, base_dir)
        else:
            print("[SKIP] TEST")

    except Exception as e:
        print("\n[FATAL] Unhandled exception:", repr(e), file=sys.stderr)
        return 1

    print("\nAll requested runs completed (or safely resumed where possible).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
