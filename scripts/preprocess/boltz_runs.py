#!/usr/bin/env python3
"""
boltz_runs.py

Long-running Boltz runner with safe resume.

Run like:
  conda activate boltz-env-torchfix
  python /home/natasha/multimodal_model/scripts/preprocess/boltz_runs.py
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
from typing import List


BASE_DIR_DEFAULT = Path("/home/natasha/multimodal_model")
RUN_ROOT_DEFAULT = BASE_DIR_DEFAULT / "outputs"  # symlink OK

DONE_MARKER = ".DONE"
FAIL_LOG = "failed_yamls.txt"      # tab-separated, append-only
FAILURES_LOG = "failures.log"      # captured stdout/stderr for failures


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def assert_correct_runtime(expected_env_substr: str, require_ld_preload: bool = True) -> None:
    py = sys.executable
    boltz = shutil.which("boltz")

    if expected_env_substr and expected_env_substr not in py:
        raise RuntimeError(
            f"[FATAL] Wrong Python interpreter:\n  {py}\n"
            f"Expected '{expected_env_substr}' in path. Activate then run."
        )

    if not boltz or (expected_env_substr and expected_env_substr not in boltz):
        raise RuntimeError(
            f"[FATAL] Wrong 'boltz' on PATH:\n  {boltz}\n"
            "Activate the correct env first."
        )

    if require_ld_preload:
        ld_preload = os.environ.get("LD_PRELOAD", "")
        if "libcublas.so.12" not in ld_preload:
            raise RuntimeError(
                "[FATAL] LD_PRELOAD missing libcublas.so.12. "
                "Your activate.d CUDA hook likely didn’t run (don’t use conda run)."
            )


def list_chunk_dirs(chunks_root: Path) -> List[Path]:
    chunks_root = chunks_root.resolve()
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
    return yaml_path.stem  # pair_000.yaml -> pair_000


def embeddings_exist_for_yaml(yaml_path: Path, outdir: Path) -> bool:
    """
    Robust skip check:
    Accept ANY embeddings*.npz inside:
      outdir/boltz_results_*/predictions/<pair>/embeddings*.npz
    """
    pair = yaml_to_pair_dirname(yaml_path)
    patterns = [
        f"boltz_results_*/predictions/{pair}/embeddings*.npz",
        f"predictions/{pair}/embeddings*.npz",  # legacy fallback
    ]
    for pat in patterns:
        if any(outdir.glob(pat)):
            return True
    return False


def all_embeddings_exist_for_dir(input_dir: Path, outdir: Path) -> bool:
    yamls = list_yamls(input_dir)
    if not yamls:
        return False
    return all(embeddings_exist_for_yaml(y, outdir) for y in yamls)


def mark_done(outdir: Path) -> None:
    (outdir / DONE_MARKER).write_text(f"done_at={time.strftime('%Y-%m-%d %H:%M:%S')}\n")


def append_fail(outdir: Path, yaml_path: Path, rc: int, note: str = "") -> None:
    p = outdir / FAIL_LOG
    with open(p, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\trc={rc}\t{yaml_path}\t{note}\n")


def append_failure_log(outdir: Path, cmd: List[str], input_path: Path, combined_output: str) -> None:
    p = outdir / FAILURES_LOG
    with open(p, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"input={input_path}\n")
        f.write(f"cmd={' '.join(map(shlex.quote, cmd))}\n\n")
        f.write((combined_output or "")[-20000:] + "\n")  # keep tail


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
    *,
    quiet: bool = True,
) -> int:
    input_path = input_path.resolve()
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = build_boltz_args(input_path, outdir, **boltz_cfg)

    if not quiet:
        print("\nCMD:", " ".join(map(shlex.quote, cmd)))
        proc = subprocess.run(cmd, cwd=str(base_dir), env=os.environ.copy())
        return proc.returncode

    # Quiet path: run with /dev/null to avoid buffering.
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=str(base_dir),
        env=os.environ.copy(),
    )
    if proc.returncode == 0:
        return 0

    # On failure, rerun once capturing output for debugging.
    proc2 = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(base_dir),
        env=os.environ.copy(),
    )
    append_failure_log(outdir, cmd, input_path, proc2.stdout or "")
    return proc2.returncode


def run_dir_with_safe_resume(
    input_dir: Path,
    outdir: Path,
    label: str,
    boltz_cfg: dict,
    base_dir: Path,
    *,
    progress_every: int,
    quiet: bool,
) -> None:
    input_dir = Path(input_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if (outdir / DONE_MARKER).exists():
        # Re-scan in case new YAMLs were added after completion
        if all_embeddings_exist_for_dir(input_dir, outdir):
            print(f"[SKIP] {label}: {DONE_MARKER} exists and embeddings complete -> {outdir}")
            return
        else:
            print(f"[RESUME] {label}: {DONE_MARKER} exists but new/missing YAMLs detected -> {outdir}")
            # continue into loop

    yamls = list_yamls(input_dir)
    if not yamls:
        print(f"[WARN] {label}: no YAMLs found in {input_dir}")
        append_fail(outdir, input_dir, 998, note="No YAML files found in directory")
        return

    if all_embeddings_exist_for_dir(input_dir, outdir):
        print(f"[SKIP] {label}: embeddings already complete -> {outdir}")
        mark_done(outdir)
        return

    print(f"\n=== {label} (per-YAML resume) ===")
    n_total = len(yamls)
    n_done = 0
    n_run = 0
    n_fail = 0

    for i, y in enumerate(yamls, start=1):
        if embeddings_exist_for_yaml(y, outdir):
            n_done += 1
            continue

        n_run += 1
        if (n_run == 1) or (n_run % progress_every == 0):
            print(f"[PROGRESS] {label}: scanned={i}/{n_total} done={n_done} run={n_run} fail={n_fail}")

        rc = run_cli(y, outdir, boltz_cfg, base_dir, quiet=quiet)
        if rc != 0:
            n_fail += 1
            append_fail(outdir, y, rc)
            continue

        if not embeddings_exist_for_yaml(y, outdir):
            n_fail += 1
            append_fail(outdir, y, 999, note="rc=0 but embeddings missing")

    if all_embeddings_exist_for_dir(input_dir, outdir):
        mark_done(outdir)
        print(f"[DONE] {label}: marked {DONE_MARKER}")
    else:
        print(f"[INCOMPLETE] {label}: missing outputs remain; see {outdir}/{FAIL_LOG} and {outdir}/{FAILURES_LOG}")


def run_chunked_dataset(chunks_root: Path, out_root: Path, label: str, boltz_cfg: dict, base_dir: Path, *, progress_every: int, quiet: bool) -> None:
    for chunk_dir in list_chunk_dirs(chunks_root):
        outdir = Path(out_root) / chunk_dir.name
        run_dir_with_safe_resume(chunk_dir, outdir, f"{label} {chunk_dir.name}", boltz_cfg, base_dir, progress_every=progress_every, quiet=quiet)


# def run_val_folder(val_yaml_dir: Path, out_root: Path, boltz_cfg: dict, base_dir: Path, *, progress_every: int, quiet: bool) -> None:
#     outdir = Path(out_root) / "val_full"
#     run_dir_with_safe_resume(Path(val_yaml_dir), outdir, "VAL val_full", boltz_cfg, base_dir, progress_every=progress_every, quiet=quiet)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Boltz over train/val/test with safe resume.")
    p.add_argument("--base_dir", type=Path, default=BASE_DIR_DEFAULT)

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

    p.add_argument("--run_root", type=Path, default=RUN_ROOT_DEFAULT)

    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_val", action="store_true")
    p.add_argument("--skip_test", action="store_true")

    p.add_argument("--progress_every", type=int, default=1000)
    p.add_argument("--quiet", action="store_true", default=True)
    p.add_argument("--debug", action="store_true", help="If set, run non-quiet (streams boltz output).")

    return p.parse_args()


def main() -> int:
    args = parse_args()

    assert_correct_runtime(
        expected_env_substr=args.expected_env,
        require_ld_preload=(not args.no_require_ld_preload),
    )

    base_dir = args.base_dir.resolve()
    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    out_train = run_root / "train"
    out_val = run_root / "val"
    out_test = run_root / "test"
    for p in (out_train, out_val, out_test):
        p.mkdir(parents=True, exist_ok=True)

    yaml_dir_train = base_dir / "data" / "train"
    yaml_dir_val = base_dir / "data" / "val"
    yaml_dir_test = base_dir / "data" / "test"

    train_chunks_root = yaml_dir_train / "_chunks"
    test_chunks_root = yaml_dir_test / "_chunks"
    val_chunks_root = yaml_dir_val / "_chunks"

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

    print("\n==============================")
    print("Boltz runner starting")
    print("BASE_DIR:", base_dir)
    print("RUN_ROOT:", run_root)
    print("boltz:", shutil.which("boltz"))
    print("python:", sys.executable)
    print("LD_PRELOAD:", os.environ.get("LD_PRELOAD", ""))
    print("BOLTZ CONFIG:", boltz_cfg)
    print("quiet:", quiet, "progress_every:", args.progress_every)
    print("==============================\n")

    try:
        if not args.skip_train:
            run_chunked_dataset(train_chunks_root, out_train, "TRAIN", boltz_cfg, base_dir, progress_every=args.progress_every, quiet=quiet)
        else:
            print("[SKIP] TRAIN")

        if not args.skip_val:
        #     run_val_folder(yaml_dir_val, out_val, boltz_cfg, base_dir, progress_every=args.progress_every, quiet=quiet)
            run_chunked_dataset(val_chunks_root, out_val, "VAL", boltz_cfg, base_dir, progress_every=args.progress_every, quiet=quiet)
        else:
            print("[SKIP] VAL")


        if not args.skip_test:
            run_chunked_dataset(test_chunks_root, out_test, "TEST", boltz_cfg, base_dir, progress_every=args.progress_every, quiet=quiet)
        else:
            print("[SKIP] TEST")

    except Exception as e:
        print("\n[FATAL] Unhandled exception:", repr(e), file=sys.stderr)
        return 1

    print("\nAll requested runs completed (or safely resumed where possible).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
