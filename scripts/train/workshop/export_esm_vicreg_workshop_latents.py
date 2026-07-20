#!/usr/bin/env python3
"""Export latents from an existing workshop ESM VICReg checkpoint without retraining.

Loads best.pt, runs final evaluation with save_latents=True, and writes
{split}_latents.npz into the existing seed output directory. Does not
overwrite predictions, history, or checkpoints.

Example
-------
    python scripts/train/workshop/export_esm_vicreg_workshop_latents.py --all-seeds
    python scripts/train/workshop/export_esm_vicreg_workshop_latents.py --seed 31
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_esm_vicreg_workshop as workshop  # noqa: E402


def shapes_from_checkpoint(ckpt: dict) -> Tuple[int, int, int, int]:
    if "shapes" in ckpt:
        s = ckpt["shapes"]
        return int(s["D"]), int(s["L_T"]), int(s["L_P"]), int(s["L_H"])
    tcr_sd = ckpt["tcr_state_dict"]
    pmhc_sd = ckpt["pmhc_state_dict"]
    D = int(tcr_sd["B_c"].shape[0])
    L_T = int(tcr_sd["A_c"].shape[0])
    L_P = int(pmhc_sd["pep_encoder.A_c"].shape[0])
    L_H = int(pmhc_sd["hla_encoder.A_c"].shape[0])
    return D, L_T, L_P, L_H


def cfg_from_checkpoint(ckpt: dict, seed: int, args) -> workshop.RunConfig:
    fields = asdict(workshop.RunConfig())
    fields.update(ckpt.get("config", {}))
    fields["seed"] = seed
    if args.checkpoint_root:
        fields["checkpoint_root"] = args.checkpoint_root
    if args.output_root:
        fields["output_root"] = args.output_root
    if args.finetuned_embed_root:
        fields["finetuned_embed_root"] = args.finetuned_embed_root
    if args.pretrained_embed_root:
        fields["pretrained_embed_root"] = args.pretrained_embed_root
    if args.finetuned_immrep_shard_dir:
        fields["finetuned_immrep_shard_dir"] = args.finetuned_immrep_shard_dir
    if args.pretrained_immrep_shard_dir:
        fields["pretrained_immrep_shard_dir"] = args.pretrained_immrep_shard_dir
    fields["batch_size"] = args.batch_size
    fields["num_workers"] = args.num_workers
    fields["save_latents"] = True
    return workshop.RunConfig(**fields)


def save_latents_npz(eval_obj: Dict, output_dir: Path, split: str) -> Path:
    if eval_obj["latents"] is None:
        raise RuntimeError(f"{split}: evaluate returned no latents")
    latent_path = output_dir / f"{split}_latents.npz"
    np.savez_compressed(latent_path, **eval_obj["latents"])
    return latent_path


def patch_summary(output_dir: Path, latent_paths: Dict[str, str]) -> None:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        print(f"Warning: no summary.json at {summary_path}; skipping summary patch", flush=True)
        return
    with open(summary_path) as f:
        summary = json.load(f)
    summary.setdefault("paths", {}).update(latent_paths)
    if "config" in summary:
        summary["config"]["save_latents"] = True
    summary["latents_exported"] = True
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def export_seed(seed: int, args) -> Dict[str, str]:
    checkpoint_path = Path(args.checkpoint_root) / f"seed_{seed}" / "best.pt"
    output_dir = Path(args.output_root) / f"seed_{seed}"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = cfg_from_checkpoint(ckpt, seed, args)
    D, L_T, L_P, L_H = shapes_from_checkpoint(ckpt)

    print("=" * 72, flush=True)
    print(f"Exporting workshop latents | seed={seed}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Shapes: D={D}, L_T={L_T}, L_P={L_P}, L_H={L_H}", flush=True)
    print(f"Splits: {args.splits}", flush=True)
    print("=" * 72, flush=True)

    tcr = workshop.ESMProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = workshop.PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
    tcr.load_state_dict(ckpt["tcr_state_dict"])
    pmhc.load_state_dict(ckpt["pmhc_state_dict"])
    tcr.eval()
    pmhc.eval()

    loaders = {}
    if "val" in args.splits:
        val_meta, _ = workshop.load_meta(cfg.val_csv, "val", positives_only=False, complete_only=True)
        val_ft, val_pre = workshop.split_dirs(cfg, "val")
        val_ds = workshop.PairedESMRowDataset(val_ft, val_pre, val_meta, "val", include_pretrained=True)
        loaders["val"] = workshop.make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    if "test" in args.splits:
        test_meta, _ = workshop.load_meta(cfg.test_csv, "test", positives_only=False, complete_only=True)
        test_ft, test_pre = workshop.split_dirs(cfg, "test")
        test_ds = workshop.PairedESMRowDataset(test_ft, test_pre, test_meta, "test", include_pretrained=True)
        loaders["test"] = workshop.make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    if "immrep_test" in args.splits:
        if not cfg.immrep_csv or str(cfg.immrep_csv).lower() in ["none", ""]:
            raise ValueError("immrep_test requested but immrep_csv is empty")
        immrep_meta, _ = workshop.load_meta(cfg.immrep_csv, "immrep_test", positives_only=False, complete_only=True)
        imm_ft, imm_pre = workshop.split_dirs(cfg, "immrep_test")
        immrep_ds = workshop.PairedESMRowDataset(imm_ft, imm_pre, immrep_meta, "immrep_test", include_pretrained=True)
        loaders["immrep_test"] = workshop.make_loader(immrep_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    latent_paths: Dict[str, str] = {}
    for split, loader in loaders.items():
        print(f"Evaluating {split} with save_latents=True ...", flush=True)
        eval_obj = workshop.evaluate(
            loader,
            tcr,
            pmhc,
            device,
            cfg,
            split,
            save_latents=True,
            model_names=workshop.ALL_EVAL_MODELS,
        )
        latent_path = save_latents_npz(eval_obj, output_dir, split)
        latent_paths[f"{split}_latents"] = str(latent_path)
        print(f"Wrote {latent_path}", flush=True)
        print(f"  keys: {sorted(eval_obj['latents'].keys())}", flush=True)
        print(f"  n_rows: {len(eval_obj['latents']['pair_id'])}", flush=True)

    patch_summary(output_dir, latent_paths)
    print(f"Updated summary: {output_dir / 'summary.json'}", flush=True)
    return latent_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--all-seeds", action="store_true", help="Export seeds 31, 37, and 43.")
    parser.add_argument("--checkpoint-root", default=workshop.RunConfig.checkpoint_root)
    parser.add_argument("--output-root", default=workshop.RunConfig.output_root)
    parser.add_argument("--finetuned-embed-root", default=None)
    parser.add_argument("--pretrained-embed-root", default=None)
    parser.add_argument("--finetuned-immrep-shard-dir", default=None)
    parser.add_argument("--pretrained-immrep-shard-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["val", "test", "immrep_test"])
    args = parser.parse_args()

    seeds = [31, 37, 43] if args.all_seeds else [args.seed]
    for seed in seeds:
        export_seed(seed, args)
    print("=" * 72, flush=True)
    print(f"Done. Exported latents for seeds: {seeds}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
