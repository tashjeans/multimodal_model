#!/usr/bin/env python3
"""Export IMMREP predictions from an already-trained plain ESM VICReg checkpoint.

This is a small helper for the case where the original run saved val/test
prediction CSVs but did not save IMMREP predictions. It does not retrain or
reselect the model. It loads the frozen checkpoint, evaluates IMMREP, and writes
prediction/per-peptide/summary files that can be used by downstream fusion
scripts.

Default paths match the user's current TULIP-style plain VICReg run.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import json
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader


def import_module_from_path(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find model script at {path}")
    spec = importlib.util.spec_from_file_location("plain_vicreg_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses expects the module to be present in sys.modules while class decorators execute.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def shapes_from_checkpoint(ckpt: dict) -> Tuple[int, int, int, int]:
    """Infer D, L_T, L_P, L_H from checkpoint state dicts."""
    tcr_sd = ckpt.get("tcr_state_dict") or ckpt.get("state", {}).get("tcr")
    pmhc_sd = ckpt.get("pmhc_state_dict") or ckpt.get("state", {}).get("pmhc")
    if tcr_sd is None or pmhc_sd is None:
        raise KeyError("Checkpoint must contain tcr_state_dict and pmhc_state_dict, or state['tcr']/state['pmhc']")

    D = int(tcr_sd["B_c"].shape[0])
    L_T = int(tcr_sd["A_c"].shape[0])

    # PMHCProjectionHead in the original script contains pep_encoder and hla_encoder.
    L_P = int(pmhc_sd["pep_encoder.A_c"].shape[0])
    L_H = int(pmhc_sd["hla_encoder.A_c"].shape[0])
    return D, L_T, L_P, L_H


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-script-path",
        default="/home/natasha/multimodal_model/scripts/train/models/train_plain_vicreg_tulip_simple_mse_score_mcclish_strict.py",
        help="Path to the original plain ESM VICReg training script, used for model/dataset/eval definitions.",
    )
    p.add_argument(
        "--checkpoint",
        default="/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple/tulip_decoys_plain_vicreg__seed31__lr0.0003__a25.0__b25.0__dlt1.0__best.pt",
    )
    p.add_argument(
        "--embed-root",
        default="/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped_tulip_decoys",
    )
    p.add_argument(
        "--immrep-csv",
        default="/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv",
    )
    p.add_argument(
        "--immrep-shard-dir",
        default="/home/natasha/multimodal_model/models/embeddings/immrep_test_set/test",
    )
    p.add_argument(
        "--out-dir",
        default="/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple",
    )
    p.add_argument("--run-tag", default="tulip_decoys_plain_vicreg")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--r-ph", type=float, default=0.7)
    p.add_argument("--partial-auc-max-fpr", type=float, default=0.1)
    args = p.parse_args()

    mod = import_module_from_path(Path(args.model_script_path))
    mod.set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    shapes = shapes_from_checkpoint(ckpt)
    D, L_T, L_P, L_H = shapes
    print("=" * 80, flush=True)
    print("Exporting IMMREP predictions from frozen ESM VICReg checkpoint", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Inferred shapes from checkpoint | D={D} | L_T={L_T} | L_P={L_P} | L_H={L_H}", flush=True)
    print(f"IMMREP CSV: {args.immrep_csv}", flush=True)
    print(f"IMMREP shards: {args.immrep_shard_dir}", flush=True)
    print("=" * 80, flush=True)

    # Create lightweight config object compatible with the original initialiser.
    cfg = mod.RunConfig(
        embed_root=args.embed_root,
        immrep_csv=args.immrep_csv,
        immrep_shard_dir=args.immrep_shard_dir,
        out_dir=args.out_dir,
        run_tag=args.run_tag,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        R_PH=args.r_ph,
    )

    # Prefer original loss params saved in checkpoint if available.
    lp = ckpt.get("loss_params")
    if lp is None:
        lp = mod.loss_params(cfg)
    lp["partial_auc_max_fpr"] = args.partial_auc_max_fpr

    tcr, pmhc = mod.initialise_models(cfg, shapes, device)

    tcr_sd = ckpt.get("tcr_state_dict") or ckpt.get("state", {}).get("tcr")
    pmhc_sd = ckpt.get("pmhc_state_dict") or ckpt.get("state", {}).get("pmhc")
    tcr.load_state_dict(tcr_sd)
    pmhc.load_state_dict(pmhc_sd)
    tcr.eval(); pmhc.eval()

    immrep_meta = mod.allowed_meta_from_csv(args.immrep_csv, "immrep_test", positives_only=False, complete_only=True)
    immrep_ds = mod.FilteredESMRowDataset(Path(args.immrep_shard_dir), immrep_meta, "immrep_test")
    immrep_loader = DataLoader(
        immrep_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mod.esm_row_collate,
    )
    pair_to_peptide = mod.load_pair_to_peptide(args.immrep_csv)

    result = mod.evaluate(
        immrep_loader,
        tcr,
        pmhc,
        device,
        lp,
        pair_to_peptide,
        split="immrep_test",
        R_PH=args.r_ph,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{args.run_tag}__seed{args.seed}__exported_immrep_from_checkpoint"
    pred_path = out_dir / f"{stem}__immrep_test_predictions.csv"
    model_pep_path = out_dir / f"{stem}__immrep_test_model_per_peptide.csv"
    raw_pep_path = out_dir / f"{stem}__immrep_test_raw_esm_per_peptide.csv"
    summary_path = out_dir / f"{stem}__immrep_test_summary.json"

    result["predictions"].to_csv(pred_path, index=False)
    result["model_peptide_table"].to_csv(model_pep_path, index=False)
    result["raw_peptide_table"].to_csv(raw_pep_path, index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "model_script_path": str(Path(args.model_script_path)),
        "immrep_csv": args.immrep_csv,
        "immrep_shard_dir": args.immrep_shard_dir,
        "metrics": result["metrics"],
        "paths": {
            "immrep_test_predictions": str(pred_path),
            "immrep_test_model_per_peptide": str(model_pep_path),
            "immrep_test_raw_esm_per_peptide": str(raw_pep_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print("=" * 80, flush=True)
    print("Done. Wrote:", flush=True)
    print(f"Predictions: {pred_path}", flush=True)
    print(f"Model per-peptide: {model_pep_path}", flush=True)
    print(f"Raw per-peptide: {raw_pep_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print("IMMREP metrics:", flush=True)
    print(json.dumps(result["metrics"], indent=2), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
