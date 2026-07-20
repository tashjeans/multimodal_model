#!/usr/bin/env python3
"""
Workshop-paper ESMC VICReg pipeline — raw/pretrained ESMC initialization.

Trains VICReg projection heads on frozen raw ESMC embedding shards.
Final evaluation reports:
  1) esm_vicreg: VICReg heads trained on raw ESMC embeddings
  2) pretrained_esmc_meanpool: frozen meanpool baseline on the same raw embeddings

Scoring convention:
  mse_distance = mean((TCR_representation - pMHC_representation)^2)
  score = -mse_distance
Higher score = more binder-like.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from esm_vicreg_common import (
    EPOCH_VAL_MODELS,
    RAW_ALL_EVAL_MODELS,
    ESMProjectionHead,
    PMHCProjectionHead,
    PairedESMRowDataset,
    evaluate,
    infer_shapes,
    load_meta,
    loss_params,
    make_loader,
    plain_vicreg_loss,
    prepare_dirs,
    raw_shard_split_dir,
    save_eval_outputs,
    set_seed,
)


REPO = Path(__file__).resolve().parents[3]


@dataclass
class RawRunConfig:
    run_tag: str = "esm_vicreg_raw_complete"
    checkpoint_root: str = str(REPO / "models/checkpoints/workshop/esm_vicreg_raw_complete")
    output_root: str = str(REPO / "models/outputs/workshop/esm_vicreg_raw_complete")
    figure_root: str = str(REPO / "models/figures/workshop/esm_vicreg_raw_complete")

    pretrained_embed_root: str = str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids")
    pretrained_immrep_shard_dir: str = str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids/immrep_test")

    train_csv: str = str(REPO / "data/train/train_multiview.csv")
    val_csv: str = str(REPO / "data/val/val_multiview.csv")
    test_csv: str = str(REPO / "data/test/test_multiview.csv")
    immrep_csv: str = str(REPO / "data/immrep_test/immrep_test_multiview.csv")

    seed: int = 31
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 30
    patience: int = 10
    min_epochs: int = 10

    rL: int = 8
    rD: int = 16
    d: int = 128
    R_PH: float = 0.7
    dropout: float = 0.1

    lr: float = 3e-4
    weight_decay: float = 1e-2

    alpha: float = 25.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    eps_var: float = 1e-4
    eps_pool: float = 1e-8
    partial_auc_max_fpr: float = 0.1

    save_latents: bool = True
    overwrite: bool = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=RawRunConfig.run_tag)
    parser.add_argument("--checkpoint-root", default=RawRunConfig.checkpoint_root)
    parser.add_argument("--output-root", default=RawRunConfig.output_root)
    parser.add_argument("--figure-root", default=RawRunConfig.figure_root)
    parser.add_argument("--pretrained-embed-root", default=RawRunConfig.pretrained_embed_root)
    parser.add_argument("--pretrained-immrep-shard-dir", default=RawRunConfig.pretrained_immrep_shard_dir)
    parser.add_argument("--train-csv", default=RawRunConfig.train_csv)
    parser.add_argument("--val-csv", default=RawRunConfig.val_csv)
    parser.add_argument("--test-csv", default=RawRunConfig.test_csv)
    parser.add_argument("--immrep-csv", default=RawRunConfig.immrep_csv)
    parser.add_argument("--seed", type=int, default=RawRunConfig.seed)
    parser.add_argument("--batch-size", type=int, default=RawRunConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=RawRunConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=RawRunConfig.epochs)
    parser.add_argument("--patience", type=int, default=RawRunConfig.patience)
    parser.add_argument("--min-epochs", type=int, default=RawRunConfig.min_epochs)
    parser.add_argument("--rL", type=int, default=RawRunConfig.rL)
    parser.add_argument("--rD", type=int, default=RawRunConfig.rD)
    parser.add_argument("--d", type=int, default=RawRunConfig.d)
    parser.add_argument("--R-PH", type=float, default=RawRunConfig.R_PH)
    parser.add_argument("--dropout", type=float, default=RawRunConfig.dropout)
    parser.add_argument("--lr", type=float, default=RawRunConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=RawRunConfig.weight_decay)
    parser.add_argument("--alpha", type=float, default=RawRunConfig.alpha)
    parser.add_argument("--beta", type=float, default=RawRunConfig.beta)
    parser.add_argument("--delta", type=float, default=RawRunConfig.delta)
    parser.add_argument("--gamma-var", type=float, default=RawRunConfig.gamma_var)
    parser.add_argument("--partial-auc-max-fpr", type=float, default=RawRunConfig.partial_auc_max_fpr)
    parser.add_argument("--save-latents", action="store_true", default=RawRunConfig.save_latents)
    parser.add_argument("--no-save-latents", action="store_false", dest="save_latents")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = RawRunConfig(
        run_tag=args.run_tag,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        figure_root=args.figure_root,
        pretrained_embed_root=args.pretrained_embed_root,
        pretrained_immrep_shard_dir=args.pretrained_immrep_shard_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        immrep_csv=args.immrep_csv,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        rL=args.rL,
        rD=args.rD,
        d=args.d,
        R_PH=args.R_PH,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        gamma_var=args.gamma_var,
        partial_auc_max_fpr=args.partial_auc_max_fpr,
        save_latents=args.save_latents,
        overwrite=args.overwrite,
    )

    set_seed(cfg.seed)
    checkpoint_dir, output_dir, figure_dir = prepare_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_root = Path(cfg.pretrained_embed_root)
    immrep_shard_dir = Path(cfg.pretrained_immrep_shard_dir)

    print("=" * 72, flush=True)
    print("Workshop ESMC VICReg run — raw ESMC initialization", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {cfg.seed}", flush=True)
    print(f"Raw ESMC shard root: {embed_root}", flush=True)
    print(f"Raw ESMC IMMREP shards: {immrep_shard_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"Figure dir: {figure_dir}", flush=True)
    print("Scoring: score = -MSE distance; no cosine metrics", flush=True)
    print("Training loader: raw ESMC shards, shuffle=True, batch_size=8", flush=True)
    print("Per-epoch val: esm_vicreg only", flush=True)
    print("Final eval: esm_vicreg + pretrained_esmc_meanpool", flush=True)
    print(f"Save latents: {cfg.save_latents}", flush=True)
    print("=" * 72, flush=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_meta, train_audit = load_meta(cfg.train_csv, "train", positives_only=True, complete_only=True)
    val_meta, val_audit = load_meta(cfg.val_csv, "val", positives_only=False, complete_only=True)
    test_meta, test_audit = load_meta(cfg.test_csv, "test", positives_only=False, complete_only=True)
    audits = [train_audit, val_audit, test_audit]
    immrep_meta = None
    if cfg.immrep_csv and str(cfg.immrep_csv).lower() not in ["none", ""]:
        immrep_meta, immrep_audit = load_meta(cfg.immrep_csv, "immrep_test", positives_only=False, complete_only=True)
        audits.append(immrep_audit)
    pd.DataFrame(audits).to_csv(output_dir / "split_filter_audit.csv", index=False)

    train_dir = raw_shard_split_dir(embed_root, immrep_shard_dir, "train")
    val_dir = raw_shard_split_dir(embed_root, immrep_shard_dir, "val")
    test_dir = raw_shard_split_dir(embed_root, immrep_shard_dir, "test")

    train_ds = PairedESMRowDataset(train_dir, None, train_meta, "train", include_pretrained=False, order_by_finetuned_shard=False)
    val_ds = PairedESMRowDataset(val_dir, None, val_meta, "val", include_pretrained=False, order_by_finetuned_shard=True)
    test_ds = PairedESMRowDataset(test_dir, None, test_meta, "test", include_pretrained=False, order_by_finetuned_shard=True)
    immrep_ds = None
    if immrep_meta is not None:
        imm_dir = raw_shard_split_dir(embed_root, immrep_shard_dir, "immrep_test")
        immrep_ds = PairedESMRowDataset(imm_dir, None, immrep_meta, "immrep_test", include_pretrained=False, order_by_finetuned_shard=True)

    print(
        f"Loaded shard rows: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}" +
        (f" | immrep_test={len(immrep_ds)}" if immrep_ds is not None else ""),
        flush=True,
    )

    D, L_T, L_P, L_H = infer_shapes([train_ds, val_ds, test_ds] + ([] if immrep_ds is None else [immrep_ds]))

    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    immrep_loader = None if immrep_ds is None else make_loader(immrep_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    tcr = ESMProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": tcr.parameters(), "lr": cfg.lr}, {"params": pmhc.parameters(), "lr": cfg.lr}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    lp = loss_params(cfg)

    best = {"metric": -np.inf, "epoch": None, "state": None, "bad_epochs": 0}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()
        running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
        n_steps = 0

        for batch in train_loader:
            zT = tcr(batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
            zPH = pmhc(
                batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
                batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
            )
            loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running["loss"] += parts["L_total"]
            for k in running:
                if k != "loss":
                    running[k] += parts[k]
            n_steps += 1
        scheduler.step()

        val_eval = evaluate(
            val_loader, tcr, pmhc, device, cfg, "val", save_latents=False,
            model_names=EPOCH_VAL_MODELS, pretrained_meanpool_from_ft=True,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, n_steps) for k, v in running.items()},
        }
        for model_name, metric_dict in val_eval["metrics"].items():
            row.update({f"val_{model_name}_{k}": v for k, v in metric_dict.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        current = val_eval["metrics"]["esm_vicreg"]["peptide_weighted_auroc"]
        improved = (not np.isnan(current)) and current > best["metric"] + 1e-4
        if improved:
            best = {
                "metric": current,
                "epoch": epoch,
                "state": {"tcr": copy.deepcopy(tcr.state_dict()), "pmhc": copy.deepcopy(pmhc.state_dict())},
                "bad_epochs": 0,
            }
        else:
            best["bad_epochs"] += 1

        print(
            f"Epoch {epoch}/{cfg.epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_inv={row['train_L_inv']:.4f} | "
            f"train_var={row['train_L_var']:.4f} | "
            f"train_cov={row['train_L_cov']:.4f} | "
            f"val_vicreg_global={val_eval['metrics']['esm_vicreg']['global_auroc']:.4f} | "
            f"val_vicreg_pep_weighted={current:.4f} | "
            f"best_epoch={best['epoch']} | bad_epochs={best['bad_epochs']}",
            flush=True,
        )

        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint selected. Check labels and validation metrics.")

    tcr.load_state_dict(best["state"]["tcr"])
    pmhc.load_state_dict(best["state"]["pmhc"])

    final_evals = {
        "val": evaluate(
            val_loader, tcr, pmhc, device, cfg, "val", cfg.save_latents,
            model_names=RAW_ALL_EVAL_MODELS, pretrained_meanpool_from_ft=True,
        ),
        "test": evaluate(
            test_loader, tcr, pmhc, device, cfg, "test", cfg.save_latents,
            model_names=RAW_ALL_EVAL_MODELS, pretrained_meanpool_from_ft=True,
        ),
    }
    if immrep_loader is not None:
        final_evals["immrep_test"] = evaluate(
            immrep_loader, tcr, pmhc, device, cfg, "immrep_test", cfg.save_latents,
            model_names=RAW_ALL_EVAL_MODELS, pretrained_meanpool_from_ft=True,
        )

    output_paths = {}
    for split, eval_obj in final_evals.items():
        output_paths.update(save_eval_outputs(eval_obj, output_dir, figure_dir, split, cfg.save_latents))

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "config": asdict(cfg),
            "shapes": {"D": D, "L_T": L_T, "L_P": L_P, "L_H": L_H},
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "best_val_esm_vicreg_peptide_weighted_auroc": best["metric"],
            "metrics": {split: obj["metrics"] for split, obj in final_evals.items()},
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(cfg),
        "model_family": "esm_raw",
        "seed": cfg.seed,
        "best_epoch": best["epoch"],
        "best_selection_metric": "val.esm_vicreg.peptide_weighted_auroc",
        "best_selection_value": best["metric"],
        "metrics": {split: obj["metrics"] for split, obj in final_evals.items()},
        "paths": {
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "history.csv"),
            "run_config": str(output_dir / "run_config.json"),
            "split_filter_audit": str(output_dir / "split_filter_audit.csv"),
            "output_dir": str(output_dir),
            "figure_dir": str(figure_dir),
            **output_paths,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72, flush=True)
    print("Done: raw ESMC workshop run", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
