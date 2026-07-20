#!/usr/bin/env python3
"""
Trace how VICReg distance magnitudes evolve over early training.

Runs short (default 10-epoch) diagnostic trainings for:
  1) onehot_vicreg
  2) finetuned_esmc_vicreg

At epoch 0 (random init) and after each epoch, evaluate on val and log:
  - matched-pair MSE on unnormalized vectors (baseline / pre-expander / post-expander)
  - TCR-TCR Euclidean within- vs between-peptide distances (same stages)
  - scale diagnostics only: mean L2 norms and per-dim std of zT / zPH

Distances are NEVER L2-normalized; they match workshop scoring:
  mse_distance = mean((zT - zPH)^2)

Outputs:
  models/outputs/workshop/distance_trace/
  models/figures/workshop/distance_trace/
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


oh = _load_module("train_onehot_vicreg_workshop", SCRIPT_DIR / "train_onehot_vicreg_workshop.py")
esm = _load_module("esm_vicreg_common", SCRIPT_DIR / "esm_vicreg_common.py")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TraceConfig:
    output_root: str = "/home/natasha/multimodal_model/models/outputs/workshop/distance_trace"
    figure_root: str = "/home/natasha/multimodal_model/models/figures/workshop/distance_trace"
    seed: int = 31
    epochs: int = 10
    models: Tuple[str, ...] = ("onehot_vicreg", "finetuned_esmc_vicreg")

    # Shared data
    train_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"

    # One-hot hyperparams (match workshop defaults)
    onehot_batch_size: int = 64
    onehot_num_workers: int = 0
    onehot_lr: float = 3e-4
    onehot_weight_decay: float = 1e-2
    onehot_dropout: float = 0.1
    onehot_rL: int = 8
    onehot_rD: int = 16
    onehot_d: int = 128
    onehot_R_PH: float = 0.7
    onehot_alpha: float = 25.0
    onehot_beta: float = 25.0
    onehot_delta: float = 1.0
    onehot_gamma_var: float = 1.0
    onehot_eps_var: float = 1e-4
    onehot_eps_pool: float = 1e-8
    missing_chain_policy: str = "complete_only"
    max_tcr_len: int = 0
    max_pep_len: int = 0
    max_hla_len: int = 0

    # ESM hyperparams (match workshop defaults)
    finetuned_embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids"
    esm_batch_size: int = 8
    esm_num_workers: int = 0
    esm_lr: float = 3e-4
    esm_weight_decay: float = 1e-2
    esm_dropout: float = 0.1
    esm_rL: int = 8
    esm_rD: int = 16
    esm_d: int = 128
    esm_R_PH: float = 0.7
    esm_alpha: float = 25.0
    esm_beta: float = 25.0
    esm_delta: float = 1.0
    esm_gamma_var: float = 1.0
    esm_eps_var: float = 1e-4
    esm_eps_pool: float = 1e-8

    # Crossreactivity subsample settings (match analysis defaults roughly)
    min_group_size: int = 5
    max_tcrs_per_peptide: int = 25
    max_between_pairs: int = 20000


# ---------------------------------------------------------------------------
# Projection helpers (pre-expander on unnormalized vectors)
# ---------------------------------------------------------------------------

def lowrank_pre_expander(module: nn.Module, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Same as LowRank/ESMProjectionHead.forward up to (but not including) expander."""
    B = emb.shape[0]
    L_true = mask.sum(dim=1)
    z_list = []
    for b in range(B):
        Lb = int(L_true[b].item())
        if Lb == 0:
            z_list.append(torch.zeros(module.d, device=emb.device, dtype=emb.dtype))
            continue
        Xb = emb[b, :Lb, :] * mask[b, :Lb].unsqueeze(-1).float()
        Yb = Xb @ module.B_c
        Ub = module.A_c[:Lb, :].T @ Yb
        z_list.append(Ub.reshape(-1) @ module.H_c)
    return torch.stack(z_list, dim=0)


def encode_tcr(tcr: nn.Module, emb: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pre = lowrank_pre_expander(tcr, emb, mask)
    post = tcr.expander(pre)
    return pre, post


def encode_pmhc(
    pmhc: nn.Module,
    emb_P: torch.Tensor,
    mask_P: torch.Tensor,
    emb_H: torch.Tensor,
    mask_H: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pre_P = lowrank_pre_expander(pmhc.pep_encoder, emb_P, mask_P)
    pre_H = lowrank_pre_expander(pmhc.hla_encoder, emb_H, mask_H)
    pre = torch.cat([pre_P, pre_H], dim=-1)
    post = torch.cat(
        [pmhc.pep_encoder.expander(pre_P), pmhc.hla_encoder.expander(pre_H)],
        dim=-1,
    )
    return pre, post


def mse_pair(zT: torch.Tensor, zPH: torch.Tensor) -> torch.Tensor:
    return (zT - zPH).pow(2).mean(dim=-1)


def vector_norm(z: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(z, ord=2, dim=-1)


# ---------------------------------------------------------------------------
# TCR-TCR distance summaries (unnormalized Euclidean)
# ---------------------------------------------------------------------------

def balanced_positive_indices(peptides: np.ndarray, labels: np.ndarray, min_group: int, max_per: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idxs: List[int] = []
    pos = np.where(labels.astype(int) == 1)[0]
    pep_pos = peptides[pos]
    for pep in np.unique(pep_pos):
        local = pos[pep_pos == pep]
        if len(local) < min_group:
            continue
        if len(local) > max_per:
            local = rng.choice(local, size=max_per, replace=False)
        idxs.extend(local.tolist())
    return np.array(sorted(idxs), dtype=int)


def within_between_euclidean(Z: np.ndarray, peptides: np.ndarray, n_between: int, seed: int) -> Tuple[float, float, int, int]:
    rng = np.random.default_rng(seed)
    within: List[float] = []
    peptide_to_idx = {p: np.where(peptides == p)[0] for p in np.unique(peptides)}
    for idx in peptide_to_idx.values():
        if len(idx) < 2:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                within.append(float(np.linalg.norm(Z[idx[a]] - Z[idx[b]])))
    within_arr = np.asarray(within, dtype=float)
    if len(within_arr) == 0:
        return float("nan"), float("nan"), 0, 0

    n = len(peptides)
    between: List[float] = []
    target = min(n_between, max(len(within_arr), 1))
    attempts = 0
    max_attempts = target * 50
    while len(between) < target and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        attempts += 1
        if i == j or peptides[i] == peptides[j]:
            continue
        between.append(float(np.linalg.norm(Z[i] - Z[j])))
    between_arr = np.asarray(between, dtype=float)
    return (
        float(np.mean(within_arr)),
        float(np.mean(between_arr)) if len(between_arr) else float("nan"),
        int(len(within_arr)),
        int(len(between_arr)),
    )


def summarize_stage(
    name: str,
    zT: np.ndarray,
    zPH: np.ndarray,
    labels: np.ndarray,
    peptides: np.ndarray,
    cfg: TraceConfig,
    seed: int,
) -> Dict:
    mse = np.mean((zT - zPH) ** 2, axis=1)
    pos = labels.astype(int) == 1
    neg = ~pos
    row = {
        "stage": name,
        "mse_all_mean": float(np.mean(mse)),
        "mse_pos_mean": float(np.mean(mse[pos])) if pos.any() else float("nan"),
        "mse_neg_mean": float(np.mean(mse[neg])) if neg.any() else float("nan"),
        "mse_pos_median": float(np.median(mse[pos])) if pos.any() else float("nan"),
        "mse_neg_median": float(np.median(mse[neg])) if neg.any() else float("nan"),
        # Scale diagnostics only (not used as distance metrics)
        "zT_norm_mean": float(np.mean(np.linalg.norm(zT, axis=1))),
        "zPH_norm_mean": float(np.mean(np.linalg.norm(zPH, axis=1))),
        "zT_std": float(np.std(zT)),
        "zPH_std": float(np.std(zPH)),
        "zT_dim": int(zT.shape[1]),
        "zPH_dim": int(zPH.shape[1]),
    }
    bal = balanced_positive_indices(peptides, labels, cfg.min_group_size, cfg.max_tcrs_per_peptide, seed)
    if len(bal) >= 3:
        Zb = zT[bal]
        pb = peptides[bal]
        w, b, nw, nb = within_between_euclidean(Zb, pb, cfg.max_between_pairs, seed)
        row.update({
            "tcr_within_mean": w,
            "tcr_between_mean": b,
            "n_within_pairs": nw,
            "n_between_pairs": nb,
            "n_balanced_positives": int(len(bal)),
        })
    else:
        row.update({
            "tcr_within_mean": float("nan"),
            "tcr_between_mean": float("nan"),
            "n_within_pairs": 0,
            "n_between_pairs": 0,
            "n_balanced_positives": int(len(bal)),
        })
    return row


# ---------------------------------------------------------------------------
# Evaluation collectors
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_onehot_traces(
    val_loader: DataLoader,
    tcr: nn.Module,
    pmhc: nn.Module,
    device: torch.device,
    cfg: TraceConfig,
) -> List[Dict]:
    tcr.eval()
    pmhc.eval()
    stores = {
        "baseline_T": [], "baseline_PH": [],
        "pre_T": [], "pre_PH": [],
        "post_T": [], "post_PH": [],
        "labels": [], "peptides": [],
    }
    for batch in val_loader:
        # Baseline: composition / meanpool of one-hot (unnormalized)
        _, _, base_T, base_PH = oh.onehot_composition_score(batch, device, cfg.onehot_R_PH, cfg.onehot_eps_pool)
        pre_T, post_T = encode_tcr(tcr, batch["emb_T"].to(device), batch["mask_T"].to(device))
        pre_PH, post_PH = encode_pmhc(
            pmhc,
            batch["emb_P"].to(device), batch["mask_P"].to(device),
            batch["emb_H"].to(device), batch["mask_H"].to(device),
        )
        stores["baseline_T"].append(base_T.detach().cpu().numpy())
        stores["baseline_PH"].append(base_PH.detach().cpu().numpy())
        stores["pre_T"].append(pre_T.detach().cpu().numpy())
        stores["pre_PH"].append(pre_PH.detach().cpu().numpy())
        stores["post_T"].append(post_T.detach().cpu().numpy())
        stores["post_PH"].append(post_PH.detach().cpu().numpy())
        stores["labels"].append(batch["binding_flag"].numpy())
        stores["peptides"].append(np.asarray(batch["peptide"], dtype=str))

    labels = np.concatenate(stores["labels"])
    peptides = np.concatenate(stores["peptides"])
    rows = []
    for stage, t_key, ph_key in [
        ("baseline_input", "baseline_T", "baseline_PH"),
        ("pre_expander", "pre_T", "pre_PH"),
        ("post_expander", "post_T", "post_PH"),
    ]:
        zT = np.concatenate(stores[t_key])
        zPH = np.concatenate(stores[ph_key])
        rows.append(summarize_stage(stage, zT, zPH, labels, peptides, cfg, cfg.seed))
    return rows


@torch.no_grad()
def collect_esm_traces(
    val_loader: DataLoader,
    tcr: nn.Module,
    pmhc: nn.Module,
    device: torch.device,
    cfg: TraceConfig,
) -> List[Dict]:
    tcr.eval()
    pmhc.eval()
    stores = {
        "baseline_T": [], "baseline_PH": [],
        "pre_T": [], "pre_PH": [],
        "post_T": [], "post_PH": [],
        "labels": [], "peptides": [],
    }
    for batch in val_loader:
        _, _, base_T, base_PH = esm.meanpool_score(
            batch["ft_emb_T"], batch["ft_mask_T"],
            batch["ft_emb_P"], batch["ft_mask_P"],
            batch["ft_emb_H"], batch["ft_mask_H"],
            device, cfg.esm_R_PH, cfg.esm_eps_pool,
        )
        pre_T, post_T = encode_tcr(tcr, batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
        pre_PH, post_PH = encode_pmhc(
            pmhc,
            batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
            batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
        )
        stores["baseline_T"].append(base_T.detach().cpu().numpy())
        stores["baseline_PH"].append(base_PH.detach().cpu().numpy())
        stores["pre_T"].append(pre_T.detach().cpu().numpy())
        stores["pre_PH"].append(pre_PH.detach().cpu().numpy())
        stores["post_T"].append(post_T.detach().cpu().numpy())
        stores["post_PH"].append(post_PH.detach().cpu().numpy())
        stores["labels"].append(batch["binding_flag"].numpy())
        stores["peptides"].append(np.asarray(batch["peptide"], dtype=str))

    labels = np.concatenate(stores["labels"])
    peptides = np.concatenate(stores["peptides"])
    rows = []
    for stage, t_key, ph_key in [
        ("baseline_input", "baseline_T", "baseline_PH"),
        ("pre_expander", "pre_T", "pre_PH"),
        ("post_expander", "post_T", "post_PH"),
    ]:
        zT = np.concatenate(stores[t_key])
        zPH = np.concatenate(stores[ph_key])
        rows.append(summarize_stage(stage, zT, zPH, labels, peptides, cfg, cfg.seed))
    return rows


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_onehot(cfg: TraceConfig, device: torch.device, out_dir: Path) -> pd.DataFrame:
    print("=" * 72, flush=True)
    print("Tracing onehot_vicreg distance magnitude", flush=True)
    print("=" * 72, flush=True)
    oh.set_seed(cfg.seed)

    train_meta, _ = oh.load_meta(cfg.train_csv, "train", positives_only=True, missing_chain_policy=cfg.missing_chain_policy)
    val_meta, _ = oh.load_meta(cfg.val_csv, "val", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
    L_T, L_P, L_H = oh.compute_max_lengths([train_meta, val_meta], cfg.max_tcr_len, cfg.max_pep_len, cfg.max_hla_len)
    print(f"Max lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}", flush=True)

    train_loader = oh.make_loader(
        oh.OneHotFullTCRDataset(train_meta, L_T, L_P, L_H),
        cfg.onehot_batch_size, True, cfg.onehot_num_workers, cfg.seed,
    )
    val_loader = oh.make_loader(
        oh.OneHotFullTCRDataset(val_meta, L_T, L_P, L_H),
        cfg.onehot_batch_size, False, cfg.onehot_num_workers, cfg.seed,
    )

    tcr = oh.LowRankProjectionHead(oh.VOCAB_SIZE, cfg.onehot_rL, cfg.onehot_rD, cfg.onehot_d, L_T, cfg.onehot_dropout).to(device)
    pmhc = oh.PMHCProjectionHead(oh.VOCAB_SIZE, cfg.onehot_rL, cfg.onehot_rD, cfg.onehot_d, L_P, L_H, cfg.onehot_R_PH, cfg.onehot_dropout).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": tcr.parameters(), "lr": cfg.onehot_lr}, {"params": pmhc.parameters(), "lr": cfg.onehot_lr}],
        weight_decay=cfg.onehot_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    lp = {
        "alpha": cfg.onehot_alpha, "beta": cfg.onehot_beta, "delta": cfg.onehot_delta,
        "gamma_var": cfg.onehot_gamma_var, "eps_var": cfg.onehot_eps_var,
    }

    rows: List[Dict] = []
    # Epoch 0: random init
    for stage_row in collect_onehot_traces(val_loader, tcr, pmhc, device, cfg):
        rows.append({"model_name": "onehot_vicreg", "epoch": 0, **stage_row})
    print(
        f"[onehot] epoch 0 | post MSE pos={rows[-1]['mse_pos_mean']:.4f} "
        f"neg={rows[-1]['mse_neg_mean']:.4f} | TCR within={rows[-1]['tcr_within_mean']:.4f}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            zT = tcr(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc(
                batch["emb_P"].to(device), batch["mask_P"].to(device),
                batch["emb_H"].to(device), batch["mask_H"].to(device),
            )
            loss, parts = oh.plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += parts["L_total"]
            n_steps += 1
        scheduler.step()

        for stage_row in collect_onehot_traces(val_loader, tcr, pmhc, device, cfg):
            rows.append({"model_name": "onehot_vicreg", "epoch": epoch, **stage_row})
        post = rows[-1]
        print(
            f"[onehot] epoch {epoch}/{cfg.epochs} | train_loss={running_loss / max(1, n_steps):.4f} | "
            f"post MSE pos={post['mse_pos_mean']:.4f} neg={post['mse_neg_mean']:.4f} | "
            f"TCR within={post['tcr_within_mean']:.4f} between={post['tcr_between_mean']:.4f} | "
            f"||zT||={post['zT_norm_mean']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "distance_trace.csv", index=False)
    return df


def train_finetuned_esm(cfg: TraceConfig, device: torch.device, out_dir: Path) -> pd.DataFrame:
    print("=" * 72, flush=True)
    print("Tracing finetuned_esmc_vicreg distance magnitude", flush=True)
    print("=" * 72, flush=True)
    esm.set_seed(cfg.seed)

    train_meta, _ = esm.load_meta(cfg.train_csv, "train", positives_only=True, complete_only=True)
    val_meta, _ = esm.load_meta(cfg.val_csv, "val", positives_only=False, complete_only=True)
    train_ft = Path(cfg.finetuned_embed_root) / "train"
    val_ft = Path(cfg.finetuned_embed_root) / "val"

    train_ds = esm.PairedESMRowDataset(train_ft, None, train_meta, "train", include_pretrained=False)
    val_ds = esm.PairedESMRowDataset(val_ft, None, val_meta, "val", include_pretrained=False)
    print(f"Loaded paired rows: train={len(train_ds)} | val={len(val_ds)}", flush=True)

    D, L_T, L_P, L_H = esm.infer_shapes([train_ds, val_ds])
    train_loader = esm.make_loader(train_ds, cfg.esm_batch_size, True, cfg.esm_num_workers, cfg.seed)
    val_loader = esm.make_loader(val_ds, cfg.esm_batch_size, False, cfg.esm_num_workers, cfg.seed)

    tcr = esm.ESMProjectionHead(D, cfg.esm_rL, cfg.esm_rD, cfg.esm_d, L_T, cfg.esm_dropout).to(device)
    pmhc = esm.PMHCProjectionHead(D, cfg.esm_rL, cfg.esm_rD, cfg.esm_d, L_P, L_H, cfg.esm_R_PH, cfg.esm_dropout).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": tcr.parameters(), "lr": cfg.esm_lr}, {"params": pmhc.parameters(), "lr": cfg.esm_lr}],
        weight_decay=cfg.esm_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    lp = {
        "alpha": cfg.esm_alpha, "beta": cfg.esm_beta, "delta": cfg.esm_delta,
        "gamma_var": cfg.esm_gamma_var, "eps_var": cfg.esm_eps_var,
    }

    rows: List[Dict] = []
    for stage_row in collect_esm_traces(val_loader, tcr, pmhc, device, cfg):
        rows.append({"model_name": "finetuned_esmc_vicreg", "epoch": 0, **stage_row})
    print(
        f"[esm] epoch 0 | post MSE pos={rows[-1]['mse_pos_mean']:.4f} "
        f"neg={rows[-1]['mse_neg_mean']:.4f} | TCR within={rows[-1]['tcr_within_mean']:.4f}",
        flush=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()
        running_loss = 0.0
        n_steps = 0
        for batch in train_loader:
            zT = tcr(batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
            zPH = pmhc(
                batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
                batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
            )
            loss, parts = esm.plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += parts["L_total"]
            n_steps += 1
        scheduler.step()

        for stage_row in collect_esm_traces(val_loader, tcr, pmhc, device, cfg):
            rows.append({"model_name": "finetuned_esmc_vicreg", "epoch": epoch, **stage_row})
        post = rows[-1]
        print(
            f"[esm] epoch {epoch}/{cfg.epochs} | train_loss={running_loss / max(1, n_steps):.4f} | "
            f"post MSE pos={post['mse_pos_mean']:.4f} neg={post['mse_neg_mean']:.4f} | "
            f"TCR within={post['tcr_within_mean']:.4f} between={post['tcr_between_mean']:.4f} | "
            f"||zT||={post['zT_norm_mean']:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "distance_trace.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

STAGE_STYLE = {
    "baseline_input": {"color": "#4daf4a", "ls": "--", "marker": "o"},
    "pre_expander": {"color": "#377eb8", "ls": "-.", "marker": "s"},
    "post_expander": {"color": "#e41a1c", "ls": "-", "marker": "^"},
}


def plot_model_traces(df: pd.DataFrame, model_name: str, fig_dir: Path) -> None:
    sub = df[df["model_name"] == model_name].copy()
    if sub.empty:
        return

    # Matched-pair MSE
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, col, title in [
        (axes[0], "mse_pos_mean", "Positive matched-pair MSE"),
        (axes[1], "mse_neg_mean", "Negative matched-pair MSE"),
    ]:
        for stage, style in STAGE_STYLE.items():
            g = sub[sub["stage"] == stage].sort_values("epoch")
            ax.plot(g["epoch"], g[col], label=stage, **style)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE (unnormalized)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name}: matched-pair MSE over early training")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{model_name}_mse_over_epochs.png", dpi=200)
    plt.close(fig)

    # TCR-TCR Euclidean
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, col, title in [
        (axes[0], "tcr_within_mean", "Within-peptide TCR Euclidean"),
        (axes[1], "tcr_between_mean", "Between-peptide TCR Euclidean"),
    ]:
        for stage, style in STAGE_STYLE.items():
            g = sub[sub["stage"] == stage].sort_values("epoch")
            ax.plot(g["epoch"], g[col], label=stage, **style)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Euclidean distance (unnormalized)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name}: TCR-TCR distances over early training")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{model_name}_tcr_euclidean_over_epochs.png", dpi=200)
    plt.close(fig)

    # Scale diagnostics only
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, col, title in [
        (axes[0], "zT_norm_mean", "Mean ||zT|| (scale diagnostic)"),
        (axes[1], "zT_std", "zT global std (scale diagnostic)"),
    ]:
        for stage, style in STAGE_STYLE.items():
            g = sub[sub["stage"] == stage].sort_values("epoch")
            ax.plot(g["epoch"], g[col], label=stage, **style)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(col)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"{model_name}: scale diagnostics (not distance metrics)")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{model_name}_scale_over_epochs.png", dpi=200)
    plt.close(fig)


def plot_combined(df: pd.DataFrame, fig_dir: Path) -> None:
    post = df[df["stage"] == "post_expander"].copy()
    if post.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for model, color in [("onehot_vicreg", "#e41a1c"), ("finetuned_esmc_vicreg", "#377eb8")]:
        g = post[post["model_name"] == model].sort_values("epoch")
        if g.empty:
            continue
        axes[0].plot(g["epoch"], g["mse_pos_mean"], marker="o", color=color, label=f"{model} pos")
        axes[0].plot(g["epoch"], g["mse_neg_mean"], marker="s", linestyle="--", color=color, label=f"{model} neg")
        axes[1].plot(g["epoch"], g["tcr_within_mean"], marker="o", color=color, label=f"{model} within")
        axes[1].plot(g["epoch"], g["tcr_between_mean"], marker="s", linestyle="--", color=color, label=f"{model} between")
    axes[0].set_title("Post-expander matched-pair MSE")
    axes[0].set_ylabel("MSE (unnormalized)")
    axes[1].set_title("Post-expander TCR-TCR Euclidean")
    axes[1].set_ylabel("Euclidean distance")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Combined post-expander distance traces")
    fig.tight_layout()
    fig.savefig(fig_dir / "combined_post_expander_over_epochs.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trace VICReg distance magnitude over early training.")
    p.add_argument("--output-root", default=TraceConfig.output_root)
    p.add_argument("--figure-root", default=TraceConfig.figure_root)
    p.add_argument("--seed", type=int, default=TraceConfig.seed)
    p.add_argument("--epochs", type=int, default=TraceConfig.epochs)
    p.add_argument(
        "--models",
        nargs="+",
        default=list(TraceConfig.models),
        choices=["onehot_vicreg", "finetuned_esmc_vicreg"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TraceConfig(
        output_root=args.output_root,
        figure_root=args.figure_root,
        seed=args.seed,
        epochs=args.epochs,
        models=tuple(args.models),
    )
    out_root = Path(cfg.output_root)
    fig_root = Path(cfg.figure_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fig_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 72, flush=True)
    print("VICReg distance-magnitude trace", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {cfg.seed}", flush=True)
    print(f"Epochs: {cfg.epochs}", flush=True)
    print(f"Models: {cfg.models}", flush=True)
    print(f"Outputs: {out_root}", flush=True)
    print(f"Figures: {fig_root}", flush=True)
    print("Distances: unnormalized only; norms are scale diagnostics", flush=True)
    print("=" * 72, flush=True)

    with open(out_root / "run_config.json", "w") as f:
        cfg_dump = asdict(cfg)
        cfg_dump["models"] = list(cfg.models)
        json.dump(cfg_dump, f, indent=2)

    frames: List[pd.DataFrame] = []
    if "onehot_vicreg" in cfg.models:
        model_out = out_root / f"onehot_vicreg_seed{cfg.seed}"
        model_out.mkdir(parents=True, exist_ok=True)
        frames.append(train_onehot(cfg, device, model_out))

    if "finetuned_esmc_vicreg" in cfg.models:
        model_out = out_root / f"finetuned_esmc_vicreg_seed{cfg.seed}"
        model_out.mkdir(parents=True, exist_ok=True)
        frames.append(train_finetuned_esm(cfg, device, model_out))

    if not frames:
        raise RuntimeError("No models selected.")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined.to_csv(out_root / "distance_trace_combined.csv", index=False)

    for model in combined["model_name"].unique():
        plot_model_traces(combined, model, fig_root)
    plot_combined(combined, fig_root)

    print("=" * 72, flush=True)
    print("Distance-magnitude trace complete", flush=True)
    print(f"Combined CSV: {out_root / 'distance_trace_combined.csv'}", flush=True)
    print(f"Figures: {fig_root}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
