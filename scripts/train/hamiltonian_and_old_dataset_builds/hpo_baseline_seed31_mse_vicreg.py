#!/usr/bin/env python3
"""
Reproducible seed-31 original VICReg-style MSE alignment grid without scout phase.

Key changes versus the original script
--------------------------------------
1. Removes the scout phase entirely.
2. Uses a single fixed seed from config for reproducibility.
3. Moves the grid to a JSON config file.
4. Searches loss-weight regimes that are equal across terms
   (all_1, all_10, all_25, all_100, all_1000 by default).
5. Monitors histogram geometry, but does NOT use geometry as the
   checkpoint-selection objective. Best checkpoint is selected by
   validation loss, with collapse guards and standard early stopping.
6. Saves best + last checkpoint for every config.
7. Adds post-hoc cross-reactivity analyses.

Assumptions
-----------
- Training shards contain positives only.
- Validation/test shards contain positives + negatives with binding_flag.
- Pair metadata CSVs contain pair_id plus peptide/TCR sequence columns, but
  exact column names may vary; flexible resolvers are implemented below.
"""

import os
import copy
import math
import json
import random
import logging
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


# ============================================================
# REPRODUCIBILITY
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================
class ShardedBatchTripletDataset(Dataset):
    def __init__(self, shards_dir):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shards_dir}")

        self.index = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for j in range(len(shard)):
                self.index.append((sp, j))

        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sp, j = self.index[idx]
        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp
        return self._cache_data[j]


# ============================================================
# MODELS
# ============================================================

class ESMProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_max, dropout=0.0):
        super().__init__()
        self.D = D
        self.rL = rL
        self.rD = rD
        self.d = d
        self.L_max = L_max

        self.B_c = nn.Parameter(torch.empty(D, rD))
        nn.init.xavier_uniform_(self.B_c)

        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        nn.init.xavier_uniform_(self.A_c)

        self.H_c = nn.Parameter(torch.empty(rL * rD, d))
        nn.init.xavier_uniform_(self.H_c)

        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, emb, mask):
        device = emb.device
        B, _, D_in = emb.shape
        assert D_in == self.D

        L_true = mask.sum(dim=1)
        z_list = []

        for b in range(B):
            Lb = int(L_true[b].item())
            if Lb == 0:
                z_list.append(torch.zeros(self.d, device=device))
                continue

            Xb = emb[b, :Lb, :]
            mb = mask[b, :Lb].unsqueeze(-1).float()
            Xb = Xb * mb

            Yb = Xb @ self.B_c
            A_pos = self.A_c[:Lb, :]
            Ub = A_pos.T @ Yb
            Ub_flat = Ub.reshape(-1)
            z_b = Ub_flat @ self.H_c
            z_list.append(z_b)

        z = torch.stack(z_list, dim=0)
        z = self.expander(z)
        return z


class PMHCProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_P_max, L_H_max, R_PH=0.7, dropout=0.0):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        assert d_P > 0 and d_H > 0

        self.pep_encoder = ESMProjectionHead(
            D, rL, rD, d_P, L_P_max, dropout=dropout
        )
        self.hla_encoder = ESMProjectionHead(
            D, rL, rD, d_H, L_H_max, dropout=dropout
        )

    def forward(self, emb_P, mask_P, emb_H, mask_H):
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)


# ============================================================
# LOSS
# ============================================================
def vicreg_variance(u, gamma=1.0, eps=1e-4):
    """VICReg variance floor on raw, unnormalised embeddings."""
    u_centered = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u_centered.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u):
    """VICReg covariance penalty using the paper-style / d scaling."""
    B, d = u.shape
    u_centered = u - u.mean(dim=0, keepdim=True)
    cov = (u_centered.T @ u_centered) / max(B - 1, 1)
    diag = torch.diag(cov)
    cov_off = cov - torch.diag_embed(diag)
    return (cov_off ** 2).sum() / d


def group_variance_floor(z, group_ids, gamma=0.05, min_group_size=3):
    """Optional peptide-aware variance floor in raw space."""
    if group_ids is None:
        return z.new_tensor(0.0), {"group_var_loss": 0.0, "group_var_n_groups": 0, "group_var_std_mean": 0.0}
    valid = group_ids >= 0
    if int(valid.sum().item()) < min_group_size:
        return z.new_tensor(0.0), {"group_var_loss": 0.0, "group_var_n_groups": 0, "group_var_std_mean": 0.0}
    losses, stds = [], []
    for gid in torch.unique(group_ids[valid]):
        idx = group_ids == gid
        if int(idx.sum().item()) < min_group_size:
            continue
        zg = z[idx]
        group_std = zg.std(dim=0, unbiased=False).mean()
        stds.append(group_std.detach())
        losses.append(F.relu(gamma - group_std))
    if not losses:
        return z.new_tensor(0.0), {"group_var_loss": 0.0, "group_var_n_groups": 0, "group_var_std_mean": 0.0}
    loss = torch.stack(losses).mean()
    return loss, {
        "group_var_loss": float(loss.detach().item()),
        "group_var_n_groups": int(len(losses)),
        "group_var_std_mean": float(torch.stack(stds).mean().item()),
    }


def vicreg_hamiltonian_loss(
    zT_raw,
    zPH_raw,
    alpha=25.0,
    beta=25.0,
    delta=1.0,
    gamma_var=1.0,
    eps=1e-4,
    peptide_group_ids=None,
    eta_peptide_group=0.0,
    group_gamma=0.05,
    group_min_size=3,
):
    """
    Original VICReg-style objective for this two-tower setup.

    No L2 normalisation and no Hamiltonian. The alignment term is raw MSE
    between the projected TCR and pMHC embeddings. Evaluation uses the raw
    squared distance ||zT - zPH||^2 as the positive/negative separation score.
    """
    sq_dist_per_example = ((zT_raw - zPH_raw) ** 2).sum(dim=-1)
    L_inv = F.mse_loss(zT_raw, zPH_raw)

    L_var_T = vicreg_variance(zT_raw, gamma=gamma_var, eps=eps)
    L_var_PH = vicreg_variance(zPH_raw, gamma=gamma_var, eps=eps)
    L_cov_T = vicreg_covariance(zT_raw)
    L_cov_PH = vicreg_covariance(zPH_raw)
    L_var_total = L_var_T + L_var_PH
    L_cov_total = L_cov_T + L_cov_PH

    L_peptide_group, group_stats = group_variance_floor(
        zT_raw, peptide_group_ids, gamma=group_gamma, min_group_size=group_min_size
    )

    L_total = alpha * L_inv + beta * L_var_total + delta * L_cov_total + eta_peptide_group * L_peptide_group

    stats = {
        "dist_mean": float(sq_dist_per_example.mean().item()),
        "dist_std": float(sq_dist_per_example.std(unbiased=False).item()),
        "L_inv": float(L_inv.item()),
        "L_var_total": float(L_var_total.item()),
        "L_cov_total": float(L_cov_total.item()),
        "L_peptide_group": float(L_peptide_group.item()),
        **group_stats,
        "zT_dim_std_mean": float(zT_raw.std(dim=0, unbiased=False).mean().item()),
        "zPH_dim_std_mean": float(zPH_raw.std(dim=0, unbiased=False).mean().item()),
        "zT_norm_mean": float(zT_raw.norm(dim=-1).mean().item()),
        "zPH_norm_mean": float(zPH_raw.norm(dim=-1).mean().item()),
    }
    return L_total, stats

# ============================================================
# METRICS + GEOMETRY
# ============================================================
def compute_binary_metrics(labels, preds):
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def apply_threshold(H_scores, threshold):
    return (H_scores <= threshold).astype(int)


def find_best_threshold_on_H(H_scores, labels):
    thresholds = np.unique(H_scores)
    best = None
    for thr in thresholds:
        preds = apply_threshold(H_scores, thr)
        metrics = compute_binary_metrics(labels, preds)
        if best is None or metrics["f1"] > best["f1"]:
            best = {
                "threshold": float(thr),
                "direction": "<=",
                **metrics,
            }
    return best


def estimate_overlap_coefficient(pos, neg, bins=60):
    if len(pos) == 0 or len(neg) == 0:
        return 1.0
    lo = min(pos.min(), neg.min())
    hi = max(pos.max(), neg.max())
    if hi <= lo:
        return 1.0
    hist_pos, edges = np.histogram(pos, bins=bins, range=(lo, hi), density=True)
    hist_neg, _ = np.histogram(neg, bins=bins, range=(lo, hi), density=True)
    bin_w = edges[1] - edges[0]
    return float(np.clip(np.minimum(hist_pos, hist_neg).sum() * bin_w, 0.0, 1.0))


def compute_geometry_metrics(H_scores, labels, overlap_bins=60):
    labels = np.asarray(labels).astype(int)
    H_scores = np.asarray(H_scores, dtype=float)

    pos = H_scores[labels == 1]
    neg = H_scores[labels == 0]

    pos_mean = float(pos.mean())
    neg_mean = float(neg.mean())
    pos_med = float(np.median(pos))
    neg_med = float(np.median(neg))

    mean_gap = float(neg_mean - pos_mean)
    median_gap = float(neg_med - pos_med)

    pos_std = float(pos.std(ddof=1)) if len(pos) > 1 else 0.0
    neg_std = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    pooled_std = math.sqrt(max((pos_std ** 2 + neg_std ** 2) / 2.0, 1e-12))
    effect_size = float(mean_gap / pooled_std)

    overlap = estimate_overlap_coefficient(pos, neg, bins=overlap_bins)
    ks_stat, ks_p = ks_2samp(neg, pos)

    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "pos_median": pos_med,
        "neg_median": neg_med,
        "mean_gap": mean_gap,
        "median_gap": median_gap,
        "effect_size": effect_size,
        "overlap": float(overlap),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_distance_histogram(H_vals, labels, title, save_path, threshold=None):
    H_vals = np.asarray(H_vals)
    labels = np.asarray(labels).astype(int)
    pos = H_vals[labels == 1]
    neg = H_vals[labels == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="negative")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="positive")
    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"thr <= {threshold:.4f}")
    plt.xlabel("Raw squared distance ||zT - zPH||² (lower = stronger predicted binding)")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_group_boxplot(observed, random_baseline, title, ylabel, save_path):
    plt.figure(figsize=(6, 5))
    plt.boxplot([observed, random_baseline], labels=["cross-reactive group", "matched random"], showfliers=False)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_frequency_scatter(df, x_col, y_col, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(6, 5))
    x = df[x_col].fillna(0).to_numpy()
    y = df[y_col].to_numpy()
    s = 20 + 15 * df["group_size"].to_numpy()
    plt.scatter(x, y, s=s, alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ============================================================
# FORWARD + EVALUATE
# ============================================================
@torch.no_grad()
def forward_batch(batch, tcr_proj, pmhc_proj, device, eps=1e-8):
    eT = batch["emb_T"].to(device)
    mT = batch["mask_T"].to(device)
    eP = batch["emb_P"].to(device)
    mP = batch["mask_P"].to(device)
    eH = batch["emb_H"].to(device)
    mH = batch["mask_H"].to(device)

    zT = tcr_proj(eT, mT)
    zPH = pmhc_proj(eP, mP, eH, mH)

    # Raw squared Euclidean distance used for evaluation histograms.
    # Lower distance = stronger predicted binder.
    dist = ((zT - zPH) ** 2).sum(dim=-1)

    labels = batch["binding_flag"]
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    return {
        "zT": zT,
        "zPH": zPH,
        "cos": dist.cpu().numpy(),  # legacy key; now stores raw distance
        "H": dist.cpu().numpy(),    # legacy key; lower distance = stronger predicted binder
        "labels": labels,
        "pair_ids": batch["pair_id"],
    }


@torch.no_grad()
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=60, keep_embeddings=False):
    tcr_proj.eval()
    pmhc_proj.eval()

    all_H, all_cos, all_lab, all_pid = [], [], [], []
    all_zT, all_zPH = [], []
    running_loss, n_steps = 0.0, 0
    stat_acc = {
        "dist_mean": [], "dist_std": [],
        "zT_dim_std_mean": [], "zPH_dim_std_mean": [],
        "zT_norm_mean": [], "zPH_norm_mean": [],
        "L_inv": [], "L_var_total": [], "L_cov_total": [],
        "L_peptide_group": [], "group_var_loss": [],
        "group_var_n_groups": [], "group_var_std_mean": [],
    }

    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device, eps=cfg["eps"])

        all_H.append(out["H"])
        all_cos.append(out["cos"])
        all_lab.append(out["labels"])
        all_pid.extend(list(out["pair_ids"]))
        if keep_embeddings:
            all_zT.append(out["zT"].cpu().numpy())
            all_zPH.append(out["zPH"].cpu().numpy())

        loss, stats = vicreg_hamiltonian_loss(
            out["zT"], out["zPH"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            delta=cfg["delta"],
            gamma_var=cfg["gamma_var"],
            eps=cfg["eps"],
        )
        running_loss += loss.item()
        n_steps += 1
        for k in stat_acc:
            stat_acc[k].append(stats.get(k, 0.0))

    H_vals = np.concatenate(all_H)
    cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_lab).astype(int)
    ranking_scores = -H_vals

    metrics = {
        "val_loss": float(running_loss / max(n_steps, 1)),
        "auroc": float(roc_auc_score(labels, ranking_scores)),
        "auprc": float(average_precision_score(labels, ranking_scores)),
        "dist_std": float(np.std(H_vals)),
        "zT_dim_std_mean": float(np.mean(stat_acc["zT_dim_std_mean"])),
        "zPH_dim_std_mean": float(np.mean(stat_acc["zPH_dim_std_mean"])),
        "zT_norm_mean": float(np.mean(stat_acc["zT_norm_mean"])),
        "zPH_norm_mean": float(np.mean(stat_acc["zPH_norm_mean"])),
        "L_inv": float(np.mean(stat_acc["L_inv"])),
        "L_var_total": float(np.mean(stat_acc["L_var_total"])),
        "L_cov_total": float(np.mean(stat_acc["L_cov_total"])),
    }
    geom = compute_geometry_metrics(H_vals, labels, overlap_bins=overlap_bins)

    out = {
        "H": H_vals,
        "cos": cos_vals,
        "labels": labels,
        "pair_ids": all_pid,
        "metrics": metrics,
        "geometry": geom,
    }
    if keep_embeddings:
        out["zT"] = np.concatenate(all_zT)
        out["zPH"] = np.concatenate(all_zPH)
    return out


# ============================================================
# COLLAPSE GUARD
# ============================================================
def collapse_flag(eval_out, guard_cfg):
    if not guard_cfg.get("enabled", True):
        return False, []

    reasons = []
    m = eval_out["metrics"]
    g = eval_out["geometry"]

    if m["dist_std"] < guard_cfg.get("dist_std_min", 1e-6):
        reasons.append(f"dist_std<{guard_cfg.get('dist_std_min', 1e-6)}")
    if m["zT_dim_std_mean"] < guard_cfg["dim_std_mean_min"]:
        reasons.append(f"zT_dim_std_mean<{guard_cfg['dim_std_mean_min']}")
    if m["zPH_dim_std_mean"] < guard_cfg["dim_std_mean_min"]:
        reasons.append(f"zPH_dim_std_mean<{guard_cfg['dim_std_mean_min']}")
    if g["overlap"] > guard_cfg["overlap_max_for_collapse"] and m["auroc"] < guard_cfg["auroc_max_for_collapse"]:
        reasons.append("near-complete distance overlap + AUROC near chance")

    return (len(reasons) > 0), reasons


# ============================================================
# CHECKPOINT HELPERS
# ============================================================
def build_models(cfg, train_loader, device):
    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], cfg["d"], L_T,
        dropout=cfg["dropout"]
    ).to(device)

    pmhc_proj = PMHCProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], cfg["d"],
        L_P, L_H, R_PH=cfg["r_ph"],
        dropout=cfg["dropout"]
    ).to(device)
    return tcr_proj, pmhc_proj


def save_epoch_checkpoint(path, epoch_state):
    torch.save(epoch_state, path)


# ============================================================
# TRAINING
# ============================================================
def run_single(cfg, train_loader, val_loader, device, dirs, overlap_bins, guard_cfg):
    set_global_seed(cfg["seed"])

    tcr_proj, pmhc_proj = build_models(cfg, train_loader, device)
    optimizer = torch.optim.AdamW(
        list(tcr_proj.parameters()) + list(pmhc_proj.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    collapse_bad_epochs = 0
    history = []

    cfg_stem = cfg["name"]
    last_ckpt_path = dirs["save_dir"] / f"{cfg_stem}__last.pt"
    best_ckpt_path = dirs["save_dir"] / f"{cfg_stem}__best.pt"

    for epoch in range(cfg["max_epochs"]):
        tcr_proj.train()
        pmhc_proj.train()
        train_running_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc_proj(
                batch["emb_P"].to(device),
                batch["mask_P"].to(device),
                batch["emb_H"].to(device),
                batch["mask_H"].to(device),
            )

            loss, _ = vicreg_hamiltonian_loss(
                zT, zPH,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                delta=cfg["delta"],
                gamma_var=cfg["gamma_var"],
                eps=cfg["eps"],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tcr_proj.parameters()) + list(pmhc_proj.parameters()),
                cfg["grad_clip_norm"],
            )
            optimizer.step()

            train_running_loss += loss.item()
            train_steps += 1

        val_out = evaluate_loader(
            val_loader, tcr_proj, pmhc_proj, device, cfg,
            overlap_bins=overlap_bins, keep_embeddings=False
        )

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_running_loss / max(train_steps, 1)),
            **val_out["metrics"],
            **{f"geom_{k}": v for k, v in val_out["geometry"].items()},
        }
        history.append(epoch_row)

        is_collapse, collapse_reasons = False, []
        if epoch + 1 >= guard_cfg["min_epochs"]:
            is_collapse, collapse_reasons = collapse_flag(val_out, guard_cfg)
            if is_collapse:
                collapse_bad_epochs += 1
            else:
                collapse_bad_epochs = 0

        log.info(
            f"[{cfg_stem}] ep{epoch+1}/{cfg['max_epochs']} | "
            f"train_loss={epoch_row['train_loss']:.4f} | val_loss={epoch_row['val_loss']:.4f} | "
            f"auroc={epoch_row['auroc']:.4f} | auprc={epoch_row['auprc']:.4f} | "
            f"mean_gap={epoch_row['geom_mean_gap']:.4f} | overlap={epoch_row['geom_overlap']:.4f} | "
            f"collapse={is_collapse}"
        )
        if collapse_reasons:
            log.info(f"[{cfg_stem}] collapse guard reasons: {collapse_reasons}")

        epoch_state = {
            "cfg": copy.deepcopy(cfg),
            "epoch": epoch + 1,
            "tcr_state_dict": copy.deepcopy(tcr_proj.state_dict()),
            "pmhc_state_dict": copy.deepcopy(pmhc_proj.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "val_metrics": copy.deepcopy(val_out["metrics"]),
            "val_geometry": copy.deepcopy(val_out["geometry"]),
            "history": copy.deepcopy(history),
            "collapse_reasons": collapse_reasons,
        }
        save_epoch_checkpoint(last_ckpt_path, epoch_state)

        if val_out["metrics"]["val_loss"] < best_val_loss:
            best_val_loss = val_out["metrics"]["val_loss"]
            best_state = copy.deepcopy(epoch_state)
            save_epoch_checkpoint(best_ckpt_path, best_state)
            bad_epochs = 0
        else:
            if epoch + 1 >= cfg["min_epochs_before_early_stop"]:
                bad_epochs += 1

        if collapse_bad_epochs >= guard_cfg["max_bad_epochs"]:
            log.info(f"[{cfg_stem}] stopped for repeated collapse evidence at epoch {epoch+1}")
            break
        if bad_epochs >= cfg["patience"]:
            log.info(f"[{cfg_stem}] early stop on validation loss at epoch {epoch+1}")
            break

    if best_state is None:
        raise RuntimeError(f"No best checkpoint saved for {cfg_stem}")

    pd.DataFrame(history).to_csv(dirs["save_dir"] / f"{cfg_stem}__history.csv", index=False)
    return best_state


# ============================================================
# FINAL EVAL
# ============================================================
def load_best_models(best_state, train_loader, device):
    cfg = best_state["cfg"]
    tcr_proj, pmhc_proj = build_models(cfg, train_loader, device)
    tcr_proj.load_state_dict(best_state["tcr_state_dict"])
    pmhc_proj.load_state_dict(best_state["pmhc_state_dict"])
    return tcr_proj, pmhc_proj


def final_evaluate(best_state, train_loader, val_loader, test_loader, device, overlap_bins):
    tcr_proj, pmhc_proj = load_best_models(best_state, train_loader, device)
    cfg = best_state["cfg"]

    val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=overlap_bins, keep_embeddings=True)
    val_thr = find_best_threshold_on_H(val_out["H"], val_out["labels"])

    test_out = evaluate_loader(test_loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=overlap_bins, keep_embeddings=True)
    test_preds = apply_threshold(test_out["H"], val_thr["threshold"])
    test_bin = compute_binary_metrics(test_out["labels"], test_preds)
    test_cm = confusion_matrix(test_out["labels"], test_preds)

    return {
        "cfg": cfg,
        "val_out": val_out,
        "test_out": test_out,
        "val_threshold": val_thr,
        "test_metrics": {
            "auroc": test_out["metrics"]["auroc"],
            "auprc": test_out["metrics"]["auprc"],
            **test_bin,
        },
        "test_confusion_matrix": test_cm.tolist(),
    }


# ============================================================
# METADATA RESOLUTION
# ============================================================
def _find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_split_metadata(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        log.warning(f"Metadata CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        log.warning(f"pair_id missing from {csv_path}; skipping metadata join")
        return None

    pep_col = _find_column(df, ["peptide", "pep", "pep_seq", "peptide_seq", "epitope", "antigen_peptide"])
    tcrb_col = _find_column(df, ["tcrb", "tcr_b", "cdr3b", "cdr3_beta", "tcr_beta", "trb_cdr3"])
    tcra_col = _find_column(df, ["tcra", "tcr_a", "cdr3a", "cdr3_alpha", "tcr_alpha", "tra_cdr3"])
    tcr_col = _find_column(df, ["tcr", "tcr_seq", "sequence_aa", "cdr3", "cdr3_sequence"])
    bind_col = _find_column(df, ["binding_flag", "label", "binder"])

    out = pd.DataFrame({"pair_id": df["pair_id"].astype(str)})
    if pep_col is not None:
        out["peptide_key"] = df[pep_col].astype(str)
    else:
        out["peptide_key"] = np.nan

    if tcra_col is not None or tcrb_col is not None:
        a = df[tcra_col].astype(str) if tcra_col is not None else pd.Series(["NA"] * len(df))
        b = df[tcrb_col].astype(str) if tcrb_col is not None else pd.Series(["NA"] * len(df))
        out["tcr_key"] = a + "|" + b
    elif tcr_col is not None:
        out["tcr_key"] = df[tcr_col].astype(str)
    else:
        out["tcr_key"] = np.nan

    if bind_col is not None:
        out["binding_flag_meta"] = df[bind_col]
    return out.drop_duplicates("pair_id")


def compute_train_exposure(train_csv):
    meta = load_split_metadata(train_csv)
    if meta is None:
        return {}, {}
    pep_counts = meta["peptide_key"].value_counts(dropna=True).to_dict() if "peptide_key" in meta.columns else {}
    tcr_counts = meta["tcr_key"].value_counts(dropna=True).to_dict() if "tcr_key" in meta.columns else {}
    return pep_counts, tcr_counts


def eval_to_df(eval_out, split_meta, pep_counts, tcr_counts):
    df = pd.DataFrame({
        "pair_id": pd.Series(eval_out["pair_ids"]).astype(str),
        "binding_flag": eval_out["labels"].astype(int),
        "H": eval_out["H"],
        "cos": eval_out["cos"],
        "row_idx": np.arange(len(eval_out["labels"])),
    })
    if split_meta is not None:
        df = df.merge(split_meta, on="pair_id", how="left")
    else:
        df["peptide_key"] = np.nan
        df["tcr_key"] = np.nan

    df["pep_train_count"] = df["peptide_key"].map(pep_counts).fillna(0).astype(int)
    df["tcr_train_count"] = df["tcr_key"].map(tcr_counts).fillna(0).astype(int)
    return df


# ============================================================
# CROSS-REACTIVITY
# ============================================================
def pairwise_cosine(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return np.array([])
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    z = arr / norms
    sims = z @ z.T
    iu = np.triu_indices(len(arr), k=1)
    return sims[iu]


def groupwise_cross_reactivity(
    df,
    variable_key,
    embedding_matrix,
    exposure_col,
    min_group_size,
    n_random_repeats,
    random_seed,
):
    rng = np.random.default_rng(random_seed)
    pos = df[df["binding_flag"] == 1].copy()
    pos = pos.dropna(subset=[variable_key])
    groups = []

    other_idx_all = pos["row_idx"].to_numpy()

    for key, grp in pos.groupby(variable_key):
        if grp["row_idx"].nunique() < min_group_size:
            continue

        idx = grp["row_idx"].to_numpy()
        obs = pairwise_cosine(embedding_matrix[idx])
        if len(obs) == 0:
            continue

        n = len(idx)
        baseline_vals = []
        eligible = pos[pos[variable_key] != key]["row_idx"].unique()
        if len(eligible) < n:
            continue

        for _ in range(n_random_repeats):
            sampled = rng.choice(eligible, size=n, replace=False)
            rand_sims = pairwise_cosine(embedding_matrix[sampled])
            if len(rand_sims) > 0:
                baseline_vals.append(np.median(rand_sims))

        if not baseline_vals:
            continue

        groups.append({
            "group_key": key,
            "group_size": int(n),
            "observed_median_pairwise_cos": float(np.median(obs)),
            "observed_mean_pairwise_cos": float(np.mean(obs)),
            "baseline_median_pairwise_cos": float(np.mean(baseline_vals)),
            "delta_vs_random": float(np.median(obs) - np.mean(baseline_vals)),
            "train_exposure": int(grp[exposure_col].iloc[0]) if exposure_col in grp else 0,
        })

    return pd.DataFrame(groups)


def save_cross_reactivity_outputs(summary_df, out_prefix, fig_dir, entity_label):
    if summary_df.empty:
        log.warning(f"No cross-reactivity groups found for {entity_label}")
        return None

    summary_df = summary_df.sort_values("delta_vs_random", ascending=False)
    summary_df.to_csv(f"{out_prefix}.csv", index=False)

    plot_group_boxplot(
        observed=summary_df["observed_median_pairwise_cos"].to_numpy(),
        random_baseline=summary_df["baseline_median_pairwise_cos"].to_numpy(),
        title=f"{entity_label}: within-group similarity vs matched random baseline",
        ylabel="Median pairwise cosine",
        save_path=fig_dir / f"{Path(out_prefix).name}__boxplot.png",
    )

    plot_frequency_scatter(
        summary_df,
        x_col="train_exposure",
        y_col="delta_vs_random",
        title=f"{entity_label}: train exposure vs excess within-group similarity",
        xlabel="Training-set exposure count",
        ylabel="Observed - matched random median cosine",
        save_path=fig_dir / f"{Path(out_prefix).name}__train_exposure_scatter.png",
    )

    corr = np.nan
    pval = np.nan
    if summary_df["train_exposure"].nunique() > 1:
        corr, pval = spearmanr(summary_df["train_exposure"], summary_df["delta_vs_random"])
    return {
        "n_groups": int(len(summary_df)),
        "median_delta_vs_random": float(summary_df["delta_vs_random"].median()),
        "spearman_train_exposure_vs_delta": None if np.isnan(corr) else float(corr),
        "spearman_p": None if np.isnan(pval) else float(pval),
    }


def run_cross_reactivity_analysis(eval_out, split_name, split_csv, train_csv, fig_dir, out_dir, cfg):
    if not cfg["cross_reactivity"]["enabled"]:
        return {}

    pep_counts, tcr_counts = compute_train_exposure(train_csv)
    split_meta = load_split_metadata(split_csv)
    eval_df = eval_to_df(eval_out, split_meta, pep_counts, tcr_counts)

    # Peptide-centric: many TCRs for one peptide -> compare TCR embeddings within peptide groups.
    peptide_df = groupwise_cross_reactivity(
        df=eval_df,
        variable_key="peptide_key",
        embedding_matrix=eval_out["zT"],
        exposure_col="pep_train_count",
        min_group_size=cfg["cross_reactivity"]["min_group_size"],
        n_random_repeats=cfg["cross_reactivity"]["n_random_repeats"],
        random_seed=cfg["cross_reactivity"]["random_seed"],
    )

    # TCR-centric: one TCR for many peptides -> compare pMHC embeddings within TCR groups.
    tcr_df = groupwise_cross_reactivity(
        df=eval_df,
        variable_key="tcr_key",
        embedding_matrix=eval_out["zPH"],
        exposure_col="tcr_train_count",
        min_group_size=cfg["cross_reactivity"]["min_group_size"],
        n_random_repeats=cfg["cross_reactivity"]["n_random_repeats"],
        random_seed=cfg["cross_reactivity"]["random_seed"],
    )

    peptide_summary = save_cross_reactivity_outputs(
        peptide_df,
        str(out_dir / f"{split_name}__peptide_centric_cross_reactivity"),
        fig_dir,
        entity_label=f"{split_name} peptide-centric cross-reactivity",
    )
    tcr_summary = save_cross_reactivity_outputs(
        tcr_df,
        str(out_dir / f"{split_name}__tcr_centric_cross_reactivity"),
        fig_dir,
        entity_label=f"{split_name} TCR-centric cross-reactivity",
    )

    return {
        "peptide_centric": peptide_summary,
        "tcr_centric": tcr_summary,
    }


# ============================================================
# CONFIG GRID
# ============================================================
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def build_config_grid(base_cfg):
    arch = base_cfg["architecture_grid"]
    configs = []
    for (rL, rD, lr, wd, dropout), lw in product(
        product(
            arch["rL"], arch["rD"], arch["lr"], arch["wd"], arch["dropout"]
        ),
        base_cfg["loss_weight_sets"],
    ):
        cfg = {
            "seed": base_cfg["seed"],
            "d": base_cfg["fixed_d"],
            "r_ph": base_cfg["r_ph"],
            "gamma_var": base_cfg["gamma_var"],
            "eps": base_cfg["eps"],
            "grad_clip_norm": base_cfg["grad_clip_norm"],
            "max_epochs": base_cfg["max_epochs"],
            "patience": base_cfg["patience"],
            "min_epochs_before_early_stop": base_cfg["min_epochs_before_early_stop"],
            "rL": rL,
            "rD": rD,
            "lr": lr,
            "wd": wd,
            "dropout": dropout,
            "alpha": lw["alpha"],
            "beta": lw["beta"],
            "delta": lw["delta"],
        }
        cfg["name"] = (
            f"seed{cfg['seed']}__rL{rL}__rD{rD}__lr{lr:.0e}__"
            f"{lw['name']}"
        )
        configs.append(cfg)
    return configs


# ============================================================
# MAIN
# ============================================================
def main(config_path):
    base_cfg = load_json(config_path)
    set_global_seed(base_cfg["seed"])

    run_name = base_cfg["run_name"]
    save_dir = Path(base_cfg["checkpoints_dir"]) / run_name
    fig_dir = Path(base_cfg["figures_dir"]) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    dirs = {"save_dir": save_dir, "fig_dir": fig_dir}

    if base_cfg["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(base_cfg["device"])
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    train_loader = DataLoader(
        ShardedBatchTripletDataset(Path(base_cfg["embed_root"]) / "train"),
        batch_size=base_cfg["batch_size"],
        shuffle=base_cfg["shuffle_train"],
        num_workers=base_cfg["num_workers"],
        collate_fn=lambda x: x[0],
    )
    val_loader = DataLoader(
        ShardedBatchTripletDataset(Path(base_cfg["embed_root"]) / "val"),
        batch_size=base_cfg["batch_size"],
        shuffle=False,
        num_workers=base_cfg["num_workers"],
        collate_fn=lambda x: x[0],
    )
    test_loader = DataLoader(
        ShardedBatchTripletDataset(Path(base_cfg["embed_root"]) / "test"),
        batch_size=base_cfg["batch_size"],
        shuffle=False,
        num_workers=base_cfg["num_workers"],
        collate_fn=lambda x: x[0],
    )

    configs = build_config_grid(base_cfg)
    log.info(f"Run name: {run_name}")
    log.info(f"Number of configs: {len(configs)}")
    log.info(f"Seed fixed at: {base_cfg['seed']}")

    summary_rows = []
    all_best_states = []
    overlap_bins = base_cfg["geometry_monitor"]["overlap_bins"]
    guard_cfg = base_cfg["collapse_guard"]

    for cfg in configs:
        log.info("=" * 100)
        log.info(f"RUNNING {cfg['name']}")
        best_state = run_single(cfg, train_loader, val_loader, device, dirs, overlap_bins, guard_cfg)
        final_eval = final_evaluate(best_state, train_loader, val_loader, test_loader, device, overlap_bins)

        stem = cfg["name"]
        with open(save_dir / f"{stem}__final_eval.json", "w") as f:
            json.dump({
                "cfg": cfg,
                "best_epoch": best_state["epoch"],
                "val_metrics": final_eval["val_out"]["metrics"],
                "val_geometry": final_eval["val_out"]["geometry"],
                "val_threshold": final_eval["val_threshold"],
                "test_metrics": final_eval["test_metrics"],
                "test_geometry": final_eval["test_out"]["geometry"],
                "test_confusion_matrix": final_eval["test_confusion_matrix"],
            }, f, indent=2)

        plot_distance_histogram(
            final_eval["val_out"]["H"],
            final_eval["val_out"]["labels"],
            title=f"{stem} | val",
            save_path=fig_dir / f"{stem}__val_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )
        plot_distance_histogram(
            final_eval["test_out"]["H"],
            final_eval["test_out"]["labels"],
            title=f"{stem} | test",
            save_path=fig_dir / f"{stem}__test_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )

        row = {
            "cfg_name": stem,
            "best_epoch": best_state["epoch"],
            "val_loss": final_eval["val_out"]["metrics"]["val_loss"],
            "val_auroc": final_eval["val_out"]["metrics"]["auroc"],
            "val_auprc": final_eval["val_out"]["metrics"]["auprc"],
            "val_mean_gap": final_eval["val_out"]["geometry"]["mean_gap"],
            "val_effect_size": final_eval["val_out"]["geometry"]["effect_size"],
            "val_overlap": final_eval["val_out"]["geometry"]["overlap"],
            "test_auroc": final_eval["test_metrics"]["auroc"],
            "test_auprc": final_eval["test_metrics"]["auprc"],
            "test_f1": final_eval["test_metrics"]["f1"],
            "test_mean_gap": final_eval["test_out"]["geometry"]["mean_gap"],
            "test_effect_size": final_eval["test_out"]["geometry"]["effect_size"],
            "test_overlap": final_eval["test_out"]["geometry"]["overlap"],
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "delta": cfg["delta"],
            "rL": cfg["rL"],
            "rD": cfg["rD"],
            "lr": cfg["lr"],
        }
        summary_rows.append(row)
        all_best_states.append({
            "cfg_name": stem,
            "best_state": best_state,
            "final_eval": final_eval,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["val_loss", "val_auroc", "test_auroc"], ascending=[True, False, False]
    )
    summary_df.to_csv(save_dir / "full_summary.csv", index=False)

    best_name = summary_df.iloc[0]["cfg_name"]
    best_bundle = next(x for x in all_best_states if x["cfg_name"] == best_name)
    best_state = best_bundle["best_state"]
    final_eval = best_bundle["final_eval"]

    log.info("=" * 100)
    log.info(f"GLOBAL BEST BY VAL LOSS: {best_name}")
    log.info(summary_df.head(10).to_string(index=False))

    cross_val = run_cross_reactivity_analysis(
        final_eval["val_out"], "val", base_cfg["val_csv"], base_cfg["train_csv"], fig_dir, save_dir, base_cfg
    )
    cross_test = run_cross_reactivity_analysis(
        final_eval["test_out"], "test", base_cfg["test_csv"], base_cfg["train_csv"], fig_dir, save_dir, base_cfg
    )


    manifest = {
        "run_name": run_name,
        "config_path": str(config_path),
        "seed": base_cfg["seed"],
        "n_configs": len(configs),
        "best_cfg_name": best_name,
        "cross_reactivity": {
            "val": cross_val,
            "test": cross_test,
        },
    }
    with open(save_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("DONE")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/mnt/data/vicreg_seed31_config.json",
        help="Path to JSON config file",
    )
    args = parser.parse_args()
    main(args.config)
