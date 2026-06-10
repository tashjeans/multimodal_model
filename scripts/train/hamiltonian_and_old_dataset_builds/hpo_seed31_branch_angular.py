#!/usr/bin/env python3
"""
Reproducible seed-31 Hamiltonian / VICReg grid without scout phase.

Key changes versus the original script
--------------------------------------
1. Removes the scout phase entirely.
2. Uses a single fixed seed from config for reproducibility.
3. Moves the grid to a JSON config file.
4. Searches loss-weight regimes that are equal across terms
   (all_1, all_10, all_25, all_100, all_1000 by default).
5. Monitors histogram geometry, but does NOT use geometry as the
   checkpoint-selection objective. Best checkpoint is selected by
   validation AUROC, with collapse guards and standard early stopping.
6. Saves best + last checkpoint for every config.
7. Adds post-hoc cross-reactivity analyses after every config run.
8. Adds optional peptide-aware group variance floor during training.
9. Keeps covariance scaling configurable.
10. Adds branch-specific angular variance weights for TCR and pMHC branches.

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
    """
    VICReg raw-space variance floor.

    This operates on the unnormalised embeddings. It prevents collapse in raw
    Euclidean space, but it does not by itself guarantee angular spread after
    Hamiltonian L2 normalisation.
    """
    u_centered = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u_centered.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u, cov_scale="d2"):
    """
    VICReg covariance penalty.

    cov_scale="d2" reproduces your earlier stable baseline:
        sum(offdiag(C)^2) / d^2

    cov_scale="d" is closer to the VICReg paper scaling:
        sum(offdiag(C)^2) / d

    Keeping this configurable lets us isolate whether instability is caused by
    the covariance rescaling or by the angular/Hamiltonian geometry.
    """
    B, d = u.shape
    u_centered = u - u.mean(dim=0, keepdim=True)
    cov = (u_centered.T @ u_centered) / max(B - 1, 1)
    diag = torch.diag(cov)
    cov_off = cov - torch.diag_embed(diag)
    raw = (cov_off ** 2).sum()

    if cov_scale == "d":
        return raw / d
    if cov_scale == "d2":
        return raw / (d * d)
    if cov_scale == "none":
        return raw
    raise ValueError(f"Unknown cov_scale={cov_scale}")


def angular_spread_loss(z, std_gamma=0.035, mean_gamma=0.20, eps=1e-8):
    """
    Angular anti-collapse loss in the unit-normalised Hamiltonian space.

    The Hamiltonian only sees e = z / ||z||. Raw VICReg variance can therefore
    look healthy while all unit vectors point into the same narrow cone. This
    loss uses two complementary diagnostics:

    1. per-dimension unit-sphere std floor: prevents individual angular
       coordinates from becoming inactive;
    2. mean resultant-length ceiling: penalises a batch-level cone collapse,
       where ||mean(e)|| approaches 1.

    For d=128, uniform directions have coordinate std around 1/sqrt(d)=0.088.
    A std floor around 0.03-0.05 is therefore a soft floor, not a demand for
    fully uniform spherical coverage.
    """
    e = F.normalize(z, dim=-1, eps=eps)
    std = e.std(dim=0, unbiased=False)
    L_std = F.relu(std_gamma - std).mean()
    mean_resultant = e.mean(dim=0).norm(p=2)
    L_mean = F.relu(mean_resultant - mean_gamma)
    loss = L_std + L_mean
    return loss, {
        "unit_dim_std_mean": float(std.mean().detach().item()),
        "unit_dim_std_min": float(std.min().detach().item()),
        "unit_dim_std_max": float(std.max().detach().item()),
        "unit_mean_resultant": float(mean_resultant.detach().item()),
        "unit_L_std": float(L_std.detach().item()),
        "unit_L_mean": float(L_mean.detach().item()),
    }


# Backward-compatible alias for older calls/configs.
def angular_variance_floor(z, gamma=0.035, eps=1e-8):
    return angular_spread_loss(z, std_gamma=gamma, mean_gamma=0.20, eps=eps)


def group_variance_floor(z, group_ids, gamma=0.05, min_group_size=3, normalise=False, eps=1e-8, prefix="group"):
    """
    Peptide-aware variance floor.

    If normalise=False, this measures within-peptide spread in raw z-space.
    If normalise=True, this measures within-peptide spread on the unit sphere,
    which is the space used by the Hamiltonian.

    It is a floor, not a repulsion term: no penalty is applied once group_std
    exceeds gamma.
    """
    if group_ids is None:
        return z.new_tensor(0.0), {
            f"{prefix}_loss": 0.0,
            f"{prefix}_n_groups": 0,
            f"{prefix}_std_mean": 0.0,
            f"{prefix}_std_min": 0.0,
        }

    valid = group_ids >= 0
    if int(valid.sum().item()) < min_group_size:
        return z.new_tensor(0.0), {
            f"{prefix}_loss": 0.0,
            f"{prefix}_n_groups": 0,
            f"{prefix}_std_mean": 0.0,
            f"{prefix}_std_min": 0.0,
        }

    z_use = F.normalize(z, dim=-1, eps=eps) if normalise else z

    losses, stds = [], []
    for gid in torch.unique(group_ids[valid]):
        idx = group_ids == gid
        if int(idx.sum().item()) < min_group_size:
            continue
        zg = z_use[idx]
        group_std = zg.std(dim=0, unbiased=False).mean()
        stds.append(group_std.detach())
        losses.append(F.relu(gamma - group_std))

    if not losses:
        return z.new_tensor(0.0), {
            f"{prefix}_loss": 0.0,
            f"{prefix}_n_groups": 0,
            f"{prefix}_std_mean": 0.0,
            f"{prefix}_std_min": 0.0,
        }

    loss = torch.stack(losses).mean()
    std_tensor = torch.stack(stds)
    return loss, {
        f"{prefix}_loss": float(loss.detach().item()),
        f"{prefix}_n_groups": int(len(losses)),
        f"{prefix}_std_mean": float(std_tensor.mean().item()),
        f"{prefix}_std_min": float(std_tensor.min().item()),
    }


def vicreg_hamiltonian_loss(
    zT_raw,
    zPH_raw,
    alpha=25.0,
    beta=25.0,
    delta=25.0,
    gamma_var=1.0,
    eps=1e-4,
    cov_scale="d2",
    peptide_group_ids=None,
    tcr_group_ids=None,
    eta_peptide_group=0.0,
    eta_tcr_group=0.0,
    group_gamma=0.05,
    tcr_group_gamma=None,
    group_min_size=3,
    rho_angular=0.0,
    rho_angular_T=None,
    rho_angular_PH=None,
    angular_gamma=0.035,
    angular_gamma_T=None,
    angular_gamma_PH=None,
    angular_mean_gamma=0.20,
    angular_mean_gamma_T=None,
    angular_mean_gamma_PH=None,
    eta_peptide_angular=0.0,
    eta_tcr_angular=0.0,
    group_angular_gamma=0.02,
    tcr_group_angular_gamma=None,
):
    # Hamiltonian invariance/alignment space. Lower H means stronger binding.
    eT = F.normalize(zT_raw, dim=-1, eps=eps)
    ePH = F.normalize(zPH_raw, dim=-1, eps=eps)

    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos
    L_inv = H.mean()

    # Raw VICReg terms remain on the non-normalised embeddings. These protect
    # magnitude/information content that the Hamiltonian itself cannot see.
    L_var_T = vicreg_variance(zT_raw, gamma=gamma_var, eps=eps)
    L_var_PH = vicreg_variance(zPH_raw, gamma=gamma_var, eps=eps)
    L_cov_T = vicreg_covariance(zT_raw, cov_scale=cov_scale)
    L_cov_PH = vicreg_covariance(zPH_raw, cov_scale=cov_scale)
    L_var_total = L_var_T + L_var_PH
    L_cov_total = L_cov_T + L_cov_PH

    # Peptide-centric group floor: many TCRs binding the same peptide should not
    # all collapse to one TCR point. This is deliberately a floor, not repulsion.
    L_peptide_group, peptide_group_stats = group_variance_floor(
        zT_raw, peptide_group_ids, gamma=group_gamma, min_group_size=group_min_size,
        normalise=False, eps=eps, prefix="pep_raw",
    )

    # TCR-centric group floor: many peptides binding the same TCR should not all
    # collapse to one pMHC point. Kept separately and usually weighted softly.
    if tcr_group_gamma is None:
        tcr_group_gamma = group_gamma
    L_tcr_group, tcr_group_stats = group_variance_floor(
        zPH_raw, tcr_group_ids, gamma=tcr_group_gamma, min_group_size=group_min_size,
        normalise=False, eps=eps, prefix="tcr_raw",
    )

    # Branch-specific angular spread in the exact space used by H.
    if rho_angular_T is None:
        rho_angular_T = rho_angular
    if rho_angular_PH is None:
        rho_angular_PH = rho_angular
    if angular_gamma_T is None:
        angular_gamma_T = angular_gamma
    if angular_gamma_PH is None:
        angular_gamma_PH = angular_gamma
    if angular_mean_gamma_T is None:
        angular_mean_gamma_T = angular_mean_gamma
    if angular_mean_gamma_PH is None:
        angular_mean_gamma_PH = angular_mean_gamma

    L_ang_T, ang_T_stats = angular_spread_loss(
        zT_raw, std_gamma=angular_gamma_T, mean_gamma=angular_mean_gamma_T, eps=eps
    )
    L_ang_PH, ang_PH_stats = angular_spread_loss(
        zPH_raw, std_gamma=angular_gamma_PH, mean_gamma=angular_mean_gamma_PH, eps=eps
    )
    L_angular_total = L_ang_T + L_ang_PH

    # Angular group floors mirror the raw group floors, but on the unit sphere.
    L_peptide_ang, peptide_group_ang_stats = group_variance_floor(
        zT_raw, peptide_group_ids, gamma=group_angular_gamma, min_group_size=group_min_size,
        normalise=True, eps=eps, prefix="pep_ang",
    )
    if tcr_group_angular_gamma is None:
        tcr_group_angular_gamma = group_angular_gamma
    L_tcr_ang, tcr_group_ang_stats = group_variance_floor(
        zPH_raw, tcr_group_ids, gamma=tcr_group_angular_gamma, min_group_size=group_min_size,
        normalise=True, eps=eps, prefix="tcr_ang",
    )

    L_total = (
        alpha * L_inv
        + beta * L_var_total
        + delta * L_cov_total
        + eta_peptide_group * L_peptide_group
        + eta_tcr_group * L_tcr_group
        + rho_angular_T * L_ang_T
        + rho_angular_PH * L_ang_PH
        + eta_peptide_angular * L_peptide_ang
        + eta_tcr_angular * L_tcr_ang
    )

    stats = {
        "cos_mean": float(cos.mean().item()),
        "cos_std": float(cos.std(unbiased=False).item()),
        "H_mean": float(H.mean().item()),
        "H_std": float(H.std(unbiased=False).item()),
        "L_inv": float(L_inv.item()),
        "L_var_total": float(L_var_total.item()),
        "L_cov_total": float(L_cov_total.item()),
        "L_peptide_group": float(L_peptide_group.item()),
        "L_tcr_group": float(L_tcr_group.item()),
        "L_angular_total": float(L_angular_total.item()),
        "L_angular_T": float(L_ang_T.item()),
        "L_angular_PH": float(L_ang_PH.item()),
        "rho_angular_T_effective": float(rho_angular_T),
        "rho_angular_PH_effective": float(rho_angular_PH),
        "L_peptide_angular": float(L_peptide_ang.item()),
        "L_tcr_angular": float(L_tcr_ang.item()),
        "zT_dim_std_mean": float(zT_raw.std(dim=0, unbiased=False).mean().item()),
        "zPH_dim_std_mean": float(zPH_raw.std(dim=0, unbiased=False).mean().item()),
        "eT_dim_std_mean": float(ang_T_stats["unit_dim_std_mean"]),
        "ePH_dim_std_mean": float(ang_PH_stats["unit_dim_std_mean"]),
        "eT_dim_std_min": float(ang_T_stats["unit_dim_std_min"]),
        "ePH_dim_std_min": float(ang_PH_stats["unit_dim_std_min"]),
        "eT_mean_resultant": float(ang_T_stats["unit_mean_resultant"]),
        "ePH_mean_resultant": float(ang_PH_stats["unit_mean_resultant"]),
        "eT_L_std": float(ang_T_stats["unit_L_std"]),
        "ePH_L_std": float(ang_PH_stats["unit_L_std"]),
        "eT_L_mean": float(ang_T_stats["unit_L_mean"]),
        "ePH_L_mean": float(ang_PH_stats["unit_L_mean"]),
        # Backward-compatible aggregate names used by old summaries/logs.
        "group_var_loss": float(L_peptide_group.item()),
        "group_var_n_groups": int(peptide_group_stats.get("pep_raw_n_groups", 0)),
        "group_var_std_mean": float(peptide_group_stats.get("pep_raw_std_mean", 0.0)),
        "group_var_std_min": float(peptide_group_stats.get("pep_raw_std_min", 0.0)),
        "group_ang_loss": float(L_peptide_ang.item()),
        "group_ang_n_groups": int(peptide_group_ang_stats.get("pep_ang_n_groups", 0)),
        "group_ang_std_mean": float(peptide_group_ang_stats.get("pep_ang_std_mean", 0.0)),
        "group_ang_std_min": float(peptide_group_ang_stats.get("pep_ang_std_min", 0.0)),
        **peptide_group_stats,
        **tcr_group_stats,
        **peptide_group_ang_stats,
        **tcr_group_ang_stats,
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
def plot_h_histogram(H_vals, labels, title, save_path, threshold=None):
    H_vals = np.asarray(H_vals)
    labels = np.asarray(labels).astype(int)
    pos = H_vals[labels == 1]
    neg = H_vals[labels == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="negative")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="positive")
    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"thr <= {threshold:.4f}")
    plt.xlabel("Hamiltonian H (lower = stronger binding)")
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
# PAIR-ID TO BIOLOGICAL GROUP-ID LOOKUP
# ============================================================
def _normalise_pair_id_list(pair_ids):
    if torch.is_tensor(pair_ids):
        return [str(x) for x in pair_ids.detach().cpu().numpy().tolist()]
    if isinstance(pair_ids, (list, tuple)):
        out = []
        for x in pair_ids:
            if torch.is_tensor(x):
                out.append(str(x.detach().cpu().item()))
            else:
                out.append(str(x))
        return out
    return [str(pair_ids)]


def build_group_lookup(csv_path, id_col):
    """
    Build pair_id -> integer group-code lookup.

    Preferred use is explicit id_col in the CSV. If that column is absent, this
    falls back to load_split_metadata(), which can derive peptide_key and tcr_key
    from common peptide/TCR column names. This avoids silently disabling group
    losses when the CSV uses biological sequence columns rather than precomputed
    *_id columns.
    """
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        raise ValueError(f"pair_id missing from {csv_path}")

    if id_col in df.columns:
        values = df[id_col].astype(str).fillna("NA")
    else:
        meta = load_split_metadata(csv_path)
        fallback = {
            "peptide_id": "peptide_key",
            "peptide_key": "peptide_key",
            "tcr_id": "tcr_key",
            "tcr_key": "tcr_key",
        }.get(id_col)
        if meta is None or fallback is None or fallback not in meta.columns:
            raise ValueError(
                f"{id_col} missing from {csv_path}, and no derivable fallback group key was found."
            )
        df = df[["pair_id"]].merge(meta[["pair_id", fallback]], on="pair_id", how="left")
        values = df[fallback].astype(str).fillna("NA")

    value_to_code = {v: i for i, v in enumerate(sorted(values.unique().tolist()))}
    pair_to_code = dict(zip(df["pair_id"].astype(str), values.map(value_to_code).astype(int)))
    return pair_to_code, value_to_code


def ids_for_batch(batch, pair_to_code, device):
    pair_ids = _normalise_pair_id_list(batch["pair_id"])
    codes = [pair_to_code.get(pid, -1) for pid in pair_ids]
    return torch.tensor(codes, dtype=torch.long, device=device)


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

    eT_n = zT / (zT.norm(dim=-1, keepdim=True) + eps)
    ePH_n = zPH / (zPH.norm(dim=-1, keepdim=True) + eps)

    cos = (eT_n * ePH_n).sum(dim=-1)
    H = -1.0 - cos

    labels = batch["binding_flag"]
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    return {
        "zT": zT,
        "zPH": zPH,
        "cos": cos.cpu().numpy(),
        "H": H.cpu().numpy(),
        "labels": labels,
        "pair_ids": batch["pair_id"],
    }


@torch.no_grad()
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=60, keep_embeddings=False, peptide_pair_to_code=None, tcr_pair_to_code=None):
    tcr_proj.eval()
    pmhc_proj.eval()

    all_H, all_cos, all_lab, all_pid = [], [], [], []
    all_zT, all_zPH = [], []
    running_loss, n_steps = 0.0, 0
    stat_keys = [
        "H_std", "cos_std",
        "zT_dim_std_mean", "zPH_dim_std_mean",
        "eT_dim_std_mean", "ePH_dim_std_mean",
        "eT_dim_std_min", "ePH_dim_std_min",
        "eT_mean_resultant", "ePH_mean_resultant",
        "eT_L_std", "ePH_L_std", "eT_L_mean", "ePH_L_mean",
        "L_inv", "L_var_total", "L_cov_total",
        "L_peptide_group", "L_tcr_group",
        "L_angular_total", "L_angular_T", "L_angular_PH",
        "rho_angular_T_effective", "rho_angular_PH_effective",
        "L_peptide_angular", "L_tcr_angular",
        "group_var_loss", "group_var_n_groups", "group_var_std_mean", "group_var_std_min",
        "group_ang_loss", "group_ang_n_groups", "group_ang_std_mean", "group_ang_std_min",
        "pep_raw_loss", "pep_raw_n_groups", "pep_raw_std_mean", "pep_raw_std_min",
        "tcr_raw_loss", "tcr_raw_n_groups", "tcr_raw_std_mean", "tcr_raw_std_min",
        "pep_ang_loss", "pep_ang_n_groups", "pep_ang_std_mean", "pep_ang_std_min",
        "tcr_ang_loss", "tcr_ang_n_groups", "tcr_ang_std_mean", "tcr_ang_std_min",
    ]
    stat_acc = {k: [] for k in stat_keys}

    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device, eps=cfg["eps"])

        all_H.append(out["H"])
        all_cos.append(out["cos"])
        all_lab.append(out["labels"])
        all_pid.extend(list(out["pair_ids"]))
        if keep_embeddings:
            all_zT.append(out["zT"].cpu().numpy())
            all_zPH.append(out["zPH"].cpu().numpy())

        peptide_group_ids = None
        if peptide_pair_to_code is not None:
            peptide_group_ids = ids_for_batch(batch, peptide_pair_to_code, device)
        tcr_group_ids = None
        if tcr_pair_to_code is not None:
            tcr_group_ids = ids_for_batch(batch, tcr_pair_to_code, device)

        loss, stats = vicreg_hamiltonian_loss(
            out["zT"], out["zPH"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            delta=cfg["delta"],
            gamma_var=cfg["gamma_var"],
            eps=cfg["eps"],
            peptide_group_ids=peptide_group_ids,
            tcr_group_ids=tcr_group_ids,
            eta_peptide_group=cfg.get("eta_peptide_group", 0.0),
            eta_tcr_group=cfg.get("eta_tcr_group", 0.0),
            group_gamma=cfg.get("group_gamma", 0.05),
            tcr_group_gamma=cfg.get("tcr_group_gamma", None),
            group_min_size=cfg.get("group_min_size", 3),
            cov_scale=cfg.get("cov_scale", "d2"),
            rho_angular=cfg.get("rho_angular", 0.0),
            rho_angular_T=cfg.get("rho_angular_T", None),
            rho_angular_PH=cfg.get("rho_angular_PH", None),
            angular_gamma=cfg.get("angular_gamma", 0.035),
            angular_gamma_T=cfg.get("angular_gamma_T", None),
            angular_gamma_PH=cfg.get("angular_gamma_PH", None),
            angular_mean_gamma=cfg.get("angular_mean_gamma", 0.20),
            angular_mean_gamma_T=cfg.get("angular_mean_gamma_T", None),
            angular_mean_gamma_PH=cfg.get("angular_mean_gamma_PH", None),
            eta_peptide_angular=cfg.get("eta_peptide_angular", 0.0),
            eta_tcr_angular=cfg.get("eta_tcr_angular", 0.0),
            group_angular_gamma=cfg.get("group_angular_gamma", 0.02),
            tcr_group_angular_gamma=cfg.get("tcr_group_angular_gamma", None),
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
        "H_std": float(np.std(H_vals)),
        "cos_std": float(np.std(cos_vals)),
        "zT_dim_std_mean": float(np.mean(stat_acc["zT_dim_std_mean"])),
        "zPH_dim_std_mean": float(np.mean(stat_acc["zPH_dim_std_mean"])),
        "eT_dim_std_mean": float(np.mean(stat_acc["eT_dim_std_mean"])),
        "ePH_dim_std_mean": float(np.mean(stat_acc["ePH_dim_std_mean"])),
        "eT_dim_std_min": float(np.mean(stat_acc["eT_dim_std_min"])),
        "ePH_dim_std_min": float(np.mean(stat_acc["ePH_dim_std_min"])),
        "eT_mean_resultant": float(np.mean(stat_acc["eT_mean_resultant"])),
        "ePH_mean_resultant": float(np.mean(stat_acc["ePH_mean_resultant"])),
        "eT_L_std": float(np.mean(stat_acc["eT_L_std"])),
        "ePH_L_std": float(np.mean(stat_acc["ePH_L_std"])),
        "eT_L_mean": float(np.mean(stat_acc["eT_L_mean"])),
        "ePH_L_mean": float(np.mean(stat_acc["ePH_L_mean"])),
        "L_inv": float(np.mean(stat_acc["L_inv"])),
        "L_var_total": float(np.mean(stat_acc["L_var_total"])),
        "L_cov_total": float(np.mean(stat_acc["L_cov_total"])),
        "L_peptide_group": float(np.mean(stat_acc["L_peptide_group"])),
        "L_tcr_group": float(np.mean(stat_acc["L_tcr_group"])),
        "L_angular_total": float(np.mean(stat_acc["L_angular_total"])),
        "L_angular_T": float(np.mean(stat_acc["L_angular_T"])),
        "L_angular_PH": float(np.mean(stat_acc["L_angular_PH"])),
        "rho_angular_T_effective": float(np.mean(stat_acc["rho_angular_T_effective"])),
        "rho_angular_PH_effective": float(np.mean(stat_acc["rho_angular_PH_effective"])),
        "L_peptide_angular": float(np.mean(stat_acc["L_peptide_angular"])),
        "L_tcr_angular": float(np.mean(stat_acc["L_tcr_angular"])),
        "group_var_loss": float(np.mean(stat_acc["group_var_loss"])),
        "group_var_n_groups": float(np.mean(stat_acc["group_var_n_groups"])),
        "group_var_std_mean": float(np.mean(stat_acc["group_var_std_mean"])),
        "group_var_std_min": float(np.mean(stat_acc["group_var_std_min"])),
        "group_ang_loss": float(np.mean(stat_acc["group_ang_loss"])),
        "group_ang_n_groups": float(np.mean(stat_acc["group_ang_n_groups"])),
        "group_ang_std_mean": float(np.mean(stat_acc["group_ang_std_mean"])),
        "group_ang_std_min": float(np.mean(stat_acc["group_ang_std_min"])),
        "pep_raw_n_groups": float(np.mean(stat_acc["pep_raw_n_groups"])),
        "pep_raw_std_mean": float(np.mean(stat_acc["pep_raw_std_mean"])),
        "tcr_raw_n_groups": float(np.mean(stat_acc["tcr_raw_n_groups"])),
        "tcr_raw_std_mean": float(np.mean(stat_acc["tcr_raw_std_mean"])),
        "pep_ang_n_groups": float(np.mean(stat_acc["pep_ang_n_groups"])),
        "pep_ang_std_mean": float(np.mean(stat_acc["pep_ang_std_mean"])),
        "tcr_ang_n_groups": float(np.mean(stat_acc["tcr_ang_n_groups"])),
        "tcr_ang_std_mean": float(np.mean(stat_acc["tcr_ang_std_mean"])),
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

    if m["H_std"] < guard_cfg["h_std_min"]:
        reasons.append(f"H_std<{guard_cfg['h_std_min']}")
    if m["cos_std"] < guard_cfg["cos_std_min"]:
        reasons.append(f"cos_std<{guard_cfg['cos_std_min']}")
    if m["zT_dim_std_mean"] < guard_cfg["dim_std_mean_min"]:
        reasons.append(f"zT_dim_std_mean<{guard_cfg['dim_std_mean_min']}")
    if m["zPH_dim_std_mean"] < guard_cfg["dim_std_mean_min"]:
        reasons.append(f"zPH_dim_std_mean<{guard_cfg['dim_std_mean_min']}")

    # Angular collapse guard: the Hamiltonian sees unit-normalised embeddings.
    # Raw z variance can be healthy while e=z/||z|| has collapsed into a narrow cone.
    if "unit_dim_std_mean_min" in guard_cfg:
        if m.get("eT_dim_std_mean", 1.0) < guard_cfg["unit_dim_std_mean_min"]:
            reasons.append(f"eT_dim_std_mean<{guard_cfg['unit_dim_std_mean_min']}")
        if m.get("ePH_dim_std_mean", 1.0) < guard_cfg["unit_dim_std_mean_min"]:
            reasons.append(f"ePH_dim_std_mean<{guard_cfg['unit_dim_std_mean_min']}")
    if "unit_dim_std_min_min" in guard_cfg:
        if m.get("eT_dim_std_min", 1.0) < guard_cfg["unit_dim_std_min_min"]:
            reasons.append(f"eT_dim_std_min<{guard_cfg['unit_dim_std_min_min']}")
        if m.get("ePH_dim_std_min", 1.0) < guard_cfg["unit_dim_std_min_min"]:
            reasons.append(f"ePH_dim_std_min<{guard_cfg['unit_dim_std_min_min']}")

    if g["overlap"] > guard_cfg["overlap_max_for_collapse"] and m["auroc"] < guard_cfg["auroc_max_for_collapse"]:
        reasons.append("near-complete overlap + AUROC near chance")

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


def auroc_selection_score(val_out):
    """
    Selection objective for early biological-signal diagnostics.

    Primary objective:
        maximise validation AUROC.

    Tie-breakers:
        maximise validation AUPRC;
        minimise positive/negative overlap;
        minimise validation loss.

    This is intentionally not the training loss. The training objective is
    self-supervised and positive-only, whereas validation AUROC is the external
    ranking diagnostic used to decide whether the learned geometry separates
    positives from negatives.
    """
    m = val_out["metrics"]
    g = val_out["geometry"]

    return (
        float(m["auroc"]),
        float(m["auprc"]),
        -float(g["overlap"]),
        -float(m["val_loss"]),
    )


# ============================================================
# TRAINING
# ============================================================
def run_single(cfg, train_loader, val_loader, device, dirs, overlap_bins, guard_cfg, train_peptide_lookup=None, val_peptide_lookup=None, train_tcr_lookup=None, val_tcr_lookup=None):
    set_global_seed(cfg["seed"])

    tcr_proj, pmhc_proj = build_models(cfg, train_loader, device)
    optimizer = torch.optim.AdamW(
        list(tcr_proj.parameters()) + list(pmhc_proj.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    # best_val_loss = float("inf")
    # best_state = None
    # bad_epochs = 0
    best_selection_score = None
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

            peptide_group_ids = None
            if train_peptide_lookup is not None:
                peptide_group_ids = ids_for_batch(batch, train_peptide_lookup, device)
            tcr_group_ids = None
            if train_tcr_lookup is not None:
                tcr_group_ids = ids_for_batch(batch, train_tcr_lookup, device)

            loss, _ = vicreg_hamiltonian_loss(
                zT, zPH,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                delta=cfg["delta"],
                gamma_var=cfg["gamma_var"],
                eps=cfg["eps"],
                peptide_group_ids=peptide_group_ids,
                tcr_group_ids=tcr_group_ids,
                eta_peptide_group=cfg.get("eta_peptide_group", 0.0),
                eta_tcr_group=cfg.get("eta_tcr_group", 0.0),
                group_gamma=cfg.get("group_gamma", 0.05),
                tcr_group_gamma=cfg.get("tcr_group_gamma", None),
                group_min_size=cfg.get("group_min_size", 3),
                cov_scale=cfg.get("cov_scale", "d2"),
                rho_angular=cfg.get("rho_angular", 0.0),
                rho_angular_T=cfg.get("rho_angular_T", None),
                rho_angular_PH=cfg.get("rho_angular_PH", None),
                angular_gamma=cfg.get("angular_gamma", 0.035),
                angular_gamma_T=cfg.get("angular_gamma_T", None),
                angular_gamma_PH=cfg.get("angular_gamma_PH", None),
                angular_mean_gamma=cfg.get("angular_mean_gamma", 0.20),
                angular_mean_gamma_T=cfg.get("angular_mean_gamma_T", None),
                angular_mean_gamma_PH=cfg.get("angular_mean_gamma_PH", None),
                eta_peptide_angular=cfg.get("eta_peptide_angular", 0.0),
                eta_tcr_angular=cfg.get("eta_tcr_angular", 0.0),
                group_angular_gamma=cfg.get("group_angular_gamma", 0.02),
                tcr_group_angular_gamma=cfg.get("tcr_group_angular_gamma", None),
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
            overlap_bins=overlap_bins, keep_embeddings=False,
            peptide_pair_to_code=val_peptide_lookup,
            tcr_pair_to_code=val_tcr_lookup,
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
            f"zTstd={epoch_row['zT_dim_std_mean']:.4f} | zPHstd={epoch_row['zPH_dim_std_mean']:.4f} | "
            f"eTstd={epoch_row.get('eT_dim_std_mean', 0.0):.4f} | ePHstd={epoch_row.get('ePH_dim_std_mean', 0.0):.4f} | "
            f"pep_raw={epoch_row.get('pep_raw_std_mean', 0.0):.4f}/n{epoch_row.get('pep_raw_n_groups', 0.0):.0f} | "
            f"tcr_raw={epoch_row.get('tcr_raw_std_mean', 0.0):.4f}/n{epoch_row.get('tcr_raw_n_groups', 0.0):.0f} | "
            f"pep_ang={epoch_row.get('pep_ang_std_mean', 0.0):.4f}/n{epoch_row.get('pep_ang_n_groups', 0.0):.0f} | "
            f"tcr_ang={epoch_row.get('tcr_ang_std_mean', 0.0):.4f}/n{epoch_row.get('tcr_ang_n_groups', 0.0):.0f} | "
            f"L_angT={epoch_row.get('L_angular_T', 0.0):.4f} | L_angPH={epoch_row.get('L_angular_PH', 0.0):.4f} | "
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

        require_noncollapsed_best = cfg.get("require_noncollapsed_best", True)
        eligible_for_best = (not require_noncollapsed_best) or (not is_collapse)

        # if eligible_for_best and val_out["metrics"]["val_loss"] < best_val_loss:
        #     best_val_loss = val_out["metrics"]["val_loss"]
        #     best_state = copy.deepcopy(epoch_state)
        #     save_epoch_checkpoint(best_ckpt_path, best_state)
        #     bad_epochs = 0
        # else:
        #     if epoch + 1 >= cfg["min_epochs_before_early_stop"]:
        #         bad_epochs += 1

        current_selection_score = auroc_selection_score(val_out)

        if eligible_for_best and (
            best_selection_score is None or current_selection_score > best_selection_score
        ):
            best_selection_score = current_selection_score
            best_state = copy.deepcopy(epoch_state)
            save_epoch_checkpoint(best_ckpt_path, best_state)
            bad_epochs = 0

            log.info(
                f"[{cfg_stem}] new best checkpoint by VAL AUROC | "
                f"epoch={epoch+1} | "
                f"auroc={val_out['metrics']['auroc']:.4f} | "
                f"auprc={val_out['metrics']['auprc']:.4f} | "
                f"overlap={val_out['geometry']['overlap']:.4f} | "
                f"val_loss={val_out['metrics']['val_loss']:.4f}"
            )
        else:
            if epoch + 1 >= cfg["min_epochs_before_early_stop"]:
                bad_epochs += 1
        

        if collapse_bad_epochs >= guard_cfg["max_bad_epochs"]:
            log.info(f"[{cfg_stem}] stopped for repeated collapse evidence at epoch {epoch+1}")
            break
        # if bad_epochs >= cfg["patience"]:
        #     log.info(f"[{cfg_stem}] early stop on validation loss at epoch {epoch+1}")
        #     break
        if bad_epochs >= cfg["patience"]:
            log.info(f"[{cfg_stem}] early stop on validation AUROC selection score at epoch {epoch+1}")
            break
        
    if best_state is None:
        log.warning(f"[{cfg_stem}] no non-collapsed best checkpoint found; falling back to last checkpoint")
        best_state = epoch_state
        save_epoch_checkpoint(best_ckpt_path, best_state)

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


def final_evaluate(best_state, train_loader, val_loader, test_loader, device, overlap_bins, val_peptide_lookup=None, test_peptide_lookup=None, val_tcr_lookup=None, test_tcr_lookup=None):
    tcr_proj, pmhc_proj = load_best_models(best_state, train_loader, device)
    cfg = best_state["cfg"]

    val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=overlap_bins, keep_embeddings=True, peptide_pair_to_code=val_peptide_lookup, tcr_pair_to_code=val_tcr_lookup)
    val_thr = find_best_threshold_on_H(val_out["H"], val_out["labels"])

    test_out = evaluate_loader(test_loader, tcr_proj, pmhc_proj, device, cfg, overlap_bins=overlap_bins, keep_embeddings=True, peptide_pair_to_code=test_peptide_lookup, tcr_pair_to_code=test_tcr_lookup)
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

    pep_col = _find_column(df, ["peptide_id", "Peptide", "peptide", "pep", "pep_seq", "peptide_seq", "epitope", "antigen_peptide"])
    tcrb_col = _find_column(df, ["tcrb", "tcr_b", "cdr3b", "cdr3_beta", "tcr_beta", "trb_cdr3"])
    tcra_col = _find_column(df, ["tcra", "tcr_a", "cdr3a", "cdr3_alpha", "tcr_alpha", "tra_cdr3"])
    tcr_col = _find_column(df, ["tcr_id", "TCR_full", "tcr", "tcr_seq", "sequence_aa", "cdr3", "cdr3_sequence"])
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
            "require_noncollapsed_best": lw.get("require_noncollapsed_best", base_cfg.get("require_noncollapsed_best", True)),
            "patience": base_cfg["patience"],
            "min_epochs_before_early_stop": base_cfg["min_epochs_before_early_stop"],
            "rL": rL,
            "rD": rD,
            "lr": lr,
            "wd": wd,
            "dropout": dropout,
            "eta_peptide_group": lw.get("eta_peptide_group", base_cfg.get("eta_peptide_group", 0.0)),
            "eta_tcr_group": lw.get("eta_tcr_group", base_cfg.get("eta_tcr_group", 0.0)),
            "group_gamma": lw.get("group_gamma", base_cfg.get("group_gamma", 0.05)),
            "tcr_group_gamma": lw.get("tcr_group_gamma", base_cfg.get("tcr_group_gamma", None)),
            "group_min_size": lw.get("group_min_size", base_cfg.get("group_min_size", 3)),
            "eta_peptide_angular": lw.get("eta_peptide_angular", base_cfg.get("eta_peptide_angular", 0.0)),
            "eta_tcr_angular": lw.get("eta_tcr_angular", base_cfg.get("eta_tcr_angular", 0.0)),
            "group_angular_gamma": lw.get("group_angular_gamma", base_cfg.get("group_angular_gamma", 0.02)),
            "tcr_group_angular_gamma": lw.get("tcr_group_angular_gamma", base_cfg.get("tcr_group_angular_gamma", None)),
            "rho_angular": lw.get("rho_angular", base_cfg.get("rho_angular", 0.0)),
            "rho_angular_T": lw.get("rho_angular_T", base_cfg.get("rho_angular_T", None)),
            "rho_angular_PH": lw.get("rho_angular_PH", base_cfg.get("rho_angular_PH", None)),
            "angular_gamma": lw.get("angular_gamma", base_cfg.get("angular_gamma", 0.035)),
            "angular_gamma_T": lw.get("angular_gamma_T", base_cfg.get("angular_gamma_T", None)),
            "angular_gamma_PH": lw.get("angular_gamma_PH", base_cfg.get("angular_gamma_PH", None)),
            "angular_mean_gamma": lw.get("angular_mean_gamma", base_cfg.get("angular_mean_gamma", 0.20)),
            "angular_mean_gamma_T": lw.get("angular_mean_gamma_T", base_cfg.get("angular_mean_gamma_T", None)),
            "angular_mean_gamma_PH": lw.get("angular_mean_gamma_PH", base_cfg.get("angular_mean_gamma_PH", None)),
            "cov_scale": lw.get("cov_scale", base_cfg.get("cov_scale", "d2")),
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

    train_peptide_lookup, _ = build_group_lookup(base_cfg["train_csv"], "peptide_id")
    val_peptide_lookup, _ = build_group_lookup(base_cfg["val_csv"], "peptide_id")
    test_peptide_lookup, _ = build_group_lookup(base_cfg["test_csv"], "peptide_id")
    train_tcr_lookup, _ = build_group_lookup(base_cfg["train_csv"], "tcr_id")
    val_tcr_lookup, _ = build_group_lookup(base_cfg["val_csv"], "tcr_id")
    test_tcr_lookup, _ = build_group_lookup(base_cfg["test_csv"], "tcr_id")

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
        best_state = run_single(
            cfg, train_loader, val_loader, device, dirs, overlap_bins, guard_cfg,
            train_peptide_lookup=train_peptide_lookup,
            val_peptide_lookup=val_peptide_lookup,
            train_tcr_lookup=train_tcr_lookup,
            val_tcr_lookup=val_tcr_lookup,
        )
        final_eval = final_evaluate(
            best_state, train_loader, val_loader, test_loader, device, overlap_bins,
            val_peptide_lookup=val_peptide_lookup,
            test_peptide_lookup=test_peptide_lookup,
            val_tcr_lookup=val_tcr_lookup,
            test_tcr_lookup=test_tcr_lookup,
        )

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

        plot_h_histogram(
            final_eval["val_out"]["H"],
            final_eval["val_out"]["labels"],
            title=f"{stem} | val",
            save_path=fig_dir / f"{stem}__val_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )
        plot_h_histogram(
            final_eval["test_out"]["H"],
            final_eval["test_out"]["labels"],
            title=f"{stem} | test",
            save_path=fig_dir / f"{stem}__test_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )

        # Cross-reactivity diagnostics can create many files. By default, only
        # generate them for a config if the selected BEST checkpoint is not
        # collapsed according to the same guard used during training. This keeps
        # collapsed runs from producing low-value cross-reactivity plots.
        cr_cfg = base_cfg.get("cross_reactivity", {})
        only_if_noncollapsed = cr_cfg.get("only_plot_if_noncollapsed", True)
        plot_splits = cr_cfg.get("plot_splits", ["val"])  # set ["val", "test"] if wanted

        best_is_collapsed, best_collapse_reasons = collapse_flag(final_eval["val_out"], guard_cfg)
        cross_summary = {
            "best_checkpoint_collapsed_on_val": bool(best_is_collapsed),
            "collapse_reasons": best_collapse_reasons,
            "plot_splits_requested": plot_splits,
            "val": None,
            "test": None,
        }

        if only_if_noncollapsed and best_is_collapsed:
            log.info(
                f"[{stem}] skipping cross-reactivity plots because best checkpoint is collapsed: "
                f"{best_collapse_reasons}"
            )
        else:
            if "val" in plot_splits:
                cross_summary["val"] = run_cross_reactivity_analysis(
                    final_eval["val_out"],
                    f"{stem}__val",
                    base_cfg["val_csv"],
                    base_cfg["train_csv"],
                    fig_dir,
                    save_dir,
                    base_cfg,
                )
            if "test" in plot_splits:
                cross_summary["test"] = run_cross_reactivity_analysis(
                    final_eval["test_out"],
                    f"{stem}__test",
                    base_cfg["test_csv"],
                    base_cfg["train_csv"],
                    fig_dir,
                    save_dir,
                    base_cfg,
                )

        with open(save_dir / f"{stem}__cross_reactivity_summary.json", "w") as f:
            json.dump(cross_summary, f, indent=2)
        row = {
            "cfg_name": stem,
            "best_epoch": best_state["epoch"],
            "val_loss": final_eval["val_out"]["metrics"]["val_loss"],
            "val_auroc": final_eval["val_out"]["metrics"]["auroc"],
            "val_auprc": final_eval["val_out"]["metrics"]["auprc"],
            "val_mean_gap": final_eval["val_out"]["geometry"]["mean_gap"],
            "val_effect_size": final_eval["val_out"]["geometry"]["effect_size"],
            "val_overlap": final_eval["val_out"]["geometry"]["overlap"],
            "val_zT_dim_std_mean": final_eval["val_out"]["metrics"]["zT_dim_std_mean"],
            "val_zPH_dim_std_mean": final_eval["val_out"]["metrics"]["zPH_dim_std_mean"],
            "val_eT_dim_std_mean": final_eval["val_out"]["metrics"].get("eT_dim_std_mean", 0.0),
            "val_ePH_dim_std_mean": final_eval["val_out"]["metrics"].get("ePH_dim_std_mean", 0.0),
            "val_eT_dim_std_min": final_eval["val_out"]["metrics"].get("eT_dim_std_min", 0.0),
            "val_ePH_dim_std_min": final_eval["val_out"]["metrics"].get("ePH_dim_std_min", 0.0),
            "val_eT_mean_resultant": final_eval["val_out"]["metrics"].get("eT_mean_resultant", 0.0),
            "val_ePH_mean_resultant": final_eval["val_out"]["metrics"].get("ePH_mean_resultant", 0.0),
            "val_pep_raw_std_mean": final_eval["val_out"]["metrics"].get("pep_raw_std_mean", 0.0),
            "val_tcr_raw_std_mean": final_eval["val_out"]["metrics"].get("tcr_raw_std_mean", 0.0),
            "val_pep_ang_std_mean": final_eval["val_out"]["metrics"].get("pep_ang_std_mean", 0.0),
            "val_tcr_ang_std_mean": final_eval["val_out"]["metrics"].get("tcr_ang_std_mean", 0.0),
            "val_group_var_std_mean": final_eval["val_out"]["metrics"].get("group_var_std_mean", 0.0),
            "val_group_ang_std_mean": final_eval["val_out"]["metrics"].get("group_ang_std_mean", 0.0),
            "val_L_peptide_group": final_eval["val_out"]["metrics"].get("L_peptide_group", 0.0),
            "val_L_tcr_group": final_eval["val_out"]["metrics"].get("L_tcr_group", 0.0),
            "val_L_angular_total": final_eval["val_out"]["metrics"].get("L_angular_total", 0.0),
            "val_L_angular_T": final_eval["val_out"]["metrics"].get("L_angular_T", 0.0),
            "val_L_angular_PH": final_eval["val_out"]["metrics"].get("L_angular_PH", 0.0),
            "val_L_peptide_angular": final_eval["val_out"]["metrics"].get("L_peptide_angular", 0.0),
            "val_L_tcr_angular": final_eval["val_out"]["metrics"].get("L_tcr_angular", 0.0),
            "test_auroc": final_eval["test_metrics"]["auroc"],
            "test_auprc": final_eval["test_metrics"]["auprc"],
            "test_f1": final_eval["test_metrics"]["f1"],
            "test_mean_gap": final_eval["test_out"]["geometry"]["mean_gap"],
            "test_effect_size": final_eval["test_out"]["geometry"]["effect_size"],
            "test_overlap": final_eval["test_out"]["geometry"]["overlap"],
            "alpha": cfg["alpha"],
            "beta": cfg["beta"],
            "delta": cfg["delta"],
            "cov_scale": cfg.get("cov_scale", "d2"),
            "eta_peptide_group": cfg.get("eta_peptide_group", 0.0),
            "eta_tcr_group": cfg.get("eta_tcr_group", 0.0),
            "group_gamma": cfg.get("group_gamma", 0.05),
            "tcr_group_gamma": cfg.get("tcr_group_gamma", None),
            "eta_peptide_angular": cfg.get("eta_peptide_angular", 0.0),
            "eta_tcr_angular": cfg.get("eta_tcr_angular", 0.0),
            "group_angular_gamma": cfg.get("group_angular_gamma", 0.02),
            "tcr_group_angular_gamma": cfg.get("tcr_group_angular_gamma", None),
            "rho_angular": cfg.get("rho_angular", 0.0),
            "rho_angular_T": cfg.get("rho_angular_T", None),
            "rho_angular_PH": cfg.get("rho_angular_PH", None),
            "angular_gamma": cfg.get("angular_gamma", 0.035),
            "angular_gamma_T": cfg.get("angular_gamma_T", None),
            "angular_gamma_PH": cfg.get("angular_gamma_PH", None),
            "angular_mean_gamma": cfg.get("angular_mean_gamma", 0.20),
            "group_min_size": cfg.get("group_min_size", 3),
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
        by=["val_auroc", "val_auprc", "val_loss", "test_auroc"],
        ascending=[False, False, True, False],
    )
    summary_df.to_csv(save_dir / "full_summary.csv", index=False)

    best_name = summary_df.iloc[0]["cfg_name"]
    best_bundle = next(x for x in all_best_states if x["cfg_name"] == best_name)
    best_state = best_bundle["best_state"]
    final_eval = best_bundle["final_eval"]

    log.info("=" * 100)
    log.info(f"GLOBAL BEST BY VAL AUROC: {best_name}")
    log.info(summary_df.head(10).to_string(index=False))

    # Cross-reactivity summaries have been generated per config. Plot creation is
    # controlled by cross_reactivity.only_plot_if_noncollapsed and
    # cross_reactivity.plot_splits.

    manifest = {
        "run_name": run_name,
        "config_path": str(config_path),
        "seed": base_cfg["seed"],
        "n_configs": len(configs),
        "best_cfg_name": best_name,
        "cross_reactivity": "generated_per_config; see <cfg_name>__cross_reactivity_summary.json plus matching CSV/PNG files",
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
        default="/home/natasha/multimodal_model/scripts/train/hpo_seed31_branch_angular_groups_config.json",
        help="Path to JSON config file",
    )
    args = parser.parse_args()
    main(args.config)
