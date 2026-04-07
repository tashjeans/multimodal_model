#!/usr/bin/env python3
"""
HPO for full Hamiltonian model with signal-aware Boltz Z*
Dual AUROC tracking: cosine (eT·ePH) and Hamiltonian (-H).
Best AUROC from either metric is used for early stopping and selection.

Run:
    tmux new -s hpo_boltz
    conda activate tcr-multimodal
    cd /home/natasha/multimodal_model/scripts/train
    python hpo_full_hamiltonian_boltz.py 2>&1 | tee hpo_full_hamiltonian_boltz.log
"""

import os, sys, gc, copy, time, math, random, json, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, precision_score, recall_score, confusion_matrix,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================
# LOGGING
# ============================================================
LOG_FILE = "hpo_full_hamiltonian_boltz.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, mode="w")],
)
log = logging.getLogger(__name__)

# ============================================================
# PATHS
# ============================================================
PROJECT = Path("/home/natasha/multimodal_model")
EMBED_ROOT = PROJECT / "models/embeddings/no_boltz"
CHECKPOINTS_DIR = PROJECT / "models/checkpoints"
FIGURE_DIR = PROJECT / "models/figures"

TRAIN_CSV = str(PROJECT / "data/train/train_df_clean.csv")
VAL_CSV   = str(PROJECT / "data/val/val_df_clean_pos_neg.csv")
TEST_CSV  = str(PROJECT / "data/test/test_df_clean_pos_neg.csv")

BOLTZ_LATENT_DIR = PROJECT / "models/embeddings/boltz_signal_bottleneck_latents"
ZSTAR_D = 128
ZSTAR_RANK = 8

FIGURE_SUBDIR = FIGURE_DIR / "hpo_full_hamiltonian_boltz"
FIGURE_SUBDIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = CHECKPOINTS_DIR / "hpo_full_hamiltonian_boltz"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HPO CONFIG
# ============================================================

SEARCH_SPACE = [
    {"rL": 8,  "rD": 16, "lr": 1e-4, "wd": 1e-2, "alpha": 1.0,  "beta": 25.0},
    {"rL": 8,  "rD": 16, "lr": 3e-4, "wd": 1e-2, "alpha": 1.0,  "beta": 25.0},
    {"rL": 4,  "rD": 8,  "lr": 1e-4, "wd": 1e-2, "alpha": 1.0,  "beta": 25.0},
    {"rL": 8,  "rD": 16, "lr": 1e-4, "wd": 1e-3, "alpha": 1.0,  "beta": 10.0},
]
NUM_EPOCHS = 30
PATIENCE = 10
D = 128
R_PH = 0.7


# ============================================================
# PEPTIDE LOOKUPS for cross-reactivity analysis
# ============================================================
def build_pep_lookup(csv_path):
    df = pd.read_csv(csv_path)
    return {str(r.get("pair_id", r.name)): str(r["Peptide"]) for _, r in df.iterrows()}

pep_lookup_train = build_pep_lookup(TRAIN_CSV)
pep_lookup_val = build_pep_lookup(VAL_CSV)
pep_lookup_test = build_pep_lookup(TEST_CSV)
log.info(f"Peptide lookups: train={len(pep_lookup_train)}, val={len(pep_lookup_val)}, test={len(pep_lookup_test)}")

# Projected Encoder to get z_T and z_pMHC
# Bilinear factored compression: z = vec(A^T X B) @ H
# Then MLP expander for nonlinear capacity (VICReg-style)
#
# Architecture:
#   1. Channel compression: (B, L, D) @ B_c -> (B, L, rD)
#   2. Positional compression: A_c^T @ Y -> (B, rL, rD)
#   3. Flatten + linear map: (rL*rD) @ H_c -> (B, d)
#   4. MLP expander: Linear(d,d) -> BN -> ReLU -> Linear(d,d)
#
# The MLP expander adds the nonlinear capacity that VICReg needs
# to avoid representational collapse. Without it, the entire
# ESM->latent pathway is a single linear map (composition of
# three linear projections), which cannot learn the nonlinear
# decision boundaries needed for binding discrimination.

eps = 1e-8

class ESMProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_max):
        """
        D    : ESM embedding dim (e.g. 960)
        rL   : positional rank
        rD   : channel rank
        d    : latent dim
        L_max: max true length for this modality in the batch
        """
        super().__init__()
        self.D   = D
        self.rL  = rL
        self.rD  = rD
        self.d   = d
        self.L_max = L_max

        # Channel mixing: D -> rD
        self.B_c = nn.Parameter(torch.empty(D, rD))
        nn.init.xavier_uniform_(self.B_c)

        # Positional mixing: positions 0..L_max-1 -> rL
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        nn.init.xavier_uniform_(self.A_c)

        # Final map: (rL * rD) -> d
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))
        nn.init.xavier_uniform_(self.H_c)

        # MLP expander — adds nonlinear capacity
        # BN is critical here: it standardises each dimension across
        # the batch, achieving a similar effect to VICReg's variance
        # regulariser but inside the network, helping gradients flow.
     

        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d, d),
        )

    def forward(self, emb, mask):
        """
        emb  : (B, L_pad, D) token embeddings
        mask : (B, L_pad)   1 = real token, 0 = pad
        returns z : (B, d)
        """
        device = emb.device
        B, L_pad, D = emb.shape
        assert D == self.D

        # Compute true lengths
        L_true = mask.sum(dim=1)            # (B,)
        z_list = []

        for b in range(B):
            Lb = int(L_true[b].item())
            if Lb == 0:
                z_b = torch.zeros(self.d, device=device)
                z_list.append(z_b)
                continue

            Xb = emb[b, :Lb, :]                      # (Lb, D)
            mb = mask[b, :Lb].unsqueeze(-1).float()  # (Lb, 1)
            Xb = Xb * mb                             # (Lb, D)

            # 1) Channel compression: D -> rD
            Yb = Xb @ self.B_c                       # (Lb, rD)

            # 2) Positional compression: Lb -> rL
            A_pos = self.A_c[:Lb, :]                 # (Lb, rL)
            Ub = A_pos.T @ Yb                        # (rL, rD)

            # 3) Flatten and map to latent d
            Ub_flat = Ub.reshape(-1)                 # (rL * rD,)
            z_b = Ub_flat @ self.H_c                 # (d,)

            z_list.append(z_b)

        z = torch.stack(z_list, dim=0)               # (B, d)

        # 4) MLP expander — nonlinear transformation
        z = self.expander(z)                         # (B, d)

        return z


# Projection Head for pMHC: separate peptide and HLA encoders
# Concatenates peptide (70% of d) and HLA (30% of d) representations
# No normalisation — unconstrained embeddings for VICReg MSE

class PMHCProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_P_max, L_H_max, R_PH=0.7):
        """
        D      : ESM embedding dim
        rL     : positional rank
        rD     : channel rank
        d      : total latent dim for pMHC
        L_P_max: max true peptide length
        L_H_max: max true HLA length
        R_PH   : fraction of d reserved for peptide (e.g. 0.7)
        """
        super().__init__()
        self.D    = D
        self.rL   = rL
        self.rD   = rD
        self.d    = d
        self.R_PH = R_PH

        # Split d into peptide and HLA sub-dims
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        assert d_P > 0 and d_H > 0, "Choose d and R_PH so both > 0"

        self.d_P = d_P
        self.d_H = d_H

        # Separate Projection Heads (each includes its own MLP expander)
        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max)

    def forward(self, emb_P, mask_P, emb_H, mask_H):
        """
        returns zPH: (B, d) with first d_P dims peptide, last d_H dims HLA
        """
        zP = self.pep_encoder(emb_P, mask_P)  # (B, d_P)
        zH = self.hla_encoder(emb_H, mask_H)  # (B, d_H)

        # Concatenate: 70% dims peptide, 30% dims HLA
        # No normalisation — VICReg variance/covariance handle scale
        zPH = torch.cat([zP, zH], dim=-1)     # (B, d)

        return zPH


# regularisers

def vicreg_variance(u, gamma=1.0, eps=1e-4):
    """
    u: (B, d) embeddings for one modality (TCR or pMHC)
    gamma: minimum desired std per dimension (VICReg uses gamma=1.0)
    """
    B, d = u.shape
    u_centered = u - u.mean(dim=0, keepdim=True)            # (B, d), make mean 0
    #std = torch.sqrt(u_centered.var(dim=0) + eps)           # (d,)
    std = torch.sqrt(u_centered.var(dim=0, unbiased=False) + eps)

    # (1/d) * sum_j ReLU(gamma - std_j)
    var_loss = F.relu(gamma - std).mean()
    return var_loss


def vicreg_covariance(u, eps=1e-4):
    """
    u: (B, d) embeddings for one modality (TCR or pMHC)
    Returns (1/d^2) * sum_{j!=k} Cov(u_j, u_k)^2
    """
    B, d = u.shape
    u_centered = u - u.mean(dim=0, keepdim=True)            # (B, d)

    # covariance matrix C = (u^T u) / (B-1)
    cov = (u_centered.T @ u_centered) / (B - 1)             # (d, d)

    # zero diag, keep off-diagonals, as we don't want diagonal terms (variances)
    diag = torch.diag(cov)
    cov_off = cov - torch.diag_embed(diag)

    cov_loss = (cov_off ** 2).sum() / (d * d)
    return cov_loss



class ShardedBatchTripletDataset(Dataset):
    def __init__(self, shards_dir):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        assert self.shard_paths, f"No shard_*.pt files found in {self.shards_dir}"

        self.index = []
        self._lens = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            self._lens.append(len(shard))
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


# threshold finder 

def find_best_threshold(scores, labels):
    thresholds = np.unique(scores)
    best = None

    for thr in thresholds:
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)

        if best is None or f1 > best["f1"]:
            best = {
                "threshold": float(thr),
                "f1": float(f1),
                "accuracy": float(accuracy_score(labels, preds)),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
            }

    return best

# ============================================================
# PLOTTING
# ============================================================

def plot_H_histogram(H_vals, labels, title, save_dir):
    pos, neg = H_vals[labels == 1], H_vals[labels == 0]
    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="negative")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="positive")
    plt.xlabel("Hamiltonian H (lower = stronger binding)")
    plt.ylabel("density"); plt.title(title); plt.legend(); plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png"); plt.close()

def plot_cosine_histogram(cos_vals, labels, title, save_dir):
    pos, neg = cos_vals[labels == 1], cos_vals[labels == 0]
    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="negative")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="positive")
    plt.xlabel("eT · ePH cosine similarity")
    plt.ylabel("density"); plt.title(title); plt.legend(); plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png"); plt.close()

def plot_cross_reactivity(cos_vals, pair_ids, labels, pep_lookup, title, save_dir):
    pos_mask = labels == 1
    pos_cos = cos_vals[pos_mask]
    pos_pids = [pid for pid, lab in zip(pair_ids, labels) if lab == 1]
    pos_dist = 1.0 - pos_cos
    pep_groups = defaultdict(list)
    for i, pid in enumerate(pos_pids):
        pep = pep_lookup.get(str(pid))
        if pep is not None:
            pep_groups[pep].append(pos_dist[i])
    same_pep = [d for pep, dists in pep_groups.items() if len(dists) >= 2 for d in dists]
    if len(same_pep) == 0:
        log.info(f"  [cross-react] No peptide groups >=2 TCRs, skipping")
        return
    plt.figure(figsize=(6, 5))
    plt.boxplot([same_pep, pos_dist.tolist()], labels=["Same peptide", "Random"])
    plt.ylabel("Cosine distance in TCR latent space"); plt.title(title); plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png"); plt.close()
    log.info(f"  [cross-react] same_pep n={len(same_pep)} med={np.median(same_pep):.4f} | "
             f"random n={len(pos_dist)} med={np.median(pos_dist):.4f}")

def plot_training_history(history, save_dir, prefix=""):
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.title(f"{prefix} Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_loss.png"); plt.close()
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_auroc"], label="val AUROC")
    plt.plot(epochs, history["val_auprc"], label="val AUPRC")
    plt.xlabel("epoch"); plt.ylabel("metric"); plt.title(f"{prefix} Metrics")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_metrics.png"); plt.close()

def plot_epoch_diagnostics(val_out, epoch, pep_lookup, run_tag, save_dir):
    labels = val_out["labels"]
    ep_dir = save_dir / run_tag
    ep_dir.mkdir(parents=True, exist_ok=True)
    plot_H_histogram(val_out["H"], labels, f"{run_tag}_val_H_ep{epoch}", ep_dir)
    plot_cosine_histogram(val_out["cos"], labels, f"{run_tag}_val_cos_ep{epoch}", ep_dir)
    plot_cross_reactivity(val_out["cos"], val_out["pair_ids"], labels, pep_lookup,
                          f"{run_tag}_val_xreact_ep{epoch}", ep_dir)

# ============================================================
# Z* construction from Signal-Aware Boltz latent — NO LEARNABLE PARAMETERS
# These are pure tensor operations matching SignalAwareOperatorBottleneck.
# The latent was learned under the new preservation-plus-separation objective
# The latent theta encodes:
#   3 diagonal vectors (A_diag, B_diag, C_diag) of size d  -> 3*d
#   3 blocks × 2 factors (U, V) of size d×r                -> 6*d*r
#   Total: 3*d + 6*d*r = 3*128 + 6*128*8 = 6528
# Each block K = diag(diag_vec) + U @ V^T / sqrt(r), giving (d, d) matrices.
# Z*_residual = [[A, C], [C^T, B]]
# Z*_full = [[I+A, I+C], [I+C^T, I+B]]
# ============================================================

import math

def _split_theta(theta, d=128, r=8):
    """
    Split theta (B, 3*d + 6*d*r) into diagonal vectors and low-rank factors.
    Matches SignalAwareOperatorBottleneck._split_theta exactly.
    """
    B = theta.shape[0]
    offset = 0
    A_diag = theta[:, offset:offset + d]; offset += d
    B_diag = theta[:, offset:offset + d]; offset += d
    C_diag = theta[:, offset:offset + d]; offset += d

    def take_mat():
        nonlocal offset
        U = theta[:, offset:offset + d * r].view(B, d, r)
        offset += d * r
        V = theta[:, offset:offset + d * r].view(B, d, r)
        offset += d * r
        return U, V

    A_U, A_V = take_mat()
    B_U, B_V = take_mat()
    C_U, C_V = take_mat()

    assert offset == theta.shape[1], f"Theta split mismatch: offset={offset}, total={theta.shape[1]}"
    return A_diag, B_diag, C_diag, A_U, A_V, B_U, B_V, C_U, C_V


def operator_latent_to_zstar_residual(theta, d=128, r=8):
    """
    Convert signal-aware operator parameters (B, 6528) to residual Z* (B, 2d, 2d).
    No learnable parameters — pure deterministic tensor math.
    
    theta: (B, 3*d + 6*d*r) = (B, 6528)
    Returns: zstar_residual (B, 2d, 2d)
    """
    B = theta.shape[0]
    device = theta.device
    dtype = theta.dtype
    
    A_diag, B_diag, C_diag, A_U, A_V, B_U, B_V, C_U, C_V = _split_theta(theta, d=d, r=r)
    
    # Low-rank products
    A    = torch.matmul(A_U, A_V.transpose(-1, -2)) / math.sqrt(r)  # (B, d, d)
    Bblk = torch.matmul(B_U, B_V.transpose(-1, -2)) / math.sqrt(r)  # (B, d, d)
    C    = torch.matmul(C_U, C_V.transpose(-1, -2)) / math.sqrt(r)  # (B, d, d)
    
    # Symmetrise intra-chain blocks
    A    = 0.5 * (A + A.transpose(1, 2))
    Bblk = 0.5 * (Bblk + Bblk.transpose(1, 2))
    
    # Add diagonal terms
    eye = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)  # (1, d, d)
    A    = A    + eye * A_diag.unsqueeze(-1)
    Bblk = Bblk + eye * B_diag.unsqueeze(-1)
    C    = C    + eye * C_diag.unsqueeze(-1)
    
    # Assemble residual Z*
    zstar_residual = torch.zeros(B, 2*d, 2*d, device=device, dtype=dtype)
    zstar_residual[:, :d, :d] = A
    zstar_residual[:, :d, d:] = C
    zstar_residual[:, d:, :d] = C.transpose(1, 2)
    zstar_residual[:, d:, d:] = Bblk
    zstar_residual = 0.5 * (zstar_residual + zstar_residual.transpose(1, 2))
    return zstar_residual


def add_identity_to_zstar(zstar_residual, d=128):
    """
    Add identity to all four blocks of Z*:
        [[I+A, I+C], [I+C^T, I+B]]
    This ensures a baseline cosine alignment term even when structural signal is weak.
    """
    B = zstar_residual.shape[0]
    device = zstar_residual.device
    dtype = zstar_residual.dtype
    
    I = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
    zstar = zstar_residual.clone()
    zstar[:, :d, :d] += I
    zstar[:, :d, d:] += I
    zstar[:, d:, :d] += I
    zstar[:, d:, d:] += I
    zstar = 0.5 * (zstar + zstar.transpose(1, 2))
    return zstar


def operator_latent_to_full_zstar(operator_params, d=128, r=8):
    """Convenience: latent -> residual -> add identity -> full Z*."""
    residual = operator_latent_to_zstar_residual(operator_params, d=d, r=r)
    return add_identity_to_zstar(residual, d=d)


# ============================================================
# Full Hamiltonian loss with per-sample Z* from Boltz
# VICReg var/cov on z_raw (unconstrained), Hamiltonian on ê with Z*
# ============================================================

def vicreg_hamiltonian_loss(
    zT_raw,
    zPH_raw,
    Zstar,
    alpha=1.0,
    beta=25.0,
    delta=1.0,
    gamma_var=1.0,
    eps=1e-4,
):
    """
    Full Hamiltonian loss with per-sample Z* from Boltz autoencoder.
    
    VICReg regularisers on z_raw (unconstrained).
    Hamiltonian H = -½ ê^T Z* ê on normalised ê with frozen per-sample Z*.
    """
    B, d = zT_raw.shape

    # ---- 1) L2 normalise for Hamiltonian ----
    eT  = zT_raw / (zT_raw.norm(dim=-1, keepdim=True) + eps)
    ePH = zPH_raw / (zPH_raw.norm(dim=-1, keepdim=True) + eps)
    e_hat = torch.cat([eT, ePH], dim=-1)  # (B, 2d)


    # ---- 2) Hamiltonian with per-sample Z* ----
    quad = torch.einsum("bi,bij,bj->b", e_hat, Zstar, e_hat)  # (B,)
    H = -0.5 * quad
    L_inv = H.mean()


    # ---- 3) VICReg var/cov on UNCONSTRAINED z_raw ----
    L_var_T  = vicreg_variance(zT_raw,  gamma=gamma_var, eps=eps)
    L_var_PH = vicreg_variance(zPH_raw, gamma=gamma_var, eps=eps)
    L_cov_T  = vicreg_covariance(zT_raw,  eps=eps)
    L_cov_PH = vicreg_covariance(zPH_raw, eps=eps)

    L_var_total = L_var_T + L_var_PH
    L_cov_total = L_cov_T + L_cov_PH

    # ---- 4) Total ----
    L_total = alpha * L_inv + beta * L_var_total + delta * L_cov_total

    # ---- 5) Diagnostics ----
    cos = (eT * ePH).sum(dim=-1)
    H_cos = -1.0 * cos
    components = {
        "L_total": L_total.item(), "L_inv_H": L_inv.item(),
        "L_var_T": L_var_T.item(), "L_var_PH": L_var_PH.item(),
        "L_cov_T": L_cov_T.item(), "L_cov_PH": L_cov_PH.item(),
        "alpha_L_inv": (alpha * L_inv).item(),
        "beta_var": (beta * L_var_total).item(),
        "delta_cov": (delta * L_cov_total).item(),
        "cos_mean": H_cos.mean().item(),
        "H_mean": H.mean().item(), "H_min": H.min().item(), "H_max": H.max().item(),
        "quad_mean": quad.mean().item(),
        "zT_norm_mean": zT_raw.norm(dim=-1).mean().item(),
        "zPH_norm_mean": zPH_raw.norm(dim=-1).mean().item(),
    }
    return L_total, components


# ============================================================
# Load Boltz operator latents and create paired ESM+Boltz dataset
# ============================================================

class OperatorLatentStore:
    """Loads exported operator latents and provides pair_id -> latent lookup."""
    def __init__(self, latent_dir, split_name):
        latent_path = os.path.join(latent_dir, f"{split_name}_latents.npz")
        meta_path = os.path.join(latent_dir, f"{split_name}_metadata.csv")
        
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Operator latents not found: {latent_path}")
        
        self.latents = np.load(latent_path)["latent"]  # (N, operator_param_dim)
        self.meta = pd.read_csv(meta_path)
        
        self.pid_to_row = {}
        for _, row in self.meta.iterrows():
            self.pid_to_row[str(row["pair_id"])] = int(row["latent_row"])
        
        print(f"[OperatorLatentStore:{split_name}] {len(self.pid_to_row)} pairs, "
              f"latent shape: {self.latents.shape}")
    
    def get(self, pair_id):
        row = self.pid_to_row.get(str(pair_id))
        if row is None:
            return None
        return self.latents[row]
    
    def has(self, pair_id):
        return str(pair_id) in self.pid_to_row


class PairedESMBoltzDataset(Dataset):
    """Inner-joins ESM shards with Boltz operator latents on pair_id."""
    def __init__(self, esm_dataset, latent_store):
        self.esm_dataset = esm_dataset
        self.latent_store = latent_store
        
        self.matched_indices = []
        skipped = 0
        
        for idx in range(len(esm_dataset)):
            sample = esm_dataset[idx]
            pair_ids = sample["pair_id"]
            
            if isinstance(pair_ids, list):
                if all(latent_store.has(pid) for pid in pair_ids):
                    self.matched_indices.append(idx)
                else:
                    skipped += 1
            else:
                if latent_store.has(pair_ids):
                    self.matched_indices.append(idx)
                else:
                    skipped += 1
        
        print(f"[PairedDataset] matched: {len(self.matched_indices)}, "
              f"skipped (no Boltz): {skipped}")
    
    def __len__(self):
        return len(self.matched_indices)
    
    def __getitem__(self, idx):
        esm_idx = self.matched_indices[idx]
        sample = self.esm_dataset[esm_idx]
        
        pair_ids = sample["pair_id"]
        if isinstance(pair_ids, list):
            latents = [torch.tensor(self.latent_store.get(pid), dtype=torch.float32) for pid in pair_ids]
            sample["operator_latent"] = torch.stack(latents, dim=0)
        else:
            latent = self.latent_store.get(pair_ids)
            sample["operator_latent"] = torch.tensor(latent, dtype=torch.float32).unsqueeze(0)
        
        return sample




# ============================================================
# FORWARD + EVALUATE — full Hamiltonian (dual AUROC: cosine + H)
# ============================================================

@torch.no_grad()
def forward_batch(batch, tcr_proj, pmhc_proj, device, eps=1e-8):
    eT = batch["emb_T"].to(device); mT = batch["mask_T"].to(device)
    eP = batch["emb_P"].to(device); mP = batch["mask_P"].to(device)
    eH = batch["emb_H"].to(device); mH = batch["mask_H"].to(device)
    op_lat = batch["operator_latent"].to(device)
    zT = tcr_proj(eT, mT); zPH = pmhc_proj(eP, mP, eH, mH)
    Zstar = operator_latent_to_full_zstar(op_lat, d=ZSTAR_D, r=ZSTAR_RANK)
    eT_n = zT/(zT.norm(dim=-1,keepdim=True)+eps); ePH_n = zPH/(zPH.norm(dim=-1,keepdim=True)+eps)
    e_hat = torch.cat([eT_n, ePH_n], dim=-1)
    quad = torch.einsum("bi,bij,bj->b", e_hat, Zstar, e_hat)
    H = -0.5 * quad
    cos = (eT_n*ePH_n).sum(dim=-1)
    labels = batch["binding_flag"]
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    return {"zT": zT, "zPH": zPH, "Zstar": Zstar, "cos": cos.cpu().numpy(),
            "H": H.cpu().numpy(), "score_cos": cos.cpu().numpy(), "score_H": (-H).cpu().numpy(),
            "labels": labels, "pair_ids": batch["pair_id"]}


@torch.no_grad()
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, alpha=1.0, beta=25.0, delta=1.0, gamma_var=1.0, eps=1e-8):
    tcr_proj.eval(); pmhc_proj.eval()
    all_scos, all_sH, all_H, all_cos, all_lab, all_pid = [], [], [], [], [], []
    rl, ns = 0.0, 0
    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device, eps)
        all_scos.append(out["score_cos"]); all_sH.append(out["score_H"])
        all_H.append(out["H"]); all_cos.append(out["cos"])
        all_lab.append(out["labels"]); all_pid.extend(out["pair_ids"])
        loss, _ = vicreg_hamiltonian_loss(out["zT"], out["zPH"], out["Zstar"],
                    alpha=alpha, beta=beta, delta=delta, gamma_var=gamma_var)
        rl += loss.item(); ns += 1
    scos = np.concatenate(all_scos); sH = np.concatenate(all_sH)
    H_vals = np.concatenate(all_H); cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_lab).astype(int)
    auroc_cos = roc_auc_score(labels, scos); auroc_H = roc_auc_score(labels, sH)
    auprc_cos = average_precision_score(labels, scos); auprc_H = average_precision_score(labels, sH)
    if auroc_cos >= auroc_H:
        scores, auroc, auprc, stype = scos, auroc_cos, auprc_cos, "cosine"
    else:
        scores, auroc, auprc, stype = sH, auroc_H, auprc_H, "hamiltonian"
    thr = find_best_threshold(scores, labels)
    return {"scores": scores, "scores_cos": scos, "scores_H": sH,
            "H": H_vals, "cos": cos_vals, "labels": labels, "pair_ids": all_pid,
            "metrics": {"auroc": auroc, "auroc_cos": auroc_cos, "auroc_H": auroc_H,
                        "auprc": auprc, "auprc_cos": auprc_cos, "auprc_H": auprc_H,
                        "val_loss": rl/max(ns,1), "score_type": stype, **thr}}

# ============================================================
# TRAINING — full Hamiltonian with frozen Boltz Z*
# ============================================================

def run_experiment(train_loader, val_loader, device, pep_lookup_val,
                   rL=8, rD=16, lr=1e-4, weight_decay=1e-2,
                   alpha=1.0, beta=25.0, delta=1.0, gamma_var=1.0,
                   run_tag="cfg0"):

    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]; L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]; D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(D_esm, rL, rD, D, L_max=L_T).to(device)
    pmhc_proj = PMHCProjectionHead(D_esm, rL, rD, D, L_P_max=L_P, L_H_max=L_H, R_PH=R_PH).to(device)

    optimizer = torch.optim.AdamW([
        {"params": tcr_proj.parameters(), "lr": lr},
        {"params": pmhc_proj.parameters(), "lr": lr},
    ], weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "val_loss": [], "val_auroc": [], "val_auprc": [], "val_f1": [],
               "val_auroc_cos": [], "val_auroc_H": []}
    best_auroc, best_state, bad_epochs = -float("inf"), None, 0

    for epoch in range(NUM_EPOCHS):
        tcr_proj.train(); pmhc_proj.train()
        rloss, ns = 0.0, 0
        for batch in train_loader:
            eT = batch["emb_T"].to(device); mT = batch["mask_T"].to(device)
            eP = batch["emb_P"].to(device); mP = batch["mask_P"].to(device)
            eH = batch["emb_H"].to(device); mH = batch["mask_H"].to(device)
            op_lat = batch["operator_latent"].to(device)
            zT = tcr_proj(eT, mT); zPH = pmhc_proj(eP, mP, eH, mH)
            with torch.no_grad():
                Zstar = operator_latent_to_full_zstar(op_lat, d=ZSTAR_D, r=ZSTAR_RANK)
            loss, _ = vicreg_hamiltonian_loss(zT, zPH, Zstar, alpha=alpha, beta=beta, delta=delta, gamma_var=gamma_var)
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            rloss += loss.item(); ns += 1

        tl = rloss / max(ns, 1); history["train_loss"].append(tl)

        val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, alpha, beta, delta, gamma_var)
        vl = val_out["metrics"]["val_loss"]; va = val_out["metrics"]["auroc"]
        vp = val_out["metrics"]["auprc"]; vf = val_out["metrics"]["f1"]
        va_cos = val_out["metrics"]["auroc_cos"]; va_H = val_out["metrics"]["auroc_H"]
        history["val_loss"].append(vl); history["val_auroc"].append(va)
        history["val_auprc"].append(vp); history["val_f1"].append(vf)
        history["val_auroc_cos"].append(va_cos); history["val_auroc_H"].append(va_H)
        log.info(f"  [{run_tag}] ep{epoch+1}/{NUM_EPOCHS} tl={tl:.4f} vl={vl:.4f} "
                 f"auroc={va:.4f}({val_out['metrics']['score_type']}) "
                 f"cos={va_cos:.4f} H={va_H:.4f} auprc={vp:.4f} f1={vf:.4f}")

        plot_epoch_diagnostics(val_out, epoch+1, pep_lookup_val, run_tag, FIGURE_SUBDIR)
        scheduler.step()

        if va > best_auroc + 1e-4:
            best_auroc = va; bad_epochs = 0
            best_state = {
                "epoch": epoch+1, "val_auroc": va, "val_loss": vl,
                "tcr_proj": copy.deepcopy(tcr_proj.state_dict()),
                "pmhc_proj": copy.deepcopy(pmhc_proj.state_dict()),
                "val_metrics": val_out["metrics"], "val_outputs": val_out,
                "config": {"rL": rL, "rD": rD, "d": D, "R_PH": R_PH, "lr": lr,
                           "weight_decay": weight_decay, "alpha": alpha, "beta": beta},
            }
            torch.save({
                "tcr_projection_state_dict": tcr_proj.state_dict(),
                "pmhc_projection_state_dict": pmhc_proj.state_dict(),
                "best_config": best_state["config"], "best_val_metrics": val_out["metrics"],
                "best_val_outputs": val_out, "history": history, "best_epoch": epoch+1,
            }, SAVE_DIR / f"best_{run_tag}.pt")
            log.info(f"  -> New best AUROC={va:.4f} ep{epoch+1} ({val_out['metrics']['score_type']}), saved")
        else:
            bad_epochs += 1
        if bad_epochs >= PATIENCE:
            log.info(f"  Early stop at ep{epoch+1} (patience={PATIENCE})"); break

    tcr_proj.load_state_dict(best_state["tcr_proj"]); pmhc_proj.load_state_dict(best_state["pmhc_proj"])
    thr = find_best_threshold(best_state["val_outputs"]["scores"], best_state["val_outputs"]["labels"])
    best_state["threshold"] = thr["threshold"]

    return {"tcr_proj": tcr_proj, "pmhc_proj": pmhc_proj, "history": history, "best_state": best_state}

# ============================================================
# MAIN
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    # Load ESM shard datasets
    train_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "train")
    val_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "val")
    test_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "test")

    # Load Boltz latents and create paired datasets
    op_train = OperatorLatentStore(BOLTZ_LATENT_DIR, "train")
    op_val = OperatorLatentStore(BOLTZ_LATENT_DIR, "val")
    op_test = OperatorLatentStore(BOLTZ_LATENT_DIR, "test")
    paired_train = PairedESMBoltzDataset(train_dataset, op_train)
    paired_val = PairedESMBoltzDataset(val_dataset, op_val)
    paired_test = PairedESMBoltzDataset(test_dataset, op_test)
    train_loader = DataLoader(paired_train, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])
    val_loader = DataLoader(paired_val, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    test_loader = DataLoader(paired_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    log.info(f"Loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

    # ---- HPO LOOP ----
    all_results = []
    best_auroc_global = -float("inf")
    best_result = None

    for ci, cfg in enumerate(SEARCH_SPACE):
        tag = f"cfg{ci}"
        log.info(f"\n===== Config {ci+1}/{len(SEARCH_SPACE)}: {cfg} =====")

        result = run_experiment(
            train_loader, val_loader, device, pep_lookup_val,
            rL=cfg["rL"], rD=cfg["rD"], lr=cfg["lr"],
            weight_decay=cfg["wd"], alpha=cfg["alpha"], beta=cfg["beta"],
            run_tag=tag,
        )
        va = result["best_state"]["val_auroc"]
        all_results.append({"config": cfg, "auroc": va, "epoch": result["best_state"]["epoch"]})
        log.info(f"Config {ci+1} best AUROC: {va:.4f} at epoch {result['best_state']['epoch']}")

        if va > best_auroc_global:
            best_auroc_global = va
            best_result = result

    log.info(f"\n===== GLOBAL BEST AUROC: {best_auroc_global:.4f} =====")
    log.info(f"Config: {best_result['best_state']['config']}")

    # ---- TEST EVALUATION ----
    best_thr = best_result["best_state"]["threshold"]
    test_out = evaluate_loader(test_loader, best_result["tcr_proj"], best_result["pmhc_proj"],
                                device, alpha=1.0, beta=25.0, delta=1.0, gamma_var=1.0)
    preds = (test_out["scores"] >= best_thr).astype(int)
    labels = test_out["labels"]
    log.info(f"\nTest AUROC: {test_out['metrics']['auroc']:.4f}")
    log.info(f"Test AUPRC: {test_out['metrics']['auprc']:.4f}")
    log.info(f"Test F1: {f1_score(labels, preds, zero_division=0):.4f}")
    log.info(f"Test confusion:\n{confusion_matrix(labels, preds)}")

    # Test plots
    plot_H_histogram(test_out["H"], labels, "test_H_best", FIGURE_SUBDIR)
    plot_cosine_histogram(test_out["cos"], labels, "test_cos_best", FIGURE_SUBDIR)
    plot_cross_reactivity(test_out["cos"], test_out["pair_ids"], labels, pep_lookup_test,
                          "test_xreact_best", FIGURE_SUBDIR)
    plot_training_history(best_result["history"], FIGURE_SUBDIR, prefix="best_config")

    # Save HPO summary
    pd.DataFrame(all_results).to_csv(SAVE_DIR / "hpo_summary.csv", index=False)
    log.info(f"HPO summary saved to {SAVE_DIR / 'hpo_summary.csv'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
