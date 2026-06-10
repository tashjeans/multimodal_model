#!/usr/bin/env python3
"""
Geometry-aware Hamiltonian model search
---------------------------------------

Goal
----
Search locally around the previously successful regime, while keeping
the Hamiltonian interpretation fixed:

    H = -1 - cosine
    lower / more negative H = more binder-like

Design
------
1. Train on positives only.
2. Use negatives only in validation/test for selection/diagnostics.
3. Search a SMALL local grid around the known good basin.
4. Use a short scout phase first.
5. Save histogram figures for scout winners and final runs.
6. Select checkpoints by validation geometry + ranking, not AUROC alone.

Run
---
tmux new -s hpo_local
conda activate tcr-multimodal
cd /home/natasha/multimodal_model/scripts/train
python hpo_local_geometry_hamiltonian.py 2>&1 | tee hpo_local_geometry_hamiltonian.log
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

from scipy.stats import ks_2samp

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
# PATHS
# ============================================================
PROJECT = Path("/home/natasha/multimodal_model")
EMBED_ROOT = PROJECT / "models/embeddings/no_boltz_train_dedup"
CHECKPOINTS_DIR = PROJECT / "models/checkpoints"
FIGURE_DIR = PROJECT / "models/figures"

RUN_NAME = "hpo_local_geometry_hamiltonian_v1"
SAVE_DIR = CHECKPOINTS_DIR / RUN_NAME
FIG_SAVE_DIR = FIGURE_DIR / RUN_NAME
SAVE_DIR.mkdir(parents=True, exist_ok=True)
FIG_SAVE_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = str(PROJECT / "data/train/train_with_ids_dedup_vs_valtest.csv")
VAL_CSV   = str(PROJECT / "data/val/val_df_clean_pos_neg.csv")
TEST_CSV  = str(PROJECT / "data/test/test_df_clean_pos_neg.csv")

# ============================================================
# SEARCH SPACE
# Local grid around the known good basin.
# d is FIXED at 128 and is NOT searched.
# ============================================================
SEEDS = [31, 47]

RLS = [6, 8, 10]
RDS = [12, 16, 20]
BETAS = [20.0, 25.0, 30.0]
LRS = [5e-5, 1e-4, 2e-4]

CONFIGS = []
for rL, rD, beta, lr in product(RLS, RDS, BETAS, LRS):
    CONFIGS.append({
        "name": f"rL{rL}_rD{rD}_beta{int(beta)}_lr{lr:.0e}",
        "rL": rL,
        "rD": rD,
        "lr": lr,
        "wd": 1e-2,
        "alpha": 1.0,
        "beta": beta,
        "dropout": 0.0,
        "norm_type": "layernorm",
    })


# ============================================================
# GLOBAL CONSTANTS
# ============================================================
D = 128                 # FIXED, not a hyperparameter
R_PH = 0.7
DELTA = 1.0
GAMMA_VAR = 1.0
GRAD_CLIP_NORM = 1.0
EPS = 1e-8

SCOUT_EPOCHS = 4
FULL_EPOCHS = 16
TOP_K_TO_CONTINUE = 4

PATIENCE = 4
MIN_EPOCHS_BEFORE_EARLY_STOP = 4

# Geometry gates
MIN_MEAN_GAP = 0.01
MIN_MEDIAN_GAP = 0.005
MIN_EFFECT_SIZE = 0.10
MAX_OVERLAP = 0.75

OVERLAP_BINS = 60

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
        assert self.shard_paths, f"No shard_*.pt files found in {self.shards_dir}"

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
def make_norm(norm_type: str, d: int):
    norm_type = norm_type.lower()
    if norm_type == "layernorm":
        return nn.LayerNorm(d)
    if norm_type == "batchnorm":
        return nn.BatchNorm1d(d)
    raise ValueError(f"Unknown norm_type: {norm_type}")

class ESMProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_max, dropout=0.0, norm_type="layernorm"):
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
            make_norm(norm_type, d),
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
    def __init__(self, D, rL, rD, d, L_P_max, L_H_max, R_PH=0.7, dropout=0.0, norm_type="layernorm"):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        assert d_P > 0 and d_H > 0

        self.pep_encoder = ESMProjectionHead(
            D, rL, rD, d_P, L_P_max, dropout=dropout, norm_type=norm_type
        )
        self.hla_encoder = ESMProjectionHead(
            D, rL, rD, d_H, L_H_max, dropout=dropout, norm_type=norm_type
        )

    def forward(self, emb_P, mask_P, emb_H, mask_H):
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)


# ============================================================
# LOSS
# ============================================================
def vicreg_variance(u, gamma=1.0, eps=1e-4):
    u_centered = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u_centered.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()

def vicreg_covariance(u):
    B, d = u.shape
    u_centered = u - u.mean(dim=0, keepdim=True)
    cov = (u_centered.T @ u_centered) / max(B - 1, 1)
    diag = torch.diag(cov)
    cov_off = cov - torch.diag_embed(diag)
    return (cov_off ** 2).sum() / (d * d)

def vicreg_hamiltonian_loss(
    zT_raw,
    zPH_raw,
    alpha=1.0,
    beta=25.0,
    delta=1.0,
    gamma_var=1.0,
    eps=1e-4,
):
    eT = zT_raw / (zT_raw.norm(dim=-1, keepdim=True) + eps)
    ePH = zPH_raw / (zPH_raw.norm(dim=-1, keepdim=True) + eps)

    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos
    L_inv = H.mean()

    L_var_T = vicreg_variance(zT_raw, gamma=gamma_var, eps=eps)
    L_var_PH = vicreg_variance(zPH_raw, gamma=gamma_var, eps=eps)
    L_cov_T = vicreg_covariance(zT_raw)
    L_cov_PH = vicreg_covariance(zPH_raw)

    L_var_total = L_var_T + L_var_PH
    L_cov_total = L_cov_T + L_cov_PH
    L_total = alpha * L_inv + beta * L_var_total + delta * L_cov_total

    return L_total, {
        "cos_mean": float(cos.mean().item()),
        "H_mean": float(H.mean().item()),
        "L_inv": float(L_inv.item()),
        "L_var_total": float(L_var_total.item()),
        "L_cov_total": float(L_cov_total.item()),
    }

# ============================================================
# METRICS
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

# ============================================================
# GEOMETRY
# ============================================================
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
    ovl = np.minimum(hist_pos, hist_neg).sum() * bin_w
    return float(np.clip(ovl, 0.0, 1.0))

def compute_geometry_metrics(H_scores, labels):
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

    overlap = estimate_overlap_coefficient(pos, neg, bins=OVERLAP_BINS)
    ks_stat, ks_p = ks_2samp(neg, pos)

    gate_pass = (
        (mean_gap >= MIN_MEAN_GAP) and
        (median_gap >= MIN_MEDIAN_GAP) and
        (effect_size >= MIN_EFFECT_SIZE) and
        (overlap <= MAX_OVERLAP)
    )

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
        "gate_pass": bool(gate_pass),
    }

def composite_val_score(metrics, geom):
    score = 0.0
    score += 1.5 * metrics["auroc"]
    score += 1.0 * metrics["auprc"]
    score += 3.0 * geom["mean_gap"]
    score += 2.0 * geom["median_gap"]
    score += 1.0 * geom["effect_size"]
    score += 0.5 * geom["ks_stat"]
    score -= 2.0 * geom["overlap"]

    if not geom["gate_pass"]:
        score -= 3.0

    return float(score)

def log_H_stats(prefix, H_scores, labels):
    pos = H_scores[labels == 1]
    neg = H_scores[labels == 0]
    log.info(
        f"{prefix} H stats | range=({H_scores.min():.6f}, {H_scores.max():.6f}) | "
        f"pos_mean={pos.mean():.6f} pos_med={np.median(pos):.6f} | "
        f"neg_mean={neg.mean():.6f} neg_med={np.median(neg):.6f}"
    )

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


# ============================================================
# FORWARD + EVALUATE
# ============================================================
@torch.no_grad()
def forward_batch(batch, tcr_proj, pmhc_proj, device, eps=EPS):
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
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, cfg):
    tcr_proj.eval()
    pmhc_proj.eval()

    all_H, all_cos, all_lab, all_pid = [], [], [], []
    running_loss, n_steps = 0.0, 0

    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device)

        all_H.append(out["H"])
        all_cos.append(out["cos"])
        all_lab.append(out["labels"])
        all_pid.extend(out["pair_ids"])

        loss, _ = vicreg_hamiltonian_loss(
            out["zT"], out["zPH"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            delta=DELTA,
            gamma_var=GAMMA_VAR,
        )
        running_loss += loss.item()
        n_steps += 1

    H_vals = np.concatenate(all_H)
    cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_lab).astype(int)

    ranking_scores = -H_vals
    metrics = {
        "auroc": float(roc_auc_score(labels, ranking_scores)),
        "auprc": float(average_precision_score(labels, ranking_scores)),
        "val_loss": float(running_loss / max(n_steps, 1)),
    }
    geom = compute_geometry_metrics(H_vals, labels)

    return {
        "H": H_vals,
        "cos": cos_vals,
        "labels": labels,
        "pair_ids": all_pid,
        "metrics": metrics,
        "geometry": geom,
    }

# ============================================================
# TRAIN ONE CONFIG/SEED
# ============================================================
def run_single(cfg, seed, train_loader, val_loader, device, max_epochs):
    set_global_seed(seed)

    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], D, L_T,
        dropout=cfg["dropout"], norm_type=cfg["norm_type"]
    ).to(device)

    pmhc_proj = PMHCProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], D,
        L_P, L_H, R_PH=R_PH,
        dropout=cfg["dropout"], norm_type=cfg["norm_type"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(tcr_proj.parameters()) + list(pmhc_proj.parameters()),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    best_score = -float("inf")
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(max_epochs):
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
                delta=DELTA,
                gamma_var=GAMMA_VAR,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tcr_proj.parameters()) + list(pmhc_proj.parameters()),
                GRAD_CLIP_NORM
            )
            optimizer.step()

            train_running_loss += loss.item()
            train_steps += 1

        val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, cfg)
        metrics = val_out["metrics"]
        geom = val_out["geometry"]
        score = composite_val_score(metrics, geom)

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_running_loss / max(train_steps, 1)),
            "val_loss": metrics["val_loss"],
            "val_auroc": metrics["auroc"],
            "val_auprc": metrics["auprc"],
            "mean_gap": geom["mean_gap"],
            "median_gap": geom["median_gap"],
            "effect_size": geom["effect_size"],
            "overlap": geom["overlap"],
            "gate_pass": geom["gate_pass"],
            "score": score,
        }
        history.append(epoch_row)

        log.info(
            f"[{cfg['name']}][seed={seed}] ep{epoch+1}/{max_epochs} | "
            f"train_loss={epoch_row['train_loss']:.4f} | "
            f"val_auroc={metrics['auroc']:.4f} | "
            f"val_auprc={metrics['auprc']:.4f} | "
            f"mean_gap={geom['mean_gap']:.4f} | "
            f"overlap={geom['overlap']:.4f} | "
            f"gate={geom['gate_pass']} | "
            f"score={score:.4f}"
        )
        log_H_stats(f"[{cfg['name']}][seed={seed}] VAL", val_out["H"], val_out["labels"])

        if score > best_score:
            best_score = score
            bad_epochs = 0
            best_state = {
                "cfg_name": cfg["name"],
                "cfg": copy.deepcopy(cfg),
                "seed": seed,
                "epoch": epoch + 1,
                "score": score,
                "metrics": copy.deepcopy(metrics),
                "geometry": copy.deepcopy(geom),
                "tcr_state_dict": copy.deepcopy(tcr_proj.state_dict()),
                "pmhc_state_dict": copy.deepcopy(pmhc_proj.state_dict()),
                "history": copy.deepcopy(history),
                "best_val_H": val_out["H"].copy(),
                "best_val_labels": val_out["labels"].copy(),
            }
        else:
            if epoch + 1 >= MIN_EPOCHS_BEFORE_EARLY_STOP:
                bad_epochs += 1

        if bad_epochs >= PATIENCE:
            log.info(f"[{cfg['name']}][seed={seed}] early stop at epoch {epoch+1}")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint saved for cfg={cfg['name']} seed={seed}")

    return best_state


# ============================================================
# REBUILD + FINAL EVAL + SAVE
# ============================================================
def build_models_from_checkpoint(best_state, train_loader, device):
    cfg = best_state["cfg"]

    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], D, L_T,
        dropout=cfg["dropout"], norm_type=cfg["norm_type"]
    ).to(device)

    pmhc_proj = PMHCProjectionHead(
        D_esm, cfg["rL"], cfg["rD"], D,
        L_P, L_H, R_PH=R_PH,
        dropout=cfg["dropout"], norm_type=cfg["norm_type"]
    ).to(device)

    tcr_proj.load_state_dict(best_state["tcr_state_dict"])
    pmhc_proj.load_state_dict(best_state["pmhc_state_dict"])

    return tcr_proj, pmhc_proj

def final_evaluate(best_state, train_loader, val_loader, test_loader, device):
    tcr_proj, pmhc_proj = build_models_from_checkpoint(best_state, train_loader, device)
    cfg = best_state["cfg"]

    val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, cfg)
    val_thr = find_best_threshold_on_H(val_out["H"], val_out["labels"])

    test_out = evaluate_loader(test_loader, tcr_proj, pmhc_proj, device, cfg)
    test_preds = apply_threshold(test_out["H"], val_thr["threshold"])
    test_bin = compute_binary_metrics(test_out["labels"], test_preds)
    test_cm = confusion_matrix(test_out["labels"], test_preds)

    return {
        "val_metrics": val_out["metrics"],
        "val_geometry": val_out["geometry"],
        "val_threshold": val_thr,
        "test_metrics": {
            "auroc": test_out["metrics"]["auroc"],
            "auprc": test_out["metrics"]["auprc"],
            **test_bin,
        },
        "test_geometry": test_out["geometry"],
        "test_confusion_matrix": test_cm.tolist(),
        "test_H": test_out["H"].tolist(),
        "test_labels": test_out["labels"].tolist(),
    }

def save_checkpoint_bundle(best_state, final_eval, out_dir: Path):
    cfg_name = best_state["cfg_name"]
    seed = best_state["seed"]
    stem = f"{cfg_name}__seed_{seed}"

    torch.save(
        {
            "best_state": best_state,
            "final_eval": final_eval,
        },
        out_dir / f"{stem}.pt"
    )

    pd.DataFrame(best_state["history"]).to_csv(out_dir / f"{stem}__history.csv", index=False)

    with open(out_dir / f"{stem}__final_eval.json", "w") as f:
        json.dump(final_eval, f, indent=2)


# ============================================================
# SCOUT PHASE
# ============================================================
def scout_phase(train_loader, val_loader, device):
    scout_rows = []
    top_candidates = []

    for cfg in CONFIGS:
        for seed in SEEDS:
            log.info("=" * 90)
            log.info(f"SCOUT | cfg={cfg['name']} | seed={seed}")

            best_state = run_single(
                cfg=cfg,
                seed=seed,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=SCOUT_EPOCHS,
            )

            row = {
                "phase": "scout",
                "cfg_name": best_state["cfg_name"],
                "seed": best_state["seed"],
                "best_epoch": best_state["epoch"],
                "score": best_state["score"],
                "val_auroc": best_state["metrics"]["auroc"],
                "val_auprc": best_state["metrics"]["auprc"],
                "val_loss": best_state["metrics"]["val_loss"],
                "mean_gap": best_state["geometry"]["mean_gap"],
                "median_gap": best_state["geometry"]["median_gap"],
                "effect_size": best_state["geometry"]["effect_size"],
                "overlap": best_state["geometry"]["overlap"],
                "ks_stat": best_state["geometry"]["ks_stat"],
                "gate_pass": best_state["geometry"]["gate_pass"],
            }
            scout_rows.append(row)
            top_candidates.append(best_state)

    scout_df = pd.DataFrame(scout_rows).sort_values(
        by=["score", "val_auroc", "mean_gap"],
        ascending=[False, False, False],
    )
    scout_df.to_csv(SAVE_DIR / "scout_summary.csv", index=False)

    log.info("=" * 90)
    log.info("SCOUT SUMMARY")
    log.info("\n" + scout_df.head(20).to_string(index=False))

    top_candidates = sorted(
        top_candidates,
        key=lambda x: (x["score"], x["metrics"]["auroc"], x["geometry"]["mean_gap"]),
        reverse=True,
    )[:TOP_K_TO_CONTINUE]

    log.info("=" * 90)
    log.info("TOP CANDIDATES TO CONTINUE")
    for i, cand in enumerate(top_candidates, start=1):
        log.info(
            f"{i}. cfg={cand['cfg_name']} | seed={cand['seed']} | "
            f"epoch={cand['epoch']} | score={cand['score']:.4f} | "
            f"auroc={cand['metrics']['auroc']:.4f} | "
            f"mean_gap={cand['geometry']['mean_gap']:.4f} | "
            f"overlap={cand['geometry']['overlap']:.4f}"
        )

    return scout_df, top_candidates


# ============================================================
# FULL PHASE
# ============================================================
def full_phase(top_candidates, train_loader, val_loader, test_loader, device):
    final_rows = []

    for cand in top_candidates:
        cfg = cand["cfg"]
        seed = cand["seed"]

        log.info("=" * 90)
        log.info(f"FULL | cfg={cfg['name']} | seed={seed}")

        best_state = run_single(
            cfg=cfg,
            seed=seed,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            max_epochs=FULL_EPOCHS,
        )

        final_eval = final_evaluate(
            best_state=best_state,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
        )

        save_checkpoint_bundle(best_state, final_eval, SAVE_DIR)

        stem = f"{best_state['cfg_name']}__seed_{best_state['seed']}"

        plot_h_histogram(
            best_state["best_val_H"],
            best_state["best_val_labels"],
            title=f"{stem} | best val epoch {best_state['epoch']}",
            save_path=FIG_SAVE_DIR / f"{stem}__best_val_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )

        plot_h_histogram(
            np.array(final_eval["test_H"]),
            np.array(final_eval["test_labels"]),
            title=f"{stem} | test",
            save_path=FIG_SAVE_DIR / f"{stem}__test_hist.png",
            threshold=final_eval["val_threshold"]["threshold"],
        )

        row = {
            "phase": "full",
            "cfg_name": best_state["cfg_name"],
            "seed": best_state["seed"],
            "best_epoch": best_state["epoch"],
            "score": best_state["score"],

            "val_auroc": final_eval["val_metrics"]["auroc"],
            "val_auprc": final_eval["val_metrics"]["auprc"],
            "val_loss": final_eval["val_metrics"]["val_loss"],
            "val_mean_gap": final_eval["val_geometry"]["mean_gap"],
            "val_median_gap": final_eval["val_geometry"]["median_gap"],
            "val_effect_size": final_eval["val_geometry"]["effect_size"],
            "val_overlap": final_eval["val_geometry"]["overlap"],
            "val_gate_pass": final_eval["val_geometry"]["gate_pass"],
            "val_threshold": final_eval["val_threshold"]["threshold"],
            "val_threshold_f1": final_eval["val_threshold"]["f1"],

            "test_auroc": final_eval["test_metrics"]["auroc"],
            "test_auprc": final_eval["test_metrics"]["auprc"],
            "test_f1": final_eval["test_metrics"]["f1"],
            "test_accuracy": final_eval["test_metrics"]["accuracy"],
            "test_precision": final_eval["test_metrics"]["precision"],
            "test_recall": final_eval["test_metrics"]["recall"],
            "test_mean_gap": final_eval["test_geometry"]["mean_gap"],
            "test_median_gap": final_eval["test_geometry"]["median_gap"],
            "test_effect_size": final_eval["test_geometry"]["effect_size"],
            "test_overlap": final_eval["test_geometry"]["overlap"],
            "test_gate_pass": final_eval["test_geometry"]["gate_pass"],
        }
        final_rows.append(row)

        log.info(
            f"FINAL | cfg={best_state['cfg_name']} | seed={best_state['seed']} | "
            f"best_epoch={best_state['epoch']} | "
            f"val_auroc={final_eval['val_metrics']['auroc']:.4f} | "
            f"test_auroc={final_eval['test_metrics']['auroc']:.4f} | "
            f"test_f1={final_eval['test_metrics']['f1']:.4f}"
        )
        log.info(f"FINAL TEST CONFUSION MATRIX: {final_eval['test_confusion_matrix']}")

    final_df = pd.DataFrame(final_rows).sort_values(
        by=["score", "test_auroc", "test_f1"],
        ascending=[False, False, False],
    )
    final_df.to_csv(SAVE_DIR / "full_summary.csv", index=False)

    log.info("=" * 90)
    log.info("FULL SUMMARY")
    log.info("\n" + final_df.to_string(index=False))

    return final_df

# ============================================================
# MAIN
# ============================================================
def main():
    log.info("Starting local geometry-aware Hamiltonian search")
    log.info(f"RUN_NAME = {RUN_NAME}")
    log.info(f"N_CONFIGS = {len(CONFIGS)}")
    log.info(f"SEEDS = {SEEDS}")
    log.info(f"SAVE_DIR = {SAVE_DIR}")
    log.info(f"FIG_SAVE_DIR = {FIG_SAVE_DIR}")
    log.info(f"D is fixed at {D}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    train_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "train")
    val_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "val")
    test_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x[0],
    )

    log.info(
        f"Loaders | train={len(train_loader)} | val={len(val_loader)} | test={len(test_loader)}"
    )

    scout_df, top_candidates = scout_phase(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    final_df = full_phase(
        top_candidates=top_candidates,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
    )

    manifest = {
        "run_name": RUN_NAME,
        "save_dir": str(SAVE_DIR),
        "fig_save_dir": str(FIG_SAVE_DIR),
        "d_fixed": D,
        "seeds": SEEDS,
        "n_configs": len(CONFIGS),
        "configs": CONFIGS,
        "scout_epochs": SCOUT_EPOCHS,
        "full_epochs": FULL_EPOCHS,
        "top_k_to_continue": TOP_K_TO_CONTINUE,
        "n_scout_rows": int(len(scout_df)),
        "n_final_rows": int(len(final_df)),
    }
    with open(SAVE_DIR / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    log.info("=" * 90)
    log.info("DONE")

if __name__ == "__main__":
    main()

    