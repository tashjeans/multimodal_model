#!/usr/bin/env python3
"""
Baseline Hamiltonian model (no Boltz, Z* = identity blocks) — stability-focused v2

Main changes vs prior v2
------------------------
1. Fixed score semantics:
   - The operative score is now the Hamiltonian H = -1 - cosine.
   - Lower / more negative H means stronger binding.
   - Thresholding is therefore fixed to preds = (H <= threshold).
   - AUROC / AUPRC are computed on -H so that higher ranking scores correspond to positives.
   This removes the previous threshold-direction flipping across epochs.

2. Improved run-to-run stability:
   - Deterministic seeds are set for Python / NumPy / Torch / CUDA.
   - BatchNorm1d is replaced with LayerNorm.
   - Dropout is removed.
   - Gradient clipping is retained.
   - Warmup + cosine schedule retained.
   - Search space is narrowed around the smaller, more regularised regime.

3. Stronger anti-collapse regularisation:
   - beta kept in the stronger regime (25.0)
   - delta increased from 1.0 to 5.0 to strengthen covariance regularisation

4. Logging changed to avoid file conflicts with tee:
   - This script logs to stdout only.
   - Use shell tee if you want a logfile.

5. Cross-reactivity kept correct:
   - Pairwise cosine distances between positive zT embeddings grouped by peptide.

Run:
    tmux new -s hpo_baseline
    conda activate tcr-multimodal
    cd /home/natasha/multimodal_model/scripts/train
    python hpo_baseline_hamiltonian.py 2>&1 | tee hpo_baseline_hamiltonian.log
"""

import os
import sys
import copy
import random
import logging
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.spatial.distance import pdist
from scipy.stats import mannwhitneyu

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
# REPRODUCIBILITY
# ============================================================
SEED = 42

def set_global_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
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
VAL_CSV = str(PROJECT / "data/val/val_df_clean_pos_neg.csv")
TEST_CSV = str(PROJECT / "data/test/test_df_clean_pos_neg.csv")

FIGURE_SUBDIR = FIGURE_DIR / "hpo_baseline_hamiltonian_v2"
FIGURE_SUBDIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = CHECKPOINTS_DIR / "hpo_baseline_hamiltonian_v2"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# HPO CONFIG
# ============================================================
# Focused stability sweep around the smaller / more regularised regime.
SEARCH_SPACE = [
    {"rL": 4, "rD": 8, "lr": 5e-5, "wd": 1e-2, "alpha": 1.0, "beta": 25.0},
    {"rL": 4, "rD": 8, "lr": 1e-4, "wd": 1e-2, "alpha": 1.0, "beta": 25.0},
]
NUM_EPOCHS = 30
PATIENCE = 10
D = 128
R_PH = 0.7
DELTA = 5.0
GAMMA_VAR = 1.0
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
EMA_WINDOW = 3
MIN_EPOCHS_BEFORE_EARLY_STOP = 8
EPS = 1e-8
PLOT_EVERY_N_EPOCHS = 5

# ============================================================
# PEPTIDE LOOKUPS
# ============================================================
def build_pep_lookup(csv_path):
    df = pd.read_csv(csv_path)
    return {str(r.get("pair_id", r.name)): str(r["Peptide"]) for _, r in df.iterrows()}

pep_lookup_train = build_pep_lookup(TRAIN_CSV)
pep_lookup_val = build_pep_lookup(VAL_CSV)
pep_lookup_test = build_pep_lookup(TEST_CSV)

# ============================================================
# MODELS
# ============================================================
class ESMProjectionHead(nn.Module):
    def __init__(self, D, rL, rD, d, L_max):
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

        # Stability change: LayerNorm instead of BatchNorm, and dropout removed.
        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.LayerNorm(d),
            nn.ReLU(),
            nn.Linear(d, d),
        )

    def forward(self, emb, mask):
        device = emb.device
        B, _, D = emb.shape
        assert D == self.D

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
    def __init__(self, D, rL, rD, d, L_P_max, L_H_max, R_PH=0.7):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        assert d_P > 0 and d_H > 0

        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max)

    def forward(self, emb_P, mask_P, emb_H, mask_H):
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)

# ============================================================
# REGULARISERS / LOSS
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
    delta=5.0,
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

    components = {
        "L_total": L_total.item(),
        "L_inv_H": L_inv.item(),
        "L_var_T": L_var_T.item(),
        "L_var_PH": L_var_PH.item(),
        "L_cov_T": L_cov_T.item(),
        "L_cov_PH": L_cov_PH.item(),
        "alpha_L_inv": (alpha * L_inv).item(),
        "beta_var": (beta * L_var_total).item(),
        "delta_cov": (delta * L_cov_total).item(),
        "cos_mean": cos.mean().item(),
        "H_mean": H.mean().item(),
        "H_min": H.min().item(),
        "H_max": H.max().item(),
    }
    return L_total, components

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
# THRESHOLDING / METRICS
# ============================================================
def compute_binary_metrics(labels, preds):
    return {
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def apply_threshold(scores, threshold):
    # Fixed semantics: lower / more negative H means positive.
    return (scores <= threshold).astype(int)


def find_best_threshold(scores, labels):
    thresholds = np.unique(scores)
    best = None
    for thr in thresholds:
        preds = apply_threshold(scores, thr)
        metrics = compute_binary_metrics(labels, preds)
        if best is None or metrics["f1"] > best["f1"]:
            best = {
                "threshold": float(thr),
                "direction": "<=",
                **metrics,
            }
    return best


def log_score_stats(prefix, scores, labels):
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    log.info(
        f"{prefix} H score stats | range=({scores.min():.6f}, {scores.max():.6f}) | "
        f"pos_mean={pos.mean():.6f} pos_med={np.median(pos):.6f} | "
        f"neg_mean={neg.mean():.6f} neg_med={np.median(neg):.6f}"
    )

# ============================================================
# PLOTTING
# ============================================================
def plot_H_histogram(H_vals, labels, title, save_dir, threshold=None):
    pos, neg = H_vals[labels == 1], H_vals[labels == 0]
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
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png")
    plt.close()


def plot_cosine_histogram(cos_vals, labels, title, save_dir):
    pos, neg = cos_vals[labels == 1], cos_vals[labels == 0]
    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.6, density=True, label="negative")
    plt.hist(pos, bins=40, alpha=0.6, density=True, label="positive")
    plt.xlabel("eT · ePH cosine similarity")
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png")
    plt.close()


def plot_cross_reactivity_zT(zT, pair_ids, labels, pep_lookup, title, save_dir, min_group_size=2, random_n=5000, seed=42):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    pos_mask = labels == 1

    zT_pos = zT[pos_mask]
    pos_pids = [pid for pid, lab in zip(pair_ids, labels) if lab == 1]

    pep_to_indices = defaultdict(list)
    for i, pid in enumerate(pos_pids):
        pep = pep_lookup.get(str(pid))
        if pep is not None:
            pep_to_indices[pep].append(i)

    within_dists = []
    for pep, idxs in pep_to_indices.items():
        if len(idxs) >= min_group_size:
            vecs = zT_pos[idxs]
            d = pdist(vecs, metric="cosine")
            within_dists.extend(d.tolist())

    within_dists = np.array(within_dists, dtype=float)

    if len(zT_pos) > 1:
        sample_size = min(len(zT_pos), random_n)
        rand_idx = rng.choice(len(zT_pos), size=sample_size, replace=False)
        random_vecs = zT_pos[rand_idx]
        random_dists = pdist(random_vecs, metric="cosine")
    else:
        random_dists = np.array([], dtype=float)

    if len(within_dists) == 0:
        log.info("  [cross-react] No peptide groups >=2 TCRs, skipping")
        return

    plt.figure(figsize=(6, 5))
    plt.boxplot([within_dists, random_dists], tick_labels=["Same peptide", "Random"])
    plt.ylabel("Cosine distance in TCR latent space")
    plt.title(title)
    plt.tight_layout()
    safe = title.replace(" ", "_").replace("/", "_")
    plt.savefig(save_dir / f"{safe}.png")
    plt.close()

    msg = (
        f"  [cross-react] same_pep n={len(within_dists)} med={np.median(within_dists):.4f} | "
        f"random n={len(random_dists)} med={np.median(random_dists):.4f}"
    )
    if len(random_dists) > 0:
        _, p_val = mannwhitneyu(within_dists, random_dists, alternative="less")
        msg += f" | MWU p={p_val:.3e}"
    log.info(msg)


def plot_training_history(history, save_dir, prefix=""):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train loss")
    plt.plot(epochs, history["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{prefix} Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_loss.png")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_auroc"], label="val AUROC")
    plt.plot(epochs, history["val_auprc"], label="val AUPRC")
    plt.plot(epochs, history["val_auroc_ema"], label="val AUROC EMA")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title(f"{prefix} Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_metrics.png")
    plt.close()


def plot_epoch_diagnostics(val_out, epoch, pep_lookup, run_tag, save_dir):
    labels = val_out["labels"]
    ep_dir = save_dir / run_tag
    ep_dir.mkdir(parents=True, exist_ok=True)
    plot_H_histogram(
        val_out["H"],
        labels,
        f"{run_tag}_val_H_ep{epoch}",
        ep_dir,
        threshold=val_out["metrics"]["threshold"],
    )
    plot_cosine_histogram(val_out["cos"], labels, f"{run_tag}_val_cos_ep{epoch}", ep_dir)
    plot_cross_reactivity_zT(
        val_out["zT"],
        val_out["pair_ids"],
        labels,
        pep_lookup,
        f"{run_tag}_val_xreact_ep{epoch}",
        ep_dir,
    )

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
        "scores": H.cpu().numpy(),  # operative score: lower/more negative is more binder-like
        "labels": labels,
        "pair_ids": batch["pair_id"],
    }


@torch.no_grad()
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, alpha=1.0, beta=25.0, delta=5.0, gamma_var=1.0, eps=EPS):
    tcr_proj.eval()
    pmhc_proj.eval()

    all_s, all_H, all_cos, all_lab, all_pid, all_zT = [], [], [], [], [], []
    running_loss, n_steps = 0.0, 0

    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device, eps)
        all_s.append(out["scores"])
        all_H.append(out["H"])
        all_cos.append(out["cos"])
        all_lab.append(out["labels"])
        all_pid.extend(out["pair_ids"])
        all_zT.append(out["zT"].cpu().numpy())

        loss, _ = vicreg_hamiltonian_loss(
            out["zT"], out["zPH"], alpha=alpha, beta=beta, delta=delta, gamma_var=gamma_var
        )
        running_loss += loss.item()
        n_steps += 1

    scores = np.concatenate(all_s)     # H
    H_vals = np.concatenate(all_H)
    cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_lab).astype(int)
    zT = np.concatenate(all_zT, axis=0)

    thr = find_best_threshold(scores, labels)
    ranking_scores = -scores  # higher means more positive for AUROC / AUPRC
    metrics = {
        "auroc": float(roc_auc_score(labels, ranking_scores)),
        "auprc": float(average_precision_score(labels, ranking_scores)),
        "val_loss": float(running_loss / max(n_steps, 1)),
        **thr,
    }

    return {
        "scores": scores,
        "H": H_vals,
        "cos": cos_vals,
        "zT": zT,
        "labels": labels,
        "pair_ids": all_pid,
        "metrics": metrics,
    }

# ============================================================
# TRAINING
# ============================================================
def make_scheduler(optimizer, total_epochs, warmup_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def run_experiment(
    train_loader,
    val_loader,
    device,
    pep_lookup_val,
    rL=4,
    rD=8,
    lr=5e-5,
    weight_decay=1e-2,
    alpha=1.0,
    beta=25.0,
    delta=5.0,
    gamma_var=1.0,
    run_tag="cfg0",
):
    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(D_esm, rL, rD, D, L_max=L_T).to(device)
    pmhc_proj = PMHCProjectionHead(D_esm, rL, rD, D, L_P_max=L_P, L_H_max=L_H, R_PH=R_PH).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": tcr_proj.parameters(), "lr": lr},
            {"params": pmhc_proj.parameters(), "lr": lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = make_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
        "val_f1": [],
        "val_auroc_ema": [],
    }
    auroc_window = deque(maxlen=EMA_WINDOW)

    best_score = -float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(NUM_EPOCHS):
        tcr_proj.train()
        pmhc_proj.train()
        running_loss, n_steps = 0.0, 0

        for batch in train_loader:
            eT = batch["emb_T"].to(device)
            mT = batch["mask_T"].to(device)
            eP = batch["emb_P"].to(device)
            mP = batch["mask_P"].to(device)
            eH = batch["emb_H"].to(device)
            mH = batch["mask_H"].to(device)

            zT = tcr_proj(eT, mT)
            zPH = pmhc_proj(eP, mP, eH, mH)
            loss, _ = vicreg_hamiltonian_loss(
                zT,
                zPH,
                alpha=alpha,
                beta=beta,
                delta=delta,
                gamma_var=gamma_var,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tcr_proj.parameters()) + list(pmhc_proj.parameters()), GRAD_CLIP_NORM)
            optimizer.step()

            running_loss += loss.item()
            n_steps += 1

        scheduler.step()
        train_loss = running_loss / max(n_steps, 1)
        history["train_loss"].append(train_loss)

        val_out = evaluate_loader(val_loader, tcr_proj, pmhc_proj, device, alpha, beta, delta, gamma_var)
        val_loss = val_out["metrics"]["val_loss"]
        val_auroc = val_out["metrics"]["auroc"]
        val_auprc = val_out["metrics"]["auprc"]
        val_f1 = val_out["metrics"]["f1"]
        val_thr = val_out["metrics"]["threshold"]

        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)
        history["val_f1"].append(val_f1)

        auroc_window.append(val_auroc)
        val_auroc_ema = float(np.mean(auroc_window))
        history["val_auroc_ema"].append(val_auroc_ema)

        log.info(
            f"  [{run_tag}] ep{epoch+1}/{NUM_EPOCHS} tl={train_loss:.4f} vl={val_loss:.4f} "
            f"auroc={val_auroc:.4f} auprc={val_auprc:.4f} f1={val_f1:.4f} "
            f"thr={val_thr:.6f} dir=<= auroc_ema={val_auroc_ema:.4f}"
        )
        log_score_stats(f"  [{run_tag}] VAL", val_out["scores"], val_out["labels"])

        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0 or epoch == 0:
            plot_epoch_diagnostics(val_out, epoch + 1, pep_lookup_val, run_tag, FIGURE_SUBDIR)

        current_score = val_auroc_ema
        if current_score > best_score + 1e-4:
            best_score = current_score
            bad_epochs = 0
            best_state = {
                "epoch": epoch + 1,
                "val_auroc": val_auroc,
                "val_auroc_ema": val_auroc_ema,
                "val_loss": val_loss,
                "tcr_proj": copy.deepcopy(tcr_proj.state_dict()),
                "pmhc_proj": copy.deepcopy(pmhc_proj.state_dict()),
                "val_metrics": val_out["metrics"],
                "val_outputs": val_out,
                "config": {
                    "rL": rL,
                    "rD": rD,
                    "d": D,
                    "R_PH": R_PH,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "alpha": alpha,
                    "beta": beta,
                    "delta": delta,
                    "gamma_var": gamma_var,
                },
            }
            torch.save(
                {
                    "tcr_projection_state_dict": tcr_proj.state_dict(),
                    "pmhc_projection_state_dict": pmhc_proj.state_dict(),
                    "best_config": best_state["config"],
                    "best_val_metrics": val_out["metrics"],
                    "best_val_outputs": val_out,
                    "history": history,
                    "best_epoch": epoch + 1,
                },
                SAVE_DIR / f"best_{run_tag}.pt",
            )
            log.info(
                f"  -> New best EMA-AUROC={val_auroc_ema:.4f} (raw AUROC={val_auroc:.4f}) ep{epoch+1}, saved"
            )
        else:
            if epoch + 1 >= MIN_EPOCHS_BEFORE_EARLY_STOP:
                bad_epochs += 1

        if bad_epochs >= PATIENCE:
            log.info(f"  Early stop at ep{epoch+1} (patience={PATIENCE})")
            break

    if best_state is None:
        raise RuntimeError(f"No best state saved for run {run_tag}")

    tcr_proj.load_state_dict(best_state["tcr_proj"])
    pmhc_proj.load_state_dict(best_state["pmhc_proj"])

    best_state["threshold"] = best_state["val_metrics"]["threshold"]
    best_state["threshold_direction"] = best_state["val_metrics"]["direction"]

    return {
        "tcr_proj": tcr_proj,
        "pmhc_proj": pmhc_proj,
        "history": history,
        "best_state": best_state,
    }

# ============================================================
# MAIN
# ============================================================
def main():
    set_global_seed(SEED)

    log.info(f"Seed: {SEED}")
    log.info(
        f"Peptide lookups: train={len(pep_lookup_train)}, val={len(pep_lookup_val)}, test={len(pep_lookup_test)}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU: {torch.cuda.get_device_name()}")

    train_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "train")
    val_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "val")
    test_dataset = ShardedBatchTripletDataset(EMBED_ROOT / "test")

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    log.info(f"Loaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")
    log.info(f"Running focused stability sweep: {len(SEARCH_SPACE)} configs | plots every {PLOT_EVERY_N_EPOCHS} epochs")

    all_results = []
    best_global_score = -float("inf")
    best_result = None

    for ci, cfg in enumerate(SEARCH_SPACE):
        tag = f"cfg{ci}"
        log.info(f"\n===== Config {ci+1}/{len(SEARCH_SPACE)}: {cfg} =====")

        result = run_experiment(
            train_loader,
            val_loader,
            device,
            pep_lookup_val,
            rL=cfg["rL"],
            rD=cfg["rD"],
            lr=cfg["lr"],
            weight_decay=cfg["wd"],
            alpha=cfg["alpha"],
            beta=cfg["beta"],
            delta=DELTA,
            gamma_var=GAMMA_VAR,
            run_tag=tag,
        )

        raw_auroc = result["best_state"]["val_auroc"]
        ema_auroc = result["best_state"]["val_auroc_ema"]
        all_results.append(
            {
                "config": cfg,
                "val_auroc": raw_auroc,
                "val_auroc_ema": ema_auroc,
                "epoch": result["best_state"]["epoch"],
                "threshold": result["best_state"]["threshold"],
                "direction": result["best_state"]["threshold_direction"],
            }
        )
        log.info(
            f"Config {ci+1} best raw AUROC: {raw_auroc:.4f} | best EMA AUROC: {ema_auroc:.4f} at epoch {result['best_state']['epoch']}"
        )

        if ema_auroc > best_global_score:
            best_global_score = ema_auroc
            best_result = result

    if best_result is None:
        raise RuntimeError("No best result found across HPO configs")

    log.info(f"\n===== GLOBAL BEST EMA AUROC: {best_global_score:.4f} =====")
    log.info(f"Config: {best_result['best_state']['config']}")

    best_thr = best_result["best_state"]["threshold"]

    test_out = evaluate_loader(
        test_loader,
        best_result["tcr_proj"],
        best_result["pmhc_proj"],
        device,
        alpha=best_result["best_state"]["config"]["alpha"],
        beta=best_result["best_state"]["config"]["beta"],
        delta=best_result["best_state"]["config"]["delta"],
        gamma_var=best_result["best_state"]["config"]["gamma_var"],
    )

    preds = apply_threshold(test_out["scores"], best_thr)
    labels = test_out["labels"]
    test_bin = compute_binary_metrics(labels, preds)

    log.info(
        f"\n[TEST] using threshold={best_thr:.6f} direction=<= | pred_pos={int(preds.sum())} pred_neg={int((1-preds).sum())}"
    )
    log_score_stats("[TEST]", test_out["scores"], labels)
    log.info(f"Test AUROC: {test_out['metrics']['auroc']:.4f}")
    log.info(f"Test AUPRC: {test_out['metrics']['auprc']:.4f}")
    log.info(f"Test F1: {test_bin['f1']:.4f}")
    log.info(f"Test accuracy: {test_bin['accuracy']:.4f}")
    log.info(f"Test precision: {test_bin['precision']:.4f}")
    log.info(f"Test recall: {test_bin['recall']:.4f}")
    log.info(f"Test confusion:\n{confusion_matrix(labels, preds)}")

    plot_H_histogram(test_out["H"], labels, "test_H_best", FIGURE_SUBDIR, threshold=best_thr)
    plot_cosine_histogram(test_out["cos"], labels, "test_cos_best", FIGURE_SUBDIR)
    plot_cross_reactivity_zT(test_out["zT"], test_out["pair_ids"], labels, pep_lookup_test, "test_xreact_best", FIGURE_SUBDIR)
    plot_training_history(best_result["history"], FIGURE_SUBDIR, prefix="best_config")

    pd.DataFrame(all_results).to_csv(SAVE_DIR / "hpo_summary.csv", index=False)
    log.info(f"HPO summary saved to {SAVE_DIR / 'hpo_summary.csv'}")
    log.info("Done.")


if __name__ == "__main__":
    main()
