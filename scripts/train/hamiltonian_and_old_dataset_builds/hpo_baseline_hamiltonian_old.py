#!/usr/bin/env python3
"""
Baseline Hamiltonian model (no Boltz, Z* = identity blocks)
Seed-stability test for a single fixed config.

Purpose
-------
Test whether run instability is primarily due to seed dependence.

Fixed config
------------
rL=4, rD=8, lr=1e-4, wd=1e-2, alpha=1.0, beta=25.0

Important choices
-----------------
1. Operative score is Hamiltonian H = -1 - cosine.
   Lower / more negative H = stronger binding.
2. AUROC / AUPRC are computed on -H so that higher = more positive for sklearn.
3. Threshold is chosen ONLY ON FINAL VALIDATION after training, using H.
4. No thresholding diagnostics during training.
5. Cross-reactivity uses pairwise cosine distances between positive zT embeddings.

Run
---
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
EMBED_ROOT = PROJECT / "models/embeddings/no_boltz_train_dedup"
CHECKPOINTS_DIR = PROJECT / "models/checkpoints"
FIGURE_DIR = PROJECT / "models/figures"

TRAIN_CSV = str(PROJECT / "data/train/train_with_ids_dedup_vs_valtest.csv")
VAL_CSV   = str(PROJECT / "data/val/val_df_clean_pos_neg.csv")
TEST_CSV  = str(PROJECT / "data/test/test_df_clean_pos_neg.csv")

FIGURE_SUBDIR = FIGURE_DIR / "hpo_baseline_hamiltonian_v3_seedscan"
FIGURE_SUBDIR.mkdir(parents=True, exist_ok=True)
SAVE_DIR = CHECKPOINTS_DIR / "hpo_baseline_hamiltonian_v3_seedscan"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# FIXED EXPERIMENT CONFIG
# ============================================================
FIXED_CFG = {
    "rL": 4,
    "rD": 8,
    "lr": 1e-4,
    "wd": 1e-2,
    "alpha": 1.0,
    "beta": 25.0,
}

SEEDS = [31, 37, 43, 47, 53, 59, 67]   # change/add if you want more seeds

NUM_EPOCHS = 20
PATIENCE = 8
MIN_EPOCHS_BEFORE_EARLY_STOP = 5

D = 128
R_PH = 0.7
DELTA = 1.0
GAMMA_VAR = 1.0
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
EPS = 1e-8
PLOT_EVERY_N_EPOCHS = 5

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

        # Keep this close to the older regime that gave good runs.
        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.BatchNorm1d(d),
            nn.ReLU(),
            nn.Dropout(0.1),
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
        "cos_mean": cos.mean().item(),
        "H_mean": H.mean().item(),
    }

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

def apply_threshold(H_scores, threshold):
    # Lower / more negative H = more binder-like
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
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title(f"{prefix} Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{prefix}_metrics.png")
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
def evaluate_loader(loader, tcr_proj, pmhc_proj, device, alpha=1.0, beta=25.0, delta=1.0, gamma_var=1.0, eps=EPS):
    tcr_proj.eval()
    pmhc_proj.eval()

    all_H, all_cos, all_lab, all_pid, all_zT = [], [], [], [], []
    running_loss, n_steps = 0.0, 0

    for batch in loader:
        out = forward_batch(batch, tcr_proj, pmhc_proj, device, eps)
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

    H_vals = np.concatenate(all_H)
    cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_lab).astype(int)
    zT = np.concatenate(all_zT, axis=0)

    ranking_scores = -H_vals  # higher = more positive for AUROC / AUPRC
    metrics = {
        "auroc": float(roc_auc_score(labels, ranking_scores)),
        "auprc": float(average_precision_score(labels, ranking_scores)),
        "val_loss": float(running_loss / max(n_steps, 1)),
    }

    return {
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

def run_experiment_for_seed(train_loader, val_loader, test_loader, device, seed, cfg):
    set_global_seed(seed)

    sample = train_loader.dataset[0]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    D_esm = sample["emb_T"].shape[2]

    tcr_proj = ESMProjectionHead(D_esm, cfg["rL"], cfg["rD"], D, L_max=L_T).to(device)
    pmhc_proj = PMHCProjectionHead(D_esm, cfg["rL"], cfg["rD"], D, L_P_max=L_P, L_H_max=L_H, R_PH=R_PH).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": tcr_proj.parameters(), "lr": cfg["lr"]},
            {"params": pmhc_proj.parameters(), "lr": cfg["lr"]},
        ],
        weight_decay=cfg["wd"],
    )
    scheduler = make_scheduler(optimizer, NUM_EPOCHS, WARMUP_EPOCHS)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "val_auprc": [],
    }

    best_auroc = -float("inf")
    best_state = None
    bad_epochs = 0
    run_tag = f"seed_{seed}"

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
                zT, zPH,
                alpha=cfg["alpha"],
                beta=cfg["beta"],
                delta=DELTA,
                gamma_var=GAMMA_VAR,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(tcr_proj.parameters()) + list(pmhc_proj.parameters()), GRAD_CLIP_NORM)
            optimizer.step()

            running_loss += loss.item()
            n_steps += 1

        scheduler.step()

        train_loss = running_loss / max(n_steps, 1)
        val_out = evaluate_loader(
            val_loader, tcr_proj, pmhc_proj, device,
            alpha=cfg["alpha"], beta=cfg["beta"], delta=DELTA, gamma_var=GAMMA_VAR
        )
        val_loss = val_out["metrics"]["val_loss"]
        val_auroc = val_out["metrics"]["auroc"]
        val_auprc = val_out["metrics"]["auprc"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auroc"].append(val_auroc)
        history["val_auprc"].append(val_auprc)

        log.info(
            f"[{run_tag}] ep{epoch+1}/{NUM_EPOCHS} "
            f"tl={train_loss:.4f} vl={val_loss:.4f} "
            f"auroc={val_auroc:.4f} auprc={val_auprc:.4f}"
        )
        log_H_stats(f"[{run_tag}] VAL", val_out["H"], val_out["labels"])

        if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0 or epoch == 0:
            plot_H_histogram(
                val_out["H"], val_out["labels"],
                f"{run_tag}_val_H_ep{epoch+1}", FIGURE_SUBDIR
            )
            plot_cross_reactivity_zT(
                val_out["zT"], val_out["pair_ids"], val_out["labels"],
                pep_lookup_val, f"{run_tag}_val_xreact_ep{epoch+1}", FIGURE_SUBDIR, seed=seed
            )

        if val_auroc > best_auroc + 1e-4:
            best_auroc = val_auroc
            bad_epochs = 0
            best_state = {
                "epoch": epoch + 1,
                "val_auroc": val_auroc,
                "val_loss": val_loss,
                "tcr_proj": copy.deepcopy(tcr_proj.state_dict()),
                "pmhc_proj": copy.deepcopy(pmhc_proj.state_dict()),
                "history": copy.deepcopy(history),
            }
            log.info(f"[{run_tag}] -> new best AUROC={val_auroc:.4f} at ep{epoch+1}")
        else:
            if epoch + 1 >= MIN_EPOCHS_BEFORE_EARLY_STOP:
                bad_epochs += 1

        if bad_epochs >= PATIENCE:
            log.info(f"[{run_tag}] early stop at ep{epoch+1}")
            break

    if best_state is None:
        raise RuntimeError(f"No best state saved for seed {seed}")

    # Reload best model
    tcr_proj.load_state_dict(best_state["tcr_proj"])
    pmhc_proj.load_state_dict(best_state["pmhc_proj"])

    # Final val threshold on H only
    best_val_out = evaluate_loader(
        val_loader, tcr_proj, pmhc_proj, device,
        alpha=cfg["alpha"], beta=cfg["beta"], delta=DELTA, gamma_var=GAMMA_VAR
    )
    val_thr = find_best_threshold_on_H(best_val_out["H"], best_val_out["labels"])

    # Final test eval
    test_out = evaluate_loader(
        test_loader, tcr_proj, pmhc_proj, device,
        alpha=cfg["alpha"], beta=cfg["beta"], delta=DELTA, gamma_var=GAMMA_VAR
    )
    preds = apply_threshold(test_out["H"], val_thr["threshold"])
    labels = test_out["labels"]
    test_bin = compute_binary_metrics(labels, preds)

    log.info(
        f"[{run_tag}] FINAL TEST | thr={val_thr['threshold']:.6f} on H | "
        f"pred_pos={int(preds.sum())} pred_neg={int((1-preds).sum())}"
    )
    log_H_stats(f"[{run_tag}] TEST", test_out["H"], labels)
    log.info(
        f"[{run_tag}] TEST METRICS | "
        f"auroc={test_out['metrics']['auroc']:.4f} "
        f"auprc={test_out['metrics']['auprc']:.4f} "
        f"f1={test_bin['f1']:.4f} "
        f"acc={test_bin['accuracy']:.4f} "
        f"prec={test_bin['precision']:.4f} "
        f"rec={test_bin['recall']:.4f}"
    )
    log.info(f"[{run_tag}] TEST CONFUSION:\n{confusion_matrix(labels, preds)}")

    # Save artefacts per seed
    torch.save(
        {
            "seed": seed,
            "config": cfg,
            "best_epoch": best_state["epoch"],
            "val_threshold_H": val_thr,
            "history": best_state["history"],
            "test_metrics": {
                "auroc": test_out["metrics"]["auroc"],
                "auprc": test_out["metrics"]["auprc"],
                **test_bin,
            },
        },
        SAVE_DIR / f"baseline_seed_{seed}.pt"
    )

    plot_H_histogram(test_out["H"], labels, f"{run_tag}_test_H_best", FIGURE_SUBDIR, threshold=val_thr["threshold"])
    plot_cross_reactivity_zT(
        test_out["zT"], test_out["pair_ids"], labels,
        pep_lookup_test, f"{run_tag}_test_xreact_best", FIGURE_SUBDIR, seed=seed
    )
    plot_training_history(best_state["history"], FIGURE_SUBDIR, prefix=run_tag)

    return {
        "seed": seed,
        "best_epoch": best_state["epoch"],
        "val_auroc": best_val_out["metrics"]["auroc"],
        "val_auprc": best_val_out["metrics"]["auprc"],
        "val_threshold_H": val_thr["threshold"],
        "test_auroc": test_out["metrics"]["auroc"],
        "test_auprc": test_out["metrics"]["auprc"],
        "test_f1": test_bin["f1"],
        "test_accuracy": test_bin["accuracy"],
        "test_precision": test_bin["precision"],
        "test_recall": test_bin["recall"],
    }

# ============================================================
# MAIN
# ============================================================
def main():
    log.info("Starting seed-stability baseline run")
    log.info(f"Fixed config: {FIXED_CFG}")
    log.info(f"Seeds: {SEEDS}")
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

    all_results = []
    for seed in SEEDS:
        log.info("\n" + "=" * 72)
        log.info(f"Running seed {seed}")
        result = run_experiment_for_seed(train_loader, val_loader, test_loader, device, seed, FIXED_CFG)
        all_results.append(result)

    df = pd.DataFrame(all_results)
    df.to_csv(SAVE_DIR / "seed_stability_summary.csv", index=False)

    log.info("\n" + "=" * 72)
    log.info("SEED STABILITY SUMMARY")
    log.info("\n" + df.to_string(index=False))

    for metric in ["val_auroc", "test_auroc", "test_auprc", "test_f1"]:
        log.info(
            f"{metric}: mean={df[metric].mean():.4f} std={df[metric].std(ddof=1):.4f}"
        )

    log.info(f"Saved summary to {SAVE_DIR / 'seed_stability_summary.csv'}")
    log.info("Done.")

if __name__ == "__main__":
    main()