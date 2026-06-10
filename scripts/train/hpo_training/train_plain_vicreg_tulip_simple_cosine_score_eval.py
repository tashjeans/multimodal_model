#!/usr/bin/env python3
"""
Plain VICReg diagnostic run on strict TULIP-style decoys.

Purpose
-------
This script is deliberately not an HPO script. It runs one clean diagnostic
configuration and compares the trained projection model against a frozen/raw
ESM cosine baseline at every epoch.

It saves:
- per-epoch history CSV
- best checkpoint
- validation/test prediction CSVs for best checkpoint
- validation/test plain VICReg histograms for model and raw ESM
- validation/test per-peptide AUROC tables
- validation/test cross-reactivity paired-distance CSVs and plots

Default shard root:
    /home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped_tulip_decoys

Expected shard batch keys:
    emb_T, mask_T, emb_P, mask_P, emb_H, mask_H, binding_flag, pair_id

Score convention
----------------
For both model and raw ESM baseline:
    cos = cosine(TCR, pMHC)
    H = -1 - cos
    score = -H = 1 + cos

Therefore higher score = more likely binder.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# Dataset
# ============================================================

class ShardedBatchDataset(Dataset):
    """
    Loads shard_*.pt files.

    Each shard is expected to contain a list-like object of pre-batched dicts.
    The DataLoader uses batch_size=1 and collate_fn=lambda x: x[0].
    """

    def __init__(self, shards_dir: Path):
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

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        sp, j = self.index[idx]
        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp
        return self._cache_data[j]


# ============================================================
# Projection heads
# ============================================================

class ESMProjectionHead(nn.Module):
    """
    Low-rank sequence projection head.

    emb:  [B, L, D]
    mask: [B, L]
    out:  [B, d]
    """

    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float = 0.1):
        super().__init__()
        self.D = D
        self.rL = rL
        self.rD = rD
        self.d = d
        self.L_max = L_max

        self.B_c = nn.Parameter(torch.empty(D, rD))
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))

        nn.init.xavier_uniform_(self.B_c)
        nn.init.xavier_uniform_(self.A_c)
        nn.init.xavier_uniform_(self.H_c)

        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = emb.device
        B, L_pad, D_in = emb.shape

        if D_in != self.D:
            raise ValueError(f"Embedding dimension mismatch: got {D_in}, expected {self.D}")
        if L_pad > self.L_max:
            raise ValueError(f"Sequence length {L_pad} exceeds L_max {self.L_max}")

        L_true = mask.sum(dim=1)
        z_list = []

        for b in range(B):
            Lb = int(L_true[b].item())
            if Lb == 0:
                z_list.append(torch.zeros(self.d, device=device, dtype=emb.dtype))
                continue

            Xb = emb[b, :Lb, :] * mask[b, :Lb].unsqueeze(-1).float()
            Yb = Xb @ self.B_c
            Ub = self.A_c[:Lb, :].T @ Yb
            zb = Ub.reshape(-1) @ self.H_c
            z_list.append(zb)

        return self.expander(torch.stack(z_list, dim=0))


class PMHCProjectionHead(nn.Module):
    """
    Peptide and HLA are encoded separately, then concatenated.
    """

    def __init__(
        self,
        D: int,
        rL: int,
        rD: int,
        d: int,
        L_P_max: int,
        L_H_max: int,
        R_PH: float = 0.7,
        dropout: float = 0.1,
    ):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")

        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)


# ============================================================
# Loss and scoring
# ============================================================

def row_normalise(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return u / (u.norm(dim=-1, keepdim=True) + eps)


def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    return (emb * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + eps)


def vicreg_variance(u: torch.Tensor, gamma: float = 1.0, eps_var: float = 1e-4) -> torch.Tensor:
    u = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u.var(dim=0, unbiased=False) + eps_var)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u = u - u.mean(dim=0, keepdim=True)
    cov = (u.T @ u) / (B - 1)
    cov_off = cov - torch.diag_embed(torch.diag(cov))
    return (cov_off ** 2).sum() / d


def plain_vicreg_loss(
    zT: torch.Tensor,
    zPH: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma_var: float,
    eps_norm: float,
    eps_var: float,
    return_parts: bool = False,
):
    """Plain VICReg objective for the positive-only cross-modal setting.

    Invariance: raw MSE between the unnormalised TCR and pMHC projected
    embeddings. This follows the standard VICReg design: the representation
    is not L2-normalised inside the loss.

    Variance/covariance: also computed on the same unnormalised projected
    embeddings. This keeps the loss as a clean VICReg baseline and avoids
    mixing in the angular/unit-sphere constraint that was causing interpretive
    tension in the plain VICReg runs.

    The L2-normalised cosine/plain VICReg values below are diagnostics and
    evaluation scores only. They are not used to optimise this plain VICReg
    objective.
    """
    L_inv = F.mse_loss(zT, zPH)
    L_var = vicreg_variance(zT, gamma_var, eps_var) + vicreg_variance(zPH, gamma_var, eps_var)
    L_cov = vicreg_covariance(zT) + vicreg_covariance(zPH)

    loss = alpha * L_inv + beta * L_var + delta * L_cov

    if not return_parts:
        return loss

    # Diagnostics only: these are not part of the plain VICReg loss.
    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos

    parts = {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((alpha * L_inv).detach().cpu()),
        "weighted_var": float((beta * L_var).detach().cpu()),
        "weighted_cov": float((delta * L_cov).detach().cpu()),
        "H_mean": float(H.mean().detach().cpu()),
        "H_std": float(H.std(unbiased=False).detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "cos_std": float(cos.std(unbiased=False).detach().cpu()),
        "zTstd": float(zT.std(unbiased=False).detach().cpu()),
        "zPHstd": float(zPH.std(unbiased=False).detach().cpu()),
        "eTstd": float(eT.std(unbiased=False).detach().cpu()),
        "ePHstd": float(ePH.std(unbiased=False).detach().cpu()),
    }
    return loss, parts


def score_from_projected(zT: torch.Tensor, zPH: torch.Tensor, eps_norm: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos
    score = -H
    return score, H, cos


def raw_esm_score(batch, device: torch.device, eps_norm: float = 1e-8, R_PH: float = 0.7):
    """
    Frozen/raw ESM proxy baseline.

    TCR = masked mean pooled TCR embedding.
    pMHC = weighted average of masked mean pooled peptide and HLA embeddings.

    This keeps TCR and pMHC in the same ESM dimension D, making the cosine
    directly computable. It is intentionally simple and diagnostic.
    """
    T = masked_mean_pool(batch["emb_T"].to(device), batch["mask_T"].to(device), eps_norm)
    P = masked_mean_pool(batch["emb_P"].to(device), batch["mask_P"].to(device), eps_norm)
    HLA = masked_mean_pool(batch["emb_H"].to(device), batch["mask_H"].to(device), eps_norm)

    PH = R_PH * P + (1.0 - R_PH) * HLA
    score, H, cos = score_from_projected(T, PH, eps_norm)
    return score, H, cos, row_normalise(T, eps_norm)


# ============================================================
# Metrics
# ============================================================

def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def per_peptide_auroc(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})

    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        auc = float(roc_auc_score(y, s)) if valid else float("nan")
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": auc,
            "valid": bool(valid),
        })

    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]].copy()

    if len(valid_table) == 0:
        summary = {"macro": float("nan"), "weighted": float("nan"), "n_total": len(table), "n_valid": 0}
    else:
        summary = {
            "macro": float(valid_table["auroc"].mean()),
            "weighted": float(np.average(valid_table["auroc"], weights=valid_table["n"])),
            "n_total": int(len(table)),
            "n_valid": int(len(valid_table)),
        }

    return table, summary


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best = None
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        row = {
            "threshold": float(thr),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best or {"threshold": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan")}


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, prefix: str) -> Dict:
    pred = (scores >= threshold).astype(int)
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_f1": float(f1_score(labels, pred, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}_precision": float(precision_score(labels, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, pred, zero_division=0)),
        f"{prefix}_cm": confusion_matrix(labels, pred).tolist(),
    }


# ============================================================
# Evaluation and cross-reactivity
# ============================================================

def get_labels_and_ids(batch) -> Tuple[np.ndarray, List[str]]:
    labels = batch["binding_flag"]
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

    pair_ids = batch["pair_id"]
    if torch.is_tensor(pair_ids):
        pair_ids = pair_ids.detach().cpu().numpy().tolist()
    pair_ids = [str(x) for x in pair_ids]
    return labels.astype(int), pair_ids


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    loss_params: Dict,
    pair_to_peptide: Dict[str, str],
    split: str,
    R_PH: float,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    model_scores, model_H, model_cos = [], [], []
    raw_scores, raw_H, raw_cos = [], [], []
    labels_all, pair_ids_all, peptides_all = [], [], []

    # TCR embeddings for cross-reactivity among positives.
    raw_tcr_pos, model_tcr_pos, pep_pos = [], [], []

    running = {k: 0.0 for k in [
        "loss", "L_inv", "L_var", "L_cov",
        "weighted_inv", "weighted_var", "weighted_cov",
        "H_std", "cos_std", "zTstd", "zPHstd", "eTstd", "ePHstd",
    ]}
    n_steps = 0

    for batch in loader:
        zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
        zPH = pmhc_proj(
            batch["emb_P"].to(device),
            batch["mask_P"].to(device),
            batch["emb_H"].to(device),
            batch["mask_H"].to(device),
        )

        loss, parts = plain_vicreg_loss(zT, zPH, **loss_params, return_parts=True)

        s_m, H_m, cos_m = score_from_projected(zT, zPH, loss_params["eps_norm"])
        s_r, H_r, cos_r, raw_T_norm = raw_esm_score(batch, device, loss_params["eps_norm"], R_PH)
        model_T_norm = row_normalise(zT, loss_params["eps_norm"])

        labels, pair_ids = get_labels_and_ids(batch)
        peptides = np.array([pair_to_peptide.get(pid, "") for pid in pair_ids], dtype=str)
        if np.any(peptides == ""):
            missing = [pid for pid, pep in zip(pair_ids, peptides) if pep == ""][:10]
            raise KeyError(f"Missing peptide mapping for {split}. Example pair_ids: {missing}")

        model_scores.append(s_m.detach().cpu().numpy())
        model_H.append(H_m.detach().cpu().numpy())
        model_cos.append(cos_m.detach().cpu().numpy())

        raw_scores.append(s_r.detach().cpu().numpy())
        raw_H.append(H_r.detach().cpu().numpy())
        raw_cos.append(cos_r.detach().cpu().numpy())

        labels_all.append(labels)
        pair_ids_all.extend(pair_ids)
        peptides_all.append(peptides)

        pos_mask = labels == 1
        if np.any(pos_mask):
            raw_tcr_pos.append(raw_T_norm.detach().cpu().numpy()[pos_mask])
            model_tcr_pos.append(model_T_norm.detach().cpu().numpy()[pos_mask])
            pep_pos.append(peptides[pos_mask])

        running["loss"] += float(loss.detach().cpu())
        for k in running:
            if k != "loss":
                running[k] += float(parts[k])
        n_steps += 1

    labels = np.concatenate(labels_all).astype(int)
    peptides = np.concatenate(peptides_all).astype(str)

    model_scores = np.concatenate(model_scores)
    model_H = np.concatenate(model_H)
    model_cos = np.concatenate(model_cos)
    raw_scores = np.concatenate(raw_scores)
    raw_H = np.concatenate(raw_H)
    raw_cos = np.concatenate(raw_cos)

    model_pep_table, model_pep = per_peptide_auroc(labels, model_scores, peptides)
    raw_pep_table, raw_pep = per_peptide_auroc(labels, raw_scores, peptides)
    best_thr = best_f1_threshold(model_scores, labels)

    metrics = {
        f"{split}_loss": running["loss"] / max(1, n_steps),
        f"{split}_model_global_auroc": safe_auroc(labels, model_scores),
        f"{split}_raw_esm_global_auroc": safe_auroc(labels, raw_scores),
        f"{split}_delta_global_auroc": safe_auroc(labels, model_scores) - safe_auroc(labels, raw_scores),
        f"{split}_model_auprc": safe_auprc(labels, model_scores),
        f"{split}_raw_esm_auprc": safe_auprc(labels, raw_scores),
        f"{split}_model_peptide_macro_auroc": model_pep["macro"],
        f"{split}_raw_esm_peptide_macro_auroc": raw_pep["macro"],
        f"{split}_delta_peptide_macro_auroc": model_pep["macro"] - raw_pep["macro"],
        f"{split}_model_peptide_weighted_auroc": model_pep["weighted"],
        f"{split}_raw_esm_peptide_weighted_auroc": raw_pep["weighted"],
        f"{split}_delta_peptide_weighted_auroc": model_pep["weighted"] - raw_pep["weighted"],
        f"{split}_n_peptides_total": model_pep["n_total"],
        f"{split}_n_peptides_valid": model_pep["n_valid"],
        f"{split}_threshold": best_thr["threshold"],
        f"{split}_f1": best_thr["f1"],
        f"{split}_accuracy": best_thr["accuracy"],
        f"{split}_precision": best_thr["precision"],
        f"{split}_recall": best_thr["recall"],
        f"{split}_H_std": float(np.std(model_H)),
        f"{split}_raw_H_std": float(np.std(raw_H)),
    }

    for k, v in running.items():
        if k != "loss":
            metrics[f"{split}_{k}"] = v / max(1, n_steps)

    predictions = pd.DataFrame({
        "pair_id": pair_ids_all,
        "peptide": peptides,
        "label": labels,
        "model_score": model_scores,
        "model_H": model_H,
        "model_cos": model_cos,
        "raw_esm_score": raw_scores,
        "raw_esm_H": raw_H,
        "raw_esm_cos": raw_cos,
    })

    if raw_tcr_pos:
        cross = {
            "raw_tcr_pos": np.concatenate(raw_tcr_pos, axis=0),
            "model_tcr_pos": np.concatenate(model_tcr_pos, axis=0),
            "peptide_pos": np.concatenate(pep_pos).astype(str),
        }
    else:
        cross = {"raw_tcr_pos": np.empty((0, 1)), "model_tcr_pos": np.empty((0, 1)), "peptide_pos": np.array([])}

    return {
        "metrics": metrics,
        "predictions": predictions,
        "model_peptide_table": model_pep_table,
        "raw_peptide_table": raw_pep_table,
        "model_H": model_H,
        "raw_H": raw_H,
        "labels": labels,
        "cross": cross,
    }


def mean_pairwise_cosine_distance(X: np.ndarray, max_pairs: int = 20000, seed: int = 31) -> float:
    n = X.shape[0]
    if n < 2:
        return float("nan")

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in idx]

    vals = []
    for i, j in pairs:
        vals.append(1.0 - float(np.dot(X[i], X[j])))
    return float(np.mean(vals))


def cross_reactivity_table(cross: Dict, min_group_size: int = 5, seed: int = 31) -> pd.DataFrame:
    raw_X = cross["raw_tcr_pos"]
    model_X = cross["model_tcr_pos"]
    peptides = cross["peptide_pos"]

    rows = []
    for pep in sorted(set(peptides.tolist())):
        idx = np.where(peptides == pep)[0]
        if len(idx) < min_group_size:
            continue

        raw_d = mean_pairwise_cosine_distance(raw_X[idx], seed=seed)
        model_d = mean_pairwise_cosine_distance(model_X[idx], seed=seed)

        rows.append({
            "peptide": pep,
            "n_positive_tcrs": int(len(idx)),
            "raw_esm_within_peptide_mean_cosine_distance": raw_d,
            "model_within_peptide_mean_cosine_distance": model_d,
            "delta_model_minus_raw": model_d - raw_d,
        })

    return pd.DataFrame(rows)


# ============================================================
# Plotting
# ============================================================

def plot_histogram(H: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = np.asarray(H)
    labels = np.asarray(labels).astype(int)

    plt.figure(figsize=(8, 5))
    if np.any(labels == 0):
        plt.hist(H[labels == 0], bins=50, density=True, alpha=0.55, label="decoy/negative")
    if np.any(labels == 1):
        plt.hist(H[labels == 1], bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("Cosine-distance score; lower = stronger binding")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cross_reactivity(table: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 6))
    if len(table) == 0:
        plt.text(0.5, 0.5, "No peptide groups passed min_group_size", ha="center", va="center")
        plt.axis("off")
    else:
        x = table["raw_esm_within_peptide_mean_cosine_distance"].to_numpy()
        y = table["model_within_peptide_mean_cosine_distance"].to_numpy()
        plt.scatter(x, y, alpha=0.75)

        lo = min(np.min(x), np.min(y))
        hi = max(np.max(x), np.max(y))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

        plt.xlabel("Raw ESM within-peptide TCR distance")
        plt.ylabel("Model within-peptide TCR distance")
        plt.title(title)
        plt.tight_layout()

    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Config
# ============================================================

@dataclass
class RunConfig:
    embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped_tulip_decoys"
    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/plain_vicreg_tulip_simple"
    run_tag: str = "tulip_decoys_plain_vicreg"

    val_csv: str = "/home/natasha/multimodal_model/data/val/val_df_clean_pos_tulip_decoys_strict.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_df_clean_pos_tulip_decoys_strict.csv"

    seed: int = 31
    batch_size: int = 1
    num_workers: int = 0

    epochs: int = 30
    patience: int = 10
    min_epochs: int = 10

    rL: int = 8
    rD: int = 16
    d: int = 128
    R_PH: float = 0.7
    dropout: float = 0.1

    lr_tcr: float = 3e-4
    lr_pmhc: float = 3e-4
    weight_decay: float = 1e-2

    alpha: float = 25.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    eps_norm: float = 1e-8
    eps_var: float = 1e-4

    crossreact_min_group_size: int = 5


def loss_params(cfg: RunConfig) -> Dict:
    return {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "delta": cfg.delta,
        "gamma_var": cfg.gamma_var,
        "eps_norm": cfg.eps_norm,
        "eps_var": cfg.eps_var,
    }


def infer_shapes(train_ds: Dataset) -> Tuple[int, int, int, int]:
    sample = train_ds[0]
    D = sample["emb_T"].shape[2]
    L_T = sample["emb_T"].shape[1]
    L_P = sample["emb_P"].shape[1]
    L_H = sample["emb_H"].shape[1]
    print(f"Detected shapes | D={D} | L_T={L_T} | L_P={L_P} | L_H={L_H}", flush=True)
    return D, L_T, L_P, L_H


def initialise_models(cfg: RunConfig, shapes: Tuple[int, int, int, int], device: torch.device) -> Tuple[nn.Module, nn.Module]:
    D, L_T, L_P, L_H = shapes
    tcr = ESMProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
    return tcr, pmhc


def load_pair_to_peptide(csv_path: str) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain pair_id")

    peptide_col = None
    for c in ["Peptide", "peptide"]:
        if c in df.columns:
            peptide_col = c
            break
    if peptide_col is None:
        raise ValueError(f"{csv_path} must contain Peptide or peptide")

    return dict(zip(df["pair_id"].astype(str), df[peptide_col].astype(str)))


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-root", default=RunConfig.embed_root)
    parser.add_argument("--out-dir", default=RunConfig.out_dir)
    parser.add_argument("--fig-dir", default=RunConfig.fig_dir)
    parser.add_argument("--run-tag", default=RunConfig.run_tag)
    parser.add_argument("--val-csv", default=RunConfig.val_csv)
    parser.add_argument("--test-csv", default=RunConfig.test_csv)

    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--patience", type=int, default=RunConfig.patience)
    parser.add_argument("--min-epochs", type=int, default=RunConfig.min_epochs)
    parser.add_argument("--num-workers", type=int, default=RunConfig.num_workers)

    parser.add_argument("--lr", type=float, default=RunConfig.lr_tcr)
    parser.add_argument("--alpha", type=float, default=RunConfig.alpha)
    parser.add_argument("--beta", type=float, default=RunConfig.beta)
    parser.add_argument("--delta", type=float, default=RunConfig.delta)
    parser.add_argument("--dropout", type=float, default=RunConfig.dropout)

    args = parser.parse_args()

    cfg = RunConfig(
        embed_root=args.embed_root,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        run_tag=args.run_tag,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        min_epochs=args.min_epochs,
        num_workers=args.num_workers,
        lr_tcr=args.lr,
        lr_pmhc=args.lr,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        dropout=args.dropout,
    )

    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{cfg.run_tag}__run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================", flush=True)
    print("Plain VICReg diagnostic run", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Embed root: {cfg.embed_root}", flush=True)
    print(f"Output dir: {cfg.out_dir}", flush=True)
    print(f"Figure dir: {cfg.fig_dir}", flush=True)
    print(f"Val CSV: {cfg.val_csv}", flush=True)
    print(f"Test CSV: {cfg.test_csv}", flush=True)
    print(f"Loss: alpha={cfg.alpha}, beta={cfg.beta}, delta={cfg.delta}; invariance=normalised MSE/cosine distance", flush=True)
    print(f"LR={cfg.lr_tcr}, epochs={cfg.epochs}, min_epochs={cfg.min_epochs}, patience={cfg.patience}", flush=True)
    print("============================================================", flush=True)

    pair_to_peptide_val = load_pair_to_peptide(cfg.val_csv)
    pair_to_peptide_test = load_pair_to_peptide(cfg.test_csv)

    train_ds = ShardedBatchDataset(Path(cfg.embed_root) / "train")
    val_ds = ShardedBatchDataset(Path(cfg.embed_root) / "val")
    test_ds = ShardedBatchDataset(Path(cfg.embed_root) / "test")

    print(f"Loaded batches | train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}", flush=True)

    shapes = infer_shapes(train_ds)

    generator = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x[0],
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x[0],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x[0],
    )

    tcr, pmhc = initialise_models(cfg, shapes, device)
    optimizer = torch.optim.AdamW(
        [
            {"params": tcr.parameters(), "lr": cfg.lr_tcr},
            {"params": pmhc.parameters(), "lr": cfg.lr_pmhc},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    lp = loss_params(cfg)

    best = {
        "val_model_peptide_weighted_auroc": -np.inf,
        "epoch": None,
        "state": None,
        "val_eval": None,
        "bad_epochs": 0,
    }

    history = []

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()

        train_running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov"]}
        n = 0

        for batch in train_loader:
            zT = tcr(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc(
                batch["emb_P"].to(device),
                batch["mask_P"].to(device),
                batch["emb_H"].to(device),
                batch["mask_H"].to(device),
            )

            loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_running["loss"] += parts["L_total"]
            for k in train_running:
                if k != "loss":
                    train_running[k] += parts[k]
            n += 1

        scheduler.step()

        val_eval = evaluate(
            val_loader,
            tcr,
            pmhc,
            device,
            lp,
            pair_to_peptide_val,
            split="val",
            R_PH=cfg.R_PH,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_running["loss"] / max(1, n),
            "train_L_inv": train_running["L_inv"] / max(1, n),
            "train_L_var": train_running["L_var"] / max(1, n),
            "train_L_cov": train_running["L_cov"] / max(1, n),
            "train_weighted_inv": train_running["weighted_inv"] / max(1, n),
            "train_weighted_var": train_running["weighted_var"] / max(1, n),
            "train_weighted_cov": train_running["weighted_cov"] / max(1, n),
            **val_eval["metrics"],
        }
        history.append(row)

        current = row["val_model_peptide_weighted_auroc"]
        raw = row["val_raw_esm_peptide_weighted_auroc"]
        improved = (not np.isnan(current)) and current > best["val_model_peptide_weighted_auroc"] + 1e-4

        if improved:
            best = {
                "val_model_peptide_weighted_auroc": current,
                "epoch": epoch,
                "state": {
                    "tcr": copy.deepcopy(tcr.state_dict()),
                    "pmhc": copy.deepcopy(pmhc.state_dict()),
                },
                "val_eval": val_eval,
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
            f"w_inv={row['train_weighted_inv']:.4f} | "
            f"w_var={row['train_weighted_var']:.4f} | "
            f"w_cov={row['train_weighted_cov']:.4f} | "
            f"val_model_global={row['val_model_global_auroc']:.4f} | "
            f"raw_global={row['val_raw_esm_global_auroc']:.4f} | "
            f"delta_global={row['val_delta_global_auroc']:.4f} | "
            f"val_model_pep_weighted={row['val_model_peptide_weighted_auroc']:.4f} | "
            f"raw_pep_weighted={raw:.4f} | "
            f"delta_pep_weighted={row['val_delta_peptide_weighted_auroc']:.4f} | "
            f"val_auprc={row['val_model_auprc']:.4f} | "
            f"raw_auprc={row['val_raw_esm_auprc']:.4f} | "
            f"H_std={row['val_H_std']:.4f} | "
            f"eTstd={row['val_eTstd']:.4f} | "
            f"ePHstd={row['val_ePHstd']:.4f} | "
            f"best_epoch={best['epoch']} | "
            f"bad_epochs={best['bad_epochs']}",
            flush=True,
        )

        pd.DataFrame(history).to_csv(out_dir / f"{cfg.run_tag}__history.csv", index=False)

        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; min_epochs={cfg.min_epochs}, patience={cfg.patience}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint was selected.")

    # Reload best model for final validation/test outputs.
    tcr.load_state_dict(best["state"]["tcr"])
    pmhc.load_state_dict(best["state"]["pmhc"])

    val_eval = evaluate(val_loader, tcr, pmhc, device, lp, pair_to_peptide_val, "val", cfg.R_PH)
    test_eval = evaluate(test_loader, tcr, pmhc, device, lp, pair_to_peptide_test, "test", cfg.R_PH)

    val_threshold = val_eval["metrics"]["val_threshold"]
    test_at_val_threshold = threshold_metrics(test_eval["predictions"]["model_score"].to_numpy(), test_eval["predictions"]["label"].to_numpy(), val_threshold, "test_at_val_threshold")

    stem = (
        f"{cfg.run_tag}"
        f"__seed{cfg.seed}"
        f"__lr{cfg.lr_tcr}"
        f"__a{cfg.alpha}"
        f"__b{cfg.beta}"
        f"__dlt{cfg.delta}"
    )

    checkpoint_path = out_dir / f"{stem}__best.pt"
    summary_path = out_dir / f"{stem}__summary.json"

    val_pred_path = out_dir / f"{stem}__val_predictions.csv"
    test_pred_path = out_dir / f"{stem}__test_predictions.csv"
    val_model_pep_path = out_dir / f"{stem}__val_model_per_peptide.csv"
    val_raw_pep_path = out_dir / f"{stem}__val_raw_esm_per_peptide.csv"
    test_model_pep_path = out_dir / f"{stem}__test_model_per_peptide.csv"
    test_raw_pep_path = out_dir / f"{stem}__test_raw_esm_per_peptide.csv"

    val_cross = cross_reactivity_table(val_eval["cross"], cfg.crossreact_min_group_size, cfg.seed)
    test_cross = cross_reactivity_table(test_eval["cross"], cfg.crossreact_min_group_size, cfg.seed)

    val_cross_path = out_dir / f"{stem}__val_crossreactivity.csv"
    test_cross_path = out_dir / f"{stem}__test_crossreactivity.csv"

    val_eval["predictions"].to_csv(val_pred_path, index=False)
    test_eval["predictions"].to_csv(test_pred_path, index=False)
    val_eval["model_peptide_table"].to_csv(val_model_pep_path, index=False)
    val_eval["raw_peptide_table"].to_csv(val_raw_pep_path, index=False)
    test_eval["model_peptide_table"].to_csv(test_model_pep_path, index=False)
    test_eval["raw_peptide_table"].to_csv(test_raw_pep_path, index=False)
    val_cross.to_csv(val_cross_path, index=False)
    test_cross.to_csv(test_cross_path, index=False)

    plot_histogram(val_eval["model_H"], val_eval["labels"], "Validation model plain VICReg", fig_dir / f"{stem}__val_model_H_hist.png")
    plot_histogram(val_eval["raw_H"], val_eval["labels"], "Validation raw ESM cosine-distance score", fig_dir / f"{stem}__val_raw_esm_H_hist.png")
    plot_histogram(test_eval["model_H"], test_eval["labels"], "Test model plain VICReg", fig_dir / f"{stem}__test_model_H_hist.png")
    plot_histogram(test_eval["raw_H"], test_eval["labels"], "Test raw ESM cosine-distance score", fig_dir / f"{stem}__test_raw_esm_H_hist.png")

    plot_cross_reactivity(val_cross, "Validation cross-reactivity: within-peptide TCR distance", fig_dir / f"{stem}__val_crossreactivity.png")
    plot_cross_reactivity(test_cross, "Test cross-reactivity: within-peptide TCR distance", fig_dir / f"{stem}__test_crossreactivity.png")

    summary = {
        "config": asdict(cfg),
        "best_epoch_by_val_model_peptide_weighted_auroc": best["epoch"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "test_at_val_threshold": test_at_val_threshold,
        "paths": {
            "history": str(out_dir / f"{cfg.run_tag}__history.csv"),
            "checkpoint": str(checkpoint_path),
            "val_predictions": str(val_pred_path),
            "test_predictions": str(test_pred_path),
            "val_model_per_peptide": str(val_model_pep_path),
            "val_raw_esm_per_peptide": str(val_raw_pep_path),
            "test_model_per_peptide": str(test_model_pep_path),
            "test_raw_esm_per_peptide": str(test_raw_pep_path),
            "val_crossreactivity": str(val_cross_path),
            "test_crossreactivity": str(test_cross_path),
            "fig_dir": str(fig_dir),
        },
    }

    torch.save(
        {
            "config": asdict(cfg),
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
        },
        checkpoint_path,
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("============================================================", flush=True)
    print("Done.", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"History: {out_dir / f'{cfg.run_tag}__history.csv'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Validation predictions: {val_pred_path}", flush=True)
    print(f"Test predictions: {test_pred_path}", flush=True)
    print(f"Figures: {fig_dir}", flush=True)
    print("Final validation metrics:", flush=True)
    print(json.dumps(val_eval["metrics"], indent=2), flush=True)
    print("Final test metrics:", flush=True)
    print(json.dumps(test_eval["metrics"], indent=2), flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
