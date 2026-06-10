#!/usr/bin/env python3
"""
HPO from precomputed shard embeddings with VICReg + Hamiltonian + angular anti-collapse.

Swapped-decoy version
---------------------
- Train shards: /home/natasha/multimodal_model/models/embeddings/no_boltz_swapped/train
- Val shards:   /home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped/val
- Test shards:  /home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped/test
- Validation CSV: /home/natasha/multimodal_model/data/val/val_df_clean_pos_swapped_with_decoys.csv
- Test CSV:       /home/natasha/multimodal_model/data/test/test_df_clean_pos_swapped_with_decoys.csv

What this script adds versus the previous version
-------------------------------------------------
1. Separate train/val/test shard directories, because your train shards now live
   under a different embedding root from val/test.
2. alpha defaults to 1.0.
3. Raw ESM baseline metrics are computed directly from frozen embeddings and
   printed beside model metrics each epoch.
4. Per-peptide AUROC is retained for both model and raw ESM baseline.
5. Validation/test pair_id uniqueness and metadata checks are explicit.

Raw ESM baseline definition
---------------------------
For each batch:
- TCR representation = masked mean pooled emb_T.
- Peptide representation = masked mean pooled emb_P.
- HLA representation = masked mean pooled emb_H.
- pMHC representation = R_PH * peptide + (1 - R_PH) * HLA.
- raw_esm_score = cosine(TCR, pMHC), where higher means more compatible.

This baseline is intentionally simple. It answers whether the projection heads
and VICReg/Hamiltonian objective improve over a direct frozen-ESM similarity
score on the same shard data.
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
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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

class ShardedBatchTripletDataset(Dataset):
    """
    Loads precomputed shard_*.pt files.

    Assumption:
    Each shard file contains a list-like object of already-batched dictionaries.
    The DataLoader uses batch_size=1 and collate_fn=lambda x: x[0], so each
    yielded object is one original precomputed batch.
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
            raise ValueError(
                f"Sequence length exceeds projection head L_max: got {L_pad}, expected <= {self.L_max}. "
                "This suggests shards may not share a consistent padded length."
            )

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
    pMHC projection head. Peptide and HLA are encoded separately, then concatenated.
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

        self.pep_encoder = ESMProjectionHead(D=D, rL=rL, rD=rD, d=d_P, L_max=L_P_max, dropout=dropout)
        self.hla_encoder = ESMProjectionHead(D=D, rL=rL, rD=rD, d=d_H, L_max=L_H_max, dropout=dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)


# ============================================================
# Loss functions
# ============================================================

def row_normalise(u: torch.Tensor, eps_norm: float = 1e-8) -> torch.Tensor:
    return u / (u.norm(dim=-1, keepdim=True) + eps_norm)


def vicreg_variance(u: torch.Tensor, gamma: float = 1.0, eps_var: float = 1e-4) -> torch.Tensor:
    u_centered = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u_centered.var(dim=0, unbiased=False) + eps_var)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u_centered = u - u.mean(dim=0, keepdim=True)
    cov = (u_centered.T @ u_centered) / (B - 1)
    cov_off = cov - torch.diag_embed(torch.diag(cov))
    return (cov_off ** 2).sum() / d


def angular_variance(e: torch.Tensor, gamma_ang: float = 0.05, eps_var: float = 1e-4) -> torch.Tensor:
    e_centered = e - e.mean(dim=0, keepdim=True)
    std = torch.sqrt(e_centered.var(dim=0, unbiased=False) + eps_var)
    return F.relu(gamma_ang - std).mean()


def vicreg_hamiltonian_loss(
    zT_raw: torch.Tensor,
    zPH_raw: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 25.0,
    delta: float = 1.0,
    lambda_ang: float = 25.0,
    gamma_var: float = 1.0,
    gamma_ang: float = 0.05,
    eps_norm: float = 1e-8,
    eps_var: float = 1e-4,
    return_parts: bool = False,
):
    eT = row_normalise(zT_raw, eps_norm=eps_norm)
    ePH = row_normalise(zPH_raw, eps_norm=eps_norm)

    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos

    L_inv = H.mean()
    L_var = vicreg_variance(zT_raw, gamma=gamma_var, eps_var=eps_var) + vicreg_variance(zPH_raw, gamma=gamma_var, eps_var=eps_var)
    L_cov = vicreg_covariance(zT_raw) + vicreg_covariance(zPH_raw)
    L_ang = angular_variance(eT, gamma_ang=gamma_ang, eps_var=eps_var) + angular_variance(ePH, gamma_ang=gamma_ang, eps_var=eps_var)

    loss = alpha * L_inv + beta * L_var + delta * L_cov + lambda_ang * L_ang

    if not return_parts:
        return loss

    parts = {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "L_ang": float(L_ang.detach().cpu()),
        "w_inv": float((alpha * L_inv).detach().cpu()),
        "w_var": float((beta * L_var).detach().cpu()),
        "w_cov": float((delta * L_cov).detach().cpu()),
        "w_ang": float((lambda_ang * L_ang).detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "cos_std": float(cos.std(unbiased=False).detach().cpu()),
        "H_mean": float(H.mean().detach().cpu()),
        "H_std": float(H.std(unbiased=False).detach().cpu()),
        "zTstd": float(zT_raw.std(unbiased=False).detach().cpu()),
        "zPHstd": float(zPH_raw.std(unbiased=False).detach().cpu()),
        "eTstd": float(eT.std(unbiased=False).detach().cpu()),
        "ePHstd": float(ePH.std(unbiased=False).detach().cpu()),
    }
    return loss, parts


# ============================================================
# Evaluation helpers
# ============================================================

@torch.no_grad()
def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.float()
    denom = mask.sum(dim=1, keepdim=True).clamp_min(eps)
    return (emb * mask.unsqueeze(-1)).sum(dim=1) / denom


@torch.no_grad()
def raw_esm_batch_score(batch, device: torch.device, R_PH: float = 0.7, eps_norm: float = 1e-8) -> Dict:
    emb_T = batch["emb_T"].to(device)
    mask_T = batch["mask_T"].to(device)
    emb_P = batch["emb_P"].to(device)
    mask_P = batch["mask_P"].to(device)
    emb_H = batch["emb_H"].to(device)
    mask_H = batch["mask_H"].to(device)

    t = masked_mean_pool(emb_T, mask_T)
    p = masked_mean_pool(emb_P, mask_P)
    h = masked_mean_pool(emb_H, mask_H)
    ph = R_PH * p + (1.0 - R_PH) * h

    t = row_normalise(t, eps_norm=eps_norm)
    ph = row_normalise(ph, eps_norm=eps_norm)
    score = (t * ph).sum(dim=-1)
    H = -1.0 - score

    labels = batch["binding_flag"]
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    return {
        "score": score.detach().cpu().numpy(),
        "H": H.detach().cpu().numpy(),
        "labels": labels,
        "pair_id": [str(x) for x in batch["pair_id"]],
    }


@torch.no_grad()
def forward_batch(batch, tcr_proj: nn.Module, pmhc_proj: nn.Module, device: torch.device, eps_norm: float = 1e-8):
    zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
    zPH = pmhc_proj(batch["emb_P"].to(device), batch["mask_P"].to(device), batch["emb_H"].to(device), batch["mask_H"].to(device))

    eT = row_normalise(zT, eps_norm=eps_norm)
    ePH = row_normalise(zPH, eps_norm=eps_norm)

    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos
    score = -H

    labels = batch["binding_flag"]
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else np.array(labels)

    return {
        "zT": zT,
        "zPH": zPH,
        "eT": eT,
        "ePH": ePH,
        "cos": cos.detach().cpu().numpy(),
        "H": H.detach().cpu().numpy(),
        "score": score.detach().cpu().numpy(),
        "labels": labels,
        "pair_id": [str(x) for x in batch["pair_id"]],
    }


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def per_peptide_auroc(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray) -> Dict:
    rows = []
    frame = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in frame.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        is_valid = len(np.unique(y)) >= 2
        auc = float(roc_auc_score(y, s)) if is_valid else float("nan")
        rows.append({"peptide": pep, "n": int(len(grp)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "auroc": auc, "valid": bool(is_valid)})
    per_pep = pd.DataFrame(rows)
    valid = per_pep[per_pep["valid"]]
    macro = float(valid["auroc"].mean()) if len(valid) else float("nan")
    weighted = float(np.average(valid["auroc"], weights=valid["n"])) if len(valid) else float("nan")
    return {
        "macro": macro,
        "weighted": weighted,
        "n_peptides_total": int(len(per_pep)),
        "n_peptides_valid": int(len(valid)),
        "table": per_pep.sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True),
    }


def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict:
    best = None
    for thr in np.unique(scores):
        preds = (scores >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        if best is None or f1 > best["f1"]:
            best = {"threshold": float(thr), "f1": float(f1), "accuracy": float(acc), "precision": float(precision), "recall": float(recall)}
    return best if best is not None else {"threshold": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan")}


@torch.no_grad()
def evaluate_raw_esm(
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    pair_to_peptide: Optional[Dict[str, str]] = None,
    R_PH: float = 0.7,
    eps_norm: float = 1e-8,
) -> Dict:
    all_scores, all_H, all_labels, all_pair_ids = [], [], [], []
    for batch in loader:
        out = raw_esm_batch_score(batch=batch, device=device, R_PH=R_PH, eps_norm=eps_norm)
        all_scores.append(out["score"])
        all_H.append(out["H"])
        all_labels.append(out["labels"])
        all_pair_ids.extend(out["pair_id"])

    scores = np.concatenate(all_scores)
    H_vals = np.concatenate(all_H)
    labels = np.concatenate(all_labels).astype(int)

    peptides = None
    pep_auc = None
    if pair_to_peptide is not None:
        peptides = np.array([pair_to_peptide.get(pid, "") for pid in all_pair_ids], dtype=str)
        if np.any(peptides == ""):
            missing = sorted({pid for pid, pep in zip(all_pair_ids, peptides) if pep == ""})[:10]
            raise KeyError(f"Missing peptide mapping for raw ESM {split_name}. Example pair_ids: {missing}")
        pep_auc = per_peptide_auroc(labels=labels, scores=scores, peptides=peptides)

    metrics = {
        f"{split_name}_raw_esm_auroc": safe_auroc(labels, scores),
        f"{split_name}_raw_esm_auprc": safe_auprc(labels, scores),
        f"{split_name}_raw_esm_score_mean": float(np.mean(scores)),
        f"{split_name}_raw_esm_score_std": float(np.std(scores)),
        f"{split_name}_raw_esm_H_mean": float(np.mean(H_vals)),
        f"{split_name}_raw_esm_H_std": float(np.std(H_vals)),
    }

    if pep_auc is not None:
        metrics[f"{split_name}_raw_esm_auroc_per_peptide_macro"] = pep_auc["macro"]
        metrics[f"{split_name}_raw_esm_auroc_per_peptide_weighted"] = pep_auc["weighted"]
        metrics[f"{split_name}_raw_esm_n_peptides_total"] = pep_auc["n_peptides_total"]
        metrics[f"{split_name}_raw_esm_n_peptides_valid_for_auroc"] = pep_auc["n_peptides_valid"]

    threshold_metrics = find_best_threshold(scores, labels)
    for k, v in threshold_metrics.items():
        metrics[f"{split_name}_raw_esm_{k}"] = v

    return {
        "scores": scores,
        "H": H_vals,
        "labels": labels,
        "peptides": peptides,
        "per_peptide_table": pep_auc["table"] if pep_auc is not None else pd.DataFrame(),
        "metrics": metrics,
    }


@torch.no_grad()
def evaluate(loader: DataLoader, tcr_proj: nn.Module, pmhc_proj: nn.Module, device: torch.device, loss_params: Dict, split_name: str = "val", pair_to_peptide: Optional[Dict[str, str]] = None) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    all_scores, all_H, all_cos, all_labels, all_pair_ids = [], [], [], [], []
    running = {"loss": 0.0, "L_inv": 0.0, "L_var": 0.0, "L_cov": 0.0, "L_ang": 0.0, "w_inv": 0.0, "w_var": 0.0, "w_cov": 0.0, "w_ang": 0.0, "zTstd": 0.0, "zPHstd": 0.0, "eTstd": 0.0, "ePHstd": 0.0}
    n_steps = 0

    for batch in loader:
        out = forward_batch(batch=batch, tcr_proj=tcr_proj, pmhc_proj=pmhc_proj, device=device, eps_norm=loss_params["eps_norm"])
        loss, parts = vicreg_hamiltonian_loss(out["zT"], out["zPH"], **loss_params, return_parts=True)

        all_scores.append(out["score"])
        all_H.append(out["H"])
        all_cos.append(out["cos"])
        all_labels.append(out["labels"])
        all_pair_ids.extend(out["pair_id"])

        running["loss"] += float(loss.detach().cpu())
        for key in ["L_inv", "L_var", "L_cov", "L_ang", "w_inv", "w_var", "w_cov", "w_ang", "zTstd", "zPHstd", "eTstd", "ePHstd"]:
            running[key] += parts[key]
        n_steps += 1

    scores = np.concatenate(all_scores)
    H_vals = np.concatenate(all_H)
    cos_vals = np.concatenate(all_cos)
    labels = np.concatenate(all_labels).astype(int)

    peptides = None
    pep_auc = None
    if pair_to_peptide is not None:
        peptides = np.array([pair_to_peptide.get(pid, "") for pid in all_pair_ids], dtype=str)
        if np.any(peptides == ""):
            missing = sorted({pid for pid, pep in zip(all_pair_ids, peptides) if pep == ""})[:10]
            raise KeyError(f"Missing peptide mapping for {split_name}. Example pair_ids: {missing}")
        pep_auc = per_peptide_auroc(labels=labels, scores=scores, peptides=peptides)

    metrics = {
        f"{split_name}_loss": float(running["loss"] / max(1, n_steps)),
        f"{split_name}_auroc": safe_auroc(labels, scores),
        f"{split_name}_auprc": safe_auprc(labels, scores),
        f"{split_name}_H_mean": float(np.mean(H_vals)),
        f"{split_name}_H_std": float(np.std(H_vals)),
        f"{split_name}_cos_mean": float(np.mean(cos_vals)),
        f"{split_name}_cos_std": float(np.std(cos_vals)),
    }
    for key in ["L_inv", "L_var", "L_cov", "L_ang", "w_inv", "w_var", "w_cov", "w_ang", "zTstd", "zPHstd", "eTstd", "ePHstd"]:
        metrics[f"{split_name}_{key}"] = float(running[key] / max(1, n_steps))

    if pep_auc is not None:
        metrics[f"{split_name}_auroc_per_peptide_macro"] = pep_auc["macro"]
        metrics[f"{split_name}_auroc_per_peptide_weighted"] = pep_auc["weighted"]
        metrics[f"{split_name}_n_peptides_total"] = pep_auc["n_peptides_total"]
        metrics[f"{split_name}_n_peptides_valid_for_auroc"] = pep_auc["n_peptides_valid"]

    threshold_metrics = find_best_threshold(scores, labels)
    for k, v in threshold_metrics.items():
        metrics[f"{split_name}_{k}"] = v

    return {
        "scores": scores,
        "H": H_vals,
        "cos": cos_vals,
        "labels": labels,
        "peptides": peptides,
        "per_peptide_table": pep_auc["table"] if pep_auc is not None else pd.DataFrame(),
        "metrics": metrics,
    }


def compute_threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, prefix: str) -> Dict:
    preds = (scores >= threshold).astype(int)
    return {
        f"{prefix}_f1": float(f1_score(labels, preds, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(labels, preds)),
        f"{prefix}_precision": float(precision_score(labels, preds, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, preds, zero_division=0)),
        f"{prefix}_cm": confusion_matrix(labels, preds).tolist(),
        f"{prefix}_threshold_from_val": float(threshold),
    }


def is_not_collapsed(metrics: Dict, split_name: str = "val") -> bool:
    H_std = metrics.get(f"{split_name}_H_std", 0.0)
    cos_std = metrics.get(f"{split_name}_cos_std", 0.0)
    eTstd = metrics.get(f"{split_name}_eTstd", 0.0)
    ePHstd = metrics.get(f"{split_name}_ePHstd", 0.0)
    return H_std >= 0.01 and cos_std >= 0.01 and eTstd >= 0.005 and ePHstd >= 0.005


# ============================================================
# Plotting
# ============================================================

def plot_hamiltonian_histogram(H: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H = np.asarray(H)
    labels = np.asarray(labels).astype(int)
    plt.figure(figsize=(8, 5))
    neg = H[labels == 0]
    pos = H[labels == 1]
    if len(neg) > 0:
        plt.hist(neg, bins=50, density=True, alpha=0.55, label="negative")
    if len(pos) > 0:
        plt.hist(pos, bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("Hamiltonian H (lower = stronger binding)")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Configuration
# ============================================================

@dataclass
class RunConfig:
    train_shards_dir: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped/train"
    val_shards_dir: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped/val"
    test_shards_dir: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped/test"

    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/vicreg_hamiltonian_swapped"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/vicreg_hamiltonian_swapped"
    run_tag: str = "swapped_decoy_validation"

    val_csv: str = "/home/natasha/multimodal_model/data/val/val_df_clean_pos_swapped_with_decoys.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_df_clean_pos_swapped_with_decoys.csv"

    seed: int = 31
    batch_size: int = 1
    num_workers: int = 0
    num_epochs: int = 20
    patience: int = 5

    R_PH: float = 0.7
    dropout: float = 0.1

    alpha: float = 1.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    lambda_ang: float = 25.0
    gamma_ang: float = 0.05
    eps_norm: float = 1e-8
    eps_var: float = 1e-4


def build_search_space() -> List[Dict]:
    return [{"rL": 8, "rD": 16, "d": 128, "lr_tcr": 1e-4, "lr_pmhc": 1e-4, "weight_decay": 1e-2}]


def make_loss_params(cfg: RunConfig) -> Dict:
    return {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "delta": cfg.delta,
        "lambda_ang": cfg.lambda_ang,
        "gamma_var": cfg.gamma_var,
        "gamma_ang": cfg.gamma_ang,
        "eps_norm": cfg.eps_norm,
        "eps_var": cfg.eps_var,
    }


# ============================================================
# Training
# ============================================================

def infer_shapes(train_ds: Dataset) -> Tuple[int, int, int, int]:
    sample = train_ds[0]
    D = sample["emb_T"].shape[2]
    L_T_max = sample["emb_T"].shape[1]
    L_P_max = sample["emb_P"].shape[1]
    L_H_max = sample["emb_H"].shape[1]
    print(f"Detected shapes | D={D} | L_T_max={L_T_max} | L_P_max={L_P_max} | L_H_max={L_H_max}", flush=True)
    return D, L_T_max, L_P_max, L_H_max


def initialise_models(cfg: RunConfig, hp: Dict, shapes: Tuple[int, int, int, int], device: torch.device) -> Tuple[nn.Module, nn.Module]:
    D, L_T_max, L_P_max, L_H_max = shapes
    tcr_proj = ESMProjectionHead(D=D, rL=hp["rL"], rD=hp["rD"], d=hp["d"], L_max=L_T_max, dropout=cfg.dropout).to(device)
    pmhc_proj = PMHCProjectionHead(D=D, rL=hp["rL"], rD=hp["rD"], d=hp["d"], L_P_max=L_P_max, L_H_max=L_H_max, R_PH=cfg.R_PH, dropout=cfg.dropout).to(device)
    return tcr_proj, pmhc_proj


def run_one(cfg: RunConfig, hp: Dict, shapes: Tuple[int, int, int, int], train_loader: DataLoader, val_loader: DataLoader, device: torch.device, pair_to_peptide_val: Dict[str, str], val_raw_esm: Dict):
    tcr_proj, pmhc_proj = initialise_models(cfg, hp, shapes, device)

    optim = torch.optim.AdamW(
        [{"params": tcr_proj.parameters(), "lr": hp["lr_tcr"]}, {"params": pmhc_proj.parameters(), "lr": hp["lr_pmhc"]}],
        weight_decay=hp["weight_decay"],
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.num_epochs)
    loss_params = make_loss_params(cfg)

    best = {"selection_auroc": -1.0, "state": None, "bad_epochs": 0, "selected_by_noncollapsed_rule": False}
    fallback_best = {"selection_auroc": -1.0, "state": None}
    history = []

    print("============================================================", flush=True)
    print("Starting VICReg Hamiltonian swapped-decoy run", flush=True)
    print(f"Hyperparameters: {hp}", flush=True)
    print(f"Loss params: {loss_params}", flush=True)
    print("Raw ESM validation baseline:", val_raw_esm["metrics"], flush=True)
    print("============================================================", flush=True)

    for ep in range(cfg.num_epochs):
        tcr_proj.train()
        pmhc_proj.train()
        tr_loss = tr_L_inv = tr_L_var = tr_L_cov = tr_L_ang = 0.0
        tr_w_inv = tr_w_var = tr_w_cov = tr_w_ang = 0.0
        n = 0

        for batch in train_loader:
            zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc_proj(batch["emb_P"].to(device), batch["mask_P"].to(device), batch["emb_H"].to(device), batch["mask_H"].to(device))
            loss, parts = vicreg_hamiltonian_loss(zT, zPH, **loss_params, return_parts=True)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            tr_loss += parts["L_total"]
            tr_L_inv += parts["L_inv"]
            tr_L_var += parts["L_var"]
            tr_L_cov += parts["L_cov"]
            tr_L_ang += parts["L_ang"]
            tr_w_inv += parts["w_inv"]
            tr_w_var += parts["w_var"]
            tr_w_cov += parts["w_cov"]
            tr_w_ang += parts["w_ang"]
            n += 1

        val = evaluate(loader=val_loader, tcr_proj=tcr_proj, pmhc_proj=pmhc_proj, device=device, loss_params=loss_params, split_name="val", pair_to_peptide=pair_to_peptide_val)

        row = {
            "epoch": ep + 1,
            "train_loss": tr_loss / max(1, n),
            "train_L_inv": tr_L_inv / max(1, n),
            "train_L_var": tr_L_var / max(1, n),
            "train_L_cov": tr_L_cov / max(1, n),
            "train_L_ang": tr_L_ang / max(1, n),
            "train_w_inv": tr_w_inv / max(1, n),
            "train_w_var": tr_w_var / max(1, n),
            "train_w_cov": tr_w_cov / max(1, n),
            "train_w_ang": tr_w_ang / max(1, n),
            **val["metrics"],
            **val_raw_esm["metrics"],
        }

        row["val_delta_auroc_vs_raw_esm"] = row["val_auroc"] - row["val_raw_esm_auroc"]
        row["val_delta_auroc_pep_macro_vs_raw_esm"] = row["val_auroc_per_peptide_macro"] - row["val_raw_esm_auroc_per_peptide_macro"]
        row["val_delta_auprc_vs_raw_esm"] = row["val_auprc"] - row["val_raw_esm_auprc"]
        row["noncollapsed"] = is_not_collapsed(val["metrics"], split_name="val")
        history.append(row)

        print(
            f"Epoch {ep + 1}/{cfg.num_epochs} | "
            f"train_loss={row['train_loss']:.4f} | "
            f"train_inv={row['train_L_inv']:.4f} | "
            f"train_var={row['train_L_var']:.4f} | "
            f"train_cov={row['train_L_cov']:.4f} | "
            f"train_ang={row['train_L_ang']:.4f} | "
            f"w_inv={row['train_w_inv']:.4f} | "
            f"w_var={row['train_w_var']:.4f} | "
            f"w_cov={row['train_w_cov']:.4f} | "
            f"w_ang={row['train_w_ang']:.4f} | "
            f"val_auroc_pep_macro={row['val_auroc_per_peptide_macro']:.4f} | "
            f"raw_esm_pep_macro={row['val_raw_esm_auroc_per_peptide_macro']:.4f} | "
            f"delta_pep={row['val_delta_auroc_pep_macro_vs_raw_esm']:.4f} | "
            f"val_auroc_global={row['val_auroc']:.4f} | "
            f"raw_esm_global={row['val_raw_esm_auroc']:.4f} | "
            f"delta_global={row['val_delta_auroc_vs_raw_esm']:.4f} | "
            f"val_auprc={row['val_auprc']:.4f} | "
            f"raw_esm_auprc={row['val_raw_esm_auprc']:.4f} | "
            f"H_std={row['val_H_std']:.4f} | "
            f"raw_H_std={row['val_raw_esm_H_std']:.4f} | "
            f"L_ang={row['val_L_ang']:.4f} | "
            f"eTstd={row['val_eTstd']:.4f} | "
            f"ePHstd={row['val_ePHstd']:.4f} | "
            f"noncollapsed={row['noncollapsed']}",
            flush=True,
        )

        val_auroc = row["val_auroc_per_peptide_macro"]
        if not np.isnan(val_auroc) and val_auroc > fallback_best["selection_auroc"] + 1e-4:
            fallback_best["selection_auroc"] = val_auroc
            fallback_best["state"] = {"tcr": copy.deepcopy(tcr_proj.state_dict()), "pmhc": copy.deepcopy(pmhc_proj.state_dict()), "val": val, "epoch": ep + 1, "noncollapsed": row["noncollapsed"]}

        if row["noncollapsed"] and not np.isnan(val_auroc) and val_auroc > best["selection_auroc"] + 1e-4:
            best["selection_auroc"] = val_auroc
            best["state"] = {"tcr": copy.deepcopy(tcr_proj.state_dict()), "pmhc": copy.deepcopy(pmhc_proj.state_dict()), "val": val, "epoch": ep + 1, "noncollapsed": row["noncollapsed"]}
            best["bad_epochs"] = 0
            best["selected_by_noncollapsed_rule"] = True
        else:
            best["bad_epochs"] += 1

        sched.step()
        if best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {ep + 1}", flush=True)
            break

    if best["state"] is None:
        print("Warning: no checkpoint passed the non-collapse filter. Using best raw validation AUROC checkpoint as fallback.", flush=True)
        best["state"] = fallback_best["state"]
        best["selection_auroc"] = fallback_best["selection_auroc"]
        best["selected_by_noncollapsed_rule"] = False

    return best, history


# ============================================================
# Main
# ============================================================

def load_and_validate_csv(path: str, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_cols = {"pair_id", "Peptide", "binding_flag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{split} CSV is missing columns {missing}: {path}")
    pair_ids = df["pair_id"].astype(str)
    if pair_ids.duplicated().any():
        dupes = pair_ids[pair_ids.duplicated()].head(10).tolist()
        raise ValueError(f"{split} CSV contains duplicate pair_id values. Examples: {dupes}")
    print(f"{split} CSV rows: {len(df)}", flush=True)
    print(f"{split} label counts: {df['binding_flag'].value_counts().to_dict()}", flush=True)
    if "decoy_type" in df.columns:
        print(f"{split} decoy types: {df['decoy_type'].value_counts().to_dict()}", flush=True)
    return df


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train-shards-dir", default=RunConfig.train_shards_dir)
    ap.add_argument("--val-shards-dir", default=RunConfig.val_shards_dir)
    ap.add_argument("--test-shards-dir", default=RunConfig.test_shards_dir)
    ap.add_argument("--out-dir", default=RunConfig.out_dir)
    ap.add_argument("--fig-dir", default=RunConfig.fig_dir)
    ap.add_argument("--run-tag", default=RunConfig.run_tag)
    ap.add_argument("--val-csv", default=RunConfig.val_csv)
    ap.add_argument("--test-csv", default=RunConfig.test_csv)

    ap.add_argument("--seed", type=int, default=RunConfig.seed)
    ap.add_argument("--epochs", type=int, default=RunConfig.num_epochs)
    ap.add_argument("--patience", type=int, default=RunConfig.patience)
    ap.add_argument("--num-workers", type=int, default=RunConfig.num_workers)

    ap.add_argument("--alpha", type=float, default=RunConfig.alpha)
    ap.add_argument("--beta", type=float, default=RunConfig.beta)
    ap.add_argument("--delta", type=float, default=RunConfig.delta)
    ap.add_argument("--lambda-ang", type=float, default=RunConfig.lambda_ang)
    ap.add_argument("--gamma-var", type=float, default=RunConfig.gamma_var)
    ap.add_argument("--gamma-ang", type=float, default=RunConfig.gamma_ang)
    ap.add_argument("--dropout", type=float, default=RunConfig.dropout)
    ap.add_argument("--R-PH", type=float, default=RunConfig.R_PH)

    args = ap.parse_args()

    cfg = RunConfig(
        train_shards_dir=args.train_shards_dir,
        val_shards_dir=args.val_shards_dir,
        test_shards_dir=args.test_shards_dir,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        run_tag=args.run_tag,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        seed=args.seed,
        num_epochs=args.epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
        lambda_ang=args.lambda_ang,
        gamma_var=args.gamma_var,
        gamma_ang=args.gamma_ang,
        dropout=args.dropout,
        R_PH=args.R_PH,
    )

    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir)
    fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("============================================================", flush=True)
    print("VICReg Hamiltonian from swapped-decoy shards", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Train shards: {cfg.train_shards_dir}", flush=True)
    print(f"Val shards: {cfg.val_shards_dir}", flush=True)
    print(f"Test shards: {cfg.test_shards_dir}", flush=True)
    print(f"Output dir: {cfg.out_dir}", flush=True)
    print(f"Figure dir: {cfg.fig_dir}", flush=True)
    print(f"Val CSV: {cfg.val_csv}", flush=True)
    print(f"Test CSV: {cfg.test_csv}", flush=True)
    print(f"Seed: {cfg.seed}", flush=True)
    print("============================================================", flush=True)

    val_df = load_and_validate_csv(cfg.val_csv, "Validation")
    test_df = load_and_validate_csv(cfg.test_csv, "Test")
    pair_to_peptide_val = dict(zip(val_df["pair_id"].astype(str), val_df["Peptide"].astype(str)))
    pair_to_peptide_test = dict(zip(test_df["pair_id"].astype(str), test_df["Peptide"].astype(str)))

    train_ds = ShardedBatchTripletDataset(Path(cfg.train_shards_dir))
    val_ds = ShardedBatchTripletDataset(Path(cfg.val_shards_dir))
    test_ds = ShardedBatchTripletDataset(Path(cfg.test_shards_dir))

    print(f"Loaded datasets | train_batches={len(train_ds)} | val_batches={len(val_ds)} | test_batches={len(test_ds)}", flush=True)

    shapes = infer_shapes(train_ds)
    gen = torch.Generator().manual_seed(cfg.seed)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=lambda x: x[0], generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=lambda x: x[0])
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=lambda x: x[0])

    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Computing raw ESM baselines...", flush=True)
    val_raw_esm = evaluate_raw_esm(val_loader, device=device, split_name="val", pair_to_peptide=pair_to_peptide_val, R_PH=cfg.R_PH, eps_norm=cfg.eps_norm)
    test_raw_esm = evaluate_raw_esm(test_loader, device=device, split_name="test", pair_to_peptide=pair_to_peptide_test, R_PH=cfg.R_PH, eps_norm=cfg.eps_norm)

    pd.DataFrame([val_raw_esm["metrics"]]).to_csv(out_dir / f"raw_esm_val_metrics__{cfg.run_tag}.csv", index=False)
    pd.DataFrame([test_raw_esm["metrics"]]).to_csv(out_dir / f"raw_esm_test_metrics__{cfg.run_tag}.csv", index=False)
    if isinstance(val_raw_esm.get("per_peptide_table"), pd.DataFrame):
        val_raw_esm["per_peptide_table"].to_csv(out_dir / f"raw_esm_val_per_peptide_stats__{cfg.run_tag}.csv", index=False)
    if isinstance(test_raw_esm.get("per_peptide_table"), pd.DataFrame):
        test_raw_esm["per_peptide_table"].to_csv(out_dir / f"raw_esm_test_per_peptide_stats__{cfg.run_tag}.csv", index=False)

    all_rows = []
    best_global = None
    search_space = build_search_space()

    for run_idx, hp in enumerate(search_space, start=1):
        print(f"===== VICReg Hamiltonian run {run_idx}/{len(search_space)} =====", flush=True)
        print(hp, flush=True)

        best, hist = run_one(cfg=cfg, hp=hp, shapes=shapes, train_loader=train_loader, val_loader=val_loader, device=device, pair_to_peptide_val=pair_to_peptide_val, val_raw_esm=val_raw_esm)

        if best["state"] is None:
            raise RuntimeError("No valid checkpoint state was produced.")

        tcr, pmhc = initialise_models(cfg, hp, shapes, device)
        tcr.load_state_dict(best["state"]["tcr"])
        pmhc.load_state_dict(best["state"]["pmhc"])

        loss_params = make_loss_params(cfg)
        val_eval = best["state"]["val"]
        test_eval = evaluate(loader=test_loader, tcr_proj=tcr, pmhc_proj=pmhc, device=device, loss_params=loss_params, split_name="test", pair_to_peptide=pair_to_peptide_test)

        val_threshold = val_eval["metrics"]["val_threshold"]
        test_threshold_metrics = compute_threshold_metrics(scores=test_eval["scores"], labels=test_eval["labels"], threshold=val_threshold, prefix="test")

        stem = (
            f"{cfg.run_tag}"
            f"__seed{cfg.seed}"
            f"__rL{hp['rL']}"
            f"__rD{hp['rD']}"
            f"__d{hp['d']}"
            f"__a{cfg.alpha}"
            f"__b{cfg.beta}"
            f"__c{cfg.delta}"
            f"__ang{cfg.lambda_ang}"
            f"__gang{cfg.gamma_ang}"
        )

        history_path = out_dir / f"{stem}__history.csv"
        checkpoint_path = out_dir / f"{stem}__best.pt"
        val_pep_stats_path = out_dir / f"{stem}__val_per_peptide_stats.csv"
        test_pep_stats_path = out_dir / f"{stem}__test_per_peptide_stats.csv"

        pd.DataFrame(hist).to_csv(history_path, index=False)
        if isinstance(val_eval.get("per_peptide_table"), pd.DataFrame):
            val_eval["per_peptide_table"].to_csv(val_pep_stats_path, index=False)
        if isinstance(test_eval.get("per_peptide_table"), pd.DataFrame):
            test_eval["per_peptide_table"].to_csv(test_pep_stats_path, index=False)

        plot_hamiltonian_histogram(H=val_eval["H"], labels=val_eval["labels"], title="Validation Hamiltonian distribution", out_path=fig_dir / f"{stem}__val_H_hist.png")
        plot_hamiltonian_histogram(H=test_eval["H"], labels=test_eval["labels"], title="Test Hamiltonian distribution", out_path=fig_dir / f"{stem}__test_H_hist.png")
        plot_hamiltonian_histogram(H=val_raw_esm["H"], labels=val_raw_esm["labels"], title="Validation raw ESM Hamiltonian proxy distribution", out_path=fig_dir / f"{stem}__val_raw_esm_H_hist.png")
        plot_hamiltonian_histogram(H=test_raw_esm["H"], labels=test_raw_esm["labels"], title="Test raw ESM Hamiltonian proxy distribution", out_path=fig_dir / f"{stem}__test_raw_esm_H_hist.png")

        row = {
            **hp,
            "seed": cfg.seed,
            "alpha": cfg.alpha,
            "beta": cfg.beta,
            "delta": cfg.delta,
            "lambda_ang": cfg.lambda_ang,
            "gamma_var": cfg.gamma_var,
            "gamma_ang": cfg.gamma_ang,
            "best_epoch": best["state"]["epoch"],
            "selected_by_noncollapsed_rule": best["selected_by_noncollapsed_rule"],
            "checkpoint_noncollapsed": best["state"]["noncollapsed"],
            **val_eval["metrics"],
            **test_eval["metrics"],
            **val_raw_esm["metrics"],
            **test_raw_esm["metrics"],
            **test_threshold_metrics,
            "val_delta_auroc_vs_raw_esm": val_eval["metrics"]["val_auroc"] - val_raw_esm["metrics"]["val_raw_esm_auroc"],
            "val_delta_auroc_pep_macro_vs_raw_esm": val_eval["metrics"]["val_auroc_per_peptide_macro"] - val_raw_esm["metrics"]["val_raw_esm_auroc_per_peptide_macro"],
            "val_delta_auprc_vs_raw_esm": val_eval["metrics"]["val_auprc"] - val_raw_esm["metrics"]["val_raw_esm_auprc"],
            "test_delta_auroc_vs_raw_esm": test_eval["metrics"]["test_auroc"] - test_raw_esm["metrics"]["test_raw_esm_auroc"],
            "test_delta_auroc_pep_macro_vs_raw_esm": test_eval["metrics"]["test_auroc_per_peptide_macro"] - test_raw_esm["metrics"]["test_raw_esm_auroc_per_peptide_macro"],
            "test_delta_auprc_vs_raw_esm": test_eval["metrics"]["test_auprc"] - test_raw_esm["metrics"]["test_raw_esm_auprc"],
            "history_path": str(history_path),
            "checkpoint_path": str(checkpoint_path),
            "val_per_peptide_stats_path": str(val_pep_stats_path),
            "test_per_peptide_stats_path": str(test_pep_stats_path),
            "val_hist_path": str(fig_dir / f"{stem}__val_H_hist.png"),
            "test_hist_path": str(fig_dir / f"{stem}__test_H_hist.png"),
            "val_raw_esm_hist_path": str(fig_dir / f"{stem}__val_raw_esm_H_hist.png"),
            "test_raw_esm_hist_path": str(fig_dir / f"{stem}__test_raw_esm_H_hist.png"),
        }

        all_rows.append(row)

        torch.save(
            {
                "cfg": asdict(cfg),
                "hp": hp,
                "loss_params": loss_params,
                "best": best,
                "val_metrics": val_eval["metrics"],
                "test_metrics": test_eval["metrics"],
                "val_raw_esm_metrics": val_raw_esm["metrics"],
                "test_raw_esm_metrics": test_raw_esm["metrics"],
                "test_threshold_metrics": test_threshold_metrics,
            },
            checkpoint_path,
        )

        print("------------------------------------------------------------", flush=True)
        print(f"Best epoch: {row['best_epoch']}", flush=True)
        print(f"Selected by non-collapse rule: {row['selected_by_noncollapsed_rule']}", flush=True)
        print(f"Validation AUROC per-peptide macro: {row['val_auroc_per_peptide_macro']:.4f}", flush=True)
        print(f"Raw ESM validation AUROC per-peptide macro: {row['val_raw_esm_auroc_per_peptide_macro']:.4f}", flush=True)
        print(f"Validation delta per-peptide macro: {row['val_delta_auroc_pep_macro_vs_raw_esm']:.4f}", flush=True)
        print(f"Validation AUROC global: {row['val_auroc']:.4f}", flush=True)
        print(f"Raw ESM validation AUROC global: {row['val_raw_esm_auroc']:.4f}", flush=True)
        print(f"Test AUROC per-peptide macro: {row['test_auroc_per_peptide_macro']:.4f}", flush=True)
        print(f"Raw ESM test AUROC per-peptide macro: {row['test_raw_esm_auroc_per_peptide_macro']:.4f}", flush=True)
        print(f"Test delta per-peptide macro: {row['test_delta_auroc_pep_macro_vs_raw_esm']:.4f}", flush=True)
        print(f"Validation threshold: {row['val_threshold']:.6f}", flush=True)
        print(f"Test F1 at val threshold: {row['test_f1']:.4f}", flush=True)
        print(f"Saved history: {history_path}", flush=True)
        print(f"Saved checkpoint: {checkpoint_path}", flush=True)
        print("------------------------------------------------------------", flush=True)

        if best_global is None or row["val_auroc_per_peptide_macro"] > best_global["val_auroc_per_peptide_macro"]:
            best_global = row

    summary_df = pd.DataFrame(all_rows).sort_values(
        ["val_auroc_per_peptide_macro", "test_auroc_per_peptide_macro", "val_auroc", "test_auroc"],
        ascending=[False, False, False, False],
    )

    summary_path = out_dir / f"hpo_summary__{cfg.run_tag}.csv"
    best_json_path = out_dir / f"hpo_best__{cfg.run_tag}.json"

    summary_df.to_csv(summary_path, index=False)
    with open(best_json_path, "w") as f:
        json.dump(to_jsonable(best_global), f, indent=2)

    print("============================================================", flush=True)
    print("Done.", flush=True)
    print(f"Summary CSV: {summary_path}", flush=True)
    print(f"Best JSON: {best_json_path}", flush=True)
    print("Best row:", flush=True)
    print(json.dumps(to_jsonable(best_global), indent=2), flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
