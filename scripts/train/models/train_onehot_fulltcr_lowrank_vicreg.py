#!/usr/bin/env python3
"""
One-hot full-TCR low-rank VICReg baseline.

Purpose
-------
Controlled scratch-sequence baseline for the TCR-pMHC VICReg experiments.
This script intentionally removes pretrained ESM/Boltz features and reads the
multiview CSVs directly. It one-hot encodes amino-acid identity, then uses the
same broad geometry as the plain ESM VICReg script:

    TCR_full one-hot -> low-rank TCR projection head -> z_T
    Peptide one-hot + HLA sequence one-hot -> low-rank pMHC projection head -> z_pMHC
    loss = VICReg(z_T, z_pMHC) on positive training pairs only
    score = -mean((z_T - z_pMHC)^2) for validation/test/IMMREP

Missing alpha/beta handling
---------------------------
The minimal controlled baseline uses TCR_full as one concatenated sequence.
If alpha or beta is missing, the sequence is still used as-is; no artificial
residue is inserted. Masks handle variable length. The script records has_alpha
and has_beta in prediction outputs, and can optionally exclude incomplete TCRs
with --missing-chain-policy complete_only.

Expected CSV columns
--------------------
Required, with flexible candidate names:
    pair_id
    TCR_full
    Peptide / peptide
    HLA_sequence / HLA_seq / hla_sequence / hla / HLA
    binding_flag / label / binder / target  (optional for train; defaults to 1)
    tcra_len, tcrb_len, pep_len, hla_len    (optional; inferred from strings where possible)

Outputs
-------
- per-epoch history CSV
- best checkpoint
- validation/test/IMMREP prediction CSVs
- per-peptide AUROC tables
- histograms for model and raw one-hot baseline
- run config and summary JSON
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
# Amino-acid one-hot utilities
# ============================================================

AA20 = "ACDEFGHIKLMNPQRSTVWY"
# PAD is not represented in the one-hot tensor; padded positions are all zero.
# X is used for unknown/non-canonical residues. SEP is available but not used by
# the default fullTCR minimal baseline.
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
VOCAB["SEP"] = len(VOCAB)
VOCAB_SIZE = len(VOCAB)
UNK_IDX = VOCAB["X"]
SEP_IDX = VOCAB["SEP"]


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    # Remove common separators/spaces if a file contains formatted sequences.
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def onehot_encode(seq: str, max_len: int, use_sep: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(max_len, VOCAB_SIZE, dtype=torch.float32)
    m = torch.zeros(max_len, dtype=torch.bool)
    seq = clean_seq(seq)
    n = min(len(seq), max_len)
    for i, aa in enumerate(seq[:n]):
        idx = VOCAB.get(aa, UNK_IDX)
        x[i, idx] = 1.0
        m[i] = True
    return x, m


# ============================================================
# CSV parsing
# ============================================================

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalise_manifest(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "pair_id" not in df.columns:
        raise ValueError(f"{source_name}: CSV must contain pair_id")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    out["binding_flag"] = 1 if label_col is None else pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    tcr_col = first_existing_col(out, ["TCR_full", "tcr_full", "full_tcr", "TCR", "tcr"])
    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    hla_col = first_existing_col(out, ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"])

    missing = []
    if tcr_col is None:
        missing.append("TCR_full")
    if pep_col is None:
        missing.append("Peptide")
    if hla_col is None:
        missing.append("HLA_sequence")
    if missing:
        raise ValueError(f"{source_name}: missing required sequence column(s): {missing}. Available columns: {list(out.columns)}")

    out["TCR_full_norm"] = out[tcr_col].map(clean_seq)
    out["Peptide_norm"] = out[pep_col].map(clean_seq)
    out["HLA_sequence_norm"] = out[hla_col].map(clean_seq)
    out["peptide_for_eval"] = out["Peptide_norm"]

    len_specs = {
        "tcra_len": ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len"],
        "tcrb_len": ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len"],
        "pep_len": ["pep_len", "peptide_len"],
        "hla_len": ["hla_len", "mhc_len", "HLA_len"],
    }
    for target, candidates in len_specs.items():
        src = first_existing_col(out, candidates)
        if src is not None:
            out[target] = pd.to_numeric(out[src], errors="coerce").fillna(0).astype(int)
        else:
            if target == "pep_len":
                out[target] = out["Peptide_norm"].str.len().astype(int)
            elif target == "hla_len":
                out[target] = out["HLA_sequence_norm"].str.len().astype(int)
            else:
                out[target] = 0

    # If TCR chain lengths are absent, infer only total length; chain-specific
    # missingness then remains unknown. Your multiview files should provide these.
    out["tcr_total_len"] = out["TCR_full_norm"].str.len().astype(int)
    out["has_alpha"] = out["tcra_len"].astype(int) > 0
    out["has_beta"] = out["tcrb_len"].astype(int) > 0

    return out


def load_meta(
    csv_path: str,
    source_name: str,
    positives_only: bool,
    missing_chain_policy: str,
    require_peptide_hla: bool = True,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    meta = normalise_manifest(raw, source_name)

    before = len(meta)
    if positives_only:
        meta = meta[meta["binding_flag"].astype(int) == 1].copy()

    # Always require peptide and HLA for this pMHC baseline.
    if require_peptide_hla:
        meta = meta[(meta["pep_len"] > 0) & (meta["hla_len"] > 0)].copy()

    # Minimal baseline keeps incomplete TCRs by default. Use complete_only for
    # a controlled complete-chain-only sensitivity run.
    if missing_chain_policy == "complete_only":
        meta = meta[(meta["has_alpha"]) & (meta["has_beta"])].copy()
    elif missing_chain_policy == "keep":
        meta = meta[meta["tcr_total_len"] > 0].copy()
    else:
        raise ValueError("missing_chain_policy must be 'keep' or 'complete_only'")

    print(
        f"{source_name}: csv_rows={before} | kept={len(meta)} | positives_only={positives_only} | "
        f"missing_chain_policy={missing_chain_policy} | "
        f"missing_alpha={(~meta['has_alpha']).sum()} | missing_beta={(~meta['has_beta']).sum()}",
        flush=True,
    )

    if len(meta) == 0:
        raise RuntimeError(f"{source_name}: no rows remain after filtering")

    return meta.reset_index(drop=True)


def compute_max_lengths(metas: List[pd.DataFrame], cap_tcr: Optional[int], cap_pep: Optional[int], cap_hla: Optional[int]) -> Tuple[int, int, int]:
    all_meta = pd.concat(metas, axis=0, ignore_index=True)
    L_T = int(all_meta["tcr_total_len"].max())
    L_P = int(all_meta["pep_len"].max())
    L_H = int(all_meta["hla_len"].max())
    if cap_tcr is not None and cap_tcr > 0:
        L_T = min(L_T, cap_tcr)
    if cap_pep is not None and cap_pep > 0:
        L_P = min(L_P, cap_pep)
    if cap_hla is not None and cap_hla > 0:
        L_H = min(L_H, cap_hla)
    if min(L_T, L_P, L_H) <= 0:
        raise ValueError(f"Invalid max lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}")
    return L_T, L_P, L_H


# ============================================================
# Dataset
# ============================================================

class OneHotFullTCRDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, L_T: int, L_P: int, L_H: int, source_name: str):
        self.meta = meta.reset_index(drop=True)
        self.L_T = int(L_T)
        self.L_P = int(L_P)
        self.L_H = int(L_H)
        self.source_name = source_name

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict:
        r = self.meta.iloc[idx]
        xT, mT = onehot_encode(r["TCR_full_norm"], self.L_T)
        xP, mP = onehot_encode(r["Peptide_norm"], self.L_P)
        xH, mH = onehot_encode(r["HLA_sequence_norm"], self.L_H)
        return {
            "emb_T": xT,
            "mask_T": mT,
            "emb_P": xP,
            "mask_P": mP,
            "emb_H": xH,
            "mask_H": mH,
            "binding_flag": int(r["binding_flag"]),
            "pair_id": str(r["pair_id"]),
            "peptide": str(r["peptide_for_eval"]),
            "has_alpha": bool(r["has_alpha"]),
            "has_beta": bool(r["has_beta"]),
            "tcra_len": int(r["tcra_len"]),
            "tcrb_len": int(r["tcrb_len"]),
            "pep_len": int(r["pep_len"]),
            "hla_len": int(r["hla_len"]),
            "tcr_total_len": int(r["tcr_total_len"]),
        }


def onehot_collate(rows: List[Dict]) -> Dict:
    tensor_keys = ["emb_T", "mask_T", "emb_P", "mask_P", "emb_H", "mask_H"]
    out = {k: torch.stack([r[k] for r in rows], dim=0) for k in tensor_keys}
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "pep_len", "hla_len", "tcr_total_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


# ============================================================
# Projection heads
# ============================================================

class LowRankProjectionHead(nn.Module):
    """Low-rank projection head compatible with [B, L, D] inputs."""

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
            Yb = Xb @ self.B_c          # [L, rD]
            Ub = self.A_c[:Lb, :].T @ Yb # [rL, rD]
            zb = Ub.reshape(-1) @ self.H_c
            z_list.append(zb)
        return self.expander(torch.stack(z_list, dim=0))


class PMHCProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P_max: int, L_H_max: int, R_PH: float = 0.7, dropout: float = 0.1):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")
        self.pep_encoder = LowRankProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = LowRankProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

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
    partial_auc_max_fpr: float = 0.1,
    return_parts: bool = False,
):
    L_inv = F.mse_loss(zT, zPH)
    L_var = vicreg_variance(zT, gamma_var, eps_var) + vicreg_variance(zPH, gamma_var, eps_var)
    L_cov = vicreg_covariance(zT) + vicreg_covariance(zPH)
    loss = alpha * L_inv + beta * L_var + delta * L_cov

    if not return_parts:
        return loss

    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)

    parts = {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((alpha * L_inv).detach().cpu()),
        "weighted_var": float((beta * L_var).detach().cpu()),
        "weighted_cov": float((delta * L_cov).detach().cpu()),
        "mse_mean": float(mse_distance.mean().detach().cpu()),
        "mse_std": float(mse_distance.std(unbiased=False).detach().cpu()),
        "cos_mean": float(cos.mean().detach().cpu()),
        "cos_std": float(cos.std(unbiased=False).detach().cpu()),
        "zTstd": float(zT.std(unbiased=False).detach().cpu()),
        "zPHstd": float(zPH.std(unbiased=False).detach().cpu()),
        "eTstd": float(eT.std(unbiased=False).detach().cpu()),
        "ePHstd": float(ePH.std(unbiased=False).detach().cpu()),
    }
    return loss, parts


def score_from_projected(zT: torch.Tensor, zPH: torch.Tensor, eps_norm: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)
    score = -mse_distance
    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    return score, mse_distance, cos


def raw_onehot_score(batch: Dict, device: torch.device, eps_norm: float = 1e-8, R_PH: float = 0.7):
    # Crude frozen baseline: mean amino-acid composition of TCR vs weighted pMHC composition.
    T = masked_mean_pool(batch["emb_T"].to(device), batch["mask_T"].to(device), eps_norm)
    P = masked_mean_pool(batch["emb_P"].to(device), batch["mask_P"].to(device), eps_norm)
    H = masked_mean_pool(batch["emb_H"].to(device), batch["mask_H"].to(device), eps_norm)
    PH = R_PH * P + (1.0 - R_PH) * H
    score, mse_distance, cos = score_from_projected(T, PH, eps_norm)
    return score, mse_distance, cos, row_normalise(T, eps_norm)


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


def safe_partial_auc_raw(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(labels, scores)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if max_fpr not in fpr:
        stop = np.searchsorted(fpr, max_fpr, side="right")
        fpr_ext = np.concatenate([fpr[:stop], [max_fpr]])
        tpr_ext = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    else:
        keep = fpr <= max_fpr
        fpr_ext = fpr[keep]
        tpr_ext = tpr[keep]
    return float(auc(fpr_ext, tpr_ext))


def safe_partial_auc_norm(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    raw = safe_partial_auc_raw(labels, scores, max_fpr=max_fpr)
    return float("nan") if np.isnan(raw) else float(raw / max_fpr)


def safe_partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def per_peptide_auroc(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})

    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        auc = float(roc_auc_score(y, s)) if valid else float("nan")
        pauc_raw = safe_partial_auc_raw(y, s, max_fpr=max_fpr) if valid else float("nan")
        pauc_mcclish = safe_partial_auc_mcclish(y, s, max_fpr=max_fpr) if valid else float("nan")
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": auc,
            f"auc{max_fpr:g}_raw": pauc_raw,
            f"auc{max_fpr:g}_raw_div_maxfpr": float(pauc_raw / max_fpr) if valid else float("nan"),
            f"auc{max_fpr:g}_norm": pauc_mcclish,
            f"auc{max_fpr:g}_mcclish": pauc_mcclish,
            "valid": bool(valid),
        })

    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]].copy()
    if len(valid_table) == 0:
        summary = {k: float("nan") for k in [
            "macro", "weighted", "macro_auc0.1_raw", "weighted_auc0.1_raw",
            "macro_auc0.1_raw_div_maxfpr", "weighted_auc0.1_raw_div_maxfpr",
            "macro_auc0.1_norm", "weighted_auc0.1_norm",
            "macro_auc0.1_mcclish", "weighted_auc0.1_mcclish",
        ]}
        summary.update({"n_total": int(len(table)), "n_valid": 0})
    else:
        summary = {
            "macro": float(valid_table["auroc"].mean()),
            "weighted": float(np.average(valid_table["auroc"], weights=valid_table["n"])),
            "macro_auc0.1_raw": float(valid_table[f"auc{max_fpr:g}_raw"].mean()),
            "weighted_auc0.1_raw": float(np.average(valid_table[f"auc{max_fpr:g}_raw"], weights=valid_table["n"])),
            "macro_auc0.1_raw_div_maxfpr": float(valid_table[f"auc{max_fpr:g}_raw_div_maxfpr"].mean()),
            "weighted_auc0.1_raw_div_maxfpr": float(np.average(valid_table[f"auc{max_fpr:g}_raw_div_maxfpr"], weights=valid_table["n"])),
            "macro_auc0.1_norm": float(valid_table[f"auc{max_fpr:g}_mcclish"].mean()),
            "weighted_auc0.1_norm": float(np.average(valid_table[f"auc{max_fpr:g}_mcclish"], weights=valid_table["n"])),
            "macro_auc0.1_mcclish": float(valid_table[f"auc{max_fpr:g}_mcclish"].mean()),
            "weighted_auc0.1_mcclish": float(np.average(valid_table[f"auc{max_fpr:g}_mcclish"], weights=valid_table["n"])),
            "n_total": int(len(table)),
            "n_valid": int(len(valid_table)),
        }
    return table, summary


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    if len(scores) == 0:
        return {"threshold": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan")}
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
    return best


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
# Evaluation and plotting
# ============================================================

@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    lp: Dict,
    split: str,
    R_PH: float,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    model_scores, model_mse, model_cos = [], [], []
    raw_scores, raw_mse, raw_cos = [], [], []
    labels_all, pair_ids_all, peptides_all = [], [], []
    has_alpha_all, has_beta_all = [], []
    tcra_len_all, tcrb_len_all, pep_len_all, hla_len_all, tcr_total_len_all = [], [], [], [], []

    running_keys = [
        "loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov",
        "mse_std", "cos_std", "zTstd", "zPHstd", "eTstd", "ePHstd",
    ]
    running = {k: 0.0 for k in running_keys}
    n_steps = 0

    for batch in loader:
        zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
        zPH = pmhc_proj(
            batch["emb_P"].to(device), batch["mask_P"].to(device),
            batch["emb_H"].to(device), batch["mask_H"].to(device),
        )
        loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
        s_m, mse_m, cos_m = score_from_projected(zT, zPH, lp["eps_norm"])
        s_r, mse_r, cos_r, _ = raw_onehot_score(batch, device, lp["eps_norm"], R_PH)

        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        peptides = np.array(batch["peptide"], dtype=str)

        model_scores.append(s_m.detach().cpu().numpy())
        model_mse.append(mse_m.detach().cpu().numpy())
        model_cos.append(cos_m.detach().cpu().numpy())
        raw_scores.append(s_r.detach().cpu().numpy())
        raw_mse.append(mse_r.detach().cpu().numpy())
        raw_cos.append(cos_r.detach().cpu().numpy())
        labels_all.append(labels)
        pair_ids_all.extend([str(x) for x in batch["pair_id"]])
        peptides_all.append(peptides)
        has_alpha_all.append(batch["has_alpha"].detach().cpu().numpy().astype(bool))
        has_beta_all.append(batch["has_beta"].detach().cpu().numpy().astype(bool))
        tcra_len_all.append(batch["tcra_len"].detach().cpu().numpy())
        tcrb_len_all.append(batch["tcrb_len"].detach().cpu().numpy())
        pep_len_all.append(batch["pep_len"].detach().cpu().numpy())
        hla_len_all.append(batch["hla_len"].detach().cpu().numpy())
        tcr_total_len_all.append(batch["tcr_total_len"].detach().cpu().numpy())

        running["loss"] += float(loss.detach().cpu())
        for k in running:
            if k != "loss":
                running[k] += float(parts[k])
        n_steps += 1

    labels = np.concatenate(labels_all).astype(int)
    peptides = np.concatenate(peptides_all).astype(str)
    model_scores = np.concatenate(model_scores)
    model_mse = np.concatenate(model_mse)
    model_cos = np.concatenate(model_cos)
    raw_scores = np.concatenate(raw_scores)
    raw_mse = np.concatenate(raw_mse)
    raw_cos = np.concatenate(raw_cos)
    has_alpha = np.concatenate(has_alpha_all)
    has_beta = np.concatenate(has_beta_all)
    tcra_len = np.concatenate(tcra_len_all)
    tcrb_len = np.concatenate(tcrb_len_all)
    pep_len = np.concatenate(pep_len_all)
    hla_len = np.concatenate(hla_len_all)
    tcr_total_len = np.concatenate(tcr_total_len_all)

    model_pep_table, model_pep = per_peptide_auroc(labels, model_scores, peptides, max_fpr=lp.get("partial_auc_max_fpr", 0.1))
    raw_pep_table, raw_pep = per_peptide_auroc(labels, raw_scores, peptides, max_fpr=lp.get("partial_auc_max_fpr", 0.1))
    best_thr = best_f1_threshold(model_scores, labels)

    metrics = {
        f"{split}_loss": running["loss"] / max(1, n_steps),
        f"{split}_model_global_auroc": safe_auroc(labels, model_scores),
        f"{split}_raw_onehot_global_auroc": safe_auroc(labels, raw_scores),
        f"{split}_delta_global_auroc": safe_auroc(labels, model_scores) - safe_auroc(labels, raw_scores),
        f"{split}_model_auprc": safe_auprc(labels, model_scores),
        f"{split}_raw_onehot_auprc": safe_auprc(labels, raw_scores),
        f"{split}_model_auc0.1_raw_div_maxfpr": safe_partial_auc_norm(labels, model_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_onehot_auc0.1_raw_div_maxfpr": safe_partial_auc_norm(labels, raw_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_auc0.1_norm": safe_partial_auc_mcclish(labels, model_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_onehot_auc0.1_norm": safe_partial_auc_mcclish(labels, raw_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_auc0.1_mcclish": safe_partial_auc_mcclish(labels, model_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_onehot_auc0.1_mcclish": safe_partial_auc_mcclish(labels, raw_scores, max_fpr=lp.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_peptide_macro_auroc": model_pep["macro"],
        f"{split}_raw_onehot_peptide_macro_auroc": raw_pep["macro"],
        f"{split}_delta_peptide_macro_auroc": model_pep["macro"] - raw_pep["macro"],
        f"{split}_model_peptide_weighted_auroc": model_pep["weighted"],
        f"{split}_raw_onehot_peptide_weighted_auroc": raw_pep["weighted"],
        f"{split}_delta_peptide_weighted_auroc": model_pep["weighted"] - raw_pep["weighted"],
        f"{split}_model_peptide_macro_auc0.1_mcclish": model_pep["macro_auc0.1_mcclish"],
        f"{split}_model_peptide_weighted_auc0.1_mcclish": model_pep["weighted_auc0.1_mcclish"],
        f"{split}_raw_onehot_peptide_macro_auc0.1_mcclish": raw_pep["macro_auc0.1_mcclish"],
        f"{split}_raw_onehot_peptide_weighted_auc0.1_mcclish": raw_pep["weighted_auc0.1_mcclish"],
        f"{split}_n_peptides_total": model_pep["n_total"],
        f"{split}_n_peptides_valid": model_pep["n_valid"],
        f"{split}_threshold": best_thr["threshold"],
        f"{split}_f1": best_thr["f1"],
        f"{split}_accuracy": best_thr["accuracy"],
        f"{split}_precision": best_thr["precision"],
        f"{split}_recall": best_thr["recall"],
        f"{split}_model_mse_std": float(np.std(model_mse)),
        f"{split}_raw_onehot_mse_std": float(np.std(raw_mse)),
        f"{split}_n_missing_alpha": int((~has_alpha).sum()),
        f"{split}_n_missing_beta": int((~has_beta).sum()),
    }
    for k, v in running.items():
        if k != "loss":
            metrics[f"{split}_{k}"] = v / max(1, n_steps)

    predictions = pd.DataFrame({
        "pair_id": pair_ids_all,
        "peptide": peptides,
        "label": labels,
        "has_alpha": has_alpha,
        "has_beta": has_beta,
        "tcra_len": tcra_len,
        "tcrb_len": tcrb_len,
        "tcr_total_len": tcr_total_len,
        "pep_len": pep_len,
        "hla_len": hla_len,
        "model_score": model_scores,
        "model_mse_distance": model_mse,
        "model_cos": model_cos,
        "raw_onehot_score": raw_scores,
        "raw_onehot_mse_distance": raw_mse,
        "raw_onehot_cos": raw_cos,
    })

    return {
        "metrics": metrics,
        "predictions": predictions,
        "model_peptide_table": model_pep_table,
        "raw_peptide_table": raw_pep_table,
        "model_mse": model_mse,
        "raw_mse": raw_mse,
        "labels": labels,
    }


def plot_histogram(distances: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    distances = np.asarray(distances)
    labels = np.asarray(labels).astype(int)
    plt.figure(figsize=(8, 5))
    if np.any(labels == 0):
        plt.hist(distances[labels == 0], bins=50, density=True, alpha=0.55, label="decoy/negative")
    if np.any(labels == 1):
        plt.hist(distances[labels == 1], bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("MSE distance; lower = stronger predicted binding")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Config
# ============================================================

@dataclass
class RunConfig:
    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/onehot_fulltcr_lowrank_vicreg"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/onehot_fulltcr_lowrank_vicreg"
    run_tag: str = "onehot_fulltcr_lowrank_vicreg"

    train_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"

    missing_chain_policy: str = "keep"  # keep or complete_only
    max_tcr_len: int = 0                 # 0 = infer from CSVs
    max_pep_len: int = 0
    max_hla_len: int = 0

    seed: int = 31
    batch_size: int = 64
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
    eps_norm: float = 1e-8
    eps_var: float = 1e-4
    partial_auc_max_fpr: float = 0.1


def loss_params(cfg: RunConfig) -> Dict:
    return {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "delta": cfg.delta,
        "gamma_var": cfg.gamma_var,
        "eps_norm": cfg.eps_norm,
        "eps_var": cfg.eps_var,
        "partial_auc_max_fpr": cfg.partial_auc_max_fpr,
    }


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=onehot_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=RunConfig.out_dir)
    parser.add_argument("--fig-dir", default=RunConfig.fig_dir)
    parser.add_argument("--run-tag", default=RunConfig.run_tag)
    parser.add_argument("--train-csv", default=RunConfig.train_csv)
    parser.add_argument("--val-csv", default=RunConfig.val_csv)
    parser.add_argument("--test-csv", default=RunConfig.test_csv)
    parser.add_argument("--immrep-csv", default=RunConfig.immrep_csv)
    parser.add_argument("--missing-chain-policy", choices=["keep", "complete_only"], default=RunConfig.missing_chain_policy)
    parser.add_argument("--max-tcr-len", type=int, default=RunConfig.max_tcr_len)
    parser.add_argument("--max-pep-len", type=int, default=RunConfig.max_pep_len)
    parser.add_argument("--max-hla-len", type=int, default=RunConfig.max_hla_len)
    parser.add_argument("--seed", type=int, default=RunConfig.seed)
    parser.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    parser.add_argument("--epochs", type=int, default=RunConfig.epochs)
    parser.add_argument("--patience", type=int, default=RunConfig.patience)
    parser.add_argument("--min-epochs", type=int, default=RunConfig.min_epochs)
    parser.add_argument("--rL", type=int, default=RunConfig.rL)
    parser.add_argument("--rD", type=int, default=RunConfig.rD)
    parser.add_argument("--d", type=int, default=RunConfig.d)
    parser.add_argument("--R-PH", type=float, default=RunConfig.R_PH)
    parser.add_argument("--dropout", type=float, default=RunConfig.dropout)
    parser.add_argument("--lr", type=float, default=RunConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=RunConfig.weight_decay)
    parser.add_argument("--alpha", type=float, default=RunConfig.alpha)
    parser.add_argument("--beta", type=float, default=RunConfig.beta)
    parser.add_argument("--delta", type=float, default=RunConfig.delta)
    parser.add_argument("--gamma-var", type=float, default=RunConfig.gamma_var)
    parser.add_argument("--partial-auc-max-fpr", type=float, default=RunConfig.partial_auc_max_fpr)
    args = parser.parse_args()

    cfg = RunConfig(
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        run_tag=args.run_tag,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        immrep_csv=args.immrep_csv,
        missing_chain_policy=args.missing_chain_policy,
        max_tcr_len=args.max_tcr_len,
        max_pep_len=args.max_pep_len,
        max_hla_len=args.max_hla_len,
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
    print("One-hot full-TCR low-rank VICReg baseline", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Run tag: {cfg.run_tag}", flush=True)
    print(f"Missing-chain policy: {cfg.missing_chain_policy}", flush=True)
    print(f"Train CSV: {cfg.train_csv}", flush=True)
    print(f"Val CSV: {cfg.val_csv}", flush=True)
    print(f"Test CSV: {cfg.test_csv}", flush=True)
    print(f"IMMREP CSV: {cfg.immrep_csv}", flush=True)
    print(f"Vocab size: {VOCAB_SIZE} ({list(VOCAB.keys())})", flush=True)
    print("============================================================", flush=True)

    train_meta = load_meta(cfg.train_csv, "train", positives_only=True, missing_chain_policy=cfg.missing_chain_policy)
    val_meta = load_meta(cfg.val_csv, "val", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
    test_meta = load_meta(cfg.test_csv, "test", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
    immrep_meta = None
    if cfg.immrep_csv and str(cfg.immrep_csv).lower() not in ["none", ""]:
        immrep_meta = load_meta(cfg.immrep_csv, "immrep_test", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)

    metas_for_lengths = [train_meta, val_meta, test_meta] + ([] if immrep_meta is None else [immrep_meta])
    L_T, L_P, L_H = compute_max_lengths(
        metas_for_lengths,
        cfg.max_tcr_len if cfg.max_tcr_len > 0 else None,
        cfg.max_pep_len if cfg.max_pep_len > 0 else None,
        cfg.max_hla_len if cfg.max_hla_len > 0 else None,
    )
    print(f"Max lengths used | L_T={L_T} | L_P={L_P} | L_H={L_H}", flush=True)

    train_ds = OneHotFullTCRDataset(train_meta, L_T, L_P, L_H, "train")
    val_ds = OneHotFullTCRDataset(val_meta, L_T, L_P, L_H, "val")
    test_ds = OneHotFullTCRDataset(test_meta, L_T, L_P, L_H, "test")
    immrep_ds = None if immrep_meta is None else OneHotFullTCRDataset(immrep_meta, L_T, L_P, L_H, "immrep_test")

    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    immrep_loader = None if immrep_ds is None else make_loader(immrep_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    tcr = LowRankProjectionHead(VOCAB_SIZE, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(VOCAB_SIZE, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": tcr.parameters(), "lr": cfg.lr},
            {"params": pmhc.parameters(), "lr": cfg.lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    lp = loss_params(cfg)

    best = {
        "val_model_peptide_weighted_auroc": -np.inf,
        "epoch": None,
        "state": None,
        "bad_epochs": 0,
    }
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()
        running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov"]}
        n_steps = 0

        for batch in train_loader:
            zT = tcr(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc(
                batch["emb_P"].to(device), batch["mask_P"].to(device),
                batch["emb_H"].to(device), batch["mask_H"].to(device),
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

        val_eval = evaluate(val_loader, tcr, pmhc, device, lp, "val", cfg.R_PH)
        row = {
            "epoch": epoch,
            "train_loss": running["loss"] / max(1, n_steps),
            "train_L_inv": running["L_inv"] / max(1, n_steps),
            "train_L_var": running["L_var"] / max(1, n_steps),
            "train_L_cov": running["L_cov"] / max(1, n_steps),
            "train_weighted_inv": running["weighted_inv"] / max(1, n_steps),
            "train_weighted_var": running["weighted_var"] / max(1, n_steps),
            "train_weighted_cov": running["weighted_cov"] / max(1, n_steps),
            **val_eval["metrics"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / f"{cfg.run_tag}__history.csv", index=False)

        current = row["val_model_peptide_weighted_auroc"]
        improved = (not np.isnan(current)) and current > best["val_model_peptide_weighted_auroc"] + 1e-4
        if improved:
            best = {
                "val_model_peptide_weighted_auroc": current,
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
            f"w_inv={row['train_weighted_inv']:.4f} | "
            f"w_var={row['train_weighted_var']:.4f} | "
            f"w_cov={row['train_weighted_cov']:.4f} | "
            f"val_model_global={row['val_model_global_auroc']:.4f} | "
            f"raw_global={row['val_raw_onehot_global_auroc']:.4f} | "
            f"delta_global={row['val_delta_global_auroc']:.4f} | "
            f"val_model_pep_weighted={row['val_model_peptide_weighted_auroc']:.4f} | "
            f"raw_pep_weighted={row['val_raw_onehot_peptide_weighted_auroc']:.4f} | "
            f"val_auprc={row['val_model_auprc']:.4f} | "
            f"raw_auprc={row['val_raw_onehot_auprc']:.4f} | "
            f"mse_std={row['val_model_mse_std']:.4f} | "
            f"zTstd={row['val_zTstd']:.4f} | "
            f"zPHstd={row['val_zPHstd']:.4f} | "
            f"missing_alpha={row['val_n_missing_alpha']} | "
            f"missing_beta={row['val_n_missing_beta']} | "
            f"best_epoch={best['epoch']} | "
            f"bad_epochs={best['bad_epochs']}",
            flush=True,
        )

        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; min_epochs={cfg.min_epochs}, patience={cfg.patience}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint was selected. Check whether validation has both positive and negative labels.")

    tcr.load_state_dict(best["state"]["tcr"])
    pmhc.load_state_dict(best["state"]["pmhc"])

    val_eval = evaluate(val_loader, tcr, pmhc, device, lp, "val", cfg.R_PH)
    test_eval = evaluate(test_loader, tcr, pmhc, device, lp, "test", cfg.R_PH)
    immrep_eval = None if immrep_loader is None else evaluate(immrep_loader, tcr, pmhc, device, lp, "immrep_test", cfg.R_PH)

    val_threshold = val_eval["metrics"]["val_threshold"]
    test_at_val_threshold = threshold_metrics(
        test_eval["predictions"]["model_score"].to_numpy(),
        test_eval["predictions"]["label"].to_numpy(),
        val_threshold,
        "test_at_val_threshold",
    )

    stem = (
        f"{cfg.run_tag}"
        f"__seed{cfg.seed}"
        f"__lr{cfg.lr}"
        f"__a{cfg.alpha}"
        f"__b{cfg.beta}"
        f"__dlt{cfg.delta}"
        f"__missing{cfg.missing_chain_policy}"
    )

    checkpoint_path = out_dir / f"{stem}__best.pt"
    summary_path = out_dir / f"{stem}__summary.json"

    val_pred_path = out_dir / f"{stem}__val_predictions.csv"
    test_pred_path = out_dir / f"{stem}__test_predictions.csv"
    val_model_pep_path = out_dir / f"{stem}__val_model_per_peptide.csv"
    val_raw_pep_path = out_dir / f"{stem}__val_raw_onehot_per_peptide.csv"
    test_model_pep_path = out_dir / f"{stem}__test_model_per_peptide.csv"
    test_raw_pep_path = out_dir / f"{stem}__test_raw_onehot_per_peptide.csv"

    val_eval["predictions"].to_csv(val_pred_path, index=False)
    test_eval["predictions"].to_csv(test_pred_path, index=False)
    val_eval["model_peptide_table"].to_csv(val_model_pep_path, index=False)
    val_eval["raw_peptide_table"].to_csv(val_raw_pep_path, index=False)
    test_eval["model_peptide_table"].to_csv(test_model_pep_path, index=False)
    test_eval["raw_peptide_table"].to_csv(test_raw_pep_path, index=False)

    immrep_pred_path = None
    immrep_model_pep_path = None
    immrep_raw_pep_path = None
    if immrep_eval is not None:
        immrep_pred_path = out_dir / f"{stem}__immrep_test_predictions.csv"
        immrep_model_pep_path = out_dir / f"{stem}__immrep_test_model_per_peptide.csv"
        immrep_raw_pep_path = out_dir / f"{stem}__immrep_test_raw_onehot_per_peptide.csv"
        immrep_eval["predictions"].to_csv(immrep_pred_path, index=False)
        immrep_eval["model_peptide_table"].to_csv(immrep_model_pep_path, index=False)
        immrep_eval["raw_peptide_table"].to_csv(immrep_raw_pep_path, index=False)

    plot_histogram(val_eval["model_mse"], val_eval["labels"], "Validation model one-hot VICReg", fig_dir / f"{stem}__val_model_mse_hist.png")
    plot_histogram(val_eval["raw_mse"], val_eval["labels"], "Validation raw one-hot composition", fig_dir / f"{stem}__val_raw_onehot_mse_hist.png")
    plot_histogram(test_eval["model_mse"], test_eval["labels"], "Test model one-hot VICReg", fig_dir / f"{stem}__test_model_mse_hist.png")
    plot_histogram(test_eval["raw_mse"], test_eval["labels"], "Test raw one-hot composition", fig_dir / f"{stem}__test_raw_onehot_mse_hist.png")
    if immrep_eval is not None:
        plot_histogram(immrep_eval["model_mse"], immrep_eval["labels"], "IMMREP model one-hot VICReg", fig_dir / f"{stem}__immrep_model_mse_hist.png")
        plot_histogram(immrep_eval["raw_mse"], immrep_eval["labels"], "IMMREP raw one-hot composition", fig_dir / f"{stem}__immrep_raw_onehot_mse_hist.png")

    torch.save(
        {
            "config": asdict(cfg),
            "vocab": VOCAB,
            "max_lengths": {"L_T": L_T, "L_P": L_P, "L_H": L_H},
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
            "immrep_test_metrics": None if immrep_eval is None else immrep_eval["metrics"],
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(cfg),
        "vocab_size": VOCAB_SIZE,
        "max_lengths": {"L_T": L_T, "L_P": L_P, "L_H": L_H},
        "best_epoch_by_val_model_peptide_weighted_auroc": best["epoch"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "immrep_test_metrics": None if immrep_eval is None else immrep_eval["metrics"],
        "test_at_val_threshold": test_at_val_threshold,
        "paths": {
            "history": str(out_dir / f"{cfg.run_tag}__history.csv"),
            "checkpoint": str(checkpoint_path),
            "val_predictions": str(val_pred_path),
            "test_predictions": str(test_pred_path),
            "val_model_per_peptide": str(val_model_pep_path),
            "val_raw_onehot_per_peptide": str(val_raw_pep_path),
            "test_model_per_peptide": str(test_model_pep_path),
            "test_raw_onehot_per_peptide": str(test_raw_pep_path),
            "immrep_test_predictions": None if immrep_pred_path is None else str(immrep_pred_path),
            "immrep_test_model_per_peptide": None if immrep_model_pep_path is None else str(immrep_model_pep_path),
            "immrep_test_raw_onehot_per_peptide": None if immrep_raw_pep_path is None else str(immrep_raw_pep_path),
            "fig_dir": str(fig_dir),
        },
    }
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
    if immrep_pred_path is not None:
        print(f"IMMREP predictions: {immrep_pred_path}", flush=True)
    print(f"Figures: {fig_dir}", flush=True)
    print("Final validation metrics:", flush=True)
    print(json.dumps(val_eval["metrics"], indent=2), flush=True)
    print("Final test metrics:", flush=True)
    print(json.dumps(test_eval["metrics"], indent=2), flush=True)
    if immrep_eval is not None:
        print("Final IMMREP test metrics:", flush=True)
        print(json.dumps(immrep_eval["metrics"], indent=2), flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
