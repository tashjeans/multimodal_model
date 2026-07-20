#!/usr/bin/env python3
"""
Workshop-paper one-hot VICReg pipeline.

Produces two paper rows from the same complete-chain multiview CSVs:
  1) onehot_composition: frozen amino-acid composition baseline
  2) onehot_vicreg: VICReg projection heads trained from one-hot sequences

Scoring convention throughout:
  mse_distance = mean((TCR_representation - pMHC_representation)^2)
  score = -mse_distance
Higher score = more binder-like.

No cosine scores are computed or saved.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
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
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
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
# One-hot utilities
# ============================================================

AA20 = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
VOCAB["SEP"] = len(VOCAB)  # retained for compatibility; not used by default
VOCAB_SIZE = len(VOCAB)
UNK_IDX = VOCAB["X"]


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def onehot_encode(seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(max_len, VOCAB_SIZE, dtype=torch.float32)
    m = torch.zeros(max_len, dtype=torch.bool)
    seq = clean_seq(seq)
    n = min(len(seq), max_len)
    for i, aa in enumerate(seq[:n]):
        x[i, VOCAB.get(aa, UNK_IDX)] = 1.0
        m[i] = True
    return x, m


# ============================================================
# CSV parsing and filtering
# ============================================================

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_length(
    df: pd.DataFrame,
    length_candidates: List[str],
    seq_candidates: List[str],
    target_name: str,
) -> Tuple[pd.Series, str]:
    length_col = first_existing_col(df, length_candidates)
    if length_col is not None:
        vals = pd.to_numeric(df[length_col], errors="coerce").fillna(0).astype(int)
        return vals, length_col

    seq_col = first_existing_col(df, seq_candidates)
    if seq_col is not None:
        vals = df[seq_col].map(clean_seq).str.len().astype(int)
        return vals, f"inferred_from:{seq_col}"

    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "__missing__"


def normalise_manifest(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
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
        missing.append("HLA_sequence/HLA")
    if missing:
        raise ValueError(f"{source_name}: missing required column(s): {missing}. Available columns: {list(out.columns)}")

    out["TCR_full_norm"] = out[tcr_col].map(clean_seq)
    out["Peptide_norm"] = out[pep_col].map(clean_seq)
    out["HLA_sequence_norm"] = out[hla_col].map(clean_seq)
    out["peptide_for_eval"] = out["Peptide_norm"]

    source_map = {
        "tcr_col": tcr_col,
        "pep_col": pep_col,
        "hla_col": hla_col,
        "label_col": "constant_1" if label_col is None else label_col,
    }

    out["tcra_len"], source_map["tcra_len"] = extract_length(
        out,
        ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
        "tcra_len",
    )
    out["tcrb_len"], source_map["tcrb_len"] = extract_length(
        out,
        ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
        "tcrb_len",
    )
    out["pep_len"], source_map["pep_len"] = extract_length(
        out,
        ["pep_len", "peptide_len"],
        ["Peptide", "peptide", "pep_seq", "peptide_seq"],
        "pep_len",
    )
    out["hla_len"], source_map["hla_len"] = extract_length(
        out,
        ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"],
        "hla_len",
    )

    out["tcr_total_len"] = out["TCR_full_norm"].str.len().astype(int)
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source_map


def load_meta(
    csv_path: str,
    source_name: str,
    positives_only: bool,
    missing_chain_policy: str = "complete_only",
) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(csv_path)
    meta, source_map = normalise_manifest(raw, source_name)

    if missing_chain_policy == "complete_only" and (
        source_map["tcra_len"] == "__missing__" or source_map["tcrb_len"] == "__missing__"
    ):
        raise ValueError(
            f"{source_name}: complete_only requested, but alpha/beta chain length information could not be found. "
            f"Source map: {source_map}. Available columns: {list(raw.columns)}"
        )

    audit = {
        "split": source_name,
        "csv_path": str(csv_path),
        "csv_rows": int(len(raw)),
        "label_source": source_map["label_col"],
        "tcr_source": source_map["tcr_col"],
        "peptide_source": source_map["pep_col"],
        "hla_source": source_map["hla_col"],
        "tcra_len_source": source_map["tcra_len"],
        "tcrb_len_source": source_map["tcrb_len"],
        "pep_len_source": source_map["pep_len"],
        "hla_len_source": source_map["hla_len"],
        "positives_only": bool(positives_only),
        "missing_chain_policy": missing_chain_policy,
        "n_positive_before_filter": int((meta["binding_flag"] == 1).sum()),
        "n_negative_before_filter": int((meta["binding_flag"] == 0).sum()),
        "n_missing_alpha_before_filter": int((~meta["has_alpha"]).sum()),
        "n_missing_beta_before_filter": int((~meta["has_beta"]).sum()),
        "n_missing_peptide_before_filter": int((meta["pep_len"] <= 0).sum()),
        "n_missing_hla_before_filter": int((meta["hla_len"] <= 0).sum()),
    }

    if positives_only:
        meta = meta[meta["binding_flag"] == 1].copy()
    audit["rows_after_positive_filter"] = int(len(meta))

    meta = meta[(meta["pep_len"] > 0) & (meta["hla_len"] > 0) & (meta["tcr_total_len"] > 0)].copy()
    audit["rows_after_required_sequence_filter"] = int(len(meta))

    if missing_chain_policy == "complete_only":
        meta = meta[meta["has_alpha"] & meta["has_beta"]].copy()
    elif missing_chain_policy == "keep":
        pass
    else:
        raise ValueError("missing_chain_policy must be 'complete_only' or 'keep'")

    audit["n_final"] = int(len(meta))
    audit["n_positive_final"] = int((meta["binding_flag"] == 1).sum())
    audit["n_negative_final"] = int((meta["binding_flag"] == 0).sum())
    audit["n_missing_alpha_final"] = int((~meta["has_alpha"]).sum())
    audit["n_missing_beta_final"] = int((~meta["has_beta"]).sum())

    print(
        f"{source_name}: csv_rows={audit['csv_rows']} | final={audit['n_final']} | "
        f"pos={audit['n_positive_final']} | neg={audit['n_negative_final']} | "
        f"complete_policy={missing_chain_policy}",
        flush=True,
    )

    if len(meta) == 0:
        raise RuntimeError(f"{source_name}: no rows remain after filtering")
    return meta.reset_index(drop=True), audit


def compute_max_lengths(metas: List[pd.DataFrame], cap_tcr: int, cap_pep: int, cap_hla: int) -> Tuple[int, int, int]:
    all_meta = pd.concat(metas, axis=0, ignore_index=True)
    L_T = int(all_meta["tcr_total_len"].max())
    L_P = int(all_meta["pep_len"].max())
    L_H = int(all_meta["hla_len"].max())
    if cap_tcr > 0:
        L_T = min(L_T, cap_tcr)
    if cap_pep > 0:
        L_P = min(L_P, cap_pep)
    if cap_hla > 0:
        L_H = min(L_H, cap_hla)
    if min(L_T, L_P, L_H) <= 0:
        raise ValueError(f"Invalid lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}")
    return L_T, L_P, L_H


# ============================================================
# Dataset
# ============================================================

class OneHotFullTCRDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, L_T: int, L_P: int, L_H: int):
        self.meta = meta.reset_index(drop=True)
        self.L_T = int(L_T)
        self.L_P = int(L_P)
        self.L_H = int(L_H)

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
            "tcr_total_len": int(r["tcr_total_len"]),
            "pep_len": int(r["pep_len"]),
            "hla_len": int(r["hla_len"]),
        }


def onehot_collate(rows: List[Dict]) -> Dict:
    tensor_keys = ["emb_T", "mask_T", "emb_P", "mask_P", "emb_H", "mask_H"]
    out = {k: torch.stack([r[k] for r in rows], dim=0) for k in tensor_keys}
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


# ============================================================
# Model
# ============================================================

class LowRankProjectionHead(nn.Module):
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
                z_list.append(torch.zeros(self.d, device=emb.device, dtype=emb.dtype))
                continue
            Xb = emb[b, :Lb, :] * mask[b, :Lb].unsqueeze(-1).float()
            Yb = Xb @ self.B_c
            Ub = self.A_c[:Lb, :].T @ Yb
            z_list.append(Ub.reshape(-1) @ self.H_c)
        return self.expander(torch.stack(z_list, dim=0))


class PMHCProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P_max: int, L_H_max: int, R_PH: float, dropout: float):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")
        self.pep_encoder = LowRankProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = LowRankProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pep_encoder(emb_P, mask_P), self.hla_encoder(emb_H, mask_H)], dim=-1)


# ============================================================
# Loss and score
# ============================================================

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
    eps_var: float,
    return_parts: bool = False,
):
    L_inv = F.mse_loss(zT, zPH)
    L_var = vicreg_variance(zT, gamma_var, eps_var) + vicreg_variance(zPH, gamma_var, eps_var)
    L_cov = vicreg_covariance(zT) + vicreg_covariance(zPH)
    loss = alpha * L_inv + beta * L_var + delta * L_cov
    if not return_parts:
        return loss
    return loss, {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((alpha * L_inv).detach().cpu()),
        "weighted_var": float((beta * L_var).detach().cpu()),
        "weighted_cov": float((delta * L_cov).detach().cpu()),
        "zT_std": float(zT.std(unbiased=False).detach().cpu()),
        "zPH_std": float(zPH.std(unbiased=False).detach().cpu()),
    }


def score_from_vectors(zT: torch.Tensor, zPH: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)
    return -mse_distance, mse_distance


def onehot_composition_score(batch: Dict, device: torch.device, R_PH: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = masked_mean_pool(batch["emb_T"].to(device), batch["mask_T"].to(device), eps)
    P = masked_mean_pool(batch["emb_P"].to(device), batch["mask_P"].to(device), eps)
    HLA = masked_mean_pool(batch["emb_H"].to(device), batch["mask_H"].to(device), eps)
    PH = R_PH * P + (1.0 - R_PH) * HLA
    score, mse = score_from_vectors(T, PH)
    return score, mse, T, PH


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


def safe_partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def per_peptide_table(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": float(roc_auc_score(y, s)) if valid else float("nan"),
            "auc0.1_raw": safe_partial_auc_raw(y, s, max_fpr) if valid else float("nan"),
            "auc0.1_mcclish": safe_partial_auc_mcclish(y, s, max_fpr) if valid else float("nan"),
            "valid": bool(valid),
        })
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid = table[table["valid"]].copy()
    if len(valid) == 0:
        summary = {
            "peptide_macro_auroc": float("nan"),
            "peptide_weighted_auroc": float("nan"),
            "peptide_macro_auc0.1_mcclish": float("nan"),
            "peptide_weighted_auc0.1_mcclish": float("nan"),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": 0,
        }
    else:
        summary = {
            "peptide_macro_auroc": float(valid["auroc"].mean()),
            "peptide_weighted_auroc": float(np.average(valid["auroc"], weights=valid["n"])),
            "peptide_macro_auc0.1_mcclish": float(valid["auc0.1_mcclish"].mean()),
            "peptide_weighted_auc0.1_mcclish": float(np.average(valid["auc0.1_mcclish"], weights=valid["n"])),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": int(len(valid)),
        }
    return table, summary


def metrics_for_scores(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    pep_table, pep_summary = per_peptide_table(labels, scores, peptides, max_fpr)
    metrics = {
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "global_auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "global_auc0.1_raw": safe_partial_auc_raw(labels, scores, max_fpr),
        "global_auc0.1_mcclish": safe_partial_auc_mcclish(labels, scores, max_fpr),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        **pep_summary,
    }
    return metrics, pep_table


# ============================================================
# Evaluation and output helpers
# ============================================================

@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    cfg: "RunConfig",
    split: str,
    save_latents: bool,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    pair_ids, peptides = [], []
    labels_all = []
    meta_cols = {k: [] for k in ["has_alpha", "has_beta", "tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len"]}
    scores = {"onehot_vicreg": [], "onehot_composition": []}
    distances = {"onehot_vicreg": [], "onehot_composition": []}
    latent_store = {"zT_vicreg": [], "zPH_vicreg": [], "T_composition": [], "PH_composition": []} if save_latents else None

    running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
    n_steps = 0

    lp = loss_params(cfg)
    for batch in loader:
        zT = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
        zPH = pmhc_proj(
            batch["emb_P"].to(device), batch["mask_P"].to(device),
            batch["emb_H"].to(device), batch["mask_H"].to(device),
        )
        loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
        vicreg_score, vicreg_mse = score_from_vectors(zT, zPH)
        comp_score, comp_mse, comp_T, comp_PH = onehot_composition_score(batch, device, cfg.R_PH, cfg.eps_pool)

        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        labels_all.append(labels)
        pair_ids.extend([str(x) for x in batch["pair_id"]])
        peptides.append(np.array(batch["peptide"], dtype=str))
        for k in meta_cols:
            meta_cols[k].append(batch[k].detach().cpu().numpy())

        scores["onehot_vicreg"].append(vicreg_score.detach().cpu().numpy())
        distances["onehot_vicreg"].append(vicreg_mse.detach().cpu().numpy())
        scores["onehot_composition"].append(comp_score.detach().cpu().numpy())
        distances["onehot_composition"].append(comp_mse.detach().cpu().numpy())

        if save_latents:
            latent_store["zT_vicreg"].append(zT.detach().cpu().numpy())
            latent_store["zPH_vicreg"].append(zPH.detach().cpu().numpy())
            latent_store["T_composition"].append(comp_T.detach().cpu().numpy())
            latent_store["PH_composition"].append(comp_PH.detach().cpu().numpy())

        running["loss"] += float(loss.detach().cpu())
        for k in running:
            if k != "loss":
                running[k] += float(parts[k])
        n_steps += 1

    labels_np = np.concatenate(labels_all).astype(int)
    peptides_np = np.concatenate(peptides).astype(str)
    scores_np = {k: np.concatenate(v) for k, v in scores.items()}
    distances_np = {k: np.concatenate(v) for k, v in distances.items()}
    meta_np = {k: np.concatenate(v) for k, v in meta_cols.items()}

    metrics = {}
    peptide_tables = {}
    for model_name in ["onehot_vicreg", "onehot_composition"]:
        m, table = metrics_for_scores(labels_np, scores_np[model_name], peptides_np, cfg.partial_auc_max_fpr)
        m.update({
            "mse_distance_mean": float(np.mean(distances_np[model_name])),
            "mse_distance_std": float(np.std(distances_np[model_name])),
        })
        metrics[model_name] = m
        peptide_tables[model_name] = table

    running_avg = {k: v / max(1, n_steps) for k, v in running.items()}
    metrics["onehot_vicreg"].update({f"eval_{k}": val for k, val in running_avg.items()})

    predictions = pd.DataFrame({
        "pair_id": pair_ids,
        "peptide": peptides_np,
        "label": labels_np,
        **{k: meta_np[k] for k in meta_np},
        "onehot_vicreg_score": scores_np["onehot_vicreg"],
        "onehot_vicreg_mse_distance": distances_np["onehot_vicreg"],
        "onehot_composition_score": scores_np["onehot_composition"],
        "onehot_composition_mse_distance": distances_np["onehot_composition"],
    })

    latents = None
    if save_latents:
        latents = {k: np.concatenate(v, axis=0) for k, v in latent_store.items()}
        latents.update({"pair_id": np.array(pair_ids, dtype=str), "peptide": peptides_np, "label": labels_np})

    return {
        "split": split,
        "metrics": metrics,
        "predictions": predictions,
        "peptide_tables": peptide_tables,
        "distances": distances_np,
        "labels": labels_np,
        "latents": latents,
    }


def plot_histogram(distances: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(int)
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


def save_eval_outputs(eval_obj: Dict, output_dir: Path, figure_dir: Path, split: str, save_latents: bool) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    pred_path = output_dir / f"{split}_predictions.csv"
    eval_obj["predictions"].to_csv(pred_path, index=False)
    paths[f"{split}_predictions"] = str(pred_path)

    for model_name, table in eval_obj["peptide_tables"].items():
        path = output_dir / f"{split}_{model_name}_per_peptide.csv"
        table.to_csv(path, index=False)
        paths[f"{split}_{model_name}_per_peptide"] = str(path)

    for model_name, dist in eval_obj["distances"].items():
        fig_path = figure_dir / f"{split}_{model_name}_mse_hist.png"
        plot_histogram(dist, eval_obj["labels"], f"{split}: {model_name} MSE distance", fig_path)
        paths[f"{split}_{model_name}_mse_hist"] = str(fig_path)

    if save_latents and eval_obj["latents"] is not None:
        latent_path = output_dir / f"{split}_latents.npz"
        np.savez_compressed(latent_path, **eval_obj["latents"])
        paths[f"{split}_latents"] = str(latent_path)

    return paths


# ============================================================
# Config
# ============================================================

@dataclass
class RunConfig:
    run_tag: str = "onehot_vicreg_complete"
    checkpoint_root: str = "/home/natasha/multimodal_model/models/checkpoints/workshop/onehot_vicreg_complete"
    output_root: str = "/home/natasha/multimodal_model/models/outputs/workshop/onehot_vicreg_complete"
    figure_root: str = "/home/natasha/multimodal_model/models/figures/workshop/onehot_vicreg_complete"

    train_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"

    missing_chain_policy: str = "complete_only"
    max_tcr_len: int = 0
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
    eps_var: float = 1e-4
    eps_pool: float = 1e-8
    partial_auc_max_fpr: float = 0.1

    save_latents: bool = False
    overwrite: bool = False


def loss_params(cfg: RunConfig) -> Dict:
    return {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "delta": cfg.delta,
        "gamma_var": cfg.gamma_var,
        "eps_var": cfg.eps_var,
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


def prepare_dirs(cfg: RunConfig) -> Tuple[Path, Path, Path]:
    seed_name = f"seed_{cfg.seed}"
    checkpoint_dir = Path(cfg.checkpoint_root) / seed_name
    output_dir = Path(cfg.output_root) / seed_name
    figure_dir = Path(cfg.figure_root) / seed_name
    if cfg.overwrite:
        for d in [checkpoint_dir, output_dir, figure_dir]:
            if d.exists():
                shutil.rmtree(d)
    for d in [checkpoint_dir, output_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, output_dir, figure_dir


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=RunConfig.run_tag)
    parser.add_argument("--checkpoint-root", default=RunConfig.checkpoint_root)
    parser.add_argument("--output-root", default=RunConfig.output_root)
    parser.add_argument("--figure-root", default=RunConfig.figure_root)
    parser.add_argument("--train-csv", default=RunConfig.train_csv)
    parser.add_argument("--val-csv", default=RunConfig.val_csv)
    parser.add_argument("--test-csv", default=RunConfig.test_csv)
    parser.add_argument("--immrep-csv", default=RunConfig.immrep_csv)
    parser.add_argument("--missing-chain-policy", choices=["complete_only", "keep"], default=RunConfig.missing_chain_policy)
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
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = RunConfig(
        run_tag=args.run_tag,
        checkpoint_root=args.checkpoint_root,
        output_root=args.output_root,
        figure_root=args.figure_root,
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
        save_latents=args.save_latents,
        overwrite=args.overwrite,
    )

    set_seed(cfg.seed)
    checkpoint_dir, output_dir, figure_dir = prepare_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 72, flush=True)
    print("Workshop one-hot VICReg run", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {cfg.seed}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"Figure dir: {figure_dir}", flush=True)
    print(f"Scoring: score = -MSE distance; no cosine metrics", flush=True)
    print("=" * 72, flush=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_meta, train_audit = load_meta(cfg.train_csv, "train", positives_only=True, missing_chain_policy=cfg.missing_chain_policy)
    val_meta, val_audit = load_meta(cfg.val_csv, "val", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
    test_meta, test_audit = load_meta(cfg.test_csv, "test", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
    immrep_meta = None
    audits = [train_audit, val_audit, test_audit]
    if cfg.immrep_csv and str(cfg.immrep_csv).lower() not in ["none", ""]:
        immrep_meta, immrep_audit = load_meta(cfg.immrep_csv, "immrep_test", positives_only=False, missing_chain_policy=cfg.missing_chain_policy)
        audits.append(immrep_audit)
    pd.DataFrame(audits).to_csv(output_dir / "split_filter_audit.csv", index=False)

    L_T, L_P, L_H = compute_max_lengths(
        [train_meta, val_meta, test_meta] + ([] if immrep_meta is None else [immrep_meta]),
        cfg.max_tcr_len,
        cfg.max_pep_len,
        cfg.max_hla_len,
    )
    print(f"Max lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}", flush=True)

    train_loader = make_loader(OneHotFullTCRDataset(train_meta, L_T, L_P, L_H), cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader = make_loader(OneHotFullTCRDataset(val_meta, L_T, L_P, L_H), cfg.batch_size, False, cfg.num_workers, cfg.seed)
    test_loader = make_loader(OneHotFullTCRDataset(test_meta, L_T, L_P, L_H), cfg.batch_size, False, cfg.num_workers, cfg.seed)
    immrep_loader = None if immrep_meta is None else make_loader(OneHotFullTCRDataset(immrep_meta, L_T, L_P, L_H), cfg.batch_size, False, cfg.num_workers, cfg.seed)

    tcr = LowRankProjectionHead(VOCAB_SIZE, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(VOCAB_SIZE, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": tcr.parameters(), "lr": cfg.lr}, {"params": pmhc.parameters(), "lr": cfg.lr}],
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    lp = loss_params(cfg)

    best = {"metric": -np.inf, "epoch": None, "state": None, "bad_epochs": 0}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tcr.train()
        pmhc.train()
        running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
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

        val_eval = evaluate(val_loader, tcr, pmhc, device, cfg, "val", save_latents=False)
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, n_steps) for k, v in running.items()},
            **{f"val_onehot_vicreg_{k}": v for k, v in val_eval["metrics"]["onehot_vicreg"].items()},
            **{f"val_onehot_composition_{k}": v for k, v in val_eval["metrics"]["onehot_composition"].items()},
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        current = val_eval["metrics"]["onehot_vicreg"]["peptide_weighted_auroc"]
        improved = (not np.isnan(current)) and current > best["metric"] + 1e-4
        if improved:
            best = {
                "metric": current,
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
            f"val_vicreg_global={val_eval['metrics']['onehot_vicreg']['global_auroc']:.4f} | "
            f"val_comp_global={val_eval['metrics']['onehot_composition']['global_auroc']:.4f} | "
            f"val_vicreg_pep_weighted={current:.4f} | "
            f"val_comp_pep_weighted={val_eval['metrics']['onehot_composition']['peptide_weighted_auroc']:.4f} | "
            f"best_epoch={best['epoch']} | bad_epochs={best['bad_epochs']}",
            flush=True,
        )

        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint selected. Check labels and validation metrics.")

    tcr.load_state_dict(best["state"]["tcr"])
    pmhc.load_state_dict(best["state"]["pmhc"])

    final_evals = {
        "val": evaluate(val_loader, tcr, pmhc, device, cfg, "val", cfg.save_latents),
        "test": evaluate(test_loader, tcr, pmhc, device, cfg, "test", cfg.save_latents),
    }
    if immrep_loader is not None:
        final_evals["immrep_test"] = evaluate(immrep_loader, tcr, pmhc, device, cfg, "immrep_test", cfg.save_latents)

    output_paths = {}
    for split, eval_obj in final_evals.items():
        output_paths.update(save_eval_outputs(eval_obj, output_dir, figure_dir, split, cfg.save_latents))

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "config": asdict(cfg),
            "vocab": VOCAB,
            "max_lengths": {"L_T": L_T, "L_P": L_P, "L_H": L_H},
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "best_val_onehot_vicreg_peptide_weighted_auroc": best["metric"],
            "metrics": {split: obj["metrics"] for split, obj in final_evals.items()},
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(cfg),
        "model_family": "onehot",
        "seed": cfg.seed,
        "best_epoch": best["epoch"],
        "best_selection_metric": "val.onehot_vicreg.peptide_weighted_auroc",
        "best_selection_value": best["metric"],
        "metrics": {split: obj["metrics"] for split, obj in final_evals.items()},
        "paths": {
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "history.csv"),
            "run_config": str(output_dir / "run_config.json"),
            "split_filter_audit": str(output_dir / "split_filter_audit.csv"),
            "output_dir": str(output_dir),
            "figure_dir": str(figure_dir),
            **output_paths,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72, flush=True)
    print("Done: one-hot workshop run", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
