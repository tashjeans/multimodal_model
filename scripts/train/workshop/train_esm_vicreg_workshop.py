#!/usr/bin/env python3
"""
Workshop-paper ESMC VICReg pipeline.

Produces three paper rows from the same complete-chain multiview CSVs:
  1) pretrained_esmc_meanpool: genuine pretrained/raw ESMC frozen baseline
  2) finetuned_esmc_meanpool: LoRA/task-adapted ESMC frozen baseline
  3) esm_vicreg: VICReg projection heads trained on fine-tuned ESMC embeddings

Scoring convention throughout:
  mse_distance = mean((TCR_representation - pMHC_representation)^2)
  score = -mse_distance
Higher score = more binder-like.

No cosine scores are computed or saved. No legacy H/Hamiltonian naming is used.
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
# Generic parsing helpers
# ============================================================

def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_length(df: pd.DataFrame, length_candidates: List[str], seq_candidates: List[str]) -> Tuple[pd.Series, str]:
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

    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    if pep_col is None:
        raise ValueError(f"{source_name}: CSV must contain Peptide/peptide. Available columns: {list(out.columns)}")
    out["peptide_for_eval"] = out[pep_col].map(clean_seq)

    source_map = {"label_col": "constant_1" if label_col is None else label_col, "pep_col": pep_col}
    out["tcra_len"], source_map["tcra_len"] = extract_length(
        out,
        ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
    )
    out["tcrb_len"], source_map["tcrb_len"] = extract_length(
        out,
        ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
    )
    out["pep_len"], source_map["pep_len"] = extract_length(
        out,
        ["pep_len", "peptide_len"],
        ["Peptide", "peptide", "pep_seq", "peptide_seq"],
    )
    out["hla_len"], source_map["hla_len"] = extract_length(
        out,
        ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"],
    )
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source_map


def load_meta(csv_path: str, source_name: str, positives_only: bool, complete_only: bool = True) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(csv_path)
    meta, source_map = normalise_manifest(raw, source_name)

    if complete_only and (source_map["tcra_len"] == "__missing__" or source_map["tcrb_len"] == "__missing__"):
        raise ValueError(
            f"{source_name}: complete_only requested, but alpha/beta chain information could not be found. "
            f"Source map: {source_map}. Available columns: {list(raw.columns)}"
        )

    audit = {
        "split": source_name,
        "csv_path": str(csv_path),
        "csv_rows": int(len(raw)),
        "label_source": source_map["label_col"],
        "peptide_source": source_map["pep_col"],
        "tcra_len_source": source_map["tcra_len"],
        "tcrb_len_source": source_map["tcrb_len"],
        "pep_len_source": source_map["pep_len"],
        "hla_len_source": source_map["hla_len"],
        "positives_only": bool(positives_only),
        "complete_only": bool(complete_only),
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

    meta = meta[(meta["pep_len"] > 0) & (meta["hla_len"] > 0)].copy()
    audit["rows_after_required_sequence_filter"] = int(len(meta))

    if complete_only:
        meta = meta[meta["has_alpha"] & meta["has_beta"]].copy()
    audit["n_final"] = int(len(meta))
    audit["n_positive_final"] = int((meta["binding_flag"] == 1).sum())
    audit["n_negative_final"] = int((meta["binding_flag"] == 0).sum())
    audit["n_missing_alpha_final"] = int((~meta["has_alpha"]).sum())
    audit["n_missing_beta_final"] = int((~meta["has_beta"]).sum())

    print(
        f"{source_name}: csv_rows={audit['csv_rows']} | final={audit['n_final']} | "
        f"pos={audit['n_positive_final']} | neg={audit['n_negative_final']} | complete_only={complete_only}",
        flush=True,
    )
    if len(meta) == 0:
        raise RuntimeError(f"{source_name}: no rows remain after filtering")
    return meta.reset_index(drop=True), audit


def to_str_list(x) -> List[str]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif not isinstance(x, (list, tuple)):
        x = [x]
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x]


# ============================================================
# Paired fine-tuned/pretrained ESM shard dataset
# ============================================================

def build_pair_index(shards_dir: Path, source_name: str, embedding_name: str) -> Dict[str, Tuple[Path, int, int]]:
    shards_dir = Path(shards_dir)
    shard_paths = sorted(shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {shards_dir} for {source_name}/{embedding_name}")

    index: Dict[str, Tuple[Path, int, int]] = {}
    seen = 0
    duplicates = []
    for sp in shard_paths:
        shard = torch.load(sp, map_location="cpu")
        for bidx, batch in enumerate(shard):
            pair_ids = to_str_list(batch["pair_id"])
            for ridx, pid in enumerate(pair_ids):
                seen += 1
                if pid in index:
                    duplicates.append(pid)
                else:
                    index[pid] = (sp, bidx, ridx)
    if duplicates:
        raise RuntimeError(f"{source_name}/{embedding_name}: duplicate pair_ids in shards. Examples: {duplicates[:10]}")
    print(f"{source_name}/{embedding_name}: shard rows indexed={seen} | unique_pair_ids={len(index)} | dir={shards_dir}", flush=True)
    return index


class PairedESMRowDataset(Dataset):
    """Rows aligned by pair_id across fine-tuned and, optionally, pretrained ESMC shard directories.

    The training split does not need pretrained embeddings. Loading them during training is
    unnecessarily slow and doubles CPU I/O. Validation/test/IMMREP include pretrained
    embeddings so that the frozen pretrained baseline can be computed.

    Rows can be ordered by fine-tuned shard location for evaluation-time cache locality.
    For training, the DataLoader should still shuffle rows, because VICReg variance
    and covariance terms depend on mixed mini-batches. Pretrained embeddings are
    deliberately not loaded for training; they are used only for evaluation baselines.
    """

    def __init__(
        self,
        finetuned_dir: Path,
        pretrained_dir: Optional[Path],
        meta: pd.DataFrame,
        source_name: str,
        include_pretrained: bool = True,
        order_by_finetuned_shard: bool = True,
    ):
        self.finetuned_dir = Path(finetuned_dir)
        self.pretrained_dir = None if pretrained_dir is None else Path(pretrained_dir)
        self.include_pretrained = bool(include_pretrained)
        self.meta = meta.reset_index(drop=True)
        self.meta_by_pid = {str(r["pair_id"]): r for _, r in self.meta.iterrows()}
        self.source_name = source_name

        self.ft_index = build_pair_index(self.finetuned_dir, source_name, "finetuned")
        self.pre_index = {}
        if self.include_pretrained:
            if self.pretrained_dir is None:
                raise ValueError(f"{source_name}: include_pretrained=True but pretrained_dir is None")
            self.pre_index = build_pair_index(self.pretrained_dir, source_name, "pretrained")

        requested = [str(x) for x in self.meta["pair_id"].tolist()]
        missing_ft = [pid for pid in requested if pid not in self.ft_index]
        missing_pre = [pid for pid in requested if self.include_pretrained and pid not in self.pre_index]
        if missing_ft or missing_pre:
            raise RuntimeError(
                f"{source_name}: pair_id alignment failure. "
                f"missing_finetuned={len(missing_ft)} examples={missing_ft[:10]} | "
                f"missing_pretrained={len(missing_pre)} examples={missing_pre[:10]}"
            )

        if order_by_finetuned_shard:
            requested = sorted(
                requested,
                key=lambda pid: (str(self.ft_index[pid][0]), int(self.ft_index[pid][1]), int(self.ft_index[pid][2])),
            )
        self.pair_ids = requested
        print(
            f"{source_name}: paired ESM rows kept={len(self.pair_ids)} | "
            f"include_pretrained={self.include_pretrained} | order_by_finetuned_shard={order_by_finetuned_shard}",
            flush=True,
        )

        self._cache_ft_path = None
        self._cache_ft_data = None
        self._cache_pre_path = None
        self._cache_pre_data = None

    def __len__(self) -> int:
        return len(self.pair_ids)

    def _load(self, sp: Path, branch: str):
        if branch == "ft":
            if self._cache_ft_path != sp:
                self._cache_ft_data = torch.load(sp, map_location="cpu")
                self._cache_ft_path = sp
            return self._cache_ft_data
        if branch == "pre":
            if self._cache_pre_path != sp:
                self._cache_pre_data = torch.load(sp, map_location="cpu")
                self._cache_pre_path = sp
            return self._cache_pre_data
        raise ValueError(branch)

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pair_ids[idx]
        row = self.meta_by_pid[pid]

        ft_sp, ft_bidx, ft_ridx = self.ft_index[pid]
        ft_batch = self._load(ft_sp, "ft")[ft_bidx]

        item = {
            "ft_emb_T": ft_batch["emb_T"][ft_ridx].float(),
            "ft_mask_T": ft_batch["mask_T"][ft_ridx].bool(),
            "ft_emb_P": ft_batch["emb_P"][ft_ridx].float(),
            "ft_mask_P": ft_batch["mask_P"][ft_ridx].bool(),
            "ft_emb_H": ft_batch["emb_H"][ft_ridx].float(),
            "ft_mask_H": ft_batch["mask_H"][ft_ridx].bool(),
            "binding_flag": int(row["binding_flag"]),
            "pair_id": pid,
            "peptide": str(row["peptide_for_eval"]),
            "has_alpha": bool(row["has_alpha"]),
            "has_beta": bool(row["has_beta"]),
            "tcra_len": int(row["tcra_len"]),
            "tcrb_len": int(row["tcrb_len"]),
            "pep_len": int(row["pep_len"]),
            "hla_len": int(row["hla_len"]),
        }

        if self.include_pretrained:
            pre_sp, pre_bidx, pre_ridx = self.pre_index[pid]
            pre_batch = self._load(pre_sp, "pre")[pre_bidx]
            item.update({
                "pre_emb_T": pre_batch["emb_T"][pre_ridx].float(),
                "pre_mask_T": pre_batch["mask_T"][pre_ridx].bool(),
                "pre_emb_P": pre_batch["emb_P"][pre_ridx].float(),
                "pre_mask_P": pre_batch["mask_P"][pre_ridx].bool(),
                "pre_emb_H": pre_batch["emb_H"][pre_ridx].float(),
                "pre_mask_H": pre_batch["mask_H"][pre_ridx].bool(),
            })
        return item


def paired_esm_collate(rows: List[Dict]) -> Dict:
    out = {}
    tensor_keys = [
        "ft_emb_T", "ft_mask_T", "ft_emb_P", "ft_mask_P", "ft_emb_H", "ft_mask_H",
    ]
    if "pre_emb_T" in rows[0]:
        tensor_keys.extend([
            "pre_emb_T", "pre_mask_T", "pre_emb_P", "pre_mask_P", "pre_emb_H", "pre_mask_H",
        ])
    for k in tensor_keys:
        out[k] = torch.stack([r[k] for r in rows], dim=0)
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "pep_len", "hla_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


# ============================================================
# Projection heads
# ============================================================

class ESMProjectionHead(nn.Module):
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
        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pep_encoder(emb_P, mask_P), self.hla_encoder(emb_H, mask_H)], dim=-1)


# ============================================================
# Loss and scoring
# ============================================================

def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
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


def meanpool_score(
    emb_T: torch.Tensor,
    mask_T: torch.Tensor,
    emb_P: torch.Tensor,
    mask_P: torch.Tensor,
    emb_H: torch.Tensor,
    mask_H: torch.Tensor,
    device: torch.device,
    R_PH: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = masked_mean_pool(emb_T.to(device), mask_T.to(device), eps)
    P = masked_mean_pool(emb_P.to(device), mask_P.to(device), eps)
    HLA = masked_mean_pool(emb_H.to(device), mask_H.to(device), eps)
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
# Evaluation
# ============================================================

ALL_EVAL_MODELS = ("esm_vicreg", "finetuned_esmc_meanpool", "pretrained_esmc_meanpool")
EPOCH_VAL_MODELS = ("esm_vicreg",)


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    cfg: "RunConfig",
    split: str,
    save_latents: bool,
    model_names: Tuple[str, ...] = ALL_EVAL_MODELS,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    model_names = tuple(model_names)
    if not model_names:
        raise ValueError("model_names must be non-empty")
    unknown = set(model_names) - set(ALL_EVAL_MODELS)
    if unknown:
        raise ValueError(f"Unknown model_names: {sorted(unknown)}")
    need_pretrained = "pretrained_esmc_meanpool" in model_names
    pair_ids, peptides = [], []
    labels_all = []
    meta_cols = {k: [] for k in ["has_alpha", "has_beta", "tcra_len", "tcrb_len", "pep_len", "hla_len"]}
    scores = {name: [] for name in model_names}
    distances = {name: [] for name in model_names}
    latent_store = {} if save_latents else None
    if save_latents:
        if "esm_vicreg" in model_names:
            latent_store["zT_esm_vicreg"] = []
            latent_store["zPH_esm_vicreg"] = []
        if "finetuned_esmc_meanpool" in model_names:
            latent_store["T_finetuned_meanpool"] = []
            latent_store["PH_finetuned_meanpool"] = []
        if "pretrained_esmc_meanpool" in model_names:
            latent_store["T_pretrained_meanpool"] = []
            latent_store["PH_pretrained_meanpool"] = []

    running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
    n_steps = 0
    lp = loss_params(cfg)

    for batch in loader:
        zT = tcr_proj(batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
        zPH = pmhc_proj(
            batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
            batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
        )
        loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
        vicreg_score, vicreg_mse = score_from_vectors(zT, zPH)

        batch_scores = {}
        batch_distances = {}
        if "esm_vicreg" in model_names:
            batch_scores["esm_vicreg"] = vicreg_score
            batch_distances["esm_vicreg"] = vicreg_mse

        ft_T = ft_PH = pre_T = pre_PH = None
        if "finetuned_esmc_meanpool" in model_names:
            ft_score, ft_mse, ft_T, ft_PH = meanpool_score(
                batch["ft_emb_T"], batch["ft_mask_T"], batch["ft_emb_P"], batch["ft_mask_P"], batch["ft_emb_H"], batch["ft_mask_H"],
                device, cfg.R_PH, cfg.eps_pool,
            )
            batch_scores["finetuned_esmc_meanpool"] = ft_score
            batch_distances["finetuned_esmc_meanpool"] = ft_mse

        if need_pretrained:
            if "pre_emb_T" not in batch:
                raise RuntimeError(
                    "pretrained_esmc_meanpool requested but loader batches lack pretrained embeddings. "
                    "Use a PairedESMRowDataset with include_pretrained=True."
                )
            pre_score, pre_mse, pre_T, pre_PH = meanpool_score(
                batch["pre_emb_T"], batch["pre_mask_T"], batch["pre_emb_P"], batch["pre_mask_P"], batch["pre_emb_H"], batch["pre_mask_H"],
                device, cfg.R_PH, cfg.eps_pool,
            )
            if "pretrained_esmc_meanpool" in model_names:
                batch_scores["pretrained_esmc_meanpool"] = pre_score
                batch_distances["pretrained_esmc_meanpool"] = pre_mse

        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        labels_all.append(labels)
        pair_ids.extend([str(x) for x in batch["pair_id"]])
        peptides.append(np.array(batch["peptide"], dtype=str))
        for k in meta_cols:
            meta_cols[k].append(batch[k].detach().cpu().numpy())

        for name in model_names:
            scores[name].append(batch_scores[name].detach().cpu().numpy())
            distances[name].append(batch_distances[name].detach().cpu().numpy())

        if save_latents:
            if "esm_vicreg" in model_names:
                latent_store["zT_esm_vicreg"].append(zT.detach().cpu().numpy())
                latent_store["zPH_esm_vicreg"].append(zPH.detach().cpu().numpy())
            if "finetuned_esmc_meanpool" in model_names:
                latent_store["T_finetuned_meanpool"].append(ft_T.detach().cpu().numpy())
                latent_store["PH_finetuned_meanpool"].append(ft_PH.detach().cpu().numpy())
            if "pretrained_esmc_meanpool" in model_names:
                latent_store["T_pretrained_meanpool"].append(pre_T.detach().cpu().numpy())
                latent_store["PH_pretrained_meanpool"].append(pre_PH.detach().cpu().numpy())

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
    for model_name in model_names:
        m, table = metrics_for_scores(labels_np, scores_np[model_name], peptides_np, cfg.partial_auc_max_fpr)
        m.update({
            "mse_distance_mean": float(np.mean(distances_np[model_name])),
            "mse_distance_std": float(np.std(distances_np[model_name])),
        })
        metrics[model_name] = m
        peptide_tables[model_name] = table

    running_avg = {k: v / max(1, n_steps) for k, v in running.items()}
    if "esm_vicreg" in model_names:
        metrics["esm_vicreg"].update({f"eval_{k}": val for k, val in running_avg.items()})

    predictions = pd.DataFrame({
        "pair_id": pair_ids,
        "peptide": peptides_np,
        "label": labels_np,
        **{k: meta_np[k] for k in meta_np},
    })
    for model_name in model_names:
        predictions[f"{model_name}_score"] = scores_np[model_name]
        predictions[f"{model_name}_mse_distance"] = distances_np[model_name]

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


# ============================================================
# Plotting/output
# ============================================================

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
    run_tag: str = "esm_vicreg_finetuned_complete"
    checkpoint_root: str = "/home/natasha/multimodal_model/models/checkpoints/workshop/esm_vicreg_finetuned_complete"
    output_root: str = "/home/natasha/multimodal_model/models/outputs/workshop/esm_vicreg_finetuned_complete"
    figure_root: str = "/home/natasha/multimodal_model/models/figures/workshop/esm_vicreg_finetuned_complete"

    finetuned_embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids"
    pretrained_embed_root: str = "/home/natasha/multimodal_model/models/embeddings/raw_esmc_300m_multiview_ids"
    finetuned_immrep_shard_dir: str = "/home/natasha/multimodal_model/models/embeddings/immrep_test_set/test"
    pretrained_immrep_shard_dir: str = "/home/natasha/multimodal_model/models/embeddings/raw_esmc_300m_multiview_ids/immrep_test"

    train_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"

    seed: int = 31
    batch_size: int = 8
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


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=paired_esm_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def split_dirs(cfg: RunConfig, split: str) -> Tuple[Path, Path]:
    if split in ["train", "val", "test"]:
        return Path(cfg.finetuned_embed_root) / split, Path(cfg.pretrained_embed_root) / split
    if split == "immrep_test":
        return Path(cfg.finetuned_immrep_shard_dir), Path(cfg.pretrained_immrep_shard_dir)
    raise ValueError(split)


def infer_shapes(datasets: List[Dataset]) -> Tuple[int, int, int, int]:
    D = None
    L_T = L_P = L_H = 0
    for ds in datasets:
        sample = ds[0]
        d_here = sample["ft_emb_T"].shape[-1]
        if D is None:
            D = d_here
        elif D != d_here:
            raise ValueError(f"Embedding dimension mismatch across datasets: {D} vs {d_here}")
        L_T = max(L_T, int(sample["ft_emb_T"].shape[0]))
        L_P = max(L_P, int(sample["ft_emb_P"].shape[0]))
        L_H = max(L_H, int(sample["ft_emb_H"].shape[0]))
    print(f"Detected projection shapes: D={D}, L_T={L_T}, L_P={L_P}, L_H={L_H}", flush=True)
    return int(D), L_T, L_P, L_H


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tag", default=RunConfig.run_tag)
    parser.add_argument("--checkpoint-root", default=RunConfig.checkpoint_root)
    parser.add_argument("--output-root", default=RunConfig.output_root)
    parser.add_argument("--figure-root", default=RunConfig.figure_root)
    parser.add_argument("--finetuned-embed-root", default=RunConfig.finetuned_embed_root)
    parser.add_argument("--pretrained-embed-root", default=RunConfig.pretrained_embed_root)
    parser.add_argument("--finetuned-immrep-shard-dir", default=RunConfig.finetuned_immrep_shard_dir)
    parser.add_argument("--pretrained-immrep-shard-dir", default=RunConfig.pretrained_immrep_shard_dir)
    parser.add_argument("--train-csv", default=RunConfig.train_csv)
    parser.add_argument("--val-csv", default=RunConfig.val_csv)
    parser.add_argument("--test-csv", default=RunConfig.test_csv)
    parser.add_argument("--immrep-csv", default=RunConfig.immrep_csv)
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
        finetuned_embed_root=args.finetuned_embed_root,
        pretrained_embed_root=args.pretrained_embed_root,
        finetuned_immrep_shard_dir=args.finetuned_immrep_shard_dir,
        pretrained_immrep_shard_dir=args.pretrained_immrep_shard_dir,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        immrep_csv=args.immrep_csv,
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
    print("Workshop ESMC VICReg run", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Seed: {cfg.seed}", flush=True)
    print(f"Fine-tuned root: {cfg.finetuned_embed_root}", flush=True)
    print(f"Pretrained root: {cfg.pretrained_embed_root}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Checkpoint dir: {checkpoint_dir}", flush=True)
    print(f"Figure dir: {figure_dir}", flush=True)
    print("Scoring: score = -MSE distance; no cosine metrics", flush=True)
    print("Training loader: fine-tuned embeddings only, shuffle=True, default batch_size=8", flush=True)
    print("Per-epoch val: esm_vicreg only (no pretrained shards, no meanpool baselines)", flush=True)
    print("Final eval: esm_vicreg + finetuned_esmc_meanpool + pretrained_esmc_meanpool", flush=True)
    print("=" * 72, flush=True)

    with open(output_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    train_meta, train_audit = load_meta(cfg.train_csv, "train", positives_only=True, complete_only=True)
    val_meta, val_audit = load_meta(cfg.val_csv, "val", positives_only=False, complete_only=True)
    test_meta, test_audit = load_meta(cfg.test_csv, "test", positives_only=False, complete_only=True)
    audits = [train_audit, val_audit, test_audit]
    immrep_meta = None
    if cfg.immrep_csv and str(cfg.immrep_csv).lower() not in ["none", ""]:
        immrep_meta, immrep_audit = load_meta(cfg.immrep_csv, "immrep_test", positives_only=False, complete_only=True)
        audits.append(immrep_audit)
    pd.DataFrame(audits).to_csv(output_dir / "split_filter_audit.csv", index=False)

    train_ft, train_pre = split_dirs(cfg, "train")
    val_ft, val_pre = split_dirs(cfg, "val")
    test_ft, test_pre = split_dirs(cfg, "test")

    # Training uses only fine-tuned embeddings. Pretrained embeddings are loaded only for evaluation baselines.
    train_ds = PairedESMRowDataset(train_ft, None, train_meta, "train", include_pretrained=False)
    # Per-epoch validation scores only esm_vicreg; no pretrained shard I/O or meanpool baselines.
    val_ds_epoch = PairedESMRowDataset(val_ft, None, val_meta, "val", include_pretrained=False)
    # Final evaluation compares all three paper rows (vicreg + both frozen meanpool baselines).
    val_ds_final = PairedESMRowDataset(val_ft, val_pre, val_meta, "val", include_pretrained=True)
    test_ds = PairedESMRowDataset(test_ft, test_pre, test_meta, "test", include_pretrained=True)
    immrep_ds = None
    if immrep_meta is not None:
        imm_ft, imm_pre = split_dirs(cfg, "immrep_test")
        immrep_ds = PairedESMRowDataset(imm_ft, imm_pre, immrep_meta, "immrep_test", include_pretrained=True)

    print(
        f"Loaded paired rows: train={len(train_ds)} | val={len(val_ds_epoch)} | test={len(test_ds)}" +
        (f" | immrep_test={len(immrep_ds)}" if immrep_ds is not None else ""),
        flush=True,
    )

    D, L_T, L_P, L_H = infer_shapes([train_ds, val_ds_epoch, test_ds] + ([] if immrep_ds is None else [immrep_ds]))

    # Training must match the original working VICReg setup: shuffled row-level
    # mini-batches, batch_size default 8. VICReg's variance/covariance terms are
    # batch-statistical, so ordered shard-local batches can materially change the
    # optimisation trajectory. Evaluation remains unshuffled for deterministic output.
    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers, cfg.seed)
    val_loader_epoch = make_loader(val_ds_epoch, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    val_loader_final = make_loader(val_ds_final, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)
    immrep_loader = None if immrep_ds is None else make_loader(immrep_ds, cfg.batch_size, False, cfg.num_workers, cfg.seed)

    tcr = ESMProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
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
            zT = tcr(batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
            zPH = pmhc(
                batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
                batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
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

        val_eval = evaluate(
            val_loader_epoch, tcr, pmhc, device, cfg, "val", save_latents=False, model_names=EPOCH_VAL_MODELS,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v / max(1, n_steps) for k, v in running.items()},
        }
        for model_name, metric_dict in val_eval["metrics"].items():
            row.update({f"val_{model_name}_{k}": v for k, v in metric_dict.items()})
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        current = val_eval["metrics"]["esm_vicreg"]["peptide_weighted_auroc"]
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
            f"val_vicreg_global={val_eval['metrics']['esm_vicreg']['global_auroc']:.4f} | "
            f"val_vicreg_pep_weighted={current:.4f} | "
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
        "val": evaluate(val_loader_final, tcr, pmhc, device, cfg, "val", cfg.save_latents),
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
            "shapes": {"D": D, "L_T": L_T, "L_P": L_P, "L_H": L_H},
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "best_val_esm_vicreg_peptide_weighted_auroc": best["metric"],
            "metrics": {split: obj["metrics"] for split, obj in final_evals.items()},
        },
        checkpoint_path,
    )

    summary = {
        "config": asdict(cfg),
        "model_family": "esm",
        "seed": cfg.seed,
        "best_epoch": best["epoch"],
        "best_selection_metric": "val.esm_vicreg.peptide_weighted_auroc",
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
    print("Done: ESMC workshop run", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
