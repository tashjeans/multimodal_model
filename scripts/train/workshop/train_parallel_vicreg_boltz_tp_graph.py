#!/usr/bin/env python3
"""Train a sparse Boltz TCR<->peptide graph scorer in parallel with frozen one-hot VICReg.

Design
------
1. Load the previously trained one-hot VICReg checkpoint and freeze it.
2. Use validation labels to fit a small supervised structural scorer.
3. The graph uses only the directed TCR-peptide Boltz blocks:
      z_TP: TCR receivers, peptide senders
      z_PT: peptide receivers, TCR senders
4. No learned projection, compression, or reconstruction of z is used.
   Raw L2 norms select and weight top-k directed edges.
5. The graph never updates the embeddings used by VICReg. It returns a separate
   scalar structural compatibility logit.
6. Fuse standardised scores on a held-out validation-selection subset:
      fused = zscore(vicreg_score) + lambda * zscore(graph_logit)
   lambda >= 0 and lambda=0 is always available, so the selected fusion cannot
   be forced to use a harmful graph branch.
7. Test and IMMREP are evaluated only after graph checkpoint and lambda selection.

This is intentionally a low-capacity first experiment. HLA remains in the frozen
VICReg pMHC encoder but is excluded from the graph branch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, auc, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Reproducibility and sequence utilities
# =============================================================================

AA20 = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
VOCAB["SEP"] = len(VOCAB)  # matches the frozen one-hot VICReg checkpoint
VOCAB_SIZE = len(VOCAB)
UNK_IDX = VOCAB["X"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def onehot_encode(seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = clean_seq(seq)
    x = torch.zeros(max_len, VOCAB_SIZE, dtype=torch.float32)
    mask = torch.zeros(max_len, dtype=torch.bool)
    n = min(len(seq), max_len)
    for i, aa in enumerate(seq[:n]):
        x[i, VOCAB.get(aa, UNK_IDX)] = 1.0
        mask[i] = True
    return x, mask


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((c for c in candidates if c in df.columns), None)


def extract_length(df: pd.DataFrame, length_cols: Sequence[str], seq_cols: Sequence[str]) -> Tuple[pd.Series, str]:
    col = first_existing_col(df, length_cols)
    if col is not None:
        return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int), col
    col = first_existing_col(df, seq_cols)
    if col is not None:
        return df[col].map(clean_seq).str.len().astype(int), f"inferred_from:{col}"
    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "__missing__"


# =============================================================================
# Manifest handling and exact Boltz alignment
# =============================================================================


def normalise_manifest(df: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if "pair_id" not in df.columns:
        raise ValueError(f"{split}: missing pair_id")
    if "boltz_embedding_npz" not in df.columns and "z_path" not in df.columns:
        raise ValueError(f"{split}: missing boltz_embedding_npz/z_path")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    out["binding_flag"] = 1 if label_col is None else pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    tcr_col = first_existing_col(out, ["TCR_full", "tcr_full", "full_tcr", "TCR", "tcr", "TCR_full_norm"])
    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq", "Peptide_norm"])
    hla_col = first_existing_col(out, [
        "HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla",
        "mhc_seq", "MHC_sequence", "HLA_sequence_norm",
    ])
    if None in (tcr_col, pep_col, hla_col):
        raise ValueError(f"{split}: required sequence columns not found; columns={list(out.columns)}")

    out["TCR_full_norm"] = out[tcr_col].map(clean_seq)
    out["Peptide_norm"] = out[pep_col].map(clean_seq)
    out["HLA_sequence_norm"] = out[hla_col].map(clean_seq)
    out["peptide_for_eval"] = out["Peptide_norm"]

    source = {
        "label_col": "constant_1" if label_col is None else str(label_col),
        "tcr_col": str(tcr_col),
        "pep_col": str(pep_col),
        "hla_col": str(hla_col),
    }
    out["tcra_len"], source["tcra_len"] = extract_length(
        out,
        ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
    )
    out["tcrb_len"], source["tcrb_len"] = extract_length(
        out,
        ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
    )
    out["pep_len"], source["pep_len"] = extract_length(
        out, ["pep_len", "peptide_len"], ["Peptide", "peptide", "pep_seq", "peptide_seq", "Peptide_norm"]
    )
    out["hla_len"], source["hla_len"] = extract_length(
        out,
        ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence", "HLA_sequence_norm"],
    )
    out["tcr_total_len"] = out["TCR_full_norm"].str.len().astype(int)
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source


def resolve_npz_path(value: str, repo_root: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else repo_root / p


def restore_meta_dtypes(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    for col in ["has_alpha", "has_beta", "z_exists", "z_aligned", "tcr_length_consistent"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["binding_flag", "tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len", "z_raw_len", "z_dim"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def load_meta(
    csv_path: str,
    split: str,
    repo_root: Path,
    filtered_meta_cache_dir: Optional[Path],
) -> Tuple[pd.DataFrame, Dict]:
    """Load complete-chain rows with aligned Boltz z.

    Reuses the audited filtered manifests produced by train_onehot_boltz_mpnn.py
    when available. Otherwise it performs lightweight path/sequence filtering;
    exact z shape is still checked when each edge cache is created.
    """
    cached_path = None if filtered_meta_cache_dir is None else filtered_meta_cache_dir / f"{split}_filtered.csv"
    source_kind = "csv"
    if cached_path is not None and cached_path.exists():
        raw = restore_meta_dtypes(pd.read_csv(cached_path))
        source_kind = "audited_filtered_cache"
    else:
        raw = pd.read_csv(csv_path)

    meta, source = normalise_manifest(raw, split)
    if source["tcra_len"] == "__missing__" or source["tcrb_len"] == "__missing__":
        raise ValueError(f"{split}: alpha/beta lengths are required for exact raw-z alignment; source={source}")

    meta["tcr_length_consistent"] = meta["tcr_total_len"] == (meta["tcra_len"] + meta["tcrb_len"])
    meta = meta[
        (meta["tcr_total_len"] > 0)
        & (meta["pep_len"] > 0)
        & (meta["hla_len"] > 0)
        & meta["has_alpha"]
        & meta["has_beta"]
        & meta["tcr_length_consistent"]
    ].copy()

    if "z_path" not in meta.columns:
        meta["z_path"] = meta["boltz_embedding_npz"].map(lambda x: str(resolve_npz_path(x, repo_root)))
    else:
        meta["z_path"] = meta["z_path"].map(lambda x: str(resolve_npz_path(x, repo_root)))

    if "z_exists" in meta.columns:
        meta = meta[meta["z_exists"]].copy()
    if "z_aligned" in meta.columns:
        meta = meta[meta["z_aligned"]].copy()

    exists = meta["z_path"].map(lambda x: Path(x).exists())
    n_missing = int((~exists).sum())
    meta = meta[exists].copy().reset_index(drop=True)
    if meta.empty:
        raise RuntimeError(f"{split}: no usable rows remain")

    audit = {
        "split": split,
        "csv_path": csv_path,
        "source_kind": source_kind,
        "n_final": int(len(meta)),
        "n_positive": int((meta["binding_flag"] == 1).sum()),
        "n_negative": int((meta["binding_flag"] == 0).sum()),
        "n_missing_z_paths": n_missing,
        **source,
    }
    print(
        f"{split}: rows={len(meta)} pos={audit['n_positive']} neg={audit['n_negative']} "
        f"source={source_kind} missing_z={n_missing}",
        flush=True,
    )
    return meta, audit


def load_raw_z(path: Path) -> np.ndarray:
    with np.load(path) as data:
        if "z" not in data.files:
            raise KeyError(f"{path}: no z key; keys={list(data.files)}")
        z = np.asarray(data["z"])
    if z.ndim == 4:
        if z.shape[0] != 1:
            raise ValueError(f"{path}: unexpected z shape {z.shape}")
        z = z[0]
    if z.ndim != 3 or z.shape[0] != z.shape[1]:
        raise ValueError(f"{path}: expected [L,L,D], got {z.shape}")
    return z


def safe_cache_name(pair_id: str, z_path: str) -> str:
    digest = hashlib.sha1(f"{pair_id}|{z_path}".encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def block_summary(q: np.ndarray) -> np.ndarray:
    flat = np.asarray(q, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return np.zeros(5, dtype=np.float32)
    n1 = max(1, int(math.ceil(0.01 * flat.size)))
    n5 = max(1, int(math.ceil(0.05 * flat.size)))
    # np.partition avoids sorting the entire block.
    top1 = np.partition(flat, flat.size - n1)[-n1:]
    top5 = np.partition(flat, flat.size - n5)[-n5:]
    return np.asarray([
        float(flat.mean()),
        float(flat.std()),
        float(flat.max()),
        float(top1.mean()),
        float(top5.mean()),
    ], dtype=np.float32)


def compute_directed_edge_norms(
    z_path: Path,
    t_len: int,
    p_len: int,
    h_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_z = load_raw_z(z_path)
    expected = t_len + p_len + h_len
    if raw_z.shape[0] != expected:
        raise ValueError(f"{z_path}: z length={raw_z.shape[0]} expected={expected}")

    z_tp = raw_z[:t_len, t_len:t_len + p_len, :]
    z_pt = raw_z[t_len:t_len + p_len, :t_len, :]
    # Norm is deterministic: z is not compressed or learned through a projection.
    q_tp = np.linalg.norm(z_tp.astype(np.float32, copy=False), axis=-1).astype(np.float32)
    q_pt = np.linalg.norm(z_pt.astype(np.float32, copy=False), axis=-1).astype(np.float32)
    summary = np.concatenate([block_summary(q_tp), block_summary(q_pt)]).astype(np.float32)
    return q_tp, q_pt, summary


# =============================================================================
# Frozen one-hot VICReg backbone (exact checkpoint architecture)
# =============================================================================


class LowRankProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float = 0.1):
        super().__init__()
        self.D, self.rL, self.rD, self.d, self.L_max = D, rL, rD, d, L_max
        self.B_c = nn.Parameter(torch.empty(D, rD))
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))
        nn.init.xavier_uniform_(self.B_c)
        nn.init.xavier_uniform_(self.A_c)
        nn.init.xavier_uniform_(self.H_c)
        self.expander = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d)
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L_pad, D_in = emb.shape
        if D_in != self.D or L_pad > self.L_max:
            raise ValueError(
                f"projection mismatch: got {tuple(emb.shape)}, expected D={self.D}, L<={self.L_max}"
            )
        lengths = mask.sum(dim=1)
        outputs = []
        for b in range(B):
            n = int(lengths[b].item())
            if n == 0:
                outputs.append(torch.zeros(self.d, device=emb.device, dtype=emb.dtype))
                continue
            x = emb[b, :n] * mask[b, :n, None].to(emb.dtype)
            y = x @ self.B_c
            u = self.A_c[:n].T @ y
            outputs.append(u.reshape(-1) @ self.H_c)
        return self.expander(torch.stack(outputs))


class PMHCProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P: int, L_H: int, R_PH: float, dropout: float):
        super().__init__()
        d_p = int(round(R_PH * d))
        d_h = d - d_p
        if min(d_p, d_h) <= 0:
            raise ValueError(f"invalid R_PH={R_PH}")
        self.pep_encoder = LowRankProjectionHead(D, rL, rD, d_p, L_P, dropout)
        self.hla_encoder = LowRankProjectionHead(D, rL, rD, d_h, L_H, dropout)

    def forward(self, x_p: torch.Tensor, m_p: torch.Tensor, x_h: torch.Tensor, m_h: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.pep_encoder(x_p, m_p),
            self.hla_encoder(x_h, m_h),
        ], dim=-1)


class FrozenOneHotVICReg(nn.Module):
    def __init__(self, checkpoint_path: Path, device: torch.device):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        lengths = checkpoint["max_lengths"]
        self.L_T = int(lengths["L_T"])
        self.L_P = int(lengths["L_P"])
        self.L_H = int(lengths["L_H"])
        checkpoint_vocab = checkpoint.get("vocab")
        if checkpoint_vocab is not None and len(checkpoint_vocab) != VOCAB_SIZE:
            raise ValueError(
                f"checkpoint vocab size={len(checkpoint_vocab)} but script vocab size={VOCAB_SIZE}"
            )
        self.tcr = LowRankProjectionHead(
            VOCAB_SIZE, int(cfg["rL"]), int(cfg["rD"]), int(cfg["d"]), self.L_T, float(cfg["dropout"])
        )
        self.pmhc = PMHCProjectionHead(
            VOCAB_SIZE, int(cfg["rL"]), int(cfg["rD"]), int(cfg["d"]),
            self.L_P, self.L_H, float(cfg["R_PH"]), float(cfg["dropout"]),
        )
        self.tcr.load_state_dict(checkpoint["tcr_state_dict"])
        self.pmhc.load_state_dict(checkpoint["pmhc_state_dict"])
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()
        self.to(device)

    @torch.no_grad()
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_t = self.tcr(batch["emb_T"], batch["mask_T"])
        z_ph = self.pmhc(batch["emb_P"], batch["mask_P"], batch["emb_H"], batch["mask_H"])
        return -(z_t - z_ph).square().mean(dim=-1)


# =============================================================================
# Dataset with small cached directed edge-norm matrices
# =============================================================================


class ParallelGraphDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        split_name: str,
        L_T: int,
        L_P: int,
        L_H: int,
        edge_cache_dir: Path,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.meta = meta.reset_index(drop=True)
        self.split_name = split_name
        self.L_T, self.L_P, self.L_H = int(L_T), int(L_P), int(L_H)
        self.edge_cache_dir = edge_cache_dir / split_name
        self.edge_cache_dir.mkdir(parents=True, exist_ok=True)
        if sample_weights is None:
            self.sample_weights = np.ones(len(self.meta), dtype=np.float32)
        else:
            if len(sample_weights) != len(self.meta):
                raise ValueError("sample_weights length mismatch")
            self.sample_weights = np.asarray(sample_weights, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.meta)

    def _load_or_create_edges(self, row: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self.edge_cache_dir / safe_cache_name(str(row.pair_id), str(row.z_path))
        if cache_path.exists():
            try:
                with np.load(cache_path) as data:
                    q_tp = np.asarray(data["q_tp"], dtype=np.float32)
                    q_pt = np.asarray(data["q_pt"], dtype=np.float32)
                    summary = np.asarray(data["summary"], dtype=np.float32)
                    if q_tp.shape == (int(row.tcr_total_len), int(row.pep_len)) and q_pt.shape == (int(row.pep_len), int(row.tcr_total_len)):
                        return q_tp, q_pt, summary
            except Exception:
                cache_path.unlink(missing_ok=True)

        q_tp, q_pt, summary = compute_directed_edge_norms(
            Path(row.z_path), int(row.tcr_total_len), int(row.pep_len), int(row.hla_len)
        )
        tmp = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            q_tp=q_tp.astype(np.float16),
            q_pt=q_pt.astype(np.float16),
            summary=summary.astype(np.float32),
        )
        try:
            os.replace(tmp, cache_path)
        except FileNotFoundError:
            # Another worker may already have completed the same cache entry.
            pass
        finally:
            tmp.unlink(missing_ok=True)
        return q_tp, q_pt, summary

    def __getitem__(self, idx: int) -> Dict:
        row = self.meta.iloc[idx]
        t_len, p_len, h_len = int(row.tcr_total_len), int(row.pep_len), int(row.hla_len)
        if t_len > self.L_T or p_len > self.L_P or h_len > self.L_H:
            raise ValueError(
                f"{row.pair_id}: sequence lengths {(t_len,p_len,h_len)} exceed checkpoint caps "
                f"{(self.L_T,self.L_P,self.L_H)}"
            )

        x_t, m_t = onehot_encode(row.TCR_full_norm, self.L_T)
        x_p, m_p = onehot_encode(row.Peptide_norm, self.L_P)
        x_h, m_h = onehot_encode(row.HLA_sequence_norm, self.L_H)
        q_tp_raw, q_pt_raw, summary = self._load_or_create_edges(row)

        q_tp = torch.zeros(self.L_T, self.L_P, dtype=torch.float32)
        q_pt = torch.zeros(self.L_P, self.L_T, dtype=torch.float32)
        q_tp[:t_len, :p_len] = torch.from_numpy(q_tp_raw)
        q_pt[:p_len, :t_len] = torch.from_numpy(q_pt_raw)

        return {
            "emb_T": x_t, "mask_T": m_t,
            "emb_P": x_p, "mask_P": m_p,
            "emb_H": x_h, "mask_H": m_h,
            "q_TP": q_tp, "q_PT": q_pt,
            "struct_summary": torch.from_numpy(summary),
            "binding_flag": int(row.binding_flag),
            "sample_weight": float(self.sample_weights[idx]),
            "pair_id": str(row.pair_id),
            "peptide": str(row.peptide_for_eval),
            "tcr_total_len": t_len, "pep_len": p_len, "hla_len": h_len,
        }


def collate_rows(rows: List[Dict]) -> Dict:
    tensor_keys = [
        "emb_T", "mask_T", "emb_P", "mask_P", "emb_H", "mask_H",
        "q_TP", "q_PT", "struct_summary",
    ]
    out = {key: torch.stack([row[key] for row in rows]) for key in tensor_keys}
    out["binding_flag"] = torch.tensor([row["binding_flag"] for row in rows], dtype=torch.float32)
    out["sample_weight"] = torch.tensor([row["sample_weight"] for row in rows], dtype=torch.float32)
    for key in ["pair_id", "peptide"]:
        out[key] = [row[key] for row in rows]
    for key in ["tcr_total_len", "pep_len", "hla_len"]:
        out[key] = torch.tensor([row[key] for row in rows], dtype=torch.long)
    return out


# =============================================================================
# Sparse deterministic edge weighting and graph scorer
# =============================================================================


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask[:, :, None].to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = x.masked_fill(~mask[:, :, None], torch.finfo(x.dtype).min)
    values = masked.max(dim=1).values
    return torch.where(torch.isfinite(values), values, torch.zeros_like(values))


def topk_normalised_weights(
    q: torch.Tensor,
    receiver_mask: torch.Tensor,
    sender_mask: torch.Tensor,
    k: int,
    temperature: float,
    edge_dropout: float,
    training: bool,
) -> torch.Tensor:
    """Return sparse [B, receiver, sender] weights from raw edge norms.

    Scores are standardised only within each receiver's valid sender set before
    top-k selection. This controls scale without learning a z projection.
    """
    pair_mask = receiver_mask[:, :, None] & sender_mask[:, None, :]
    q_float = q.float()
    count = pair_mask.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = (q_float * pair_mask).sum(dim=-1, keepdim=True) / count
    var = (((q_float - mean) ** 2) * pair_mask).sum(dim=-1, keepdim=True) / count
    scores = (q_float - mean) / torch.sqrt(var + 1e-6)
    scores = scores.masked_fill(~pair_mask, -1e9)

    k_eff = min(max(1, int(k)), q.shape[-1])
    values, indices = torch.topk(scores, k=k_eff, dim=-1)
    valid_top = torch.gather(pair_mask, dim=-1, index=indices)
    logits = values / max(float(temperature), 1e-6)
    top_weights = torch.softmax(logits, dim=-1) * valid_top.to(logits.dtype)

    if training and edge_dropout > 0:
        keep = (torch.rand_like(top_weights) >= edge_dropout).to(top_weights.dtype)
        dropped = top_weights * keep
        use_dropped = dropped.sum(dim=-1, keepdim=True) > 0
        top_weights = torch.where(use_dropped, dropped, top_weights)

    top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    dense = torch.zeros_like(q_float)
    dense.scatter_(dim=-1, index=indices, src=top_weights)
    return dense * receiver_mask[:, :, None].to(dense.dtype)


class SparseDirectionalTPGraph(nn.Module):
    """One-step directional TCR<->peptide message passing and scalar scoring."""

    def __init__(
        self,
        hidden_dim: int,
        top_k: int,
        temperature: float,
        edge_dropout: float,
        dropout: float,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.edge_dropout = float(edge_dropout)

        self.tcr_node = nn.Sequential(nn.Linear(VOCAB_SIZE, hidden_dim), nn.GELU())
        self.pep_node = nn.Sequential(nn.Linear(VOCAB_SIZE, hidden_dim), nn.GELU())
        self.p_to_t = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.t_to_p = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.update_t = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_p = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_scale_t = nn.Parameter(torch.tensor(0.1))
        self.update_scale_p = nn.Parameter(torch.tensor(0.1))
        # log1p controls scale while preserving absolute between-pair magnitude;
        # no learned z projection or per-sample normalisation is applied.
        self.summary_transform = nn.Identity()

        # mean, max, absolute mean difference, product of means = 6*hidden;
        # plus 10 log-scaled deterministic raw-z norm summaries.
        feature_dim = 6 * hidden_dim + 10
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor], z_mode: str = "boltz") -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x_t, x_p = batch["emb_T"], batch["emb_P"]
        m_t, m_p = batch["mask_T"], batch["mask_P"]
        q_tp, q_pt = batch["q_TP"], batch["q_PT"]
        summary = batch["struct_summary"]

        if z_mode == "zero":
            q_tp = torch.zeros_like(q_tp)
            q_pt = torch.zeros_like(q_pt)
            summary = torch.zeros_like(summary)
        elif z_mode == "shuffled":
            if q_tp.shape[0] > 1:
                q_tp = torch.roll(q_tp, shifts=1, dims=0)
                q_pt = torch.roll(q_pt, shifts=1, dims=0)
                summary = torch.roll(summary, shifts=1, dims=0)
            else:
                q_tp = torch.zeros_like(q_tp)
                q_pt = torch.zeros_like(q_pt)
                summary = torch.zeros_like(summary)
        elif z_mode != "boltz":
            raise ValueError(f"unknown z_mode={z_mode}")

        h_t = self.tcr_node(x_t) * m_t[:, :, None].to(x_t.dtype)
        h_p = self.pep_node(x_p) * m_p[:, :, None].to(x_p.dtype)

        w_tp = topk_normalised_weights(
            q_tp, m_t, m_p, self.top_k, self.temperature, self.edge_dropout, self.training
        )
        w_pt = topk_normalised_weights(
            q_pt, m_p, m_t, self.top_k, self.temperature, self.edge_dropout, self.training
        )
        msg_t = torch.bmm(w_tp, self.p_to_t(h_p))
        msg_p = torch.bmm(w_pt, self.t_to_p(h_t))

        delta_t = torch.tanh(self.update_t(torch.cat([h_t, msg_t], dim=-1)))
        delta_p = torch.tanh(self.update_p(torch.cat([h_p, msg_p], dim=-1)))
        h_t_new = (h_t + self.update_scale_t * delta_t) * m_t[:, :, None].to(h_t.dtype)
        h_p_new = (h_p + self.update_scale_p * delta_p) * m_p[:, :, None].to(h_p.dtype)

        mean_t, max_t = masked_mean(h_t_new, m_t), masked_max(h_t_new, m_t)
        mean_p, max_p = masked_mean(h_p_new, m_p), masked_max(h_p_new, m_p)
        features = torch.cat([
            mean_t,
            max_t,
            mean_p,
            max_p,
            torch.abs(mean_t - mean_p),
            mean_t * mean_p,
            self.summary_transform(torch.log1p(summary.clamp_min(0.0))),
        ], dim=-1)
        logit = self.scorer(features).squeeze(-1)
        diagnostics = {
            "update_scale_t": self.update_scale_t.detach(),
            "update_scale_p": self.update_scale_p.detach(),
            "mean_abs_delta_t": delta_t.abs().mean().detach(),
            "mean_abs_delta_p": delta_p.abs().mean().detach(),
            "mean_selected_weight_tp": w_tp[w_tp > 0].mean().detach() if torch.any(w_tp > 0) else torch.zeros((), device=w_tp.device),
            "mean_selected_weight_pt": w_pt[w_pt > 0].mean().detach() if torch.any(w_pt > 0) else torch.zeros((), device=w_pt.device),
        }
        return logit, diagnostics


# =============================================================================
# Validation split and weighting
# =============================================================================


def stratified_peptide_label_split(meta: pd.DataFrame, select_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each peptide-label cell, preserving both labels where counts allow."""
    rng = np.random.default_rng(seed)
    fit_indices: List[int] = []
    select_indices: List[int] = []
    assignments = []

    for (peptide, label), group in meta.groupby(["peptide_for_eval", "binding_flag"], sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        if n <= 1:
            n_select = 0
        else:
            n_select = max(1, int(round(select_fraction * n)))
            n_select = min(n_select, n - 1)
        selected = indices[:n_select]
        fitted = indices[n_select:]
        select_indices.extend(selected.tolist())
        fit_indices.extend(fitted.tolist())
        assignments.extend({"row_index": int(i), "subset": "select", "peptide": peptide, "label": int(label)} for i in selected)
        assignments.extend({"row_index": int(i), "subset": "fit", "peptide": peptide, "label": int(label)} for i in fitted)

    fit = meta.loc[sorted(fit_indices)].reset_index(drop=True)
    select = meta.loc[sorted(select_indices)].reset_index(drop=True)
    assignment_df = pd.DataFrame(assignments).sort_values("row_index").reset_index(drop=True)
    if fit.empty or select.empty:
        raise RuntimeError("validation split produced an empty fit or selection subset")
    return fit, select, assignment_df


def peptide_label_sample_weights(meta: pd.DataFrame) -> np.ndarray:
    counts = meta.groupby(["peptide_for_eval", "binding_flag"])["pair_id"].transform("count").to_numpy(dtype=np.float64)
    weights = 1.0 / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights.astype(np.float32)


# =============================================================================
# Metrics, score fusion, evaluation
# =============================================================================


def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(average_precision_score(y, s))


def safe_partial_auc_raw(y: np.ndarray, s: np.ndarray, max_fpr: float) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    fpr_e = np.concatenate([fpr[:stop], [max_fpr]])
    tpr_e = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    return float(auc(fpr_e, tpr_e))


def safe_partial_auc_mcclish(y: np.ndarray, s: np.ndarray, max_fpr: float) -> float:
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s, max_fpr=max_fpr))


def per_peptide_table(y: np.ndarray, s: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    frame = pd.DataFrame({"label": y.astype(int), "score": s.astype(float), "peptide": peptides.astype(str)})
    rows = []
    for peptide, group in frame.groupby("peptide", sort=True):
        yy, ss = group.label.to_numpy(), group.score.to_numpy()
        valid = len(np.unique(yy)) == 2
        rows.append({
            "peptide": peptide,
            "n": int(len(group)),
            "n_pos": int(yy.sum()),
            "n_neg": int((yy == 0).sum()),
            "valid": bool(valid),
            "auroc": safe_auroc(yy, ss) if valid else float("nan"),
            "auc0.1_raw": safe_partial_auc_raw(yy, ss, max_fpr) if valid else float("nan"),
            "auc0.1_mcclish": safe_partial_auc_mcclish(yy, ss, max_fpr) if valid else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid = table[table.valid]
    if valid.empty:
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
            "peptide_macro_auroc": float(valid.auroc.mean()),
            "peptide_weighted_auroc": float(np.average(valid.auroc, weights=valid.n)),
            "peptide_macro_auc0.1_mcclish": float(valid["auc0.1_mcclish"].mean()),
            "peptide_weighted_auc0.1_mcclish": float(np.average(valid["auc0.1_mcclish"], weights=valid.n)),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": int(len(valid)),
        }
    return table, summary


def metrics_for_scores(y: np.ndarray, s: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[Dict, pd.DataFrame]:
    table, peptide_summary = per_peptide_table(y, s, peptides, max_fpr)
    metrics = {
        "n_examples": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()),
        "global_auroc": safe_auroc(y, s),
        "auprc": safe_auprc(y, s),
        "global_auc0.1_raw": safe_partial_auc_raw(y, s, max_fpr),
        "global_auc0.1_mcclish": safe_partial_auc_mcclish(y, s, max_fpr),
        "score_mean": float(np.mean(s)),
        "score_std": float(np.std(s)),
        **peptide_summary,
    }
    return metrics, table


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (values - mean) / max(std, 1e-8)


def fit_score_stats(fit_predictions: pd.DataFrame) -> Dict[str, float]:
    return {
        "vicreg_mean": float(fit_predictions.vicreg_score.mean()),
        "vicreg_std": float(fit_predictions.vicreg_score.std(ddof=0)),
        "graph_mean": float(fit_predictions.graph_logit.mean()),
        "graph_std": float(fit_predictions.graph_logit.std(ddof=0)),
    }


def add_fused_score(predictions: pd.DataFrame, stats: Dict[str, float], fusion_lambda: float) -> pd.DataFrame:
    out = predictions.copy()
    out["vicreg_score_standardised"] = zscore(out.vicreg_score.to_numpy(), stats["vicreg_mean"], stats["vicreg_std"])
    out["graph_logit_standardised"] = zscore(out.graph_logit.to_numpy(), stats["graph_mean"], stats["graph_std"])
    out["fused_score"] = out["vicreg_score_standardised"] + float(fusion_lambda) * out["graph_logit_standardised"]
    return out


def select_fusion_lambda(
    fit_predictions: pd.DataFrame,
    select_predictions: pd.DataFrame,
    lambda_max: float,
    lambda_step: float,
    max_fpr: float,
) -> Tuple[float, Dict[str, float], pd.DataFrame]:
    stats = fit_score_stats(fit_predictions)
    rows = []
    best_lambda, best_metric = 0.0, -math.inf
    lambdas = np.arange(0.0, lambda_max + 0.5 * lambda_step, lambda_step)
    for value in lambdas:
        fused = add_fused_score(select_predictions, stats, float(value))
        metrics, _ = metrics_for_scores(
            fused.label.to_numpy(dtype=int), fused.fused_score.to_numpy(), fused.peptide.to_numpy(dtype=str), max_fpr
        )
        current = metrics["peptide_weighted_auc0.1_mcclish"]
        rows.append({"lambda": float(value), **metrics})
        if np.isfinite(current) and (current > best_metric + 1e-12 or (abs(current - best_metric) <= 1e-12 and value < best_lambda)):
            best_metric = float(current)
            best_lambda = float(value)
    return best_lambda, stats, pd.DataFrame(rows)


@torch.no_grad()
def predict(
    loader: DataLoader,
    vicreg: FrozenOneHotVICReg,
    graph: SparseDirectionalTPGraph,
    device: torch.device,
    z_mode: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    graph.eval()
    rows = []
    diagnostics = {
        "update_scale_t": 0.0,
        "update_scale_p": 0.0,
        "mean_abs_delta_t": 0.0,
        "mean_abs_delta_p": 0.0,
        "mean_selected_weight_tp": 0.0,
        "mean_selected_weight_pt": 0.0,
    }
    steps = 0
    for batch_cpu in loader:
        batch = {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch_cpu.items()}
        vicreg_score = vicreg(batch)
        graph_logit, diag = graph(batch, z_mode=z_mode)
        labels = batch["binding_flag"].cpu().numpy().astype(int)
        v = vicreg_score.cpu().numpy()
        g = graph_logit.cpu().numpy()
        for i in range(len(labels)):
            rows.append({
                "pair_id": batch["pair_id"][i],
                "peptide": batch["peptide"][i],
                "label": int(labels[i]),
                "vicreg_score": float(v[i]),
                "graph_logit": float(g[i]),
            })
        for key in diagnostics:
            diagnostics[key] += float(diag[key].cpu())
        steps += 1
    return pd.DataFrame(rows), {key: value / max(steps, 1) for key, value in diagnostics.items()}


def evaluate_prediction_table(predictions: pd.DataFrame, max_fpr: float) -> Tuple[Dict[str, Dict], Dict[str, pd.DataFrame]]:
    metrics, tables = {}, {}
    for score_name in ["vicreg_score", "graph_logit", "fused_score"]:
        if score_name not in predictions.columns:
            continue
        metric, table = metrics_for_scores(
            predictions.label.to_numpy(dtype=int),
            predictions[score_name].to_numpy(dtype=float),
            predictions.peptide.to_numpy(dtype=str),
            max_fpr,
        )
        metrics[score_name] = metric
        tables[score_name] = table
    return metrics, tables


# =============================================================================
# Configuration, loaders, plots, outputs
# =============================================================================


@dataclass
class RunConfig:
    run_tag: str = "tp_topk8_rawznorm"
    repo_root: str = "/home/natasha/multimodal_model"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    vicreg_checkpoint: str = "/home/natasha/multimodal_model/models/checkpoints/workshop/onehot_vicreg_complete/seed_31/best.pt"
    filtered_meta_cache_dir: str = "/home/natasha/multimodal_model/data/cache/onehot_boltz_mpnn"
    edge_cache_dir: str = "/home/natasha/multimodal_model/data/cache/boltz_tp_edge_norms"
    checkpoint_root: str = "/home/natasha/multimodal_model/models/checkpoints/workshop/onehot_boltz_parallel_graph"
    output_root: str = "/home/natasha/multimodal_model/models/outputs/workshop/onehot_boltz_parallel_graph"
    figure_root: str = "/home/natasha/multimodal_model/models/figures/workshop/onehot_boltz_parallel_graph"

    seed: int = 31
    val_select_fraction: float = 0.20
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 30
    min_epochs: int = 5
    patience: int = 8
    hidden_dim: int = 32
    top_k: int = 8
    temperature: float = 1.0
    edge_dropout: float = 0.10
    dropout: float = 0.10
    z_mode: str = "boltz"

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip_norm: float = 5.0
    partial_auc_max_fpr: float = 0.1
    fusion_lambda_max: float = 2.0
    fusion_lambda_step: float = 0.05
    overwrite: bool = False


def prepare_dirs(cfg: RunConfig) -> Tuple[Path, Path, Path]:
    suffix = Path(cfg.run_tag) / f"seed_{cfg.seed}"
    checkpoint_dir = Path(cfg.checkpoint_root) / suffix
    output_dir = Path(cfg.output_root) / suffix
    figure_dir = Path(cfg.figure_root) / suffix
    if cfg.overwrite:
        for path in [checkpoint_dir, output_dir, figure_dir]:
            if path.exists():
                shutil.rmtree(path)
    for path in [checkpoint_dir, output_dir, figure_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, output_dir, figure_dir


def make_loader(dataset: Dataset, cfg: RunConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed) if shuffle else None
    kwargs = {}
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate_rows,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )


def plot_training(history: pd.DataFrame, out_path: Path) -> None:
    if history.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history.epoch, history.train_loss, label="fit BCE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Parallel graph training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_lambda_search(search: pd.DataFrame, out_path: Path) -> None:
    if search.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(search["lambda"], search["peptide_weighted_auc0.1_mcclish"])
    plt.xlabel("Fusion lambda")
    plt.ylabel("Selection peptide-weighted McClish AUC0.1")
    plt.title("Fusion weight search")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_split_outputs(
    split: str,
    predictions: pd.DataFrame,
    metrics: Dict[str, Dict],
    tables: Dict[str, pd.DataFrame],
    output_dir: Path,
) -> Dict[str, str]:
    paths = {}
    pred_path = output_dir / f"{split}_predictions.csv"
    predictions.to_csv(pred_path, index=False)
    paths[f"{split}_predictions"] = str(pred_path)
    metrics_path = output_dir / f"{split}_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)
    paths[f"{split}_metrics"] = str(metrics_path)
    for score_name, table in tables.items():
        path = output_dir / f"{split}_{score_name}_per_peptide.csv"
        table.to_csv(path, index=False)
        paths[f"{split}_{score_name}_per_peptide"] = str(path)
    return paths


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in [
        "run_tag", "repo_root", "val_csv", "test_csv", "immrep_csv", "vicreg_checkpoint",
        "filtered_meta_cache_dir", "edge_cache_dir", "checkpoint_root", "output_root", "figure_root",
    ]:
        parser.add_argument("--" + name.replace("_", "-"), default=getattr(RunConfig, name))
    for name in ["seed", "batch_size", "num_workers", "epochs", "min_epochs", "patience", "hidden_dim", "top_k"]:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(RunConfig, name))
    for name in [
        "val_select_fraction", "temperature", "edge_dropout", "dropout", "lr", "weight_decay",
        "grad_clip_norm", "partial_auc_max_fpr", "fusion_lambda_max", "fusion_lambda_step",
    ]:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(RunConfig, name))
    parser.add_argument("--z-mode", choices=["boltz", "zero", "shuffled"], default=RunConfig.z_mode)
    parser.add_argument("--overwrite", action="store_true")
    return RunConfig(**vars(parser.parse_args()))


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    cfg = parse_args()
    if not (0.0 < cfg.val_select_fraction < 0.5):
        raise ValueError("val_select_fraction must be in (0, 0.5)")
    if cfg.top_k <= 0:
        raise ValueError("top_k must be positive")

    set_seed(cfg.seed)
    checkpoint_dir, output_dir, figure_dir = prepare_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(output_dir / "run_config.json", "w") as handle:
        json.dump(asdict(cfg), handle, indent=2)

    print("=" * 88, flush=True)
    print("Frozen one-hot VICReg + parallel sparse Boltz TCR-peptide graph", flush=True)
    print(f"device={device} seed={cfg.seed} z_mode={cfg.z_mode}", flush=True)
    print(f"VICReg checkpoint: {cfg.vicreg_checkpoint}", flush=True)
    print(f"output: {output_dir}", flush=True)
    print("No z projection/reconstruction; raw z norms -> directed top-k masks and weights.", flush=True)
    print("Test and IMMREP are evaluated only after val-fit/val-select model locking.", flush=True)
    print("=" * 88, flush=True)

    vicreg = FrozenOneHotVICReg(Path(cfg.vicreg_checkpoint), device)
    print(f"frozen VICReg lengths: T={vicreg.L_T} P={vicreg.L_P} H={vicreg.L_H}", flush=True)

    repo_root = Path(cfg.repo_root)
    filtered_cache = Path(cfg.filtered_meta_cache_dir) if cfg.filtered_meta_cache_dir else None
    val_meta, val_audit = load_meta(cfg.val_csv, "val", repo_root, filtered_cache)
    test_meta, test_audit = load_meta(cfg.test_csv, "test", repo_root, filtered_cache)
    immrep_meta, immrep_audit = load_meta(cfg.immrep_csv, "immrep_test", repo_root, filtered_cache)
    pd.DataFrame([val_audit, test_audit, immrep_audit]).to_csv(output_dir / "split_filter_audit.csv", index=False)

    val_fit, val_select, assignments = stratified_peptide_label_split(val_meta, cfg.val_select_fraction, cfg.seed)
    assignments.to_csv(output_dir / "val_fit_select_assignments.csv", index=False)
    print(
        f"validation split: fit={len(val_fit)} select={len(val_select)} "
        f"fit_pos={(val_fit.binding_flag == 1).sum()} select_pos={(val_select.binding_flag == 1).sum()}",
        flush=True,
    )

    edge_cache_dir = Path(cfg.edge_cache_dir)
    fit_weights = peptide_label_sample_weights(val_fit)
    datasets = {
        "val_fit": ParallelGraphDataset(val_fit, "val", vicreg.L_T, vicreg.L_P, vicreg.L_H, edge_cache_dir, fit_weights),
        "val_select": ParallelGraphDataset(val_select, "val", vicreg.L_T, vicreg.L_P, vicreg.L_H, edge_cache_dir),
        "val_full": ParallelGraphDataset(val_meta, "val", vicreg.L_T, vicreg.L_P, vicreg.L_H, edge_cache_dir),
        "test": ParallelGraphDataset(test_meta, "test", vicreg.L_T, vicreg.L_P, vicreg.L_H, edge_cache_dir),
        "immrep_test": ParallelGraphDataset(immrep_meta, "immrep_test", vicreg.L_T, vicreg.L_P, vicreg.L_H, edge_cache_dir),
    }
    loaders = {
        name: make_loader(dataset, cfg, shuffle=(name == "val_fit")) for name, dataset in datasets.items()
    }

    graph = SparseDirectionalTPGraph(
        cfg.hidden_dim, cfg.top_k, cfg.temperature, cfg.edge_dropout, cfg.dropout
    ).to(device)
    optimizer = torch.optim.AdamW(graph.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {
        "metric": -math.inf,
        "epoch": None,
        "state": None,
        "lambda": 0.0,
        "score_stats": None,
        "fusion_search": None,
        "bad_epochs": 0,
    }
    history_rows = []

    for epoch in range(1, cfg.epochs + 1):
        graph.train()
        loss_sum, n_examples = 0.0, 0
        for batch_cpu in loaders["val_fit"]:
            batch = {key: (value.to(device, non_blocking=True) if torch.is_tensor(value) else value) for key, value in batch_cpu.items()}
            logits, _ = graph(batch, z_mode=cfg.z_mode)
            losses = F.binary_cross_entropy_with_logits(logits, batch["binding_flag"], reduction="none")
            loss = (losses * batch["sample_weight"]).sum() / batch["sample_weight"].sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(graph.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(logits)
            n_examples += len(logits)
        scheduler.step()

        fit_pred, fit_diag = predict(loaders["val_fit"], vicreg, graph, device, cfg.z_mode)
        select_pred, select_diag = predict(loaders["val_select"], vicreg, graph, device, cfg.z_mode)
        fusion_lambda, score_stats, fusion_search = select_fusion_lambda(
            fit_pred,
            select_pred,
            cfg.fusion_lambda_max,
            cfg.fusion_lambda_step,
            cfg.partial_auc_max_fpr,
        )
        fit_fused = add_fused_score(fit_pred, score_stats, fusion_lambda)
        select_fused = add_fused_score(select_pred, score_stats, fusion_lambda)
        fit_metrics, _ = evaluate_prediction_table(fit_fused, cfg.partial_auc_max_fpr)
        select_metrics, _ = evaluate_prediction_table(select_fused, cfg.partial_auc_max_fpr)
        current = select_metrics["fused_score"]["peptide_weighted_auc0.1_mcclish"]

        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": loss_sum / max(n_examples, 1),
            "fusion_lambda": fusion_lambda,
            "fit_graph_global_auroc": fit_metrics["graph_logit"]["global_auroc"],
            "select_vicreg_global_auroc": select_metrics["vicreg_score"]["global_auroc"],
            "select_graph_global_auroc": select_metrics["graph_logit"]["global_auroc"],
            "select_fused_global_auroc": select_metrics["fused_score"]["global_auroc"],
            "select_vicreg_pep_w_mcclish": select_metrics["vicreg_score"]["peptide_weighted_auc0.1_mcclish"],
            "select_graph_pep_w_mcclish": select_metrics["graph_logit"]["peptide_weighted_auc0.1_mcclish"],
            "select_fused_pep_w_mcclish": current,
            **{f"fit_diag_{key}": value for key, value in fit_diag.items()},
            **{f"select_diag_{key}": value for key, value in select_diag.items()},
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(output_dir / "history.csv", index=False)

        improved = np.isfinite(current) and current > best["metric"] + 1e-4
        if improved:
            best = {
                "metric": float(current),
                "epoch": epoch,
                "state": copy.deepcopy(graph.state_dict()),
                "lambda": float(fusion_lambda),
                "score_stats": score_stats,
                "fusion_search": fusion_search.copy(),
                "bad_epochs": 0,
            }
        else:
            best["bad_epochs"] += 1

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} loss={row['train_loss']:.4f} "
            f"select_vicreg_global={row['select_vicreg_global_auroc']:.4f} "
            f"select_graph_global={row['select_graph_global_auroc']:.4f} "
            f"select_fused_global={row['select_fused_global_auroc']:.4f} "
            f"select_fused_pep_w_mcclish={current:.4f} lambda={fusion_lambda:.2f} "
            f"best_epoch={best['epoch']} bad={best['bad_epochs']}",
            flush=True,
        )
        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best["state"] is None or best["score_stats"] is None:
        raise RuntimeError("No graph checkpoint selected; inspect validation peptide labels and metrics")
    graph.load_state_dict(best["state"])

    best["fusion_search"].to_csv(output_dir / "best_fusion_lambda_search.csv", index=False)
    plot_training(pd.DataFrame(history_rows), figure_dir / "training_loss.png")
    plot_lambda_search(best["fusion_search"], figure_dir / "best_fusion_lambda_search.png")

    # Model is now locked. Test and IMMREP are first evaluated here.
    final_predictions = {}
    final_diagnostics = {}
    for split in ["val_fit", "val_select", "val_full", "test", "immrep_test"]:
        pred, diag = predict(loaders[split], vicreg, graph, device, cfg.z_mode)
        final_predictions[split] = add_fused_score(pred, best["score_stats"], best["lambda"])
        final_diagnostics[split] = diag

    all_metrics = {}
    output_paths = {}
    for split, predictions in final_predictions.items():
        metrics, tables = evaluate_prediction_table(predictions, cfg.partial_auc_max_fpr)
        all_metrics[split] = metrics
        output_paths.update(save_split_outputs(split, predictions, metrics, tables, output_dir))

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save({
        "config": asdict(cfg),
        "graph_state_dict": graph.state_dict(),
        "best_epoch": best["epoch"],
        "best_selection_metric": "val_select.fused_score.peptide_weighted_auc0.1_mcclish",
        "best_selection_value": best["metric"],
        "fusion_lambda": best["lambda"],
        "score_stats_from_val_fit": best["score_stats"],
        "frozen_vicreg_checkpoint": cfg.vicreg_checkpoint,
        "metrics": all_metrics,
    }, checkpoint_path)

    summary = {
        "config": asdict(cfg),
        "model_family": "frozen_onehot_vicreg_plus_parallel_sparse_tp_graph",
        "method": {
            "vicreg": "frozen independently computed TCR and pMHC representations",
            "graph": "one-step directional TCR<->peptide message passing",
            "z_usage": "raw z L2 norms; receiver-wise top-k edge selection and fixed softmax weights; no learned z projection",
            "fusion": "standardised VICReg score + non-negative lambda * standardised graph logit",
            "graph_training_data": "val_fit only",
            "selection_data": "val_select only",
            "test_policy": "test and IMMREP evaluated after locking graph checkpoint and lambda",
        },
        "best_epoch": best["epoch"],
        "best_selection_metric": "val_select.fused_score.peptide_weighted_auc0.1_mcclish",
        "best_selection_value": best["metric"],
        "fusion_lambda": best["lambda"],
        "score_stats_from_val_fit": best["score_stats"],
        "diagnostics": final_diagnostics,
        "metrics": all_metrics,
        "paths": {
            "checkpoint": str(checkpoint_path),
            "history": str(output_dir / "history.csv"),
            "run_config": str(output_dir / "run_config.json"),
            "split_filter_audit": str(output_dir / "split_filter_audit.csv"),
            "val_assignments": str(output_dir / "val_fit_select_assignments.csv"),
            "fusion_search": str(output_dir / "best_fusion_lambda_search.csv"),
            "output_dir": str(output_dir),
            "figure_dir": str(figure_dir),
            **output_paths,
        },
    }
    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 88, flush=True)
    print(
        f"Done. best_epoch={best['epoch']} selection McClish={best['metric']:.4f} "
        f"fusion_lambda={best['lambda']:.2f}",
        flush=True,
    )
    for split in ["test", "immrep_test"]:
        m = all_metrics[split]
        print(
            f"{split}: VICReg global={m['vicreg_score']['global_auroc']:.4f} "
            f"graph global={m['graph_logit']['global_auroc']:.4f} "
            f"fused global={m['fused_score']['global_auroc']:.4f} "
            f"fused pep-w McClish={m['fused_score']['peptide_weighted_auc0.1_mcclish']:.4f}",
            flush=True,
        )
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 88, flush=True)


if __name__ == "__main__":
    main()
