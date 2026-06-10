#!/usr/bin/env python3
"""
Boltz-s VICReg diagnostic run with strict MSE scoring and IMMREP McClish metrics.

Purpose
-------
This script trains a positive-only VICReg model using Boltz single/token
representations (`s`) rather than ESM embeddings. It is intended to be the direct
Boltz-s analogue of the existing plain ESM VICReg diagnostic script.

This version includes optional input LayerNorm inside each Boltz-s token projection
head. LayerNorm is applied to the input token features before the low-rank
projection and does not L2-normalise the final VICReg latents.

    TCR view  = Boltz s tokens for TCR alpha + TCR beta
    pMHC view = Boltz s tokens for peptide + HLA/MHC

The model is evaluated on validation/test/IMMREP examples containing positives
and decoys/negatives. Training uses positives only.

Expected Boltz embedding format
-------------------------------
Each Boltz embedding file should contain an array named `s` with shape:

    (1, L, 384) or (L, 384)

where L is the total complex length. The script slices the token dimension using
lengths from the multiview CSVs:

    [TCRA | TCRB | peptide | HLA]

Only complete complexes are retained: tcra_len > 0, tcrb_len > 0, pep_len > 0,
hla_len > 0.

Score convention
----------------
Evaluation uses the same geometry as the VICReg invariance term:

    mse_distance = mean((zT - zPH)^2)
    score = -mse_distance

Higher score therefore means more binder-like. Cosine is saved only as a
diagnostic.

IMMREP metric
-------------
The script preserves the McClish-standardised partial AUROC convention:

    roc_auc_score(labels, scores, max_fpr=0.1)

This is reported globally and per peptide, with macro and weighted per-peptide
summaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
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
    roc_curve,
    auc,
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
# CSV/meta handling
# ============================================================

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalise_manifest(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if "pair_id" not in df.columns:
        raise ValueError(f"{source_name} must contain pair_id")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    out["binding_flag"] = 1 if label_col is None else out[label_col].astype(int)

    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    out["peptide_for_eval"] = out[pep_col].astype(str) if pep_col is not None else out["pair_id"].astype(str)

    length_specs = {
        "tcra_len": ["tcra_len", "tcr_alpha_len", "cdr3a_len", "alpha_len"],
        "tcrb_len": ["tcrb_len", "tcr_beta_len", "cdr3b_len", "beta_len"],
        "pep_len": ["pep_len", "peptide_len"],
        "hla_len": ["hla_len", "mhc_len", "mhca_len"],
    }
    seq_specs = {
        "tcra_len": ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
        "tcrb_len": ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
        "pep_len": ["Peptide", "peptide", "pep_seq", "peptide_seq"],
        "hla_len": ["hla", "HLA", "hla_seq", "mhc", "mhc_seq"],
    }

    for target_col, candidates in length_specs.items():
        src = first_existing_col(out, candidates)
        if src is not None:
            out[target_col] = pd.to_numeric(out[src], errors="coerce").fillna(0).astype(int)
            continue
        seq_col = first_existing_col(out, seq_specs[target_col])
        if seq_col is not None:
            out[target_col] = out[seq_col].fillna("").astype(str).str.len().astype(int)
        else:
            out[target_col] = 0

    return out


def allowed_meta_from_csv(
    csv_path: str,
    source_name: str,
    positives_only: bool = False,
    complete_only: bool = True,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    meta = normalise_manifest(raw, source_name)
    before = len(meta)
    if positives_only:
        meta = meta[meta["binding_flag"].astype(int) == 1].copy()
    after_label = len(meta)
    if complete_only:
        meta = meta[
            (meta["tcra_len"] > 0)
            & (meta["tcrb_len"] > 0)
            & (meta["pep_len"] > 0)
            & (meta["hla_len"] > 0)
        ].copy()
    print(
        f"{source_name}: csv_rows={before} | after_label_filter={after_label} | "
        f"complete_complexes={len(meta)} | positives_only={positives_only}",
        flush=True,
    )
    return meta.reset_index(drop=True)


# ============================================================
# Boltz-s file loading
# ============================================================

def _normalise_pair_stem(pair_id: str) -> List[str]:
    """Return plausible file stem variants for a pair_id."""
    pid = str(pair_id)
    stems = [pid]
    if not pid.startswith("embeddings_"):
        stems.append(f"embeddings_{pid}")
    if not pid.startswith("prediction_"):
        stems.append(f"prediction_{pid}")
    return list(dict.fromkeys(stems))


class BoltzSFileResolver:
    """Locate Boltz .npz embedding files for pair_ids.

    The resolver is deliberately permissive because Boltz output folders vary
    between runs. It first tries common direct paths, then optionally builds a
    recursive filename index under the supplied root.
    """

    def __init__(self, root: Path, source_name: str, recursive_index: bool = True):
        self.root = Path(root)
        self.source_name = source_name
        self.recursive_index = recursive_index
        if not self.root.exists():
            raise FileNotFoundError(f"{source_name}: Boltz root does not exist: {self.root}")
        self._index: Optional[Dict[str, Path]] = None

    def _build_index(self) -> Dict[str, Path]:
        print(f"{self.source_name}: building recursive .npz index under {self.root}", flush=True)
        index: Dict[str, Path] = {}
        for p in self.root.rglob("*.npz"):
            index[p.name] = p
            index[p.stem] = p
        print(f"{self.source_name}: indexed {len(index)} .npz name/stem entries", flush=True)
        return index

    def resolve(self, pair_id: str) -> Optional[Path]:
        stems = _normalise_pair_stem(pair_id)
        candidates: List[Path] = []
        for stem in stems:
            candidates.extend([
                self.root / f"{stem}.npz",
                self.root / str(pair_id) / f"{stem}.npz",
                self.root / str(pair_id) / "embeddings.npz",
                self.root / str(pair_id) / "prediction_embeddings.npz",
                self.root / str(pair_id) / "boltz_embeddings.npz",
            ])
        for c in candidates:
            if c.exists():
                return c

        if self.recursive_index:
            if self._index is None:
                self._index = self._build_index()
            for stem in stems:
                for key in [f"{stem}.npz", stem]:
                    if key in self._index:
                        return self._index[key]

        return None




def resolve_npz_from_csv_or_index(
    row: pd.Series,
    resolver: BoltzSFileResolver,
    repo_root: Path,
) -> Optional[Path]:
    """Resolve the Boltz embedding npz path for one CSV row.

    Preferred route: use the multiview CSV column `boltz_embedding_npz`, which is
    repo-relative in the current pipeline, e.g.
    outputs/train/chunk_000/boltz_results_pair_000/predictions/pair_000/embeddings_pair_000.npz.

    Fallback route: use the recursive resolver under the split root.
    """
    if "boltz_embedding_npz" in row.index and pd.notna(row["boltz_embedding_npz"]):
        raw_path = str(row["boltz_embedding_npz"]).strip()
        if raw_path:
            p = Path(raw_path)
            if not p.is_absolute():
                p = repo_root / p
            if p.exists():
                return p
    return resolver.resolve(str(row["pair_id"]))


def load_boltz_s(npz_path: Path) -> np.ndarray:
    """Load Boltz single/token representation `s`.

    This function is strict once called: it raises if `s` is absent or malformed.
    Missing-`s` files are filtered during dataset indexing by `has_boltz_s`, so
    the DataLoader should only call this for usable examples.
    """
    with np.load(npz_path) as data:
        if "s" not in data.files:
            raise KeyError(f"{npz_path} does not contain key 's'. Available keys: {data.files}")
        s = data["s"]
    if s.ndim == 3:
        if s.shape[0] != 1:
            raise ValueError(f"Expected s shape (1, L, D) or (L, D), got {s.shape} at {npz_path}")
        s = s[0]
    if s.ndim != 2:
        raise ValueError(f"Expected s shape (L, D), got {s.shape} at {npz_path}")
    return s.astype(np.float32, copy=False)


def has_boltz_s(npz_path: Path) -> bool:
    """Return True if the npz file contains key `s`.

    This opens only the npz directory/metadata; it does not materialise the full
    array into memory. Use this during indexing to skip old files where `s` was
    deleted while keeping `z`/confidence arrays.
    """
    try:
        with np.load(npz_path) as data:
            return "s" in data.files
    except Exception:
        return False


def boltz_s_shape(npz_path: Path) -> Tuple[int, int]:
    """Return (L, D) for `s` without keeping the full array loaded."""
    with np.load(npz_path) as data:
        if "s" not in data.files:
            raise KeyError(f"{npz_path} does not contain key 's'. Available keys: {data.files}")
        shp = data["s"].shape
    if len(shp) == 3:
        if shp[0] != 1:
            raise ValueError(f"Expected s shape (1, L, D) or (L, D), got {shp} at {npz_path}")
        return int(shp[1]), int(shp[2])
    if len(shp) == 2:
        return int(shp[0]), int(shp[1])
    raise ValueError(f"Expected s shape (1, L, D) or (L, D), got {shp} at {npz_path}")


# ============================================================
# Dataset
# ============================================================

class BoltzSRowDataset(Dataset):
    """Row-level dataset that slices Boltz s into TCR, peptide and HLA blocks."""

    def __init__(
        self,
        boltz_root: Path,
        meta: pd.DataFrame,
        source_name: str,
        recursive_index: bool = True,
        strict_length_match: bool = True,
        repo_root: Path = Path("/home/natasha/multimodal_model"),
    ):
        self.boltz_root = Path(boltz_root)
        self.meta = meta.copy().reset_index(drop=True)
        self.source_name = source_name
        self.strict_length_match = strict_length_match
        self.repo_root = Path(repo_root)
        self.resolver = BoltzSFileResolver(self.boltz_root, source_name, recursive_index=recursive_index)

        self.rows: List[Dict] = []
        missing = 0
        missing_s = 0
        unreadable = 0
        length_bad = 0

        for _, r in self.meta.iterrows():
            pid = str(r["pair_id"])
            npz_path = resolve_npz_from_csv_or_index(r, self.resolver, self.repo_root)
            if npz_path is None:
                missing += 1
                continue

            # Some historical Boltz files may still exist but no longer contain
            # the `s` array. Skip them here so training/evaluation does not fail
            # inside the DataLoader.
            if not has_boltz_s(npz_path):
                missing_s += 1
                continue

            la = int(r["tcra_len"])
            lb = int(r["tcrb_len"])
            lp = int(r["pep_len"])
            lh = int(r["hla_len"])
            expected_L = la + lb + lp + lh

            # Fast shape check without keeping the array loaded.
            try:
                L, D = boltz_s_shape(npz_path)
            except Exception as exc:
                unreadable += 1
                print(
                    f"{source_name}: skipping pair_id={pid}; could not read s shape at {npz_path}: {exc}",
                    flush=True,
                )
                continue

            if strict_length_match and L != expected_L:
                length_bad += 1
                continue
            if L < expected_L:
                length_bad += 1
                continue

            self.rows.append({
                "pair_id": pid,
                "npz_path": str(npz_path),
                "binding_flag": int(r["binding_flag"]),
                "peptide": str(r["peptide_for_eval"]),
                "tcra_len": la,
                "tcrb_len": lb,
                "pep_len": lp,
                "hla_len": lh,
                "L": L,
                "D": D,
            })

        if not self.rows:
            raise RuntimeError(
                f"{source_name}: no usable Boltz-s rows found. "
                f"missing_files={missing}, missing_s={missing_s}, unreadable_s={unreadable}, "
                f"length_mismatches={length_bad}, root={self.boltz_root}"
            )

        dims = sorted(set(r["D"] for r in self.rows))
        if len(dims) != 1:
            raise ValueError(f"{source_name}: inconsistent s embedding dimensions: {dims}")

        print(
            f"{source_name}: usable_rows={len(self.rows)} | missing_files={missing} | "
            f"missing_s={missing_s} | unreadable_s={unreadable} | "
            f"length_mismatches={length_bad} | D={dims[0]}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict:
        row = self.rows[idx]
        s = load_boltz_s(Path(row["npz_path"]))
        la = row["tcra_len"]
        lb = row["tcrb_len"]
        lp = row["pep_len"]
        lh = row["hla_len"]

        a0 = 0
        b0 = a0 + la
        p0 = b0 + lb
        h0 = p0 + lp
        h1 = h0 + lh

        # TCR view = alpha + beta; pMHC view = peptide + HLA encoded separately.
        s_T = s[a0:p0, :]
        s_P = s[p0:h0, :]
        s_H = s[h0:h1, :]

        return {
            "emb_T": torch.from_numpy(s_T).float(),
            "emb_P": torch.from_numpy(s_P).float(),
            "emb_H": torch.from_numpy(s_H).float(),
            "binding_flag": int(row["binding_flag"]),
            "pair_id": row["pair_id"],
            "peptide": row["peptide"],
        }


def boltz_s_collate(rows: List[Dict]) -> Dict:
    def pad_stack(key: str) -> Tuple[torch.Tensor, torch.Tensor]:
        xs = [r[key] for r in rows]
        max_len = max(x.shape[0] for x in xs)
        D = xs[0].shape[1]
        out = torch.zeros(len(xs), max_len, D, dtype=torch.float32)
        mask = torch.zeros(len(xs), max_len, dtype=torch.bool)
        for i, x in enumerate(xs):
            L = x.shape[0]
            out[i, :L, :] = x
            mask[i, :L] = True
        return out, mask

    emb_T, mask_T = pad_stack("emb_T")
    emb_P, mask_P = pad_stack("emb_P")
    emb_H, mask_H = pad_stack("emb_H")

    return {
        "emb_T": emb_T,
        "mask_T": mask_T,
        "emb_P": emb_P,
        "mask_P": mask_P,
        "emb_H": emb_H,
        "mask_H": mask_H,
        "binding_flag": torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long),
        "pair_id": [r["pair_id"] for r in rows],
        "peptide": [r["peptide"] for r in rows],
    }


# ============================================================
# Projection heads
# ============================================================

class TokenProjectionHead(nn.Module):
    """Low-rank token projection head for Boltz s tokens.

    emb:  [B, L, D]
    mask: [B, L]
    out:  [B, d]
    """

    def __init__(
        self,
        D: int,
        rL: int,
        rD: int,
        d: int,
        L_max: int,
        dropout: float = 0.1,
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        self.D = D
        self.rL = rL
        self.rD = rD
        self.d = d
        self.L_max = L_max
        self.use_input_layernorm = bool(use_input_layernorm)

        # This normalises each token's 384-dimensional Boltz-s feature vector before
        # projection. It is an input-conditioning step only: the final projected
        # VICReg latents remain unnormalised.
        self.input_norm = nn.LayerNorm(D) if self.use_input_layernorm else nn.Identity()

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

        emb = self.input_norm(emb)

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
    """Peptide and HLA are encoded separately, then concatenated."""

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
        use_input_layernorm: bool = True,
    ):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")

        self.pep_encoder = TokenProjectionHead(D, rL, rD, d_P, L_P_max, dropout, use_input_layernorm)
        self.hla_encoder = TokenProjectionHead(D, rL, rD, d_H, L_H_max, dropout, use_input_layernorm)

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
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)
    score = -mse_distance
    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    return score, mse_distance, cos


def raw_boltz_s_score(batch, device: torch.device, eps_norm: float = 1e-8, R_PH: float = 0.7):
    """Frozen/raw Boltz-s diagnostic baseline using masked mean pooling."""
    T = masked_mean_pool(batch["emb_T"].to(device), batch["mask_T"].to(device), eps_norm)
    P = masked_mean_pool(batch["emb_P"].to(device), batch["mask_P"].to(device), eps_norm)
    HLA = masked_mean_pool(batch["emb_H"].to(device), batch["mask_H"].to(device), eps_norm)
    PH = R_PH * P + (1.0 - R_PH) * HLA
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
    """McClish-standardised partial AUROC; random is ~0.5, perfect is 1.0."""
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
        auc_val = float(roc_auc_score(y, s)) if valid else float("nan")
        pauc_raw = safe_partial_auc_raw(y, s, max_fpr=max_fpr) if valid else float("nan")
        pauc_mcclish = safe_partial_auc_mcclish(y, s, max_fpr=max_fpr) if valid else float("nan")
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": auc_val,
            f"auc{max_fpr:g}_raw": pauc_raw,
            f"auc{max_fpr:g}_raw_div_maxfpr": float(pauc_raw / max_fpr) if valid else float("nan"),
            f"auc{max_fpr:g}_norm": pauc_mcclish,
            f"auc{max_fpr:g}_mcclish": pauc_mcclish,
            "valid": bool(valid),
        })

    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]].copy()

    if len(valid_table) == 0:
        summary = {
            "macro": float("nan"),
            "weighted": float("nan"),
            "macro_auc0.1_raw": float("nan"),
            "weighted_auc0.1_raw": float("nan"),
            "macro_auc0.1_raw_div_maxfpr": float("nan"),
            "weighted_auc0.1_raw_div_maxfpr": float("nan"),
            "macro_auc0.1_norm": float("nan"),
            "weighted_auc0.1_norm": float("nan"),
            "macro_auc0.1_mcclish": float("nan"),
            "weighted_auc0.1_mcclish": float("nan"),
            "n_total": len(table),
            "n_valid": 0,
        }
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
# Evaluation and plots
# ============================================================

@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    loss_params: Dict,
    split: str,
    R_PH: float,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    model_scores, model_mse, model_cos = [], [], []
    raw_scores, raw_mse, raw_cos = [], [], []
    labels_all, pair_ids_all, peptides_all = [], [], []

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

        s_m, d_m, cos_m = score_from_projected(zT, zPH, loss_params["eps_norm"])
        s_r, d_r, cos_r, _ = raw_boltz_s_score(batch, device, loss_params["eps_norm"], R_PH)

        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        pair_ids = [str(x) for x in batch["pair_id"]]
        peptides = np.array([str(x) for x in batch["peptide"]], dtype=str)

        model_scores.append(s_m.detach().cpu().numpy())
        model_mse.append(d_m.detach().cpu().numpy())
        model_cos.append(cos_m.detach().cpu().numpy())

        raw_scores.append(s_r.detach().cpu().numpy())
        raw_mse.append(d_r.detach().cpu().numpy())
        raw_cos.append(cos_r.detach().cpu().numpy())

        labels_all.append(labels)
        pair_ids_all.extend(pair_ids)
        peptides_all.append(peptides)

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

    model_pep_table, model_pep = per_peptide_auroc(labels, model_scores, peptides, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1))
    raw_pep_table, raw_pep = per_peptide_auroc(labels, raw_scores, peptides, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1))
    best_thr = best_f1_threshold(model_scores, labels)

    metrics = {
        f"{split}_loss": running["loss"] / max(1, n_steps),
        f"{split}_model_global_auroc": safe_auroc(labels, model_scores),
        f"{split}_raw_boltz_s_global_auroc": safe_auroc(labels, raw_scores),
        f"{split}_delta_global_auroc": safe_auroc(labels, model_scores) - safe_auroc(labels, raw_scores),
        f"{split}_model_auprc": safe_auprc(labels, model_scores),
        f"{split}_raw_boltz_s_auprc": safe_auprc(labels, raw_scores),
        f"{split}_model_auc0.1_raw_div_maxfpr": safe_partial_auc_norm(labels, model_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_boltz_s_auc0.1_raw_div_maxfpr": safe_partial_auc_norm(labels, raw_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_auc0.1_norm": safe_partial_auc_mcclish(labels, model_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_boltz_s_auc0.1_norm": safe_partial_auc_mcclish(labels, raw_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_auc0.1_mcclish": safe_partial_auc_mcclish(labels, model_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_raw_boltz_s_auc0.1_mcclish": safe_partial_auc_mcclish(labels, raw_scores, max_fpr=loss_params.get("partial_auc_max_fpr", 0.1)),
        f"{split}_model_peptide_macro_auroc": model_pep["macro"],
        f"{split}_raw_boltz_s_peptide_macro_auroc": raw_pep["macro"],
        f"{split}_delta_peptide_macro_auroc": model_pep["macro"] - raw_pep["macro"],
        f"{split}_model_peptide_weighted_auroc": model_pep["weighted"],
        f"{split}_raw_boltz_s_peptide_weighted_auroc": raw_pep["weighted"],
        f"{split}_delta_peptide_weighted_auroc": model_pep["weighted"] - raw_pep["weighted"],
        f"{split}_model_peptide_macro_auc0.1_raw_div_maxfpr": model_pep["macro_auc0.1_raw_div_maxfpr"],
        f"{split}_model_peptide_weighted_auc0.1_raw_div_maxfpr": model_pep["weighted_auc0.1_raw_div_maxfpr"],
        f"{split}_raw_boltz_s_peptide_macro_auc0.1_raw_div_maxfpr": raw_pep["macro_auc0.1_raw_div_maxfpr"],
        f"{split}_raw_boltz_s_peptide_weighted_auc0.1_raw_div_maxfpr": raw_pep["weighted_auc0.1_raw_div_maxfpr"],
        f"{split}_model_peptide_macro_auc0.1_norm": model_pep["macro_auc0.1_mcclish"],
        f"{split}_model_peptide_weighted_auc0.1_norm": model_pep["weighted_auc0.1_mcclish"],
        f"{split}_raw_boltz_s_peptide_macro_auc0.1_norm": raw_pep["macro_auc0.1_mcclish"],
        f"{split}_raw_boltz_s_peptide_weighted_auc0.1_norm": raw_pep["weighted_auc0.1_mcclish"],
        f"{split}_model_peptide_macro_auc0.1_mcclish": model_pep["macro_auc0.1_mcclish"],
        f"{split}_model_peptide_weighted_auc0.1_mcclish": model_pep["weighted_auc0.1_mcclish"],
        f"{split}_raw_boltz_s_peptide_macro_auc0.1_mcclish": raw_pep["macro_auc0.1_mcclish"],
        f"{split}_raw_boltz_s_peptide_weighted_auc0.1_mcclish": raw_pep["weighted_auc0.1_mcclish"],
        f"{split}_n_peptides_total": model_pep["n_total"],
        f"{split}_n_peptides_valid": model_pep["n_valid"],
        f"{split}_threshold": best_thr["threshold"],
        f"{split}_f1": best_thr["f1"],
        f"{split}_accuracy": best_thr["accuracy"],
        f"{split}_precision": best_thr["precision"],
        f"{split}_recall": best_thr["recall"],
        f"{split}_model_mse_std": float(np.std(model_mse)),
        f"{split}_raw_boltz_s_mse_std": float(np.std(raw_mse)),
    }

    for k, v in running.items():
        if k != "loss":
            metrics[f"{split}_{k}"] = v / max(1, n_steps)

    predictions = pd.DataFrame({
        "pair_id": pair_ids_all,
        "peptide": peptides,
        "label": labels,
        "model_score": model_scores,
        "model_mse_distance": model_mse,
        "model_cos": model_cos,
        "raw_boltz_s_score": raw_scores,
        "raw_boltz_s_mse_distance": raw_mse,
        "raw_boltz_s_cos": raw_cos,
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


def plot_histogram(mse_distance: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mse_distance = np.asarray(mse_distance)
    labels = np.asarray(labels).astype(int)

    plt.figure(figsize=(8, 5))
    if np.any(labels == 0):
        plt.hist(mse_distance[labels == 0], bins=50, density=True, alpha=0.55, label="decoy/negative")
    if np.any(labels == 1):
        plt.hist(mse_distance[labels == 1], bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("MSE distance; lower = stronger binding")
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
    # Boltz roots. These can be either split-specific roots containing .npz files,
    # or parent roots where recursive search can find each pair_id file.
    train_boltz_root: str = "/home/natasha/multimodal_model/outputs/train"
    val_boltz_root: str = "/home/natasha/multimodal_model/outputs/val"
    test_boltz_root: str = "/home/natasha/multimodal_model/outputs/test"
    immrep_boltz_root: str = "/home/natasha/multimodal_model/outputs_data/immrep_test"
    repo_root: str = "/home/natasha/multimodal_model"

    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/boltz_s_vicreg"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/boltz_s_vicreg"
    run_tag: str = "boltz_s_vicreg_mse_mcclish_strict"

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
    input_layernorm: bool = True

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    alpha: float = 25.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    eps_norm: float = 1e-8
    eps_var: float = 1e-4

    recursive_index: bool = True
    strict_length_match: bool = True
    partial_auc_max_fpr: float = 0.1
    selection_metric: str = "val_model_peptide_weighted_auroc"


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


def infer_shapes(train_ds: Dataset) -> Tuple[int, int, int, int]:
    max_T = max(r["tcra_len"] + r["tcrb_len"] for r in train_ds.rows)
    max_P = max(r["pep_len"] for r in train_ds.rows)
    max_H = max(r["hla_len"] for r in train_ds.rows)
    D = int(train_ds.rows[0]["D"])
    print(f"Detected Boltz-s shapes | D={D} | L_T_max={max_T} | L_P_max={max_P} | L_H_max={max_H}", flush=True)
    return D, max_T, max_P, max_H


def initialise_models(cfg: RunConfig, shapes: Tuple[int, int, int, int], device: torch.device) -> Tuple[nn.Module, nn.Module]:
    D, L_T, L_P, L_H = shapes
    tcr = TokenProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout, cfg.input_layernorm).to(device)
    pmhc = PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout, cfg.input_layernorm).to(device)
    return tcr, pmhc


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    defaults = asdict(RunConfig())
    for k, v in defaults.items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            parser.add_argument(arg, action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, int):
            parser.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            parser.add_argument(arg, type=float, default=v)
        else:
            parser.add_argument(arg, default=v)
    return RunConfig(**vars(parser.parse_args()))


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{cfg.run_tag}__run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80, flush=True)
    print("Boltz-s VICReg diagnostic run", flush=True)
    print(f"Device: {device}", flush=True)
    print(json.dumps(asdict(cfg), indent=2), flush=True)
    print("=" * 80, flush=True)

    train_meta = allowed_meta_from_csv(cfg.train_csv, "train", positives_only=True, complete_only=True)
    val_meta = allowed_meta_from_csv(cfg.val_csv, "val", positives_only=False, complete_only=True)
    test_meta = allowed_meta_from_csv(cfg.test_csv, "test", positives_only=False, complete_only=True)
    immrep_meta = allowed_meta_from_csv(cfg.immrep_csv, "immrep_test", positives_only=False, complete_only=True) if cfg.immrep_csv else None

    repo_root = Path(cfg.repo_root)
    train_ds = BoltzSRowDataset(Path(cfg.train_boltz_root), train_meta, "train", cfg.recursive_index, cfg.strict_length_match, repo_root)
    val_ds = BoltzSRowDataset(Path(cfg.val_boltz_root), val_meta, "val", cfg.recursive_index, cfg.strict_length_match, repo_root)
    test_ds = BoltzSRowDataset(Path(cfg.test_boltz_root), test_meta, "test", cfg.recursive_index, cfg.strict_length_match, repo_root)
    immrep_ds = None
    if immrep_meta is not None and cfg.immrep_boltz_root:
        immrep_ds = BoltzSRowDataset(Path(cfg.immrep_boltz_root), immrep_meta, "immrep_test", cfg.recursive_index, cfg.strict_length_match, repo_root)

    print(
        f"Loaded usable examples | train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
        + (f" | immrep={len(immrep_ds)}" if immrep_ds is not None else ""),
        flush=True,
    )

    shapes = infer_shapes(train_ds)

    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=boltz_s_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=boltz_s_collate, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=boltz_s_collate, pin_memory=torch.cuda.is_available())
    immrep_loader = None if immrep_ds is None else DataLoader(immrep_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=boltz_s_collate, pin_memory=torch.cuda.is_available())

    tcr, pmhc = initialise_models(cfg, shapes, device)
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
        "selection_value": -np.inf,
        "epoch": None,
        "state": None,
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
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(list(tcr.parameters()) + list(pmhc.parameters()), cfg.grad_clip)
            optimizer.step()

            train_running["loss"] += parts["L_total"]
            for k in train_running:
                if k != "loss":
                    train_running[k] += parts[k]
            n += 1

        scheduler.step()

        val_eval = evaluate(val_loader, tcr, pmhc, device, lp, split="val", R_PH=cfg.R_PH)
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
        pd.DataFrame(history).to_csv(out_dir / f"{cfg.run_tag}__history.csv", index=False)

        current = row.get(cfg.selection_metric, float("nan"))
        if not isinstance(current, (int, float)) or math.isnan(float(current)):
            current = row.get("val_model_peptide_weighted_auroc", float("nan"))
        improved = (not math.isnan(float(current))) and float(current) > best["selection_value"] + 1e-4

        if improved:
            best = {
                "selection_value": float(current),
                "epoch": epoch,
                "state": {
                    "tcr": copy.deepcopy(tcr.state_dict()),
                    "pmhc": copy.deepcopy(pmhc.state_dict()),
                },
                "bad_epochs": 0,
            }
            torch.save(
                {
                    "config": asdict(cfg),
                    "loss_params": lp,
                    "tcr_state_dict": best["state"]["tcr"],
                    "pmhc_state_dict": best["state"]["pmhc"],
                    "best_epoch": epoch,
                    "selection_value": float(current),
                    "val_metrics": val_eval["metrics"],
                },
                out_dir / f"{cfg.run_tag}__best.pt",
            )
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
            f"raw_global={row['val_raw_boltz_s_global_auroc']:.4f} | "
            f"delta_global={row['val_delta_global_auroc']:.4f} | "
            f"val_model_pep_weighted={row['val_model_peptide_weighted_auroc']:.4f} | "
            f"raw_pep_weighted={row['val_raw_boltz_s_peptide_weighted_auroc']:.4f} | "
            f"delta_pep_weighted={row['val_delta_peptide_weighted_auroc']:.4f} | "
            f"val_auprc={row['val_model_auprc']:.4f} | "
            f"raw_auprc={row['val_raw_boltz_s_auprc']:.4f} | "
            f"zTstd={row['val_zTstd']:.4f} | "
            f"zPHstd={row['val_zPHstd']:.4f} | "
            f"best_epoch={best['epoch']} | "
            f"bad_epochs={best['bad_epochs']}",
            flush=True,
        )

        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}; min_epochs={cfg.min_epochs}, patience={cfg.patience}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint was selected.")

    tcr.load_state_dict(best["state"]["tcr"])
    pmhc.load_state_dict(best["state"]["pmhc"])

    val_eval = evaluate(val_loader, tcr, pmhc, device, lp, "val", cfg.R_PH)
    test_eval = evaluate(test_loader, tcr, pmhc, device, lp, "test", cfg.R_PH)
    immrep_eval = None if immrep_loader is None else evaluate(immrep_loader, tcr, pmhc, device, lp, "immrep_test", cfg.R_PH)

    val_threshold = val_eval["metrics"]["val_threshold"]
    test_at_val_threshold = threshold_metrics(test_eval["predictions"]["model_score"].to_numpy(), test_eval["predictions"]["label"].to_numpy(), val_threshold, "test_at_val_threshold")

    stem = (
        f"{cfg.run_tag}"
        f"__seed{cfg.seed}"
        f"__lr{cfg.lr}"
        f"__a{cfg.alpha}"
        f"__b{cfg.beta}"
        f"__dlt{cfg.delta}"
    )

    checkpoint_path = out_dir / f"{stem}__best.pt"
    summary_path = out_dir / f"{stem}__summary.json"

    paths = {
        "history": out_dir / f"{cfg.run_tag}__history.csv",
        "checkpoint": checkpoint_path,
        "val_predictions": out_dir / f"{stem}__val_predictions.csv",
        "test_predictions": out_dir / f"{stem}__test_predictions.csv",
        "val_model_per_peptide": out_dir / f"{stem}__val_model_per_peptide.csv",
        "val_raw_boltz_s_per_peptide": out_dir / f"{stem}__val_raw_boltz_s_per_peptide.csv",
        "test_model_per_peptide": out_dir / f"{stem}__test_model_per_peptide.csv",
        "test_raw_boltz_s_per_peptide": out_dir / f"{stem}__test_raw_boltz_s_per_peptide.csv",
        "immrep_test_predictions": out_dir / f"{stem}__immrep_test_predictions.csv",
        "immrep_test_model_per_peptide": out_dir / f"{stem}__immrep_test_model_per_peptide.csv",
        "immrep_test_raw_boltz_s_per_peptide": out_dir / f"{stem}__immrep_test_raw_boltz_s_per_peptide.csv",
    }

    val_eval["predictions"].to_csv(paths["val_predictions"], index=False)
    test_eval["predictions"].to_csv(paths["test_predictions"], index=False)
    val_eval["model_peptide_table"].to_csv(paths["val_model_per_peptide"], index=False)
    val_eval["raw_peptide_table"].to_csv(paths["val_raw_boltz_s_per_peptide"], index=False)
    test_eval["model_peptide_table"].to_csv(paths["test_model_per_peptide"], index=False)
    test_eval["raw_peptide_table"].to_csv(paths["test_raw_boltz_s_per_peptide"], index=False)

    if immrep_eval is not None:
        immrep_eval["predictions"].to_csv(paths["immrep_test_predictions"], index=False)
        immrep_eval["model_peptide_table"].to_csv(paths["immrep_test_model_per_peptide"], index=False)
        immrep_eval["raw_peptide_table"].to_csv(paths["immrep_test_raw_boltz_s_per_peptide"], index=False)

    plot_histogram(val_eval["model_mse"], val_eval["labels"], "Validation model Boltz-s VICReg", fig_dir / f"{stem}__val_model_mse_hist.png")
    plot_histogram(val_eval["raw_mse"], val_eval["labels"], "Validation raw Boltz-s MSE-distance score", fig_dir / f"{stem}__val_raw_boltz_s_mse_hist.png")
    plot_histogram(test_eval["model_mse"], test_eval["labels"], "Test model Boltz-s VICReg", fig_dir / f"{stem}__test_model_mse_hist.png")
    plot_histogram(test_eval["raw_mse"], test_eval["labels"], "Test raw Boltz-s MSE-distance score", fig_dir / f"{stem}__test_raw_boltz_s_mse_hist.png")
    if immrep_eval is not None:
        plot_histogram(immrep_eval["model_mse"], immrep_eval["labels"], "IMMREP model Boltz-s VICReg", fig_dir / f"{stem}__immrep_model_mse_hist.png")
        plot_histogram(immrep_eval["raw_mse"], immrep_eval["labels"], "IMMREP raw Boltz-s MSE-distance score", fig_dir / f"{stem}__immrep_raw_boltz_s_mse_hist.png")

    summary = {
        "config": asdict(cfg),
        "best_epoch_by_selection_metric": best["epoch"],
        "selection_metric": cfg.selection_metric,
        "selection_value": best["selection_value"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "immrep_test_metrics": None if immrep_eval is None else immrep_eval["metrics"],
        "test_at_val_threshold": test_at_val_threshold,
        "paths": {k: str(v) for k, v in paths.items()} | {"fig_dir": str(fig_dir)},
    }

    torch.save(
        {
            "config": asdict(cfg),
            "loss_params": lp,
            "tcr_state_dict": tcr.state_dict(),
            "pmhc_state_dict": pmhc.state_dict(),
            "best_epoch": best["epoch"],
            "selection_metric": cfg.selection_metric,
            "selection_value": best["selection_value"],
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
            "immrep_test_metrics": None if immrep_eval is None else immrep_eval["metrics"],
        },
        checkpoint_path,
    )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80, flush=True)
    print("Done.", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"History: {out_dir / f'{cfg.run_tag}__history.csv'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)
    print(f"Figures: {fig_dir}", flush=True)
    print("Final validation metrics:", flush=True)
    print(json.dumps(val_eval["metrics"], indent=2), flush=True)
    print("Final test metrics:", flush=True)
    print(json.dumps(test_eval["metrics"], indent=2), flush=True)
    if immrep_eval is not None:
        print("Final IMMREP test metrics:", flush=True)
        print(json.dumps(immrep_eval["metrics"], indent=2), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
