#!/usr/bin/env python3
"""Train a one-hot, Boltz-conditioned residue message-passing VICReg model.

One configurable script covers the principal ablations:
  --z-ablation none      one-hot VICReg, no message passing
  --z-ablation zero      message-passing architecture with zeroed Boltz z
  --z-ablation shuffled  message passing with another batch item's Boltz z
  --z-ablation boltz     correctly aligned Boltz z

Graph modes:
  --graph-mode tp        TCR <-> peptide only
  --graph-mode cross     all cross-component edges
  --graph-mode all       all valid residue-pair edges

The model is trained on positive pairs only. Checkpoint selection uses validation
peptide-weighted McClish partial AUC at max_fpr=0.1. Test and IMMREP are evaluated
only after selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import zipfile
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


# -----------------------------------------------------------------------------
# Reproducibility and sequence utilities
# -----------------------------------------------------------------------------

AA20 = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
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


def normalise_manifest(df: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if "pair_id" not in df.columns:
        raise ValueError(f"{split}: missing pair_id")
    if "boltz_embedding_npz" not in df.columns:
        raise ValueError(f"{split}: missing boltz_embedding_npz")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    out["binding_flag"] = 1 if label_col is None else pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    tcr_col = first_existing_col(out, ["TCR_full", "tcr_full", "full_tcr", "TCR", "tcr"])
    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    hla_col = first_existing_col(out, ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"])
    if None in (tcr_col, pep_col, hla_col):
        raise ValueError(f"{split}: required sequence columns not found; columns={list(out.columns)}")

    out["TCR_full_norm"] = out[tcr_col].map(clean_seq)
    out["Peptide_norm"] = out[pep_col].map(clean_seq)
    out["HLA_sequence_norm"] = out[hla_col].map(clean_seq)
    out["peptide_for_eval"] = out["Peptide_norm"]

    source = {
        "label_col": "constant_1" if label_col is None else label_col,
        "tcr_col": str(tcr_col), "pep_col": str(pep_col), "hla_col": str(hla_col),
    }
    out["tcra_len"], source["tcra_len"] = extract_length(
        out, ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
    )
    out["tcrb_len"], source["tcrb_len"] = extract_length(
        out, ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
    )
    out["pep_len"], source["pep_len"] = extract_length(
        out, ["pep_len", "peptide_len"], ["Peptide", "peptide", "pep_seq", "peptide_seq"],
    )
    out["hla_len"], source["hla_len"] = extract_length(
        out, ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"],
    )
    out["tcr_total_len"] = out["TCR_full_norm"].str.len().astype(int)
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source


def resolve_npz_path(value: str, repo_root: Path) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else repo_root / p


def _normalise_z_shape(shape: Tuple[int, ...]) -> Tuple[int, int]:
    if len(shape) == 4 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 3 or shape[0] != shape[1]:
        raise ValueError(f"expected z [L,L,D] or [1,L,L,D], got {shape}")
    return int(shape[0]), int(shape[2])


def inspect_z_shape(path: Path) -> Tuple[int, int]:
    try:
        return inspect_z_shape_fast(path)
    except Exception:
        with np.load(path) as data:
            if "z" not in data.files:
                raise KeyError(f"{path}: no z key; keys={list(data.files)}")
            return _normalise_z_shape(tuple(data["z"].shape))


def inspect_z_shape_fast(path: Path) -> Tuple[int, int]:
    """Read z shape from the .npz header without decompressing the full array."""
    from numpy.lib.format import read_array_header_1_0, read_magic

    with zipfile.ZipFile(path, "r") as archive:
        member = next((name for name in archive.namelist() if name.endswith("z.npy")), None)
        if member is None:
            raise KeyError(f"{path}: no z.npy member; keys={archive.namelist()}")
        with archive.open(member, "r") as handle:
            read_magic(handle)
            shape, _, _ = read_array_header_1_0(handle)
    return _normalise_z_shape(tuple(shape))


def meta_cache_paths(cache_dir: Path, split: str) -> Tuple[Path, Path]:
    return cache_dir / f"{split}_filtered.csv", cache_dir / f"{split}_filtered.stamp.json"


def restore_meta_dtypes(meta: pd.DataFrame) -> pd.DataFrame:
    out = meta.copy()
    for col in ["has_alpha", "has_beta", "z_exists", "z_aligned", "tcr_length_consistent"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in ["binding_flag", "tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len", "z_raw_len", "z_dim"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    return out


def load_meta_cache(cache_dir: Path, split: str, csv_path: str, positives_only: bool) -> Optional[Tuple[pd.DataFrame, Dict[str, str], int]]:
    cache_csv, stamp_path = meta_cache_paths(cache_dir, split)
    if not cache_csv.exists() or not stamp_path.exists():
        return None
    with open(stamp_path) as f:
        stamp = json.load(f)
    csv = Path(csv_path)
    if stamp.get("csv_path") != str(csv):
        return None
    if stamp.get("positives_only") != bool(positives_only):
        return None
    if csv.exists() and stamp.get("csv_mtime") != csv.stat().st_mtime:
        return None
    meta = restore_meta_dtypes(pd.read_csv(cache_csv))
    source = stamp.get("source", {})
    csv_rows = int(stamp.get("csv_rows", len(meta)))
    print(f"{split}: loaded cached manifest rows={len(meta)} from {cache_csv}", flush=True)
    return meta, source, csv_rows


def save_meta_cache(
    cache_dir: Path,
    split: str,
    csv_path: str,
    positives_only: bool,
    meta: pd.DataFrame,
    source: Dict[str, str],
    csv_rows: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_csv, stamp_path = meta_cache_paths(cache_dir, split)
    meta.to_csv(cache_csv, index=False)
    stamp = {
        "split": split,
        "csv_path": str(csv_path),
        "csv_mtime": Path(csv_path).stat().st_mtime if Path(csv_path).exists() else None,
        "positives_only": bool(positives_only),
        "n_rows": int(len(meta)),
        "csv_rows": int(csv_rows),
        "source": source,
    }
    with open(stamp_path, "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"{split}: wrote manifest cache rows={len(meta)} -> {cache_csv}", flush=True)


def limit_rows(meta: pd.DataFrame, max_rows: int, seed: int, split: str) -> pd.DataFrame:
    if max_rows <= 0 or len(meta) <= max_rows:
        return meta
    out = meta.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    print(f"{split}: subsampled rows {len(meta)} -> {len(out)} (max_rows={max_rows})", flush=True)
    return out


def audit_from_meta(meta: pd.DataFrame, split: str, csv_path: str, positives_only: bool, source: Dict[str, str], csv_rows: int) -> Tuple[pd.DataFrame, Dict]:
    audit = {
        "split": split,
        "csv_path": csv_path,
        "csv_rows": int(csv_rows),
        "positives_only": bool(positives_only),
        **source,
        "n_bad_tcr_length": int((~meta["tcr_length_consistent"]).sum()) if "tcr_length_consistent" in meta.columns else 0,
        "n_missing_z": int((~meta["z_exists"]).sum()),
        "n_bad_z_alignment": int((meta["z_exists"] & ~meta["z_aligned"]).sum()),
        "n_final": int(len(meta)),
        "n_positive_final": int((meta["binding_flag"] == 1).sum()),
        "n_negative_final": int((meta["binding_flag"] == 0).sum()),
    }
    dims = sorted(meta["z_dim"].unique().tolist())
    if len(dims) != 1:
        raise ValueError(f"{split}: inconsistent z dimensions: {dims}")
    print(
        f"{split}: rows={len(meta)} pos={audit['n_positive_final']} neg={audit['n_negative_final']} "
        f"z_dim={dims[0]} dropped_missing={audit['n_missing_z']} dropped_misaligned={audit['n_bad_z_alignment']}",
        flush=True,
    )
    return meta, audit


def load_meta(
    csv_path: str,
    split: str,
    positives_only: bool,
    repo_root: Path,
    *,
    skip_z_audit: bool = False,
    meta_cache_dir: Optional[Path] = None,
    write_meta_cache: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    if meta_cache_dir is not None:
        if skip_z_audit:
            cached = load_meta_cache(meta_cache_dir, split, csv_path, positives_only)
            if cached is None:
                raise FileNotFoundError(
                    f"{split}: --skip-z-audit set but no valid cache in {meta_cache_dir}. "
                    "Run once with --write-meta-cache to build it."
                )
            meta, source, csv_rows = cached
            if len(meta) == 0:
                raise RuntimeError(f"{split}: cached manifest is empty")
            return audit_from_meta(meta.reset_index(drop=True), split, csv_path, positives_only, source, csv_rows)

        if not write_meta_cache:
            cached = load_meta_cache(meta_cache_dir, split, csv_path, positives_only)
            if cached is not None:
                meta, source, csv_rows = cached
                return audit_from_meta(meta.reset_index(drop=True), split, csv_path, positives_only, source, csv_rows)

    raw = pd.read_csv(csv_path)
    meta, source = normalise_manifest(raw, split)

    if source["tcra_len"] == "__missing__" or source["tcrb_len"] == "__missing__":
        raise ValueError(f"{split}: alpha/beta lengths are required for Boltz alignment; source={source}")

    if positives_only:
        meta = meta[meta["binding_flag"] == 1].copy()

    meta = meta[
        (meta["tcr_total_len"] > 0) & (meta["pep_len"] > 0) & (meta["hla_len"] > 0)
        & meta["has_alpha"] & meta["has_beta"]
    ].copy()

    # TCR_full must be exactly alpha + beta length for alignment with the raw z order.
    meta["tcr_length_consistent"] = meta["tcr_total_len"] == (meta["tcra_len"] + meta["tcrb_len"])
    n_bad_tcr = int((~meta["tcr_length_consistent"]).sum())
    meta = meta[meta["tcr_length_consistent"]].copy()

    paths, exists, z_len, z_dim, z_ok, errors = [], [], [], [], [], []
    n_rows = len(meta)
    for i, (_, row) in enumerate(meta.iterrows(), start=1):
        if i == 1 or i % 1000 == 0 or i == n_rows:
            print(f"{split}: auditing z files {i}/{n_rows}", flush=True)
        p = resolve_npz_path(row["boltz_embedding_npz"], repo_root)
        paths.append(str(p))
        if not p.exists():
            exists.append(False); z_len.append(-1); z_dim.append(-1); z_ok.append(False); errors.append("missing_file")
            continue
        exists.append(True)
        try:
            length, dim = inspect_z_shape(p)
            expected = int(row["tcr_total_len"] + row["pep_len"] + row["hla_len"])
            z_len.append(length); z_dim.append(dim); z_ok.append(length == expected)
            errors.append("" if length == expected else f"length_{length}_expected_{expected}")
        except Exception as exc:  # audit malformed files without crashing the whole split
            z_len.append(-1); z_dim.append(-1); z_ok.append(False); errors.append(type(exc).__name__)

    meta["z_path"] = paths
    meta["z_exists"] = exists
    meta["z_raw_len"] = z_len
    meta["z_dim"] = z_dim
    meta["z_aligned"] = z_ok
    meta["z_error"] = errors

    meta = meta[meta["z_exists"] & meta["z_aligned"]].copy().reset_index(drop=True)
    if len(meta) == 0:
        raise RuntimeError(f"{split}: no correctly aligned rows remain")

    if write_meta_cache and meta_cache_dir is not None:
        save_meta_cache(meta_cache_dir, split, csv_path, positives_only, meta, source, len(raw))

    audit_source = {**source, "n_bad_tcr_length": n_bad_tcr}
    return audit_from_meta(meta, split, csv_path, positives_only, audit_source, len(raw))


def infer_padded_lengths(metas: Sequence[pd.DataFrame], cap_tcr: int, cap_pep: int, cap_hla: int) -> Tuple[int, int, int, int]:
    all_meta = pd.concat(metas, ignore_index=True)
    maxima = [int(all_meta[c].max()) for c in ["tcr_total_len", "pep_len", "hla_len"]]
    caps = [cap_tcr, cap_pep, cap_hla]
    lengths = [min(v, cap) if cap > 0 else v for v, cap in zip(maxima, caps)]
    if any(v <= 0 for v in lengths):
        raise ValueError(f"invalid padded lengths: {lengths}")
    z_dims = sorted(all_meta["z_dim"].unique().tolist())
    if len(z_dims) != 1:
        raise ValueError(f"inconsistent z dimensions across splits: {z_dims}")
    return lengths[0], lengths[1], lengths[2], int(z_dims[0])


# -----------------------------------------------------------------------------
# Dataset: exact raw-z to fixed padded block alignment
# -----------------------------------------------------------------------------


def load_raw_z(path: Path) -> torch.Tensor:
    with np.load(path) as data:
        z = np.asarray(data["z"])
    if z.ndim == 4:
        z = z[0]
    return torch.from_numpy(z.astype(np.float32, copy=False))


def copy_block(src: torch.Tensor, dst: torch.Tensor, src_rows: slice, src_cols: slice, dst_rows: slice, dst_cols: slice) -> None:
    dst[dst_rows, dst_cols, :] = src[src_rows, src_cols, :]


def align_z_to_padded_layout(
    raw_z: torch.Tensor,
    t_len: int,
    p_len: int,
    h_len: int,
    L_T: int,
    L_P: int,
    L_H: int,
    cross_component_only: bool = False,
) -> torch.Tensor:
    """Move raw [T,P,H] blocks into fixed padded [T|P|H] regions.

    When cross_component_only=True, same-component blocks (T-T, P-P, H-H) stay zero.
    This is numerically identical to full alignment under graph_mode=cross because
    those edges are masked out before message aggregation.
    """
    if t_len > L_T or p_len > L_P or h_len > L_H:
        raise ValueError(
            f"sequence exceeds configured caps: actual={(t_len,p_len,h_len)} caps={(L_T,L_P,L_H)}"
        )
    expected = t_len + p_len + h_len
    if raw_z.shape[0] != expected or raw_z.shape[1] != expected:
        raise ValueError(f"raw z length {tuple(raw_z.shape[:2])} != expected {expected}")

    D = raw_z.shape[-1]
    L = L_T + L_P + L_H
    out = torch.zeros(L, L, D, dtype=torch.float32)

    raw = {
        "T": slice(0, t_len),
        "P": slice(t_len, t_len + p_len),
        "H": slice(t_len + p_len, expected),
    }
    pad = {
        "T": slice(0, t_len),
        "P": slice(L_T, L_T + p_len),
        "H": slice(L_T + L_P, L_T + L_P + h_len),
    }
    block_pairs = (
        [("T", "P"), ("T", "H"), ("P", "T"), ("P", "H"), ("H", "T"), ("H", "P")]
        if cross_component_only
        else [(r, s) for r in ("T", "P", "H") for s in ("T", "P", "H")]
    )
    for receiver, sender in block_pairs:
        copy_block(raw_z, out, raw[receiver], raw[sender], pad[receiver], pad[sender])
    return out


class OneHotBoltzDataset(Dataset):
    def __init__(
        self,
        meta: pd.DataFrame,
        L_T: int,
        L_P: int,
        L_H: int,
        source_name: str,
        graph_mode: str = "cross",
    ):
        self.meta = meta.reset_index(drop=True)
        self.L_T, self.L_P, self.L_H = int(L_T), int(L_P), int(L_H)
        self.source_name = source_name
        self.cross_component_only = graph_mode == "cross"

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Dict:
        row = self.meta.iloc[idx]
        t_len, p_len, h_len = int(row.tcr_total_len), int(row.pep_len), int(row.hla_len)

        xT, mT = onehot_encode(row.TCR_full_norm, self.L_T)
        xP, mP = onehot_encode(row.Peptide_norm, self.L_P)
        xH, mH = onehot_encode(row.HLA_sequence_norm, self.L_H)
        raw_z = load_raw_z(Path(row.z_path))
        z = align_z_to_padded_layout(
            raw_z, t_len, p_len, h_len, self.L_T, self.L_P, self.L_H, self.cross_component_only
        )

        return {
            "emb_T": xT, "mask_T": mT,
            "emb_P": xP, "mask_P": mP,
            "emb_H": xH, "mask_H": mH,
            "z": z,
            "binding_flag": int(row.binding_flag),
            "pair_id": str(row.pair_id),
            "peptide": str(row.peptide_for_eval),
            "has_alpha": bool(row.has_alpha), "has_beta": bool(row.has_beta),
            "tcra_len": int(row.tcra_len), "tcrb_len": int(row.tcrb_len),
            "pep_len": p_len, "hla_len": h_len,
        }


def collate_rows(rows: List[Dict]) -> Dict:
    tensor_keys = ["emb_T", "mask_T", "emb_P", "mask_P", "emb_H", "mask_H", "z"]
    out = {k: torch.stack([r[k] for r in rows]) for k in tensor_keys}
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "pep_len", "hla_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class LowRankProjectionHead(nn.Module):
    """Existing residue-and-feature low-rank projection head; no extra pooling."""
    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float):
        super().__init__()
        self.D, self.rL, self.rD, self.d, self.L_max = D, rL, rD, d, L_max
        self.B_c = nn.Parameter(torch.empty(D, rD))
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))
        nn.init.xavier_uniform_(self.B_c)
        nn.init.xavier_uniform_(self.A_c)
        nn.init.xavier_uniform_(self.H_c)
        self.expander = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, d))

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, L_pad, D = emb.shape
        if D != self.D or L_pad > self.L_max:
            raise ValueError(f"projection shape mismatch: got {tuple(emb.shape)}, expected D={self.D}, L<={self.L_max}")
        outputs = []
        lengths = mask.sum(dim=1)
        for b in range(B):
            n = int(lengths[b].item())
            if n == 0:
                outputs.append(torch.zeros(self.d, device=emb.device, dtype=emb.dtype))
                continue
            X = emb[b, :n] * mask[b, :n, None].to(emb.dtype)
            Y = X @ self.B_c
            U = self.A_c[:n].T @ Y
            outputs.append(U.reshape(-1) @ self.H_c)
        return self.expander(torch.stack(outputs))


class PMHCProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P: int, L_H: int, R_PH: float, dropout: float):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if min(d_P, d_H) <= 0:
            raise ValueError(f"invalid R_PH={R_PH}")
        self.peptide = LowRankProjectionHead(D, rL, rD, d_P, L_P, dropout)
        self.hla = LowRankProjectionHead(D, rL, rD, d_H, L_H, dropout)

    def forward(self, hP: torch.Tensor, mP: torch.Tensor, hH: torch.Tensor, mH: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.peptide(hP, mP), self.hla(hH, mH)], dim=-1)


def component_ids(B: int, L_T: int, L_P: int, L_H: int, device: torch.device) -> torch.Tensor:
    return torch.cat([
        torch.zeros(B, L_T, dtype=torch.long, device=device),
        torch.ones(B, L_P, dtype=torch.long, device=device),
        torch.full((B, L_H), 2, dtype=torch.long, device=device),
    ], dim=1)


def graph_block_mask(ids: torch.Tensor, graph_mode: str) -> torch.Tensor:
    """Return [B, receiver, sender] biological block mask."""
    receiver = ids[:, :, None]
    sender = ids[:, None, :]
    if graph_mode == "all":
        return torch.ones_like(receiver == sender)
    if graph_mode == "cross":
        return receiver != sender
    if graph_mode == "tp":
        return ((receiver == 0) & (sender == 1)) | ((receiver == 1) & (sender == 0))
    raise ValueError(f"unknown graph_mode={graph_mode}")


def stochastic_symmetric_edge_mask(pair_mask: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not training or p <= 0:
        return pair_mask
    B, L, _ = pair_mask.shape
    random_upper = torch.rand(B, L, L, device=pair_mask.device)
    keep_upper = torch.triu(random_upper >= p, diagonal=1)
    keep = keep_upper | keep_upper.transpose(1, 2)
    return pair_mask & keep


class ChunkedBoltzMessageLayer(nn.Module):
    """Feature-wise edge gating with sender chunking to limit GPU memory."""
    def __init__(self, d_hidden: int, d_z: int, dropout: float, sender_chunk: int):
        super().__init__()
        self.sender_chunk = int(sender_chunk)
        self.source = nn.Linear(d_hidden, d_hidden, bias=False)
        self.edge_gate = nn.Sequential(nn.Linear(d_z, d_hidden), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Linear(2 * d_hidden, d_hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_hidden, d_hidden)
        )
        self.message_norm = nn.LayerNorm(d_hidden)
        self.update_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, h: torch.Tensor, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = h.shape
        source = self.source(h)
        message_sum = torch.zeros_like(h)
        degree = torch.zeros(B, L, 1, dtype=h.dtype, device=h.device)

        chunk = self.sender_chunk if self.sender_chunk > 0 else L
        for start in range(0, L, chunk):
            end = min(L, start + chunk)
            # receiver i, sender j in [start:end]
            gate = self.edge_gate(z[:, :, start:end, :])                    # [B,L,C,D]
            messages = gate * source[:, None, start:end, :]                 # [B,L,C,D]
            edge = pair_mask[:, :, start:end, None].to(messages.dtype)      # [B,L,C,1]
            message_sum += (messages * edge).sum(dim=2)
            degree += edge.sum(dim=2)

        aggregated = self.message_norm(message_sum / degree.clamp_min(1.0))
        delta = self.update(torch.cat([h, aggregated], dim=-1))
        return h + self.update_scale * delta


class OneHotBoltzMPNN(nn.Module):
    def __init__(
        self, L_T: int, L_P: int, L_H: int, d_z: int, d_hidden: int,
        component_dim: int, rL: int, rD: int, d_out: int, R_PH: float,
        dropout: float, graph_mode: str, z_ablation: str, edge_mask_prob: float,
        sender_chunk: int,
    ):
        super().__init__()
        self.L_T, self.L_P, self.L_H = L_T, L_P, L_H
        self.graph_mode = graph_mode
        self.z_ablation = z_ablation
        self.edge_mask_prob = edge_mask_prob
        self.component_embedding = nn.Embedding(3, component_dim)
        self.node_encoder = nn.Linear(VOCAB_SIZE + component_dim, d_hidden)
        self.message = None if z_ablation == "none" else ChunkedBoltzMessageLayer(d_hidden, d_z, dropout, sender_chunk)
        self.tcr_projector = LowRankProjectionHead(d_hidden, rL, rD, d_out, L_T, dropout)
        self.pmhc_projector = PMHCProjectionHead(d_hidden, rL, rD, d_out, L_P, L_H, R_PH, dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        xT, xP, xH = batch["emb_T"], batch["emb_P"], batch["emb_H"]
        mT, mP, mH = batch["mask_T"], batch["mask_P"], batch["mask_H"]
        x = torch.cat([xT, xP, xH], dim=1)
        token_mask = torch.cat([mT, mP, mH], dim=1)
        B, L, _ = x.shape

        ids = component_ids(B, self.L_T, self.L_P, self.L_H, x.device)
        h = self.node_encoder(torch.cat([x, self.component_embedding(ids)], dim=-1))
        h = h * token_mask[:, :, None].to(h.dtype)

        pair_mask = None
        if self.message is not None:
            valid = token_mask[:, :, None] & token_mask[:, None, :]
            eye = torch.eye(L, dtype=torch.bool, device=x.device)[None]
            pair_mask = valid & ~eye & graph_block_mask(ids, self.graph_mode)
            pair_mask = stochastic_symmetric_edge_mask(pair_mask, self.edge_mask_prob, self.training)

            z = batch["z"]
            if self.z_ablation == "zero":
                z = torch.zeros_like(z)
            elif self.z_ablation == "shuffled":
                z = torch.roll(z, shifts=1, dims=0) if B > 1 else torch.zeros_like(z)
            elif self.z_ablation != "boltz":
                raise ValueError(self.z_ablation)
            h = self.message(h, z, pair_mask)
            h = h * token_mask[:, :, None].to(h.dtype)

        t_end = self.L_T
        p_end = self.L_T + self.L_P
        hT, hP, hH = h[:, :t_end], h[:, t_end:p_end], h[:, p_end:]
        zT = self.tcr_projector(hT, mT)
        zPH = self.pmhc_projector(hP, mP, hH, mH)
        diagnostics = {"h_initial_or_updated": h, "pair_mask": pair_mask}
        return zT, zPH, diagnostics


# -----------------------------------------------------------------------------
# VICReg, scores, metrics
# -----------------------------------------------------------------------------


def vicreg_variance(u: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    u = u - u.mean(0, keepdim=True)
    std = torch.sqrt(u.var(0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.zeros((), device=u.device, dtype=u.dtype)
    u = u - u.mean(0, keepdim=True)
    cov = u.T @ u / (B - 1)
    off = cov - torch.diag_embed(torch.diag(cov))
    return off.square().sum() / d


def vicreg_loss(zT: torch.Tensor, zPH: torch.Tensor, cfg: "RunConfig", return_parts: bool = False):
    inv = F.mse_loss(zT, zPH)
    var = vicreg_variance(zT, cfg.gamma_var, cfg.eps_var) + vicreg_variance(zPH, cfg.gamma_var, cfg.eps_var)
    cov = vicreg_covariance(zT) + vicreg_covariance(zPH)
    loss = cfg.alpha * inv + cfg.beta * var + cfg.delta * cov
    if not return_parts:
        return loss
    return loss, {
        "loss": float(loss.detach()), "L_inv": float(inv.detach()), "L_var": float(var.detach()), "L_cov": float(cov.detach()),
        "weighted_inv": float((cfg.alpha * inv).detach()), "weighted_var": float((cfg.beta * var).detach()),
        "weighted_cov": float((cfg.delta * cov).detach()), "zT_std": float(zT.std(unbiased=False).detach()),
        "zPH_std": float(zPH.std(unbiased=False).detach()),
    }


def score_from_vectors(zT: torch.Tensor, zPH: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    distance = (zT - zPH).square().mean(-1)
    return -distance, distance


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
    rows = []
    frame = pd.DataFrame({"label": y.astype(int), "score": s.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in frame.groupby("peptide", sort=True):
        yy, ss = grp.label.to_numpy(), grp.score.to_numpy()
        valid = len(np.unique(yy)) == 2
        rows.append({
            "peptide": pep, "n": len(grp), "n_pos": int(yy.sum()), "n_neg": int((yy == 0).sum()), "valid": valid,
            "auroc": safe_auroc(yy, ss) if valid else float("nan"),
            "auc0.1_raw": safe_partial_auc_raw(yy, ss, max_fpr) if valid else float("nan"),
            "auc0.1_mcclish": safe_partial_auc_mcclish(yy, ss, max_fpr) if valid else float("nan"),
        })
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid = table[table.valid]
    if valid.empty:
        summary = {k: float("nan") for k in [
            "peptide_macro_auroc", "peptide_weighted_auroc",
            "peptide_macro_auc0.1_mcclish", "peptide_weighted_auc0.1_mcclish",
        ]}
        summary.update({"n_peptides_total": len(table), "n_peptides_valid": 0})
    else:
        summary = {
            "peptide_macro_auroc": float(valid.auroc.mean()),
            "peptide_weighted_auroc": float(np.average(valid.auroc, weights=valid.n)),
            "peptide_macro_auc0.1_mcclish": float(valid["auc0.1_mcclish"].mean()),
            "peptide_weighted_auc0.1_mcclish": float(np.average(valid["auc0.1_mcclish"], weights=valid.n)),
            "n_peptides_total": int(len(table)), "n_peptides_valid": int(len(valid)),
        }
    return table, summary


def metrics_for_scores(y: np.ndarray, s: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[Dict, pd.DataFrame]:
    table, pep = per_peptide_table(y, s, peptides, max_fpr)
    metrics = {
        "n_examples": int(len(y)), "n_positive": int(y.sum()), "n_negative": int((y == 0).sum()),
        "global_auroc": safe_auroc(y, s), "auprc": safe_auprc(y, s),
        "global_auc0.1_raw": safe_partial_auc_raw(y, s, max_fpr),
        "global_auc0.1_mcclish": safe_partial_auc_mcclish(y, s, max_fpr),
        "score_mean": float(s.mean()), "score_std": float(s.std()), **pep,
    }
    return metrics, table


@torch.no_grad()
def evaluate(loader: DataLoader, model: nn.Module, device: torch.device, cfg: "RunConfig", split: str, save_latents: bool) -> Dict:
    model.eval()
    labels, scores, distances, peptides, pair_ids = [], [], [], [], []
    latent_T, latent_PH = [], []
    metadata = {k: [] for k in ["has_alpha", "has_beta", "tcra_len", "tcrb_len", "pep_len", "hla_len"]}
    running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
    steps = 0

    for batch_cpu in loader:
        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch_cpu.items()}
        zT, zPH, _ = model(batch)
        _, parts = vicreg_loss(zT, zPH, cfg, return_parts=True)
        score, distance = score_from_vectors(zT, zPH)
        labels.append(batch["binding_flag"].cpu().numpy())
        scores.append(score.cpu().numpy()); distances.append(distance.cpu().numpy())
        peptides.extend(batch["peptide"]); pair_ids.extend(batch["pair_id"])
        for k in metadata:
            metadata[k].append(batch[k].cpu().numpy())
        if save_latents:
            latent_T.append(zT.cpu().numpy()); latent_PH.append(zPH.cpu().numpy())
        for k in running:
            running[k] += parts[k]
        steps += 1

    y = np.concatenate(labels).astype(int)
    s = np.concatenate(scores); d = np.concatenate(distances)
    pep = np.asarray(peptides, dtype=str)
    metrics, per_peptide = metrics_for_scores(y, s, pep, cfg.partial_auc_max_fpr)
    metrics.update({"mse_distance_mean": float(d.mean()), "mse_distance_std": float(d.std())})
    metrics.update({f"eval_{k}": v / max(1, steps) for k, v in running.items()})

    predictions = pd.DataFrame({
        "pair_id": pair_ids, "peptide": pep, "label": y, "score": s, "mse_distance": d,
        **{k: np.concatenate(v) for k, v in metadata.items()},
    })
    latents = None
    if save_latents:
        latents = {
            "zT": np.concatenate(latent_T), "zPH": np.concatenate(latent_PH),
            "pair_id": np.asarray(pair_ids, dtype=str), "peptide": pep, "label": y,
        }
    return {"split": split, "metrics": metrics, "per_peptide": per_peptide, "predictions": predictions, "latents": latents, "labels": y, "distances": d}


# -----------------------------------------------------------------------------
# Configuration and output
# -----------------------------------------------------------------------------


@dataclass
class RunConfig:
    run_tag: str = "onehot_mp_boltz_cross_mask10"
    repo_root: str = "/home/natasha/multimodal_model"
    train_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    checkpoint_root: str = "/home/natasha/multimodal_model/models/checkpoints/workshop/onehot_boltz_mpnn"
    output_root: str = "/home/natasha/multimodal_model/models/outputs/workshop/onehot_boltz_mpnn"
    figure_root: str = "/home/natasha/multimodal_model/models/figures/workshop/onehot_boltz_mpnn"

    seed: int = 31
    batch_size: int = 8
    num_workers: int = 4
    epochs: int = 30
    patience: int = 10
    min_epochs: int = 10

    max_train_rows: int = 0
    max_val_rows: int = 0
    max_test_rows: int = 0
    max_immrep_rows: int = 0
    max_steps_per_epoch: int = 0
    log_every: int = 0  # 0 = no per-step logs; only print the epoch summary line

    meta_cache_dir: str = "/home/natasha/multimodal_model/data/cache/onehot_boltz_mpnn"
    skip_z_audit: bool = False
    write_meta_cache: bool = False
    eval_splits: str = "val,test,immrep_test"

    cap_tcr: int = 300
    cap_pep: int = 0  # 0 = use observed max peptide length (train max is 24)
    cap_hla: int = 400
    d_hidden: int = 128
    component_dim: int = 8
    sender_chunk: int = 32
    rL: int = 8
    rD: int = 16
    d: int = 128
    R_PH: float = 0.7
    dropout: float = 0.1

    graph_mode: str = "cross"
    z_ablation: str = "boltz"
    edge_mask_prob: float = 0.10

    lr: float = 3e-4
    weight_decay: float = 1e-2
    alpha: float = 25.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    eps_var: float = 1e-4
    partial_auc_max_fpr: float = 0.1
    grad_clip_norm: float = 5.0

    save_latents: bool = False
    overwrite: bool = False


def prepare_dirs(cfg: RunConfig) -> Tuple[Path, Path, Path]:
    suffix = Path(cfg.run_tag) / f"seed_{cfg.seed}"
    dirs = [Path(cfg.checkpoint_root) / suffix, Path(cfg.output_root) / suffix, Path(cfg.figure_root) / suffix]
    if cfg.overwrite:
        for path in dirs:
            if path.exists():
                shutil.rmtree(path)
    for path in dirs:
        path.mkdir(parents=True, exist_ok=True)
    return tuple(dirs)  # type: ignore[return-value]


def make_loader(ds: Dataset, cfg: RunConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed) if shuffle else None
    loader_kwargs = {}
    if cfg.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True
    return DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers,
        collate_fn=collate_rows, generator=generator, pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )


def plot_histogram(distances: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for label, name in [(0, "negative"), (1, "positive")]:
        if np.any(labels == label):
            plt.hist(distances[labels == label], bins=50, density=True, alpha=0.55, label=name)
    plt.xlabel("MSE distance; lower = stronger predicted binding")
    plt.ylabel("Density"); plt.title(title); plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()


def save_eval(obj: Dict, output_dir: Path, figure_dir: Path, split: str, save_latents: bool) -> Dict[str, str]:
    paths = {}
    pred = output_dir / f"{split}_predictions.csv"; obj["predictions"].to_csv(pred, index=False); paths[f"{split}_predictions"] = str(pred)
    pep = output_dir / f"{split}_per_peptide.csv"; obj["per_peptide"].to_csv(pep, index=False); paths[f"{split}_per_peptide"] = str(pep)
    fig = figure_dir / f"{split}_mse_hist.png"; plot_histogram(obj["distances"], obj["labels"], f"{split}: MSE distance", fig); paths[f"{split}_hist"] = str(fig)
    if save_latents and obj["latents"] is not None:
        lp = output_dir / f"{split}_latents.npz"; np.savez_compressed(lp, **obj["latents"]); paths[f"{split}_latents"] = str(lp)
    return paths


def parse_eval_splits(value: str) -> List[str]:
    allowed = {"val", "test", "immrep_test"}
    splits = [part.strip() for part in value.split(",") if part.strip()]
    if not splits:
        raise ValueError("eval_splits must list at least one split")
    unknown = [split for split in splits if split not in allowed]
    if unknown:
        raise ValueError(f"unknown eval_splits entries: {unknown}; allowed={sorted(allowed)}")
    return splits


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name in ["run_tag", "repo_root", "train_csv", "val_csv", "test_csv", "immrep_csv", "checkpoint_root", "output_root", "figure_root", "meta_cache_dir", "eval_splits"]:
        p.add_argument("--" + name.replace("_", "-"), default=getattr(RunConfig, name))
    for name in [
        "seed", "batch_size", "num_workers", "epochs", "patience", "min_epochs",
        "max_train_rows", "max_val_rows", "max_test_rows", "max_immrep_rows",
        "max_steps_per_epoch", "log_every",
        "cap_tcr", "cap_pep", "cap_hla", "d_hidden", "component_dim", "sender_chunk", "rL", "rD", "d",
    ]:
        p.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(RunConfig, name))
    for name in ["R_PH", "dropout", "edge_mask_prob", "lr", "weight_decay", "alpha", "beta", "delta", "gamma_var", "eps_var", "partial_auc_max_fpr", "grad_clip_norm"]:
        p.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(RunConfig, name))
    p.add_argument("--graph-mode", choices=["tp", "cross", "all"], default=RunConfig.graph_mode)
    p.add_argument("--z-ablation", choices=["none", "zero", "shuffled", "boltz"], default=RunConfig.z_ablation)
    p.add_argument("--skip-z-audit", action="store_true", default=RunConfig.skip_z_audit)
    p.add_argument("--write-meta-cache", action="store_true", default=RunConfig.write_meta_cache)
    p.add_argument("--save-latents", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = vars(p.parse_args())
    return RunConfig(**args)


# -----------------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------------


def main() -> None:
    cfg = parse_args()
    if cfg.z_ablation == "none" and cfg.edge_mask_prob != 0:
        print("z_ablation=none: edge_mask_prob is ignored", flush=True)
    if cfg.z_ablation in {"zero", "shuffled"} and cfg.edge_mask_prob > 0:
        print("Control run uses stochastic edge masking; set --edge-mask-prob 0 for a pure control.", flush=True)

    set_seed(cfg.seed)
    checkpoint_dir, output_dir, figure_dir = prepare_dirs(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    repo_root = Path(cfg.repo_root)
    meta_cache_dir = Path(cfg.meta_cache_dir)
    load_kwargs = {
        "skip_z_audit": cfg.skip_z_audit,
        "meta_cache_dir": meta_cache_dir,
        "write_meta_cache": cfg.write_meta_cache,
    }
    train_meta, train_audit = load_meta(cfg.train_csv, "train", True, repo_root, **load_kwargs)
    val_meta, val_audit = load_meta(cfg.val_csv, "val", False, repo_root, **load_kwargs)
    test_meta, test_audit = load_meta(cfg.test_csv, "test", False, repo_root, **load_kwargs)
    immrep_meta, immrep_audit = load_meta(cfg.immrep_csv, "immrep_test", False, repo_root, **load_kwargs)
    pd.DataFrame([train_audit, val_audit, test_audit, immrep_audit]).to_csv(output_dir / "split_filter_audit.csv", index=False)

    train_meta = limit_rows(train_meta, cfg.max_train_rows, cfg.seed, "train")
    val_meta = limit_rows(val_meta, cfg.max_val_rows, cfg.seed, "val")
    test_meta = limit_rows(test_meta, cfg.max_test_rows, cfg.seed, "test")
    immrep_meta = limit_rows(immrep_meta, cfg.max_immrep_rows, cfg.seed, "immrep_test")
    eval_splits = parse_eval_splits(cfg.eval_splits)

    L_T, L_P, L_H, d_z = infer_padded_lengths(
        [train_meta, val_meta, test_meta, immrep_meta], cfg.cap_tcr, cfg.cap_pep, cfg.cap_hla
    )
    print(f"device={device} lengths: T={L_T} P={L_P} H={L_H} total={L_T+L_P+L_H} d_z={d_z}", flush=True)
    print(f"model: z_ablation={cfg.z_ablation} graph_mode={cfg.graph_mode} edge_mask_prob={cfg.edge_mask_prob}", flush=True)
    print(f"data: train={len(train_meta)} val={len(val_meta)} test={len(test_meta)} immrep={len(immrep_meta)}", flush=True)
    if cfg.max_steps_per_epoch > 0:
        print(f"train capped at max_steps_per_epoch={cfg.max_steps_per_epoch}", flush=True)
    print(f"final eval splits: {eval_splits}", flush=True)
    print("selection metric: val peptide_weighted_auc0.1_mcclish", flush=True)

    datasets = {
        name: OneHotBoltzDataset(meta, L_T, L_P, L_H, name, cfg.graph_mode)
        for name, meta in [
            ("train", train_meta),
            ("val", val_meta),
            ("test", test_meta),
            ("immrep_test", immrep_meta),
        ]
    }
    loaders = {
        name: make_loader(ds, cfg, shuffle=(name == "train")) for name, ds in datasets.items()
    }

    model = OneHotBoltzMPNN(
        L_T, L_P, L_H, d_z, cfg.d_hidden, cfg.component_dim, cfg.rL, cfg.rD, cfg.d,
        cfg.R_PH, cfg.dropout, cfg.graph_mode, cfg.z_ablation, cfg.edge_mask_prob, cfg.sender_chunk,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best = {"metric": -math.inf, "epoch": None, "state": None, "bad_epochs": 0}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
        steps = 0
        for batch_cpu in loaders["train"]:
            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch_cpu.items()}
            zT, zPH, _ = model(batch)
            loss, parts = vicreg_loss(zT, zPH, cfg, return_parts=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            for k in running:
                running[k] += parts[k]
            steps += 1
            if cfg.log_every > 0 and (steps == 1 or steps % cfg.log_every == 0):
                print(
                    f"Epoch {epoch:02d} step {steps} loss={parts['loss']:.4f} "
                    f"inv={parts['L_inv']:.4f} var={parts['L_var']:.4f} cov={parts['L_cov']:.4f}",
                    flush=True,
                )
            if cfg.max_steps_per_epoch > 0 and steps >= cfg.max_steps_per_epoch:
                print(f"Epoch {epoch:02d}: reached max_steps_per_epoch={cfg.max_steps_per_epoch}", flush=True)
                break
        scheduler.step()

        val_obj = evaluate(loaders["val"], model, device, cfg, "val", False)
        current = val_obj["metrics"]["peptide_weighted_auc0.1_mcclish"]
        row = {"epoch": epoch, "lr": scheduler.get_last_lr()[0], **{f"train_{k}": v / max(1, steps) for k, v in running.items()}}
        row.update({f"val_{k}": v for k, v in val_obj["metrics"].items()})
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        improved = np.isfinite(current) and current > best["metric"] + 1e-4
        if improved:
            best = {"metric": float(current), "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "bad_epochs": 0}
        else:
            best["bad_epochs"] += 1

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} loss={row['train_loss']:.4f} inv={row['train_L_inv']:.4f} "
            f"var={row['train_L_var']:.4f} cov={row['train_L_cov']:.4f} "
            f"val_global={val_obj['metrics']['global_auroc']:.4f} "
            f"val_pep_w_auc0.1_mcclish={current:.4f} best_epoch={best['epoch']} bad={best['bad_epochs']}",
            flush=True,
        )
        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No validation checkpoint selected; inspect peptide labels and metrics.")
    model.load_state_dict(best["state"])

    final = {
        split: evaluate(loaders[split], model, device, cfg, split, cfg.save_latents)
        for split in eval_splits
    }
    paths = {}
    for split, obj in final.items():
        paths.update(save_eval(obj, output_dir, figure_dir, split, cfg.save_latents))

    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save({
        "config": asdict(cfg), "shapes": {"L_T": L_T, "L_P": L_P, "L_H": L_H, "d_z": d_z},
        "model_state_dict": model.state_dict(), "best_epoch": best["epoch"],
        "best_val_peptide_weighted_auc0.1_mcclish": best["metric"],
        "metrics": {k: v["metrics"] for k, v in final.items()},
    }, checkpoint_path)

    summary = {
        "config": asdict(cfg), "model_family": "onehot_boltz_mpnn", "seed": cfg.seed,
        "best_epoch": best["epoch"],
        "best_selection_metric": "val.peptide_weighted_auc0.1_mcclish",
        "best_selection_value": best["metric"],
        "metrics": {k: v["metrics"] for k, v in final.items()},
        "paths": {
            "checkpoint": str(checkpoint_path), "history": str(output_dir / "history.csv"),
            "run_config": str(output_dir / "run_config.json"), "split_filter_audit": str(output_dir / "split_filter_audit.csv"),
            "output_dir": str(output_dir), "figure_dir": str(figure_dir), **paths,
        },
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80, flush=True)
    print(f"Done. Best epoch={best['epoch']} val weighted McClish AUC0.1={best['metric']:.4f}", flush=True)
    if "immrep_test" in final:
        imm = final["immrep_test"]["metrics"]
        print(f"IMMREP weighted McClish AUC0.1={imm['peptide_weighted_auc0.1_mcclish']:.4f}", flush=True)
    print(f"Summary: {output_dir / 'summary.json'}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
