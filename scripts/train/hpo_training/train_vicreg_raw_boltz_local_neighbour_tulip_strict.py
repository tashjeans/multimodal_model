#!/usr/bin/env python3
"""
Plain VICReg + raw-Boltz local-neighbour structural guidance.

This script tests a softer use of Boltz structure than the earlier scalar-confidence
or Hamiltonian-style losses. Boltz is used only during training, only for positive
examples with available raw z tensors, and only to preserve local structural
neighbourhoods among related positives. Validation/test/IMMREP inference remains
sequence-only.

Default placement on the server:
    /home/natasha/multimodal_model/scripts/train/hpo_training/train_vicreg_raw_boltz_local_neighbour_tulip_strict.py

Typical launch from that folder:
    cd /home/natasha/multimodal_model/scripts/train/hpo_training
    conda activate tcr-multimodal
    export TOKENIZERS_PARALLELISM=false
    python train_vicreg_raw_boltz_local_neighbour_tulip_strict.py 2>&1 | tee /home/natasha/multimodal_model/models/checkpoints/hpo_training/raw_boltz_local_neighbour_tulip_strict/train.log

Main outputs:
    /home/natasha/multimodal_model/models/checkpoints/hpo_training/raw_boltz_local_neighbour
    /home/natasha/multimodal_model/models/figures/hpo_training/raw_boltz_local_neighbour

Core idea:
    1. Extract compact raw-Boltz z interface vectors from embeddings_pair_*.npz.
    2. Build kNN neighbourhoods within local biological groups only, e.g. peptide_group
       or hla_group. No global Boltz ranking is imposed.
    3. Train the usual sequence-only VICReg model, plus a small auxiliary loss that
       makes model pair embeddings preserve Boltz-defined local neighbour relations.

Score convention:
    cos = cosine(TCR, pMHC)
    H = -1 - cos
    score = -H = 1 + cos
    higher score = more likely binder.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Utilities
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


def to_str_list(x: Any) -> List[str]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif not isinstance(x, (list, tuple)):
        x = [x]
    out = []
    for v in x:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))
    return out


def row_normalise(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return u / (u.norm(dim=-1, keepdim=True) + eps)


def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    return (emb * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + eps)


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


# ============================================================
# Flat shard dataset
# ============================================================

class FlatShardedDataset(Dataset):
    """Flatten the existing shard_*.pt files from pre-batched dicts into examples.

    Existing shards look like a list of dicts. Each dict contains tensors with
    first dimension B and a list of pair_id values. This dataset indexes each
    individual row, but still caches one shard at a time to avoid repeated disk
    reads inside a worker.
    """

    def __init__(self, shards_dir: Path, pair_to_group: Optional[Dict[str, str]] = None):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shards_dir}")
        self.pair_to_group = pair_to_group or {}
        self.index: List[Tuple[Path, int, int]] = []
        self.meta: List[Dict[str, Any]] = []
        self._cache_path: Optional[Path] = None
        self._cache_data: Optional[Any] = None

        print(f"Indexing flat dataset: {self.shards_dir}", flush=True)
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for bidx, batch in enumerate(shard):
                pair_ids = to_str_list(batch["pair_id"])
                labels = batch["binding_flag"]
                labels = labels.detach().cpu().numpy().tolist() if torch.is_tensor(labels) else list(labels)
                for ridx, pid in enumerate(pair_ids):
                    self.index.append((sp, bidx, ridx))
                    self.meta.append({
                        "pair_id": str(pid),
                        "binding_flag": int(labels[ridx]),
                        "group": str(self.pair_to_group.get(str(pid), "NA")),
                    })
        print(f"Flat dataset rows={len(self.index)} from {len(self.shard_paths)} shards", flush=True)

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, sp: Path):
        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp
        return self._cache_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sp, bidx, ridx = self.index[idx]
        batch = self._load_shard(sp)[bidx]
        out: Dict[str, Any] = {}
        for key in ["emb_T", "emb_P", "emb_H", "mask_T", "mask_P", "mask_H"]:
            out[key] = batch[key][ridx]
        labels = batch["binding_flag"]
        out["binding_flag"] = int(labels[ridx].item() if torch.is_tensor(labels) else labels[ridx])
        pair_ids = to_str_list(batch["pair_id"])
        out["pair_id"] = str(pair_ids[ridx])
        out["group"] = str(self.pair_to_group.get(out["pair_id"], "NA"))
        return out


def flat_collate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "emb_T": torch.stack([r["emb_T"] for r in rows], dim=0),
        "emb_P": torch.stack([r["emb_P"] for r in rows], dim=0),
        "emb_H": torch.stack([r["emb_H"] for r in rows], dim=0),
        "mask_T": torch.stack([r["mask_T"] for r in rows], dim=0),
        "mask_P": torch.stack([r["mask_P"] for r in rows], dim=0),
        "mask_H": torch.stack([r["mask_H"] for r in rows], dim=0),
        "binding_flag": torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long),
        "pair_id": [r["pair_id"] for r in rows],
        "group": [r["group"] for r in rows],
    }


# ============================================================
# Model
# ============================================================

class ESMProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float = 0.1):
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
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P_max: int, L_H_max: int, R_PH: float = 0.7, dropout: float = 0.1):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; d_P={d_P}, d_H={d_H}")
        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pep_encoder(emb_P, mask_P), self.hla_encoder(emb_H, mask_H)], dim=-1)


# ============================================================
# VICReg + scoring
# ============================================================

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


def plain_vicreg_loss(zT: torch.Tensor, zPH: torch.Tensor, alpha: float, beta: float, delta: float, gamma_var: float, eps_norm: float, eps_var: float, return_parts: bool = False):
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
        "H_std": float(H.std(unbiased=False).detach().cpu()),
        "eTstd": float(eT.std(unbiased=False).detach().cpu()),
        "ePHstd": float(ePH.std(unbiased=False).detach().cpu()),
    }
    return loss, parts


def score_from_projected(zT: torch.Tensor, zPH: torch.Tensor, eps_norm: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    eT = row_normalise(zT, eps_norm)
    ePH = row_normalise(zPH, eps_norm)
    cos = (eT * ePH).sum(dim=-1)
    H = -1.0 - cos
    return -H, H, cos


def raw_esm_score(batch: Dict[str, Any], device: torch.device, eps_norm: float = 1e-8, R_PH: float = 0.7):
    T = masked_mean_pool(batch["emb_T"].to(device), batch["mask_T"].to(device), eps_norm)
    P = masked_mean_pool(batch["emb_P"].to(device), batch["mask_P"].to(device), eps_norm)
    HLA = masked_mean_pool(batch["emb_H"].to(device), batch["mask_H"].to(device), eps_norm)
    PH = R_PH * P + (1.0 - R_PH) * HLA
    score, H, cos = score_from_projected(T, PH, eps_norm)
    return score, H, cos, row_normalise(T, eps_norm)


# ============================================================
# Raw Boltz z feature extraction + neighbourhood cache
# ============================================================

def pair_id_aliases(pair_id: str) -> List[str]:
    aliases = {str(pair_id)}
    s = str(pair_id)
    if s.startswith("pair_"):
        tail = s.split("pair_", 1)[1]
        if tail.isdigit():
            n = int(tail)
            aliases.update({f"pair_{n}", f"pair_{n:03d}", f"pair_{n:06d}"})
    return sorted(aliases)


def build_boltz_prediction_index(outputs_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for pattern in ["chunk_*/boltz_results_pair_*/predictions/pair_*", "**/boltz_results_pair_*/predictions/pair_*"]:
        for p in Path(outputs_root).glob(pattern):
            if not p.is_dir():
                continue
            for alias in pair_id_aliases(p.name):
                index.setdefault(alias, p)
        if index:
            break
    return index


def find_pred_dir(index: Dict[str, Path], pair_id: str) -> Optional[Path]:
    for alias in pair_id_aliases(pair_id):
        if alias in index:
            return index[alias]
    return None


def first_z_array(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "z" in data:
        z = data["z"]
    else:
        z_keys = [k for k in data.keys() if k.lower() == "z" or "z" in k.lower()]
        if not z_keys:
            raise KeyError(f"No z-like array in {npz_path}; keys={list(data.keys())}")
        z = data[z_keys[0]]
    z = np.asarray(z)
    if z.ndim == 4:
        z = z[0]
    if z.ndim != 3:
        raise ValueError(f"Expected z shape (L,L,D) or (B,L,L,D), got {z.shape} in {npz_path}")
    return z.astype(np.float32, copy=False)


def chain_slices(row: pd.Series, chain_order: str) -> Dict[str, slice]:
    lengths = {
        "tcra": int(row.get("tcra_len", 0)),
        "tcrb": int(row.get("tcrb_len", 0)),
        "pep": int(row.get("pep_len", 0)),
        "hla": int(row.get("hla_len", 0)),
    }
    pos = 0
    sl: Dict[str, slice] = {}
    for name in [x.strip().lower() for x in chain_order.split(",") if x.strip()]:
        if name not in lengths:
            raise ValueError(f"Unknown chain '{name}' in chain order '{chain_order}'")
        L = lengths[name]
        sl[name] = slice(pos, pos + L)
        pos += L
    starts = [sl[c].start for c in ["tcra", "tcrb"] if c in sl and sl[c].stop > sl[c].start]
    stops = [sl[c].stop for c in ["tcra", "tcrb"] if c in sl and sl[c].stop > sl[c].start]
    if starts and stops:
        sl["tcr"] = slice(min(starts), max(stops))
    else:
        sl["tcr"] = slice(0, 0)
    return sl


def block_stats(z: np.ndarray, row: pd.Series, chain_order: str, rep: str) -> np.ndarray:
    sl = chain_slices(row, chain_order)
    pep = sl["pep"]
    hla = sl["hla"]
    tcr = sl["tcr"]
    if rep == "z_tcr_pep":
        blocks = [z[tcr, pep, :], z[pep, tcr, :]]
    elif rep == "z_tcr_hla":
        blocks = [z[tcr, hla, :], z[hla, tcr, :]]
    elif rep == "z_pep_hla":
        blocks = [z[pep, hla, :], z[hla, pep, :]]
    elif rep == "z_tcr_pmhc":
        pmhc = slice(min(pep.start, hla.start), max(pep.stop, hla.stop))
        blocks = [z[tcr, pmhc, :], z[pmhc, tcr, :]]
    elif rep == "concat_all_interfaces":
        blocks = [z[tcr, pep, :], z[pep, tcr, :], z[tcr, hla, :], z[hla, tcr, :], z[pep, hla, :], z[hla, pep, :]]
    else:
        raise ValueError(f"Unknown z representation: {rep}")

    feats = []
    for b in blocks:
        arr = np.asarray(b, dtype=np.float32)
        if arr.size == 0:
            continue
        flat = arr.reshape(-1, arr.shape[-1])
        feats.append(np.nanmean(flat, axis=0))
        feats.append(np.nanstd(flat, axis=0))
    if not feats:
        raise ValueError("Empty interface block after slicing")
    return np.concatenate(feats).astype(np.float32)


def build_or_load_raw_z_feature_cache(cfg: "RunConfig") -> Tuple[pd.DataFrame, np.ndarray]:
    cache_csv = Path(cfg.raw_z_feature_cache_csv)
    cache_npz = Path(cfg.raw_z_feature_cache_npz)
    if cache_csv.exists() and cache_npz.exists() and not cfg.force_rebuild_raw_z_cache:
        meta = pd.read_csv(cache_csv)
        X = np.load(cache_npz)["X"].astype(np.float32)
        print(f"Loaded raw-z feature cache | rows={len(meta)} | dim={X.shape[1]} | {cache_csv}", flush=True)
        return meta, X

    manifest = pd.read_csv(cfg.train_manifest_csv)
    required = {"pair_id", "binding_flag", "pep_len", "tcra_len", "tcrb_len", "hla_len"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    pos = manifest[manifest["binding_flag"].astype(int) == 1].copy()
    if cfg.max_raw_z_pairs > 0:
        pos = pos.head(cfg.max_raw_z_pairs).copy()

    print(f"Indexing raw Boltz prediction dirs under {cfg.boltz_train_root}", flush=True)
    pred_index = build_boltz_prediction_index(Path(cfg.boltz_train_root))
    print(f"Indexed aliases={len(pred_index)} unique_dirs={len(set(map(str, pred_index.values())))}", flush=True)

    rows = []
    feats = []
    for i, (_, row) in enumerate(pos.iterrows(), start=1):
        pid = str(row["pair_id"])
        pred_dir = find_pred_dir(pred_index, pid)
        if pred_dir is None:
            continue
        npz_path = pred_dir / f"embeddings_{pid}.npz"
        if not npz_path.exists():
            # tolerate zero-padding mismatch between directory name and manifest pair_id
            candidates = sorted(pred_dir.glob("embeddings_pair_*.npz"))
            if not candidates:
                continue
            npz_path = candidates[0]
        try:
            z = first_z_array(npz_path)
            f = block_stats(z, row, cfg.boltz_chain_order, cfg.raw_z_representation)
            group = str(row[cfg.local_group_col]) if cfg.local_group_col in row and pd.notna(row[cfg.local_group_col]) else str(row.get("Peptide", pid))
            rows.append({
                "pair_id": pid,
                "group": group,
                "peptide": str(row.get("Peptide", "")),
                "hla_group": str(row.get("hla_group", "")),
                "peptide_group": str(row.get("peptide_group", "")),
                "npz_path": str(npz_path),
            })
            feats.append(f)
        except Exception as e:
            if cfg.verbose_raw_z_errors:
                print(f"raw-z extraction failed for {pid}: {repr(e)}", flush=True)
        if i % cfg.raw_z_progress_every == 0:
            print(f"  scanned {i}/{len(pos)} positives | usable={len(rows)}", flush=True)

    if not feats:
        raise RuntimeError("No raw Boltz z features were extracted. Check paths, chain order, and npz schema.")
    max_dim = max(len(f) for f in feats)
    X = np.zeros((len(feats), max_dim), dtype=np.float32)
    for i, f in enumerate(feats):
        X[i, :len(f)] = f
    meta = pd.DataFrame(rows)

    # Standardise, then optional PCA. The standardised/PCA representation is what the loss sees.
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    Xs = (X - mu) / sd
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if cfg.raw_z_pca_dim > 0 and Xs.shape[1] > cfg.raw_z_pca_dim and Xs.shape[0] > cfg.raw_z_pca_dim:
        pca = PCA(n_components=cfg.raw_z_pca_dim, random_state=cfg.seed)
        Xs = pca.fit_transform(Xs).astype(np.float32)
        print(f"Applied PCA to raw-z features | dim={Xs.shape[1]} | explained={pca.explained_variance_ratio_.sum():.3f}", flush=True)

    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(cache_csv, index=False)
    np.savez_compressed(cache_npz, X=Xs)
    print(f"Saved raw-z feature cache | rows={len(meta)} | dim={Xs.shape[1]} | {cache_csv}", flush=True)
    return meta, Xs


class RawBoltzLocalNeighbourGuidance:
    def __init__(self, meta: pd.DataFrame, X: np.ndarray, k: int = 10, min_group_size: int = 3):
        self.meta = meta.copy()
        self.X = X.astype(np.float32)
        self.k = int(k)
        self.min_group_size = int(min_group_size)
        self.pid_to_idx = {str(pid): i for i, pid in enumerate(self.meta["pair_id"].astype(str).tolist())}
        self.pid_to_group = dict(zip(self.meta["pair_id"].astype(str), self.meta["group"].astype(str)))
        self.pid_to_vec = {str(pid): self.X[i] for i, pid in enumerate(self.meta["pair_id"].astype(str).tolist())}
        self.neighbour_edges = self._build_edges()
        print(f"Raw Boltz local guidance | usable_pair_ids={len(self.pid_to_idx)} | directed_edges={len(self.neighbour_edges)}", flush=True)

    def _build_edges(self) -> set[Tuple[str, str]]:
        edges: set[Tuple[str, str]] = set()
        df = self.meta.reset_index(drop=True)
        for group, g in df.groupby("group", sort=False):
            idx = g.index.to_numpy()
            if len(idx) < self.min_group_size:
                continue
            n_nei = min(self.k + 1, len(idx))
            nn = NearestNeighbors(n_neighbors=n_nei, metric="cosine")
            nn.fit(self.X[idx])
            _, neigh = nn.kneighbors(self.X[idx])
            pids = g["pair_id"].astype(str).to_numpy()
            for local_i, row_neigh in enumerate(neigh):
                src = pids[local_i]
                for local_j in row_neigh[1:]:
                    dst = pids[int(local_j)]
                    edges.add((src, dst))
        return edges

    def tensors_for_batch(self, pair_ids: List[str], labels: torch.Tensor, groups: List[str], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vecs = []
        has = []
        for pid, y in zip(pair_ids, labels.detach().cpu().numpy().tolist()):
            ok = int(y) == 1 and str(pid) in self.pid_to_vec
            has.append(ok)
            if ok:
                vecs.append(self.pid_to_vec[str(pid)])
            else:
                vecs.append(np.zeros(self.X.shape[1], dtype=np.float32))
        teacher = torch.tensor(np.stack(vecs), dtype=torch.float32, device=device)
        mask = torch.tensor(has, dtype=torch.bool, device=device)

        B = len(pair_ids)
        same_group = torch.zeros((B, B), dtype=torch.bool, device=device)
        knn_edge = torch.zeros((B, B), dtype=torch.bool, device=device)
        for i in range(B):
            for j in range(B):
                if i == j:
                    continue
                if mask[i] and mask[j] and str(groups[i]) == str(groups[j]):
                    same_group[i, j] = True
                if (str(pair_ids[i]), str(pair_ids[j])) in self.neighbour_edges or (str(pair_ids[j]), str(pair_ids[i])) in self.neighbour_edges:
                    knn_edge[i, j] = True
        return mask, teacher, same_group, knn_edge


def local_neighbour_loss(
    pair_rep: torch.Tensor,
    teacher_vec: torch.Tensor,
    has_mask: torch.Tensor,
    same_group: torch.Tensor,
    knn_edge: torch.Tensor,
    lambda_rel: float,
    lambda_attr: float,
    teacher_temp: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if has_mask.sum().item() < 2 or (lambda_rel <= 0 and lambda_attr <= 0):
        z = torch.tensor(0.0, device=pair_rep.device, dtype=pair_rep.dtype)
        return z, {"boltz_local_rel": 0.0, "boltz_local_attr": 0.0, "n_local_pairs": 0.0, "n_knn_pairs": 0.0}

    q = row_normalise(pair_rep, 1e-8)
    t = row_normalise(teacher_vec.to(pair_rep.dtype), 1e-8)
    student_sim = q @ q.T
    teacher_sim = t @ t.T

    valid_rel = same_group & has_mask[:, None] & has_mask[None, :]
    valid_rel.fill_diagonal_(False)
    valid_knn = knn_edge & has_mask[:, None] & has_mask[None, :]
    valid_knn.fill_diagonal_(False)

    rel_loss = torch.tensor(0.0, device=pair_rep.device, dtype=pair_rep.dtype)
    attr_loss = torch.tensor(0.0, device=pair_rep.device, dtype=pair_rep.dtype)

    if valid_rel.sum().item() > 0 and lambda_rel > 0:
        # Weight more local/near pairs, but only within the selected biological group.
        w = torch.softmax(teacher_sim[valid_rel] / teacher_temp, dim=0).detach()
        rel_loss = lambda_rel * (w * (student_sim[valid_rel] - teacher_sim[valid_rel].detach()).pow(2)).sum()

    if valid_knn.sum().item() > 0 and lambda_attr > 0:
        # Pull Boltz-neighbour positives together in the learned pair space.
        w = torch.clamp((teacher_sim[valid_knn].detach() + 1.0) / 2.0, min=0.0, max=1.0)
        attr_loss = lambda_attr * (w * (1.0 - student_sim[valid_knn])).mean()

    total = rel_loss + attr_loss
    return total, {
        "boltz_local_rel": float(rel_loss.detach().cpu()),
        "boltz_local_attr": float(attr_loss.detach().cpu()),
        "n_local_pairs": float(valid_rel.sum().detach().cpu()),
        "n_knn_pairs": float(valid_knn.sum().detach().cpu()),
    }


# ============================================================
# Metrics and plots
# ============================================================

def per_peptide_auroc(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        auc = float(roc_auc_score(y, s)) if valid else float("nan")
        rows.append({"peptide": pep, "n": int(len(grp)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "auroc": auc, "valid": bool(valid)})
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]].copy()
    if len(valid_table) == 0:
        summary = {"macro": float("nan"), "weighted": float("nan"), "n_total": len(table), "n_valid": 0}
    else:
        summary = {"macro": float(valid_table["auroc"].mean()), "weighted": float(np.average(valid_table["auroc"], weights=valid_table["n"])), "n_total": int(len(table)), "n_valid": int(len(valid_table))}
    return table, summary


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best = None
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        row = {"threshold": float(thr), "f1": float(f1_score(labels, pred, zero_division=0)), "accuracy": float(accuracy_score(labels, pred)), "precision": float(precision_score(labels, pred, zero_division=0)), "recall": float(recall_score(labels, pred, zero_division=0))}
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best or {"threshold": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan")}


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, prefix: str) -> Dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_f1": float(f1_score(labels, pred, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}_precision": float(precision_score(labels, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, pred, zero_division=0)),
        f"{prefix}_cm": confusion_matrix(labels, pred).tolist(),
    }


def load_pair_to_col(csv_path: str, col_candidates: List[str], required: bool = False) -> Dict[str, str]:
    if not csv_path:
        return {}
    p = Path(csv_path)
    if not p.exists():
        if required:
            raise FileNotFoundError(csv_path)
        print(f"Optional CSV not found: {csv_path}", flush=True)
        return {}
    df = pd.read_csv(p)
    if "pair_id" not in df.columns:
        if required:
            raise ValueError(f"{csv_path} must contain pair_id")
        return {}
    col = next((c for c in col_candidates if c in df.columns), None)
    if col is None:
        if required:
            raise ValueError(f"{csv_path} must contain one of {col_candidates}")
        return {}
    return dict(zip(df["pair_id"].astype(str), df[col].astype(str)))


@torch.no_grad()
def evaluate(loader: DataLoader, tcr: nn.Module, pmhc: nn.Module, device: torch.device, cfg: "RunConfig", pair_to_peptide: Dict[str, str], split: str) -> Dict[str, Any]:
    tcr.eval(); pmhc.eval()
    model_scores, model_H, model_cos = [], [], []
    raw_scores, raw_H, raw_cos = [], [], []
    labels_all, pair_ids_all, peptides_all = [], [], []
    running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "H_std", "eTstd", "ePHstd"]}
    n_steps = 0
    for batch in loader:
        zT = tcr(batch["emb_T"].to(device), batch["mask_T"].to(device))
        zPH = pmhc(batch["emb_P"].to(device), batch["mask_P"].to(device), batch["emb_H"].to(device), batch["mask_H"].to(device))
        loss, parts = plain_vicreg_loss(zT, zPH, cfg.alpha, cfg.beta, cfg.delta, cfg.gamma_var, cfg.eps_norm, cfg.eps_var, return_parts=True)
        s_m, H_m, cos_m = score_from_projected(zT, zPH, cfg.eps_norm)
        s_r, H_r, cos_r, _ = raw_esm_score(batch, device, cfg.eps_norm, cfg.R_PH)
        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        pair_ids = to_str_list(batch["pair_id"])
        peptides = np.asarray([pair_to_peptide.get(pid, pid) for pid in pair_ids], dtype=str)
        model_scores.append(s_m.detach().cpu().numpy()); model_H.append(H_m.detach().cpu().numpy()); model_cos.append(cos_m.detach().cpu().numpy())
        raw_scores.append(s_r.detach().cpu().numpy()); raw_H.append(H_r.detach().cpu().numpy()); raw_cos.append(cos_r.detach().cpu().numpy())
        labels_all.append(labels); pair_ids_all.extend(pair_ids); peptides_all.append(peptides)
        running["loss"] += float(loss.detach().cpu())
        for k in running:
            if k != "loss": running[k] += float(parts[k])
        n_steps += 1

    labels = np.concatenate(labels_all).astype(int)
    peptides = np.concatenate(peptides_all).astype(str)
    model_scores = np.concatenate(model_scores); model_H = np.concatenate(model_H); model_cos = np.concatenate(model_cos)
    raw_scores = np.concatenate(raw_scores); raw_H = np.concatenate(raw_H); raw_cos = np.concatenate(raw_cos)
    model_pep_table, model_pep = per_peptide_auroc(labels, model_scores, peptides)
    raw_pep_table, raw_pep = per_peptide_auroc(labels, raw_scores, peptides)
    thr = best_f1_threshold(model_scores, labels)
    metrics = {
        f"{split}_loss": running["loss"] / max(1, n_steps),
        f"{split}_model_global_auroc": safe_auroc(labels, model_scores),
        f"{split}_raw_esm_global_auroc": safe_auroc(labels, raw_scores),
        f"{split}_delta_global_auroc": safe_auroc(labels, model_scores) - safe_auroc(labels, raw_scores),
        f"{split}_model_auprc": safe_auprc(labels, model_scores),
        f"{split}_raw_esm_auprc": safe_auprc(labels, raw_scores),
        f"{split}_model_peptide_weighted_auroc": model_pep["weighted"],
        f"{split}_raw_esm_peptide_weighted_auroc": raw_pep["weighted"],
        f"{split}_delta_peptide_weighted_auroc": model_pep["weighted"] - raw_pep["weighted"],
        f"{split}_model_peptide_macro_auroc": model_pep["macro"],
        f"{split}_n_peptides_valid": model_pep["n_valid"],
        f"{split}_threshold": thr["threshold"],
        f"{split}_f1": thr["f1"],
        f"{split}_accuracy": thr["accuracy"],
        f"{split}_precision": thr["precision"],
        f"{split}_recall": thr["recall"],
        f"{split}_H_std": float(np.std(model_H)),
        f"{split}_raw_H_std": float(np.std(raw_H)),
    }
    for k, v in running.items():
        if k != "loss": metrics[f"{split}_{k}"] = v / max(1, n_steps)
    predictions = pd.DataFrame({"pair_id": pair_ids_all, "peptide": peptides, "label": labels, "model_score": model_scores, "model_H": model_H, "model_cos": model_cos, "raw_esm_score": raw_scores, "raw_esm_H": raw_H, "raw_esm_cos": raw_cos})
    return {"metrics": metrics, "predictions": predictions, "model_peptide_table": model_pep_table, "raw_peptide_table": raw_pep_table, "model_H": model_H, "raw_H": raw_H, "labels": labels}


def plot_histogram(H: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if np.any(labels == 0): plt.hist(H[labels == 0], bins=50, density=True, alpha=0.55, label="negative")
    if np.any(labels == 1): plt.hist(H[labels == 1], bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("H = -1 - cosine; lower = stronger binding")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend(); plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()


# ============================================================
# Config
# ============================================================

@dataclass
class RunConfig:
    project: str = "/home/natasha/multimodal_model"
    embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_swapped_tulip_decoys"
    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/raw_boltz_local_neighbour_tulip_strict"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/raw_boltz_local_neighbour_tulip_strict"
    run_tag: str = "plain_vicreg_raw_boltz_local_neighbour_tulip_strict"

    train_manifest_csv: str = "/home/natasha/multimodal_model/manifests/train_manifest.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_df_clean_pos_tulip_decoys_strict.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_df_clean_pos_tulip_decoys_strict.csv"
    immrep_test_root: str = "/home/natasha/multimodal_model/models/embeddings/immrep_test_set"
    immrep_csv: str = "/home/natasha/multimodal_model/data/test/immrep_test_set_pair_id.csv"

    boltz_train_root: str = "/home/natasha/multimodal_model/outputs/train"
    boltz_chain_order: str = "tcra,tcrb,pep,hla"
    raw_z_representation: str = "z_tcr_pep"
    local_group_col: str = "peptide_group"
    raw_z_feature_cache_csv: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/raw_boltz_local_neighbour_tulip_strict/raw_z_train_features.csv"
    raw_z_feature_cache_npz: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/raw_boltz_local_neighbour_tulip_strict/raw_z_train_features.npz"
    force_rebuild_raw_z_cache: bool = False
    max_raw_z_pairs: int = 0
    raw_z_pca_dim: int = 64
    raw_z_progress_every: int = 500
    verbose_raw_z_errors: bool = False

    use_boltz_local: bool = True
    boltz_k: int = 10
    boltz_min_group_size: int = 3
    lambda_boltz_rel: float = 0.05
    lambda_boltz_attr: float = 0.05
    boltz_teacher_temp: float = 0.2

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


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--embed-root", default=RunConfig.embed_root)
    p.add_argument("--out-dir", default=RunConfig.out_dir)
    p.add_argument("--fig-dir", default=RunConfig.fig_dir)
    p.add_argument("--run-tag", default=RunConfig.run_tag)
    p.add_argument("--train-manifest-csv", default=RunConfig.train_manifest_csv)
    p.add_argument("--val-csv", default=RunConfig.val_csv)
    p.add_argument("--test-csv", default=RunConfig.test_csv)
    p.add_argument("--immrep-test-root", default=RunConfig.immrep_test_root)
    p.add_argument("--immrep-csv", default=RunConfig.immrep_csv)
    p.add_argument("--boltz-train-root", default=RunConfig.boltz_train_root)
    p.add_argument("--boltz-chain-order", default=RunConfig.boltz_chain_order)
    p.add_argument("--raw-z-representation", default=RunConfig.raw_z_representation, choices=["z_tcr_pep", "z_tcr_hla", "z_pep_hla", "z_tcr_pmhc", "concat_all_interfaces"])
    p.add_argument("--local-group-col", default=RunConfig.local_group_col)
    p.add_argument("--force-rebuild-raw-z-cache", action="store_true")
    p.add_argument("--max-raw-z-pairs", type=int, default=RunConfig.max_raw_z_pairs)
    p.add_argument("--raw-z-pca-dim", type=int, default=RunConfig.raw_z_pca_dim)
    p.add_argument("--use-boltz-local", action=argparse.BooleanOptionalAction, default=RunConfig.use_boltz_local)
    p.add_argument("--boltz-k", type=int, default=RunConfig.boltz_k)
    p.add_argument("--lambda-boltz-rel", type=float, default=RunConfig.lambda_boltz_rel)
    p.add_argument("--lambda-boltz-attr", type=float, default=RunConfig.lambda_boltz_attr)
    p.add_argument("--batch-size", type=int, default=RunConfig.batch_size)
    p.add_argument("--num-workers", type=int, default=RunConfig.num_workers)
    p.add_argument("--epochs", type=int, default=RunConfig.epochs)
    p.add_argument("--patience", type=int, default=RunConfig.patience)
    p.add_argument("--min-epochs", type=int, default=RunConfig.min_epochs)
    p.add_argument("--lr", type=float, default=RunConfig.lr)
    p.add_argument("--alpha", type=float, default=RunConfig.alpha)
    p.add_argument("--beta", type=float, default=RunConfig.beta)
    p.add_argument("--delta", type=float, default=RunConfig.delta)
    p.add_argument("--dropout", type=float, default=RunConfig.dropout)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    args = p.parse_args()
    cfg = RunConfig(**{**asdict(RunConfig()), **vars(args)})
    cfg.raw_z_feature_cache_csv = str(Path(cfg.out_dir) / f"{cfg.run_tag}__raw_z_{cfg.raw_z_representation}_{cfg.local_group_col}_features.csv")
    cfg.raw_z_feature_cache_npz = str(Path(cfg.out_dir) / f"{cfg.run_tag}__raw_z_{cfg.raw_z_representation}_{cfg.local_group_col}_features.npz")
    return cfg


def infer_shapes(ds: Dataset) -> Tuple[int, int, int, int]:
    sample = ds[0]
    return sample["emb_T"].shape[-1], sample["emb_T"].shape[0], sample["emb_P"].shape[0], sample["emb_H"].shape[0]


def dataset_root_has_shards(path: str) -> bool:
    p = Path(path)
    return p.exists() and any(p.glob("shard_*.pt"))


def resolve_shard_dir(root_or_split: str, preferred_split: Optional[str] = None) -> Optional[Path]:
    """Return the directory containing shard_*.pt files.

    Allows either:
      /root/with/shard_*.pt
    or:
      /root/train, /root/val, /root/test

    This is mainly to keep IMMREP robust, because the user-facing path may be
    /models/embeddings/immrep_test_set while the actual shards may sit under
    /models/embeddings/immrep_test_set/test.
    """
    p = Path(root_or_split)
    if dataset_root_has_shards(str(p)):
        return p
    if preferred_split is not None and dataset_root_has_shards(str(p / preferred_split)):
        return p / preferred_split
    return None


# ============================================================
# Main
# ============================================================

def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = Path(cfg.out_dir); fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{cfg.run_tag}__config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60, flush=True)
    print("Plain VICReg + raw-Boltz local-neighbour guidance", flush=True)
    print("SCRIPT VERSION: tulip_decoys_strict_csvs__swapped_decoys_embeddings__immrep_full_test_root", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Embed root: {cfg.embed_root}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Raw z rep: {cfg.raw_z_representation} | group: {cfg.local_group_col} | chain_order: {cfg.boltz_chain_order}", flush=True)
    print(f"Boltz local: {cfg.use_boltz_local} | lambda_rel={cfg.lambda_boltz_rel} | lambda_attr={cfg.lambda_boltz_attr}", flush=True)
    print("=" * 60, flush=True)

    train_meta = pd.read_csv(cfg.train_manifest_csv)
    if cfg.local_group_col in train_meta.columns:
        train_pair_to_group = dict(zip(train_meta["pair_id"].astype(str), train_meta[cfg.local_group_col].astype(str)))
    elif "Peptide" in train_meta.columns:
        train_pair_to_group = dict(zip(train_meta["pair_id"].astype(str), train_meta["Peptide"].astype(str)))
    else:
        train_pair_to_group = {}

    val_pair_to_pep = load_pair_to_col(cfg.val_csv, ["Peptide", "peptide", "pep_seq", "peptide_seq"], required=True)
    test_pair_to_pep = load_pair_to_col(cfg.test_csv, ["Peptide", "peptide", "pep_seq", "peptide_seq"], required=True)
    immrep_pair_to_pep = load_pair_to_col(cfg.immrep_csv, ["Peptide", "peptide", "pep_seq", "peptide_seq"], required=False)

    train_dir = resolve_shard_dir(cfg.embed_root, "train")
    val_dir = resolve_shard_dir(str(Path(cfg.embed_root) / "val"), None) or resolve_shard_dir(cfg.embed_root, "val")
    test_dir = resolve_shard_dir(str(Path(cfg.embed_root) / "test"), None) or resolve_shard_dir(cfg.embed_root, "test")
    immrep_dir = resolve_shard_dir(cfg.immrep_test_root, "test")

    if train_dir is None or val_dir is None or test_dir is None:
        raise FileNotFoundError(
            f"Could not resolve train/val/test shard directories under embed_root={cfg.embed_root}. "
            f"Resolved train={train_dir}, val={val_dir}, test={test_dir}"
        )

    print(f"Resolved train shards: {train_dir}", flush=True)
    print(f"Resolved val shards:   {val_dir}", flush=True)
    print(f"Resolved test shards:  {test_dir}", flush=True)
    print(f"Resolved IMMREP shards: {immrep_dir if immrep_dir is not None else 'NONE'}", flush=True)

    train_ds = FlatShardedDataset(train_dir, pair_to_group=train_pair_to_group)
    val_ds = FlatShardedDataset(val_dir)
    test_ds = FlatShardedDataset(test_dir)
    immrep_ds = FlatShardedDataset(immrep_dir) if immrep_dir is not None else None

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=flat_collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=flat_collate, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=flat_collate, pin_memory=torch.cuda.is_available())
    immrep_loader = DataLoader(immrep_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=flat_collate, pin_memory=torch.cuda.is_available()) if immrep_ds is not None else None

    D, L_T, L_P, L_H = infer_shapes(train_ds)
    print(f"Detected shapes | D={D} | L_T={L_T} | L_P={L_P} | L_H={L_H}", flush=True)
    tcr = ESMProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_T, cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(D, cfg.rL, cfg.rD, cfg.d, L_P, L_H, cfg.R_PH, cfg.dropout).to(device)
    opt = torch.optim.AdamW(list(tcr.parameters()) + list(pmhc.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    guidance = None
    if cfg.use_boltz_local and (cfg.lambda_boltz_rel > 0 or cfg.lambda_boltz_attr > 0):
        raw_meta, raw_X = build_or_load_raw_z_feature_cache(cfg)
        guidance = RawBoltzLocalNeighbourGuidance(raw_meta, raw_X, k=cfg.boltz_k, min_group_size=cfg.boltz_min_group_size)

    best = {"metric": -np.inf, "epoch": None, "state": None, "bad_epochs": 0}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tcr.train(); pmhc.train()
        running = {k: 0.0 for k in ["loss", "base_loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "boltz_local", "boltz_local_rel", "boltz_local_attr", "n_local_pairs", "n_knn_pairs"]}
        n = 0
        for batch in train_loader:
            zT = tcr(batch["emb_T"].to(device), batch["mask_T"].to(device))
            zPH = pmhc(batch["emb_P"].to(device), batch["mask_P"].to(device), batch["emb_H"].to(device), batch["mask_H"].to(device))
            base_loss, parts = plain_vicreg_loss(zT, zPH, cfg.alpha, cfg.beta, cfg.delta, cfg.gamma_var, cfg.eps_norm, cfg.eps_var, return_parts=True)
            boltz_loss = torch.tensor(0.0, device=device)
            aux = {"boltz_local_rel": 0.0, "boltz_local_attr": 0.0, "n_local_pairs": 0.0, "n_knn_pairs": 0.0}
            if guidance is not None:
                has, teacher, same_group, knn_edge = guidance.tensors_for_batch(batch["pair_id"], batch["binding_flag"], batch["group"], device)
                pair_rep = 0.5 * (zT + zPH)
                boltz_loss, aux = local_neighbour_loss(pair_rep, teacher, has, same_group, knn_edge, cfg.lambda_boltz_rel, cfg.lambda_boltz_attr, cfg.boltz_teacher_temp)
            loss = base_loss + boltz_loss
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            running["loss"] += float(loss.detach().cpu())
            running["base_loss"] += parts["L_total"]
            running["boltz_local"] += float(boltz_loss.detach().cpu())
            for k in ["L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov"]:
                running[k] += parts[k]
            for k, v in aux.items():
                running[k] += float(v)
            n += 1
        sched.step()

        val_eval = evaluate(val_loader, tcr, pmhc, device, cfg, val_pair_to_pep, "val")
        row = {"epoch": epoch, **{f"train_{k}": v / max(1, n) for k, v in running.items()}, **val_eval["metrics"]}
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / f"{cfg.run_tag}__history.csv", index=False)

        metric = row["val_model_peptide_weighted_auroc"]
        improved = (not np.isnan(metric)) and metric > best["metric"] + 1e-4
        if improved:
            best = {"metric": metric, "epoch": epoch, "state": {"tcr": copy.deepcopy(tcr.state_dict()), "pmhc": copy.deepcopy(pmhc.state_dict())}, "bad_epochs": 0}
        else:
            best["bad_epochs"] += 1

        print(
            f"Epoch {epoch}/{cfg.epochs} | train_loss={row['train_loss']:.4f} | base={row['train_base_loss']:.4f} | "
            f"boltz={row['train_boltz_local']:.4f} rel={row['train_boltz_local_rel']:.4f} attr={row['train_boltz_local_attr']:.4f} | "
            f"local_pairs/b={row['train_n_local_pairs']:.1f} knn_pairs/b={row['train_n_knn_pairs']:.1f} | "
            f"val_global={row['val_model_global_auroc']:.4f} raw={row['val_raw_esm_global_auroc']:.4f} delta={row['val_delta_global_auroc']:.4f} | "
            f"val_pep_w={row['val_model_peptide_weighted_auroc']:.4f} raw_pep_w={row['val_raw_esm_peptide_weighted_auroc']:.4f} | "
            f"AUPRC={row['val_model_auprc']:.4f} raw_AUPRC={row['val_raw_esm_auprc']:.4f} | "
            f"H_std={row['val_H_std']:.4f} best_epoch={best['epoch']} bad={best['bad_epochs']}",
            flush=True,
        )
        if epoch >= cfg.min_epochs and best["bad_epochs"] >= cfg.patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best["state"] is None:
        raise RuntimeError("No best checkpoint selected")
    tcr.load_state_dict(best["state"]["tcr"]); pmhc.load_state_dict(best["state"]["pmhc"])
    val_eval = evaluate(val_loader, tcr, pmhc, device, cfg, val_pair_to_pep, "val")
    test_eval = evaluate(test_loader, tcr, pmhc, device, cfg, test_pair_to_pep, "test")
    immrep_eval = evaluate(immrep_loader, tcr, pmhc, device, cfg, immrep_pair_to_pep, "immrep") if immrep_loader is not None else None

    stem = f"{cfg.run_tag}__seed{cfg.seed}__lr{cfg.lr}__rel{cfg.lambda_boltz_rel}__attr{cfg.lambda_boltz_attr}__rep{cfg.raw_z_representation}__grp{cfg.local_group_col}"
    checkpoint_path = out_dir / f"{stem}__best.pt"
    torch.save({"config": asdict(cfg), "tcr_state_dict": tcr.state_dict(), "pmhc_state_dict": pmhc.state_dict(), "best_epoch": best["epoch"], "val_metrics": val_eval["metrics"], "test_metrics": test_eval["metrics"], "immrep_metrics": immrep_eval["metrics"] if immrep_eval is not None else None}, checkpoint_path)

    val_eval["predictions"].to_csv(out_dir / f"{stem}__val_predictions.csv", index=False)
    test_eval["predictions"].to_csv(out_dir / f"{stem}__test_predictions.csv", index=False)
    val_eval["model_peptide_table"].to_csv(out_dir / f"{stem}__val_model_per_peptide.csv", index=False)
    test_eval["model_peptide_table"].to_csv(out_dir / f"{stem}__test_model_per_peptide.csv", index=False)
    if immrep_eval is not None:
        immrep_eval["predictions"].to_csv(out_dir / f"{stem}__immrep_predictions.csv", index=False)
        immrep_eval["model_peptide_table"].to_csv(out_dir / f"{stem}__immrep_model_per_peptide.csv", index=False)

    plot_histogram(val_eval["model_H"], val_eval["labels"], "Validation model H", fig_dir / f"{stem}__val_model_H_hist.png")
    plot_histogram(test_eval["model_H"], test_eval["labels"], "Test model H", fig_dir / f"{stem}__test_model_H_hist.png")
    if immrep_eval is not None:
        plot_histogram(immrep_eval["model_H"], immrep_eval["labels"], "IMMREP model H", fig_dir / f"{stem}__immrep_model_H_hist.png")

    val_thr = val_eval["metrics"]["val_threshold"]
    summary = {
        "config": asdict(cfg),
        "best_epoch_by_val_model_peptide_weighted_auroc": best["epoch"],
        "val_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "immrep_metrics": immrep_eval["metrics"] if immrep_eval is not None else None,
        "test_at_val_threshold": threshold_metrics(test_eval["predictions"]["model_score"].to_numpy(), test_eval["predictions"]["label"].to_numpy(), val_thr, "test_at_val_threshold"),
        "immrep_at_val_threshold": threshold_metrics(immrep_eval["predictions"]["model_score"].to_numpy(), immrep_eval["predictions"]["label"].to_numpy(), val_thr, "immrep_at_val_threshold") if immrep_eval is not None else None,
        "paths": {"history": str(out_dir / f"{cfg.run_tag}__history.csv"), "checkpoint": str(checkpoint_path), "out_dir": str(out_dir), "fig_dir": str(fig_dir)},
    }
    with open(out_dir / f"{stem}__summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60, flush=True)
    print("DONE", flush=True)
    print(f"Best epoch: {best['epoch']}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"Summary: {out_dir / f'{stem}__summary.json'}", flush=True)
    print("Final validation metrics:", json.dumps(val_eval["metrics"], indent=2), flush=True)
    print("Final test metrics:", json.dumps(test_eval["metrics"], indent=2), flush=True)
    if immrep_eval is not None:
        print("Final IMMREP metrics:", json.dumps(immrep_eval["metrics"], indent=2), flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
