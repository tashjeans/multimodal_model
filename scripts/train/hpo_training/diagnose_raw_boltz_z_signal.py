#!/usr/bin/env python3
"""
Diagnose whether raw Boltz z embeddings contain useful structural signal and whether
that signal is sequence-learnable from ESM embeddings.

Intended location:
  /home/natasha/multimodal_model/scripts/train/hpo_training/diagnose_raw_boltz_z_signal.py

Run from:
  cd /home/natasha/multimodal_model/scripts/train/hpo_training
  python diagnose_raw_boltz_z_signal.py 2>&1 | tee diagnose_raw_boltz_z_signal.log

This script does not train the TCR-pMHC model. It is a diagnostic/oracle audit.

Main outputs:
  /home/natasha/multimodal_model/models/checkpoints/raw_boltz_z_signal_diagnostics/

What it tests:
  1. Coverage: which pair_ids have raw Boltz embeddings_pair_*.npz with key 'z'.
  2. Raw z interface representations: TCR-peptide, alpha-peptide, beta-peptide,
     peptide-HLA, TCR-HLA, and optionally global mean.
  3. Biological structure: same-peptide/same-HLA examples closer than random?
     Compared against sequence ESM pooled representations where available.
  4. Sequence-learnability: can sequence-only ESM features predict raw Boltz-z
     interface representations on held-out train and val/test positives?

Assumed chain order by default:
  A = TCR alpha, B = TCR beta, C = peptide, D = HLA
  i.e. --boltz-chain-order tcra,tcrb,pep,hla

The script uses the lengths in the manifest to slice the z tensor directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
except Exception:
    torch = None

try:
    from scipy.stats import mannwhitneyu, spearmanr
except Exception:
    mannwhitneyu = None
    spearmanr = None

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Defaults
# ============================================================
PROJECT = Path("/home/natasha/multimodal_model")
EMBED_ROOT = PROJECT / "models/embeddings/no_boltz_train_clean_immrep_A"
CHECKPOINTS_DIR = PROJECT / "models/checkpoints"

TRAIN_CSV = PROJECT / "data/train/train_df_clean.csv"
VAL_CSV = PROJECT / "data/val/val_df_clean_pos_neg_A.csv"
TEST_CSV = PROJECT / "data/test/test_df_clean_pos_neg_A.csv"

TRAIN_MANIFEST = PROJECT / "manifests/train_manifest.csv"
VAL_MANIFEST = PROJECT / "manifests/val_manifest_A.csv"
TEST_MANIFEST = PROJECT / "manifests/test_manifest_A.csv"

BOLTZ_ROOTS = {
    "train": PROJECT / "outputs/train",
    "val": PROJECT / "outputs/val",
    "test": PROJECT / "outputs/test",
}

DEFAULT_OUTDIR = CHECKPOINTS_DIR / "raw_boltz_z_signal_diagnostics"


# ============================================================
# Logging
# ============================================================
def setup_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("raw_boltz_z_diag")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(str(log_file), mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


# ============================================================
# Utility functions
# ============================================================
def canonical_pair_id(x: Any) -> str:
    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        return s
    m = re.search(r"pair[_-]?(\d+)$", s)
    if m:
        return f"pair_{int(m.group(1))}"
    # keep IMMREP ids etc unchanged
    return s


def pair_aliases(pid: Any) -> List[str]:
    s = str(pid).strip()
    aliases = {s, canonical_pair_id(s)}
    m = re.search(r"(\d+)$", s)
    if m:
        n = int(m.group(1))
        aliases.update({
            f"pair_{n}", f"pair_{n:03d}", f"pair_{n:04d}", f"pair_{n:05d}", f"pair_{n:06d}",
            str(n), f"{n:03d}", f"{n:04d}", f"{n:05d}", f"{n:06d}",
        })
    return sorted(aliases)


def infer_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    # looser contains matching
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def cosine_distance_matrix_sample(X: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sims = np.sum(Xn[pairs[:, 0]] * Xn[pairs[:, 1]], axis=1)
    return 1.0 - sims


def sample_same_and_random_pairs(groups: np.ndarray, max_pairs: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Return same-group pairs and random pairs as arrays of shape (n, 2)."""
    n = len(groups)
    by_group: Dict[Any, np.ndarray] = {}
    for g in pd.Series(groups).dropna().unique():
        idx = np.where(groups == g)[0]
        if len(idx) >= 2:
            by_group[g] = idx

    same_pairs = []
    group_keys = list(by_group.keys())
    if group_keys:
        per_group = max(1, max_pairs // max(1, len(group_keys)))
        for g in group_keys:
            idx = by_group[g]
            k = min(per_group, len(idx) * (len(idx) - 1) // 2)
            for _ in range(k):
                a, b = rng.choice(idx, size=2, replace=False)
                same_pairs.append((int(a), int(b)))
                if len(same_pairs) >= max_pairs:
                    break
            if len(same_pairs) >= max_pairs:
                break

    rand_pairs = []
    tries = 0
    while len(rand_pairs) < min(max_pairs, max(1, len(same_pairs))) and tries < max_pairs * 20:
        a, b = rng.integers(0, n, size=2)
        tries += 1
        if a == b:
            continue
        rand_pairs.append((int(a), int(b)))

    if len(same_pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(same_pairs, dtype=np.int64), np.asarray(rand_pairs, dtype=np.int64)


def biological_structure_test(
    X: np.ndarray,
    meta: pd.DataFrame,
    group_col: str,
    rep_name: str,
    split: str,
    max_pairs: int,
    seed: int,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    ok = meta[group_col].notna().to_numpy()
    X2 = X[ok]
    g = meta.loc[ok, group_col].astype(str).to_numpy()
    if len(X2) < 5 or len(np.unique(g)) < 2:
        return {
            "split": split, "representation": rep_name, "group_col": group_col,
            "n": int(len(X2)), "n_groups": int(len(np.unique(g))),
            "n_same_pairs": 0, "n_random_pairs": 0,
            "same_mean_dist": np.nan, "random_mean_dist": np.nan,
            "effect_random_minus_same": np.nan, "mw_pvalue": np.nan,
        }
    same_pairs, rand_pairs = sample_same_and_random_pairs(g, max_pairs=max_pairs, rng=rng)
    if len(same_pairs) == 0 or len(rand_pairs) == 0:
        return {
            "split": split, "representation": rep_name, "group_col": group_col,
            "n": int(len(X2)), "n_groups": int(len(np.unique(g))),
            "n_same_pairs": int(len(same_pairs)), "n_random_pairs": int(len(rand_pairs)),
            "same_mean_dist": np.nan, "random_mean_dist": np.nan,
            "effect_random_minus_same": np.nan, "mw_pvalue": np.nan,
        }
    same_d = cosine_distance_matrix_sample(X2, same_pairs)
    rand_d = cosine_distance_matrix_sample(X2, rand_pairs)
    p = np.nan
    if mannwhitneyu is not None:
        try:
            p = float(mannwhitneyu(same_d, rand_d, alternative="less").pvalue)
        except Exception:
            p = np.nan
    return {
        "split": split, "representation": rep_name, "group_col": group_col,
        "n": int(len(X2)), "n_groups": int(len(np.unique(g))),
        "n_same_pairs": int(len(same_d)), "n_random_pairs": int(len(rand_d)),
        "same_mean_dist": float(np.mean(same_d)),
        "same_median_dist": float(np.median(same_d)),
        "random_mean_dist": float(np.mean(rand_d)),
        "random_median_dist": float(np.median(rand_d)),
        "effect_random_minus_same": float(np.mean(rand_d) - np.mean(same_d)),
        "mw_pvalue": p,
    }


def effective_rank_from_svals(s: np.ndarray) -> float:
    s = np.asarray(s, dtype=np.float64)
    if len(s) == 0 or np.sum(s) <= 0:
        return np.nan
    p = s / np.sum(s)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def variation_summary(X: np.ndarray, split: str, rep_name: str, max_pca_components: int = 64) -> Dict[str, Any]:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    col_std = np.nanstd(X, axis=0)
    out = {
        "split": split,
        "representation": rep_name,
        "n": int(n),
        "dim": int(d),
        "mean_abs_value": float(np.nanmean(np.abs(X))),
        "mean_col_std": float(np.nanmean(col_std)),
        "median_col_std": float(np.nanmedian(col_std)),
        "min_col_std": float(np.nanmin(col_std)),
        "max_col_std": float(np.nanmax(col_std)),
        "near_constant_dims_std_lt_1e-6": int(np.sum(col_std < 1e-6)),
    }
    if n >= 5 and d >= 2:
        k = min(max_pca_components, n - 1, d)
        try:
            Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
            pca = PCA(n_components=k, svd_solver="randomized", random_state=31)
            pca.fit(Xs)
            evr = pca.explained_variance_ratio_
            cs = np.cumsum(evr)
            out.update({
                "pca_k": int(k),
                "effective_rank_singular_values": effective_rank_from_svals(pca.singular_values_),
                "pc1_var_ratio": float(evr[0]) if len(evr) else np.nan,
                "pc2_var_ratio": float(evr[1]) if len(evr) > 1 else np.nan,
                "n_pcs_50pct": int(np.searchsorted(cs, 0.50) + 1),
                "n_pcs_80pct": int(np.searchsorted(cs, 0.80) + 1),
                "n_pcs_90pct": int(np.searchsorted(cs, 0.90) + 1),
                "n_pcs_95pct": int(np.searchsorted(cs, 0.95) + 1),
            })
        except Exception as e:
            out["pca_error"] = str(e)
    return out


# ============================================================
# Data loading / metadata
# ============================================================
def load_csv_metadata(csv_path: Path, manifest_path: Optional[Path], split: str, logger: logging.Logger) -> pd.DataFrame:
    parts = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "pair_id" not in df.columns:
            df["pair_id"] = [f"{split}_{i}" for i in range(len(df))]
        df["pair_id"] = df["pair_id"].map(canonical_pair_id)
        parts.append(df)
    else:
        logger.warning(f"{split}: CSV not found: {csv_path}")

    if manifest_path is not None and manifest_path.exists():
        man = pd.read_csv(manifest_path)
        if "pair_id" in man.columns:
            man["pair_id"] = man["pair_id"].map(canonical_pair_id)
            if parts:
                df = parts[0].merge(man, on="pair_id", how="left", suffixes=("", "_manifest"))
                parts = [df]
            else:
                parts = [man]
        else:
            logger.warning(f"{split}: manifest has no pair_id column: {manifest_path}")
    else:
        logger.warning(f"{split}: manifest not found: {manifest_path}")

    if not parts:
        return pd.DataFrame(columns=["pair_id"])
    df = parts[0]

    # Standardise length columns.
    col_map = {
        "tcra_len": ["tcra_len", "tcr_a_len", "TRA_len", "alpha_len", "tcr_alpha_len"],
        "tcrb_len": ["tcrb_len", "tcr_b_len", "TRB_len", "beta_len", "tcr_beta_len"],
        "pep_len": ["pep_len", "peptide_len", "Peptide_len", "epitope_len"],
        "hla_len": ["hla_len", "mhc_len", "MHC_len", "HLA_len"],
    }
    for std, cands in col_map.items():
        if std not in df.columns:
            c = infer_col(df, cands)
            if c is not None:
                df[std] = df[c]
        df[std] = df[std].map(lambda x: safe_int(x, 0)) if std in df.columns else 0

    # Standardise group columns.
    pep_col = infer_col(df, ["Peptide", "peptide", "pep", "epitope", "peptide_seq", "pep_seq"])
    hla_col = infer_col(df, ["HLA", "hla", "mhc", "MHC", "hla_seq", "mhc_seq"])
    if pep_col is not None and pep_col != "peptide_group":
        df["peptide_group"] = df[pep_col].astype(str)
    if hla_col is not None and hla_col != "hla_group":
        df["hla_group"] = df[hla_col].astype(str)

    if "binding_flag" not in df.columns:
        bcol = infer_col(df, ["binding_flag", "label", "target", "binder", "y"])
        if bcol:
            df["binding_flag"] = df[bcol]
    if "binding_flag" in df.columns:
        df["binding_flag"] = df["binding_flag"].map(lambda x: safe_int(x, 1))

    return df


# ============================================================
# Boltz z extraction
# ============================================================
def index_boltz_embedding_files(root: Path, logger: logging.Logger) -> Dict[str, Path]:
    logger.info(f"Indexing Boltz raw embedding files under: {root}")
    index: Dict[str, Path] = {}
    if not root.exists():
        logger.warning(f"Boltz root does not exist: {root}")
        return index
    n_files = 0
    for p in root.rglob("embeddings_pair_*.npz"):
        n_files += 1
        # Prefer pair id in parent predictions/pair_x, but filename also works.
        candidates = [p.parent.name, p.stem.replace("embeddings_", "")]
        for c in candidates:
            for a in pair_aliases(c):
                index.setdefault(a, p)
    logger.info(f"Indexed {n_files} embeddings npz files; alias entries={len(index)}")
    return index


@dataclass
class ChainSlices:
    tcra: slice
    tcrb: slice
    pep: slice
    hla: slice
    tcr: np.ndarray
    pmhc: np.ndarray
    total_len: int


def make_chain_slices(row: pd.Series, chain_order: List[str], z_len: int, allow_offset: bool = True) -> Optional[ChainSlices]:
    lengths = {
        "tcra": safe_int(row.get("tcra_len", 0)),
        "tcrb": safe_int(row.get("tcrb_len", 0)),
        "pep": safe_int(row.get("pep_len", 0)),
        "hla": safe_int(row.get("hla_len", 0)),
    }
    if any(lengths[k] <= 0 for k in ["pep", "hla"]):
        return None
    total = sum(lengths.get(k, 0) for k in chain_order)
    if total <= 0 or z_len < total:
        return None

    # Boltz usually has residue-only L == total. If there are extra tokens, we
    # default to offset 0 because your inspected CIF scheme starts at A/B/C/D residues.
    offset = 0
    if allow_offset and z_len != total:
        offset = 0

    starts: Dict[str, int] = {}
    pos = offset
    for k in chain_order:
        starts[k] = pos
        pos += lengths[k]

    slices = {k: slice(starts[k], starts[k] + lengths[k]) for k in chain_order}
    tcra_idx = np.arange(slices["tcra"].start, slices["tcra"].stop) if "tcra" in slices else np.array([], dtype=int)
    tcrb_idx = np.arange(slices["tcrb"].start, slices["tcrb"].stop) if "tcrb" in slices else np.array([], dtype=int)
    pep_idx = np.arange(slices["pep"].start, slices["pep"].stop)
    hla_idx = np.arange(slices["hla"].start, slices["hla"].stop)
    tcr_idx = np.concatenate([tcra_idx, tcrb_idx])
    pmhc_idx = np.concatenate([pep_idx, hla_idx])

    return ChainSlices(
        tcra=slices.get("tcra", slice(0, 0)),
        tcrb=slices.get("tcrb", slice(0, 0)),
        pep=slices["pep"],
        hla=slices["hla"],
        tcr=tcr_idx,
        pmhc=pmhc_idx,
        total_len=total,
    )


def mean_block(z: np.ndarray, rows: np.ndarray | slice, cols: np.ndarray | slice, symmetric: bool = True) -> np.ndarray:
    a = z[rows, :, :][:, cols, :].mean(axis=(0, 1))
    if symmetric:
        b = z[cols, :, :][:, rows, :].mean(axis=(0, 1))
        return ((a + b) / 2.0).astype(np.float32)
    return a.astype(np.float32)


def extract_z_representations(z_path: Path, row: pd.Series, chain_order: List[str], include_global: bool) -> Tuple[Optional[Dict[str, np.ndarray]], Dict[str, Any]]:
    info: Dict[str, Any] = {"z_path": str(z_path), "status": "ok"}
    try:
        data = np.load(z_path)
        if "z" not in data.files:
            info["status"] = "no_z_key"
            info["keys"] = ",".join(data.files)
            return None, info
        z = data["z"]
        if z.ndim == 4:
            z = z[0]
        if z.ndim != 3:
            info["status"] = f"bad_z_ndim_{z.ndim}"
            info["z_shape"] = str(z.shape)
            return None, info
        z = np.asarray(z, dtype=np.float32)
        L, L2, D = z.shape
        info["z_shape"] = str(z.shape)
        if L != L2:
            info["status"] = "z_not_square"
            return None, info
        cs = make_chain_slices(row, chain_order=chain_order, z_len=L)
        if cs is None:
            info["status"] = "bad_chain_lengths_or_z_too_short"
            info["z_L"] = L
            info["manifest_total_len"] = int(sum(safe_int(row.get(k, 0)) for k in ["tcra_len", "tcrb_len", "pep_len", "hla_len"]))
            return None, info

        pep_idx = np.arange(cs.pep.start, cs.pep.stop)
        hla_idx = np.arange(cs.hla.start, cs.hla.stop)
        tcra_idx = np.arange(cs.tcra.start, cs.tcra.stop)
        tcrb_idx = np.arange(cs.tcrb.start, cs.tcrb.stop)

        reps: Dict[str, np.ndarray] = {}
        reps["z_tcr_pep"] = mean_block(z, cs.tcr, pep_idx, symmetric=True)
        reps["z_tcra_pep"] = mean_block(z, tcra_idx, pep_idx, symmetric=True) if len(tcra_idx) else np.zeros(D, dtype=np.float32)
        reps["z_tcrb_pep"] = mean_block(z, tcrb_idx, pep_idx, symmetric=True) if len(tcrb_idx) else np.zeros(D, dtype=np.float32)
        reps["z_pep_hla"] = mean_block(z, pep_idx, hla_idx, symmetric=True)
        reps["z_tcr_hla"] = mean_block(z, cs.tcr, hla_idx, symmetric=True)
        reps["z_tcr_pmhc"] = mean_block(z, cs.tcr, cs.pmhc, symmetric=True)

        if include_global:
            reps["z_global"] = z.mean(axis=(0, 1)).astype(np.float32)

        info.update({
            "z_L": int(L), "z_D": int(D), "manifest_total_len": int(cs.total_len),
            "tcra_len": safe_int(row.get("tcra_len", 0)),
            "tcrb_len": safe_int(row.get("tcrb_len", 0)),
            "pep_len": safe_int(row.get("pep_len", 0)),
            "hla_len": safe_int(row.get("hla_len", 0)),
        })
        return reps, info
    except Exception as e:
        info["status"] = "exception"
        info["error"] = repr(e)
        return None, info


def build_raw_z_feature_cache(
    split: str,
    meta: pd.DataFrame,
    boltz_index: Dict[str, Path],
    outdir: Path,
    chain_order: List[str],
    include_global: bool,
    force: bool,
    max_rows: Optional[int],
    progress_every: int,
    logger: logging.Logger,
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    cache_npz = outdir / f"{split}_raw_boltz_z_representations.npz"
    cache_meta = outdir / f"{split}_raw_boltz_z_metadata.csv"
    if cache_npz.exists() and cache_meta.exists() and not force:
        logger.info(f"{split}: loading cached raw z representations: {cache_npz}")
        data = np.load(cache_npz)
        reps = {k: data[k] for k in data.files if k != "pair_ids"}
        meta_out = pd.read_csv(cache_meta)
        return reps, meta_out

    logger.info(f"{split}: extracting raw z interface representations | rows={len(meta)}")
    records = []
    rep_lists: Dict[str, List[np.ndarray]] = {}
    pair_ids: List[str] = []

    rows = meta.reset_index(drop=True)
    if max_rows is not None and max_rows > 0:
        rows = rows.iloc[:max_rows].copy()
        logger.info(f"{split}: max_rows active: {len(rows)}")

    n_found_path = 0
    n_ok = 0
    for i, row in rows.iterrows():
        pid = canonical_pair_id(row.get("pair_id", f"{split}_{i}"))
        z_path = None
        for a in pair_aliases(pid):
            if a in boltz_index:
                z_path = boltz_index[a]
                break
        if z_path is None:
            records.append({"pair_id": pid, "status": "no_embedding_file"})
            continue
        n_found_path += 1
        reps, info = extract_z_representations(z_path, row, chain_order=chain_order, include_global=include_global)
        info["pair_id"] = pid
        records.append(info)
        if reps is not None:
            pair_ids.append(pid)
            for k, v in reps.items():
                rep_lists.setdefault(k, []).append(v)
            n_ok += 1
        if (i + 1) % progress_every == 0:
            logger.info(f"{split}: scanned {i+1}/{len(rows)} | paths_found={n_found_path} | usable_z={n_ok}")

    meta_status = pd.DataFrame(records)
    usable_meta = meta_status[meta_status["status"].eq("ok")].copy()
    if len(pair_ids) == 0:
        logger.warning(f"{split}: no usable z representations extracted")
        meta_status.to_csv(cache_meta, index=False)
        return {}, meta_status

    reps_arr = {k: np.vstack(v).astype(np.float32) for k, v in rep_lists.items()}
    np.savez_compressed(cache_npz, pair_ids=np.asarray(pair_ids, dtype=object), **reps_arr)

    # Merge original metadata for biological grouping.
    keep_cols = [c for c in ["pair_id", "binding_flag", "peptide_group", "hla_group", "tcra_len", "tcrb_len", "pep_len", "hla_len"] if c in meta.columns]
    original = meta[keep_cols].drop_duplicates("pair_id") if keep_cols else pd.DataFrame({"pair_id": []})
    usable_meta = pd.DataFrame({"pair_id": pair_ids}).merge(original, on="pair_id", how="left")
    usable_meta = usable_meta.merge(meta_status, on="pair_id", how="left", suffixes=("", "_extract"))
    usable_meta.to_csv(cache_meta, index=False)

    # Also save full status including failures.
    meta_status.to_csv(outdir / f"{split}_raw_boltz_z_extraction_status_all_rows.csv", index=False)
    logger.info(f"{split}: extraction complete | usable_z={n_ok} | saved={cache_npz}")
    return reps_arr, usable_meta


# ============================================================
# ESM sequence embedding loading
# ============================================================
def to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    try:
        return np.asarray(x)
    except Exception:
        return None


def pool_embedding_array(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        # Already pooled: (N, D)
        return arr.astype(np.float32)
    if arr.ndim == 3:
        # Token embeddings: (N, L, D)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape[:2] == arr.shape[:2]:
                denom = mask.sum(axis=1, keepdims=True).clip(min=1.0)
                return ((arr * mask[..., None]).sum(axis=1) / denom).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    return None


def extract_pair_ids_from_obj(obj: Any) -> Optional[List[str]]:
    if isinstance(obj, dict):
        for k in ["pair_ids", "pair_id", "ids", "id", "names", "sample_ids"]:
            if k in obj:
                vals = obj[k]
                if torch is not None and isinstance(vals, torch.Tensor):
                    vals = vals.detach().cpu().numpy().tolist()
                if isinstance(vals, np.ndarray):
                    vals = vals.tolist()
                if not isinstance(vals, list):
                    vals = list(vals)
                return [canonical_pair_id(v.decode() if isinstance(v, bytes) else v) for v in vals]
    return None


def candidate_embedding_keys(keys: Iterable[str]) -> List[str]:
    out = []
    bad = ["label", "binding", "flag", "mask", "len", "length", "pair", "id", "target", "y"]
    good = ["tcr", "pmhc", "pep", "hla", "mhc", "emb", "embedding", "esm", "x"]
    for k in keys:
        kl = k.lower()
        if any(b in kl for b in bad):
            continue
        if any(g in kl for g in good):
            out.append(k)
    return out


def load_esm_features_for_split(split: str, embed_root: Path, outdir: Path, force: bool, logger: logging.Logger) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    cache_npz = outdir / f"{split}_sequence_esm_pooled_features.npz"
    cache_meta = outdir / f"{split}_sequence_esm_pooled_metadata.csv"
    if cache_npz.exists() and cache_meta.exists() and not force:
        logger.info(f"{split}: loading cached ESM pooled features: {cache_npz}")
        d = np.load(cache_npz)
        return d["X"].astype(np.float32), pd.read_csv(cache_meta)

    split_dir = embed_root / split
    if not split_dir.exists():
        logger.warning(f"{split}: ESM split directory not found: {split_dir}")
        return None, None

    files = []
    for ext in ["*.pt", "*.pth", "*.npz"]:
        files.extend(sorted(split_dir.rglob(ext)))
    if not files:
        logger.warning(f"{split}: no ESM shard files found under {split_dir}")
        return None, None

    logger.info(f"{split}: loading ESM pooled features from {len(files)} shard files under {split_dir}")
    X_chunks = []
    pid_all: List[str] = []
    inspected = 0

    for fp in files:
        try:
            if fp.suffix in [".pt", ".pth"]:
                if torch is None:
                    continue
                obj = torch.load(fp, map_location="cpu")
            else:
                npz = np.load(fp, allow_pickle=True)
                obj = {k: npz[k] for k in npz.files}
        except Exception as e:
            logger.warning(f"{split}: failed to load shard {fp}: {e}")
            continue

        if isinstance(obj, list):
            # Common project format: each .pt shard is a list of batch dictionaries.
            # Each batch dictionary contains:
            #   emb_T: (B, L_T, D), emb_P: (B, L_P, D), emb_H: (B, L_H, D)
            #   mask_T/mask_P/mask_H: (B, L), pair_id: list[str] length B
            # We mean-pool each modality using its mask and concatenate [T, P, H].
            pids: List[str] = []
            feats: List[np.ndarray] = []
            for r in obj:
                if not isinstance(r, dict):
                    continue
                pid_vals = r.get("pair_id", r.get("pair_ids", r.get("id", None)))
                if pid_vals is None:
                    continue

                # Normalise pair_id field to a list. In this project it is normally length B.
                if torch is not None and isinstance(pid_vals, torch.Tensor):
                    pid_vals = pid_vals.detach().cpu().numpy().tolist()
                if isinstance(pid_vals, np.ndarray):
                    pid_vals = pid_vals.tolist()
                if not isinstance(pid_vals, list):
                    pid_vals = [pid_vals]
                pid_vals = [canonical_pair_id(v.decode() if isinstance(v, bytes) else v) for v in pid_vals]
                n = len(pid_vals)

                arrays = []
                explicit_modalities = [
                    ("emb_T", "mask_T"),
                    ("emb_P", "mask_P"),
                    ("emb_H", "mask_H"),
                    # fallback names used in some variants
                    ("tcr_emb", "tcr_mask"),
                    ("pep_emb", "pep_mask"),
                    ("hla_emb", "hla_mask"),
                    ("pmhc_emb", "pmhc_mask"),
                ]
                used = set()
                for ek, mk in explicit_modalities:
                    if ek not in r or ek in used:
                        continue
                    arr = to_numpy(r.get(ek))
                    mask = to_numpy(r.get(mk)) if mk in r else None
                    if arr is None:
                        continue
                    pooled = pool_embedding_array(arr, mask=mask)
                    if pooled is not None and pooled.shape[0] == n:
                        arrays.append(pooled)
                        used.add(ek)

                # Generic fallback: recognise any batch-level embedding arrays.
                if not arrays:
                    keys = candidate_embedding_keys(r.keys())
                    keys = sorted(keys, key=lambda k: (0 if "emb_t" in k.lower() or "tcr" in k.lower() else 1 if ("emb_p" in k.lower() or "pep" in k.lower()) else 2 if ("emb_h" in k.lower() or "hla" in k.lower() or "mhc" in k.lower()) else 3, k))
                    for k in keys:
                        arr = to_numpy(r.get(k))
                        if arr is None:
                            continue
                        if arr.shape[0] != n:
                            continue
                        mask = None
                        for mk in [f"{k}_mask", k.replace("emb", "mask"), "mask", "attention_mask"]:
                            if mk in r:
                                mask = to_numpy(r[mk])
                                break
                        pooled = pool_embedding_array(arr, mask=mask)
                        if pooled is not None and pooled.shape[0] == n:
                            arrays.append(pooled)
                        if len(arrays) >= 4:
                            break

                if arrays:
                    Xb = np.concatenate(arrays, axis=1).astype(np.float32)
                    for i, pid in enumerate(pid_vals):
                        pids.append(pid)
                        feats.append(Xb[i])

            if feats:
                X_chunks.append(np.vstack(feats).astype(np.float32))
                pid_all.extend(pids)
            continue

        if not isinstance(obj, dict):
            continue

        pids = extract_pair_ids_from_obj(obj)
        if pids is None:
            continue
        n = len(pids)
        if inspected < 5:
            logger.info(f"{split}: shard example {fp.name} keys={list(obj.keys())}")
            inspected += 1

        arrays = []
        # Prefer explicit TCR/pMHC-like arrays; concatenate all recognised modalities.
        keys = candidate_embedding_keys(obj.keys())
        # Make order deterministic and put TCR/pMHC first when present.
        keys = sorted(keys, key=lambda k: (0 if "tcr" in k.lower() else 1 if ("pmhc" in k.lower() or "pep" in k.lower() or "hla" in k.lower() or "mhc" in k.lower()) else 2, k))
        for k in keys:
            arr = to_numpy(obj.get(k))
            if arr is None:
                continue
            if arr.shape[0] != n:
                continue
            mask = None
            # Try matched mask key.
            for mk in [f"{k}_mask", k.replace("emb", "mask"), "mask", "attention_mask"]:
                if mk in obj:
                    mask = to_numpy(obj[mk])
                    break
            pooled = pool_embedding_array(arr, mask=mask)
            if pooled is not None and pooled.shape[0] == n:
                arrays.append(pooled)
        if not arrays:
            continue
        # Avoid extremely wide accidental concatenation: cap to first 4 recognised arrays.
        arrays = arrays[:4]
        X = np.concatenate(arrays, axis=1).astype(np.float32)
        X_chunks.append(X)
        pid_all.extend(pids)

    if not X_chunks:
        logger.warning(f"{split}: could not extract any ESM features. Sequence-learnability will be skipped for this split.")
        return None, None

    X = np.vstack(X_chunks).astype(np.float32)
    meta = pd.DataFrame({"pair_id": pid_all})
    # Deduplicate by pair_id, keeping first.
    meta["_row"] = np.arange(len(meta))
    dedup = meta.drop_duplicates("pair_id", keep="first")
    X = X[dedup["_row"].to_numpy()]
    meta = dedup.drop(columns=["_row"]).reset_index(drop=True)

    np.savez_compressed(cache_npz, pair_ids=meta["pair_id"].to_numpy(dtype=object), X=X)
    meta.to_csv(cache_meta, index=False)
    logger.info(f"{split}: saved ESM pooled features | n={len(meta)} dim={X.shape[1]}")
    return X, meta


# ============================================================
# Diagnostics
# ============================================================
def split_shift_summary(train_X: np.ndarray, other_X: np.ndarray, rep_name: str, other_split: str) -> Dict[str, Any]:
    tr_mean = np.mean(train_X, axis=0)
    ot_mean = np.mean(other_X, axis=0)
    tr_std = np.std(train_X, axis=0) + 1e-8
    ot_std = np.std(other_X, axis=0) + 1e-8
    zmean = (ot_mean - tr_mean) / tr_std
    return {
        "representation": rep_name,
        "split": other_split,
        "train_n": int(len(train_X)),
        "split_n": int(len(other_X)),
        "mean_abs_z_shift_vs_train": float(np.mean(np.abs(zmean))),
        "median_abs_z_shift_vs_train": float(np.median(np.abs(zmean))),
        "l2_z_shift_vs_train": float(np.linalg.norm(zmean) / math.sqrt(len(zmean))),
        "mean_std_ratio_vs_train": float(np.mean(ot_std / tr_std)),
        "median_std_ratio_vs_train": float(np.median(ot_std / tr_std)),
    }


def align_by_pair_id(X: np.ndarray, meta: pd.DataFrame, target_pair_ids: Iterable[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    row = {pid: i for i, pid in enumerate(meta["pair_id"].astype(str).tolist())}
    idx = []
    keep = []
    for pid in target_pair_ids:
        pid = str(pid)
        if pid in row:
            idx.append(row[pid])
            keep.append(pid)
    if not idx:
        return np.zeros((0, X.shape[1]), dtype=np.float32), pd.DataFrame({"pair_id": []})
    return X[np.asarray(idx)], pd.DataFrame({"pair_id": keep})


def sequence_learnability(
    rep_name: str,
    z_train: np.ndarray,
    z_meta_train: pd.DataFrame,
    esm_by_split: Dict[str, Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]],
    z_by_split: Dict[str, Tuple[Dict[str, np.ndarray], pd.DataFrame]],
    outdir: Path,
    n_pca_targets: int,
    seed: int,
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """Predict Boltz representation PCA targets from ESM pooled features."""
    X_train_all, Xmeta_train = esm_by_split.get("train", (None, None))
    if X_train_all is None or Xmeta_train is None:
        logger.warning(f"{rep_name}: no train ESM features; skipping sequence-learnability")
        return []

    # Align train ESM with z rows.
    esm_row = {pid: i for i, pid in enumerate(Xmeta_train["pair_id"].astype(str).tolist())}
    z_pids = z_meta_train["pair_id"].astype(str).tolist()
    ix_esm, ix_z, keep_pids = [], [], []
    for iz, pid in enumerate(z_pids):
        if pid in esm_row:
            ix_esm.append(esm_row[pid])
            ix_z.append(iz)
            keep_pids.append(pid)
    if len(ix_esm) < 100:
        logger.warning(f"{rep_name}: too few train intersections between ESM and Boltz z: {len(ix_esm)}")
        return []

    X = X_train_all[np.asarray(ix_esm)].astype(np.float32)
    Y_raw = z_train[np.asarray(ix_z)].astype(np.float32)

    # PCA target fitted on train only. This tests whether sequence can predict the dominant raw-z axes.
    k = min(n_pca_targets, Y_raw.shape[1], len(Y_raw) - 2)
    y_scaler = StandardScaler(with_mean=True, with_std=True)
    Y_std = y_scaler.fit_transform(Y_raw)
    pca_y = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    Y = pca_y.fit_transform(Y_std).astype(np.float32)

    x_scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = x_scaler.fit_transform(X).astype(np.float32)

    tr_idx, ho_idx = train_test_split(np.arange(len(Xs)), test_size=0.20, random_state=seed)
    alphas = np.logspace(-3, 4, 12)
    model = RidgeCV(alphas=alphas)
    logger.info(f"{rep_name}: fitting RidgeCV sequence->raw-z-PC | n_train={len(tr_idx)} n_holdout={len(ho_idx)} Xdim={Xs.shape[1]} Ypc={k}")
    model.fit(Xs[tr_idx], Y[tr_idx])

    rows = []

    def eval_block(split_name: str, X_eval_raw: np.ndarray, Y_eval_raw: np.ndarray, n_available: int) -> Dict[str, Any]:
        X_eval = x_scaler.transform(X_eval_raw).astype(np.float32)
        Y_eval = pca_y.transform(y_scaler.transform(Y_eval_raw)).astype(np.float32)
        pred = model.predict(X_eval).astype(np.float32)
        cos = np.sum(pred * Y_eval, axis=1) / ((np.linalg.norm(pred, axis=1) + 1e-8) * (np.linalg.norm(Y_eval, axis=1) + 1e-8))
        # Retrieval: nearest predicted target to true target within split.
        predn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        Yn = Y_eval / (np.linalg.norm(Y_eval, axis=1, keepdims=True) + 1e-8)
        top1 = np.nan
        top10 = np.nan
        if len(Yn) <= 5000:
            sims = predn @ Yn.T
            ranks = np.argsort(-sims, axis=1)
            true = np.arange(len(Yn))
            top1 = float(np.mean(ranks[:, :1] == true[:, None]))
            top10 = float(np.mean(np.any(ranks[:, :10] == true[:, None], axis=1)))
        out = {
            "representation": rep_name,
            "eval_split": split_name,
            "n": int(len(X_eval_raw)),
            "n_available_before_intersection": int(n_available),
            "target_pcs": int(k),
            "ridge_alpha": float(model.alpha_) if hasattr(model, "alpha_") else np.nan,
            "target_pc_explained_var_sum": float(np.sum(pca_y.explained_variance_ratio_)),
            "r2_uniform": float(r2_score(Y_eval, pred, multioutput="uniform_average")),
            "r2_variance_weighted": float(r2_score(Y_eval, pred, multioutput="variance_weighted")),
            "mean_cosine_pred_true": float(np.mean(cos)),
            "median_cosine_pred_true": float(np.median(cos)),
            "top1_retrieval": top1,
            "top10_retrieval": top10,
        }
        if spearmanr is not None:
            # Flattened rank correlation is crude but useful as a sanity check.
            try:
                out["flattened_spearman"] = float(spearmanr(Y_eval.ravel(), pred.ravel()).correlation)
            except Exception:
                out["flattened_spearman"] = np.nan
        return out

    rows.append(eval_block("train_holdout", X[ho_idx], Y_raw[ho_idx], len(ho_idx)))

    for split in ["val", "test"]:
        X_split, Xmeta_split = esm_by_split.get(split, (None, None))
        z_reps, z_meta = z_by_split.get(split, ({}, pd.DataFrame()))
        if X_split is None or Xmeta_split is None or rep_name not in z_reps or len(z_meta) == 0:
            continue
        esm_row2 = {pid: i for i, pid in enumerate(Xmeta_split["pair_id"].astype(str).tolist())}
        ix_e, ix_y = [], []
        for iy, pid in enumerate(z_meta["pair_id"].astype(str).tolist()):
            if pid in esm_row2:
                ix_e.append(esm_row2[pid])
                ix_y.append(iy)
        if len(ix_e) < 20:
            logger.warning(f"{rep_name}: {split} too few intersections for learnability: {len(ix_e)}")
            continue
        rows.append(eval_block(split, X_split[np.asarray(ix_e)], z_reps[rep_name][np.asarray(ix_y)], len(z_meta)))

    # Save model diagnostic artefacts for this rep.
    pd.DataFrame(rows).to_csv(outdir / f"sequence_learnability_{rep_name}.csv", index=False)
    return rows


def make_basic_figures(var_df: pd.DataFrame, bio_df: pd.DataFrame, learn_df: pd.DataFrame, figdir: Path, logger: logging.Logger) -> None:
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        if len(bio_df):
            for group in sorted(bio_df["group_col"].dropna().unique()):
                sub = bio_df[bio_df["group_col"].eq(group)].copy()
                sub = sub.dropna(subset=["effect_random_minus_same"])
                if len(sub) == 0:
                    continue
                plt.figure(figsize=(12, max(5, 0.35 * len(sub))))
                labels = sub["split"].astype(str) + " | " + sub["representation"].astype(str)
                plt.barh(labels, sub["effect_random_minus_same"].astype(float))
                plt.axvline(0, linewidth=1)
                plt.xlabel("Random mean cosine distance - same-group mean cosine distance")
                plt.title(f"Biological grouping effect: {group}")
                plt.tight_layout()
                plt.savefig(figdir / f"biological_grouping_effect_{group}.png", dpi=200)
                plt.close()
        if len(learn_df):
            sub = learn_df.copy().dropna(subset=["r2_uniform"])
            if len(sub):
                plt.figure(figsize=(12, max(5, 0.35 * len(sub))))
                labels = sub["eval_split"].astype(str) + " | " + sub["representation"].astype(str)
                plt.barh(labels, sub["r2_uniform"].astype(float))
                plt.axvline(0, linewidth=1)
                plt.xlabel("Uniform R²: sequence ESM -> raw Boltz-z PCs")
                plt.title("Sequence-learnability of raw Boltz z representations")
                plt.tight_layout()
                plt.savefig(figdir / "sequence_learnability_r2_uniform.png", dpi=200)
                plt.close()
    except Exception as e:
        logger.warning(f"Figure generation failed: {e}")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose raw Boltz z signal and sequence-learnability.")
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--embed-root", type=Path, default=EMBED_ROOT)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    p.add_argument("--val-csv", type=Path, default=VAL_CSV)
    p.add_argument("--test-csv", type=Path, default=TEST_CSV)
    p.add_argument("--train-manifest", type=Path, default=TRAIN_MANIFEST)
    p.add_argument("--val-manifest", type=Path, default=VAL_MANIFEST)
    p.add_argument("--test-manifest", type=Path, default=TEST_MANIFEST)
    p.add_argument("--train-boltz-root", type=Path, default=BOLTZ_ROOTS["train"])
    p.add_argument("--val-boltz-root", type=Path, default=BOLTZ_ROOTS["val"])
    p.add_argument("--test-boltz-root", type=Path, default=BOLTZ_ROOTS["test"])
    p.add_argument("--boltz-chain-order", type=str, default="tcra,tcrb,pep,hla")
    p.add_argument("--include-global-z", action="store_true", help="Also compute global z mean over all LxL pairs. Interface reps are computed regardless.")
    p.add_argument("--force-rebuild-z-cache", action="store_true")
    p.add_argument("--force-rebuild-esm-cache", action="store_true")
    p.add_argument("--max-rows-per-split", type=int, default=0, help="0 means all rows. Use e.g. 500 for a dry run.")
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--max-pairs-biostruct", type=int, default=50000)
    p.add_argument("--sequence-target-pcs", type=int, default=32)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--log-file", type=Path, default=Path("diagnose_raw_boltz_z_signal.log"), help="Relative path logs into current working directory by default.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_file)
    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    chain_order = [x.strip().lower() for x in args.boltz_chain_order.split(",") if x.strip()]
    required = {"tcra", "tcrb", "pep", "hla"}
    if set(chain_order) != required:
        raise ValueError(f"--boltz-chain-order must contain exactly {required}; got {chain_order}")

    max_rows = args.max_rows_per_split if args.max_rows_per_split and args.max_rows_per_split > 0 else None

    logger.info("============================================================")
    logger.info("Raw Boltz z signal diagnostic v2-list-shard-fix")
    logger.info(f"Project: {args.project}")
    logger.info(f"Embed root: {args.embed_root}")
    logger.info(f"Outdir: {outdir}")
    logger.info(f"Chain order: {chain_order}")
    logger.info(f"Include global z: {args.include_global_z}")
    logger.info("============================================================")

    # Save run config.
    with open(outdir / "run_config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    # Metadata.
    split_paths = {
        "train": (args.train_csv, args.train_manifest, args.train_boltz_root),
        "val": (args.val_csv, args.val_manifest, args.val_boltz_root),
        "test": (args.test_csv, args.test_manifest, args.test_boltz_root),
    }
    metadata: Dict[str, pd.DataFrame] = {}
    for split, (csvp, manp, _) in split_paths.items():
        metadata[split] = load_csv_metadata(csvp, manp, split, logger)
        logger.info(f"{split}: metadata rows={len(metadata[split])} columns={metadata[split].columns.tolist()}")

    # Boltz indexes and raw z reps.
    z_by_split: Dict[str, Tuple[Dict[str, np.ndarray], pd.DataFrame]] = {}
    coverage_rows = []
    for split, (_, _, root) in split_paths.items():
        idx = index_boltz_embedding_files(root, logger)
        reps, zmeta = build_raw_z_feature_cache(
            split=split,
            meta=metadata[split],
            boltz_index=idx,
            outdir=outdir,
            chain_order=chain_order,
            include_global=args.include_global_z,
            force=args.force_rebuild_z_cache,
            max_rows=max_rows,
            progress_every=args.progress_every,
            logger=logger,
        )
        z_by_split[split] = (reps, zmeta)
        coverage_rows.append({
            "split": split,
            "metadata_rows": int(len(metadata[split])),
            "usable_raw_z_rows": int(len(zmeta[zmeta.get("status", "ok").eq("ok")]) if "status" in zmeta.columns else len(zmeta)),
            "representations": ",".join(sorted(reps.keys())),
        })
    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(outdir / "coverage_summary.csv", index=False)
    logger.info("Coverage summary:\n" + coverage_df.to_string(index=False))

    # Load ESM features.
    esm_by_split: Dict[str, Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]] = {}
    for split in ["train", "val", "test"]:
        Xesm, Mesm = load_esm_features_for_split(split, args.embed_root, outdir, args.force_rebuild_esm_cache, logger)
        esm_by_split[split] = (Xesm, Mesm)

    # Variation, split shift and biological structure.
    var_rows = []
    shift_rows = []
    bio_rows = []

    rep_names = sorted(set().union(*[set(z_by_split[s][0].keys()) for s in z_by_split]))
    logger.info(f"Raw-z representation names: {rep_names}")

    for split, (reps, zmeta) in z_by_split.items():
        for rep_name, X in reps.items():
            var_rows.append(variation_summary(X, split, rep_name))
            for group_col in ["peptide_group", "hla_group"]:
                if group_col in zmeta.columns:
                    bio_rows.append(biological_structure_test(X, zmeta, group_col, rep_name, split, args.max_pairs_biostruct, args.seed))

    # ESM biological structure baseline on same metadata where possible.
    for split, (Xesm, Mesm) in esm_by_split.items():
        if Xesm is None or Mesm is None:
            continue
        # Merge ESM pair ids with split metadata to get biological groups.
        m = Mesm.merge(metadata[split][[c for c in ["pair_id", "peptide_group", "hla_group", "binding_flag"] if c in metadata[split].columns]].drop_duplicates("pair_id"), on="pair_id", how="left")
        for group_col in ["peptide_group", "hla_group"]:
            if group_col in m.columns:
                bio_rows.append(biological_structure_test(Xesm, m, group_col, "esm_pooled", split, args.max_pairs_biostruct, args.seed))
        var_rows.append(variation_summary(Xesm, split, "esm_pooled"))

    # Split shift versus train for each raw-z rep.
    for rep_name in rep_names:
        train_reps = z_by_split["train"][0]
        if rep_name not in train_reps:
            continue
        Xtr = train_reps[rep_name]
        for split in ["val", "test"]:
            reps = z_by_split[split][0]
            if rep_name in reps and len(reps[rep_name]) > 2:
                shift_rows.append(split_shift_summary(Xtr, reps[rep_name], rep_name, split))

    var_df = pd.DataFrame(var_rows)
    shift_df = pd.DataFrame(shift_rows)
    bio_df = pd.DataFrame(bio_rows)
    var_df.to_csv(outdir / "variation_summary_all_representations.csv", index=False)
    shift_df.to_csv(outdir / "split_shift_vs_train.csv", index=False)
    bio_df.to_csv(outdir / "biological_structure_tests_all_representations.csv", index=False)

    logger.info(f"Saved variation summary: {outdir / 'variation_summary_all_representations.csv'}")
    logger.info(f"Saved biological structure tests: {outdir / 'biological_structure_tests_all_representations.csv'}")

    # Sequence learnability for raw-z representations.
    learn_rows = []
    for rep_name in rep_names:
        train_reps, train_zmeta = z_by_split["train"]
        if rep_name not in train_reps:
            continue
        rows = sequence_learnability(
            rep_name=rep_name,
            z_train=train_reps[rep_name],
            z_meta_train=train_zmeta,
            esm_by_split=esm_by_split,
            z_by_split=z_by_split,
            outdir=outdir,
            n_pca_targets=args.sequence_target_pcs,
            seed=args.seed,
            logger=logger,
        )
        learn_rows.extend(rows)

    learn_df = pd.DataFrame(learn_rows)
    learn_df.to_csv(outdir / "sequence_learnability_summary.csv", index=False)
    logger.info(f"Saved sequence learnability summary: {outdir / 'sequence_learnability_summary.csv'}")

    # Figures.
    make_basic_figures(var_df, bio_df, learn_df, figdir, logger)

    # Write a simple text interpretation scaffold.
    with open(outdir / "READ_ME_INTERPRETATION.md", "w") as f:
        f.write("# Raw Boltz z signal diagnostic v2-list-shard-fixs\n\n")
        f.write("Inspect these files first:\n\n")
        f.write("- `coverage_summary.csv`\n")
        f.write("- `variation_summary_all_representations.csv`\n")
        f.write("- `biological_structure_tests_all_representations.csv`\n")
        f.write("- `split_shift_vs_train.csv`\n")
        f.write("- `sequence_learnability_summary.csv`\n\n")
        f.write("Positive signs:\n\n")
        f.write("- Raw-z interface reps have non-trivial variance and are not extremely low-rank.\n")
        f.write("- Same-peptide or same-HLA pairs are closer than random pairs, especially for `z_tcr_pep`, `z_tcra_pep`, or `z_tcrb_pep`.\n")
        f.write("- The raw-z biological grouping effect is comparable to or stronger than `esm_pooled`.\n")
        f.write("- Sequence-learnability generalises: val/test R² and cosine are positive, not just train-holdout.\n\n")
        f.write("Negative signs:\n\n")
        f.write("- Raw-z is much weaker than ESM on biological grouping.\n")
        f.write("- Val/test split shift is large relative to train.\n")
        f.write("- Sequence->raw-z learnability collapses on val/test.\n")

    logger.info("============================================================")
    logger.info("DONE")
    logger.info(f"All outputs saved under: {outdir}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
