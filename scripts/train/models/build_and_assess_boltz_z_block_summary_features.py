#!/usr/bin/env python3
"""Build and assess deterministic Boltz z block-summary features.

Purpose
-------
This script is deliberately diagnostic. It does NOT train a neural compressor over
Boltz z. Instead, it extracts deterministic, biologically structured block-level
statistics from the pairwise z tensor, then assesses which blocks/features carry
binding signal on val/test/IMMREP.

Important data assumption
-------------------------
Only val, test and immrep_test have negatives. The train split is therefore not
used for supervised assessment by default. IMMREP negatives are the closest thing
here to true negatives, so IMMREP should be treated as the external benchmark.

Expected Boltz NPZ structure
----------------------------
Each row in the multiview CSV should contain a repo-relative path in
`boltz_embedding_npz`, for example:

    outputs/val/chunk_000/boltz_results_pair_000/predictions/pair_000/embeddings_pair_000.npz

Each NPZ should contain `z` with shape either:
    (1, L, L, Dz) or (L, L, Dz)

The residue order is assumed to be:
    TCRA | TCRB | peptide | HLA

Outputs
-------
For each split:
    - {split}_z_block_features.csv
    - {split}_z_feature_audit.csv

Assessment outputs:
    - univariate_feature_metrics_by_split.csv
    - univariate_feature_metrics_val_oriented.csv
    - block_group_logreg_valfit_metrics.csv
    - block_group_logreg_coefficients.csv
    - feature_schema.json
    - run_summary.json

The assessment includes AUROC, AUPRC, raw partial AUC@0.1, McClish-standardised
partial AUC@0.1, and per-peptide macro/weighted AUROC/AUC@0.1 metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# -----------------------------------------------------------------------------
# Manifest handling
# -----------------------------------------------------------------------------

def first_existing_col(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalise_manifest(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if "pair_id" not in df.columns:
        raise ValueError(f"{source}: CSV must contain pair_id")
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


def complete_complex_filter(meta: pd.DataFrame) -> pd.Series:
    return (
        (meta["tcra_len"] > 0)
        & (meta["tcrb_len"] > 0)
        & (meta["pep_len"] > 0)
        & (meta["hla_len"] > 0)
    )


def resolve_npz_path(repo_root: Path, row: pd.Series, split_root: Optional[Path] = None) -> Optional[Path]:
    """Resolve the Boltz embedding NPZ path for one manifest row.

    Primary route is the `boltz_embedding_npz` column. Fallback is recursive
    filename search under split_root for embeddings_{pair_id}.npz.
    """
    pid = str(row["pair_id"])
    if "boltz_embedding_npz" in row.index and pd.notna(row["boltz_embedding_npz"]):
        p = Path(str(row["boltz_embedding_npz"]))
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            return p

    if split_root is not None and split_root.exists():
        target = f"embeddings_{pid}.npz"
        hits = list(split_root.rglob(target))
        if hits:
            return hits[0]
    return None


# -----------------------------------------------------------------------------
# z feature extraction
# -----------------------------------------------------------------------------

def load_z(npz_path: Path) -> np.ndarray:
    with np.load(npz_path, allow_pickle=False) as data:
        if "z" not in data.files:
            raise KeyError(f"No key 'z'. Available keys: {data.files}")
        z = data["z"]
    if z.ndim == 4:
        if z.shape[0] != 1:
            raise ValueError(f"Expected z shape (1,L,L,D) or (L,L,D), got {z.shape}")
        z = z[0]
    if z.ndim != 3:
        raise ValueError(f"Expected z shape (L,L,D), got {z.shape}")
    if z.shape[0] != z.shape[1]:
        raise ValueError(f"Expected square pair matrix, got {z.shape}")
    return z.astype(np.float32, copy=False)


def chain_slices(tcra_len: int, tcrb_len: int, pep_len: int, hla_len: int) -> Dict[str, slice]:
    a0 = 0
    a1 = a0 + tcra_len
    b1 = a1 + tcrb_len
    p1 = b1 + pep_len
    h1 = p1 + hla_len
    return {
        "tcra": slice(a0, a1),
        "tcrb": slice(a1, b1),
        "pep": slice(b1, p1),
        "hla": slice(p1, h1),
        "tcr": slice(a0, b1),
        "pmhc": slice(b1, h1),
        "all": slice(a0, h1),
    }


def safe_stats_1d(x: np.ndarray, prefix: str, top_fracs: Tuple[float, ...] = (0.01, 0.05, 0.10)) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    out: Dict[str, float] = {f"{prefix}_n": float(x.size)}
    if x.size == 0:
        for name in ["mean", "std", "min", "max", "median", "q90", "q95", "q99"]:
            out[f"{prefix}_{name}"] = np.nan
        for frac in top_fracs:
            out[f"{prefix}_top{int(frac*100):02d}_mean"] = np.nan
        return out
    out.update({
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_q90": float(np.quantile(x, 0.90)),
        f"{prefix}_q95": float(np.quantile(x, 0.95)),
        f"{prefix}_q99": float(np.quantile(x, 0.99)),
    })
    xs = np.sort(x)
    n = x.size
    for frac in top_fracs:
        k = max(1, int(math.ceil(frac * n)))
        out[f"{prefix}_top{int(frac*100):02d}_mean"] = float(np.mean(xs[-k:]))
    return out


def cosine_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + eps)
    if den <= eps:
        return np.nan
    return float(np.dot(a, b) / den)


def mse_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.mean((a - b) ** 2))


def block_features(block: np.ndarray, prefix: str, save_mean_vectors: bool = False) -> Tuple[Dict[str, float], np.ndarray]:
    """Compute scalar features for a z block of shape [L1, L2, Dz]."""
    out: Dict[str, float] = {}
    if block.size == 0 or block.shape[0] == 0 or block.shape[1] == 0:
        return {f"{prefix}_empty": 1.0}, np.zeros((0,), dtype=np.float32)

    # Pairwise vector magnitudes. This preserves the hypothesis that magnitude may
    # carry structural/confidence signal.
    norms = np.linalg.norm(block, axis=-1)
    out.update(safe_stats_1d(norms, f"{prefix}_norm"))

    # Per-feature first and second moments. These keep some channel-level signal
    # without storing all D dimensions downstream.
    flat = block.reshape(-1, block.shape[-1]).astype(np.float32, copy=False)
    mean_vec = flat.mean(axis=0)
    std_vec = flat.std(axis=0)
    abs_mean_vec = np.abs(mean_vec)

    out.update(safe_stats_1d(mean_vec, f"{prefix}_meanvec"))
    out.update(safe_stats_1d(abs_mean_vec, f"{prefix}_absmeanvec"))
    out.update(safe_stats_1d(std_vec, f"{prefix}_stdvec"))
    out[f"{prefix}_meanvec_l2"] = float(np.linalg.norm(mean_vec))
    out[f"{prefix}_stdvec_l2"] = float(np.linalg.norm(std_vec))
    out[f"{prefix}_pair_count"] = float(block.shape[0] * block.shape[1])
    out[f"{prefix}_empty"] = 0.0

    return out, mean_vec.astype(np.float32, copy=False)


def extract_z_features_for_row(repo_root: Path, row: pd.Series, split_root: Optional[Path], strict_length_match: bool) -> Tuple[Optional[Dict[str, float]], Dict[str, object]]:
    pid = str(row["pair_id"])
    audit = {"pair_id": pid, "status": "ok", "reason": ""}
    npz_path = resolve_npz_path(repo_root, row, split_root)
    if npz_path is None:
        audit.update(status="skip", reason="missing_npz")
        return None, audit

    try:
        z = load_z(npz_path)
    except Exception as exc:
        audit.update(status="skip", reason=f"load_z_failed:{type(exc).__name__}:{exc}")
        return None, audit

    tcra_len = int(row["tcra_len"])
    tcrb_len = int(row["tcrb_len"])
    pep_len = int(row["pep_len"])
    hla_len = int(row["hla_len"])
    L_expected = tcra_len + tcrb_len + pep_len + hla_len
    if strict_length_match and int(z.shape[0]) != L_expected:
        audit.update(status="skip", reason=f"length_mismatch:zL={z.shape[0]}:expected={L_expected}")
        return None, audit
    if int(z.shape[0]) < L_expected:
        audit.update(status="skip", reason=f"z_shorter_than_manifest:zL={z.shape[0]}:expected={L_expected}")
        return None, audit

    sl = chain_slices(tcra_len, tcrb_len, pep_len, hla_len)

    feature_row: Dict[str, float] = {
        "pair_id": pid,
        "peptide": str(row["peptide_for_eval"]),
        "binding_flag": int(row["binding_flag"]),
        "tcra_len": tcra_len,
        "tcrb_len": tcrb_len,
        "pep_len": pep_len,
        "hla_len": hla_len,
        "z_L": int(z.shape[0]),
        "z_D": int(z.shape[-1]),
    }

    block_defs = {
        "tcra_pep": ("tcra", "pep"),
        "tcrb_pep": ("tcrb", "pep"),
        "tcr_pep": ("tcr", "pep"),
        "tcra_hla": ("tcra", "hla"),
        "tcrb_hla": ("tcrb", "hla"),
        "tcr_hla": ("tcr", "hla"),
        "pep_hla": ("pep", "hla"),
        "tcra_tcrb": ("tcra", "tcrb"),
        "tcr_pmhc": ("tcr", "pmhc"),
        "global": ("all", "all"),
    }

    mean_vecs: Dict[str, np.ndarray] = {}
    for name, (rkey, ckey) in block_defs.items():
        block = z[sl[rkey], sl[ckey], :]
        feats, mvec = block_features(block, name)
        feature_row.update(feats)
        mean_vecs[name] = mvec

        # Also include reverse-direction block where it is distinct. z is not
        # guaranteed to be symmetric in representation space.
        if rkey != ckey:
            rev_name = f"{ckey}_{rkey}"
            rev_block = z[sl[ckey], sl[rkey], :]
            rev_feats, rev_mvec = block_features(rev_block, rev_name)
            feature_row.update(rev_feats)
            mean_vecs[rev_name] = rev_mvec

    # Directional / block-comparison features. These are often useful because they
    # test whether interaction blocks differ in geometry or magnitude, without
    # storing full block vectors.
    comparisons = [
        ("tcrb_pep", "tcra_pep"),
        ("tcr_pep", "tcr_hla"),
        ("tcr_pep", "pep_hla"),
        ("tcrb_hla", "tcra_hla"),
        ("tcr_pmhc", "global"),
        ("pep_hla", "global"),
    ]
    for a, b in comparisons:
        if a in mean_vecs and b in mean_vecs and mean_vecs[a].size and mean_vecs[b].size:
            feature_row[f"cos_meanvec_{a}_vs_{b}"] = cosine_np(mean_vecs[a], mean_vecs[b])
            feature_row[f"mse_meanvec_{a}_vs_{b}"] = mse_np(mean_vecs[a], mean_vecs[b])
            feature_row[f"norm_ratio_{a}_vs_{b}"] = float((np.linalg.norm(mean_vecs[a]) + 1e-8) / (np.linalg.norm(mean_vecs[b]) + 1e-8))

    audit.update(status="ok", reason="")
    return feature_row, audit


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[m], scores[m]))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    return float(average_precision_score(labels[m], scores[m]))


def partial_auc_raw(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels[m], scores[m])
    stop = np.searchsorted(fpr, max_fpr, side="right")
    f = np.concatenate([fpr[:stop], [max_fpr]])
    t = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    return float(auc(f, t))


def partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return float("nan")
    return float(roc_auc_score(labels[m], scores[m], max_fpr=max_fpr))


def per_peptide_metrics(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float = 0.1) -> Dict[str, float]:
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    rows = []
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy(int)
        s = grp["score"].to_numpy(float)
        valid = len(np.unique(y)) == 2 and np.isfinite(s).sum() >= 2
        if valid:
            rows.append({
                "peptide": pep,
                "n": len(grp),
                "auroc": safe_auroc(y, s),
                "auc0.1_mcclish": partial_auc_mcclish(y, s, max_fpr),
            })
    if not rows:
        return {
            "pep_macro_auroc": float("nan"),
            "pep_weighted_auroc": float("nan"),
            "pep_macro_auc0.1_mcclish": float("nan"),
            "pep_weighted_auc0.1_mcclish": float("nan"),
            "n_valid_peptides": 0,
        }
    tab = pd.DataFrame(rows)
    return {
        "pep_macro_auroc": float(tab["auroc"].mean()),
        "pep_weighted_auroc": float(np.average(tab["auroc"], weights=tab["n"])),
        "pep_macro_auc0.1_mcclish": float(tab["auc0.1_mcclish"].mean()),
        "pep_weighted_auc0.1_mcclish": float(np.average(tab["auc0.1_mcclish"], weights=tab["n"])),
        "n_valid_peptides": int(len(tab)),
    }


def metrics_dict(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float = 0.1) -> Dict[str, float]:
    pr = partial_auc_raw(labels, scores, max_fpr)
    pm = partial_auc_mcclish(labels, scores, max_fpr)
    return {
        "auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "auc0.1_raw": pr,
        "auc0.1_raw_div_maxfpr": float(pr / max_fpr) if not math.isnan(pr) else float("nan"),
        "auc0.1_mcclish": pm,
        **per_peptide_metrics(labels, scores, peptides, max_fpr),
    }


# -----------------------------------------------------------------------------
# Assessment
# -----------------------------------------------------------------------------

def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"pair_id", "peptide", "binding_flag", "tcra_len", "tcrb_len", "pep_len", "hla_len", "z_L", "z_D"}
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def infer_feature_group(col: str) -> str:
    # Specific biologically meaningful groups first.
    for group in [
        "tcra_pep", "tcrb_pep", "tcr_pep",
        "tcra_hla", "tcrb_hla", "tcr_hla",
        "pep_hla", "tcra_tcrb", "tcr_pmhc", "global",
        "pep_tcra", "pep_tcrb", "pep_tcr",
        "hla_tcra", "hla_tcrb", "hla_tcr", "hla_pep",
        "tcrb_tcra", "pmhc_tcr",
    ]:
        if col.startswith(group + "_"):
            return group
    if col.startswith("cos_meanvec") or col.startswith("mse_meanvec") or col.startswith("norm_ratio"):
        return "block_comparison"
    return "other"


def assess_univariate_by_split(split_dfs: Dict[str, pd.DataFrame], out_dir: Path, max_fpr: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_cols = sorted(set().union(*[set(feature_columns(df)) for df in split_dfs.values()]))
    rows = []
    for split, df in split_dfs.items():
        y = df["binding_flag"].to_numpy(int)
        peps = df["peptide"].to_numpy(str)
        for c in all_cols:
            if c not in df.columns:
                continue
            s_raw = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            m_pos = metrics_dict(y, s_raw, peps, max_fpr)
            m_neg = metrics_dict(y, -s_raw, peps, max_fpr)
            # Split-wise best orientation is only diagnostic and can overstate signal.
            if (m_neg["auroc"] if np.isfinite(m_neg["auroc"]) else -np.inf) > (m_pos["auroc"] if np.isfinite(m_pos["auroc"]) else -np.inf):
                best = m_neg
                sign = -1
            else:
                best = m_pos
                sign = 1
            rows.append({
                "split": split,
                "feature": c,
                "group": infer_feature_group(c),
                "splitwise_best_sign": sign,
                **{f"best_{k}": v for k, v in best.items()},
                **{f"plus_{k}": v for k, v in m_pos.items()},
                **{f"minus_{k}": v for k, v in m_neg.items()},
            })
    out = pd.DataFrame(rows).sort_values(["split", "best_auroc"], ascending=[True, False])
    out.to_csv(out_dir / "univariate_feature_metrics_by_split.csv", index=False)

    # Choose orientation on val only, then apply to test/IMMREP. This is more honest
    # for detecting portable signal than picking sign separately on each split.
    val_rows = out[out["split"] == "val"].copy()
    val_sign = dict(zip(val_rows["feature"], val_rows["splitwise_best_sign"]))
    rows2 = []
    for split, df in split_dfs.items():
        y = df["binding_flag"].to_numpy(int)
        peps = df["peptide"].to_numpy(str)
        for c, sign in val_sign.items():
            if c not in df.columns:
                continue
            s = sign * pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            m = metrics_dict(y, s, peps, max_fpr)
            rows2.append({"split": split, "feature": c, "group": infer_feature_group(c), "val_oriented_sign": sign, **m})
    oriented = pd.DataFrame(rows2).sort_values(["split", "auroc"], ascending=[True, False])
    oriented.to_csv(out_dir / "univariate_feature_metrics_val_oriented.csv", index=False)
    return out, oriented


def fit_eval_logreg_groups(split_dfs: Dict[str, pd.DataFrame], out_dir: Path, max_fpr: float = 0.1, seed: int = 31) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fit small L2 logistic models on val decoys and evaluate test/IMMREP.

    This is a calibration diagnostic, not the final training protocol. Because train
    has no negatives, val is used for fitting. Test/IMMREP are therefore the only
    meaningful held-out readouts. IMMREP has true negatives and is the most important.
    """
    if "val" not in split_dfs:
        return pd.DataFrame(), pd.DataFrame()

    val_df = split_dfs["val"].copy()
    all_cols = sorted(set().union(*[set(feature_columns(df)) for df in split_dfs.values()]))
    groups = sorted(set(infer_feature_group(c) for c in all_cols))

    group_to_cols = {"all_z_features": all_cols}
    for g in groups:
        group_to_cols[f"group_{g}"] = [c for c in all_cols if infer_feature_group(c) == g]

    # Biologically useful aggregate groups.
    tcr_pep_groups = {"tcra_pep", "tcrb_pep", "tcr_pep", "pep_tcra", "pep_tcrb", "pep_tcr"}
    tcr_hla_groups = {"tcra_hla", "tcrb_hla", "tcr_hla", "hla_tcra", "hla_tcrb", "hla_tcr"}
    group_to_cols["aggregate_tcr_peptide_blocks"] = [c for c in all_cols if infer_feature_group(c) in tcr_pep_groups]
    group_to_cols["aggregate_tcr_hla_blocks"] = [c for c in all_cols if infer_feature_group(c) in tcr_hla_groups]
    group_to_cols["aggregate_no_global"] = [c for c in all_cols if infer_feature_group(c) != "global"]
    group_to_cols["aggregate_no_comparisons"] = [c for c in all_cols if infer_feature_group(c) != "block_comparison"]
    group_to_cols = {k: v for k, v in group_to_cols.items() if len(v) > 0}

    y_val = val_df["binding_flag"].to_numpy(int)
    if len(np.unique(y_val)) < 2:
        print("val has only one class; skipping fitted logistic diagnostics", flush=True)
        return pd.DataFrame(), pd.DataFrame()

    metric_rows = []
    coef_rows = []
    for group_name, cols in group_to_cols.items():
        # Remove all-NaN and constant columns on val.
        usable_cols = []
        for c in cols:
            if c not in val_df.columns:
                continue
            x = pd.to_numeric(val_df[c], errors="coerce").to_numpy(float)
            finite = x[np.isfinite(x)]
            if finite.size < 3:
                continue
            if np.nanstd(finite) <= 1e-12:
                continue
            usable_cols.append(c)
        if len(usable_cols) == 0:
            continue

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=0.1,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
                max_iter=2000,
            )),
        ])
        X_val = val_df[usable_cols].apply(pd.to_numeric, errors="coerce")
        pipe.fit(X_val, y_val)

        clf = pipe.named_steps["clf"]
        coefs = clf.coef_.reshape(-1)
        for c, w in sorted(zip(usable_cols, coefs), key=lambda x: abs(x[1]), reverse=True)[:100]:
            coef_rows.append({"model_group": group_name, "feature": c, "feature_group": infer_feature_group(c), "coef": float(w)})

        for split, df in split_dfs.items():
            cols_here = [c for c in usable_cols if c in df.columns]
            if len(cols_here) != len(usable_cols):
                # Add missing columns as NaN to keep schema.
                tmp = df.copy()
                for c in usable_cols:
                    if c not in tmp.columns:
                        tmp[c] = np.nan
                X = tmp[usable_cols].apply(pd.to_numeric, errors="coerce")
            else:
                X = df[usable_cols].apply(pd.to_numeric, errors="coerce")
            y = df["binding_flag"].to_numpy(int)
            peps = df["peptide"].to_numpy(str)
            try:
                s = pipe.predict_proba(X)[:, 1]
            except Exception:
                s = pipe.decision_function(X)
            m = metrics_dict(y, s, peps, max_fpr)
            metric_rows.append({
                "model_group": group_name,
                "fit_split": "val",
                "eval_split": split,
                "n_features": len(usable_cols),
                **m,
            })

    metric_df = pd.DataFrame(metric_rows).sort_values(["eval_split", "auroc"], ascending=[True, False])
    coef_df = pd.DataFrame(coef_rows)
    metric_df.to_csv(out_dir / "block_group_logreg_valfit_metrics.csv", index=False)
    coef_df.to_csv(out_dir / "block_group_logreg_coefficients.csv", index=False)
    return metric_df, coef_df


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    repo_root: str = "/home/natasha/multimodal_model"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    val_boltz_root: str = "/home/natasha/multimodal_model/outputs/val"
    test_boltz_root: str = "/home/natasha/multimodal_model/outputs/test"
    immrep_boltz_root: str = "/home/natasha/multimodal_model/outputs_data/immrep_test"
    out_dir: str = "/home/natasha/multimodal_model/outputs_data/z_block_features"
    strict_length_match: bool = True
    recompute: bool = False
    max_rows_per_split: int = 0
    partial_auc_max_fpr: float = 0.1
    seed: int = 31


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=RunConfig.repo_root)
    p.add_argument("--val-csv", default=RunConfig.val_csv)
    p.add_argument("--test-csv", default=RunConfig.test_csv)
    p.add_argument("--immrep-csv", default=RunConfig.immrep_csv)
    p.add_argument("--val-boltz-root", default=RunConfig.val_boltz_root)
    p.add_argument("--test-boltz-root", default=RunConfig.test_boltz_root)
    p.add_argument("--immrep-boltz-root", default=RunConfig.immrep_boltz_root)
    p.add_argument("--out-dir", default=RunConfig.out_dir)
    p.add_argument("--strict-length-match", action=argparse.BooleanOptionalAction, default=RunConfig.strict_length_match)
    p.add_argument("--recompute", action=argparse.BooleanOptionalAction, default=RunConfig.recompute)
    p.add_argument("--max-rows-per-split", type=int, default=RunConfig.max_rows_per_split)
    p.add_argument("--partial-auc-max-fpr", type=float, default=RunConfig.partial_auc_max_fpr)
    p.add_argument("--seed", type=int, default=RunConfig.seed)
    return RunConfig(**vars(p.parse_args()))


def build_features_for_split(split: str, csv_path: Path, split_root: Path, cfg: RunConfig, out_dir: Path) -> pd.DataFrame:
    feat_path = out_dir / f"{split}_z_block_features.csv"
    audit_path = out_dir / f"{split}_z_feature_audit.csv"
    if feat_path.exists() and not cfg.recompute:
        print(f"{split}: loading existing features from {feat_path}", flush=True)
        return pd.read_csv(feat_path)

    repo_root = Path(cfg.repo_root)
    raw = pd.read_csv(csv_path)
    meta = normalise_manifest(raw, split)
    meta = meta[complete_complex_filter(meta)].copy().reset_index(drop=True)
    if cfg.max_rows_per_split and cfg.max_rows_per_split > 0:
        meta = meta.head(cfg.max_rows_per_split).copy()

    print("=" * 80, flush=True)
    print(f"Building z block features for {split}", flush=True)
    print(f"CSV rows={len(raw)} | complete_complex_rows={len(meta)}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"Boltz root fallback: {split_root}", flush=True)
    print("=" * 80, flush=True)

    rows: List[Dict[str, float]] = []
    audits: List[Dict[str, object]] = []
    for i, row in meta.iterrows():
        feats, audit = extract_z_features_for_row(repo_root, row, split_root, cfg.strict_length_match)
        audits.append(audit)
        if feats is not None:
            rows.append(feats)
        if (i + 1) % 250 == 0:
            print(f"{split}: processed {i+1}/{len(meta)} | features_written={len(rows)}", flush=True)

    feat_df = pd.DataFrame(rows)
    audit_df = pd.DataFrame(audits)
    feat_df.to_csv(feat_path, index=False)
    audit_df.to_csv(audit_path, index=False)

    print(f"{split}: wrote features={len(feat_df)} to {feat_path}", flush=True)
    print(f"{split}: audit status counts:\n{audit_df['reason'].value_counts(dropna=False).head(20)}", flush=True)
    return feat_df


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    split_specs = {
        "val": (Path(cfg.val_csv), Path(cfg.val_boltz_root)),
        "test": (Path(cfg.test_csv), Path(cfg.test_boltz_root)),
        "immrep_test": (Path(cfg.immrep_csv), Path(cfg.immrep_boltz_root)),
    }

    split_dfs: Dict[str, pd.DataFrame] = {}
    for split, (csv_path, root) in split_specs.items():
        if not csv_path.exists():
            print(f"{split}: missing CSV {csv_path}; skipping", flush=True)
            continue
        df = build_features_for_split(split, csv_path, root, cfg, out_dir)
        if len(df) == 0:
            print(f"{split}: no features written; skipping assessment for this split", flush=True)
            continue
        split_dfs[split] = df

    if not split_dfs:
        raise RuntimeError("No split features available for assessment")

    schema = []
    for c in sorted(set().union(*[set(feature_columns(df)) for df in split_dfs.values()])):
        schema.append({"feature": c, "group": infer_feature_group(c)})
    (out_dir / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    print("=" * 80, flush=True)
    print("Assessing univariate z block-summary features", flush=True)
    print("=" * 80, flush=True)
    univ, oriented = assess_univariate_by_split(split_dfs, out_dir, cfg.partial_auc_max_fpr)

    print("=" * 80, flush=True)
    print("Fitting small L2 logistic models on val decoys and evaluating test/IMMREP", flush=True)
    print("NOTE: this is a calibration diagnostic because train has no negatives.", flush=True)
    print("=" * 80, flush=True)
    logreg_metrics, logreg_coefs = fit_eval_logreg_groups(split_dfs, out_dir, cfg.partial_auc_max_fpr, cfg.seed)

    summary = {
        "config": asdict(cfg),
        "split_counts": {
            split: {
                "n": int(len(df)),
                "n_pos": int(df["binding_flag"].sum()),
                "n_neg": int((df["binding_flag"] == 0).sum()),
                "n_peptides": int(df["peptide"].nunique()),
                "n_features": int(len(feature_columns(df))),
            }
            for split, df in split_dfs.items()
        },
        "top_univariate_by_split": univ.groupby("split").head(20).to_dict(orient="records") if len(univ) else [],
        "top_val_oriented_immrep": oriented[oriented["split"] == "immrep_test"].head(20).to_dict(orient="records") if len(oriented) else [],
        "top_logreg_immrep": logreg_metrics[logreg_metrics["eval_split"] == "immrep_test"].head(20).to_dict(orient="records") if len(logreg_metrics) else [],
        "outputs": {
            "feature_schema": str(out_dir / "feature_schema.json"),
            "univariate_by_split": str(out_dir / "univariate_feature_metrics_by_split.csv"),
            "univariate_val_oriented": str(out_dir / "univariate_feature_metrics_val_oriented.csv"),
            "logreg_metrics": str(out_dir / "block_group_logreg_valfit_metrics.csv"),
            "logreg_coefficients": str(out_dir / "block_group_logreg_coefficients.csv"),
        },
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))

    print("=" * 80, flush=True)
    print("Done.", flush=True)
    print(f"Outputs written to: {out_dir}", flush=True)
    print("Split counts:", json.dumps(summary["split_counts"], indent=2), flush=True)
    if len(oriented):
        print("Top IMMREP val-oriented univariate features:", flush=True)
        cols = ["feature", "group", "val_oriented_sign", "auroc", "auprc", "auc0.1_mcclish", "pep_weighted_auroc", "pep_macro_auc0.1_mcclish"]
        print(oriented[oriented["split"] == "immrep_test"][cols].head(15).to_string(index=False), flush=True)
    if len(logreg_metrics):
        print("Top IMMREP val-fit logistic groups:", flush=True)
        cols = ["model_group", "n_features", "auroc", "auprc", "auc0.1_mcclish", "pep_weighted_auroc", "pep_macro_auc0.1_mcclish"]
        print(logreg_metrics[logreg_metrics["eval_split"] == "immrep_test"][cols].head(15).to_string(index=False), flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
