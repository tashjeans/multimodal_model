#!/usr/bin/env python3
"""
Assess Boltz signal injection into an existing ESM VICReg model.

Purpose
-------
This script does NOT retrain VICReg. It evaluates whether Boltz-derived
structural information adds conditional signal beyond a frozen ESM VICReg score.

It compares:
  1. ESM only
  2. ESM + confidence/summary metrics
  3. ESM + selected z block-summary features
  4. ESM + block-group z features
  5. ESM + confidence + z features
  6. z/confidence-only diagnostic models

Protocol
--------
Because the main train set contains positives only, this is a calibration and
signal-diagnostic script:
  - Fit fusion/calibration models on VAL labels/decoys.
  - Evaluate unchanged on TEST and IMMREP_TEST.
  - Treat VAL as calibration, TEST as internal held-out, and IMMREP as the
    external true-negative benchmark.

Inputs
------
- Existing ESM VICReg prediction CSVs for val/test/immrep_test.
- Deterministic Boltz z block-summary CSVs built separately.
- Optional confidence/summary metrics from multiview CSVs, z feature CSVs, and
  Boltz confidence_*.json files colocated with embeddings_*.npz.

Expected ESM prediction columns:
  pair_id, peptide, label, model_score

If the column names differ, the script attempts sensible fallbacks.

Outputs
-------
- model_group_metrics.csv
- model_group_predictions.csv
- model_group_coefficients.csv
- feature_group_manifest.json
- per_peptide_metrics_by_model.csv
- run_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def partial_auc_raw(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.1) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    stop = np.searchsorted(fpr, max_fpr, side="right")
    fpr_ext = np.concatenate([fpr[:stop], [max_fpr]])
    tpr_ext = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    return float(auc(fpr_ext, tpr_ext))


def partial_auc_mcclish(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.1) -> float:
    """sklearn's standardised partial AUC; random ~= 0.5, perfect = 1.0."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    m = np.isfinite(s)
    y, s = y[m], s[m]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


def per_peptide_metrics(
    y: np.ndarray,
    s: np.ndarray,
    peptide: np.ndarray,
    max_fpr: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = pd.DataFrame({
        "label": np.asarray(y).astype(int),
        "score": np.asarray(s).astype(float),
        "peptide": np.asarray(peptide).astype(str),
    })
    df = df[np.isfinite(df["score"].to_numpy())].copy()
    rows = []
    for pep, g in df.groupby("peptide", sort=True):
        yy = g["label"].to_numpy(int)
        ss = g["score"].to_numpy(float)
        valid = len(np.unique(yy)) == 2
        rows.append({
            "peptide": pep,
            "n": int(len(g)),
            "n_pos": int(yy.sum()),
            "n_neg": int((yy == 0).sum()),
            "auroc": safe_auroc(yy, ss) if valid else float("nan"),
            f"auc{max_fpr:g}_raw": partial_auc_raw(yy, ss, max_fpr) if valid else float("nan"),
            f"auc{max_fpr:g}_mcclish": partial_auc_mcclish(yy, ss, max_fpr) if valid else float("nan"),
            "valid": bool(valid),
        })
    tab = pd.DataFrame(rows)
    if len(tab) == 0 or not tab["valid"].any():
        return tab, {
            "pep_macro_auroc": float("nan"),
            "pep_weighted_auroc": float("nan"),
            f"pep_macro_auc{max_fpr:g}_mcclish": float("nan"),
            f"pep_weighted_auc{max_fpr:g}_mcclish": float("nan"),
            "n_valid_peptides": 0,
            "n_peptides_total": int(len(tab)),
        }
    vt = tab[tab["valid"]].copy()
    return tab, {
        "pep_macro_auroc": float(vt["auroc"].mean()),
        "pep_weighted_auroc": float(np.average(vt["auroc"], weights=vt["n"])),
        f"pep_macro_auc{max_fpr:g}_mcclish": float(vt[f"auc{max_fpr:g}_mcclish"].mean()),
        f"pep_weighted_auc{max_fpr:g}_mcclish": float(np.average(vt[f"auc{max_fpr:g}_mcclish"], weights=vt["n"])),
        "n_valid_peptides": int(len(vt)),
        "n_peptides_total": int(len(tab)),
    }


def metric_summary(y: np.ndarray, s: np.ndarray, peptide: np.ndarray, max_fpr: float = 0.1) -> Dict[str, float]:
    _, pep = per_peptide_metrics(y, s, peptide, max_fpr=max_fpr)
    raw = partial_auc_raw(y, s, max_fpr=max_fpr)
    return {
        "n": int(len(y)),
        "n_pos": int(np.asarray(y).sum()),
        "n_neg": int((np.asarray(y) == 0).sum()),
        "auroc": safe_auroc(y, s),
        "auprc": safe_auprc(y, s),
        f"auc{max_fpr:g}_raw": raw,
        f"auc{max_fpr:g}_raw_div_maxfpr": float(raw / max_fpr) if not math.isnan(raw) else float("nan"),
        f"auc{max_fpr:g}_mcclish": partial_auc_mcclish(y, s, max_fpr=max_fpr),
        **pep,
    }


# -----------------------------------------------------------------------------
# Loading and feature handling
# -----------------------------------------------------------------------------

def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalise_esm_predictions(path: str, split: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing ESM prediction CSV for {split}: {p}")
    df = pd.read_csv(p)
    pair_col = first_existing_col(df, ["pair_id", "id", "complex_id"])
    label_col = first_existing_col(df, ["label", "binding_flag", "binder", "target"])
    pep_col = first_existing_col(df, ["peptide", "Peptide", "pep_seq", "peptide_for_eval"])
    score_col = first_existing_col(df, ["model_score", "esm_score", "score", "prediction_score"])
    raw_col = first_existing_col(df, ["raw_esm_score", "raw_score"])
    if pair_col is None or label_col is None or score_col is None:
        raise ValueError(
            f"{path} must contain pair_id, label/binding_flag and model_score-like columns. "
            f"Columns: {list(df.columns)}"
        )
    out = pd.DataFrame({
        "pair_id": df[pair_col].astype(str),
        "label": df[label_col].astype(int),
        "esm_score": pd.to_numeric(df[score_col], errors="coerce"),
    })
    out["peptide"] = df[pep_col].astype(str) if pep_col is not None else out["pair_id"]
    if raw_col is not None:
        out["raw_esm_score"] = pd.to_numeric(df[raw_col], errors="coerce")
    out["split"] = split
    return out


def load_z_features(path: str, split: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing z feature CSV for {split}: {p}")
    df = pd.read_csv(p)
    if "pair_id" not in df.columns:
        raise ValueError(f"{path} must contain pair_id")
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    return df


def load_manifest(path: str, split: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"[warn] missing manifest for {split}: {p}")
        return pd.DataFrame(columns=["pair_id"])
    df = pd.read_csv(p)
    if "pair_id" not in df.columns:
        return pd.DataFrame(columns=["pair_id"])
    df = df.copy()
    df["pair_id"] = df["pair_id"].astype(str)
    return df




def _safe_float(x) -> Optional[float]:
    try:
        if isinstance(x, (bool, str)):
            # Avoid converting arbitrary strings; JSON scalar scores should already be numeric.
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def flatten_numeric_json(obj, prefix: str = "") -> Dict[str, float]:
    """Flatten numeric values from nested Boltz confidence JSON.

    Examples:
      confidence_score -> confidence_score
      chains_ptm["0"] -> chains_ptm_0
      pair_chains_iptm["0"]["2"] -> pair_chains_iptm_0_2
    """
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            kk = re.sub(r"[^A-Za-z0-9]+", "_", str(k)).strip("_")
            name = f"{prefix}_{kk}" if prefix else kk
            out.update(flatten_numeric_json(v, name))
    elif isinstance(obj, (list, tuple)):
        # Keep short numeric arrays as individual entries; summarise longer arrays.
        vals = [_safe_float(v) for v in obj]
        vals = [v for v in vals if v is not None]
        if vals:
            arr = np.asarray(vals, dtype=float)
            out[f"{prefix}_mean"] = float(arr.mean())
            out[f"{prefix}_std"] = float(arr.std())
            out[f"{prefix}_min"] = float(arr.min())
            out[f"{prefix}_max"] = float(arr.max())
            if len(vals) <= 16:
                for i, v in enumerate(vals):
                    out[f"{prefix}_{i}"] = float(v)
    else:
        v = _safe_float(obj)
        if v is not None and prefix:
            out[prefix] = float(v)
    return out


def resolve_confidence_json(row: pd.Series, repo_root: Path) -> Optional[Path]:
    """Resolve Boltz confidence JSON from a multiview row.

    Preferred source is a JSON path column if present. Otherwise infer it as the
    sibling of embeddings_{pair_id}.npz inside predictions/{pair_id}/.
    """
    # Direct path columns, if any exist in future CSVs.
    for c in [
        "boltz_confidence_json", "confidence_json", "confidence_path",
        "boltz_confidence_path", "confidence_file",
    ]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            q = Path(str(row[c]))
            if not q.is_absolute():
                q = repo_root / q
            if q.exists():
                return q

    pair_id = str(row.get("pair_id", ""))
    emb_col = None
    for c in ["boltz_embedding_npz", "embedding_npz", "boltz_npz"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            emb_col = c
            break
    if emb_col is None:
        return None

    emb = Path(str(row[emb_col]))
    if not emb.is_absolute():
        emb = repo_root / emb
    parent = emb.parent
    candidates = []
    if pair_id:
        candidates.extend([
            parent / f"confidence_{pair_id}_model_0.json",
            parent / f"confidence_{pair_id}.json",
        ])
    candidates.extend(sorted(parent.glob("confidence_*_model_*.json")))
    candidates.extend(sorted(parent.glob("confidence_*.json")))
    for q in candidates:
        if q.exists():
            return q
    return None


def load_confidence_json_features(manifest: pd.DataFrame, repo_root: str, split: str) -> pd.DataFrame:
    """Load scalar Boltz confidence metrics from confidence_*.json files.

    Returns pair_id + numeric confidence features. Missing files are allowed and
    simply produce no feature row for that pair_id.
    """
    repo = Path(repo_root)
    rows = []
    n_missing = 0
    n_bad = 0
    for _, row in manifest.iterrows():
        if "pair_id" not in row.index:
            continue
        pid = str(row["pair_id"])
        q = resolve_confidence_json(row, repo)
        if q is None:
            n_missing += 1
            continue
        try:
            with open(q, "r") as f:
                payload = json.load(f)
            flat = flatten_numeric_json(payload)
            if flat:
                flat = {f"conf__{k}": v for k, v in flat.items()}
                flat["pair_id"] = pid
                rows.append(flat)
        except Exception as exc:
            n_bad += 1
            if n_bad <= 5:
                print(f"[warn] {split}: failed reading confidence JSON for pair_id={pid} at {q}: {type(exc).__name__}: {exc}", flush=True)
    out = pd.DataFrame(rows)
    if len(out) == 0:
        out = pd.DataFrame(columns=["pair_id"])
    else:
        out["pair_id"] = out["pair_id"].astype(str)
    print(
        f"{split}: confidence_json rows={len(out)} | missing={n_missing} | bad={n_bad}",
        flush=True,
    )
    return out

def numeric_feature_cols(df: pd.DataFrame, exclude: Iterable[str]) -> List[str]:
    ex = set(exclude)
    cols = []
    for c in df.columns:
        if c in ex:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def prefixed_join(base: pd.DataFrame, extra: pd.DataFrame, prefix: str, exclude_cols: Sequence[str]) -> pd.DataFrame:
    if extra is None or len(extra) == 0:
        return base
    extra = extra.copy()
    if "pair_id" not in extra.columns:
        return base
    rename = {}
    for c in extra.columns:
        if c == "pair_id" or c in exclude_cols:
            continue
        if pd.api.types.is_numeric_dtype(extra[c]):
            rename[c] = f"{prefix}{c}"
    slim = extra[["pair_id"] + list(rename.keys())].rename(columns=rename)
    return base.merge(slim, on="pair_id", how="left")


def build_split_table(esm: pd.DataFrame, z: pd.DataFrame, manifest: pd.DataFrame, conf: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = esm.copy()
    # z features may include metadata; prefix all numeric z-derived fields to avoid collisions.
    z_exclude = [
        "label", "binding_flag", "binder", "target", "split", "peptide", "Peptide",
        "tcra", "tcrb", "hla", "HLA", "pep_seq", "peptide_seq", "boltz_embedding_npz",
    ]
    out = prefixed_join(out, z, "z__", z_exclude)
    # Manifests may include confidence and length columns; prefix numeric fields.
    man_exclude = [
        "label", "binding_flag", "binder", "target", "split", "peptide", "Peptide",
        "tcra", "tcrb", "hla", "HLA", "pep_seq", "peptide_seq", "boltz_embedding_npz",
    ]
    out = prefixed_join(out, manifest, "meta__", man_exclude)
    # Boltz confidence JSON features are already prefixed as conf__. Join without re-prefixing names.
    if conf is not None and len(conf) > 0 and "pair_id" in conf.columns:
        conf = conf.copy()
        keep = ["pair_id"] + [c for c in conf.columns if c.startswith("conf__") and pd.api.types.is_numeric_dtype(conf[c])]
        if len(keep) > 1:
            out = out.merge(conf[keep], on="pair_id", how="left")
    return out


# -----------------------------------------------------------------------------
# Feature groups
# -----------------------------------------------------------------------------

def contains_any(name: str, tokens: Sequence[str]) -> bool:
    return any(tok in name for tok in tokens)


def build_feature_groups(df: pd.DataFrame, max_selected_z: int = 30) -> Dict[str, List[str]]:
    all_cols = numeric_feature_cols(df, exclude=["label"])
    esm_cols = ["esm_score"] if "esm_score" in df.columns else []
    raw_esm_cols = ["raw_esm_score"] if "raw_esm_score" in df.columns else []

    z_cols = [c for c in all_cols if c.startswith("z__")]
    meta_cols = [c for c in all_cols if c.startswith("meta__")]
    conf_json_cols = [c for c in all_cols if c.startswith("conf__")]

    # Confidence/summary features. This intentionally errs on inclusiveness, but
    # is still separated from z block summaries.
    confidence_tokens = [
        "iptm", "ptm", "plddt", "confidence", "ranking", "pae", "ipae",
        "prob", "score", "complex", "interface",
    ]
    length_tokens = ["len", "length"]
    confidence_cols = [
        c for c in meta_cols
        if contains_any(c.lower(), confidence_tokens)
    ]
    # If the z feature files themselves contain confidence-like columns, include those separately.
    z_conf_cols = [
        c for c in z_cols
        if contains_any(c.lower(), ["iptm", "ptm", "plddt", "confidence", "ranking", "pae"])
    ]
    confidence_cols = sorted(set(confidence_cols + z_conf_cols + conf_json_cols))
    length_cols = [c for c in meta_cols if contains_any(c.lower(), length_tokens)]

    # z block groups. These rely on the deterministic extractor's naming pattern.
    def zmatch(tokens: Sequence[str], exclude_tokens: Sequence[str] = ()) -> List[str]:
        out = []
        for c in z_cols:
            cl = c.lower()
            if all(t in cl for t in tokens) and not any(t in cl for t in exclude_tokens):
                out.append(c)
        return sorted(set(out))

    tcr_pep = sorted(set(
        zmatch(["tcr", "pep"]) + zmatch(["pep", "tcr"]) +
        zmatch(["tcra", "pep"]) + zmatch(["tcrb", "pep"]) +
        zmatch(["pep", "tcra"]) + zmatch(["pep", "tcrb"])
    ))
    tcrb_pep = sorted(set(zmatch(["tcrb", "pep"]) + zmatch(["pep", "tcrb"])))
    tcra_pep = sorted(set(zmatch(["tcra", "pep"]) + zmatch(["pep", "tcra"])))
    tcr_hla = sorted(set(
        zmatch(["tcr", "hla"]) + zmatch(["hla", "tcr"]) +
        zmatch(["tcra", "hla"]) + zmatch(["tcrb", "hla"]) +
        zmatch(["hla", "tcra"]) + zmatch(["hla", "tcrb"])
    ))
    pep_hla = sorted(set(zmatch(["pep", "hla"]) + zmatch(["hla", "pep"])))
    global_z = sorted(set([c for c in z_cols if "global" in c.lower()]))
    comparison_z = sorted(set([c for c in z_cols if "comparison" in c.lower() or "_vs_" in c.lower()]))

    # Selected z features: biologically constrained and based on previous outputs.
    # We match by suffix because all z columns have z__ prefix.
    preferred_suffixes = [
        "pep_tcr_norm_top01_mean",
        "pep_tcr_norm_max",
        "tcrb_pep_norm_top01_mean",
        "pep_tcr_absmeanvec_q95",
        "mse_meanvec_pep_hla_vs_global",
        "tcr_pep_norm_top01_mean",
        "tcr_pep_norm_top10_mean",
        "pep_tcrb_absmeanvec_q95",
        "tcra_pep_absmeanvec_top10_mean",
        "pep_tcra_absmeanvec_q95",
        "tcr_pep_absmeanvec_top05_mean",
        "cos_meanvec_pep_hla_vs_global",
        "tcr_pep_stdvec_max",
        "pep_tcr_norm_q99",
        "hla_pep_meanvec_top05_mean",
    ]
    selected = []
    for suffix in preferred_suffixes:
        hits = [c for c in z_cols if c.endswith(suffix)]
        selected.extend(hits)
    selected = sorted(set(selected))

    # Fallback: if naming differs, use a compact subset from the biologically relevant blocks.
    if not selected:
        candidate = tcr_pep + tcrb_pep + pep_hla + comparison_z
        selected = candidate[:max_selected_z]

    groups = {
        "esm_only": esm_cols,
        "raw_esm_only": raw_esm_cols,
        "confidence_only": confidence_cols,
        "confidence_plus_lengths": sorted(set(confidence_cols + length_cols)),
        "z_selected_only": selected,
        "z_tcr_peptide_only": tcr_pep,
        "z_tcrb_peptide_only": tcrb_pep,
        "z_tcra_peptide_only": tcra_pep,
        "z_tcr_hla_only": tcr_hla,
        "z_pep_hla_only": pep_hla,
        "z_global_only": global_z,
        "z_comparison_only": comparison_z,
        "z_all_only": z_cols,
        "esm_plus_confidence": sorted(set(esm_cols + confidence_cols)),
        "esm_plus_confidence_lengths": sorted(set(esm_cols + confidence_cols + length_cols)),
        "esm_plus_z_selected": sorted(set(esm_cols + selected)),
        "esm_plus_z_tcr_peptide": sorted(set(esm_cols + tcr_pep)),
        "esm_plus_z_tcrb_peptide": sorted(set(esm_cols + tcrb_pep)),
        "esm_plus_z_tcra_peptide": sorted(set(esm_cols + tcra_pep)),
        "esm_plus_z_tcr_hla": sorted(set(esm_cols + tcr_hla)),
        "esm_plus_z_pep_hla": sorted(set(esm_cols + pep_hla)),
        "esm_plus_z_global": sorted(set(esm_cols + global_z)),
        "esm_plus_z_comparison": sorted(set(esm_cols + comparison_z)),
        "esm_plus_z_all": sorted(set(esm_cols + z_cols)),
        "esm_plus_confidence_z_selected": sorted(set(esm_cols + confidence_cols + selected)),
        "esm_plus_confidence_z_all": sorted(set(esm_cols + confidence_cols + z_cols)),
    }

    # Additional explicit confidence-score ablations. These are crucial controls:
    # they test whether expensive z features add more than cheap Boltz confidence JSON metrics.
    def conf_like(tokens: Sequence[str]) -> List[str]:
        return sorted([c for c in confidence_cols if contains_any(c.lower(), tokens)])

    explicit_conf = {
        "conf_confidence_score_only": conf_like(["confidence_score"]),
        "conf_iptm_only": [c for c in conf_like(["iptm"]) if "pair_chains" not in c.lower()],
        "conf_ptm_only": [c for c in conf_like(["ptm"]) if "iptm" not in c.lower() and "chains" not in c.lower()],
        "conf_plddt_only": conf_like(["plddt", "iplddt"]),
        "conf_pde_only": conf_like(["pde", "ipde"]),
        "conf_chain_pair_iptm_only": conf_like(["pair_chains_iptm"]),
        "esm_plus_confidence_score": sorted(set(esm_cols + conf_like(["confidence_score"]))),
        "esm_plus_iptm": sorted(set(esm_cols + [c for c in conf_like(["iptm"]) if "pair_chains" not in c.lower()])),
        "esm_plus_ptm": sorted(set(esm_cols + [c for c in conf_like(["ptm"]) if "iptm" not in c.lower() and "chains" not in c.lower()])),
        "esm_plus_plddt": sorted(set(esm_cols + conf_like(["plddt", "iplddt"]))),
        "esm_plus_chain_pair_iptm": sorted(set(esm_cols + conf_like(["pair_chains_iptm"]))),
    }
    groups.update(explicit_conf)
    # Remove empty and duplicate-only invalid groups.
    groups = {k: v for k, v in groups.items() if len(v) > 0}
    return groups


# -----------------------------------------------------------------------------
# Fitting/evaluation
# -----------------------------------------------------------------------------

def get_model(kind: str, C: float, max_iter: int, random_state: int):
    if kind == "logreg":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                C=C,
                solver="liblinear",
                max_iter=max_iter,
                random_state=random_state,
            )),
        ])
    if kind == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RidgeClassifier(alpha=1.0 / max(C, 1e-12), random_state=random_state)),
        ])
    raise ValueError(f"Unknown model kind: {kind}")


def predict_score(model, X: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "logreg":
        return model.predict_proba(X)[:, 1]
    if kind == "ridge":
        return model.decision_function(X)
    raise ValueError(kind)


def fit_and_eval_group(
    name: str,
    features: List[str],
    tables: Dict[str, pd.DataFrame],
    kind: str,
    C: float,
    max_iter: int,
    max_fpr: float,
    random_state: int,
) -> Tuple[List[Dict], List[pd.DataFrame], List[Dict], List[pd.DataFrame]]:
    val = tables["val"]
    X_val = val[features]
    y_val = val["label"].to_numpy(int)
    if len(np.unique(y_val)) < 2:
        raise RuntimeError("Val split must contain positives and negatives for calibration")
    model = get_model(kind, C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_val, y_val)

    metric_rows = []
    pred_frames = []
    coef_rows = []
    pep_frames = []

    # Coefficients in standardised space; useful for relative feature influence.
    clf = model.named_steps["clf"]
    if hasattr(clf, "coef_"):
        coefs = np.ravel(clf.coef_)
        for f, b in zip(features, coefs):
            coef_rows.append({"model_group": name, "feature": f, "coef_standardised": float(b), "abs_coef": float(abs(b))})

    for split, df in tables.items():
        X = df[features]
        y = df["label"].to_numpy(int)
        score = predict_score(model, X, kind=kind)
        peptide = df["peptide"].to_numpy(str)
        m = metric_summary(y, score, peptide, max_fpr=max_fpr)
        metric_rows.append({
            "model_group": name,
            "fit_split": "val",
            "eval_split": split,
            "n_features": len(features),
            "model_kind": kind,
            **m,
        })
        pred_frames.append(pd.DataFrame({
            "split": split,
            "pair_id": df["pair_id"].astype(str),
            "peptide": df["peptide"].astype(str),
            "label": y,
            "model_group": name,
            "score": score,
        }))
        pep_tab, _ = per_peptide_metrics(y, score, peptide, max_fpr=max_fpr)
        if len(pep_tab):
            pep_tab.insert(0, "model_group", name)
            pep_tab.insert(1, "eval_split", split)
            pep_frames.append(pep_tab)

    return metric_rows, pred_frames, coef_rows, pep_frames


def add_delta_columns(metrics: pd.DataFrame, baseline_group: str = "esm_only") -> pd.DataFrame:
    """Add deltas versus ESM-only and ESM+confidence baselines when available."""
    out = metrics.copy()
    if len(out) == 0 or "model_group" not in out.columns:
        return out

    metric_cols = [
        "auroc", "auprc", "auc0.1_mcclish", "pep_weighted_auroc", "pep_macro_auc0.1_mcclish"
    ]
    metric_cols = [c for c in metric_cols if c in out.columns]

    if baseline_group in set(out["model_group"]):
        base = out[out["model_group"] == baseline_group][["eval_split"] + metric_cols].copy()
        base = base.rename(columns={c: f"{baseline_group}__{c}" for c in metric_cols})
        out = out.merge(base, on="eval_split", how="left")
        for c in metric_cols:
            b = f"{baseline_group}__{c}"
            if b in out.columns:
                out[f"delta_vs_{baseline_group}__{c}"] = out[c] - out[b]

    conf_group = "esm_plus_confidence"
    if conf_group in set(out["model_group"]):
        base2 = out[out["model_group"] == conf_group][["eval_split"] + metric_cols].copy()
        base2 = base2.rename(columns={c: f"{conf_group}__{c}" for c in metric_cols})
        out = out.merge(base2, on="eval_split", how="left")
        for c in metric_cols:
            b = f"{conf_group}__{c}"
            if b in out.columns:
                out[f"delta_vs_{conf_group}__{c}"] = out[c] - out[b]
    return out


# -----------------------------------------------------------------------------
# Config/main
# -----------------------------------------------------------------------------

@dataclass
class RunConfig:
    repo_root: str = "/home/natasha/multimodal_model"
    out_dir: str = "/home/natasha/multimodal_model/outputs_data/boltz_signal_injection"

    esm_val_pred: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple/tulip_decoys_plain_vicreg__seed31__lr0.0003__a25.0__b25.0__dlt1.0__val_predictions.csv"
    esm_test_pred: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple/tulip_decoys_plain_vicreg__seed31__lr0.0003__a25.0__b25.0__dlt1.0__test_predictions.csv"
    esm_immrep_pred: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple/tulip_decoys_plain_vicreg__seed31__exported_immrep_from_checkpoint__immrep_test_predictions.csv"

    z_val_features: str = "/home/natasha/multimodal_model/outputs_data/z_block_features/val_z_block_features.csv"
    z_test_features: str = "/home/natasha/multimodal_model/outputs_data/z_block_features/test_z_block_features.csv"
    z_immrep_features: str = "/home/natasha/multimodal_model/outputs_data/z_block_features/immrep_test_z_block_features.csv"

    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"

    model_kind: str = "logreg"
    C: float = 0.1
    max_iter: int = 5000
    partial_auc_max_fpr: float = 0.1
    random_state: int = 31

    # Use only intersection rows across ESM prediction table and z features.
    require_z_features: bool = True
    extract_confidence_json: bool = True


def parse_args() -> RunConfig:
    defaults = asdict(RunConfig())
    p = argparse.ArgumentParser()
    for k, v in defaults.items():
        arg = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            p.add_argument(arg, action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, default=v)
    return RunConfig(**vars(p.parse_args()))


def main():
    cfg = parse_args()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 90, flush=True)
    print("Boltz signal injection assessment", flush=True)
    print(json.dumps(asdict(cfg), indent=2), flush=True)
    print("=" * 90, flush=True)

    # Load split tables.
    esm = {
        "val": normalise_esm_predictions(cfg.esm_val_pred, "val"),
        "test": normalise_esm_predictions(cfg.esm_test_pred, "test"),
        "immrep_test": normalise_esm_predictions(cfg.esm_immrep_pred, "immrep_test"),
    }
    z = {
        "val": load_z_features(cfg.z_val_features, "val"),
        "test": load_z_features(cfg.z_test_features, "test"),
        "immrep_test": load_z_features(cfg.z_immrep_features, "immrep_test"),
    }
    manifest = {
        "val": load_manifest(cfg.val_csv, "val"),
        "test": load_manifest(cfg.test_csv, "test"),
        "immrep_test": load_manifest(cfg.immrep_csv, "immrep_test"),
    }
    conf = {
        split: load_confidence_json_features(manifest[split], cfg.repo_root, split)
        for split in ["val", "test", "immrep_test"]
    } if cfg.extract_confidence_json else {split: pd.DataFrame(columns=["pair_id"]) for split in ["val", "test", "immrep_test"]}

    tables = {}
    for split in ["val", "test", "immrep_test"]:
        tab = build_split_table(esm[split], z[split], manifest[split], conf[split])
        before = len(tab)
        if cfg.require_z_features:
            # Keep rows that matched z features. Any z__ column non-null suffices.
            zcols = [c for c in tab.columns if c.startswith("z__")]
            if zcols:
                tab = tab[tab[zcols].notna().any(axis=1)].copy()
        # Drop rows without ESM score or label.
        tab = tab[np.isfinite(tab["esm_score"].to_numpy())].copy()
        tables[split] = tab.reset_index(drop=True)
        print(
            f"{split}: rows before={before} | rows used={len(tab)} | "
            f"n_pos={int(tab['label'].sum())} | n_neg={int((tab['label']==0).sum())}",
            flush=True,
        )

    val_labels = tables["val"]["label"].to_numpy(int)
    if len(np.unique(val_labels)) < 2:
        raise RuntimeError(
            "The joined VAL table contains only one class after matching ESM predictions to z features. "
            "This usually means the ESM prediction CSVs were generated on a different val/test CSV "
            "than the z-block features. Export aligned ESM predictions on val_multiview/test_multiview/"
            "immrep_test_multiview first using export_esm_vicreg_aligned_predictions.py, then rerun this script. "
            f"Current VAL counts: n_pos={int(val_labels.sum())}, n_neg={int((val_labels==0).sum())}."
        )

    groups = build_feature_groups(tables["val"])
    with open(out_dir / "feature_group_manifest.json", "w") as f:
        json.dump({k: {"n": len(v), "features": v} for k, v in groups.items()}, f, indent=2)

    print("\nFeature groups:", flush=True)
    for k, v in groups.items():
        print(f"  {k}: {len(v)}", flush=True)

    # Fit all groups on val and evaluate on all splits.
    all_metric_rows: List[Dict] = []
    all_pred_frames: List[pd.DataFrame] = []
    all_coef_rows: List[Dict] = []
    all_pep_frames: List[pd.DataFrame] = []

    for name, features in groups.items():
        # Skip huge all-z models if they are empty or impossible.
        features = [f for f in features if f in tables["val"].columns]
        if not features:
            continue
        try:
            metric_rows, pred_frames, coef_rows, pep_frames = fit_and_eval_group(
                name=name,
                features=features,
                tables=tables,
                kind=cfg.model_kind,
                C=cfg.C,
                max_iter=cfg.max_iter,
                max_fpr=cfg.partial_auc_max_fpr,
                random_state=cfg.random_state,
            )
            all_metric_rows.extend(metric_rows)
            all_pred_frames.extend(pred_frames)
            all_coef_rows.extend(coef_rows)
            all_pep_frames.extend(pep_frames)
            imm = [r for r in metric_rows if r["eval_split"] == "immrep_test"][0]
            print(
                f"{name:32s} | n={len(features):4d} | "
                f"IMMREP auroc={imm['auroc']:.4f} | auprc={imm['auprc']:.4f} | "
                f"mcclish={imm['auc0.1_mcclish']:.4f} | pep_w={imm['pep_weighted_auroc']:.4f}",
                flush=True,
            )
        except Exception as e:
            print(f"[warn] failed group {name}: {type(e).__name__}: {e}", flush=True)

    metrics = pd.DataFrame(all_metric_rows)
    metrics = add_delta_columns(metrics, baseline_group="esm_only")
    metrics.to_csv(out_dir / "model_group_metrics.csv", index=False)

    preds = pd.concat(all_pred_frames, ignore_index=True) if all_pred_frames else pd.DataFrame()
    preds.to_csv(out_dir / "model_group_predictions.csv", index=False)

    coefs = pd.DataFrame(all_coef_rows)
    if len(coefs):
        coefs = coefs.sort_values(["model_group", "abs_coef"], ascending=[True, False])
    coefs.to_csv(out_dir / "model_group_coefficients.csv", index=False)

    pep = pd.concat(all_pep_frames, ignore_index=True) if all_pep_frames else pd.DataFrame()
    pep.to_csv(out_dir / "per_peptide_metrics_by_model.csv", index=False)

    # Useful focused views.
    if len(metrics):
        for split in ["val", "test", "immrep_test"]:
            view = metrics[metrics["eval_split"] == split].copy()
            view = view.sort_values(["pep_weighted_auroc", "auc0.1_mcclish", "auroc"], ascending=False)
            view.to_csv(out_dir / f"model_group_metrics__{split}.csv", index=False)

        imm = metrics[metrics["eval_split"] == "immrep_test"].copy()
        imm = imm.sort_values(["pep_weighted_auroc", "auc0.1_mcclish", "auroc"], ascending=False)
        print("\nTop IMMREP models by peptide-weighted AUROC:", flush=True)
        cols = [
            "model_group", "n_features", "auroc", "auprc", "auc0.1_mcclish",
            "pep_weighted_auroc", "pep_macro_auc0.1_mcclish",
            "delta_vs_esm_only__pep_weighted_auroc", "delta_vs_esm_only__auc0.1_mcclish",
        ]
        cols = [c for c in cols if c in imm.columns]
        print(imm[cols].head(20).to_string(index=False), flush=True)

    summary = {
        "config": asdict(cfg),
        "n_rows": {k: int(len(v)) for k, v in tables.items()},
        "n_groups": int(len(groups)),
        "outputs": {
            "metrics": str(out_dir / "model_group_metrics.csv"),
            "predictions": str(out_dir / "model_group_predictions.csv"),
            "coefficients": str(out_dir / "model_group_coefficients.csv"),
            "per_peptide": str(out_dir / "per_peptide_metrics_by_model.csv"),
            "feature_groups": str(out_dir / "feature_group_manifest.json"),
        },
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 90, flush=True)
    print(f"Done. Outputs written to: {out_dir}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
