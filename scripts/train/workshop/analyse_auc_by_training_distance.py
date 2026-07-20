#!/usr/bin/env python3
"""
AUROC versus peptide distance from the training set.

This is a read-only post-hoc analysis for the NeurIPS 2026 workshop runs. It:

1. Reads the positive training peptide set from train_multiview.csv.
2. Loads pair-level predictions from each workshop run and seed.
3. Computes each evaluation peptide's minimum Levenshtein distance to training.
4. Computes AUROC and standardised McClish AUC0.1 separately for each peptide.
5. Aggregates peptide-level performance by absolute and normalised distance.
6. Bootstraps confidence intervals by resampling peptides.
7. Produces test and IMMREP figures, plus learned-minus-baseline delta analyses.

Expected prediction layout:
  {outputs_root}/{run_name}/seed_{seed}/{split}_predictions.csv

Expected prediction columns:
  pair_id, peptide, label,
  {model}_mse_distance,
  optionally {model}_score

Default outputs:
  Tables:
    /home/natasha/multimodal_model/models/outputs/workshop/
      paper_analysis/distance_to_training/

  Figures:
    /home/natasha/multimodal_model/models/figures/workshop/
      paper_analysis/distance_to_training/

Notes
-----
- Distance is defined using training peptides only. Validation peptides are not
  included in the reference set.
- A distance of zero means the peptide was observed in training.
- AUROC is calculated per peptide, then averaged equally over peptides.
- Smaller MSE distance means stronger predicted binding, so when a saved score
  is unavailable the script uses score = -mse_distance.
- The bootstrap unit is the peptide, not the TCR-pMHC pair.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]

MODEL_LABELS = {
    "onehot_composition": "One-hot composition",
    "pretrained_esmc_meanpool": "Raw ESMC mean pool",
    "finetuned_esmc_meanpool": "LoRA-ESMC mean pool",
    "onehot_vicreg": "One-hot + VICReg",
    "raw_esmc_vicreg": "Raw ESMC + VICReg",
    "finetuned_esmc_vicreg": "LoRA-ESMC + VICReg",
}

DELTA_PAIRS = [
    ("onehot_vicreg", "onehot_composition", "One-hot: VICReg minus raw"),
    ("raw_esmc_vicreg", "pretrained_esmc_meanpool", "Raw ESMC: VICReg minus raw"),
    ("finetuned_esmc_vicreg", "finetuned_esmc_meanpool", "LoRA-ESMC: VICReg minus raw"),
]


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def canonicalise_peptide(value: object) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).strip().upper().split())


def infer_column(df: pd.DataFrame, requested: Optional[str], candidates: Sequence[str], role: str) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise KeyError(
                f"Requested {role} column '{requested}' was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return requested

    lower_to_original = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return str(lower_to_original[candidate.lower()])

    raise KeyError(
        f"Could not infer the {role} column. Tried {list(candidates)}. "
        f"Available columns: {list(df.columns)}"
    )


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & pd.notna(labels)
    labels = labels[mask].astype(int)
    scores = scores[mask]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_mcclish_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    mask = np.isfinite(scores) & pd.notna(labels)
    labels = labels[mask].astype(int)
    scores = scores[mask]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def is_raw_run(run_name: str) -> bool:
    s = run_name.lower()
    return ("raw" in s or "pretrained" in s) and "finetuned" not in s


def is_finetuned_run(run_name: str) -> bool:
    s = run_name.lower()
    return "finetuned" in s or "lora" in s or "adapted" in s


def alias_model(run_name: str, model_name: str) -> str:
    """Match the paper-level aliases used by analyse_workshop_paper_results.py."""
    rn = run_name.lower()
    m = model_name.lower()

    if m == "esm_vicreg":
        if is_raw_run(rn):
            return "raw_esmc_vicreg"
        if is_finetuned_run(rn):
            return "finetuned_esmc_vicreg"
        return f"{run_name}__esm_vicreg"

    if m in {
        "onehot_vicreg",
        "onehot_composition",
        "pretrained_esmc_meanpool",
        "finetuned_esmc_meanpool",
    }:
        return m

    if m in {"raw_esmc_vicreg", "pretrained_esmc_vicreg"}:
        return "raw_esmc_vicreg"
    if m in {"lora_esmc_vicreg", "adapted_esmc_vicreg"}:
        return "finetuned_esmc_vicreg"

    return model_name


def should_skip_prediction_model(
    run_name: str,
    model_name: str,
    skip_duplicate_baselines_from_raw_runs: bool,
) -> bool:
    if not skip_duplicate_baselines_from_raw_runs:
        return False
    if not is_raw_run(run_name):
        return False
    return model_name in {"finetuned_esmc_meanpool", "pretrained_esmc_meanpool"}


def stable_seed(*parts: object, base_seed: int = 31) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return (base_seed + zlib.crc32(text)) % (2**32 - 1)


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------

try:
    from rapidfuzz.distance import Levenshtein as _RapidLevenshtein

    @lru_cache(maxsize=None)
    def levenshtein_distance(a: str, b: str) -> int:
        return int(_RapidLevenshtein.distance(a, b))

    LEVENSHTEIN_BACKEND = "rapidfuzz"

except Exception:

    @lru_cache(maxsize=None)
    def levenshtein_distance(a: str, b: str) -> int:
        """Memory-efficient pure-Python Levenshtein distance fallback."""
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        if len(a) < len(b):
            a, b = b, a

        previous = list(range(len(b) + 1))
        for i, char_a in enumerate(a, start=1):
            current = [i]
            for j, char_b in enumerate(b, start=1):
                insertion = current[j - 1] + 1
                deletion = previous[j] + 1
                substitution = previous[j - 1] + (char_a != char_b)
                current.append(min(insertion, deletion, substitution))
            previous = current
        return int(previous[-1])

    LEVENSHTEIN_BACKEND = "python_fallback"


def read_training_peptides(
    train_csv: Path,
    peptide_col: Optional[str],
    label_col: Optional[str],
) -> Tuple[List[str], str]:
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    pep_col = infer_column(
        df,
        peptide_col,
        candidates=["peptide", "pep", "epitope", "peptide_seq", "peptide_sequence"],
        role="training peptide",
    )

    # The reference set should contain observed positive training peptides only.
    if label_col is not None:
        if label_col not in df.columns:
            raise KeyError(f"Training label column '{label_col}' not found.")
        labels = pd.to_numeric(df[label_col], errors="coerce")
        df = df[labels == 1].copy()
    else:
        inferred_label = None
        for candidate in ["label", "target", "binding", "binder"]:
            if candidate in df.columns:
                inferred_label = candidate
                break
        if inferred_label is not None:
            labels = pd.to_numeric(df[inferred_label], errors="coerce")
            if (labels == 0).any() and (labels == 1).any():
                warnings.warn(
                    f"Training CSV contains both classes in '{inferred_label}'. "
                    "Only label==1 rows will define the training peptide set."
                )
                df = df[labels == 1].copy()

    peptides = sorted({
        canonicalise_peptide(x)
        for x in df[pep_col].tolist()
        if canonicalise_peptide(x)
    })
    if not peptides:
        raise ValueError(f"No valid training peptides found in {train_csv} column '{pep_col}'.")

    return peptides, pep_col


def compute_nearest_training_distances(
    evaluation_peptides: Sequence[str],
    train_peptides: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict] = []
    eval_unique = sorted({canonicalise_peptide(p) for p in evaluation_peptides if canonicalise_peptide(p)})

    print(
        f"Computing distances for {len(eval_unique)} unique evaluation peptides "
        f"against {len(train_peptides)} unique training peptides "
        f"using {LEVENSHTEIN_BACKEND}.",
        flush=True,
    )

    for index, peptide in enumerate(eval_unique, start=1):
        best_abs = math.inf
        best_abs_peptide = ""
        best_norm = math.inf
        best_norm_peptide = ""
        best_norm_abs_distance = math.inf

        for train_peptide in train_peptides:
            distance = levenshtein_distance(peptide, train_peptide)
            denominator = max(len(peptide), len(train_peptide))
            normalised = distance / denominator if denominator > 0 else float("nan")

            # Deterministic tie-break: shorter nearest peptide, then lexicographic order.
            abs_key = (distance, abs(len(peptide) - len(train_peptide)), train_peptide)
            current_abs_key = (
                best_abs,
                abs(len(peptide) - len(best_abs_peptide)) if best_abs_peptide else math.inf,
                best_abs_peptide,
            )
            if abs_key < current_abs_key:
                best_abs = distance
                best_abs_peptide = train_peptide

            norm_key = (normalised, distance, train_peptide)
            current_norm_key = (best_norm, best_norm_abs_distance, best_norm_peptide)
            if norm_key < current_norm_key:
                best_norm = normalised
                best_norm_abs_distance = distance
                best_norm_peptide = train_peptide

        rows.append({
            "peptide": peptide,
            "nearest_train_peptide_absolute": best_abs_peptide,
            "distance_absolute": int(best_abs),
            "nearest_train_peptide_normalised": best_norm_peptide,
            "distance_absolute_at_normalised_minimum": int(best_norm_abs_distance),
            "distance_normalised": float(best_norm),
            "peptide_seen_in_train": bool(best_abs == 0),
            "evaluation_peptide_length": int(len(peptide)),
            "nearest_absolute_train_peptide_length": int(len(best_abs_peptide)),
            "nearest_normalised_train_peptide_length": int(len(best_norm_peptide)),
        })

        if index % 25 == 0 or index == len(eval_unique):
            print(f"  distance lookup: {index}/{len(eval_unique)} peptides", flush=True)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------

def prediction_path(outputs_root: Path, run_name: str, seed: int, split: str) -> Path:
    return outputs_root / run_name / f"seed_{seed}" / f"{split}_predictions.csv"


def read_prediction_models(
    path: Path,
    run_name: str,
    skip_duplicate_baselines_from_raw_runs: bool,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"pair_id", "peptide", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    split = path.name.replace("_predictions.csv", "")
    seed_text = path.parent.name.replace("seed_", "")
    seed = int(seed_text) if seed_text.isdigit() else np.nan

    frames: List[pd.DataFrame] = []
    for distance_col in df.columns:
        if not distance_col.endswith("_mse_distance"):
            continue

        raw_model_name = distance_col[:-len("_mse_distance")]
        if should_skip_prediction_model(
            run_name,
            raw_model_name,
            skip_duplicate_baselines_from_raw_runs,
        ):
            continue

        model_name = alias_model(run_name, raw_model_name)
        score_col = f"{raw_model_name}_score"

        mse_distance = pd.to_numeric(df[distance_col], errors="coerce")
        if score_col in df.columns:
            score = pd.to_numeric(df[score_col], errors="coerce")
            score_source = score_col
        else:
            score = -mse_distance
            score_source = f"-{distance_col}"

        frame = pd.DataFrame({
            "run_name": run_name,
            "seed": seed,
            "split": split,
            "model_name": model_name,
            "raw_model_name": raw_model_name,
            "pair_id": df["pair_id"].astype(str),
            "peptide": df["peptide"].map(canonicalise_peptide),
            "label": pd.to_numeric(df["label"], errors="coerce"),
            "mse_distance": mse_distance,
            "score": score,
            "score_source": score_source,
            "prediction_path": str(path),
        })
        frame = frame.dropna(subset=["label", "score"])
        frame = frame[frame["peptide"] != ""].copy()
        frame["label"] = frame["label"].astype(int)
        frames.append(frame)

    if not frames:
        warnings.warn(f"No '*_mse_distance' columns found in {path}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def collect_predictions(
    outputs_root: Path,
    run_names: Sequence[str],
    seeds: Sequence[int],
    splits: Sequence[str],
    skip_duplicate_baselines_from_raw_runs: bool,
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    run_priority = {name: index for index, name in enumerate(run_names)}

    for run_name in run_names:
        for seed in seeds:
            for split in splits:
                path = prediction_path(outputs_root, run_name, seed, split)
                if not path.exists():
                    warnings.warn(f"Missing predictions: {path}")
                    continue
                frames.append(
                    read_prediction_models(
                        path,
                        run_name,
                        skip_duplicate_baselines_from_raw_runs,
                    )
                )

    if not frames:
        raise FileNotFoundError(
            "No prediction rows were loaded. Check --outputs-root, --run-names, "
            "--seeds and --splits."
        )

    predictions = pd.concat(frames, ignore_index=True)
    predictions["_run_priority"] = predictions["run_name"].map(run_priority).fillna(999)

    # Baselines can be emitted by more than one run. Keep one copy per
    # model/seed/split/pair, but warn if duplicate copies disagree.
    key_cols = ["model_name", "seed", "split", "pair_id"]
    duplicate_mask = predictions.duplicated(key_cols, keep=False)
    if duplicate_mask.any():
        duplicate_rows = predictions.loc[duplicate_mask]
        conflict_count = 0
        for _, group in duplicate_rows.groupby(key_cols, dropna=False):
            finite_scores = group["score"].to_numpy(float)
            finite_scores = finite_scores[np.isfinite(finite_scores)]
            if len(finite_scores) > 1 and np.ptp(finite_scores) > 1e-8:
                conflict_count += 1
        if conflict_count:
            warnings.warn(
                f"{conflict_count} duplicated model/seed/split/pair groups had "
                "non-identical scores. The earliest run in --run-names was retained."
            )

    predictions = (
        predictions
        .sort_values(key_cols + ["_run_priority"])
        .drop_duplicates(key_cols, keep="first")
        .drop(columns="_run_priority")
        .reset_index(drop=True)
    )
    return predictions


# ---------------------------------------------------------------------------
# Per-peptide metrics and binning
# ---------------------------------------------------------------------------

def compute_per_peptide_seed_metrics(
    predictions: pd.DataFrame,
    distance_lookup: pd.DataFrame,
    min_positives: int,
    min_negatives: int,
) -> pd.DataFrame:
    rows: List[Dict] = []

    for (model, seed, split, peptide), group in predictions.groupby(
        ["model_name", "seed", "split", "peptide"],
        dropna=False,
        sort=False,
    ):
        labels = group["label"].to_numpy(int)
        scores = group["score"].to_numpy(float)
        n_positive = int((labels == 1).sum())
        n_negative = int((labels == 0).sum())

        valid = n_positive >= min_positives and n_negative >= min_negatives
        rows.append({
            "model_name": model,
            "seed": int(seed),
            "split": split,
            "peptide": peptide,
            "n_pairs": int(len(group)),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "valid_for_auc": bool(valid),
            "auc": safe_auroc(labels, scores) if valid else np.nan,
            "auc0.1_mcclish": safe_mcclish_auc(labels, scores) if valid else np.nan,
            "score_source": ",".join(sorted(set(group["score_source"].astype(str)))),
            "run_names": ",".join(sorted(set(group["run_name"].astype(str)))),
        })

    per_seed = pd.DataFrame(rows)
    per_seed = per_seed.merge(distance_lookup, on="peptide", how="left", validate="many_to_one")

    if per_seed["distance_absolute"].isna().any():
        missing = sorted(per_seed.loc[per_seed["distance_absolute"].isna(), "peptide"].unique())
        raise RuntimeError(f"Distance lookup failed for evaluation peptides: {missing[:10]}")

    return per_seed


def average_per_peptide_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []
    for (model, split, peptide), group in per_seed.groupby(
        ["model_name", "split", "peptide"],
        sort=False,
        dropna=False,
    ):
        valid_auc = group["auc"].dropna()
        valid_partial = group["auc0.1_mcclish"].dropna()
        first = group.iloc[0]

        rows.append({
            "model_name": model,
            "split": split,
            "peptide": peptide,
            "auc": float(valid_auc.mean()) if len(valid_auc) else np.nan,
            "auc_seed_std": float(valid_auc.std(ddof=1)) if len(valid_auc) > 1 else 0.0,
            "auc0.1_mcclish": float(valid_partial.mean()) if len(valid_partial) else np.nan,
            "auc0.1_mcclish_seed_std": (
                float(valid_partial.std(ddof=1)) if len(valid_partial) > 1 else 0.0
            ),
            "n_seeds_auc": int(valid_auc.shape[0]),
            "n_seeds_auc0.1": int(valid_partial.shape[0]),
            "n_pairs": int(first["n_pairs"]),
            "n_positive": int(first["n_positive"]),
            "n_negative": int(first["n_negative"]),
            "distance_absolute": int(first["distance_absolute"]),
            "distance_normalised": float(first["distance_normalised"]),
            "nearest_train_peptide_absolute": first["nearest_train_peptide_absolute"],
            "nearest_train_peptide_normalised": first["nearest_train_peptide_normalised"],
            "peptide_seen_in_train": bool(first["peptide_seen_in_train"]),
            "evaluation_peptide_length": int(first["evaluation_peptide_length"]),
        })

    return pd.DataFrame(rows)


def add_distance_bins(
    df: pd.DataFrame,
    absolute_max_bin: int,
    normalised_edges: Sequence[float],
) -> pd.DataFrame:
    out = df.copy()

    abs_distance = pd.to_numeric(out["distance_absolute"], errors="coerce")
    out["absolute_bin_order"] = np.where(
        abs_distance <= absolute_max_bin,
        abs_distance,
        absolute_max_bin + 1,
    ).astype(int)
    out["absolute_bin"] = out["absolute_bin_order"].map(
        lambda x: str(x) if x <= absolute_max_bin else f"{absolute_max_bin + 1}+"
    )

    edges = list(map(float, normalised_edges))
    if len(edges) < 2 or edges[0] > 0 or edges[-1] < 1:
        raise ValueError(
            "--normalised-edges must begin at 0 (or lower) and end at 1 (or higher)."
        )
    if any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise ValueError("--normalised-edges must be strictly increasing.")

    labels = []
    for left, right in zip(edges[:-1], edges[1:]):
        left_pct = int(round(left * 100))
        right_pct = int(round(right * 100))
        labels.append(f"{left_pct}-{right_pct}%")

    out["normalised_bin"] = pd.cut(
        pd.to_numeric(out["distance_normalised"], errors="coerce"),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    # Include exact 1.0 in the final bin when the final edge is exactly 1.0.
    exact_upper = pd.to_numeric(out["distance_normalised"], errors="coerce") == edges[-1]
    if exact_upper.any():
        out.loc[exact_upper, "normalised_bin"] = labels[-1]

    out["normalised_bin"] = out["normalised_bin"].astype(str)
    normalised_order = {label: index for index, label in enumerate(labels)}
    out["normalised_bin_order"] = out["normalised_bin"].map(normalised_order)

    return out


def bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap_reps: int,
    confidence_level: float,
    seed: int,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")

    mean = float(np.mean(values))
    if len(values) == 1 or bootstrap_reps <= 0:
        return mean, mean, mean

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        len(values),
        size=(bootstrap_reps, len(values)),
    )
    bootstrap_means = values[sampled_indices].mean(axis=1)
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(bootstrap_means, alpha / 2))
    upper = float(np.quantile(bootstrap_means, 1 - alpha / 2))
    return mean, lower, upper


def summarise_binned_metrics(
    peptide_mean: pd.DataFrame,
    bin_kind: str,
    bootstrap_reps: int,
    confidence_level: float,
    random_seed: int,
) -> pd.DataFrame:
    if bin_kind == "absolute":
        bin_col = "absolute_bin"
        order_col = "absolute_bin_order"
    elif bin_kind == "normalised":
        bin_col = "normalised_bin"
        order_col = "normalised_bin_order"
    else:
        raise ValueError("bin_kind must be 'absolute' or 'normalised'.")

    rows: List[Dict] = []
    metrics = ["auc", "auc0.1_mcclish"]

    grouped = peptide_mean.groupby(
        ["model_name", "split", bin_col, order_col],
        dropna=False,
        sort=False,
    )
    for (model, split, bin_label, bin_order), group in grouped:
        if pd.isna(bin_order) or str(bin_label) == "nan":
            continue

        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            mean, lower, upper = bootstrap_mean_ci(
                values,
                bootstrap_reps=bootstrap_reps,
                confidence_level=confidence_level,
                seed=stable_seed(
                    model,
                    split,
                    bin_kind,
                    bin_label,
                    metric,
                    base_seed=random_seed,
                ),
            )
            rows.append({
                "model_name": model,
                "split": split,
                "distance_kind": bin_kind,
                "distance_bin": str(bin_label),
                "distance_bin_order": int(bin_order),
                "metric": metric,
                "mean": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_peptides": int(len(values)),
                "confidence_level": float(confidence_level),
                "bootstrap_reps": int(bootstrap_reps),
            })

    return pd.DataFrame(rows)


def summarise_seed_level_bins(per_seed: pd.DataFrame, bin_kind: str) -> pd.DataFrame:
    if bin_kind == "absolute":
        bin_col = "absolute_bin"
        order_col = "absolute_bin_order"
    else:
        bin_col = "normalised_bin"
        order_col = "normalised_bin_order"

    rows: List[Dict] = []
    for (model, seed, split, bin_label, bin_order), group in per_seed.groupby(
        ["model_name", "seed", "split", bin_col, order_col],
        dropna=False,
        sort=False,
    ):
        if pd.isna(bin_order) or str(bin_label) == "nan":
            continue
        for metric in ["auc", "auc0.1_mcclish"]:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            rows.append({
                "model_name": model,
                "seed": int(seed),
                "split": split,
                "distance_kind": bin_kind,
                "distance_bin": str(bin_label),
                "distance_bin_order": int(bin_order),
                "metric": metric,
                "mean": float(values.mean()) if len(values) else np.nan,
                "n_peptides": int(len(values)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Learned-minus-baseline deltas
# ---------------------------------------------------------------------------

def compute_model_deltas(peptide_mean: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    index_cols = [
        "split",
        "peptide",
        "distance_absolute",
        "distance_normalised",
        "absolute_bin",
        "absolute_bin_order",
        "normalised_bin",
        "normalised_bin_order",
    ]

    for learned, baseline, label in DELTA_PAIRS:
        learned_df = peptide_mean[peptide_mean["model_name"] == learned].copy()
        baseline_df = peptide_mean[peptide_mean["model_name"] == baseline].copy()
        if learned_df.empty or baseline_df.empty:
            continue

        keep_learned = index_cols + ["auc", "auc0.1_mcclish"]
        keep_baseline = ["split", "peptide", "auc", "auc0.1_mcclish"]

        merged = learned_df[keep_learned].merge(
            baseline_df[keep_baseline],
            on=["split", "peptide"],
            how="inner",
            suffixes=("_learned", "_baseline"),
            validate="one_to_one",
        )
        if merged.empty:
            continue

        merged["learned_model"] = learned
        merged["baseline_model"] = baseline
        merged["delta_label"] = label
        merged["delta_auc"] = merged["auc_learned"] - merged["auc_baseline"]
        merged["delta_auc0.1_mcclish"] = (
            merged["auc0.1_mcclish_learned"] - merged["auc0.1_mcclish_baseline"]
        )
        rows.append(merged)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise_delta_bins(
    delta_df: pd.DataFrame,
    bin_kind: str,
    bootstrap_reps: int,
    confidence_level: float,
    random_seed: int,
) -> pd.DataFrame:
    if delta_df.empty:
        return pd.DataFrame()

    if bin_kind == "absolute":
        bin_col = "absolute_bin"
        order_col = "absolute_bin_order"
    else:
        bin_col = "normalised_bin"
        order_col = "normalised_bin_order"

    metric_map = {
        "delta_auc": "auc",
        "delta_auc0.1_mcclish": "auc0.1_mcclish",
    }
    rows: List[Dict] = []

    for (
        learned,
        baseline,
        delta_label,
        split,
        bin_label,
        bin_order,
    ), group in delta_df.groupby(
        [
            "learned_model",
            "baseline_model",
            "delta_label",
            "split",
            bin_col,
            order_col,
        ],
        dropna=False,
        sort=False,
    ):
        if pd.isna(bin_order) or str(bin_label) == "nan":
            continue

        for delta_col, metric in metric_map.items():
            values = pd.to_numeric(group[delta_col], errors="coerce").dropna().to_numpy(float)
            mean, lower, upper = bootstrap_mean_ci(
                values,
                bootstrap_reps=bootstrap_reps,
                confidence_level=confidence_level,
                seed=stable_seed(
                    learned,
                    baseline,
                    split,
                    bin_kind,
                    bin_label,
                    metric,
                    base_seed=random_seed,
                ),
            )
            rows.append({
                "learned_model": learned,
                "baseline_model": baseline,
                "delta_label": delta_label,
                "split": split,
                "distance_kind": bin_kind,
                "distance_bin": str(bin_label),
                "distance_bin_order": int(bin_order),
                "metric": metric,
                "mean_delta": mean,
                "ci_lower": lower,
                "ci_upper": upper,
                "n_peptides": int(len(values)),
                "confidence_level": float(confidence_level),
                "bootstrap_reps": int(bootstrap_reps),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def ordered_models(values: Iterable[str]) -> List[str]:
    values = list(dict.fromkeys(values))
    return [m for m in MODEL_ORDER if m in values] + sorted(m for m in values if m not in MODEL_ORDER)


def model_style_map(models: Sequence[str]) -> Dict[str, Dict]:
    # Use Matplotlib's active default colour cycle, rather than hard-coding colours.
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    line_styles = {
        "onehot_composition": "--",
        "pretrained_esmc_meanpool": "--",
        "finetuned_esmc_meanpool": "--",
        "onehot_vicreg": "-",
        "raw_esmc_vicreg": "-",
        "finetuned_esmc_vicreg": "-",
    }
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    styles: Dict[str, Dict] = {}
    for index, model in enumerate(models):
        style = {
            "linestyle": line_styles.get(model, "-"),
            "marker": markers[index % len(markers)],
        }
        if colours:
            style["color"] = colours[index % len(colours)]
        styles[model] = style
    return styles


def plot_single_split_auc(
    summary: pd.DataFrame,
    split: str,
    distance_kind: str,
    metric: str,
    output_base: Path,
    y_min: float,
    y_max: float,
) -> None:
    subset = summary[
        (summary["split"] == split)
        & (summary["distance_kind"] == distance_kind)
        & (summary["metric"] == metric)
        & (summary["n_peptides"] > 0)
    ].copy()
    if subset.empty:
        return

    models = ordered_models(subset["model_name"].unique())
    styles = model_style_map(models)
    bin_table = (
        subset[["distance_bin", "distance_bin_order"]]
        .drop_duplicates()
        .sort_values("distance_bin_order")
    )
    bin_labels = bin_table["distance_bin"].tolist()
    x_map = {label: index for index, label in enumerate(bin_labels)}

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for model in models:
        group = subset[subset["model_name"] == model].sort_values("distance_bin_order")
        x = np.array([x_map[label] for label in group["distance_bin"]], dtype=float)
        y = group["mean"].to_numpy(float)
        lower = group["ci_lower"].to_numpy(float)
        upper = group["ci_upper"].to_numpy(float)
        yerr = np.vstack([y - lower, upper - y])

        ax.errorbar(
            x,
            y,
            yerr=yerr,
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABELS.get(model, model),
            **styles[model],
        )

    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Random")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("Peptide-level AUROC" if metric == "auc" else "Peptide-level McClish AUC0.1")
    ax.set_xlabel(
        "Minimum Levenshtein distance to a training peptide"
        if distance_kind == "absolute"
        else "Minimum normalised Levenshtein distance to training"
    )
    split_label = "IMMREP" if split == "immrep_test" else split.replace("_", " ").title()
    ax.set_title(split_label)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_auc(
    summary: pd.DataFrame,
    splits: Sequence[str],
    distance_kind: str,
    metric: str,
    output_base: Path,
    y_min: float,
    y_max: float,
) -> None:
    available_splits = [
        split
        for split in splits
        if not summary[
            (summary["split"] == split)
            & (summary["distance_kind"] == distance_kind)
            & (summary["metric"] == metric)
            & (summary["n_peptides"] > 0)
        ].empty
    ]
    if not available_splits:
        return

    models = ordered_models(
        summary[
            (summary["split"].isin(available_splits))
            & (summary["distance_kind"] == distance_kind)
            & (summary["metric"] == metric)
        ]["model_name"].unique()
    )
    styles = model_style_map(models)

    fig, axes = plt.subplots(
        1,
        len(available_splits),
        figsize=(7.1 * len(available_splits), 5.2),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    legend_handles = []
    legend_labels = []

    for ax, split in zip(axes, available_splits):
        subset = summary[
            (summary["split"] == split)
            & (summary["distance_kind"] == distance_kind)
            & (summary["metric"] == metric)
            & (summary["n_peptides"] > 0)
        ].copy()

        bin_table = (
            subset[["distance_bin", "distance_bin_order"]]
            .drop_duplicates()
            .sort_values("distance_bin_order")
        )
        bin_labels = bin_table["distance_bin"].tolist()
        x_map = {label: index for index, label in enumerate(bin_labels)}

        for model in models:
            group = subset[subset["model_name"] == model].sort_values("distance_bin_order")
            if group.empty:
                continue
            x = np.array([x_map[label] for label in group["distance_bin"]], dtype=float)
            y = group["mean"].to_numpy(float)
            lower = group["ci_lower"].to_numpy(float)
            upper = group["ci_upper"].to_numpy(float)
            yerr = np.vstack([y - lower, upper - y])

            handle = ax.errorbar(
                x,
                y,
                yerr=yerr,
                capsize=3,
                linewidth=1.5,
                markersize=5,
                label=MODEL_LABELS.get(model, model),
                **styles[model],
            )
            if model not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(model)

        ax.axhline(0.5, linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel(
            "Minimum edit distance to training"
            if distance_kind == "absolute"
            else "Minimum normalised edit distance to training"
        )
        split_label = "IMMREP" if split == "immrep_test" else split.replace("_", " ").title()
        ax.set_title(split_label)

    axes[0].set_ylabel(
        "Peptide-level AUROC"
        if metric == "auc"
        else "Peptide-level McClish AUC0.1"
    )

    display_labels = [MODEL_LABELS.get(model, model) for model in legend_labels]
    fig.legend(
        legend_handles,
        display_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, max(1, len(display_labels))),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_combined_delta(
    summary: pd.DataFrame,
    splits: Sequence[str],
    distance_kind: str,
    metric: str,
    output_base: Path,
) -> None:
    if summary.empty:
        return

    available_splits = [
        split
        for split in splits
        if not summary[
            (summary["split"] == split)
            & (summary["distance_kind"] == distance_kind)
            & (summary["metric"] == metric)
            & (summary["n_peptides"] > 0)
        ].empty
    ]
    if not available_splits:
        return

    labels = list(dict.fromkeys(summary["delta_label"].tolist()))
    styles = model_style_map(labels)

    fig, axes = plt.subplots(
        1,
        len(available_splits),
        figsize=(7.1 * len(available_splits), 5.2),
        squeeze=False,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, split in zip(axes, available_splits):
        subset = summary[
            (summary["split"] == split)
            & (summary["distance_kind"] == distance_kind)
            & (summary["metric"] == metric)
            & (summary["n_peptides"] > 0)
        ].copy()

        bin_table = (
            subset[["distance_bin", "distance_bin_order"]]
            .drop_duplicates()
            .sort_values("distance_bin_order")
        )
        bin_labels = bin_table["distance_bin"].tolist()
        x_map = {label: index for index, label in enumerate(bin_labels)}

        for label in labels:
            group = subset[subset["delta_label"] == label].sort_values("distance_bin_order")
            if group.empty:
                continue
            x = np.array([x_map[value] for value in group["distance_bin"]], dtype=float)
            y = group["mean_delta"].to_numpy(float)
            lower = group["ci_lower"].to_numpy(float)
            upper = group["ci_upper"].to_numpy(float)
            yerr = np.vstack([y - lower, upper - y])
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                capsize=3,
                linewidth=1.5,
                markersize=5,
                label=label,
                **styles[label],
            )

        ax.axhline(0.0, linestyle="--", linewidth=1.2)
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel(
            "Minimum edit distance to training"
            if distance_kind == "absolute"
            else "Minimum normalised edit distance to training"
        )
        split_label = "IMMREP" if split == "immrep_test" else split.replace("_", " ").title()
        ax.set_title(split_label)

    axes[0].set_ylabel(
        r"$\Delta$ peptide-level AUROC"
        if metric == "auc"
        else r"$\Delta$ peptide-level McClish AUC0.1"
    )
    handles, labels_display = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_display,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, max(1, len(labels_display))),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def update_manifest(manifest_path: Path, new_rows: List[Dict]) -> None:
    if not new_rows:
        return
    mkdir(manifest_path.parent)
    new_df = pd.DataFrame(new_rows)

    if manifest_path.exists():
        try:
            existing = pd.read_csv(manifest_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:
            warnings.warn(f"Could not read existing manifest {manifest_path}: {exc}")
            combined = new_df
    else:
        combined = new_df

    dedupe_cols = [c for c in ["artifact", "path"] if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(dedupe_cols, keep="last")
    combined.to_csv(manifest_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse peptide-level AUROC as a function of distance from training."
    )
    parser.add_argument(
        "--train-csv",
        default="/home/natasha/multimodal_model/data/train/train_multiview.csv",
    )
    parser.add_argument("--train-peptide-col", default=None)
    parser.add_argument(
        "--train-label-col",
        default=None,
        help="Optional. If supplied, only label==1 rows define the training peptide set.",
    )

    parser.add_argument(
        "--outputs-root",
        default="/home/natasha/multimodal_model/models/outputs/workshop",
    )
    parser.add_argument(
        "--analysis-out-dir",
        default=(
            "/home/natasha/multimodal_model/models/outputs/workshop/"
            "paper_analysis/distance_to_training"
        ),
    )
    parser.add_argument(
        "--analysis-fig-dir",
        default=(
            "/home/natasha/multimodal_model/models/figures/workshop/"
            "paper_analysis/distance_to_training"
        ),
    )
    parser.add_argument(
        "--manifest-path",
        default=(
            "/home/natasha/multimodal_model/models/outputs/workshop/"
            "paper_analysis/diagnostics_manifest.csv"
        ),
    )

    parser.add_argument(
        "--run-names",
        nargs="+",
        default=[
            "onehot_vicreg_complete",
            "esm_vicreg_raw_complete",
            "esm_vicreg_finetuned_complete",
        ],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43])
    parser.add_argument("--splits", nargs="+", default=["test", "immrep_test"])

    parser.add_argument("--min-positives", type=int, default=1)
    parser.add_argument("--min-negatives", type=int, default=1)
    parser.add_argument(
        "--absolute-max-bin",
        type=int,
        default=8,
        help="Distances above this value are pooled into the next '+' bin.",
    )
    parser.add_argument(
        "--normalised-edges",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0000001],
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=31)

    parser.add_argument(
        "--plot-metric",
        choices=["auc", "auc0.1_mcclish"],
        default="auc",
    )
    parser.add_argument("--plot-y-min", type=float, default=0.30)
    parser.add_argument("--plot-y-max", type=float, default=1.00)
    parser.add_argument(
        "--include-duplicate-baselines-from-raw-runs",
        action="store_true",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir = mkdir(Path(args.analysis_out_dir))
    fig_dir = mkdir(Path(args.analysis_fig_dir))
    manifest_path = Path(args.manifest_path)

    config = vars(args).copy()
    config["levenshtein_backend"] = LEVENSHTEIN_BACKEND
    config_path = out_dir / "distance_analysis_config.json"
    with open(config_path, "w") as handle:
        json.dump(config, handle, indent=2)

    print("[1/8] Reading positive training peptides", flush=True)
    train_peptides, train_peptide_col = read_training_peptides(
        Path(args.train_csv),
        peptide_col=args.train_peptide_col,
        label_col=args.train_label_col,
    )
    pd.DataFrame({"training_peptide": train_peptides}).to_csv(
        out_dir / "training_peptide_reference_set.csv",
        index=False,
    )
    print(
        f"Loaded {len(train_peptides)} unique training peptides "
        f"from column '{train_peptide_col}'.",
        flush=True,
    )

    print("[2/8] Loading pair-level predictions", flush=True)
    predictions = collect_predictions(
        outputs_root=outputs_root,
        run_names=args.run_names,
        seeds=args.seeds,
        splits=args.splits,
        skip_duplicate_baselines_from_raw_runs=(
            not args.include_duplicate_baselines_from_raw_runs
        ),
    )
    print(
        f"Loaded {len(predictions):,} pair-model prediction rows across "
        f"{predictions['model_name'].nunique()} paper-level models.",
        flush=True,
    )

    print("[3/8] Computing peptide distance to training", flush=True)
    distance_lookup = compute_nearest_training_distances(
        evaluation_peptides=predictions["peptide"].unique(),
        train_peptides=train_peptides,
    )
    split_presence = (
        predictions[["split", "peptide"]]
        .drop_duplicates()
        .assign(present=True)
        .pivot(index="peptide", columns="split", values="present")
        .fillna(False)
        .reset_index()
    )
    distance_lookup = distance_lookup.merge(
        split_presence,
        on="peptide",
        how="left",
        validate="one_to_one",
    )
    distance_lookup_path = out_dir / "peptide_distance_lookup.csv"
    distance_lookup.to_csv(distance_lookup_path, index=False)

    print("[4/8] Computing peptide-level AUROC for each seed", flush=True)
    per_seed = compute_per_peptide_seed_metrics(
        predictions=predictions,
        distance_lookup=distance_lookup.drop(
            columns=[c for c in args.splits if c in distance_lookup.columns]
        ),
        min_positives=args.min_positives,
        min_negatives=args.min_negatives,
    )
    per_seed = add_distance_bins(
        per_seed,
        absolute_max_bin=args.absolute_max_bin,
        normalised_edges=args.normalised_edges,
    )
    per_seed_path = out_dir / "per_peptide_seed_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)

    print("[5/8] Averaging peptide-level metrics across seeds", flush=True)
    peptide_mean = average_per_peptide_across_seeds(per_seed)
    peptide_mean = add_distance_bins(
        peptide_mean,
        absolute_max_bin=args.absolute_max_bin,
        normalised_edges=args.normalised_edges,
    )
    peptide_mean_path = out_dir / "per_peptide_metrics_across_seeds.csv"
    peptide_mean.to_csv(peptide_mean_path, index=False)

    print("[6/8] Aggregating and bootstrapping distance bins", flush=True)
    absolute_summary = summarise_binned_metrics(
        peptide_mean,
        bin_kind="absolute",
        bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )
    normalised_summary = summarise_binned_metrics(
        peptide_mean,
        bin_kind="normalised",
        bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )
    absolute_seed_summary = summarise_seed_level_bins(per_seed, "absolute")
    normalised_seed_summary = summarise_seed_level_bins(per_seed, "normalised")

    absolute_summary_path = out_dir / "binned_auc_absolute.csv"
    normalised_summary_path = out_dir / "binned_auc_normalised.csv"
    absolute_seed_path = out_dir / "binned_auc_absolute_seed.csv"
    normalised_seed_path = out_dir / "binned_auc_normalised_seed.csv"

    absolute_summary.to_csv(absolute_summary_path, index=False)
    normalised_summary.to_csv(normalised_summary_path, index=False)
    absolute_seed_summary.to_csv(absolute_seed_path, index=False)
    normalised_seed_summary.to_csv(normalised_seed_path, index=False)

    print("[7/8] Computing learned-minus-baseline deltas", flush=True)
    delta_df = compute_model_deltas(peptide_mean)
    delta_path = out_dir / "model_delta_per_peptide.csv"
    delta_df.to_csv(delta_path, index=False)

    delta_absolute = summarise_delta_bins(
        delta_df,
        bin_kind="absolute",
        bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )
    delta_normalised = summarise_delta_bins(
        delta_df,
        bin_kind="normalised",
        bootstrap_reps=args.bootstrap_reps,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
    )
    delta_absolute_path = out_dir / "model_delta_binned_absolute.csv"
    delta_normalised_path = out_dir / "model_delta_binned_normalised.csv"
    delta_absolute.to_csv(delta_absolute_path, index=False)
    delta_normalised.to_csv(delta_normalised_path, index=False)

    print("[8/8] Producing figures", flush=True)
    figures_created: List[Path] = []

    for distance_kind, summary in [
        ("absolute", absolute_summary),
        ("normalised", normalised_summary),
    ]:
        for split in args.splits:
            base = fig_dir / f"{split}_{args.plot_metric}_by_{distance_kind}_distance"
            plot_single_split_auc(
                summary=summary,
                split=split,
                distance_kind=distance_kind,
                metric=args.plot_metric,
                output_base=base,
                y_min=args.plot_y_min,
                y_max=args.plot_y_max,
            )
            if base.with_suffix(".png").exists():
                figures_created.extend([base.with_suffix(".png"), base.with_suffix(".pdf")])

        combined_base = fig_dir / f"paper_{args.plot_metric}_by_{distance_kind}_distance"
        plot_combined_auc(
            summary=summary,
            splits=args.splits,
            distance_kind=distance_kind,
            metric=args.plot_metric,
            output_base=combined_base,
            y_min=args.plot_y_min,
            y_max=args.plot_y_max,
        )
        if combined_base.with_suffix(".png").exists():
            figures_created.extend([
                combined_base.with_suffix(".png"),
                combined_base.with_suffix(".pdf"),
            ])

    for distance_kind, summary in [
        ("absolute", delta_absolute),
        ("normalised", delta_normalised),
    ]:
        base = fig_dir / f"paper_delta_{args.plot_metric}_by_{distance_kind}_distance"
        plot_combined_delta(
            summary=summary,
            splits=args.splits,
            distance_kind=distance_kind,
            metric=args.plot_metric,
            output_base=base,
        )
        if base.with_suffix(".png").exists():
            figures_created.extend([base.with_suffix(".png"), base.with_suffix(".pdf")])

    artifact_rows = [
        {
            "artifact": "distance_analysis_config",
            "path": str(config_path),
            "description": "Configuration for AUROC-versus-training-distance analysis.",
        },
        {
            "artifact": "training_peptide_reference_set",
            "path": str(out_dir / "training_peptide_reference_set.csv"),
            "description": "Unique positive training peptides used as the distance reference set.",
        },
        {
            "artifact": "peptide_distance_lookup",
            "path": str(distance_lookup_path),
            "description": "Nearest training peptide and absolute/normalised Levenshtein distance.",
        },
        {
            "artifact": "per_peptide_seed_metrics",
            "path": str(per_seed_path),
            "description": "Seed-level peptide-specific AUROC and McClish AUC0.1.",
        },
        {
            "artifact": "per_peptide_metrics_across_seeds",
            "path": str(peptide_mean_path),
            "description": "Peptide-specific metrics averaged across model seeds.",
        },
        {
            "artifact": "binned_auc_absolute",
            "path": str(absolute_summary_path),
            "description": "Peptide-bootstrap AUROC summaries by absolute edit distance.",
        },
        {
            "artifact": "binned_auc_normalised",
            "path": str(normalised_summary_path),
            "description": "Peptide-bootstrap AUROC summaries by normalised edit distance.",
        },
        {
            "artifact": "model_delta_per_peptide",
            "path": str(delta_path),
            "description": "Learned-minus-corresponding-baseline peptide-level performance deltas.",
        },
        {
            "artifact": "model_delta_binned_absolute",
            "path": str(delta_absolute_path),
            "description": "Binned learned-minus-baseline deltas by absolute edit distance.",
        },
        {
            "artifact": "model_delta_binned_normalised",
            "path": str(delta_normalised_path),
            "description": "Binned learned-minus-baseline deltas by normalised edit distance.",
        },
    ]
    for path in figures_created:
        artifact_rows.append({
            "artifact": path.stem,
            "path": str(path),
            "description": "AUROC-versus-training-distance workshop figure.",
        })

    update_manifest(manifest_path, artifact_rows)

    print("=" * 78)
    print("Distance-from-training analysis complete")
    print(f"Training CSV: {args.train_csv}")
    print(f"Tables:       {out_dir}")
    print(f"Figures:      {fig_dir}")
    print(f"Manifest:     {manifest_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
