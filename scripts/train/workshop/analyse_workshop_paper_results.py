#!/usr/bin/env python3
"""
Workshop-paper analysis and diagnostics for TCR-pMHC VICReg experiments.

This script is read-only with respect to training outputs. It consolidates
metrics, computes ablation deltas, and runs representation diagnostics from
saved prediction CSVs and latent NPZ files.

Expected folder layout:
  outputs:     /home/natasha/multimodal_model/models/outputs/workshop/{run_name}/seed_{seed}/
  figures:     /home/natasha/multimodal_model/models/figures/workshop/{run_name}/seed_{seed}/
  checkpoints: /home/natasha/multimodal_model/models/checkpoints/workshop/{run_name}/seed_{seed}/best.pt

Core diagnostics:
  1. Score distributions by model and split.
  2. CKA / Procrustes / pairwise-distance correlation between learned spaces.
  3. Effective rank and singular-value spectrum.
  4. Cross-reactivity: within-peptide vs between-peptide distances and kNN peptide purity.
  5. Nuisance correlations with TCR/peptide/HLA lengths.
  6. t-SNE qualitative TCR-space plots for top peptides.

The representation diagnostics require split_latents.npz files. If these are
missing, rerun the relevant training/evaluation script with --save-latents.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]

CORE_METRICS = [
    "global_auroc",
    "auprc",
    "global_auc0.1_mcclish",
    "peptide_macro_auroc",
    "peptide_weighted_auroc",
    "peptide_macro_auc0.1_mcclish",
    "peptide_weighted_auc0.1_mcclish",
]

LENGTH_COLS = ["tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len"]

VICREG_T_MODELS = ["onehot_vicreg", "raw_esmc_vicreg", "finetuned_esmc_vicreg"]

# ColourBrewer Set1 + Set2 + Dark2 hexes: 15+ mutually distinct categorical colours.
_DISTINCT_HEX = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
    "#e6ab02", "#a6761d", "#666666",
]
_DISTINCT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p", "8", "H", "d"]


def peptide_color_map(peptides: Sequence[str]) -> Dict[str, str]:
    """Assign a unique categorical colour to each peptide (no cycle collisions)."""
    n = len(peptides)
    if n == 0:
        return {}
    if n <= len(_DISTINCT_HEX):
        cols = _DISTINCT_HEX[:n]
    else:
        cols = [plt.cm.hsv(i / (n + 1)) for i in range(n)]
    return {pep: cols[i] for i, pep in enumerate(peptides)}


def peptide_marker_map(peptides: Sequence[str]) -> Dict[str, str]:
    """Assign a cycling but peptide-stable marker so similar hues stay separable."""
    return {pep: _DISTINCT_MARKERS[i % len(_DISTINCT_MARKERS)] for i, pep in enumerate(peptides)}


@dataclass
class RunConfig:
    outputs_root: str
    figures_root: str
    analysis_out_dir: str
    analysis_fig_dir: str
    run_names: List[str]
    seeds: List[int]
    plot_seed: int
    splits: List[str]
    min_group_size: int
    max_tcrs_per_peptide: int
    knn_k: List[int]
    top_n_peptides_tsne: int
    max_tsne_points: int
    max_cka_rows: int
    max_pair_distance_pairs: int
    skip_tsne: bool
    skip_duplicate_baselines_from_raw_runs: bool
    random_seed: int
    cache_merged_reps: bool = False


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Subdirectory layout under paper_analysis/ (kept in sync with README.md).
ARTIFACT_SUBDIR = {
    "paper_main_table.csv": "tables",
    "paper_ablation_deltas.csv": "tables",
    "paper_model_metrics_long.csv": "tables",
    "paper_model_metrics_summary.csv": "tables",
    "score_distribution_summary.csv": "score_distributions",
    "representation_similarity.csv": "representation_similarity",
    "effective_rank.csv": "effective_rank",
    "crossreactivity_within_between.csv": "crossreactivity",
    "crossreactivity_pair_auc.csv": "crossreactivity",
    "knn_peptide_purity.csv": "crossreactivity",
    "peptide_frequency_compactness.csv": "nuisance",
    "nuisance_correlations.csv": "nuisance",
    "diagnostics_manifest.csv": "meta",
    "diagnostics_run_config.json": "meta",
}


def artifact_path(out_dir: Path, name: str) -> Path:
    sub = ARTIFACT_SUBDIR.get(name, "")
    base = mkdir(out_dir / sub) if sub else out_dir
    return base / name


def write_manifest(rows: List[Dict], out_dir: Path) -> None:
    if not rows:
        return
    pd.DataFrame(rows).to_csv(artifact_path(out_dir, "diagnostics_manifest.csv"), index=False)


def finite_float(x) -> float:
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float("nan")


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    mask = np.isfinite(scores)
    labels = labels[mask]
    scores = scores[mask]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def safe_mcclish_auc(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    mask = np.isfinite(scores)
    labels = labels[mask]
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
    """Map script-internal names to paper-level names."""
    rn = run_name.lower()
    m = model_name.lower()

    if m == "esm_vicreg":
        if is_raw_run(rn):
            return "raw_esmc_vicreg"
        if is_finetuned_run(rn):
            return "finetuned_esmc_vicreg"
        return f"{run_name}__esm_vicreg"

    if m in {"onehot_vicreg", "onehot_composition", "pretrained_esmc_meanpool", "finetuned_esmc_meanpool"}:
        return m

    # If a raw ESM script used a more explicit name already, standardise it.
    if m in {"raw_esmc_vicreg", "pretrained_esmc_vicreg"}:
        return "raw_esmc_vicreg"
    if m in {"lora_esmc_vicreg", "adapted_esmc_vicreg"}:
        return "finetuned_esmc_vicreg"

    return model_name


def sort_model_df(df: pd.DataFrame) -> pd.DataFrame:
    if "model_name" not in df.columns:
        return df
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    tmp = df.copy()
    tmp["_model_order"] = tmp["model_name"].map(order).fillna(999).astype(int)
    sort_cols = [c for c in ["split", "_model_order", "model_name"] if c in tmp.columns]
    tmp = tmp.sort_values(sort_cols).drop(columns=["_model_order"])
    return tmp


def subset_rows(X: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    if max_rows <= 0 or n <= max_rows:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_rows, replace=False))


def zscore_matrix(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mu = np.nanmean(X, axis=0, keepdims=True)
    sd = np.nanstd(X, axis=0, keepdims=True)
    return (X - mu) / (sd + eps)


def center_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - np.mean(X, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# Metrics consolidation and ablation deltas
# ---------------------------------------------------------------------------

def should_skip_summary_model(run_name: str, model_name: str, skip_duplicate: bool) -> bool:
    if not skip_duplicate:
        return False
    if not is_raw_run(run_name):
        return False
    # Raw ESM runs often still emit meanpool baselines with script-internal names.
    # Keep only the VICReg row from raw runs to avoid double-counting deterministic baselines.
    return model_name in {"finetuned_esmc_meanpool", "pretrained_esmc_meanpool"}


def flatten_summary(summary_path: Path, skip_duplicate_baselines_from_raw_runs: bool = True) -> List[Dict]:
    with open(summary_path, "r") as f:
        summary = json.load(f)

    cfg = summary.get("config", {})
    seed = summary.get("seed", cfg.get("seed"))
    run_name = summary_path.parents[1].name
    family = summary.get("model_family", run_name)
    best_epoch = summary.get("best_epoch")
    best_selection_metric = summary.get("best_selection_metric")
    best_selection_value = summary.get("best_selection_value")

    rows = []
    for split, models in summary.get("metrics", {}).items():
        for raw_model_name, metrics in models.items():
            if should_skip_summary_model(run_name, raw_model_name, skip_duplicate_baselines_from_raw_runs):
                continue
            model_name = alias_model(run_name, raw_model_name)
            row = {
                "summary_path": str(summary_path),
                "run_name": run_name,
                "model_family": family,
                "raw_model_name": raw_model_name,
                "model_name": model_name,
                "split": split,
                "seed": int(seed) if seed is not None else np.nan,
                "deterministic_baseline": model_name in {
                    "onehot_composition", "pretrained_esmc_meanpool", "finetuned_esmc_meanpool"
                },
                "best_epoch": best_epoch,
                "best_selection_metric": best_selection_metric,
                "best_selection_value": best_selection_value,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def consolidate_metrics(cfg: RunConfig, out_dir: Path, manifest: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outputs_root = Path(cfg.outputs_root)
    rows = []
    summary_paths = []
    for rn in cfg.run_names:
        for seed in cfg.seeds:
            p = outputs_root / rn / f"seed_{seed}" / "summary.json"
            if p.exists():
                summary_paths.append(p)
                rows.extend(flatten_summary(p, cfg.skip_duplicate_baselines_from_raw_runs))
            else:
                warnings.warn(f"Missing summary: {p}")

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise FileNotFoundError("No summary rows found. Check --run-names, --seeds and --outputs-root.")
    long_df = sort_model_df(long_df)
    long_path = artifact_path(out_dir, "paper_model_metrics_long.csv")
    long_df.to_csv(long_path, index=False)
    manifest.append({"artifact": "paper_model_metrics_long", "path": str(long_path), "description": "Flattened seed-level metrics across run folders."})

    meta_cols = {
        "summary_path", "run_name", "model_family", "raw_model_name", "model_name", "split", "seed",
        "deterministic_baseline", "best_epoch", "best_selection_metric", "best_selection_value",
    }
    metric_cols = [c for c in long_df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(long_df[c])]

    grouped_rows = []
    for (model_name, split), grp in long_df.groupby(["model_name", "split"], dropna=False):
        row = {
            "model_name": model_name,
            "split": split,
            "deterministic_baseline": bool(grp["deterministic_baseline"].iloc[0]),
            "n_seed_rows": int(grp["seed"].nunique(dropna=True)),
            "seeds": ",".join(str(int(x)) for x in sorted(grp["seed"].dropna().unique())),
            "run_names": ",".join(sorted(set(map(str, grp["run_name"].dropna().unique())))),
        }
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        grouped_rows.append(row)

    summary_df = sort_model_df(pd.DataFrame(grouped_rows))
    summary_path = artifact_path(out_dir, "paper_model_metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    manifest.append({"artifact": "paper_model_metrics_summary", "path": str(summary_path), "description": "Mean/std metrics by model and split."})

    compact_cols = ["model_name", "split", "deterministic_baseline", "n_seed_rows", "seeds", "run_names"]
    for m in CORE_METRICS + ["n_examples", "n_positive", "n_negative", "n_peptides_total", "n_peptides_valid", "mse_distance_mean", "mse_distance_std"]:
        for suffix in ["_mean", "_std"]:
            c = f"{m}{suffix}"
            if c in summary_df.columns:
                compact_cols.append(c)
    compact_df = summary_df[[c for c in compact_cols if c in summary_df.columns]].copy()
    compact_path = artifact_path(out_dir, "paper_main_table.csv")
    compact_df.to_csv(compact_path, index=False)
    manifest.append({"artifact": "paper_main_table", "path": str(compact_path), "description": "Compact paper-facing metric table."})

    deltas = compute_ablation_deltas(summary_df)
    delta_path = artifact_path(out_dir, "paper_ablation_deltas.csv")
    deltas.to_csv(delta_path, index=False)
    manifest.append({"artifact": "paper_ablation_deltas", "path": str(delta_path), "description": "Controlled performance deltas for interpretation."})

    return long_df, summary_df


def compute_ablation_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("VICReg gain from one-hot", "onehot_vicreg", "onehot_composition"),
        ("VICReg gain from raw ESMC", "raw_esmc_vicreg", "pretrained_esmc_meanpool"),
        ("VICReg gain from fine-tuned ESMC", "finetuned_esmc_vicreg", "finetuned_esmc_meanpool"),
        ("LoRA gain before VICReg", "finetuned_esmc_meanpool", "pretrained_esmc_meanpool"),
        ("LoRA gain after VICReg", "finetuned_esmc_vicreg", "raw_esmc_vicreg"),
        ("Raw ESMC VICReg gain over one-hot VICReg", "raw_esmc_vicreg", "onehot_vicreg"),
        ("Fine-tuned ESMC VICReg gain over one-hot VICReg", "finetuned_esmc_vicreg", "onehot_vicreg"),
    ]
    rows = []
    for split in sorted(summary_df["split"].dropna().unique()):
        sub = summary_df[summary_df["split"] == split].set_index("model_name")
        for label, hi, lo in pairs:
            if hi not in sub.index or lo not in sub.index:
                continue
            for metric in CORE_METRICS:
                c = f"{metric}_mean"
                if c not in sub.columns:
                    continue
                rows.append({
                    "split": split,
                    "delta_name": label,
                    "metric": metric,
                    "higher_model": hi,
                    "lower_model": lo,
                    "higher_value": finite_float(sub.loc[hi, c]),
                    "lower_value": finite_float(sub.loc[lo, c]),
                    "delta": finite_float(sub.loc[hi, c]) - finite_float(sub.loc[lo, c]),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Predictions and score distributions
# ---------------------------------------------------------------------------

def prediction_path(outputs_root: Path, run_name: str, seed: int, split: str) -> Path:
    return outputs_root / run_name / f"seed_{seed}" / f"{split}_predictions.csv"


def read_prediction_models(path: Path, run_name: str, skip_duplicate_baselines_from_raw_runs: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows = []
    split = path.name.replace("_predictions.csv", "")
    seed_str = path.parent.name.replace("seed_", "")
    seed = int(seed_str) if seed_str.isdigit() else np.nan
    for col in df.columns:
        if not col.endswith("_mse_distance"):
            continue
        raw_model = col[:-len("_mse_distance")]
        if should_skip_summary_model(run_name, raw_model, skip_duplicate_baselines_from_raw_runs):
            continue
        model = alias_model(run_name, raw_model)
        score_col = f"{raw_model}_score"
        if score_col in df.columns:
            score = pd.to_numeric(df[score_col], errors="coerce")
        else:
            score = -pd.to_numeric(df[col], errors="coerce")
        base = pd.DataFrame({
            "run_name": run_name,
            "seed": seed,
            "split": split,
            "model_name": model,
            "raw_model_name": raw_model,
            "pair_id": df["pair_id"].astype(str),
            "peptide": df["peptide"].astype(str),
            "label": pd.to_numeric(df["label"], errors="coerce").astype(int),
            "mse_distance": pd.to_numeric(df[col], errors="coerce"),
            "score": score,
        })
        for lc in LENGTH_COLS:
            if lc in df.columns:
                base[lc] = pd.to_numeric(df[lc], errors="coerce")
        rows.append(base)
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def collect_prediction_rows(cfg: RunConfig, seeds: Optional[List[int]] = None) -> pd.DataFrame:
    outputs_root = Path(cfg.outputs_root)
    rows = []
    use_seeds = seeds if seeds is not None else cfg.seeds
    for rn in cfg.run_names:
        for seed in use_seeds:
            for split in cfg.splits:
                p = prediction_path(outputs_root, rn, seed, split)
                if p.exists():
                    rows.append(read_prediction_models(p, rn, cfg.skip_duplicate_baselines_from_raw_runs))
                else:
                    warnings.warn(f"Missing predictions: {p}")
    return pd.concat(rows, axis=0, ignore_index=True) if rows else pd.DataFrame()


def score_distribution_diagnostics(cfg: RunConfig, out_dir: Path, fig_dir: Path, manifest: List[Dict]) -> pd.DataFrame:
    pred_df = collect_prediction_rows(cfg)
    if pred_df.empty:
        warnings.warn("No predictions found for score-distribution diagnostics.")
        return pred_df

    rows = []
    for (seed, split, model), grp in pred_df.groupby(["seed", "split", "model_name"], dropna=False):
        y = grp["label"].to_numpy().astype(int)
        dist = grp["mse_distance"].to_numpy(float)
        score = grp["score"].to_numpy(float)
        pos = dist[y == 1]
        neg = dist[y == 0]
        rows.append({
            "seed": seed,
            "split": split,
            "model_name": model,
            "n_positive": int((y == 1).sum()),
            "n_negative": int((y == 0).sum()),
            "positive_mse_mean": finite_float(np.nanmean(pos)) if len(pos) else np.nan,
            "negative_mse_mean": finite_float(np.nanmean(neg)) if len(neg) else np.nan,
            "positive_mse_median": finite_float(np.nanmedian(pos)) if len(pos) else np.nan,
            "negative_mse_median": finite_float(np.nanmedian(neg)) if len(neg) else np.nan,
            "median_gap_negative_minus_positive": finite_float(np.nanmedian(neg) - np.nanmedian(pos)) if len(pos) and len(neg) else np.nan,
            "auroc": safe_auroc(y, score),
            "auprc": safe_auprc(y, score),
            "global_auc0.1_mcclish": safe_mcclish_auc(y, score),
        })
    score_summary = sort_model_df(pd.DataFrame(rows))
    out_path = artifact_path(out_dir, "score_distribution_summary.csv")
    score_summary.to_csv(out_path, index=False)
    manifest.append({"artifact": "score_distribution_summary", "path": str(out_path), "description": "Positive/negative MSE separation by model and split."})

    score_fig_dir = mkdir(fig_dir / "score_distributions")
    plot_df = pred_df[pred_df["seed"] == cfg.plot_seed].copy()
    for split in cfg.splits:
        split_df = plot_df[plot_df["split"] == split]
        if split_df.empty:
            continue
        models = [m for m in MODEL_ORDER if m in set(split_df["model_name"])] + [m for m in sorted(set(split_df["model_name"])) if m not in MODEL_ORDER]
        n = len(models)
        ncols = 2
        nrows = int(math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, 3.5 * nrows)))
        axes = np.asarray(axes).reshape(-1)
        for ax, model in zip(axes, models):
            g = split_df[split_df["model_name"] == model]
            y = g["label"].to_numpy().astype(int)
            d = g["mse_distance"].to_numpy(float)
            if np.any(y == 0):
                ax.hist(d[y == 0], bins=50, density=True, alpha=0.55, label="negative")
            if np.any(y == 1):
                ax.hist(d[y == 1], bins=50, density=True, alpha=0.55, label="positive")
            ax.set_title(model)
            ax.set_xlabel("MSE distance")
            ax.set_ylabel("Density")
            ax.legend(fontsize=8)
        for ax in axes[len(models):]:
            ax.axis("off")
        fig.suptitle(f"{split}: score distributions, seed {cfg.plot_seed}")
        fig.tight_layout()
        fig_path = score_fig_dir / f"{split}_score_distribution_grid_seed{cfg.plot_seed}.png"
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        manifest.append({"artifact": f"{split}_score_distribution_grid", "path": str(fig_path), "description": "MSE-distance histograms for positives and negatives."})

    return score_summary


# ---------------------------------------------------------------------------
# Latent representation loading
# ---------------------------------------------------------------------------

def latent_path(outputs_root: Path, run_name: str, seed: int, split: str) -> Path:
    return outputs_root / run_name / f"seed_{seed}" / f"{split}_latents.npz"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def latent_arrays_from_npz(run_name: str, npz: Dict[str, np.ndarray], skip_duplicate_baselines_from_raw_runs: bool = True) -> Dict[str, np.ndarray]:
    """Return {model__side: array}; side is T or PH."""
    arrays: Dict[str, np.ndarray] = {}
    rn = run_name.lower()

    if "zT_vicreg" in npz:
        arrays["onehot_vicreg__T"] = np.asarray(npz["zT_vicreg"])
    if "zPH_vicreg" in npz:
        arrays["onehot_vicreg__PH"] = np.asarray(npz["zPH_vicreg"])
    if "T_composition" in npz:
        arrays["onehot_composition__T"] = np.asarray(npz["T_composition"])
    if "PH_composition" in npz:
        arrays["onehot_composition__PH"] = np.asarray(npz["PH_composition"])

    if "zT_esm_vicreg" in npz:
        model = "raw_esmc_vicreg" if is_raw_run(rn) else "finetuned_esmc_vicreg" if is_finetuned_run(rn) else f"{run_name}__esm_vicreg"
        arrays[f"{model}__T"] = np.asarray(npz["zT_esm_vicreg"])
    if "zPH_esm_vicreg" in npz:
        model = "raw_esmc_vicreg" if is_raw_run(rn) else "finetuned_esmc_vicreg" if is_finetuned_run(rn) else f"{run_name}__esm_vicreg"
        arrays[f"{model}__PH"] = np.asarray(npz["zPH_esm_vicreg"])

    if not (skip_duplicate_baselines_from_raw_runs and is_raw_run(rn)):
        if "T_finetuned_meanpool" in npz:
            arrays["finetuned_esmc_meanpool__T"] = np.asarray(npz["T_finetuned_meanpool"])
        if "PH_finetuned_meanpool" in npz:
            arrays["finetuned_esmc_meanpool__PH"] = np.asarray(npz["PH_finetuned_meanpool"])
        if "T_pretrained_meanpool" in npz:
            arrays["pretrained_esmc_meanpool__T"] = np.asarray(npz["T_pretrained_meanpool"])
        if "PH_pretrained_meanpool" in npz:
            arrays["pretrained_esmc_meanpool__PH"] = np.asarray(npz["PH_pretrained_meanpool"])
    else:
        # In raw/pretrained ESM VICReg runs, the meanpool baselines are usually duplicates.
        pass

    return arrays


def load_representations_for_seed_split(cfg: RunConfig, seed: int, split: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    outputs_root = Path(cfg.outputs_root)
    meta: Optional[pd.DataFrame] = None
    arrays: Dict[str, np.ndarray] = {}
    missing = []
    for rn in cfg.run_names:
        p = latent_path(outputs_root, rn, seed, split)
        if not p.exists():
            missing.append(str(p))
            continue
        npz = load_npz(p)
        if meta is None:
            meta = pd.DataFrame({
                "pair_id": np.asarray(npz["pair_id"]).astype(str),
                "peptide": np.asarray(npz["peptide"]).astype(str),
                "label": np.asarray(npz["label"]).astype(int),
            })
            pred_path = prediction_path(outputs_root, rn, seed, split)
            if pred_path.exists():
                pred = pd.read_csv(pred_path)
                cols = ["pair_id"] + [c for c in LENGTH_COLS if c in pred.columns]
                if len(cols) > 1:
                    meta = meta.merge(pred[cols].drop_duplicates("pair_id"), on="pair_id", how="left")
        else:
            # Verify row order where possible.
            pid = np.asarray(npz["pair_id"]).astype(str)
            if len(pid) != len(meta) or not np.all(pid == meta["pair_id"].to_numpy(str)):
                warnings.warn(f"Latent row order mismatch for {p}. Diagnostics require aligned rows; this run will be skipped for latent arrays.")
                continue
        arrays.update(latent_arrays_from_npz(rn, npz, cfg.skip_duplicate_baselines_from_raw_runs))

    if missing:
        warnings.warn(
            "Missing latent files for some runs. Representation diagnostics will use the available files only. "
            "To generate missing files, rerun those model scripts with --save-latents. Examples:\n" + "\n".join(missing[:8])
        )
    if meta is None:
        return pd.DataFrame(), {}
    return meta, arrays


def cache_representations(cfg: RunConfig, out_dir: Path, manifest: List[Dict]) -> None:
    """Optional merged latent cache. Prefer reading per-run latents directly.

    Skipped unless cfg.cache_merged_reps is True — the merged NPZ tree is large and
    redundant with seed-level latents already stored under each model output dir.
    """
    if not getattr(cfg, "cache_merged_reps", False):
        print("Skipping merged reps cache (use --cache-merged-reps to enable).", flush=True)
        return
    # Keep original function name compatibility if renamed on disk.
    fn_name = "cache_reps"
    rep_dir = mkdir(out_dir / "representations")
    for seed in cfg.seeds:
        for split in cfg.splits:
            meta, arrays = load_representations_for_seed_split(cfg, seed, split)
            if meta.empty or not arrays:
                continue
            out = {c: meta[c].to_numpy() for c in meta.columns}
            out.update(arrays)
            p = rep_dir / f"seed_{seed}_{split}_representations.npz"
            np.savez_compressed(p, **out)
            manifest.append({
                "artifact": f"representations_seed{seed}_{split}",
                "path": str(p),
                "description": "Merged latent representation cache from available runs.",
            })


# ---------------------------------------------------------------------------
# Representation similarity: CKA, Procrustes, distance correlation
# ---------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = center_matrix(X)
    Yc = center_matrix(Y)
    numerator = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    denom = np.linalg.norm(Xc.T @ Xc, ord="fro") * np.linalg.norm(Yc.T @ Yc, ord="fro")
    return float(numerator / denom) if denom > 0 else float("nan")


def procrustes_r2(X: np.ndarray, Y: np.ndarray) -> float:
    Xc = center_matrix(X)
    Yc = center_matrix(Y)
    if Xc.shape[1] != Yc.shape[1]:
        return float("nan")
    M = Xc.T @ Yc
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    resid = np.linalg.norm(Xc @ R - Yc, ord="fro") ** 2
    denom = np.linalg.norm(Yc, ord="fro") ** 2
    return float(1.0 - resid / denom) if denom > 0 else float("nan")


def sample_pairwise_distances(X: np.ndarray, n_pairs: int, seed: int) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        return np.array([])
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    return np.linalg.norm(X[i] - X[j], axis=1)


def pairwise_distance_corr(X: np.ndarray, Y: np.ndarray, n_pairs: int, seed: int) -> Tuple[float, float]:
    n = min(X.shape[0], Y.shape[0])
    if n < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    dx = np.linalg.norm(X[i] - X[j], axis=1)
    dy = np.linalg.norm(Y[i] - Y[j], axis=1)
    if len(dx) < 3:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(dx, dy)[0, 1])
    if stats is not None:
        spearman = float(stats.spearmanr(dx, dy, nan_policy="omit").correlation)
    else:
        spearman = float(pd.Series(dx).rank().corr(pd.Series(dy).rank()))
    return pearson, spearman


def representation_similarity_diagnostics(cfg: RunConfig, out_dir: Path, fig_dir: Path, manifest: List[Dict]) -> pd.DataFrame:
    rows = []
    for seed in cfg.seeds:
        for split in cfg.splits:
            meta, arrays = load_representations_for_seed_split(cfg, seed, split)
            if meta.empty or not arrays:
                continue
            for side in ["T", "PH", "DELTA"]:
                side_arrays = {}
                if side == "DELTA":
                    models = sorted(set(k.split("__")[0] for k in arrays if k.endswith("__T")))
                    for m in models:
                        kt, kp = f"{m}__T", f"{m}__PH"
                        if kt in arrays and kp in arrays and arrays[kt].shape == arrays[kp].shape:
                            side_arrays[m] = arrays[kt] - arrays[kp]
                else:
                    suffix = f"__{side}"
                    side_arrays = {k[:-len(suffix)]: v for k, v in arrays.items() if k.endswith(suffix)}
                models = [m for m in MODEL_ORDER if m in side_arrays] + [m for m in sorted(side_arrays) if m not in MODEL_ORDER]
                for a_i, m1 in enumerate(models):
                    for m2 in models[a_i + 1:]:
                        X = side_arrays[m1]
                        Y = side_arrays[m2]
                        n = min(X.shape[0], Y.shape[0])
                        idx = subset_rows(X[:n], cfg.max_cka_rows, seed + 17)
                        Xs, Ys = X[:n][idx], Y[:n][idx]
                        rows.append({
                            "seed": seed,
                            "split": split,
                            "side": side,
                            "model_a": m1,
                            "model_b": m2,
                            "n_rows": int(len(idx)),
                            "cka": linear_cka(Xs, Ys),
                            "procrustes_r2": procrustes_r2(Xs, Ys),
                            "pairwise_distance_pearson": pairwise_distance_corr(Xs, Ys, cfg.max_pair_distance_pairs, seed + 31)[0],
                            "pairwise_distance_spearman": pairwise_distance_corr(Xs, Ys, cfg.max_pair_distance_pairs, seed + 31)[1],
                        })
    df = pd.DataFrame(rows)
    out_path = artifact_path(out_dir, "representation_similarity.csv")
    df.to_csv(out_path, index=False)
    manifest.append({"artifact": "representation_similarity", "path": str(out_path), "description": "CKA, Procrustes and pairwise-distance correlations between latent spaces."})

    if not df.empty:
        sim_fig_dir = mkdir(fig_dir / "representation_similarity")
        for split in cfg.splits:
            sub = df[(df["seed"] == cfg.plot_seed) & (df["split"] == split) & (df["side"] == "T")]
            if sub.empty:
                continue
            models = sorted(set(sub["model_a"]).union(set(sub["model_b"])), key=lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999)
            mat = pd.DataFrame(np.eye(len(models)), index=models, columns=models)
            for _, r in sub.iterrows():
                mat.loc[r["model_a"], r["model_b"]] = r["cka"]
                mat.loc[r["model_b"], r["model_a"]] = r["cka"]
            fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.2), max(5, len(models) * 1.0)))
            im = ax.imshow(mat.to_numpy(float), vmin=0, vmax=1)
            ax.set_xticks(range(len(models)))
            ax.set_yticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right")
            ax.set_yticklabels(models)
            ax.set_title(f"{split}: linear CKA, TCR spaces, seed {cfg.plot_seed}")
            for i in range(len(models)):
                for j in range(len(models)):
                    ax.text(j, i, f"{mat.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            p = sim_fig_dir / f"{split}_cka_heatmap_T_seed{cfg.plot_seed}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            manifest.append({"artifact": f"{split}_cka_heatmap_T", "path": str(p), "description": "Linear CKA heatmap for TCR spaces."})

    return df


# ---------------------------------------------------------------------------
# Effective rank diagnostics
# ---------------------------------------------------------------------------

def spectrum_metrics(Z: np.ndarray, seed: int, max_rows: int = 5000) -> Tuple[Dict, np.ndarray]:
    idx = subset_rows(Z, max_rows, seed)
    X = center_matrix(np.asarray(Z[idx], dtype=np.float64))
    if X.shape[0] < 2:
        return {}, np.array([])
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    s = np.maximum(s, 0)
    if s.sum() > 0:
        p = s / s.sum()
        eff_rank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    else:
        eff_rank = float("nan")
    s2 = s ** 2
    pr = float((s2.sum() ** 2) / (np.sum(s2 ** 2) + 1e-12))
    ve = {}
    total = s2.sum()
    for k in [1, 5, 10, 20, 50]:
        ve[f"variance_explained_top{k}"] = float(s2[:min(k, len(s2))].sum() / total) if total > 0 else np.nan
    metrics = {
        "n_rows": int(X.shape[0]),
        "d": int(X.shape[1]),
        "effective_rank": eff_rank,
        "participation_ratio": pr,
        **ve,
    }
    return metrics, s


def effective_rank_diagnostics(cfg: RunConfig, out_dir: Path, fig_dir: Path, manifest: List[Dict]) -> pd.DataFrame:
    rows = []
    spectra = {}
    label_subset_specs = [
        ("all", lambda m: np.ones(len(m), dtype=bool)),
        ("positive", lambda m: m["label"].astype(int).to_numpy() == 1),
        ("negative", lambda m: m["label"].astype(int).to_numpy() == 0),
    ]
    for seed in cfg.seeds:
        for split in cfg.splits:
            meta, arrays = load_representations_for_seed_split(cfg, seed, split)
            if meta.empty or not arrays:
                continue
            for key, Z in arrays.items():
                if "__" not in key:
                    continue
                model, side = key.split("__", 1)
                Z = np.asarray(Z)
                for label_subset, mask_fn in label_subset_specs:
                    sel = mask_fn(meta)
                    Z_sub = Z[sel]
                    if len(Z_sub) < 2:
                        continue
                    metrics, s = spectrum_metrics(Z_sub, seed=seed, max_rows=cfg.max_cka_rows)
                    if not metrics:
                        continue
                    rows.append({
                        "seed": seed,
                        "split": split,
                        "model_name": model,
                        "side": side,
                        "label_subset": label_subset,
                        **metrics,
                    })
                    if seed == cfg.plot_seed and label_subset == "all":
                        spectra[(split, model, side)] = s
    df = sort_model_df(pd.DataFrame(rows)) if rows else pd.DataFrame()
    out_path = artifact_path(out_dir, "effective_rank.csv")
    df.to_csv(out_path, index=False)
    manifest.append({
        "artifact": "effective_rank",
        "path": str(out_path),
        "description": "Effective rank, participation ratio and variance spectrum metrics (all / positive / negative).",
    })

    if not df.empty:
        er_fig_dir = mkdir(fig_dir / "effective_rank")
        subset_colors = {"positive": "#2ca02c", "negative": "#d62728"}
        plot_sub = df[
            (df["seed"] == cfg.plot_seed)
            & (df["side"].isin(["T", "PH"]))
            & (df["label_subset"].isin(["positive", "negative"]))
        ]
        for split in cfg.splits:
            sub = plot_sub[plot_sub["split"] == split]
            if sub.empty:
                continue
            group_keys = []
            for model in MODEL_ORDER:
                for side in ["T", "PH"]:
                    if ((sub["model_name"] == model) & (sub["side"] == side)).any():
                        group_keys.append((model, side))
            for model in sorted(set(sub["model_name"]) - set(MODEL_ORDER)):
                for side in ["T", "PH"]:
                    if ((sub["model_name"] == model) & (sub["side"] == side)).any():
                        group_keys.append((model, side))

            x = np.arange(len(group_keys))
            width = 0.36
            fig, ax = plt.subplots(figsize=(max(8, len(group_keys) * 0.85), 5))
            for i, label_subset in enumerate(["positive", "negative"]):
                vals = []
                for model, side in group_keys:
                    g = sub[
                        (sub["model_name"] == model)
                        & (sub["side"] == side)
                        & (sub["label_subset"] == label_subset)
                    ]
                    vals.append(float(g["effective_rank"].iloc[0]) if len(g) else np.nan)
                offset = (i - 0.5) * width
                ax.bar(
                    x + offset,
                    vals,
                    width=width,
                    label=label_subset,
                    color=subset_colors[label_subset],
                    alpha=0.9,
                )
            labels = [f"{model}\n{side}" for model, side in group_keys]
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Effective rank")
            ax.set_title(f"{split}: effective rank by label, seed {cfg.plot_seed}")
            ax.legend(title="label subset")
            fig.tight_layout()
            p = er_fig_dir / f"{split}_effective_rank_seed{cfg.plot_seed}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            manifest.append({
                "artifact": f"{split}_effective_rank_plot",
                "path": str(p),
                "description": "Effective rank by model/side, positive vs negative.",
            })

            fig, ax = plt.subplots(figsize=(8, 5))
            for (sp, model, side), s in spectra.items():
                if sp == split and side == "T" and model in VICREG_T_MODELS:
                    frac = (s ** 2) / np.sum(s ** 2) if np.sum(s ** 2) > 0 else s
                    ax.plot(np.arange(1, len(frac) + 1), frac, label=model)
            ax.set_xlabel("Component index")
            ax.set_ylabel("Variance fraction")
            ax.set_title(f"{split}: TCR singular-value spectrum, seed {cfg.plot_seed}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = er_fig_dir / f"{split}_singular_value_spectrum_T_seed{cfg.plot_seed}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            manifest.append({"artifact": f"{split}_singular_value_spectrum_T", "path": str(p), "description": "TCR singular-value spectrum for VICReg models."})
    return df


# ---------------------------------------------------------------------------
# Cross-reactivity and neighbourhood diagnostics
# ---------------------------------------------------------------------------

def balanced_positive_indices(meta: pd.DataFrame, min_group_size: int, max_per_peptide: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = meta[meta["label"].astype(int) == 1].copy()
    idxs = []
    for pep, grp in pos.groupby("peptide"):
        if len(grp) < min_group_size:
            continue
        indices = grp.index.to_numpy()
        if len(indices) > max_per_peptide:
            indices = rng.choice(indices, size=max_per_peptide, replace=False)
        idxs.extend(indices.tolist())
    return np.array(sorted(idxs), dtype=int)


def within_between_distances(Z: np.ndarray, peptides: np.ndarray, n_between: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    within = []
    peptide_to_idx = {p: np.where(peptides == p)[0] for p in np.unique(peptides)}
    for p, idx in peptide_to_idx.items():
        if len(idx) < 2:
            continue
        # Cap pair enumeration for very large groups, though groups are already capped.
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                within.append(np.linalg.norm(Z[idx[a]] - Z[idx[b]]))
    within = np.asarray(within, dtype=float)
    if len(within) == 0:
        return within, np.array([])

    n = len(peptides)
    between = []
    attempts = 0
    target = min(n_between, max(len(within), 1))
    max_attempts = target * 50
    while len(between) < target and attempts < max_attempts:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        attempts += 1
        if i == j or peptides[i] == peptides[j]:
            continue
        between.append(np.linalg.norm(Z[i] - Z[j]))
    return within, np.asarray(between, dtype=float)


def knn_purity_entropy(Z: np.ndarray, peptides: np.ndarray, ks: Sequence[int]) -> List[Dict]:
    n = Z.shape[0]
    if n < 3:
        return []
    max_k = min(max(ks), n - 1)
    nbrs = NearestNeighbors(n_neighbors=max_k + 1, metric="euclidean")
    nbrs.fit(Z)
    _, indices = nbrs.kneighbors(Z)
    indices = indices[:, 1:]  # remove self
    rows = []
    unique, counts = np.unique(peptides, return_counts=True)
    freq = dict(zip(unique, counts))
    for k in ks:
        if k > max_k:
            continue
        purities = []
        entropies = []
        random_purities = []
        for i in range(n):
            neigh = indices[i, :k]
            neigh_peps = peptides[neigh]
            purities.append(float(np.mean(neigh_peps == peptides[i])))
            vals, cnts = np.unique(neigh_peps, return_counts=True)
            q = cnts / cnts.sum()
            entropies.append(float(-np.sum(q * np.log(q + 1e-12))))
            random_purities.append(float((freq.get(peptides[i], 1) - 1) / max(n - 1, 1)))
        obs = float(np.mean(purities))
        rnd = float(np.mean(random_purities))
        rows.append({
            "k": int(k),
            "purity_mean": obs,
            "purity_std": float(np.std(purities)),
            "random_purity_expected": rnd,
            "purity_enrichment": float(obs / rnd) if rnd > 0 else np.nan,
            "entropy_mean": float(np.mean(entropies)),
            "entropy_std": float(np.std(entropies)),
        })
    return rows


def crossreactivity_diagnostics(cfg: RunConfig, out_dir: Path, fig_dir: Path, manifest: List[Dict]) -> None:
    wb_rows = []
    auc_rows = []
    knn_rows = []
    peptide_rows = []
    fig_records = []

    for seed in cfg.seeds:
        for split in cfg.splits:
            meta, arrays = load_representations_for_seed_split(cfg, seed, split)
            if meta.empty or not arrays:
                continue
            idx = balanced_positive_indices(meta, cfg.min_group_size, cfg.max_tcrs_per_peptide, seed)
            if len(idx) < 3:
                continue
            peps = meta.loc[idx, "peptide"].to_numpy(str)
            for key, Z_all in arrays.items():
                if not key.endswith("__T"):
                    continue
                model = key[:-len("__T")]
                Z = np.asarray(Z_all[idx], dtype=float)
                within, between = within_between_distances(Z, peps, n_between=cfg.max_pair_distance_pairs, seed=seed)
                if len(within) == 0 or len(between) == 0:
                    continue
                wb_rows.append({
                    "seed": seed,
                    "split": split,
                    "model_name": model,
                    "n_positive_balanced": int(len(idx)),
                    "n_peptides_balanced": int(len(np.unique(peps))),
                    "n_within_pairs": int(len(within)),
                    "n_between_pairs": int(len(between)),
                    "within_mean": float(np.mean(within)),
                    "between_mean": float(np.mean(between)),
                    "delta_between_minus_within": float(np.mean(between) - np.mean(within)),
                    "within_median": float(np.median(within)),
                    "between_median": float(np.median(between)),
                    "median_delta_between_minus_within": float(np.median(between) - np.median(within)),
                })
                labels = np.concatenate([np.ones(len(within)), np.zeros(len(between))])
                scores = -np.concatenate([within, between])
                auc_rows.append({
                    "seed": seed,
                    "split": split,
                    "model_name": model,
                    "same_peptide_pair_auroc": safe_auroc(labels, scores),
                    "same_peptide_pair_auprc": safe_auprc(labels, scores),
                    "n_pairs": int(len(labels)),
                })
                for row in knn_purity_entropy(Z, peps, cfg.knn_k):
                    knn_rows.append({"seed": seed, "split": split, "model_name": model, **row})
                # Peptide compactness by group.
                for pep in np.unique(peps):
                    local = np.where(peps == pep)[0]
                    if len(local) < 2:
                        continue
                    dvals = []
                    for a in range(len(local)):
                        for b in range(a + 1, len(local)):
                            dvals.append(np.linalg.norm(Z[local[a]] - Z[local[b]]))
                    peptide_rows.append({
                        "seed": seed,
                        "split": split,
                        "model_name": model,
                        "peptide": pep,
                        "n_tcrs_balanced": int(len(local)),
                        "within_mean_distance": float(np.mean(dvals)),
                        "within_median_distance": float(np.median(dvals)),
                    })
                if seed == cfg.plot_seed:
                    fig_records.append((split, model, within, between))

    cr_dir = mkdir(fig_dir / "crossreactivity")
    wb_df = sort_model_df(pd.DataFrame(wb_rows)) if wb_rows else pd.DataFrame()
    auc_df = sort_model_df(pd.DataFrame(auc_rows)) if auc_rows else pd.DataFrame()
    knn_df = sort_model_df(pd.DataFrame(knn_rows)) if knn_rows else pd.DataFrame()
    pep_df = sort_model_df(pd.DataFrame(peptide_rows)) if peptide_rows else pd.DataFrame()

    for name, df, desc in [
        ("crossreactivity_within_between.csv", wb_df, "Balanced within- vs between-peptide TCR distances."),
        ("crossreactivity_pair_auc.csv", auc_df, "Same-peptide recovery AUROC from pairwise TCR distances."),
        ("knn_peptide_purity.csv", knn_df, "kNN peptide purity and entropy in TCR latent spaces."),
        ("peptide_frequency_compactness.csv", pep_df, "Peptide-wise compactness after balanced sampling."),
    ]:
        p = artifact_path(out_dir, name)
        df.to_csv(p, index=False)
        manifest.append({"artifact": name.replace(".csv", ""), "path": str(p), "description": desc})

    # Boxplots for plot seed: twin y-axes so VICReg and non-VICReg scales are both readable.
    for split in cfg.splits:
        recs = [(m, w, b) for sp, m, w, b in fig_records if sp == split and m in MODEL_ORDER]
        recs += [(m, w, b) for sp, m, w, b in fig_records if sp == split and m not in MODEL_ORDER]
        if not recs:
            continue
        other_recs = [(m, w, b) for m, w, b in recs if m not in VICREG_T_MODELS]
        vicreg_recs = [(m, w, b) for m, w, b in recs if m in VICREG_T_MODELS]
        # Preserve MODEL_ORDER: non-VICReg first, then VICReg.
        ordered = other_recs + vicreg_recs

        fig, ax_left = plt.subplots(figsize=(max(8, len(ordered) * 1.25), 5))
        ax_right = ax_left.twinx()
        all_labels = []
        all_positions = []
        pos = 1
        left_data, left_pos = [], []
        right_data, right_pos = [], []
        for model, within, between in ordered:
            for series, tag in [(within, "within"), (between, "between")]:
                all_positions.append(pos)
                all_labels.append(f"{model}\n{tag}")
                if model in VICREG_T_MODELS:
                    right_data.append(series)
                    right_pos.append(pos)
                else:
                    left_data.append(series)
                    left_pos.append(pos)
                pos += 1

        box_kwargs = dict(showfliers=False, widths=0.55, patch_artist=True)
        if left_data:
            bp_l = ax_left.boxplot(left_data, positions=left_pos, **box_kwargs)
            for patch in bp_l["boxes"]:
                patch.set_facecolor("#9ecae1")
                patch.set_alpha(0.85)
            for key in ("medians", "whiskers", "caps"):
                for line in bp_l[key]:
                    line.set_color("#08519c")
            ax_left.set_ylabel("Euclidean distance (composition / meanpool)", color="#08519c")
            ax_left.tick_params(axis="y", labelcolor="#08519c")
        else:
            ax_left.set_ylabel("Euclidean distance (composition / meanpool)", color="#08519c")
            ax_left.tick_params(axis="y", labelcolor="#08519c")

        if right_data:
            bp_r = ax_right.boxplot(right_data, positions=right_pos, **box_kwargs)
            for patch in bp_r["boxes"]:
                patch.set_facecolor("#fdae6b")
                patch.set_alpha(0.85)
            for key in ("medians", "whiskers", "caps"):
                for line in bp_r[key]:
                    line.set_color("#a63603")
            ax_right.set_ylabel("Euclidean distance (VICReg)", color="#a63603")
            ax_right.tick_params(axis="y", labelcolor="#a63603")
        else:
            ax_right.set_ylabel("Euclidean distance (VICReg)", color="#a63603")
            ax_right.tick_params(axis="y", labelcolor="#a63603")

        ax_left.set_xlim(0.4, pos - 0.4)
        ax_right.set_xlim(0.4, pos - 0.4)
        ax_left.set_xticks(all_positions)
        ax_left.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=8)
        ax_right.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        ax_left.set_title(f"{split}: same-peptide vs between-peptide TCR distances, seed {cfg.plot_seed}")
        # Visual separator between scale groups when both are present.
        if left_data and right_data:
            sep = 0.5 * (max(left_pos) + min(right_pos))
            ax_left.axvline(sep, color="0.6", linestyle="--", linewidth=1.0, zorder=0)
        fig.tight_layout()
        p = cr_dir / f"{split}_within_between_boxplot_seed{cfg.plot_seed}.png"
        fig.savefig(p, dpi=200, bbox_inches="tight")

        plt.close(fig)
        manifest.append({"artifact": f"{split}_crossreactivity_boxplot", "path": str(p), "description": "Within/between peptide TCR-distance boxplot (twin y-axes for VICReg vs baselines)."})

    # kNN purity barplot for plot seed.
    if not knn_df.empty:
        for split in cfg.splits:
            sub = knn_df[(knn_df["seed"] == cfg.plot_seed) & (knn_df["split"] == split)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(9, 5))
            models = [m for m in MODEL_ORDER if m in set(sub["model_name"])] + [m for m in sorted(set(sub["model_name"])) if m not in MODEL_ORDER]
            width = 0.8 / max(1, len(cfg.knn_k))
            x = np.arange(len(models))
            for i, k in enumerate(cfg.knn_k):
                vals = []
                for m in models:
                    g = sub[(sub["model_name"] == m) & (sub["k"] == k)]
                    vals.append(float(g["purity_enrichment"].iloc[0]) if len(g) else np.nan)
                ax.bar(x + (i - (len(cfg.knn_k)-1)/2) * width, vals, width=width, label=f"k={k}")
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Purity enrichment vs frequency baseline")
            ax.set_title(f"{split}: kNN peptide-purity enrichment, seed {cfg.plot_seed}")
            ax.legend()
            fig.tight_layout()
            p = cr_dir / f"{split}_knn_purity_enrichment_seed{cfg.plot_seed}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            manifest.append({"artifact": f"{split}_knn_purity_enrichment", "path": str(p), "description": "kNN peptide-purity enrichment plot."})


# ---------------------------------------------------------------------------
# Nuisance-variable correlations
# ---------------------------------------------------------------------------

def corr_pair(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x, y)[0, 1])
    if stats is not None:
        spearman = float(stats.spearmanr(x, y, nan_policy="omit").correlation)
    else:
        spearman = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return pearson, spearman


def nuisance_correlation_diagnostics(cfg: RunConfig, out_dir: Path, fig_dir: Path, manifest: List[Dict]) -> pd.DataFrame:
    pred_df = collect_prediction_rows(cfg)
    rows = []
    if pred_df.empty:
        warnings.warn("No predictions found for nuisance diagnostics.")
        return pd.DataFrame()
    available_lengths = [c for c in LENGTH_COLS if c in pred_df.columns]
    for (seed, split, model), grp in pred_df.groupby(["seed", "split", "model_name"], dropna=False):
        for label_subset, g in [
            ("all", grp),
            ("positive", grp[grp["label"] == 1]),
            ("negative", grp[grp["label"] == 0]),
        ]:
            d = g["mse_distance"].to_numpy(float)
            for col in available_lengths:
                x = g[col].to_numpy(float)
                pearson, spearman = corr_pair(d, x)
                rows.append({
                    "seed": seed,
                    "split": split,
                    "model_name": model,
                    "label_subset": label_subset,
                    "variable": col,
                    "n": int(np.isfinite(d).sum()),
                    "pearson_r": pearson,
                    "spearman_rho": spearman,
                })
    df = sort_model_df(pd.DataFrame(rows)) if rows else pd.DataFrame()
    p = artifact_path(out_dir, "nuisance_correlations.csv")
    df.to_csv(p, index=False)
    manifest.append({"artifact": "nuisance_correlations", "path": str(p), "description": "Correlation of MSE distances with length variables."})

    if not df.empty:
        nui_dir = mkdir(fig_dir / "nuisance")
        for split in cfg.splits:
            sub = df[(df["seed"] == cfg.plot_seed) & (df["split"] == split) & (df["label_subset"] == "all")]
            if sub.empty:
                continue
            models = [m for m in MODEL_ORDER if m in set(sub["model_name"])] + [m for m in sorted(set(sub["model_name"])) if m not in MODEL_ORDER]
            variables = [v for v in LENGTH_COLS if v in set(sub["variable"])]
            mat = pd.DataFrame(np.nan, index=models, columns=variables)
            for _, r in sub.iterrows():
                mat.loc[r["model_name"], r["variable"]] = r["spearman_rho"]
            fig, ax = plt.subplots(figsize=(max(6, len(variables) * 1.2), max(5, len(models) * 0.55)))
            im = ax.imshow(mat.to_numpy(float), vmin=-1, vmax=1)
            ax.set_xticks(range(len(variables)))
            ax.set_yticks(range(len(models)))
            ax.set_xticklabels(variables, rotation=45, ha="right")
            ax.set_yticklabels(models)
            ax.set_title(f"{split}: Spearman correlation with MSE distance, seed {cfg.plot_seed}")
            for i in range(len(models)):
                for j in range(len(variables)):
                    val = mat.iloc[i, j]
                    ax.text(j, i, "" if pd.isna(val) else f"{val:.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fp = nui_dir / f"{split}_score_length_spearman_heatmap_seed{cfg.plot_seed}.png"
            fig.savefig(fp, dpi=200)
            plt.close(fig)
            manifest.append({"artifact": f"{split}_nuisance_heatmap", "path": str(fp), "description": "Spearman length-correlation heatmap."})
    return df


# ---------------------------------------------------------------------------
# t-SNE plots
# ---------------------------------------------------------------------------

def tsne_diagnostics(cfg: RunConfig, fig_dir: Path, manifest: List[Dict]) -> None:
    if cfg.skip_tsne:
        return
    tsne_dir = mkdir(fig_dir / "tsne")
    seed = cfg.plot_seed
    for split in cfg.splits:
        meta, arrays = load_representations_for_seed_split(cfg, seed, split)
        if meta.empty or not arrays:
            continue
        pos_meta = meta[meta["label"].astype(int) == 1].copy()
        if pos_meta.empty:
            continue
        top_peps = pos_meta["peptide"].value_counts().head(cfg.top_n_peptides_tsne).index.tolist()
        keep = pos_meta.index[pos_meta["peptide"].isin(top_peps)].to_numpy()
        if len(keep) < 10:
            continue
        if len(keep) > cfg.max_tsne_points:
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(keep, size=cfg.max_tsne_points, replace=False))
        plot_meta = meta.loc[keep].copy()
        for model in VICREG_T_MODELS:
            key = f"{model}__T"
            if key not in arrays:
                continue
            Z = np.asarray(arrays[key][keep], dtype=float)
            Z = zscore_matrix(Z)
            perplexity = min(30, max(5, (len(Z) - 1) // 3))
            try:
                emb = TSNE(n_components=2, perplexity=perplexity, learning_rate="auto", init="pca", random_state=seed).fit_transform(Z)
            except TypeError:
                emb = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(Z)
            fig, ax = plt.subplots(figsize=(8, 6))
            pep_colors = peptide_color_map(top_peps)
            pep_markers = peptide_marker_map(top_peps)
            for pep in top_peps:
                mask = plot_meta["peptide"].to_numpy(str) == pep
                if not np.any(mask):
                    continue
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=18,
                    alpha=0.8,
                    label=pep,
                    color=pep_colors[pep],
                    marker=pep_markers[pep],
                    edgecolors="none",
                )
            ax.set_title(f"{split}: {model} TCR t-SNE, positives, seed {seed}")
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")
            ax.legend(fontsize=6, loc="best", ncol=2, markerscale=1.4)
            fig.tight_layout()
            p = tsne_dir / f"{split}_tsne_tcr_top_peptides_{model}_seed{seed}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            manifest.append({"artifact": f"{split}_tsne_{model}", "path": str(p), "description": "t-SNE of positive TCR latent space coloured by peptide."})

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse workshop-paper TCR-pMHC representation-learning results.")
    parser.add_argument("--outputs-root", default="/home/natasha/multimodal_model/models/outputs/workshop")
    parser.add_argument("--figures-root", default="/home/natasha/multimodal_model/models/figures/workshop")
    parser.add_argument("--analysis-out-dir", default="/home/natasha/multimodal_model/models/outputs/workshop/paper_analysis")
    parser.add_argument("--analysis-fig-dir", default="/home/natasha/multimodal_model/models/figures/workshop/paper_analysis")
    parser.add_argument("--run-names", nargs="+", default=["onehot_vicreg_complete", "esm_vicreg_raw_complete", "esm_vicreg_finetuned_complete"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43, 49, 55])
    parser.add_argument("--plot-seed", type=int, default=31)
    parser.add_argument("--splits", nargs="+", default=["val", "test", "immrep_test"])
    parser.add_argument("--min-group-size", type=int, default=5)
    parser.add_argument("--max-tcrs-per-peptide", type=int, default=25)
    parser.add_argument("--knn-k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--top-n-peptides-tsne", type=int, default=15)
    parser.add_argument("--max-tsne-points", type=int, default=3000)
    parser.add_argument("--max-cka-rows", type=int, default=5000)
    parser.add_argument("--max-pair-distance-pairs", type=int, default=20000)
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--include-duplicate-baselines-from-raw-runs", action="store_true")
    parser.add_argument("--random-seed", type=int, default=31)
    parser.add_argument(
        "--cache-merged-reps",
        action="store_true",
        help="Write a merged NPZ cache under paper_analysis/reps/ (large; off by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(
        outputs_root=args.outputs_root,
        figures_root=args.figures_root,
        analysis_out_dir=args.analysis_out_dir,
        analysis_fig_dir=args.analysis_fig_dir,
        run_names=args.run_names,
        seeds=args.seeds,
        plot_seed=args.plot_seed,
        splits=args.splits,
        min_group_size=args.min_group_size,
        max_tcrs_per_peptide=args.max_tcrs_per_peptide,
        knn_k=args.knn_k,
        top_n_peptides_tsne=args.top_n_peptides_tsne,
        max_tsne_points=args.max_tsne_points,
        max_cka_rows=args.max_cka_rows,
        max_pair_distance_pairs=args.max_pair_distance_pairs,
        skip_tsne=args.skip_tsne,
        skip_duplicate_baselines_from_raw_runs=not args.include_duplicate_baselines_from_raw_runs,
        random_seed=args.random_seed,
        cache_merged_reps=args.cache_merged_reps,
    )

    out_dir = mkdir(Path(cfg.analysis_out_dir))
    fig_dir = mkdir(Path(cfg.analysis_fig_dir))
    manifest: List[Dict] = []

    with open(artifact_path(out_dir, "diagnostics_run_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    manifest.append({"artifact": "diagnostics_run_config", "path": str(artifact_path(out_dir, "diagnostics_run_config.json")), "description": "Configuration for this analysis run."})

    print("[1/7] Consolidating metrics", flush=True)
    consolidate_metrics(cfg, out_dir, manifest)

    print("[2/7] Score-distribution diagnostics", flush=True)
    score_distribution_diagnostics(cfg, out_dir, fig_dir, manifest)

    print("[3/7] Caching available latent representations", flush=True)
    cache_representations(cfg, out_dir, manifest)

    print("[4/7] Representation similarity diagnostics", flush=True)
    representation_similarity_diagnostics(cfg, out_dir, fig_dir, manifest)

    print("[5/7] Effective-rank diagnostics", flush=True)
    effective_rank_diagnostics(cfg, out_dir, fig_dir, manifest)

    print("[6/7] Cross-reactivity and nuisance diagnostics", flush=True)
    crossreactivity_diagnostics(cfg, out_dir, fig_dir, manifest)
    nuisance_correlation_diagnostics(cfg, out_dir, fig_dir, manifest)

    print("[7/7] t-SNE diagnostics", flush=True)
    tsne_diagnostics(cfg, fig_dir, manifest)

    write_manifest(manifest, out_dir)

    print("=" * 72)
    print("Workshop paper analysis complete")
    print(f"Outputs: {out_dir}")
    print(f"Figures: {fig_dir}")
    print(f"Manifest: {out_dir / 'diagnostics_manifest.csv'}")
    print("=" * 72)


if __name__ == "__main__":
    main()
