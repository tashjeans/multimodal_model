#!/usr/bin/env python3
"""Unnormalised Euclidean (pairwise MSE) extension of the IMMREP stage diagnostic.

Re-extracts TCR representations at input / pre_expander / final_latent from
validation-selected VICReg checkpoints (no retraining) and compares internal
test vs IMMREP using the same peptide balancing / pair sampling / 2000-peptide
bootstrap as the cosine stage diagnostic.

Primary comparison across stages: IMMREP / test ratio (scales differ by stage).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyse_immrep_transfer_stage_diagnostic as base  # noqa: E402

OUT_DIR = base.OUT_DIR
FIG_DIR = base.FIG_DIR
N_BOOT = base.N_BOOT
N_PAIR_MAX = base.N_PAIR_MAX
BOOT_SEED = base.BOOT_SEED
STAGES = base.STAGES
MODELS = base.MODELS
SEEDS = base.SEEDS
SPLITS = base.SPLITS


def zlib_salt(s: str) -> int:
    return base.zlib_salt(s)


def bootstrap_mean_ci(vals: np.ndarray, n_boot: int, rng: np.random.Generator):
    return base.bootstrap_mean_ci(vals, n_boot, rng)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    return base.cliffs_delta(x, y)


def pairwise_mse_sample(
    A: np.ndarray,
    B: Optional[np.ndarray],
    rng: np.random.Generator,
    n_sample: int,
    same_set: bool,
) -> np.ndarray:
    """Unnormalised pairwise MSE = mean_d (z_i - z_j)^2."""
    A = np.asarray(A, dtype=np.float64)
    if same_set:
        n = len(A)
        if n < 2:
            return np.array([], dtype=np.float64)
        i = rng.integers(0, n, size=n_sample)
        j = rng.integers(0, n - 1, size=n_sample)
        j = j + (j >= i)
        diff = A[i] - A[j]
    else:
        B = np.asarray(B, dtype=np.float64)
        if len(A) == 0 or len(B) == 0:
            return np.array([], dtype=np.float64)
        i = rng.integers(0, len(A), size=n_sample)
        j = rng.integers(0, len(B), size=n_sample)
        diff = A[i] - B[j]
    return np.mean(diff * diff, axis=1)


def geometry_mse_peptide(X: np.ndarray, rng: np.random.Generator) -> dict:
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n == 0:
        return {
            "n_tcr": 0,
            "median_pairwise_mse": np.nan,
            "mean_latent_norm": np.nan,
            "median_latent_norm": np.nan,
            "covariance_trace": np.nan,
            "mean_var_per_dim": np.nan,
        }
    norms = np.linalg.norm(X, axis=1)
    out = {
        "n_tcr": int(n),
        "mean_latent_norm": float(np.mean(norms)),
        "median_latent_norm": float(np.median(norms)),
    }
    if n >= 2:
        var_dim = np.var(X, axis=0, ddof=1)
        out["mean_var_per_dim"] = float(np.mean(var_dim))
        out["covariance_trace"] = float(np.sum(var_dim))
        n_sample = min(N_PAIR_MAX, n * (n - 1) // 2)
        mse = pairwise_mse_sample(X, None, rng, n_sample, same_set=True)
        out["median_pairwise_mse"] = float(np.median(mse))
    else:
        out["mean_var_per_dim"] = float("nan")
        out["covariance_trace"] = float("nan")
        out["median_pairwise_mse"] = float("nan")
    return out


def class_sep_mse_peptide(Z: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> Optional[dict]:
    pos = Z[labels == 1]
    neg = Z[labels == 0]
    if len(pos) < 1 or len(neg) < 1:
        return None
    avail = {"pn": len(pos) * len(neg)}
    if len(pos) >= 2:
        avail["pp"] = len(pos) * (len(pos) - 1) // 2
    if len(neg) >= 2:
        avail["nn"] = len(neg) * (len(neg) - 1) // 2
    n_sample = min(N_PAIR_MAX, min(avail.values()))
    pn = pairwise_mse_sample(pos, neg, rng, n_sample, same_set=False)
    out = {
        "n_positive": int(len(pos)),
        "n_negative": int(len(neg)),
        "n_pair_sampled": int(n_sample),
        "median_pn_mse": float(np.median(pn)),
        "median_pp_mse": float("nan"),
        "median_nn_mse": float("nan"),
        "pn_minus_pp": float("nan"),
        "cliffs_delta_pn_vs_pp": float("nan"),
    }
    if "pp" in avail:
        pp = pairwise_mse_sample(pos, pos, rng, n_sample, same_set=True)
        out["median_pp_mse"] = float(np.median(pp))
        out["pn_minus_pp"] = float(out["median_pn_mse"] - out["median_pp_mse"])
        out["cliffs_delta_pn_vs_pp"] = cliffs_delta(pn, pp)
    if "nn" in avail:
        nn = pairwise_mse_sample(neg, neg, rng, n_sample, same_set=True)
        out["median_nn_mse"] = float(np.median(nn))
    return out


def analyse_extracted(extracted: dict, model_key: str, seed: int, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    peptides = extracted["peptide"]
    labels = extracted["label"]
    geom_rows: List[dict] = []
    sep_rows: List[dict] = []
    uniq = sorted(set(peptides.tolist()))
    pep_to_idx = {pep: np.where(peptides == pep)[0] for pep in uniq}

    for stage in STAGES:
        Z = extracted[f"{stage}__tcr"]
        for pep, idx in pep_to_idx.items():
            rng = np.random.default_rng(
                BOOT_SEED + zlib_salt(f"msegeom|{model_key}|{seed}|{split}|{stage}|{pep}")
            )
            g = geometry_mse_peptide(Z[idx], rng)
            geom_rows.append({
                "model": model_key,
                "model_label": MODELS[model_key]["label"],
                "seed": seed,
                "split": split,
                "stage": stage,
                "side": "tcr",
                "peptide": pep,
                **g,
            })
            if split != "train":
                rng2 = np.random.default_rng(
                    BOOT_SEED + zlib_salt(f"msesep|{model_key}|{seed}|{split}|{stage}|{pep}")
                )
                sep = class_sep_mse_peptide(Z[idx], labels[idx], rng2)
                if sep is not None:
                    sep_rows.append({
                        "model": model_key,
                        "model_label": MODELS[model_key]["label"],
                        "seed": seed,
                        "split": split,
                        "stage": stage,
                        "side": "tcr",
                        "peptide": pep,
                        **sep,
                    })
    return pd.DataFrame(geom_rows), pd.DataFrame(sep_rows)


def average_over_seeds(df: pd.DataFrame, value_cols: List[str], extra_keys: List[str]) -> pd.DataFrame:
    keys = ["model", "model_label", "split", "stage", "side", "peptide"] + [
        k for k in extra_keys if k in df.columns
    ]
    # Keep unique key set
    keys = list(dict.fromkeys(keys))
    rows = []
    for key, g in df.groupby([k for k in keys if k in df.columns], sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        use_keys = [k for k in keys if k in df.columns]
        row = dict(zip(use_keys, key))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in value_cols:
            if c not in g.columns:
                continue
            vals = g[c].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[c] = float(np.mean(vals)) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_metric(per_pep: pd.DataFrame, metric: str, analysis: str) -> pd.DataFrame:
    rows = []
    lookup: Dict[Tuple[str, str, str], Tuple[dict, np.ndarray]] = {}
    for (model, split, stage), g in per_pep.groupby(["model", "split", "stage"], sort=True):
        vals = g[metric].to_numpy(float)
        rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"mseagg|{analysis}|{metric}|{model}|{split}|{stage}"))
        mean, lo, hi = bootstrap_mean_ci(vals, N_BOOT, rng)
        med = float(np.median(vals[np.isfinite(vals)])) if np.any(np.isfinite(vals)) else float("nan")
        row = {
            "analysis": analysis,
            "metric": metric,
            "model": model,
            "model_label": g["model_label"].iloc[0],
            "split": split,
            "stage": stage,
            "side": "tcr",
            "peptide_balanced_mean": mean,
            "peptide_median": med,
            "ci_low": lo,
            "ci_high": hi,
            "n_peptides": int(np.sum(np.isfinite(vals))),
            "n_bootstrap": N_BOOT,
        }
        rows.append(row)
        lookup[(model, split, stage)] = (row, vals[np.isfinite(vals)])

    for model in MODELS:
        for stage in STAGES:
            if (model, "test", stage) not in lookup or (model, "immrep_test", stage) not in lookup:
                continue
            test_row, test_vals = lookup[(model, "test", stage)]
            imm_row, imm_vals = lookup[(model, "immrep_test", stage)]
            rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"msediff|{analysis}|{metric}|{model}|{stage}"))
            diffs = np.empty(N_BOOT)
            ratios = np.empty(N_BOOT)
            for b in range(N_BOOT):
                t = test_vals[rng.integers(0, len(test_vals), len(test_vals))].mean()
                i = imm_vals[rng.integers(0, len(imm_vals), len(imm_vals))].mean()
                diffs[b] = i - t
                ratios[b] = i / t if abs(t) > 1e-12 else np.nan
            t_mean = test_row["peptide_balanced_mean"]
            i_mean = imm_row["peptide_balanced_mean"]
            rows.append({
                "analysis": analysis,
                "metric": metric,
                "model": model,
                "model_label": test_row["model_label"],
                "split": "comparison_immrep_vs_test",
                "stage": stage,
                "side": "tcr",
                "test_peptide_balanced_mean": t_mean,
                "test_ci_low": test_row["ci_low"],
                "test_ci_high": test_row["ci_high"],
                "immrep_peptide_balanced_mean": i_mean,
                "immrep_ci_low": imm_row["ci_low"],
                "immrep_ci_high": imm_row["ci_high"],
                "immrep_minus_test": float(i_mean - t_mean),
                "immrep_minus_test_ci_low": float(np.nanquantile(diffs, 0.025)),
                "immrep_minus_test_ci_high": float(np.nanquantile(diffs, 0.975)),
                "immrep_div_test": float(i_mean / t_mean) if abs(t_mean) > 1e-12 else float("nan"),
                "immrep_div_test_ci_low": float(np.nanquantile(ratios, 0.025)),
                "immrep_div_test_ci_high": float(np.nanquantile(ratios, 0.975)),
                "n_peptides_test": test_row["n_peptides"],
                "n_peptides_immrep": imm_row["n_peptides"],
                "n_bootstrap": N_BOOT,
            })
    return pd.DataFrame(rows)


def build_combined_table(
    mse_summary: pd.DataFrame,
    cosine_geom: pd.DataFrame,
    cosine_sep: pd.DataFrame,
    mse_sep_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for stage in STAGES:
            def get_mse(metric: str, field: str):
                r = mse_summary[
                    (mse_summary.model == model)
                    & (mse_summary.stage == stage)
                    & (mse_summary.metric == metric)
                    & (mse_summary.split == "comparison_immrep_vs_test")
                ]
                return float(r.iloc[0][field]) if len(r) else float("nan")

            def get_split_mse(metric: str, split: str):
                r = mse_summary[
                    (mse_summary.model == model)
                    & (mse_summary.stage == stage)
                    & (mse_summary.metric == metric)
                    & (mse_summary.split == split)
                ]
                return float(r.iloc[0]["peptide_balanced_mean"]) if len(r) else float("nan")

            # Cosine difference from existing diagnostic (seed-averaged summary)
            cos_t = cosine_geom[
                (cosine_geom.model == model)
                & (cosine_geom.split == "test")
                & (cosine_geom.stage == stage)
                & (cosine_geom.side == "tcr")
                & (cosine_geom.metric == "median_pairwise_cosine")
            ]
            cos_i = cosine_geom[
                (cosine_geom.model == model)
                & (cosine_geom.split == "immrep_test")
                & (cosine_geom.stage == stage)
                & (cosine_geom.side == "tcr")
                & (cosine_geom.metric == "median_pairwise_cosine")
            ]
            cos_diff = (
                float(cos_i.iloc[0].peptide_balanced_mean - cos_t.iloc[0].peptide_balanced_mean)
                if len(cos_t) and len(cos_i)
                else float("nan")
            )

            # Norm / cov ratios from this MSE run
            test_norm = get_split_mse("mean_latent_norm", "test")
            imm_norm = get_split_mse("mean_latent_norm", "immrep_test")
            test_cov = get_split_mse("covariance_trace", "test")
            imm_cov = get_split_mse("covariance_trace", "immrep_test")

            def get_sep(metric: str, split: str, source: str):
                if source == "mse":
                    r = mse_sep_summary[
                        (mse_sep_summary.model == model)
                        & (mse_sep_summary.stage == stage)
                        & (mse_sep_summary.metric == metric)
                        & (mse_sep_summary.split == split)
                    ]
                    return float(r.iloc[0]["peptide_balanced_mean"]) if len(r) else float("nan")
                r = cosine_sep[
                    (cosine_sep.model == model)
                    & (cosine_sep.stage == stage)
                    & (cosine_sep.metric == metric)
                    & (cosine_sep.split == split)
                ]
                return float(r.iloc[0]["peptide_balanced_mean"]) if len(r) else float("nan")

            test_mse = get_split_mse("median_pairwise_mse", "test")
            imm_mse = get_split_mse("median_pairwise_mse", "immrep_test")
            ratio = get_mse("median_pairwise_mse", "immrep_div_test")

            # Classification tags
            angular = cos_diff < -0.05
            euclidean = (not np.isnan(ratio)) and ratio < 0.7
            radial = (abs(test_norm) > 1e-12) and (imm_norm / test_norm < 0.7)
            common_dir = angular and (not np.isnan(ratio)) and ratio > 0.85
            cliffs_t = get_sep("cliffs_delta_pn_vs_pp", "test", "mse")
            cliffs_i = get_sep("cliffs_delta_pn_vs_pp", "immrep_test", "mse")
            sep_lost = (not np.isnan(cliffs_t)) and (not np.isnan(cliffs_i)) and (
                cliffs_i < 0.5 * max(cliffs_t, 1e-6)
            )
            fully_collapsed = euclidean and sep_lost

            rows.append({
                "model": model,
                "model_label": MODELS[model]["label"],
                "stage": stage,
                "test_pairwise_mse": test_mse,
                "immrep_pairwise_mse": imm_mse,
                "immrep_test_ratio": ratio,
                "cosine_difference": cos_diff,
                "norm_ratio": float(imm_norm / test_norm) if abs(test_norm) > 1e-12 else float("nan"),
                "covariance_trace_ratio": float(imm_cov / test_cov) if abs(test_cov) > 1e-12 else float("nan"),
                "test_pn_minus_pp": get_sep("pn_minus_pp", "test", "mse"),
                "immrep_pn_minus_pp": get_sep("pn_minus_pp", "immrep_test", "mse"),
                "test_cliffs_delta": cliffs_t,
                "immrep_cliffs_delta": cliffs_i,
                "tag_angular_concentration": bool(angular),
                "tag_euclidean_contraction": bool(euclidean),
                "tag_radial_contraction": bool(radial),
                "tag_common_direction_mapping": bool(common_dir),
                "tag_fully_collapsed": bool(fully_collapsed),
            })
    return pd.DataFrame(rows)


def plot_mse_figure(combined: pd.DataFrame, out_pdf: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.6), sharey=False)
    models = list(MODELS.keys())
    stage_x = {s: i for i, s in enumerate(STAGES)}
    colors = {
        "onehot_vicreg": "#252525",
        "raw_esmc_vicreg": "#08519c",
        "lora_esmc_vicreg": "#006d2c",
    }

    # Panel A: IMMREP/test MSE ratio
    ax = axes[0]
    for model in models:
        sub = combined[combined.model == model]
        xs = [stage_x[s] for s in STAGES]
        ys = [float(sub.loc[sub.stage == s, "immrep_test_ratio"].iloc[0]) for s in STAGES]
        ax.plot(xs, ys, marker="o", color=colors[model], label=MODELS[model]["label"], lw=2)
    ax.axhline(1.0, color="0.5", ls=":", lw=0.9)
    ax.set_xticks(list(stage_x.values()))
    ax.set_xticklabels(["input", "pre-exp", "final"], fontsize=8)
    ax.set_ylabel("IMMREP / test pairwise MSE")
    ax.set_title("(a) Euclidean contraction ratio", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, fontsize=7)

    # Panel B: cosine difference
    ax = axes[1]
    for model in models:
        sub = combined[combined.model == model]
        xs = [stage_x[s] for s in STAGES]
        ys = [float(sub.loc[sub.stage == s, "cosine_difference"].iloc[0]) for s in STAGES]
        ax.plot(xs, ys, marker="s", color=colors[model], label=MODELS[model]["label"], lw=2)
    ax.axhline(0.0, color="0.5", ls=":", lw=0.9)
    ax.set_xticks(list(stage_x.values()))
    ax.set_xticklabels(["input", "pre-exp", "final"], fontsize=8)
    ax.set_ylabel("IMMREP − test cosine distance")
    ax.set_title("(b) Angular concentration", fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.3)

    # Panel C: Cliff's delta test vs IMMREP at final
    ax = axes[2]
    x = np.arange(len(models))
    w = 0.35
    test_vals = [float(combined[(combined.model == m) & (combined.stage == "final_latent")]["test_cliffs_delta"].iloc[0]) for m in models]
    imm_vals = [float(combined[(combined.model == m) & (combined.stage == "final_latent")]["immrep_cliffs_delta"].iloc[0]) for m in models]
    ax.bar(x - w / 2, test_vals, w, color="#4c78a8", label="Internal test")
    ax.bar(x + w / 2, imm_vals, w, color="#f58518", label="IMMREP")
    ax.set_xticks(x)
    ax.set_xticklabels([MODELS[m]["label"].replace(" + ", "\n+ ") for m in models], fontsize=7)
    ax.set_ylabel("Cliff's δ (PN vs PP), MSE")
    ax.set_title("(c) Class separation at final latent", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Unnormalised Euclidean stage diagnostic (TCR)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def append_report(out_dir: Path, combined: pd.DataFrame) -> None:
    lines = [
        "",
        "## Unnormalised Euclidean (pairwise MSE) extension",
        "",
        "Distances use `mean((z_i − z_j)²)` without L2 normalisation. "
        "Across-stage comparisons use the **IMMREP / test ratio** because "
        "input, pre-expander and final latent live on different scales.",
        "",
        "### Terminology",
        "- **Angular concentration**: cosine distance decreases (IMMREP − test < 0).",
        "- **Euclidean contraction**: pairwise MSE and/or covariance-trace ratio ≪ 1.",
        "- **Radial contraction**: mean latent-norm ratio ≪ 1.",
        "- **Common-direction mapping**: cosine decreases but pairwise MSE ratio stays near 1.",
        "- **Fully collapsed**: only if Euclidean contraction *and* class-separating variation "
        "(Cliff’s δ / PN−PP) are substantially reduced.",
        "",
        "### Combined interpretation table",
        "",
        combined.to_string(index=False),
        "",
    ]
    report = out_dir / "STAGE_DIAGNOSTIC_REPORT.md"
    if report.exists():
        report.write_text(report.read_text() + "\n".join(lines))
    else:
        report.write_text("# Stage diagnostic\n" + "\n".join(lines))


def run_one(model_key: str, seed: int, split: str, device: torch.device, batch_size: int):
    print(f"\n=== MSE {model_key} seed={seed} split={split} ===", flush=True)
    tcr, pmhc, ckpt, shapes = base.load_checkpoint(model_key, seed, device)
    R_PH = float(ckpt["config"]["R_PH"])
    _, L_T, L_P, L_H = shapes
    if model_key == "onehot_vicreg":
        loader = base.make_onehot_loader(seed, split, L_T, L_P, L_H, batch_size)
    else:
        loader = base.make_esm_loader(model_key, seed, split, batch_size)
    extracted = base.extract_split(model_key, seed, split, tcr, pmhc, loader, device, R_PH)
    geom, sep = analyse_extracted(extracted, model_key, seed, split)
    del extracted
    return geom, sep


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()), choices=list(MODELS.keys()))
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--batch-size-onehot", type=int, default=256)
    parser.add_argument("--batch-size-esm", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    out_dir = args.out_dir
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    geom_parts, sep_parts = [], []
    for model_key in args.models:
        bs = args.batch_size_onehot if model_key == "onehot_vicreg" else args.batch_size_esm
        for seed in args.seeds:
            for split in args.splits:
                geom, sep = run_one(model_key, seed, split, device, bs)
                geom_parts.append(geom)
                if len(sep):
                    sep_parts.append(sep)

    geom_df = pd.concat(geom_parts, ignore_index=True)
    sep_df = pd.concat(sep_parts, ignore_index=True) if sep_parts else pd.DataFrame()
    detail_dir = out_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    geom_df.to_csv(detail_dir / "stagewise_unnormalised_mse_per_peptide.csv", index=False)
    sep_df.to_csv(detail_dir / "stagewise_unnormalised_mse_class_separation_per_peptide.csv", index=False)

    geom_avg = average_over_seeds(
        geom_df,
        ["median_pairwise_mse", "mean_latent_norm", "median_latent_norm", "covariance_trace", "mean_var_per_dim"],
        ["n_tcr"],
    )
    sep_avg = average_over_seeds(
        sep_df,
        ["median_pn_mse", "median_pp_mse", "median_nn_mse", "pn_minus_pp", "cliffs_delta_pn_vs_pp"],
        ["n_positive", "n_negative"],
    ) if len(sep_df) else pd.DataFrame()

    mse_summary_parts = [
        summarise_metric(geom_avg, "median_pairwise_mse", "unnormalised_mse"),
        summarise_metric(geom_avg, "mean_latent_norm", "unnormalised_mse"),
        summarise_metric(geom_avg, "median_latent_norm", "unnormalised_mse"),
        summarise_metric(geom_avg, "covariance_trace", "unnormalised_mse"),
        summarise_metric(geom_avg, "mean_var_per_dim", "unnormalised_mse"),
    ]
    mse_summary = pd.concat(mse_summary_parts, ignore_index=True)
    mse_summary.to_csv(out_dir / "stagewise_unnormalised_mse_summary.csv", index=False)

    sep_summary_parts = []
    for col in ["median_pn_mse", "median_pp_mse", "median_nn_mse", "pn_minus_pp", "cliffs_delta_pn_vs_pp"]:
        if len(sep_avg) and col in sep_avg.columns:
            sep_summary_parts.append(summarise_metric(sep_avg, col, "unnormalised_mse_class_sep"))
    sep_summary = pd.concat(sep_summary_parts, ignore_index=True) if sep_summary_parts else pd.DataFrame()
    sep_summary.to_csv(out_dir / "stagewise_unnormalised_mse_class_separation.csv", index=False)

    cosine_geom = pd.read_csv(out_dir / "stagewise_geometry_summary.csv")
    cosine_sep = pd.read_csv(out_dir / "stagewise_class_separation.csv")
    combined = build_combined_table(mse_summary, cosine_geom, cosine_sep, sep_summary)
    combined.to_csv(out_dir / "cosine_vs_mse_concentration_summary.csv", index=False)

    plot_mse_figure(combined, out_dir / "stagewise_unnormalised_mse_figure.pdf")
    plot_mse_figure(combined, fig_dir / "stagewise_unnormalised_mse_figure.pdf")
    append_report(out_dir, combined)

    print("\n=== IMMREP/test pairwise-MSE ratios (TCR) ===", flush=True)
    for model in args.models:
        print(f"{MODELS[model]['label']}:", flush=True)
        for stage in STAGES:
            r = combined[(combined.model == model) & (combined.stage == stage)].iloc[0]
            print(
                f"  {stage:14s} ratio={r.immrep_test_ratio:.3f}  "
                f"cosΔ={r.cosine_difference:+.3f}  "
                f"norm_ratio={r.norm_ratio:.3f}  "
                f"cov_ratio={r.covariance_trace_ratio:.3f}  "
                f"collapsed={r.tag_fully_collapsed}",
                flush=True,
            )

    (out_dir / "unnormalised_mse_manifest.json").write_text(json.dumps({
        "models": args.models,
        "seeds": args.seeds,
        "splits": args.splits,
        "n_boot": N_BOOT,
        "n_pair_max": N_PAIR_MAX,
        "distance": "unnormalised_pairwise_mse",
    }, indent=2))
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
