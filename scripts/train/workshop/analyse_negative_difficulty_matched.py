#!/usr/bin/env python3
"""Matched positive-reference-set negative difficulty (no retraining).

Controls the positive reference-set size explicitly to avoid confounding
IMMREP (50 positives/peptide) with internal test (often 1–2).

PRIMARY (m=1)
  For each peptide: for each positive reference (exhaustive, or 1,000
  bootstrap refs if n_pos > 1,000), take median distance of all negatives
  to that single reference; average reference-specific medians.

SECONDARY (m=2)
  Peptides with ≥2 positives: 1,000 times sample two positives; for each
  negative take distance to the nearer; record median over negatives;
  average the 1,000 medians → one peptide value.

Also:
  - count-independent pairwise TCR separation (PN / PP / NN)
  - latent concentration (pairwise cosine, variance, cov trace, eff. rank, norm)

Distances: Euclidean for one-hot composition; cosine for ESMC / VICReg.
Aggregation: equal peptide weight; 2,000 peptide bootstraps for 95% CIs.
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
OUT_DIR = REPO / "models/outputs/workshop/paper_analysis/negative_set_difficulty"

SEEDS = [31, 37, 43, 49, 55]
SPLITS = ["test", "immrep_test"]
N_BOOT_PEP = 2000
N_REF_BOOT = 1000
N_PAIR_MAX = 5000
EXHAUSTIVE_M1_MAX = 1000
BOOT_SEED = 20260806

REPRESENTATIONS = {
    "onehot_composition": {
        "run": "onehot_vicreg_complete",
        "key": "T_composition",
        "distance": "euclidean",
        "label": "One-hot composition (Euclidean)",
        "family": "composition",
    },
    "raw_esmc_meanpool": {
        "run": "esm_vicreg_raw_complete",
        "key": "T_pretrained_meanpool",
        "distance": "cosine",
        "label": "Raw ESMC mean-pool (cosine)",
        "family": "raw_esmc",
    },
    "lora_esmc_meanpool": {
        "run": "esm_vicreg_finetuned_complete",
        "key": "T_finetuned_meanpool",
        "distance": "cosine",
        "label": "LoRA ESMC mean-pool (cosine)",
        "family": "lora_esmc",
    },
    "onehot_vicreg": {
        "run": "onehot_vicreg_complete",
        "key": "zT_vicreg",
        "distance": "cosine",
        "label": "One-hot VICReg latent (cosine)",
        "family": "learned_latent",
    },
    "raw_esmc_vicreg": {
        "run": "esm_vicreg_raw_complete",
        "key": "zT_esm_vicreg",
        "distance": "cosine",
        "label": "Raw ESMC VICReg latent (cosine)",
        "family": "learned_latent",
    },
    "lora_esmc_vicreg": {
        "run": "esm_vicreg_finetuned_complete",
        "key": "zT_esm_vicreg",
        "distance": "cosine",
        "label": "LoRA ESMC VICReg latent (cosine)",
        "family": "learned_latent",
    },
}


def zlib_salt(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def prepare_space(Z: np.ndarray, distance: str) -> Tuple[np.ndarray, str]:
    """Return (matrix, mode) with mode in {'euclidean','cosine_unit'}."""
    if distance == "euclidean":
        return np.asarray(Z, dtype=np.float64), "euclidean"
    if distance == "cosine":
        return l2_normalize(np.asarray(Z, dtype=np.float64)), "cosine_unit"
    raise ValueError(distance)


def dist_to_refs(neg: np.ndarray, refs: np.ndarray, mode: str) -> np.ndarray:
    """Distances from each negative to each reference. Shape (n_neg, n_ref)."""
    if mode == "euclidean":
        aa = np.sum(neg * neg, axis=1, keepdims=True)
        bb = np.sum(refs * refs, axis=1, keepdims=True).T
        d2 = np.maximum(aa + bb - 2.0 * (neg @ refs.T), 0.0)
        return np.sqrt(d2)
    if mode == "cosine_unit":
        return 1.0 - (neg @ refs.T)
    raise ValueError(mode)


def pairwise_sample_distances(
    A: np.ndarray,
    B: np.ndarray,
    mode: str,
    n_sample: int,
    rng: np.random.Generator,
    same_set: bool,
) -> np.ndarray:
    """Sample pairwise distances between rows of A and B (or within A if same_set)."""
    if same_set:
        n = len(A)
        if n < 2:
            return np.array([], dtype=np.float64)
        # Sample unordered pairs via two distinct indices
        i = rng.integers(0, n, size=n_sample)
        j = rng.integers(0, n - 1, size=n_sample)
        j = j + (j >= i)  # shift to avoid i==j
        X, Y = A[i], A[j]
    else:
        if len(A) == 0 or len(B) == 0:
            return np.array([], dtype=np.float64)
        i = rng.integers(0, len(A), size=n_sample)
        j = rng.integers(0, len(B), size=n_sample)
        X, Y = A[i], B[j]

    if mode == "euclidean":
        return np.linalg.norm(X - Y, axis=1)
    if mode == "cosine_unit":
        return 1.0 - np.sum(X * Y, axis=1)
    raise ValueError(mode)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x>y) - P(x<y). Positive ⇒ x tends larger than y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    y_sorted = np.sort(y)
    # For each x_i: #{y < x} = cases x>y; #{y > x} = cases x<y
    n_x_gt_y = np.searchsorted(y_sorted, x, side="left")
    n_x_lt_y = len(y) - np.searchsorted(y_sorted, x, side="right")
    n = len(x) * len(y)
    return float((n_x_gt_y.sum() - n_x_lt_y.sum()) / n)


def load_split_arrays(
    split: str, seed: int, rep_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = REPRESENTATIONS[rep_key]
    path = REPO / "models/outputs/workshop" / cfg["run"] / f"seed_{seed}" / f"{split}_latents.npz"
    npz = np.load(path, allow_pickle=True)
    peptide = np.asarray(npz["peptide"]).astype(str)
    label = np.asarray(npz["label"]).astype(int)
    Z = np.asarray(npz[cfg["key"]], dtype=np.float64)
    return peptide, label, Z


def matched_m1_peptide(
    pos: np.ndarray,
    neg: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[float, int, str]:
    """Return (mean of reference-specific medians, n_refs_used, ref_scheme)."""
    n_pos = len(pos)
    if n_pos == 0 or len(neg) == 0:
        return float("nan"), 0, "none"
    # Precompute all neg→pos distances once.
    D_all = dist_to_refs(neg, pos, mode)  # (n_neg, n_pos)
    if n_pos <= EXHAUSTIVE_M1_MAX:
        ref_medians = np.median(D_all, axis=0)
        return float(np.mean(ref_medians)), int(n_pos), "exhaustive"
    idx = rng.integers(0, n_pos, size=N_REF_BOOT)
    ref_medians = np.median(D_all[:, idx], axis=0)
    return float(np.mean(ref_medians)), int(N_REF_BOOT), "bootstrap_1000"


def matched_m2_peptide(
    pos: np.ndarray,
    neg: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[float, int]:
    """Return (mean of 1,000 two-ref medians, n_replicates)."""
    n_pos = len(pos)
    if n_pos < 2 or len(neg) == 0:
        return float("nan"), 0
    D_all = dist_to_refs(neg, pos, mode)  # (n_neg, n_pos)
    # Sample 1,000 unordered pairs of distinct positives
    i = rng.integers(0, n_pos, size=N_REF_BOOT)
    j = rng.integers(0, n_pos - 1, size=N_REF_BOOT)
    j = j + (j >= i)
    nn = np.minimum(D_all[:, i], D_all[:, j])  # (n_neg, n_boot)
    medians = np.median(nn, axis=0)
    return float(np.mean(medians)), N_REF_BOOT


def pairwise_peptide(
    pos: np.ndarray,
    neg: np.ndarray,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    n_pos, n_neg = len(pos), len(neg)
    avail = {}
    if n_pos >= 1 and n_neg >= 1:
        avail["pn"] = n_pos * n_neg
    if n_pos >= 2:
        avail["pp"] = n_pos * (n_pos - 1) // 2
    if n_neg >= 2:
        avail["nn"] = n_neg * (n_neg - 1) // 2
    if not avail:
        return {
            "n_pair_sampled": 0,
            "median_pn": np.nan,
            "median_pp": np.nan,
            "median_nn": np.nan,
            "pn_minus_pp": np.nan,
            "pn_div_pp": np.nan,
            "cliffs_delta_pn_vs_pp": np.nan,
        }

    n_sample = min(N_PAIR_MAX, min(avail.values()))
    out: dict = {"n_pair_sampled": int(n_sample)}
    dists: Dict[str, np.ndarray] = {}
    if "pn" in avail:
        dists["pn"] = pairwise_sample_distances(pos, neg, mode, n_sample, rng, same_set=False)
        out["median_pn"] = float(np.median(dists["pn"]))
    else:
        out["median_pn"] = float("nan")
    if "pp" in avail:
        dists["pp"] = pairwise_sample_distances(pos, pos, mode, n_sample, rng, same_set=True)
        out["median_pp"] = float(np.median(dists["pp"]))
    else:
        out["median_pp"] = float("nan")
    if "nn" in avail:
        dists["nn"] = pairwise_sample_distances(neg, neg, mode, n_sample, rng, same_set=True)
        out["median_nn"] = float(np.median(dists["nn"]))
    else:
        out["median_nn"] = float("nan")

    if "pn" in dists and "pp" in dists:
        out["pn_minus_pp"] = float(out["median_pn"] - out["median_pp"])
        out["pn_div_pp"] = (
            float(out["median_pn"] / out["median_pp"]) if out["median_pp"] > 0 else float("nan")
        )
        out["cliffs_delta_pn_vs_pp"] = cliffs_delta(dists["pn"], dists["pp"])
    else:
        out["pn_minus_pp"] = float("nan")
        out["pn_div_pp"] = float("nan")
        out["cliffs_delta_pn_vs_pp"] = float("nan")
    return out


def concentration_peptide(
    Z_all: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """Latent concentration on all TCR vectors for a peptide (cosine pairs)."""
    X = np.asarray(Z_all, dtype=np.float64)
    n, d = X.shape
    norms = np.linalg.norm(X, axis=1)
    mean_norm = float(np.mean(norms)) if n else float("nan")

    # Per-dimension variance / covariance (unbiased if n>1)
    if n >= 2:
        var_dim = np.var(X, axis=0, ddof=1)
        mean_var = float(np.mean(var_dim))
        # Covariance eigenvalues for effective rank
        Xc = X - X.mean(axis=0, keepdims=True)
        # Use SVD of centred data: eigenvalues of cov = s^2 / (n-1)
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)
        eig = (s ** 2) / max(n - 1, 1)
        eig = np.maximum(eig, 0.0)
        trace = float(np.sum(eig))
        if trace > 0:
            p = eig / trace
            p = p[p > 0]
            eff_rank = float(np.exp(-np.sum(p * np.log(p))))
        else:
            eff_rank = float("nan")
    else:
        mean_var = float("nan")
        trace = float("nan")
        eff_rank = float("nan")

    # Median pairwise cosine distance
    if n >= 2:
        Xu = l2_normalize(X)
        n_pairs = min(N_PAIR_MAX, n * (n - 1) // 2)
        dcos = pairwise_sample_distances(Xu, Xu, "cosine_unit", n_pairs, rng, same_set=True)
        med_cos = float(np.median(dcos))
    else:
        med_cos = float("nan")
        n_pairs = 0

    return {
        "n_tcr": int(n),
        "n_pairwise_cosine_sampled": int(n_pairs if n >= 2 else 0),
        "median_pairwise_cosine_distance": med_cos,
        "mean_per_dim_variance": mean_var,
        "covariance_trace": trace,
        "effective_covariance_rank": eff_rank,
        "mean_latent_norm": mean_norm,
        "latent_dim": int(d),
    }


def analyse_seed_split_rep(
    split: str,
    seed: int,
    rep_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    peptide, label, Z_raw = load_split_arrays(split, seed, rep_key)
    cfg = REPRESENTATIONS[rep_key]
    Z, mode = prepare_space(Z_raw, cfg["distance"])
    # Raw (unnormalised) copy for concentration variance / norms
    Z_raw = np.asarray(Z_raw, dtype=np.float64)

    m1_rows: List[dict] = []
    m2_rows: List[dict] = []
    pair_rows: List[dict] = []
    conc_rows: List[dict] = []

    for pep in sorted(set(peptide.tolist())):
        idx = np.where(peptide == pep)[0]
        pos_idx = idx[label[idx] == 1]
        neg_idx = idx[label[idx] == 0]
        pos, neg = Z[pos_idx], Z[neg_idx]
        rng = np.random.default_rng(
            BOOT_SEED + zlib_salt(f"{rep_key}|{split}|{seed}|{pep}")
        )

        meta = {
            "split": split,
            "seed": int(seed),
            "representation": rep_key,
            "representation_label": cfg["label"],
            "family": cfg["family"],
            "distance_metric": cfg["distance"],
            "peptide": pep,
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
        }

        if len(pos_idx) >= 1 and len(neg_idx) >= 1:
            val, n_refs, scheme = matched_m1_peptide(pos, neg, mode, rng)
            m1_rows.append({
                **meta,
                "m": 1,
                "matched_median_neg_to_ref": val,
                "n_references_used": n_refs,
                "reference_scheme": scheme,
            })

        if len(pos_idx) >= 2 and len(neg_idx) >= 1:
            val2, n_rep = matched_m2_peptide(pos, neg, mode, rng)
            m2_rows.append({
                **meta,
                "m": 2,
                "matched_median_neg_to_nearest_of_two": val2,
                "n_bootstrap_refs": n_rep,
            })

        if len(pos_idx) >= 1 or len(neg_idx) >= 1:
            pw = pairwise_peptide(pos, neg, mode, rng)
            pair_rows.append({**meta, **pw})

        # Concentration: all TCR vectors for the peptide (pos + neg)
        conc = concentration_peptide(Z_raw[idx], rng)
        conc_rows.append({
            "split": split,
            "seed": int(seed),
            "representation": rep_key,
            "representation_label": cfg["label"],
            "family": cfg["family"],
            "distance_metric": cfg["distance"],
            "peptide": pep,
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            **conc,
        })

    return (
        pd.DataFrame(m1_rows),
        pd.DataFrame(m2_rows),
        pd.DataFrame(pair_rows),
        pd.DataFrame(conc_rows),
    )


def average_over_seeds(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keys = [
        "split",
        "representation",
        "representation_label",
        "family",
        "distance_metric",
        "peptide",
    ]
    # Keep first-available count columns
    count_cols = [c for c in ("n_positive", "n_negative", "n_tcr", "latent_dim") if c in df.columns]
    rows = []
    for key, g in df.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row["n_seeds"] = int(g["seed"].nunique())
        for c in count_cols:
            row[c] = int(g[c].iloc[0]) if pd.notna(g[c].iloc[0]) else 0
        for c in value_cols:
            if c not in g.columns:
                continue
            vals = g[c].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            row[c] = float(np.mean(vals)) if len(vals) else float("nan")
            if len(vals) > 1:
                row[f"{c}_seed_std"] = float(np.std(vals, ddof=1))
            else:
                row[f"{c}_seed_std"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    med = float(np.median(values))
    if len(values) == 1:
        return mean, med, mean, mean
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return mean, med, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarise_metric(
    per_pep_avg: pd.DataFrame,
    value_col: str,
    analysis: str,
) -> pd.DataFrame:
    """Peptide-balanced mean/median + CI; IMMREP−test difference."""
    rows: List[dict] = []
    lookup: Dict[Tuple[str, str], Tuple[dict, np.ndarray]] = {}

    for (split, rep), g in per_pep_avg.groupby(["split", "representation"], sort=True):
        vals = g[value_col].to_numpy(float)
        rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"{analysis}|{value_col}|{split}|{rep}"))
        mean, med, lo, hi = bootstrap_mean_ci(vals, N_BOOT_PEP, rng)
        row = {
            "analysis": analysis,
            "metric": value_col,
            "split": split,
            "representation": rep,
            "representation_label": g["representation_label"].iloc[0],
            "family": g["family"].iloc[0],
            "distance_metric": g["distance_metric"].iloc[0],
            "n_peptides": int(np.sum(np.isfinite(vals))),
            "peptide_balanced_mean": mean,
            "peptide_median": med,
            "ci_low": lo,
            "ci_high": hi,
            "n_bootstrap": N_BOOT_PEP,
        }
        rows.append(row)
        lookup[(split, rep)] = (row, vals[np.isfinite(vals)])

    for rep in REPRESENTATIONS:
        if ("test", rep) not in lookup or ("immrep_test", rep) not in lookup:
            continue
        test_row, test_vals = lookup[("test", rep)]
        imm_row, imm_vals = lookup[("immrep_test", rep)]
        rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"diff|{analysis}|{value_col}|{rep}"))
        diffs = np.empty(N_BOOT_PEP, dtype=np.float64)
        for b in range(N_BOOT_PEP):
            t = test_vals[rng.integers(0, len(test_vals), size=len(test_vals))].mean()
            i = imm_vals[rng.integers(0, len(imm_vals), size=len(imm_vals))].mean()
            diffs[b] = i - t
        delta = imm_row["peptide_balanced_mean"] - test_row["peptide_balanced_mean"]
        rows.append({
            "analysis": analysis,
            "metric": value_col,
            "split": "comparison_immrep_minus_test",
            "representation": rep,
            "representation_label": test_row["representation_label"],
            "family": test_row["family"],
            "distance_metric": test_row["distance_metric"],
            "n_peptides_test": test_row["n_peptides"],
            "n_peptides_immrep": imm_row["n_peptides"],
            "test_peptide_balanced_mean": test_row["peptide_balanced_mean"],
            "test_peptide_median": test_row["peptide_median"],
            "test_ci_low": test_row["ci_low"],
            "test_ci_high": test_row["ci_high"],
            "immrep_peptide_balanced_mean": imm_row["peptide_balanced_mean"],
            "immrep_peptide_median": imm_row["peptide_median"],
            "immrep_ci_low": imm_row["ci_low"],
            "immrep_ci_high": imm_row["ci_high"],
            "immrep_minus_test": float(delta),
            "immrep_minus_test_ci_low": float(np.quantile(diffs, 0.025)),
            "immrep_minus_test_ci_high": float(np.quantile(diffs, 0.975)),
            "n_bootstrap": N_BOOT_PEP,
            "note": (
                "No shared peptides between splits. "
                "Negative Δ for distance metrics ⇒ IMMREP negatives closer / harder."
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    global N_BOOT_PEP, N_REF_BOOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-boot-pep", type=int, default=N_BOOT_PEP)
    parser.add_argument("--n-ref-boot", type=int, default=N_REF_BOOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--reps",
        nargs="+",
        default=list(REPRESENTATIONS.keys()),
        choices=list(REPRESENTATIONS.keys()),
    )
    args = parser.parse_args()

    N_BOOT_PEP = args.n_boot_pep
    N_REF_BOOT = args.n_ref_boot
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    m1_parts, m2_parts, pair_parts, conc_parts = [], [], [], []
    for rep in args.reps:
        for split in SPLITS:
            for seed in args.seeds:
                print(f"[matched] {rep} {split} seed={seed}", flush=True)
                m1, m2, pair, conc = analyse_seed_split_rep(split, seed, rep)
                m1_parts.append(m1)
                m2_parts.append(m2)
                pair_parts.append(pair)
                conc_parts.append(conc)

    m1_df = pd.concat(m1_parts, ignore_index=True)
    m2_df = pd.concat(m2_parts, ignore_index=True)
    pair_df = pd.concat(pair_parts, ignore_index=True)
    conc_df = pd.concat(conc_parts, ignore_index=True)

    detail_dir = out_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    m1_path = detail_dir / "negative_difficulty_matched_m1.csv"
    m2_path = detail_dir / "negative_difficulty_matched_m2.csv"
    m1_df.to_csv(m1_path, index=False)
    m2_df.to_csv(m2_path, index=False)

    # Seed-average per-peptide tables (full distributions for plotting)
    m1_avg = average_over_seeds(m1_df, ["matched_median_neg_to_ref"])
    m2_avg = average_over_seeds(m2_df, ["matched_median_neg_to_nearest_of_two"])
    pair_avg = average_over_seeds(
        pair_df,
        [
            "median_pn",
            "median_pp",
            "median_nn",
            "pn_minus_pp",
            "pn_div_pp",
            "cliffs_delta_pn_vs_pp",
        ],
    )
    conc_avg = average_over_seeds(
        conc_df,
        [
            "median_pairwise_cosine_distance",
            "mean_per_dim_variance",
            "covariance_trace",
            "effective_covariance_rank",
            "mean_latent_norm",
        ],
    )

    m1_avg.to_csv(out_dir / "negative_difficulty_matched_m1_per_peptide_avg.csv", index=False)
    m2_avg.to_csv(out_dir / "negative_difficulty_matched_m2_per_peptide_avg.csv", index=False)
    pair_df.to_csv(detail_dir / "pairwise_tcr_separation_per_peptide.csv", index=False)
    pair_avg.to_csv(out_dir / "pairwise_tcr_separation_per_peptide_avg.csv", index=False)
    conc_df.to_csv(detail_dir / "latent_concentration_per_peptide.csv", index=False)
    conc_avg.to_csv(out_dir / "latent_concentration_per_peptide_avg.csv", index=False)

    # Summaries
    summary_parts = [
        summarise_metric(m1_avg, "matched_median_neg_to_ref", "matched_m1"),
        summarise_metric(m2_avg, "matched_median_neg_to_nearest_of_two", "matched_m2"),
    ]
    for col in [
        "median_pn",
        "median_pp",
        "median_nn",
        "pn_minus_pp",
        "pn_div_pp",
        "cliffs_delta_pn_vs_pp",
    ]:
        summary_parts.append(summarise_metric(pair_avg, col, "pairwise_separation"))
    for col in [
        "median_pairwise_cosine_distance",
        "mean_per_dim_variance",
        "covariance_trace",
        "effective_covariance_rank",
        "mean_latent_norm",
    ]:
        summary_parts.append(summarise_metric(conc_avg, col, "latent_concentration"))

    summary = pd.concat(summary_parts, ignore_index=True)
    sum_path = out_dir / "negative_difficulty_matched_summary.csv"
    summary.to_csv(sum_path, index=False)

    # Compact matched-only console report
    print("\n=== Matched-m negative difficulty (peptide-balanced mean [95% CI]) ===", flush=True)
    for analysis, col in [
        ("matched_m1", "matched_median_neg_to_ref"),
        ("matched_m2", "matched_median_neg_to_nearest_of_two"),
    ]:
        print(f"\n-- {analysis} --", flush=True)
        for rep in args.reps:
            st = summary[
                (summary.analysis == analysis)
                & (summary.metric == col)
                & (summary.split == "test")
                & (summary.representation == rep)
            ]
            si = summary[
                (summary.analysis == analysis)
                & (summary.metric == col)
                & (summary.split == "immrep_test")
                & (summary.representation == rep)
            ]
            sc = summary[
                (summary.analysis == analysis)
                & (summary.metric == col)
                & (summary.split == "comparison_immrep_minus_test")
                & (summary.representation == rep)
            ]
            if st.empty or si.empty or sc.empty:
                continue
            st, si, sc = st.iloc[0], si.iloc[0], sc.iloc[0]
            print(
                f"{REPRESENTATIONS[rep]['label']}\n"
                f"  test   {st.peptide_balanced_mean:.4f} "
                f"[{st.ci_low:.4f}, {st.ci_high:.4f}]  "
                f"median={st.peptide_median:.4f}  (n_pep={int(st.n_peptides)})\n"
                f"  IMMREP {si.peptide_balanced_mean:.4f} "
                f"[{si.ci_low:.4f}, {si.ci_high:.4f}]  "
                f"median={si.peptide_median:.4f}  (n_pep={int(si.n_peptides)})\n"
                f"  Δ(IMMREP−test) {sc.immrep_minus_test:.4f} "
                f"[{sc.immrep_minus_test_ci_low:.4f}, {sc.immrep_minus_test_ci_high:.4f}]",
                flush=True,
            )

    print("\n=== Pairwise PN−PP (peptide-balanced) ===", flush=True)
    for rep in args.reps:
        sc = summary[
            (summary.analysis == "pairwise_separation")
            & (summary.metric == "pn_minus_pp")
            & (summary.split == "comparison_immrep_minus_test")
            & (summary.representation == rep)
        ]
        st = summary[
            (summary.analysis == "pairwise_separation")
            & (summary.metric == "pn_minus_pp")
            & (summary.split == "test")
            & (summary.representation == rep)
        ]
        si = summary[
            (summary.analysis == "pairwise_separation")
            & (summary.metric == "pn_minus_pp")
            & (summary.split == "immrep_test")
            & (summary.representation == rep)
        ]
        if sc.empty:
            continue
        st, si, sc = st.iloc[0], si.iloc[0], sc.iloc[0]
        print(
            f"{REPRESENTATIONS[rep]['label']}: "
            f"test {st.peptide_balanced_mean:.4f} | "
            f"IMMREP {si.peptide_balanced_mean:.4f} | "
            f"Δ {sc.immrep_minus_test:.4f} "
            f"[{sc.immrep_minus_test_ci_low:.4f}, {sc.immrep_minus_test_ci_high:.4f}]",
            flush=True,
        )

    print("\n=== Latent concentration: median pairwise cosine (IMMREP−test) ===", flush=True)
    for rep in args.reps:
        sc = summary[
            (summary.analysis == "latent_concentration")
            & (summary.metric == "median_pairwise_cosine_distance")
            & (summary.split == "comparison_immrep_minus_test")
            & (summary.representation == rep)
        ]
        if sc.empty:
            continue
        sc = sc.iloc[0]
        print(
            f"{REPRESENTATIONS[rep]['label']}: Δ={sc.immrep_minus_test:.4f} "
            f"[{sc.immrep_minus_test_ci_low:.4f}, {sc.immrep_minus_test_ci_high:.4f}]",
            flush=True,
        )

    manifest = {
        "seeds": args.seeds,
        "n_boot_pep": N_BOOT_PEP,
        "n_ref_boot": N_REF_BOOT,
        "n_pair_max": N_PAIR_MAX,
        "exhaustive_m1_max": EXHAUSTIVE_M1_MAX,
        "reps": args.reps,
        "out_dir": str(out_dir),
    }
    (out_dir / "matched_run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {m1_path}", flush=True)
    print(f"Wrote {m2_path}", flush=True)
    print(f"Wrote {sum_path}", flush=True)
    print(f"Wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
