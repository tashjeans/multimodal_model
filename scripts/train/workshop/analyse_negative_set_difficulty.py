#!/usr/bin/env python3
"""Negative-set difficulty: nearest-positive TCR distances (no retraining).

For each split × representation × peptide:
  - positives = binder TCRs for that peptide
  - for each negative TCR, distance to nearest same-peptide positive TCR
  - summarise median nearest-positive distance per peptide

Compare internal-test decoys vs IMMREP negatives with peptide-balanced means
(equal peptide weight) and peptide-bootstrap 95% CIs.

Note: workshop internal-test and IMMREP share **zero** peptides, so
"corresponding" comparisons are distributional (IMMREP peptide median vs
internal-test peptide-balanced reference), not matched peptide IDs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
OUT_DIR = REPO / "models/outputs/workshop/paper_analysis/negative_set_difficulty"
FIG_DIR = REPO / "models/figures/workshop/paper_analysis/negative_set_difficulty"

SEEDS = [31, 37, 43, 49, 55]
SPLITS = ["test", "immrep_test"]
N_BOOT = 2000
BOOT_SEED = 20260806

# Representation loaders: (run_dir, array_key, distance)
# distance: "euclidean" | "cosine"
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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def pairwise_min_distance(
    neg: np.ndarray,
    pos: np.ndarray,
    distance: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_dist per neg, index of nearest pos)."""
    if distance == "euclidean":
        # (n_neg, n_pos)
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        aa = np.sum(neg * neg, axis=1, keepdims=True)
        bb = np.sum(pos * pos, axis=1, keepdims=True).T
        dots = neg @ pos.T
        d2 = np.maximum(aa + bb - 2.0 * dots, 0.0)
        nn_idx = np.argmin(d2, axis=1)
        nn_dist = np.sqrt(d2[np.arange(len(neg)), nn_idx])
        return nn_dist.astype(np.float64), nn_idx.astype(np.int64)
    if distance == "cosine":
        neg_u = l2_normalize(neg)
        pos_u = l2_normalize(pos)
        sims = neg_u @ pos_u.T
        nn_idx = np.argmax(sims, axis=1)
        nn_dist = 1.0 - sims[np.arange(len(neg)), nn_idx]
        return nn_dist.astype(np.float64), nn_idx.astype(np.int64)
    raise ValueError(distance)


def load_split_arrays(split: str, seed: int, rep_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = REPRESENTATIONS[rep_key]
    path = REPO / "models/outputs/workshop" / cfg["run"] / f"seed_{seed}" / f"{split}_latents.npz"
    npz = np.load(path, allow_pickle=True)
    pair_id = np.asarray(npz["pair_id"]).astype(str)
    peptide = np.asarray(npz["peptide"]).astype(str)
    label = np.asarray(npz["label"]).astype(int)
    Z = np.asarray(npz[cfg["key"]], dtype=np.float64)
    return pair_id, peptide, label, Z


def analyse_seed_split_rep(
    split: str,
    seed: int,
    rep_key: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pair_id, peptide, label, Z = load_split_arrays(split, seed, rep_key)
    distance = REPRESENTATIONS[rep_key]["distance"]
    row_rows: List[dict] = []
    pep_rows: List[dict] = []

    for pep in sorted(set(peptide.tolist())):
        idx = np.where(peptide == pep)[0]
        pos_idx = idx[label[idx] == 1]
        neg_idx = idx[label[idx] == 0]
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue
        nn_dist, nn_local = pairwise_min_distance(Z[neg_idx], Z[pos_idx], distance)
        nearest_pos_pair = pair_id[pos_idx[nn_local]]
        for j, row_i in enumerate(neg_idx):
            row_rows.append({
                "split": split,
                "seed": seed,
                "representation": rep_key,
                "representation_label": REPRESENTATIONS[rep_key]["label"],
                "family": REPRESENTATIONS[rep_key]["family"],
                "distance_metric": distance,
                "peptide": pep,
                "pair_id": pair_id[row_i],
                "nearest_positive_pair_id": nearest_pos_pair[j],
                "nearest_positive_distance": float(nn_dist[j]),
                "n_positives_for_peptide": int(len(pos_idx)),
                "n_negatives_for_peptide": int(len(neg_idx)),
            })
        pep_rows.append({
            "split": split,
            "seed": seed,
            "representation": rep_key,
            "representation_label": REPRESENTATIONS[rep_key]["label"],
            "family": REPRESENTATIONS[rep_key]["family"],
            "distance_metric": distance,
            "peptide": pep,
            "n_positive": int(len(pos_idx)),
            "n_negative": int(len(neg_idx)),
            "median_nearest_positive_distance": float(np.median(nn_dist)),
            "mean_nearest_positive_distance": float(np.mean(nn_dist)),
            "q25_nearest_positive_distance": float(np.quantile(nn_dist, 0.25)),
            "q75_nearest_positive_distance": float(np.quantile(nn_dist, 0.75)),
        })
    return pd.DataFrame(row_rows), pd.DataFrame(pep_rows)


def average_peptides_over_seeds(per_pep: pd.DataFrame) -> pd.DataFrame:
    """Mean of per-seed peptide medians (same peptide across seeds)."""
    keys = ["split", "representation", "representation_label", "family", "distance_metric", "peptide"]
    rows = []
    for key, g in per_pep.groupby(keys, sort=True):
        rows.append({
            **dict(zip(keys, key)),
            "n_seeds": int(g["seed"].nunique()),
            "n_positive": int(g["n_positive"].iloc[0]),
            "n_negative": int(g["n_negative"].iloc[0]),
            "median_nearest_positive_distance": float(g["median_nearest_positive_distance"].mean()),
            "median_nearest_positive_distance_seed_std": float(g["median_nearest_positive_distance"].std(ddof=1))
            if len(g) > 1 else 0.0,
            "mean_nearest_positive_distance": float(g["mean_nearest_positive_distance"].mean()),
        })
    return pd.DataFrame(rows)


def bootstrap_peptide_mean(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(np.mean(values))
    if len(values) == 1:
        return point, point, point
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = values[idx].mean(axis=1)
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarise_across_peptides(per_pep_avg: pd.DataFrame) -> pd.DataFrame:
    """Peptide-balanced means + bootstrap CIs; IMMREP vs internal comparison."""
    rows = []
    # First compute per split×rep summaries
    summary_lookup = {}
    for (split, rep), g in per_pep_avg.groupby(["split", "representation"], sort=True):
        vals = g["median_nearest_positive_distance"].to_numpy(float)
        rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"{split}|{rep}"))
        mean, lo, hi = bootstrap_peptide_mean(vals, N_BOOT, rng)
        med = float(np.median(vals))
        row = {
            "split": split,
            "representation": rep,
            "representation_label": g["representation_label"].iloc[0],
            "family": g["family"].iloc[0],
            "distance_metric": g["distance_metric"].iloc[0],
            "n_peptides": int(len(g)),
            "peptide_balanced_mean_median_nn_dist": mean,
            "peptide_balanced_mean_ci_low": lo,
            "peptide_balanced_mean_ci_high": hi,
            "peptide_median_of_medians": med,
            "peptide_std_of_medians": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n_bootstrap": N_BOOT,
        }
        rows.append(row)
        summary_lookup[(split, rep)] = (row, vals)

    # Comparison rows
    for rep in REPRESENTATIONS:
        if ("test", rep) not in summary_lookup or ("immrep_test", rep) not in summary_lookup:
            continue
        test_row, test_vals = summary_lookup[("test", rep)]
        imm_row, imm_vals = summary_lookup[("immrep_test", rep)]
        # Peptide-bootstrap of difference: resample each split's peptides independently
        rng = np.random.default_rng(BOOT_SEED + zlib_salt(f"diff|{rep}"))
        n_boot = N_BOOT
        diffs = np.empty(n_boot, dtype=np.float64)
        for b in range(n_boot):
            t = test_vals[rng.integers(0, len(test_vals), size=len(test_vals))].mean()
            i = imm_vals[rng.integers(0, len(imm_vals), size=len(imm_vals))].mean()
            diffs[b] = i - t  # IMMREP minus test: negative => IMMREP harder (closer)
        delta = imm_row["peptide_balanced_mean_median_nn_dist"] - test_row["peptide_balanced_mean_median_nn_dist"]
        # Proportion of IMMREP peptides closer than internal peptide-balanced mean
        ref_mean = test_row["peptide_balanced_mean_median_nn_dist"]
        ref_median = test_row["peptide_median_of_medians"]
        prop_vs_mean = float(np.mean(imm_vals < ref_mean))
        prop_vs_median = float(np.mean(imm_vals < ref_median))
        rows.append({
            "split": "comparison_immrep_vs_test",
            "representation": rep,
            "representation_label": test_row["representation_label"],
            "family": test_row["family"],
            "distance_metric": test_row["distance_metric"],
            "n_peptides_test": test_row["n_peptides"],
            "n_peptides_immrep": imm_row["n_peptides"],
            "test_peptide_balanced_mean": test_row["peptide_balanced_mean_median_nn_dist"],
            "test_ci_low": test_row["peptide_balanced_mean_ci_low"],
            "test_ci_high": test_row["peptide_balanced_mean_ci_high"],
            "immrep_peptide_balanced_mean": imm_row["peptide_balanced_mean_median_nn_dist"],
            "immrep_ci_low": imm_row["peptide_balanced_mean_ci_low"],
            "immrep_ci_high": imm_row["peptide_balanced_mean_ci_high"],
            "immrep_minus_test_mean": float(delta),
            "immrep_minus_test_ci_low": float(np.quantile(diffs, 0.025)),
            "immrep_minus_test_ci_high": float(np.quantile(diffs, 0.975)),
            "prop_immrep_peptides_closer_than_test_mean": prop_vs_mean,
            "prop_immrep_peptides_closer_than_test_median": prop_vs_median,
            "note": (
                "No shared peptides between splits. "
                "prop_* = fraction of IMMREP peptides whose median NN-pos distance "
                "is smaller (harder) than the internal-test peptide-balanced mean/median."
            ),
            "n_bootstrap": N_BOOT,
        })
    return pd.DataFrame(rows)


def zlib_salt(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8")) & 0x7FFFFFFF


def plot_per_peptide(per_pep_avg: pd.DataFrame, summary: pd.DataFrame, out_pdf: Path) -> None:
    reps = list(REPRESENTATIONS.keys())
    n = len(reps)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=False)
    axes = axes.ravel()
    for ax, rep in zip(axes, reps):
        test = per_pep_avg[(per_pep_avg.representation == rep) & (per_pep_avg.split == "test")]
        imm = per_pep_avg[(per_pep_avg.representation == rep) & (per_pep_avg.split == "immrep_test")]
        # Strip / swarm-like: jittered points
        rng = np.random.default_rng(0)
        yt = test["median_nearest_positive_distance"].to_numpy()
        yi = imm["median_nearest_positive_distance"].to_numpy()
        ax.scatter(
            np.full(len(yt), 0) + rng.uniform(-0.12, 0.12, len(yt)),
            yt, s=10, alpha=0.35, color="#4c78a8", label="Internal test peptides",
        )
        ax.scatter(
            np.full(len(yi), 1) + rng.uniform(-0.12, 0.12, len(yi)),
            yi, s=28, alpha=0.85, color="#f58518", label="IMMREP peptides",
        )
        # Peptide-balanced means with CI
        st = summary[(summary.split == "test") & (summary.representation == rep)]
        si = summary[(summary.split == "immrep_test") & (summary.representation == rep)]
        if len(st) and len(si):
            for x, sub, c in [(0, st.iloc[0], "#4c78a8"), (1, si.iloc[0], "#f58518")]:
                m = sub["peptide_balanced_mean_median_nn_dist"]
                lo = sub["peptide_balanced_mean_ci_low"]
                hi = sub["peptide_balanced_mean_ci_high"]
                ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt="D", color=c,
                            ecolor=c, elinewidth=1.5, capsize=4, markersize=6, zorder=5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal\ntest", "IMMREP"])
        ax.set_title(REPRESENTATIONS[rep]["label"], fontsize=9)
        ax.set_ylabel("Median NN-positive distance")
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        "Per-peptide median nearest-positive TCR distance\n"
        "(diamonds = peptide-balanced mean ± 95% peptide-bootstrap CI; no shared peptides)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path) -> None:
    text = f"""# Negative-set difficulty (nearest-positive TCR distances)

Computed from saved embeddings only (no retraining).

## Method
For each split, representation and peptide with ≥1 positive and ≥1 negative:
1. Collect positive TCR embeddings for that peptide.
2. For every negative TCR, compute distance to the **nearest** positive TCR.
3. Summarise the **median** nearest-positive distance within the peptide.

Representations:
- `onehot_composition`: Euclidean on amino-acid composition (`T_composition`)
- `raw_esmc_meanpool`: cosine distance on pretrained ESMC mean-pool
- `lora_esmc_meanpool`: cosine distance on LoRA-finetuned ESMC mean-pool
- `*_vicreg`: cosine distance on learned VICReg TCR latents

Cosine distance = 1 − cosine similarity.

## Peptide balancing
Across-peptide summaries use the **mean of per-peptide medians** (one vote per peptide).
95% CIs are peptide-level bootstrap (2000 resamples). Do not interpret pooled-row
means.

Seeds 31/37/43: per-peptide medians are averaged across seeds before summarising.

## Internal test vs IMMREP
The workshop internal-test and IMMREP peptide sets are **disjoint** (0 shared peptides).
Comparisons are therefore distributional:
- peptide-balanced mean ± CI on each split
- Δ = IMMREP mean − internal mean (negative ⇒ IMMREP negatives closer / harder)
- `prop_immrep_peptides_closer_than_test_mean`: fraction of IMMREP peptides whose
  median NN-pos distance is below the internal-test peptide-balanced mean

## Outputs
| File | Content |
|--|--|
| `nearest_positive_distances_rowlevel.csv` | Every negative row |
| `nearest_positive_distances_per_peptide.csv` | Per seed×peptide medians |
| `nearest_positive_distances_per_peptide_avg.csv` | Seed-averaged peptide table |
| `nearest_positive_distances_summary.csv` | Peptide-balanced means, CIs, comparison |
| `nearest_positive_distance_test_vs_immrep.pdf` | Per-peptide plot |
| `README_negative_set_difficulty.md` | This file |
"""
    (out_dir / "README_negative_set_difficulty.md").write_text(text)


def main() -> None:
    global N_BOOT

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    N_BOOT = args.n_boot
    seeds = args.seeds
    out_dir = args.out_dir
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    row_parts = []
    pep_parts = []
    for rep in REPRESENTATIONS:
        for split in SPLITS:
            for seed in seeds:
                print(f"[nn] {rep} {split} seed={seed}", flush=True)
                rows, peps = analyse_seed_split_rep(split, seed, rep)
                row_parts.append(rows)
                pep_parts.append(peps)

    row_df = pd.concat(row_parts, ignore_index=True)
    pep_df = pd.concat(pep_parts, ignore_index=True)
    pep_avg = average_peptides_over_seeds(pep_df)
    summary = summarise_across_peptides(pep_avg)

    detail_dir = out_dir / "detail"
    detail_dir.mkdir(parents=True, exist_ok=True)
    # Row-level nearest-positive distances are huge; keep only peptide aggregates by default.
    # Set WRITE_ROWLEVEL=1 to dump them under detail/.
    row_path = detail_dir / "nearest_positive_distances_rowlevel.csv"
    pep_path = detail_dir / "nearest_positive_distances_per_peptide.csv"
    pep_avg_path = out_dir / "nearest_positive_distances_per_peptide_avg.csv"
    sum_path = out_dir / "nearest_positive_distances_summary.csv"
    if os.environ.get("WRITE_ROWLEVEL", "0") == "1":
        row_df.to_csv(row_path, index=False)
    pep_df.to_csv(pep_path, index=False)
    pep_avg.to_csv(pep_avg_path, index=False)
    summary.to_csv(sum_path, index=False)

    plot_path = out_dir / "nearest_positive_distance_test_vs_immrep.pdf"
    plot_per_peptide(pep_avg, summary, plot_path)
    plot_per_peptide(pep_avg, summary, fig_dir / "nearest_positive_distance_test_vs_immrep.pdf")
    write_readme(out_dir)

    # Console report
    print("\n=== Peptide-balanced mean median-NN distance (mean [95% CI]) ===", flush=True)
    for rep in REPRESENTATIONS:
        st = summary[(summary.split == "test") & (summary.representation == rep)]
        si = summary[(summary.split == "immrep_test") & (summary.representation == rep)]
        sc = summary[(summary.split == "comparison_immrep_vs_test") & (summary.representation == rep)]
        if len(st) == 0:
            continue
        st, si, sc = st.iloc[0], si.iloc[0], sc.iloc[0]
        print(
            f"{REPRESENTATIONS[rep]['label']}\n"
            f"  test   {st.peptide_balanced_mean_median_nn_dist:.4f} "
            f"[{st.peptide_balanced_mean_ci_low:.4f}, {st.peptide_balanced_mean_ci_high:.4f}] "
            f"(n_pep={int(st.n_peptides)})\n"
            f"  IMMREP {si.peptide_balanced_mean_median_nn_dist:.4f} "
            f"[{si.peptide_balanced_mean_ci_low:.4f}, {si.peptide_balanced_mean_ci_high:.4f}] "
            f"(n_pep={int(si.n_peptides)})\n"
            f"  Δ(IMMREP−test) {sc.immrep_minus_test_mean:.4f} "
            f"[{sc.immrep_minus_test_ci_low:.4f}, {sc.immrep_minus_test_ci_high:.4f}]\n"
            f"  prop IMMREP peptides closer than test mean: "
            f"{sc.prop_immrep_peptides_closer_than_test_mean:.2f}",
            flush=True,
        )

    # Overlap check
    tpeps = set(pep_avg.loc[pep_avg.split == "test", "peptide"])
    ipeps = set(pep_avg.loc[pep_avg.split == "immrep_test", "peptide"])
    print(f"\nShared peptides test∩IMMREP: {len(tpeps & ipeps)} (expected 0)", flush=True)
    print(f"Wrote {out_dir}", flush=True)
    (out_dir / "run_manifest.json").write_text(json.dumps({
        "seeds": seeds, "n_boot": N_BOOT, "representationss": list(REPRESENTATIONS), "out_dir": str(out_dir),
    }, indent=2))


if __name__ == "__main__":
    main()
