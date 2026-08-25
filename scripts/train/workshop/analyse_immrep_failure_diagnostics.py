#!/usr/bin/env python3
"""IMMREP failure diagnostics from saved VICReg latents (no retraining).

Analyses (one-hot / raw ESMC / LoRA ESMC VICReg; seeds 31/37/43;
internal test + IMMREP):

  1) Score decomposition AUROCs from z_T, z_pMHC
  2) Per-peptide IMMREP diagnostics + forest plot
  3) Inference-time pMHC-unit permutation dependence diagnostic

Does not load checkpoints or change selection; uses {split}_latents.npz only
(and multiview CSVs solely for HLA_sequence to define pMHC units).
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/home/natasha/multimodal_model")
DEFAULT_OUT = REPO / "models/outputs/workshop/paper_analysis/immrep_failure_analysis"
DEFAULT_FIG = REPO / "models/figures/workshop/paper_analysis/immrep_failure_analysis"

MODELS = {
    "onehot_vicreg": {
        "run": "onehot_vicreg_complete",
        "zT": "zT_vicreg",
        "zPH": "zPH_vicreg",
        "label": "One-hot VICReg",
    },
    "raw_esmc_vicreg": {
        "run": "esm_vicreg_raw_complete",
        "zT": "zT_esm_vicreg",
        "zPH": "zPH_esm_vicreg",
        "label": "Raw ESMC VICReg",
    },
    "finetuned_esmc_vicreg": {
        "run": "esm_vicreg_finetuned_complete",
        "zT": "zT_esm_vicreg",
        "zPH": "zPH_esm_vicreg",
        "label": "LoRA ESMC VICReg",
    },
}

SEEDS = [31, 37, 43, 49, 55]
SPLITS = ["test", "immrep_test"]

# HLA lookup for pairing latents to pMHC units.
SPLIT_META_CSV = {
    "test": REPO / "data/test/test_multiview.csv",
    "immrep_test": REPO / "data/immrep_test/immrep_test_multiview.csv",
}

N_BOOT = 2000
N_PERM = 100
MAX_FPR = 0.1
PERM_SEED = 20260806


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def safe_auroc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_mcclish_pauc01(y: np.ndarray, s: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s, max_fpr=max_fpr))


def peptide_weighted_auroc(y: np.ndarray, s: np.ndarray, peptides: np.ndarray) -> float:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(s).astype(float), "p": np.asarray(peptides).astype(str)})
    rows = []
    for _, g in df.groupby("p"):
        if len(np.unique(g["y"])) < 2:
            continue
        rows.append((len(g), float(roc_auc_score(g["y"], g["s"]))))
    if not rows:
        return float("nan")
    ns, aucs = zip(*rows)
    return float(np.average(aucs, weights=ns))


def peptide_macro_auroc(y: np.ndarray, s: np.ndarray, peptides: np.ndarray) -> float:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(s).astype(float), "p": np.asarray(peptides).astype(str)})
    aucs = []
    for _, g in df.groupby("p"):
        if len(np.unique(g["y"])) < 2:
            continue
        aucs.append(float(roc_auc_score(g["y"], g["s"])))
    return float(np.mean(aucs)) if aucs else float("nan")


def peptide_macro_mcclish(y: np.ndarray, s: np.ndarray, peptides: np.ndarray) -> float:
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "s": np.asarray(s).astype(float), "p": np.asarray(peptides).astype(str)})
    vals = []
    for _, g in df.groupby("p"):
        v = safe_mcclish_pauc01(g["y"].to_numpy(), g["s"].to_numpy())
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Score decomposition
# ---------------------------------------------------------------------------

def score_components(zT: np.ndarray, zPH: np.ndarray) -> Dict[str, np.ndarray]:
    zT = np.asarray(zT, dtype=np.float64)
    zPH = np.asarray(zPH, dtype=np.float64)
    diff = zT - zPH
    full = -np.mean(diff * diff, axis=1)
    tcr_norm = -np.mean(zT * zT, axis=1)
    dot = np.mean(zT * zPH, axis=1)

    nt = np.linalg.norm(zT, axis=1)
    nph = np.linalg.norm(zPH, axis=1)
    denom = nt * nph
    cosine = np.divide(np.sum(zT * zPH, axis=1), denom, out=np.full(len(zT), np.nan), where=denom > 0)

    eps = 1e-12
    zT_u = zT / np.maximum(nt[:, None], eps)
    zPH_u = zPH / np.maximum(nph[:, None], eps)
    diff_u = zT_u - zPH_u
    norm_mse = -np.mean(diff_u * diff_u, axis=1)

    return {
        "full_score": full.astype(np.float64),
        "tcr_norm_score": tcr_norm.astype(np.float64),
        "dot_product_score": dot.astype(np.float64),
        "cosine_score": cosine.astype(np.float64),
        "normalised_mse_score": norm_mse.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hla_lookup(split: str) -> pd.DataFrame:
    path = SPLIT_META_CSV[split]
    df = pd.read_csv(path, usecols=["pair_id", "Peptide", "HLA_sequence", "binding_flag"])
    df["pair_id"] = df["pair_id"].astype(str)
    df["Peptide"] = df["Peptide"].astype(str)
    df["HLA_sequence"] = df["HLA_sequence"].astype(str)
    return df.drop_duplicates("pair_id").set_index("pair_id")


def load_latents(model_key: str, seed: int, split: str, hla_lookup: pd.DataFrame) -> Dict:
    cfg = MODELS[model_key]
    path = REPO / "models/outputs/workshop" / cfg["run"] / f"seed_{seed}" / f"{split}_latents.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    npz = np.load(path, allow_pickle=True)
    pair_id = np.asarray(npz["pair_id"]).astype(str)
    peptide = np.asarray(npz["peptide"]).astype(str)
    label = np.asarray(npz["label"]).astype(int)
    zT = np.asarray(npz[cfg["zT"]], dtype=np.float32)
    zPH = np.asarray(npz[cfg["zPH"]], dtype=np.float32)

    missing = [p for p in pair_id if p not in hla_lookup.index]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} pair_ids missing HLA join (e.g. {missing[:3]})")

    hla = np.asarray([hla_lookup.loc[p, "HLA_sequence"] for p in pair_id], dtype=object)
    # Prefer CSV peptide for consistency with HLA unit definition
    pep_csv = np.asarray([hla_lookup.loc[p, "Peptide"] for p in pair_id], dtype=object).astype(str)
    if not np.all(pep_csv == peptide):
        n_mismatch = int(np.sum(pep_csv != peptide))
        print(f"WARN {path.name}: {n_mismatch} peptide string mismatches vs CSV; using CSV Peptide for units", flush=True)
        peptide = pep_csv

    y_csv = np.asarray([int(hla_lookup.loc[p, "binding_flag"]) for p in pair_id])
    if not np.all(y_csv == label):
        raise RuntimeError(f"{path}: label mismatch vs binding_flag")

    return {
        "pair_id": pair_id,
        "peptide": peptide,
        "hla": hla.astype(str),
        "label": label,
        "zT": zT,
        "zPH": zPH,
        "path": str(path),
    }


# ---------------------------------------------------------------------------
# Bootstrap / derangement
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(
    y: np.ndarray,
    s: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    point = safe_auroc(y, s)
    n = len(y)
    if n < 2 or not np.isfinite(point):
        return point, float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    k = 0
    tries = 0
    max_tries = n_boot * 20
    while k < n_boot and tries < max_tries:
        tries += 1
        idx = rng.integers(0, n, size=n)
        yy, ss = y[idx], s[idx]
        if len(np.unique(yy)) < 2:
            continue
        boots[k] = float(roc_auc_score(yy, ss))
        k += 1
    if k < max(50, n_boot // 10):
        return point, float("nan"), float("nan")
    boots = boots[:k]
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def random_derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 2:
        raise ValueError("derangement needs n>=2")
    # Rejection sampling is fine for n up to a few hundred.
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def unit_prototypes(zPH: np.ndarray, unit_ids: np.ndarray, n_units: int) -> np.ndarray:
    """Mean z_pMHC per unique unit (deterministic prototype)."""
    d = zPH.shape[1]
    proto = np.zeros((n_units, d), dtype=np.float64)
    counts = np.zeros(n_units, dtype=np.int64)
    for i, uid in enumerate(unit_ids):
        proto[uid] += zPH[i]
        counts[uid] += 1
    if np.any(counts == 0):
        raise RuntimeError("empty pMHC unit")
    proto /= counts[:, None]
    return proto


def permute_pmhc_aurocs(
    zT: np.ndarray,
    zPH: np.ndarray,
    labels: np.ndarray,
    unit_ids: np.ndarray,
    n_units: int,
    n_perm: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    scores0 = score_components(zT, zPH)["full_score"]
    orig = safe_auroc(labels, scores0)
    proto = unit_prototypes(zPH, unit_ids, n_units)
    perm_aucs = np.empty(n_perm, dtype=np.float64)
    for k in range(n_perm):
        der = random_derangement(n_units, rng)
        zPH_perm = proto[der[unit_ids]]
        sc = score_components(zT, zPH_perm)["full_score"]
        perm_aucs[k] = safe_auroc(labels, sc)
    return {
        "original_auroc": float(orig),
        "permuted_auroc_mean": float(np.mean(perm_aucs)),
        "permuted_auroc_std": float(np.std(perm_aucs, ddof=1)) if n_perm > 1 else 0.0,
        "permuted_auroc_p2.5": float(np.quantile(perm_aucs, 0.025)),
        "permuted_auroc_p97.5": float(np.quantile(perm_aucs, 0.975)),
        "original_minus_permuted_mean": float(orig - np.mean(perm_aucs)),
        "n_unique_pmhc_units": int(n_units),
        "n_permutations": int(n_perm),
    }


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def run_score_decomposition(hla_lookups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_key in MODELS:
        for seed in SEEDS:
            for split in SPLITS:
                data = load_latents(model_key, seed, split, hla_lookups[split])
                comps = score_components(data["zT"], data["zPH"])
                y, pep = data["label"], data["peptide"]
                for score_name, scores in comps.items():
                    row = {
                        "model": model_key,
                        "model_label": MODELS[model_key]["label"],
                        "seed": seed,
                        "split": split,
                        "score_type": score_name,
                        "n": int(len(y)),
                        "global_auroc": safe_auroc(y, scores),
                        "peptide_weighted_auroc": peptide_weighted_auroc(y, scores, pep),
                        "peptide_macro_auroc": peptide_macro_auroc(y, scores, pep),
                        "peptide_macro_mcclish_pauc0.1": peptide_macro_mcclish(y, scores, pep),
                        "global_mcclish_pauc0.1": safe_mcclish_pauc01(y, scores),
                    }
                    rows.append(row)
                print(f"[decomp] {model_key} seed={seed} {split} full_global={rows[-5]['global_auroc']:.4f}", flush=True)
    return pd.DataFrame(rows)


def aggregate_decomposition(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean±std across seeds for the metrics requested in the brief."""
    metric_cols = [
        "global_auroc",
        "peptide_weighted_auroc",
        "peptide_macro_auroc",
        "peptide_macro_mcclish_pauc0.1",
        "global_mcclish_pauc0.1",
    ]
    rows = []
    for (model, split, score_type), g in raw.groupby(["model", "split", "score_type"]):
        row = {
            "model": model,
            "model_label": g["model_label"].iloc[0],
            "split": split,
            "score_type": score_type,
            "n_seeds": int(g["seed"].nunique()),
            "n": int(g["n"].iloc[0]),
        }
        for m in metric_cols:
            row[f"{m}_mean"] = float(g[m].mean())
            row[f"{m}_std"] = float(g[m].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run_per_peptide_immrep(hla_lookup: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Per seed diagnostics + seed-aggregated table for forest plot."""
    per_seed_rows = []
    # Collect scores for shared-index bootstrap across seeds
    # peptide -> model -> list of (y, s) aligned by pair_id
    store: Dict[str, Dict[str, Dict[int, pd.DataFrame]]] = {}

    for model_key in MODELS:
        for seed in SEEDS:
            data = load_latents(model_key, seed, "immrep_test", hla_lookup)
            scores = score_components(data["zT"], data["zPH"])["full_score"]
            df = pd.DataFrame({
                "pair_id": data["pair_id"],
                "peptide": data["peptide"],
                "y": data["label"],
                "score": scores,
            })
            rng = np.random.default_rng(seed * 1009 + 17)
            for pep, g in df.groupby("peptide", sort=True):
                y = g["y"].to_numpy()
                s = g["score"].to_numpy()
                auroc, lo, hi = bootstrap_auroc_ci(y, s, N_BOOT, rng)
                pos = s[y == 1]
                neg = s[y == 0]
                med_pos = float(np.median(pos)) if len(pos) else float("nan")
                med_neg = float(np.median(neg)) if len(neg) else float("nan")
                per_seed_rows.append({
                    "model": model_key,
                    "model_label": MODELS[model_key]["label"],
                    "seed": seed,
                    "peptide": str(pep),
                    "n_positive": int((y == 1).sum()),
                    "n_negative": int((y == 0).sum()),
                    "auroc": auroc,
                    "auroc_ci_low": lo,
                    "auroc_ci_high": hi,
                    "mcclish_pauc0.1": safe_mcclish_pauc01(y, s),
                    "median_positive_score": med_pos,
                    "median_negative_score": med_neg,
                    "median_score_difference": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else float("nan"),
                    "n_bootstrap": N_BOOT,
                })
                store.setdefault(str(pep), {}).setdefault(model_key, {})[seed] = g[["pair_id", "y", "score"]].copy()
            print(f"[perpep] {model_key} seed={seed}", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)

    # Aggregated forest rows: mean AUROC across seeds; shared-index bootstrap of mean AUROC
    agg_rows = []
    for pep in sorted(store.keys()):
        for model_key in MODELS:
            by_seed = store[pep][model_key]
            # Align on pair_id intersection (should be identical)
            base = by_seed[SEEDS[0]].sort_values("pair_id").reset_index(drop=True)
            aligned = []
            ok = True
            for seed in SEEDS:
                g = by_seed[seed].sort_values("pair_id").reset_index(drop=True)
                if not np.array_equal(g["pair_id"].to_numpy(), base["pair_id"].to_numpy()):
                    ok = False
                if not np.array_equal(g["y"].to_numpy(), base["y"].to_numpy()):
                    ok = False
                aligned.append(g)
            if not ok:
                raise RuntimeError(f"IMMREP pair alignment failed for {model_key} peptide={pep}")

            seed_aucs = [safe_auroc(g["y"], g["score"]) for g in aligned]
            y = base["y"].to_numpy().astype(int)
            n = len(y)
            rng = np.random.default_rng(
                zlib.crc32(f"{pep}|{model_key}".encode("utf-8")) & 0x7FFFFFFF
            )
            boots = []
            tries = 0
            while len(boots) < N_BOOT and tries < N_BOOT * 20:
                tries += 1
                idx = rng.integers(0, n, size=n)
                if len(np.unique(y[idx])) < 2:
                    continue
                aucs = [float(roc_auc_score(y[idx], g["score"].to_numpy()[idx])) for g in aligned]
                boots.append(float(np.mean(aucs)))
            ci_lo = float(np.quantile(boots, 0.025)) if boots else float("nan")
            ci_hi = float(np.quantile(boots, 0.975)) if boots else float("nan")

            pos_meds = []
            neg_meds = []
            paucs = []
            for g in aligned:
                yy = g["y"].to_numpy()
                ss = g["score"].to_numpy()
                pos_meds.append(float(np.median(ss[yy == 1])))
                neg_meds.append(float(np.median(ss[yy == 0])))
                paucs.append(safe_mcclish_pauc01(yy, ss))

            agg_rows.append({
                "model": model_key,
                "model_label": MODELS[model_key]["label"],
                "peptide": pep,
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "auroc_mean": float(np.mean(seed_aucs)),
                "auroc_std": float(np.std(seed_aucs, ddof=1)),
                "auroc_ci_low": ci_lo,
                "auroc_ci_high": ci_hi,
                "mcclish_pauc0.1_mean": float(np.mean(paucs)),
                "mcclish_pauc0.1_std": float(np.std(paucs, ddof=1)),
                "median_positive_score_mean": float(np.mean(pos_meds)),
                "median_negative_score_mean": float(np.mean(neg_meds)),
                "median_score_difference_mean": float(np.mean(pos_meds) - np.mean(neg_meds)),
                "n_bootstrap": N_BOOT,
                "n_seeds": len(SEEDS),
            })
    return per_seed, pd.DataFrame(agg_rows)


def run_pmhc_permutation(hla_lookups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_key in MODELS:
        for seed in SEEDS:
            for split in SPLITS:
                data = load_latents(model_key, seed, split, hla_lookups[split])
                # Encode unique (peptide, HLA) units
                keys = list(zip(data["peptide"].tolist(), data["hla"].tolist()))
                uniq = {k: i for i, k in enumerate(sorted(set(keys)))}
                unit_ids = np.array([uniq[k] for k in keys], dtype=np.int32)
                n_units = len(uniq)
                model_salt = zlib.crc32(model_key.encode("utf-8")) % 997
                rng = np.random.default_rng(
                    PERM_SEED + seed * 10007 + (0 if split == "test" else 1) + model_salt
                )
                stats = permute_pmhc_aurocs(
                    data["zT"], data["zPH"], data["label"], unit_ids, n_units, N_PERM, rng
                )
                rows.append({
                    "model": model_key,
                    "model_label": MODELS[model_key]["label"],
                    "seed": seed,
                    "split": split,
                    "diagnostic": "inference_time_pmhc_unit_derangement",
                    "note": (
                        "Inference-time dependence diagnostic only: TCR embeddings and labels "
                        "fixed; unique (peptide,HLA) unit embeddings deranged. Not a valid "
                        "biological evaluation of the permuted pairs."
                    ),
                    **stats,
                })
                print(
                    f"[perm] {model_key} seed={seed} {split} "
                    f"orig={stats['original_auroc']:.4f} perm_mean={stats['permuted_auroc_mean']:.4f} "
                    f"delta={stats['original_minus_permuted_mean']:.4f} units={n_units}",
                    flush=True,
                )
    return pd.DataFrame(rows)


def plot_forest(agg: pd.DataFrame, out_pdf: Path) -> None:
    models = list(MODELS.keys())
    peptides = sorted(agg["peptide"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), sharey=True)
    for ax, model_key in zip(axes, models):
        sub = agg[agg["model"] == model_key].set_index("peptide").loc[peptides]
        y = np.arange(len(peptides))
        x = sub["auroc_mean"].to_numpy()
        lo = sub["auroc_ci_low"].to_numpy()
        hi = sub["auroc_ci_high"].to_numpy()
        ax.axvline(0.5, color="0.6", lw=0.8, ls="--")
        ax.errorbar(
            x, y,
            xerr=[x - lo, hi - x],
            fmt="o",
            color="#1f4e79",
            ecolor="#1f4e79",
            elinewidth=1.2,
            capsize=2,
            markersize=4,
        )
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("AUROC")
        ax.set_title(MODELS[model_key]["label"])
        ax.set_yticks(y)
        ax.set_yticklabels(peptides if model_key == models[0] else [])
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_ylabel("Peptide")
    fig.suptitle(
        "IMMREP per-peptide AUROC (mean over seeds; 95% bootstrap CI of seed-mean AUROC)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir: Path, fig_dir: Path) -> None:
    text = f"""# IMMREP failure analysis

Diagnostics computed from **saved VICReg latent embeddings only**
(`{{split}}_latents.npz`). No retraining; checkpoint selection unchanged.

## Models and seeds
- onehot_vicreg (`onehot_vicreg_complete`)
- raw_esmc_vicreg (`esm_vicreg_raw_complete`)
- finetuned_esmc_vicreg / LoRA (`esm_vicreg_finetuned_complete`)
- Seeds: 31, 37, 43
- Splits: internal `test` (workshop complete-chain rows) and `immrep_test`

## Score definitions
For embeddings \\(z_T, z_{{PH}}\\in\\mathbb{{R}}^d\\):
- `full_score` = \\(-\\mathrm{{mean}}((z_T-z_{{PH}})^2)\\) (training score)
- `tcr_norm_score` = \\(-\\mathrm{{mean}}(z_T^2)\\)
- `dot_product_score` = \\(\\mathrm{{mean}}(z_T\\odot z_{{PH}})\\)
- `cosine_score` = standard cosine similarity
- `normalised_mse_score` = full_score after independently unit-normalising \\(z_T\\) and \\(z_{{PH}}\\)

## Outputs
| File | Contents |
|--|--|
| `immrep_score_decomposition.csv` | Per model×seed×split×score_type metrics |
| `immrep_score_decomposition_agg.csv` | Mean±std across seeds |
| `immrep_per_peptide_diagnostics.csv` | Per model×seed×peptide IMMREP diagnostics |
| `immrep_per_peptide_diagnostics_agg.csv` | Seed-aggregated peptide table (forest source) |
| `pmhc_permutation_diagnostics.csv` | Inference-time pMHC-unit derangement AUROCs |
| `immrep_per_peptide_forestplot.pdf` | Forest plot (also `.png`) |
| `README_immrep_failure_analysis.md` | This file |

Figure copy: `{fig_dir}/immrep_per_peptide_forestplot.pdf`

## Inference-time pMHC permutation (important)
This is an **inference-time dependence diagnostic**, not a valid biological
evaluation of the permuted pairs. Procedure:
1. Define a pMHC unit by exact `(peptide, HLA_sequence)`.
2. Build one prototype embedding per unit (mean of saved \\(z_{{PH}}\\) over rows of that unit).
3. Draw 100 derangements of the unique units (no fixed points).
4. Remap every row's pMHC embedding to the prototype of the deranged unit.
5. Keep TCR embeddings and original labels fixed; recompute `full_score` AUROC.

HLA sequences are joined from:
- test: `data/test/test_multiview.csv`
- IMMREP: `data/immrep_test/immrep_test_multiview.csv`

## Metrics notes
- Internal test reporting focus: global AUROC and peptide-weighted AUROC.
- IMMREP reporting focus: peptide-macro AUROC and McClish pAUC@FPR≤0.1
  (`sklearn.roc_auc_score(..., max_fpr=0.1)`).
- Bootstrap: 2000 resamples; 95% CI = 2.5/97.5 percentiles.
"""
    (out_dir / "README_immrep_failure_analysis.md").write_text(text)


def main() -> None:
    global N_BOOT, N_PERM, SEEDS

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--fig-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = parser.parse_args()

    N_BOOT = args.n_boot
    N_PERM = args.n_perm
    SEEDS = list(args.seeds)

    out_dir = args.out_dir
    fig_dir = args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("Loading HLA lookups...", flush=True)
    hla_lookups = {split: load_hla_lookup(split) for split in SPLITS}

    print("=== 1) Score decomposition ===", flush=True)
    decomp = run_score_decomposition(hla_lookups)
    decomp_path = out_dir / "immrep_score_decomposition.csv"
    decomp.to_csv(decomp_path, index=False)
    agg = aggregate_decomposition(decomp)
    agg.to_csv(out_dir / "immrep_score_decomposition_agg.csv", index=False)
    print(f"wrote {decomp_path}", flush=True)

    print("=== 2) Per-peptide IMMREP ===", flush=True)
    per_seed, per_agg = run_per_peptide_immrep(hla_lookups["immrep_test"])
    pep_path = out_dir / "immrep_per_peptide_diagnostics.csv"
    per_seed.to_csv(pep_path, index=False)
    per_agg.to_csv(out_dir / "immrep_per_peptide_diagnostics_agg.csv", index=False)
    forest_pdf = out_dir / "immrep_per_peptide_forestplot.pdf"
    plot_forest(per_agg, forest_pdf)
    # also copy figure to figures tree
    plot_forest(per_agg, fig_dir / "immrep_per_peptide_forestplot.pdf")
    print(f"wrote {pep_path} and {forest_pdf}", flush=True)

    print("=== 3) Inference-time pMHC permutation ===", flush=True)
    perm = run_pmhc_permutation(hla_lookups)
    perm_path = out_dir / "pmhc_permutation_diagnostics.csv"
    perm.to_csv(perm_path, index=False)
    print(f"wrote {perm_path}", flush=True)

    write_readme(out_dir, fig_dir)

    # Brief console summary for full_score
    print("\n=== SUMMARY full_score (mean±std over seeds) ===", flush=True)
    for split in SPLITS:
        print(f"-- {split} --")
        sub = agg[(agg.split == split) & (agg.score_type == "full_score")]
        for _, r in sub.iterrows():
            if split == "test":
                print(
                    f"  {r.model_label}: global {r.global_auroc_mean:.3f}±{r.global_auroc_std:.3f} | "
                    f"pep-w {r.peptide_weighted_auroc_mean:.3f}±{r.peptide_weighted_auroc_std:.3f}"
                )
            else:
                print(
                    f"  {r.model_label}: pep-macro {r.peptide_macro_auroc_mean:.3f}±{r.peptide_macro_auroc_std:.3f} | "
                    f"McClish pAUC0.1 {r['peptide_macro_mcclish_pauc0.1_mean']:.3f}±{r['peptide_macro_mcclish_pauc0.1_std']:.3f}"
                )

    print("\n=== SUMMARY permutation delta (orig - perm_mean), mean over seeds ===", flush=True)
    for split in SPLITS:
        print(f"-- {split} --")
        for model_key in MODELS:
            g = perm[(perm.model == model_key) & (perm.split == split)]
            print(
                f"  {MODELS[model_key]['label']}: "
                f"orig {g.original_auroc.mean():.3f} | perm {g.permuted_auroc_mean.mean():.3f} | "
                f"delta {g.original_minus_permuted_mean.mean():.3f}"
            )

    manifest = {
        "out_dir": str(out_dir),
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "models": list(MODELS.keys()),
        "seeds": SEEDS,
        "splits": SPLITS,
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Outputs in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
