#!/usr/bin/env python3
"""Peptide-balanced organisation AUROC + hardened multi-cognate retrieval.

1) Same-peptide recovery AUROC sensitivity analysis
   - Original (frequency-capped) protocol for reference.
   - Peptide-balanced protocol:
       * fixed number of within-peptide TCR pairs per peptide;
       * between pairs formed by sampling two peptides uniformly, then one TCR each.
     Pair indices are drawn once with --eval-seed (independent of training
     seeds) and reused for every representation, so deterministic baselines
     have one AUROC and VICReg seed spread reflects only the learned spaces.

2) Multi-cognate TCR → pMHC retrieval
   - Gallery items are unique (peptide, HLA) complexes.
   - Audit: max L2 difference of z_T across occurrences of the same TCR.
   - Metrics: Recall@10, mAP (MRR supplementary).
   - Random-ranking baseline per query.
   - 95% CI by bootstrapping TCRs (plus seed points).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path("/home/natasha/multimodal_model")
OUTPUTS = REPO / "models/outputs/workshop"
OUT_DIR = REPO / "models/outputs/workshop/paper_analysis/refined_geometry"
FIG_DIR = REPO / "models/figures/workshop/paper_analysis/refined_geometry"

MODEL_ORDER = [
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
    "onehot_vicreg",
    "raw_esmc_vicreg",
    "finetuned_esmc_vicreg",
]
MODEL_LABELS = {
    "onehot_composition": "One-hot",
    "pretrained_esmc_meanpool": "Raw ESMC",
    "finetuned_esmc_meanpool": "LoRA ESMC",
    "onehot_vicreg": "One-hot+VICReg",
    "raw_esmc_vicreg": "Raw ESMC+VICReg",
    "finetuned_esmc_vicreg": "LoRA ESMC+VICReg",
}
COLORS = {
    "onehot_composition": "#636363",
    "pretrained_esmc_meanpool": "#3182bd",
    "finetuned_esmc_meanpool": "#31a354",
    "onehot_vicreg": "#e6550d",
    "raw_esmc_vicreg": "#de2d26",
    "finetuned_esmc_vicreg": "#756bb1",
}
RUNS = [
    "onehot_vicreg_complete",
    "esm_vicreg_raw_complete",
    "esm_vicreg_finetuned_complete",
]
DETERMINISTIC_MODELS = {
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
}


def mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_seq(s) -> str:
    if pd.isna(s):
        return ""
    return "".join(ch for ch in str(s).strip().upper() if ch.isalpha())


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def attach_meta(split: str, pair_ids: Sequence[str]) -> pd.DataFrame:
    path = {
        "test": REPO / "data/test/test_multiview.csv",
        "val": REPO / "data/val/val_multiview.csv",
    }[split]
    meta = pd.read_csv(path, usecols=["pair_id", "TCR_full", "Peptide", "HLA_sequence", "binding_flag"])
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta = meta.set_index("pair_id").reindex(list(pair_ids)).reset_index()
    meta["tcr"] = meta["TCR_full"].map(clean_seq)
    meta["pep"] = meta["Peptide"].map(clean_seq)
    meta["hla"] = meta["HLA_sequence"].map(clean_seq)
    meta["pmhc"] = meta["pep"] + "|" + meta["hla"]
    return meta


def model_arrays_from_run(run_name: str, npz: Dict[str, np.ndarray]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    rn = run_name.lower()
    if "zT_vicreg" in npz and "zPH_vicreg" in npz:
        out["onehot_vicreg"] = (np.asarray(npz["zT_vicreg"], float), np.asarray(npz["zPH_vicreg"], float))
    if "T_composition" in npz and "PH_composition" in npz:
        out["onehot_composition"] = (np.asarray(npz["T_composition"], float), np.asarray(npz["PH_composition"], float))
    if "zT_esm_vicreg" in npz and "zPH_esm_vicreg" in npz:
        key = "raw_esmc_vicreg" if "raw" in rn else "finetuned_esmc_vicreg"
        out[key] = (np.asarray(npz["zT_esm_vicreg"], float), np.asarray(npz["zPH_esm_vicreg"], float))
    if "T_pretrained_meanpool" in npz and "PH_pretrained_meanpool" in npz:
        out["pretrained_esmc_meanpool"] = (
            np.asarray(npz["T_pretrained_meanpool"], float),
            np.asarray(npz["PH_pretrained_meanpool"], float),
        )
    if "T_finetuned_meanpool" in npz and "PH_finetuned_meanpool" in npz:
        out["finetuned_esmc_meanpool"] = (
            np.asarray(npz["T_finetuned_meanpool"], float),
            np.asarray(npz["PH_finetuned_meanpool"], float),
        )
    return out


def load_seed_split(split: str, seed: int) -> Tuple[pd.DataFrame, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    models: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    meta = None
    pair_ids = None
    for run_name in RUNS:
        p = OUTPUTS / run_name / f"seed_{seed}" / f"{split}_latents.npz"
        if not p.exists():
            print(f"MISSING {p}", flush=True)
            continue
        npz = dict(np.load(p, allow_pickle=True))
        if pair_ids is None:
            pair_ids = np.asarray(npz["pair_id"]).astype(str)
            labels = np.asarray(npz["label"]).astype(int)
            peptides = np.asarray(npz["peptide"]).astype(str)
            meta = attach_meta(split, pair_ids)
            meta["label"] = labels
            # Prefer latent peptide strings when present
            meta["peptide_latent"] = peptides
        else:
            pid = np.asarray(npz["pair_id"]).astype(str)
            if len(pid) != len(pair_ids) or not np.all(pid == pair_ids):
                raise RuntimeError(f"Row-order mismatch in {p}")
        models.update(model_arrays_from_run(run_name, npz))
    if meta is None:
        raise FileNotFoundError(f"No latents found for {split} seed {seed}")
    return meta, models


# ---------------------------------------------------------------------------
# Within / between AUROC protocols
# ---------------------------------------------------------------------------

def capped_positive_indices(
    meta: pd.DataFrame,
    min_group: int,
    max_per_pep: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (row indices into meta, peptide labels for those rows)."""
    rng = np.random.default_rng(seed)
    pos = meta[meta["label"].astype(int) == 1]
    idxs = []
    for pep, grp in pos.groupby("pep"):
        if len(grp) < min_group:
            continue
        ix = grp.index.to_numpy()
        if len(ix) > max_per_pep:
            ix = rng.choice(ix, size=max_per_pep, replace=False)
        idxs.extend(ix.tolist())
    idxs = np.array(sorted(idxs), dtype=int)
    return idxs, meta.loc[idxs, "pep"].to_numpy(str)


def sample_capped_pair_indices(
    peps: np.ndarray,
    seed: int,
    max_between: int = 20000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Original protocol: all within pairs on capped set; between matched in count.

    Returns integer index arrays of shape (n, 2) into the capped TCR set.
    """
    rng = np.random.default_rng(seed)
    within = []
    pep_to_idx = {p: np.where(peps == p)[0] for p in np.unique(peps)}
    for idx in pep_to_idx.values():
        if len(idx) < 2:
            continue
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                within.append((int(idx[a]), int(idx[b])))
    within = np.asarray(within, dtype=int).reshape(-1, 2)
    if len(within) == 0:
        empty = np.zeros((0, 2), dtype=int)
        return empty, empty

    target = min(max_between, len(within))
    between = []
    n = len(peps)
    attempts = 0
    while len(between) < target and attempts < target * 50:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        attempts += 1
        if i == j or peps[i] == peps[j]:
            continue
        between.append((i, j))
    return within, np.asarray(between, dtype=int).reshape(-1, 2)


def sample_peptide_balanced_pair_indices(
    peps: np.ndarray,
    seed: int,
    pairs_per_peptide: int,
    min_tcrs: int,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Fixed within pairs per peptide; between via uniform peptide→TCR sampling.

    Returns integer index arrays of shape (n, 2) into the positive TCR set.
    """
    rng = np.random.default_rng(seed)
    pep_to_idx = {p: np.where(peps == p)[0] for p in np.unique(peps)}
    eligible = [p for p, idx in pep_to_idx.items() if len(idx) >= min_tcrs]
    # Need C(n,2) >= pairs_per_peptide
    eligible = [p for p in eligible if len(pep_to_idx[p]) * (len(pep_to_idx[p]) - 1) // 2 >= pairs_per_peptide]
    within = []
    for p in eligible:
        idx = pep_to_idx[p]
        # enumerate all pair index combinations then sample
        pairs = [(a, b) for a in range(len(idx)) for b in range(a + 1, len(idx))]
        chosen = rng.choice(len(pairs), size=pairs_per_peptide, replace=False)
        for c in chosen:
            a, b = pairs[int(c)]
            within.append((int(idx[a]), int(idx[b])))
    within = np.asarray(within, dtype=int).reshape(-1, 2)

    between = []
    if len(eligible) < 2 or len(within) == 0:
        return within, np.zeros((0, 2), dtype=int), {"n_peptides_eligible": len(eligible)}

    target = len(within)
    attempts = 0
    while len(between) < target and attempts < target * 80:
        attempts += 1
        p1, p2 = rng.choice(eligible, size=2, replace=False)
        i = int(rng.choice(pep_to_idx[p1]))
        j = int(rng.choice(pep_to_idx[p2]))
        between.append((i, j))

    info = {
        "n_peptides_eligible": len(eligible),
        "pairs_per_peptide": pairs_per_peptide,
        "n_within": int(len(within)),
        "n_between": int(len(between)),
        "n_attempts_between": attempts,
    }
    return within, np.asarray(between, dtype=int).reshape(-1, 2), info


def pair_l2(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.asarray([], dtype=float)
    return np.linalg.norm(Z[pairs[:, 0]] - Z[pairs[:, 1]], axis=1)


def lift_pair_indices(subset_idx: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """Map (n, 2) indices from a subset array back to full-row indices."""
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=int)
    return np.column_stack([subset_idx[pairs[:, 0]], subset_idx[pairs[:, 1]]])


def auroc_from_pairs(within: np.ndarray, between: np.ndarray) -> float:
    if len(within) == 0 or len(between) == 0:
        return float("nan")
    labels = np.concatenate([np.ones(len(within)), np.zeros(len(between))])
    scores = -np.concatenate([within, between])  # smaller distance => higher score
    return safe_auroc(labels, scores)


def eval_pair_table(
    meta: pd.DataFrame,
    pairs: np.ndarray,
    same_peptide: int,
) -> pd.DataFrame:
    pair_id = meta["pair_id"].astype(str).to_numpy()
    pep = meta["pep"].astype(str).to_numpy()
    tcr = meta["tcr"].astype(str).to_numpy()
    rows = []
    for a, b in pairs:
        rows.append(
            {
                "pair_id_a": pair_id[a],
                "pair_id_b": pair_id[b],
                "tcr_a": tcr[a],
                "tcr_b": tcr[b],
                "pep_a": pep[a],
                "pep_b": pep[b],
                "same_peptide": int(same_peptide),
            }
        )
    return pd.DataFrame(rows)


def run_auroc_protocols(
    split: str,
    seeds: Sequence[int],
    min_group: int,
    max_per_pep: int,
    pairs_per_peptide: int,
    eval_seed: int,
    pair_out_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Score every representation on one fixed peptide-balanced pair set.

    Pair indices are sampled once with ``eval_seed`` from the split metadata.
    Deterministic input baselines are scored once; VICReg models are scored
    in each training-seed latent space on those same pairs.
    """
    meta0, _ = load_seed_split(split, seeds[0])
    meta0 = meta0.reset_index(drop=True)
    pair_ids0 = meta0["pair_id"].astype(str).to_numpy()

    pos_idx = np.where(meta0["label"].astype(int).to_numpy() == 1)[0]
    pos_peps = meta0.loc[pos_idx, "pep"].to_numpy(str)
    within_pos, between_pos, info = sample_peptide_balanced_pair_indices(
        pos_peps, eval_seed, pairs_per_peptide=pairs_per_peptide, min_tcrs=min_group
    )
    within_b = lift_pair_indices(pos_idx, within_pos)
    between_b = lift_pair_indices(pos_idx, between_pos)

    capped_idx, capped_peps = capped_positive_indices(meta0, min_group, max_per_pep, eval_seed)
    within_c_rel, between_c_rel = sample_capped_pair_indices(capped_peps, eval_seed)
    within_c = lift_pair_indices(capped_idx, within_c_rel)
    between_c = lift_pair_indices(capped_idx, between_c_rel)

    if pair_out_path is not None:
        pairs_df = pd.concat(
            [
                eval_pair_table(meta0, within_b, 1),
                eval_pair_table(meta0, between_b, 0),
            ],
            ignore_index=True,
        )
        pairs_df.insert(0, "eval_seed", eval_seed)
        pairs_df.to_csv(pair_out_path, index=False)
        print(f"Wrote {pair_out_path} ({len(pairs_df)} pairs)", flush=True)

    rows = []
    seen_deterministic: Dict[str, Dict[str, float]] = {}
    for seed in seeds:
        meta, models = load_seed_split(split, seed)
        meta = meta.reset_index(drop=True)
        pids = meta["pair_id"].astype(str).to_numpy()
        if len(pids) != len(pair_ids0) or not np.all(pids == pair_ids0):
            raise RuntimeError(f"pair_id order mismatch for {split} seed {seed}")

        for model, (zT, _zPH) in models.items():
            auc_c = auroc_from_pairs(pair_l2(zT, within_c), pair_l2(zT, between_c))
            auc_b = auroc_from_pairs(pair_l2(zT, within_b), pair_l2(zT, between_b))
            is_det = model in DETERMINISTIC_MODELS
            if is_det:
                prev = seen_deterministic.get(model)
                if prev is not None:
                    if not (np.isclose(prev["frequency_capped"], auc_c) and np.isclose(prev["peptide_balanced"], auc_b)):
                        raise RuntimeError(
                            f"Deterministic {model} AUROC changed across training seeds "
                            f"(seed {seed} vs first evaluation)"
                        )
                    continue
                seen_deterministic[model] = {"frequency_capped": auc_c, "peptide_balanced": auc_b}
                seed_out = eval_seed
            else:
                seed_out = seed

            rows.append(
                {
                    "split": split,
                    "seed": seed_out,
                    "eval_seed": eval_seed,
                    "training_seed": np.nan if is_det else seed,
                    "model_name": model,
                    "protocol": "frequency_capped",
                    "auroc": auc_c,
                    "n_within": int(len(within_c)),
                    "n_between": int(len(between_c)),
                    "n_peptides": int(len(np.unique(capped_peps))),
                    "n_tcrs": int(len(capped_idx)),
                }
            )
            rows.append(
                {
                    "split": split,
                    "seed": seed_out,
                    "eval_seed": eval_seed,
                    "training_seed": np.nan if is_det else seed,
                    "model_name": model,
                    "protocol": "peptide_balanced",
                    "auroc": auc_b,
                    "n_within": int(len(within_b)),
                    "n_between": int(len(between_b)),
                    "n_peptides": int(info.get("n_peptides_eligible", 0)),
                    "n_tcrs": int(len(pos_idx)),
                    "pairs_per_peptide": pairs_per_peptide,
                }
            )
    return pd.DataFrame(rows)


def plot_auroc_comparison(df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    protocols = [
        ("frequency_capped", "Frequency-capped (original)"),
        ("peptide_balanced", "Peptide-balanced (sensitivity)"),
    ]
    for ax, (protocol, title) in zip(axes, protocols):
        sub = df[df["protocol"] == protocol]
        models = [m for m in MODEL_ORDER if m in set(sub["model_name"])]
        x = np.arange(len(models))
        for i, model in enumerate(models):
            pts = sub.loc[sub["model_name"] == model, "auroc"].to_numpy(float)
            mean = float(np.mean(pts))
            ax.scatter(
                np.full(len(pts), i),
                pts,
                color=COLORS.get(model, "0.3"),
                s=34,
                alpha=0.85,
                zorder=3,
                edgecolors="white",
                linewidths=0.4,
            )
            ax.hlines(mean, i - 0.28, i + 0.28, colors=COLORS.get(model, "0.2"), linewidths=2.2)
        ax.axhline(0.5, color="0.55", linestyle="--", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(r"Same-peptide recovery AUROC")
        ax.set_ylim(0.45, 0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # annotate n peptides from peptide-balanced
    bal = df[df.protocol == "peptide_balanced"]
    if not bal.empty:
        n_pep = int(bal["n_peptides"].iloc[0])
        ppp = int(bal["pairs_per_peptide"].dropna().iloc[0]) if "pairs_per_peptide" in bal.columns else "?"
        fig.suptitle(
            f"{split}: $P(d_\\mathrm{{within}}<d_\\mathrm{{between}})$ — "
            f"balanced uses {ppp} within-pairs / peptide across {n_pep} peptides"
        )
    fig.tight_layout()
    p = fig_dir / f"{split}_same_peptide_auroc_capped_vs_balanced.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}", flush=True)


# ---------------------------------------------------------------------------
# z_T consistency audit + multi-cognate retrieval
# ---------------------------------------------------------------------------

def zt_consistency_audit(meta: pd.DataFrame, models: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> pd.DataFrame:
    pos = meta[meta["label"].astype(int) == 1]
    rows = []
    for model, (zT, _) in models.items():
        max_diff = 0.0
        n_multi = 0
        for tcr, grp in pos.groupby("tcr"):
            if not tcr or len(grp) < 2:
                continue
            n_multi += 1
            vecs = zT[grp.index.to_numpy()]
            # pairwise max L2
            for a in range(len(vecs)):
                for b in range(a + 1, len(vecs)):
                    max_diff = max(max_diff, float(np.linalg.norm(vecs[a] - vecs[b])))
        rows.append(
            {
                "model_name": model,
                "n_tcrs_with_multiple_rows": n_multi,
                "max_l2_within_tcr": max_diff,
                "approx_zero": bool(max_diff < 1e-5),
            }
        )
    return pd.DataFrame(rows)


def average_precision(relevant_ranks_1based: Sequence[int], n_relevant: int) -> float:
    if n_relevant <= 0 or len(relevant_ranks_1based) == 0:
        return 0.0
    ap = 0.0
    hit = 0
    for r in relevant_ranks_1based:
        hit += 1
        ap += hit / float(r)
    return ap / float(n_relevant)


def random_baseline_for_query(n_gallery: int, n_relevant: int, ks: Sequence[int], n_sims: int, rng: np.random.Generator) -> Dict[str, float]:
    """Monte-Carlo random ranking baseline for one query."""
    if n_relevant <= 0 or n_gallery <= 0:
        return {f"recall@{k}": 0.0 for k in ks} | {"mrr": 0.0, "map": 0.0}
    recalls = {k: [] for k in ks}
    mrrs = []
    maps = []
    for _ in range(n_sims):
        # random permutation ranks for relevant items ~ sample without replacement
        ranks = np.sort(rng.choice(n_gallery, size=n_relevant, replace=False)) + 1
        mrrs.append(1.0 / float(ranks[0]))
        maps.append(average_precision(ranks.tolist(), n_relevant))
        for k in ks:
            recalls[k].append(float(np.mean(ranks <= k)))
    out = {"mrr": float(np.mean(mrrs)), "map": float(np.mean(maps))}
    out.update({f"recall@{k}": float(np.mean(recalls[k])) for k in ks})
    return out


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> Tuple[float, float, float]:
    vals = np.asarray(values, float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(vals.mean())
    if len(vals) == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        boots.append(float(sample.mean()))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return mean, lo, hi


def multicognate_retrieval_hardened(
    split: str,
    seeds: Sequence[int],
    n_boot: int,
    random_sims: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    audit_rows = []
    per_tcr_rows = []
    summary_rows = []
    ks = [5, 10, 20]

    for seed in seeds:
        meta, models = load_seed_split(split, seed)
        meta = meta.reset_index(drop=True)
        audit = zt_consistency_audit(meta, models)
        audit["split"] = split
        audit["seed"] = seed
        audit_rows.append(audit)

        pos = meta[meta["label"].astype(int) == 1].copy()
        # Gallery: unique pep-HLA among positives
        pmhc_to_idx = {pmhc: grp.index.to_numpy() for pmhc, grp in pos.groupby("pmhc") if pmhc and not pmhc.startswith("|")}
        gallery_pmhcs = sorted(pmhc_to_idx.keys())
        # Multi-cognate TCRs by pep-HLA
        tcr_to_pmhc = pos.groupby("tcr")["pmhc"].apply(lambda s: set(x for x in s if x and not str(x).startswith("|")))
        multi = {t: set(p) for t, p in tcr_to_pmhc.items() if t and len(p) >= 2}

        for model, (zT, zPH) in models.items():
            # Gallery centroids in PH space
            gallery = np.asarray([zPH[pmhc_to_idx[p]].mean(axis=0) for p in gallery_pmhcs], float)
            n_gal = len(gallery_pmhcs)
            pmhc_to_gal = {p: j for j, p in enumerate(gallery_pmhcs)}

            query_metrics = []
            for tcr, cognates in multi.items():
                relevant = [p for p in cognates if p in pmhc_to_gal]
                if not relevant:
                    continue
                # Query: any single occurrence (they should be identical); use first for audit clarity
                q_rows = pos.index[pos["tcr"] == tcr].to_numpy()
                q = zT[q_rows[0]]
                # Optional sanity: if variation exists, still use mean but flag later via audit
                if len(q_rows) > 1:
                    q = zT[q_rows].mean(axis=0)

                dists = np.linalg.norm(gallery - q[None, :], axis=1)
                order = np.argsort(dists)
                ranks = {gallery_pmhcs[j]: r + 1 for r, j in enumerate(order)}
                rel_ranks = sorted(ranks[p] for p in relevant)

                rng_q = np.random.default_rng(seed + (abs(hash(tcr)) % 10_000))
                rnd = random_baseline_for_query(n_gal, len(relevant), ks, random_sims, rng_q)

                row = {
                    "split": split,
                    "seed": seed,
                    "model_name": model,
                    "tcr_id": str(abs(hash(tcr)) % 10**12),
                    "n_cognates": len(relevant),
                    "n_gallery": n_gal,
                    "best_rank": int(rel_ranks[0]),
                    "mrr": 1.0 / float(rel_ranks[0]),
                    "map": average_precision(rel_ranks, len(relevant)),
                    "random_mrr": rnd["mrr"],
                    "random_map": rnd["map"],
                }
                for k in ks:
                    row[f"recall@{k}"] = float(np.mean([1.0 if ranks[p] <= k else 0.0 for p in relevant]))
                    row[f"random_recall@{k}"] = rnd[f"recall@{k}"]
                per_tcr_rows.append(row)
                query_metrics.append(row)

            if not query_metrics:
                continue
            qdf = pd.DataFrame(query_metrics)
            summary = {
                "split": split,
                "seed": seed,
                "model_name": model,
                "n_multi_tcrs": len(multi),
                "n_queries": len(qdf),
                "n_gallery_pmhc": n_gal,
            }
            for metric in ["map", "mrr", "recall@5", "recall@10", "recall@20",
                           "random_map", "random_mrr", "random_recall@5", "random_recall@10", "random_recall@20"]:
                mean, lo, hi = bootstrap_ci(qdf[metric].to_numpy(float), n_boot=n_boot, seed=seed + 17)
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_ci_lo"] = lo
                summary[f"{metric}_ci_hi"] = hi
            summary_rows.append(summary)

    audit_df = pd.concat(audit_rows, ignore_index=True) if audit_rows else pd.DataFrame()
    per_tcr_df = pd.DataFrame(per_tcr_rows)
    summary_df = pd.DataFrame(summary_rows)
    return audit_df, per_tcr_df, summary_df


def plot_retrieval(summary_df: pd.DataFrame, split: str, fig_dir: Path) -> None:
    # Aggregate across seeds: mean of seed-level means; CI = mean of TCR-bootstrap CIs (approx)
    # Better: pool is hard; show seed points + average of per-seed TCR-bootstrap intervals
    models = [m for m in MODEL_ORDER if m in set(summary_df["model_name"])]
    x = np.arange(len(models))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for ax, metric, ylabel in [
        (axes[0], "recall@10", "Recall@10"),
        (axes[1], "map", "mAP"),
    ]:
        means, lo, hi = [], [], []
        rnd_means = []
        for m in models:
            g = summary_df[summary_df.model_name == m]
            means.append(float(g[f"{metric}_mean"].mean()))
            lo.append(float(g[f"{metric}_ci_lo"].mean()))
            hi.append(float(g[f"{metric}_ci_hi"].mean()))
            rnd_means.append(float(g[f"random_{metric}_mean"].mean()))
            # seed points
            pts = g[f"{metric}_mean"].to_numpy(float)
            jitter = (np.arange(len(pts)) - (len(pts) - 1) / 2.0) * 0.08
            ax.scatter(np.full(len(pts), models.index(m)) + jitter, pts, color="0.15", s=18, zorder=4)

        yerr = np.vstack([np.asarray(means) - np.asarray(lo), np.asarray(hi) - np.asarray(means)])
        colors = [COLORS.get(m, "0.4") for m in models]
        ax.bar(x, means, color=colors, alpha=0.85, edgecolor="white", yerr=yerr, capsize=3, zorder=2)
        ax.plot(x, rnd_means, color="0.45", linestyle="--", marker="x", linewidth=1.3, label="random ranking", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8, frameon=False, loc="upper left")

    n_q = int(summary_df["n_queries"].iloc[0])
    n_gal = int(summary_df["n_gallery_pmhc"].iloc[0])
    fig.suptitle(
        f"{split}: multi-cognate TCR → (peptide, HLA) retrieval\n"
        f"{n_q} TCRs with ≥2 cognates; gallery={n_gal} pMHCs; "
        f"error bars = mean 95% CI over TCR bootstrap (per seed)"
    )
    fig.tight_layout()
    p = fig_dir / f"{split}_multicognate_retrieval_hardened.png"
    fig.savefig(p, dpi=220, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {p}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43, 49, 55])
    p.add_argument("--min-group-size", type=int, default=5)
    p.add_argument("--max-tcrs-per-peptide", type=int, default=25)
    p.add_argument("--pairs-per-peptide", type=int, default=10)
    p.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Pair-sampling seed, independent of model-training seeds.",
    )
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--random-sims", type=int, default=200)
    p.add_argument(
        "--write-diagnostic-plots",
        action="store_true",
        help="Also write intermediate diagnostic PNGs (default: CSV tables only).",
    )
    p.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Recompute same-peptide AUROC tables only; leave retrieval CSVs unchanged.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = mkdir(OUT_DIR)
    fig_dir = mkdir(FIG_DIR)

    print("[1/2] Same-peptide AUROC: capped vs peptide-balanced (fixed eval pairs)", flush=True)
    auroc_df = run_auroc_protocols(
        args.split,
        args.seeds,
        min_group=args.min_group_size,
        max_per_pep=args.max_tcrs_per_peptide,
        pairs_per_peptide=args.pairs_per_peptide,
        eval_seed=args.eval_seed,
        pair_out_path=out_dir / f"{args.split}_peptide_balanced_eval_pairs.csv",
    )
    auroc_df.to_csv(out_dir / f"{args.split}_same_peptide_auroc_protocols_by_seed.csv", index=False)
    summary = (
        auroc_df.groupby(["protocol", "model_name"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.to_csv(out_dir / f"{args.split}_same_peptide_auroc_protocols_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    if args.write_diagnostic_plots:
        plot_auroc_comparison(auroc_df, args.split, fig_dir)

    if args.skip_retrieval:
        print("Skipping retrieval (--skip-retrieval).", flush=True)
        print(f"\nTables: {out_dir}", flush=True)
        print("Paper figure: python scripts/train/workshop/plot_geometry_multipanel.py", flush=True)
        return

    print("[2/2] z_T consistency audit + hardened multi-cognate retrieval", flush=True)
    audit_df, per_tcr_df, summary_df = multicognate_retrieval_hardened(
        args.split, args.seeds, n_boot=args.n_bootstrap, random_sims=args.random_sims
    )
    audit_df.to_csv(out_dir / f"{args.split}_zt_within_tcr_audit.csv", index=False)
    per_tcr_df.to_csv(out_dir / f"{args.split}_multicognate_retrieval_per_tcr_hardened.csv", index=False)
    summary_df.to_csv(out_dir / f"{args.split}_multicognate_retrieval_summary_hardened.csv", index=False)
    print("\nz_T audit (max L2 within TCR):", flush=True)
    print(
        audit_df.groupby("model_name")["max_l2_within_tcr"].agg(["max", "mean"]).round(8).to_string(),
        flush=True,
    )
    print("\nRetrieval summary (seed-level means):", flush=True)
    cols = [
        "model_name", "seed", "n_queries", "n_gallery_pmhc",
        "recall@10_mean", "recall@10_ci_lo", "recall@10_ci_hi",
        "map_mean", "map_ci_lo", "map_ci_hi",
        "random_recall@10_mean", "random_map_mean",
        "mrr_mean",
    ]
    print(summary_df[cols].sort_values(["model_name", "seed"]).to_string(index=False), flush=True)
    if args.write_diagnostic_plots:
        plot_retrieval(summary_df, args.split, fig_dir)

    print(f"\nTables: {out_dir}", flush=True)
    print("Paper figure: python scripts/train/workshop/plot_geometry_multipanel.py", flush=True)


if __name__ == "__main__":
    main()
