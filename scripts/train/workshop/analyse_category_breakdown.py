#!/usr/bin/env python3
"""Novelty-category AUROC breakdown for workshop VICReg runs.

Reconstructs unseen_HLA / completely_unseen / unseen_TCR / unseen_peptide /
both_seen labels from train positives, joins saved prediction CSVs, and
aggregates global / peptide-weighted AUROC over seeds.

Outputs:
  models/outputs/workshop/paper_analysis/category_breakdown/
    category_coverage.csv
    category_metrics_by_seed.csv
    category_metrics_agg.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path("/home/natasha/multimodal_model")

DEFAULT_MODELS = {
    "onehot_composition": {
        "root": REPO / "models/outputs/workshop/onehot_vicreg_complete",
        "score_col": "onehot_composition_score",
    },
    "pretrained_esmc_meanpool": {
        "root": REPO / "models/outputs/workshop/esm_vicreg_raw_complete",
        "score_col": "pretrained_esmc_meanpool_score",
    },
    "finetuned_esmc_meanpool": {
        "root": REPO / "models/outputs/workshop/esm_vicreg_finetuned_complete",
        "score_col": "finetuned_esmc_meanpool_score",
    },
    "onehot_vicreg": {
        "root": REPO / "models/outputs/workshop/onehot_vicreg_complete",
        "score_col": "onehot_vicreg_score",
    },
    "esm_vicreg_raw": {
        "root": REPO / "models/outputs/workshop/esm_vicreg_raw_complete",
        "score_col": "esm_vicreg_score",
    },
    "esm_vicreg_finetuned": {
        "root": REPO / "models/outputs/workshop/esm_vicreg_finetuned_complete",
        "score_col": "esm_vicreg_score",
    },
}

CATS = ["unseen_HLA", "completely_unseen", "unseen_TCR", "unseen_peptide", "both_seen"]


def clean_seq(s) -> str:
    if pd.isna(s):
        return ""
    return "".join(ch for ch in str(s).strip().upper() if ch.isalpha())


def complete_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["tcra_len"] > 0)
        & (df["tcrb_len"] > 0)
        & (df["pep_len"] > 0)
        & (df["hla_len"] > 0)
    )


def safe_auroc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def safe_auprc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def pep_metrics(y, s, peptides, min_pos: int = 1, min_neg: int = 1) -> Tuple[Dict, pd.DataFrame]:
    df = pd.DataFrame(
        {
            "y": np.asarray(y).astype(int),
            "s": np.asarray(s).astype(float),
            "p": np.asarray(peptides).astype(str),
        }
    )
    rows = []
    for pep, g in df.groupby("p"):
        npos = int(g.y.sum())
        nneg = int((g.y == 0).sum())
        valid = npos >= min_pos and nneg >= min_neg and len(np.unique(g.y)) == 2
        rows.append(
            {
                "peptide": pep,
                "n": len(g),
                "n_pos": npos,
                "n_neg": nneg,
                "auroc": float(roc_auc_score(g.y, g.s)) if valid else np.nan,
                "valid": valid,
            }
        )
    tab = pd.DataFrame(rows)
    valid = tab[tab.valid]
    if len(valid) == 0:
        pw = float("nan")
        macro = float("nan")
    else:
        pw = float(np.average(valid.auroc, weights=valid.n))
        macro = float(valid.auroc.mean())
    return {
        "peptide_weighted_auroc": pw,
        "peptide_macro_auroc": macro,
        "n_peptides": int(len(tab)),
        "n_peptides_valid": int(len(valid)),
        "n_peptides_n_le_2": int((tab.n <= 2).sum()),
        "frac_pairs_in_n_le_2_peps": float(
            tab.loc[tab.n <= 2, "n"].sum() / max(1, int(tab.n.sum()))
        ),
    }, tab


def assign_category(tcr: str, pep: str, hla: str, train_tcr, train_pep, train_hla) -> str:
    ts, ps, hs = tcr in train_tcr, pep in train_pep, hla in train_hla
    if not hs:
        return "unseen_HLA"
    if (not ts) and (not ps):
        return "completely_unseen"
    if (not ts) and ps:
        return "unseen_TCR"
    if ts and (not ps):
        return "unseen_peptide"
    return "both_seen"


def label_split(df: pd.DataFrame, train_tcr, train_pep, train_hla) -> pd.DataFrame:
    out = df.copy()
    by_id = out.set_index("pair_id", drop=False)
    cats = []
    for _, r in out.iterrows():
        if int(r.binding_flag) == 1:
            src = r
        else:
            sid = r.source_pair_id
            if pd.notna(sid) and sid in by_id.index:
                src = by_id.loc[sid]
                if isinstance(src, pd.DataFrame):
                    src = src.iloc[0]
            else:
                src = r
        cats.append(
            assign_category(
                clean_seq(src.TCR_full),
                clean_seq(src.Peptide),
                clean_seq(src.HLA_sequence),
                train_tcr,
                train_pep,
                train_hla,
            )
        )
    out["category"] = cats
    out["peptide_norm"] = out.Peptide.map(clean_seq)
    return out


def discover_seeds(root: Path, splits: Sequence[str], requested: Sequence[int]) -> List[int]:
    found = []
    for seed in requested:
        ok = all((root / f"seed_{seed}" / f"{split}_predictions.csv").exists() for split in splits)
        if ok:
            found.append(int(seed))
        else:
            missing = [
                split
                for split in splits
                if not (root / f"seed_{seed}" / f"{split}_predictions.csv").exists()
            ]
            print(f"SKIP {root.name} seed_{seed}: missing {missing}", flush=True)
    return found


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-csv", default=str(REPO / "data/train/train_multiview.csv"))
    p.add_argument(
        "--val-csv",
        default=str(REPO / "data/val/val_multiview.csv"),
        help="Val split used by the current VICReg runs (occurrence-matched).",
    )
    p.add_argument(
        "--test-csv",
        default=str(REPO / "data/test/test_multiview.csv"),
        help="Test split used by the current VICReg runs (occurrence-matched).",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO / "models/outputs/workshop/paper_analysis/category_breakdown"),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[31, 37, 43, 49, 55])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train_csv)
    val = pd.read_csv(args.val_csv)
    test = pd.read_csv(args.test_csv)
    train = train[complete_mask(train)].copy()
    val = val[complete_mask(val)].copy()
    test = test[complete_mask(test)].copy()

    train_tcr = set(train.TCR_full.map(clean_seq))
    train_pep = set(train.Peptide.map(clean_seq))
    train_hla = set(train.HLA_sequence.map(clean_seq))

    labeled = {
        "val": label_split(val, train_tcr, train_pep, train_hla),
        "test": label_split(test, train_tcr, train_pep, train_hla),
    }
    labeled = {k: v for k, v in labeled.items() if k in args.splits}

    models = {}
    for name, cfg in DEFAULT_MODELS.items():
        seeds = discover_seeds(cfg["root"], args.splits, args.seeds)
        models[name] = {**cfg, "seeds": seeds}
        print(f"{name}: seeds={seeds}", flush=True)

    rows = []
    for split_name, meta in labeled.items():
        for cat, g in meta.groupby("category"):
            rows.append(
                {
                    "kind": "coverage",
                    "split": split_name,
                    "category": cat,
                    "model": np.nan,
                    "seed": np.nan,
                    "n_pairs": len(g),
                    "n_pos": int((g.binding_flag == 1).sum()),
                    "n_neg": int((g.binding_flag == 0).sum()),
                    "n_peptides": int(g.peptide_norm.nunique()),
                    "global_auroc": np.nan,
                    "global_auprc": np.nan,
                    "pep_w_auroc": np.nan,
                    "pep_w_auroc_min2each": np.nan,
                    "n_pep_valid": np.nan,
                    "n_pep_n_le_2": np.nan,
                    "frac_pairs_n_le_2": np.nan,
                    "note": np.nan,
                }
            )

        for model_name, cfg in models.items():
            for seed in cfg["seeds"]:
                pred_path = cfg["root"] / f"seed_{seed}" / f"{split_name}_predictions.csv"
                pred = pd.read_csv(pred_path)
                merged = meta.merge(pred[["pair_id", cfg["score_col"]]], on="pair_id", how="inner")
                if len(merged) != len(meta):
                    print(
                        f"WARN {model_name} seed{seed} {split_name}: "
                        f"meta={len(meta)} merged={len(merged)} pred={len(pred)}",
                        flush=True,
                    )
                score = merged[cfg["score_col"]].to_numpy()
                y = merged.binding_flag.to_numpy().astype(int)
                pm, _ = pep_metrics(y, score, merged.peptide_norm, 1, 1)
                pm2, _ = pep_metrics(y, score, merged.peptide_norm, 2, 2)
                rows.append(
                    {
                        "kind": "metric",
                        "split": split_name,
                        "category": "ALL",
                        "model": model_name,
                        "seed": seed,
                        "n_pairs": len(merged),
                        "n_pos": int(y.sum()),
                        "n_neg": int((y == 0).sum()),
                        "n_peptides": int(merged.peptide_norm.nunique()),
                        "global_auroc": safe_auroc(y, score),
                        "global_auprc": safe_auprc(y, score),
                        "pep_w_auroc": pm["peptide_weighted_auroc"],
                        "pep_w_auroc_min2each": pm2["peptide_weighted_auroc"],
                        "n_pep_valid": pm["n_peptides_valid"],
                        "n_pep_n_le_2": pm["n_peptides_n_le_2"],
                        "frac_pairs_n_le_2": pm["frac_pairs_in_n_le_2_peps"],
                        "note": "",
                    }
                )
                for cat in CATS:
                    g = merged[merged.category == cat]
                    if len(g) == 0:
                        continue
                    ys = g.binding_flag.to_numpy().astype(int)
                    ss = g[cfg["score_col"]].to_numpy()
                    ps = g.peptide_norm
                    pm, _ = pep_metrics(ys, ss, ps, 1, 1)
                    pm2, _ = pep_metrics(ys, ss, ps, 2, 2)
                    note = []
                    if g.peptide_norm.nunique() < 5:
                        note.append("few_peptides")
                    if pm["frac_pairs_in_n_le_2_peps"] > 0.5:
                        note.append("majority_pairs_in_n<=2_peptides")
                    if (g.binding_flag == 1).sum() < 10 or (g.binding_flag == 0).sum() < 10:
                        note.append("low_class_counts")
                    if pm["n_peptides_valid"] <= 3:
                        note.append("pep_w_unstable")
                    rows.append(
                        {
                            "kind": "metric",
                            "split": split_name,
                            "category": cat,
                            "model": model_name,
                            "seed": seed,
                            "n_pairs": len(g),
                            "n_pos": int(ys.sum()),
                            "n_neg": int((ys == 0).sum()),
                            "n_peptides": int(g.peptide_norm.nunique()),
                            "global_auroc": safe_auroc(ys, ss),
                            "global_auprc": safe_auprc(ys, ss),
                            "pep_w_auroc": pm["peptide_weighted_auroc"],
                            "pep_w_auroc_min2each": pm2["peptide_weighted_auroc"],
                            "n_pep_valid": pm["n_peptides_valid"],
                            "n_pep_valid_min2each": pm2["n_peptides_valid"],
                            "n_pep_n_le_2": pm["n_peptides_n_le_2"],
                            "frac_pairs_n_le_2": pm["frac_pairs_in_n_le_2_peps"],
                            "note": ";".join(note),
                        }
                    )

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "category_metrics_by_seed.csv", index=False)

    metric = res[res.kind == "metric"].copy()
    agg_rows = []
    for (split, cat, model), g in metric.groupby(["split", "category", "model"]):
        def msd(col: str) -> Tuple[float, float]:
            v = pd.to_numeric(g[col], errors="coerce")
            mean = float(v.mean()) if v.notna().any() else float("nan")
            std = float(v.std(ddof=1)) if v.notna().sum() > 1 else 0.0
            return mean, std

        for col in ["global_auroc", "global_auprc", "pep_w_auroc", "pep_w_auroc_min2each"]:
            mean, std = msd(col)
            g0 = g.iloc[0]
            agg_rows.append(
                {
                    "split": split,
                    "category": cat,
                    "model": model,
                    "metric": col,
                    "mean": mean,
                    "std": std,
                    "n_seeds": int(g.seed.nunique()),
                    "seeds": ",".join(str(int(x)) for x in sorted(g.seed.dropna().unique())),
                    "n_pairs": int(g0.n_pairs),
                    "n_pos": int(g0.n_pos),
                    "n_neg": int(g0.n_neg),
                    "n_peptides": int(g0.n_peptides),
                    "n_pep_n_le_2": g0.n_pep_n_le_2,
                    "frac_pairs_n_le_2": g0.frac_pairs_n_le_2,
                    "note": g0.note,
                }
            )
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(out_dir / "category_metrics_agg.csv", index=False)

    cov = res[res.kind == "coverage"][
        ["split", "category", "n_pairs", "n_pos", "n_neg", "n_peptides"]
    ]
    cov.to_csv(out_dir / "category_coverage.csv", index=False)

    for split in args.splits:
        print(f"\n======== {split.upper()} global AUROC mean±std ========", flush=True)
        sub = agg[(agg.split == split) & (agg.metric == "global_auroc")]
        for cat in ["ALL"] + CATS:
            s = sub[sub.category == cat]
            if len(s) == 0:
                continue
            bits = [
                f"{cat:20s} n={int(s.iloc[0].n_pairs):4d} "
                f"pep={int(s.iloc[0].n_peptides):3d} seeds={s.iloc[0].n_seeds}"
            ]
            for _, r in s.iterrows():
                bits.append(f"{r.model}={r['mean']:.3f}±{r['std']:.3f}")
            print(" | ".join(bits), flush=True)
            if isinstance(s.iloc[0].note, str) and s.iloc[0].note:
                print("   NOTE:", s.iloc[0].note, flush=True)

    print(f"\nWrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
