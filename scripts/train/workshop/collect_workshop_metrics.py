#!/usr/bin/env python3
"""Collect workshop experiment metrics into long and summary CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DETERMINISTIC_MODELS = {
    "onehot_composition",
    "pretrained_esmc_meanpool",
    "finetuned_esmc_meanpool",
}


def flatten_summary(summary_path: Path) -> List[Dict]:
    with open(summary_path, "r") as f:
        summary = json.load(f)

    cfg = summary.get("config", {})
    seed = summary.get("seed", cfg.get("seed"))
    family = summary.get("model_family", summary_path.parents[1].name if len(summary_path.parents) > 1 else "unknown")
    best_epoch = summary.get("best_epoch")
    best_selection_metric = summary.get("best_selection_metric")
    best_selection_value = summary.get("best_selection_value")

    rows = []
    for split, models in summary.get("metrics", {}).items():
        for model_name, metrics in models.items():
            row = {
                "summary_path": str(summary_path),
                "model_family": family,
                "model_name": model_name,
                "split": split,
                "seed": int(seed) if seed is not None else np.nan,
                "deterministic_baseline": model_name in DETERMINISTIC_MODELS,
                "best_epoch": best_epoch,
                "best_selection_metric": best_selection_metric,
                "best_selection_value": best_selection_value,
            }
            row.update(metrics)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        default="/home/natasha/multimodal_model/models/outputs/workshop",
        help="Root folder containing model/seed_x/summary.json files.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/natasha/multimodal_model/models/outputs/workshop/consolidated",
        help="Where consolidated CSVs should be written.",
    )
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_paths = sorted(outputs_root.glob("*/seed_*/summary.json"))
    if not summary_paths:
        raise FileNotFoundError(f"No summary.json files found under {outputs_root} using */seed_*/summary.json")

    rows = []
    for path in summary_paths:
        rows.extend(flatten_summary(path))

    long_df = pd.DataFrame(rows)
    long_path = out_dir / "workshop_metrics_long.csv"
    long_df.to_csv(long_path, index=False)

    metadata_cols = {
        "summary_path", "model_family", "model_name", "split", "seed", "deterministic_baseline",
        "best_epoch", "best_selection_metric", "best_selection_value",
    }
    metric_cols = [c for c in long_df.columns if c not in metadata_cols and pd.api.types.is_numeric_dtype(long_df[c])]

    grouped_rows = []
    for (model_family, model_name, split), grp in long_df.groupby(["model_family", "model_name", "split"], dropna=False):
        row = {
            "model_family": model_family,
            "model_name": model_name,
            "split": split,
            "deterministic_baseline": bool(grp["deterministic_baseline"].iloc[0]),
            "n_seed_rows": int(grp["seed"].nunique(dropna=True)),
            "seeds": ",".join(str(int(x)) for x in sorted(grp["seed"].dropna().unique())),
        }
        for col in metric_cols:
            vals = pd.to_numeric(grp[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
            row[f"{col}_std"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else 0.0
        grouped_rows.append(row)

    summary_df = pd.DataFrame(grouped_rows).sort_values(["split", "model_family", "model_name"])
    summary_path = out_dir / "workshop_metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # A compact table focused on the core paper metrics.
    core_metrics = [
        "global_auroc", "auprc", "global_auc0.1_mcclish",
        "peptide_macro_auroc", "peptide_weighted_auroc",
        "peptide_macro_auc0.1_mcclish", "peptide_weighted_auc0.1_mcclish",
        "n_examples", "n_peptides_total", "n_peptides_valid",
    ]
    available = []
    for metric in core_metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in summary_df.columns:
            available.extend([mean_col, std_col])
    compact_cols = ["model_family", "model_name", "split", "deterministic_baseline", "n_seed_rows", "seeds"] + available
    compact_df = summary_df[compact_cols].copy()
    compact_path = out_dir / "workshop_metrics_compact.csv"
    compact_df.to_csv(compact_path, index=False)

    print("Collected summaries:", len(summary_paths))
    print("Long metrics:", long_path)
    print("Summary metrics:", summary_path)
    print("Compact metrics:", compact_path)


if __name__ == "__main__":
    main()
