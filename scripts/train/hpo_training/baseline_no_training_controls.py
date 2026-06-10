#!/usr/bin/env python3
"""
No-training control baselines on precomputed shards.

Purpose
-------
This script evaluates whether the validation/test signal exists before training.

It compares:

1) Raw ESM mean-pooling baselines:
   a. TCR-peptide cosine
   b. TCR-HLA cosine
   c. TCR-pMHC average cosine = 0.5 * cos(TCR, peptide) + 0.5 * cos(TCR, HLA)

2) Random projection-head baseline:
   Same low-rank architecture as the trained model, but random initialisation only.
   No optimisation is performed.
   Repeated across multiple random seeds.

Outputs
-------
- JSON with full metrics.
- CSV summary table.
- Histogram plots for validation and test scores.
- Random-seed distribution summary.

Assumptions
-----------
- Shards are precomputed and stored under:
    embed_root/train
    embed_root/val
    embed_root/test

- Each shard item contains:
    emb_T, mask_T
    emb_P, mask_P
    emb_H, mask_H
    binding_flag
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DATASET
# ============================================================

class ShardedBatchTripletDataset(Dataset):
    def __init__(self, shards_dir: Path):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))

        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shards_dir}")

        self.index = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for j in range(len(shard)):
                self.index.append((sp, j))

        self._cache_path = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sp, j = self.index[idx]

        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp

        return self._cache_data[j]


# ============================================================
# MODEL ARCHITECTURE FOR RANDOM PROJECTION CONTROL
# ============================================================

class ESMProjectionHead(nn.Module):
    def __init__(
        self,
        D: int,
        rL: int,
        rD: int,
        d: int,
        L_max: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.D = D
        self.rL = rL
        self.rD = rD
        self.d = d
        self.L_max = L_max

        self.B_c = nn.Parameter(torch.empty(D, rD))
        self.A_c = nn.Parameter(torch.empty(L_max, rL))
        self.H_c = nn.Parameter(torch.empty(rL * rD, d))

        nn.init.xavier_uniform_(self.B_c)
        nn.init.xavier_uniform_(self.A_c)
        nn.init.xavier_uniform_(self.H_c)

        self.expander = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d, d),
        )

    def forward(self, emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        device = emb.device
        B = emb.shape[0]
        L_true = mask.sum(dim=1)

        z_list = []

        for b in range(B):
            Lb = int(L_true[b].item())

            if Lb == 0:
                z_list.append(torch.zeros(self.d, device=device))
                continue

            Xb = emb[b, :Lb, :]
            mb = mask[b, :Lb].unsqueeze(-1).float()
            Xb = Xb * mb

            Yb = Xb @ self.B_c
            A_pos = self.A_c[:Lb, :]
            Ub = A_pos.T @ Yb

            z_b = Ub.reshape(-1) @ self.H_c
            z_list.append(z_b)

        z = torch.stack(z_list, dim=0)
        return self.expander(z)


class PMHCProjectionHead(nn.Module):
    def __init__(
        self,
        D: int,
        rL: int,
        rD: int,
        d: int,
        L_P_max: int,
        L_H_max: int,
        R_PH: float = 0.7,
        dropout: float = 0.0,
    ):
        super().__init__()

        d_P = int(round(R_PH * d))
        d_H = d - d_P

        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid pMHC split: d_P={d_P}, d_H={d_H}")

        self.pep_encoder = ESMProjectionHead(
            D=D,
            rL=rL,
            rD=rD,
            d=d_P,
            L_max=L_P_max,
            dropout=dropout,
        )

        self.hla_encoder = ESMProjectionHead(
            D=D,
            rL=rL,
            rD=rD,
            d=d_H,
            L_max=L_H_max,
            dropout=dropout,
        )

    def forward(
        self,
        emb_P: torch.Tensor,
        mask_P: torch.Tensor,
        emb_H: torch.Tensor,
        mask_H: torch.Tensor,
    ) -> torch.Tensor:
        zP = self.pep_encoder(emb_P, mask_P)
        zH = self.hla_encoder(emb_H, mask_H)
        return torch.cat([zP, zH], dim=-1)


# ============================================================
# BASIC HELPERS
# ============================================================

def masked_mean(emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).float()
    summed = (emb * m).sum(dim=1)
    denom = m.sum(dim=1).clamp(min=1.0)
    return summed / denom


def get_labels(batch) -> np.ndarray:
    labels = batch["binding_flag"]
    labels = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.array(labels)
    return labels.astype(int)


def check_binary_labels(labels: np.ndarray, split_name: str):
    unique = np.unique(labels)
    if len(unique) != 2:
        raise ValueError(
            f"{split_name} labels must contain both classes for AUROC/AUPRC. "
            f"Found labels: {unique}"
        )


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Select threshold on validation scores by F1.

    Higher score = more likely binder.
    """
    check_binary_labels(labels, "threshold-selection split")

    best = None

    for thr in np.unique(scores):
        preds = (scores >= thr).astype(int)

        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)

        candidate = {
            "threshold": float(thr),
            "f1": float(f1),
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
        }

        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    return best


def metrics_from_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    peptides: np.ndarray,
    threshold: float,
    split_name: str,
) -> Dict:
    """
    Compute ranking metrics and thresholded classification metrics.

    Higher score = more likely binder.
    """
    check_binary_labels(labels, split_name)

    preds = (scores >= threshold).astype(int)

    out = {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "n": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "score_mean_pos": float(scores[labels == 1].mean()),
        "score_mean_neg": float(scores[labels == 0].mean()),
        "score_gap_pos_minus_neg": float(scores[labels == 1].mean() - scores[labels == 0].mean()),
        "score_std_pos": float(scores[labels == 1].std()),
        "score_std_neg": float(scores[labels == 0].std()),
    }
    pep = per_peptide_auroc(labels=labels, scores=scores, peptides=peptides)
    out.update(
        {
            "auroc_per_peptide_macro": pep["auroc_per_peptide_macro"],
            "auroc_per_peptide_weighted": pep["auroc_per_peptide_weighted"],
            "n_peptides_total": pep["n_peptides_total"],
            "n_peptides_valid_for_auroc": pep["n_peptides_valid_for_auroc"],
        }
    )
    return out


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    threshold: float | None = None,
    peptides: np.ndarray | None = None,
    n_peptides_hist: int = 10,
):
    labels = labels.astype(int)

    pos = scores[labels == 1]
    neg = scores[labels == 0]

    plt.figure(figsize=(7, 5))
    plt.hist(neg, bins=40, alpha=0.25, density=True, label="negative (all)", color="#1f77b4")
    plt.hist(pos, bins=40, alpha=0.25, density=True, label="positive (all)", color="#ff7f0e")

    if peptides is not None:
        pep_series = pd.Series(peptides.astype(str))
        top_peptides = pep_series.value_counts().head(n_peptides_hist).index.tolist()
        colors = [
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#d62728",
            "#bcbd22",
            "#e377c2",
            "#7f7f7f",
            "#1f77b4",
            "#ff7f0e",
        ]
        for i, pep in enumerate(top_peptides):
            mask_pep = pep_series.values == pep
            pos_pep = scores[mask_pep & (labels == 1)]
            neg_pep = scores[mask_pep & (labels == 0)]
            color = colors[i % len(colors)]
            if len(pos_pep):
                plt.hist(
                    pos_pep,
                    bins=35,
                    alpha=0.35,
                    density=True,
                    color=color,
                    label=f"{pep[:10]}... pos",
                )
            if len(neg_pep):
                plt.hist(
                    neg_pep,
                    bins=35,
                    alpha=0.70,
                    density=True,
                    color=color,
                    label=f"{pep[:10]}... neg",
                )

    if threshold is not None:
        plt.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=f"val threshold = {threshold:.4f}",
        )

    plt.xlabel("Score: higher = more likely binder")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def summarise_random_metric(
    random_results: List[Dict],
    split: str,
    metric: str,
) -> Dict[str, float]:
    vals = np.array([r[split][metric] for r in random_results], dtype=float)

    return {
        f"{split}_{metric}_mean": float(vals.mean()),
        f"{split}_{metric}_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        f"{split}_{metric}_min": float(vals.min()),
        f"{split}_{metric}_max": float(vals.max()),
        f"{split}_{metric}_p05": float(np.percentile(vals, 5)),
        f"{split}_{metric}_p50": float(np.percentile(vals, 50)),
        f"{split}_{metric}_p95": float(np.percentile(vals, 95)),
    }


def per_peptide_auroc(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray) -> Dict[str, float]:
    frame = pd.DataFrame(
        {"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)}
    )
    rows = []
    for pep, grp in frame.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) >= 2
        auc = float(roc_auc_score(y, s)) if valid else float("nan")
        rows.append(
            {
                "peptide": pep,
                "n": int(len(grp)),
                "n_pos": int(y.sum()),
                "n_neg": int((y == 0).sum()),
                "auroc": auc,
                "valid": bool(valid),
            }
        )
    table = pd.DataFrame(rows)
    valid_table = table[table["valid"]]
    if len(valid_table) == 0:
        macro = float("nan")
        weighted = float("nan")
    else:
        macro = float(valid_table["auroc"].mean())
        weighted = float(np.average(valid_table["auroc"], weights=valid_table["n"]))
    return {
        "auroc_per_peptide_macro": macro,
        "auroc_per_peptide_weighted": weighted,
        "n_peptides_total": int(len(table)),
        "n_peptides_valid_for_auroc": int(len(valid_table)),
        "table": table.sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True),
    }


# ============================================================
# RAW ESM BASELINES
# ============================================================

@torch.no_grad()
def evaluate_raw_meanpool_baselines(
    loader: DataLoader,
    device: torch.device,
    eps: float = 1e-8,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Raw ESM mean-pooling controls.

    These avoid the earlier concat/truncate flaw.

    Scores returned:
    - raw_TCR_peptide_cos
    - raw_TCR_HLA_cos
    - raw_TCR_pMHC_avg_cos
    """
    all_scores = {
        "raw_TCR_peptide_cos": [],
        "raw_TCR_HLA_cos": [],
        "raw_TCR_pMHC_avg_cos": [],
    }
    all_labels = []
    all_pair_ids = []

    for batch in loader:
        eT = masked_mean(batch["emb_T"].to(device), batch["mask_T"].to(device))
        eP = masked_mean(batch["emb_P"].to(device), batch["mask_P"].to(device))
        eH = masked_mean(batch["emb_H"].to(device), batch["mask_H"].to(device))

        eT = F.normalize(eT, dim=-1, eps=eps)
        eP = F.normalize(eP, dim=-1, eps=eps)
        eH = F.normalize(eH, dim=-1, eps=eps)

        cos_TP = (eT * eP).sum(dim=-1)
        cos_TH = (eT * eH).sum(dim=-1)
        cos_avg = 0.5 * cos_TP + 0.5 * cos_TH

        all_scores["raw_TCR_peptide_cos"].append(cos_TP.detach().cpu().numpy())
        all_scores["raw_TCR_HLA_cos"].append(cos_TH.detach().cpu().numpy())
        all_scores["raw_TCR_pMHC_avg_cos"].append(cos_avg.detach().cpu().numpy())

        all_labels.append(get_labels(batch))
        all_pair_ids.extend([str(x) for x in batch["pair_id"]])

    scores = {k: np.concatenate(v) for k, v in all_scores.items()}
    labels = np.concatenate(all_labels).astype(int)
    pair_ids = np.array(all_pair_ids, dtype=str)

    return scores, labels, pair_ids


# ============================================================
# RANDOM PROJECTION BASELINE
# ============================================================

@torch.no_grad()
def evaluate_random_projection(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random projection-head control.

    Higher score = more likely binder.
    Uses score = cosine(zT, zPH).
    This is equivalent to -H when H = -1 - cosine.
    """
    tcr_proj.eval()
    pmhc_proj.eval()

    all_scores = []
    all_labels = []
    all_pair_ids = []

    for batch in loader:
        zT = tcr_proj(
            batch["emb_T"].to(device),
            batch["mask_T"].to(device),
        )

        zPH = pmhc_proj(
            batch["emb_P"].to(device),
            batch["mask_P"].to(device),
            batch["emb_H"].to(device),
            batch["mask_H"].to(device),
        )

        eT = F.normalize(zT, dim=-1, eps=eps)
        ePH = F.normalize(zPH, dim=-1, eps=eps)

        cos = (eT * ePH).sum(dim=-1)

        all_scores.append(cos.detach().cpu().numpy())
        all_labels.append(get_labels(batch))
        all_pair_ids.extend([str(x) for x in batch["pair_id"]])

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels).astype(int)
    pair_ids = np.array(all_pair_ids, dtype=str)

    return scores, labels, pair_ids


def build_random_projection_models(
    sample,
    cfg,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    D = sample["emb_T"].shape[2]
    L_T_max = sample["emb_T"].shape[1]
    L_P_max = sample["emb_P"].shape[1]
    L_H_max = sample["emb_H"].shape[1]

    tcr_proj = ESMProjectionHead(
        D=D,
        rL=cfg.rL,
        rD=cfg.rD,
        d=cfg.d,
        L_max=L_T_max,
        dropout=0.0,
    ).to(device)

    pmhc_proj = PMHCProjectionHead(
        D=D,
        rL=cfg.rL,
        rD=cfg.rD,
        d=cfg.d,
        L_P_max=L_P_max,
        L_H_max=L_H_max,
        R_PH=cfg.r_ph,
        dropout=0.0,
    ).to(device)

    return tcr_proj, pmhc_proj


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_train_clean_immrep_A"

    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/no_training_baselines_immrepA"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/no_training_baselines_immrepA"
    run_tag: str = "immrepA_trainclean"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_df_clean_pos_neg_A.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_df_clean_pos_neg_A.csv"

    seed: int = 31
    n_random_seeds: int = 50

    batch_size: int = 1
    num_workers: int = 0

    rL: int = 8
    rD: int = 16
    d: int = 128
    r_ph: float = 0.7

    eps: float = 1e-8
    n_peptides_hist: int = 10


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--embed-root",
        type=str,
        default=Config.embed_root,
        help="Root containing train/val/test shard folders.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=Config.out_dir,
        help="Directory for JSON/CSV outputs.",
    )
    ap.add_argument(
        "--fig-dir",
        type=str,
        default=Config.fig_dir,
        help="Directory for histogram plots.",
    )
    ap.add_argument("--seed", type=int, default=Config.seed)
    ap.add_argument("--n-random-seeds", type=int, default=Config.n_random_seeds)
    ap.add_argument("--run-tag", type=str, default=Config.run_tag)
    ap.add_argument("--val-csv", type=str, default=Config.val_csv)
    ap.add_argument("--test-csv", type=str, default=Config.test_csv)

    ap.add_argument("--rL", type=int, default=Config.rL)
    ap.add_argument("--rD", type=int, default=Config.rD)
    ap.add_argument("--d", type=int, default=Config.d)
    ap.add_argument("--r-ph", type=float, default=Config.r_ph)
    ap.add_argument("--n-peptides-hist", type=int, default=Config.n_peptides_hist)

    args = ap.parse_args()

    cfg = Config(
        embed_root=args.embed_root,
        out_dir=args.out_dir,
        fig_dir=args.fig_dir,
        seed=args.seed,
        n_random_seeds=args.n_random_seeds,
        run_tag=args.run_tag,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        rL=args.rL,
        rD=args.rD,
        d=args.d,
        r_ph=args.r_ph,
        n_peptides_hist=args.n_peptides_hist,
    )

    if cfg.batch_size != 1:
        raise ValueError(
            "This script expects pre-batched shard items and requires batch_size=1."
        )

    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 100)
    print("NO-TRAINING BASELINE CONTROLS")
    print("=" * 100)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Embedding root: {cfg.embed_root}")
    print(f"Val CSV: {cfg.val_csv}")
    print(f"Test CSV: {cfg.test_csv}")
    print(f"Output directory: {out_dir}")
    print(f"Figure directory: {fig_dir}")
    print(f"Random projection seeds: {cfg.n_random_seeds}")
    print("=" * 100)

    train_ds = ShardedBatchTripletDataset(Path(cfg.embed_root) / "train")
    val_ds = ShardedBatchTripletDataset(Path(cfg.embed_root) / "val")
    test_ds = ShardedBatchTripletDataset(Path(cfg.embed_root) / "test")

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x[0],
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda x: x[0],
    )

    sample = train_ds[0]
    val_df = pd.read_csv(cfg.val_csv)
    test_df = pd.read_csv(cfg.test_csv)
    if "pair_id" not in val_df.columns or "Peptide" not in val_df.columns:
        raise ValueError(f"Validation CSV must contain pair_id and Peptide: {cfg.val_csv}")
    if "pair_id" not in test_df.columns or "Peptide" not in test_df.columns:
        raise ValueError(f"Test CSV must contain pair_id and Peptide: {cfg.test_csv}")
    val_pair_to_peptide = dict(zip(val_df["pair_id"].astype(str), val_df["Peptide"].astype(str)))
    test_pair_to_peptide = dict(zip(test_df["pair_id"].astype(str), test_df["Peptide"].astype(str)))

    full_results = {
        "config": asdict(cfg),
        "raw_esm_meanpool": {},
        "random_projection_no_training": {},
        "random_projection_summary": {},
    }

    summary_rows = []
    per_peptide_rows = []

    # ------------------------------------------------------------
    # 1. RAW ESM MEAN-POOL BASELINES
    # ------------------------------------------------------------
    print("\nEvaluating raw ESM mean-pooling baselines...")

    val_raw_scores, val_raw_labels, val_raw_pair_ids = evaluate_raw_meanpool_baselines(
        val_loader,
        device,
        eps=cfg.eps,
    )

    test_raw_scores, test_raw_labels, test_raw_pair_ids = evaluate_raw_meanpool_baselines(
        test_loader,
        device,
        eps=cfg.eps,
    )
    val_raw_peptides = np.array([val_pair_to_peptide.get(pid, "") for pid in val_raw_pair_ids], dtype=str)
    test_raw_peptides = np.array([test_pair_to_peptide.get(pid, "") for pid in test_raw_pair_ids], dtype=str)

    for baseline_name in val_raw_scores.keys():
        print(f"  Raw baseline: {baseline_name}")

        val_scores = val_raw_scores[baseline_name]
        test_scores = test_raw_scores[baseline_name]

        thr = best_threshold(val_scores, val_raw_labels)["threshold"]

        val_metrics = metrics_from_scores(
            val_scores,
            val_raw_labels,
            val_raw_peptides,
            threshold=thr,
            split_name=f"val/{baseline_name}",
        )

        test_metrics = metrics_from_scores(
            test_scores,
            test_raw_labels,
            test_raw_peptides,
            threshold=thr,
            split_name=f"test/{baseline_name}",
        )

        full_results["raw_esm_meanpool"][baseline_name] = {
            "val": val_metrics,
            "test": test_metrics,
        }
        val_pep_table = per_peptide_auroc(val_raw_labels, val_scores, val_raw_peptides)["table"]
        test_pep_table = per_peptide_auroc(test_raw_labels, test_scores, test_raw_peptides)["table"]
        for split_name, table in [("val", val_pep_table), ("test", test_pep_table)]:
            t = table.copy()
            t["baseline"] = baseline_name
            t["seed"] = np.nan
            t["split"] = split_name
            per_peptide_rows.extend(t.to_dict(orient="records"))

        plot_score_histogram(
            val_scores,
            val_raw_labels,
            title=f"{baseline_name} | validation",
            save_path=fig_dir / f"{baseline_name}__val_hist.png",
            threshold=thr,
            peptides=val_raw_peptides,
            n_peptides_hist=cfg.n_peptides_hist,
        )

        plot_score_histogram(
            test_scores,
            test_raw_labels,
            title=f"{baseline_name} | test",
            save_path=fig_dir / f"{baseline_name}__test_hist.png",
            threshold=thr,
            peptides=test_raw_peptides,
            n_peptides_hist=cfg.n_peptides_hist,
        )

        summary_rows.append({
            "baseline": baseline_name,
            "seed": None,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_auroc_per_peptide_macro": val_metrics["auroc_per_peptide_macro"],
            "val_score_gap_pos_minus_neg": val_metrics["score_gap_pos_minus_neg"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_auroc_per_peptide_macro": test_metrics["auroc_per_peptide_macro"],
            "test_score_gap_pos_minus_neg": test_metrics["score_gap_pos_minus_neg"],
            "threshold": thr,
        })

        print(
            f"    val AUROC={val_metrics['auroc']:.4f} | "
            f"test AUROC={test_metrics['auroc']:.4f} | "
            f"test AUPRC={test_metrics['auprc']:.4f}"
        )

    # ------------------------------------------------------------
    # 2. RANDOM PROJECTION BASELINE ACROSS MANY SEEDS
    # ------------------------------------------------------------
    print("\nEvaluating random projection-head baseline...")

    random_results = []
    best_random_by_val = None

    for i in range(cfg.n_random_seeds):
        seed = cfg.seed + i
        set_seed(seed)

        tcr_proj, pmhc_proj = build_random_projection_models(sample, cfg, device)

        val_scores, val_labels, val_pair_ids = evaluate_random_projection(
            val_loader,
            tcr_proj,
            pmhc_proj,
            device,
            eps=cfg.eps,
        )

        test_scores, test_labels, test_pair_ids = evaluate_random_projection(
            test_loader,
            tcr_proj,
            pmhc_proj,
            device,
            eps=cfg.eps,
        )
        val_peptides = np.array([val_pair_to_peptide.get(pid, "") for pid in val_pair_ids], dtype=str)
        test_peptides = np.array([test_pair_to_peptide.get(pid, "") for pid in test_pair_ids], dtype=str)

        thr = best_threshold(val_scores, val_labels)["threshold"]

        val_metrics = metrics_from_scores(
            val_scores,
            val_labels,
            val_peptides,
            threshold=thr,
            split_name=f"val/random_projection_seed{seed}",
        )

        test_metrics = metrics_from_scores(
            test_scores,
            test_labels,
            test_peptides,
            threshold=thr,
            split_name=f"test/random_projection_seed{seed}",
        )

        seed_result = {
            "seed": seed,
            "val": val_metrics,
            "test": test_metrics,
        }
        val_pep_table = per_peptide_auroc(val_labels, val_scores, val_peptides)["table"]
        test_pep_table = per_peptide_auroc(test_labels, test_scores, test_peptides)["table"]
        for split_name, table in [("val", val_pep_table), ("test", test_pep_table)]:
            t = table.copy()
            t["baseline"] = "random_projection_no_training"
            t["seed"] = int(seed)
            t["split"] = split_name
            per_peptide_rows.extend(t.to_dict(orient="records"))

        random_results.append(seed_result)

        summary_rows.append({
            "baseline": "random_projection_no_training",
            "seed": seed,
            "val_auroc": val_metrics["auroc"],
            "val_auprc": val_metrics["auprc"],
            "val_f1": val_metrics["f1"],
            "val_auroc_per_peptide_macro": val_metrics["auroc_per_peptide_macro"],
            "val_score_gap_pos_minus_neg": val_metrics["score_gap_pos_minus_neg"],
            "test_auroc": test_metrics["auroc"],
            "test_auprc": test_metrics["auprc"],
            "test_f1": test_metrics["f1"],
            "test_auroc_per_peptide_macro": test_metrics["auroc_per_peptide_macro"],
            "test_score_gap_pos_minus_neg": test_metrics["score_gap_pos_minus_neg"],
            "threshold": thr,
        })

        if best_random_by_val is None or val_metrics["auroc"] > best_random_by_val["val"]["auroc"]:
            best_random_by_val = {
                "seed": seed,
                "val": val_metrics,
                "test": test_metrics,
                "val_scores": val_scores,
                "val_labels": val_labels,
                "test_scores": test_scores,
                "test_labels": test_labels,
                "val_peptides": val_peptides,
                "test_peptides": test_peptides,
                "threshold": thr,
            }

        print(
            f"  Seed {seed} | "
            f"val AUROC={val_metrics['auroc']:.4f} | "
            f"test AUROC={test_metrics['auroc']:.4f}"
        )

    random_summary = {}
    for split in ["val", "test"]:
        for metric in ["auroc", "auroc_per_peptide_macro", "auprc", "f1", "score_gap_pos_minus_neg"]:
            random_summary.update(
                summarise_random_metric(random_results, split=split, metric=metric)
            )

    full_results["random_projection_no_training"] = random_results
    full_results["random_projection_summary"] = random_summary

    # Plot best random seed by validation AUROC.
    if best_random_by_val is not None:
        best_seed = best_random_by_val["seed"]
        thr = best_random_by_val["threshold"]

        plot_score_histogram(
            best_random_by_val["val_scores"],
            best_random_by_val["val_labels"],
            title=f"random_projection_no_training | best val seed {best_seed} | validation",
            save_path=fig_dir / f"random_projection_best_seed{best_seed}__val_hist.png",
            threshold=thr,
            peptides=best_random_by_val["val_peptides"],
            n_peptides_hist=cfg.n_peptides_hist,
        )

        plot_score_histogram(
            best_random_by_val["test_scores"],
            best_random_by_val["test_labels"],
            title=f"random_projection_no_training | best val seed {best_seed} | test",
            save_path=fig_dir / f"random_projection_best_seed{best_seed}__test_hist.png",
            threshold=thr,
            peptides=best_random_by_val["test_peptides"],
            n_peptides_hist=cfg.n_peptides_hist,
        )

    # Plot random AUROC distribution.
    rand_val_aurocs = np.array([r["val"]["auroc"] for r in random_results])
    rand_test_aurocs = np.array([r["test"]["auroc"] for r in random_results])

    plt.figure(figsize=(7, 5))
    plt.hist(rand_val_aurocs, bins=20, alpha=0.7, density=True, label="validation")
    plt.hist(rand_test_aurocs, bins=20, alpha=0.7, density=True, label="test")
    plt.xlabel("AUROC")
    plt.ylabel("Density")
    plt.title("Random projection no-training AUROC distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "random_projection_auroc_distribution.png", dpi=200)
    plt.close()

    # ------------------------------------------------------------
    # SAVE OUTPUTS
    # ------------------------------------------------------------
    summary_df = pd.DataFrame(summary_rows)

    summary_csv = out_dir / f"no_training_baseline_summary__{cfg.run_tag}.csv"
    per_peptide_csv = out_dir / f"no_training_baseline_per_peptide_stats__{cfg.run_tag}.csv"
    full_json = out_dir / f"no_training_baseline_full_results__{cfg.run_tag}.json"
    random_summary_json = out_dir / f"random_projection_summary__{cfg.run_tag}.json"

    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(per_peptide_rows).to_csv(per_peptide_csv, index=False)

    with open(full_json, "w") as f:
        json.dump(full_results, f, indent=2)

    with open(random_summary_json, "w") as f:
        json.dump(random_summary, f, indent=2)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote per-peptide CSV: {per_peptide_csv}")
    print(f"Wrote full JSON: {full_json}")
    print(f"Wrote random summary JSON: {random_summary_json}")
    print(f"Wrote figures to: {fig_dir}")

    print("\nTop rows by validation AUROC:")
    print(
        summary_df.sort_values("val_auroc", ascending=False)
        .head(15)
        .to_string(index=False)
    )

    print("\nRandom projection AUROC summary:")
    for k, v in random_summary.items():
        if "auroc" in k:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()