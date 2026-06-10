#!/usr/bin/env python3
"""
Post-hoc cross-reactivity plots from a saved HPO checkpoint.

This script:
1) loads projection heads from a saved checkpoint
2) runs inference on val/test shard splits
3) joins embeddings to metadata via pair_id from val/test CSVs
4) computes simple cross-reactivity group metrics on positives:
   - peptide-centric: pairwise cosine among TCR embeddings (eT)
   - TCR-centric: pairwise cosine among pMHC embeddings (ePH)
5) saves histograms + group summary CSVs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hpo_vicreg_hamiltonian_from_shards import (
    ESMProjectionHead,
    PMHCProjectionHead,
    ShardedBatchTripletDataset,
    row_normalise,
    set_seed,
)


def pairwise_cos_values(x: np.ndarray) -> np.ndarray:
    if x.shape[0] < 2:
        return np.array([])
    sim = x @ x.T
    iu = np.triu_indices(sim.shape[0], k=1)
    return sim[iu]


@torch.no_grad()
def infer_split(
    split_dir: Path,
    tcr_proj: torch.nn.Module,
    pmhc_proj: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    ds = ShardedBatchTripletDataset(split_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])

    rows: List[Dict] = []
    for batch in loader:
        zt = tcr_proj(batch["emb_T"].to(device), batch["mask_T"].to(device))
        zph = pmhc_proj(
            batch["emb_P"].to(device),
            batch["mask_P"].to(device),
            batch["emb_H"].to(device),
            batch["mask_H"].to(device),
        )
        zt_np = zt.detach().cpu().numpy()
        zph_np = zph.detach().cpu().numpy()
        et = row_normalise(zt).detach().cpu().numpy()
        eph = row_normalise(zph).detach().cpu().numpy()
        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        pair_ids = list(batch["pair_id"])

        for i, pid in enumerate(pair_ids):
            rows.append(
                {
                    "pair_id": str(pid),
                    "binding_flag": int(labels[i]),
                    "rT": zt_np[i],
                    "rPH": zph_np[i],
                    "eT": et[i],
                    "ePH": eph[i],
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def infer_split_raw_meanpool(split_dir: Path, device: torch.device) -> pd.DataFrame:
    ds = ShardedBatchTripletDataset(split_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
    rows: List[Dict] = []
    for batch in loader:
        emb_t = batch["emb_T"].to(device)
        emb_p = batch["emb_P"].to(device)
        emb_h = batch["emb_H"].to(device)
        m_t = batch["mask_T"].to(device).unsqueeze(-1).float()
        m_p = batch["mask_P"].to(device).unsqueeze(-1).float()
        m_h = batch["mask_H"].to(device).unsqueeze(-1).float()

        t = (emb_t * m_t).sum(dim=1) / m_t.sum(dim=1).clamp(min=1.0)
        p = (emb_p * m_p).sum(dim=1) / m_p.sum(dim=1).clamp(min=1.0)
        h = (emb_h * m_h).sum(dim=1) / m_h.sum(dim=1).clamp(min=1.0)

        ph = 0.5 * p + 0.5 * h
        t_np = t.detach().cpu().numpy()
        ph_np = ph.detach().cpu().numpy()
        et = F.normalize(t, dim=-1).detach().cpu().numpy()
        eph = F.normalize(ph, dim=-1).detach().cpu().numpy()
        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        pair_ids = list(batch["pair_id"])
        for i, pid in enumerate(pair_ids):
            rows.append(
                {
                    "pair_id": str(pid),
                    "binding_flag": int(labels[i]),
                    "rT": t_np[i],
                    "rPH": ph_np[i],
                    "eT": et[i],
                    "ePH": eph[i],
                }
            )
    return pd.DataFrame(rows)


@torch.no_grad()
def infer_split_random_projection(
    split_dir: Path,
    device: torch.device,
    seed: int,
    rL: int,
    rD: int,
    d: int,
    r_ph: float,
) -> pd.DataFrame:
    set_seed(seed)
    ds = ShardedBatchTripletDataset(split_dir)
    sample = ds[0]
    D = sample["emb_T"].shape[2]
    L_T_max = sample["emb_T"].shape[1]
    L_P_max = sample["emb_P"].shape[1]
    L_H_max = sample["emb_H"].shape[1]
    tcr_proj = ESMProjectionHead(D, rL, rD, d, L_T_max, dropout=0.0).to(device)
    pmhc_proj = PMHCProjectionHead(D, rL, rD, d, L_P_max, L_H_max, R_PH=r_ph, dropout=0.0).to(device)
    tcr_proj.eval()
    pmhc_proj.eval()
    return infer_split(split_dir, tcr_proj, pmhc_proj, device)


def compute_group_reactivity(
    df_pos: pd.DataFrame,
    group_col: str,
    emb_col: str,
    rng: np.random.Generator,
    n_random: int = 100,
) -> pd.DataFrame:
    all_idx = df_pos["row_idx"].to_numpy()
    emb_mat = np.stack(df_pos[emb_col].values, axis=0)
    out = []
    for key, grp in df_pos.groupby(group_col):
        idx = grp["row_idx"].to_numpy()
        if len(idx) < 2:
            continue

        obs_vals = pairwise_cos_values(emb_mat[idx])
        if obs_vals.size == 0:
            continue

        eligible = np.setdiff1d(all_idx, idx)
        if len(eligible) < len(idx):
            continue

        baseline_medians = []
        for _ in range(n_random):
            sampled = rng.choice(eligible, size=len(idx), replace=False)
            rand_vals = pairwise_cos_values(emb_mat[sampled])
            if rand_vals.size > 0:
                baseline_medians.append(float(np.median(rand_vals)))

        if len(baseline_medians) == 0:
            continue

        obs_median = float(np.median(obs_vals))
        baseline_median = float(np.mean(baseline_medians))
        out.append(
            {
                "group_key": key,
                "group_size": int(len(idx)),
                "observed_median_pairwise_cos": obs_median,
                "baseline_median_pairwise_cos": baseline_median,
                "delta_vs_random": obs_median - baseline_median,
            }
        )
    out_df = pd.DataFrame(out).sort_values(
        ["group_size", "observed_median_pairwise_cos"], ascending=[False, False]
    )
    return out_df


def plot_hist(vals: np.ndarray, title: str, out_path: Path, xlabel: str) -> None:
    vals = vals[np.isfinite(vals)]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if vals.size > 0:
        plt.hist(vals, bins=40, alpha=0.8, density=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_scatter(df: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = df["group_size"].values
    y = df["delta_vs_random"].values
    m = np.isfinite(y)
    plt.figure(figsize=(7, 5))
    plt.scatter(x[m], y[m], s=14, alpha=0.7)
    plt.title(title)
    plt.xlabel("Group size")
    plt.ylabel("Observed - matched random median cosine")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_box_observed_vs_random(df: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    obs = df["observed_median_pairwise_cos"].to_numpy()
    base = df["baseline_median_pairwise_cos"].to_numpy()
    obs = obs[np.isfinite(obs)]
    base = base[np.isfinite(base)]
    plt.figure(figsize=(6, 5))
    if obs.size > 0 and base.size > 0:
        plt.boxplot([obs, base], tick_labels=["Cross-reactive group", "Matched random"], showfliers=True)
    plt.ylabel("Median pairwise cosine")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multi_baseline_boxplot(
    group_tables: Dict[str, pd.DataFrame],
    metric_col: str,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = []
    data = []
    for name, df in group_tables.items():
        vals = df[metric_col].to_numpy()
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        labels.append(name)
        data.append(vals)
    plt.figure(figsize=(8, 5))
    if data:
        plt.boxplot(data, tick_labels=labels, showfliers=True)
    plt.ylabel("Median pairwise cosine")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_multi_baseline_boxplot_subplots(
    group_tables_norm: Dict[str, pd.DataFrame],
    group_tables_raw: Dict[str, pd.DataFrame],
    metric_col: str,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def collect(group_tables: Dict[str, pd.DataFrame]):
        labels = []
        data = []
        for name, df in group_tables.items():
            vals = df[metric_col].to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            labels.append(name)
            data.append(vals)
        return labels, data

    labels_norm, data_norm = collect(group_tables_norm)
    labels_raw, data_raw = collect(group_tables_raw)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    if data_norm:
        axes[0].boxplot(data_norm, tick_labels=labels_norm, showfliers=True)
    axes[0].set_title("Normalized space")
    axes[0].set_ylabel("Median pairwise cosine")
    axes[0].tick_params(axis="x", labelrotation=15)

    if data_raw:
        axes[1].boxplot(data_raw, tick_labels=labels_raw, showfliers=True)
    axes[1].set_title("Raw space")
    axes[1].tick_params(axis="x", labelrotation=15)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="/home/natasha/multimodal_model/models/checkpoints/hpo_training/angular_variance_rerun/seed31__rL8__rD16__d128__a0.1__b25.0__c1.0__ang25.0__gang0.05__best.pt",
    )
    ap.add_argument(
        "--embed-root",
        default="/home/natasha/multimodal_model/models/embeddings/no_boltz_train_dedup",
    )
    ap.add_argument(
        "--val-csv",
        default="/home/natasha/multimodal_model/data/val/val_df_clean_pos_neg.csv",
    )
    ap.add_argument(
        "--test-csv",
        default="/home/natasha/multimodal_model/data/test/test_df_clean_pos_neg.csv",
    )
    ap.add_argument(
        "--out-dir",
        default="/home/natasha/multimodal_model/models/figures/hpo_training/angular_variance_rerun/cross_reactivity",
    )
    ap.add_argument(
        "--baseline-full-json",
        default="/home/natasha/multimodal_model/models/checkpoints/no_training_baselines/no_training_baseline_full_results.json",
    )
    ap.add_argument(
        "--baseline-summary-csv",
        default="/home/natasha/multimodal_model/models/checkpoints/no_training_baselines/no_training_baseline_summary.csv",
    )
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--n-random", type=int, default=100)
    args = ap.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    hp = ckpt["hp"]
    state = ckpt["best"]["state"]

    train_ds = ShardedBatchTripletDataset(Path(args.embed_root) / "train")
    sample = train_ds[0]
    D = sample["emb_T"].shape[2]
    L_T_max = sample["emb_T"].shape[1]
    L_P_max = sample["emb_P"].shape[1]
    L_H_max = sample["emb_H"].shape[1]

    tcr_proj = ESMProjectionHead(D, hp["rL"], hp["rD"], hp["d"], L_T_max, dropout=cfg["dropout"]).to(device)
    pmhc_proj = PMHCProjectionHead(
        D, hp["rL"], hp["rD"], hp["d"], L_P_max, L_H_max, R_PH=cfg["R_PH"], dropout=cfg["dropout"]
    ).to(device)
    tcr_proj.load_state_dict(state["tcr"])
    pmhc_proj.load_state_dict(state["pmhc"])
    tcr_proj.eval()
    pmhc_proj.eval()

    val_emb = infer_split(Path(args.embed_root) / "val", tcr_proj, pmhc_proj, device)
    test_emb = infer_split(Path(args.embed_root) / "test", tcr_proj, pmhc_proj, device)

    meta = pd.concat([pd.read_csv(args.val_csv), pd.read_csv(args.test_csv)], ignore_index=True)
    meta["pair_id"] = meta["pair_id"].astype(str)
    meta = meta[["pair_id", "binding_flag", "TCR_full", "Peptide", "HLA_sequence"]].copy()
    meta = meta.drop_duplicates(subset=["pair_id"], keep="first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_full = json.loads(Path(args.baseline_full_json).read_text())
    baseline_cfg = baseline_full["config"]
    summary_df = pd.read_csv(args.baseline_summary_csv)
    rand_rows = summary_df[summary_df["baseline"] == "random_projection_no_training"].copy()
    rand_rows = rand_rows.sort_values("val_auroc", ascending=False)
    best_random_seed = int(rand_rows.iloc[0]["seed"]) if len(rand_rows) else int(args.seed)

    val_raw_emb = infer_split_raw_meanpool(Path(args.embed_root) / "val", device)
    test_raw_emb = infer_split_raw_meanpool(Path(args.embed_root) / "test", device)
    val_rand_emb = infer_split_random_projection(
        Path(args.embed_root) / "val",
        device=device,
        seed=best_random_seed,
        rL=int(baseline_cfg["rL"]),
        rD=int(baseline_cfg["rD"]),
        d=int(baseline_cfg["d"]),
        r_ph=float(baseline_cfg["r_ph"]),
    )
    test_rand_emb = infer_split_random_projection(
        Path(args.embed_root) / "test",
        device=device,
        seed=best_random_seed,
        rL=int(baseline_cfg["rL"]),
        rD=int(baseline_cfg["rD"]),
        d=int(baseline_cfg["d"]),
        r_ph=float(baseline_cfg["r_ph"]),
    )

    split_tables = {
        "val": {
            "trained": val_emb,
            "raw_esm_meanpool": val_raw_emb,
            f"random_no_training_seed{best_random_seed}": val_rand_emb,
        },
        "test": {
            "trained": test_emb,
            "raw_esm_meanpool": test_raw_emb,
            f"random_no_training_seed{best_random_seed}": test_rand_emb,
        },
    }

    for split_name, variant_tables in split_tables.items():
        pep_variant_group_tables: Dict[str, pd.DataFrame] = {}
        tcr_variant_group_tables: Dict[str, pd.DataFrame] = {}
        pep_variant_group_tables_raw: Dict[str, pd.DataFrame] = {}
        tcr_variant_group_tables_raw: Dict[str, pd.DataFrame] = {}

        for variant_name, emb_df in variant_tables.items():
            m = emb_df.merge(meta, on="pair_id", how="left", suffixes=("_shard", "_csv"))
            m["binding_flag"] = m["binding_flag_shard"].astype(int)

            pos = m[m["binding_flag"] == 1].copy().reset_index(drop=True)
            pos["row_idx"] = np.arange(len(pos))

            pep_df = compute_group_reactivity(
                pos, group_col="Peptide", emb_col="eT", rng=rng, n_random=args.n_random
            )
            tcr_df = compute_group_reactivity(
                pos, group_col="TCR_full", emb_col="ePH", rng=rng, n_random=args.n_random
            )
            pep_df_raw = compute_group_reactivity(
                pos, group_col="Peptide", emb_col="rT", rng=rng, n_random=args.n_random
            )
            tcr_df_raw = compute_group_reactivity(
                pos, group_col="TCR_full", emb_col="rPH", rng=rng, n_random=args.n_random
            )

            pep_variant_group_tables[variant_name] = pep_df
            tcr_variant_group_tables[variant_name] = tcr_df
            pep_variant_group_tables_raw[variant_name] = pep_df_raw
            tcr_variant_group_tables_raw[variant_name] = tcr_df_raw
            pep_df.to_csv(out_dir / f"{split_name}_{variant_name}_peptide_centric.csv", index=False)
            tcr_df.to_csv(out_dir / f"{split_name}_{variant_name}_tcr_centric.csv", index=False)
            pep_df_raw.to_csv(out_dir / f"{split_name}_{variant_name}_peptide_centric_raw.csv", index=False)
            tcr_df_raw.to_csv(out_dir / f"{split_name}_{variant_name}_tcr_centric_raw.csv", index=False)

            if variant_name == "trained":
                plot_hist(
                    pep_df["observed_median_pairwise_cos"].to_numpy(),
                    f"{split_name} peptide-centric observed cross-reactivity",
                    out_dir / f"{split_name}_peptide_centric_hist.png",
                    xlabel="Observed median pairwise cosine",
                )
                plot_hist(
                    pep_df["baseline_median_pairwise_cos"].to_numpy(),
                    f"{split_name} peptide-centric matched random baseline",
                    out_dir / f"{split_name}_peptide_centric_baseline_hist.png",
                    xlabel="Matched random median pairwise cosine",
                )
                plot_scatter(
                    pep_df,
                    f"{split_name} peptide-centric (size vs delta)",
                    out_dir / f"{split_name}_peptide_centric_scatter.png",
                )
                plot_box_observed_vs_random(
                    pep_df,
                    f"{split_name} peptide-centric: within-group vs matched random",
                    out_dir / f"{split_name}_peptide_centric_boxplot.png",
                )

                plot_hist(
                    tcr_df["observed_median_pairwise_cos"].to_numpy(),
                    f"{split_name} TCR-centric observed cross-reactivity",
                    out_dir / f"{split_name}_tcr_centric_hist.png",
                    xlabel="Observed median pairwise cosine",
                )
                plot_hist(
                    tcr_df["baseline_median_pairwise_cos"].to_numpy(),
                    f"{split_name} TCR-centric matched random baseline",
                    out_dir / f"{split_name}_tcr_centric_baseline_hist.png",
                    xlabel="Matched random median pairwise cosine",
                )
                plot_scatter(
                    tcr_df,
                    f"{split_name} TCR-centric (size vs delta)",
                    out_dir / f"{split_name}_tcr_centric_scatter.png",
                )
                plot_box_observed_vs_random(
                    tcr_df,
                    f"{split_name} TCR-centric: within-group vs matched random",
                    out_dir / f"{split_name}_tcr_centric_boxplot.png",
                )

            print(
                f"[{split_name}/{variant_name}] positives={len(pos)} | "
                f"peptide-groups={len(pep_df)} | tcr-groups={len(tcr_df)}",
                flush=True,
            )

        plot_multi_baseline_boxplot(
            pep_variant_group_tables,
            metric_col="observed_median_pairwise_cos",
            title=f"{split_name} peptide-centric: observed cross-reactivity comparison",
            out_path=out_dir / f"{split_name}_peptide_centric_model_comparison_boxplot.png",
        )
        plot_multi_baseline_boxplot_subplots(
            pep_variant_group_tables,
            pep_variant_group_tables_raw,
            metric_col="observed_median_pairwise_cos",
            title=f"{split_name} peptide-centric: observed comparison (normalized vs raw)",
            out_path=out_dir / f"{split_name}_peptide_centric_model_comparison_subplots.png",
        )
        plot_multi_baseline_boxplot(
            pep_variant_group_tables,
            metric_col="delta_vs_random",
            title=f"{split_name} peptide-centric: delta-vs-random comparison",
            out_path=out_dir / f"{split_name}_peptide_centric_delta_comparison_boxplot.png",
        )
        plot_multi_baseline_boxplot_subplots(
            pep_variant_group_tables,
            pep_variant_group_tables_raw,
            metric_col="delta_vs_random",
            title=f"{split_name} peptide-centric: delta comparison (normalized vs raw)",
            out_path=out_dir / f"{split_name}_peptide_centric_delta_comparison_subplots.png",
        )
        plot_multi_baseline_boxplot(
            tcr_variant_group_tables,
            metric_col="observed_median_pairwise_cos",
            title=f"{split_name} TCR-centric: observed cross-reactivity comparison",
            out_path=out_dir / f"{split_name}_tcr_centric_model_comparison_boxplot.png",
        )
        plot_multi_baseline_boxplot_subplots(
            tcr_variant_group_tables,
            tcr_variant_group_tables_raw,
            metric_col="observed_median_pairwise_cos",
            title=f"{split_name} TCR-centric: observed comparison (normalized vs raw)",
            out_path=out_dir / f"{split_name}_tcr_centric_model_comparison_subplots.png",
        )
        plot_multi_baseline_boxplot(
            tcr_variant_group_tables,
            metric_col="delta_vs_random",
            title=f"{split_name} TCR-centric: delta-vs-random comparison",
            out_path=out_dir / f"{split_name}_tcr_centric_delta_comparison_boxplot.png",
        )
        plot_multi_baseline_boxplot_subplots(
            tcr_variant_group_tables,
            tcr_variant_group_tables_raw,
            metric_col="delta_vs_random",
            title=f"{split_name} TCR-centric: delta comparison (normalized vs raw)",
            out_path=out_dir / f"{split_name}_tcr_centric_delta_comparison_subplots.png",
        )

    print(f"Saved cross-reactivity outputs (trained + baselines) to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()

