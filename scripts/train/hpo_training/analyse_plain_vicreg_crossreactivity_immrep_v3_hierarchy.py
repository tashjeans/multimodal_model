#!/usr/bin/env python3
"""
Post-hoc analysis for a trained plain-VICReg TCR-pMHC model.

This script is intended to live in:
    /home/natasha/multimodal_model/scripts/train/hpo_training

It reuses the model classes and scoring utilities from:
    train_plain_vicreg_tulip_simple.py

Main questions answered
----------------------
1. Cross-reactivity:
   Are TCRs that bind the same peptide more directionally similar in the trained TCR latent space
   than they are in raw ESM space?

2. Repeated-TCR control:
   Is the apparent same-peptide closeness driven by the model seeing the same
   TCR multiple times? The script therefore reports both:
     - all positive TCRs; and
     - deduplicated positive TCRs using a hash of the masked raw TCR ESM tensor.
   It also stratifies peptide groups by group size and by TCR repeat frequency.

3. Collapse check:
   Have same-peptide TCRs collapsed to almost the same point in the latent space?
   The script computes within-peptide pairwise cosine similarities, centroid dispersion,
   nearest-neighbour distances and several collapse flags.

4. IMMREP evaluation:
   Runs the trained model on IMMREP test shards and computes peptide-level
   McClish-standardised partial ROC AUC at max_fpr=0.1, then macro-averages
   across valid peptides. This matches the stated Macro AUC0.1 evaluation logic.

Score convention
----------------
Higher score = more likely binder.

Primary scores used here:
    model_score_mse = -MSE(zT, zPH)
    raw_esm_score_mse = -MSE(mean_pool(TCR_ESM), weighted_mean_pool(pMHC_ESM))

The script also saves cosine/Hamiltonian-style scores for continuity:
    score_cos = 1 + cosine = -H

Why MSE is primary here
-----------------------
The trained plain VICReg model optimises an invariance MSE between unnormalised
zT and zPH. Therefore -MSE is the most direct scoring analogue for both the
trained model and raw ESM baseline. Cosine/H scores are still saved as secondary
comparators because earlier diagnostics used them.
"""

from __future__ import annotations

SCRIPT_VERSION = "v3_hierarchy_multimetric_seqcontrol_checkpoint_shape_safe_2026_05_10"

import argparse
import glob
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# The script should sit next to train_plain_vicreg_tulip_simple.py.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from train_plain_vicreg_tulip_simple import (  # noqa: E402
    ShardedBatchDataset,
    initialise_models,
    row_normalise,
    masked_mean_pool,
)


def set_seed(seed: int) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def latest_checkpoint(checkpoint_dir: Path) -> Path:
    candidates = sorted(checkpoint_dir.glob("*__best.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No *__best.pt checkpoint found in {checkpoint_dir}")
    return candidates[0]


def load_checkpoint_and_models(checkpoint_path: Path, shard_dir: Path, device: torch.device):
    """
    Load the checkpoint using model dimensions inferred from the checkpoint itself.

    The IMMREP test shards may have shorter padded sequence lengths than the
    training shards. The learned low-rank positional matrices must therefore be
    reconstructed with the checkpoint shapes, not the test-shard shapes.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    tcr_sd = ckpt["tcr_state_dict"]
    pmhc_sd = ckpt["pmhc_state_dict"]

    D = int(tcr_sd["B_c"].shape[0])
    L_T = int(tcr_sd["A_c"].shape[0])
    L_P = int(pmhc_sd["pep_encoder.A_c"].shape[0])
    L_H = int(pmhc_sd["hla_encoder.A_c"].shape[0])

    class Cfg:
        pass

    cfg = Cfg()
    cfg.rL = int(cfg_dict.get("rL", tcr_sd["A_c"].shape[1]))
    cfg.rD = int(cfg_dict.get("rD", tcr_sd["B_c"].shape[1]))
    cfg.d = int(cfg_dict.get("d", tcr_sd["H_c"].shape[1]))
    cfg.R_PH = float(cfg_dict.get("R_PH", 0.7))
    cfg.dropout = float(cfg_dict.get("dropout", 0.0))

    print(f"Loaded checkpoint model shapes | D={D} | L_T={L_T} | L_P={L_P} | L_H={L_H}", flush=True)

    tcr, pmhc = initialise_models(cfg, (D, L_T, L_P, L_H), device)
    tcr.load_state_dict(tcr_sd)
    pmhc.load_state_dict(pmhc_sd)
    tcr.eval()
    pmhc.eval()

    ds = ShardedBatchDataset(shard_dir)
    return ckpt, cfg_dict, tcr, pmhc, ds


def tensor_hash(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> str:
    """Stable hash for detecting repeated TCRs from raw ESM tensors."""
    x_cpu = x.detach().cpu()
    if mask is not None:
        mask_cpu = mask.detach().cpu().bool()
        if x_cpu.ndim == 2:
            x_cpu = x_cpu[mask_cpu]
        elif x_cpu.ndim == 3:
            x_cpu = x_cpu[:, mask_cpu]
    arr = np.ascontiguousarray(x_cpu.numpy().astype(np.float32))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def load_pair_metadata(csv_path: Optional[str]) -> pd.DataFrame:
    """Load pair_id-level metadata used for peptide grouping and sequence controls."""
    if not csv_path:
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        raise ValueError(f"{csv_path} must contain pair_id")

    peptide_col = None
    for c in ["Peptide", "peptide", "peptide_seq", "epitope", "Epitope"]:
        if c in df.columns:
            peptide_col = c
            break
    if peptide_col is None:
        raise ValueError(f"{csv_path} must contain a peptide column")

    seq_col = None
    for c in ["TCR_full", "tcr_full", "TCR", "tcr", "tcr_sequence", "sequence"]:
        if c in df.columns:
            seq_col = c
            break

    out = pd.DataFrame({
        "pair_id": df["pair_id"].astype(str),
        "peptide": df[peptide_col].astype(str),
    })
    if seq_col is not None:
        out["tcr_sequence"] = df[seq_col].astype(str)
    return out.drop_duplicates("pair_id", keep="first")


def metadata_maps(meta: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    if meta is None or len(meta) == 0:
        return {}, {}
    pair_to_peptide = dict(zip(meta["pair_id"].astype(str), meta["peptide"].astype(str)))
    if "tcr_sequence" in meta.columns:
        pair_to_tcr_sequence = dict(zip(meta["pair_id"].astype(str), meta["tcr_sequence"].astype(str)))
    else:
        pair_to_tcr_sequence = {}
    return pair_to_peptide, pair_to_tcr_sequence


def batch_pair_ids(batch) -> List[str]:
    pids = batch["pair_id"]
    if torch.is_tensor(pids):
        pids = pids.detach().cpu().numpy().tolist()
    return [str(x) for x in pids]


def batch_labels(batch) -> np.ndarray:
    y = batch["binding_flag"]
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return np.asarray(y).astype(int)


def peptide_from_batch(batch, pair_ids: List[str], pair_to_peptide: Dict[str, str]) -> List[str]:
    # Prefer explicit mapping when supplied, because it is the least ambiguous.
    if pair_to_peptide:
        missing = [pid for pid in pair_ids if pid not in pair_to_peptide]
        if missing:
            raise KeyError(f"Missing peptide mapping for {len(missing)} pair_ids. Examples: {missing[:10]}")
        return [pair_to_peptide[pid] for pid in pair_ids]

    # Otherwise try common batch keys.
    for key in ["peptide", "Peptide", "peptide_seq", "epitope", "Epitope"]:
        if key in batch:
            vals = batch[key]
            if torch.is_tensor(vals):
                vals = vals.detach().cpu().numpy().tolist()
            return [str(x) for x in vals]

    raise ValueError(
        "Could not infer peptide labels. Provide --immrep-csv with pair_id and peptide/Peptide, "
        "or include a peptide-like field in the shard batches."
    )


def per_sample_mse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b) ** 2).mean(dim=-1)


def pad_or_truncate_to_match(a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Only used defensively. Raw ESM TCR and pMHC should normally share D."""
    da = a.shape[-1]
    db = b.shape[-1]
    if da == db:
        return a, b
    d = min(da, db)
    return a[..., :d], b[..., :d]


@torch.no_grad()
def collect_split(
    shard_dir: Path,
    tcr,
    pmhc,
    device: torch.device,
    pair_to_peptide: Dict[str, str],
    pair_to_tcr_sequence: Dict[str, str],
    R_PH: float,
    batch_size: int,
    num_workers: int,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    ds = ShardedBatchDataset(shard_dir)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x[0])

    rows: List[Dict] = []
    arrays: Dict[str, List[np.ndarray]] = {
        "zT": [], "zPH": [], "model_T_norm": [], "raw_T_norm": [], "raw_PH_norm": [],
        "raw_T": [], "raw_PH": [],
    }

    for batch in loader:
        pair_ids = batch_pair_ids(batch)
        labels = batch_labels(batch)
        peptides = peptide_from_batch(batch, pair_ids, pair_to_peptide)

        emb_T = batch["emb_T"].to(device)
        mask_T = batch["mask_T"].to(device)
        emb_P = batch["emb_P"].to(device)
        mask_P = batch["mask_P"].to(device)
        emb_H = batch["emb_H"].to(device)
        mask_H = batch["mask_H"].to(device)

        zT = tcr(emb_T, mask_T)
        zPH = pmhc(emb_P, mask_P, emb_H, mask_H)

        model_mse = per_sample_mse(zT, zPH)
        model_score_mse = -model_mse
        model_T_norm = row_normalise(zT)
        model_PH_norm = row_normalise(zPH)
        model_cos = (model_T_norm * model_PH_norm).sum(dim=-1)
        model_H = -1.0 - model_cos
        model_score_cos = -model_H

        raw_T = masked_mean_pool(emb_T, mask_T)
        raw_P = masked_mean_pool(emb_P, mask_P)
        raw_HLA = masked_mean_pool(emb_H, mask_H)
        raw_PH = R_PH * raw_P + (1.0 - R_PH) * raw_HLA
        raw_T, raw_PH = pad_or_truncate_to_match(raw_T, raw_PH)

        raw_mse = per_sample_mse(raw_T, raw_PH)
        raw_score_mse = -raw_mse
        raw_T_norm = row_normalise(raw_T)
        raw_PH_norm = row_normalise(raw_PH)
        raw_cos = (raw_T_norm * raw_PH_norm).sum(dim=-1)
        raw_H_score = -1.0 - raw_cos
        raw_score_cos = -raw_H_score

        # Hash individual TCRs to detect exact repeats across the evaluation set.
        t_hashes = []
        for i in range(emb_T.shape[0]):
            L = int(mask_T[i].sum().item())
            t_hashes.append(tensor_hash(emb_T[i, :L, :]))

        zT_np = zT.detach().cpu().numpy()
        zPH_np = zPH.detach().cpu().numpy()
        model_T_norm_np = model_T_norm.detach().cpu().numpy()
        raw_T_norm_np = raw_T_norm.detach().cpu().numpy()
        raw_PH_norm_np = raw_PH_norm.detach().cpu().numpy()
        raw_T_np = raw_T.detach().cpu().numpy()
        raw_PH_np = raw_PH.detach().cpu().numpy()

        start_idx = len(rows)
        arrays["zT"].append(zT_np)
        arrays["zPH"].append(zPH_np)
        arrays["model_T_norm"].append(model_T_norm_np)
        arrays["raw_T_norm"].append(raw_T_norm_np)
        arrays["raw_PH_norm"].append(raw_PH_norm_np)
        arrays["raw_T"].append(raw_T_np)
        arrays["raw_PH"].append(raw_PH_np)

        for i, pid in enumerate(pair_ids):
            rows.append({
                "row_index": start_idx + i,
                "pair_id": pid,
                "label": int(labels[i]),
                "peptide": str(peptides[i]),
                "tcr_hash": t_hashes[i],
                "tcr_sequence": pair_to_tcr_sequence.get(pid, ""),
                "model_score_mse": float(model_score_mse[i].detach().cpu()),
                "model_mse": float(model_mse[i].detach().cpu()),
                "model_score_cos": float(model_score_cos[i].detach().cpu()),
                "model_H": float(model_H[i].detach().cpu()),
                "model_cos": float(model_cos[i].detach().cpu()),
                "raw_esm_score_mse": float(raw_score_mse[i].detach().cpu()),
                "raw_esm_mse": float(raw_mse[i].detach().cpu()),
                "raw_esm_score_cos": float(raw_score_cos[i].detach().cpu()),
                "raw_esm_H": float(raw_H_score[i].detach().cpu()),
                "raw_esm_cos": float(raw_cos[i].detach().cpu()),
            })

    df = pd.DataFrame(rows)
    counts = df["tcr_hash"].value_counts().to_dict()
    df["tcr_seen_count_in_split"] = df["tcr_hash"].map(counts).astype(int)
    out_arrays = {k: np.concatenate(v, axis=0) if v else np.empty((0, 1)) for k, v in arrays.items()}
    return df, out_arrays


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return X @ X.T


def pairwise_cosine_similarities(X: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    if n < 2:
        return np.array([], dtype=float)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        S = cosine_similarity_matrix(X)
        iu = np.triu_indices(n, k=1)
        return S[iu]
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n - 1, size=max_pairs)
    j = np.where(j >= i, j + 1, j)
    return np.sum(X[i] * X[j], axis=1)


def levenshtein_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein implementation for TCR sequence controls."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            substitute = previous[j - 1] + (ca != cb)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def normalised_edit_distance(a: str, b: str) -> float:
    a = "" if pd.isna(a) else str(a)
    b = "" if pd.isna(b) else str(b)
    denom = max(len(a), len(b), 1)
    return float(levenshtein_distance(a, b) / denom)


def sequence_distance_bin(x: float) -> str:
    if np.isnan(x):
        return "missing"
    if x <= 0.10:
        return "00-0.10 very similar"
    if x <= 0.25:
        return "0.10-0.25 similar"
    if x <= 0.50:
        return "0.25-0.50 dissimilar"
    return ">0.50 very dissimilar"


def pairwise_metric_values(X: np.ndarray, i: int, j: int, metric: str) -> float:
    """Pairwise embedding metric. For cosine, higher means closer. For distances, lower means closer."""
    xi = X[i]
    xj = X[j]
    if metric == "cosine":
        return float(np.sum(xi * xj))
    if metric == "euclidean":
        return float(np.linalg.norm(xi - xj))
    if metric == "mse":
        diff = xi - xj
        return float(np.mean(diff * diff))
    raise ValueError(f"Unknown metric: {metric}")


def metric_direction(metric: str) -> str:
    return "higher_is_closer" if metric == "cosine" else "lower_is_closer"


def is_within_closer_than_control(metric: str, within_value: float, control_value: float) -> bool:
    if math.isnan(within_value) or math.isnan(control_value):
        return False
    return within_value > control_value if metric == "cosine" else within_value < control_value


def signed_closeness_effect(metric: str, within_value: float, control_value: float) -> float:
    """Positive means same-peptide pairs are closer than the control under that metric."""
    if math.isnan(within_value) or math.isnan(control_value):
        return float("nan")
    return within_value - control_value if metric == "cosine" else control_value - within_value


def add_embedding_pair_metrics(row: dict, arrays: Dict[str, np.ndarray], i: int, j: int) -> None:
    """
    Add the full metric hierarchy for one TCR-TCR pair.

    raw_esm_cosine uses row-normalised raw ESM TCR embeddings.
    model_cosine uses row-normalised model TCR latents.
    raw_esm_euclidean/mse use unnormalised mean-pooled raw ESM TCR embeddings.
    model_euclidean/mse use unnormalised model zT latents.
    """
    metric_specs = [
        ("model", "cosine", "model_T_norm"),
        ("raw_esm", "cosine", "raw_T_norm"),
        ("model", "euclidean", "zT"),
        ("raw_esm", "euclidean", "raw_T"),
        ("model", "mse", "zT"),
        ("raw_esm", "mse", "raw_T"),
    ]
    for space, metric, key in metric_specs:
        col = f"{space}_tcr_{metric}"
        row[col] = pairwise_metric_values(arrays[key], i, j, metric)


def build_pairwise_tcr_table(
    df_pos: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    same_peptide_only: bool,
    max_random_pairs: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Build a TCR-TCR pair table with sequence distance and embedding cosine similarity.

    For same_peptide_only=True, all within-peptide positive TCR pairs are included.
    For same_peptide_only=False, a matched-size sample of different-peptide positive
    TCR pairs is drawn. This gives the sequence-distance-controlled random baseline.
    """
    rows = []
    if "tcr_sequence" not in df_pos.columns or df_pos["tcr_sequence"].replace("", np.nan).isna().all():
        return pd.DataFrame()

    df_pos = df_pos.set_index("row_index", drop=False)

    def add_pair(i: int, j: int, same_peptide: bool):
        seq_i = df_pos.loc[i, "tcr_sequence"]
        seq_j = df_pos.loc[j, "tcr_sequence"]
        seq_dist = normalised_edit_distance(seq_i, seq_j)
        row = {
            "same_peptide": bool(same_peptide),
            "peptide_i": df_pos.loc[i, "peptide"],
            "peptide_j": df_pos.loc[j, "peptide"],
            "row_i": int(i),
            "row_j": int(j),
            "pair_id_i": df_pos.loc[i, "pair_id"],
            "pair_id_j": df_pos.loc[j, "pair_id"],
            "normalised_edit_distance": seq_dist,
            "sequence_distance_bin": sequence_distance_bin(seq_dist),
        }
        add_embedding_pair_metrics(row, arrays, int(i), int(j))
        row["delta_model_minus_raw_cosine"] = row["model_tcr_cosine"] - row["raw_esm_tcr_cosine"]
        row["delta_model_minus_raw_euclidean"] = row["model_tcr_euclidean"] - row["raw_esm_tcr_euclidean"]
        row["delta_model_minus_raw_mse"] = row["model_tcr_mse"] - row["raw_esm_tcr_mse"]
        row["model_more_similar_than_raw_cosine"] = bool(row["model_tcr_cosine"] > row["raw_esm_tcr_cosine"])
        rows.append(row)

    if same_peptide_only:
        for _, grp in df_pos.groupby("peptide", sort=True):
            idx = grp.index.to_numpy()
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    add_pair(int(idx[a]), int(idx[b]), True)
    else:
        idx_all = df_pos.index.to_numpy()
        peptides = df_pos.loc[idx_all, "peptide"].to_numpy()
        n_attempts = 0
        while len(rows) < max_random_pairs and n_attempts < max_random_pairs * 20:
            n_attempts += 1
            i, j = rng.choice(idx_all, size=2, replace=False)
            if peptides[np.where(idx_all == i)[0][0]] == peptides[np.where(idx_all == j)[0][0]]:
                continue
            add_pair(int(i), int(j), False)
    return pd.DataFrame(rows)


def sequence_controlled_crossreactivity(
    df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    rng: np.random.Generator,
    deduplicate_tcrs: bool,
    max_random_pairs: int = 25000,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_pos = df[df["label"] == 1].copy()
    if deduplicate_tcrs:
        df_pos = df_pos.sort_values("pair_id").drop_duplicates("tcr_hash", keep="first").copy()

    same = build_pairwise_tcr_table(df_pos, arrays, same_peptide_only=True, max_random_pairs=max_random_pairs, rng=rng)
    if len(same) == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    random_pairs = build_pairwise_tcr_table(
        df_pos, arrays, same_peptide_only=False, max_random_pairs=min(max_random_pairs, len(same)), rng=rng
    )
    pair_table = pd.concat([same, random_pairs], ignore_index=True)
    pair_table["deduplicated_tcrs"] = bool(deduplicate_tcrs)

    metric_cols = [
        "model_tcr_cosine", "raw_esm_tcr_cosine",
        "model_tcr_euclidean", "raw_esm_tcr_euclidean",
        "model_tcr_mse", "raw_esm_tcr_mse",
    ]
    agg_spec = {
        "n_pairs": ("normalised_edit_distance", "count"),
        "mean_normalised_edit_distance": ("normalised_edit_distance", "mean"),
    }
    for col in metric_cols:
        agg_spec[f"mean_{col}"] = (col, "mean")
        agg_spec[f"median_{col}"] = (col, "median")
    summary = pair_table.groupby(["deduplicated_tcrs", "same_peptide", "sequence_distance_bin"], dropna=False).agg(**agg_spec).reset_index()

    # Metric hierarchy: same-peptide enrichment versus sampled different-peptide controls.
    hierarchy_rows = []
    for seq_bin, grp_bin in pair_table.groupby("sequence_distance_bin", dropna=False):
        same_grp = grp_bin[grp_bin["same_peptide"]]
        ctrl_grp = grp_bin[~grp_bin["same_peptide"]]
        if len(same_grp) == 0 or len(ctrl_grp) == 0:
            continue
        for space in ["model", "raw_esm"]:
            for metric in ["cosine", "euclidean", "mse"]:
                col = f"{space}_tcr_{metric}"
                within = float(same_grp[col].mean())
                control = float(ctrl_grp[col].mean())
                hierarchy_rows.append({
                    "deduplicated_tcrs": bool(deduplicate_tcrs),
                    "sequence_distance_bin": seq_bin,
                    "embedding_space": space,
                    "metric": metric,
                    "closer_direction": metric_direction(metric),
                    "n_same_peptide_pairs": int(len(same_grp)),
                    "n_control_pairs": int(len(ctrl_grp)),
                    "mean_same_peptide": within,
                    "mean_control_different_peptide": control,
                    "signed_closeness_effect_vs_control": signed_closeness_effect(metric, within, control),
                    "same_peptide_closer_than_control": is_within_closer_than_control(metric, within, control),
                })

    # Also add an all-sequence-bin aggregate so the first read is simple.
    same_all = pair_table[pair_table["same_peptide"]]
    ctrl_all = pair_table[~pair_table["same_peptide"]]
    if len(same_all) and len(ctrl_all):
        for space in ["model", "raw_esm"]:
            for metric in ["cosine", "euclidean", "mse"]:
                col = f"{space}_tcr_{metric}"
                within = float(same_all[col].mean())
                control = float(ctrl_all[col].mean())
                hierarchy_rows.append({
                    "deduplicated_tcrs": bool(deduplicate_tcrs),
                    "sequence_distance_bin": "ALL",
                    "embedding_space": space,
                    "metric": metric,
                    "closer_direction": metric_direction(metric),
                    "n_same_peptide_pairs": int(len(same_all)),
                    "n_control_pairs": int(len(ctrl_all)),
                    "mean_same_peptide": within,
                    "mean_control_different_peptide": control,
                    "signed_closeness_effect_vs_control": signed_closeness_effect(metric, within, control),
                    "same_peptide_closer_than_control": is_within_closer_than_control(metric, within, control),
                })

    hierarchy_summary = pd.DataFrame(hierarchy_rows)

    # Residual control: regress each metric against sequence distance over all sampled pairs.
    residual_rows = []
    for space in ["model", "raw_esm"]:
        for metric in ["cosine", "euclidean", "mse"]:
            col = f"{space}_tcr_{metric}"
            tmp = pair_table[["same_peptide", "normalised_edit_distance", col]].dropna().copy()
            if len(tmp) >= 3 and tmp["normalised_edit_distance"].nunique() > 1:
                x = tmp["normalised_edit_distance"].to_numpy(dtype=float)
                y = tmp[col].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, deg=1)
                tmp["residual"] = y - (slope * x + intercept)
                for same_flag, grp in tmp.groupby("same_peptide"):
                    residual_rows.append({
                        "deduplicated_tcrs": bool(deduplicate_tcrs),
                        "embedding_space": space,
                        "metric": metric,
                        "closer_direction": metric_direction(metric),
                        "same_peptide": bool(same_flag),
                        "n_pairs": int(len(grp)),
                        "mean_residual_after_sequence_control": float(grp["residual"].mean()),
                        "median_residual_after_sequence_control": float(grp["residual"].median()),
                        "sequence_similarity_slope": float(slope),
                        "sequence_similarity_intercept": float(intercept),
                    })
    residual_summary = pd.DataFrame(residual_rows)
    return pair_table, summary, residual_summary, hierarchy_summary


def group_size_bin(n: int) -> str:
    if n < 2:
        return "01"
    if n == 2:
        return "02"
    if 3 <= n <= 5:
        return "03-05"
    if 6 <= n <= 10:
        return "06-10"
    if 11 <= n <= 25:
        return "11-25"
    if 26 <= n <= 50:
        return "26-50"
    return "51+"


def repeat_bin(x: float) -> str:
    if x <= 1:
        return "1"
    if x <= 2:
        return "2"
    if x <= 5:
        return "3-5"
    if x <= 10:
        return "6-10"
    return "11+"


def matched_random_similarity(
    all_indices: np.ndarray,
    exclude_peptide: str,
    df_pos: pd.DataFrame,
    X: np.ndarray,
    n: int,
    repeats: int,
    rng: np.random.Generator,
) -> float:
    candidates = all_indices[df_pos.loc[all_indices, "peptide"].to_numpy() != exclude_peptide]
    if len(candidates) < 2:
        return float("nan")
    vals = []
    draw_n = min(n, len(candidates))
    for _ in range(repeats):
        sample_idx = rng.choice(candidates, size=draw_n, replace=False)
        s = pairwise_cosine_similarities(X[sample_idx], max_pairs=20000, rng=rng)
        if len(s):
            vals.append(float(np.mean(s)))
    return float(np.mean(vals)) if vals else float("nan")


def crossreactivity_analysis(
    df: pd.DataFrame,
    arrays: Dict[str, np.ndarray],
    rng: np.random.Generator,
    max_pairs: int,
    random_repeats: int,
    deduplicate_tcrs: bool,
) -> pd.DataFrame:
    df_pos = df[df["label"] == 1].copy()
    if deduplicate_tcrs:
        df_pos = df_pos.sort_values("pair_id").drop_duplicates("tcr_hash", keep="first").copy()
    df_pos = df_pos.set_index("row_index", drop=False)
    all_indices = df_pos.index.to_numpy()

    rows = []
    for pep, grp in df_pos.groupby("peptide", sort=True):
        idx = grp.index.to_numpy()
        n = len(idx)
        if n < 2:
            continue

        model_s = pairwise_cosine_similarities(arrays["model_T_norm"][idx], max_pairs=max_pairs, rng=rng)
        raw_s = pairwise_cosine_similarities(arrays["raw_T_norm"][idx], max_pairs=max_pairs, rng=rng)

        model_rand = matched_random_similarity(all_indices, pep, df_pos, arrays["model_T_norm"], n, random_repeats, rng)
        raw_rand = matched_random_similarity(all_indices, pep, df_pos, arrays["raw_T_norm"], n, random_repeats, rng)

        repeat_counts = grp["tcr_seen_count_in_split"].to_numpy()
        rows.append({
            "peptide": pep,
            "n_positive_rows_used": int(n),
            "n_unique_tcr_hashes": int(grp["tcr_hash"].nunique()),
            "deduplicated_tcrs": bool(deduplicate_tcrs),
            "group_size_bin": group_size_bin(n),
            "mean_tcr_seen_count": float(np.mean(repeat_counts)),
            "max_tcr_seen_count": int(np.max(repeat_counts)),
            "mean_repeat_bin": repeat_bin(float(np.mean(repeat_counts))),
            "model_within_mean_cossim": float(np.mean(model_s)),
            "model_within_median_cossim": float(np.median(model_s)),
            "model_within_p90_cossim": float(np.quantile(model_s, 0.90)),
            "model_within_max_cossim": float(np.max(model_s)),
            "raw_within_mean_cossim": float(np.mean(raw_s)),
            "raw_within_median_cossim": float(np.median(raw_s)),
            "raw_within_p90_cossim": float(np.quantile(raw_s, 0.90)),
            "raw_within_max_cossim": float(np.max(raw_s)),
            "model_matched_random_mean_cossim": model_rand,
            "raw_matched_random_mean_cossim": raw_rand,
            "model_within_minus_random_cossim": float(np.mean(model_s) - model_rand) if not math.isnan(model_rand) else float("nan"),
            "raw_within_minus_random_cossim": float(np.mean(raw_s) - raw_rand) if not math.isnan(raw_rand) else float("nan"),
            "delta_model_minus_raw_within_cossim": float(np.mean(model_s) - np.mean(raw_s)),
            "model_more_similar_than_raw": bool(np.mean(model_s) > np.mean(raw_s)),
            "model_more_similar_than_random": bool((not math.isnan(model_rand)) and np.mean(model_s) > model_rand),
        })
    return pd.DataFrame(rows)


def collapse_analysis(df: pd.DataFrame, arrays: Dict[str, np.ndarray], rng: np.random.Generator, deduplicate_tcrs: bool) -> pd.DataFrame:
    df_pos = df[df["label"] == 1].copy()
    if deduplicate_tcrs:
        df_pos = df_pos.sort_values("pair_id").drop_duplicates("tcr_hash", keep="first").copy()
    df_pos = df_pos.set_index("row_index", drop=False)

    rows = []
    for pep, grp in df_pos.groupby("peptide", sort=True):
        idx = grp.index.to_numpy()
        n = len(idx)
        if n < 2:
            continue

        z = arrays["zT"][idx]
        z_norm = arrays["model_T_norm"][idx]
        raw_norm = arrays["raw_T_norm"][idx]

        model_s = pairwise_cosine_similarities(z_norm, max_pairs=50000, rng=rng)
        raw_s = pairwise_cosine_similarities(raw_norm, max_pairs=50000, rng=rng)
        centroid = z.mean(axis=0, keepdims=True)
        l2_to_centroid = np.linalg.norm(z - centroid, axis=1)
        coord_std_mean = float(z.std(axis=0).mean())
        coord_std_min = float(z.std(axis=0).min())
        angular_concentration = float(np.linalg.norm(z_norm.mean(axis=0)))
        raw_angular_concentration = float(np.linalg.norm(raw_norm.mean(axis=0)))

        rows.append({
            "peptide": pep,
            "deduplicated_tcrs": bool(deduplicate_tcrs),
            "n_positive_rows_used": int(n),
            "n_unique_tcr_hashes": int(grp["tcr_hash"].nunique()),
            "unique_fraction": float(grp["tcr_hash"].nunique() / n),
            "group_size_bin": group_size_bin(n),
            "mean_tcr_seen_count": float(grp["tcr_seen_count_in_split"].mean()),
            "model_mean_pairwise_cossim": float(np.mean(model_s)),
            "model_median_pairwise_cossim": float(np.median(model_s)),
            "model_max_pairwise_cossim": float(np.max(model_s)),
            "model_p99_pairwise_cossim": float(np.quantile(model_s, 0.99)),
            "raw_mean_pairwise_cossim": float(np.mean(raw_s)),
            "raw_median_pairwise_cossim": float(np.median(raw_s)),
            "raw_max_pairwise_cossim": float(np.max(raw_s)),
            "model_angular_concentration": angular_concentration,
            "raw_angular_concentration": raw_angular_concentration,
            "zT_mean_l2_to_centroid": float(np.mean(l2_to_centroid)),
            "zT_median_l2_to_centroid": float(np.median(l2_to_centroid)),
            "zT_max_l2_to_centroid": float(np.max(l2_to_centroid)),
            "zT_coord_std_mean": coord_std_mean,
            "zT_coord_std_min": coord_std_min,
            "collapse_flag_mean_cossim_gt_0_999": bool(np.mean(model_s) > 0.999),
            "collapse_flag_mean_cossim_gt_0_99": bool(np.mean(model_s) > 0.99),
            "near_duplicate_flag_max_cossim_gt_0_99999": bool(np.max(model_s) > 0.99999),
        })
    return pd.DataFrame(rows)


def partial_auc01_table(df: pd.DataFrame, score_col: str, peptide_col: str = "peptide") -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    for pep, grp in df.groupby(peptide_col, sort=True):
        y = grp["label"].to_numpy().astype(int)
        s = grp[score_col].to_numpy().astype(float)
        valid = len(np.unique(y)) == 2
        auc01 = float(roc_auc_score(y, s, max_fpr=0.1)) if valid else float("nan")
        full_auc = float(roc_auc_score(y, s)) if valid else float("nan")
        rows.append({
            "peptide": pep,
            "score_col": score_col,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "valid": bool(valid),
            "auc0_1_mcclish_standardised": auc01,
            "full_auroc": full_auc,
        })
    table = pd.DataFrame(rows)
    valid_table = table[table["valid"]].copy()
    summary = {
        "score_col": score_col,
        "macro_auc0_1_mcclish_standardised": float(valid_table["auc0_1_mcclish_standardised"].mean()) if len(valid_table) else float("nan"),
        "weighted_auc0_1_mcclish_standardised": float(np.average(valid_table["auc0_1_mcclish_standardised"], weights=valid_table["n"])) if len(valid_table) else float("nan"),
        "macro_full_auroc": float(valid_table["full_auroc"].mean()) if len(valid_table) else float("nan"),
        "n_peptides_total": int(len(table)),
        "n_peptides_valid": int(len(valid_table)),
    }
    return table, summary


def summarise_cross_table(table: pd.DataFrame) -> pd.DataFrame:
    if len(table) == 0:
        return pd.DataFrame()
    agg = table.groupby(["deduplicated_tcrs", "group_size_bin"], dropna=False).agg(
        n_peptides=("peptide", "count"),
        mean_n_positive_rows=("n_positive_rows_used", "mean"),
        mean_unique_tcrs=("n_unique_tcr_hashes", "mean"),
        mean_repeat_count=("mean_tcr_seen_count", "mean"),
        model_within_mean_cossim=("model_within_mean_cossim", "mean"),
        raw_within_mean_cossim=("raw_within_mean_cossim", "mean"),
        model_random_mean_cossim=("model_matched_random_mean_cossim", "mean"),
        raw_random_mean_cossim=("raw_matched_random_mean_cossim", "mean"),
        mean_delta_model_minus_raw_cossim=("delta_model_minus_raw_within_cossim", "mean"),
        frac_model_more_similar_than_raw=("model_more_similar_than_raw", "mean"),
        frac_model_more_similar_than_random=("model_more_similar_than_random", "mean"),
    ).reset_index()
    return agg


def plot_cross_scatter(table: pd.DataFrame, out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 6))
    if len(table) == 0:
        plt.text(0.5, 0.5, "No valid peptide groups", ha="center", va="center")
        plt.axis("off")
    else:
        x = table["raw_within_mean_cossim"].to_numpy()
        y = table["model_within_mean_cossim"].to_numpy()
        plt.scatter(x, y, alpha=0.75)
        lo = float(np.nanmin([x.min(), y.min()]))
        hi = float(np.nanmax([x.max(), y.max()]))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlabel("Raw ESM same-peptide TCR cosine similarity")
        plt.ylabel("Model same-peptide TCR cosine similarity")
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to trained *__best.pt checkpoint. If omitted, latest checkpoint in --checkpoint-dir is used.")
    parser.add_argument("--checkpoint-dir", default="/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple")
    parser.add_argument("--immrep-shard-dir", default="/home/natasha/multimodal_model/models/embeddings/immrep_test_set/test")
    parser.add_argument("--immrep-csv", default="/home/natasha/multimodal_model/data/test/immrep_test_set_pair_id.csv", help="CSV with pair_id, Peptide and preferably TCR_full for sequence-similarity controls.")
    parser.add_argument("--out-dir", default="/home/natasha/multimodal_model/models/checkpoints/hpo_training/plain_vicreg_tulip_simple/posthoc_immrep_analysis")
    parser.add_argument("--fig-dir", default="/home/natasha/multimodal_model/models/figures/hpo_training/plain_vicreg_tulip_simple/posthoc_immrep_analysis")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-pairs", type=int, default=50000)
    parser.add_argument("--random-repeats", type=int, default=200)
    args = parser.parse_args()

    rng = set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(Path(args.checkpoint_dir))
    shard_dir = Path(args.immrep_shard_dir)
    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print("============================================================", flush=True)
    print("Plain VICReg post-hoc cross-reactivity and IMMREP analysis", flush=True)
    print(f"Script version: {SCRIPT_VERSION}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)
    print(f"IMMREP shard dir: {shard_dir}", flush=True)
    print(f"IMMREP CSV: {args.immrep_csv}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print("============================================================", flush=True)

    ckpt, cfg_dict, tcr, pmhc, _ = load_checkpoint_and_models(checkpoint_path, shard_dir, device)
    R_PH = float(cfg_dict.get("R_PH", 0.7))
    meta_df = load_pair_metadata(args.immrep_csv)
    pair_to_peptide, pair_to_tcr_sequence = metadata_maps(meta_df)

    pred_df, arrays = collect_split(
        shard_dir=shard_dir,
        tcr=tcr,
        pmhc=pmhc,
        device=device,
        pair_to_peptide=pair_to_peptide,
        pair_to_tcr_sequence=pair_to_tcr_sequence,
        R_PH=R_PH,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    pred_path = out_dir / "immrep_predictions_with_model_and_raw_esm_scores.csv"
    pred_df.to_csv(pred_path, index=False)

    # IMMREP Macro AUC0.1, primary MSE score plus cosine continuity scores.
    score_cols = ["model_score_mse", "raw_esm_score_mse", "model_score_cos", "raw_esm_score_cos"]
    metric_summaries = []
    for sc in score_cols:
        table, summary = partial_auc01_table(pred_df, sc)
        table.to_csv(out_dir / f"immrep_per_peptide_auc0_1__{sc}.csv", index=False)
        metric_summaries.append(summary)
    metrics_df = pd.DataFrame(metric_summaries)
    metrics_df.to_csv(out_dir / "immrep_macro_auc0_1_summary.csv", index=False)

    # Cross-reactivity and collapse analyses, with and without TCR deduplication.
    cross_tables = []
    collapse_tables = []
    for dedup in [False, True]:
        cross = crossreactivity_analysis(pred_df, arrays, rng, args.max_pairs, args.random_repeats, deduplicate_tcrs=dedup)
        collapse = collapse_analysis(pred_df, arrays, rng, deduplicate_tcrs=dedup)
        tag = "dedup_tcr" if dedup else "all_rows"
        cross.to_csv(out_dir / f"crossreactivity_same_peptide_vs_raw_esm__{tag}.csv", index=False)
        collapse.to_csv(out_dir / f"collapse_same_peptide_tcrs__{tag}.csv", index=False)
        plot_cross_scatter(cross, fig_dir / f"crossreactivity_scatter__{tag}.png", f"Same-peptide TCR cosine similarity: model vs raw ESM ({tag})")
        cross_tables.append(cross)
        collapse_tables.append(collapse)

    # Sequence-similarity controlled cross-reactivity analysis.
    # This tests whether same-peptide TCRs are unusually close after controlling
    # for TCR sequence homology, rather than just being close because they are
    # near-neighbours in sequence space.
    sequence_pair_tables = []
    sequence_summary_tables = []
    sequence_residual_tables = []
    sequence_hierarchy_tables = []
    if "tcr_sequence" in pred_df.columns and pred_df["tcr_sequence"].replace("", np.nan).notna().any():
        for dedup in [False, True]:
            pair_table, seq_summary, residual_summary, hierarchy_summary = sequence_controlled_crossreactivity(
                pred_df, arrays, rng, deduplicate_tcrs=dedup, max_random_pairs=25000
            )
            tag = "dedup_tcr" if dedup else "all_rows"
            if len(pair_table):
                pair_table.to_csv(out_dir / f"sequence_control_pairwise_tcr_pairs__{tag}.csv", index=False)
                sequence_pair_tables.append(pair_table)
            if len(seq_summary):
                seq_summary.to_csv(out_dir / f"sequence_control_summary_by_edit_distance__{tag}.csv", index=False)
                sequence_summary_tables.append(seq_summary)
            if len(residual_summary):
                residual_summary.to_csv(out_dir / f"sequence_control_residual_summary__{tag}.csv", index=False)
                sequence_residual_tables.append(residual_summary)
            if len(hierarchy_summary):
                hierarchy_summary.to_csv(out_dir / f"sequence_control_metric_hierarchy__{tag}.csv", index=False)
                sequence_hierarchy_tables.append(hierarchy_summary)

    sequence_summary_all = pd.concat(sequence_summary_tables, ignore_index=True) if sequence_summary_tables else pd.DataFrame()
    sequence_residual_all = pd.concat(sequence_residual_tables, ignore_index=True) if sequence_residual_tables else pd.DataFrame()
    sequence_hierarchy_all = pd.concat(sequence_hierarchy_tables, ignore_index=True) if sequence_hierarchy_tables else pd.DataFrame()
    sequence_summary_all.to_csv(out_dir / "sequence_control_summary_by_edit_distance.csv", index=False)
    sequence_residual_all.to_csv(out_dir / "sequence_control_residual_summary.csv", index=False)
    sequence_hierarchy_all.to_csv(out_dir / "sequence_control_metric_hierarchy.csv", index=False)

    cross_all = pd.concat(cross_tables, ignore_index=True) if cross_tables else pd.DataFrame()
    collapse_all = pd.concat(collapse_tables, ignore_index=True) if collapse_tables else pd.DataFrame()
    cross_summary = summarise_cross_table(cross_all)
    cross_summary.to_csv(out_dir / "crossreactivity_group_size_summary.csv", index=False)

    collapse_summary = pd.DataFrame({
        "deduplicated_tcrs": [False, True],
        "n_peptide_groups": [int((collapse_all["deduplicated_tcrs"] == False).sum()) if len(collapse_all) else 0,
                             int((collapse_all["deduplicated_tcrs"] == True).sum()) if len(collapse_all) else 0],
        "n_groups_mean_cossim_gt_0_999": [int(((collapse_all["deduplicated_tcrs"] == False) & (collapse_all["collapse_flag_mean_cossim_gt_0_999"])).sum()) if len(collapse_all) else 0,
                                          int(((collapse_all["deduplicated_tcrs"] == True) & (collapse_all["collapse_flag_mean_cossim_gt_0_999"])).sum()) if len(collapse_all) else 0],
        "n_groups_mean_cossim_gt_0_99": [int(((collapse_all["deduplicated_tcrs"] == False) & (collapse_all["collapse_flag_mean_cossim_gt_0_99"])).sum()) if len(collapse_all) else 0,
                                         int(((collapse_all["deduplicated_tcrs"] == True) & (collapse_all["collapse_flag_mean_cossim_gt_0_99"])).sum()) if len(collapse_all) else 0],
    })
    collapse_summary.to_csv(out_dir / "collapse_summary.csv", index=False)

    run_summary = {
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint_path),
        "immrep_shard_dir": str(shard_dir),
        "immrep_csv": args.immrep_csv,
        "R_PH": R_PH,
        "n_rows": int(len(pred_df)),
        "n_positive": int(pred_df["label"].sum()),
        "n_negative": int((pred_df["label"] == 0).sum()),
        "n_peptides": int(pred_df["peptide"].nunique()),
        "n_unique_tcr_hashes": int(pred_df["tcr_hash"].nunique()),
        "metric_summary": metric_summaries,
        "paths": {
            "predictions": str(pred_path),
            "macro_auc0_1_summary": str(out_dir / "immrep_macro_auc0_1_summary.csv"),
            "crossreactivity_group_size_summary": str(out_dir / "crossreactivity_group_size_summary.csv"),
            "collapse_summary": str(out_dir / "collapse_summary.csv"),
            "sequence_control_summary": str(out_dir / "sequence_control_summary_by_edit_distance.csv"),
            "sequence_control_residual_summary": str(out_dir / "sequence_control_residual_summary.csv"),
            "out_dir": str(out_dir),
            "fig_dir": str(fig_dir),
        },
    }
    with open(out_dir / "posthoc_immrep_analysis_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2)

    print("============================================================", flush=True)
    print("Done.", flush=True)
    print(f"Predictions: {pred_path}", flush=True)
    print(f"Macro AUC0.1 summary: {out_dir / 'immrep_macro_auc0_1_summary.csv'}", flush=True)
    print(f"Cross-reactivity summary: {out_dir / 'crossreactivity_group_size_summary.csv'}", flush=True)
    print(f"Collapse summary: {out_dir / 'collapse_summary.csv'}", flush=True)
    print(f"Sequence-control summary: {out_dir / 'sequence_control_summary_by_edit_distance.csv'}", flush=True)
    print(f"Sequence-control residuals: {out_dir / 'sequence_control_residual_summary.csv'}", flush=True)
    print("Metric summary:", flush=True)
    print(metrics_df.to_string(index=False), flush=True)
    print("============================================================", flush=True)


if __name__ == "__main__":
    main()
