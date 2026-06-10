#!/usr/bin/env python3
"""
Diagnose whether compressed Boltz Z* latents are biologically meaningful and sequence-learnable.

This is a diagnostic script, not a training script.

It tests two questions:
  1. Meaningfulness: do Z* latents show non-trivial variation and biological structure
     such as same-peptide / same-HLA neighbourhood enrichment?
  2. Sequence learnability: can sequence-only ESM features predict compressed Z* latents?

Expected location on server:
  /home/natasha/multimodal_model/scripts/train/hpo_training/diagnose_zstar_signal_learnability.py

Run from that folder:
  conda activate tcr-multimodal
  cd /home/natasha/multimodal_model/scripts/train/hpo_training
  python diagnose_zstar_signal_learnability.py 2>&1 | tee diagnose_zstar_signal_learnability.log

Outputs are written to:
  /home/natasha/multimodal_model/models/checkpoints/zstar_signal_diagnostics
"""

import os
import sys
import json
import math
import time
import random
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection
from sklearn.metrics.pairwise import cosine_distances

try:
    from scipy.stats import mannwhitneyu, spearmanr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# Default paths
# ============================================================
PROJECT = Path("/home/natasha/multimodal_model")
EMBED_ROOT = PROJECT / "models/embeddings/no_boltz_train_clean_immrep_A"
CHECKPOINTS_DIR = PROJECT / "models/checkpoints"

TRAIN_CSV = PROJECT / "data/train/train_df_clean.csv"
VAL_CSV   = PROJECT / "data/val/val_df_clean_pos_neg_A.csv"
TEST_CSV  = PROJECT / "data/test/test_df_clean_pos_neg_A.csv"

BOLTZ_LATENT_DIR = PROJECT / "models/embeddings/boltz_signal_bottleneck_latents"
ZSTAR_D = 128
ZSTAR_RANK = 8

DEFAULT_OUTDIR = CHECKPOINTS_DIR / "zstar_signal_diagnostics"
DEFAULT_LOG_FILE = "diagnose_zstar_signal_learnability.log"


# ============================================================
# Logging
# ============================================================
def setup_logging(log_file: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w")],
        force=True,
    )
    return logging.getLogger("zstar_diag")


# ============================================================
# Dataset and latent store utilities
# ============================================================
class ShardedBatchTripletDataset(Dataset):
    """Matches the shard loader style used in the existing training scripts."""
    def __init__(self, shards_dir: Path):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shards_dir}")

        self.index: List[Tuple[Path, int]] = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for j in range(len(shard)):
                self.index.append((sp, j))

        self._cache_path: Optional[Path] = None
        self._cache_data: Any = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sp, j = self.index[idx]
        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp
        return self._cache_data[j]


class OperatorLatentStore:
    """pair_id -> compressed Boltz Z* operator latent."""
    def __init__(self, latent_dir: Path, split_name: str):
        self.latent_dir = Path(latent_dir)
        self.split_name = split_name
        latent_path = self.latent_dir / f"{split_name}_latents.npz"
        meta_path = self.latent_dir / f"{split_name}_metadata.csv"
        if not latent_path.exists():
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        npz = np.load(latent_path)
        if "latent" in npz.files:
            key = "latent"
        elif "latents" in npz.files:
            key = "latents"
        else:
            # Fall back to the first array, but record it.
            key = npz.files[0]
        self.latents = np.asarray(npz[key], dtype=np.float32)
        self.meta = pd.read_csv(meta_path)

        if "latent_row" not in self.meta.columns:
            self.meta["latent_row"] = np.arange(len(self.meta))
        if "pair_id" not in self.meta.columns:
            raise ValueError(f"No pair_id column in {meta_path}")

        self.pid_to_row: Dict[str, int] = {}
        for _, row in self.meta.iterrows():
            self.pid_to_row[str(row["pair_id"])] = int(row["latent_row"])

        self.key = key

    def has(self, pair_id: Any) -> bool:
        return str(pair_id) in self.pid_to_row

    def get(self, pair_id: Any) -> Optional[np.ndarray]:
        row = self.pid_to_row.get(str(pair_id))
        if row is None:
            return None
        return self.latents[row]

    def __len__(self) -> int:
        return len(self.pid_to_row)


# ============================================================
# Sequence / metadata utilities
# ============================================================
def normalise_pair_id(pid: Any) -> str:
    return str(pid)


def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def load_label_table(csv_path: Path, split: str, log: logging.Logger) -> pd.DataFrame:
    if not Path(csv_path).exists():
        log.warning(f"{split}: CSV not found: {csv_path}")
        return pd.DataFrame(columns=["pair_id"])
    df = pd.read_csv(csv_path)
    if "pair_id" not in df.columns:
        log.warning(f"{split}: CSV has no pair_id column: {csv_path}")
        return pd.DataFrame(columns=["pair_id"])

    pep_col = detect_column(df, ["Peptide", "peptide", "pep", "epitope", "peptide_seq", "pep_seq"])
    hla_col = detect_column(df, ["HLA", "hla", "hla_seq", "hla_sequence", "MHC", "mhc", "mhc_seq", "mhc_sequence"])
    bind_col = detect_column(df, ["binding_flag", "binder", "label", "y"])

    keep = ["pair_id"]
    rename = {}
    if pep_col is not None:
        keep.append(pep_col); rename[pep_col] = "peptide"
    if hla_col is not None:
        keep.append(hla_col); rename[hla_col] = "hla"
    if bind_col is not None:
        keep.append(bind_col); rename[bind_col] = "binding_flag_csv"

    out = df[keep].rename(columns=rename).copy()
    out["pair_id"] = out["pair_id"].astype(str)
    log.info(f"{split}: loaded {len(out)} rows from {csv_path}; peptide_col={pep_col}; hla_col={hla_col}; binding_col={bind_col}")
    return out


def to_list_pair_ids(pair_ids: Any, B: int) -> List[str]:
    if isinstance(pair_ids, (list, tuple)):
        return [str(x) for x in pair_ids]
    if isinstance(pair_ids, np.ndarray):
        return [str(x) for x in pair_ids.tolist()]
    if torch.is_tensor(pair_ids):
        vals = pair_ids.detach().cpu().tolist()
        if isinstance(vals, list):
            return [str(x) for x in vals]
        return [str(vals)]
    return [str(pair_ids)] * B


def row_tensor(x: Any, i: int, B: int) -> torch.Tensor:
    """Return row i from a possibly batched tensor."""
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if x.ndim >= 1 and x.shape[0] == B:
        return x[i]
    return x


def mean_pool(emb: torch.Tensor, mask: torch.Tensor) -> np.ndarray:
    emb = emb.detach().cpu().float()
    mask = mask.detach().cpu().float()
    if emb.ndim != 2:
        raise ValueError(f"Expected emb (L,D), got shape {tuple(emb.shape)}")
    if mask.ndim != 1:
        mask = mask.view(-1)
    L = min(emb.shape[0], mask.shape[0])
    emb = emb[:L]
    mask = mask[:L].unsqueeze(-1)
    denom = float(mask.sum().item())
    if denom <= 0:
        return emb.mean(dim=0).numpy().astype(np.float32)
    return ((emb * mask).sum(dim=0) / max(denom, 1.0)).numpy().astype(np.float32)


def collect_split_arrays(
    split: str,
    embed_root: Path,
    latent_store: OperatorLatentStore,
    label_table: pd.DataFrame,
    max_pairs: int,
    seed: int,
    log: logging.Logger,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Collect matched rows with both ESM shard data and Z* latent.

    Returns:
      meta_df with pair_id, split, optional peptide/hla/label
      X_esm: mean-pooled ESM sequence features [T, P, H]
      Y_z: compressed Boltz operator latent
    """
    ds = ShardedBatchTripletDataset(embed_root / split)
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    if max_pairs and max_pairs > 0:
        # We still have to inspect shard batches, but randomising avoids only taking early shards.
        rng.shuffle(indices)

    label_lookup = label_table.set_index("pair_id") if len(label_table) else pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    X: List[np.ndarray] = []
    Y: List[np.ndarray] = []
    inspected_batches = 0
    inspected_rows = 0
    matched = 0

    for idx in indices:
        sample = ds[idx]
        inspected_batches += 1

        emb_T = sample["emb_T"]
        emb_P = sample["emb_P"]
        emb_H = sample["emb_H"]
        mask_T = sample["mask_T"]
        mask_P = sample["mask_P"]
        mask_H = sample["mask_H"]

        # These shard samples are usually mini-batches, e.g. B=8.
        if torch.is_tensor(emb_T) and emb_T.ndim == 3:
            B = int(emb_T.shape[0])
        else:
            B = 1
        pids = to_list_pair_ids(sample.get("pair_id", [f"unknown_{idx}"]), B)

        labels = sample.get("binding_flag", None)

        for i, pid in enumerate(pids):
            inspected_rows += 1
            z = latent_store.get(pid)
            if z is None:
                continue

            try:
                t_vec = mean_pool(row_tensor(emb_T, i, B), row_tensor(mask_T, i, B))
                p_vec = mean_pool(row_tensor(emb_P, i, B), row_tensor(mask_P, i, B))
                h_vec = mean_pool(row_tensor(emb_H, i, B), row_tensor(mask_H, i, B))
            except Exception as e:
                log.warning(f"{split}: failed ESM pooling for pair_id={pid}: {e}")
                continue

            x = np.concatenate([t_vec, p_vec, h_vec], axis=0).astype(np.float32)
            y = np.asarray(z, dtype=np.float32)

            row: Dict[str, Any] = {"pair_id": str(pid), "split": split}
            if labels is not None:
                try:
                    if torch.is_tensor(labels):
                        if labels.ndim > 0 and labels.shape[0] == B:
                            row["binding_flag"] = int(labels[i].item())
                        else:
                            row["binding_flag"] = int(labels.item())
                    elif isinstance(labels, (list, tuple, np.ndarray)):
                        row["binding_flag"] = int(labels[i])
                    else:
                        row["binding_flag"] = int(labels)
                except Exception:
                    pass

            if len(label_lookup) and str(pid) in label_lookup.index:
                lr = label_lookup.loc[str(pid)]
                if isinstance(lr, pd.DataFrame):
                    lr = lr.iloc[0]
                for col in ["peptide", "hla", "binding_flag_csv"]:
                    if col in lr.index:
                        row[col] = lr[col]

            rows.append(row)
            X.append(x)
            Y.append(y)
            matched += 1

            if max_pairs and max_pairs > 0 and matched >= max_pairs:
                break

        if inspected_batches % 250 == 0:
            log.info(f"{split}: inspected_batches={inspected_batches:,}; inspected_rows={inspected_rows:,}; matched={matched:,}")
        if max_pairs and max_pairs > 0 and matched >= max_pairs:
            break

    if not X:
        raise RuntimeError(f"{split}: no matched ESM+Z* rows found")

    meta = pd.DataFrame(rows)
    X_arr = np.vstack(X).astype(np.float32)
    Y_arr = np.vstack(Y).astype(np.float32)
    log.info(f"{split}: collected matched rows={len(meta):,}; X={X_arr.shape}; Y={Y_arr.shape}; latent_store={len(latent_store):,}")
    return meta, X_arr, Y_arr


# ============================================================
# Diagnostics
# ============================================================
def safe_effective_rank(singular_values: np.ndarray) -> float:
    s = np.asarray(singular_values, dtype=np.float64)
    s = s[s > 0]
    if len(s) == 0:
        return float("nan")
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(entropy))


def latent_variation_summary(Y: np.ndarray, outdir: Path, prefix: str, log: logging.Logger) -> pd.DataFrame:
    Y = np.asarray(Y, dtype=np.float32)
    n, d = Y.shape
    col_std = Y.std(axis=0)
    col_mean = Y.mean(axis=0)
    zeroish = float(np.mean(col_std < 1e-6))

    # Use SVD on a centred subsample if needed.
    Yc = Y - Y.mean(axis=0, keepdims=True)
    max_svd_n = min(n, 8000)
    if n > max_svd_n:
        rng = np.random.default_rng(123)
        idx = rng.choice(n, size=max_svd_n, replace=False)
        Ysvd = Yc[idx]
    else:
        Ysvd = Yc
    # Direct SVD is acceptable on <= 8000 x 6528 for diagnostics, but can still be heavy.
    # TruncatedSVD gives stable enough spectrum diagnostics at much lower cost.
    k = int(min(256, Ysvd.shape[0] - 1, Ysvd.shape[1] - 1))
    if k >= 2:
        svd = TruncatedSVD(n_components=k, random_state=123)
        svd.fit(Ysvd)
        evr = svd.explained_variance_ratio_
        singular = svd.singular_values_
        eff_rank = safe_effective_rank(singular)
        n90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1) if np.cumsum(evr)[-1] >= 0.90 else -1
        n95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1) if np.cumsum(evr)[-1] >= 0.95 else -1
        pd.DataFrame({
            "component": np.arange(1, len(evr) + 1),
            "explained_variance_ratio": evr,
            "cumulative_explained_variance_ratio": np.cumsum(evr),
            "singular_value": singular,
        }).to_csv(outdir / f"{prefix}_zstar_spectrum.csv", index=False)
    else:
        eff_rank = float("nan"); n90 = -1; n95 = -1

    rows = [{
        "split": prefix,
        "n": int(n),
        "latent_dim": int(d),
        "mean_abs_column_mean": float(np.mean(np.abs(col_mean))),
        "mean_column_std": float(np.mean(col_std)),
        "median_column_std": float(np.median(col_std)),
        "min_column_std": float(np.min(col_std)),
        "max_column_std": float(np.max(col_std)),
        "frac_near_constant_dims_std_lt_1e_minus_6": zeroish,
        "truncated_effective_rank": eff_rank,
        "n_components_90pct_variance_within_truncation": n90,
        "n_components_95pct_variance_within_truncation": n95,
    }]
    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{prefix}_zstar_variation_summary.csv", index=False)
    log.info(f"{prefix}: Z* variation summary: {rows[0]}")
    return df


def reduce_for_distance(Y: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    n, d = Y.shape
    k = int(min(n_components, n - 1, d - 1))
    if k < 2:
        return Y.astype(np.float32)
    pipe = make_pipeline(StandardScaler(with_mean=True, with_std=True), TruncatedSVD(n_components=k, random_state=seed))
    return pipe.fit_transform(Y).astype(np.float32)


def sample_same_group_pairs(groups: np.ndarray, max_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = []
    s = pd.Series(groups)
    for _, idxs in s.groupby(s).groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        if len(idxs) < 2:
            continue
        local_n = min(len(idxs) * 2, max(10, max_pairs // max(1, s.nunique())))
        for _ in range(local_n):
            a, b = rng.choice(idxs, size=2, replace=False)
            pairs.append((int(a), int(b)))
            if len(pairs) >= max_pairs:
                break
        if len(pairs) >= max_pairs:
            break
    return np.asarray(pairs, dtype=int) if pairs else np.empty((0, 2), dtype=int)


def sample_random_pairs(n: int, max_pairs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(max_pairs):
        a, b = rng.choice(n, size=2, replace=False)
        pairs.append((int(a), int(b)))
    return np.asarray(pairs, dtype=int)


def pair_cosine_distances(Z: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.asarray([], dtype=np.float32)
    A = Z[pairs[:, 0]]
    B = Z[pairs[:, 1]]
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return (1.0 - np.sum(A * B, axis=1)).astype(np.float32)


def biological_structure_tests(
    meta: pd.DataFrame,
    representations: Dict[str, np.ndarray],
    outdir: Path,
    prefix: str,
    max_pairs: int,
    seed: int,
    log: logging.Logger,
) -> pd.DataFrame:
    rows = []
    n = len(meta)
    random_pairs = sample_random_pairs(n, max_pairs=max_pairs, seed=seed + 17)

    for group_col in ["peptide", "hla"]:
        if group_col not in meta.columns:
            log.info(f"{prefix}: skipping group test for {group_col}; column unavailable")
            continue
        groups = meta[group_col].astype(str).replace({"nan": np.nan}).to_numpy()
        valid = pd.notna(groups)
        # Keep only valid rows for this grouping.
        sub_idx = np.where(valid)[0]
        if len(sub_idx) < 10:
            continue
        sub_meta = meta.iloc[sub_idx].reset_index(drop=True)
        sub_groups = sub_meta[group_col].astype(str).to_numpy()
        same_pairs_sub = sample_same_group_pairs(sub_groups, max_pairs=max_pairs, seed=seed + 31)
        random_pairs_sub = sample_random_pairs(len(sub_meta), max_pairs=max_pairs, seed=seed + 37)
        if len(same_pairs_sub) == 0:
            continue

        for rep_name, Z_all in representations.items():
            Z = Z_all[sub_idx]
            same_d = pair_cosine_distances(Z, same_pairs_sub)
            rand_d = pair_cosine_distances(Z, random_pairs_sub)
            if len(same_d) == 0 or len(rand_d) == 0:
                continue
            effect = float(np.mean(rand_d) - np.mean(same_d))
            # Positive effect means same-group examples are closer than random examples.
            if SCIPY_OK:
                try:
                    u = mannwhitneyu(same_d, rand_d, alternative="less")
                    pval = float(u.pvalue)
                except Exception:
                    pval = float("nan")
            else:
                pval = float("nan")
            rows.append({
                "split": prefix,
                "representation": rep_name,
                "group": group_col,
                "n_rows_with_group": int(len(sub_meta)),
                "n_unique_groups": int(pd.Series(sub_groups).nunique()),
                "n_same_pairs": int(len(same_d)),
                "n_random_pairs": int(len(rand_d)),
                "same_group_cosine_distance_mean": float(np.mean(same_d)),
                "same_group_cosine_distance_median": float(np.median(same_d)),
                "random_cosine_distance_mean": float(np.mean(rand_d)),
                "random_cosine_distance_median": float(np.median(rand_d)),
                "effect_random_minus_same_positive_means_same_group_closer": effect,
                "mannwhitney_p_same_less_than_random": pval,
            })

            # Plot distributions.
            plt.figure(figsize=(7, 5))
            plt.hist(rand_d, bins=60, alpha=0.55, density=True, label="random pairs")
            plt.hist(same_d, bins=60, alpha=0.55, density=True, label=f"same {group_col}")
            plt.xlabel("cosine distance")
            plt.ylabel("density")
            plt.title(f"{prefix}: {rep_name}, same {group_col} vs random")
            plt.legend()
            plt.tight_layout()
            fig_name = f"{prefix}_{rep_name}_same_{group_col}_vs_random_cosdist.png".replace("/", "_")
            plt.savefig(outdir / "figures" / fig_name, dpi=160)
            plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{prefix}_biological_structure_tests.csv", index=False)
    log.info(f"{prefix}: biological structure tests saved with {len(df)} rows")
    return df


def build_representations_for_comparison(X_esm: np.ndarray, Y_z: np.ndarray, seed: int, n_components: int) -> Dict[str, np.ndarray]:
    reps = {}
    reps["zstar_latent_reduced"] = reduce_for_distance(Y_z, n_components=n_components, seed=seed)
    reps["esm_meanpool_reduced"] = reduce_for_distance(X_esm, n_components=n_components, seed=seed + 1)
    return reps


def predictability_train_holdout(
    train_meta: pd.DataFrame,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    external: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
    outdir: Path,
    seed: int,
    x_components: int,
    y_components: int,
    ridge_alpha: float,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Test whether Z* latents are predictable from sequence-only ESM features.

    To keep this computationally manageable, both X and Y are projected to lower-dimensional
    spaces. X uses TruncatedSVD after scaling. Y uses TruncatedSVD/PCA-like compression.
    Ridge then predicts Y components from X components.
    """
    n = X_train.shape[0]
    xk = int(min(x_components, X_train.shape[1] - 1, n - 1))
    yk = int(min(y_components, Y_train.shape[1] - 1, n - 1))
    if xk < 2 or yk < 2:
        raise ValueError(f"Not enough rows/components for predictability: X={X_train.shape}, Y={Y_train.shape}")

    log.info(f"Predictability: fitting X reducer to {xk} components and Y reducer to {yk} components on train")
    x_reducer = make_pipeline(StandardScaler(with_mean=True, with_std=True), TruncatedSVD(n_components=xk, random_state=seed))
    y_reducer = make_pipeline(StandardScaler(with_mean=True, with_std=True), TruncatedSVD(n_components=yk, random_state=seed + 1))

    Xr = x_reducer.fit_transform(X_train).astype(np.float32)
    Yr = y_reducer.fit_transform(Y_train).astype(np.float32)
    y_svd = y_reducer.named_steps["truncatedsvd"]
    y_var_cum = np.cumsum(y_svd.explained_variance_ratio_)
    pd.DataFrame({
        "component": np.arange(1, len(y_svd.explained_variance_ratio_) + 1),
        "y_explained_variance_ratio": y_svd.explained_variance_ratio_,
        "y_cumulative_explained_variance_ratio": y_var_cum,
    }).to_csv(outdir / "sequence_learnability_target_zstar_svd_spectrum.csv", index=False)

    idx_train, idx_hold = train_test_split(np.arange(n), test_size=0.2, random_state=seed, shuffle=True)
    model = Ridge(alpha=ridge_alpha, random_state=seed)
    model.fit(Xr[idx_train], Yr[idx_train])

    rows = []

    def eval_block(name: str, meta: pd.DataFrame, X: np.ndarray, Y: np.ndarray, is_train_reduced: bool = False, idx: Optional[np.ndarray] = None):
        if is_train_reduced:
            X_eval = Xr[idx]
            Y_eval = Yr[idx]
            Y_orig = Y_train[idx]
        else:
            X_eval = x_reducer.transform(X).astype(np.float32)
            Y_eval = y_reducer.transform(Y).astype(np.float32)
            Y_orig = Y
        Y_pred = model.predict(X_eval).astype(np.float32)

        # Metrics in compressed target space.
        r2_uniform = float(r2_score(Y_eval, Y_pred, multioutput="uniform_average"))
        r2_varw = float(r2_score(Y_eval, Y_pred, multioutput="variance_weighted"))
        mse = float(np.mean((Y_eval - Y_pred) ** 2))

        # Cosine similarity in compressed target space.
        Yn = Y_eval / (np.linalg.norm(Y_eval, axis=1, keepdims=True) + 1e-8)
        Pn = Y_pred / (np.linalg.norm(Y_pred, axis=1, keepdims=True) + 1e-8)
        cos = np.sum(Yn * Pn, axis=1)
        mean_cos = float(np.mean(cos))
        median_cos = float(np.median(cos))

        # Retrieval: is the true Z* among nearest predicted-to-true targets?
        # Computed on compressed target space for efficiency.
        max_ret = min(3000, len(Y_eval))
        if len(Y_eval) > max_ret:
            rng = np.random.default_rng(seed + 99)
            ridx = rng.choice(len(Y_eval), size=max_ret, replace=False)
            Y_ret = Y_eval[ridx]
            P_ret = Y_pred[ridx]
        else:
            ridx = np.arange(len(Y_eval))
            Y_ret = Y_eval
            P_ret = Y_pred
        dist = cosine_distances(P_ret, Y_ret)
        ranks = np.argsort(dist, axis=1)
        true_rank = np.array([np.where(ranks[i] == i)[0][0] + 1 for i in range(len(ridx))])
        top1 = float(np.mean(true_rank <= 1))
        top5 = float(np.mean(true_rank <= 5))
        top10 = float(np.mean(true_rank <= 10))
        median_rank = float(np.median(true_rank))

        row = {
            "eval_set": name,
            "n": int(len(Y_eval)),
            "x_components": int(xk),
            "y_components": int(yk),
            "y_cumulative_variance_explained_by_components": float(y_var_cum[-1]),
            "ridge_alpha": float(ridge_alpha),
            "r2_uniform_average_y_components": r2_uniform,
            "r2_variance_weighted_y_components": r2_varw,
            "mse_y_components": mse,
            "mean_cosine_pred_vs_true_y_components": mean_cos,
            "median_cosine_pred_vs_true_y_components": median_cos,
            "retrieval_top1": top1,
            "retrieval_top5": top5,
            "retrieval_top10": top10,
            "retrieval_median_rank": median_rank,
        }
        rows.append(row)
        log.info(f"Predictability {name}: {row}")

        # Save per-row predictions for later inspection, but only compact diagnostics.
        pred_df = meta.copy().reset_index(drop=True)
        if idx is not None and is_train_reduced:
            pred_df = train_meta.iloc[idx].copy().reset_index(drop=True)
        pred_df["cosine_pred_vs_true_y_components"] = cos
        pred_df.to_csv(outdir / f"sequence_learnability_{name}_per_pair.csv", index=False)

        plt.figure(figsize=(7, 5))
        plt.hist(cos, bins=60, alpha=0.75, density=True)
        plt.xlabel("cosine(predicted Z* components, true Z* components)")
        plt.ylabel("density")
        plt.title(f"Sequence→Z* predictability: {name}")
        plt.tight_layout()
        plt.savefig(outdir / "figures" / f"sequence_learnability_{name}_cosine_hist.png", dpi=160)
        plt.close()

    eval_block("train_holdout", train_meta.iloc[idx_hold].reset_index(drop=True), X_train, Y_train, is_train_reduced=True, idx=idx_hold)

    # Refit on all train before external split evaluation.
    model.fit(Xr, Yr)
    for name, (meta, X, Y) in external.items():
        if len(meta) >= 5:
            eval_block(name, meta, X, Y, is_train_reduced=False, idx=None)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "sequence_learnability_summary.csv", index=False)
    return df


def split_shift_tests(train_Y: np.ndarray, split_arrays: Dict[str, np.ndarray], outdir: Path, log: logging.Logger) -> pd.DataFrame:
    rows = []
    train_mean = train_Y.mean(axis=0)
    train_std = train_Y.std(axis=0) + 1e-8
    for split, Y in split_arrays.items():
        if len(Y) == 0:
            continue
        mean_shift = (Y.mean(axis=0) - train_mean) / train_std
        rows.append({
            "split": split,
            "n": int(len(Y)),
            "mean_abs_standardised_mean_shift_vs_train": float(np.mean(np.abs(mean_shift))),
            "median_abs_standardised_mean_shift_vs_train": float(np.median(np.abs(mean_shift))),
            "max_abs_standardised_mean_shift_vs_train": float(np.max(np.abs(mean_shift))),
            "mean_column_std_ratio_vs_train": float(np.mean(Y.std(axis=0) / train_std)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "zstar_split_shift_vs_train.csv", index=False)
    log.info(f"Split shift rows: {len(df)}")
    return df


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Diagnose Boltz compressed Z* meaningfulness and sequence learnability")
    p.add_argument("--project", type=Path, default=PROJECT)
    p.add_argument("--embed-root", type=Path, default=EMBED_ROOT)
    p.add_argument("--boltz-latent-dir", type=Path, default=BOLTZ_LATENT_DIR)
    p.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    p.add_argument("--val-csv", type=Path, default=VAL_CSV)
    p.add_argument("--test-csv", type=Path, default=TEST_CSV)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--log-file", type=str, default=DEFAULT_LOG_FILE)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument("--max-train-pairs", type=int, default=12000, help="0 means use all matched train pairs")
    p.add_argument("--max-eval-pairs", type=int, default=4000, help="0 means use all matched val/test pairs")
    p.add_argument("--distance-components", type=int, default=128)
    p.add_argument("--distance-pairs", type=int, default=20000)
    p.add_argument("--x-components", type=int, default=256)
    p.add_argument("--y-components", type=int, default=128)
    p.add_argument("--ridge-alpha", type=float, default=10.0)
    p.add_argument("--skip-val-test", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "figures").mkdir(parents=True, exist_ok=True)
    log = setup_logging(args.log_file)

    t0 = time.time()
    log.info("Starting Z* signal diagnostics")
    log.info(f"Project: {args.project}")
    log.info(f"Embed root: {args.embed_root}")
    log.info(f"Boltz latent dir: {args.boltz_latent_dir}")
    log.info(f"Output dir: {args.outdir}")
    log.info(f"Log file: {Path(args.log_file).resolve()}")
    log.info(f"max_train_pairs={args.max_train_pairs}; max_eval_pairs={args.max_eval_pairs}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    label_tables = {
        "train": load_label_table(args.train_csv, "train", log),
        "val": load_label_table(args.val_csv, "val", log),
        "test": load_label_table(args.test_csv, "test", log),
    }

    stores = {}
    for split in ["train", "val", "test"]:
        try:
            stores[split] = OperatorLatentStore(args.boltz_latent_dir, split)
            log.info(f"{split}: latent store loaded; n={len(stores[split]):,}; shape={stores[split].latents.shape}; npz_key={stores[split].key}")
        except Exception as e:
            if split == "train":
                raise
            log.warning(f"{split}: could not load latent store: {e}")

    # Coverage summary from stores and CSVs.
    coverage_rows = []
    for split, store in stores.items():
        csv_n = len(label_tables.get(split, pd.DataFrame()))
        csv_pids = set(label_tables.get(split, pd.DataFrame()).get("pair_id", pd.Series([], dtype=str)).astype(str).tolist())
        store_pids = set(store.pid_to_row.keys())
        coverage_rows.append({
            "split": split,
            "zstar_latent_pairs": int(len(store_pids)),
            "csv_rows": int(csv_n),
            "zstar_pairs_in_csv": int(len(store_pids & csv_pids)) if csv_n else 0,
            "csv_rows_with_zstar": int(len(csv_pids & store_pids)) if csv_n else 0,
        })
    pd.DataFrame(coverage_rows).to_csv(args.outdir / "coverage_summary.csv", index=False)
    log.info(f"Coverage summary: {coverage_rows}")

    # Collect matched arrays.
    train_meta, X_train, Y_train = collect_split_arrays(
        "train", args.embed_root, stores["train"], label_tables["train"], args.max_train_pairs, args.seed, log
    )
    train_meta.to_csv(args.outdir / "train_matched_metadata.csv", index=False)

    split_data: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]] = {"train": (train_meta, X_train, Y_train)}
    if not args.skip_val_test:
        for split in ["val", "test"]:
            if split not in stores:
                continue
            try:
                meta, X, Y = collect_split_arrays(
                    split, args.embed_root, stores[split], label_tables[split], args.max_eval_pairs, args.seed + 10, log
                )
                meta.to_csv(args.outdir / f"{split}_matched_metadata.csv", index=False)
                split_data[split] = (meta, X, Y)
            except Exception as e:
                log.warning(f"{split}: failed to collect matched arrays: {e}")

    # Variation summaries.
    variation_tables = []
    for split, (_, _, Y) in split_data.items():
        variation_tables.append(latent_variation_summary(Y, args.outdir, split, log))
    pd.concat(variation_tables, ignore_index=True).to_csv(args.outdir / "zstar_variation_summary_all_splits.csv", index=False)

    # Split shift.
    split_shift_tests(Y_train, {k: v[2] for k, v in split_data.items() if k != "train"}, args.outdir, log)

    # Biological structure tests in Z* versus raw ESM mean-pool.
    all_structure = []
    for split, (meta, X, Y) in split_data.items():
        reps = build_representations_for_comparison(X, Y, seed=args.seed, n_components=args.distance_components)
        df = biological_structure_tests(
            meta, reps, args.outdir, split, max_pairs=args.distance_pairs, seed=args.seed, log=log
        )
        if len(df):
            all_structure.append(df)
    if all_structure:
        pd.concat(all_structure, ignore_index=True).to_csv(args.outdir / "biological_structure_tests_all_splits.csv", index=False)

    # Sequence learnability.
    external = {k: v for k, v in split_data.items() if k != "train"}
    predictability_train_holdout(
        train_meta=train_meta,
        X_train=X_train,
        Y_train=Y_train,
        external=external,
        outdir=args.outdir,
        seed=args.seed,
        x_components=args.x_components,
        y_components=args.y_components,
        ridge_alpha=args.ridge_alpha,
        log=log,
    )

    # Save run config.
    with open(args.outdir / "run_config.json", "w") as f:
        json.dump({k: str(v) for k, v in vars(args).items()}, f, indent=2)

    log.info(f"Done in {(time.time() - t0) / 60.0:.2f} minutes")
    log.info(f"Outputs saved to: {args.outdir}")


if __name__ == "__main__":
    main()
