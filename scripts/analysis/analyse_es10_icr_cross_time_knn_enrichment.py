#!/usr/bin/env python3
"""Preliminary ES10 longitudinal cross-time kNN enrichment in frozen ESMC+VICReg space.

Question
--------
Does the pMHC-aligned TCR representation contain longitudinal structure such that
non-identical TCRs from the same patient at different timepoints are enriched in
one another's local latent-space neighbourhoods?

This script does NOT train or fine-tune. It reuses the workshop raw-ESMC + VICReg
checkpoint and the same ESMC tokenisation / TCR-projection pathway used by the
workshop evaluation / shard-export code.

IMPORTANT sequence mapping
--------------------------
Workshop VICReg training embeds ``TCR_full`` = concatenated TCRα ∥ TCRβ variable
regions. ICR AIRR files do not provide paired α chains. Following the existing
ICR analysis notebook pathway, model inputs are ICR ``sequence_aa`` (full
productive TCRβ variable-region amino-acid sequence). Clonotype identity for
collapsing and neighbour exclusion is CDR3β ``junction_aa``. This is NOT a silent
substitution of CDR3 for ``TCR_full``.

Typical launch
--------------
    cd /home/natasha/multimodal_model
    conda activate tcr-multimodal
    python scripts/analysis/analyse_es10_icr_cross_time_knn_enrichment.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
WORKSHOP_DIR = REPO / "scripts" / "train" / "workshop"
sys.path.insert(0, str(WORKSHOP_DIR))

from esm_vicreg_common import (  # noqa: E402
    ESMProjectionHead,
    clean_seq,
    masked_mean_pool,
    set_seed,
)

AA_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
PRODUCTIVE_TRUE = {"T", "TRUE", "1", "YES", "Y", "1.0"}
ES10_TIMEPOINTS = ("T1", "T4", "T5", "T6", "T7")
K_VALUES = (5, 10, 20, 50)
PRIMARY_K = 20
SEED = 31
N_BOOT = 1000
N_PERM = 1000

DEFAULT_DATA_DIR = REPO / "data" / "ICR" / "data_2025_renamed"
DEFAULT_CKPT = (
    REPO
    / "models"
    / "checkpoints"
    / "workshop"
    / "esm_vicreg_raw_complete"
    / "seed_31"
    / "best.pt"
)
DEFAULT_OUT = REPO / "models" / "outputs" / "icr_preliminary" / "es10_esmc_vicreg"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_icr_filename(path: Path) -> Dict[str, str]:
    """Parse ES10_T1_Bo_C1.tsv.gz -> patient, timepoint, site, sample_id."""
    stem = path.name
    for suffix in (".tsv.gz", ".tsv", ".csv.gz", ".csv"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected ICR filename (need patient_timepoint_site_sample): {path.name}")
    return {
        "patient": parts[0],
        "timepoint": parts[1],
        "site": parts[2],
        "sample_id": "_".join(parts[3:]),
        "file_stem": stem,
    }


def coerce_productive(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.upper()
    return x.isin(PRODUCTIVE_TRUE)


def load_icr_repertoire(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all ICR TSVs; return (observations, summary, file_summary).

    Observations are unique (patient, timepoint, CDR3=junction_aa) after pooling
    files within each patient×timepoint. ``sequence_aa`` retained for embedding.
    """
    files = sorted(data_dir.glob("*.tsv.gz")) + sorted(data_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No ICR TSV files in {data_dir}")

    file_rows: List[Dict] = []
    buckets: Dict[Tuple[str, str], List[pd.DataFrame]] = {}

    for path in files:
        meta = parse_icr_filename(path)
        opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".tsv.gz") else open
        with opener(path, "rt") as fh:
            raw = pd.read_csv(fh, sep="\t", low_memory=False)

        required = ["junction_aa", "sequence_aa", "duplicate_count", "productive"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")

        df = raw.copy()
        df = df.loc[coerce_productive(df["productive"])].copy()
        df["junction_aa"] = df["junction_aa"].map(clean_seq)
        df["sequence_aa"] = df["sequence_aa"].map(clean_seq)
        df = df[df["junction_aa"].map(lambda s: bool(AA_RE.match(s)))].copy()
        df = df[df["sequence_aa"].map(lambda s: bool(AA_RE.match(s)))].copy()
        df["duplicate_count"] = pd.to_numeric(df["duplicate_count"], errors="coerce").fillna(1).astype(np.int64)

        key = (meta["patient"], meta["timepoint"])
        buckets.setdefault(key, []).append(df)
        file_rows.append(
            {
                **meta,
                "path": str(path),
                "n_productive_rows": int(len(df)),
                "n_unique_cdr3_in_file": int(df["junction_aa"].nunique()) if len(df) else 0,
                "total_duplicate_count_file": int(df["duplicate_count"].sum()) if len(df) else 0,
            }
        )

    obs_parts: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []
    for (patient, timepoint), parts in sorted(buckets.items()):
        d = pd.concat(parts, ignore_index=True)
        # Choose modal sequence_aa by summed count for each CDR3.
        pair_counts = (
            d.groupby(["junction_aa", "sequence_aa"], as_index=False)["duplicate_count"]
            .sum()
            .sort_values("duplicate_count", ascending=False)
        )
        best_seq = pair_counts.groupby("junction_aa", as_index=False).first()
        cdr3_counts = d.groupby("junction_aa")["duplicate_count"].sum()
        best_seq["duplicate_count"] = best_seq["junction_aa"].map(cdr3_counts).astype(np.int64)
        best_seq["patient"] = patient
        best_seq["timepoint"] = timepoint
        best_seq["cdr3"] = best_seq["junction_aa"]
        best_seq["cdr3_len"] = best_seq["cdr3"].str.len().astype(int)
        best_seq["n_files"] = len(parts)
        obs_parts.append(best_seq)

        summary_rows.append(
            {
                "patient": patient,
                "timepoint": timepoint,
                "number_of_files": len(parts),
                "n_unique_productive_tcrs": int(len(best_seq)),
                "total_duplicate_count": int(best_seq["duplicate_count"].sum()),
            }
        )

    observations = pd.concat(obs_parts, ignore_index=True)
    observations["obs_id"] = np.arange(len(observations), dtype=np.int64)
    summary = pd.DataFrame(summary_rows).sort_values(["patient", "timepoint"]).reset_index(drop=True)
    file_summary = pd.DataFrame(file_rows)
    return observations, summary, file_summary


def print_sequence_mapping() -> None:
    print("=" * 80)
    print("SEQUENCE REPRESENTATION MAPPING")
    print("=" * 80)
    print(
        "Workshop VICReg training / evaluation embeds column TCR_full =\n"
        "  concatenated full variable-region TCRα ∥ TCRβ amino-acid sequences."
    )
    print(
        "ICR AIRR files do NOT provide paired TCRα. Available fields include:\n"
        "  - junction_aa  : CDR3β amino-acid sequence (clonotype identity)\n"
        "  - sequence_aa  : full productive TCRβ variable-region AA sequence"
    )
    print(
        "MAPPING USED (explicit, not silent):\n"
        "  Model input  -> ICR sequence_aa  (TCRβ full; matches prior ICR notebook)\n"
        "  Clonotype ID -> ICR junction_aa  (CDR3β) for collapse / neighbour exclusion\n"
        "  Do NOT feed junction_aa alone into ESMC+VICReg; training never used CDR3-only."
    )
    print("=" * 80)


# ---------------------------------------------------------------------------
# Model load + embedding
# ---------------------------------------------------------------------------

def load_vicreg_tcr_head(checkpoint_path: Path, device: torch.device) -> Tuple[ESMProjectionHead, Dict]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required = ["tcr_state_dict", "config", "shapes"]
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"Checkpoint missing keys {missing}: {checkpoint_path}")

    cfg = ckpt["config"]
    shapes = ckpt["shapes"]
    tcr = ESMProjectionHead(
        int(shapes["D"]),
        int(cfg["rL"]),
        int(cfg["rD"]),
        int(cfg["d"]),
        int(shapes["L_T"]),
        float(cfg.get("dropout", 0.1)),
    )
    tcr.load_state_dict(ckpt["tcr_state_dict"])
    tcr.to(device)
    tcr.eval()

    # Confirm weights are trained (not freshly initialised).
    with torch.no_grad():
        w = tcr.B_c.float().cpu().numpy()
        if not np.isfinite(w).all() or float(np.abs(w).mean()) < 1e-8:
            raise RuntimeError("TCR projection weights look uninitialised / empty")

    meta = {
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": ckpt.get("best_epoch"),
        "best_val_metric": ckpt.get("best_val_esm_vicreg_peptide_weighted_auroc"),
        "shapes": {k: int(v) for k, v in shapes.items()},
        "config_subset": {
            "rL": int(cfg["rL"]),
            "rD": int(cfg["rD"]),
            "d": int(cfg["d"]),
            "dropout": float(cfg.get("dropout", 0.1)),
            "seed": cfg.get("seed"),
            "run_tag": cfg.get("run_tag"),
        },
        "has_pmhc_state_dict": "pmhc_state_dict" in ckpt,
    }
    return tcr, meta


class SeqDataset(Dataset):
    def __init__(self, sequences: Sequence[str]):
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def embed_sequences(
    sequences: Sequence[str],
    esmc,
    tcr_proj: ESMProjectionHead,
    device: torch.device,
    batch_size: int,
    L_max: int,
    eps_pool: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (raw_meanpool [N,D], z_T [N,d]) for unique AA strings.

    Tokenisation matches export_raw_esmc_embedding_shards.py:
    model.tokenizer(seqs, return_tensors='pt', padding=True).
    """
    tokenizer = esmc.tokenizer
    esmc.eval()
    tcr_proj.eval()

    raw_list: List[np.ndarray] = []
    z_list: List[np.ndarray] = []

    loader = DataLoader(
        SeqDataset(sequences),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda xs: list(xs),
    )

    n_done = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_seqs in loader:
            enc = tokenizer(batch_seqs, return_tensors="pt", padding=True)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device=device, dtype=torch.bool)

            if input_ids.size(1) > L_max:
                input_ids = input_ids[:, :L_max]
                attn = attn[:, :L_max]

            emb = esmc(sequence_tokens=input_ids).embeddings  # [B, L, D]
            if emb.size(1) > L_max:
                emb = emb[:, :L_max]
                attn = attn[:, :L_max]

            z = tcr_proj(emb, attn)
            raw = masked_mean_pool(emb, attn, eps_pool)

            raw_np = raw.detach().float().cpu().numpy()
            z_np = z.detach().float().cpu().numpy()
            if not np.isfinite(raw_np).all() or not np.isfinite(z_np).all():
                raise RuntimeError("Non-finite embeddings encountered")
            raw_list.append(raw_np)
            z_list.append(z_np)

            n_done += len(batch_seqs)
            if n_done % max(batch_size * 50, 1) < batch_size or n_done == len(sequences):
                rate = n_done / max(time.time() - t0, 1e-6)
                print(
                    f"  embedded {n_done}/{len(sequences)} ({100.0 * n_done / len(sequences):.1f}%) "
                    f"[{rate:.1f} seq/s]",
                    flush=True,
                )

            del input_ids, attn, emb, z, raw
            if device.type == "cuda" and (n_done // batch_size) % 40 == 0:
                torch.cuda.empty_cache()

    return np.concatenate(raw_list, axis=0), np.concatenate(z_list, axis=0)


def attach_embeddings(
    observations: pd.DataFrame,
    cache_dir: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
    force_reembed: bool,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    seq_path = cache_dir / "unique_sequence_aa.npy"
    raw_path = cache_dir / "raw_esmc_meanpool.npy"
    z_path = cache_dir / "zT_vicreg.npy"
    meta_path = cache_dir / "embed_cache_meta.json"

    unique_seqs = sorted(observations["sequence_aa"].unique().tolist())
    tcr_proj, ckpt_meta = load_vicreg_tcr_head(checkpoint_path, device)
    L_max = int(ckpt_meta["shapes"]["L_T"])
    D = int(ckpt_meta["shapes"]["D"])
    d = int(ckpt_meta["config_subset"]["d"])

    raw: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    use_cache = (
        not force_reembed
        and seq_path.exists()
        and raw_path.exists()
        and z_path.exists()
        and meta_path.exists()
    )
    if use_cache:
        cached_seqs = np.load(seq_path, allow_pickle=True).tolist()
        cache_meta = json.loads(meta_path.read_text())
        if cached_seqs == unique_seqs and cache_meta.get("checkpoint_path") == str(checkpoint_path):
            print(f"Loading cached embeddings from {cache_dir}", flush=True)
            raw = np.load(raw_path)
            z = np.load(z_path)
        else:
            print("Cache mismatch; re-embedding.", flush=True)
            use_cache = False

    if not use_cache:
        print("Loading frozen ESMC-300m ...", flush=True)
        from esm.models.esmc import ESMC

        esmc = ESMC.from_pretrained("esmc_300m").to(device).eval()
        print(
            f"Embedding {len(unique_seqs)} unique sequence_aa strings "
            f"(raw mean-pool dim={D}, VICReg z_T dim={d}, L_max={L_max})",
            flush=True,
        )
        raw, z = embed_sequences(
            unique_seqs,
            esmc=esmc,
            tcr_proj=tcr_proj,
            device=device,
            batch_size=batch_size,
            L_max=L_max,
        )
        np.save(seq_path, np.array(unique_seqs, dtype=object))
        np.save(raw_path, raw.astype(np.float32))
        np.save(z_path, z.astype(np.float32))
        cache_meta = {
            "checkpoint_path": str(checkpoint_path),
            "n_unique_sequences": len(unique_seqs),
            "raw_shape": list(raw.shape),
            "z_shape": list(z.shape),
            "L_max": L_max,
            "model_input_field": "sequence_aa",
            "clonotype_field": "junction_aa",
            "training_field": "TCR_full",
            **{k: ckpt_meta[k] for k in ("best_epoch", "best_val_metric", "shapes", "config_subset")},
        }
        meta_path.write_text(json.dumps(cache_meta, indent=2))
        del esmc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert raw is not None and z is not None
    seq_to_i = {s: i for i, s in enumerate(unique_seqs)}
    emb_idx = observations["sequence_aa"].map(seq_to_i).to_numpy(dtype=np.int64)
    if emb_idx.min() < 0:
        raise RuntimeError("Failed to map all observations to embedding cache")

    print(f"Raw ESMC embedding shape (unique): {raw.shape}", flush=True)
    print(f"VICReg latent shape (unique):      {z.shape}", flush=True)
    print(f"Number of unique sequences embedded: {len(unique_seqs)}", flush=True)
    print(f"Checkpoint used: {checkpoint_path}", flush=True)
    print(
        f"  best_epoch={ckpt_meta.get('best_epoch')}  "
        f"best_val_peptide_weighted_auroc={ckpt_meta.get('best_val_metric')}",
        flush=True,
    )
    print(f"Model eval mode: tcr_proj.training={tcr_proj.training}", flush=True)

    return observations.assign(emb_idx=emb_idx), raw.astype(np.float32), z.astype(np.float32), {
        **ckpt_meta,
        "n_unique_sequences": len(unique_seqs),
        "raw_shape": list(raw.shape),
        "z_shape": list(z.shape),
    }


# ---------------------------------------------------------------------------
# kNN enrichment
# ---------------------------------------------------------------------------

def squared_euclidean_chunked(
    query: np.ndarray,
    gallery: np.ndarray,
    chunk: int = 2048,
) -> np.ndarray:
    """Return [n_query, n_gallery] squared Euclidean distances."""
    q = query.astype(np.float32, copy=False)
    g = gallery.astype(np.float32, copy=False)
    q2 = np.sum(q * q, axis=1, keepdims=True)
    out = np.empty((q.shape[0], g.shape[0]), dtype=np.float32)
    for start in range(0, g.shape[0], chunk):
        end = min(start + chunk, g.shape[0])
        gc = g[start:end]
        g2 = np.sum(gc * gc, axis=1, keepdims=True).T
        out[:, start:end] = q2 + g2 - 2.0 * (q @ gc.T)
    return out


def precompute_knn_for_queries(
    query_idx: np.ndarray,
    emb: np.ndarray,
    timepoint: np.ndarray,
    cdr3: np.ndarray,
    max_k: int,
    dist_chunk: int = 4096,
) -> Dict[str, np.ndarray]:
    """For each query, find top-max_k eligible neighbours (diff timepoint, diff CDR3)."""
    n_q = len(query_idx)
    top_idx = np.full((n_q, max_k), -1, dtype=np.int64)
    top_dist = np.full((n_q, max_k), np.inf, dtype=np.float32)
    n_eligible = np.zeros(n_q, dtype=np.int64)

    all_idx = np.arange(len(timepoint), dtype=np.int64)
    tp_to_gallery = {tp: all_idx[timepoint != tp] for tp in np.unique(timepoint)}

    cdr3_to_idx: Dict[str, np.ndarray] = {}
    for i, c in enumerate(cdr3):
        cdr3_to_idx.setdefault(c, []).append(i)
    cdr3_to_idx = {c: np.asarray(v, dtype=np.int64) for c, v in cdr3_to_idx.items()}

    q_tp = timepoint[query_idx]
    unique_q_tps = list(dict.fromkeys(q_tp.tolist()))
    pos_in_query = {int(qi): j for j, qi in enumerate(query_idx.tolist())}

    for tp in unique_q_tps:
        q_local = np.array([qi for qi in query_idx if timepoint[qi] == tp], dtype=np.int64)
        gallery = tp_to_gallery[tp]
        if len(gallery) == 0 or len(q_local) == 0:
            for qi in q_local:
                n_eligible[pos_in_query[int(qi)]] = 0
            continue

        q_chunk = 64 if emb.shape[1] >= 512 else 256
        g_cdr3 = cdr3[gallery]
        for start in range(0, len(q_local), q_chunk):
            end = min(start + q_chunk, len(q_local))
            ql = q_local[start:end]
            dists = squared_euclidean_chunked(emb[ql], emb[gallery], chunk=dist_chunk)

            for row, qi in enumerate(ql):
                j = pos_in_query[int(qi)]
                c = cdr3[qi]
                elig_local = np.where(g_cdr3 != c)[0]
                n_eligible[j] = int(len(elig_local))
                if len(elig_local) == 0:
                    continue
                drow = dists[row, elig_local]
                take = min(max_k, len(elig_local))
                if take < len(elig_local):
                    part = np.argpartition(drow, take - 1)[:take]
                    order = part[np.argsort(drow[part])]
                else:
                    order = np.argsort(drow)[:take]
                chosen = gallery[elig_local[order]]
                top_idx[j, :take] = chosen
                top_dist[j, :take] = drow[order]

        print(f"  knn queries timepoint={tp}: {len(q_local)} queries, gallery={len(gallery)}", flush=True)

    same_cdr3_other: List[np.ndarray] = []
    for qi in query_idx:
        tp = timepoint[qi]
        idxs = cdr3_to_idx[cdr3[qi]]
        same_cdr3_other.append(idxs[timepoint[idxs] != tp])

    return {
        "top_idx": top_idx,
        "top_dist": top_dist,
        "n_eligible": n_eligible,
        "same_cdr3_other": same_cdr3_other,
        "query_idx": query_idx,
    }


def n_es10_eligible(
    query_pos: int,
    query_tp: str,
    patient: np.ndarray,
    timepoint: np.ndarray,
    same_cdr3_other: Sequence[np.ndarray],
    es10_code: str = "ES10",
) -> int:
    base = int(np.sum((timepoint != query_tp) & (patient == es10_code)))
    same = same_cdr3_other[query_pos]
    if len(same):
        base -= int(np.sum(patient[same] == es10_code))
    return base


def enrichment_per_query(
    knn: Dict[str, np.ndarray],
    patient: np.ndarray,
    timepoint: np.ndarray,
    cdr3: np.ndarray,
    k_values: Sequence[int],
    es10: str = "ES10",
) -> pd.DataFrame:
    q_idx = knn["query_idx"]
    rows = []
    for j, qi in enumerate(q_idx):
        q_tp = timepoint[qi]
        q_cdr3 = cdr3[qi]
        n_elig = int(knn["n_eligible"][j])
        n_pos = n_es10_eligible(j, q_tp, patient, timepoint, knn["same_cdr3_other"], es10)
        expected = (n_pos / n_elig) if n_elig > 0 else np.nan
        row = {
            "query_obs_id": int(qi),
            "query_timepoint": q_tp,
            "query_cdr3": q_cdr3,
            "n_eligible": n_elig,
            "n_es10_other_eligible": n_pos,
            "expected": expected,
        }
        for k in k_values:
            neigh = knn["top_idx"][j, :k]
            neigh = neigh[neigh >= 0]
            if len(neigh) == 0 or not np.isfinite(expected) or expected <= 0:
                row[f"observed_k{k}"] = np.nan
                row[f"enrichment_k{k}"] = np.nan
                row[f"n_success_k{k}"] = 0
                continue
            # Sanity: neighbours must differ in timepoint and CDR3
            assert np.all(timepoint[neigh] != q_tp)
            assert np.all(cdr3[neigh] != q_cdr3)
            success = (patient[neigh] == es10) & (timepoint[neigh] != q_tp) & (cdr3[neigh] != q_cdr3)
            # success already implies the last two; keep explicit for clarity
            obs = float(np.mean(success))
            row[f"observed_k{k}"] = obs
            row[f"enrichment_k{k}"] = obs / expected
            row[f"n_success_k{k}"] = int(success.sum())
            # Critical: never count exact CDR3 match
            assert not np.any(cdr3[neigh] == q_cdr3)
        rows.append(row)
    return pd.DataFrame(rows)


def macro_average_by_timepoint(
    per_query: pd.DataFrame,
    value_col: str,
    timepoints: Sequence[str] = ES10_TIMEPOINTS,
) -> Tuple[float, Dict[str, float]]:
    per_tp = {}
    for tp in timepoints:
        vals = per_query.loc[per_query["query_timepoint"] == tp, value_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        per_tp[tp] = float(np.mean(vals)) if len(vals) else np.nan
    macro = float(np.nanmean([per_tp[tp] for tp in timepoints]))
    return macro, per_tp


def bootstrap_macro_ci(
    per_query: pd.DataFrame,
    value_col: str,
    n_boot: int,
    seed: int,
    timepoints: Sequence[str] = ES10_TIMEPOINTS,
) -> Tuple[float, float, float, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    groups = {tp: per_query.loc[per_query["query_timepoint"] == tp].index.to_numpy() for tp in timepoints}
    macros = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        tp_means = []
        for tp in timepoints:
            idx = groups[tp]
            if len(idx) == 0:
                tp_means.append(np.nan)
                continue
            draw = rng.choice(idx, size=len(idx), replace=True)
            vals = per_query.loc[draw, value_col].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            tp_means.append(float(np.mean(vals)) if len(vals) else np.nan)
        macros[b] = float(np.nanmean(tp_means))
    point, per_tp = macro_average_by_timepoint(per_query, value_col, timepoints)
    lo, hi = np.nanpercentile(macros, [2.5, 97.5])
    return point, float(lo), float(hi), per_tp


def run_permutation_null(
    knn: Dict,
    patient: np.ndarray,
    timepoint: np.ndarray,
    cdr3_len: np.ndarray,
    per_query_template: pd.DataFrame,
    k_values: Sequence[int],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    """Shuffle patient labels within (timepoint, cdr3_len) strata."""
    rng = np.random.default_rng(seed)
    patient = patient.copy()
    strata = pd.DataFrame({"timepoint": timepoint, "cdr3_len": cdr3_len})
    stratum_ids = strata.groupby(["timepoint", "cdr3_len"], sort=False).ngroup().to_numpy()

    q_idx = knn["query_idx"]
    q_tps = timepoint[q_idx]

    # Observed macros for reference
    obs_macros = {}
    for k in k_values:
        obs_macros[k], _ = macro_average_by_timepoint(per_query_template, f"enrichment_k{k}")

    null_macros = {k: np.empty(n_perm, dtype=np.float64) for k in k_values}

    # Precompute index lists per stratum for shuffling
    stratum_members = {}
    for s in np.unique(stratum_ids):
        stratum_members[int(s)] = np.where(stratum_ids == s)[0]

    for p in range(n_perm):
        perm_patient = patient.copy()
        for members in stratum_members.values():
            if len(members) <= 1:
                continue
            perm_patient[members] = rng.permutation(perm_patient[members])

        # Per-timepoint base ES10 counts among tp!=T
        base_es10 = {tp: int(np.sum((timepoint != tp) & (perm_patient == "ES10"))) for tp in ES10_TIMEPOINTS}

        # Query enrichments
        tp_vals = {k: {tp: [] for tp in ES10_TIMEPOINTS} for k in k_values}
        for j, qi in enumerate(q_idx):
            q_tp = q_tps[j]
            n_elig = int(knn["n_eligible"][j])
            n_pos = base_es10[q_tp]
            same = knn["same_cdr3_other"][j]
            if len(same):
                n_pos -= int(np.sum(perm_patient[same] == "ES10"))
            expected = (n_pos / n_elig) if n_elig > 0 else np.nan
            if not np.isfinite(expected) or expected <= 0:
                continue
            for k in k_values:
                neigh = knn["top_idx"][j, :k]
                neigh = neigh[neigh >= 0]
                if len(neigh) == 0:
                    continue
                obs = float(np.mean(perm_patient[neigh] == "ES10"))
                tp_vals[k][q_tp].append(obs / expected)

        for k in k_values:
            means = [float(np.mean(tp_vals[k][tp])) if tp_vals[k][tp] else np.nan for tp in ES10_TIMEPOINTS]
            null_macros[k][p] = float(np.nanmean(means))

        if (p + 1) % 100 == 0 or p == 0:
            print(f"  permutation {p + 1}/{n_perm}", flush=True)

    rows = []
    for k in k_values:
        null = null_macros[k]
        obs = obs_macros[k]
        # two-sided p-value
        p_val = float((np.sum(np.abs(null - np.nanmean(null)) >= np.abs(obs - np.nanmean(null))) + 1) / (n_perm + 1))
        # also report one-sided enrichment p
        p_right = float((np.sum(null >= obs) + 1) / (n_perm + 1))
        rows.append(
            {
                "k": k,
                "observed_macro_enrichment": obs,
                "null_mean": float(np.nanmean(null)),
                "null_std": float(np.nanstd(null)),
                "null_p2.5": float(np.nanpercentile(null, 2.5)),
                "null_p97.5": float(np.nanpercentile(null, 97.5)),
                "permutation_pvalue_twosided": p_val,
                "permutation_pvalue_right_tail": p_right,
                "n_permutations": n_perm,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Descriptive + examples + figures
# ---------------------------------------------------------------------------

def descriptive_es10(obs: pd.DataFrame) -> pd.DataFrame:
    es = obs[obs["patient"] == "ES10"].copy()
    # unique CDR3 across all ES10
    by_cdr3 = es.groupby("cdr3")["timepoint"].nunique()
    total_unique = int(es["cdr3"].nunique())
    recurring = int((by_cdr3 >= 2).sum())
    rows = [
        {"metric": "total_unique_productive_cdr3b", "value": total_unique},
        {"metric": "n_appearing_ge_2_timepoints", "value": recurring},
        {
            "metric": "pct_recurring_ge_2_timepoints",
            "value": 100.0 * recurring / total_unique if total_unique else np.nan,
        },
    ]
    sets = {tp: set(es.loc[es["timepoint"] == tp, "cdr3"]) for tp in ES10_TIMEPOINTS}
    for a, b in [("T1", "T4"), ("T4", "T5"), ("T5", "T6"), ("T6", "T7")]:
        inter = len(sets[a] & sets[b])
        rows.append({"metric": f"exact_overlap_{a}_{b}", "value": inter})
        rows.append(
            {
                "metric": f"jaccard_{a}_{b}",
                "value": inter / len(sets[a] | sets[b]) if (sets[a] | sets[b]) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def example_neighbours(
    knn_vicreg: Dict,
    knn_raw: Dict,
    obs: pd.DataFrame,
    k: int = 10,
    n_examples: int = 25,
) -> pd.DataFrame:
    patient = obs["patient"].to_numpy()
    timepoint = obs["timepoint"].to_numpy()
    cdr3 = obs["cdr3"].to_numpy()
    q_idx = knn_vicreg["query_idx"]

    scores = []
    for j, qi in enumerate(q_idx):
        neigh = knn_vicreg["top_idx"][j, :k]
        neigh = neigh[neigh >= 0]
        n_succ = int(np.sum((patient[neigh] == "ES10") & (timepoint[neigh] != timepoint[qi]) & (cdr3[neigh] != cdr3[qi])))
        scores.append((n_succ, j, int(qi)))
    scores.sort(reverse=True)

    # Map obs_id -> row position in knn_raw query list
    raw_pos = {int(qi): j for j, qi in enumerate(knn_raw["query_idx"].tolist())}

    rows = []
    for n_succ, j, qi in scores[:n_examples]:
        if n_succ <= 0:
            continue
        neigh = knn_vicreg["top_idx"][j, :k]
        dists = knn_vicreg["top_dist"][j, :k]
        for ni, nd in zip(neigh, dists):
            if ni < 0:
                continue
            if patient[ni] != "ES10" or timepoint[ni] == timepoint[qi] or cdr3[ni] == cdr3[qi]:
                continue
            # raw distance for same pair
            jr = raw_pos[qi]
            raw_neigh = knn_raw["top_idx"][jr]
            raw_dist_map = {int(a): float(b) for a, b in zip(raw_neigh, knn_raw["top_dist"][jr]) if a >= 0}
            # If neighbour not in raw top-k, compute exact squared distance
            if int(ni) in raw_dist_map:
                raw_d = raw_dist_map[int(ni)]
            else:
                # fallback: will fill later if we pass emb — store nan and fix in caller if needed
                raw_d = np.nan
            qa = cdr3[qi]
            na = cdr3[ni]
            rows.append(
                {
                    "query_cdr3": qa,
                    "query_timepoint": timepoint[qi],
                    "neighbour_cdr3": na,
                    "neighbour_timepoint": timepoint[ni],
                    "vicreg_sq_euclidean": float(nd),
                    "raw_esmc_sq_euclidean": raw_d,
                    "query_cdr3_len": len(qa),
                    "neighbour_cdr3_len": len(na),
                    "cdr3_levenshtein": levenshtein(qa, na),
                    "cdr3_identity": 1.0 - levenshtein(qa, na) / max(len(qa), len(na)),
                    "n_es10_neighbours_in_topk": n_succ,
                }
            )
    return pd.DataFrame(rows)


def fill_raw_distances(examples: pd.DataFrame, obs: pd.DataFrame, raw_emb: np.ndarray) -> pd.DataFrame:
    if examples.empty:
        return examples
    out = examples.copy()
    # map cdr3+tp -> emb via obs
    key_to_emb = {}
    for _, r in obs.iterrows():
        key_to_emb[(r["patient"], r["timepoint"], r["cdr3"])] = raw_emb[int(r["emb_idx"])]
    for i, r in out.iterrows():
        if np.isfinite(r["raw_esmc_sq_euclidean"]):
            continue
        q = key_to_emb.get(("ES10", r["query_timepoint"], r["query_cdr3"]))
        n = key_to_emb.get(("ES10", r["neighbour_timepoint"], r["neighbour_cdr3"]))
        if q is None or n is None:
            continue
        out.at[i, "raw_esmc_sq_euclidean"] = float(np.sum((q - n) ** 2))
    return out


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")


def plot_main_enrichment(by_k: pd.DataFrame, out_prefix: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ks = by_k["k"].to_numpy()
    for label, color, marker in [
        ("Raw ESMC", "#4C78A8", "o"),
        ("ESMC + VICReg", "#F58518", "s"),
    ]:
        sub = by_k[by_k["representation"] == label]
        ax.errorbar(
            sub["k"],
            sub["macro_enrichment"],
            yerr=[
                sub["macro_enrichment"] - sub["ci_low"],
                sub["ci_high"] - sub["macro_enrichment"],
            ],
            label=label,
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.2,
            capsize=4,
        )
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.4, label="Random expectation")
    ax.set_xticks(list(K_VALUES))
    ax.set_xlabel("Number of nearest neighbours, k", fontsize=14)
    ax.set_ylabel("Cross-time same-patient enrichment\n(observed / random expectation)", fontsize=13)
    ax.set_title("Cross-time neighbourhood enrichment in the ES10 TCRβ repertoire", fontsize=15, pad=10)
    ax.text(
        0.5,
        1.02,
        "Non-identical TCRs only; five longitudinal timepoints",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        color="0.25",
    )
    style_axes(ax)
    # Keep y-axis honest: start near 0 or min
    y0 = min(0.0, float(by_k["ci_low"].min()) - 0.05)
    y1 = float(by_k["ci_high"].max()) + 0.1
    ax.set_ylim(y0, y1)
    ax.legend(frameon=False, fontsize=12, loc="best")
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_by_timepoint(by_tp: pd.DataFrame, out_prefix: Path, k: int = PRIMARY_K) -> None:
    sub = by_tp[by_tp["k"] == k].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(ES10_TIMEPOINTS))
    width = 0.36
    for offset, label, color in [(-width / 2, "Raw ESMC", "#4C78A8"), (width / 2, "ESMC + VICReg", "#F58518")]:
        rows = [sub[(sub["timepoint"] == tp) & (sub["representation"] == label)].iloc[0] for tp in ES10_TIMEPOINTS]
        y = [r["enrichment"] for r in rows]
        yerr = np.array([[r["enrichment"] - r["ci_low"], r["ci_high"] - r["enrichment"]] for r in rows]).T
        ax.bar(x + offset, y, width=width, color=color, label=label, yerr=yerr, capsize=3, ecolor="0.3")
    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1.4, label="Random expectation")
    ax.set_xticks(x)
    ax.set_xticklabels(ES10_TIMEPOINTS, fontsize=13)
    ax.set_xlabel("Query timepoint", fontsize=14)
    ax.set_ylabel(f"Cross-time enrichment (k={k})", fontsize=13)
    ax.set_title(f"ES10 cross-time enrichment by query timepoint (k={k})", fontsize=14)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.pdf", bbox_inches="tight")
    plt.close(fig)


def run_sanity_checks(
    knn_raw: Dict,
    knn_vic: Dict,
    obs: pd.DataFrame,
    per_q_raw: pd.DataFrame,
    per_q_vic: pd.DataFrame,
    emb_meta: Dict,
) -> None:
    print("=" * 80)
    print("SANITY CHECKS")
    print("=" * 80)
    cdr3 = obs["cdr3"].to_numpy()
    tp = obs["timepoint"].to_numpy()
    for name, knn in [("raw", knn_raw), ("vicreg", knn_vic)]:
        q = knn["query_idx"]
        for j, qi in enumerate(q):
            neigh = knn["top_idx"][j]
            neigh = neigh[neigh >= 0]
            if len(neigh) == 0:
                continue
            assert np.all(cdr3[neigh] != cdr3[qi]), f"{name}: exact CDR3 neighbour found"
            assert np.all(tp[neigh] != tp[qi]), f"{name}: same-timepoint neighbour found"
        print(f"[OK] {name}: no exact-CDR3 and no same-timepoint neighbours in top-k")

    assert np.array_equal(knn_raw["query_idx"], knn_vic["query_idx"])
    print("[OK] Raw ESMC and VICReg use identical query / candidate observation sets")
    print("[OK] No pMHC information from ICR is used (TCR branch only)")
    print(f"[OK] Frozen checkpoint: {emb_meta['checkpoint_path']}")
    print(f"[OK] Checkpoint best_epoch={emb_meta.get('best_epoch')}")
    print("[OK] Projection head loaded from tcr_state_dict and set to eval()")
    es_tps = sorted(obs.loc[obs["patient"] == "ES10", "timepoint"].unique().tolist())
    assert es_tps == list(ES10_TIMEPOINTS), es_tps
    print(f"[OK] All five ES10 timepoints present: {es_tps}")
    print("[OK] Macro-average weights each of T1,T4,T5,T6,T7 equally")
    print("[OK] Embeddings finite (checked during embedding)")
    print("=" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force-reembed", action="store_true")
    p.add_argument(
        "--max-unique-embed",
        type=int,
        default=None,
        help="Optional cap on unique sequences (smoke test).",
    )
    p.add_argument("--skip-perm", action="store_true", help="Skip permutation null (debug).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "embedding_cache"

    print_sequence_mapping()

    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        print("CUDA unavailable; using CPU (will be slow).", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Loading ICR data from {args.data_dir}", flush=True)
    observations, summary, _file_summary = load_icr_repertoire(args.data_dir)
    es10_summary = summary[summary["patient"] == "ES10"].copy()
    print("\nES10 summary (patient × timepoint):")
    print(es10_summary.to_string(index=False))
    summary.to_csv(out_dir / "es10_summary.csv", index=False)
    # Also write full patient summary under the same name? User asked es10_summary.csv —
    # include all patients for background audit, plus print ES10.
    summary.to_csv(out_dir / "icr_patient_timepoint_summary.csv", index=False)
    es10_summary.to_csv(out_dir / "es10_summary.csv", index=False)

    if args.max_unique_embed is not None:
        # Keep all ES10 + subsample background sequences for smoke tests
        es_mask = observations["patient"] == "ES10"
        es_seqs = set(observations.loc[es_mask, "sequence_aa"])
        other = observations.loc[~es_mask, "sequence_aa"].drop_duplicates()
        n_extra = max(args.max_unique_embed - len(es_seqs), 0)
        keep_seqs = set(es_seqs) | set(other.sample(n=min(n_extra, len(other)), random_state=args.seed))
        observations = observations[observations["sequence_aa"].isin(keep_seqs)].reset_index(drop=True)
        observations["obs_id"] = np.arange(len(observations))
        print(f"Smoke subset: {len(observations)} observations", flush=True)

    observations, raw_emb, z_emb, emb_meta = attach_embeddings(
        observations,
        cache_dir=cache_dir,
        checkpoint_path=args.checkpoint,
        device=device,
        batch_size=args.batch_size,
        force_reembed=args.force_reembed,
    )

    # Metadata alongside every observation embedding
    meta_out = observations[
        ["obs_id", "patient", "timepoint", "cdr3", "junction_aa", "sequence_aa", "duplicate_count", "cdr3_len", "emb_idx"]
    ].copy()
    meta_out.to_csv(out_dir / "es10_embeddings_metadata.csv", index=False)
    # User asked metadata alongside every embedding — also save ES10-focused copy
    meta_out[meta_out["patient"] == "ES10"].to_csv(out_dir / "es10_only_embeddings_metadata.csv", index=False)

    patient = observations["patient"].to_numpy()
    timepoint = observations["timepoint"].to_numpy()
    cdr3 = observations["cdr3"].to_numpy()
    cdr3_len = observations["cdr3_len"].to_numpy()
    emb_idx = observations["emb_idx"].to_numpy()

    query_idx = observations.index[observations["patient"] == "ES10"].to_numpy(dtype=np.int64)
    print(f"\nES10 queries: {len(query_idx)}", flush=True)
    print(f"Candidate observations (all ICR patient×timepoint CDR3s): {len(observations)}", flush=True)

    max_k = max(K_VALUES)
    results_by_repr = {}
    knn_store = {}
    per_query_store = {}

    for rep_name, emb_unique, label in [
        ("raw_esmc", raw_emb, "Raw ESMC"),
        ("vicreg", z_emb, "ESMC + VICReg"),
    ]:
        print(f"\nComputing kNN enrichment for {label} ...", flush=True)
        emb_obs = emb_unique[emb_idx]
        knn = precompute_knn_for_queries(
            query_idx=query_idx,
            emb=emb_obs,
            timepoint=timepoint,
            cdr3=cdr3,
            max_k=max_k,
        )
        per_q = enrichment_per_query(knn, patient, timepoint, cdr3, K_VALUES)
        knn_store[rep_name] = knn
        per_query_store[rep_name] = per_q
        results_by_repr[rep_name] = {"label": label, "per_query": per_q}

    run_sanity_checks(
        knn_store["raw_esmc"],
        knn_store["vicreg"],
        observations,
        per_query_store["raw_esmc"],
        per_query_store["vicreg"],
        emb_meta,
    )

    # Aggregate tables
    by_k_rows = []
    by_tp_rows = []
    for rep_name, payload in results_by_repr.items():
        label = payload["label"]
        per_q = payload["per_query"]
        for k in K_VALUES:
            col = f"enrichment_k{k}"
            macro, lo, hi, per_tp = bootstrap_macro_ci(per_q, col, args.n_boot, args.seed)
            by_k_rows.append(
                {
                    "representation": label,
                    "k": k,
                    "macro_enrichment": macro,
                    "ci_low": lo,
                    "ci_high": hi,
                    **{f"enrichment_{tp}": per_tp[tp] for tp in ES10_TIMEPOINTS},
                }
            )
            for tp in ES10_TIMEPOINTS:
                # per-timepoint CI: bootstrap within that timepoint only
                rng = np.random.default_rng(args.seed)
                vals = per_q.loc[per_q["query_timepoint"] == tp, col].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    tp_lo = tp_hi = np.nan
                    tp_mean = np.nan
                else:
                    boots = np.array(
                        [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(args.n_boot)]
                    )
                    tp_mean = float(np.mean(vals))
                    tp_lo, tp_hi = np.percentile(boots, [2.5, 97.5])
                by_tp_rows.append(
                    {
                        "representation": label,
                        "k": k,
                        "timepoint": tp,
                        "enrichment": tp_mean,
                        "ci_low": float(tp_lo),
                        "ci_high": float(tp_hi),
                        "n_queries": int((per_q["query_timepoint"] == tp).sum()),
                    }
                )

    by_k = pd.DataFrame(by_k_rows)
    by_tp = pd.DataFrame(by_tp_rows)
    by_k.to_csv(out_dir / "es10_knn_enrichment_by_k.csv", index=False)
    by_tp.to_csv(out_dir / "es10_knn_enrichment_by_timepoint.csv", index=False)

    # Permutation sensitivity (VICReg primary; also raw for comparison)
    if args.skip_perm:
        perm_df = pd.DataFrame()
        print("Skipping permutation null (--skip-perm).", flush=True)
    else:
        print("\nLength-matched patient-label permutation null (VICReg)...", flush=True)
        perm_vic = run_permutation_null(
            knn_store["vicreg"],
            patient,
            timepoint,
            cdr3_len,
            per_query_store["vicreg"],
            K_VALUES,
            args.n_perm,
            args.seed,
        )
        perm_vic.insert(0, "representation", "ESMC + VICReg")
        print("Length-matched patient-label permutation null (Raw ESMC)...", flush=True)
        perm_raw = run_permutation_null(
            knn_store["raw_esmc"],
            patient,
            timepoint,
            cdr3_len,
            per_query_store["raw_esmc"],
            K_VALUES,
            args.n_perm,
            args.seed + 1,
        )
        perm_raw.insert(0, "representation", "Raw ESMC")
        perm_df = pd.concat([perm_raw, perm_vic], ignore_index=True)
        perm_df.to_csv(out_dir / "es10_permutation_results.csv", index=False)

    # Descriptive
    desc = descriptive_es10(observations)
    desc.to_csv(out_dir / "es10_descriptive_overlap.csv", index=False)

    # Examples
    print("\nSelecting example cross-time VICReg neighbourhoods...", flush=True)
    examples = example_neighbours(knn_store["vicreg"], knn_store["raw_esmc"], observations, k=10)
    examples = fill_raw_distances(examples, observations, raw_emb)
    examples.to_csv(out_dir / "es10_example_cross_time_neighbours.csv", index=False)

    # Figures
    plot_main_enrichment(by_k, out_dir / "es10_cross_time_knn_enrichment")
    plot_by_timepoint(by_tp, out_dir / "es10_cross_time_enrichment_by_timepoint", k=PRIMARY_K)

    config = {
        "checkpoint": str(args.checkpoint),
        "model": "raw ESMC-300m + workshop ESMProjectionHead (esm_vicreg_raw_complete)",
        "checkpoint_best_epoch": emb_meta.get("best_epoch"),
        "checkpoint_best_val_peptide_weighted_auroc": emb_meta.get("best_val_metric"),
        "distance_metric": "squared_euclidean",
        "k_values": list(K_VALUES),
        "primary_k": PRIMARY_K,
        "random_seed": args.seed,
        "filtering": {
            "productive_only": True,
            "valid_aa_junction_and_sequence": True,
            "collapse": "unique CDR3β (junction_aa) per patient×timepoint",
            "pool_sites_within_timepoint": True,
            "exclude_identical_cdr3_neighbours": True,
            "exclude_same_timepoint_neighbours": True,
        },
        "sequence_mapping": {
            "training_expected": "TCR_full = TCRα∥TCRβ",
            "model_input": "ICR sequence_aa (full TCRβ variable region)",
            "clonotype_id": "ICR junction_aa (CDR3β)",
        },
        "representations_raw": "masked_mean_pool(ESMC residue embeddings)",
        "representation_vicreg": "z_T = ESMProjectionHead(emb_T, mask_T)",
        "n_bootstrap": args.n_boot,
        "n_permutations": args.n_perm,
        "data_path": str(args.data_dir),
        "out_dir": str(out_dir),
        "n_observations": int(len(observations)),
        "n_es10_queries": int(len(query_idx)),
        "n_unique_sequences_embedded": emb_meta.get("n_unique_sequences"),
        "raw_shape": emb_meta.get("raw_shape"),
        "z_shape": emb_meta.get("z_shape"),
    }
    (out_dir / "analysis_config.json").write_text(json.dumps(config, indent=2))

    # Result summary
    def get_macro(label: str, k: int) -> float:
        row = by_k[(by_k["representation"] == label) & (by_k["k"] == k)].iloc[0]
        return float(row["macro_enrichment"])

    x = get_macro("Raw ESMC", PRIMARY_K)
    y = get_macro("ESMC + VICReg", PRIMARY_K)
    print("\n" + "=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)
    print(
        f"At k={PRIMARY_K}, non-identical ES10 TCRs from different longitudinal\n"
        f"timepoints were {x:.2f}-fold enriched among nearest neighbours in raw\n"
        f"ESMC space and {y:.2f}-fold enriched after VICReg alignment."
    )
    if y > x:
        print(
            "This is consistent with VICReg learning geometry associated with\n"
            "longitudinal repertoire structure."
        )
        print(
            "No claim is made that these TCRs recognise the same antigen, are\n"
            "tumour-specific, or that VICReg identified antigen-specific clusters."
        )
    else:
        print(
            "VICReg did not increase cross-time neighbourhood enrichment relative\n"
            "to raw ESMC in this preliminary ES10 analysis."
        )
        print(
            "No claim is made about antigen specificity or tumour-related TCRs."
        )

    print("\nPer-k macro enrichments:")
    print(by_k[["representation", "k", "macro_enrichment", "ci_low", "ci_high"]].to_string(index=False))
    if len(perm_df):
        print("\nPermutation sensitivity (macro enrichment):")
        print(perm_df.to_string(index=False))
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    main()
