#!/usr/bin/env python3
"""
Sequence-structure VICReg for TCR:pMHC complexes using canonical multiview IDs.

This script trains a two-view model of the same positive binding event:

    view 1: sequence-side TCR:pMHC representation from existing ESM shards
    view 2: structure-side TCR:pMHC representation from Boltz z pair embeddings

The model is deliberately not a TCR-vs-pMHC alignment model. It is a full-complex
view-alignment model. I use the complete sequence complex on one branch and the
complete structural interface representation on the other branch, project both into
one latent coordinate system, and train them with a VICReg objective.

Important design choices:
    1. Training uses positives only. The invariance term is therefore only imposed on
       observed binders. Validation/test can contain positives and negatives.
    2. Any example with missing TCR alpha or missing TCR beta is excluded from train,
       validation and test. I do not use incomplete TCRs because the structural view is
       no longer biologically comparable across examples.
    3. The inference score is the same geometry as the invariance term: negative MSE
       between the sequence projection and the structure projection. Higher score means
       more binder-like. Cosine is not used as the evaluation score.
    4. The sequence and structure branches do not share a projection head. They share
       only the target latent dimensionality. This is intentional because ESM residue
       embeddings and Boltz pair embeddings have different token semantics.

Typical launch:
    cd /home/natasha/multimodal_model/scripts/train/hpo_training
    conda activate tcr-multimodal
    python train_vicreg_sequence_structure_multiview.py \
      --embed-root /home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids \
      --train-manifest-csv /home/natasha/multimodal_model/data/train/train_multiview.csv \
      --val-csv /home/natasha/multimodal_model/data/val/val_multiview.csv \
      --test-csv /home/natasha/multimodal_model/data/test/test_multiview.csv \
      --boltz-train-root /home/natasha/multimodal_model/outputs/train \
      --boltz-val-root /home/natasha/multimodal_model/outputs/val \
      --boltz-test-root /home/natasha/multimodal_model/outputs/test \
      2>&1 | tee /home/natasha/multimodal_model/models/checkpoints/hpo_training/seq_struct_multiview/train.log
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


# ============================================================
# General utilities
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_str_list(x: Any) -> List[str]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif not isinstance(x, (list, tuple)):
        x = [x]
    out = []
    for v in x:
        out.append(v.decode("utf-8") if isinstance(v, bytes) else str(v))
    return out


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float("nan") if len(np.unique(labels)) < 2 else float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float("nan") if len(np.unique(labels)) < 2 else float(average_precision_score(labels, scores))


def safe_partial_auc_raw(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Return the unstandardised partial ROC area from FPR=0 to max_fpr.

    The IMMREP-style AUC0.1 metric is usually described as the area under the ROC
    curve restricted to low false-positive rates. I therefore report the raw integral
    as written in the competition definition, and separately report raw/max_fpr as a
    0-to-1-normalised convenience metric.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(labels, scores)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0, 1]")

    # Ensure the curve contains an exact point at max_fpr by linear interpolation.
    if max_fpr not in fpr:
        stop = np.searchsorted(fpr, max_fpr, side="right")
        fpr_ext = np.concatenate([fpr[:stop], [max_fpr]])
        tpr_ext = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    else:
        keep = fpr <= max_fpr
        fpr_ext = fpr[keep]
        tpr_ext = tpr[keep]
    return float(auc(fpr_ext, tpr_ext))


def safe_partial_auc_norm(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    raw = safe_partial_auc_raw(labels, scores, max_fpr=max_fpr)
    return float("nan") if math.isnan(raw) else float(raw / max_fpr)





def safe_partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """Return McClish-standardised partial AUROC up to max_fpr.

    This matches sklearn's roc_auc_score(..., max_fpr=max_fpr) convention and is the
    appropriate scale for IMMREP-style Macro AUC0.1 reporting: random performance is
    ~0.5 and perfect performance is 1.0. It is distinct from raw/max_fpr, where random
    performance at max_fpr=0.1 is ~0.05.
    """
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def resolve_shard_dir(embed_root: str, split: str) -> Optional[Path]:
    """Resolve shard directories while tolerating either root/split or direct split paths."""
    root = Path(embed_root)
    candidates = [root / split, root]
    for c in candidates:
        if c.exists() and list(c.glob("shard_*.pt")):
            return c
    return None


# ============================================================
# Manifest filtering
# ============================================================


def normalise_manifest(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Standardise metadata needed for filtering, labels, peptides and chain lengths."""
    if "pair_id" not in df.columns:
        raise ValueError(f"{source_name} must contain pair_id")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    if label_col is None:
        # Train manifests should normally have binding_flag, but validation/test files
        # sometimes arrive as positives-only files. I default to positive only when no
        # label is available, and print this explicitly in the calling function.
        out["binding_flag"] = 1
    elif label_col != "binding_flag":
        out["binding_flag"] = out[label_col].astype(int)
    else:
        out["binding_flag"] = out["binding_flag"].astype(int)

    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    if pep_col is not None:
        out["peptide_for_eval"] = out[pep_col].astype(str)
    else:
        out["peptide_for_eval"] = out["pair_id"].astype(str)

    # I prefer explicit lengths because Boltz z slicing must match the chain order used
    # during structure prediction. Sequence strings are only a fallback.
    length_specs = {
        "tcra_len": ["tcra_len", "tcr_alpha_len", "cdr3a_len", "alpha_len"],
        "tcrb_len": ["tcrb_len", "tcr_beta_len", "cdr3b_len", "beta_len"],
        "pep_len": ["pep_len", "peptide_len"],
        "hla_len": ["hla_len", "mhc_len", "mhca_len"],
    }
    seq_specs = {
        "tcra_len": ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
        "tcrb_len": ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
        "pep_len": ["Peptide", "peptide", "pep_seq", "peptide_seq"],
        "hla_len": ["hla", "HLA", "hla_seq", "mhc", "mhc_seq"],
    }

    for target_col, candidates in length_specs.items():
        src = first_existing_col(out, candidates)
        if src is not None:
            out[target_col] = pd.to_numeric(out[src], errors="coerce").fillna(0).astype(int)
            continue
        seq_col = first_existing_col(out, seq_specs[target_col])
        if seq_col is not None:
            out[target_col] = out[seq_col].fillna("").astype(str).str.len().astype(int)
        else:
            out[target_col] = 0

    return out


def complete_complex_pair_ids(df: pd.DataFrame, positives_only: bool, source_name: str) -> Tuple[pd.DataFrame, set[str]]:
    meta = normalise_manifest(df, source_name)
    before = len(meta)
    if positives_only:
        meta = meta[meta["binding_flag"].astype(int) == 1].copy()
    after_label = len(meta)

    complete = (
        (meta["tcra_len"] > 0)
        & (meta["tcrb_len"] > 0)
        & (meta["pep_len"] > 0)
        & (meta["hla_len"] > 0)
    )
    meta = meta[complete].copy()
    print(
        f"{source_name}: rows={before} | after_label_filter={after_label} | complete_alpha_beta_pmhc={len(meta)}",
        flush=True,
    )
    return meta, set(meta["pair_id"].astype(str))


# ============================================================
# Boltz prediction indexing and z-token extraction
# ============================================================


def pair_id_aliases(pair_id: str) -> List[str]:
    aliases = {str(pair_id)}
    s = str(pair_id)
    if s.startswith("pair_"):
        tail = s.split("pair_", 1)[1]
        if tail.isdigit():
            n = int(tail)
            aliases.update({f"pair_{n}", f"pair_{n:03d}", f"pair_{n:06d}"})
    return sorted(aliases)


def build_boltz_prediction_index(outputs_root: str) -> Dict[str, Path]:
    root = Path(outputs_root)
    index: Dict[str, Path] = {}
    if not root.exists():
        print(f"Boltz root does not exist: {root}", flush=True)
        return index
    patterns = [
        "chunk_*/boltz_results_pair_*/predictions/pair_*",
        "boltz_results_pair_*/predictions/pair_*",
        "**/boltz_results_pair_*/predictions/pair_*",
        "**/predictions/pair_*",
    ]
    for pattern in patterns:
        for p in root.glob(pattern):
            if p.is_dir():
                for alias in pair_id_aliases(p.name):
                    index.setdefault(alias, p)
        if index:
            break
    print(f"Indexed Boltz predictions under {root} | aliases={len(index)} | dirs={len(set(map(str, index.values())))}", flush=True)
    return index


def find_pred_dir(index: Dict[str, Path], pair_id: str) -> Optional[Path]:
    for alias in pair_id_aliases(pair_id):
        if alias in index:
            return index[alias]
    return None


def find_embedding_npz(pred_dir: Path, pair_id: str) -> Optional[Path]:
    candidates = [pred_dir / f"embeddings_{a}.npz" for a in pair_id_aliases(pair_id)]
    candidates.extend(sorted(pred_dir.glob("embeddings_pair_*.npz")))
    candidates.extend(sorted(pred_dir.glob("*embedding*.npz")))
    for c in candidates:
        if c.exists():
            return c
    return None


def first_z_array(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "z" in data:
        z = data["z"]
    else:
        z_keys = [k for k in data.keys() if k.lower() == "z" or "z" in k.lower()]
        if not z_keys:
            raise KeyError(f"No z-like array in {npz_path}; keys={list(data.keys())}")
        z = data[z_keys[0]]
    z = np.asarray(z)
    if z.ndim == 4:
        z = z[0]
    if z.ndim != 3:
        raise ValueError(f"Expected z shape (L,L,Dz) or (B,L,L,Dz), got {z.shape} in {npz_path}")
    return z.astype(np.float32, copy=False)


def chain_slices(row: pd.Series, chain_order: str) -> Dict[str, slice]:
    lengths = {
        "tcra": int(row["tcra_len"]),
        "tcrb": int(row["tcrb_len"]),
        "pep": int(row["pep_len"]),
        "hla": int(row["hla_len"]),
    }
    pos = 0
    out: Dict[str, slice] = {}
    for name in [x.strip().lower() for x in chain_order.split(",") if x.strip()]:
        if name not in lengths:
            raise ValueError(f"Unknown chain '{name}' in chain_order='{chain_order}'")
        L = lengths[name]
        out[name] = slice(pos, pos + L)
        pos += L
    return out


def deterministic_subsample(x: np.ndarray, max_tokens: int) -> np.ndarray:
    if max_tokens <= 0 or x.shape[0] <= max_tokens:
        return x
    idx = np.linspace(0, x.shape[0] - 1, num=max_tokens, dtype=np.int64)
    return x[idx]


def extract_interface_tokens(
    z: np.ndarray,
    row: pd.Series,
    chain_order: str,
    include_bidirectional: bool,
    max_tokens_per_block: int,
    interfaces: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Boltz z into variable-length pair tokens plus pair-type ids.

    I keep the structural branch token-based rather than collapsing z to scalar block
    statistics. This preserves more information for the attention encoder. The token
    budget prevents very large TCR-HLA blocks from dominating memory.
    """
    sl = chain_slices(row, chain_order)
    block_specs = []
    requested = [x.strip().lower() for x in interfaces.split(",") if x.strip()]

    def add_block(name: str, a: str, b: str) -> None:
        block_specs.append((name, sl[a], sl[b]))
        if include_bidirectional:
            block_specs.append((name + "_rev", sl[b], sl[a]))

    if "tcr_pep" in requested:
        add_block("tcra_pep", "tcra", "pep")
        add_block("tcrb_pep", "tcrb", "pep")
    if "tcr_hla" in requested:
        add_block("tcra_hla", "tcra", "hla")
        add_block("tcrb_hla", "tcrb", "hla")
    if "pep_hla" in requested:
        add_block("pep_hla", "pep", "hla")

    tokens: List[np.ndarray] = []
    type_ids: List[np.ndarray] = []
    for type_id, (_, a, b) in enumerate(block_specs):
        block = z[a, b, :]
        if block.size == 0:
            continue
        flat = block.reshape(-1, block.shape[-1]).astype(np.float32, copy=False)
        flat = deterministic_subsample(flat, max_tokens_per_block)
        tokens.append(flat)
        type_ids.append(np.full((flat.shape[0],), type_id, dtype=np.int64))

    if not tokens:
        raise ValueError("No interface tokens were extracted from Boltz z")
    return np.concatenate(tokens, axis=0), np.concatenate(type_ids, axis=0)


def resolve_npz_from_csv_value(value: Any, boltz_root: str, project_root: str) -> Optional[Path]:
    """Resolve a Boltz npz path stored in the metadata CSV.

    The new multiview CSVs may contain a direct `boltz_embedding_npz` column. This
    is preferable to reconstructing paths from folder names, because it preserves
    the exact Boltz output used when the manifest was built. The values may be
    absolute paths or paths relative to the project root, for example:

        outputs/val/chunk_000/boltz_results_pair_000/.../embeddings_pair_000.npz

    I test several grounded locations and return the first file that exists.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null"}:
        return None

    p = Path(raw)
    candidates = []
    if p.is_absolute():
        candidates.append(p)
    else:
        root = Path(project_root)
        br = Path(boltz_root)
        candidates.extend([
            Path.cwd() / p,
            root / p,
            br / p,
            br.parent / p,
            br.parent.parent / p if br.parent.parent != br.parent else br.parent / p,
        ])
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand
    return None


def make_boltz_path_map(meta: pd.DataFrame, boltz_root: str, required: bool, source_name: str, project_root: str = "/home/natasha/multimodal_model") -> Dict[str, Path]:
    """Map pair_id to the Boltz pair-embedding npz.

    I first use direct npz paths from the merged multiview CSV when available. This
    is the safest mode because the CSV is the source of truth for the structure file
    attached to each row. If no direct path exists or a particular path cannot be
    resolved, I fall back to the older folder-indexing logic based on pair_id aliases.
    """
    direct_col = first_existing_col(meta, ["boltz_embedding_npz", "boltz_npz", "npz_path", "embedding_npz", "boltz_z_npz"])
    path_map: Dict[str, Path] = {}

    if direct_col is not None:
        for _, row in meta.iterrows():
            pid = str(row["pair_id"])
            npz = resolve_npz_from_csv_value(row[direct_col], boltz_root=boltz_root, project_root=project_root)
            if npz is not None:
                path_map[pid] = npz
        print(f"{source_name}: Boltz npz matched directly from {direct_col}: {len(path_map)}/{len(meta)} complete metadata rows", flush=True)

    if len(path_map) < len(meta):
        index = build_boltz_prediction_index(boltz_root)
        for pid in meta["pair_id"].astype(str).tolist():
            if pid in path_map:
                continue
            pred_dir = find_pred_dir(index, pid)
            if pred_dir is None:
                continue
            npz = find_embedding_npz(pred_dir, pid)
            if npz is not None:
                path_map[pid] = npz
        print(f"{source_name}: Boltz npz matched after direct/fallback lookup {len(path_map)}/{len(meta)} complete metadata rows", flush=True)

    if required and not path_map:
        raise RuntimeError(f"No Boltz npz files matched for {source_name}; check --boltz-{source_name}-root, direct npz paths and pair_id aliases")
    return path_map


# ============================================================
# Precomputed structure shard store
# ============================================================


class StructShardStore:
    """Read precomputed Boltz interface shards built by build_struct_shards.py.

    Structure tokens are no longer sliced from raw Boltz .npz during training.
    Instead they are looked up by pair_id from padded float16 shards (cap 128 per
    directional block, N_max 512). This removes per-epoch .npz IO and slicing.
    """

    def __init__(self, shards_dir: Path, source_name: str, cache_size: int = 8):
        self.dir = Path(shards_dir)
        self.source_name = source_name
        index_path = self.dir / "struct_shard_index.json"
        if not index_path.exists():
            raise FileNotFoundError(
                f"{source_name}: struct_shard_index.json not found in {self.dir}. "
                f"Run scripts/preprocess/build_struct_shards.py for this split first."
            )
        payload = json.loads(index_path.read_text())
        self.index: Dict[str, Dict[str, Any]] = payload["index"]
        self.cap_per_block = int(payload.get("cap_per_block", 128))
        self.n_max = int(payload.get("n_max", 512))
        self.dz = int(payload.get("dz", 128))
        # Small LRU cache of loaded shards. Shuffled training jumps across many shards,
        # so caching only the last shard thrashes the disk. Keep this modest: each shard
        # can be large.
        self.cache_size = max(1, int(cache_size))
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        print(
            f"{source_name}: loaded struct shard index {self.dir} | examples={len(self.index)} "
            f"| cap_per_block={self.cap_per_block} | n_max={self.n_max} | dz={self.dz} "
            f"| shard_cache_size={self.cache_size}",
            flush=True,
        )

    def pair_ids(self) -> set[str]:
        return set(self.index.keys())

    def __contains__(self, pair_id: str) -> bool:
        return str(pair_id) in self.index

    def _load_shard(self, shard_name: str):
        if shard_name in self._cache:
            self._cache.move_to_end(shard_name)
            return self._cache[shard_name]

        shard = torch.load(self.dir / shard_name, map_location="cpu")
        self._cache[shard_name] = shard
        self._cache.move_to_end(shard_name)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    def get(self, pair_id: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rec = self.index[str(pair_id)]
        shard = self._load_shard(rec["shard"])
        row = int(rec["row"])
        # struct_tokens are stored as float16; cast to float32 for downstream use.
        tokens = shard["struct_tokens"][row].float()
        type_ids = shard["struct_type_ids"][row].long()
        mask = shard["struct_mask"][row].bool()
        return tokens, type_ids, mask


def subsample_tokens_per_type(
    tokens: torch.Tensor,
    type_ids: torch.Tensor,
    mask: torch.Tensor,
    cap_per_type: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop padding and deterministically subsample each type-id group to cap_per_type.

    Shards store up to 128 tokens per directional block. At load time we can cap each
    block lower (e.g. 64) so the same 128-cap shards serve both 64- and 128-token
    experiments. Selection uses evenly spaced indices, matching the build-time
    deterministic_subsample. This is a subsample-of-a-subsample relative to slicing the
    full Boltz z block straight to 64 tokens; that is deterministic and acceptable.
    """
    valid = mask.bool()
    tokens = tokens[valid]
    type_ids = type_ids[valid]

    keep_tok: List[torch.Tensor] = []
    keep_ty: List[torch.Tensor] = []
    # torch.unique returns sorted type ids, preserving block order 0,1,2,3.
    for tid in torch.unique(type_ids):
        sel = (type_ids == tid).nonzero(as_tuple=True)[0]
        if cap_per_type > 0 and sel.numel() > cap_per_type:
            idx = torch.linspace(0, sel.numel() - 1, steps=cap_per_type).round().long()
            sel = sel[idx]
        keep_tok.append(tokens[sel])
        keep_ty.append(type_ids[sel])

    if not keep_tok:
        return tokens.new_zeros((0, tokens.shape[-1])), type_ids.new_zeros((0,))
    return torch.cat(keep_tok, dim=0), torch.cat(keep_ty, dim=0)


# ============================================================
# Sharded multimodal dataset
# ============================================================


class SequenceStructureDataset(Dataset):
    """Flatten existing ESM shards and attach precomputed structure tokens by pair_id.

    The canonical join key is pair_id. ESM token embeddings come from the existing ESM
    shards; the structure view is read from precomputed struct shards (built offline by
    build_struct_shards.py) rather than from raw Boltz .npz. A row-order fallback exists
    only as an optional diagnostic switch; it is disabled by default because final
    experiments should not silently pair mismatched sequence and structure examples.
    """

    def __init__(
        self,
        shards_dir: Path,
        meta: pd.DataFrame,
        allowed_pair_ids: set[str],
        struct_store: "StructShardStore",
        cfg: "RunConfig",
        source_name: str,
    ):
        self.shards_dir = Path(shards_dir)
        self.shard_paths = sorted(self.shards_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shards_dir}")
        self.cfg = cfg
        self.source_name = source_name
        self.meta = meta.reset_index(drop=True).copy()
        self.meta_by_pid = {str(r["pair_id"]): r for _, r in self.meta.iterrows()}
        self.struct_store = struct_store
        self.allowed = set(allowed_pair_ids) & struct_store.pair_ids()
        self.index: List[Tuple[Path, int, int, str, int, str]] = []
        self._cache_path: Optional[Path] = None
        self._cache_data: Optional[Any] = None

        print(f"Indexing {source_name} ESM shards: {self.shards_dir}", flush=True)

        flat_rows: List[Tuple[Path, int, int, str]] = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for bidx, batch in enumerate(shard):
                pair_ids = to_str_list(batch["pair_id"])
                for ridx, shard_pid in enumerate(pair_ids):
                    flat_rows.append((sp, bidx, ridx, str(shard_pid)))

        seen = len(flat_rows)
        self.n_shard_rows_seen = seen
        kept_by_pair_id = 0
        kept_by_row_order = 0

        # First pass: exact pair_id join between ESM shard metadata and the multiview CSV.
        for global_i, (sp, bidx, ridx, shard_pid) in enumerate(flat_rows):
            if shard_pid in self.allowed:
                self.index.append((sp, bidx, ridx, shard_pid, global_i, shard_pid))
                kept_by_pair_id += 1

        # Fallback: if exact pair_id matching fails or is sparse, use row-order alignment.
        # This is appropriate when the ESM shards were created directly from the same split
        # CSV in the same order but before Boltz-specific pair_id renaming/aliasing.
        if kept_by_pair_id == 0 and self.cfg.allow_row_order_fallback and seen == len(self.meta):
            self.index = []
            for global_i, (sp, bidx, ridx, shard_pid) in enumerate(flat_rows):
                meta_pid = str(self.meta.iloc[global_i]["pair_id"])
                if meta_pid in self.allowed:
                    self.index.append((sp, bidx, ridx, shard_pid, global_i, meta_pid))
                    kept_by_row_order += 1
            print(
                f"{source_name}: exact shard pair_id matching kept 0 rows; "
                f"row-order fallback was used because shard rows ({seen}) == CSV rows ({len(self.meta)})",
                flush=True,
            )
        elif kept_by_pair_id == 0:
            print(
                f"{source_name}: exact shard pair_id matching kept 0 rows and row-order fallback was not used "
                f"because shard rows ({seen}) != CSV rows ({len(self.meta)})",
                flush=True,
            )

        kept = len(self.index)
        print(
            f"{source_name}: shard rows seen={seen} | kept_complete_with_struct={kept} "
            f"| kept_by_pair_id={kept_by_pair_id} | kept_by_row_order={kept_by_row_order}",
            flush=True,
        )
        if not self.index:
            example_shard_ids = [r[3] for r in flat_rows[:5]]
            example_meta_ids = self.meta["pair_id"].astype(str).head(5).tolist() if "pair_id" in self.meta.columns else []
            raise RuntimeError(
                f"{source_name}: no examples left after complete-chain and struct-shard filtering. "
                f"Example shard pair_ids={example_shard_ids}; example CSV pair_ids={example_meta_ids}. "
                f"This usually means the ESM shard IDs and struct shard IDs still differ, or the "
                f"complete-chain filtering removed all matching rows."
            )

    def __len__(self) -> int:
        return len(self.index)

    def _load_shard(self, sp: Path):
        if self._cache_path != sp:
            self._cache_data = torch.load(sp, map_location="cpu")
            self._cache_path = sp
        return self._cache_data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sp, bidx, ridx, shard_pid, global_i, meta_pid = self.index[idx]
        batch = self._load_shard(sp)[bidx]
        row = self.meta_by_pid[meta_pid]

        # Structure view comes from precomputed shards (no raw Boltz .npz at train time).
        # Shards hold up to cap_per_block (128) tokens per directional block; we cap each
        # type-id group down to cfg.tokens_per_interface_at_load (e.g. 64) here.
        tokens, type_ids, mask = self.struct_store.get(meta_pid)
        struct_tokens, struct_type_ids = subsample_tokens_per_type(
            tokens, type_ids, mask, self.cfg.tokens_per_interface_at_load
        )

        # Labels are taken from the manifest after filtering, not from the shard, because
        # the validation/test CSV is the authoritative source once negatives are present.
        return {
            "emb_T": batch["emb_T"][ridx].float(),
            "emb_P": batch["emb_P"][ridx].float(),
            "emb_H": batch["emb_H"][ridx].float(),
            "mask_T": batch["mask_T"][ridx].bool(),
            "mask_P": batch["mask_P"][ridx].bool(),
            "mask_H": batch["mask_H"][ridx].bool(),
            "struct_tokens": struct_tokens.float(),
            "struct_type_ids": struct_type_ids.long(),
            "binding_flag": int(row["binding_flag"]),
            "pair_id": str(row["pair_id"]),
            "peptide": str(row["peptide_for_eval"]),
        }


def seq_struct_collate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_n = max(r["struct_tokens"].shape[0] for r in rows)
    dz = rows[0]["struct_tokens"].shape[1]
    B = len(rows)
    struct_tokens = torch.zeros((B, max_n, dz), dtype=torch.float32)
    struct_type_ids = torch.zeros((B, max_n), dtype=torch.long)
    struct_mask = torch.zeros((B, max_n), dtype=torch.bool)
    for i, r in enumerate(rows):
        n = r["struct_tokens"].shape[0]
        struct_tokens[i, :n] = r["struct_tokens"]
        struct_type_ids[i, :n] = r["struct_type_ids"]
        struct_mask[i, :n] = True

    return {
        "emb_T": torch.stack([r["emb_T"] for r in rows], dim=0),
        "emb_P": torch.stack([r["emb_P"] for r in rows], dim=0),
        "emb_H": torch.stack([r["emb_H"] for r in rows], dim=0),
        "mask_T": torch.stack([r["mask_T"] for r in rows], dim=0),
        "mask_P": torch.stack([r["mask_P"] for r in rows], dim=0),
        "mask_H": torch.stack([r["mask_H"] for r in rows], dim=0),
        "struct_tokens": struct_tokens,
        "struct_type_ids": struct_type_ids,
        "struct_mask": struct_mask,
        "binding_flag": torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long),
        "pair_id": [r["pair_id"] for r in rows],
        "peptide": [r["peptide"] for r in rows],
    }


# ============================================================
# Model components
# ============================================================


class TransformerAttentionPooler(nn.Module):
    """CLS-token transformer pooling for variable-length biological tokens."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, n_layers: int, dropout: float, n_token_types: int):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.type_embedding = nn.Embedding(n_token_types, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        x = self.input_proj(tokens) + self.type_embedding(type_ids.clamp_min(0))
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        cls_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
        full_mask = torch.cat([cls_mask, mask], dim=1)
        x = self.encoder(x, src_key_padding_mask=~full_mask)
        return self.mlp(self.norm(x[:, 0]))


class AttentionPooler(nn.Module):
    """Multi-query attention pooling over variable-length tokens.

    Unlike the transformer pooler, this does NOT compute token-token self-attention.
    ``n_queries`` learned queries each attend once over the input tokens
    (cross-attention); their pooled outputs are concatenated and projected to the latent
    dimension. This is the default for the structure branch: Boltz z tokens are already
    pairwise structural representations, so the immediate objective is to learn which
    interface tokens are most informative rather than to recompute all token-token
    interactions among interface-pair tokens. Multiple queries let the branch learn
    several complementary summaries of the interface (e.g. TCR-peptide, TCR-HLA and
    mixed structural patterns).
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, n_heads: int, dropout: float, n_token_types: int, n_queries: int = 4):
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by n_heads={n_heads}")
        self.n_queries = max(1, int(n_queries))
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.type_embedding = nn.Embedding(n_token_types, hidden_dim)
        self.query = nn.Parameter(torch.zeros(1, self.n_queries, hidden_dim))
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim * self.n_queries),
            nn.Linear(hidden_dim * self.n_queries, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.normal_(self.query, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        x = self.input_proj(tokens) + self.type_embedding(type_ids.clamp_min(0))
        q = self.query.expand(B, -1, -1)
        # key_padding_mask marks positions to IGNORE, so invert the validity mask.
        # Guard against fully-padded rows (would produce NaN): force at least one key.
        key_padding = ~mask
        all_pad = key_padding.all(dim=1)
        if all_pad.any():
            key_padding = key_padding.clone()
            key_padding[all_pad, 0] = False
        pooled, _ = self.attn(q, x, x, key_padding_mask=key_padding, need_weights=False)
        pooled = pooled.reshape(B, -1)
        return self.mlp(pooled)


class SequenceComplexEncoder(nn.Module):
    """Attention encoder over TCR alpha+beta, peptide and HLA ESM tokens."""

    def __init__(self, esm_dim: int, hidden_dim: int, latent_dim: int, n_heads: int, n_layers: int, dropout: float):
        super().__init__()
        # token types: 0=TCR, 1=peptide, 2=HLA. CLS is injected inside the pooler.
        self.pooler = TransformerAttentionPooler(
            input_dim=esm_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            n_token_types=3,
        )

    def forward(self, emb_T: torch.Tensor, mask_T: torch.Tensor, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        tokens = torch.cat([emb_T, emb_P, emb_H], dim=1)
        mask = torch.cat([mask_T, mask_P, mask_H], dim=1)
        B = tokens.shape[0]
        t_ids = torch.zeros((B, emb_T.shape[1]), dtype=torch.long, device=tokens.device)
        p_ids = torch.ones((B, emb_P.shape[1]), dtype=torch.long, device=tokens.device)
        h_ids = torch.full((B, emb_H.shape[1]), 2, dtype=torch.long, device=tokens.device)
        type_ids = torch.cat([t_ids, p_ids, h_ids], dim=1)
        return self.pooler(tokens, mask, type_ids)


class StructureComplexEncoder(nn.Module):
    """Encoder over Boltz z interface pair tokens.

    Defaults to single-query attention pooling (``encoder_mode="attention_pool"``).
    Full transformer self-attention over the interface tokens is retained as an
    optional ablation (``encoder_mode="transformer"``).
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        latent_dim: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        max_pair_types: int,
        encoder_mode: str = "attention_pool",
        n_queries: int = 4,
    ):
        super().__init__()
        self.encoder_mode = encoder_mode
        if encoder_mode == "attention_pool":
            self.pooler = AttentionPooler(
                input_dim=z_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                n_heads=n_heads,
                dropout=dropout,
                n_token_types=max_pair_types,
                n_queries=n_queries,
            )
        elif encoder_mode == "transformer":
            self.pooler = TransformerAttentionPooler(
                input_dim=z_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
                n_token_types=max_pair_types,
            )
        else:
            raise ValueError(f"Unknown struct encoder_mode='{encoder_mode}' (use 'attention_pool' or 'transformer')")

    def forward(self, struct_tokens: torch.Tensor, struct_mask: torch.Tensor, struct_type_ids: torch.Tensor) -> torch.Tensor:
        return self.pooler(struct_tokens, struct_mask, struct_type_ids)


class SequenceStructureVICReg(nn.Module):
    def __init__(self, esm_dim: int, z_dim: int, cfg: "RunConfig"):
        super().__init__()
        self.sequence_encoder = SequenceComplexEncoder(
            esm_dim=esm_dim,
            hidden_dim=cfg.seq_hidden_dim,
            latent_dim=cfg.latent_dim,
            n_heads=cfg.seq_heads,
            n_layers=cfg.seq_layers,
            dropout=cfg.dropout,
        )
        self.structure_encoder = StructureComplexEncoder(
            z_dim=z_dim,
            hidden_dim=cfg.struct_hidden_dim,
            latent_dim=cfg.latent_dim,
            n_heads=cfg.struct_heads,
            n_layers=cfg.struct_layers,
            dropout=cfg.dropout,
            max_pair_types=cfg.max_pair_types,
            encoder_mode=cfg.struct_encoder_mode,
            n_queries=cfg.struct_pool_queries,
        )

    def forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        z_seq = self.sequence_encoder(
            batch["emb_T"], batch["mask_T"], batch["emb_P"], batch["mask_P"], batch["emb_H"], batch["mask_H"]
        )
        z_struct = self.structure_encoder(batch["struct_tokens"], batch["struct_mask"], batch["struct_type_ids"])
        return z_seq, z_struct


# ============================================================
# VICReg objective and MSE scoring
# ============================================================


def vicreg_variance(u: torch.Tensor, gamma: float, eps: float) -> torch.Tensor:
    u = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u.var(dim=0, unbiased=False) + eps)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u = u - u.mean(dim=0, keepdim=True)
    cov = (u.T @ u) / (B - 1)
    off = cov - torch.diag_embed(torch.diag(cov))
    return (off ** 2).sum() / d


def vicreg_two_view_loss(z_seq: torch.Tensor, z_struct: torch.Tensor, cfg: "RunConfig", return_parts: bool = False):
    # This is the only alignment term. It uses raw latent coordinates, not cosine.
    L_inv = F.mse_loss(z_seq, z_struct)
    L_var = vicreg_variance(z_seq, cfg.gamma_var, cfg.eps_var) + vicreg_variance(z_struct, cfg.gamma_var, cfg.eps_var)
    L_cov = vicreg_covariance(z_seq) + vicreg_covariance(z_struct)
    loss = cfg.alpha * L_inv + cfg.beta * L_var + cfg.delta * L_cov
    if not return_parts:
        return loss
    return loss, {
        "loss": float(loss.detach().cpu()),
        "L_inv_mse": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((cfg.alpha * L_inv).detach().cpu()),
        "weighted_var": float((cfg.beta * L_var).detach().cpu()),
        "weighted_cov": float((cfg.delta * L_cov).detach().cpu()),
        "seq_std": float(z_seq.std(unbiased=False).detach().cpu()),
        "struct_std": float(z_struct.std(unbiased=False).detach().cpu()),
    }


def mse_distance_and_score(z_seq: torch.Tensor, z_struct: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mse_dist = (z_seq - z_struct).pow(2).mean(dim=-1)
    score = -mse_dist
    return mse_dist, score


# ============================================================
# Evaluation, plots and summaries
# ============================================================


def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    best: Optional[Dict[str, float]] = None
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        row = {
            "threshold": float(thr),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "accuracy": float(accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
        }
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best or {"threshold": float("nan"), "f1": float("nan"), "accuracy": float("nan"), "precision": float("nan"), "recall": float("nan")}



def per_peptide_metrics(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float = 0.1) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Calculate per-peptide AUROC and low-FPR partial AUC metrics.

    The IMMREP benchmark uses McClish-standardised partial AUC at FPR <= 0.1 and
    averages that score arithmetically across peptides. I therefore report both:
      - raw partial area and raw/max_fpr diagnostics; and
      - McClish-standardised partial AUC, the benchmark-comparable value.
    """
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        pauc_raw = safe_partial_auc_raw(y, s, max_fpr=max_fpr) if valid else float("nan")
        pauc_mcclish = safe_partial_auc_mcclish(y, s, max_fpr=max_fpr) if valid else float("nan")
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": float(roc_auc_score(y, s)) if valid else float("nan"),
            f"auc{max_fpr:g}_raw": float(pauc_raw) if valid else float("nan"),
            f"auc{max_fpr:g}_raw_div_maxfpr": float(pauc_raw / max_fpr) if valid else float("nan"),
            f"auc{max_fpr:g}_norm": float(pauc_mcclish) if valid else float("nan"),
            f"auc{max_fpr:g}_mcclish": float(pauc_mcclish) if valid else float("nan"),
            "valid": bool(valid),
        })
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid_table = table[table["valid"]].copy()
    if len(valid_table) == 0:
        summary = {
            "macro_per_peptide_auroc": float("nan"),
            "weighted_per_peptide_auroc": float("nan"),
            f"macro_per_peptide_auc{max_fpr:g}_raw": float("nan"),
            f"weighted_per_peptide_auc{max_fpr:g}_raw": float("nan"),
            f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr": float("nan"),
            f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr": float("nan"),
            f"macro_per_peptide_auc{max_fpr:g}_norm": float("nan"),
            f"weighted_per_peptide_auc{max_fpr:g}_norm": float("nan"),
            f"macro_per_peptide_auc{max_fpr:g}_mcclish": float("nan"),
            f"weighted_per_peptide_auc{max_fpr:g}_mcclish": float("nan"),
            "n_valid_peptides": 0,
        }
    else:
        summary = {
            "macro_per_peptide_auroc": float(valid_table["auroc"].mean()),
            "weighted_per_peptide_auroc": float(np.average(valid_table["auroc"], weights=valid_table["n"])),
            f"macro_per_peptide_auc{max_fpr:g}_raw": float(valid_table[f"auc{max_fpr:g}_raw"].mean()),
            f"weighted_per_peptide_auc{max_fpr:g}_raw": float(np.average(valid_table[f"auc{max_fpr:g}_raw"], weights=valid_table["n"])),
            f"macro_per_peptide_auc{max_fpr:g}_raw_div_maxfpr": float(valid_table[f"auc{max_fpr:g}_raw_div_maxfpr"].mean()),
            f"weighted_per_peptide_auc{max_fpr:g}_raw_div_maxfpr": float(np.average(valid_table[f"auc{max_fpr:g}_raw_div_maxfpr"], weights=valid_table["n"])),
            f"macro_per_peptide_auc{max_fpr:g}_norm": float(valid_table[f"auc{max_fpr:g}_mcclish"].mean()),
            f"weighted_per_peptide_auc{max_fpr:g}_norm": float(np.average(valid_table[f"auc{max_fpr:g}_mcclish"], weights=valid_table["n"])),
            f"macro_per_peptide_auc{max_fpr:g}_mcclish": float(valid_table[f"auc{max_fpr:g}_mcclish"].mean()),
            f"weighted_per_peptide_auc{max_fpr:g}_mcclish": float(np.average(valid_table[f"auc{max_fpr:g}_mcclish"], weights=valid_table["n"])),
            "n_valid_peptides": int(len(valid_table)),
        }
    return table, summary

def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float, prefix: str) -> Dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    return {
        f"{prefix}_threshold": float(threshold),
        f"{prefix}_f1": float(f1_score(labels, pred, zero_division=0)),
        f"{prefix}_accuracy": float(accuracy_score(labels, pred)),
        f"{prefix}_precision": float(precision_score(labels, pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(labels, pred, zero_division=0)),
        f"{prefix}_cm": confusion_matrix(labels, pred).tolist(),
    }


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: "RunConfig", split: str) -> Dict[str, Any]:
    model.eval()
    rows = []
    running: Dict[str, float] = {}
    n_steps = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        z_seq, z_struct = model(batch)
        _, parts = vicreg_two_view_loss(z_seq, z_struct, cfg, return_parts=True)
        mse_dist, score = mse_distance_and_score(z_seq, z_struct)
        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        for i, pid in enumerate(batch["pair_id"]):
            rows.append({
                "split": split,
                "pair_id": pid,
                "peptide": batch["peptide"][i],
                "label": int(labels[i]),
                "mse_distance": float(mse_dist[i].detach().cpu()),
                "model_score": float(score[i].detach().cpu()),
            })
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + float(v)
        n_steps += 1

    pred = pd.DataFrame(rows)
    labels_np = pred["label"].to_numpy(dtype=int)
    scores_np = pred["model_score"].to_numpy(dtype=float)
    pep_table, pep_summary = per_peptide_metrics(labels_np, scores_np, pred["peptide"].to_numpy(dtype=str), max_fpr=cfg.partial_auc_max_fpr)
    best_thr = best_f1_threshold(scores_np, labels_np)
    metrics = {
        "split": split,
        "n": int(len(pred)),
        "n_pos": int(labels_np.sum()),
        "n_neg": int((labels_np == 0).sum()),
        "auroc": safe_auroc(labels_np, scores_np),
        "auprc": safe_auprc(labels_np, scores_np),
        f"auc{cfg.partial_auc_max_fpr:g}_raw": safe_partial_auc_raw(labels_np, scores_np, max_fpr=cfg.partial_auc_max_fpr),
        f"auc{cfg.partial_auc_max_fpr:g}_raw_div_maxfpr": safe_partial_auc_norm(labels_np, scores_np, max_fpr=cfg.partial_auc_max_fpr),
        f"auc{cfg.partial_auc_max_fpr:g}_norm": safe_partial_auc_mcclish(labels_np, scores_np, max_fpr=cfg.partial_auc_max_fpr),
        f"auc{cfg.partial_auc_max_fpr:g}_mcclish": safe_partial_auc_mcclish(labels_np, scores_np, max_fpr=cfg.partial_auc_max_fpr),
        **pep_summary,
        **{f"mean_{k}": v / max(n_steps, 1) for k, v in running.items()},
        **{f"best_f1_{k}": v for k, v in best_thr.items()},
    }
    return {"predictions": pred, "per_peptide": pep_table, "metrics": metrics}


def plot_score_histogram(pred: pd.DataFrame, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    pos = pred[pred["label"] == 1]["model_score"].to_numpy()
    neg = pred[pred["label"] == 0]["model_score"].to_numpy()
    if len(pos):
        plt.hist(pos, bins=40, alpha=0.6, label="positive")
    if len(neg):
        plt.hist(neg, bins=40, alpha=0.6, label="negative")
    plt.xlabel("Model score = -MSE(sequence view, structure view)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Configuration
# ============================================================


@dataclass
class RunConfig:
    project: str = "/home/natasha/multimodal_model"
    embed_root: str = "/home/natasha/multimodal_model/models/embeddings/no_boltz_multiview_ids"
    train_manifest_csv: str = "/home/natasha/multimodal_model/data/train/train_multiview.csv"
    val_csv: str = "/home/natasha/multimodal_model/data/val/val_multiview.csv"
    test_csv: str = "/home/natasha/multimodal_model/data/test/test_multiview.csv"
    immrep_test_csv: str = "/home/natasha/multimodal_model/data/immrep_test/immrep_test_multiview.csv"
    immrep_shard_dir: str = ""

    # --- LEGACY (retained for backward compatibility / config files; NOT used) ---
    # Training now reads structure from precomputed shards, so these raw-Boltz roots and
    # on-the-fly slicing controls are no longer consulted when struct_*_root are supplied.
    boltz_train_root: str = "/home/natasha/multimodal_model/outputs/train"
    boltz_val_root: str = "/home/natasha/multimodal_model/outputs/val"
    boltz_test_root: str = "/home/natasha/multimodal_model/outputs/test"
    boltz_immrep_test_root: str = "/home/natasha/multimodal_model/outputs/immrep_test"
    structure_interfaces: str = "tcr_pep,tcr_hla,pep_hla"  # legacy: defined offline now
    include_bidirectional_z: bool = True  # legacy: bidirectional blocks are baked into shards
    max_tokens_per_interface: int = 512  # legacy: see tokens_per_interface_at_load instead
    # -----------------------------------------------------------------------------

    # Precomputed structure shards (built by scripts/preprocess/build_struct_shards.py).
    # Training reads structure tokens from these instead of opening raw Boltz .npz.
    struct_train_root: str = "/home/natasha/multimodal_model/outputs_data/train_struct_shards"
    struct_val_root: str = "/home/natasha/multimodal_model/outputs_data/val_struct_shards"
    struct_test_root: str = "/home/natasha/multimodal_model/outputs_data/test_struct_shards"
    struct_immrep_test_root: str = "/home/natasha/multimodal_model/outputs_data/immrep_test_struct_shards"
    out_dir: str = "/home/natasha/multimodal_model/models/checkpoints/hpo_training/seq_struct_multiview_vicreg_v4_canonical_ids"
    fig_dir: str = "/home/natasha/multimodal_model/models/figures/hpo_training/seq_struct_multiview_vicreg_v4_canonical_ids"
    run_tag: str = "seq_struct_multiview_vicreg_v4_canonical_ids"

    boltz_chain_order: str = "tcra,tcrb,pep,hla"
    max_pair_types: int = 16

    # Structure shards store up to 128 tokens per directional block. At load time each
    # type-id group is subsampled down to this cap, so the same shards serve both the
    # initial 64-token runs and the later 128-token runs without re-preprocessing.
    tokens_per_interface_at_load: int = 64
    # Structure branch encoder: "attention_pool" (default) or "transformer" (ablation).
    struct_encoder_mode: str = "attention_pool"
    # Number of learned pooling queries for the attention-pool structure encoder.
    struct_pool_queries: int = 4
    # LRU cache of loaded structure shards (modest: each shard can be large).
    struct_shard_cache_size: int = 8

    seed: int = 31
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 30
    patience: int = 10
    min_epochs: int = 5

    latent_dim: int = 128
    seq_hidden_dim: int = 256
    struct_hidden_dim: int = 128
    seq_heads: int = 8
    struct_heads: int = 8
    seq_layers: int = 2
    struct_layers: int = 1
    dropout: float = 0.1

    lr: float = 3e-4
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    alpha: float = 25.0
    beta: float = 25.0
    delta: float = 1.0
    gamma_var: float = 1.0
    eps_var: float = 1e-4

    partial_auc_max_fpr: float = 0.1
    allow_row_order_fallback: bool = False
    # Checkpoint selection metric. Defaults to the IMMREP-style low-FPR macro per-peptide
    # partial AUC. get_selection_metric() falls back gracefully if the key is absent.
    selection_metric: str = "macro_per_peptide_auc0.1_norm"


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser()
    defaults = asdict(RunConfig())
    for k, v in defaults.items():
        arg = "--" + k.replace("_", "-")
        if k == "immrep_test_csv":
            # Both spellings are accepted. The shorter --immrep-csv matches the current
            # multiview dataset naming, while --immrep-test-csv is retained for backward
            # compatibility with earlier script versions.
            p.add_argument("--immrep-csv", "--immrep-test-csv", dest=k, default=v)
            continue
        if isinstance(v, bool):
            p.add_argument(arg, action=argparse.BooleanOptionalAction, default=v)
        elif isinstance(v, int):
            p.add_argument(arg, type=int, default=v)
        elif isinstance(v, float):
            p.add_argument(arg, type=float, default=v)
        else:
            p.add_argument(arg, default=v)
    args = p.parse_args()
    return RunConfig(**vars(args))


# ============================================================
# Main training loop
# ============================================================


def infer_dims(ds: SequenceStructureDataset) -> Tuple[int, int]:
    sample = ds[0]
    esm_dim = int(sample["emb_T"].shape[-1])
    z_dim = int(sample["struct_tokens"].shape[-1])
    print(f"Detected dimensions | ESM token dim={esm_dim} | Boltz z token dim={z_dim}", flush=True)
    return esm_dim, z_dim


def build_dataset_for_split(cfg: RunConfig, split: str, shards_dir: Path, csv_path: str, struct_root: str, positives_only: bool) -> SequenceStructureDataset:
    raw = pd.read_csv(csv_path)
    meta, allowed = complete_complex_pair_ids(raw, positives_only=positives_only, source_name=split)
    struct_store = StructShardStore(Path(struct_root), source_name=split, cache_size=cfg.struct_shard_cache_size)
    return SequenceStructureDataset(shards_dir, meta, allowed, struct_store, cfg, split)


def get_selection_metric(metrics: Dict[str, Any], cfg: RunConfig) -> float:
    """Return the checkpoint-selection value, robust to which keys exist.

    The benchmark objective is the IMMREP-style low-FPR partial AUC, ideally macro
    per-peptide. We prefer that, then fall back to the global partial AUC and finally
    to global AUROC so selection never silently returns NaN when the preferred metric
    is undefined (e.g. a split with no per-peptide validity).
    """
    candidates = [
        cfg.selection_metric,
        "weighted_per_peptide_auroc",
        "auroc",
        f"macro_per_peptide_auc{cfg.partial_auc_max_fpr:g}_mcclish",
        f"auc{cfg.partial_auc_max_fpr:g}_mcclish",
    ]
    for key in candidates:
        val = metrics.get(key)
        if isinstance(val, (int, float)) and not math.isnan(float(val)):
            return float(val)
    return float("nan")


def describe_dataset(ds: SequenceStructureDataset, split: str, cfg: RunConfig) -> None:
    """Short sanity print so the intended token budget is easy to confirm."""
    print(
        f"[dataset] split={split} | esm_shard_rows_seen={getattr(ds, 'n_shard_rows_seen', 'NA')} "
        f"| kept_complete_and_struct_matched={len(ds)} "
        f"| struct_examples_indexed={len(ds.struct_store.index)} "
        f"| tokens_per_interface_at_load={cfg.tokens_per_interface_at_load} "
        f"| struct_encoder_mode={cfg.struct_encoder_mode}",
        flush=True,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir); fig_dir = Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True); fig_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80, flush=True)
    print("Sequence-structure full-complex VICReg", flush=True)
    print(f"Device: {device}", flush=True)
    print(json.dumps(asdict(cfg), indent=2), flush=True)
    print("=" * 80, flush=True)

    train_dir = resolve_shard_dir(cfg.embed_root, "train")
    val_dir = resolve_shard_dir(cfg.embed_root, "val")
    test_dir = resolve_shard_dir(cfg.embed_root, "test")
    immrep_dir = Path(cfg.immrep_shard_dir) if cfg.immrep_shard_dir else resolve_shard_dir(cfg.embed_root, "immrep_test")
    if train_dir is None or val_dir is None or test_dir is None:
        raise FileNotFoundError(f"Could not resolve shard dirs under {cfg.embed_root}: train={train_dir}, val={val_dir}, test={test_dir}")

    train_ds = build_dataset_for_split(cfg, "train", train_dir, cfg.train_manifest_csv, cfg.struct_train_root, positives_only=True)
    val_ds = build_dataset_for_split(cfg, "val", val_dir, cfg.val_csv, cfg.struct_val_root, positives_only=False)
    test_ds = build_dataset_for_split(cfg, "test", test_dir, cfg.test_csv, cfg.struct_test_root, positives_only=False)
    immrep_ds = None
    if cfg.immrep_test_csv:
        if immrep_dir is None or not Path(immrep_dir).exists():
            raise FileNotFoundError(f"--immrep-test-csv was provided, but immrep_test ESM shards were not found. Set --immrep-shard-dir explicitly or create {Path(cfg.embed_root) / 'immrep_test'}")
        immrep_ds = build_dataset_for_split(cfg, "immrep_test", Path(immrep_dir), cfg.immrep_test_csv, cfg.struct_immrep_test_root, positives_only=False)

    # Sanity prints: per-split row accounting and the intended token budget.
    describe_dataset(train_ds, "train", cfg)
    describe_dataset(val_ds, "val", cfg)
    describe_dataset(test_ds, "test", cfg)
    if immrep_ds is not None:
        describe_dataset(immrep_ds, "immrep_test", cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=seq_struct_collate, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=seq_struct_collate, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=seq_struct_collate, pin_memory=torch.cuda.is_available())
    immrep_loader = None if immrep_ds is None else DataLoader(immrep_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=seq_struct_collate, pin_memory=torch.cuda.is_available())

    # One collated batch shape print, so the actual token budget per batch is visible.
    _sample_batch = next(iter(train_loader))
    print(
        "[batch] emb_T={} emb_P={} emb_H={} struct_tokens={} struct_type_ids={} struct_mask={}".format(
            tuple(_sample_batch["emb_T"].shape),
            tuple(_sample_batch["emb_P"].shape),
            tuple(_sample_batch["emb_H"].shape),
            tuple(_sample_batch["struct_tokens"].shape),
            tuple(_sample_batch["struct_type_ids"].shape),
            tuple(_sample_batch["struct_mask"].shape),
        ),
        flush=True,
    )
    del _sample_batch

    esm_dim, z_dim = infer_dims(train_ds)
    model = SequenceStructureVICReg(esm_dim=esm_dim, z_dim=z_dim, cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"epoch": -1, "selection_value": -math.inf, "val_auroc": -math.inf, "state": None, "metrics": None}
    history = []
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running: Dict[str, float] = {}
        n_steps = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            opt.zero_grad(set_to_none=True)
            z_seq, z_struct = model(batch)
            loss, parts = vicreg_two_view_loss(z_seq, z_struct, cfg, return_parts=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            for k, v in parts.items():
                running[k] = running.get(k, 0.0) + float(v)
            n_steps += 1

        train_parts = {f"train_{k}": v / max(n_steps, 1) for k, v in running.items()}
        val_eval = evaluate(model, val_loader, device, cfg, "val")
        val_metrics = val_eval["metrics"]
        epoch_row = {"epoch": epoch, **train_parts, **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float, str))}}
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(out_dir / f"{cfg.run_tag}__history.csv", index=False)

        val_auc = val_metrics["auroc"]
        selection_value = get_selection_metric(val_metrics, cfg)
        print(
            f"Epoch {epoch:03d} | train_loss={train_parts.get('train_loss', float('nan')):.4f} "
            f"| val_auroc={val_auc:.4f} | val_auprc={val_metrics['auprc']:.4f} "
            f"| sel[{cfg.selection_metric}]={selection_value:.4f}",
            flush=True,
        )

        if not math.isnan(selection_value) and selection_value > best["selection_value"]:
            best = {
                "epoch": epoch,
                "selection_value": selection_value,
                "val_auroc": val_auc,
                "state": copy.deepcopy(model.state_dict()),
                "metrics": val_metrics,
            }
            bad_epochs = 0
            torch.save(
                {
                    "config": asdict(cfg),
                    "model_state_dict": best["state"],
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "selection_metric": cfg.selection_metric,
                    "selection_value": selection_value,
                },
                out_dir / f"{cfg.run_tag}__best.pt",
            )
        else:
            bad_epochs += 1

        if epoch >= cfg.min_epochs and bad_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch} after {bad_epochs} non-improving epochs", flush=True)
            break

    if best["state"] is not None:
        model.load_state_dict(best["state"])

    val_eval = evaluate(model, val_loader, device, cfg, "val")
    test_eval = evaluate(model, test_loader, device, cfg, "test")
    immrep_eval = None if immrep_loader is None else evaluate(model, immrep_loader, device, cfg, "immrep_test")

    val_eval["predictions"].to_csv(out_dir / f"{cfg.run_tag}__val_predictions.csv", index=False)
    test_eval["predictions"].to_csv(out_dir / f"{cfg.run_tag}__test_predictions.csv", index=False)
    val_eval["per_peptide"].to_csv(out_dir / f"{cfg.run_tag}__val_per_peptide_metrics.csv", index=False)
    test_eval["per_peptide"].to_csv(out_dir / f"{cfg.run_tag}__test_per_peptide_metrics.csv", index=False)
    if immrep_eval is not None:
        immrep_eval["predictions"].to_csv(out_dir / f"{cfg.run_tag}__immrep_test_predictions.csv", index=False)
        immrep_eval["per_peptide"].to_csv(out_dir / f"{cfg.run_tag}__immrep_test_per_peptide_metrics.csv", index=False)

    val_thr = val_eval["metrics"].get("best_f1_threshold", float("nan"))
    summary = {
        "config": asdict(cfg),
        "best_epoch": best["epoch"],
        "selection_metric": cfg.selection_metric,
        "selection_value": best["selection_value"],
        "best_val_auroc_during_training": best["val_auroc"],
        "final_val_metrics": val_eval["metrics"],
        "final_test_metrics": test_eval["metrics"],
        "final_immrep_test_metrics": None if immrep_eval is None else immrep_eval["metrics"],
        "test_at_val_threshold": threshold_metrics(test_eval["predictions"]["model_score"].to_numpy(), test_eval["predictions"]["label"].to_numpy(), val_thr, "test_at_val_threshold"),
    }
    if immrep_eval is not None:
        summary["immrep_test_at_val_threshold"] = threshold_metrics(immrep_eval["predictions"]["model_score"].to_numpy(), immrep_eval["predictions"]["label"].to_numpy(), val_thr, "immrep_test_at_val_threshold")

    with open(out_dir / f"{cfg.run_tag}__summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_score_histogram(val_eval["predictions"], "Validation score distribution", fig_dir / f"{cfg.run_tag}__val_score_hist.png")
    plot_score_histogram(test_eval["predictions"], "Test score distribution", fig_dir / f"{cfg.run_tag}__test_score_hist.png")
    if immrep_eval is not None:
        plot_score_histogram(immrep_eval["predictions"], "IMMREP test score distribution", fig_dir / f"{cfg.run_tag}__immrep_test_score_hist.png")

    torch.save({"config": asdict(cfg), "model_state_dict": model.state_dict(), "summary": summary}, out_dir / f"{cfg.run_tag}__final.pt")
    print("Final validation metrics:", json.dumps(val_eval["metrics"], indent=2), flush=True)
    print("Final test metrics:", json.dumps(test_eval["metrics"], indent=2), flush=True)
    if immrep_eval is not None:
        print("Final IMMREP test metrics:", json.dumps(immrep_eval["metrics"], indent=2), flush=True)
    print(f"Outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
