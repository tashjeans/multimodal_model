#!/usr/bin/env python3
"""
Script to run message passing with one-hot encoding initialisations of embeddings.
This is used to test whether message passing is able to learn something generalisable where we can insert information from Boltz.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


REPO = Path(__file__).resolve().parents[3]

# embedding directory - commented out for now as we will initialise using one-hot encoding
# pretrained_embed_root: str = str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids")
# pretrained_immrep_shard_dir: str = str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids/immrep_test")

train_csv: str = str(REPO / "data/train/train_multiview.csv")
val_csv: str = str(REPO / "data/val/val_multiview.csv")
test_csv: str = str(REPO / "data/test/test_multiview.csv")
immrep_csv: str = str(REPO / "data/immrep_test/immrep_test_multiview.csv")

# parameters for VICReg model
seed: int = 31
batch_size: int = 8
num_workers: int = 0
epochs: int = 30
patience: int = 10
min_epochs: int = 10

rL: int = 8
rD: int = 16
d: int = 128
R_PH: float = 0.7
dropout: float = 0.1

lr: float = 3e-4
weight_decay: float = 1e-2

alpha: float = 25.0
beta: float = 25.0
delta: float = 1.0
gamma_var: float = 1.0
eps_var: float = 1e-4
eps_pool: float = 1e-8
partial_auc_max_fpr: float = 0.1

save_latents: bool = True
overwrite: bool = False

# one-hot encoding utilities

AA20 = "ACDEFGHIKLMNPQRSTVWY"
VOCAB = {aa: i for i, aa in enumerate(AA20)}
VOCAB["X"] = len(VOCAB)
VOCAB["SEP"] = len(VOCAB)  # retained for compatibility; not used by default
VOCAB_SIZE = len(VOCAB)
UNK_IDX = VOCAB["X"]


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def onehot_encode(seq: str, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.zeros(max_len, VOCAB_SIZE, dtype=torch.float32)
    m = torch.zeros(max_len, dtype=torch.bool)
    seq = clean_seq(seq)
    n = min(len(seq), max_len)
    for i, aa in enumerate(seq[:n]):
        x[i, VOCAB.get(aa, UNK_IDX)] = 1.0
        m[i] = True
    return x, m


# ============================================================
# CSV parsing and filtering
# ============================================================

def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_length(
    df: pd.DataFrame,
    length_candidates: List[str],
    seq_candidates: List[str],
    target_name: str,
) -> Tuple[pd.Series, str]:
    length_col = first_existing_col(df, length_candidates)
    if length_col is not None:
        vals = pd.to_numeric(df[length_col], errors="coerce").fillna(0).astype(int)
        return vals, length_col

    seq_col = first_existing_col(df, seq_candidates)
    if seq_col is not None:
        vals = df[seq_col].map(clean_seq).str.len().astype(int)
        return vals, f"inferred_from:{seq_col}"

    return pd.Series(np.zeros(len(df), dtype=int), index=df.index), "__missing__"


def normalise_manifest(df: pd.DataFrame, source_name: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if "pair_id" not in df.columns:
        raise ValueError(f"{source_name}: CSV must contain pair_id")

    out = df.copy()
    out["pair_id"] = out["pair_id"].astype(str)

    label_col = first_existing_col(out, ["binding_flag", "label", "binder", "target"])
    out["binding_flag"] = 1 if label_col is None else pd.to_numeric(out[label_col], errors="coerce").fillna(0).astype(int)

    tcr_col = first_existing_col(out, ["TCR_full", "tcr_full", "full_tcr", "TCR", "tcr"])
    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    hla_col = first_existing_col(out, ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"])

    missing = []
    if tcr_col is None:
        missing.append("TCR_full")
    if pep_col is None:
        missing.append("Peptide")
    if hla_col is None:
        missing.append("HLA_sequence/HLA")
    if missing:
        raise ValueError(f"{source_name}: missing required column(s): {missing}. Available columns: {list(out.columns)}")

    out["TCR_full_norm"] = out[tcr_col].map(clean_seq)
    out["Peptide_norm"] = out[pep_col].map(clean_seq)
    out["HLA_sequence_norm"] = out[hla_col].map(clean_seq)
    out["peptide_for_eval"] = out["Peptide_norm"]

    source_map = {
        "tcr_col": tcr_col,
        "pep_col": pep_col,
        "hla_col": hla_col,
        "label_col": "constant_1" if label_col is None else label_col,
    }

    out["tcra_len"], source_map["tcra_len"] = extract_length(
        out,
        ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
        "tcra_len",
    )
    out["tcrb_len"], source_map["tcrb_len"] = extract_length(
        out,
        ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
        "tcrb_len",
    )
    out["pep_len"], source_map["pep_len"] = extract_length(
        out,
        ["pep_len", "peptide_len"],
        ["Peptide", "peptide", "pep_seq", "peptide_seq"],
        "pep_len",
    )
    out["hla_len"], source_map["hla_len"] = extract_length(
        out,
        ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"],
        "hla_len",
    )

    out["tcr_total_len"] = out["TCR_full_norm"].str.len().astype(int)
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source_map


def load_meta(
    csv_path: str,
    source_name: str,
    positives_only: bool,
    missing_chain_policy: str = "complete_only",
) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(csv_path)
    meta, source_map = normalise_manifest(raw, source_name)

    if missing_chain_policy == "complete_only" and (
        source_map["tcra_len"] == "__missing__" or source_map["tcrb_len"] == "__missing__"
    ):
        raise ValueError(
            f"{source_name}: complete_only requested, but alpha/beta chain length information could not be found. "
            f"Source map: {source_map}. Available columns: {list(raw.columns)}"
        )

    audit = {
        "split": source_name,
        "csv_path": str(csv_path),
        "csv_rows": int(len(raw)),
        "label_source": source_map["label_col"],
        "tcr_source": source_map["tcr_col"],
        "peptide_source": source_map["pep_col"],
        "hla_source": source_map["hla_col"],
        "tcra_len_source": source_map["tcra_len"],
        "tcrb_len_source": source_map["tcrb_len"],
        "pep_len_source": source_map["pep_len"],
        "hla_len_source": source_map["hla_len"],
        "positives_only": bool(positives_only),
        "missing_chain_policy": missing_chain_policy,
        "n_positive_before_filter": int((meta["binding_flag"] == 1).sum()),
        "n_negative_before_filter": int((meta["binding_flag"] == 0).sum()),
        "n_missing_alpha_before_filter": int((~meta["has_alpha"]).sum()),
        "n_missing_beta_before_filter": int((~meta["has_beta"]).sum()),
        "n_missing_peptide_before_filter": int((meta["pep_len"] <= 0).sum()),
        "n_missing_hla_before_filter": int((meta["hla_len"] <= 0).sum()),
    }

    if positives_only:
        meta = meta[meta["binding_flag"] == 1].copy()
    audit["rows_after_positive_filter"] = int(len(meta))

    meta = meta[(meta["pep_len"] > 0) & (meta["hla_len"] > 0) & (meta["tcr_total_len"] > 0)].copy()
    audit["rows_after_required_sequence_filter"] = int(len(meta))

    if missing_chain_policy == "complete_only":
        meta = meta[meta["has_alpha"] & meta["has_beta"]].copy()
    elif missing_chain_policy == "keep":
        pass
    else:
        raise ValueError("missing_chain_policy must be 'complete_only' or 'keep'")

    audit["n_final"] = int(len(meta))
    audit["n_positive_final"] = int((meta["binding_flag"] == 1).sum())
    audit["n_negative_final"] = int((meta["binding_flag"] == 0).sum())
    audit["n_missing_alpha_final"] = int((~meta["has_alpha"]).sum())
    audit["n_missing_beta_final"] = int((~meta["has_beta"]).sum())

    print(
        f"{source_name}: csv_rows={audit['csv_rows']} | final={audit['n_final']} | "
        f"pos={audit['n_positive_final']} | neg={audit['n_negative_final']} | "
        f"complete_policy={missing_chain_policy}",
        flush=True,
    )

    if len(meta) == 0:
        raise RuntimeError(f"{source_name}: no rows remain after filtering")
    return meta.reset_index(drop=True), audit


def compute_max_lengths(metas: List[pd.DataFrame], cap_tcr: int, cap_pep: int, cap_hla: int) -> Tuple[int, int, int]:
    all_meta = pd.concat(metas, axis=0, ignore_index=True)
    L_T = int(all_meta["tcr_total_len"].max())
    L_P = int(all_meta["pep_len"].max())
    L_H = int(all_meta["hla_len"].max())
    if cap_tcr > 0:
        L_T = min(L_T, cap_tcr)
    if cap_pep > 0:
        L_P = min(L_P, cap_pep)
    if cap_hla > 0:
        L_H = min(L_H, cap_hla)
    if min(L_T, L_P, L_H) <= 0:
        raise ValueError(f"Invalid lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}")
    return L_T, L_P, L_H

# ============================================================
# Datasets
# ============================================================

def first_z_array(npz_path: Path, max_len: int) -> np.ndarray:
    """Load one complex's Boltz z from embeddings_{pair_id}.npz → (L, L, D)."""
    with np.load(npz_path) as data:
        if "z" not in data.files:
            raise KeyError(f"No key 'z' in {npz_path}; keys={list(data.files)}")
        z = np.asarray(data["z"])
    if z.ndim == 4:
        z = z[0]
    if z.ndim != 3:
        raise ValueError(f"Expected z shape (L,L,D) or (1,L,L,D), got {z.shape} in {npz_path}")
    # pad to max len
    D = z.shape[-1]
    L_b = z.shape[0]
    z = torch.from_numpy(z.astype(np.float32, copy=False))
    z_pad = torch.zeros(max_len, max_len, D, dtype=torch.float32)
    #z_out = z.astype(np.float32, copy=False)
    mask = torch.zeros(max_len, max_len, dtype=torch.bool)
    # multiply them together?
    z_pad[:L_b, :L_b, :] = z
    mask[:L_b, :L_b] = True
    #z_pad = z.astype(np.float32, copy=False)
    return(z_pad, mask)
    #return z.astype(np.float32, copy=False)


def make_pair_id_to_npz(meta: pd.DataFrame, repo_root: Path) -> Dict[str, Path]:
    """Build pair_id → npz Path once from the CSV column boltz_embedding_npz."""
    path_map: Dict[str, Path] = {}
    col = "boltz_embedding_npz"
    if col not in meta.columns:
        raise ValueError(f"{col} not in meta.columns")
    for _, row in meta.iterrows():
        pid = str(row["pair_id"])
        p = Path(str(row[col]))
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            path_map[pid] = p
    return path_map


class SeqPlusBoltzDataset(Dataset):
    """
    One training example = one pair_id.
    Sequence side: one-hot (later ESM). Structure side: Boltz z.
    """

    def __init__(
        self,
        meta: pd.DataFrame,
        pair_id_to_npz: Dict[str, Path],
        L_T: int,
        L_P: int,
        L_H: int,
        L_Z_max: int,
    ):
        self.paths = pair_id_to_npz
        self.L_T, self.L_P, self.L_H = int(L_T), int(L_P), int(L_H)
        self.L_z_max = int(L_z_max)

        # Keep only rows that have a Boltz file (paths was built once, outside this class)
        keep = [str(pid) in self.paths for pid in meta["pair_id"]]
        self.meta = meta.loc[keep].reset_index(drop=True)
        self.pair_ids = self.meta["pair_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, idx: int) -> Dict:
        row = self.meta.iloc[idx]
        pid = str(row["pair_id"])

        # Structure: look up the path we already stored; do NOT rebuild the map here
        z, z_mask = first_z_array(self.paths[pid], self.L_z_max)

        # Sequence: one-hot (swap for ESM lookup by pid later)
        xT, mT = onehot_encode(row["TCR_full_norm"], self.L_T)
        xP, mP = onehot_encode(row["Peptide_norm"], self.L_P)
        xH, mH = onehot_encode(row["HLA_sequence_norm"], self.L_H)

        return {
            "emb_T": xT,
            "mask_T": mT,
            "emb_P": xP,
            "mask_P": mP,
            "emb_H": xH,
            "mask_H": mH,
            "z": z,
            "z_mask": z_mask,
            "binding_flag": int(row["binding_flag"]),
            "pair_id": pid,
            "peptide": str(row["peptide_for_eval"]),
            "has_alpha": bool(row["has_alpha"]),
            "has_beta": bool(row["has_beta"]),
            "tcra_len": int(row["tcra_len"]),
            "tcrb_len": int(row["tcrb_len"]),
            "tcr_total_len": int(row["tcr_total_len"]),
            "pep_len": int(row["pep_len"]),
            "hla_len": int(row["hla_len"]),
        }


def compute_max_z(metas: List[pd.DataFrame], cap_tcr: int, cap_pep: int, cap_hla: int):
    all_meta = pd.concat(metas, axis=0, ignore_index=True)
    L_T = int(all_meta["tcr_total_len"].max())
    L_P = int(all_meta["pep_len"].max())
    L_H = int(all_meta["hla_len"].max())
    if cap_tcr > 0:
        L_T = min(L_T, cap_tcr)
    if cap_pep > 0:
        L_P = min(L_P, cap_pep)
    if cap_hla > 0:
        L_H = min(L_H, cap_hla)
    if min(L_T, L_P, L_H) <= 0:
        raise ValueError(f"Invalid lengths: L_T={L_T}, L_P={L_P}, L_H={L_H}")
    L_z_max = L_T + L_P + L_H
    return L_z_max


def onehot_collate(rows: List[Dict]) -> Dict:
    tensor_keys = ["emb_T", "mask_T", "emb_P", "mask_P", "emb_H", "mask_H"]
    out = {k: torch.stack([r[k] for r in rows], dim=0) for k in tensor_keys}
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "tcr_total_len", "pep_len", "hla_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


def seq_plus_boltz_collate(rows: List[Dict]) -> Dict:
    """Stack fixed-size one-hot fields; leave variable-size z as a list for now."""
    out = onehot_collate(rows)
    out["z"] = torch.stack([r["z"] for r in rows], dim=0) # padded
    out["z_mask"] = torch.stack([r["z_mask"] for r in rows])
    # also probably have to change this? 
    return out


# parameters 
# B     = bath size
# L_T   = padded TCR length
# L_P   = padded peptide length
# L_H   = padded HLA length
# L     = L_T + L_P + L_H
# d_in  = input node dimension (21 for one hot composition, 960 for ESM)
# d_z   = boltz pairwise tensor dimension (128)
# d_h   = hidden node dimension
# d_out = VICReg projection dimension

# example calling the data

# define necessary parameters before the networks
train, audit     = load_meta(train_csv, "train", positives_only=True, missing_chain_policy="complete_only")
pair_id_to_npz  = make_pair_id_to_npz(train, REPO)
L_T, L_P, L_H   = compute_max_lengths([train], 300, 20, 400)
L_z_max         = compute_max_z([train], 300, 20, 400)

# do for val, test and immrep too - later on!


# build algorithm with one example
ds = SeqPlusBoltzDataset(train, pair_id_to_npz, L_T, L_P, L_H, L_z_max)
ex = ds[0]
# run the collate functions
rows = [ds[i] for i in range(8)]
batch = seq_plus_boltz_collate(rows)

print(batch["z"].shape)

# print(len(ds))
# print(ex.keys())
# print(ex["emb_T"].shape, ex["emb_P"].shape, ex["emb_H"].shape)
# print(ex["z"])

# for one batch
# message passing
# define algorithm and then VICReg functions below

# will go inside a class

x_tcr           = batch["emb_T"]
x_peptide       = batch["emb_P"]
x_hla           = batch["emb_H"]
mask_tcr        = batch["mask_T"]
mask_peptide    = batch["mask_P"]
mask_hla        = batch["mask_H"]
z               = batch["z"]

# concatenate the graph nodes ()
x_all = torch.cat(
    [x_tcr, x_peptide, x_hla],
    dim=1,
)

token_mask = torch.cat(
    [mask_tcr, mask_peptide, mask_hla],
    dim=1,
)

# add node type information - is it a TCR, a peptide or an HLA
self.component_
