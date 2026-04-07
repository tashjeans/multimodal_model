#!/usr/bin/env python3
"""
Prepare baseline inputs for downstream classification.


-----------------------------
This script prepares data for comparing two models, prepping for KNN:
1. plain ESM
2. fine-tuned ESM

What this script produces
-------------------------
For each split (train / val / test), we save:

A) plain ESM pair vectors
B) fine-tuned ESM pair vectors

Each saved file contains:
- X: the feature matrix
- y: the labels
- pair_ids: the pair identifiers

These files can then be loaded to build the KNN models
"""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, EsmModel


# ============================================================
# PATHS
# I only need to change paths here.
# ============================================================

PROJECT_ROOT = Path("/home/natasha/multimodal_model")

TRAIN_CSV = PROJECT_ROOT / "data/train/train_df_clean.csv"
VAL_CSV = PROJECT_ROOT / "data/val/val_df_clean_pos_neg.csv"
TEST_CSV = PROJECT_ROOT / "data/test/test_df_clean_pos_neg.csv"

FINETUNED_EMBED_ROOT = PROJECT_ROOT / "models/embeddings/no_boltz"

OUTPUT_DIR = Path("./prepared_baseline_inputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLAIN_ESM_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42


# ============================================================
# COLUMN NAMES
# If my CSV headers are different, I change them here.
# ============================================================

@dataclass
class ColumnNames:
    pair_id: str = "pair_id"
    label: str = "binding_flag"
    tcra: str = "TCRA"
    tcrb: str = "TCRB"
    peptide: str = "Peptide"
    hla: str = "HLA"


COLS = ColumnNames()


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SMALL HELPERS
# ============================================================

def clean_sequence(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def safe_mean_pool(emb, mask):
    """
    emb  = [L, D]
    mask = [L]

    I mean-pool only over valid positions.
    """
    mask = mask.astype(np.float32)
    denom = max(mask.sum(), 1.0)
    return (emb * mask[:, None]).sum(axis=0) / denom


# ============================================================
# RAW CSV DATASET
# I use this for the plain ESM side.
# ============================================================

class PairSequenceDataset(Dataset):
    def __init__(self, csv_path, cols: ColumnNames):
        self.df = pd.read_csv(csv_path).copy()
        self.cols = cols

        required = [
            cols.pair_id,
            cols.label,
            cols.tcra,
            cols.tcrb,
            cols.peptide,
            cols.hla,
        ]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")

        for c in [cols.tcra, cols.tcrb, cols.peptide, cols.hla]:
            self.df[c] = self.df[c].apply(clean_sequence)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "pair_id": str(row[self.cols.pair_id]),
            "binding_flag": int(row[self.cols.label]),
            "tcra": row[self.cols.tcra],
            "tcrb": row[self.cols.tcrb],
            "peptide": row[self.cols.peptide],
            "hla": row[self.cols.hla],
        }


# ============================================================
# PLAIN ESM EMBEDDER
# I use this to generate embeddings directly from the original ESM model.
# ============================================================

class PlainESMEmbedder:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

        print(f"Loading plain ESM model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def embed_sequence(self, seq: str):
        """
        I return one fixed vector for one sequence.

        If the sequence is empty, I return zeros of the correct size.
        """
        hidden_size = self.model.config.hidden_size

        if seq == "":
            return np.zeros(hidden_size, dtype=np.float32)

        toks = self.tokenizer(
            seq,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
        )
        toks = {k: v.to(self.device) for k, v in toks.items()}

        out = self.model(**toks)
        x = out.last_hidden_state[0]
        attn = toks["attention_mask"][0]

        # I remove special tokens if they are present.
        if x.shape[0] > 2:
            x = x[1:-1]
            attn = attn[1:-1]

        x = x.detach().cpu().numpy()
        attn = attn.detach().cpu().numpy().astype(np.float32)

        denom = max(attn.sum(), 1.0)
        pooled = (x * attn[:, None]).sum(axis=0) / denom
        return pooled.astype(np.float32)

    def embed_pair(self, tcra: str, tcrb: str, peptide: str, hla: str):
        """
        I create one pair-level vector as:

        TCR vector  = average of alpha and beta embeddings
        Final vector = [TCR | peptide | HLA]

        This gives the student one fixed vector per pair.
        """
        a = self.embed_sequence(tcra)
        b = self.embed_sequence(tcrb)
        p = self.embed_sequence(peptide)
        h = self.embed_sequence(hla)

        tcr = 0.5 * (a + b)
        pair_vec = np.concatenate([tcr, p, h], axis=0).astype(np.float32)
        return pair_vec


# ============================================================
# FINE-TUNED SHARD LOADER
# I use this to load the saved fine-tuned embeddings.
# This matches the storage style from my baseline pipeline.
# ============================================================

class ShardedFineTunedEmbeddingDataset(Dataset):
    def __init__(self, shard_dir):
        self.shard_dir = Path(shard_dir)
        self.shard_paths = sorted(self.shard_dir.glob("shard_*.pt"))
        if not self.shard_paths:
            raise FileNotFoundError(f"No shard_*.pt files found in {self.shard_dir}")

        self.index = []
        for sp in self.shard_paths:
            shard = torch.load(sp, map_location="cpu")
            for j in range(len(shard)):
                self.index.append((sp, j))

        self._cached_path = None
        self._cached_data = None

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        sp, j = self.index[idx]
        if self._cached_path != sp:
            self._cached_data = torch.load(sp, map_location="cpu")
            self._cached_path = sp
        return self._cached_data[j]


def pair_vector_from_finetuned_item(item):
    """
    I convert one saved fine-tuned example into one fixed vector.

    The shard item already contains token-level embeddings and masks:
    - emb_T, mask_T
    - emb_P, mask_P
    - emb_H, mask_H

    I mean-pool each one and then concatenate:
    [TCR | peptide | HLA]
    """
    emb_T = item["emb_T"].cpu().numpy()
    emb_P = item["emb_P"].cpu().numpy()
    emb_H = item["emb_H"].cpu().numpy()

    mask_T = item["mask_T"].cpu().numpy().astype(np.float32)
    mask_P = item["mask_P"].cpu().numpy().astype(np.float32)
    mask_H = item["mask_H"].cpu().numpy().astype(np.float32)

    tcr = safe_mean_pool(emb_T, mask_T)
    pep = safe_mean_pool(emb_P, mask_P)
    hla = safe_mean_pool(emb_H, mask_H)

    pair_vec = np.concatenate([tcr, pep, hla], axis=0).astype(np.float32)

    return {
        "pair_id": str(item["pair_id"]),
        "binding_flag": int(item["binding_flag"]),
        "x": pair_vec,
    }


# ============================================================
# EXTRACTION FUNCTIONS
# ============================================================

def extract_plain_esm_split(csv_path, out_npz, cols: ColumnNames, model_name: str, device: str):
    """
    I read one CSV split and save plain ESM pair vectors.
    """
    dataset = PairSequenceDataset(csv_path, cols)
    embedder = PlainESMEmbedder(model_name, device)

    X = []
    y = []
    pair_ids = []

    print(f"\nExtracting plain ESM features from: {csv_path}")
    for i in range(len(dataset)):
        item = dataset[i]

        x = embedder.embed_pair(
            item["tcra"],
            item["tcrb"],
            item["peptide"],
            item["hla"],
        )

        X.append(x)
        y.append(item["binding_flag"])
        pair_ids.append(item["pair_id"])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(dataset)}")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    pair_ids = np.asarray(pair_ids)

    np.savez_compressed(out_npz, X=X, y=y, pair_ids=pair_ids)
    print(f"Saved: {out_npz}")
    print(f"Shape: X={X.shape}, y={y.shape}")


def extract_finetuned_split(shard_dir, out_npz):
    """
    I read one fine-tuned shard split and save pair vectors.
    """
    dataset = ShardedFineTunedEmbeddingDataset(shard_dir)

    X = []
    y = []
    pair_ids = []

    print(f"\nExtracting fine-tuned features from: {shard_dir}")
    for i in range(len(dataset)):
        item = dataset[i]
        out = pair_vector_from_finetuned_item(item)

        X.append(out["x"])
        y.append(out["binding_flag"])
        pair_ids.append(out["pair_id"])

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} / {len(dataset)}")

    X = np.stack(X).astype(np.float32)
    y = np.asarray(y, dtype=np.int64)
    pair_ids = np.asarray(pair_ids)

    np.savez_compressed(out_npz, X=X, y=y, pair_ids=pair_ids)
    print(f"Saved: {out_npz}")
    print(f"Shape: X={X.shape}, y={y.shape}")


# ============================================================
# OPTIONAL LOAD EXAMPLE
# I include this so the student can see how to load the output.
# ============================================================

def show_example_load(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    pair_ids = data["pair_ids"]

    print(f"\nExample load from: {npz_path}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"pair_ids shape: {pair_ids.shape}")
    print(f"First pair_id: {pair_ids[0]}")
    print(f"First label: {y[0]}")
    print(f"Feature dimension: {X.shape[1]}")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # --------------------------------------------------------
    # Plain ESM outputs
    # --------------------------------------------------------
    plain_train_out = OUTPUT_DIR / "plain_esm_train.npz"
    plain_val_out = OUTPUT_DIR / "plain_esm_val.npz"
    plain_test_out = OUTPUT_DIR / "plain_esm_test.npz"

    if not plain_train_out.exists():
        extract_plain_esm_split(TRAIN_CSV, plain_train_out, COLS, PLAIN_ESM_MODEL_NAME, DEVICE)
    if not plain_val_out.exists():
        extract_plain_esm_split(VAL_CSV, plain_val_out, COLS, PLAIN_ESM_MODEL_NAME, DEVICE)
    if not plain_test_out.exists():
        extract_plain_esm_split(TEST_CSV, plain_test_out, COLS, PLAIN_ESM_MODEL_NAME, DEVICE)

    # --------------------------------------------------------
    # Fine-tuned ESM outputs
    # --------------------------------------------------------
    ft_train_out = OUTPUT_DIR / "finetuned_esm_train.npz"
    ft_val_out = OUTPUT_DIR / "finetuned_esm_val.npz"
    ft_test_out = OUTPUT_DIR / "finetuned_esm_test.npz"

    if not ft_train_out.exists():
        extract_finetuned_split(FINETUNED_EMBED_ROOT / "train", ft_train_out)
    if not ft_val_out.exists():
        extract_finetuned_split(FINETUNED_EMBED_ROOT / "val", ft_val_out)
    if not ft_test_out.exists():
        extract_finetuned_split(FINETUNED_EMBED_ROOT / "test", ft_test_out)

    # --------------------------------------------------------
    # I show one example of how the saved outputs can be loaded.
    # --------------------------------------------------------
    show_example_load(plain_train_out)
    show_example_load(ft_train_out)

    print("\nDone.")
    print("The student can now load these .npz files and build their own classifier.")


if __name__ == "__main__":
    main()