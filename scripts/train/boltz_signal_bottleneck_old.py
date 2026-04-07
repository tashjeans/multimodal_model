#!/usr/bin/env python3
"""
Signal-aware structured Boltz bottleneck operator model (v2)
============================================================

Design goals
------------
1. Keep the manifest-driven file discovery and filtering logic.
2. Learn a compact latent theta that is itself signal-bearing.
3. Construct residual Z* deterministically from theta.
4. Preserve structure by supervising on cheap, blockwise, signal-bearing summaries
   of the symmetrised channel-averaged raw Boltz matrix.
5. Save intermediate checkpoints and preview exports during training.

This is NOT a raw autoencoder.
It is a structured latent operator model with losses on:
- latent feature prediction
- Z* feature matching
- binding classification from theta
- latent spread/decorrelation
- Z* scale control

Notes on how to run in tmux:
----------------------------
tmux new -s boltz_signal_v2
conda activate tcr-multimodal
cd /home/natasha/multimodal_model/scripts/train
python boltz_signal_bottleneck.py 2>&1 | tee boltz_signal_bottleneck.log
"""

import os
import time
import glob
import math
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

@dataclass
class Config:
    base_path: str = "/home/natasha/multimodal_model"
    train_manifest: str = "/home/natasha/multimodal_model/manifests/train_manifest.csv"
    val_manifest: str = "/home/natasha/multimodal_model/manifests/val_manifest.csv"
    test_manifest: str = "/home/natasha/multimodal_model/manifests/test_manifest.csv"

    checkpoint_dir: str = "/home/natasha/multimodal_model/models/boltz_signal_bottleneck_checkpoints"
    latent_dir: str = "/home/natasha/multimodal_model/models/embeddings/boltz_signal_bottleneck_latents"
    preview_dir: str = "/home/natasha/multimodal_model/models/embeddings/boltz_signal_bottleneck_previews"
    feature_cache_dir: str = "/home/natasha/multimodal_model/models/feature_cache/boltz_signal_targets"

    batch_size: int = 4
    accum_steps: int = 4
    num_workers: int = 4
    max_pad_len: int = 704
    patch_size: int = 8

    max_epochs: int = 10
    min_epochs: int = 3
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    grad_clip: float = 1.0
    early_stop_patience: int = 3
    min_delta: float = 1e-4
    seed: int = 42

    enc_channels: int = 64
    num_res_blocks: int = 3
    dropout: float = 0.10

    zstar_d: int = 128
    zstar_rank: int = 8
    reg_proj_dim: int = 256

    # Loss weights
    lambda_latent_feat: float = 2.0
    lambda_zstar_feat: float = 2.0
    lambda_bind: float = 0.50
    lambda_var: float = 1.0
    lambda_cov: float = 1.0
    lambda_theta_l2: float = 1e-4
    lambda_zstar_scale: float = 1e-4

    var_target_std: float = 1.0

    save_epoch_checkpoints: bool = True
    export_preview_every: int = 1
    preview_split: str = "val"
    preview_max_batches: int = 64
    export_full_valtest_on_best: bool = True


CFG = Config()

TARGET_FEATURE_NAMES = [
    "A_diag_mean",
    "A_offdiag_mean",
    "B_diag_mean",
    "B_offdiag_mean",
    "C_mean",
    "trace_mean",
]

TARGET_FEATURE_WEIGHTS = {
    "A_diag_mean": 1.0,
    "A_offdiag_mean": 2.0,
    "B_diag_mean": 1.0,
    "B_offdiag_mean": 2.0,
    "C_mean": 2.0,
    "trace_mean": 1.5,
}


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None, keepdim=False, eps: float = 1e-6) -> torch.Tensor:
    w = mask.to(dtype=x.dtype)
    num = (x * w).sum(dim=dim, keepdim=keepdim)
    den = w.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def strict_upper_mean_np(mat: np.ndarray) -> float:
    n = mat.shape[0]
    if n <= 1:
        return 0.0
    iu = np.triu_indices(n, k=1)
    vals = mat[iu]
    if vals.size == 0:
        return 0.0
    return float(vals.mean())


def strict_upper_mean_torch(mat: torch.Tensor) -> torch.Tensor:
    n = mat.shape[-1]
    if n <= 1:
        return torch.zeros(mat.shape[:-2], device=mat.device, dtype=mat.dtype)
    iu = torch.triu_indices(n, n, offset=1, device=mat.device)
    vals = mat[..., iu[0], iu[1]]
    return vals.mean(dim=-1)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * weights.view(1, -1)
    return loss.mean()


# ============================================================
# CHEAP RAW TARGET FEATURE EXTRACTION
# ============================================================

def compute_raw_target_features_fast(
    z: np.ndarray,
    pep_len: int,
    tcra_len: int,
    tcrb_len: int,
    hla_len: int,
) -> Dict[str, float]:
    """
    Fast feature extraction:
      1. symmetrise channelwise
      2. average over channels to get S_avg in R^{L x L}
      3. compute blockwise diag/offdiag/cross/trace summaries on S_avg

    This is materially cheaper than looping over channels.
    """
    if z.ndim != 3:
        raise ValueError(f"Expected z ndim=3, got {z.shape}")

    z = z.astype(np.float32)
    z_sym = 0.5 * (z + np.transpose(z, (1, 0, 2)))
    S_avg = z_sym.mean(axis=-1)  # [L, L]

    L_pad = S_avg.shape[0]
    L_T = int(tcra_len + tcrb_len)
    L_PH = int(pep_len + hla_len)
    L_total = min(L_T + L_PH, L_pad)

    if L_total <= 0:
        raise ValueError("Computed non-positive L_total from lengths")

    L_T = min(L_T, L_total)
    L_PH = min(L_PH, L_total - L_T)

    S_avg = S_avg[:L_total, :L_total]

    if L_T > 0:
        A = S_avg[:L_T, :L_T]
        A_diag_mean = float(np.diag(A).mean())
        A_offdiag_mean = strict_upper_mean_np(A)
    else:
        A_diag_mean = 0.0
        A_offdiag_mean = 0.0

    if L_PH > 0:
        B = S_avg[L_T:L_T + L_PH, L_T:L_T + L_PH]
        B_diag_mean = float(np.diag(B).mean())
        B_offdiag_mean = strict_upper_mean_np(B)
    else:
        B_diag_mean = 0.0
        B_offdiag_mean = 0.0

    if L_T > 0 and L_PH > 0:
        C = S_avg[:L_T, L_T:L_T + L_PH]
        C_mean = float(C.mean())
    else:
        C_mean = 0.0

    trace_mean = float(np.trace(S_avg) / max(L_total, 1))

    return {
        "A_diag_mean": A_diag_mean,
        "A_offdiag_mean": A_offdiag_mean,
        "B_diag_mean": B_diag_mean,
        "B_offdiag_mean": B_offdiag_mean,
        "C_mean": C_mean,
        "trace_mean": trace_mean,
    }


def build_or_load_feature_cache(
    manifest_df: pd.DataFrame,
    cache_csv_path: str,
) -> pd.DataFrame:
    os.makedirs(os.path.dirname(cache_csv_path), exist_ok=True)

    if os.path.exists(cache_csv_path):
        cached = pd.read_csv(cache_csv_path)
        needed = ["pair_id"] + TARGET_FEATURE_NAMES
        missing = [c for c in needed if c not in cached.columns]
        if len(missing) == 0:
            merged = manifest_df.merge(cached[needed], on="pair_id", how="left")
            if not merged[TARGET_FEATURE_NAMES].isna().any().any():
                print(f"[feature cache] loaded {cache_csv_path}")
                return merged
            print(f"[feature cache] incomplete cache; rebuilding {cache_csv_path}")

    print(f"[feature cache] building {cache_csv_path}")
    rows = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="feature cache"):
        with np.load(row["emb_path"]) as arr:
            z = arr["z"]

        z = np.squeeze(z, axis=0) if z.ndim == 4 and z.shape[0] == 1 else z

        feats = compute_raw_target_features_fast(
            z=z,
            pep_len=int(row["pep_len"]),
            tcra_len=int(row["tcra_len"]),
            tcrb_len=int(row["tcrb_len"]),
            hla_len=int(row["hla_len"]),
        )
        feats["pair_id"] = str(row["pair_id"])
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(cache_csv_path, index=False)
    print(f"[feature cache] saved {cache_csv_path}")

    return manifest_df.merge(feat_df, on="pair_id", how="left")


# ============================================================
# DATASET
# ============================================================

class BoltzDataset(Dataset):
    def __init__(self, manifest_path, base_path, split_dir, feature_cache_dir, strict=False):
        self.base_path = base_path
        self.split_dir = split_dir
        manifest = pd.read_csv(manifest_path)

        required_cols = ["pair_id", "pep_len", "tcra_len", "tcrb_len", "hla_len", "binding_flag"]
        missing_cols = [c for c in required_cols if c not in manifest.columns]
        if missing_cols:
            raise ValueError(f"Manifest {manifest_path} missing columns: {missing_cols}")

        valid_rows, missing = [], 0
        for i in range(len(manifest)):
            row = manifest.iloc[i].copy()
            pair_id = str(row["pair_id"])
            emb_path = self._find_embedding_path(pair_id)
            if emb_path is not None:
                row["pair_id"] = pair_id
                row["emb_path"] = emb_path
                row["split_dir"] = self.split_dir
                row["manifest_idx"] = i
                valid_rows.append(row)
            else:
                missing += 1
                if strict:
                    raise FileNotFoundError(f"Missing embedding for pair_id={pair_id}")

        self.manifest = pd.DataFrame(valid_rows).reset_index(drop=True)
        print(f"[BoltzDataset:{split_dir}] kept {len(self.manifest)}, filtered {missing} missing")

        cache_csv = os.path.join(feature_cache_dir, f"{split_dir}_raw_target_features.csv")
        self.manifest = build_or_load_feature_cache(self.manifest, cache_csv)

        if self.manifest[TARGET_FEATURE_NAMES].isna().any().any():
            bad = self.manifest[self.manifest[TARGET_FEATURE_NAMES].isna().any(axis=1)][["pair_id"]]
            raise ValueError(f"Target feature cache incomplete for split={split_dir}; examples:\n{bad.head()}")

    def _find_embedding_path(self, pair_id):
        pattern = os.path.join(
            self.base_path, "outputs", self.split_dir, "chunk_*",
            f"boltz_results_{pair_id}", "predictions", pair_id,
            f"embeddings_{pair_id}.npz",
        )
        matches = glob.glob(pattern)
        if len(matches) == 0:
            return None
        if len(matches) > 1:
            raise RuntimeError(f"Multiple matches for {pair_id}: {matches}")
        return matches[0]

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        with np.load(row["emb_path"]) as arr:
            z = arr["z"]

        z = np.squeeze(z, axis=0) if z.ndim == 4 and z.shape[0] == 1 else z
        if z.ndim != 3:
            raise ValueError(f"Unexpected z shape: {z.shape}")

        L_pad = z.shape[0]
        pep_len = int(row["pep_len"])
        tcra_len = int(row["tcra_len"])
        tcrb_len = int(row["tcrb_len"])
        hla_len = int(row["hla_len"])

        L_T = tcra_len + tcrb_len
        L_PH = pep_len + hla_len
        L_total = L_T + L_PH
        if L_total > L_pad:
            L_T_new = min(L_T, L_pad)
            remaining = L_pad - L_T_new
            L_PH_new = min(L_PH, remaining)
            tcra_len = min(tcra_len, L_T_new)
            tcrb_len = min(tcrb_len, L_T_new - tcra_len)
            pep_len = min(pep_len, L_PH_new)
            hla_len = min(hla_len, L_PH_new - pep_len)
            L_total = tcra_len + tcrb_len + pep_len + hla_len

        target_features = np.array([float(row[name]) for name in TARGET_FEATURE_NAMES], dtype=np.float32)

        return {
            "z": z.astype(np.float32),
            "orig_len": int(min(L_total, z.shape[0])),
            "pep_len": pep_len,
            "tcra_len": tcra_len,
            "tcrb_len": tcrb_len,
            "hla_len": hla_len,
            "pair_id": str(row["pair_id"]),
            "binding_flag": int(row["binding_flag"]),
            "split_dir": row["split_dir"],
            "manifest_idx": int(row["manifest_idx"]),
            "emb_path": row["emb_path"],
            "target_features": target_features,
        }


def boltz_collate_fn(batch):
    zs = []
    for item in batch:
        z = item["z"]
        if z.ndim == 4 and z.shape[0] == 1:
            z = z[0]
        elif z.ndim != 3:
            raise ValueError(f"Unexpected z ndim={z.ndim}")
        zs.append(z)

    max_len_in_batch = max(z.shape[0] for z in zs)
    max_len_in_batch = min(max_len_in_batch, CFG.max_pad_len)
    padded_len = int(math.ceil(max_len_in_batch / CFG.patch_size) * CFG.patch_size)
    padded_len = min(padded_len, CFG.max_pad_len)

    d_in = zs[0].shape[-1]
    padded = np.zeros((len(zs), padded_len, padded_len, d_in), dtype=np.float32)
    for i, z in enumerate(zs):
        L = min(z.shape[0], padded_len)
        padded[i, :L, :L, :] = z[:L, :L, :]

    return {
        "z": torch.from_numpy(padded).float(),
        "orig_len": torch.tensor([min(it["orig_len"], padded_len) for it in batch], dtype=torch.long),
        "pep_len": torch.tensor([it["pep_len"] for it in batch], dtype=torch.long),
        "tcra_len": torch.tensor([it["tcra_len"] for it in batch], dtype=torch.long),
        "tcrb_len": torch.tensor([it["tcrb_len"] for it in batch], dtype=torch.long),
        "hla_len": torch.tensor([it["hla_len"] for it in batch], dtype=torch.long),
        "pair_id": [it["pair_id"] for it in batch],
        "binding_flag": torch.tensor([it["binding_flag"] for it in batch], dtype=torch.float32),
        "split_dir": [it["split_dir"] for it in batch],
        "manifest_idx": torch.tensor([it["manifest_idx"] for it in batch], dtype=torch.long),
        "emb_path": [it["emb_path"] for it in batch],
        "target_features": torch.from_numpy(np.stack([it["target_features"] for it in batch], axis=0)).float(),
    }


# ============================================================
# MODEL
# ============================================================

class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(self.dropout(F.gelu(self.norm2(x))))
        return residual + x


class SignalAwareOperatorBottleneck(nn.Module):
    """
    Encoder -> theta
            -> latent feature head
            -> binding head
            -> deterministic symmetric residual Z*
            -> Z* feature computation
    """
    def __init__(
        self,
        d_in: int,
        enc_channels: int,
        patch_size: int,
        num_res_blocks: int,
        dropout: float,
        max_pad_len: int,
        zstar_d: int,
        zstar_rank: int,
        reg_proj_dim: int,
        target_feature_dim: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.enc_channels = enc_channels
        self.patch_size = patch_size
        self.max_pad_len = max_pad_len
        self.zstar_d = zstar_d
        self.zstar_rank = zstar_rank
        self.target_feature_dim = target_feature_dim

        # theta parameterisation:
        #   A_diag, B_diag, C_diag: 3*d
        #   A_U, A_V, B_U, B_V, C_U, C_V: 6*d*r
        self.operator_param_dim = 3 * zstar_d + 6 * zstar_d * zstar_rank

        self.patch_embed = nn.Conv2d(d_in, enc_channels, kernel_size=patch_size, stride=patch_size)
        self.encoder_blocks = nn.Sequential(*[ResidualBlock2D(enc_channels, dropout) for _ in range(num_res_blocks)])
        self.enc_norm = nn.GroupNorm(8, enc_channels)

        self.to_theta = nn.Sequential(
            nn.Linear(2 * enc_channels, 4 * enc_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * enc_channels, self.operator_param_dim),
        )

        reg_proj = torch.randn(self.operator_param_dim, reg_proj_dim) / math.sqrt(self.operator_param_dim)
        self.register_buffer("reg_proj", reg_proj)

        self.latent_feature_head = nn.Sequential(
            nn.Linear(self.operator_param_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, target_feature_dim),
        )

        self.binding_head = nn.Sequential(
            nn.Linear(self.operator_param_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.register_buffer("target_mean", torch.zeros(target_feature_dim))
        self.register_buffer("target_std", torch.ones(target_feature_dim))
        self.register_buffer(
            "target_loss_weights",
            torch.tensor([TARGET_FEATURE_WEIGHTS[n] for n in TARGET_FEATURE_NAMES], dtype=torch.float32),
        )

    def set_target_normaliser(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.detach())
        self.target_std.copy_(std.detach().clamp_min(1e-6))

    @staticmethod
    def build_valid_pair_mask(orig_len: torch.Tensor, L_pad: int, device: torch.device) -> torch.Tensor:
        idx = torch.arange(L_pad, device=device).unsqueeze(0)
        tok = (idx < orig_len.unsqueeze(1)).float()
        pair = tok.unsqueeze(1) * tok.unsqueeze(2)
        return pair.unsqueeze(1)

    def encode(self, z_boltz: torch.Tensor, orig_len: torch.Tensor):
        B, L_pad, _, d_in = z_boltz.shape
        assert d_in == self.d_in, f"Expected input channels {self.d_in}, got {d_in}"

        pair_mask = self.build_valid_pair_mask(orig_len, L_pad, z_boltz.device)

        x = z_boltz.permute(0, 3, 1, 2)
        x = x * pair_mask

        patch_mask = F.avg_pool2d(pair_mask, kernel_size=self.patch_size, stride=self.patch_size)
        patch_mask = (patch_mask > 0).to(dtype=x.dtype)

        x = self.patch_embed(x)
        x = x * patch_mask
        x = self.encoder_blocks(x)
        x = self.enc_norm(x)
        x = x * patch_mask

        pooled_mean = masked_mean(x, patch_mask, dim=(2, 3), keepdim=False)
        centered = (x - pooled_mean[:, :, None, None]) * patch_mask
        pooled_var = masked_mean(centered.pow(2), patch_mask, dim=(2, 3), keepdim=False)
        pooled_std = torch.sqrt(pooled_var + 1e-5)

        pooled = torch.cat([pooled_mean, pooled_std], dim=1)
        theta = self.to_theta(pooled)
        reg_code = theta @ self.reg_proj
        return theta, reg_code

    def _split_theta(self, theta: torch.Tensor):
        B = theta.shape[0]
        d = self.zstar_d
        r = self.zstar_rank

        offset = 0
        A_diag = theta[:, offset:offset + d]; offset += d
        B_diag = theta[:, offset:offset + d]; offset += d
        C_diag = theta[:, offset:offset + d]; offset += d

        def take_mat():
            nonlocal offset
            U = theta[:, offset:offset + d * r].view(B, d, r)
            offset += d * r
            V = theta[:, offset:offset + d * r].view(B, d, r)
            offset += d * r
            return U, V

        A_U, A_V = take_mat()
        B_U, B_V = take_mat()
        C_U, C_V = take_mat()

        assert offset == theta.shape[1], f"Theta split mismatch: offset={offset}, total={theta.shape[1]}"
        return A_diag, B_diag, C_diag, A_U, A_V, B_U, B_V, C_U, C_V

    def latent_to_zstar(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Build symmetric residual Z* of shape [B, 2d, 2d].
        Symmetry is by construction. There is no separate symmetry loss.
        """
        Bsz = theta.shape[0]
        d = self.zstar_d
        r = self.zstar_rank
        dtype = theta.dtype
        device = theta.device

        A_diag, B_diag, C_diag, A_U, A_V, B_U, B_V, C_U, C_V = self._split_theta(theta)

        A = torch.matmul(A_U, A_V.transpose(-1, -2)) / math.sqrt(r)
        Bblk = torch.matmul(B_U, B_V.transpose(-1, -2)) / math.sqrt(r)
        C = torch.matmul(C_U, C_V.transpose(-1, -2)) / math.sqrt(r)

        A = 0.5 * (A + A.transpose(1, 2))
        Bblk = 0.5 * (Bblk + Bblk.transpose(1, 2))

        eye = torch.eye(d, device=device, dtype=dtype).unsqueeze(0)
        A = A + eye * A_diag.unsqueeze(-1)
        Bblk = Bblk + eye * B_diag.unsqueeze(-1)
        C = C + eye * C_diag.unsqueeze(-1)

        zstar_residual = torch.zeros(Bsz, 2 * d, 2 * d, device=device, dtype=dtype)
        zstar_residual[:, :d, :d] = A
        zstar_residual[:, :d, d:] = C
        zstar_residual[:, d:, :d] = C.transpose(1, 2)
        zstar_residual[:, d:, d:] = Bblk
        zstar_residual = 0.5 * (zstar_residual + zstar_residual.transpose(1, 2))
        return zstar_residual

    def add_identity_to_zstar(self, zstar_residual: torch.Tensor) -> torch.Tensor:
        """
        Construct full Z* by adding identity to all four blocks,
        matching your old downstream convention.
        """
        B, D, _ = zstar_residual.shape
        d = D // 2
        device = zstar_residual.device
        dtype = zstar_residual.dtype

        I = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        zstar = zstar_residual.clone()
        zstar[:, :d, :d] += I
        zstar[:, :d, d:] += I
        zstar[:, d:, :d] += I
        zstar[:, d:, d:] += I
        zstar = 0.5 * (zstar + zstar.transpose(1, 2))
        return zstar

    def zstar_features(self, zstar_residual: torch.Tensor) -> torch.Tensor:
        """
        Compute the same blockwise feature family on residual Z*.
        """
        d = self.zstar_d
        A = zstar_residual[:, :d, :d]
        C = zstar_residual[:, :d, d:]
        Bblk = zstar_residual[:, d:, d:]

        A_diag_mean = torch.diagonal(A, dim1=1, dim2=2).mean(dim=1)
        B_diag_mean = torch.diagonal(Bblk, dim1=1, dim2=2).mean(dim=1)

        A_offdiag_mean = strict_upper_mean_torch(A)
        B_offdiag_mean = strict_upper_mean_torch(Bblk)

        C_mean = C.mean(dim=(1, 2))
        trace_mean = torch.diagonal(zstar_residual, dim1=1, dim2=2).mean(dim=1)

        feats = torch.stack(
            [A_diag_mean, A_offdiag_mean, B_diag_mean, B_offdiag_mean, C_mean, trace_mean],
            dim=1,
        )
        return feats

    def zstar_scale_penalty(self, zstar_residual: torch.Tensor) -> torch.Tensor:
        """
        Cheap operator regulariser to stop pathological scale/spectrum growth.
        """
        d = self.zstar_d
        A = zstar_residual[:, :d, :d]
        C = zstar_residual[:, :d, d:]
        Bblk = zstar_residual[:, d:, d:]

        penalty = (
            A.pow(2).mean()
            + Bblk.pow(2).mean()
            + C.pow(2).mean()
        )
        return penalty

    def normalise_targets(self, target_raw: torch.Tensor) -> torch.Tensor:
        return (target_raw - self.target_mean.view(1, -1)) / self.target_std.view(1, -1)

    def normalise_feature_tensor(self, feat_raw: torch.Tensor) -> torch.Tensor:
        return (feat_raw - self.target_mean.view(1, -1)) / self.target_std.view(1, -1)

    def forward(self, z_boltz: torch.Tensor, orig_len: torch.Tensor):
        theta, reg_code = self.encode(z_boltz, orig_len)

        latent_feat_pred_norm = self.latent_feature_head(theta)
        binding_logit = self.binding_head(theta).squeeze(-1)

        zstar_residual = self.latent_to_zstar(theta)
        zstar_feat_raw = self.zstar_features(zstar_residual)
        zstar_feat_norm = self.normalise_feature_tensor(zstar_feat_raw)
        zstar_scale = self.zstar_scale_penalty(zstar_residual)

        return {
            "latent": theta,
            "reg_code": reg_code,
            "latent_feat_pred_norm": latent_feat_pred_norm,
            "binding_logit": binding_logit,
            "zstar_residual": zstar_residual,
            "zstar_feat_raw": zstar_feat_raw,
            "zstar_feat_norm": zstar_feat_norm,
            "zstar_scale": zstar_scale,
        }

    def encode_only(self, z_boltz: torch.Tensor, orig_len: torch.Tensor) -> torch.Tensor:
        theta, _ = self.encode(z_boltz, orig_len)
        return theta


# ============================================================
# LOSSES
# ============================================================

def latent_regularisation(reg_code, target_std=1.0):
    B, D = reg_code.shape
    if B < 2:
        zero = reg_code.new_tensor(0.0)
        return {"var": zero, "cov": zero, "total": zero}

    z = reg_code - reg_code.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
    var_loss = F.relu(target_std - std).mean()

    cov = (z.T @ z) / max(B - 1, 1)
    cov_loss = off_diagonal(cov).pow(2).mean()

    return {"var": var_loss, "cov": cov_loss, "total": var_loss + cov_loss}


def compute_losses(model, out, batch):
    target_raw = batch["target_features"].to(out["latent"].device, non_blocking=True)
    target_norm = model.normalise_targets(target_raw)

    latent_feat_loss = weighted_smooth_l1(
        out["latent_feat_pred_norm"], target_norm, model.target_loss_weights
    )

    zstar_feat_loss = weighted_smooth_l1(
        out["zstar_feat_norm"], target_norm, model.target_loss_weights
    )

    bind_target = batch["binding_flag"].to(out["binding_logit"].device, non_blocking=True)
    bind_loss = F.binary_cross_entropy_with_logits(out["binding_logit"], bind_target)

    reg_terms = latent_regularisation(out["reg_code"], target_std=CFG.var_target_std)
    theta_l2 = out["latent"].pow(2).mean()

    total = (
        CFG.lambda_latent_feat * latent_feat_loss
        + CFG.lambda_zstar_feat * zstar_feat_loss
        + CFG.lambda_bind * bind_loss
        + CFG.lambda_var * reg_terms["var"]
        + CFG.lambda_cov * reg_terms["cov"]
        + CFG.lambda_theta_l2 * theta_l2
        + CFG.lambda_zstar_scale * out["zstar_scale"]
    )

    return {
        "total": total,
        "latent_feat": latent_feat_loss,
        "zstar_feat": zstar_feat_loss,
        "bind": bind_loss,
        "var": reg_terms["var"],
        "cov": reg_terms["cov"],
        "theta_l2": theta_l2,
        "zstar_scale": out["zstar_scale"],
    }


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_epoch(model, loader, optimizer, scheduler, device, scaler, train: bool):
    model.train() if train else model.eval()

    running = {
        "loss": 0.0,
        "latent_feat": 0.0,
        "zstar_feat": 0.0,
        "bind": 0.0,
        "var": 0.0,
        "cov": 0.0,
        "theta_l2": 0.0,
        "zstar_scale": 0.0,
    }
    total_n = 0

    pbar = tqdm(loader, desc=("train" if train else "val  "), ncols=130, miniters=20)
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        z = batch["z"].to(device, non_blocking=True)
        orig_len = batch["orig_len"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(z, orig_len)
                loss_terms = compute_losses(model, out, batch)
                loss = loss_terms["total"]
                loss_to_backprop = loss / CFG.accum_steps

            if train:
                scaler.scale(loss_to_backprop).backward()
                if (step + 1) % CFG.accum_steps == 0 or (step + 1) == len(loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CFG.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        bs = z.shape[0]
        running["loss"] += loss.item() * bs
        for k in ["latent_feat", "zstar_feat", "bind", "var", "cov", "theta_l2", "zstar_scale"]:
            running[k] += loss_terms[k].item() * bs
        total_n += bs

        pbar.set_postfix(loss=f"{running['loss'] / max(total_n, 1):.4f}")

    return {k: v / max(total_n, 1) for k, v in running.items()}


# ============================================================
# EXPORTS / CHECKPOINTS
# ============================================================

@torch.no_grad()
def export_operator_latents(model, loader, device, out_dir, split_name, max_batches=None, include_zstar=False):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)

    all_latents = []
    all_pids, all_bf, all_sd, all_mi = [], [], [], []
    all_zstars = []

    n_seen_batches = 0
    for batch in tqdm(loader, desc=f"Export {split_name}"):
        if max_batches is not None and n_seen_batches >= max_batches:
            break

        z = batch["z"].to(device, non_blocking=True)
        orig_len = batch["orig_len"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            theta = model.encode_only(z, orig_len)

        all_latents.append(theta.float().cpu().numpy())
        all_pids.extend(batch["pair_id"])
        all_bf.extend(batch["binding_flag"].cpu().numpy().tolist())
        all_sd.extend(batch["split_dir"])
        all_mi.extend(batch["manifest_idx"].cpu().numpy().tolist())

        if include_zstar:
            zstar_res = model.latent_to_zstar(theta)
            all_zstars.append(zstar_res.float().cpu().numpy())

        n_seen_batches += 1

    latents = np.concatenate(all_latents, axis=0)
    meta = pd.DataFrame({
        "pair_id": all_pids,
        "binding_flag": all_bf,
        "split_dir": all_sd,
        "manifest_idx": all_mi,
        "latent_row": np.arange(len(all_pids), dtype=np.int64),
    })

    meta.to_csv(os.path.join(out_dir, f"{split_name}_metadata.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, f"{split_name}_latents.npz"), latent=latents)

    if include_zstar:
        zstars = np.concatenate(all_zstars, axis=0)
        np.savez_compressed(os.path.join(out_dir, f"{split_name}_zstar_residual.npz"), zstar_residual=zstars)

    print(f"[{split_name}] latent shape: {latents.shape}")


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, d_in):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "d_in": d_in,
        "config": asdict(CFG),
        "target_feature_names": TARGET_FEATURE_NAMES,
    }, path)


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(CFG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.latent_dir, exist_ok=True)
    os.makedirs(CFG.preview_dir, exist_ok=True)
    os.makedirs(CFG.feature_cache_dir, exist_ok=True)

    with open(os.path.join(CFG.checkpoint_dir, "boltz_signal_bottleneck_v2_config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("\n=== Loading datasets ===")
    train_ds = BoltzDataset(CFG.train_manifest, CFG.base_path, "train", CFG.feature_cache_dir)
    val_ds = BoltzDataset(CFG.val_manifest, CFG.base_path, "val", CFG.feature_cache_dir)
    test_ds = BoltzDataset(CFG.test_manifest, CFG.base_path, "test", CFG.feature_cache_dir)

    loader_kwargs = dict(
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        pin_memory=True,
        collate_fn=boltz_collate_fn,
        persistent_workers=(CFG.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    sample = next(iter(train_loader))
    actual_dB = sample["z"].shape[-1]
    print(f"\nDetected dB from data: {actual_dB}")

    train_feat_df = train_ds.manifest[TARGET_FEATURE_NAMES].copy()
    target_mean = torch.tensor(train_feat_df.mean(axis=0).values, dtype=torch.float32)
    target_std = torch.tensor(train_feat_df.std(axis=0).replace(0.0, 1.0).values, dtype=torch.float32)

    model = SignalAwareOperatorBottleneck(
        d_in=actual_dB,
        enc_channels=CFG.enc_channels,
        patch_size=CFG.patch_size,
        num_res_blocks=CFG.num_res_blocks,
        dropout=CFG.dropout,
        max_pad_len=CFG.max_pad_len,
        zstar_d=CFG.zstar_d,
        zstar_rank=CFG.zstar_rank,
        reg_proj_dim=CFG.reg_proj_dim,
        target_feature_dim=len(TARGET_FEATURE_NAMES),
    ).to(device)
    model.set_target_normaliser(target_mean.to(device), target_std.to(device))

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    print(f"Operator bottleneck dim: {model.operator_param_dim:,}")
    print(f"Target features: {TARGET_FEATURE_NAMES}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    optim_steps_per_epoch = math.ceil(len(train_loader) / CFG.accum_steps)
    warmup_steps = CFG.warmup_epochs * optim_steps_per_epoch
    total_steps = CFG.max_epochs * optim_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    current_ckpt_path = os.path.join(CFG.checkpoint_dir, "current_boltz_signal_bottleneck_v2.pt")
    best_ckpt_path = os.path.join(CFG.checkpoint_dir, "best_boltz_signal_bottleneck_v2.pt")

    preview_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[CFG.preview_split]

    print(f"\n=== Signal-aware bottleneck training (v2): up to {CFG.max_epochs} epochs ===")
    print(f"    Early stop patience: {CFG.early_stop_patience} | min_delta: {CFG.min_delta}")
    print(f"    Train batches/epoch: {len(train_loader)}, Val batches/epoch: {len(val_loader)}")
    print(f"    Batch size: {CFG.batch_size}, Accum steps: {CFG.accum_steps}")
    print(f"    Effective batch size: {CFG.batch_size * CFG.accum_steps}")
    print(f"    Preview split: {CFG.preview_split} | every {CFG.export_preview_every} epoch(s)\n")

    t_start = time.time()

    for epoch in range(1, CFG.max_epochs + 1):
        t_ep = time.time()

        train_m = run_epoch(model, train_loader, optimizer, scheduler, device, scaler, train=True)
        with torch.no_grad():
            val_m = run_epoch(model, val_loader, optimizer=None, scheduler=None, device=device, scaler=scaler, train=False)

        elapsed = time.time() - t_ep
        total_elapsed = time.time() - t_start

        history_row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_latent_feat": train_m["latent_feat"],
            "train_zstar_feat": train_m["zstar_feat"],
            "train_bind": train_m["bind"],
            "train_var": train_m["var"],
            "train_cov": train_m["cov"],
            "train_theta_l2": train_m["theta_l2"],
            "train_zstar_scale": train_m["zstar_scale"],
            "val_loss": val_m["loss"],
            "val_latent_feat": val_m["latent_feat"],
            "val_zstar_feat": val_m["zstar_feat"],
            "val_bind": val_m["bind"],
            "val_var": val_m["var"],
            "val_cov": val_m["cov"],
            "val_theta_l2": val_m["theta_l2"],
            "val_zstar_scale": val_m["zstar_scale"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)

        print(
            f"Epoch {epoch:2d}/{CFG.max_epochs} | "
            f"train={train_m['loss']:.4f} latent={train_m['latent_feat']:.4f} zstar={train_m['zstar_feat']:.4f} bind={train_m['bind']:.4f} scale={train_m['zstar_scale']:.4f} | "
            f"val={val_m['loss']:.4f} latent={val_m['latent_feat']:.4f} zstar={val_m['zstar_feat']:.4f} bind={val_m['bind']:.4f} scale={val_m['zstar_scale']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s (total {total_elapsed/60:.1f}m)"
        )

        save_checkpoint(current_ckpt_path, epoch, model, optimizer, scheduler, val_m["loss"], actual_dB)
        print(f"  -> Saved current checkpoint: {current_ckpt_path}")

        if CFG.save_epoch_checkpoints:
            epoch_ckpt_path = os.path.join(CFG.checkpoint_dir, f"epoch_{epoch:03d}_boltz_signal_bottleneck_v2.pt")
            save_checkpoint(epoch_ckpt_path, epoch, model, optimizer, scheduler, val_m["loss"], actual_dB)
            print(f"  -> Saved epoch checkpoint: {epoch_ckpt_path}")

        history_path = os.path.join(CFG.checkpoint_dir, "boltz_signal_bottleneck_v2_history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)

        if (epoch % CFG.export_preview_every) == 0:
            preview_epoch_dir = os.path.join(CFG.preview_dir, f"epoch_{epoch:03d}")
            print(f"  -> Exporting preview artifacts to {preview_epoch_dir}")
            export_operator_latents(
                model=model,
                loader=preview_loader,
                device=device,
                out_dir=preview_epoch_dir,
                split_name=f"{CFG.preview_split}_preview",
                max_batches=CFG.preview_max_batches,
                include_zstar=True,
            )

        if val_m["loss"] < (best_val - CFG.min_delta):
            best_val = val_m["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, scheduler, best_val, actual_dB)
            print(f"  -> New best val_loss={best_val:.4f}, saved to {best_ckpt_path}")

            if CFG.export_full_valtest_on_best:
                best_export_dir = os.path.join(CFG.preview_dir, f"best_epoch_{epoch:03d}")
                print(f"  -> Exporting full val/test best artifacts to {best_export_dir}")
                export_operator_latents(model, val_loader, device, best_export_dir, "val", include_zstar=True)
                export_operator_latents(model, test_loader, device, best_export_dir, "test", include_zstar=True)
        else:
            patience_counter += 1
            if epoch >= CFG.min_epochs and patience_counter >= CFG.early_stop_patience:
                print(f"\nStopping early at epoch {epoch}: validation loss has stabilised.")
                break

    total_time = time.time() - t_start
    print(f"\n=== Training complete in {total_time/60:.1f} minutes | best epoch: {best_epoch} ===")

    history_path = os.path.join(CFG.checkpoint_dir, "boltz_signal_bottleneck_v2_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== Exporting final operator latents ===")
    export_operator_latents(model, train_loader, device, CFG.latent_dir, "train", include_zstar=False)
    export_operator_latents(model, val_loader, device, CFG.latent_dir, "val", include_zstar=False)
    export_operator_latents(model, test_loader, device, CFG.latent_dir, "test", include_zstar=False)

    print("\nDone.")


if __name__ == "__main__":
    main()