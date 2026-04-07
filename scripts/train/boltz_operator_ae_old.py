#!/usr/bin/env python3
"""
Boltz operator-bottleneck autoencoder
====================================

Goal
----
Train an autoencoder whose bottleneck directly parameterises the residual part of Z*.

Raw masked Boltz -> encoder -> operator parameters (the bottleneck)
                    -> fixed deterministic construction of residual Z*
                    -> decoder -> reconstructed raw masked Boltz

The learned compressed object is the compact parameterisation from which residual Z* is constructed.

What is saved
-------------
- "latent" in the exported NPZ files = the operator parameters (the bottleneck)
- current checkpoint after every epoch (overwritten)
- best checkpoint based on validation loss
- training history CSV

How to use later
----------------
1. Load the checkpoint
2. Run encode_only(...) to get operator parameters
3. Call latent_to_zstar(...) to construct residual Z*
4. Call add_identity_to_zstar(...) when you want full Z* downstream

Run in tmux:
    tmux new -s boltz_op_ae
    conda activate tcr-multimodal
    cd /home/natasha/multimodal_model/scripts/train
    python boltz_operator_ae.py 2>&1 | tee boltz_operator_ae.log
"""

import os
import time
import glob
import math
import json
import random
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm


@dataclass
class Config:
    base_path: str = "/home/natasha/multimodal_model"
    train_manifest: str = "/home/natasha/multimodal_model/manifests/train_manifest.csv"
    val_manifest: str = "/home/natasha/multimodal_model/manifests/val_manifest.csv"
    test_manifest: str = "/home/natasha/multimodal_model/manifests/test_manifest.csv"
    checkpoint_dir: str = "/home/natasha/multimodal_model/models/boltz_operator_ae_checkpoints"
    latent_dir: str = "/home/natasha/multimodal_model/models/embeddings/boltz_operator_latents"

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
    dec_channels: int = 64
    num_res_blocks: int = 3
    dropout: float = 0.10

    zstar_d: int = 128
    zstar_rank: int = 8

    reg_proj_dim: int = 256
    lambda_recon_abs: float = 0.70
    lambda_recon_rel: float = 0.30
    lambda_var: float = 1.0
    lambda_cov: float = 1.0
    var_target_std: float = 1.0
    eps: float = 1e-6


CFG = Config()


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


class BoltzDataset(Dataset):
    def __init__(self, manifest_path, base_path, split_dir, strict=False):
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
        "binding_flag": torch.tensor([it["binding_flag"] for it in batch], dtype=torch.long),
        "split_dir": [it["split_dir"] for it in batch],
        "manifest_idx": torch.tensor([it["manifest_idx"] for it in batch], dtype=torch.long),
        "emb_path": [it["emb_path"] for it in batch],
    }


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


class OperatorBottleneckBoltzAutoencoder(nn.Module):
    def __init__(
        self,
        d_in: int,
        enc_channels: int,
        dec_channels: int,
        patch_size: int,
        num_res_blocks: int,
        dropout: float,
        max_pad_len: int,
        zstar_d: int,
        zstar_rank: int,
        reg_proj_dim: int,
    ):
        super().__init__()
        self.d_in = d_in
        self.enc_channels = enc_channels
        self.dec_channels = dec_channels
        self.patch_size = patch_size
        self.max_pad_len = max_pad_len
        self.max_patch_len = max_pad_len // patch_size
        self.zstar_d = zstar_d
        self.zstar_rank = zstar_rank
        self.operator_param_dim = 3 * 2 * zstar_d * zstar_rank

        self.patch_embed = nn.Conv2d(d_in, enc_channels, kernel_size=patch_size, stride=patch_size)
        self.encoder_blocks = nn.Sequential(*[ResidualBlock2D(enc_channels, dropout) for _ in range(num_res_blocks)])
        self.enc_norm = nn.GroupNorm(8, enc_channels)
        self.to_operator_params = nn.Sequential(
            nn.Linear(2 * enc_channels, 4 * enc_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * enc_channels, self.operator_param_dim),
        )

        reg_proj = torch.randn(self.operator_param_dim, reg_proj_dim) / math.sqrt(self.operator_param_dim)
        self.register_buffer("reg_proj", reg_proj)

        self.zstar_in = nn.Conv2d(1, dec_channels, kernel_size=5, padding=2)
        self.zstar_blocks = nn.Sequential(*[ResidualBlock2D(dec_channels, dropout) for _ in range(2)])
        self.row_embed = nn.Parameter(torch.randn(self.max_patch_len, dec_channels) * 0.02)
        self.col_embed = nn.Parameter(torch.randn(self.max_patch_len, dec_channels) * 0.02)
        self.decoder_blocks = nn.Sequential(*[ResidualBlock2D(dec_channels, dropout) for _ in range(num_res_blocks)])
        self.dec_norm = nn.GroupNorm(8, dec_channels)
        self.patch_decode = nn.ConvTranspose2d(dec_channels, d_in, kernel_size=patch_size, stride=patch_size)

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
        operator_params = self.to_operator_params(pooled)
        reg_code = operator_params @ self.reg_proj
        return operator_params, reg_code, pair_mask, x.shape[-2], x.shape[-1]

    def _params_to_blocks(self, operator_params: torch.Tensor):
        B = operator_params.shape[0]
        d = self.zstar_d
        r = self.zstar_rank

        params = operator_params.view(B, 3, 2, d, r)
        U = params[:, :, 0]
        V = params[:, :, 1]
        K = torch.matmul(U, V.transpose(-1, -2)) / math.sqrt(r)

        A = K[:, 0]
        C = K[:, 1]
        Bblk = K[:, 2]

        A = 0.5 * (A + A.transpose(1, 2))
        Bblk = 0.5 * (Bblk + Bblk.transpose(1, 2))
        return A, C, Bblk

    def latent_to_zstar(self, operator_params: torch.Tensor):
        """
        Construct residual Z* only:
            [ A    C  ]
            [ C^T  B  ]
        """
        B = operator_params.shape[0]
        d = self.zstar_d
        device = operator_params.device
        dtype = operator_params.dtype

        A, C, Bblk = self._params_to_blocks(operator_params)

        zstar_residual = torch.zeros(B, 2 * d, 2 * d, device=device, dtype=dtype)
        zstar_residual[:, :d, :d] = A
        zstar_residual[:, :d, d:] = C
        zstar_residual[:, d:, :d] = C.transpose(1, 2)
        zstar_residual[:, d:, d:] = Bblk
        zstar_residual = 0.5 * (zstar_residual + zstar_residual.transpose(1, 2))
        return zstar_residual

    def add_identity_to_zstar(self, zstar_residual: torch.Tensor):
        """
        Construct full Z* from residual Z* by adding identity to all four blocks:
            [ I + A    I + C   ]
            [ I + C^T  I + B   ]
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

    def decode(self, zstar_residual: torch.Tensor, Hp: int, Wp: int, L_pad: int):
        z = zstar_residual.unsqueeze(1)
        x = self.zstar_in(z)
        x = self.zstar_blocks(x)
        x = F.interpolate(x, size=(Hp, Wp), mode="bilinear", align_corners=False)

        row = self.row_embed[:Hp].transpose(0, 1).unsqueeze(0).unsqueeze(-1)
        col = self.col_embed[:Wp].transpose(0, 1).unsqueeze(0).unsqueeze(-2)
        x = x + row + col
        x = self.decoder_blocks(x)
        x = F.gelu(self.dec_norm(x))
        recon = self.patch_decode(x)
        recon = recon[:, :, :L_pad, :L_pad]
        recon = recon.permute(0, 2, 3, 1).contiguous()
        return recon

    def forward(self, z_boltz: torch.Tensor, orig_len: torch.Tensor):
        operator_params, reg_code, pair_mask, Hp, Wp = self.encode(z_boltz, orig_len)
        zstar_residual = self.latent_to_zstar(operator_params)
        recon = self.decode(zstar_residual, Hp=Hp, Wp=Wp, L_pad=z_boltz.shape[1])

        return {
            "latent": operator_params,
            "reg_code": reg_code,
            "recon": recon,
            "pair_mask": pair_mask,
            "zstar_residual": zstar_residual,
        }

    def encode_only(self, z_boltz: torch.Tensor, orig_len: torch.Tensor) -> torch.Tensor:
        operator_params, _, _, _, _ = self.encode(z_boltz, orig_len)
        return operator_params


def reconstruction_loss(recon, target, pair_mask, lambda_abs, lambda_rel, eps=1e-6):
    mask = pair_mask.permute(0, 2, 3, 1).expand_as(target)

    abs_err = F.smooth_l1_loss(recon, target, reduction="none")
    abs_loss = masked_mean(abs_err, mask, dim=(1, 2, 3)).mean()

    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    sample_scale = ((target.abs() * mask).sum(dim=(1, 2, 3)) / denom).detach().clamp_min(1e-3)
    sample_scale = sample_scale.view(-1, 1, 1, 1)

    rel_err = F.smooth_l1_loss(recon / sample_scale, target / sample_scale, reduction="none")
    rel_loss = masked_mean(rel_err, mask, dim=(1, 2, 3)).mean()

    total = lambda_abs * abs_loss + lambda_rel * rel_loss
    return {"total": total, "abs": abs_loss, "rel": rel_loss}


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


def run_epoch(model, loader, optimizer, scheduler, device, scaler, train: bool):
    model.train() if train else model.eval()
    running = {"loss": 0.0, "recon": 0.0, "recon_abs": 0.0, "recon_rel": 0.0, "var": 0.0, "cov": 0.0}
    total_n = 0

    pbar = tqdm(loader, desc=("train" if train else "val  "), ncols=115, miniters=20)
    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(pbar):
        z = batch["z"].to(device, non_blocking=True)
        orig_len = batch["orig_len"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(device_type="cuda", enabled=(device.type == "cuda")):
                out = model(z, orig_len)
                recon_terms = reconstruction_loss(
                    recon=out["recon"],
                    target=z,
                    pair_mask=out["pair_mask"],
                    lambda_abs=CFG.lambda_recon_abs,
                    lambda_rel=CFG.lambda_recon_rel,
                    eps=CFG.eps,
                )
                reg_terms = latent_regularisation(out["reg_code"], target_std=CFG.var_target_std)
                loss = recon_terms["total"] + CFG.lambda_var * reg_terms["var"] + CFG.lambda_cov * reg_terms["cov"]
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
        running["recon"] += recon_terms["total"].item() * bs
        running["recon_abs"] += recon_terms["abs"].item() * bs
        running["recon_rel"] += recon_terms["rel"].item() * bs
        running["var"] += reg_terms["var"].item() * bs
        running["cov"] += reg_terms["cov"].item() * bs
        total_n += bs
        pbar.set_postfix(loss=f"{running['loss'] / max(total_n, 1):.4f}")

    return {k: v / max(total_n, 1) for k, v in running.items()}


@torch.no_grad()
def export_operator_latents(model, loader, device, out_dir, split_name):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    all_latents, all_pids, all_bf, all_sd, all_mi = [], [], [], [], []

    for batch in tqdm(loader, desc=f"Export {split_name}"):
        z = batch["z"].to(device, non_blocking=True)
        orig_len = batch["orig_len"].to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=(device.type == "cuda")):
            latent = model.encode_only(z, orig_len)

        all_latents.append(latent.float().cpu().numpy())
        all_pids.extend(batch["pair_id"])
        all_bf.extend(batch["binding_flag"].cpu().numpy().tolist())
        all_sd.extend(batch["split_dir"])
        all_mi.extend(batch["manifest_idx"].cpu().numpy().tolist())

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
    print(f"[{split_name}] operator latent shape: {latents.shape}")


def save_checkpoint(path, epoch, model, optimizer, scheduler, val_loss, d_in):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "val_loss": val_loss,
        "d_in": d_in,
        "config": asdict(CFG),
    }, path)


def main():
    set_seed(CFG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    os.makedirs(CFG.latent_dir, exist_ok=True)
    with open(os.path.join(CFG.checkpoint_dir, "boltz_operator_ae_config.json"), "w") as f:
        json.dump(asdict(CFG), f, indent=2)

    print("\n=== Loading datasets ===")
    train_ds = BoltzDataset(CFG.train_manifest, CFG.base_path, "train")
    val_ds = BoltzDataset(CFG.val_manifest, CFG.base_path, "val")
    test_ds = BoltzDataset(CFG.test_manifest, CFG.base_path, "test")

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

    model = OperatorBottleneckBoltzAutoencoder(
        d_in=actual_dB,
        enc_channels=CFG.enc_channels,
        dec_channels=CFG.dec_channels,
        patch_size=CFG.patch_size,
        num_res_blocks=CFG.num_res_blocks,
        dropout=CFG.dropout,
        max_pad_len=CFG.max_pad_len,
        zstar_d=CFG.zstar_d,
        zstar_rank=CFG.zstar_rank,
        reg_proj_dim=CFG.reg_proj_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_params:,}")
    print(f"Operator bottleneck dim: {model.operator_param_dim:,}")

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

    current_ckpt_path = os.path.join(CFG.checkpoint_dir, "current_boltz_operator_ae.pt")
    best_ckpt_path = os.path.join(CFG.checkpoint_dir, "best_boltz_operator_ae.pt")

    print(f"\n=== Operator-bottleneck AE training: up to {CFG.max_epochs} epochs ===")
    print(f"    Early stop patience: {CFG.early_stop_patience} | min_delta: {CFG.min_delta}")
    print(f"    Train batches/epoch: {len(train_loader)}, Val batches/epoch: {len(val_loader)}")
    print(f"    Batch size: {CFG.batch_size}, Accum steps: {CFG.accum_steps}")
    print(f"    Effective batch size: {CFG.batch_size * CFG.accum_steps}\n")

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
            "train_recon": train_m["recon"],
            "train_abs": train_m["recon_abs"],
            "train_rel": train_m["recon_rel"],
            "train_var": train_m["var"],
            "train_cov": train_m["cov"],
            "val_loss": val_m["loss"],
            "val_recon": val_m["recon"],
            "val_abs": val_m["recon_abs"],
            "val_rel": val_m["recon_rel"],
            "val_var": val_m["var"],
            "val_cov": val_m["cov"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(history_row)

        print(
            f"Epoch {epoch:2d}/{CFG.max_epochs} | "
            f"train={train_m['loss']:.4f} recon={train_m['recon']:.4f} var={train_m['var']:.4f} cov={train_m['cov']:.4f} | "
            f"val={val_m['loss']:.4f} recon={val_m['recon']:.4f} var={val_m['var']:.4f} cov={val_m['cov']:.4f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s (total {total_elapsed/60:.1f}m)"
        )

        save_checkpoint(current_ckpt_path, epoch, model, optimizer, scheduler, val_m["loss"], actual_dB)
        print(f"  -> Saved current checkpoint to {current_ckpt_path}")

        if val_m["loss"] < (best_val - CFG.min_delta):
            best_val = val_m["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(best_ckpt_path, epoch, model, optimizer, scheduler, best_val, actual_dB)
            print(f"  -> New best val_loss={best_val:.4f}, saved to {best_ckpt_path}")
        else:
            patience_counter += 1
            if epoch >= CFG.min_epochs and patience_counter >= CFG.early_stop_patience:
                print(f"\nStopping early at epoch {epoch}: validation loss has stabilised.")
                break

        pd.DataFrame(history).to_csv(os.path.join(CFG.checkpoint_dir, "boltz_operator_ae_history.csv"), index=False)

    total_time = time.time() - t_start
    print(f"\n=== Training complete in {total_time/60:.1f} minutes | best epoch: {best_epoch} ===")

    history_path = os.path.join(CFG.checkpoint_dir, "boltz_operator_ae_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"Saved training history to {history_path}")

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\n=== Exporting operator latents ===")
    export_operator_latents(model, train_loader, device, CFG.latent_dir, "train")
    export_operator_latents(model, val_loader, device, CFG.latent_dir, "val")
    export_operator_latents(model, test_loader, device, CFG.latent_dir, "test")

    print("\nDone.")


if __name__ == "__main__":
    main()