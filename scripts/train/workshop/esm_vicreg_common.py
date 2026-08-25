"""Shared ESMC VICReg workshop utilities for fine-tuned and raw embedding runs."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader, Dataset


FINETUNED_ALL_EVAL_MODELS = ("esm_vicreg", "finetuned_esmc_meanpool", "pretrained_esmc_meanpool")
RAW_ALL_EVAL_MODELS = ("esm_vicreg", "pretrained_esmc_meanpool")
EPOCH_VAL_MODELS = ("esm_vicreg",)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_seq(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    for ch in [" ", "-", ":", "|", ";", ","]:
        s = s.replace(ch, "")
    return s


def first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def extract_length(df: pd.DataFrame, length_candidates: List[str], seq_candidates: List[str]) -> Tuple[pd.Series, str]:
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

    pep_col = first_existing_col(out, ["Peptide", "peptide", "pep_seq", "peptide_seq"])
    if pep_col is None:
        raise ValueError(f"{source_name}: CSV must contain Peptide/peptide. Available columns: {list(out.columns)}")
    out["peptide_for_eval"] = out[pep_col].map(clean_seq)

    source_map = {"label_col": "constant_1" if label_col is None else label_col, "pep_col": pep_col}
    out["tcra_len"], source_map["tcra_len"] = extract_length(
        out,
        ["tcra_len", "tcr_alpha_len", "alpha_len", "TRA_len", "cdr3a_len"],
        ["tcra", "tcr_alpha", "TRA", "cdr3a", "alpha", "alpha_seq", "tcr_a"],
    )
    out["tcrb_len"], source_map["tcrb_len"] = extract_length(
        out,
        ["tcrb_len", "tcr_beta_len", "beta_len", "TRB_len", "cdr3b_len"],
        ["tcrb", "tcr_beta", "TRB", "cdr3b", "beta", "beta_seq", "tcr_b"],
    )
    out["pep_len"], source_map["pep_len"] = extract_length(
        out,
        ["pep_len", "peptide_len"],
        ["Peptide", "peptide", "pep_seq", "peptide_seq"],
    )
    out["hla_len"], source_map["hla_len"] = extract_length(
        out,
        ["hla_len", "mhc_len", "HLA_len", "mhca_len"],
        ["HLA_sequence", "HLA_seq", "hla_sequence", "hla_seq", "HLA", "hla", "mhc_seq", "MHC_sequence"],
    )
    out["has_alpha"] = out["tcra_len"] > 0
    out["has_beta"] = out["tcrb_len"] > 0
    return out, source_map


def load_meta(csv_path: str, source_name: str, positives_only: bool, complete_only: bool = True) -> Tuple[pd.DataFrame, Dict]:
    raw = pd.read_csv(csv_path)
    meta, source_map = normalise_manifest(raw, source_name)

    if complete_only and (source_map["tcra_len"] == "__missing__" or source_map["tcrb_len"] == "__missing__"):
        raise ValueError(
            f"{source_name}: complete_only requested, but alpha/beta chain information could not be found. "
            f"Source map: {source_map}. Available columns: {list(raw.columns)}"
        )

    audit = {
        "split": source_name,
        "csv_path": str(csv_path),
        "csv_rows": int(len(raw)),
        "label_source": source_map["label_col"],
        "peptide_source": source_map["pep_col"],
        "tcra_len_source": source_map["tcra_len"],
        "tcrb_len_source": source_map["tcrb_len"],
        "pep_len_source": source_map["pep_len"],
        "hla_len_source": source_map["hla_len"],
        "positives_only": bool(positives_only),
        "complete_only": bool(complete_only),
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

    meta = meta[(meta["pep_len"] > 0) & (meta["hla_len"] > 0)].copy()
    audit["rows_after_required_sequence_filter"] = int(len(meta))

    if complete_only:
        meta = meta[meta["has_alpha"] & meta["has_beta"]].copy()
    audit["n_final"] = int(len(meta))
    audit["n_positive_final"] = int((meta["binding_flag"] == 1).sum())
    audit["n_negative_final"] = int((meta["binding_flag"] == 0).sum())
    audit["n_missing_alpha_final"] = int((~meta["has_alpha"]).sum())
    audit["n_missing_beta_final"] = int((~meta["has_beta"]).sum())

    print(
        f"{source_name}: csv_rows={audit['csv_rows']} | final={audit['n_final']} | "
        f"pos={audit['n_positive_final']} | neg={audit['n_negative_final']} | complete_only={complete_only}",
        flush=True,
    )
    if len(meta) == 0:
        raise RuntimeError(f"{source_name}: no rows remain after filtering")
    return meta.reset_index(drop=True), audit


def to_str_list(x) -> List[str]:
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().tolist()
    elif isinstance(x, np.ndarray):
        x = x.tolist()
    elif not isinstance(x, (list, tuple)):
        x = [x]
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in x]


def build_pair_index(shards_dir: Path, source_name: str, embedding_name: str) -> Dict[str, Tuple[Path, int, int]]:
    shards_dir = Path(shards_dir)
    shard_paths = sorted(shards_dir.glob("shard_*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.pt files found in {shards_dir} for {source_name}/{embedding_name}")

    index: Dict[str, Tuple[Path, int, int]] = {}
    seen = 0
    duplicates = []
    for sp in shard_paths:
        shard = torch.load(sp, map_location="cpu")
        for bidx, batch in enumerate(shard):
            pair_ids = to_str_list(batch["pair_id"])
            for ridx, pid in enumerate(pair_ids):
                seen += 1
                if pid in index:
                    duplicates.append(pid)
                else:
                    index[pid] = (sp, bidx, ridx)
    if duplicates:
        raise RuntimeError(f"{source_name}/{embedding_name}: duplicate pair_ids in shards. Examples: {duplicates[:10]}")
    print(
        f"{source_name}/{embedding_name}: shard rows indexed={seen} | unique_pair_ids={len(index)} | dir={shards_dir}",
        flush=True,
    )
    return index


class PairedESMRowDataset(Dataset):
    """Rows aligned by pair_id from ESMC shard directories (fine-tuned and/or raw)."""

    def __init__(
        self,
        finetuned_dir: Path,
        pretrained_dir: Optional[Path],
        meta: pd.DataFrame,
        source_name: str,
        include_pretrained: bool = True,
        order_by_finetuned_shard: bool = True,
    ):
        self.finetuned_dir = Path(finetuned_dir)
        self.pretrained_dir = None if pretrained_dir is None else Path(pretrained_dir)
        self.include_pretrained = bool(include_pretrained)
        self.meta = meta.reset_index(drop=True)
        self.meta_by_pid = {str(r["pair_id"]): r for _, r in self.meta.iterrows()}
        self.source_name = source_name

        self.ft_index = build_pair_index(self.finetuned_dir, source_name, "finetuned")
        self.pre_index = {}
        if self.include_pretrained:
            if self.pretrained_dir is None:
                raise ValueError(f"{source_name}: include_pretrained=True but pretrained_dir is None")
            self.pre_index = build_pair_index(self.pretrained_dir, source_name, "pretrained")

        requested = [str(x) for x in self.meta["pair_id"].tolist()]
        missing_ft = [pid for pid in requested if pid not in self.ft_index]
        missing_pre = [pid for pid in requested if self.include_pretrained and pid not in self.pre_index]
        if missing_ft or missing_pre:
            raise RuntimeError(
                f"{source_name}: pair_id alignment failure. "
                f"missing_finetuned={len(missing_ft)} examples={missing_ft[:10]} | "
                f"missing_pretrained={len(missing_pre)} examples={missing_pre[:10]}"
            )

        if order_by_finetuned_shard:
            requested = sorted(
                requested,
                key=lambda pid: (str(self.ft_index[pid][0]), int(self.ft_index[pid][1]), int(self.ft_index[pid][2])),
            )
        self.pair_ids = requested
        print(
            f"{source_name}: paired ESM rows kept={len(self.pair_ids)} | "
            f"include_pretrained={self.include_pretrained} | order_by_finetuned_shard={order_by_finetuned_shard}",
            flush=True,
        )

        # Keep every loaded shard in RAM. A single-slot cache thrashs under DataLoader
        # shuffle (train reloads ~GB .pt files almost every sample). Values and sample
        # order are unchanged; only repeated torch.load I/O is removed.
        self._ft_shard_cache: Dict[str, object] = {}
        self._pre_shard_cache: Dict[str, object] = {}

    def __len__(self) -> int:
        return len(self.pair_ids)

    def clear_shard_cache(self) -> None:
        self._ft_shard_cache.clear()
        self._pre_shard_cache.clear()

    def _load(self, sp: Path, branch: str):
        key = str(sp)
        if branch == "ft":
            cache = self._ft_shard_cache
        elif branch == "pre":
            cache = self._pre_shard_cache
        else:
            raise ValueError(branch)
        if key not in cache:
            cache[key] = torch.load(sp, map_location="cpu")
            print(
                f"{self.source_name}: cached {branch} shard {Path(key).name} "
                f"({len(cache)} unique shards)",
                flush=True,
            )
        return cache[key]

    def __getitem__(self, idx: int) -> Dict:
        pid = self.pair_ids[idx]
        row = self.meta_by_pid[pid]

        ft_sp, ft_bidx, ft_ridx = self.ft_index[pid]
        ft_batch = self._load(ft_sp, "ft")[ft_bidx]

        item = {
            "ft_emb_T": ft_batch["emb_T"][ft_ridx].float(),
            "ft_mask_T": ft_batch["mask_T"][ft_ridx].bool(),
            "ft_emb_P": ft_batch["emb_P"][ft_ridx].float(),
            "ft_mask_P": ft_batch["mask_P"][ft_ridx].bool(),
            "ft_emb_H": ft_batch["emb_H"][ft_ridx].float(),
            "ft_mask_H": ft_batch["mask_H"][ft_ridx].bool(),
            "binding_flag": int(row["binding_flag"]),
            "pair_id": pid,
            "peptide": str(row["peptide_for_eval"]),
            "has_alpha": bool(row["has_alpha"]),
            "has_beta": bool(row["has_beta"]),
            "tcra_len": int(row["tcra_len"]),
            "tcrb_len": int(row["tcrb_len"]),
            "pep_len": int(row["pep_len"]),
            "hla_len": int(row["hla_len"]),
        }

        if self.include_pretrained:
            pre_sp, pre_bidx, pre_ridx = self.pre_index[pid]
            pre_batch = self._load(pre_sp, "pre")[pre_bidx]
            item.update({
                "pre_emb_T": pre_batch["emb_T"][pre_ridx].float(),
                "pre_mask_T": pre_batch["mask_T"][pre_ridx].bool(),
                "pre_emb_P": pre_batch["emb_P"][pre_ridx].float(),
                "pre_mask_P": pre_batch["mask_P"][pre_ridx].bool(),
                "pre_emb_H": pre_batch["emb_H"][pre_ridx].float(),
                "pre_mask_H": pre_batch["mask_H"][pre_ridx].bool(),
            })
        return item


def esm_collate(rows: List[Dict]) -> Dict:
    out = {}
    tensor_keys = [
        "ft_emb_T", "ft_mask_T", "ft_emb_P", "ft_mask_P", "ft_emb_H", "ft_mask_H",
    ]
    if "pre_emb_T" in rows[0]:
        tensor_keys.extend([
            "pre_emb_T", "pre_mask_T", "pre_emb_P", "pre_mask_P", "pre_emb_H", "pre_mask_H",
        ])
    for k in tensor_keys:
        out[k] = torch.stack([r[k] for r in rows], dim=0)
    out["binding_flag"] = torch.tensor([r["binding_flag"] for r in rows], dtype=torch.long)
    for k in ["pair_id", "peptide"]:
        out[k] = [r[k] for r in rows]
    for k in ["has_alpha", "has_beta"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.bool)
    for k in ["tcra_len", "tcrb_len", "pep_len", "hla_len"]:
        out[k] = torch.tensor([r[k] for r in rows], dtype=torch.long)
    return out


class ESMProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_max: int, dropout: float = 0.1):
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
        B, L_pad, D_in = emb.shape
        if D_in != self.D:
            raise ValueError(f"Embedding dimension mismatch: got {D_in}, expected {self.D}")
        if L_pad > self.L_max:
            raise ValueError(f"Sequence length {L_pad} exceeds L_max {self.L_max}")
        L_true = mask.sum(dim=1)
        z_list = []
        for b in range(B):
            Lb = int(L_true[b].item())
            if Lb == 0:
                z_list.append(torch.zeros(self.d, device=emb.device, dtype=emb.dtype))
                continue
            Xb = emb[b, :Lb, :] * mask[b, :Lb].unsqueeze(-1).float()
            Yb = Xb @ self.B_c
            Ub = self.A_c[:Lb, :].T @ Yb
            z_list.append(Ub.reshape(-1) @ self.H_c)
        return self.expander(torch.stack(z_list, dim=0))


class PMHCProjectionHead(nn.Module):
    def __init__(self, D: int, rL: int, rD: int, d: int, L_P_max: int, L_H_max: int, R_PH: float, dropout: float):
        super().__init__()
        d_P = int(round(R_PH * d))
        d_H = d - d_P
        if d_P <= 0 or d_H <= 0:
            raise ValueError(f"Invalid R_PH={R_PH}; produced d_P={d_P}, d_H={d_H}")
        self.pep_encoder = ESMProjectionHead(D, rL, rD, d_P, L_P_max, dropout)
        self.hla_encoder = ESMProjectionHead(D, rL, rD, d_H, L_H_max, dropout)

    def forward(self, emb_P: torch.Tensor, mask_P: torch.Tensor, emb_H: torch.Tensor, mask_H: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.pep_encoder(emb_P, mask_P), self.hla_encoder(emb_H, mask_H)], dim=-1)


def masked_mean_pool(emb: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    mask_f = mask.float().unsqueeze(-1)
    return (emb * mask_f).sum(dim=1) / (mask_f.sum(dim=1) + eps)


def vicreg_variance(u: torch.Tensor, gamma: float = 1.0, eps_var: float = 1e-4) -> torch.Tensor:
    u = u - u.mean(dim=0, keepdim=True)
    std = torch.sqrt(u.var(dim=0, unbiased=False) + eps_var)
    return F.relu(gamma - std).mean()


def vicreg_covariance(u: torch.Tensor) -> torch.Tensor:
    B, d = u.shape
    if B <= 1:
        return torch.tensor(0.0, device=u.device, dtype=u.dtype)
    u = u - u.mean(dim=0, keepdim=True)
    cov = (u.T @ u) / (B - 1)
    cov_off = cov - torch.diag_embed(torch.diag(cov))
    return (cov_off ** 2).sum() / d


def plain_vicreg_loss(
    zT: torch.Tensor,
    zPH: torch.Tensor,
    alpha: float,
    beta: float,
    delta: float,
    gamma_var: float,
    eps_var: float,
    return_parts: bool = False,
):
    L_inv = F.mse_loss(zT, zPH)
    L_var = vicreg_variance(zT, gamma_var, eps_var) + vicreg_variance(zPH, gamma_var, eps_var)
    L_cov = vicreg_covariance(zT) + vicreg_covariance(zPH)
    loss = alpha * L_inv + beta * L_var + delta * L_cov
    if not return_parts:
        return loss
    return loss, {
        "L_total": float(loss.detach().cpu()),
        "L_inv": float(L_inv.detach().cpu()),
        "L_var": float(L_var.detach().cpu()),
        "L_cov": float(L_cov.detach().cpu()),
        "weighted_inv": float((alpha * L_inv).detach().cpu()),
        "weighted_var": float((beta * L_var).detach().cpu()),
        "weighted_cov": float((delta * L_cov).detach().cpu()),
        "zT_std": float(zT.std(unbiased=False).detach().cpu()),
        "zPH_std": float(zPH.std(unbiased=False).detach().cpu()),
    }


def score_from_vectors(zT: torch.Tensor, zPH: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mse_distance = (zT - zPH).pow(2).mean(dim=-1)
    return -mse_distance, mse_distance


def meanpool_score(
    emb_T: torch.Tensor,
    mask_T: torch.Tensor,
    emb_P: torch.Tensor,
    mask_P: torch.Tensor,
    emb_H: torch.Tensor,
    mask_H: torch.Tensor,
    device: torch.device,
    R_PH: float,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    T = masked_mean_pool(emb_T.to(device), mask_T.to(device), eps)
    P = masked_mean_pool(emb_P.to(device), mask_P.to(device), eps)
    HLA = masked_mean_pool(emb_H.to(device), mask_H.to(device), eps)
    PH = R_PH * P + (1.0 - R_PH) * HLA
    score, mse = score_from_vectors(T, PH)
    return score, mse, T, PH


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_auprc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def safe_partial_auc_raw(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("max_fpr must be in (0, 1]")
    if max_fpr not in fpr:
        stop = np.searchsorted(fpr, max_fpr, side="right")
        fpr_ext = np.concatenate([fpr[:stop], [max_fpr]])
        tpr_ext = np.concatenate([tpr[:stop], [np.interp(max_fpr, fpr, tpr)]])
    else:
        keep = fpr <= max_fpr
        fpr_ext = fpr[keep]
        tpr_ext = tpr[keep]
    return float(auc(fpr_ext, tpr_ext))


def safe_partial_auc_mcclish(labels: np.ndarray, scores: np.ndarray, max_fpr: float = 0.1) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores, max_fpr=max_fpr))


def per_peptide_table(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    df = pd.DataFrame({"label": labels.astype(int), "score": scores.astype(float), "peptide": peptides.astype(str)})
    for pep, grp in df.groupby("peptide", sort=True):
        y = grp["label"].to_numpy()
        s = grp["score"].to_numpy()
        valid = len(np.unique(y)) == 2
        rows.append({
            "peptide": pep,
            "n": int(len(grp)),
            "n_pos": int(y.sum()),
            "n_neg": int((y == 0).sum()),
            "auroc": float(roc_auc_score(y, s)) if valid else float("nan"),
            "auc0.1_raw": safe_partial_auc_raw(y, s, max_fpr) if valid else float("nan"),
            "auc0.1_mcclish": safe_partial_auc_mcclish(y, s, max_fpr) if valid else float("nan"),
            "valid": bool(valid),
        })
    table = pd.DataFrame(rows).sort_values(["valid", "n"], ascending=[False, False]).reset_index(drop=True)
    valid = table[table["valid"]].copy()
    if len(valid) == 0:
        summary = {
            "peptide_macro_auroc": float("nan"),
            "peptide_weighted_auroc": float("nan"),
            "peptide_macro_auc0.1_mcclish": float("nan"),
            "peptide_weighted_auc0.1_mcclish": float("nan"),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": 0,
        }
    else:
        summary = {
            "peptide_macro_auroc": float(valid["auroc"].mean()),
            "peptide_weighted_auroc": float(np.average(valid["auroc"], weights=valid["n"])),
            "peptide_macro_auc0.1_mcclish": float(valid["auc0.1_mcclish"].mean()),
            "peptide_weighted_auc0.1_mcclish": float(np.average(valid["auc0.1_mcclish"], weights=valid["n"])),
            "n_peptides_total": int(len(table)),
            "n_peptides_valid": int(len(valid)),
        }
    return table, summary


def metrics_for_scores(labels: np.ndarray, scores: np.ndarray, peptides: np.ndarray, max_fpr: float) -> Tuple[Dict[str, float], pd.DataFrame]:
    pep_table, pep_summary = per_peptide_table(labels, scores, peptides, max_fpr)
    metrics = {
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int((labels == 0).sum()),
        "global_auroc": safe_auroc(labels, scores),
        "auprc": safe_auprc(labels, scores),
        "global_auc0.1_raw": safe_partial_auc_raw(labels, scores, max_fpr),
        "global_auc0.1_mcclish": safe_partial_auc_mcclish(labels, scores, max_fpr),
        "score_mean": float(np.mean(scores)),
        "score_std": float(np.std(scores)),
        **pep_summary,
    }
    return metrics, pep_table


def loss_params(cfg) -> Dict:
    return {
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "delta": cfg.delta,
        "gamma_var": cfg.gamma_var,
        "eps_var": cfg.eps_var,
    }


def prepare_dirs(cfg) -> Tuple[Path, Path, Path]:
    seed_name = f"seed_{cfg.seed}"
    checkpoint_dir = Path(cfg.checkpoint_root) / seed_name
    output_dir = Path(cfg.output_root) / seed_name
    figure_dir = Path(cfg.figure_root) / seed_name
    if cfg.overwrite:
        for d in [checkpoint_dir, output_dir, figure_dir]:
            if d.exists():
                shutil.rmtree(d)
    for d in [checkpoint_dir, output_dir, figure_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir, output_dir, figure_dir


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=esm_collate,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def infer_shapes(datasets: List[Dataset]) -> Tuple[int, int, int, int]:
    D = None
    L_T = L_P = L_H = 0
    for ds in datasets:
        sample = ds[0]
        d_here = sample["ft_emb_T"].shape[-1]
        if D is None:
            D = d_here
        elif D != d_here:
            raise ValueError(f"Embedding dimension mismatch across datasets: {D} vs {d_here}")
        L_T = max(L_T, int(sample["ft_emb_T"].shape[0]))
        L_P = max(L_P, int(sample["ft_emb_P"].shape[0]))
        L_H = max(L_H, int(sample["ft_emb_H"].shape[0]))
    print(f"Detected projection shapes: D={D}, L_T={L_T}, L_P={L_P}, L_H={L_H}", flush=True)
    return int(D), L_T, L_P, L_H


def raw_shard_split_dir(embed_root: Path, immrep_shard_dir: Path, split: str) -> Path:
    if split == "immrep_test":
        return Path(immrep_shard_dir)
    return Path(embed_root) / split


@torch.no_grad()
def evaluate(
    loader: DataLoader,
    tcr_proj: nn.Module,
    pmhc_proj: nn.Module,
    device: torch.device,
    cfg,
    split: str,
    save_latents: bool,
    model_names: Tuple[str, ...],
    pretrained_meanpool_from_ft: bool = False,
) -> Dict:
    tcr_proj.eval()
    pmhc_proj.eval()

    model_names = tuple(model_names)
    allowed = set(FINETUNED_ALL_EVAL_MODELS) | set(RAW_ALL_EVAL_MODELS)
    if not model_names:
        raise ValueError("model_names must be non-empty")
    unknown = set(model_names) - allowed
    if unknown:
        raise ValueError(f"Unknown model_names: {sorted(unknown)}")

    pair_ids, peptides = [], []
    labels_all = []
    meta_cols = {k: [] for k in ["has_alpha", "has_beta", "tcra_len", "tcrb_len", "pep_len", "hla_len"]}
    scores = {name: [] for name in model_names}
    distances = {name: [] for name in model_names}
    latent_store = {} if save_latents else None
    if save_latents:
        if "esm_vicreg" in model_names:
            latent_store["zT_esm_vicreg"] = []
            latent_store["zPH_esm_vicreg"] = []
        if "finetuned_esmc_meanpool" in model_names:
            latent_store["T_finetuned_meanpool"] = []
            latent_store["PH_finetuned_meanpool"] = []
        if "pretrained_esmc_meanpool" in model_names:
            latent_store["T_pretrained_meanpool"] = []
            latent_store["PH_pretrained_meanpool"] = []

    running = {k: 0.0 for k in ["loss", "L_inv", "L_var", "L_cov", "weighted_inv", "weighted_var", "weighted_cov", "zT_std", "zPH_std"]}
    n_steps = 0
    lp = loss_params(cfg)

    for batch in loader:
        zT = tcr_proj(batch["ft_emb_T"].to(device), batch["ft_mask_T"].to(device))
        zPH = pmhc_proj(
            batch["ft_emb_P"].to(device), batch["ft_mask_P"].to(device),
            batch["ft_emb_H"].to(device), batch["ft_mask_H"].to(device),
        )
        loss, parts = plain_vicreg_loss(zT, zPH, **lp, return_parts=True)
        vicreg_score, vicreg_mse = score_from_vectors(zT, zPH)

        batch_scores = {}
        batch_distances = {}
        if "esm_vicreg" in model_names:
            batch_scores["esm_vicreg"] = vicreg_score
            batch_distances["esm_vicreg"] = vicreg_mse

        ft_T = ft_PH = pre_T = pre_PH = None
        if "finetuned_esmc_meanpool" in model_names:
            ft_score, ft_mse, ft_T, ft_PH = meanpool_score(
                batch["ft_emb_T"], batch["ft_mask_T"], batch["ft_emb_P"], batch["ft_mask_P"], batch["ft_emb_H"], batch["ft_mask_H"],
                device, cfg.R_PH, cfg.eps_pool,
            )
            batch_scores["finetuned_esmc_meanpool"] = ft_score
            batch_distances["finetuned_esmc_meanpool"] = ft_mse

        if "pretrained_esmc_meanpool" in model_names:
            if pretrained_meanpool_from_ft or "pre_emb_T" not in batch:
                pre_score, pre_mse, pre_T, pre_PH = meanpool_score(
                    batch["ft_emb_T"], batch["ft_mask_T"], batch["ft_emb_P"], batch["ft_mask_P"], batch["ft_emb_H"], batch["ft_mask_H"],
                    device, cfg.R_PH, cfg.eps_pool,
                )
            else:
                pre_score, pre_mse, pre_T, pre_PH = meanpool_score(
                    batch["pre_emb_T"], batch["pre_mask_T"], batch["pre_emb_P"], batch["pre_mask_P"], batch["pre_emb_H"], batch["pre_mask_H"],
                    device, cfg.R_PH, cfg.eps_pool,
                )
            batch_scores["pretrained_esmc_meanpool"] = pre_score
            batch_distances["pretrained_esmc_meanpool"] = pre_mse

        labels = batch["binding_flag"].detach().cpu().numpy().astype(int)
        labels_all.append(labels)
        pair_ids.extend([str(x) for x in batch["pair_id"]])
        peptides.append(np.array(batch["peptide"], dtype=str))
        for k in meta_cols:
            meta_cols[k].append(batch[k].detach().cpu().numpy())

        for name in model_names:
            scores[name].append(batch_scores[name].detach().cpu().numpy())
            distances[name].append(batch_distances[name].detach().cpu().numpy())

        if save_latents:
            if "esm_vicreg" in model_names:
                latent_store["zT_esm_vicreg"].append(zT.detach().cpu().numpy())
                latent_store["zPH_esm_vicreg"].append(zPH.detach().cpu().numpy())
            if "finetuned_esmc_meanpool" in model_names:
                latent_store["T_finetuned_meanpool"].append(ft_T.detach().cpu().numpy())
                latent_store["PH_finetuned_meanpool"].append(ft_PH.detach().cpu().numpy())
            if "pretrained_esmc_meanpool" in model_names:
                latent_store["T_pretrained_meanpool"].append(pre_T.detach().cpu().numpy())
                latent_store["PH_pretrained_meanpool"].append(pre_PH.detach().cpu().numpy())

        running["loss"] += float(loss.detach().cpu())
        for k in running:
            if k != "loss":
                running[k] += float(parts[k])
        n_steps += 1

    labels_np = np.concatenate(labels_all).astype(int)
    peptides_np = np.concatenate(peptides).astype(str)
    scores_np = {k: np.concatenate(v) for k, v in scores.items()}
    distances_np = {k: np.concatenate(v) for k, v in distances.items()}
    meta_np = {k: np.concatenate(v) for k, v in meta_cols.items()}

    metrics = {}
    peptide_tables = {}
    for model_name in model_names:
        m, table = metrics_for_scores(labels_np, scores_np[model_name], peptides_np, cfg.partial_auc_max_fpr)
        m.update({
            "mse_distance_mean": float(np.mean(distances_np[model_name])),
            "mse_distance_std": float(np.std(distances_np[model_name])),
        })
        metrics[model_name] = m
        peptide_tables[model_name] = table

    running_avg = {k: v / max(1, n_steps) for k, v in running.items()}
    if "esm_vicreg" in model_names:
        metrics["esm_vicreg"].update({f"eval_{k}": val for k, val in running_avg.items()})

    predictions = pd.DataFrame({
        "pair_id": pair_ids,
        "peptide": peptides_np,
        "label": labels_np,
        **{k: meta_np[k] for k in meta_np},
    })
    for model_name in model_names:
        predictions[f"{model_name}_score"] = scores_np[model_name]
        predictions[f"{model_name}_mse_distance"] = distances_np[model_name]

    latents = None
    if save_latents:
        latents = {k: np.concatenate(v, axis=0) for k, v in latent_store.items()}
        latents.update({"pair_id": np.array(pair_ids, dtype=str), "peptide": peptides_np, "label": labels_np})

    return {
        "split": split,
        "metrics": metrics,
        "predictions": predictions,
        "peptide_tables": peptide_tables,
        "distances": distances_np,
        "labels": labels_np,
        "latents": latents,
    }


def plot_histogram(distances: np.ndarray, labels: np.ndarray, title: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(int)
    plt.figure(figsize=(8, 5))
    if np.any(labels == 0):
        plt.hist(distances[labels == 0], bins=50, density=True, alpha=0.55, label="decoy/negative")
    if np.any(labels == 1):
        plt.hist(distances[labels == 1], bins=50, density=True, alpha=0.55, label="positive")
    plt.xlabel("MSE distance; lower = stronger predicted binding")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_eval_outputs(eval_obj: Dict, output_dir: Path, figure_dir: Path, split: str, save_latents: bool) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    pred_path = output_dir / f"{split}_predictions.csv"
    eval_obj["predictions"].to_csv(pred_path, index=False)
    paths[f"{split}_predictions"] = str(pred_path)

    for model_name, table in eval_obj["peptide_tables"].items():
        path = output_dir / f"{split}_{model_name}_per_peptide.csv"
        table.to_csv(path, index=False)
        paths[f"{split}_{model_name}_per_peptide"] = str(path)

    for model_name, dist in eval_obj["distances"].items():
        fig_path = figure_dir / f"{split}_{model_name}_mse_hist.png"
        plot_histogram(dist, eval_obj["labels"], f"{split}: {model_name} MSE distance", fig_path)
        paths[f"{split}_{model_name}_mse_hist"] = str(fig_path)

    if save_latents and eval_obj["latents"] is not None:
        latent_path = output_dir / f"{split}_latents.npz"
        np.savez_compressed(latent_path, **eval_obj["latents"])
        paths[f"{split}_latents"] = str(latent_path)

    return paths
