#!/usr/bin/env python3
"""Export frozen ESMC-300m per-residue embedding shards for VICReg baselines.

This mirrors the shard layout produced by ``save_embedding_shards_for_split`` in
``scripts/train/model_baseline_vicreg_hamiltonian.ipynb``, but uses a single
shared pretrained ``ESMC.from_pretrained("esmc_300m")`` model with no LoRA
adapters and no MLM fine-tuning.

Pipeline
--------
CSV strings -> ESMC tokenizer (clean, unmasked) -> frozen ESMC forward -> shards

Each ``shard_XXXXX.pt`` file is a list of batch dicts with keys:
    emb_T, emb_P, emb_H, mask_T, mask_P, mask_H,
    binding_flag, pair_id, batch_idx, split

Typical launch (full export)
----------------------------
    cd /home/natasha/multimodal_model
    conda activate tcr-multimodal
    python scripts/preprocess/export_raw_esmc_embedding_shards.py 2>&1 | tee models/log_files/export_raw_esmc_multiview_ids.log

Smoke test first
----------------
    python scripts/preprocess/export_raw_esmc_embedding_shards.py \\
        --embed-root models/embeddings/raw_esmc_300m_multiview_ids_smoke \\
        --splits val \\
        --max-rows 16 \\
        --batch-size 4 \\
        --chunk-size 2 \\
        --verify
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]

REQUIRED_COLUMNS = ("TCR_full", "Peptide", "HLA_sequence", "binding_flag", "pair_id")


@dataclass
class ExportConfig:
    train_csv: str = str(REPO / "data/train/train_multiview.csv")
    val_csv: str = str(REPO / "data/val/val_multiview.csv")
    test_csv: str = str(REPO / "data/test/test_multiview.csv")
    immrep_csv: str = str(REPO / "data/immrep_test/immrep_test_multiview.csv")
    embed_root: str = str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids")
    model_name: str = "esmc_300m"
    batch_size: int = 8
    num_workers: int = 0
    chunk_size: int = 100
    dtype: str = "float16"
    device: str = "cuda"
    splits: Tuple[str, ...] = ("train", "val", "test", "immrep_test")
    max_rows: Optional[int] = None
    clear_existing: bool = True


def load_split_sequences(
    csv_path: str,
    max_rows: Optional[int] = None,
    tcr_col: str = "TCR_full",
    peptide_col: str = "Peptide",
    hla_col: str = "HLA_sequence",
    binding_col: str = "binding_flag",
    pair_id_col: str = "pair_id",
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path, low_memory=False).reset_index(drop=True)
    if max_rows is not None:
        df = df.iloc[: int(max_rows)].copy()

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}. Available: {list(df.columns)}")

    return {
        "df": df,
        "tcrs_data": df[tcr_col].astype(str).tolist(),
        "peptides_data": df[peptide_col].astype(str).tolist(),
        "hlas_data": df[hla_col].astype(str).tolist(),
        "binding_flags": df[binding_col].astype(int).tolist(),
        "pair_ids": df[pair_id_col].astype(str).tolist(),
        "source_csv": str(csv_path),
        "n_rows": int(len(df)),
    }


class CleanSeqDataset(Dataset):
    def __init__(self, sequences: Sequence[str], tokenizer):
        self.sequences = list(sequences)
        enc = tokenizer(self.sequences, return_tensors="pt", padding=True)
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }


def clean_collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([f["input_ids"] for f in features], dim=0),
        "attention_mask": torch.stack([f["attention_mask"] for f in features], dim=0),
    }


def build_clean_loaders(
    tokenizer,
    tcrs_data: Sequence[str],
    peptides_data: Sequence[str],
    hlas_data: Sequence[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    tcr_ds = CleanSeqDataset(tcrs_data, tokenizer)
    pep_ds = CleanSeqDataset(peptides_data, tokenizer)
    hla_ds = CleanSeqDataset(hlas_data, tokenizer)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=clean_collate,
    )
    return (
        DataLoader(tcr_ds, **loader_kwargs),
        DataLoader(pep_ds, **loader_kwargs),
        DataLoader(hla_ds, **loader_kwargs),
    )


def cast_embeddings(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if dtype == "float16":
        return tensor.to(torch.float16)
    if dtype == "float32":
        return tensor.to(torch.float32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def clear_split_shards(split_dir: Path) -> int:
    """Remove existing shard_*.pt in one split dir only (never touches sibling splits)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for sp in split_dir.glob("shard_*.pt"):
        sp.unlink()
        removed += 1
    return removed


@torch.inference_mode()
def save_embedding_shards_for_split(
    split_name: str,
    split_data: Dict[str, Any],
    model,
    tokenizer,
    save_root: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    chunk_size: int,
    dtype: str,
    clear_existing: bool = True,
) -> Dict[str, Any]:
    split_dir = save_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    n_cleared = 0
    if clear_existing:
        n_cleared = clear_split_shards(split_dir)
        if n_cleared:
            print(f"{split_name}: cleared {n_cleared} existing shard(s) in {split_dir}", flush=True)

    tcr_loader, pep_loader, hla_loader = build_clean_loaders(
        tokenizer=tokenizer,
        tcrs_data=split_data["tcrs_data"],
        peptides_data=split_data["peptides_data"],
        hlas_data=split_data["hlas_data"],
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    chunk: List[Dict[str, Any]] = []
    shard_id = 0
    pair_ptr = 0
    binding_flags = split_data["binding_flags"]
    pair_ids = split_data["pair_ids"]
    shard_paths: List[str] = []

    model.eval()

    for i, (tcr_batch, pep_batch, hla_batch) in enumerate(zip(tcr_loader, pep_loader, hla_loader)):
        tcr_ids = tcr_batch["input_ids"].to(device, non_blocking=True)
        pep_ids = pep_batch["input_ids"].to(device, non_blocking=True)
        hla_ids = hla_batch["input_ids"].to(device, non_blocking=True)

        tcr_mask = tcr_batch["attention_mask"].to(torch.bool).cpu()
        pep_mask = pep_batch["attention_mask"].to(torch.bool).cpu()
        hla_mask = hla_batch["attention_mask"].to(torch.bool).cpu()

        emb_T = cast_embeddings(model(sequence_tokens=tcr_ids).embeddings, dtype).cpu()
        emb_P = cast_embeddings(model(sequence_tokens=pep_ids).embeddings, dtype).cpu()
        emb_H = cast_embeddings(model(sequence_tokens=hla_ids).embeddings, dtype).cpu()

        bsz = emb_T.shape[0]
        batch_binding = binding_flags[pair_ptr : pair_ptr + bsz]
        batch_pair_ids = pair_ids[pair_ptr : pair_ptr + bsz]

        chunk.append(
            {
                "emb_T": emb_T,
                "emb_P": emb_P,
                "emb_H": emb_H,
                "mask_T": tcr_mask,
                "mask_P": pep_mask,
                "mask_H": hla_mask,
                "binding_flag": torch.tensor(batch_binding, dtype=torch.long),
                "pair_id": batch_pair_ids,
                "batch_idx": i,
                "split": split_name,
            }
        )
        pair_ptr += bsz

        del tcr_ids, pep_ids, hla_ids

        if device.type == "cuda" and (i % 50 == 0):
            torch.cuda.empty_cache()

        if len(chunk) >= chunk_size:
            out_path = split_dir / f"shard_{shard_id:05d}.pt"
            torch.save(chunk, out_path)
            shard_paths.append(str(out_path))
            shard_id += 1
            chunk.clear()
            gc.collect()

    if chunk:
        out_path = split_dir / f"shard_{shard_id:05d}.pt"
        torch.save(chunk, out_path)
        shard_paths.append(str(out_path))
        chunk.clear()
        gc.collect()

    if pair_ptr != len(pair_ids):
        raise RuntimeError(
            f"{split_name}: processed {pair_ptr} rows but expected {len(pair_ids)}. "
            "Loader batching may be misaligned across modalities."
        )

    summary = {
        "split": split_name,
        "n_rows": int(len(pair_ids)),
        "n_shards": len(shard_paths),
        "n_cleared": n_cleared,
        "shard_paths": shard_paths,
        "split_dir": str(split_dir),
        "source_csv": split_data["source_csv"],
    }
    print(
        f"{split_name}: saved {summary['n_shards']} shard(s) | rows={summary['n_rows']} | dir={split_dir}",
        flush=True,
    )
    return summary


def verify_shards(
    embed_root: Path,
    splits: Sequence[str],
    reference_root: Optional[Path] = None,
) -> Dict[str, Any]:
    expected_keys = {
        "emb_T",
        "emb_P",
        "emb_H",
        "mask_T",
        "mask_P",
        "mask_H",
        "binding_flag",
        "pair_id",
        "batch_idx",
        "split",
    }
    report: Dict[str, Any] = {"splits": {}, "ok": True}

    for split in splits:
        split_dir = embed_root / split
        shard_paths = sorted(split_dir.glob("shard_*.pt"))
        if not shard_paths:
            report["ok"] = False
            report["splits"][split] = {"ok": False, "error": f"No shards in {split_dir}"}
            continue

        shard = torch.load(shard_paths[0], map_location="cpu")
        if not shard:
            report["ok"] = False
            report["splits"][split] = {"ok": False, "error": f"Empty shard: {shard_paths[0]}"}
            continue

        batch = shard[0]
        keys = set(batch.keys())
        missing = expected_keys - keys
        extra = keys - expected_keys
        split_report: Dict[str, Any] = {
            "ok": not missing,
            "n_shards": len(shard_paths),
            "first_shard": str(shard_paths[0]),
            "keys": sorted(keys),
            "missing_keys": sorted(missing),
            "extra_keys": sorted(extra),
            "emb_T_shape": tuple(batch["emb_T"].shape),
            "emb_P_shape": tuple(batch["emb_P"].shape),
            "emb_H_shape": tuple(batch["emb_H"].shape),
            "emb_T_dtype": str(batch["emb_T"].dtype),
            "mask_T_dtype": str(batch["mask_T"].dtype),
            "binding_flag_dtype": str(batch["binding_flag"].dtype),
            "sample_pair_id": batch["pair_id"][0],
            "sample_split": batch["split"],
        }

        if reference_root is not None:
            ref_dir = reference_root / split
            ref_shards = sorted(ref_dir.glob("shard_*.pt"))
            if ref_shards:
                ref_batch = torch.load(ref_shards[0], map_location="cpu")[0]
                split_report["reference_root"] = str(reference_root)
                split_report["reference_emb_T_shape"] = tuple(ref_batch["emb_T"].shape[1:])
                split_report["reference_emb_P_shape"] = tuple(ref_batch["emb_P"].shape[1:])
                split_report["reference_emb_H_shape"] = tuple(ref_batch["emb_H"].shape[1:])
                split_report["reference_keys_match"] = set(ref_batch.keys()) == keys

        if missing:
            report["ok"] = False
        report["splits"][split] = split_report

    return report


def split_csv_map(cfg: ExportConfig) -> Dict[str, str]:
    return {
        "train": cfg.train_csv,
        "val": cfg.val_csv,
        "test": cfg.test_csv,
        "immrep_test": cfg.immrep_csv,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frozen ESMC embedding shards.")
    parser.add_argument("--train-csv", default=ExportConfig.train_csv)
    parser.add_argument("--val-csv", default=ExportConfig.val_csv)
    parser.add_argument("--test-csv", default=ExportConfig.test_csv)
    parser.add_argument("--immrep-csv", default=ExportConfig.immrep_csv)
    parser.add_argument("--embed-root", default=ExportConfig.embed_root)
    parser.add_argument("--model-name", default=ExportConfig.model_name)
    parser.add_argument("--batch-size", type=int, default=ExportConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=ExportConfig.num_workers)
    parser.add_argument("--chunk-size", type=int, default=ExportConfig.chunk_size)
    parser.add_argument("--dtype", choices=("float16", "float32"), default=ExportConfig.dtype)
    parser.add_argument("--device", default=ExportConfig.device)
    parser.add_argument(
        "--splits",
        default="train,val,test,immrep_test",
        help="Comma-separated splits to export.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap per split for smoke tests.",
    )
    parser.add_argument(
        "--clear-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear shard_*.pt in each requested split dir before writing (default: true).",
    )
    parser.add_argument(
        "--reference-root",
        default=str(REPO / "models/embeddings/no_boltz_multiview_ids"),
        help="Optional fine-tuned shard root for structural verification.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify exported shard keys/shapes after writing.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification on an existing embed root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
    embed_root = Path(args.embed_root)
    embed_root.mkdir(parents=True, exist_ok=True)

    cfg = ExportConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        immrep_csv=args.immrep_csv,
        embed_root=str(embed_root),
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        device=args.device,
        splits=splits,
        max_rows=args.max_rows,
        clear_existing=bool(args.clear_existing),
    )

    run_config_path = embed_root / "export_raw_esmc_run_config.json"
    with open(run_config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 80, flush=True)
    print("Raw ESMC shard export", flush=True)
    print(f"Embed root: {embed_root}", flush=True)
    print(f"Run config: {run_config_path}", flush=True)
    print(f"Model: {cfg.model_name}", flush=True)
    print(f"Splits: {', '.join(splits)}", flush=True)
    print(f"Clear existing shards: {cfg.clear_existing}", flush=True)
    if cfg.max_rows is not None:
        print(f"Max rows per split: {cfg.max_rows}", flush=True)
    print("=" * 80, flush=True)

    if args.verify_only:
        ref_root = Path(args.reference_root) if args.reference_root else None
        report = verify_shards(embed_root, splits, reference_root=ref_root)
        report_path = embed_root / "export_raw_esmc_verify_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        print(f"Verify report: {report_path}", flush=True)
        if not report["ok"]:
            sys.exit(1)
        return

    if not torch.cuda.is_available() and cfg.device.startswith("cuda"):
        print("CUDA not available; falling back to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    from esm.models.esmc import ESMC

    print(f"Loading {cfg.model_name} on {device} ...", flush=True)
    model = ESMC.from_pretrained(cfg.model_name).to(device).eval()
    tokenizer = model.tokenizer

    csv_map = split_csv_map(cfg)
    split_summaries: List[Dict[str, Any]] = []

    for split_name in splits:
        if split_name not in csv_map:
            raise ValueError(f"Unknown split '{split_name}'. Expected one of {list(csv_map)}")
        csv_path = csv_map[split_name]
        print(f"\n[{split_name}] loading {csv_path}", flush=True)
        split_data = load_split_sequences(csv_path, max_rows=cfg.max_rows)
        summary = save_embedding_shards_for_split(
            split_name=split_name,
            split_data=split_data,
            model=model,
            tokenizer=tokenizer,
            save_root=embed_root,
            device=device,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            chunk_size=cfg.chunk_size,
            dtype=cfg.dtype,
            clear_existing=cfg.clear_existing,
        )
        split_summaries.append(summary)

    manifest = {
        "config": asdict(cfg),
        "device_used": str(device),
        "model_name": cfg.model_name,
        "embed_root": str(embed_root),
        "run_config_path": str(run_config_path),
        "splits": split_summaries,
    }
    manifest_path = embed_root / "export_raw_esmc_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("Export complete.", flush=True)
    print(f"Manifest: {manifest_path}", flush=True)
    for s in split_summaries:
        print(f"  {s['split']}: {s['n_rows']} rows -> {s['split_dir']}", flush=True)

    if args.verify:
        ref_root = Path(args.reference_root) if args.reference_root else None
        report = verify_shards(embed_root, splits, reference_root=ref_root)
        report_path = embed_root / "export_raw_esmc_verify_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Verify report: {report_path}", flush=True)
        print(json.dumps(report, indent=2), flush=True)
        if not report["ok"]:
            sys.exit(1)


if __name__ == "__main__":
    main()
