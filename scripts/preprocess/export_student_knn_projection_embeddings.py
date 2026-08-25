#!/usr/bin/env python3
"""Export raw-ESMC VICReg projection embeddings for student KNN delivery.

Assumes raw ESMC shards already exist for:
  - train positives: models/embeddings/raw_esmc_300m_multiview_ids/train
  - train matched negatives: models/embeddings/raw_esmc_300m_matched_negatives_knn/train
  - val/test: models/embeddings/raw_esmc_300m_multiview_ids/{val,test}

Loads workshop raw VICReg seed checkpoint and writes compact npz files:
  zT_esm_vicreg, zPH_esm_vicreg, pair_id, peptide, label
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
WORKSHOP = REPO / "scripts" / "train" / "workshop"
if str(WORKSHOP) not in sys.path:
    sys.path.insert(0, str(WORKSHOP))

from esm_vicreg_common import (  # noqa: E402
    ESMProjectionHead,
    PMHCProjectionHead,
    PairedESMRowDataset,
    build_pair_index,
    esm_collate,
    evaluate,
    load_meta,
    make_loader,
)


DEFAULT_SPLITS = {
    "train": REPO / "data/train/train_multiview_negative_decoys.csv",
    "val": REPO / "data/val/val_multiview.csv",
    "test": REPO / "data/test/test_multiview.csv",
}


def link_combined_train_shards(
    positive_dir: Path,
    negative_dir: Path,
    combined_dir: Path,
) -> Path:
    combined_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous links/files but keep directory.
    for p in combined_dir.glob("shard_*.pt"):
        p.unlink()

    idx = 0
    for src_dir, tag in ((positive_dir, "pos"), (negative_dir, "neg")):
        shards = sorted(Path(src_dir).glob("shard_*.pt"))
        if not shards:
            raise FileNotFoundError(f"No shards in {src_dir}")
        for sp in shards:
            dst = combined_dir / f"shard_{idx:05d}_{tag}.pt"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(sp.resolve())
            idx += 1
    print(f"Combined train shards: {idx} links -> {combined_dir}", flush=True)
    return combined_dir


class MultiDirESMRowDataset(PairedESMRowDataset):
    """Like PairedESMRowDataset but indexes one or more raw-shard directories."""

    def __init__(
        self,
        shard_dirs: List[Path],
        meta: pd.DataFrame,
        source_name: str,
        order_by_shard: bool = True,
    ):
        self.finetuned_dir = Path(shard_dirs[0])
        self.pretrained_dir = None
        self.include_pretrained = False
        self.meta = meta.reset_index(drop=True)
        self.meta_by_pid = {str(r["pair_id"]): r for _, r in self.meta.iterrows()}
        self.source_name = source_name

        self.ft_index: Dict[str, Tuple[Path, int, int]] = {}
        for d in shard_dirs:
            part = build_pair_index(Path(d), source_name, Path(d).name)
            overlap = set(self.ft_index) & set(part)
            if overlap:
                raise RuntimeError(f"{source_name}: overlapping pair_ids across shard dirs, e.g. {list(overlap)[:5]}")
            self.ft_index.update(part)
        self.pre_index = {}

        requested = [str(x) for x in self.meta["pair_id"].tolist()]
        missing = [pid for pid in requested if pid not in self.ft_index]
        if missing:
            raise RuntimeError(
                f"{source_name}: missing shard rows for {len(missing)} pair_ids. Examples: {missing[:10]}"
            )
        if order_by_shard:
            requested = sorted(
                requested,
                key=lambda pid: (str(self.ft_index[pid][0]), int(self.ft_index[pid][1]), int(self.ft_index[pid][2])),
            )
        self.pair_ids = requested
        print(f"{source_name}: paired ESM rows kept={len(self.pair_ids)} | dirs={list(map(str, shard_dirs))}", flush=True)
        # Ordered export only needs the current shard resident (full multi-shard
        # cache would try to hold ~60GB of LoRA residue shards in RAM).
        self._ft_shard_cache: Dict[str, object] = {}
        self._pre_shard_cache: Dict[str, object] = {}
        self._max_cached_shards = 1

    def _load(self, sp: Path, branch: str):
        key = str(sp)
        if branch == "ft":
            cache = self._ft_shard_cache
        elif branch == "pre":
            cache = self._pre_shard_cache
        else:
            raise ValueError(branch)
        if key not in cache:
            # Evict oldest entries if over limit (Python 3.7+ dicts are ordered).
            while len(cache) >= int(self._max_cached_shards):
                cache.pop(next(iter(cache)))
            cache[key] = torch.load(sp, map_location="cpu")
            print(
                f"{self.source_name}: cached {branch} shard {Path(key).name} "
                f"({len(cache)} unique shards)",
                flush=True,
            )
        return cache[key]


def shapes_from_checkpoint(ckpt: dict) -> Tuple[int, int, int, int]:
    if "shapes" in ckpt:
        s = ckpt["shapes"]
        return int(s["D"]), int(s["L_T"]), int(s["L_P"]), int(s["L_H"])
    tcr_sd = ckpt["tcr_state_dict"]
    pmhc_sd = ckpt["pmhc_state_dict"]
    return (
        int(tcr_sd["B_c"].shape[0]),
        int(tcr_sd["A_c"].shape[0]),
        int(pmhc_sd["pep_encoder.A_c"].shape[0]),
        int(pmhc_sd["hla_encoder.A_c"].shape[0]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=31)
    p.add_argument(
        "--checkpoint",
        default=str(REPO / "models/checkpoints/workshop/esm_vicreg_raw_complete/seed_31/best.pt"),
    )
    p.add_argument(
        "--raw-embed-root",
        default=str(REPO / "models/embeddings/raw_esmc_300m_multiview_ids"),
    )
    p.add_argument(
        "--train-neg-embed-dir",
        default=str(REPO / "models/embeddings/raw_esmc_300m_matched_negatives_knn/train"),
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO / "models/outputs/student_knn_embeddings/raw_esmc_vicreg_seed31"),
    )
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--complete-only", action="store_true", default=False)
    p.add_argument(
        "--name-prefix",
        default="raw_esmc_vicreg",
        help="Filename stem prefix, e.g. raw_esmc_vicreg or finetuned_esmc_vicreg",
    )
    p.add_argument(
        "--model-label",
        default="Raw ESMC + VICReg",
        help="Human-readable model label written into README/manifest",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    D, L_T, L_P, L_H = shapes_from_checkpoint(ckpt)

    class _Cfg:
        pass

    run_cfg = _Cfg()
    for k, v in {
        "rL": 8, "rD": 16, "d": 128, "R_PH": 0.7, "dropout": 0.1,
        "alpha": 25.0, "beta": 25.0, "delta": 1.0, "gamma_var": 1.0, "eps_var": 1e-4,
        "eps_pool": 1e-8, "partial_auc_max_fpr": 0.1, "seed": args.seed,
        "batch_size": args.batch_size, "num_workers": args.num_workers,
    }.items():
        setattr(run_cfg, k, cfg.get(k, v))

    print("=" * 72, flush=True)
    print(f"Student KNN projection export | {args.model_label}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Output: {out_dir}", flush=True)
    print(f"Shapes: D={D}, L_T={L_T}, L_P={L_P}, L_H={L_H}, d={run_cfg.d}", flush=True)
    print(f"Splits: {splits} | complete_only={args.complete_only}", flush=True)
    print(f"Name prefix: {args.name_prefix}", flush=True)
    print("=" * 72, flush=True)

    tcr = ESMProjectionHead(D, run_cfg.rL, run_cfg.rD, run_cfg.d, L_T, run_cfg.dropout).to(device)
    pmhc = PMHCProjectionHead(D, run_cfg.rL, run_cfg.rD, run_cfg.d, L_P, L_H, run_cfg.R_PH, run_cfg.dropout).to(device)
    tcr.load_state_dict(ckpt["tcr_state_dict"])
    pmhc.load_state_dict(ckpt["pmhc_state_dict"])
    tcr.eval()
    pmhc.eval()

    raw_root = Path(args.raw_embed_root)
    train_neg_dir = Path(args.train_neg_embed_dir)
    name_prefix = args.name_prefix
    manifest = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "model": name_prefix,
        "model_label": args.model_label,
        "embed_root": str(raw_root),
        "train_neg_embed_dir": str(train_neg_dir),
        "embedding_keys": ["zT_esm_vicreg", "zPH_esm_vicreg"],
        "splits": {},
    }

    for split in splits:
        csv_path = DEFAULT_SPLITS[split]
        meta, audit = load_meta(str(csv_path), split, positives_only=False, complete_only=args.complete_only)

        if split == "train":
            ds = MultiDirESMRowDataset(
                shard_dirs=[raw_root / "train", train_neg_dir],
                meta=meta,
                source_name="train",
                order_by_shard=True,
            )
        else:
            ds = PairedESMRowDataset(
                finetuned_dir=raw_root / split,
                pretrained_dir=None,
                meta=meta,
                source_name=split,
                include_pretrained=False,
                order_by_finetuned_shard=True,
            )

        loader = make_loader(ds, args.batch_size, False, args.num_workers, args.seed)
        print(f"Projecting {split} ({len(ds)} rows) ...", flush=True)
        eval_obj = evaluate(
            loader,
            tcr,
            pmhc,
            device,
            run_cfg,
            split,
            save_latents=True,
            model_names=("esm_vicreg",),
            pretrained_meanpool_from_ft=True,
        )
        lat = eval_obj["latents"]
        # Compact delivery: projection outputs only.
        payload = {
            "zT_esm_vicreg": lat["zT_esm_vicreg"].astype(np.float32),
            "zPH_esm_vicreg": lat["zPH_esm_vicreg"].astype(np.float32),
            "pair_id": np.array(lat["pair_id"], dtype=str),
            "peptide": np.array(lat["peptide"], dtype=str),
            "label": np.array(lat["label"], dtype=np.int64),
        }
        out_path = out_dir / f"{split}_{name_prefix}_seed{args.seed}_latents.npz"
        np.savez_compressed(out_path, **payload)
        # Also save a lightweight CSV index for easy inspection.
        pd.DataFrame({
            "pair_id": payload["pair_id"],
            "peptide": payload["peptide"],
            "label": payload["label"],
        }).to_csv(out_dir / f"{split}_{name_prefix}_seed{args.seed}_index.csv", index=False)

        manifest["splits"][split] = {
            "csv": str(csv_path),
            "n_rows": int(payload["zT_esm_vicreg"].shape[0]),
            "z_dim": int(payload["zT_esm_vicreg"].shape[1]),
            "latents_npz": str(out_path),
            "audit": audit,
            "metrics_esm_vicreg": eval_obj["metrics"]["esm_vicreg"],
        }
        print(f"Wrote {out_path} shape={payload['zT_esm_vicreg'].shape}", flush=True)
        if hasattr(ds, "clear_shard_cache"):
            ds.clear_shard_cache()

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    readme = out_dir / "README.txt"
    readme.write_text(
        "\n".join([
            f"{args.model_label} projection embeddings (seed {args.seed})",
            "=" * max(20, len(args.model_label) + 40),
            "",
            f"Checkpoint: {args.checkpoint}",
            f"Embed root: {raw_root}",
            f"Train negatives: {train_neg_dir}",
            "Outputs per split npz:",
            "  - zT_esm_vicreg:  float32 [N, 128]  TCR projection",
            "  - zPH_esm_vicreg: float32 [N, 128]  pMHC projection",
            "  - pair_id, peptide, label",
            "",
            "Scoring used in training/eval: score = -MSE(zT, zPH)",
            "Higher score => more binder-like.",
            "",
            "Splits:",
            "  train = train_multiview_negative_decoys.csv (positives + matched negatives)",
            "  val   = val_multiview.csv",
            "  test  = test_multiview.csv",
            "",
            "Load example:",
            "  import numpy as np",
            f"  z = np.load('train_{name_prefix}_seed{args.seed}_latents.npz')",
            "  zT, zPH = z['zT_esm_vicreg'], z['zPH_esm_vicreg']",
            "",
        ])
    )
    print(f"Manifest: {out_dir / 'manifest.json'}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
