#!/usr/bin/env python3
"""Surgically remove val/test-leaking rows from train embedding shards (CPU only).

Atomic per-shard replace: write shard.tmp then os.replace over the original.
Keep-sets come from the already-cleaned train CSVs.
"""
from __future__ import annotations
import glob, os, sys
import pandas as pd
import torch

REPO = "/home/natasha/multimodal_model"
ROW_TENSOR_KEYS = ["emb_T", "emb_P", "emb_H", "mask_T", "mask_P", "mask_H", "binding_flag"]

def load_keepsets():
    tm = set(pd.read_csv(f"{REPO}/data/train/train_multiview.csv").pair_id.astype(str))
    bal_path = f"{REPO}/data/train/train_multiview_balanced_full_hard_decoys.csv"
    bal = pd.read_csv(bal_path, low_memory=False)
    # drop the single negative-vs-negative row that matches a val tulip decoy
    drop_id = "negative_train_full_hard_000459"
    before = len(bal)
    bal = bal[bal.pair_id.astype(str) != drop_id].copy()
    if len(bal) != before:
        bal.to_csv(bal_path, index=False)
        print(f"[balanced csv] dropped {drop_id}: {before} -> {len(bal)} "
              f"(binding {bal.binding_flag.value_counts().to_dict()})", flush=True)
    return {
        "raw_esmc_300m_multiview_ids": tm,
        "no_boltz_multiview_ids": tm,
        "KNN_embeddings_dir": set(bal.pair_id.astype(str)),
    }

def filter_batch(b, keep):
    pids = [str(x) for x in b["pair_id"]]
    mask = torch.tensor([pid in keep for pid in pids], dtype=torch.bool)
    if mask.all():
        return b, len(pids), 0  # unchanged
    nb = dict(b)
    for k in ROW_TENSOR_KEYS:
        if k in nb and torch.is_tensor(nb[k]):
            nb[k] = nb[k][mask]
    nb["pair_id"] = [pid for pid, m in zip(pids, mask.tolist()) if m]
    return nb, int(mask.sum()), int((~mask).sum())

def process_dir(d, keep):
    shard_paths = sorted(glob.glob(f"{REPO}/models/embeddings/{d}/train/shard_*.pt"))
    tot_kept = tot_removed = shards_rewritten = empty_dropped = 0
    for p in shard_paths:
        shard = torch.load(p, map_location="cpu", weights_only=False)
        new_shard = []
        changed = False
        for b in shard:
            nb, kept, removed = filter_batch(b, keep)
            tot_kept += kept; tot_removed += removed
            if removed:
                changed = True
            if kept == 0:
                empty_dropped += 1
                changed = True
                continue
            new_shard.append(nb)
        if changed:
            tmp = p + ".tmp"
            torch.save(new_shard, tmp)
            os.replace(tmp, p)
            shards_rewritten += 1
    print(f"[{d}] kept={tot_kept} removed={tot_removed} "
          f"shards_rewritten={shards_rewritten}/{len(shard_paths)} empty_batches_dropped={empty_dropped}",
          flush=True)
    return tot_kept

def verify(d, keep):
    kept_ids = set()
    rows = 0
    for p in sorted(glob.glob(f"{REPO}/models/embeddings/{d}/train/shard_*.pt")):
        for b in torch.load(p, map_location="cpu", weights_only=False):
            pids = [str(x) for x in b["pair_id"]]
            rows += len(pids); kept_ids.update(pids)
            bad = [pid for pid in pids if pid not in keep]
            if bad:
                raise SystemExit(f"VERIFY FAIL {d}: {len(bad)} leaky ids remain e.g. {bad[:5]}")
    missing = keep - kept_ids
    print(f"[verify {d}] rows={rows} unique={len(kept_ids)} keepset={len(keep)} "
          f"missing_from_shards={len(missing)} leaky_remaining=0", flush=True)

def main():
    keepsets = load_keepsets()
    for d, keep in keepsets.items():
        process_dir(d, keep)
    print("--- verification ---", flush=True)
    for d, keep in keepsets.items():
        verify(d, keep)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
