#!/usr/bin/env python3
"""Is the TCR representation (zT) identical for the same TCR across the two
delivery blocks (positives vs positives+matched-negatives)?

Context
-------
The VICReg model is a bi-encoder: ``zT = ESMProjectionHead(emb_T, mask_T)`` never
sees the peptide. Therefore, for a FIXED model (here: seed 31, raw ESMC + VICReg),
an identical ``TCR_full`` string must map to an identical ``zT`` regardless of
whether it appears in a positive or a (matched) negative pair.

Crucially, in the delivered artifacts the positives' residue embeddings were
exported in one ESMC pass (``raw_esmc_300m_multiview_ids/train``) and the matched
negatives' in a *separate* pass (``raw_esmc_300m_matched_negatives_knn/train``).
So a matched negative and its source positive share the same ``TCR_full`` but were
embedded by two different export runs -- the analogue of "I sent the data in two
blocks". If those two passes used the same model generation, ``zT`` must match to
numerical precision; a non-trivial difference is a provenance / model-generation
artifact, not biology.

This script quantifies the difference using the already-computed seed-31 raw
latents (no re-embedding needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/home/natasha/multimodal_model")
NPZ = REPO / "models/outputs/student_knn_embeddings/raw_esmc_vicreg_seed31/train_raw_esmc_vicreg_seed31_latents.npz"
CSV = REPO / "data/train/train_multiview_negative_decoys.csv"
OUT = REPO / "models/outputs/embedding_consistency"


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(NPZ, allow_pickle=True)
    zT = z["zT_esm_vicreg"].astype(np.float64)
    pair_id = z["pair_id"].astype(str)
    label = z["label"].astype(int)
    pid2row = {p: i for i, p in enumerate(pair_id)}

    # Typical scale of a zT vector, so distances are interpretable.
    zT_norm_median = float(np.median(np.linalg.norm(zT, axis=1)))

    df = pd.read_csv(CSV, dtype=str)
    pid2tcr = dict(zip(df["pair_id"], df["TCR_full"]))

    neg = df[df["binding_flag"] == "0"].copy()
    print(f"Loaded {len(pair_id)} latent rows | positives={int((label==1).sum())} "
          f"negatives={int((label==0).sum())}")
    print(f"Median ||zT|| = {zT_norm_median:.4f} (128-dim); interpret L2 distances "
          f"relative to this.\n")

    # --- Analysis A: matched negative zT vs its source positive zT -------------
    # These share TCR_full but were embedded in two SEPARATE ESMC export passes.
    l2, cos, tcr_mismatch, missing = [], [], 0, 0
    rows_worst = []
    for _, r in neg.iterrows():
        npid, spid = r["pair_id"], r.get("source_positive_pair_id")
        if not isinstance(spid, str) or npid not in pid2row or spid not in pid2row:
            missing += 1
            continue
        if pid2tcr.get(npid) != pid2tcr.get(spid):
            tcr_mismatch += 1
            continue
        a, b = zT[pid2row[npid]], zT[pid2row[spid]]
        d = float(np.linalg.norm(a - b))
        c = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        l2.append(d)
        cos.append(c)
        rows_worst.append((npid, spid, d, c, pid2tcr.get(npid)))

    l2 = np.asarray(l2)
    cos = np.asarray(cos)
    print("=" * 72)
    print("A) Matched negative zT vs source positive zT (same TCR, two export passes)")
    print("=" * 72)
    print(f"pairs compared        : {len(l2)}")
    print(f"skipped (missing pid) : {missing}")
    print(f"skipped (TCR mismatch): {tcr_mismatch}")
    if len(l2):
        print(f"L2 distance  mean={l2.mean():.6g} median={np.median(l2):.6g} "
              f"p99={pct(l2,99):.6g} max={l2.max():.6g}")
        print(f"L2 / ||zT||  median={np.median(l2)/zT_norm_median:.3%} "
              f"max={l2.max()/zT_norm_median:.3%}")
        print(f"cosine sim   mean={cos.mean():.8f} min={cos.min():.8f}")
        print(f"frac L2 < 1e-3 : {(l2 < 1e-3).mean():.3%}")
        print(f"frac L2 < 1e-2 : {(l2 < 1e-2).mean():.3%}")

    # --- Analysis B: exact TCR_full groups across ALL rows (pos+neg) -----------
    # Robust catch-all: any TCR_full appearing on >1 row should have ~identical zT.
    tcr_by_row = np.array([pid2tcr.get(p, "") for p in pair_id])
    order = np.argsort(tcr_by_row, kind="stable")
    group_spreads = []
    i = 0
    sorted_tcr = tcr_by_row[order]
    while i < len(order):
        j = i
        while j < len(order) and sorted_tcr[j] == sorted_tcr[i]:
            j += 1
        if sorted_tcr[i] and (j - i) > 1:
            idx = order[i:j]
            v = zT[idx]
            centroid = v.mean(axis=0)
            spread = float(np.max(np.linalg.norm(v - centroid, axis=1)))
            group_spreads.append(spread)
        i = j
    group_spreads = np.asarray(group_spreads)
    print()
    print("=" * 72)
    print("B) Exact TCR_full groups (max within-group L2 to centroid)")
    print("=" * 72)
    print(f"n TCR groups with >1 row : {len(group_spreads)}")
    if len(group_spreads):
        print(f"within-group spread  mean={group_spreads.mean():.6g} "
              f"median={np.median(group_spreads):.6g} "
              f"p99={pct(group_spreads,99):.6g} max={group_spreads.max():.6g}")

    # --- Persist results -------------------------------------------------------
    summary = {
        "checkpoint": "esm_vicreg_raw_complete/seed_31/best.pt",
        "latents_npz": str(NPZ),
        "csv": str(CSV),
        "zT_norm_median": zT_norm_median,
        "matched_neg_vs_source_pos": {
            "n_pairs": int(len(l2)),
            "n_skipped_missing": int(missing),
            "n_skipped_tcr_mismatch": int(tcr_mismatch),
            "l2_mean": float(l2.mean()) if len(l2) else None,
            "l2_median": float(np.median(l2)) if len(l2) else None,
            "l2_p99": pct(l2, 99),
            "l2_max": float(l2.max()) if len(l2) else None,
            "cos_mean": float(cos.mean()) if len(cos) else None,
            "cos_min": float(cos.min()) if len(cos) else None,
            "frac_l2_lt_1e-3": float((l2 < 1e-3).mean()) if len(l2) else None,
        },
        "exact_tcr_groups": {
            "n_groups": int(len(group_spreads)),
            "spread_mean": float(group_spreads.mean()) if len(group_spreads) else None,
            "spread_median": float(np.median(group_spreads)) if len(group_spreads) else None,
            "spread_max": float(group_spreads.max()) if len(group_spreads) else None,
        },
    }
    (OUT / "tcr_representation_consistency_summary.json").write_text(json.dumps(summary, indent=2))

    if len(rows_worst):
        worst = sorted(rows_worst, key=lambda t: -t[2])[:50]
        pd.DataFrame(worst, columns=["neg_pair_id", "source_pos_pair_id", "l2", "cosine", "TCR_full"]).to_csv(
            OUT / "worst_offenders.csv", index=False)

    # Histogram (log-x) of the matched-pair L2 distances.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if len(l2):
            safe = np.clip(l2, 1e-8, None)
            ax[0].hist(np.log10(safe), bins=60, color="#4C72B0")
            ax[0].axvline(np.log10(zT_norm_median), color="crimson", ls="--",
                          label=f"median ||zT||={zT_norm_median:.2f}")
            ax[0].set_xlabel("log10 L2( zT_neg , zT_source_pos )")
            ax[0].set_ylabel("count")
            ax[0].set_title("A) Same TCR across two export blocks")
            ax[0].legend()
        if len(group_spreads):
            safe = np.clip(group_spreads, 1e-8, None)
            ax[1].hist(np.log10(safe), bins=60, color="#55A868")
            ax[1].set_xlabel("log10 within-group L2 spread")
            ax[1].set_title("B) Exact TCR_full groups")
        fig.tight_layout()
        fig.savefig(OUT / "tcr_representation_consistency.png", dpi=140)
        print(f"\nWrote figure -> {OUT / 'tcr_representation_consistency.png'}")
    except Exception as e:  # pragma: no cover
        print(f"(plot skipped: {e})")

    print(f"Wrote summary -> {OUT / 'tcr_representation_consistency_summary.json'}")


if __name__ == "__main__":
    main()
