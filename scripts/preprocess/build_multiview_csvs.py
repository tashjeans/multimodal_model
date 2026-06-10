#!/usr/bin/env python3
"""Build train/val/test/immrep *_multiview.csv from data CSVs + manifests + Boltz NPZ paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

SPLITS: Dict[str, Dict[str, Path]] = {
    "train": {
        "data": REPO / "data/train/train_df_clean.csv",
        "manifest": REPO / "manifests/train_manifest.csv",
        "boltz_root": REPO / "outputs/train",
        "out": REPO / "data/train/train_multiview.csv",
    },
    "val": {
        "data": REPO / "data/val/val_df_clean_pos_tulip_decoys.csv",
        "manifest": REPO / "manifests/val_manifest.csv",
        "boltz_root": REPO / "outputs/val",
        "out": REPO / "data/val/val_multiview.csv",
    },
    "test": {
        "data": REPO / "data/test/test_df_clean_pos_tulip_decoys.csv",
        "manifest": REPO / "manifests/test_manifest.csv",
        "boltz_root": REPO / "outputs/test",
        "out": REPO / "data/test/test_multiview.csv",
    },
    "immrep_test": {
        "data": REPO / "data/immrep_test/immrep_test_set_pair_id.csv",
        "manifest": REPO / "manifests/immrep_test_manifest.csv",
        "boltz_root": REPO / "outputs_data/immrep_test",
        "out": REPO / "data/immrep_test/immrep_test_multiview.csv",
    },
}

MANIFEST_EXTRA = ["yaml_path", "pep_len", "tcra_len", "tcrb_len", "hla_len"]


def rel_repo_path(path: Path) -> str:
    p = path.resolve()
    for prefix in ("outputs", "outputs_data"):
        root = (REPO / prefix).resolve()
        try:
            return str(Path(prefix) / p.relative_to(root))
        except ValueError:
            continue
    try:
        return str(p.relative_to(REPO.resolve()))
    except ValueError:
        return str(p)


def is_pair_embedding_npz(path: Path) -> bool:
    name = path.name
    return (
        name.startswith("embeddings_")
        and "pae" not in name
        and "pde" not in name
        and "plddt" not in name
    )


def build_full_boltz_npz_index(boltz_root: Path) -> Dict[str, Path]:
    """Map pair_id -> pair-embedding NPZ for any Boltz results layout.

    Indexes pair_*, val_tulip_*, test_tulip_*, immrep_pair_*, etc. under
    chunk_*/boltz_results_*/predictions/<pair_id>/.
    """
    index: Dict[str, Path] = {}
    if not boltz_root.exists():
        return index

    seen_pred_dirs: set[Path] = set()
    patterns = [
        "chunk_*/boltz_results_*/predictions/*",
        "boltz_results_*/predictions/*",
    ]
    for pattern in patterns:
        for pred_dir in boltz_root.glob(pattern):
            if not pred_dir.is_dir() or pred_dir in seen_pred_dirs:
                continue
            seen_pred_dirs.add(pred_dir)
            pid = pred_dir.name
            candidates = sorted(pred_dir.glob(f"embeddings_{pid}.npz"))
            candidates.extend(sorted(pred_dir.glob("embeddings*.npz")))
            for candidate in candidates:
                if is_pair_embedding_npz(candidate):
                    index.setdefault(pid, candidate)
                    break
    return index


def build_split(name: str, cfg: Dict[str, Path]) -> Dict[str, Any]:
    data = pd.read_csv(cfg["data"])
    manifest = pd.read_csv(cfg["manifest"])

    if data["pair_id"].duplicated().any():
        dupes = data.loc[data["pair_id"].duplicated(keep=False), "pair_id"].unique()[:5]
        raise ValueError(f"{name}: duplicate pair_id in data: {dupes}")

    mcols = ["pair_id"] + [c for c in MANIFEST_EXTRA if c in manifest.columns]
    manifest_sub = manifest[mcols].drop_duplicates(subset=["pair_id"], keep="first")
    if manifest_sub["pair_id"].duplicated().any():
        raise ValueError(f"{name}: duplicate pair_id in manifest")

    missing_manifest = set(data["pair_id"].astype(str)) - set(manifest_sub["pair_id"].astype(str))
    if missing_manifest:
        sample = sorted(missing_manifest)[:5]
        raise ValueError(f"{name}: {len(missing_manifest)} data pair_id not in manifest (e.g. {sample})")

    extra_manifest = set(manifest_sub["pair_id"].astype(str)) - set(data["pair_id"].astype(str))
    if extra_manifest:
        raise ValueError(f"{name}: manifest has {len(extra_manifest)} pair_id not in data")

    merged = data.merge(manifest_sub, on="pair_id", how="left", suffixes=("", "_manifest"))
    if len(merged) != len(data):
        raise ValueError(f"{name}: merge changed row count {len(data)} -> {len(merged)}")

    if "binding_flag" in data.columns and "binding_flag" in manifest.columns:
        manifest_flags = manifest.set_index("pair_id").loc[merged["pair_id"], "binding_flag"].astype(int).values
        flag_mismatch = int((merged["binding_flag"].astype(int).values != manifest_flags).sum())
        if flag_mismatch:
            raise ValueError(f"{name}: binding_flag mismatch vs manifest on {flag_mismatch} rows")

    boltz_index = build_full_boltz_npz_index(cfg["boltz_root"])
    npz_paths = [
        rel_repo_path(boltz_index[pid]) if pid in boltz_index else ""
        for pid in merged["pair_id"].astype(str)
    ]
    merged["boltz_embedding_npz"] = npz_paths

    length_cols = ["pep_len", "tcra_len", "tcrb_len", "hla_len"]
    complete_mask = (merged[length_cols] > 0).all(axis=1)

    cfg["out"].parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(cfg["out"], index=False)

    has_npz = merged["boltz_embedding_npz"].astype(str).str.len() > 0
    report: Dict[str, Any] = {
        "split": name,
        "output": str(cfg["out"]),
        "rows": int(len(merged)),
        "unique_pair_id": int(merged["pair_id"].nunique()),
        "boltz_index_size": len(boltz_index),
        "boltz_npz_matched": int(has_npz.sum()),
        "boltz_npz_missing": int((~has_npz).sum()),
        "complete_alpha_beta_pmhc": int(complete_mask.sum()),
        "incomplete_chain_lengths": int((~complete_mask).sum()),
    }

    if "binding_flag" in merged.columns:
        for label in sorted(merged["binding_flag"].astype(int).unique()):
            sub = merged[merged["binding_flag"].astype(int) == label]
            report[f"boltz_npz_binding_{label}"] = int(sub["boltz_embedding_npz"].astype(str).str.len().gt(0).sum())
            report[f"rows_binding_{label}"] = int(len(sub))

    if "decoy_type" in merged.columns:
        report["decoy_type_counts"] = {str(k): int(v) for k, v in merged["decoy_type"].value_counts().items()}

    missing_files = []
    for p in merged.loc[has_npz, "boltz_embedding_npz"]:
        if not (REPO / p).exists():
            missing_files.append(p)
    if missing_files:
        raise ValueError(f"{name}: {len(missing_files)} listed NPZ paths do not exist (e.g. {missing_files[:3]})")

    manifest_lengths = manifest.set_index("pair_id")
    for col in length_cols:
        if not (merged[col].astype(int).values == manifest_lengths.loc[merged["pair_id"], col].astype(int).values).all():
            raise ValueError(f"{name}: {col} does not match manifest after merge")

    return report


def main() -> None:
    reports = [build_split(name, cfg) for name, cfg in SPLITS.items()]
    report_path = REPO / "data/multiview_build_report.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
    for r in reports:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
