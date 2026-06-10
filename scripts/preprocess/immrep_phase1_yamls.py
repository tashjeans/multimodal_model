#!/usr/bin/env python3
"""
Phase 1: IMMREP YAML archive + chunk symlinks (no MSA/Boltz moves).

- Migrate 4k legacy negative YAMLs -> data/immrep_test/_archive_root_yamls/immrep_pair_*.yaml
- Generate YAMLs for remaining pairs from master CSV + raw TCR splits
- Build data/immrep_test/_chunks/chunk_*/ symlinks (2000 per chunk, sorted)
"""

from __future__ import annotations

import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Optional

import pandas as pd

BASE_DIR = Path("/home/natasha/multimodal_model")
RELOC_MAP = BASE_DIR / "data/immrep_test/immrep_negative_reloc_map.csv"
IMMREP_MASTER = BASE_DIR / "data/immrep_test/immrep_test_set_pair_id.csv"
IMMREP_RAW = BASE_DIR / "data/raw/immrep2025_test_set.csv"
MANIFEST = BASE_DIR / "manifests/immrep_test_manifest.csv"
ARCHIVE_DIR = BASE_DIR / "data/immrep_test/_archive_root_yamls"
CHUNK_ROOT = BASE_DIR / "data/immrep_test/_chunks"
MSA_ROOT = BASE_DIR / "data/raw/MSA/jackhmmer_msas/immrep_test"
REPORT_OUT = BASE_DIR / "data/immrep_test/phase1_validation_report.json"

CHUNK_SIZE = 2000
MISSING_TOKENS = {"", "<unk>", "<UNK>", "UNK", "UNKNOWN", "NA", "N/A", "NONE", "NULL", "NAN"}


def clean_seq(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^A-Za-z]", "", s).upper()


def normalise_seq_for_yaml(s) -> str:
    s0 = "" if not isinstance(s, str) else str(s).strip()
    if s0 in {"<unk>", "<UNK>"}:
        return ""
    s = clean_seq(s0)
    return "" if s in MISSING_TOKENS else s


def msa_path_if_exists(pair_id: str, stem: str) -> Optional[Path]:
    p = MSA_ROOT / pair_id / f"{stem}.filt.a3m"
    return p if p.is_file() else None


def relpath_from_base(p: Path, base: Path) -> str:
    return str(p.relative_to(base))


def make_yaml(
    base_dir: Path,
    tcra_seq: str,
    tcrb_seq: str,
    pep: str,
    mhc_seq: str,
    tcra_msa: Optional[Path],
    tcrb_msa: Optional[Path],
    mhc_msa: Optional[Path],
) -> str:
    def msa_field(p: Optional[Path]) -> str:
        return "empty" if p is None else relpath_from_base(p, base_dir)

    tcra_seq = normalise_seq_for_yaml(tcra_seq)
    tcrb_seq = normalise_seq_for_yaml(tcrb_seq)
    pep = normalise_seq_for_yaml(pep)
    mhc_seq = normalise_seq_for_yaml(mhc_seq)

    lines = ["version: 1", "sequences:"]

    def add(pid: str, seq: str, msa: str):
        lines.append(
            textwrap.dedent(
                f"""\
        - protein:
            id: {pid}
            sequence: {seq}
            msa: {msa}
        """
            ).rstrip()
        )

    if tcra_seq:
        add("A", tcra_seq, msa_field(tcra_msa))
    if tcrb_seq:
        add("B", tcrb_seq, msa_field(tcrb_msa))
    if pep:
        add("C", pep, "empty")
    if mhc_seq:
        add("D", mhc_seq, msa_field(mhc_msa))

    return "\n".join(lines) + "\n"


def attach_tcra_tcrb(master: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(IMMREP_RAW)
    raw["TCR_full_key"] = raw["tcra_trimmed"].map(normalise_seq_for_yaml) + raw["tcrb_trimmed"].map(
        normalise_seq_for_yaml
    )
    master = master.copy()
    master["TCR_full_key"] = master["TCR_full"].map(normalise_seq_for_yaml)
    raw_lut = raw.drop_duplicates("TCR_full_key").set_index("TCR_full_key")
    master["TCRa"] = master["TCR_full_key"].map(
        lambda k: raw_lut.loc[k, "tcra_trimmed"] if k in raw_lut.index else ""
    )
    master["TCRb"] = master["TCR_full_key"].map(
        lambda k: raw_lut.loc[k, "tcrb_trimmed"] if k in raw_lut.index else ""
    )
    return master


def migrate_legacy_yaml(row: pd.Series) -> tuple[bool, str]:
    """Copy legacy YAML with MSA paths retargeted to immrep_test/{immrep_pair_id}."""
    old_path = BASE_DIR / str(row["old_yaml_path"])
    new_path = BASE_DIR / str(row["new_yaml_path"])
    if not old_path.is_file():
        return False, "missing_old_yaml"
    if new_path.exists() and new_path.stat().st_size > 50:
        return True, "skipped_exists"

    old_msa = str(row["old_msa_dir"]).replace("\\", "/")
    new_msa = str(row["new_msa_dir"]).replace("\\", "/")
    text = old_path.read_text()
    # Replace full old MSA dir prefix; also handle legacy split paths embedded in yaml.
    text = text.replace(old_msa, new_msa)
    legacy = str(row["legacy_pair_id"])
    text = text.replace(
        f"jackhmmer_msas/test/{legacy}",
        f"jackhmmer_msas/immrep_test/{row['immrep_pair_id']}",
    )
    text = text.replace(
        f"jackhmmer_msas/val/{legacy}",
        f"jackhmmer_msas/immrep_test/{row['immrep_pair_id']}",
    )

    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(text)
    return True, "migrated"


def generate_yaml_row(row: pd.Series) -> tuple[bool, str]:
    pair_id = str(row["pair_id"])
    new_path = ARCHIVE_DIR / f"{pair_id}.yaml"
    if new_path.exists() and new_path.stat().st_size > 50:
        return True, "skipped_exists"

    tcra = row.get("TCRa", "")
    tcrb = row.get("TCRb", "")
    pep = row["Peptide"]
    mhc = row["HLA_sequence"]

    tcra_msa = msa_path_if_exists(pair_id, "tcra")
    tcrb_msa = msa_path_if_exists(pair_id, "tcrb")
    mhc_msa = msa_path_if_exists(pair_id, "mhc")

    content = make_yaml(BASE_DIR, tcra, tcrb, pep, mhc, tcra_msa, tcrb_msa, mhc_msa)
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_text(content)
    return True, "generated"


def build_chunk_symlinks(manifest: pd.DataFrame, *, overwrite: bool = False) -> dict:
    CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    yamls = sorted(ARCHIVE_DIR.glob("*.yaml"), key=lambda p: p.name)
    if len(yamls) != len(manifest):
        print(f"[warn] archive yamls={len(yamls)} manifest rows={len(manifest)}", flush=True)

    n_chunks = math.ceil(len(yamls) / CHUNK_SIZE)
    created = 0
    skipped = 0
    for i in range(n_chunks):
        chunk_dir = CHUNK_ROOT / f"chunk_{i:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for src in yamls[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]:
            dst = chunk_dir / src.name
            if dst.exists() or dst.is_symlink():
                if overwrite:
                    dst.unlink()
                else:
                    skipped += 1
                    continue
            os.symlink(src.resolve(), dst)
            created += 1

    chunk_counts = {
        p.name: len(list(p.glob("*.yaml")) + list(p.glob("*.yml")))
        for p in sorted(CHUNK_ROOT.glob("chunk_*"))
    }
    return {"symlinks_created": created, "symlinks_skipped": skipped, "chunk_counts": chunk_counts}


def main() -> None:
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading relocation map...", flush=True)
    reloc = pd.read_csv(RELOC_MAP)

    print("Migrating 4k legacy YAMLs...", flush=True)
    mig_stats: dict[str, int] = {}
    for _, row in reloc.iterrows():
        _, status = migrate_legacy_yaml(row)
        mig_stats[status] = mig_stats.get(status, 0) + 1
    print(f"  migrate: {mig_stats}", flush=True)

    print("Loading master CSV for remaining YAML generation...", flush=True)
    master = pd.read_csv(IMMREP_MASTER)
    master = attach_tcra_tcrb(master)
    master = master.sort_values("pair_id").reset_index(drop=True)

    reloc_ids = set(reloc["immrep_pair_id"])
    remaining = master[~master["pair_id"].isin(reloc_ids)]
    print(f"Generating YAMLs for {len(remaining)} pairs not in reloc map...", flush=True)
    gen_stats: dict[str, int] = {}
    for _, row in remaining.iterrows():
        ok, status = generate_yaml_row(row)
        gen_stats[status] = gen_stats.get(status, 0) + 1

    # Also ensure any reloc row whose migrate failed gets a generated fallback
    archive_ids = {p.stem for p in ARCHIVE_DIR.glob("*.yaml")}
    missing_reloc = reloc[~reloc["immrep_pair_id"].isin(archive_ids)]
    if len(missing_reloc):
        print(f"Fallback generate for {len(missing_reloc)} failed migrations...", flush=True)
        lut = master.set_index("pair_id")
        for _, r in missing_reloc.iterrows():
            pid = r["immrep_pair_id"]
            ok, status = generate_yaml_row(lut.loc[pid])
            gen_stats[f"fallback_{status}"] = gen_stats.get(f"fallback_{status}", 0) + 1

    print(f"  generate: {gen_stats}", flush=True)

    print("Building _chunks symlinks...", flush=True)
    manifest = pd.read_csv(MANIFEST)
    chunk_report = build_chunk_symlinks(manifest)

    archive_count = len(list(ARCHIVE_DIR.glob("*.yaml")))
    report = {
        "archive_yaml_count": archive_count,
        "manifest_rows": len(manifest),
        "migrate_stats": mig_stats,
        "generate_stats": gen_stats,
        "chunk_symlinks": chunk_report,
        "msa_ready_under_immrep_test": sum(
            1
            for p in MSA_ROOT.iterdir()
            if p.is_dir() and (p / "tcra.filt.a3m").exists()
        )
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("Phase 1 complete.", flush=True)


if __name__ == "__main__":
    main()
