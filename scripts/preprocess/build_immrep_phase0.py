#!/usr/bin/env python3
"""
Phase 0: build IMMREP relocation map (4k legacy negatives) and full 10k manifest.

No file moves or renames — CSV outputs and validation report only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

BASE_DIR = Path("/home/natasha/multimodal_model")
IMMREP_MASTER = BASE_DIR / "data/immrep_test/immrep_test_set_pair_id.csv"
IMMREP_RAW = BASE_DIR / "data/raw/immrep2025_test_set.csv"
TEST_CSV = BASE_DIR / "data/test/test_df_clean_pos_neg.csv"
VAL_CSV = BASE_DIR / "data/val/val_df_clean_pos_neg.csv"
OUTPUTS_ROOT = BASE_DIR / "outputs"
OUTPUTS_DATA_IMMREP = BASE_DIR / "outputs_data/immrep_test"

RELOC_MAP_OUT = BASE_DIR / "data/immrep_test/immrep_negative_reloc_map.csv"
MANIFEST_OUT = BASE_DIR / "manifests/immrep_test_manifest.csv"
REPORT_OUT = BASE_DIR / "data/immrep_test/phase0_validation_report.json"

CHUNK_SIZE = 2000
ARCHIVE_YAML = "data/immrep_test/_archive_root_yamls"
CHUNK_YAML_TMPL = "data/immrep_test/_chunks/{chunk}/{pair_id}.yaml"
MSA_ROOT = "data/raw/MSA/jackhmmer_msas/immrep_test"
BOLTZ_ROOT = "outputs_data/immrep_test"


def normalise_seq(s: str) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = str(s).strip().upper()
    return s if s and s.lower() != "nan" else ""


def seq_key(peptide: str, hla: str, tcr_full: str) -> str:
    return "|".join([normalise_seq(peptide), normalise_seq(hla), normalise_seq(tcr_full)])


def chunk_name_for_index(idx: int, chunk_size: int = CHUNK_SIZE) -> str:
    return f"chunk_{idx // chunk_size:03d}"


def build_boltz_index(splits: tuple[str, ...] = ("test", "val")) -> dict[tuple[str, str], str]:
    """One-time scan: (split, legacy_pair_id) -> relative boltz results dir."""
    index: dict[tuple[str, str], str] = {}
    for split in splits:
        root = OUTPUTS_ROOT / split
        if not root.is_dir():
            continue
        for p in root.glob("chunk_*/boltz_results_*"):
            if not p.is_dir():
                continue
            legacy = p.name.replace("boltz_results_", "", 1)
            index[(split, legacy)] = str(p.relative_to(BASE_DIR))
    return index


def msa_done(msa_dir: Path) -> bool:
    if not msa_dir.is_dir():
        return False
    for stem in ("tcra", "tcrb", "mhc"):
        if not (msa_dir / f"{stem}.filt.a3m").exists():
            return False
    return True


def yaml_done(yaml_path: Path) -> bool:
    return yaml_path.is_file() and yaml_path.stat().st_size > 50


def attach_tcra_tcrb_from_raw(master: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(IMMREP_RAW)
    raw = raw.copy()
    raw["TCR_full_key"] = (
        raw["tcra_trimmed"].map(normalise_seq) + raw["tcrb_trimmed"].map(normalise_seq)
    )
    raw["seq_key"] = (
        raw["peptide"].map(normalise_seq)
        + "|"
        + raw["hla_sequence"].map(normalise_seq)
        + "|"
        + raw["TCR_full_key"]
    )
    master = master.copy()
    master["TCR_full_key"] = master["TCR_full"].map(normalise_seq)
    master["seq_key"] = (
        master["Peptide"].map(normalise_seq)
        + "|"
        + master["HLA_sequence"].map(normalise_seq)
        + "|"
        + master["TCR_full_key"]
    )
    raw_lut = raw.drop_duplicates("seq_key").set_index("seq_key")
    master["TCRa"] = master["seq_key"].map(lambda k: raw_lut.loc[k, "tcra_trimmed"] if k in raw_lut.index else "")
    master["TCRb"] = master["seq_key"].map(lambda k: raw_lut.loc[k, "tcrb_trimmed"] if k in raw_lut.index else "")
    n_miss = (master["TCRa"] == "").sum()
    if n_miss:
        print(f"[warn] {n_miss} rows missing TCRa/TCRb after raw join; lengths may be 0", flush=True)
    return master


def build_reloc_map(master: pd.DataFrame, boltz_index: dict[tuple[str, str], str]) -> pd.DataFrame:
    test = pd.read_csv(TEST_CSV)
    val = pd.read_csv(VAL_CSV)
    neg_test = test[test["binding_flag"] == 0].copy()
    neg_val = val[val["binding_flag"] == 0].copy()
    neg_test["from_test"] = True
    neg_test["from_val"] = False
    neg_val["from_test"] = False
    neg_val["from_val"] = True
    neg_all = pd.concat([neg_test, neg_val], ignore_index=True)

    master_keys = master["Peptide"].map(normalise_seq) + "|" + master["HLA_sequence"].map(normalise_seq) + "|" + master["TCR_full"].map(normalise_seq)
    immrep_lut = dict(zip(master_keys, master["pair_id"]))
    pair_to_chunk = dict(zip(master["pair_id"], [chunk_name_for_index(i) for i in range(len(master))]))

    rows = []
    for _, r in neg_all.iterrows():
        legacy = str(r["pair_id"]).strip()
        split = "test" if r["from_test"] else "val"
        key = seq_key(r["Peptide"], r["HLA_sequence"], r["TCR_full"])
        immrep_id = immrep_lut.get(key)
        if immrep_id is None:
            rows.append(
                {
                    "immrep_pair_id": "",
                    "legacy_pair_id": legacy,
                    "from_test": bool(r["from_test"]),
                    "from_val": bool(r["from_val"]),
                    "pair_id_negative": r.get("pair_id_negative", ""),
                    "relocate_status": "sequence_not_in_master",
                }
            )
            continue

        chunk = pair_to_chunk[immrep_id]

        old_yaml = BASE_DIR / f"data/{split}/_archive_root_yamls/{legacy}.yaml"
        old_msa = BASE_DIR / f"data/raw/MSA/jackhmmer_msas/{split}/{legacy}"
        old_boltz = boltz_index.get((split, legacy))

        new_yaml = f"{ARCHIVE_YAML}/{immrep_id}.yaml"
        new_chunk_yaml = CHUNK_YAML_TMPL.format(chunk=chunk, pair_id=immrep_id)
        new_msa = f"{MSA_ROOT}/{immrep_id}"
        new_boltz = f"{BOLTZ_ROOT}/{chunk}/boltz_results_{immrep_id}"

        status = "ready"
        if not yaml_done(old_yaml):
            status = "missing_old_yaml"
        elif not msa_done(old_msa):
            status = "missing_old_msa"
        elif old_boltz is None:
            status = "missing_old_boltz"

        rows.append(
            {
                "immrep_pair_id": immrep_id,
                "legacy_pair_id": legacy,
                "from_test": bool(r["from_test"]),
                "from_val": bool(r["from_val"]),
                "pair_id_negative": r.get("pair_id_negative", ""),
                "Peptide": r["Peptide"],
                "HLA_sequence": r["HLA_sequence"],
                "TCR_full": r["TCR_full"],
                "binding_flag": int(r["binding_flag"]),
                "immrep_chunk": chunk,
                "old_yaml_path": str(old_yaml.relative_to(BASE_DIR)) if old_yaml.exists() else "",
                "old_msa_dir": str(old_msa.relative_to(BASE_DIR)) if old_msa.is_dir() else "",
                "old_boltz_dir": old_boltz or "",
                "new_yaml_path": new_yaml,
                "new_chunk_yaml_path": new_chunk_yaml,
                "new_msa_dir": new_msa,
                "new_boltz_dir": new_boltz,
                "relocate_status": status,
            }
        )

    df = pd.DataFrame(rows)
    # stable sort for review
    return df.sort_values(["immrep_pair_id", "from_test"], ascending=[True, False]).reset_index(drop=True)


def build_manifest(master: pd.DataFrame) -> pd.DataFrame:
    master = master.sort_values("pair_id").reset_index(drop=True)
    master["chunk"] = [chunk_name_for_index(i) for i in range(len(master))]

    rows = []
    for i, r in master.iterrows():
        pid = r["pair_id"]
        pep = normalise_seq(r["Peptide"])
        mhc = normalise_seq(r["HLA_sequence"])
        tcra = normalise_seq(r.get("TCRa", ""))
        tcrb = normalise_seq(r.get("TCRb", ""))
        rows.append(
            {
                "pair_id": pid,
                "yaml_path": f"{ARCHIVE_YAML}/{pid}.yaml",
                "pep_len": len(pep),
                "tcra_len": len(tcra),
                "tcrb_len": len(tcrb),
                "hla_len": len(mhc),
                "binding_flag": int(r["binding_flag"]),
            }
        )
    return pd.DataFrame(rows)


def validate(reloc: pd.DataFrame, manifest: pd.DataFrame, master: pd.DataFrame) -> dict:
    report: dict = {}
    report["master_rows"] = len(master)
    report["manifest_rows"] = len(manifest)
    report["reloc_rows"] = len(reloc)
    report["reloc_unique_immrep_pair_id"] = int(reloc["immrep_pair_id"].nunique())
    report["reloc_status_counts"] = reloc["relocate_status"].value_counts().to_dict()
    report["reloc_from_test"] = int(reloc["from_test"].sum())
    report["reloc_from_val"] = int(reloc["from_val"].sum())
    report["manifest_unique_pair_id"] = int(manifest["pair_id"].nunique())
    report["binding_flag_counts"] = manifest["binding_flag"].value_counts().to_dict()
    report["chunk_counts"] = (
        master.assign(chunk=[chunk_name_for_index(i) for i in range(len(master))])["chunk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # collision checks on planned new boltz dirs
    dup_boltz = reloc.groupby("new_boltz_dir").size()
    report["duplicate_new_boltz_dir"] = int((dup_boltz > 1).sum())

    legacy_dup = reloc.groupby("legacy_pair_id").filter(lambda g: len(g) > 1)
    report["legacy_pair_id_rows_gt1"] = len(legacy_dup)
    if len(legacy_dup):
        report["legacy_pair_id_collision_resolved"] = bool(
            (legacy_dup.groupby("legacy_pair_id")["immrep_pair_id"].nunique() > 1).all()
        )

    miss_seq = reloc[reloc["relocate_status"] == "sequence_not_in_master"]
    report["sequence_not_in_master"] = len(miss_seq)

    zero_lens = manifest[(manifest["pep_len"] == 0) | (manifest["tcra_len"] == 0) | (manifest["hla_len"] == 0)]
    report["manifest_zero_length_rows"] = len(zero_lens)

    return report


def main() -> None:
    print("Loading master CSV...", flush=True)
    master = pd.read_csv(IMMREP_MASTER)
    assert len(master) == master["pair_id"].nunique() == 10_000

    print("Attaching TCRa/TCRb from raw IMMREP...", flush=True)
    master = attach_tcra_tcrb_from_raw(master)
    master = master.sort_values("pair_id").reset_index(drop=True)

    print("Indexing Boltz output dirs (test/val)...", flush=True)
    boltz_index = build_boltz_index()
    print(f"  indexed {len(boltz_index)} boltz result dirs", flush=True)

    print("Building relocation map (4k)...", flush=True)
    reloc = build_reloc_map(master, boltz_index)
    RELOC_MAP_OUT.parent.mkdir(parents=True, exist_ok=True)
    reloc.to_csv(RELOC_MAP_OUT, index=False)
    print(f"Wrote {RELOC_MAP_OUT} ({len(reloc)} rows)", flush=True)

    print("Building immrep_test_manifest.csv (10k)...", flush=True)
    manifest = build_manifest(master)
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"Wrote {MANIFEST_OUT} ({len(manifest)} rows)", flush=True)

    report = validate(reloc, manifest, master)
    REPORT_OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {REPORT_OUT}", flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
