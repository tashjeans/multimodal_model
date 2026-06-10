#!/usr/bin/env python3
"""
Build Boltz YAMLs + jackhmmer MSAs for pending val/test tulip decoys and IMMREP test pairs.

Mirrors scripts/preprocess/merge_positives_negatives.ipynb (MSA + YAML logic) without
modifying that notebook. Typical workload (~10k jackhmmer runs):

  - val tulip decoys:  1,944
  - test tulip decoys: 2,182
  - immrep_test:       ~6,000 pairs still missing tcra/tcrb/mhc .filt.a3m (4k already done)

Outputs:
  - YAMLs: data/{val,test}/_archive_root_yamls/{pair_id}.yaml
           data/immrep_test/_archive_root_yamls/immrep_pair_*.yaml (updated when MSAs built)
  - MSAs:  data/raw/MSA/jackhmmer_msas/{val,test,immrep_test}/{pair_id}/

Usage:
  python scripts/preprocess/build_pending_boltz_yamls_msas.py --dry-run
  python scripts/preprocess/build_pending_boltz_yamls_msas.py --targets val-decoys test-decoys immrep
  python scripts/preprocess/build_pending_boltz_yamls_msas.py --link-chunks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BASE_DIR = Path("/home/natasha/multimodal_model")
IMMREP_RAW = BASE_DIR / "data/raw/immrep2025_test_set.csv"
IMMREP_MASTER = BASE_DIR / "data/immrep_test/immrep_test_set_pair_id.csv"

DB_COMBINED = BASE_DIR / "data/raw/MSA/big_combo_subset_tcrs_50000_w_mhc_seqs.fasta"
DBS = {
    "tcra": BASE_DIR / "data/raw/MSA/db_split/alpha.fasta",
    "tcrb": BASE_DIR / "data/raw/MSA/db_split/beta.fasta",
    "mhc": BASE_DIR / "data/raw/MSA/db_split/mhc.fasta",
}

TARGETS = {
    "val-decoys": {
        "csv": BASE_DIR / "data/val/val_df_clean_pos_tulip_decoys.csv",
        "yaml_arch": BASE_DIR / "data/val/_archive_root_yamls",
        "msa_split": "val",
        "chunk_root": BASE_DIR / "data/val/_chunks",
    },
    "test-decoys": {
        "csv": BASE_DIR / "data/test/test_df_clean_pos_tulip_decoys.csv",
        "yaml_arch": BASE_DIR / "data/test/_archive_root_yamls",
        "msa_split": "test",
        "chunk_root": BASE_DIR / "data/test/_chunks",
    },
    "immrep": {
        "csv": IMMREP_MASTER,
        "yaml_arch": BASE_DIR / "data/immrep_test/_archive_root_yamls",
        "msa_split": "immrep_test",
        "chunk_root": BASE_DIR / "data/immrep_test/_chunks",
    },
}

REPORT_PATH = BASE_DIR / "data/build_pending_boltz_yamls_msas_report.json"
MISSING_TOKENS = {"", "<unk>", "<UNK>", "UNK", "UNKNOWN", "NA", "N/A", "NONE", "NULL", "NAN"}


# ---------------------------------------------------------------------------
# MSA + YAML core (from merge_positives_negatives.ipynb)
# ---------------------------------------------------------------------------
@dataclass
class MSAConfig:
    out_root: Path
    db_combined: Path
    dbs: Dict[str, Path]
    verbose: bool = False
    keep_intermediates: bool = False
    jack_iters: int = 1
    evalue: float = 1e-10
    cpu_threads: int = os.cpu_count() or 4
    max_seqs: int = 64
    id_thr: int = 100
    cov_thr_tcr: int = 50
    cov_thr_mhc: int = 30


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def run(cmd: List[str]) -> tuple[int, str, str]:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    return p.returncode, out, err


def pick_db_for(cfg: MSAConfig, stem: str) -> Path:
    return cfg.dbs.get(stem, cfg.db_combined)


def clean_seq(s) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"[^A-Za-z]", "", s).upper()


def has_seq(s: str, min_len: int = 1) -> bool:
    return isinstance(s, str) and len(s) >= min_len


def normalise_seq_for_yaml(s) -> str:
    s0 = "" if not isinstance(s, str) else str(s).strip()
    if s0 in {"<unk>", "<UNK>"}:
        return ""
    s = clean_seq(s0)
    return "" if s in MISSING_TOKENS else s


def sto_to_a3m(sto_path: Path, a3m_path: Path) -> bool:
    if have("esl-reformat"):
        code, out, _err = run(["esl-reformat", "a3m", str(sto_path)])
        if code == 0 and out:
            a3m_path.write_text(out)
            return True
    if have("reformat.pl"):
        code, _out, _err = run(["reformat.pl", "sto", "a3m", str(sto_path), str(a3m_path)])
        return code == 0 and a3m_path.exists()
    return False


def count_a3m(p: Path) -> int:
    if not p.exists():
        return -1
    return sum(1 for ln in p.open() if ln.startswith(">"))


def hhfilter_cap(cfg: MSAConfig, in_a3m: Path, out_a3m: Path, cov_thr: int) -> bool:
    if not have("hhfilter"):
        if cfg.verbose:
            print("WARN: hhfilter not found; copying input → output")
        shutil.copyfile(in_a3m, out_a3m)
        return True

    code, out, err = run(["hhfilter", "-h"])
    use_maxseq = ("-maxseq" in (out or "")) or ("-maxseq" in (err or ""))
    cmd = [
        "hhfilter",
        "-i",
        str(in_a3m),
        "-o",
        str(out_a3m),
        "-id",
        str(cfg.id_thr),
        "-cov",
        str(cov_thr),
    ]
    cmd += (["-maxseq", str(cfg.max_seqs)] if use_maxseq else ["-n", str(cfg.max_seqs)])
    if cfg.verbose:
        print("[CMD]", " ".join(map(str, cmd)))
    code, _out, err = run(cmd)
    if cfg.verbose and err:
        print("[HHFILTER][stderr]\n", err.strip()[:800])
    return code == 0 and out_a3m.exists()


def build_msa_for_chain(cfg: MSAConfig, seq: str, out_dir: Path, stem: str) -> Optional[Path]:
    seq = normalise_seq_for_yaml(seq)
    if not has_seq(seq, min_len=1):
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    qfa = out_dir / f"{stem}.fa"
    qfa.write_text(f">{stem}\n{seq}\n")

    sto = out_dir / f"{stem}.sto"
    raw_a3m = out_dir / f"{stem}.a3m"
    filt_a3m = out_dir / f"{stem}.filt.a3m"
    tbl = out_dir / f"{stem}.tbl"
    db_fasta = pick_db_for(cfg, stem)

    cmd = [
        "jackhmmer",
        "-N",
        str(cfg.jack_iters),
        "-A",
        str(sto),
        "--tblout",
        str(tbl),
        "-E",
        str(cfg.evalue),
        "--incE",
        str(cfg.evalue),
        "--incdomE",
        str(cfg.evalue),
        "--cpu",
        str(cfg.cpu_threads),
        str(qfa),
        str(db_fasta),
    ]
    if cfg.verbose:
        print(f"\n=== {stem} ===")
        print("[CMD]", " ".join(map(str, cmd)))

    code, _out, err = run(cmd)
    if code != 0 or (not sto.exists()) or sto.stat().st_size < 200:
        if cfg.verbose:
            print("WARN: bad/empty .sto → fallback to single-seq A3M")
            if err:
                print("[JACK][stderr]\n", err.strip()[:800])
        raw_a3m.write_text(f">{stem}\n{seq}\n")
        raw = raw_a3m
    else:
        ok = sto_to_a3m(sto, raw_a3m)
        if not ok:
            if cfg.verbose:
                print("WARN: sto->a3m failed; using single-seq fallback")
            raw_a3m.write_text(f">{stem}\n{seq}\n")
        raw = raw_a3m

    cov_thr = cfg.cov_thr_tcr if stem in ("tcra", "tcrb") else cfg.cov_thr_mhc
    ok = hhfilter_cap(cfg, raw, filt_a3m, cov_thr=cov_thr)
    if cfg.verbose:
        print("[CHK] filt a3m:", filt_a3m, "nseq=", count_a3m(filt_a3m))

    if not cfg.keep_intermediates:
        for p in (sto, tbl, raw_a3m, qfa):
            try:
                p.unlink()
            except OSError:
                pass

    return filt_a3m if ok else None


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


def get_pair_id(row: dict, i: int, pad: int = 3) -> str:
    pid = row.get("pair_id", None)
    if isinstance(pid, str) and pid.strip():
        return pid.strip()
    return f"pair_{i:0{pad}d}"


def _yaml_done(yml_path: Path) -> bool:
    return yml_path.exists() and yml_path.stat().st_size > 50


def _msas_done(pair_msa_dir: Path, stems: tuple[str, ...] = ("tcra", "tcrb", "mhc")) -> bool:
    for st in stems:
        if not (pair_msa_dir / f"{st}.filt.a3m").exists():
            return False
    return True


def _process_one_row(args: tuple) -> tuple[str, bool, str]:
    i, row_dict, split_name, base_dir_str, yaml_dir_str, msa_root_str, cfg, pad = args
    base_dir = Path(base_dir_str)
    yaml_dir = Path(yaml_dir_str)
    msa_root = Path(msa_root_str)

    pair_id = get_pair_id(row_dict, i=i, pad=pad)
    yml_path = yaml_dir / f"{pair_id}.yaml"
    pair_msa_dir = msa_root / split_name / pair_id

    pep = normalise_seq_for_yaml(row_dict.get("Peptide", ""))
    mhc = normalise_seq_for_yaml(row_dict.get("HLA_sequence", ""))
    tcra = normalise_seq_for_yaml(row_dict.get("TCRa", ""))
    tcrb = normalise_seq_for_yaml(row_dict.get("TCRb", ""))

    tcra_a3m = build_msa_for_chain(cfg, tcra, pair_msa_dir, "tcra")
    tcrb_a3m = build_msa_for_chain(cfg, tcrb, pair_msa_dir, "tcrb")
    mhc_a3m = build_msa_for_chain(cfg, mhc, pair_msa_dir, "mhc")

    yml_text = make_yaml(base_dir, tcra, tcrb, pep, mhc, tcra_a3m, tcrb_a3m, mhc_a3m)
    yml_path.write_text(yml_text)
    return pair_id, True, ""


def process_split_parallel_resume(
    df: pd.DataFrame,
    split_name: str,
    base_dir: Path,
    yaml_dir: Path,
    msa_root: Path,
    cfg: MSAConfig,
    *,
    pad: int = 3,
    resume: bool = True,
    require_msas_for_skip: bool = True,
    max_workers: Optional[int] = None,
) -> dict:
    yaml_dir.mkdir(parents=True, exist_ok=True)

    if max_workers is None:
        ncpu = os.cpu_count() or 8
        max_workers = max(2, min(6, ncpu // max(1, cfg.cpu_threads)))

    df2 = df.reset_index(drop=True)
    todo = []
    skipped = 0

    for i, row in df2.iterrows():
        pair_id = get_pair_id(row.to_dict(), i=i, pad=pad)
        yml_path = yaml_dir / f"{pair_id}.yaml"
        pair_msa_dir = msa_root / split_name / pair_id

        if resume:
            if require_msas_for_skip and _yaml_done(yml_path) and _msas_done(pair_msa_dir):
                skipped += 1
                continue
            if not require_msas_for_skip and _yaml_done(yml_path):
                skipped += 1
                continue

        todo.append((i, row.to_dict(), split_name, str(base_dir), str(yaml_dir), str(msa_root), cfg, pad))

    print(
        f"[{split_name}] total={len(df2)} todo={len(todo)} skipped={skipped} workers={max_workers}",
        flush=True,
    )

    n_ok, n_fail = 0, 0
    if todo:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_process_one_row, t) for t in todo]
            for fut in as_completed(futures):
                try:
                    _pair_id, wrote, _err = fut.result()
                    n_ok += int(wrote)
                except Exception as e:
                    n_fail += 1
                    print("[ERR]", repr(e), flush=True)

    print(f"[{split_name}] completed={n_ok} failed={n_fail}", flush=True)
    return {"split": split_name, "total": len(df2), "todo": len(todo), "skipped": skipped, "ok": n_ok, "fail": n_fail}


# ---------------------------------------------------------------------------
# Input preparation
# ---------------------------------------------------------------------------
def split_tcr_from_lengths(tcr_full: str, tcra_len: int, tcrb_len: int) -> tuple[str, str]:
    tf = normalise_seq_for_yaml(tcr_full)
    la, lb = int(tcra_len), int(tcrb_len)
    if la > 0 and lb > 0 and len(tf) >= la + lb:
        return tf[:la], tf[la : la + lb]
    if la > 0 and lb == 0:
        return tf, ""
    if la == 0 and lb > 0:
        return "", tf
    return "", ""


def build_donor_meta_by_split() -> dict[tuple[str, str], dict]:
    """
    Per-split donor metadata. pair_id is NOT globally unique (pair_000 in val != pair_000 in test).
    """
    meta: dict[tuple[str, str], dict] = {}

    def ingest(split: str, df: pd.DataFrame, manifest: pd.DataFrame) -> None:
        merged = df.merge(manifest[["pair_id", "tcra_len", "tcrb_len"]], on="pair_id", how="inner")
        for _, r in merged.iterrows():
            pid = str(r["pair_id"])
            meta[(split, pid)] = {
                "tcr_full": normalise_seq_for_yaml(r["TCR_full"]),
                "tcra_len": int(r["tcra_len"]),
                "tcrb_len": int(r["tcrb_len"]),
            }

    for split in ("train", "val", "test"):
        clean = pd.read_csv(BASE_DIR / f"data/{split}/{split}_df_clean.csv")
        manifest = pd.read_csv(BASE_DIR / f"manifests/{split}_manifest.csv")
        ingest(split, clean, manifest)

        pos_neg_path = BASE_DIR / f"data/{split}/{split}_df_clean_pos_neg.csv"
        if pos_neg_path.exists():
            pn = pd.read_csv(pos_neg_path)
            pos = pn[pn["binding_flag"] == 1]
            ingest(split, pos, manifest)

    return meta


def resolve_tcra_tcrb_for_decoy(row: pd.Series, target_split: str, meta: dict[tuple[str, str], dict]) -> tuple[str, str]:
    """
    Tulip decoys: TCR_full is the donor TCR sequence on the decoy row; chain lengths come from
    the donor_pair_id entry in *this* split (val/test), not from a global pair_id map.
    """
    decoy_tcr = normalise_seq_for_yaml(row.get("TCR_full", ""))
    donor = str(row.get("donor_pair_id", "") or "").strip()
    if not decoy_tcr:
        return "", ""

    def try_split(info: dict) -> tuple[str, str]:
        tcra, tcrb = split_tcr_from_lengths(decoy_tcr, info["tcra_len"], info["tcrb_len"])
        if tcra or tcrb:
            return tcra, tcrb
        # Manifest can have tcra_len=tcrb_len=0 while TCR_full is non-empty; use beta-only fallback.
        if decoy_tcr:
            return "", decoy_tcr
        return "", ""

    key = (target_split, donor)
    if key in meta and meta[key]["tcr_full"] == decoy_tcr:
        tcra, tcrb = try_split(meta[key])
        if tcra or tcrb:
            return tcra, tcrb

    for split in (target_split, "val", "test", "train"):
        k = (split, donor)
        if k in meta and meta[k]["tcr_full"] == decoy_tcr:
            tcra, tcrb = try_split(meta[k])
            if tcra or tcrb:
                return tcra, tcrb

    for (_split, _pid), info in meta.items():
        if info["tcr_full"] == decoy_tcr:
            tcra, tcrb = try_split(info)
            if tcra or tcrb:
                return tcra, tcrb

    return "", ""


def load_tulip_decoys(split: str, meta: dict[tuple[str, str], dict]) -> pd.DataFrame:
    path = BASE_DIR / f"data/{split}/{split}_df_clean_pos_tulip_decoys.csv"
    df = pd.read_csv(path)
    decoys = df[df["binding_flag"] == 0].copy()
    decoys["pair_id"] = decoys["pair_id"].astype(str)

    tcra_list, tcrb_list = [], []
    missing = []
    for _, r in decoys.iterrows():
        tcra, tcrb = resolve_tcra_tcrb_for_decoy(r, split, meta)
        tcra_list.append(tcra)
        tcrb_list.append(tcrb)
        if not tcra and not tcrb:
            missing.append(str(r["pair_id"]))

    decoys["TCRa"] = tcra_list
    decoys["TCRb"] = tcrb_list

    if missing:
        raise ValueError(
            f"{split}: could not resolve TCRa/TCRb for {len(missing)} tulip decoys "
            f"(e.g. {missing[:5]}). Check manifests / donor_pair_id."
        )

    return decoys[["pair_id", "Peptide", "HLA_sequence", "TCRa", "TCRb", "binding_flag"]]


def attach_immrep_tcra_tcrb(master: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_csv(IMMREP_RAW)
    raw = raw.copy()
    raw["TCR_full_key"] = raw["tcra_trimmed"].map(normalise_seq_for_yaml) + raw["tcrb_trimmed"].map(
        normalise_seq_for_yaml
    )
    raw["seq_key"] = (
        raw["peptide"].map(normalise_seq_for_yaml)
        + "|"
        + raw["hla_sequence"].map(normalise_seq_for_yaml)
        + "|"
        + raw["TCR_full_key"]
    )
    master = master.copy()
    master["TCR_full_key"] = master["TCR_full"].map(normalise_seq_for_yaml)
    master["seq_key"] = (
        master["Peptide"].map(normalise_seq_for_yaml)
        + "|"
        + master["HLA_sequence"].map(normalise_seq_for_yaml)
        + "|"
        + master["TCR_full_key"]
    )
    raw_lut = raw.drop_duplicates("seq_key").set_index("seq_key")
    master["TCRa"] = master["seq_key"].map(
        lambda k: raw_lut.loc[k, "tcra_trimmed"] if k in raw_lut.index else ""
    )
    master["TCRb"] = master["seq_key"].map(
        lambda k: raw_lut.loc[k, "tcrb_trimmed"] if k in raw_lut.index else ""
    )
    n_miss = ((master["TCRa"] == "") & (master["TCRb"] == "")).sum()
    if n_miss:
        raise ValueError(f"immrep: {n_miss} rows missing TCRa/TCRb after raw join")
    return master


def filter_immrep_needing_msa(df: pd.DataFrame, msa_root: Path) -> pd.DataFrame:
    split = "immrep_test"
    need = []
    for _, row in df.iterrows():
        pid = str(row["pair_id"])
        d = msa_root / split / pid
        if not _msas_done(d):
            need.append(row)
    return pd.DataFrame(need).reset_index(drop=True)


def count_todo(df: pd.DataFrame, split_name: str, yaml_dir: Path, msa_root: Path) -> int:
    n = 0
    for _, row in df.iterrows():
        pid = str(row["pair_id"])
        yml = yaml_dir / f"{pid}.yaml"
        msa_dir = msa_root / split_name / pid
        if not (_yaml_done(yml) and _msas_done(msa_dir)):
            n += 1
    return n


def ensure_chunk_symlinks(yaml_arch: Path, chunk_root: Path, per_chunk: int = 2000) -> dict:
    chunk_root.mkdir(parents=True, exist_ok=True)
    yamls = sorted(list(yaml_arch.glob("*.yaml")) + list(yaml_arch.glob("*.yml")))
    if not yamls:
        return {"archive": 0, "to_link": 0}

    chunks = sorted(p for p in chunk_root.iterdir() if p.is_dir() and p.name.startswith("chunk_"))
    if not chunks:
        chunks = [chunk_root / "chunk_000"]
        chunks[0].mkdir(parents=True, exist_ok=True)

    def count_yaml(d: Path) -> int:
        return len(list(d.glob("*.y*")))

    existing: set[str] = set()
    for c in chunks:
        for p in list(c.glob("*.yaml")) + list(c.glob("*.yml")):
            existing.add(p.stem)

    to_link = [p for p in yamls if p.stem not in existing]
    ci = 0
    linked = 0
    for y in to_link:
        while True:
            if ci >= len(chunks):
                new = chunk_root / f"chunk_{ci:03d}"
                new.mkdir(parents=True, exist_ok=True)
                chunks.append(new)
            if count_yaml(chunks[ci]) < per_chunk:
                break
            ci += 1
        link = chunks[ci] / y.name
        if not link.exists():
            os.symlink(y.resolve(), link)
            linked += 1

    return {
        "archive": len(yamls),
        "already_linked": len(existing),
        "to_link": len(to_link),
        "new_symlinks": linked,
        "chunks": len(chunks),
        "last_chunk": chunks[-1].name,
        "last_chunk_count": count_yaml(chunks[-1]),
    }


def default_cfg(verbose: bool, jack_cpu: Optional[int]) -> MSAConfig:
    cpu = jack_cpu if jack_cpu is not None else (os.cpu_count() or 4)
    return MSAConfig(
        out_root=BASE_DIR / "outputs",
        db_combined=DB_COMBINED,
        dbs=DBS,
        verbose=verbose,
        keep_intermediates=False,
        jack_iters=1,
        evalue=1e-10,
        cpu_threads=cpu,
        max_seqs=64,
        id_thr=100,
        cov_thr_tcr=50,
        cov_thr_mhc=30,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--targets",
        nargs="+",
        default=["val-decoys", "test-decoys", "immrep"],
        choices=list(TARGETS.keys()),
    )
    ap.add_argument("--dry-run", action="store_true", help="Print counts only; do not run jackhmmer")
    ap.add_argument("--no-resume", action="store_true", help="Reprocess even if YAML+MSAs already exist")
    ap.add_argument("--max-workers", type=int, default=None)
    ap.add_argument("--jack-cpu", type=int, default=None, help="Threads per jackhmmer process")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--link-chunks", action="store_true", help="After processing, symlink new YAMLs into _chunks/")
    ap.add_argument(
        "--link-chunks-only",
        action="store_true",
        help="Only create _chunks symlinks (no jackhmmer); uses same --targets",
    )
    args = ap.parse_args()

    msa_root = BASE_DIR / "data/raw/MSA/jackhmmer_msas"
    cfg = default_cfg(args.verbose, args.jack_cpu)
    resume = not args.no_resume

    print("Building per-split donor metadata for tulip decoys...", flush=True)
    donor_meta = build_donor_meta_by_split()

    plan: list[tuple[str, pd.DataFrame, dict]] = []

    if "val-decoys" in args.targets:
        df = load_tulip_decoys("val", donor_meta)
        plan.append(("val-decoys", df, TARGETS["val-decoys"]))

    if "test-decoys" in args.targets:
        df = load_tulip_decoys("test", donor_meta)
        plan.append(("test-decoys", df, TARGETS["test-decoys"]))

    if "immrep" in args.targets:
        master = pd.read_csv(IMMREP_MASTER)
        master = attach_immrep_tcra_tcrb(master)
        master = filter_immrep_needing_msa(master, msa_root)
        plan.append(("immrep", master, TARGETS["immrep"]))

    summary = []
    for label, df, meta in plan:
        split_name = meta["msa_split"]
        todo_n = count_todo(df, split_name, meta["yaml_arch"], msa_root)
        entry = {
            "target": label,
            "rows_in_df": len(df),
            "would_process": todo_n,
            "yaml_arch": str(meta["yaml_arch"]),
            "msa_dir": str(msa_root / split_name),
        }
        summary.append(entry)
        print(json.dumps(entry), flush=True)

    if args.dry_run:
        REPORT_PATH.write_text(json.dumps({"dry_run": True, "plan": summary}, indent=2) + "\n")
        print(f"Dry run complete. Report: {REPORT_PATH}", flush=True)
        return

    if args.link_chunks_only:
        chunk_reports = {}
        for label, _df, meta in plan:
            chunk_reports[label] = ensure_chunk_symlinks(meta["yaml_arch"], meta["chunk_root"])
            print(json.dumps({"target": label, **chunk_reports[label]}), flush=True)
        report = {"plan": summary, "chunk_symlinks": chunk_reports, "link_chunks_only": True}
        REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Chunk symlinks done. Report: {REPORT_PATH}", flush=True)
        return

    results = []
    for label, df, meta in plan:
        if len(df) == 0:
            print(f"[{label}] nothing to do", flush=True)
            results.append({"target": label, "skipped_all": True})
            continue

        rep = process_split_parallel_resume(
            df=df,
            split_name=meta["msa_split"],
            base_dir=BASE_DIR,
            yaml_dir=meta["yaml_arch"],
            msa_root=msa_root,
            cfg=cfg,
            resume=resume,
            require_msas_for_skip=True,
            max_workers=args.max_workers,
        )
        rep["target"] = label
        results.append(rep)

    chunk_reports = {}
    if args.link_chunks:
        for label, _df, meta in plan:
            chunk_reports[label] = ensure_chunk_symlinks(meta["yaml_arch"], meta["chunk_root"])

    report = {"plan": summary, "results": results, "chunk_symlinks": chunk_reports}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Done. Report: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
