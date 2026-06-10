#!/usr/bin/env python3
"""
Build a PDB/RCSB shortlist for Boltz vs AF3 TCR:pMHC structural comparison.

v3 fix:
- Does NOT use RCSB sequence-similarity search for short antigen peptides.
- Does NOT use "text" search against entity_poly.pdbx_seq_one_letter_code_can,
  because RCSB does not enable the text service on that attribute.
- Instead uses RCSB full_text search for the peptide string, then validates each
  returned PDB entry through the RCSB Data API by checking polymer entity
  sequences/descriptions.

Default input:
    /home/natasha/multimodal_model/data/train/train_for_pdb_search.csv

Typical run:
    python build_rcsb_tcrpmhc_shortlist_v3.py \
      --input /home/natasha/multimodal_model/data/train/train_for_pdb_search.csv \
      --outdir /home/natasha/multimodal_model/data/train/pdb_search \
      --prefix train_pdb \
      --max-peptides 300 \
      --shortlist-size 30
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_POLYMER_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"

AA_RE = re.compile(r"[^A-Z]")


def clean_seq(x: Any) -> str:
    if pd.isna(x):
        return ""
    return AA_RE.sub("", str(x).strip().upper())


def request_json(method: str, url: str, *, json_payload: Optional[dict] = None,
                 timeout: int = 30, retries: int = 3, sleep: float = 0.5) -> Optional[dict]:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if method.upper() == "POST":
                r = requests.post(url, json=json_payload, timeout=timeout)
            else:
                r = requests.get(url, timeout=timeout)

            if r.status_code == 404:
                return None

            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(sleep * attempt)
                continue

            if r.status_code >= 400:
                return {
                    "_error": True,
                    "status_code": r.status_code,
                    "text": r.text[:1000],
                    "url": url,
                }

            try:
                return r.json()
            except json.JSONDecodeError:
                return {
                    "_error": True,
                    "status_code": r.status_code,
                    "text": r.text[:1000],
                    "url": url,
                }

        except Exception as e:
            last_err = repr(e)
            time.sleep(sleep * attempt)

    return {"_error": True, "status_code": None, "text": str(last_err), "url": url}


def rcsb_full_text_query(value: str, return_type: str = "entry") -> Tuple[List[str], Optional[str]]:
    """
    Full-text query across RCSB-indexed fields.

    This is intentionally broad. We validate exact peptide presence later by
    fetching polymer entity sequences from each returned entry.
    """
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": value
            },
        },
        "return_type": return_type,
        "request_options": {
            "return_all_hits": True,
            "scoring_strategy": "combined",
        },
    }

    data = request_json("POST", RCSB_SEARCH_URL, json_payload=payload)
    if data is None:
        return [], None
    if data.get("_error"):
        return [], f"{data.get('status_code')}: {data.get('text')}"

    hits = data.get("result_set", []) or []
    ids = [h.get("identifier") for h in hits if h.get("identifier")]
    return sorted(set(ids)), None


def get_entry(pdb_id: str) -> Optional[dict]:
    data = request_json("GET", RCSB_ENTRY_URL.format(pdb_id=pdb_id.lower()))
    if data and data.get("_error"):
        return None
    return data


def get_polymer_entity(pdb_id: str, entity_id: str) -> Optional[dict]:
    data = request_json("GET", RCSB_POLYMER_ENTITY_URL.format(pdb_id=pdb_id.lower(), entity_id=entity_id))
    if data and data.get("_error"):
        return None
    return data


def extract_entity_ids(entry: dict) -> List[str]:
    ids = []
    ids += [str(x) for x in (entry.get("rcsb_entry_container_identifiers", {}) or {}).get("polymer_entity_ids", []) or []]
    ids += [str(x) for x in entry.get("polymer_entity_ids", []) or []]
    return sorted(set(ids), key=lambda x: int(x) if x.isdigit() else x)


def entity_sequence(entity: dict) -> str:
    ep = entity.get("entity_poly", {}) or {}
    return clean_seq(ep.get("pdbx_seq_one_letter_code_can") or ep.get("pdbx_seq_one_letter_code") or "")


def entity_description(entity: dict) -> str:
    parts = []
    rpe = entity.get("rcsb_polymer_entity", {}) or {}
    ep = entity.get("entity_poly", {}) or {}
    if rpe.get("pdbx_description"):
        parts.append(str(rpe.get("pdbx_description")))
    if ep.get("type"):
        parts.append(str(ep.get("type")))
    return " ".join(parts).strip()


def classify_entities(pdb_id: str, peptide: str, sleep: float = 0.1) -> dict:
    entry = get_entry(pdb_id)
    if not entry:
        return {
            "pdb_id": pdb_id,
            "entry_fetch_ok": False,
            "has_exact_peptide_entity": False,
            "has_hla_or_mhc": False,
            "has_tcr_alpha": False,
            "has_tcr_beta": False,
            "is_tcr_pmhc_candidate": False,
            "entity_summary": "",
            "resolution": None,
            "experimental_method": "",
            "title": "",
        }

    title = (entry.get("struct", {}) or {}).get("title", "") or ""

    methods = []
    for e in entry.get("exptl", []) or []:
        if e.get("method"):
            methods.append(e["method"])
    method = "; ".join(sorted(set(methods)))

    res = None
    res_list = (entry.get("rcsb_entry_info", {}) or {}).get("resolution_combined") or []
    if res_list:
        try:
            res = float(res_list[0])
        except Exception:
            res = None

    has_exact_peptide = False
    has_hla = False
    has_tcra = False
    has_tcrb = False
    summary_bits = []

    for eid in extract_entity_ids(entry):
        ent = get_polymer_entity(pdb_id, eid)
        time.sleep(sleep)
        if not ent:
            continue

        seq = entity_sequence(ent)
        desc = entity_description(ent)
        desc_u = desc.upper()
        length = len(seq)

        is_peptide = seq == peptide

        is_hla = (
            ("HLA" in desc_u)
            or ("MHC" in desc_u)
            or ("H-2" in desc_u)
            or ("MAJOR HISTOCOMPATIBILITY" in desc_u)
        ) and length > 50

        is_tcra = (
            ("T-CELL RECEPTOR ALPHA" in desc_u)
            or ("T CELL RECEPTOR ALPHA" in desc_u)
            or ("TCR ALPHA" in desc_u)
            or ("T-CELL RECEPTOR A" in desc_u)
        ) and length > 50

        is_tcrb = (
            ("T-CELL RECEPTOR BETA" in desc_u)
            or ("T CELL RECEPTOR BETA" in desc_u)
            or ("TCR BETA" in desc_u)
            or ("T-CELL RECEPTOR B" in desc_u)
        ) and length > 50

        has_exact_peptide = has_exact_peptide or is_peptide
        has_hla = has_hla or is_hla
        has_tcra = has_tcra or is_tcra
        has_tcrb = has_tcrb or is_tcrb

        labels = []
        if is_peptide:
            labels.append("PEPTIDE")
        if is_hla:
            labels.append("HLA/MHC")
        if is_tcra:
            labels.append("TCRa")
        if is_tcrb:
            labels.append("TCRb")
        label = ",".join(labels) if labels else "other"

        seq_preview = seq[:25] + ("..." if len(seq) > 25 else "")
        summary_bits.append(f"{eid}:{label}:len{length}:{desc[:80]}:seq={seq_preview}")

    return {
        "pdb_id": pdb_id,
        "entry_fetch_ok": True,
        "has_exact_peptide_entity": has_exact_peptide,
        "has_hla_or_mhc": has_hla,
        "has_tcr_alpha": has_tcra,
        "has_tcr_beta": has_tcrb,
        "is_tcr_pmhc_candidate": bool(has_exact_peptide and has_hla and has_tcra and has_tcrb),
        "entity_summary": " | ".join(summary_bits),
        "resolution": res,
        "experimental_method": method,
        "title": title,
    }


def find_col(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Missing required column. Tried {candidates}. Available: {list(df.columns)}")
    return None


def representative_rows(df: pd.DataFrame, peptide_col: str, peptide: str, max_rows: int = 3) -> pd.DataFrame:
    sub = df[df[peptide_col] == peptide].copy()
    return sub.head(max_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/home/natasha/multimodal_model/data/train/train_for_pdb_search.csv")
    ap.add_argument("--outdir", default="/home/natasha/multimodal_model/data/train/pdb_search")
    ap.add_argument("--prefix", default="train_pdb")
    ap.add_argument("--max-peptides", type=int, default=300)
    ap.add_argument("--shortlist-size", type=int, default=30)
    ap.add_argument("--sleep", type=float, default=0.25)
    ap.add_argument("--include-negatives", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} rows from {input_path}")
    print(f"Columns: {list(df.columns)}")

    pair_col = find_col(df, ["pair_id", "pairid", "id"], required=False)
    pep_col = find_col(df, ["Peptide", "peptide", "pep"])
    hla_col = find_col(df, ["HLA_sequence", "hla_sequence", "mhc_sequence", "MHC_sequence", "HLA", "hla"])
    tcra_col = find_col(df, ["TCRa", "tcra", "TRA", "tra", "tcr_alpha", "alpha"])
    tcrb_col = find_col(df, ["TCRb", "tcrb", "TRB", "trb", "tcr_beta", "beta"])
    label_col = find_col(df, ["binding_flag", "label", "binder"], required=False)

    for col in [pep_col, hla_col, tcra_col, tcrb_col]:
        df[col] = df[col].map(clean_seq)

    if pair_col:
        before = len(df)
        df = df.drop_duplicates(subset=[pair_col], keep="first").copy()
        print(f"Deduplicated by {pair_col}: {before:,} -> {len(df):,}")

    before = len(df)
    df = df[(df[pep_col] != "") & (df[hla_col] != "") & (df[tcra_col] != "") & (df[tcrb_col] != "")].copy()
    print(f"Complete rows: {before:,} -> {len(df):,}")

    if label_col and not args.include_negatives:
        before = len(df)
        df = df[df[label_col].astype(str).isin(["1", "1.0", "True", "true", "TRUE"])].copy()
        print(f"Positive rows only: {before:,} -> {len(df):,}")

    peptide_counts = df[pep_col].value_counts()
    peptides = list(peptide_counts.head(args.max_peptides).index)
    print(f"Searching top {len(peptides)} peptides")

    query_log = []
    pdb_rows = []
    row_matches = []

    # Cache by (pdb_id, peptide), because exact peptide validation is peptide-specific.
    entity_cache: Dict[tuple, dict] = {}

    for i, pep in enumerate(peptides, 1):
        n_train = int(peptide_counts[pep])
        print(f"[{i}/{len(peptides)}] Searching peptide {pep} ({n_train} train rows)")

        entry_ids, err = rcsb_full_text_query(pep)
        if err:
            print(f"  QUERY ERROR for {pep}: {err}")
            query_log.append({
                "peptide": pep,
                "train_rows_for_peptide": n_train,
                "n_entry_ids_returned": 0,
                "n_valid_exact_peptide_entries": 0,
                "n_valid_tcr_pmhc_entries": 0,
                "error": err,
            })
            time.sleep(args.sleep)
            continue

        print(f"  Full-text returned {len(entry_ids)} candidate RCSB entries")

        valid_exact = 0
        valid_tcrpmhc = 0

        for pdb_id in entry_ids:
            key = (pdb_id, pep)
            if key not in entity_cache:
                entity_cache[key] = classify_entities(pdb_id, pep, sleep=max(args.sleep / 2, 0.05))
                time.sleep(args.sleep)

            rec = dict(entity_cache[key])
            rec["peptide"] = pep
            rec["train_rows_for_peptide"] = n_train

            # Keep only entries where the peptide was actually validated as an
            # exact polymer entity. Full-text hits without this are noise.
            if not rec.get("has_exact_peptide_entity", False):
                continue

            valid_exact += 1
            if rec.get("is_tcr_pmhc_candidate", False):
                valid_tcrpmhc += 1

            pdb_rows.append(rec)

            reps = representative_rows(df, pep_col, pep, max_rows=3)
            for _, row in reps.iterrows():
                if rec.get("is_tcr_pmhc_candidate", False):
                    tier = 3
                    rationale = "Tier 3: exact peptide with validated TCR:pMHC PDB structure"
                elif rec.get("has_hla_or_mhc", False):
                    tier = 4
                    rationale = "Tier 4: exact peptide with pMHC-like PDB structure; TCR alpha/beta not both detected"
                else:
                    tier = 5
                    rationale = "Tier 5: exact peptide found, but complex composition unclear"

                row_matches.append({
                    "tier": tier,
                    "rationale": rationale,
                    "pair_id": row.get(pair_col, "") if pair_col else "",
                    "peptide": pep,
                    "HLA_sequence": row[hla_col],
                    "TCRa": row[tcra_col],
                    "TCRb": row[tcrb_col],
                    "pdb_id": pdb_id,
                    "pdb_title": rec.get("title", ""),
                    "experimental_method": rec.get("experimental_method", ""),
                    "resolution": rec.get("resolution", None),
                    "is_tcr_pmhc_candidate": rec.get("is_tcr_pmhc_candidate", False),
                    "has_exact_peptide_entity": rec.get("has_exact_peptide_entity", False),
                    "has_hla_or_mhc": rec.get("has_hla_or_mhc", False),
                    "has_tcr_alpha": rec.get("has_tcr_alpha", False),
                    "has_tcr_beta": rec.get("has_tcr_beta", False),
                    "entity_summary": rec.get("entity_summary", ""),
                    "train_rows_for_peptide": n_train,
                })

        print(f"  Valid exact peptide entries: {valid_exact}; valid TCR:pMHC candidates: {valid_tcrpmhc}")

        query_log.append({
            "peptide": pep,
            "train_rows_for_peptide": n_train,
            "n_entry_ids_returned": len(entry_ids),
            "n_valid_exact_peptide_entries": valid_exact,
            "n_valid_tcr_pmhc_entries": valid_tcrpmhc,
            "error": "",
        })

    qlog = pd.DataFrame(query_log)
    pdb_df = pd.DataFrame(pdb_rows).drop_duplicates(subset=["peptide", "pdb_id"], keep="first") if pdb_rows else pd.DataFrame()
    row_df = pd.DataFrame(row_matches).drop_duplicates(subset=["pair_id", "peptide", "pdb_id"], keep="first") if row_matches else pd.DataFrame()

    qlog_path = outdir / f"{args.prefix}_peptide_query_log.csv"
    pdb_path = outdir / f"{args.prefix}_pdb_entries_by_peptide.csv"
    row_path = outdir / f"{args.prefix}_row_level_matches.csv"
    shortlist_path = outdir / f"{args.prefix}_boltz_af3_shortlist.csv"
    external_path = outdir / f"{args.prefix}_external_pdb_benchmark_candidates.csv"

    qlog.to_csv(qlog_path, index=False)
    pdb_df.to_csv(pdb_path, index=False)
    row_df.to_csv(row_path, index=False)

    if not row_df.empty:
        tmp = row_df.copy()
        tmp["resolution_sort"] = tmp["resolution"].fillna(999.0)
        tmp = tmp.sort_values(
            ["tier", "is_tcr_pmhc_candidate", "resolution_sort", "train_rows_for_peptide"],
            ascending=[True, False, True, False],
        )

        shortlist = tmp.drop_duplicates(subset=["peptide"], keep="first").head(args.shortlist_size)
        if len(shortlist) < args.shortlist_size:
            extra = tmp[~tmp.index.isin(shortlist.index)]
            shortlist = pd.concat([shortlist, extra.head(args.shortlist_size - len(shortlist))], ignore_index=True)

        shortlist.drop(columns=["resolution_sort"], errors="ignore").to_csv(shortlist_path, index=False)
    else:
        pd.DataFrame().to_csv(shortlist_path, index=False)

    if not pdb_df.empty and "is_tcr_pmhc_candidate" in pdb_df.columns:
        ext = pdb_df[pdb_df["is_tcr_pmhc_candidate"] == True].copy()
        if not ext.empty:
            ext["resolution_sort"] = ext["resolution"].fillna(999.0)
            ext = ext.sort_values(["resolution_sort", "train_rows_for_peptide"], ascending=[True, False])
            ext.drop(columns=["resolution_sort"], errors="ignore").to_csv(external_path, index=False)
        else:
            pd.DataFrame().to_csv(external_path, index=False)
    else:
        pd.DataFrame().to_csv(external_path, index=False)

    print("\nDone.")
    print(f"Query log:          {qlog_path}")
    print(f"PDB by peptide:     {pdb_path}")
    print(f"Row-level matches:  {row_path}")
    print(f"Shortlist:          {shortlist_path}")
    print(f"External candidates:{external_path}")

    if not qlog.empty:
        print("\nSummary:")
        print(qlog[["n_entry_ids_returned", "n_valid_exact_peptide_entries", "n_valid_tcr_pmhc_entries"]].describe())

    if not row_df.empty and "tier" in row_df.columns:
        print("\nTier counts:")
        print(row_df["tier"].value_counts().sort_index())


if __name__ == "__main__":
    main()
