#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


def clean(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip()


def require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] missing required columns: {missing}")


def unique_negative_hla(df: pd.DataFrame, split_name: str) -> str:
    neg = df[df["binding_flag"] == 0].copy()
    if neg.empty:
        raise ValueError(f"[{split_name}] no negative rows found")
    hlas = clean(neg["HLA_sequence"]).value_counts()
    if len(hlas) != 1:
        raise ValueError(
            f"[{split_name}] expected exactly one HLA in negatives, found {len(hlas)}. "
            f"Top counts: {hlas.head(10).to_dict()}"
        )
    return hlas.index[0]


def build_yaml_path(pair_id: str, split_name: str, existing_manifest: pd.DataFrame) -> str:
    if "yaml_path" in existing_manifest.columns and len(existing_manifest) > 0:
        sample = str(existing_manifest["yaml_path"].iloc[0])
        if sample.endswith(".yaml"):
            prefix = sample.rsplit("/", 1)[0]
            return f"{prefix}/{pair_id}.yaml"
    return f"data/{split_name}/_archive_root_yamls/{pair_id}.yaml"


def make_immrep_positive_rows(
    immrep_pos_hla: pd.DataFrame,
    target_columns: list[str],
    split_name: str,
) -> pd.DataFrame:
    rows = []
    prefix = f"immrep_positive_{split_name}"
    for i, (_, r) in enumerate(immrep_pos_hla.reset_index(drop=True).iterrows()):
        pair_id = f"{prefix}_{i:06d}"
        tcra = str(r["tcra_trimmed"])
        tcrb = str(r["tcrb_trimmed"])
        tcr_full = tcra + tcrb
        row = {
            "pair_id": pair_id,
            "Peptide": str(r["peptide"]),
            "HLA_sequence": str(r["hla_sequence"]),
            "TCR_full": tcr_full,
            "binding_flag": 1,
            "pair_id_negative": "",
            "TCRa": tcra,
            "TCRb": tcrb,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    # Keep exact dataframe schema of existing files.
    return out.reindex(columns=target_columns)


def make_new_manifest_rows(
    immrep_pos_hla: pd.DataFrame,
    split_name: str,
    manifest_columns: list[str],
    base_manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    prefix = f"immrep_positive_{split_name}"
    for i, (_, r) in enumerate(immrep_pos_hla.reset_index(drop=True).iterrows()):
        pair_id = f"{prefix}_{i:06d}"
        tcra = str(r["tcra_trimmed"])
        tcrb = str(r["tcrb_trimmed"])
        peptide = str(r["peptide"])
        hla = str(r["hla_sequence"])
        row = {
            "pair_id": pair_id,
            "yaml_path": build_yaml_path(pair_id, split_name, base_manifest),
            "pep_len": len(peptide),
            "tcra_len": len(tcra),
            "tcrb_len": len(tcrb),
            "hla_len": len(hla),
            "binding_flag": 1,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.reindex(columns=manifest_columns)


def assert_pair_id_integrity(df: pd.DataFrame, manifest: pd.DataFrame, split_name: str) -> None:
    df_ids = set(clean(df["pair_id"]))
    m_ids = set(clean(manifest["pair_id"]))
    if "" in df_ids:
        raise ValueError(f"[{split_name}] dataframe has missing pair_id values")
    if "" in m_ids:
        raise ValueError(f"[{split_name}] manifest has missing pair_id values")
    if df_ids != m_ids:
        only_df = sorted(df_ids - m_ids)[:10]
        only_m = sorted(m_ids - df_ids)[:10]
        raise ValueError(
            f"[{split_name}] pair_id mismatch between dataframe and manifest. "
            f"only_in_dataframe={only_df}, only_in_manifest={only_m}"
        )
    if clean(df["pair_id"]).duplicated().any():
        raise ValueError(f"[{split_name}] duplicate pair_id values in dataframe")
    if clean(manifest["pair_id"]).duplicated().any():
        raise ValueError(f"[{split_name}] duplicate pair_id values in manifest")


def print_summary(
    val_new: pd.DataFrame,
    test_new: pd.DataFrame,
    val_manifest_new: pd.DataFrame,
    test_manifest_new: pd.DataFrame,
    peptide_col: str,
    tcr_a_col: str,
    tcr_b_col: str,
    tcr_full_col: str,
    hla_col: str,
) -> None:
    print("\n=== Final Summary ===")
    print(
        f"new validation rows={len(val_new)} "
        f"(pos={(val_new['binding_flag']==1).sum()}, neg={(val_new['binding_flag']==0).sum()})"
    )
    print(
        f"new test rows={len(test_new)} "
        f"(pos={(test_new['binding_flag']==1).sum()}, neg={(test_new['binding_flag']==0).sum()})"
    )
    print(f"new validation manifest rows={len(val_manifest_new)}")
    print(f"new test manifest rows={len(test_manifest_new)}")

    val_match = set(clean(val_new["pair_id"])) == set(clean(val_manifest_new["pair_id"]))
    test_match = set(clean(test_new["pair_id"])) == set(clean(test_manifest_new["pair_id"]))
    print(f"validation dataframe-manifest pair_id match: {val_match}")
    print(f"test dataframe-manifest pair_id match: {test_match}")

    val_dup = clean(val_new["pair_id"]).duplicated().any()
    test_dup = clean(test_new["pair_id"]).duplicated().any()
    overlap = len(set(clean(val_new["pair_id"])) & set(clean(test_new["pair_id"])))
    print(f"no duplicate pair_ids within validation: {not val_dup}")
    print(f"no duplicate pair_ids within test: {not test_dup}")
    print(f"no pair_id overlap between validation and test: {overlap == 0} (overlap_count={overlap})")

    print("HLA counts in new validation:")
    print(clean(val_new[hla_col]).value_counts().to_dict())
    print("HLA counts in new test:")
    print(clean(test_new[hla_col]).value_counts().to_dict())

    print("missing/null counts (validation):")
    print(
        {
            "peptide": val_new[peptide_col].isna().sum() + clean(val_new[peptide_col]).eq("").sum(),
            "tcr_a": val_new[tcr_a_col].isna().sum() + clean(val_new[tcr_a_col]).eq("").sum(),
            "tcr_b": val_new[tcr_b_col].isna().sum() + clean(val_new[tcr_b_col]).eq("").sum(),
            "tcr_full": val_new[tcr_full_col].isna().sum() + clean(val_new[tcr_full_col]).eq("").sum(),
            "hla_sequence": val_new[hla_col].isna().sum() + clean(val_new[hla_col]).eq("").sum(),
            "binding_flag": val_new["binding_flag"].isna().sum(),
            "pair_id": val_new["pair_id"].isna().sum() + clean(val_new["pair_id"]).eq("").sum(),
        }
    )
    print("missing/null counts (test):")
    print(
        {
            "peptide": test_new[peptide_col].isna().sum() + clean(test_new[peptide_col]).eq("").sum(),
            "tcr_a": test_new[tcr_a_col].isna().sum() + clean(test_new[tcr_a_col]).eq("").sum(),
            "tcr_b": test_new[tcr_b_col].isna().sum() + clean(test_new[tcr_b_col]).eq("").sum(),
            "tcr_full": test_new[tcr_full_col].isna().sum() + clean(test_new[tcr_full_col]).eq("").sum(),
            "hla_sequence": test_new[hla_col].isna().sum() + clean(test_new[hla_col]).eq("").sum(),
            "binding_flag": test_new["binding_flag"].isna().sum(),
            "pair_id": test_new["pair_id"].isna().sum() + clean(test_new["pair_id"]).eq("").sum(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create IMMREP-based val/test A datasets: existing negatives + 500 IMMREP positives each."
    )
    parser.add_argument("--val-csv", default="data/val/val_df_clean_pos_neg.csv")
    parser.add_argument("--test-csv", default="data/test/test_df_clean_pos_neg.csv")
    parser.add_argument("--val-manifest", default="manifests/val_manifest.csv")
    parser.add_argument("--test-manifest", default="manifests/test_manifest.csv")
    parser.add_argument("--immrep-csv", default="data/raw/immrep2025_test_set.csv")
    parser.add_argument("--out-val-csv", default="data/val/val_df_clean_pos_neg_A.csv")
    parser.add_argument("--out-test-csv", default="data/test/test_df_clean_pos_neg_A.csv")
    parser.add_argument("--out-val-manifest", default="manifests/val_manifest_A.csv")
    parser.add_argument("--out-test-manifest", default="manifests/test_manifest_A.csv")
    args = parser.parse_args()

    val_df = pd.read_csv(args.val_csv)
    test_df = pd.read_csv(args.test_csv)
    val_manifest = pd.read_csv(args.val_manifest)
    test_manifest = pd.read_csv(args.test_manifest)
    immrep = pd.read_csv(args.immrep_csv)

    require_columns(val_df, ["pair_id", "Peptide", "HLA_sequence", "TCR_full", "binding_flag"], "val_df")
    require_columns(test_df, ["pair_id", "Peptide", "HLA_sequence", "TCR_full", "binding_flag"], "test_df")
    require_columns(val_manifest, ["pair_id", "binding_flag"], "val_manifest")
    require_columns(test_manifest, ["pair_id", "binding_flag"], "test_manifest")
    require_columns(immrep, ["peptide", "tcra_trimmed", "tcrb_trimmed", "hla_sequence", "label"], "immrep")

    val_neg = val_df[val_df["binding_flag"] == 0].copy().reset_index(drop=True)
    test_neg = test_df[test_df["binding_flag"] == 0].copy().reset_index(drop=True)
    val_neg_ids = set(clean(val_neg["pair_id"]))
    test_neg_ids = set(clean(test_neg["pair_id"]))

    val_hla = unique_negative_hla(val_df, "validation")
    test_hla = unique_negative_hla(test_df, "test")
    if val_hla == test_hla:
        raise ValueError("validation and test negative HLAs are identical; expected two distinct HLAs")

    immrep_pos = immrep[immrep["label"] == 1][["peptide", "tcra_trimmed", "tcrb_trimmed", "hla_sequence"]].copy()
    immrep_pos["peptide"] = clean(immrep_pos["peptide"])
    immrep_pos["tcra_trimmed"] = clean(immrep_pos["tcra_trimmed"])
    immrep_pos["tcrb_trimmed"] = clean(immrep_pos["tcrb_trimmed"])
    immrep_pos["hla_sequence"] = clean(immrep_pos["hla_sequence"])

    hla_counts = immrep_pos["hla_sequence"].value_counts()
    if len(hla_counts) != 2:
        raise ValueError(f"IMMREP positives must have exactly 2 HLA sequences, found {len(hla_counts)}")
    if sorted(hla_counts.tolist()) != [500, 500]:
        raise ValueError(f"IMMREP positives must be 500 rows per HLA, found counts={hla_counts.to_dict()}")

    immrep_hlas = set(hla_counts.index.tolist())
    if val_hla not in immrep_hlas:
        raise ValueError("validation negative HLA does not match either IMMREP positive HLA sequence")
    if test_hla not in immrep_hlas:
        raise ValueError("test negative HLA does not match either IMMREP positive HLA sequence")

    immrep_val_pos = immrep_pos[immrep_pos["hla_sequence"] == val_hla].copy().reset_index(drop=True)
    immrep_test_pos = immrep_pos[immrep_pos["hla_sequence"] == test_hla].copy().reset_index(drop=True)

    if len(immrep_val_pos) != 500 or len(immrep_test_pos) != 500:
        raise ValueError(
            f"IMMREP split by mapped HLA must yield 500/500 rows. "
            f"val={len(immrep_val_pos)}, test={len(immrep_test_pos)}"
        )

    val_new_pos = make_immrep_positive_rows(immrep_val_pos, list(val_df.columns), "val",)
    test_new_pos = make_immrep_positive_rows(immrep_test_pos, list(test_df.columns), "test",)

    # Collision checks against existing negatives.
    if set(clean(val_new_pos["pair_id"])) & val_neg_ids:
        raise ValueError("new validation IMMREP pair_ids collide with existing validation negative pair_ids")
    if set(clean(test_new_pos["pair_id"])) & test_neg_ids:
        raise ValueError("new test IMMREP pair_ids collide with existing test negative pair_ids")
    if set(clean(val_new_pos["pair_id"])) & set(clean(test_new_pos["pair_id"])):
        raise ValueError("new validation/test IMMREP pair_ids overlap")

    val_new = pd.concat([val_neg, val_new_pos], ignore_index=True)
    test_new = pd.concat([test_neg, test_new_pos], ignore_index=True)

    # Build manifests from retained negative pair_ids + new IMMREP positives.
    val_manifest_neg = val_manifest[val_manifest["pair_id"].isin(val_neg_ids)].copy().reset_index(drop=True)
    test_manifest_neg = test_manifest[test_manifest["pair_id"].isin(test_neg_ids)].copy().reset_index(drop=True)

    val_manifest_new_pos = make_new_manifest_rows(
        immrep_val_pos, "val", list(val_manifest.columns), val_manifest
    )
    test_manifest_new_pos = make_new_manifest_rows(
        immrep_test_pos, "test", list(test_manifest.columns), test_manifest
    )
    val_manifest_new = pd.concat([val_manifest_neg, val_manifest_new_pos], ignore_index=True)
    test_manifest_new = pd.concat([test_manifest_neg, test_manifest_new_pos], ignore_index=True)

    assert_pair_id_integrity(val_new, val_manifest_new, "validation")
    assert_pair_id_integrity(test_new, test_manifest_new, "test")

    Path(args.out_val_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_val_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_test_manifest).parent.mkdir(parents=True, exist_ok=True)

    val_new.to_csv(args.out_val_csv, index=False)
    test_new.to_csv(args.out_test_csv, index=False)
    val_manifest_new.to_csv(args.out_val_manifest, index=False)
    test_manifest_new.to_csv(args.out_test_manifest, index=False)

    print(f"validation negative HLA mapped to IMMREP positives: {val_hla}")
    print(f"test negative HLA mapped to IMMREP positives: {test_hla}")
    print_summary(
        val_new,
        test_new,
        val_manifest_new,
        test_manifest_new,
        peptide_col="Peptide",
        tcr_a_col="TCRa",
        tcr_b_col="TCRb",
        tcr_full_col="TCR_full",
        hla_col="HLA_sequence",
    )
    print("\nWrote:")
    print(f"- {args.out_val_csv}")
    print(f"- {args.out_test_csv}")
    print(f"- {args.out_val_manifest}")
    print(f"- {args.out_test_manifest}")


if __name__ == "__main__":
    main()
