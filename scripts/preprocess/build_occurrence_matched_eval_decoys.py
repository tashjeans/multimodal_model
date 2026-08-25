#!/usr/bin/env python3
"""Build occurrence-matched TULIP-style validation/test decoys.

This script implements the exact negative-sampling strategy used for the
corrected internal validation and test sets:

1. Keep every observed positive row unchanged.
2. Create exactly one negative for every positive, holding the target pMHC
   (Peptide + HLA_sequence) fixed.
3. Use TCRs only from positive rows in the SAME evaluation split.
4. Use every positive TCR occurrence exactly once as a negative donor.
   Therefore the exact TCR-sequence marginal is identical in positives and
   negatives. Because the target pMHC is held fixed one-for-one, the exact
   pMHC marginal is also identical.
5. Require donor_peptide != target Peptide.
6. Exclude any generated full pair (TCR_full, Peptide, HLA_sequence) that is
   known positive or otherwise supplied as forbidden.
7. Require unique negative full molecular pairs.
8. Maximise reuse of eligible existing negative complexes, so existing YAML,
   MSA and Boltz outputs can be retained where possible.
9. Build validation first; when building test, forbid every final validation
   full pair so there is no exact val/test full-pair overlap.
10. Reused negatives retain their historical pair_id and provenance metadata.
    Newly generated negatives receive pipeline-safe IDs:
      val_tulip_occurrence_matched_decoy_######
      test_tulip_occurrence_matched_decoy_######

Dependencies: networkx, numpy, scipy
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

Row = Dict[str, str]
PairKey = Tuple[str, str, str]        # TCR_full, Peptide, HLA_sequence
PMHCKey = Tuple[str, str]            # Peptide, HLA_sequence


def read_csv(path: Path) -> List[Row]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Row], fieldnames: List[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"Cannot infer fields for empty output: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def is_positive(row: Row) -> bool:
    return float(row["binding_flag"]) == 1.0


def full_pair(row: Row) -> PairKey:
    return row["TCR_full"], row["Peptide"], row["HLA_sequence"]


def pmhc(row: Row) -> PMHCKey:
    return row["Peptide"], row["HLA_sequence"]


def proposed_pair(donor: Row, target: Row) -> PairKey:
    return donor["TCR_full"], target["Peptide"], target["HLA_sequence"]


def select_maximum_reuse(
    positives: List[Row], old_negatives: List[Row], forbidden: Set[PairKey]
) -> Tuple[List[Row], int]:
    """Select the largest feasible set of old molecular negative complexes.

    A capacitated max-flow is solved from TCR sequence -> target pMHC.
    Sequence capacities equal their positive occurrence counts; pMHC demands
    equal their positive occurrence counts. Each existing molecular complex
    can be reused at most once.
    """
    edge_to_row: Dict[Tuple[str, PMHCKey], Row] = {}
    for row in old_negatives:
        key = full_pair(row)
        if key in forbidden:
            continue
        if row.get("donor_peptide", "") == row["Peptide"]:
            continue
        edge = (row["TCR_full"], pmhc(row))
        edge_to_row.setdefault(edge, row)

    sequence_capacity = collections.Counter(r["TCR_full"] for r in positives)
    pmhc_demand = collections.Counter(pmhc(r) for r in positives)

    graph = nx.DiGraph()
    source, sink = ("root", "source"), ("root", "sink")
    for seq, cap in sequence_capacity.items():
        graph.add_edge(source, ("seq", seq), capacity=cap)
    for target_pmhc, demand in pmhc_demand.items():
        graph.add_edge(("pmhc", target_pmhc), sink, capacity=demand)
    for seq, target_pmhc in edge_to_row:
        graph.add_edge(("seq", seq), ("pmhc", target_pmhc), capacity=1)

    flow_value, flow = nx.maximum_flow(
        graph, source, sink, flow_func=nx.algorithms.flow.dinitz
    )

    selected: List[Row] = []
    for (seq, target_pmhc), row in edge_to_row.items():
        if flow.get(("seq", seq), {}).get(("pmhc", target_pmhc), 0) > 0:
            selected.append(row)

    assert len(selected) == flow_value
    return selected, flow_value


def assign_occurrences_to_reused(
    positives: List[Row], reused: List[Row]
) -> Dict[str, int]:
    """Assign each retained old complex to a unique positive donor occurrence.

    Multiple positive rows can contain the same exact TCR_full. Assignment is
    done separately within each TCR sequence. Historical donor_pair_id is
    preferred (cost 0); an alternative positive row carrying the identical
    TCR sequence costs 1. The donor's cognate peptide must differ from the
    negative target peptide.

    Returns: target source_pair_id -> positive-row index used for accounting.
    Historical donor metadata on reused negative rows is NOT overwritten.
    """
    by_sequence_positive: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, row in enumerate(positives):
        by_sequence_positive[row["TCR_full"]].append(idx)

    by_sequence_reused: Dict[str, List[Row]] = collections.defaultdict(list)
    for row in reused:
        by_sequence_reused[row["TCR_full"]].append(row)

    assignment: Dict[str, int] = {}
    for seq, rows in by_sequence_reused.items():
        donor_indices = by_sequence_positive[seq]
        m, n = len(rows), len(donor_indices)
        if m > n:
            raise RuntimeError(f"More retained uses than positive occurrences for TCR {seq}")

        cost = np.full((m, n), 1000.0)
        for i, neg in enumerate(rows):
            for j, donor_idx in enumerate(donor_indices):
                donor = positives[donor_idx]
                if donor["Peptide"] == neg["Peptide"]:
                    continue
                cost[i, j] = 0.0 if neg.get("donor_pair_id") == donor["pair_id"] else 1.0

        row_idx, col_idx = linear_sum_assignment(cost)
        if len(row_idx) != m or np.any(cost[row_idx, col_idx] >= 1000):
            raise RuntimeError(f"Could not assign donor occurrences for TCR {seq}")

        for i, j in zip(row_idx, col_idx):
            assignment[rows[i]["source_pair_id"]] = donor_indices[j]

    return assignment


def match_remaining_occurrences(
    positives: List[Row],
    remaining_targets: List[int],
    remaining_donors: List[int],
    forbidden: Set[PairKey],
    fixed_negative_keys: Set[PairKey],
    seed: int,
    max_attempts: int = 20,
) -> Dict[int, int]:
    """Perfect-match remaining positive targets to remaining donor occurrences."""
    rng = random.Random(seed)
    n = len(remaining_targets)
    if n != len(remaining_donors):
        raise ValueError("Target and donor occurrence counts differ")

    fixed_counts = collections.Counter(fixed_negative_keys)

    def valid(target_idx: int, donor_idx: int) -> bool:
        target, donor = positives[target_idx], positives[donor_idx]
        if donor["Peptide"] == target["Peptide"]:
            return False
        key = proposed_pair(donor, target)
        if key in forbidden or fixed_counts[key] > 0:
            return False
        return True

    for _ in range(max_attempts):
        targets = remaining_targets[:]
        donors = remaining_donors[:]
        rng.shuffle(targets)
        rng.shuffle(donors)

        matrix_rows, matrix_cols = [], []
        for i, target_idx in enumerate(targets):
            target = positives[target_idx]
            for j, donor_idx in enumerate(donors):
                donor = positives[donor_idx]
                if donor["Peptide"] == target["Peptide"]:
                    continue
                key = proposed_pair(donor, target)
                if key in forbidden or fixed_counts[key] > 0:
                    continue
                matrix_rows.append(i)
                matrix_cols.append(j)

        adjacency = csr_matrix(
            (np.ones(len(matrix_rows), dtype=np.int8), (matrix_rows, matrix_cols)),
            shape=(n, n),
        )
        match = maximum_bipartite_matching(adjacency, perm_type="column")
        if len(match) != n or np.any(match < 0):
            continue

        assignment = {targets[i]: donors[int(match[i])] for i in range(n)}
        if len(set(assignment.values())) != n:
            continue

        # Different positive rows can carry the same TCR sequence, so a perfect
        # occurrence matching can still create duplicate molecular negatives.
        # Repair any such duplicates by valid donor swaps.
        key_counts = collections.Counter(fixed_negative_keys)
        for target_idx, donor_idx in assignment.items():
            key_counts[proposed_pair(positives[donor_idx], positives[target_idx])] += 1

        def duplicate_targets() -> List[int]:
            seen = collections.Counter(fixed_negative_keys)
            bad = []
            for target_idx in assignment:
                key = proposed_pair(positives[assignment[target_idx]], positives[target_idx])
                seen[key] += 1
                if seen[key] > 1:
                    bad.append(target_idx)
            return bad

        def can_swap(i: int, j: int) -> bool:
            if i == j:
                return False
            di, dj = assignment[i], assignment[j]
            if not valid(i, dj) or not valid(j, di):
                return False
            old_i = proposed_pair(positives[di], positives[i])
            old_j = proposed_pair(positives[dj], positives[j])
            new_i = proposed_pair(positives[dj], positives[i])
            new_j = proposed_pair(positives[di], positives[j])
            if new_i == new_j:
                return False
            rem_i = key_counts[new_i] - int(new_i == old_i) - int(new_i == old_j)
            rem_j = key_counts[new_j] - int(new_j == old_i) - int(new_j == old_j)
            return rem_i == 0 and rem_j == 0

        def do_swap(i: int, j: int) -> None:
            old_i = proposed_pair(positives[assignment[i]], positives[i])
            old_j = proposed_pair(positives[assignment[j]], positives[j])
            key_counts[old_i] -= 1
            key_counts[old_j] -= 1
            assignment[i], assignment[j] = assignment[j], assignment[i]
            key_counts[proposed_pair(positives[assignment[i]], positives[i])] += 1
            key_counts[proposed_pair(positives[assignment[j]], positives[j])] += 1

        ok = True
        for _repair_round in range(20):
            bad = duplicate_targets()
            if not bad:
                break
            rng.shuffle(bad)
            for target_idx in bad:
                candidates = list(assignment.keys())
                rng.shuffle(candidates)
                repaired = False
                for other_idx in candidates[: min(len(candidates), 5000)]:
                    if can_swap(target_idx, other_idx):
                        do_swap(target_idx, other_idx)
                        repaired = True
                        break
                if not repaired:
                    ok = False
                    break
            if not ok:
                break

        if ok and not duplicate_targets():
            return assignment

    raise RuntimeError("Could not find a valid perfect occurrence matching")


def build_split(
    split: str,
    input_rows: List[Row],
    forbidden: Set[PairKey],
    seed: int,
) -> Tuple[List[Row], List[Row], List[Row], List[Row], int]:
    positives = [r for r in input_rows if is_positive(r)]
    old_negatives = [r for r in input_rows if not is_positive(r)]
    positive_id_to_index = {r["pair_id"]: i for i, r in enumerate(positives)}

    reused, max_reuse = select_maximum_reuse(positives, old_negatives, forbidden)
    reused = [r for r in reused if r.get("source_pair_id") in positive_id_to_index]

    accounting_assignment = assign_occurrences_to_reused(positives, reused)
    reused_by_target = {
        positive_id_to_index[r["source_pair_id"]]: r for r in reused
    }
    consumed_donors = set(accounting_assignment.values())
    fixed_keys = {full_pair(r) for r in reused_by_target.values()}

    remaining_targets = [i for i in range(len(positives)) if i not in reused_by_target]
    remaining_donors = [i for i in range(len(positives)) if i not in consumed_donors]
    new_assignment = match_remaining_occurrences(
        positives,
        remaining_targets,
        remaining_donors,
        forbidden,
        fixed_keys,
        seed,
    )

    all_assignment = {
        target_idx: accounting_assignment[positives[target_idx]["pair_id"]]
        for target_idx in reused_by_target
    }
    all_assignment.update(new_assignment)

    if len(all_assignment) != len(positives):
        raise AssertionError("Not every positive target received a negative")
    if len(set(all_assignment.values())) != len(positives):
        raise AssertionError("Positive donor occurrences were not used one-to-one")

    fieldnames = list(input_rows[0].keys())
    output = [dict(r) for r in positives]
    new_only: List[Row] = []
    mapping: List[Row] = []
    new_counter = 0

    for target_idx, target in enumerate(positives):
        donor_idx = all_assignment[target_idx]
        donor = positives[donor_idx]

        if target_idx in reused_by_target:
            old = dict(reused_by_target[target_idx])
            row = old  # preserve pair_id, donor provenance, YAML/MSA/Boltz paths
            status = "reused_existing_boltz"
            historical_donor = old.get("donor_pair_id", "")
            accounting_donor = donor["pair_id"]
        else:
            pair_id = f"{split}_tulip_occurrence_matched_decoy_{new_counter:06d}"
            new_counter += 1
            row = {k: "" for k in fieldnames}
            row.update(
                {
                    "pair_id": pair_id,
                    "Peptide": target["Peptide"],
                    "HLA_sequence": target["HLA_sequence"],
                    "TCR_full": donor["TCR_full"],
                    "binding_flag": "0",
                    "decoy_type": "tulip_occurrence_matched_same_pmhc_donor_tcr",
                    "source_pair_id": target["pair_id"],
                    "donor_pair_id": donor["pair_id"],
                    "donor_peptide": donor["Peptide"],
                    "donor_hla_sequence": donor["HLA_sequence"],
                    "original_pair_id": target["pair_id"],
                    "yaml_path": "",
                    "pep_len": target.get("pep_len", ""),
                    "tcra_len": donor.get("tcra_len", ""),
                    "tcrb_len": donor.get("tcrb_len", ""),
                    "hla_len": target.get("hla_len", ""),
                    "boltz_embedding_npz": "",
                }
            )
            status = "new_boltz_required"
            historical_donor = donor["pair_id"]
            accounting_donor = donor["pair_id"]
            new_only.append(dict(row))

        output.append(dict(row))
        mapping.append(
            {
                "status": status,
                "pair_id": row["pair_id"],
                "target_positive_pair_id": target["pair_id"],
                "provenance_donor_pair_id": historical_donor,
                "accounting_donor_positive_pair_id": accounting_donor,
                "donor_metadata_differs_from_accounting_occurrence": str(
                    historical_donor != accounting_donor
                ),
                "TCR_full": row["TCR_full"],
                "Peptide": row["Peptide"],
                "HLA_sequence": row["HLA_sequence"],
                "yaml_path": row.get("yaml_path", ""),
                "boltz_embedding_npz": row.get("boltz_embedding_npz", ""),
            }
        )

    return output, new_only, mapping, positives, max_reuse


def audit_split(
    split: str,
    output: List[Row],
    positives: List[Row],
    known_positive_keys: Set[PairKey],
    forbidden_before_split: Set[PairKey],
    old_negative_ids: Set[str],
    other_split_negative_keys: Set[PairKey] | None = None,
) -> dict:
    negatives = [r for r in output if not is_positive(r)]
    keys = [full_pair(r) for r in negatives]
    reused = [r for r in negatives if r["pair_id"] in old_negative_ids]
    new = [r for r in negatives if r["pair_id"] not in old_negative_ids]

    return {
        "split": split,
        "positive_rows": len(positives),
        "negative_rows": len(negatives),
        "reused_existing_negative_complexes": len(reused),
        "new_negative_complexes": len(new),
        "tcr_sequence_marginal_exact": collections.Counter(r["TCR_full"] for r in positives)
        == collections.Counter(r["TCR_full"] for r in negatives),
        "pmhc_marginal_exact": collections.Counter(pmhc(r) for r in positives)
        == collections.Counter(pmhc(r) for r in negatives),
        "same_recorded_donor_target_peptide": sum(
            r.get("donor_peptide", "") == r["Peptide"] for r in negatives
        ),
        "known_positive_collision_count": sum(k in known_positive_keys for k in keys),
        "forbidden_full_pair_collision_count": sum(k in forbidden_before_split for k in keys),
        "duplicate_negative_full_pairs": len(keys) - len(set(keys)),
        "cross_split_negative_full_pair_overlap": 0
        if other_split_negative_keys is None
        else len(set(keys) & other_split_negative_keys),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-positives", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True, help="Original val CSV containing positives + old decoys")
    parser.add_argument("--test", type=Path, required=True, help="Original test CSV containing positives + old decoys")
    parser.add_argument(
        "--forbid-csv",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional CSV whose full molecular pairs must not appear as eval negatives. "
            "Repeat as needed. For exact reproduction of the current data, include the "
            "corrected train-decoy CSV here."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-seed", type=int, default=31)
    parser.add_argument("--test-seed", type=int, default=37)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_csv(args.train_positives)
    val_rows = read_csv(args.val)
    test_rows = read_csv(args.test)

    # All recorded positives across train/val/test are forbidden as synthetic negatives.
    known_positive_keys = {
        full_pair(r)
        for rows in (train_rows, val_rows, test_rows)
        for r in rows
        if is_positive(r)
    }

    # Optional additional forbidden pairs (e.g. corrected supervised train decoys)
    # are included to guarantee exact full-pair separation from those files too.
    extra_forbidden: Set[PairKey] = set()
    for path in args.forbid_csv:
        extra_forbidden.update(full_pair(r) for r in read_csv(path))

    base_forbidden = known_positive_keys | extra_forbidden

    val_output, val_new, val_map, val_positives, val_max_reuse = build_split(
        "val", val_rows, base_forbidden, args.val_seed
    )

    # Test cannot duplicate ANY final validation molecular pair.
    val_all_keys = {full_pair(r) for r in val_output}
    test_forbidden = base_forbidden | val_all_keys
    test_output, test_new, test_map, test_positives, test_max_reuse = build_split(
        "test", test_rows, test_forbidden, args.test_seed
    )

    val_fields = list(val_rows[0].keys())
    test_fields = list(test_rows[0].keys())

    write_csv(args.output_dir / "val_multiview_occurrence_matched_max_reuse.csv", val_output, val_fields)
    write_csv(args.output_dir / "test_multiview_occurrence_matched_max_reuse.csv", test_output, test_fields)
    write_csv(args.output_dir / "val_new_negatives_boltz_required_occurrence_matched.csv", val_new, val_fields)
    write_csv(args.output_dir / "test_new_negatives_boltz_required_occurrence_matched.csv", test_new, test_fields)
    write_csv(args.output_dir / "val_negative_reuse_mapping_occurrence_matched.csv", val_map)
    write_csv(args.output_dir / "test_negative_reuse_mapping_occurrence_matched.csv", test_map)

    val_old_ids = {r["pair_id"] for r in val_rows if not is_positive(r)}
    test_old_ids = {r["pair_id"] for r in test_rows if not is_positive(r)}
    val_negative_keys = {full_pair(r) for r in val_output if not is_positive(r)}

    audit = {
        "method": "TULIP-style occurrence-matched same-pMHC donor-TCR decoys",
        "seeds": {"val": args.val_seed, "test": args.test_seed},
        "val_maximum_reusable_existing_complexes": val_max_reuse,
        "test_maximum_reusable_existing_complexes_after_val_exclusions": test_max_reuse,
        "validation": audit_split(
            "val",
            val_output,
            val_positives,
            known_positive_keys,
            base_forbidden,
            val_old_ids,
        ),
        "test": audit_split(
            "test",
            test_output,
            test_positives,
            known_positive_keys,
            test_forbidden,
            test_old_ids,
            val_negative_keys,
        ),
    }

    with (args.output_dir / "occurrence_matched_decoy_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
