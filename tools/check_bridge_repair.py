#!/usr/bin/env python3
"""Run the exact finite Stage 2-I2 bridge-repair audit."""

from __future__ import annotations

from itertools import combinations, product
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.bridge_repair import (  # noqa: E402
    BinaryObservation,
    FiniteBridgeCode,
    joint_bridge_repairs,
    one_sided_relation_repair,
    threshold_profile,
    thresholds_separate_distance_alphabet,
)


def subsets(values):
    values = tuple(values)
    return tuple(
        frozenset(
            values[index]
            for index in range(len(values))
            if mask & (1 << index)
        )
        for mask in range(1 << len(values))
    )


def neighborhood(word, max_errors, max_erasures):
    return {
        observation
        for observation in product((0, 1, None), repeat=len(word))
        if sum(value is None for value in observation) <= max_erasures
        and sum(
            value is not None and value != bit
            for value, bit in zip(observation, word, strict=True)
        )
        <= max_errors
    }


def local_repair_audit() -> dict[str, object]:
    cells = (0, 1, 2)
    relations = subsets(cells)
    regimes = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    violations = []
    case_count = 0
    for relation, induced, (relation_weight, induced_weight) in product(
        relations,
        relations,
        regimes,
    ):
        repairs = joint_bridge_repairs(
            cells,
            relation,
            induced,
            relation_weights=relation_weight,
            induced_weights=induced_weight,
        )
        brute_costs = {
            consensus: (
                relation_weight * len(consensus.symmetric_difference(relation))
                + induced_weight * len(consensus.symmetric_difference(induced))
            )
            for consensus in relations
        }
        minimum = min(brute_costs.values())
        expected = {
            consensus
            for consensus, cost in brute_costs.items()
            if cost == minimum
        }
        actual = {repair.consensus for repair in repairs}
        conflicts = len(relation.symmetric_difference(induced))
        expected_count = (
            2**conflicts
            if relation_weight == induced_weight
            else 1
        )
        if (
            actual != expected
            or len(repairs) != expected_count
            or any(repair.cost != minimum for repair in repairs)
        ):
            violations.append(
                {
                    "relation": sorted(relation),
                    "induced": sorted(induced),
                    "weights": [relation_weight, induced_weight],
                }
            )
        case_count += 1

    one_sided = one_sided_relation_repair(
        ("00", "01", "10", "11"),
        {"00", "01", "11"},
        {"00", "10"},
    )
    minimal = joint_bridge_repairs(("cell",), (), ("cell",))
    return {
        "case_count": case_count,
        "violations": violations,
        "one_sided": {
            "flip_count": len(one_sided.relation_flips),
            "normalized_defect": one_sided.normalized_defect,
            "unique": True,
        },
        "minimal_equal_weight_conflict": {
            "cell_count": 1,
            "repair_count": len(minimal),
            "minimum_cost": minimal[0].cost,
        },
    }


def error_erasure_audit() -> dict[str, object]:
    words = tuple(product((0, 1), repeat=3))
    violations = []
    case_count = 0
    for candidate_count in (2, 3):
        for selected_words in combinations(words, candidate_count):
            code = FiniteBridgeCode(
                probes=("p0", "p1", "p2"),
                codewords=tuple(
                    (f"c{index}", word)
                    for index, word in enumerate(selected_words)
                ),
            )
            for max_errors, max_erasures in product(range(3), repeat=2):
                neighborhoods = [
                    neighborhood(word, max_errors, max_erasures)
                    for word in selected_words
                ]
                disjoint = all(
                    left.isdisjoint(right)
                    for left, right in combinations(neighborhoods, 2)
                )
                criterion = code.error_erasure_identifiable(
                    max_errors,
                    max_erasures,
                )
                if criterion != disjoint:
                    violations.append(
                        {
                            "words": selected_words,
                            "max_errors": max_errors,
                            "max_erasures": max_erasures,
                        }
                    )
                case_count += 1
    return {
        "case_count": case_count,
        "violations": violations,
    }


def identifiability_audit() -> dict[str, object]:
    code = FiniteBridgeCode(
        probes=("p0", "p1", "p2"),
        codewords=(
            ("a", (0, 0, 0)),
            ("b", (0, 1, 1)),
            ("c", (1, 0, 1)),
        ),
    )
    supports = [
        set(support)
        for _, _, support in code.pairwise_difference_supports()
    ]
    violations = []
    subset_count = 0
    for size in range(4):
        for selected in combinations(code.probes, size):
            hitting = all(set(selected).intersection(support) for support in supports)
            if code.is_identifiable(selected) != hitting:
                violations.append(selected)
            subset_count += 1

    weighted_code = FiniteBridgeCode(
        probes=("p0", "p1", "p2", "p3"),
        codewords=(
            ("zero", (0, 0, 0, 0)),
            ("one", (1, 1, 1, 1)),
            ("mixed", (1, 0, 1, 0)),
        ),
    )
    weights = {"p0": 1.0, "p1": 2.0, "p2": 4.0, "p3": 8.0}
    observations = tuple(
        BinaryObservation(tuple(word))
        for word in product((0, 1), repeat=4)
    )
    margin_violations = []
    certified_pairs = 0
    for left, right in product(observations, repeat=2):
        result = weighted_code.nearest_repair(left, weights=weights)
        budget = weighted_code.observation_distance(
            left,
            right,
            weights=weights,
        )
        if result.is_unique and 2 * budget < result.margin:
            certified_pairs += 1
            perturbed = weighted_code.nearest_repair(right, weights=weights)
            if (
                not perturbed.is_unique
                or perturbed.candidates != result.candidates
            ):
                margin_violations.append(
                    {
                        "left": left.values,
                        "right": right.values,
                    }
                )
    return {
        "probe_subset_count": subset_count,
        "hitting_criterion_violations": violations,
        "minimum_identifying_probe_sets": code.minimum_identifying_probe_sets(),
        "information_lower_bound": code.information_lower_bound,
        "minimum_distance": code.minimum_distance(),
        "margin_certified_pair_count": certified_pairs,
        "margin_violations": margin_violations,
    }


def specialization_audit() -> dict[str, object]:
    complex_code = FiniteBridgeCode(
        probes=("edge01", "edge02", "edge12", "triangle012"),
        codewords=(
            ("boundary", (1, 1, 1, 0)),
            ("filled", (1, 1, 1, 1)),
        ),
    )
    category_code = FiniteBridgeCode(
        probes=("composable_a_a", "a_squared_is_e"),
        codewords=(("C4", (1, 0)), ("V4", (1, 1))),
    )
    dynamics_code = FiniteBridgeCode(
        probes=("reach01", "reach12", "reach02", "step02"),
        codewords=(("path", (1, 1, 1, 0)), ("shortcut", (1, 1, 1, 1))),
    )
    order_code = FiniteBridgeCode(
        probes=("cross_a0_b", "cross_a1_b", "within_a0_a1"),
        codewords=(
            ("equality_order", (0, 0, 0)),
            ("within_type_order", (0, 0, 1)),
        ),
    )
    alphabet = (1.0, 2.0, 4.0)
    separating_thresholds = (1.0, 2.0)
    nonseparating_thresholds = (5.0,)
    return {
        "adjacency_only_identifiable": complex_code.is_identifiable(
            ("edge01", "edge02", "edge12")
        ),
        "all_simplex_probes_identifiable": complex_code.is_identifiable(),
        "composability_only_identifiable": category_code.is_identifiable(
            ("composable_a_a",)
        ),
        "composition_probe_identifiable": category_code.is_identifiable(),
        "reachability_only_identifiable": dynamics_code.is_identifiable(
            ("reach01", "reach12", "reach02")
        ),
        "transition_probe_identifiable": dynamics_code.is_identifiable(),
        "cross_order_only_identifiable": order_code.is_identifiable(
            ("cross_a0_b", "cross_a1_b")
        ),
        "full_order_probe_identifiable": order_code.is_identifiable(),
        "distance_alphabet": alphabet,
        "separating_thresholds": separating_thresholds,
        "separating_profiles": [
            threshold_profile(distance, separating_thresholds)
            for distance in alphabet
        ],
        "separated": thresholds_separate_distance_alphabet(
            alphabet,
            separating_thresholds,
        ),
        "nonseparating_thresholds": nonseparating_thresholds,
        "nonseparated": not thresholds_separate_distance_alphabet(
            alphabet,
            nonseparating_thresholds,
        ),
    }


def main() -> int:
    result = {
        "local_repair": local_repair_audit(),
        "error_erasure": error_erasure_audit(),
        "identifiability": identifiability_audit(),
        "specializations": specialization_audit(),
    }
    result["status"] = (
        "passed"
        if not result["local_repair"]["violations"]
        and not result["error_erasure"]["violations"]
        and not result["identifiability"]["hitting_criterion_violations"]
        and not result["identifiability"]["margin_violations"]
        and not result["specializations"]["adjacency_only_identifiable"]
        and result["specializations"]["all_simplex_probes_identifiable"]
        and not result["specializations"]["composability_only_identifiable"]
        and result["specializations"]["composition_probe_identifiable"]
        and not result["specializations"]["reachability_only_identifiable"]
        and result["specializations"]["transition_probe_identifiable"]
        and not result["specializations"]["cross_order_only_identifiable"]
        and result["specializations"]["full_order_probe_identifiable"]
        and result["specializations"]["separated"]
        and result["specializations"]["nonseparated"]
        else "failed"
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output = REPOSITORY_ROOT / "experiments" / "bridge_repair" / "results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{rendered}\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
