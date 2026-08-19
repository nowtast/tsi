#!/usr/bin/env python3
"""Exhaustively audit the finite Stage 2-I1 correspondence spectrum."""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from tsi.coherence_spectrum import (  # noqa: E402
    LayerDistortionVector,
    alignment_frustration,
    audit_pareto_triangle,
    coherent_correspondence_spectrum,
    signature_weights,
)
from tsi.coherent import (  # noqa: E402
    CoherenceSignature,
    CoherentStructuralState,
    coherent_structural_discrepancy,
)
from tsi.dynamical import IntegratedStructuralState  # noqa: E402
from tsi.order_topology import FinitePreorder  # noqa: E402
from tsi.relational import (  # noqa: E402
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


SCHEMA = FiniteRelationalSchema(
    objects=("entity",),
    arrows=(ArrowSpec("rel", "entity", "entity"),),
)


def state(
    entities: tuple[object, ...],
    *,
    labels: tuple[object, ...] | None = None,
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    spacing: float = 1.0,
    edge: bool = False,
    linear_order: bool = False,
) -> CoherentStructuralState:
    labels = labels or tuple("same" for _ in entities)
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": FiniteRelation(entities, entities, relation_pairs)},
    )
    tagged = tuple(("entity", entity) for entity in entities)
    simplices = {
        frozenset(),
        *(frozenset((entity,)) for entity in tagged),
    }
    if edge and len(tagged) == 2:
        simplices.add(frozenset(tagged))
    distances = tuple(
        tuple(
            abs(left_index - right_index) * spacing
            for right_index in range(len(entities))
        )
        for left_index in range(len(entities))
    )
    core = IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )
    if linear_order:
        order_relation = frozenset(
            (left, right)
            for left_index, left in enumerate(tagged)
            for right_index, right in enumerate(tagged)
            if left_index <= right_index
        )
    else:
        order_relation = frozenset((entity, entity) for entity in tagged)
    return CoherentStructuralState(
        core,
        FinitePreorder(tagged, order_relation, core.tagged_labels),
        CoherenceSignature(),
    )


def conflict_pair() -> tuple[CoherentStructuralState, CoherentStructuralState]:
    labels = ("red", "blue")
    return (
        state(
            (0, 1),
            labels=labels,
            relation_pairs=frozenset({(0, 0)}),
        ),
        state(
            (0, 1),
            labels=labels,
            relation_pairs=frozenset({(1, 1)}),
        ),
    )


def audit_family() -> tuple[CoherentStructuralState, ...]:
    conflict_left, conflict_right = conflict_pair()
    return (
        state((0,)),
        state((0, 1)),
        state((0, 1), spacing=2.0),
        conflict_left,
        conflict_right,
        state((0, 1), edge=True, linear_order=True),
    )


def run_audit() -> dict[str, object]:
    states = audit_family()
    spectra = {
        (i, j): coherent_correspondence_spectrum(left, right)
        for i, left in enumerate(states)
        for j, right in enumerate(states)
    }

    symmetry_violations: list[dict[str, int]] = []
    diagonal_violations: list[int] = []
    frustration_violations: list[dict[str, int]] = []
    scalarization_violations: list[dict[str, object]] = []
    max_scalarization_error = 0.0
    zero = LayerDistortionVector.zero()

    for i, j in product(range(len(states)), repeat=2):
        forward = spectra[(i, j)]
        backward = spectra[(j, i)]
        if (
            forward.attainable != backward.attainable
            or forward.pareto != backward.pareto
        ):
            symmetry_violations.append({"left": i, "right": j})

        weights = signature_weights(states[i])
        scalarized = forward.scalarized_value(weights)
        i0_value = coherent_structural_discrepancy(states[i], states[j])
        error = abs(scalarized - i0_value)
        max_scalarization_error = max(max_scalarization_error, error)
        if error > 1e-12:
            scalarization_violations.append(
                {
                    "left": i,
                    "right": j,
                    "spectrum_value": scalarized,
                    "i0_value": i0_value,
                    "absolute_error": error,
                }
            )

        frustration = alignment_frustration(forward, weights)
        if frustration.is_zero != forward.ideal_is_attainable:
            frustration_violations.append({"left": i, "right": j})
        if i == j and forward.pareto != (zero,):
            diagonal_violations.append(i)

    triangle_violations: list[dict[str, object]] = []
    tested_frontier_pairs = 0
    for i, j, k in product(range(len(states)), repeat=3):
        audit = audit_pareto_triangle(
            spectra[(i, j)],
            spectra[(j, k)],
            spectra[(i, k)],
        )
        tested_frontier_pairs += audit.tested_frontier_pairs
        if not audit.passed:
            triangle_violations.append(
                {
                    "first": i,
                    "middle": j,
                    "last": k,
                    "violations": [
                        {
                            "left": left.as_dict(),
                            "right": right.as_dict(),
                        }
                        for left, right in audit.violations
                    ],
                }
            )

    conflict_left, conflict_right = conflict_pair()
    conflict = coherent_correspondence_spectrum(conflict_left, conflict_right)
    expected_frontier = (
        LayerDistortionVector(0, 0, 0, 1, 0),
        LayerDistortionVector(1, 0, 0, 0, 0),
    )
    witness_weights = (2.0, 3.0, 4.0, 5.0, 6.0)
    conflict_frustration = alignment_frustration(conflict, witness_weights)
    singleton_counts = tuple(
        coherent_correspondence_spectrum(left, right).correspondence_count
        for left, right in (
            (state((0,)), state(("a",))),
            (state((0,)), state(("a", "b"))),
            (state((0, 1)), state(("a",))),
        )
    )
    strict_example_passed = (
        conflict.pareto == expected_frontier
        and conflict.ideal == zero
        and not conflict.ideal_is_attainable
        and conflict_frustration.gap
        == min(witness_weights[0], witness_weights[3])
    )
    singleton_minimality_passed = singleton_counts == (1, 1, 1)

    status = "passed"
    if any(
        (
            symmetry_violations,
            diagonal_violations,
            frustration_violations,
            scalarization_violations,
            triangle_violations,
        )
    ):
        status = "failed"
    if not strict_example_passed or not singleton_minimality_passed:
        status = "failed"

    return {
        "status": status,
        "layer_order": [
            "label",
            "simplicial",
            "metric",
            "relation",
            "order",
        ],
        "audit_family": {
            "state_count": len(states),
            "ordered_pair_count": len(states) ** 2,
            "ordered_triple_count": len(states) ** 3,
            "tested_frontier_pair_sums": tested_frontier_pairs,
        },
        "checks": {
            "symmetry_violations": symmetry_violations,
            "diagonal_violations": diagonal_violations,
            "scalarization_violations": scalarization_violations,
            "max_scalarization_absolute_error": max_scalarization_error,
            "frustration_equivalence_violations": frustration_violations,
            "pareto_triangle_violations": triangle_violations,
        },
        "strict_separation_example": {
            "passed": strict_example_passed,
            "spectrum": conflict.as_dict(),
            "weights": witness_weights,
            "frustration": conflict_frustration.as_dict(),
        },
        "one_sided_singleton_minimality": {
            "passed": singleton_minimality_passed,
            "correspondence_counts": singleton_counts,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-report",
        type=Path,
        default=Path("experiments/coherence_spectrum/results.json"),
    )
    args = parser.parse_args()

    result = run_audit()
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    output = args.write_report
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"{rendered}\n")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
