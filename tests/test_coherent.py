from __future__ import annotations

import math
import unittest
from itertools import product

from tsi.coherent import (
    BridgeSpec,
    CoherenceSignature,
    CoherentStructuralState,
    TypedCorrespondence,
    are_coherently_isomorphic,
    bridge_defects,
    coherent_structural_discrepancy,
    compose_correspondences,
    correspondence_costs,
    induced_bridge_relation,
    lipschitz_task_error_bound,
    typed_correspondences,
)
from tsi.dynamical import IntegratedStructuralState
from tsi.order_topology import FinitePreorder
from tsi.relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


SCHEMA = FiniteRelationalSchema(
    objects=("entity",),
    arrows=(ArrowSpec("rel", "entity", "entity"),),
)


def make_core(
    entities: tuple[object, ...],
    *,
    labels: tuple[object, ...] | None = None,
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    edges: frozenset[frozenset[object]] = frozenset(),
    filled_triangles: frozenset[frozenset[object]] = frozenset(),
    spacing: float = 1.0,
) -> IntegratedStructuralState:
    labels = labels or tuple("same" for _ in entities)
    relational = FiniteRelationAssignment(
        SCHEMA,
        {"entity": entities},
        {"entity": labels},
        {"rel": FiniteRelation(entities, entities, relation_pairs)},
    )
    tagged = {entity: ("entity", entity) for entity in entities}
    simplices = {
        frozenset(),
        *(frozenset((tagged[entity],)) for entity in entities),
    }
    for edge in edges:
        simplices.add(frozenset(tagged[entity] for entity in edge))
    for triangle in filled_triangles:
        tagged_triangle = frozenset(tagged[entity] for entity in triangle)
        simplices.add(tagged_triangle)
        for first, second in product(triangle, repeat=2):
            if first != second:
                simplices.add(frozenset((tagged[first], tagged[second])))
    distances = tuple(
        tuple(
            abs(left_index - right_index) * spacing
            for right_index, _ in enumerate(entities)
        )
        for left_index, _ in enumerate(entities)
    )
    return IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )


def equality_order(core: IntegratedStructuralState) -> FinitePreorder:
    return FinitePreorder(
        core.tagged_entities,
        frozenset((value, value) for value in core.tagged_entities),
        core.tagged_labels,
    )


def linear_order(core: IntegratedStructuralState) -> FinitePreorder:
    return FinitePreorder(
        core.tagged_entities,
        frozenset(
            (left, right)
            for left_index, left in enumerate(core.tagged_entities)
            for right_index, right in enumerate(core.tagged_entities)
            if left_index <= right_index
        ),
        core.tagged_labels,
    )


def state(
    entities: tuple[object, ...],
    *,
    labels: tuple[object, ...] | None = None,
    relation_pairs: frozenset[tuple[object, object]] = frozenset(),
    edges: frozenset[frozenset[object]] = frozenset(),
    spacing: float = 1.0,
    signature: CoherenceSignature | None = None,
    order_kind: str = "equality",
) -> CoherentStructuralState:
    core = make_core(
        entities,
        labels=labels,
        relation_pairs=relation_pairs,
        edges=edges,
        spacing=spacing,
    )
    order = equality_order(core) if order_kind == "equality" else linear_order(core)
    return CoherentStructuralState(
        core,
        order,
        signature or CoherenceSignature(),
    )


class CommonCorrespondenceMetricTest(unittest.TestCase):
    def test_unequal_carriers_and_label_mismatches_have_finite_cost(self) -> None:
        left = state((0,), labels=("red",))
        right = state(("a", "b"), labels=("red", "blue"))

        value = coherent_structural_discrepancy(left, right)
        self.assertTrue(math.isfinite(value))
        self.assertGreater(value, 0.0)

    def test_zero_exactly_detects_integrated_structural_isomorphism(self) -> None:
        left = state(
            ("a", "b"),
            labels=("red", "blue"),
            relation_pairs=frozenset({("a", "b")}),
            edges=frozenset({frozenset(("a", "b"))}),
            spacing=2.0,
        )
        right = state(
            ("v", "u"),
            labels=("blue", "red"),
            relation_pairs=frozenset({("u", "v")}),
            edges=frozenset({frozenset(("u", "v"))}),
            spacing=2.0,
        )
        changed = state(
            ("v", "u"),
            labels=("blue", "red"),
            relation_pairs=frozenset({("v", "u")}),
            edges=frozenset({frozenset(("u", "v"))}),
            spacing=2.0,
        )

        self.assertTrue(are_coherently_isomorphic(left, right))
        self.assertEqual(coherent_structural_discrepancy(left, right), 0.0)
        self.assertGreater(coherent_structural_discrepancy(left, changed), 0.0)

    def test_metric_axioms_on_mixed_cardinality_family(self) -> None:
        states = (
            state((0,)),
            state((0,), labels=("red",)),
            state((0, 1)),
            state((0, 1), spacing=2.0),
            state(
                (0, 1),
                relation_pairs=frozenset({(0, 1)}),
                edges=frozenset({frozenset((0, 1))}),
            ),
            state((0, 1), order_kind="linear"),
        )
        distances = {
            (i, j): coherent_structural_discrepancy(left, right)
            for i, left in enumerate(states)
            for j, right in enumerate(states)
        }
        for i in range(len(states)):
            self.assertEqual(distances[(i, i)], 0.0)
        for i, j in product(range(len(states)), repeat=2):
            self.assertAlmostEqual(distances[(i, j)], distances[(j, i)])
        for i, j, k in product(range(len(states)), repeat=3):
            self.assertLessEqual(
                distances[(i, k)],
                distances[(i, j)] + distances[(j, k)] + 1e-9,
            )

    def test_composition_respects_each_component_triangle_bound(self) -> None:
        left = state((0,))
        middle = state(("a", "b"))
        right = state(("u", "v"))
        first = next(typed_correspondences(left, middle))
        second = next(typed_correspondences(middle, right))
        composite = compose_correspondences(first, second)

        first_cost = correspondence_costs(first, left, middle)
        second_cost = correspondence_costs(second, middle, right)
        composite_cost = correspondence_costs(composite, left, right)
        for name in ("label", "simplicial", "metric", "relation", "order", "total"):
            self.assertLessEqual(
                getattr(composite_cost, name),
                getattr(first_cost, name) + getattr(second_cost, name) + 1e-9,
            )

    def test_typed_correspondence_must_cover_both_sides(self) -> None:
        left = state((0, 1))
        right = state(("a", "b"))
        incomplete = TypedCorrespondence.from_mapping(
            SCHEMA.objects,
            {"entity": frozenset({(0, "a"), (1, "a")})},
        )
        with self.assertRaisesRegex(ValueError, "right carrier"):
            correspondence_costs(incomplete, left, right)


class BridgeCompatibilityTest(unittest.TestCase):
    def test_adjacency_metric_and_order_bridges(self) -> None:
        edge_pairs = frozenset({(0, 1), (1, 0)})
        edge_signature = CoherenceSignature(
            bridges=(BridgeSpec("rel", "adjacency"),)
        )
        edge_state = state(
            (0, 1),
            relation_pairs=edge_pairs,
            edges=frozenset({frozenset((0, 1))}),
            signature=edge_signature,
        )
        self.assertEqual(
            bridge_defects(edge_state.core, edge_state.order, edge_signature)["rel"],
            0.0,
        )

        threshold_signature = CoherenceSignature(
            bridges=(BridgeSpec("rel", "metric_threshold", 1.0),)
        )
        threshold_pairs = frozenset(product((0, 1), repeat=2))
        threshold_state = state(
            (0, 1),
            relation_pairs=threshold_pairs,
            signature=threshold_signature,
        )
        self.assertEqual(
            induced_bridge_relation(
                threshold_state.core,
                threshold_state.order,
                threshold_signature.bridges[0],
            ),
            threshold_pairs,
        )

        order_signature = CoherenceSignature(
            bridges=(BridgeSpec("rel", "order"),)
        )
        order_pairs = frozenset({(0, 0), (0, 1), (1, 1)})
        order_state = state(
            (0, 1),
            relation_pairs=order_pairs,
            signature=order_signature,
            order_kind="linear",
        )
        self.assertEqual(
            bridge_defects(order_state.core, order_state.order, order_signature)[
                "rel"
            ],
            0.0,
        )

    def test_incoherent_bridge_is_rejected_with_measured_defect(self) -> None:
        signature = CoherenceSignature(
            bridges=(BridgeSpec("rel", "adjacency"),)
        )
        core = make_core(
            (0, 1),
            relation_pairs=frozenset(),
            edges=frozenset({frozenset((0, 1))}),
        )
        order = equality_order(core)
        self.assertEqual(bridge_defects(core, order, signature)["rel"], 0.5)
        with self.assertRaisesRegex(ValueError, "bridge constraints"):
            CoherentStructuralState(core, order, signature)

    def test_adjacency_does_not_determine_higher_simplices(self) -> None:
        entities = (0, 1, 2)
        edge_pairs = frozenset(
            (left, right)
            for left in entities
            for right in entities
            if left != right
        )
        signature = CoherenceSignature(
            bridges=(BridgeSpec("rel", "adjacency"),)
        )
        edges = frozenset(
            {
                frozenset((0, 1)),
                frozenset((0, 2)),
                frozenset((1, 2)),
            }
        )
        boundary_core = make_core(
            entities,
            relation_pairs=edge_pairs,
            edges=edges,
        )
        filled_core = make_core(
            entities,
            relation_pairs=edge_pairs,
            edges=edges,
            filled_triangles=frozenset({frozenset(entities)}),
        )
        boundary = CoherentStructuralState(
            boundary_core,
            equality_order(boundary_core),
            signature,
        )
        filled = CoherentStructuralState(
            filled_core,
            equality_order(filled_core),
            signature,
        )

        self.assertEqual(
            induced_bridge_relation(
                boundary.core, boundary.order, signature.bridges[0]
            ),
            induced_bridge_relation(
                filled.core, filled.order, signature.bridges[0]
            ),
        )
        self.assertGreater(coherent_structural_discrepancy(boundary, filled), 0.0)


class RecoveryBoundaryTest(unittest.TestCase):
    def test_lipschitz_readout_certificate_and_constant_readout_limit(self) -> None:
        target = state((0, 1), spacing=1.0)
        prediction = state((0, 1), spacing=2.0)
        error = coherent_structural_discrepancy(prediction, target)

        self.assertGreater(error, 0.0)
        self.assertEqual(lipschitz_task_error_bound(error, 0.0), 0.0)
        self.assertAlmostEqual(
            lipschitz_task_error_bound(error, 3.0),
            3.0 * error,
        )


if __name__ == "__main__":
    unittest.main()
