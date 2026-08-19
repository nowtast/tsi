"""Exact finite reference model for TSI Extension 2A-X1.

The module implements finite preorders, their upper-set Alexandrov topologies,
specialization, monotonicity/continuity checks, and the exact order errors used
in the accompanying paper. Exhaustive use is intended only for small carriers.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import permutations, product
from math import factorial, inf, isclose, isfinite
from types import MappingProxyType
from typing import Hashable, Iterator, Mapping


Element = Hashable
Label = Hashable
Relation = frozenset[tuple[Element, Element]]
Topology = frozenset[frozenset[Element]]
_TOLERANCE = 1e-12


@dataclass(frozen=True)
class FinitePreorder:
    """A nonempty finite labeled carrier with a reflexive transitive relation."""

    elements: tuple[Element, ...]
    relation: Relation
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        elements = tuple(self.elements)
        labels = tuple(self.labels)
        raw_relation = tuple(self.relation)
        if not elements:
            raise ValueError("a finite preorder carrier must be nonempty")
        if len(set(elements)) != len(elements):
            raise ValueError("preorder carrier elements must be unique")
        if len(labels) != len(elements):
            raise ValueError("there must be exactly one label per carrier element")
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in raw_relation):
            raise ValueError("every relation member must be an ordered pair")

        relation = frozenset((pair[0], pair[1]) for pair in raw_relation)
        carrier = frozenset(elements)
        if any(left not in carrier or right not in carrier for left, right in relation):
            raise ValueError("the relation contains an element outside its carrier")

        missing_reflexive = {(value, value) for value in elements} - relation
        if missing_reflexive:
            raise ValueError("a preorder relation must be reflexive")

        for left, middle in relation:
            for second_middle, right in relation:
                if middle == second_middle and (left, right) not in relation:
                    raise ValueError("a preorder relation must be transitive")

        object.__setattr__(self, "elements", elements)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "labels", labels)

    @property
    def label_map(self) -> Mapping[Element, Label]:
        return MappingProxyType(dict(zip(self.elements, self.labels, strict=True)))

    @property
    def is_antisymmetric(self) -> bool:
        return all(
            left == right or (right, left) not in self.relation
            for left, right in self.relation
        )

    @property
    def is_partial_order(self) -> bool:
        return self.is_antisymmetric

    @property
    def is_equality_order(self) -> bool:
        return self.relation == frozenset((value, value) for value in self.elements)

    def principal_upper_set(self, value: Element) -> frozenset[Element]:
        if value not in frozenset(self.elements):
            raise ValueError("the requested point is outside the preorder carrier")
        return frozenset(
            target for source, target in self.relation if source == value
        )

    @property
    def upper_topology(self) -> Topology:
        """Return all upper sets of the preorder."""

        opens: set[frozenset[Element]] = set()
        size = len(self.elements)
        for mask in range(1 << size):
            candidate = frozenset(
                self.elements[index] for index in range(size) if mask & (1 << index)
            )
            if all(
                left not in candidate or right in candidate
                for left, right in self.relation
            ):
                opens.add(candidate)
        return frozenset(opens)

    @property
    def equivalence_classes(self) -> tuple[frozenset[Element], ...]:
        """Return classes for x ~ y iff x <= y and y <= x."""

        unseen = set(self.elements)
        classes: list[frozenset[Element]] = []
        for value in self.elements:
            if value not in unseen:
                continue
            equivalence_class = frozenset(
                other
                for other in self.elements
                if (value, other) in self.relation
                and (other, value) in self.relation
            )
            classes.append(equivalence_class)
            unseen.difference_update(equivalence_class)
        return tuple(classes)

    def kolmogorov_quotient(self) -> "FinitePreorder":
        """Return the antisymmetric quotient by topological indistinguishability."""

        classes = self.equivalence_classes
        class_of = {
            value: equivalence_class
            for equivalence_class in classes
            for value in equivalence_class
        }
        quotient_relation = frozenset(
            (class_of[left], class_of[right]) for left, right in self.relation
        )
        return FinitePreorder(classes, quotient_relation, classes)


def is_topology(
    elements: tuple[Element, ...],
    topology: Topology,
) -> bool:
    """Return whether ``topology`` is a topology on the finite carrier."""

    carrier_tuple = tuple(elements)
    if len(set(carrier_tuple)) != len(carrier_tuple):
        return False
    carrier = frozenset(carrier_tuple)
    opens = frozenset(frozenset(open_set) for open_set in topology)
    if any(not open_set.issubset(carrier) for open_set in opens):
        return False
    if frozenset() not in opens or carrier not in opens:
        return False
    return all(
        left.union(right) in opens and left.intersection(right) in opens
        for left in opens
        for right in opens
    )


def specialization_relation(
    elements: tuple[Element, ...],
    topology: Topology,
) -> Relation:
    """Return x <= y iff every open neighborhood of x contains y."""

    normalized = frozenset(frozenset(open_set) for open_set in topology)
    if not is_topology(elements, normalized):
        raise ValueError("specialization requires a topology on the stated carrier")
    return frozenset(
        (left, right)
        for left in elements
        for right in elements
        if all(left not in open_set or right in open_set for open_set in normalized)
    )


def is_t0_topology(
    elements: tuple[Element, ...],
    topology: Topology,
) -> bool:
    relation = specialization_relation(elements, topology)
    return all(
        left == right or (right, left) not in relation
        for left, right in relation
    )


def is_t1_topology(
    elements: tuple[Element, ...],
    topology: Topology,
) -> bool:
    normalized = frozenset(frozenset(open_set) for open_set in topology)
    if not is_topology(elements, normalized):
        raise ValueError("T1 separation requires a topology")
    carrier = frozenset(elements)
    return all(carrier - {value} in normalized for value in elements)


def is_discrete_topology(
    elements: tuple[Element, ...],
    topology: Topology,
) -> bool:
    normalized = frozenset(frozenset(open_set) for open_set in topology)
    if not is_topology(elements, normalized):
        raise ValueError("discreteness requires a topology")
    return len(normalized) == 1 << len(elements)


def _normalized_map(
    mapping: Mapping[Element, Element],
    source: FinitePreorder,
    target: FinitePreorder,
) -> Mapping[Element, Element]:
    normalized = dict(mapping)
    if set(normalized) != set(source.elements):
        raise ValueError("a map must be total and defined exactly on the source")
    if any(value not in frozenset(target.elements) for value in normalized.values()):
        raise ValueError("a map value lies outside the target carrier")
    return MappingProxyType(normalized)


def is_monotone(
    mapping: Mapping[Element, Element],
    source: FinitePreorder,
    target: FinitePreorder,
) -> bool:
    normalized = _normalized_map(mapping, source, target)
    return all(
        (normalized[left], normalized[right]) in target.relation
        for left, right in source.relation
    )


def is_continuous(
    mapping: Mapping[Element, Element],
    source: FinitePreorder,
    target: FinitePreorder,
) -> bool:
    """Check continuity between the two upper-set topologies."""

    normalized = _normalized_map(mapping, source, target)
    source_opens = source.upper_topology
    return all(
        frozenset(
            value for value in source.elements if normalized[value] in target_open
        )
        in source_opens
        for target_open in target.upper_topology
    )


def monotonicity_defect(
    mapping: Mapping[Element, Element],
    source: FinitePreorder,
    target: FinitePreorder,
) -> float:
    """Return the exact fraction of source-order pairs violated by a map."""

    normalized = _normalized_map(mapping, source, target)
    violations = sum(
        (normalized[left], normalized[right]) not in target.relation
        for left, right in source.relation
    )
    return violations / len(source.relation)


def label_preserving_bijections(
    source: FinitePreorder,
    target: FinitePreorder,
    *,
    max_alignments: int = 100_000,
) -> Iterator[Mapping[Element, Element]]:
    """Yield all label-preserving carrier bijections."""

    if max_alignments <= 0:
        raise ValueError("max_alignments must be positive")
    if len(source.elements) != len(target.elements):
        return
    if Counter(source.labels) != Counter(target.labels):
        return

    labels: list[Label] = []
    for label in source.labels:
        if label not in labels:
            labels.append(label)

    source_groups = {
        label: tuple(
            value
            for value, value_label in zip(
                source.elements, source.labels, strict=True
            )
            if value_label == label
        )
        for label in labels
    }
    target_groups = {
        label: tuple(
            value
            for value, value_label in zip(
                target.elements, target.labels, strict=True
            )
            if value_label == label
        )
        for label in labels
    }
    alignment_count = 1
    for label in labels:
        alignment_count *= factorial(len(source_groups[label]))
    if alignment_count > max_alignments:
        raise ValueError(
            f"{alignment_count} admissible alignments exceed max_alignments"
        )

    choices = tuple(permutations(target_groups[label]) for label in labels)
    for selected_targets in product(*choices):
        mapping: dict[Element, Element] = {}
        for label, target_group in zip(labels, selected_targets, strict=True):
            mapping.update(zip(source_groups[label], target_group, strict=True))
        yield MappingProxyType(mapping)


def transported_relation(
    state: FinitePreorder,
    alignment: Mapping[Element, Element],
) -> Relation:
    """Transport a relation along a total carrier map."""

    normalized = dict(alignment)
    if set(normalized) != set(state.elements):
        raise ValueError("relation transport requires a total source map")
    return frozenset(
        (normalized[left], normalized[right]) for left, right in state.relation
    )


def order_discrepancy(
    left: FinitePreorder,
    right: FinitePreorder,
    *,
    max_alignments: int = 100_000,
) -> float:
    """Return minimum normalized relation error over label-preserving bijections."""

    best = inf
    denominator = len(left.elements) ** 2
    for alignment in label_preserving_bijections(
        left,
        right,
        max_alignments=max_alignments,
    ):
        transported = transported_relation(left, alignment)
        best = min(
            best,
            len(transported.symmetric_difference(right.relation)) / denominator,
        )
    return best


def are_order_isomorphic(
    left: FinitePreorder,
    right: FinitePreorder,
) -> bool:
    value = order_discrepancy(left, right)
    return isfinite(value) and isclose(value, 0.0, abs_tol=_TOLERANCE)


def containment_prediction_error(
    source: FinitePreorder,
    predicted_target: FinitePreorder,
    true_target: FinitePreorder,
    predicted_map: Mapping[Element, Element],
    *,
    relation_weight: float = 1.0,
    monotonicity_weight: float = 1.0,
    max_alignments: int = 100_000,
) -> float:
    """Combine target-order discrepancy with predicted-map monotonicity defect."""

    weights = (float(relation_weight), float(monotonicity_weight))
    if any(weight <= 0 or not isfinite(weight) for weight in weights):
        raise ValueError("prediction-error weights must be finite and positive")
    relation_error = order_discrepancy(
        predicted_target,
        true_target,
        max_alignments=max_alignments,
    )
    map_defect = monotonicity_defect(
        predicted_map,
        source,
        predicted_target,
    )
    return weights[0] * relation_error + weights[1] * map_defect
