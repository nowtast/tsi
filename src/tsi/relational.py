"""Exact finite reference model for the categorical layer of TSI Paper 2C.

The alignment search is deliberately exhaustive. These routines audit theorem
statements on small fixtures; they are not intended as scalable learning code.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import inf, isclose
from types import MappingProxyType
from typing import Hashable, Iterator, Mapping

from .geometric import FiniteMetricState


Entity = Hashable
Label = Hashable
ObjectType = Hashable
ArrowId = Hashable
Pair = tuple[Entity, Entity]
TypedAlignment = Mapping[ObjectType, Mapping[Entity, Entity]]


@dataclass(frozen=True)
class FiniteRelation:
    """A binary relation with explicit finite source and target carriers."""

    source: tuple[Entity, ...]
    target: tuple[Entity, ...]
    pairs: frozenset[Pair]

    def __post_init__(self) -> None:
        source = tuple(self.source)
        target = tuple(self.target)
        pairs = frozenset(tuple(pair) for pair in self.pairs)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "pairs", pairs)

        if len(set(source)) != len(source) or len(set(target)) != len(target):
            raise ValueError("relation carriers must contain unique entities")
        source_set = set(source)
        target_set = set(target)
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("every relation pair must have two entries")
        if any(left not in source_set or right not in target_set for left, right in pairs):
            raise ValueError("relation pair lies outside its source or target carrier")

    @classmethod
    def identity(cls, carrier: tuple[Entity, ...]) -> "FiniteRelation":
        """Return the diagonal relation on ``carrier``."""

        normalized = tuple(carrier)
        return cls(normalized, normalized, frozenset((value, value) for value in normalized))

    @classmethod
    def graph(
        cls,
        source: tuple[Entity, ...],
        target: tuple[Entity, ...],
        mapping: Mapping[Entity, Entity],
    ) -> "FiniteRelation":
        """Return a function graph after checking its declared domain."""

        if set(mapping) != set(source):
            raise ValueError("graph mapping must be defined on the entire source")
        return cls(
            tuple(source),
            tuple(target),
            frozenset((value, mapping[value]) for value in source),
        )

    def compose(self, before: "FiniteRelation") -> "FiniteRelation":
        """Return ``self o before``."""

        if before.target != self.source:
            raise ValueError("relation carriers do not match for composition")
        return FiniteRelation(
            before.source,
            self.target,
            frozenset(
                (left, right)
                for left, middle in before.pairs
                for candidate, right in self.pairs
                if middle == candidate
            ),
        )

    def converse(self) -> "FiniteRelation":
        """Reverse every pair and swap the source and target."""

        return FiniteRelation(
            self.target,
            self.source,
            frozenset((right, left) for left, right in self.pairs),
        )

    def symmetric_difference_size(self, other: "FiniteRelation") -> int:
        """Return the pairwise Hamming difference on matching carriers."""

        if self.source != other.source or self.target != other.target:
            raise ValueError("symmetric difference requires identical carriers")
        return len(self.pairs.symmetric_difference(other.pairs))


def graph_bijection(relation: FiniteRelation) -> Mapping[Entity, Entity] | None:
    """Recover the bijection represented by a relation, or return ``None``."""

    by_source: dict[Entity, list[Entity]] = {value: [] for value in relation.source}
    for left, right in relation.pairs:
        by_source[left].append(right)
    if any(len(values) != 1 for values in by_source.values()):
        return None
    mapping = {left: values[0] for left, values in by_source.items()}
    if len(set(mapping.values())) != len(mapping):
        return None
    if set(mapping.values()) != set(relation.target):
        return None
    return MappingProxyType(mapping)


def is_relation_isomorphism(
    relation: FiniteRelation,
    inverse: FiniteRelation,
) -> bool:
    """Check the two categorical inverse equations in ``FinRel``."""

    if relation.source != inverse.target or relation.target != inverse.source:
        return False
    return (
        inverse.compose(relation) == FiniteRelation.identity(relation.source)
        and relation.compose(inverse) == FiniteRelation.identity(relation.target)
    )


@dataclass(frozen=True)
class ArrowSpec:
    """A named quiver arrow."""

    name: ArrowId
    source: ObjectType
    target: ObjectType


@dataclass(frozen=True)
class TypedPath:
    """A path encoded in source-to-target arrow order."""

    source: ObjectType
    arrows: tuple[ArrowId, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "arrows", tuple(self.arrows))


@dataclass(frozen=True)
class PathEquation:
    """A declared equation between two parallel paths."""

    left: TypedPath
    right: TypedPath


@dataclass(frozen=True)
class FiniteRelationalSchema:
    """A category presentation by a finite quiver and finite path equations."""

    objects: tuple[ObjectType, ...]
    arrows: tuple[ArrowSpec, ...]
    equations: tuple[PathEquation, ...] = ()

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        arrows = tuple(self.arrows)
        equations = tuple(self.equations)
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "arrows", arrows)
        object.__setattr__(self, "equations", equations)

        if len(set(objects)) != len(objects):
            raise ValueError("schema objects must be unique")
        object_set = set(objects)
        if len({arrow.name for arrow in arrows}) != len(arrows):
            raise ValueError("schema arrow names must be unique")
        if any(arrow.source not in object_set or arrow.target not in object_set for arrow in arrows):
            raise ValueError("schema arrow endpoint is not a declared object")
        for equation in equations:
            left_target = self.path_target(equation.left)
            right_target = self.path_target(equation.right)
            if equation.left.source != equation.right.source or left_target != right_target:
                raise ValueError("schema equations must relate parallel paths")

    def arrow(self, name: ArrowId) -> ArrowSpec:
        """Return the uniquely named arrow."""

        for arrow in self.arrows:
            if arrow.name == name:
                return arrow
        raise KeyError(f"unknown schema arrow: {name!r}")

    def path_target(self, path: TypedPath) -> ObjectType:
        """Validate a path and return its target object."""

        if path.source not in set(self.objects):
            raise ValueError("path source is not a schema object")
        current = path.source
        for name in path.arrows:
            arrow = self.arrow(name)
            if arrow.source != current:
                raise ValueError("path contains noncomposable arrows")
            current = arrow.target
        return current


@dataclass(frozen=True)
class FiniteRelationAssignment:
    """Finite carriers and generator relations on a presented schema."""

    schema: FiniteRelationalSchema
    carriers: Mapping[ObjectType, tuple[Entity, ...]]
    labels: Mapping[ObjectType, tuple[Label, ...]]
    generators: Mapping[ArrowId, FiniteRelation]

    def __post_init__(self) -> None:
        carriers = {name: tuple(values) for name, values in self.carriers.items()}
        labels = {name: tuple(values) for name, values in self.labels.items()}
        generators = dict(self.generators)

        if set(carriers) != set(self.schema.objects):
            raise ValueError("assignment must provide every schema carrier")
        if set(labels) != set(self.schema.objects):
            raise ValueError("assignment must provide labels for every carrier")
        if set(generators) != {arrow.name for arrow in self.schema.arrows}:
            raise ValueError("assignment must provide every generator relation")

        for object_name in self.schema.objects:
            carrier = carriers[object_name]
            if not carrier:
                raise ValueError("relational state carriers must be nonempty")
            if len(set(carrier)) != len(carrier):
                raise ValueError("carrier entities must be unique within each type")
            if len(labels[object_name]) != len(carrier):
                raise ValueError("labels must have one entry per carrier entity")

        for arrow in self.schema.arrows:
            relation = generators[arrow.name]
            if relation.source != carriers[arrow.source] or relation.target != carriers[arrow.target]:
                raise ValueError("generator relation carrier does not match its schema arrow")

        object.__setattr__(self, "carriers", MappingProxyType(carriers))
        object.__setattr__(self, "labels", MappingProxyType(labels))
        object.__setattr__(self, "generators", MappingProxyType(generators))

    def label_map(self, object_name: ObjectType) -> Mapping[Entity, Label]:
        """Return labels indexed by entity for one object type."""

        return MappingProxyType(
            dict(zip(self.carriers[object_name], self.labels[object_name], strict=True))
        )

    def path_relation(self, path: TypedPath) -> FiniteRelation:
        """Evaluate a path by the unique free relational extension."""

        self.schema.path_target(path)
        value = FiniteRelation.identity(self.carriers[path.source])
        for arrow_name in path.arrows:
            value = self.generators[arrow_name].compose(value)
        return value

    def equation_defects(self) -> tuple[float, ...]:
        """Return normalized symmetric differences for declared equations."""

        defects: list[float] = []
        for equation in self.schema.equations:
            left = self.path_relation(equation.left)
            right = self.path_relation(equation.right)
            denominator = len(left.source) * len(left.target)
            defects.append(left.symmetric_difference_size(right) / denominator)
        return tuple(defects)

    @property
    def composition_defect(self) -> float:
        """Return the maximum declared path-equation defect."""

        defects = self.equation_defects()
        return max(defects, default=0.0)

    @property
    def is_functorial(self) -> bool:
        """Whether the free extension descends through every equation."""

        return self.composition_defect == 0.0


def label_preserving_alignments(
    left: FiniteRelationAssignment,
    right: FiniteRelationAssignment,
    *,
    max_alignments: int = 100_000,
) -> Iterator[TypedAlignment]:
    """Enumerate all typed label-preserving carrier bijections."""

    if left.schema != right.schema:
        raise ValueError("states must use the same presented schema")

    choices_by_object: list[tuple[Mapping[Entity, Entity], ...]] = []
    alignment_count = 1
    for object_name in left.schema.objects:
        left_carrier = left.carriers[object_name]
        right_carrier = right.carriers[object_name]
        if len(left_carrier) != len(right_carrier):
            return
        left_labels = left.labels[object_name]
        right_label_map = right.label_map(object_name)
        choices = tuple(
            MappingProxyType(dict(zip(left_carrier, candidate, strict=True)))
            for candidate in permutations(right_carrier)
            if all(
                left_labels[index] == right_label_map[candidate[index]]
                for index in range(len(left_carrier))
            )
        )
        if not choices:
            return
        choices_by_object.append(choices)
        alignment_count *= len(choices)
        if alignment_count > max_alignments:
            raise ValueError(
                "exact typed alignment enumeration is restricted to small states; "
                f"found at least {alignment_count} alignments"
            )

    for choices in product(*choices_by_object):
        yield MappingProxyType(dict(zip(left.schema.objects, choices, strict=True)))


def _transport_generator(
    assignment: FiniteRelationAssignment,
    arrow: ArrowSpec,
    alignment: TypedAlignment,
    target_assignment: FiniteRelationAssignment,
) -> FiniteRelation:
    relation = assignment.generators[arrow.name]
    source_map = alignment[arrow.source]
    target_map = alignment[arrow.target]
    return FiniteRelation(
        target_assignment.carriers[arrow.source],
        target_assignment.carriers[arrow.target],
        frozenset((source_map[left], target_map[right]) for left, right in relation.pairs),
    )


def relational_discrepancy(
    left: FiniteRelationAssignment,
    right: FiniteRelationAssignment,
    *,
    weights: Mapping[ArrowId, float] | None = None,
    max_alignments: int = 100_000,
) -> float:
    """Compute Paper 2C's exact generator discrepancy."""

    if left.schema != right.schema:
        raise ValueError("states must use the same presented schema")
    if not left.is_functorial or not right.is_functorial:
        raise ValueError("relational discrepancy is defined on functorial states")
    if not left.schema.arrows:
        raise ValueError("generator discrepancy requires at least one schema arrow")

    if weights is None:
        uniform = 1.0 / len(left.schema.arrows)
        normalized_weights = {arrow.name: uniform for arrow in left.schema.arrows}
    else:
        normalized_weights = {name: float(value) for name, value in weights.items()}
        if set(normalized_weights) != {arrow.name for arrow in left.schema.arrows}:
            raise ValueError("weights must cover every generator exactly")
        if any(value <= 0 for value in normalized_weights.values()):
            raise ValueError("generator weights must be strictly positive")
        if not isclose(sum(normalized_weights.values()), 1.0):
            raise ValueError("generator weights must sum to one")

    best = inf
    for alignment in label_preserving_alignments(
        left,
        right,
        max_alignments=max_alignments,
    ):
        score = 0.0
        for arrow in left.schema.arrows:
            transported = _transport_generator(left, arrow, alignment, right)
            target = right.generators[arrow.name]
            denominator = len(target.source) * len(target.target)
            score += (
                normalized_weights[arrow.name]
                * transported.symmetric_difference_size(target)
                / denominator
            )
        best = min(best, score)
    return best


def are_naturally_isomorphic(
    left: FiniteRelationAssignment,
    right: FiniteRelationAssignment,
) -> bool:
    """Return whether two small labeled states have discrepancy zero."""

    return relational_discrepancy(left, right) == 0.0


def composition_error_bound(
    relation: FiniteRelation,
    perturbed_relation: FiniteRelation,
    after: FiniteRelation,
    perturbed_after: FiniteRelation,
) -> tuple[int, int]:
    """Return the actual composite error and Paper 2C's upper bound."""

    if relation.source != perturbed_relation.source or relation.target != perturbed_relation.target:
        raise ValueError("first relation pair must use matching carriers")
    if after.source != perturbed_after.source or after.target != perturbed_after.target:
        raise ValueError("second relation pair must use matching carriers")
    if relation.target != after.source:
        raise ValueError("relations are not composable")

    actual = after.compose(relation).symmetric_difference_size(
        perturbed_after.compose(perturbed_relation)
    )
    bound = (
        len(after.target) * relation.symmetric_difference_size(perturbed_relation)
        + len(relation.source) * after.symmetric_difference_size(perturbed_after)
    )
    return actual, bound


def face_inclusion_relation(
    simplices: tuple[frozenset[Entity], ...],
) -> FiniteRelation:
    """Return the thin inclusion relation on nonempty faces of a complex."""

    faces = tuple(frozenset(simplex) for simplex in simplices)
    if not faces or any(not face for face in faces):
        raise ValueError("face category requires nonempty simplices")
    if len(set(faces)) != len(faces):
        raise ValueError("simplices must be unique")
    face_set = set(faces)
    for face in faces:
        values = tuple(face)
        for size in range(1, len(values) + 1):
            for subset in combinations(values, size):
                if frozenset(subset) not in face_set:
                    raise ValueError("simplices are not closed under nonempty faces")
    return FiniteRelation(
        faces,
        faces,
        frozenset((left, right) for left in faces for right in faces if left <= right),
    )


def metric_threshold_relation(
    state: FiniteMetricState,
    threshold: float,
) -> FiniteRelation:
    """Return the relation of pairs at distance at most ``threshold``."""

    if threshold < 0:
        raise ValueError("metric threshold must be nonnegative")
    return FiniteRelation(
        state.entities,
        state.entities,
        frozenset(
            (state.entities[i], state.entities[j])
            for i in range(len(state.entities))
            for j in range(len(state.entities))
            if state.distances[i][j] <= threshold
        ),
    )


def threshold_profile_preserved(
    left: FiniteMetricState,
    right: FiniteMetricState,
    mapping: Mapping[Entity, Entity],
) -> bool:
    """Check the finite critical-threshold criterion for a metric isometry."""

    if set(mapping) != set(left.entities):
        return False
    if set(mapping.values()) != set(right.entities) or len(set(mapping.values())) != len(mapping):
        return False
    critical = {
        value
        for matrix in (left.distances, right.distances)
        for row in matrix
        for value in row
    }
    for threshold in critical:
        left_relation = metric_threshold_relation(left, threshold)
        right_relation = metric_threshold_relation(right, threshold)
        transported = frozenset((mapping[x], mapping[y]) for x, y in left_relation.pairs)
        if transported != right_relation.pairs:
            return False
    return True
