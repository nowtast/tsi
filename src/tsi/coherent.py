"""Finite reference model for the Stage 2-I0 coherence package.

The routines in this module are exhaustive theorem-audit tools for small
states.  They deliberately favor definitions that expose every alignment and
every compatibility condition over algorithms intended for training at scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import inf, isclose, isfinite
from types import MappingProxyType
from typing import Hashable, Iterator, Mapping

from .dynamical import IntegratedStructuralState, TaggedEntity
from .order_topology import FinitePreorder
from .relational import ArrowId, Entity, ObjectType


_TOLERANCE = 1e-9
_BRIDGE_KINDS = frozenset({"adjacency", "metric_threshold", "order"})
Pair = tuple[Entity, Entity]


@dataclass(frozen=True)
class BridgeSpec:
    """Identify a generator relation with one relation induced by another layer."""

    arrow: ArrowId
    kind: str
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _BRIDGE_KINDS:
            raise ValueError(f"unknown bridge kind: {self.kind!r}")
        if self.kind == "metric_threshold":
            if (
                self.threshold is None
                or not isfinite(float(self.threshold))
                or float(self.threshold) < 0
            ):
                raise ValueError("a metric-threshold bridge needs a finite threshold >= 0")
            object.__setattr__(self, "threshold", float(self.threshold))
        elif self.threshold is not None:
            raise ValueError("only metric-threshold bridges accept a threshold")


@dataclass(frozen=True)
class CoherenceSignature:
    """Fixed comparison scale, positive layer weights, and bridge declarations."""

    metric_scale: float = 1.0
    label_weight: float = 0.2
    simplicial_weight: float = 0.2
    metric_weight: float = 0.2
    relation_weight: float = 0.2
    order_weight: float = 0.2
    bridges: tuple[BridgeSpec, ...] = ()

    def __post_init__(self) -> None:
        scale = float(self.metric_scale)
        weights = tuple(
            float(value)
            for value in (
                self.label_weight,
                self.simplicial_weight,
                self.metric_weight,
                self.relation_weight,
                self.order_weight,
            )
        )
        bridges = tuple(self.bridges)
        if not isfinite(scale) or scale <= 0:
            raise ValueError("metric_scale must be finite and strictly positive")
        if any(not isfinite(value) or value <= 0 for value in weights):
            raise ValueError("all coherence weights must be finite and positive")
        if not isclose(sum(weights), 1.0, abs_tol=_TOLERANCE):
            raise ValueError("coherence weights must sum to one")
        if len({bridge.arrow for bridge in bridges}) != len(bridges):
            raise ValueError("at most one bridge may be declared for each generator")
        object.__setattr__(self, "metric_scale", scale)
        (
            label_weight,
            simplicial_weight,
            metric_weight,
            relation_weight,
            order_weight,
        ) = weights
        object.__setattr__(self, "label_weight", label_weight)
        object.__setattr__(self, "simplicial_weight", simplicial_weight)
        object.__setattr__(self, "metric_weight", metric_weight)
        object.__setattr__(self, "relation_weight", relation_weight)
        object.__setattr__(self, "order_weight", order_weight)
        object.__setattr__(self, "bridges", bridges)


def _validate_signature(
    core: IntegratedStructuralState,
    signature: CoherenceSignature,
) -> None:
    arrow_names = {arrow.name for arrow in core.relational.schema.arrows}
    unknown = {bridge.arrow for bridge in signature.bridges} - arrow_names
    if unknown:
        raise ValueError(f"bridges reference unknown generators: {unknown!r}")


def induced_bridge_relation(
    core: IntegratedStructuralState,
    order: FinitePreorder,
    bridge: BridgeSpec,
) -> frozenset[Pair]:
    """Return the untagged relation induced by a declared layer bridge."""

    arrow = core.relational.schema.arrow(bridge.arrow)
    source = core.relational.carriers[arrow.source]
    target = core.relational.carriers[arrow.target]
    index = core.distance_index

    if bridge.kind == "adjacency":
        return frozenset(
            (left, right)
            for left in source
            for right in target
            if (arrow.source, left) != (arrow.target, right)
            and frozenset(((arrow.source, left), (arrow.target, right)))
            in core.simplices
        )
    if bridge.kind == "metric_threshold":
        assert bridge.threshold is not None
        return frozenset(
            (left, right)
            for left in source
            for right in target
            if core.distances[index[(arrow.source, left)]][
                index[(arrow.target, right)]
            ]
            <= bridge.threshold + _TOLERANCE
        )
    return frozenset(
        (left, right)
        for left in source
        for right in target
        if ((arrow.source, left), (arrow.target, right)) in order.relation
    )


def bridge_defects(
    core: IntegratedStructuralState,
    order: FinitePreorder,
    signature: CoherenceSignature,
) -> Mapping[ArrowId, float]:
    """Return normalized symmetric differences for all declared bridges."""

    _validate_signature(core, signature)
    defects: dict[ArrowId, float] = {}
    for bridge in signature.bridges:
        arrow = core.relational.schema.arrow(bridge.arrow)
        relation = core.relational.generators[bridge.arrow]
        induced = induced_bridge_relation(core, order, bridge)
        denominator = (
            len(core.relational.carriers[arrow.source])
            * len(core.relational.carriers[arrow.target])
        )
        defects[bridge.arrow] = len(relation.pairs.symmetric_difference(induced)) / denominator
    return MappingProxyType(defects)


@dataclass(frozen=True)
class CoherentStructuralState:
    """An integrated state whose declared cross-layer bridges commute exactly."""

    core: IntegratedStructuralState
    order: FinitePreorder
    signature: CoherenceSignature

    def __post_init__(self) -> None:
        _validate_signature(self.core, self.signature)
        if self.order.elements != self.core.tagged_entities:
            raise ValueError("the preorder carrier must equal the tagged core carrier")
        if self.order.labels != self.core.tagged_labels:
            raise ValueError("preorder labels must equal the tagged core labels")
        defects = bridge_defects(self.core, self.order, self.signature)
        if any(not isclose(value, 0.0, abs_tol=_TOLERANCE) for value in defects.values()):
            raise ValueError(f"the state violates declared bridge constraints: {dict(defects)!r}")

    @property
    def schema(self):
        return self.core.relational.schema


@dataclass(frozen=True)
class TypedCorrespondence:
    """A type-indexed family of finite relations covering both carriers."""

    components: tuple[tuple[ObjectType, frozenset[Pair]], ...]

    @classmethod
    def from_mapping(
        cls,
        object_order: tuple[ObjectType, ...],
        components: Mapping[ObjectType, frozenset[Pair]],
    ) -> "TypedCorrespondence":
        if set(components) != set(object_order):
            raise ValueError("a typed correspondence needs one component per object")
        return cls(
            tuple(
                (object_name, frozenset(components[object_name]))
                for object_name in object_order
            )
        )

    @property
    def mapping(self) -> Mapping[ObjectType, frozenset[Pair]]:
        return MappingProxyType(dict(self.components))

    def inverse(self) -> "TypedCorrespondence":
        return TypedCorrespondence(
            tuple(
                (
                    object_name,
                    frozenset((right, left) for left, right in pairs),
                )
                for object_name, pairs in self.components
            )
        )


def _validate_correspondence(
    correspondence: TypedCorrespondence,
    left: CoherentStructuralState,
    right: CoherentStructuralState,
) -> None:
    if left.schema != right.schema:
        raise ValueError("coherent states must use the same schema")
    if left.signature != right.signature:
        raise ValueError("coherent states must use the same signature")
    components = correspondence.mapping
    if tuple(components) != left.schema.objects:
        raise ValueError("correspondence object order does not match the schema")
    for object_name in left.schema.objects:
        left_carrier = frozenset(left.core.relational.carriers[object_name])
        right_carrier = frozenset(right.core.relational.carriers[object_name])
        pairs = components[object_name]
        if not pairs:
            raise ValueError("each correspondence component must be nonempty")
        if any(x not in left_carrier or y not in right_carrier for x, y in pairs):
            raise ValueError("correspondence pair lies outside its typed carriers")
        if frozenset(x for x, _ in pairs) != left_carrier:
            raise ValueError("correspondence does not cover its left carrier")
        if frozenset(y for _, y in pairs) != right_carrier:
            raise ValueError("correspondence does not cover its right carrier")


def _component_correspondences(
    left: tuple[Entity, ...],
    right: tuple[Entity, ...],
) -> tuple[frozenset[Pair], ...]:
    possible = tuple(product(left, right))
    complete_left = frozenset(left)
    complete_right = frozenset(right)
    candidates: list[frozenset[Pair]] = []
    for mask in range(1, 1 << len(possible)):
        pairs = frozenset(
            possible[index]
            for index in range(len(possible))
            if mask & (1 << index)
        )
        if (
            frozenset(x for x, _ in pairs) == complete_left
            and frozenset(y for _, y in pairs) == complete_right
        ):
            candidates.append(pairs)
    return tuple(candidates)


def typed_correspondences(
    left: CoherentStructuralState,
    right: CoherentStructuralState,
    *,
    max_correspondences: int = 100_000,
) -> Iterator[TypedCorrespondence]:
    """Enumerate every typed correspondence between two small finite states."""

    if max_correspondences <= 0:
        raise ValueError("max_correspondences must be positive")
    if left.schema != right.schema:
        raise ValueError("coherent states must use the same schema")
    if left.signature != right.signature:
        raise ValueError("coherent states must use the same signature")

    choices: list[tuple[frozenset[Pair], ...]] = []
    count = 1
    for object_name in left.schema.objects:
        component_choices = _component_correspondences(
            left.core.relational.carriers[object_name],
            right.core.relational.carriers[object_name],
        )
        choices.append(component_choices)
        count *= len(component_choices)
        if count > max_correspondences:
            raise ValueError(
                "exact correspondence enumeration is restricted to small states; "
                f"found at least {count} correspondences"
            )
    for selected in product(*choices):
        yield TypedCorrespondence.from_mapping(
            left.schema.objects,
            dict(zip(left.schema.objects, selected, strict=True)),
        )


def compose_correspondences(
    first: TypedCorrespondence,
    second: TypedCorrespondence,
) -> TypedCorrespondence:
    """Return ``second o first`` using relational composition in every type."""

    if tuple(name for name, _ in first.components) != tuple(
        name for name, _ in second.components
    ):
        raise ValueError("correspondences use different object orders")
    first_map = first.mapping
    second_map = second.mapping
    return TypedCorrespondence(
        tuple(
            (
                object_name,
                frozenset(
                    (left, right)
                    for left, middle in first_map[object_name]
                    for candidate, right in second_map[object_name]
                    if middle == candidate
                ),
            )
            for object_name, _ in first.components
        )
    )


@dataclass(frozen=True)
class CorrespondenceCosts:
    """The five bounded component distortions and their weighted total."""

    label: float
    simplicial: float
    metric: float
    relation: float
    order: float
    total: float


def _tagged_pairs(
    correspondence: TypedCorrespondence,
) -> tuple[tuple[TaggedEntity, TaggedEntity], ...]:
    return tuple(
        ((object_name, left), (object_name, right))
        for object_name, pairs in correspondence.components
        for left, right in pairs
    )


def _simplicial_distortion(
    pairs: tuple[tuple[TaggedEntity, TaggedEntity], ...],
    left: CoherentStructuralState,
    right: CoherentStructuralState,
) -> float:
    """Return the exact binary simplex-membership distortion."""

    for mask in range(1, 1 << len(pairs)):
        left_vertices = frozenset(
            pairs[index][0]
            for index in range(len(pairs))
            if mask & (1 << index)
        )
        right_vertices = frozenset(
            pairs[index][1]
            for index in range(len(pairs))
            if mask & (1 << index)
        )
        if (left_vertices in left.core.simplices) != (
            right_vertices in right.core.simplices
        ):
            return 1.0
    return 0.0


def correspondence_costs(
    correspondence: TypedCorrespondence,
    left: CoherentStructuralState,
    right: CoherentStructuralState,
) -> CorrespondenceCosts:
    """Evaluate every component of the common-correspondence discrepancy."""

    _validate_correspondence(correspondence, left, right)
    tagged_pairs = _tagged_pairs(correspondence)
    left_labels = {
        entity: label
        for entity, label in zip(
            left.core.tagged_entities,
            left.core.tagged_labels,
            strict=True,
        )
    }
    right_labels = {
        entity: label
        for entity, label in zip(
            right.core.tagged_entities,
            right.core.tagged_labels,
            strict=True,
        )
    }
    label_cost = float(
        any(left_labels[x] != right_labels[y] for x, y in tagged_pairs)
    )
    simplicial_cost = _simplicial_distortion(tagged_pairs, left, right)

    left_index = left.core.distance_index
    right_index = right.core.distance_index
    raw_metric = max(
        abs(
            left.core.distances[left_index[x]][left_index[x_prime]]
            - right.core.distances[right_index[y]][right_index[y_prime]]
        )
        for x, y in tagged_pairs
        for x_prime, y_prime in tagged_pairs
    )
    metric_cost = min(1.0, raw_metric / left.signature.metric_scale)

    components = correspondence.mapping
    relation_cost = 0.0
    for arrow in left.schema.arrows:
        left_relation = left.core.relational.generators[arrow.name].pairs
        right_relation = right.core.relational.generators[arrow.name].pairs
        relation_cost = max(
            relation_cost,
            float(
                any(
                    ((x, x_prime) in left_relation)
                    != ((y, y_prime) in right_relation)
                    for x, y in components[arrow.source]
                    for x_prime, y_prime in components[arrow.target]
                )
            ),
        )

    left_order = left.order.relation
    right_order = right.order.relation
    order_cost = float(
        any(
            ((x, x_prime) in left_order) != ((y, y_prime) in right_order)
            for x, y in tagged_pairs
            for x_prime, y_prime in tagged_pairs
        )
    )
    signature = left.signature
    total = (
        signature.label_weight * label_cost
        + signature.simplicial_weight * simplicial_cost
        + signature.metric_weight * metric_cost
        + signature.relation_weight * relation_cost
        + signature.order_weight * order_cost
    )
    return CorrespondenceCosts(
        label=label_cost,
        simplicial=simplicial_cost,
        metric=metric_cost,
        relation=relation_cost,
        order=order_cost,
        total=total,
    )


def coherent_structural_discrepancy(
    left: CoherentStructuralState,
    right: CoherentStructuralState,
    *,
    max_correspondences: int = 100_000,
) -> float:
    """Return the minimum common-correspondence cost."""

    best = inf
    for correspondence in typed_correspondences(
        left,
        right,
        max_correspondences=max_correspondences,
    ):
        best = min(best, correspondence_costs(correspondence, left, right).total)
    return best


def are_coherently_isomorphic(
    left: CoherentStructuralState,
    right: CoherentStructuralState,
) -> bool:
    """Return whether the two coherent states are structurally isomorphic."""

    value = coherent_structural_discrepancy(left, right)
    return isfinite(value) and isclose(value, 0.0, abs_tol=_TOLERANCE)


def structural_recovery_error(
    predicted: CoherentStructuralState,
    target: CoherentStructuralState,
) -> float:
    """Evaluate a decoded state against its latent structural target."""

    return coherent_structural_discrepancy(predicted, target)


def lipschitz_task_error_bound(
    structural_error: float,
    lipschitz_constant: float,
) -> float:
    """Return the certified task-error bound ``L * structural_error``."""

    error = float(structural_error)
    constant = float(lipschitz_constant)
    if not isfinite(error) or error < 0:
        raise ValueError("structural_error must be finite and nonnegative")
    if not isfinite(constant) or constant < 0:
        raise ValueError("lipschitz_constant must be finite and nonnegative")
    return constant * error
