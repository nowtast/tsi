"""Exact finite reference model for the dynamical layer of TSI Paper 2D.

The implementation is intentionally exhaustive and small-scale.  It provides
computational witnesses for the finite theorems in the paper; it is not a
training-time dynamics engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import comb, inf, isclose, isfinite
from types import MappingProxyType
from typing import Callable, Hashable, Mapping, Sequence

from .geometric import FiniteMetricState
from .relational import (
    ArrowId,
    Entity,
    FiniteRelationAssignment,
    Label,
    ObjectType,
    TypedAlignment,
    label_preserving_alignments,
)


TaggedEntity = tuple[ObjectType, Entity]
Action = Hashable
ActionWord = tuple[Action, ...]
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class PartialBijection:
    """A bijection between a subset of ``source`` and a subset of ``target``."""

    source: tuple[Hashable, ...]
    target: tuple[Hashable, ...]
    pairs: frozenset[tuple[Hashable, Hashable]]

    def __post_init__(self) -> None:
        source = tuple(self.source)
        target = tuple(self.target)
        pairs = frozenset(tuple(pair) for pair in self.pairs)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "pairs", pairs)

        if len(set(source)) != len(source) or len(set(target)) != len(target):
            raise ValueError("partial-bijection carriers must contain unique values")
        if any(len(pair) != 2 for pair in pairs):
            raise ValueError("every partial-bijection pair must have two entries")
        source_set = set(source)
        target_set = set(target)
        if any(left not in source_set or right not in target_set for left, right in pairs):
            raise ValueError("partial-bijection pair lies outside a declared carrier")

        left_values = [left for left, _ in pairs]
        right_values = [right for _, right in pairs]
        if len(set(left_values)) != len(left_values):
            raise ValueError("a partial bijection must be single-valued")
        if len(set(right_values)) != len(right_values):
            raise ValueError("a partial bijection must be injective")

    @classmethod
    def identity(cls, carrier: tuple[Hashable, ...]) -> "PartialBijection":
        """Return the total identity partial bijection."""

        normalized = tuple(carrier)
        return cls(
            normalized,
            normalized,
            frozenset((value, value) for value in normalized),
        )

    @property
    def mapping(self) -> Mapping[Hashable, Hashable]:
        """Return the represented partial function."""

        return MappingProxyType(dict(self.pairs))

    @property
    def domain(self) -> frozenset[Hashable]:
        return frozenset(left for left, _ in self.pairs)

    @property
    def image(self) -> frozenset[Hashable]:
        return frozenset(right for _, right in self.pairs)

    @property
    def is_total(self) -> bool:
        return self.domain == frozenset(self.source) and self.image == frozenset(self.target)

    def compose(self, before: "PartialBijection") -> "PartialBijection":
        """Return ``self o before``."""

        if before.target != self.source:
            raise ValueError("partial-bijection carriers do not match for composition")
        after_map = self.mapping
        return PartialBijection(
            before.source,
            self.target,
            frozenset(
                (left, after_map[middle])
                for left, middle in before.pairs
                if middle in after_map
            ),
        )

    def inverse(self) -> "PartialBijection":
        """Return the inverse partial bijection between image and domain."""

        return PartialBijection(
            self.target,
            self.source,
            frozenset((right, left) for left, right in self.pairs),
        )

    def graph_difference(self, other: "PartialBijection") -> int:
        """Return graph symmetric-difference size on fixed carriers."""

        if self.source != other.source or self.target != other.target:
            raise ValueError("graph difference requires identical carriers")
        return len(self.pairs.symmetric_difference(other.pairs))


@dataclass(frozen=True)
class IntegratedStructuralState:
    """A finite state carrying relational, topological, and metric structure."""

    relational: FiniteRelationAssignment
    simplices: frozenset[frozenset[TaggedEntity]]
    distances: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if not self.relational.is_functorial:
            raise ValueError("integrated states require a functorial relational state")

        simplices = frozenset(frozenset(simplex) for simplex in self.simplices)
        distances = tuple(tuple(float(value) for value in row) for row in self.distances)
        object.__setattr__(self, "simplices", simplices)
        object.__setattr__(self, "distances", distances)

        vertices = frozenset(self.tagged_entities)
        if frozenset() not in simplices:
            raise ValueError("a simplicial complex must contain the empty simplex")
        if any(not simplex.issubset(vertices) for simplex in simplices):
            raise ValueError("a simplex contains a vertex outside the tagged carrier")
        if any(frozenset((vertex,)) not in simplices for vertex in vertices):
            raise ValueError("every tagged carrier element must be a complex vertex")

        for simplex in simplices:
            ordered = tuple(simplex)
            for size in range(len(ordered) + 1):
                for face in combinations(ordered, size):
                    if frozenset(face) not in simplices:
                        raise ValueError("simplices must be downward closed")

        FiniteMetricState(
            self.tagged_entities,
            distances,
            self.tagged_labels,
        )

    @property
    def tagged_entities(self) -> tuple[TaggedEntity, ...]:
        return tuple(
            (object_name, entity)
            for object_name in self.relational.schema.objects
            for entity in self.relational.carriers[object_name]
        )

    @property
    def tagged_labels(self) -> tuple[tuple[ObjectType, Label], ...]:
        return tuple(
            (object_name, label)
            for object_name in self.relational.schema.objects
            for label in self.relational.labels[object_name]
        )

    @property
    def distance_index(self) -> Mapping[TaggedEntity, int]:
        return MappingProxyType(
            {entity: index for index, entity in enumerate(self.tagged_entities)}
        )

    def induced_simplices(
        self,
        vertices: frozenset[TaggedEntity],
    ) -> frozenset[frozenset[TaggedEntity]]:
        """Return the subcomplex induced by ``vertices``."""

        if not vertices.issubset(frozenset(self.tagged_entities)):
            raise ValueError("induced vertices lie outside the state carrier")
        return frozenset(simplex for simplex in self.simplices if simplex.issubset(vertices))


def _typed_tag_map(alignment: TypedAlignment) -> Mapping[TaggedEntity, TaggedEntity]:
    return MappingProxyType(
        {
            (object_name, source): (object_name, target)
            for object_name, component in alignment.items()
            for source, target in component.items()
        }
    )


def _relation_weights(
    state: IntegratedStructuralState,
    weights: Mapping[ArrowId, float] | None,
) -> Mapping[ArrowId, float]:
    arrow_names = tuple(arrow.name for arrow in state.relational.schema.arrows)
    if not arrow_names:
        if weights and len(weights) != 0:
            raise ValueError("relation weights were supplied for an arrow-free schema")
        return MappingProxyType({})

    if weights is None:
        uniform = 1.0 / len(arrow_names)
        return MappingProxyType({name: uniform for name in arrow_names})

    normalized = {name: float(value) for name, value in weights.items()}
    if set(normalized) != set(arrow_names):
        raise ValueError("relation weights must cover every generator exactly")
    if any(value <= 0 or not isfinite(value) for value in normalized.values()):
        raise ValueError("relation weights must be finite and strictly positive")
    if not isclose(sum(normalized.values()), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("relation weights must sum to one")
    return MappingProxyType(normalized)


def integrated_structural_discrepancy(
    left: IntegratedStructuralState,
    right: IntegratedStructuralState,
    *,
    topology_weight: float = 1.0,
    geometry_weight: float = 1.0,
    relation_weight: float = 1.0,
    generator_weights: Mapping[ArrowId, float] | None = None,
    metric_scale: float = 1.0,
    max_alignments: int = 100_000,
) -> float:
    """Compute the exact integrated discrepancy from Paper 2D."""

    if left.relational.schema != right.relational.schema:
        raise ValueError("integrated states must use the same schema")
    layer_weights = (topology_weight, geometry_weight, relation_weight)
    if any(value <= 0 or not isfinite(value) for value in layer_weights):
        raise ValueError("layer weights must be finite and strictly positive")
    relation_weights = _relation_weights(left, generator_weights)
    metric_scale = float(metric_scale)
    if metric_scale <= 0 or not isfinite(metric_scale):
        raise ValueError("metric_scale must be finite and strictly positive")

    left_index = left.distance_index
    right_index = right.distance_index
    vertex_count = len(left.tagged_entities)
    best = inf
    for alignment in label_preserving_alignments(
        left.relational,
        right.relational,
        max_alignments=max_alignments,
    ):
        tag_map = _typed_tag_map(alignment)
        transported_simplices = frozenset(
            frozenset(tag_map[vertex] for vertex in simplex)
            for simplex in left.simplices
        )
        topology_cost = max(
            (
                len(
                    frozenset(
                        simplex
                        for simplex in transported_simplices
                        if len(simplex) == dimension + 1
                    ).symmetric_difference(
                        frozenset(
                            simplex
                            for simplex in right.simplices
                            if len(simplex) == dimension + 1
                        )
                    )
                )
                / comb(vertex_count, dimension + 1)
                for dimension in range(vertex_count)
            ),
            default=0.0,
        )

        raw_geometry_cost = max(
            abs(
                left.distances[left_index[first]][left_index[second]]
                - right.distances[right_index[tag_map[first]]][right_index[tag_map[second]]]
            )
            for first in left.tagged_entities
            for second in left.tagged_entities
        )
        geometry_cost = min(1.0, raw_geometry_cost / metric_scale)

        relational_cost = 0.0
        for arrow in left.relational.schema.arrows:
            source_map = alignment[arrow.source]
            target_map = alignment[arrow.target]
            transported_pairs = frozenset(
                (source_map[source], target_map[target])
                for source, target in left.relational.generators[arrow.name].pairs
            )
            target_relation = right.relational.generators[arrow.name]
            denominator = len(target_relation.source) * len(target_relation.target)
            relational_cost += (
                relation_weights[arrow.name]
                * len(transported_pairs.symmetric_difference(target_relation.pairs))
                / denominator
            )

        score = (
            topology_weight * topology_cost
            + geometry_weight * geometry_cost
            + relation_weight * relational_cost
        )
        best = min(best, score)
    return best


def are_integrated_isomorphic(
    left: IntegratedStructuralState,
    right: IntegratedStructuralState,
) -> bool:
    """Return whether the integrated discrepancy is zero."""

    value = integrated_structural_discrepancy(left, right)
    return isfinite(value) and isclose(value, 0.0, abs_tol=_TOLERANCE)


@dataclass(frozen=True)
class TransitionDefects:
    """Unnormalized defects of one tracked transition."""

    turnover: int
    topological: int
    geometric: float
    relational: int


@dataclass(frozen=True)
class TrackedTransition:
    """A family of typed, label-preserving partial bijections between states."""

    source: IntegratedStructuralState
    target: IntegratedStructuralState
    components: Mapping[ObjectType, PartialBijection]

    def __post_init__(self) -> None:
        if self.source.relational.schema != self.target.relational.schema:
            raise ValueError("tracked transition endpoints must use the same schema")
        components = dict(self.components)
        expected = set(self.source.relational.schema.objects)
        if set(components) != expected:
            raise ValueError("a tracked transition needs one component per schema object")

        for object_name in self.source.relational.schema.objects:
            component = components[object_name]
            if component.source != self.source.relational.carriers[object_name]:
                raise ValueError("tracking source carrier does not match its state")
            if component.target != self.target.relational.carriers[object_name]:
                raise ValueError("tracking target carrier does not match its state")
            source_labels = self.source.relational.label_map(object_name)
            target_labels = self.target.relational.label_map(object_name)
            if any(
                source_labels[left] != target_labels[right]
                for left, right in component.pairs
            ):
                raise ValueError("tracked persistence must preserve labels")

        object.__setattr__(self, "components", MappingProxyType(components))

    @classmethod
    def identity(cls, state: IntegratedStructuralState) -> "TrackedTransition":
        """Return the full identity tracking on a state."""

        return cls(
            state,
            state,
            {
                object_name: PartialBijection.identity(
                    state.relational.carriers[object_name]
                )
                for object_name in state.relational.schema.objects
            },
        )

    @property
    def tagged_mapping(self) -> Mapping[TaggedEntity, TaggedEntity]:
        return MappingProxyType(
            {
                (object_name, left): (object_name, right)
                for object_name, component in self.components.items()
                for left, right in component.pairs
            }
        )

    @property
    def tagged_domain(self) -> frozenset[TaggedEntity]:
        return frozenset(self.tagged_mapping)

    @property
    def tagged_image(self) -> frozenset[TaggedEntity]:
        return frozenset(self.tagged_mapping.values())

    @property
    def is_full(self) -> bool:
        return all(component.is_total for component in self.components.values())

    @property
    def turnover(self) -> int:
        return sum(
            len(component.source)
            - len(component.domain)
            + len(component.target)
            - len(component.image)
            for component in self.components.values()
        )

    def compose(self, before: "TrackedTransition") -> "TrackedTransition":
        """Return ``self o before``."""

        if before.target != self.source:
            raise ValueError("tracked transition endpoints do not match")
        return TrackedTransition(
            before.source,
            self.target,
            {
                object_name: self.components[object_name].compose(
                    before.components[object_name]
                )
                for object_name in self.source.relational.schema.objects
            },
        )

    def inverse(self) -> "TrackedTransition":
        """Reverse all persistence components and exchange endpoint states."""

        return TrackedTransition(
            self.target,
            self.source,
            {
                object_name: component.inverse()
                for object_name, component in self.components.items()
            },
        )

    @property
    def preserves_topology(self) -> bool:
        mapping = self.tagged_mapping
        transported = frozenset(
            frozenset(mapping[vertex] for vertex in simplex)
            for simplex in self.source.induced_simplices(self.tagged_domain)
        )
        return transported == self.target.induced_simplices(self.tagged_image)

    @property
    def preserves_geometry(self) -> bool:
        mapping = self.tagged_mapping
        source_index = self.source.distance_index
        target_index = self.target.distance_index
        return all(
            isclose(
                self.source.distances[source_index[first]][source_index[second]],
                self.target.distances[target_index[mapping[first]]][
                    target_index[mapping[second]]
                ],
                rel_tol=_TOLERANCE,
                abs_tol=_TOLERANCE,
            )
            for first in self.tagged_domain
            for second in self.tagged_domain
        )

    @property
    def preserves_relations(self) -> bool:
        for arrow in self.source.relational.schema.arrows:
            source_component = self.components[arrow.source]
            target_component = self.components[arrow.target]
            source_map = source_component.mapping
            target_map = target_component.mapping
            source_relation = self.source.relational.generators[arrow.name]
            target_relation = self.target.relational.generators[arrow.name]
            transported = frozenset(
                (source_map[left], target_map[right])
                for left, right in source_relation.pairs
                if left in source_component.domain and right in target_component.domain
            )
            restricted_target = frozenset(
                (left, right)
                for left, right in target_relation.pairs
                if left in source_component.image and right in target_component.image
            )
            if transported != restricted_target:
                return False
        return True

    @property
    def preservation_signature(self) -> frozenset[str]:
        preserved = set()
        if self.preserves_topology:
            preserved.add("topology")
        if self.preserves_geometry:
            preserved.add("geometry")
        if self.preserves_relations:
            preserved.add("relation")
        return frozenset(preserved)

    @property
    def is_exact_conservative(self) -> bool:
        return self.is_full and self.preservation_signature == frozenset(
            {"topology", "geometry", "relation"}
        )

    @property
    def defects(self) -> TransitionDefects:
        mapping = self.tagged_mapping
        source_complex = self.source.induced_simplices(self.tagged_domain)
        target_complex = self.target.induced_simplices(self.tagged_image)
        transported_complex = frozenset(
            frozenset(mapping[vertex] for vertex in simplex)
            for simplex in source_complex
        )
        topology_defect = len(transported_complex.symmetric_difference(target_complex))

        source_index = self.source.distance_index
        target_index = self.target.distance_index
        geometry_defect = max(
            (
                abs(
                    self.source.distances[source_index[first]][source_index[second]]
                    - self.target.distances[target_index[mapping[first]]][
                        target_index[mapping[second]]
                    ]
                )
                for first in self.tagged_domain
                for second in self.tagged_domain
            ),
            default=0.0,
        )

        relation_defect = 0
        for arrow in self.source.relational.schema.arrows:
            source_component = self.components[arrow.source]
            target_component = self.components[arrow.target]
            source_map = source_component.mapping
            target_map = target_component.mapping
            source_relation = self.source.relational.generators[arrow.name]
            target_relation = self.target.relational.generators[arrow.name]
            transported = frozenset(
                (source_map[left], target_map[right])
                for left, right in source_relation.pairs
                if left in source_component.domain and right in target_component.domain
            )
            restricted_target = frozenset(
                (left, right)
                for left, right in target_relation.pairs
                if left in source_component.image and right in target_component.image
            )
            relation_defect += len(transported.symmetric_difference(restricted_target))

        return TransitionDefects(
            turnover=self.turnover,
            topological=topology_defect,
            geometric=geometry_defect,
            relational=relation_defect,
        )


def tracking_difference(
    left: TrackedTransition,
    right: TrackedTransition,
) -> int:
    """Return typed graph symmetric-difference size for aligned transitions."""

    if left.source != right.source or left.target != right.target:
        raise ValueError("tracking difference requires identical endpoint states")
    return sum(
        left.components[object_name].graph_difference(right.components[object_name])
        for object_name in left.source.relational.schema.objects
    )


def tracking_composition_error_bound(
    before: TrackedTransition,
    perturbed_before: TrackedTransition,
    after: TrackedTransition,
    perturbed_after: TrackedTransition,
) -> tuple[int, int]:
    """Return actual composite tracking error and its additive upper bound."""

    if before.source != perturbed_before.source or before.target != perturbed_before.target:
        raise ValueError("first transition pair must have matching endpoints")
    if after.source != perturbed_after.source or after.target != perturbed_after.target:
        raise ValueError("second transition pair must have matching endpoints")
    if before.target != after.source:
        raise ValueError("transition pairs are not composable")

    actual = tracking_difference(
        after.compose(before),
        perturbed_after.compose(perturbed_before),
    )
    bound = tracking_difference(before, perturbed_before) + tracking_difference(
        after,
        perturbed_after,
    )
    return actual, bound


def _all_action_words(actions: tuple[Action, ...], horizon: int) -> tuple[ActionWord, ...]:
    return tuple(
        word
        for length in range(horizon + 1)
        for word in product(actions, repeat=length)
    )


@dataclass(frozen=True)
class FiniteActionHistory:
    """A complete finite action-prefix tree and its tracked edge transitions."""

    actions: tuple[Action, ...]
    horizon: int
    states: Mapping[ActionWord, IntegratedStructuralState]
    edges: Mapping[tuple[ActionWord, Action], TrackedTransition]

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions or len(set(actions)) != len(actions):
            raise ValueError("action alphabet must be finite, nonempty, and unique")
        if self.horizon < 0:
            raise ValueError("history horizon must be nonnegative")

        states = {tuple(word): state for word, state in self.states.items()}
        edges = {
            (tuple(word), action): transition
            for (word, action), transition in self.edges.items()
        }
        expected_words = set(_all_action_words(actions, self.horizon))
        if set(states) != expected_words:
            raise ValueError("history must provide exactly every word through the horizon")
        expected_edges = {
            (word, action)
            for word in expected_words
            if len(word) < self.horizon
            for action in actions
        }
        if set(edges) != expected_edges:
            raise ValueError("history must provide exactly every one-step action edge")

        for (word, action), transition in edges.items():
            if transition.source != states[word]:
                raise ValueError("history edge source does not match its prefix state")
            if transition.target != states[word + (action,)]:
                raise ValueError("history edge target does not match its child state")

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "states", MappingProxyType(states))
        object.__setattr__(self, "edges", MappingProxyType(edges))

    def transition(
        self,
        prefix: ActionWord,
        descendant: ActionWord,
    ) -> TrackedTransition:
        """Return the unique composite transition along a prefix extension."""

        prefix = tuple(prefix)
        descendant = tuple(descendant)
        if prefix not in self.states or descendant not in self.states:
            raise KeyError("history word is outside the finite horizon")
        if descendant[: len(prefix)] != prefix:
            raise ValueError("the first word is not a prefix of the second")

        result = TrackedTransition.identity(self.states[prefix])
        current = prefix
        for action in descendant[len(prefix) :]:
            edge = self.edges[(current, action)]
            result = edge.compose(result)
            current = current + (action,)
        return result


def build_action_history(
    initial: IntegratedStructuralState,
    actions: Sequence[Action],
    horizon: int,
    update: Callable[
        [IntegratedStructuralState, Action, ActionWord],
        TrackedTransition,
    ],
) -> FiniteActionHistory:
    """Build the unique history extension generated by one-step updates."""

    action_tuple = tuple(actions)
    if not action_tuple or len(set(action_tuple)) != len(action_tuple):
        raise ValueError("action alphabet must be finite, nonempty, and unique")
    if horizon < 0:
        raise ValueError("history horizon must be nonnegative")

    states: dict[ActionWord, IntegratedStructuralState] = {(): initial}
    edges: dict[tuple[ActionWord, Action], TrackedTransition] = {}
    for depth in range(horizon):
        for word in product(action_tuple, repeat=depth):
            source = states[word]
            for action in action_tuple:
                transition = update(source, action, word)
                if transition.source != source:
                    raise ValueError("update returned a transition from the wrong state")
                child = word + (action,)
                states[child] = transition.target
                edges[(word, action)] = transition
    return FiniteActionHistory(action_tuple, horizon, states, edges)


def rollout_error_bound(
    one_step_errors: Sequence[float],
    lipschitz_constants: Sequence[float],
) -> float:
    """Return the sharp recursive bound ``e_i <= eps_i + L_i e_{i-1}``."""

    if len(one_step_errors) != len(lipschitz_constants):
        raise ValueError("one-step errors and Lipschitz constants must have equal length")
    errors = tuple(float(value) for value in one_step_errors)
    constants = tuple(float(value) for value in lipschitz_constants)
    if any(value < 0 or not isfinite(value) for value in errors + constants):
        raise ValueError("rollout parameters must be finite and nonnegative")

    bound = 0.0
    for error, constant in zip(errors, constants, strict=True):
        bound = error + constant * bound
    return bound


def intervene_actions(
    factual_actions: Sequence[Action],
    interventions: Mapping[int, Action],
) -> tuple[Action, ...]:
    """Replace selected action equations at specified time indices."""

    result = list(factual_actions)
    for index, action in interventions.items():
        if index < 0 or index >= len(result):
            raise IndexError(f"intervention index out of range: {index}")
        result[index] = action
    return tuple(result)


def causal_structural_rollout(
    initial: IntegratedStructuralState,
    actions: Sequence[Action],
    exogenous: Sequence[Hashable],
    update: Callable[
        [IntegratedStructuralState, Action, Hashable, int],
        TrackedTransition,
    ],
) -> tuple[tuple[IntegratedStructuralState, ...], tuple[TrackedTransition, ...]]:
    """Solve a finite acyclic structural update model by forward recursion."""

    if len(actions) != len(exogenous):
        raise ValueError("actions and exogenous inputs must have equal length")
    states = [initial]
    transitions: list[TrackedTransition] = []
    for time, (action, noise) in enumerate(zip(actions, exogenous, strict=True)):
        transition = update(states[-1], action, noise, time)
        if transition.source != states[-1]:
            raise ValueError("causal update returned a transition from the wrong state")
        transitions.append(transition)
        states.append(transition.target)
    return tuple(states), tuple(transitions)


def collision_pairs(
    positions: Mapping[Hashable, Sequence[float]],
    radii: Mapping[Hashable, float],
    *,
    tolerance: float = _TOLERANCE,
) -> frozenset[frozenset[Hashable]]:
    """Return unordered pairs of closed Euclidean balls that intersect."""

    if set(positions) != set(radii):
        raise ValueError("positions and radii must use identical entity sets")
    if tolerance < 0 or not isfinite(tolerance):
        raise ValueError("collision tolerance must be finite and nonnegative")
    normalized_positions = {
        entity: tuple(float(value) for value in coordinate)
        for entity, coordinate in positions.items()
    }
    if not normalized_positions:
        return frozenset()
    dimensions = {len(coordinate) for coordinate in normalized_positions.values()}
    if dimensions == {0} or len(dimensions) != 1:
        raise ValueError("all positions must share one positive dimension")
    if any(
        not isfinite(value)
        for coordinate in normalized_positions.values()
        for value in coordinate
    ):
        raise ValueError("positions must be finite")
    normalized_radii = {entity: float(value) for entity, value in radii.items()}
    if any(value < 0 or not isfinite(value) for value in normalized_radii.values()):
        raise ValueError("radii must be finite and nonnegative")

    result = set()
    entities = tuple(normalized_positions)
    for first_index, first in enumerate(entities):
        for second in entities[first_index + 1 :]:
            squared_distance = sum(
                (left - right) ** 2
                for left, right in zip(
                    normalized_positions[first],
                    normalized_positions[second],
                    strict=True,
                )
            )
            threshold = normalized_radii[first] + normalized_radii[second]
            if squared_distance <= threshold**2 + tolerance:
                result.add(frozenset((first, second)))
    return frozenset(result)
