"""Deterministic oracle-transition benchmark for the first Paper 3 gate.

This module implements ``P3-1`` under the frozen ``P3-I0-FIXED-v1``
interface.  It is deliberately a small finite benchmark:

* every source and target is an exact :class:`CoherentStructuralState`;
* local identifiers are fixed within each state;
* cross-time identity is supplied only by an explicit tracking map;
* train, validation, and test sets are split by source state;
* predictions are decoded into a finite family of valid coherent states; and
* evaluation is delegated to the exact Paper 3 interface evaluator.

The linear predictor is a reference implementation for pipeline validation.
It is not evidence that a learned JEPA encoder is superior.  Objective
ablation and trainable-encoder comparisons belong to the later ``P3-2`` gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from itertools import combinations
import json
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Hashable, Mapping, Protocol, Sequence

import numpy as np

from .coherent import (
    BridgeSpec,
    CoherenceSignature,
    CoherentStructuralState,
    bridge_defects,
)
from .dynamical import (
    IntegratedStructuralState,
    PartialBijection,
    TrackedTransition,
)
from .order_topology import FinitePreorder
from .paper3_interface import (
    FROZEN_PAPER3_INTERFACE,
    Paper3Evaluation,
    StructuralTransitionExample,
    evaluate_decoded_prediction,
)
from .relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


ENTITY_TYPE = "entity"
ENTITY_IDS = (0, 1, 2)
BASE_LABELS = ("red", "red", "blue")
SPLIT_NAMES = ("train", "validation", "test")
P3_ORACLE_BENCHMARK_ID = "P3-1-ORACLE-v1"

ORACLE_SCHEMA = FiniteRelationalSchema(
    objects=(ENTITY_TYPE,),
    arrows=(ArrowSpec("adjacent", ENTITY_TYPE, ENTITY_TYPE),),
)
ORACLE_SIGNATURE = CoherenceSignature(
    metric_scale=5.0,
    label_weight=0.2,
    simplicial_weight=0.2,
    metric_weight=0.2,
    relation_weight=0.2,
    order_weight=0.2,
    bridges=(BridgeSpec("adjacent", "adjacency"),),
)


class SyntheticAction(str, Enum):
    """Actions in the finite oracle transition system."""

    HOLD = "hold"
    ROTATE = "rotate"
    DEFORM = "deform"
    COUPLED = "coupled"


ACTION_AGNOSTIC_FIXED_EXACT_UPPER_BOUND = 1.0 / len(SyntheticAction)


@dataclass(frozen=True)
class ActionRule:
    """Modular update rule and explicit tracking shift for one action."""

    label_delta: int
    topology_delta: int
    metric_delta: int
    order_delta: int
    tracking_shift: int


ACTION_RULES: Mapping[SyntheticAction, ActionRule] = MappingProxyType(
    {
        SyntheticAction.HOLD: ActionRule(0, 0, 0, 0, 0),
        SyntheticAction.ROTATE: ActionRule(1, 1, 0, 0, 1),
        SyntheticAction.DEFORM: ActionRule(0, 1, 1, 1, 0),
        SyntheticAction.COUPLED: ActionRule(2, 2, 1, 2, 2),
    }
)


def _validate_ternary_coordinate(name: str, value: int) -> None:
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError(f"{name} must be one of 0, 1, or 2")


@dataclass(frozen=True, order=True)
class SyntheticStateCode:
    """Four independent ternary coordinates identifying an oracle state."""

    label_phase: int
    topology_mode: int
    metric_mode: int
    order_mode: int

    def __post_init__(self) -> None:
        _validate_ternary_coordinate("label_phase", self.label_phase)
        _validate_ternary_coordinate("topology_mode", self.topology_mode)
        _validate_ternary_coordinate("metric_mode", self.metric_mode)
        _validate_ternary_coordinate("order_mode", self.order_mode)

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return a stable serialization tuple."""

        return (
            self.label_phase,
            self.topology_mode,
            self.metric_mode,
            self.order_mode,
        )


def all_state_codes() -> tuple[SyntheticStateCode, ...]:
    """Enumerate the complete finite state family in lexical order."""

    return tuple(
        SyntheticStateCode(label, topology, metric, order)
        for label in range(3)
        for topology in range(3)
        for metric in range(3)
        for order in range(3)
    )


def _topology_edges(mode: int) -> tuple[tuple[int, int], ...]:
    if mode == 0:
        return ((0, 1), (1, 2))
    if mode == 1:
        return ((0, 1), (0, 2), (1, 2))
    return ((0, 2),)


def _metric_coordinates(mode: int) -> tuple[float, float, float]:
    if mode == 0:
        return (0.0, 1.0, 3.0)
    if mode == 1:
        return (0.0, 2.0, 3.0)
    return (0.0, 1.0, 5.0)


def _order_relation(
    tagged_entities: tuple[tuple[Hashable, Hashable], ...],
    mode: int,
) -> frozenset[tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]]]:
    if mode == 0:
        return frozenset((entity, entity) for entity in tagged_entities)

    index = {entity: position for position, entity in enumerate(tagged_entities)}
    if mode == 1:
        return frozenset(
            (left, right)
            for left in tagged_entities
            for right in tagged_entities
            if index[left] <= index[right]
        )
    return frozenset(
        (left, right)
        for left in tagged_entities
        for right in tagged_entities
        if index[left] >= index[right]
    )


def build_oracle_state(code: SyntheticStateCode) -> CoherentStructuralState:
    """Decode one state code into an exact coherent structural state."""

    labels = tuple(
        BASE_LABELS[(identifier - code.label_phase) % len(ENTITY_IDS)]
        for identifier in ENTITY_IDS
    )
    tagged_entities = tuple((ENTITY_TYPE, identifier) for identifier in ENTITY_IDS)

    simplices: set[frozenset[tuple[Hashable, Hashable]]] = {frozenset()}
    simplices.update(frozenset((entity,)) for entity in tagged_entities)
    edges = _topology_edges(code.topology_mode)
    simplices.update(
        frozenset(
            (
                (ENTITY_TYPE, left),
                (ENTITY_TYPE, right),
            )
        )
        for left, right in edges
    )

    coordinates = _metric_coordinates(code.metric_mode)
    distances = tuple(
        tuple(abs(coordinates[left] - coordinates[right]) for right in ENTITY_IDS)
        for left in ENTITY_IDS
    )

    relation_pairs = frozenset(
        (left, right) for edge in edges for left, right in (edge, tuple(reversed(edge)))
    )
    core = IntegratedStructuralState(
        relational=FiniteRelationAssignment(
            schema=ORACLE_SCHEMA,
            carriers={ENTITY_TYPE: ENTITY_IDS},
            labels={ENTITY_TYPE: labels},
            generators={
                "adjacent": FiniteRelation(
                    ENTITY_IDS,
                    ENTITY_IDS,
                    relation_pairs,
                )
            },
        ),
        simplices=frozenset(simplices),
        distances=distances,
    )
    order = FinitePreorder(
        core.tagged_entities,
        _order_relation(core.tagged_entities, code.order_mode),
        core.tagged_labels,
    )
    return CoherentStructuralState(
        core=core,
        order=order,
        signature=ORACLE_SIGNATURE,
    )


def normalize_action(action: SyntheticAction | str) -> SyntheticAction:
    """Normalize an action while rejecting undeclared interventions."""

    try:
        return (
            action if isinstance(action, SyntheticAction) else SyntheticAction(action)
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown synthetic action: {action!r}") from error


def successor_code(
    source: SyntheticStateCode,
    action: SyntheticAction | str,
) -> SyntheticStateCode:
    """Apply the declared modular action rule to a state code."""

    rule = ACTION_RULES[normalize_action(action)]
    return SyntheticStateCode(
        (source.label_phase + rule.label_delta) % 3,
        (source.topology_mode + rule.topology_delta) % 3,
        (source.metric_mode + rule.metric_delta) % 3,
        (source.order_mode + rule.order_delta) % 3,
    )


def build_oracle_tracking(
    source: CoherentStructuralState,
    target: CoherentStructuralState,
    action: SyntheticAction | str,
) -> TrackedTransition:
    """Construct the explicit full tracking map declared by an action."""

    shift = ACTION_RULES[normalize_action(action)].tracking_shift
    pairs = frozenset(
        (identifier, (identifier + shift) % len(ENTITY_IDS))
        for identifier in ENTITY_IDS
    )
    return TrackedTransition(
        source=source.core,
        target=target.core,
        components={
            ENTITY_TYPE: PartialBijection(
                source.core.relational.carriers[ENTITY_TYPE],
                target.core.relational.carriers[ENTITY_TYPE],
                pairs,
            )
        },
    )


@dataclass(frozen=True)
class OracleTransitionCase:
    """One benchmark example together with its finite state codes."""

    split: str
    source_code: SyntheticStateCode
    action: SyntheticAction
    target_code: SyntheticStateCode
    example: StructuralTransitionExample

    def __post_init__(self) -> None:
        if self.split not in SPLIT_NAMES:
            raise ValueError(f"unknown benchmark split: {self.split!r}")
        if successor_code(self.source_code, self.action) != self.target_code:
            raise ValueError("target code does not follow the declared action rule")


@dataclass(frozen=True)
class P3OracleBenchmarkSpec:
    """Reproducibility and optimization parameters for ``P3-1``."""

    seed: int = 20_260_728
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    ridge: float = 1.0e-8
    target_momentum: float = 0.99

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if not isfinite(self.train_fraction) or self.train_fraction <= 0.0:
            raise ValueError("train_fraction must be positive and finite")
        if not isfinite(self.validation_fraction) or self.validation_fraction <= 0.0:
            raise ValueError("validation_fraction must be positive and finite")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must sum to less than 1")
        if not isfinite(self.ridge) or self.ridge <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if not isfinite(self.target_momentum) or not 0.0 <= self.target_momentum < 1.0:
            raise ValueError("target_momentum must lie in [0, 1)")


@dataclass(frozen=True)
class SyntheticTransitionBenchmark:
    """Complete state family and source-disjoint transition splits."""

    spec: P3OracleBenchmarkSpec
    states: Mapping[SyntheticStateCode, CoherentStructuralState]
    splits: Mapping[str, tuple[OracleTransitionCase, ...]]
    digest: str

    def __post_init__(self) -> None:
        if set(self.states) != set(all_state_codes()):
            raise ValueError("benchmark must contain the complete state family")
        if tuple(self.splits) != SPLIT_NAMES:
            raise ValueError("benchmark splits must use the declared stable order")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(
            self,
            "splits",
            MappingProxyType(
                {name: tuple(cases) for name, cases in self.splits.items()}
            ),
        )

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def transition_count(self) -> int:
        return sum(len(cases) for cases in self.splits.values())

    def source_codes(self, split: str) -> frozenset[SyntheticStateCode]:
        return frozenset(case.source_code for case in self.splits[split])


def _state_digest_payload(
    code: SyntheticStateCode,
    state: CoherentStructuralState,
) -> dict[str, object]:
    return {
        "code": code.as_tuple(),
        "labels": list(state.core.relational.labels[ENTITY_TYPE]),
        "simplices": [
            [list(entity) for entity in sorted(simplex, key=repr)]
            for simplex in sorted(
                state.core.simplices,
                key=lambda item: (
                    len(item),
                    tuple(sorted(map(repr, item))),
                ),
            )
        ],
        "distances": [list(row) for row in state.core.distances],
        "relation": [
            list(pair)
            for pair in sorted(state.core.relational.generators["adjacent"].pairs)
        ],
        "order": [
            [list(left), list(right)]
            for left, right in sorted(state.order.relation, key=repr)
        ],
        "signature": {
            "metric_scale": state.signature.metric_scale,
            "weights": [
                state.signature.label_weight,
                state.signature.simplicial_weight,
                state.signature.metric_weight,
                state.signature.relation_weight,
                state.signature.order_weight,
            ],
            "bridges": [
                [bridge.arrow, bridge.kind, bridge.threshold]
                for bridge in state.signature.bridges
            ],
        },
    }


def _benchmark_digest(
    spec: P3OracleBenchmarkSpec,
    states: Mapping[SyntheticStateCode, CoherentStructuralState],
    splits: Mapping[str, tuple[OracleTransitionCase, ...]],
) -> str:
    payload = {
        "benchmark_id": P3_ORACLE_BENCHMARK_ID,
        "split_spec": {
            "seed": spec.seed,
            "train_fraction": spec.train_fraction,
            "validation_fraction": spec.validation_fraction,
        },
        "states": [
            _state_digest_payload(code, states[code]) for code in sorted(states)
        ],
        "transitions": [
            {
                "split": split,
                "source": case.source_code.as_tuple(),
                "action": case.action.value,
                "target": case.target_code.as_tuple(),
                "tracking": sorted(
                    list(pair)
                    for pair in case.example.tracking.components[ENTITY_TYPE].pairs
                ),
            }
            for split in SPLIT_NAMES
            for case in splits[split]
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def build_oracle_transition_benchmark(
    spec: P3OracleBenchmarkSpec | None = None,
) -> SyntheticTransitionBenchmark:
    """Build all transitions and split them without source-state leakage."""

    resolved = spec or P3OracleBenchmarkSpec()
    codes = list(all_state_codes())
    shuffled = list(codes)
    Random(resolved.seed).shuffle(shuffled)

    train_count = round(len(shuffled) * resolved.train_fraction)
    validation_count = round(len(shuffled) * resolved.validation_fraction)
    if min(train_count, validation_count) <= 0:
        raise ValueError("benchmark fractions yield an empty training split")
    if train_count + validation_count >= len(shuffled):
        raise ValueError("benchmark fractions yield an empty test split")

    split_codes = {
        "train": frozenset(shuffled[:train_count]),
        "validation": frozenset(shuffled[train_count : train_count + validation_count]),
        "test": frozenset(shuffled[train_count + validation_count :]),
    }
    states = {code: build_oracle_state(code) for code in codes}
    splits: dict[str, tuple[OracleTransitionCase, ...]] = {}
    action_order = tuple(SyntheticAction)
    for split in SPLIT_NAMES:
        cases: list[OracleTransitionCase] = []
        for source_code in sorted(split_codes[split]):
            source = states[source_code]
            for action in action_order:
                target_code = successor_code(source_code, action)
                target = states[target_code]
                tracking = build_oracle_tracking(source, target, action)
                cases.append(
                    OracleTransitionCase(
                        split=split,
                        source_code=source_code,
                        action=action,
                        target_code=target_code,
                        example=StructuralTransitionExample(
                            source=source,
                            action=action,
                            target=target,
                            tracking=tracking,
                        ),
                    )
                )
        splits[split] = tuple(cases)

    return SyntheticTransitionBenchmark(
        spec=resolved,
        states=states,
        splits=splits,
        digest=_benchmark_digest(resolved, states, splits),
    )


def _fixed_carrier_signature(
    state: CoherentStructuralState,
) -> tuple[tuple[Hashable, tuple[Hashable, ...]], ...]:
    return tuple(
        (object_name, state.core.relational.carriers[object_name])
        for object_name in state.schema.objects
    )


@dataclass(frozen=True)
class StructuralFeatureLayout:
    """Injective exact-layer feature layout for the finite state codebook."""

    schema: FiniteRelationalSchema
    signature: CoherenceSignature
    carriers: tuple[tuple[Hashable, tuple[Hashable, ...]], ...]
    tagged_entities: tuple[tuple[Hashable, Hashable], ...]
    label_vocabulary: tuple[tuple[Hashable, Hashable], ...]
    simplex_probes: tuple[frozenset[tuple[Hashable, Hashable]], ...]
    metric_pairs: tuple[
        tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]], ...
    ]
    relation_cells: tuple[tuple[Hashable, Hashable, Hashable], ...]
    order_cells: tuple[tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]], ...]

    @classmethod
    def from_states(
        cls,
        states: Sequence[CoherentStructuralState],
    ) -> StructuralFeatureLayout:
        if not states:
            raise ValueError("at least one state is required")
        reference = states[0]
        carriers = _fixed_carrier_signature(reference)
        for state in states[1:]:
            if state.schema != reference.schema:
                raise ValueError("all feature states must share one schema")
            if state.signature != reference.signature:
                raise ValueError("all feature states must share one signature")
            if _fixed_carrier_signature(state) != carriers:
                raise ValueError("all feature states must share fixed carriers")

        tagged = reference.core.tagged_entities
        vocabulary = tuple(
            sorted(
                {label for state in states for label in state.core.tagged_labels},
                key=repr,
            )
        )
        simplex_probes = tuple(
            frozenset(subset)
            for size in range(2, len(tagged) + 1)
            for subset in combinations(tagged, size)
        )
        metric_pairs = tuple(combinations(tagged, 2))
        relation_cells = tuple(
            (arrow.name, left, right)
            for arrow in reference.schema.arrows
            for left in reference.core.relational.carriers[arrow.source]
            for right in reference.core.relational.carriers[arrow.target]
        )
        order_cells = tuple((left, right) for left in tagged for right in tagged)
        return cls(
            schema=reference.schema,
            signature=reference.signature,
            carriers=carriers,
            tagged_entities=tagged,
            label_vocabulary=vocabulary,
            simplex_probes=simplex_probes,
            metric_pairs=metric_pairs,
            relation_cells=relation_cells,
            order_cells=order_cells,
        )

    @property
    def dimension(self) -> int:
        """Number of coordinates in the exact structural representation."""

        return (
            len(self.tagged_entities) * len(self.label_vocabulary)
            + len(self.simplex_probes)
            + len(self.metric_pairs)
            + len(self.relation_cells)
            + len(self.order_cells)
        )

    def _validate_state(self, state: CoherentStructuralState) -> None:
        if state.schema != self.schema:
            raise ValueError("feature state has the wrong schema")
        if state.signature != self.signature:
            raise ValueError("feature state has the wrong signature")
        if _fixed_carrier_signature(state) != self.carriers:
            raise ValueError("feature state has the wrong fixed carriers")

    def encode(self, state: CoherentStructuralState) -> np.ndarray:
        """Encode every frozen exact layer in a deterministic coordinate order."""

        self._validate_state(state)
        values: list[float] = []
        labels = dict(
            zip(
                state.core.tagged_entities,
                state.core.tagged_labels,
                strict=True,
            )
        )
        for entity in self.tagged_entities:
            values.extend(
                1.0 if labels[entity] == label else 0.0
                for label in self.label_vocabulary
            )

        values.extend(
            1.0 if simplex in state.core.simplices else 0.0
            for simplex in self.simplex_probes
        )
        for left, right in self.metric_pairs:
            index = state.core.distance_index
            values.append(
                state.core.distances[index[left]][index[right]]
                / self.signature.metric_scale
            )

        values.extend(
            1.0
            if (left, right) in state.core.relational.generators[arrow].pairs
            else 0.0
            for arrow, left, right in self.relation_cells
        )
        values.extend(
            1.0 if cell in state.order.relation else 0.0 for cell in self.order_cells
        )
        result = np.asarray(values, dtype=np.float64)
        if result.shape != (self.dimension,):
            raise RuntimeError("structural feature layout has inconsistent dimension")
        return result

    def encode_many(
        self,
        states: Sequence[CoherentStructuralState],
    ) -> np.ndarray:
        """Encode a nonempty state sequence as a matrix."""

        if not states:
            raise ValueError("at least one state is required")
        return np.stack([self.encode(state) for state in states], axis=0)


@dataclass(frozen=True)
class DecodedTransitionPrediction:
    """A valid decoded target, explicit tracking, and optional latent estimate."""

    target: CoherentStructuralState
    tracking: TrackedTransition
    latent: np.ndarray | None


class TransitionPredictor(Protocol):
    """Minimal predictor protocol used by the benchmark evaluator."""

    def predict(
        self,
        source: CoherentStructuralState,
        action: SyntheticAction | str,
    ) -> DecodedTransitionPrediction:
        """Predict a valid target and explicit tracking map."""


def _tracking_graph(case: OracleTransitionCase) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(case.example.tracking.components[ENTITY_TYPE].pairs))


class LinearStructuralJEPA:
    """Action-conditioned ridge predictor over a noncollapsed exact encoder.

    The context and target encoders share a standardized structural layout.
    The target copy is stop-gradient and initialized by an EMA-consistent copy
    of the context statistics.  Only the linear latent predictor is fitted.
    This restricted construction isolates the transition/evaluation pipeline;
    ``P3-2`` must test learned encoders and the nine objective families.
    """

    def __init__(
        self,
        *,
        ridge: float = 1.0e-8,
        condition_on_action: bool = True,
        target_momentum: float = 0.99,
    ) -> None:
        if not isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be positive and finite")
        if not isfinite(target_momentum) or not 0.0 <= target_momentum < 1.0:
            raise ValueError("target_momentum must lie in [0, 1)")
        self.ridge = float(ridge)
        self.condition_on_action = condition_on_action
        self.target_momentum = float(target_momentum)
        self._fitted = False

    def _key(self, action: SyntheticAction | str) -> str:
        normalized = normalize_action(action)
        return normalized.value if self.condition_on_action else "__pooled__"

    def fit(
        self,
        benchmark: SyntheticTransitionBenchmark,
    ) -> LinearStructuralJEPA:
        """Fit encoder statistics, latent predictors, and tracking templates."""

        training = benchmark.splits["train"]
        if not training:
            raise ValueError("training split must be nonempty")
        states = tuple(benchmark.states[code] for code in sorted(benchmark.states))
        layout = StructuralFeatureLayout.from_states(states)
        training_states = tuple(
            state
            for case in training
            for state in (case.example.source, case.example.target)
        )
        matrix = layout.encode_many(training_states)
        mean = np.mean(matrix, axis=0)
        standard_deviation = np.std(matrix, axis=0)
        active = np.flatnonzero(standard_deviation > 1.0e-12)
        if active.size == 0:
            raise ValueError("training features are completely collapsed")
        scale = standard_deviation.copy()
        scale[scale <= 1.0e-12] = 1.0

        self.layout = layout
        self.context_mean = mean
        self.context_scale = scale
        self.active_coordinates = active
        # Equal initialization is an EMA fixed point.  The target copy remains
        # stop-gradient while only the predictor coefficients are optimized.
        self.target_mean = (
            self.target_momentum * mean + (1.0 - self.target_momentum) * mean
        )
        self.target_scale = (
            self.target_momentum * scale + (1.0 - self.target_momentum) * scale
        )
        self.target_ema_updates = 1

        candidate_codes = tuple(sorted(benchmark.states))
        candidate_matrix = np.stack(
            [
                self._encode_target_unchecked(benchmark.states[code])
                for code in candidate_codes
            ],
            axis=0,
        )
        unique_rows = np.unique(np.round(candidate_matrix, decimals=12), axis=0)
        if unique_rows.shape[0] != len(candidate_codes):
            raise ValueError("target encoder does not separate the state codebook")

        grouped: dict[str, list[OracleTransitionCase]] = {}
        for case in training:
            grouped.setdefault(self._key(case.action), []).append(case)

        coefficients: dict[str, np.ndarray] = {}
        tracking_templates: dict[str, tuple[tuple[int, int], ...]] = {}
        for key, cases in grouped.items():
            source_latent = np.stack(
                [self._encode_context_unchecked(case.example.source) for case in cases],
                axis=0,
            )
            target_latent = np.stack(
                [self._encode_target_unchecked(case.example.target) for case in cases],
                axis=0,
            )
            design = np.concatenate(
                (source_latent, np.ones((len(cases), 1), dtype=np.float64)),
                axis=1,
            )
            penalty = np.eye(design.shape[1], dtype=np.float64)
            penalty[-1, -1] = 0.0
            coefficients[key] = np.linalg.solve(
                design.T @ design + self.ridge * penalty,
                design.T @ target_latent,
            )
            counts = Counter(_tracking_graph(case) for case in cases)
            tracking_templates[key] = min(
                counts,
                key=lambda graph: (-counts[graph], graph),
            )

        expected_keys = (
            {action.value for action in SyntheticAction}
            if self.condition_on_action
            else {"__pooled__"}
        )
        if set(coefficients) != expected_keys:
            raise ValueError("training split does not cover every predictor key")

        self.candidate_codes = candidate_codes
        self.candidate_states = MappingProxyType(dict(benchmark.states))
        self.candidate_latent = candidate_matrix
        self.coefficients = MappingProxyType(coefficients)
        self.tracking_templates = MappingProxyType(tracking_templates)
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("predictor must be fitted before use")

    def _encode_context_unchecked(
        self,
        state: CoherentStructuralState,
    ) -> np.ndarray:
        features = self.layout.encode(state)
        standardized = (features - self.context_mean) / self.context_scale
        return standardized[self.active_coordinates]

    def _encode_target_unchecked(
        self,
        state: CoherentStructuralState,
    ) -> np.ndarray:
        features = self.layout.encode(state)
        standardized = (features - self.target_mean) / self.target_scale
        return standardized[self.active_coordinates]

    def context_embedding(
        self,
        state: CoherentStructuralState,
    ) -> np.ndarray:
        """Return the fitted context representation."""

        self._require_fitted()
        return self._encode_context_unchecked(state)

    def target_embedding(
        self,
        state: CoherentStructuralState,
    ) -> np.ndarray:
        """Return the frozen stop-gradient target representation."""

        self._require_fitted()
        return self._encode_target_unchecked(state)

    @property
    def latent_dimension(self) -> int:
        self._require_fitted()
        return int(self.active_coordinates.size)

    def predict(
        self,
        source: CoherentStructuralState,
        action: SyntheticAction | str,
    ) -> DecodedTransitionPrediction:
        """Predict a nearest valid state and a label-preserving tracking map."""

        self._require_fitted()
        normalized = normalize_action(action)
        key = self._key(normalized)
        context = self._encode_context_unchecked(source)
        design = np.concatenate((context, np.ones(1, dtype=np.float64)))
        latent = design @ self.coefficients[key]
        squared_distances = np.sum(
            (self.candidate_latent - latent[None, :]) ** 2,
            axis=1,
        )
        candidate_index = int(np.argmin(squared_distances))
        target = self.candidate_states[self.candidate_codes[candidate_index]]

        source_labels = source.core.relational.label_map(ENTITY_TYPE)
        target_labels = target.core.relational.label_map(ENTITY_TYPE)
        projected_pairs = frozenset(
            (left, right)
            for left, right in self.tracking_templates[key]
            if source_labels[left] == target_labels[right]
        )
        tracking = TrackedTransition(
            source=source.core,
            target=target.core,
            components={
                ENTITY_TYPE: PartialBijection(
                    source.core.relational.carriers[ENTITY_TYPE],
                    target.core.relational.carriers[ENTITY_TYPE],
                    projected_pairs,
                )
            },
        )
        return DecodedTransitionPrediction(
            target=target,
            tracking=tracking,
            latent=latent,
        )


class IdentityTransitionBaseline:
    """Action-insensitive baseline that copies the exact source state."""

    def predict(
        self,
        source: CoherentStructuralState,
        action: SyntheticAction | str,
    ) -> DecodedTransitionPrediction:
        normalize_action(action)
        tracking = TrackedTransition.identity(source.core)
        return DecodedTransitionPrediction(
            target=source,
            tracking=tracking,
            latent=None,
        )


@dataclass(frozen=True)
class SplitEvaluation:
    """Aggregated exact and latent metrics for one benchmark split."""

    example_count: int
    state_isomorphic_rate: float
    fixed_exact_rate: float
    tracking_exact_rate: float
    quotient_joint_exact_rate: float
    fixed_joint_exact_rate: float
    mean_quotient_distance: float
    mean_fixed_total: float
    mean_label_error: float
    mean_simplicial_error: float
    mean_metric_error: float
    mean_relation_error: float
    mean_order_error: float
    mean_tracking_error: float
    mean_latent_error: float | None
    bridge_violation_rate: float

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "example_count": self.example_count,
            "state_isomorphic_rate": self.state_isomorphic_rate,
            "fixed_exact_rate": self.fixed_exact_rate,
            "tracking_exact_rate": self.tracking_exact_rate,
            "quotient_joint_exact_rate": self.quotient_joint_exact_rate,
            "fixed_joint_exact_rate": self.fixed_joint_exact_rate,
            "mean_quotient_distance": self.mean_quotient_distance,
            "mean_fixed_total": self.mean_fixed_total,
            "mean_label_error": self.mean_label_error,
            "mean_simplicial_error": self.mean_simplicial_error,
            "mean_metric_error": self.mean_metric_error,
            "mean_relation_error": self.mean_relation_error,
            "mean_order_error": self.mean_order_error,
            "mean_tracking_error": self.mean_tracking_error,
            "mean_latent_error": self.mean_latent_error,
            "bridge_violation_rate": self.bridge_violation_rate,
        }


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric sequence")
    return float(sum(values) / len(values))


def evaluate_transition_predictor(
    predictor: TransitionPredictor,
    benchmark: SyntheticTransitionBenchmark,
    split: str,
) -> SplitEvaluation:
    """Evaluate one predictor using exact state and tracking metrics."""

    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown benchmark split: {split!r}")
    cases = benchmark.splits[split]
    if not cases:
        raise ValueError("evaluation split must be nonempty")

    evaluations: list[Paper3Evaluation] = []
    latent_errors: list[float] = []
    bridge_violations: list[float] = []
    fixed_joint: list[float] = []
    for case in cases:
        prediction = predictor.predict(case.example.source, case.action)
        latent_error: float | None = None
        target_embedding = getattr(predictor, "target_embedding", None)
        if prediction.latent is not None and callable(target_embedding):
            difference = prediction.latent - target_embedding(case.example.target)
            latent_error = float(np.sqrt(np.mean(difference**2)))
            latent_errors.append(latent_error)
        evaluation = evaluate_decoded_prediction(
            case.example,
            prediction.target,
            prediction.tracking,
            latent_prediction_error=latent_error,
        )
        evaluations.append(evaluation)
        fixed_joint.append(
            1.0
            if evaluation.fixed_carrier.is_zero and evaluation.tracking == 0.0
            else 0.0
        )
        defects = bridge_defects(
            prediction.target.core,
            prediction.target.order,
            prediction.target.signature,
        )
        bridge_violations.append(1.0 if any(defects.values()) else 0.0)

    return SplitEvaluation(
        example_count=len(evaluations),
        state_isomorphic_rate=_mean(
            [1.0 if result.state_isomorphic else 0.0 for result in evaluations]
        ),
        fixed_exact_rate=_mean(
            [1.0 if result.fixed_carrier.is_zero else 0.0 for result in evaluations]
        ),
        tracking_exact_rate=_mean(
            [1.0 if result.tracking == 0.0 else 0.0 for result in evaluations]
        ),
        quotient_joint_exact_rate=_mean(
            [1.0 if result.jointly_exact else 0.0 for result in evaluations]
        ),
        fixed_joint_exact_rate=_mean(fixed_joint),
        mean_quotient_distance=_mean([result.quotient.total for result in evaluations]),
        mean_fixed_total=_mean([result.fixed_carrier.total for result in evaluations]),
        mean_label_error=_mean([result.fixed_carrier.label for result in evaluations]),
        mean_simplicial_error=_mean(
            [result.fixed_carrier.simplicial for result in evaluations]
        ),
        mean_metric_error=_mean(
            [result.fixed_carrier.metric for result in evaluations]
        ),
        mean_relation_error=_mean(
            [result.fixed_carrier.relation for result in evaluations]
        ),
        mean_order_error=_mean([result.fixed_carrier.order for result in evaluations]),
        mean_tracking_error=_mean([result.tracking for result in evaluations]),
        mean_latent_error=_mean(latent_errors) if latent_errors else None,
        bridge_violation_rate=_mean(bridge_violations),
    )


@dataclass(frozen=True)
class P3OracleBenchmarkReport:
    """Machine-readable audit record for the ``P3-1`` gate."""

    interface_id: str
    benchmark_id: str
    benchmark_digest: str
    seed: int
    state_count: int
    transition_count: int
    split_source_counts: Mapping[str, int]
    feature_dimension: int
    latent_dimension: int
    models: Mapping[str, Mapping[str, SplitEvaluation]]
    action_conditioning_gain: float
    action_agnostic_fixed_exact_upper_bound: float
    audit_errors: tuple[str, ...]
    method_contract: Mapping[str, str]
    claim_status: str = "empirical"
    gate: str = "P3-1"

    @property
    def passed(self) -> bool:
        return not self.audit_errors

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "passed": self.passed,
            "claim_status": self.claim_status,
            "interface_id": self.interface_id,
            "benchmark_id": self.benchmark_id,
            "benchmark_digest": self.benchmark_digest,
            "seed": self.seed,
            "state_count": self.state_count,
            "transition_count": self.transition_count,
            "split_source_counts": dict(self.split_source_counts),
            "feature_dimension": self.feature_dimension,
            "latent_dimension": self.latent_dimension,
            "models": {
                model: {
                    split: evaluation.as_dict()
                    for split, evaluation in split_results.items()
                }
                for model, split_results in self.models.items()
            },
            "action_conditioning_gain": self.action_conditioning_gain,
            "action_agnostic_fixed_exact_upper_bound": (
                self.action_agnostic_fixed_exact_upper_bound
            ),
            "audit_errors": list(self.audit_errors),
            "method_contract": dict(self.method_contract),
        }


def _audit_benchmark(
    benchmark: SyntheticTransitionBenchmark,
    models: Mapping[str, Mapping[str, SplitEvaluation]],
) -> tuple[str, ...]:
    errors: list[str] = []
    source_sets = {split: benchmark.source_codes(split) for split in SPLIT_NAMES}
    for left, right in combinations(SPLIT_NAMES, 2):
        if source_sets[left] & source_sets[right]:
            errors.append(f"source-state leakage between {left} and {right}")
    if benchmark.state_count != 81:
        errors.append("finite state family must contain exactly 81 states")
    if benchmark.transition_count != 324:
        errors.append("benchmark must contain exactly 324 transitions")
    for code in all_state_codes():
        targets = {successor_code(code, action) for action in SyntheticAction}
        if len(targets) != len(SyntheticAction):
            errors.append("one source has action-colliding successor codes")
            break
    for state in benchmark.states.values():
        if any(bridge_defects(state.core, state.order, state.signature).values()):
            errors.append("oracle state family contains a bridge defect")
            break
    for split in SPLIT_NAMES:
        actions = {case.action for case in benchmark.splits[split]}
        if actions != set(SyntheticAction):
            errors.append(f"{split} split does not cover every action")
        for case in benchmark.splits[split]:
            graph = case.example.tracking.components[ENTITY_TYPE]
            if len(graph.pairs) != len(ENTITY_IDS):
                errors.append(f"{split} contains a non-total oracle tracking map")
                break

    full_test = models["action_conditioned"]["test"]
    pooled_test = models["action_agnostic"]["test"]
    if full_test.fixed_joint_exact_rate != 1.0:
        errors.append("action-conditioned model is not fixed-joint exact on test")
    if full_test.bridge_violation_rate != 0.0:
        errors.append("action-conditioned decoder violates a declared bridge")
    if pooled_test.fixed_joint_exact_rate >= full_test.fixed_joint_exact_rate:
        errors.append("action conditioning has no positive fixed-joint test gain")
    if pooled_test.fixed_exact_rate > ACTION_AGNOSTIC_FIXED_EXACT_UPPER_BOUND:
        errors.append("action-agnostic exact rate exceeds its construction bound")
    return tuple(errors)


def run_p3_oracle_benchmark(
    spec: P3OracleBenchmarkSpec | None = None,
) -> P3OracleBenchmarkReport:
    """Run all ``P3-1`` baselines and return the complete audit report."""

    benchmark = build_oracle_transition_benchmark(spec)
    full = LinearStructuralJEPA(
        ridge=benchmark.spec.ridge,
        condition_on_action=True,
        target_momentum=benchmark.spec.target_momentum,
    ).fit(benchmark)
    pooled = LinearStructuralJEPA(
        ridge=benchmark.spec.ridge,
        condition_on_action=False,
        target_momentum=benchmark.spec.target_momentum,
    ).fit(benchmark)
    identity = IdentityTransitionBaseline()

    predictors: Mapping[str, TransitionPredictor] = {
        "action_conditioned": full,
        "action_agnostic": pooled,
        "identity": identity,
    }
    model_results = {
        name: {
            split: evaluate_transition_predictor(predictor, benchmark, split)
            for split in SPLIT_NAMES
        }
        for name, predictor in predictors.items()
    }
    errors = _audit_benchmark(benchmark, model_results)
    full_test = model_results["action_conditioned"]["test"]
    pooled_test = model_results["action_agnostic"]["test"]
    return P3OracleBenchmarkReport(
        interface_id=FROZEN_PAPER3_INTERFACE.identifier,
        benchmark_id=P3_ORACLE_BENCHMARK_ID,
        benchmark_digest=benchmark.digest,
        seed=benchmark.spec.seed,
        state_count=benchmark.state_count,
        transition_count=benchmark.transition_count,
        split_source_counts={
            split: len(benchmark.source_codes(split)) for split in SPLIT_NAMES
        },
        feature_dimension=full.layout.dimension,
        latent_dimension=full.latent_dimension,
        models=MappingProxyType(
            {
                model: MappingProxyType(dict(split_results))
                for model, split_results in model_results.items()
            }
        ),
        action_conditioning_gain=(
            full_test.fixed_joint_exact_rate - pooled_test.fixed_joint_exact_rate
        ),
        action_agnostic_fixed_exact_upper_bound=(
            ACTION_AGNOSTIC_FIXED_EXACT_UPPER_BOUND
        ),
        audit_errors=errors,
        method_contract=MappingProxyType(
            {
                "split": "source-state-disjoint deterministic split",
                "decoder": (
                    "nearest valid state in the complete declared 81-state "
                    "admissible codebook"
                ),
                "encoder": (
                    "exact standardized layer features with statistics fitted "
                    "only on training endpoints; frozen stop-gradient target copy"
                ),
                "scope": (
                    "finite oracle pipeline baseline; not evidence for learned "
                    "representation superiority or P3-2 objective ablation"
                ),
            }
        ),
    )
