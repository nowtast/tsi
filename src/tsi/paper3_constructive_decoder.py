"""Codebook-free constructive decoder for the TSI P3-3 exact-state gate."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from inspect import signature
from itertools import combinations
import json
from math import inf, isfinite
from types import MappingProxyType
from typing import Hashable, Mapping

import numpy as np

from .coherent import CoherentStructuralState, bridge_defects
from .dynamical import (
    IntegratedStructuralState,
    PartialBijection,
    TrackedTransition,
)
from .order_topology import FinitePreorder
from .paper3_multiworld import (
    BASE_LABELS,
    ENTITY_IDS,
    ENTITY_TYPE,
    P3_MULTIWORLD_SCHEMA,
    P3_MULTIWORLD_SIGNATURE,
    PRIMITIVE_ACTIONS,
    all_multiworld_state_codes,
    build_multiworld_state,
    build_world_mechanism,
    successor_code,
)
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_oracle_benchmark import StructuralFeatureLayout
from .relational import FiniteRelation, FiniteRelationAssignment


P3_CONSTRUCTIVE_DECODER_ID = "P3-3A-CONSTRUCTIVE-DECODER-v1"
PRIMARY_THRESHOLD = 0.5
MINIMUM_DISTINCT_DISTANCE = 1.0e-6


def build_multiworld_feature_layout() -> StructuralFeatureLayout:
    """Build the frozen local feature schema without enumerating target states."""

    tagged = tuple((ENTITY_TYPE, identifier) for identifier in ENTITY_IDS)
    vocabulary = tuple(
        sorted(
            {(ENTITY_TYPE, label) for label in BASE_LABELS},
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
        for arrow in P3_MULTIWORLD_SCHEMA.arrows
        for left in ENTITY_IDS
        for right in ENTITY_IDS
    )
    order_cells = tuple((left, right) for left in tagged for right in tagged)
    return StructuralFeatureLayout(
        schema=P3_MULTIWORLD_SCHEMA,
        signature=P3_MULTIWORLD_SIGNATURE,
        carriers=((ENTITY_TYPE, ENTITY_IDS),),
        tagged_entities=tagged,
        label_vocabulary=vocabulary,
        simplex_probes=simplex_probes,
        metric_pairs=metric_pairs,
        relation_cells=relation_cells,
        order_cells=order_cells,
    )


@dataclass(frozen=True)
class DecoderSlices:
    label: slice
    topology: slice
    metric: slice
    relation: slice
    order: slice
    dimension: int

    @classmethod
    def from_layout(cls, layout: StructuralFeatureLayout) -> "DecoderSlices":
        sizes = (
            len(layout.tagged_entities) * len(layout.label_vocabulary),
            len(layout.simplex_probes),
            len(layout.metric_pairs),
            len(layout.relation_cells),
            len(layout.order_cells),
        )
        slices: list[slice] = []
        cursor = 0
        for size in sizes:
            slices.append(slice(cursor, cursor + size))
            cursor += size
        if cursor != layout.dimension:
            raise RuntimeError("decoder slices do not cover the feature layout")
        return cls(*slices, dimension=cursor)


@dataclass(frozen=True)
class ConstructiveDecodedPrediction:
    target: CoherentStructuralState
    tracking: TrackedTransition
    raw_features: np.ndarray


class ConstructiveStructuralDecoder:
    """Construct valid layers from local coordinates, never global candidates."""

    def __init__(
        self,
        layout: StructuralFeatureLayout,
        *,
        threshold: float = PRIMARY_THRESHOLD,
        minimum_distinct_distance: float = MINIMUM_DISTINCT_DISTANCE,
    ) -> None:
        if layout.schema != P3_MULTIWORLD_SCHEMA:
            raise ValueError("decoder requires the frozen P3-3 schema")
        if layout.signature != P3_MULTIWORLD_SIGNATURE:
            raise ValueError("decoder requires the frozen P3-3 signature")
        if not isfinite(threshold):
            raise ValueError("decoder threshold must be finite")
        if not isfinite(minimum_distinct_distance) or minimum_distinct_distance <= 0.0:
            raise ValueError(
                "minimum distinct-point distance must be finite and positive"
            )
        self.layout = layout
        self.slices = DecoderSlices.from_layout(layout)
        self.threshold = float(threshold)
        self.minimum_distinct_distance = float(minimum_distinct_distance)

    def _validate_raw(self, raw_features: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_features, dtype=np.float64)
        if values.shape != (self.layout.dimension,):
            raise ValueError(
                f"raw feature vector must have shape ({self.layout.dimension},)"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("raw feature vector must be finite")
        return values

    def _decode_labels(self, raw: np.ndarray) -> tuple[Hashable, ...]:
        label_values = raw[self.slices.label].reshape(
            len(self.layout.tagged_entities),
            len(self.layout.label_vocabulary),
        )
        labels: list[Hashable] = []
        for row, entity in zip(
            label_values,
            self.layout.tagged_entities,
            strict=True,
        ):
            object_name = entity[0]
            compatible = tuple(
                (index, label)
                for index, label in enumerate(self.layout.label_vocabulary)
                if label[0] == object_name
            )
            if not compatible:
                raise ValueError("label vocabulary misses an entity type")
            best_index, best_label = max(
                compatible,
                key=lambda item: (row[item[0]], -item[0]),
            )
            del best_index
            labels.append(best_label[1])
        return tuple(labels)

    def _decode_simplices(
        self,
        raw: np.ndarray,
    ) -> frozenset[frozenset[tuple[Hashable, Hashable]]]:
        simplices: set[frozenset[tuple[Hashable, Hashable]]] = {frozenset()}
        simplices.update(frozenset((entity,)) for entity in self.layout.tagged_entities)
        selected = (
            simplex
            for simplex, value in zip(
                self.layout.simplex_probes,
                raw[self.slices.topology],
                strict=True,
            )
            if value >= self.threshold
        )
        for simplex in selected:
            ordered = tuple(simplex)
            for size in range(len(ordered) + 1):
                simplices.update(
                    frozenset(face) for face in combinations(ordered, size)
                )
        return frozenset(simplices)

    def _decode_metric(
        self,
        raw: np.ndarray,
    ) -> tuple[tuple[float, ...], ...]:
        count = len(self.layout.tagged_entities)
        index = {
            entity: position
            for position, entity in enumerate(self.layout.tagged_entities)
        }
        matrix = [[inf] * count for _ in range(count)]
        for position in range(count):
            matrix[position][position] = 0.0
        for (left, right), normalized in zip(
            self.layout.metric_pairs,
            raw[self.slices.metric],
            strict=True,
        ):
            distance = max(
                abs(float(normalized)) * self.layout.signature.metric_scale,
                self.minimum_distinct_distance,
            )
            left_index = index[left]
            right_index = index[right]
            matrix[left_index][right_index] = distance
            matrix[right_index][left_index] = distance

        for middle in range(count):
            for left in range(count):
                for right in range(count):
                    candidate = matrix[left][middle] + matrix[middle][right]
                    if candidate < matrix[left][right]:
                        matrix[left][right] = candidate
        return tuple(tuple(value for value in row) for row in matrix)

    def _decode_relations(
        self,
        raw: np.ndarray,
        simplices: frozenset[frozenset[tuple[Hashable, Hashable]]],
    ) -> Mapping[Hashable, FiniteRelation]:
        raw_by_cell = dict(
            zip(
                self.layout.relation_cells,
                raw[self.slices.relation],
                strict=True,
            )
        )
        generators: dict[Hashable, FiniteRelation] = {}
        for arrow in self.layout.schema.arrows:
            if arrow.name == "adjacent":
                pairs = frozenset(
                    (left, right)
                    for left in ENTITY_IDS
                    for right in ENTITY_IDS
                    if left != right
                    and frozenset(((ENTITY_TYPE, left), (ENTITY_TYPE, right)))
                    in simplices
                )
            else:
                pairs = frozenset(
                    (left, right)
                    for left in ENTITY_IDS
                    for right in ENTITY_IDS
                    if raw_by_cell[(arrow.name, left, right)] >= self.threshold
                )
            generators[arrow.name] = FiniteRelation(
                ENTITY_IDS,
                ENTITY_IDS,
                pairs,
            )
        return MappingProxyType(generators)

    def _decode_order(
        self,
        raw: np.ndarray,
        tagged_entities: tuple[tuple[Hashable, Hashable], ...],
        tagged_labels: tuple[tuple[Hashable, Hashable], ...],
    ) -> FinitePreorder:
        count = len(tagged_entities)
        relation = [[False] * count for _ in range(count)]
        index = {entity: position for position, entity in enumerate(tagged_entities)}
        for position in range(count):
            relation[position][position] = True
        for (left, right), value in zip(
            self.layout.order_cells,
            raw[self.slices.order],
            strict=True,
        ):
            if value >= self.threshold:
                relation[index[left]][index[right]] = True
        for middle in range(count):
            for left in range(count):
                for right in range(count):
                    relation[left][right] = relation[left][right] or (
                        relation[left][middle] and relation[middle][right]
                    )
        pairs = frozenset(
            (tagged_entities[left], tagged_entities[right])
            for left in range(count)
            for right in range(count)
            if relation[left][right]
        )
        return FinitePreorder(tagged_entities, pairs, tagged_labels)

    def decode_state(
        self,
        raw_features: np.ndarray,
    ) -> CoherentStructuralState:
        """Decode one valid state without a target-state candidate collection."""

        raw = self._validate_raw(raw_features)
        labels = self._decode_labels(raw)
        simplices = self._decode_simplices(raw)
        distances = self._decode_metric(raw)
        generators = self._decode_relations(raw, simplices)
        relational = FiniteRelationAssignment(
            schema=self.layout.schema,
            carriers={ENTITY_TYPE: ENTITY_IDS},
            labels={ENTITY_TYPE: labels},
            generators=generators,
        )
        core = IntegratedStructuralState(
            relational=relational,
            simplices=simplices,
            distances=distances,
        )
        order = self._decode_order(
            raw,
            core.tagged_entities,
            core.tagged_labels,
        )
        return CoherentStructuralState(
            core=core,
            order=order,
            signature=self.layout.signature,
        )

    def _best_tracking_pairs(
        self,
        source: CoherentStructuralState,
        target: CoherentStructuralState,
        tracking_scores: np.ndarray,
    ) -> frozenset[tuple[Hashable, Hashable]]:
        scores = np.asarray(tracking_scores, dtype=np.float64)
        if scores.shape != (len(ENTITY_IDS), len(ENTITY_IDS)):
            raise ValueError("tracking scores must have one row and column per entity")
        if not np.all(np.isfinite(scores)):
            raise ValueError("tracking scores must be finite")

        source_labels = source.core.relational.label_map(ENTITY_TYPE)
        target_labels = target.core.relational.label_map(ENTITY_TYPE)
        best_key: tuple[float, int, tuple[tuple[int, int], ...]] | None = None
        best_pairs: tuple[tuple[int, int], ...] = ()

        def search(
            source_position: int,
            used_targets: frozenset[int],
            pairs: tuple[tuple[int, int], ...],
            total_score: float,
        ) -> None:
            nonlocal best_key, best_pairs
            if source_position == len(ENTITY_IDS):
                key = (total_score, len(pairs), tuple(reversed(pairs)))
                if best_key is None or key > best_key:
                    best_key = key
                    best_pairs = pairs
                return
            search(
                source_position + 1,
                used_targets,
                pairs,
                total_score,
            )
            source_id = ENTITY_IDS[source_position]
            for target_position, target_id in enumerate(ENTITY_IDS):
                if target_id in used_targets:
                    continue
                if source_labels[source_id] != target_labels[target_id]:
                    continue
                score = float(scores[source_position, target_position])
                search(
                    source_position + 1,
                    used_targets.union((target_id,)),
                    (*pairs, (source_id, target_id)),
                    total_score + score,
                )

        search(0, frozenset(), (), 0.0)
        return frozenset(best_pairs)

    def decode_transition(
        self,
        source: CoherentStructuralState,
        raw_target_features: np.ndarray,
        tracking_scores: np.ndarray,
    ) -> ConstructiveDecodedPrediction:
        target = self.decode_state(raw_target_features)
        tracking_pairs = self._best_tracking_pairs(
            source,
            target,
            tracking_scores,
        )
        tracking = TrackedTransition(
            source=source.core,
            target=target.core,
            components={
                ENTITY_TYPE: PartialBijection(
                    ENTITY_IDS,
                    ENTITY_IDS,
                    tracking_pairs,
                )
            },
        )
        return ConstructiveDecodedPrediction(
            target=target,
            tracking=tracking,
            raw_features=np.asarray(
                raw_target_features,
                dtype=np.float64,
            ).copy(),
        )


def exact_tracking_scores(
    tracking: TrackedTransition,
) -> np.ndarray:
    """Encode a tracking graph as separated positive/negative scores."""

    pairs = tracking.components[ENTITY_TYPE].pairs
    return np.asarray(
        [
            [1.0 if (left, right) in pairs else -1.0 for right in ENTITY_IDS]
            for left in ENTITY_IDS
        ],
        dtype=np.float64,
    )


def constructive_decoder_digest() -> str:
    """Digest the semantic local-decoding policy."""

    layout = build_multiworld_feature_layout()
    payload = {
        "identifier": P3_CONSTRUCTIVE_DECODER_ID,
        "layout_dimension": layout.dimension,
        "label_vocabulary": [list(label) for label in layout.label_vocabulary],
        "simplex_probes": [
            [list(entity) for entity in sorted(simplex, key=repr)]
            for simplex in layout.simplex_probes
        ],
        "metric_pairs": [
            [list(left), list(right)] for left, right in layout.metric_pairs
        ],
        "relation_cells": [
            [arrow, left, right] for arrow, left, right in layout.relation_cells
        ],
        "order_cells": [
            [list(left), list(right)] for left, right in layout.order_cells
        ],
        "threshold": PRIMARY_THRESHOLD,
        "minimum_distinct_distance": MINIMUM_DISTINCT_DISTANCE,
        "global_target_state_candidates": 0,
        "full_codebook_branch": False,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ConstructiveDecoderAudit:
    decoder_digest: str
    layout_dimension: int
    exact_state_decodes: int
    exact_tracking_decodes: int
    adversarial_valid_decodes: int
    target_state_collection_parameters: int
    global_candidate_states: int
    bridge_violations: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "decoder_digest": self.decoder_digest,
            "layout_dimension": self.layout_dimension,
            "exact_state_decodes": self.exact_state_decodes,
            "exact_tracking_decodes": self.exact_tracking_decodes,
            "adversarial_valid_decodes": self.adversarial_valid_decodes,
            "target_state_collection_parameters": (
                self.target_state_collection_parameters
            ),
            "global_candidate_states": self.global_candidate_states,
            "bridge_violations": self.bridge_violations,
            "errors": list(self.errors),
            "passed": self.passed,
        }


def audit_constructive_decoder() -> ConstructiveDecoderAudit:
    """Exhaustively reconstruct the finite state/action family."""

    errors: list[str] = []
    layout = build_multiworld_feature_layout()
    decoder = ConstructiveStructuralDecoder(layout)
    constructor_parameters = signature(
        ConstructiveStructuralDecoder.__init__
    ).parameters
    forbidden_parameters = tuple(
        name
        for name in constructor_parameters
        if "candidate" in name or "target_state" in name or "codebook" in name
    )
    if forbidden_parameters:
        errors.append(
            "decoder constructor exposes global candidates: "
            + ", ".join(forbidden_parameters)
        )

    exact_state_decodes = 0
    bridge_violations = 0
    for code in all_multiworld_state_codes():
        state = build_multiworld_state(code)
        decoded = decoder.decode_state(layout.encode(state))
        if decoded != state:
            errors.append(f"exact state decode failed for {code.as_tuple()}")
            break
        defects = bridge_defects(decoded.core, decoded.order, decoded.signature)
        bridge_violations += sum(value != 0.0 for value in defects.values())
        exact_state_decodes += 1

    adversarial_vectors = (
        np.zeros(layout.dimension, dtype=np.float64),
        np.ones(layout.dimension, dtype=np.float64),
        np.linspace(-2.0, 2.0, layout.dimension, dtype=np.float64),
        np.asarray(
            [(-1.0) ** index * (index + 1) for index in range(layout.dimension)],
            dtype=np.float64,
        ),
    )
    adversarial_valid_decodes = 0
    for vector in adversarial_vectors:
        try:
            decoded = decoder.decode_state(vector)
        except ValueError as error:
            errors.append(f"adversarial constructive decode failed: {error}")
            continue
        defects = bridge_defects(decoded.core, decoded.order, decoded.signature)
        bridge_violations += sum(value != 0.0 for value in defects.values())
        adversarial_valid_decodes += 1
    invalid = np.zeros(layout.dimension, dtype=np.float64)
    invalid[0] = np.nan
    try:
        decoder.decode_state(invalid)
    except ValueError:
        pass
    else:
        errors.append("nonfinite raw features were not rejected")

    mechanism = build_world_mechanism(
        WorldFamily.BRIDGE_COUPLED,
        BenchmarkSplit.DEVELOPMENT,
        0,
    )
    exact_tracking_decodes = 0
    for code in all_multiworld_state_codes():
        source = build_multiworld_state(code)
        for action in PRIMITIVE_ACTIONS:
            target_code = successor_code(code, action, mechanism)
            target = build_multiworld_state(target_code)
            label_delta = (
                mechanism.layer_multipliers[0] * action.mapping["label"]
            ) % len(ENTITY_IDS)
            expected_pairs = frozenset(
                (
                    identifier,
                    (identifier + label_delta) % len(ENTITY_IDS),
                )
                for identifier in ENTITY_IDS
            )
            expected_tracking = TrackedTransition(
                source=source.core,
                target=target.core,
                components={
                    ENTITY_TYPE: PartialBijection(
                        ENTITY_IDS,
                        ENTITY_IDS,
                        expected_pairs,
                    )
                },
            )
            prediction = decoder.decode_transition(
                source,
                layout.encode(target),
                exact_tracking_scores(expected_tracking),
            )
            if prediction.target != target:
                errors.append(
                    "transition target decode failed for "
                    f"{code.as_tuple()}/{action.name}"
                )
                break
            if (
                prediction.tracking.components[ENTITY_TYPE]
                != expected_tracking.components[ENTITY_TYPE]
            ):
                errors.append(
                    f"tracking decode failed for {code.as_tuple()}/{action.name}"
                )
                break
            exact_tracking_decodes += 1
        if errors:
            break

    return ConstructiveDecoderAudit(
        decoder_digest=constructive_decoder_digest(),
        layout_dimension=layout.dimension,
        exact_state_decodes=exact_state_decodes,
        exact_tracking_decodes=exact_tracking_decodes,
        adversarial_valid_decodes=adversarial_valid_decodes,
        target_state_collection_parameters=len(forbidden_parameters),
        global_candidate_states=0,
        bridge_violations=bridge_violations,
        errors=tuple(errors),
    )
