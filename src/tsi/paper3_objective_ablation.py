"""Matched objective-ablation benchmark for TSI Paper 3 gate ``P3-2``.

The gate keeps the frozen ``P3-I0-FIXED-v1`` state and evaluator contracts but
replaces the fixed P3-1 encoder with a trainable NumPy reference model.  The
benchmark is deliberately finite and exact:

* source states are split by a pre-registered interaction residue;
* every condition has identical parameters, updates, data, and initialization
  seed;
* context and target encoders are separated by stop-gradient and EMA;
* label, topology, metric, relation, order, and tracking use independent heads;
* bridge and finite-domain validity are differentiable soft surrogates;
* hard outputs are projected to the declared coherent codebook before exact
  I0, fixed-carrier, and tracking evaluation; and
* five paired seeds are summarized without requiring a positive result.

This is a small reference experiment, not a scalable neural implementation and
not a theorem that soft losses recover hidden structure.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from hashlib import sha256
from itertools import permutations
import json
from math import cos, isfinite, pi, sqrt
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .coherent import CoherentStructuralState
from .dynamical import PartialBijection, TrackedTransition
from .paper3_interface import (
    FROZEN_PAPER3_INTERFACE,
)
from .paper3_oracle_benchmark import (
    ENTITY_IDS,
    ENTITY_TYPE,
    SPLIT_NAMES,
    OracleTransitionCase,
    StructuralFeatureLayout,
    SyntheticAction,
    SyntheticStateCode,
    all_state_codes,
    build_oracle_state,
    build_oracle_tracking,
    successor_code,
)
from .paper3_interface import StructuralTransitionExample


P3_ABLATION_BENCHMARK_ID = "P3-2-ABLATION-v1"
DEFAULT_ABLATION_SEEDS = (
    20_260_728,
    20_260_729,
    20_260_730,
    20_260_731,
    20_260_732,
)
INTERACTION_RESIDUE_TO_SPLIT = MappingProxyType(
    {0: "train", 1: "validation", 2: "test"}
)
T_CRITICAL_95_DF4 = 2.7764451051977987
_EPSILON = 1.0e-12


class ObjectiveCondition(str, Enum):
    """The eight pre-registered P3-2 training conditions."""

    JEPA_ONLY = "jepa_only"
    NO_TOPOLOGY = "no_topology"
    NO_METRIC = "no_metric"
    NO_RELATION = "no_relation"
    NO_ORDER = "no_order"
    NO_BRIDGE = "no_bridge"
    NO_TRACKING = "no_tracking"
    FULL = "full"


@dataclass(frozen=True)
class ObjectiveMask:
    """Nonnegative switches for the frozen nine empirical surrogates."""

    jepa_latent: float
    label_surrogate: float
    simplicial_surrogate: float
    metric_surrogate: float
    relation_surrogate: float
    order_surrogate: float
    bridge_surrogate: float
    tracking_surrogate: float
    validity_surrogate: float

    def __post_init__(self) -> None:
        for item in fields(self):
            value = float(getattr(self, item.name))
            if not isfinite(value) or value < 0.0:
                raise ValueError("objective masks must be finite and nonnegative")
            object.__setattr__(self, item.name, value)

    def as_dict(self) -> dict[str, float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


def _mask_without(name: str) -> ObjectiveMask:
    values = {item.name: 1.0 for item in fields(ObjectiveMask)}
    values[name] = 0.0
    return ObjectiveMask(**values)


OBJECTIVE_MASKS: Mapping[ObjectiveCondition, ObjectiveMask] = MappingProxyType(
    {
        ObjectiveCondition.JEPA_ONLY: ObjectiveMask(
            jepa_latent=1.0,
            label_surrogate=0.0,
            simplicial_surrogate=0.0,
            metric_surrogate=0.0,
            relation_surrogate=0.0,
            order_surrogate=0.0,
            bridge_surrogate=0.0,
            tracking_surrogate=0.0,
            validity_surrogate=0.0,
        ),
        ObjectiveCondition.NO_TOPOLOGY: _mask_without("simplicial_surrogate"),
        ObjectiveCondition.NO_METRIC: _mask_without("metric_surrogate"),
        ObjectiveCondition.NO_RELATION: _mask_without("relation_surrogate"),
        ObjectiveCondition.NO_ORDER: _mask_without("order_surrogate"),
        ObjectiveCondition.NO_BRIDGE: _mask_without("bridge_surrogate"),
        ObjectiveCondition.NO_TRACKING: _mask_without("tracking_surrogate"),
        ObjectiveCondition.FULL: ObjectiveMask(
            jepa_latent=1.0,
            label_surrogate=1.0,
            simplicial_surrogate=1.0,
            metric_surrogate=1.0,
            relation_surrogate=1.0,
            order_surrogate=1.0,
            bridge_surrogate=1.0,
            tracking_surrogate=1.0,
            validity_surrogate=1.0,
        ),
    }
)


@dataclass(frozen=True)
class P3AblationSpec:
    """Fixed optimization and replication contract for the P3-2 gate."""

    seeds: tuple[int, ...] = DEFAULT_ABLATION_SEEDS
    training_steps: int = 1000
    learning_rate: float = 0.01
    minimum_learning_rate_fraction: float = 0.05
    latent_dimension: int = 16
    ema_momentum: float = 0.98
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    gradient_clip: float = 5.0

    def __post_init__(self) -> None:
        seeds = tuple(self.seeds)
        if not seeds or any(type(seed) is not int for seed in seeds):
            raise ValueError("seeds must be a nonempty integer tuple")
        if len(set(seeds)) != len(seeds):
            raise ValueError("ablation seeds must be unique")
        if type(self.training_steps) is not int or self.training_steps <= 0:
            raise ValueError("training_steps must be a positive integer")
        if type(self.latent_dimension) is not int or self.latent_dimension <= 1:
            raise ValueError("latent_dimension must be an integer greater than one")
        finite_positive = {
            "learning_rate": self.learning_rate,
            "adam_epsilon": self.adam_epsilon,
            "gradient_clip": self.gradient_clip,
        }
        if any(
            not isfinite(float(value)) or float(value) <= 0.0
            for value in finite_positive.values()
        ):
            raise ValueError("positive optimizer parameters must be finite")
        if not 0.0 < self.minimum_learning_rate_fraction <= 1.0:
            raise ValueError("minimum learning-rate fraction must lie in (0, 1]")
        if not 0.0 <= self.ema_momentum < 1.0:
            raise ValueError("ema_momentum must lie in [0, 1)")
        if not 0.0 <= self.adam_beta1 < 1.0:
            raise ValueError("adam_beta1 must lie in [0, 1)")
        if not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("adam_beta2 must lie in [0, 1)")
        object.__setattr__(self, "seeds", seeds)

    def as_dict(self) -> dict[str, object]:
        return {
            item.name: (
                list(getattr(self, item.name))
                if item.name == "seeds"
                else getattr(self, item.name)
            )
            for item in fields(self)
        }


@dataclass(frozen=True)
class P3AblationBenchmark:
    """Interaction-disjoint exact transition benchmark."""

    states: Mapping[SyntheticStateCode, CoherentStructuralState]
    splits: Mapping[str, tuple[OracleTransitionCase, ...]]
    layout: StructuralFeatureLayout
    digest: str

    def __post_init__(self) -> None:
        if set(self.states) != set(all_state_codes()):
            raise ValueError("ablation benchmark must contain all 81 states")
        if tuple(self.splits) != SPLIT_NAMES:
            raise ValueError("ablation splits must use the stable split order")
        object.__setattr__(self, "states", MappingProxyType(dict(self.states)))
        object.__setattr__(
            self,
            "splits",
            MappingProxyType(
                {name: tuple(cases) for name, cases in self.splits.items()}
            ),
        )

    def source_codes(self, split: str) -> frozenset[SyntheticStateCode]:
        return frozenset(case.source_code for case in self.splits[split])

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def transition_count(self) -> int:
        return sum(len(cases) for cases in self.splits.values())


def interaction_residue(code: SyntheticStateCode) -> int:
    """Return the pre-registered label/topology interaction class."""

    return (code.label_phase + code.topology_mode) % 3


def _ablation_benchmark_digest(
    states: Mapping[SyntheticStateCode, CoherentStructuralState],
    splits: Mapping[str, tuple[OracleTransitionCase, ...]],
    layout: StructuralFeatureLayout,
) -> str:
    payload = {
        "benchmark_id": P3_ABLATION_BENCHMARK_ID,
        "interface_id": FROZEN_PAPER3_INTERFACE.identifier,
        "split_rule": "(label_phase + topology_mode) mod 3",
        "residue_to_split": dict(INTERACTION_RESIDUE_TO_SPLIT),
        "states": [
            {
                "code": code.as_tuple(),
                "features": layout.encode(states[code]).tolist(),
                "bridges": [
                    [bridge.arrow, bridge.kind, bridge.threshold]
                    for bridge in states[code].signature.bridges
                ],
            }
            for code in sorted(states)
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


def build_p3_ablation_benchmark() -> P3AblationBenchmark:
    """Build the fixed interaction-residue benchmark."""

    codes = all_state_codes()
    states = {code: build_oracle_state(code) for code in codes}
    layout = StructuralFeatureLayout.from_states(tuple(states[code] for code in codes))
    split_lists: dict[str, list[OracleTransitionCase]] = {
        name: [] for name in SPLIT_NAMES
    }
    for source_code in codes:
        split = INTERACTION_RESIDUE_TO_SPLIT[interaction_residue(source_code)]
        source = states[source_code]
        for action in SyntheticAction:
            target_code = successor_code(source_code, action)
            target = states[target_code]
            tracking = build_oracle_tracking(source, target, action)
            split_lists[split].append(
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
    splits = {name: tuple(split_lists[name]) for name in SPLIT_NAMES}
    return P3AblationBenchmark(
        states=states,
        splits=splits,
        layout=layout,
        digest=_ablation_benchmark_digest(states, splits, layout),
    )


@dataclass(frozen=True)
class FeatureSlices:
    """Stable coordinates of the five exact state heads."""

    label: slice
    simplicial: slice
    metric: slice
    relation: slice
    order: slice
    dimension: int

    @classmethod
    def from_layout(cls, layout: StructuralFeatureLayout) -> FeatureSlices:
        label_size = len(layout.tagged_entities) * len(layout.label_vocabulary)
        simplicial_size = len(layout.simplex_probes)
        metric_size = len(layout.metric_pairs)
        relation_size = len(layout.relation_cells)
        order_size = len(layout.order_cells)
        cursor = 0
        label = slice(cursor, cursor + label_size)
        cursor = label.stop
        simplicial = slice(cursor, cursor + simplicial_size)
        cursor = simplicial.stop
        metric = slice(cursor, cursor + metric_size)
        cursor = metric.stop
        relation = slice(cursor, cursor + relation_size)
        cursor = relation.stop
        order = slice(cursor, cursor + order_size)
        cursor = order.stop
        if cursor != layout.dimension:
            raise RuntimeError("feature slices do not cover the structural layout")
        return cls(label, simplicial, metric, relation, order, cursor)

    def dimensions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                "label": self.label.stop - self.label.start,
                "simplicial": self.simplicial.stop - self.simplicial.start,
                "metric": self.metric.stop - self.metric.start,
                "relation": self.relation.stop - self.relation.start,
                "order": self.order.stop - self.order.start,
            }
        )


@dataclass(frozen=True)
class NumericSplit:
    """Exact transition examples represented as matched NumPy arrays."""

    cases: tuple[OracleTransitionCase, ...]
    source_inputs: np.ndarray
    target_inputs: np.ndarray
    actions: np.ndarray
    target_features: np.ndarray
    tracking_targets: np.ndarray


@dataclass(frozen=True)
class P3AblationDataset:
    """Numerical representation and candidate codebook for P3-2."""

    benchmark: P3AblationBenchmark
    slices: FeatureSlices
    input_mean: np.ndarray
    input_scale: np.ndarray
    active_coordinates: np.ndarray
    splits: Mapping[str, NumericSplit]
    candidate_codes: tuple[SyntheticStateCode, ...]
    candidate_features: np.ndarray
    coordinate_weights: np.ndarray
    bridge_links: tuple[tuple[int, int], ...]

    @property
    def input_dimension(self) -> int:
        return int(self.active_coordinates.size)

    def standardize(self, features: np.ndarray) -> np.ndarray:
        values = (features - self.input_mean) / self.input_scale
        return values[..., self.active_coordinates]


def _tracking_target(case: OracleTransitionCase) -> np.ndarray:
    target = np.zeros((len(ENTITY_IDS), len(ENTITY_IDS)), dtype=np.float64)
    for left, right in case.example.tracking.components[ENTITY_TYPE].pairs:
        target[ENTITY_IDS.index(left), ENTITY_IDS.index(right)] = 1.0
    return target.reshape(-1)


def _action_vector(action: SyntheticAction) -> np.ndarray:
    result = np.zeros(len(SyntheticAction), dtype=np.float64)
    result[tuple(SyntheticAction).index(action)] = 1.0
    return result


def _bridge_links(layout: StructuralFeatureLayout) -> tuple[tuple[int, int], ...]:
    relation_index = {cell: index for index, cell in enumerate(layout.relation_cells)}
    links: list[tuple[int, int]] = []
    for simplex_index, simplex in enumerate(layout.simplex_probes):
        if len(simplex) != 2:
            continue
        left, right = sorted(simplex, key=repr)
        if left[0] != ENTITY_TYPE or right[0] != ENTITY_TYPE:
            raise ValueError("P3-2 adjacency bridge expects the entity type")
        links.append(
            (
                simplex_index,
                relation_index[("adjacent", left[1], right[1])],
            )
        )
        links.append(
            (
                simplex_index,
                relation_index[("adjacent", right[1], left[1])],
            )
        )
    if len(links) != 6:
        raise RuntimeError("three undirected edges must induce six bridge links")
    return tuple(links)


def build_p3_ablation_dataset(
    benchmark: P3AblationBenchmark | None = None,
) -> P3AblationDataset:
    """Encode the P3-2 benchmark without fitting on held-out source states."""

    resolved = benchmark or build_p3_ablation_benchmark()
    slices = FeatureSlices.from_layout(resolved.layout)
    raw: dict[str, tuple[np.ndarray, ...]] = {}
    for split in SPLIT_NAMES:
        cases = resolved.splits[split]
        source_features = np.stack(
            [resolved.layout.encode(case.example.source) for case in cases]
        )
        target_features = np.stack(
            [resolved.layout.encode(case.example.target) for case in cases]
        )
        actions = np.stack([_action_vector(case.action) for case in cases])
        tracking = np.stack([_tracking_target(case) for case in cases])
        raw[split] = (
            source_features,
            target_features,
            actions,
            tracking,
        )

    train_endpoints = np.concatenate((raw["train"][0], raw["train"][1]), axis=0)
    mean = np.mean(train_endpoints, axis=0)
    deviation = np.std(train_endpoints, axis=0)
    active = np.flatnonzero(deviation > 1.0e-12)
    scale = deviation.copy()
    scale[scale <= 1.0e-12] = 1.0

    numeric_splits: dict[str, NumericSplit] = {}
    for split in SPLIT_NAMES:
        source_features, target_features, actions, tracking = raw[split]
        numeric_splits[split] = NumericSplit(
            cases=resolved.splits[split],
            source_inputs=((source_features - mean) / scale)[:, active],
            target_inputs=((target_features - mean) / scale)[:, active],
            actions=actions,
            target_features=target_features,
            tracking_targets=tracking,
        )

    candidate_codes = tuple(sorted(resolved.states))
    candidate_features = np.stack(
        [resolved.layout.encode(resolved.states[code]) for code in candidate_codes]
    )
    coordinate_weights = np.zeros(slices.dimension, dtype=np.float64)
    for block_slice in (
        slices.label,
        slices.simplicial,
        slices.metric,
        slices.relation,
        slices.order,
    ):
        coordinate_weights[block_slice] = 0.2 / (block_slice.stop - block_slice.start)
    return P3AblationDataset(
        benchmark=resolved,
        slices=slices,
        input_mean=mean,
        input_scale=scale,
        active_coordinates=active,
        splits=MappingProxyType(numeric_splits),
        candidate_codes=candidate_codes,
        candidate_features=candidate_features,
        coordinate_weights=coordinate_weights,
        bridge_links=_bridge_links(resolved.layout),
    )


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0.0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _softmax_pairs(logits: np.ndarray) -> np.ndarray:
    shaped = logits.reshape(logits.shape[0], len(ENTITY_IDS), -1)
    shifted = shaped - np.max(shaped, axis=2, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=2, keepdims=True)


def _binary_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    losses = (
        np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    )
    return float(np.mean(losses))


@dataclass(frozen=True)
class ForwardPass:
    """Cached arrays needed for exact manual reverse-mode differentiation."""

    context: np.ndarray
    predictor_operators: np.ndarray
    predicted_latent: np.ndarray
    target_latent: np.ndarray
    logits: Mapping[str, np.ndarray]
    probabilities: Mapping[str, np.ndarray]
    soft_state: np.ndarray
    nearest_candidate_indices: np.ndarray
    nearest_candidate_scores: np.ndarray


@dataclass(frozen=True)
class TrainingSnapshot:
    """Raw loss ledger at one optimization point."""

    total: float
    losses: Mapping[str, float]
    gradient_norm: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "total": self.total,
            "losses": dict(self.losses),
            "gradient_norm": self.gradient_norm,
        }


class TrainableStructuralJEPA:
    """Small trainable JEPA-style model with exact manual NumPy gradients."""

    _HEADS = ("label", "simplicial", "metric", "relation", "order", "tracking")

    def __init__(
        self,
        dataset: P3AblationDataset,
        condition: ObjectiveCondition | str,
        seed: int,
        spec: P3AblationSpec,
    ) -> None:
        try:
            self.condition = (
                condition
                if isinstance(condition, ObjectiveCondition)
                else ObjectiveCondition(condition)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown objective condition: {condition!r}") from error
        if type(seed) is not int:
            raise ValueError("model seed must be an integer")
        self.dataset = dataset
        self.seed = seed
        self.spec = spec
        self.mask = OBJECTIVE_MASKS[self.condition]
        self.rng = np.random.default_rng(seed)
        dimensions = dataset.slices.dimensions()
        latent = spec.latent_dimension
        input_dimension = dataset.input_dimension

        def glorot(rows: int, columns: int) -> np.ndarray:
            bound = sqrt(6.0 / (rows + columns))
            return self.rng.uniform(-bound, bound, size=(rows, columns))

        self.parameters: dict[str, np.ndarray] = {
            "encoder_weight": glorot(input_dimension, latent),
            "encoder_bias": np.zeros(latent, dtype=np.float64),
            "predictor_weight": np.stack(
                [glorot(latent, latent) for _ in SyntheticAction],
                axis=0,
            ),
            "predictor_bias": np.zeros(
                (len(SyntheticAction), latent),
                dtype=np.float64,
            ),
        }
        for head in self._HEADS:
            output_dimension = (
                len(ENTITY_IDS) * len(ENTITY_IDS)
                if head == "tracking"
                else dimensions[head]
            )
            self.parameters[f"{head}_weight"] = glorot(
                latent,
                output_dimension,
            )
            self.parameters[f"{head}_bias"] = np.zeros(
                output_dimension,
                dtype=np.float64,
            )
        self.target_weight = self.parameters["encoder_weight"].copy()
        self.target_bias = self.parameters["encoder_bias"].copy()
        self.adam_first = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.adam_second = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        self.update_count = 0
        self.initial_snapshot: TrainingSnapshot | None = None
        self.final_snapshot: TrainingSnapshot | None = None

    @property
    def parameter_count(self) -> int:
        """Return the matched number of independently optimized parameters."""

        return sum(value.size for value in self.parameters.values())

    def _forward(
        self,
        source_inputs: np.ndarray,
        actions: np.ndarray,
        target_inputs: np.ndarray,
    ) -> ForwardPass:
        context = np.tanh(
            source_inputs @ self.parameters["encoder_weight"]
            + self.parameters["encoder_bias"]
        )
        predictor_operators = np.einsum(
            "ba,aij->bij",
            actions,
            self.parameters["predictor_weight"],
        )
        predictor_bias = actions @ self.parameters["predictor_bias"]
        predicted_latent = np.tanh(
            np.einsum("bi,bij->bj", context, predictor_operators) + predictor_bias
        )
        target_latent = np.tanh(target_inputs @ self.target_weight + self.target_bias)
        logits = {
            head: (
                predicted_latent @ self.parameters[f"{head}_weight"]
                + self.parameters[f"{head}_bias"]
            )
            for head in self._HEADS
        }
        label_probabilities = _softmax_pairs(logits["label"]).reshape(
            source_inputs.shape[0],
            -1,
        )
        probabilities = {
            "label": label_probabilities,
            **{
                head: _stable_sigmoid(logits[head])
                for head in self._HEADS
                if head != "label"
            },
        }
        soft_state = np.concatenate(
            tuple(probabilities[head] for head in self._HEADS[:-1]),
            axis=1,
        )
        differences = (
            soft_state[:, None, :] - self.dataset.candidate_features[None, :, :]
        )
        candidate_scores = np.sum(
            differences**2 * self.dataset.coordinate_weights[None, None, :],
            axis=2,
        )
        nearest = np.argmin(candidate_scores, axis=1)
        return ForwardPass(
            context=context,
            predictor_operators=predictor_operators,
            predicted_latent=predicted_latent,
            target_latent=target_latent,
            logits=MappingProxyType(logits),
            probabilities=MappingProxyType(probabilities),
            soft_state=soft_state,
            nearest_candidate_indices=nearest,
            nearest_candidate_scores=candidate_scores[
                np.arange(source_inputs.shape[0]), nearest
            ],
        )

    def _losses_and_gradients(
        self,
        split: NumericSplit,
        *,
        gradients: bool,
    ) -> tuple[TrainingSnapshot, Mapping[str, np.ndarray] | None, ForwardPass]:
        forward = self._forward(
            split.source_inputs,
            split.actions,
            split.target_inputs,
        )
        batch_size = split.source_inputs.shape[0]
        slices = self.dataset.slices
        targets = {
            "label": split.target_features[:, slices.label],
            "simplicial": split.target_features[:, slices.simplicial],
            "metric": split.target_features[:, slices.metric],
            "relation": split.target_features[:, slices.relation],
            "order": split.target_features[:, slices.order],
            "tracking": split.tracking_targets,
        }

        label_probabilities = forward.probabilities["label"].reshape(
            batch_size,
            len(ENTITY_IDS),
            -1,
        )
        label_targets = targets["label"].reshape(
            batch_size,
            len(ENTITY_IDS),
            -1,
        )
        losses: dict[str, float] = {
            "jepa_latent": float(
                np.mean((forward.predicted_latent - forward.target_latent) ** 2)
            ),
            "label_surrogate": float(
                -np.sum(
                    label_targets * np.log(np.clip(label_probabilities, _EPSILON, 1.0))
                )
                / (batch_size * len(ENTITY_IDS))
            ),
            "simplicial_surrogate": _binary_cross_entropy(
                forward.logits["simplicial"],
                targets["simplicial"],
            ),
            "metric_surrogate": float(
                np.mean((forward.probabilities["metric"] - targets["metric"]) ** 2)
            ),
            "relation_surrogate": _binary_cross_entropy(
                forward.logits["relation"],
                targets["relation"],
            ),
            "order_surrogate": _binary_cross_entropy(
                forward.logits["order"],
                targets["order"],
            ),
            "tracking_surrogate": _binary_cross_entropy(
                forward.logits["tracking"],
                targets["tracking"],
            ),
        }

        bridge_differences = np.stack(
            [
                forward.probabilities["simplicial"][:, simplex_index]
                - forward.probabilities["relation"][:, relation_index]
                for simplex_index, relation_index in self.dataset.bridge_links
            ],
            axis=1,
        )
        losses["bridge_surrogate"] = float(np.mean(bridge_differences**2))
        losses["validity_surrogate"] = float(np.mean(forward.nearest_candidate_scores))
        mask_values = self.mask.as_dict()
        total = sum(losses[name] * mask_values[name] for name in losses)
        if not gradients:
            return TrainingSnapshot(total, MappingProxyType(losses)), None, forward

        d_logits = {head: np.zeros_like(forward.logits[head]) for head in self._HEADS}
        d_predicted = (
            self.mask.jepa_latent
            * 2.0
            * (forward.predicted_latent - forward.target_latent)
            / forward.predicted_latent.size
        )

        label_direct = (label_probabilities - label_targets) / (
            batch_size * len(ENTITY_IDS)
        )
        d_logits["label"] += self.mask.label_surrogate * label_direct.reshape(
            batch_size,
            -1,
        )
        for head, mask_name in (
            ("simplicial", "simplicial_surrogate"),
            ("relation", "relation_surrogate"),
            ("order", "order_surrogate"),
            ("tracking", "tracking_surrogate"),
        ):
            d_logits[head] += (
                getattr(self.mask, mask_name)
                * (forward.probabilities[head] - targets[head])
                / forward.probabilities[head].size
            )
        metric_probabilities = forward.probabilities["metric"]
        d_logits["metric"] += (
            self.mask.metric_surrogate
            * 2.0
            * (metric_probabilities - targets["metric"])
            * metric_probabilities
            * (1.0 - metric_probabilities)
            / metric_probabilities.size
        )

        probability_gradients = {
            head: np.zeros_like(forward.probabilities[head]) for head in self._HEADS
        }
        bridge_normalizer = batch_size * len(self.dataset.bridge_links)
        for link_index, (simplex_index, relation_index) in enumerate(
            self.dataset.bridge_links
        ):
            difference = bridge_differences[:, link_index]
            derivative = (
                self.mask.bridge_surrogate * 2.0 * difference / bridge_normalizer
            )
            probability_gradients["simplicial"][:, simplex_index] += derivative
            probability_gradients["relation"][:, relation_index] -= derivative

        nearest_features = self.dataset.candidate_features[
            forward.nearest_candidate_indices
        ]
        validity_gradient = (
            self.mask.validity_surrogate
            * 2.0
            * (forward.soft_state - nearest_features)
            * self.dataset.coordinate_weights[None, :]
            / batch_size
        )
        for head, block_slice in (
            ("label", slices.label),
            ("simplicial", slices.simplicial),
            ("metric", slices.metric),
            ("relation", slices.relation),
            ("order", slices.order),
        ):
            probability_gradients[head] += validity_gradient[:, block_slice]

        label_probability_gradient = probability_gradients["label"].reshape(
            batch_size,
            len(ENTITY_IDS),
            -1,
        )
        label_projection = np.sum(
            label_probability_gradient * label_probabilities,
            axis=2,
            keepdims=True,
        )
        d_logits["label"] += (
            label_probabilities * (label_probability_gradient - label_projection)
        ).reshape(batch_size, -1)
        for head in ("simplicial", "metric", "relation", "order"):
            probability = forward.probabilities[head]
            d_logits[head] += (
                probability_gradients[head] * probability * (1.0 - probability)
            )

        parameter_gradients = {
            name: np.zeros_like(value) for name, value in self.parameters.items()
        }
        for head in self._HEADS:
            parameter_gradients[f"{head}_weight"] = (
                forward.predicted_latent.T @ d_logits[head]
            )
            parameter_gradients[f"{head}_bias"] = np.sum(
                d_logits[head],
                axis=0,
            )
            d_predicted += d_logits[head] @ self.parameters[f"{head}_weight"].T

        predictor_pre_gradient = d_predicted * (1.0 - forward.predicted_latent**2)
        parameter_gradients["predictor_weight"] = np.einsum(
            "ba,bi,bj->aij",
            split.actions,
            forward.context,
            predictor_pre_gradient,
        )
        parameter_gradients["predictor_bias"] = split.actions.T @ predictor_pre_gradient
        context_gradient = np.einsum(
            "bj,bij->bi",
            predictor_pre_gradient,
            forward.predictor_operators,
        )
        encoder_pre_gradient = context_gradient * (1.0 - forward.context**2)
        parameter_gradients["encoder_weight"] = (
            split.source_inputs.T @ encoder_pre_gradient
        )
        parameter_gradients["encoder_bias"] = np.sum(
            encoder_pre_gradient,
            axis=0,
        )

        gradient_norm = sqrt(
            sum(float(np.sum(gradient**2)) for gradient in parameter_gradients.values())
        )
        if gradient_norm > self.spec.gradient_clip:
            factor = self.spec.gradient_clip / gradient_norm
            parameter_gradients = {
                name: gradient * factor
                for name, gradient in parameter_gradients.items()
            }
        snapshot = TrainingSnapshot(
            total=total,
            losses=MappingProxyType(losses),
            gradient_norm=gradient_norm,
        )
        return snapshot, MappingProxyType(parameter_gradients), forward

    def _learning_rate(self, step: int) -> float:
        progress = (step - 1) / max(1, self.spec.training_steps - 1)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        fraction = self.spec.minimum_learning_rate_fraction
        return self.spec.learning_rate * (fraction + (1.0 - fraction) * cosine)

    def _update(self, gradients: Mapping[str, np.ndarray], step: int) -> None:
        self.update_count += 1
        beta1 = self.spec.adam_beta1
        beta2 = self.spec.adam_beta2
        learning_rate = self._learning_rate(step)
        for name, parameter in self.parameters.items():
            gradient = gradients[name]
            self.adam_first[name] = (
                beta1 * self.adam_first[name] + (1.0 - beta1) * gradient
            )
            self.adam_second[name] = (
                beta2 * self.adam_second[name] + (1.0 - beta2) * gradient**2
            )
            first_hat = self.adam_first[name] / (1.0 - beta1**self.update_count)
            second_hat = self.adam_second[name] / (1.0 - beta2**self.update_count)
            parameter -= (
                learning_rate
                * first_hat
                / (np.sqrt(second_hat) + self.spec.adam_epsilon)
            )
        momentum = self.spec.ema_momentum
        self.target_weight = (
            momentum * self.target_weight
            + (1.0 - momentum) * self.parameters["encoder_weight"]
        )
        self.target_bias = (
            momentum * self.target_bias
            + (1.0 - momentum) * self.parameters["encoder_bias"]
        )

    def fit(self) -> TrainableStructuralJEPA:
        """Run the fixed number of full-batch matched Adam updates."""

        training = self.dataset.splits["train"]
        self.initial_snapshot = self._losses_and_gradients(
            training,
            gradients=False,
        )[0]
        for step in range(1, self.spec.training_steps + 1):
            _, gradients, _ = self._losses_and_gradients(
                training,
                gradients=True,
            )
            assert gradients is not None
            self._update(gradients, step)
        self.final_snapshot = self._losses_and_gradients(
            training,
            gradients=False,
        )[0]
        return self

    def target_embedding(self, target_inputs: np.ndarray) -> np.ndarray:
        """Return stop-gradient target representations."""

        return np.tanh(target_inputs @ self.target_weight + self.target_bias)

    def context_embeddings(self, source_inputs: np.ndarray) -> np.ndarray:
        """Return learned context representations."""

        return np.tanh(
            source_inputs @ self.parameters["encoder_weight"]
            + self.parameters["encoder_bias"]
        )

    def forward_split(self, split: NumericSplit) -> ForwardPass:
        """Return all soft predictions for one numerical split."""

        return self._forward(
            split.source_inputs,
            split.actions,
            split.target_inputs,
        )


@dataclass(frozen=True)
class AblationPrediction:
    """One coherent projected state and its pre-projection diagnostics."""

    target_code: SyntheticStateCode
    target: CoherentStructuralState
    tracking: TrackedTransition
    latent: np.ndarray
    soft_bridge_defect: float
    soft_validity_defect: float
    projection_correction: float


def _decode_tracking(
    source: CoherentStructuralState,
    target: CoherentStructuralState,
    probabilities: np.ndarray,
) -> TrackedTransition:
    source_labels = source.core.relational.label_map(ENTITY_TYPE)
    target_labels = target.core.relational.label_map(ENTITY_TYPE)
    matrix = probabilities.reshape(len(ENTITY_IDS), len(ENTITY_IDS))
    clipped = np.clip(matrix, 1.0e-9, 1.0 - 1.0e-9)
    best_score = -float("inf")
    best_pairs: frozenset[tuple[int, int]] | None = None
    for target_order in permutations(ENTITY_IDS):
        pairs = frozenset(zip(ENTITY_IDS, target_order, strict=True))
        if any(source_labels[left] != target_labels[right] for left, right in pairs):
            continue
        selected = np.zeros_like(matrix)
        for left, right in pairs:
            selected[ENTITY_IDS.index(left), ENTITY_IDS.index(right)] = 1.0
        score = float(
            np.sum(
                selected * np.log(clipped) + (1.0 - selected) * np.log(1.0 - clipped)
            )
        )
        if score > best_score:
            best_score = score
            best_pairs = pairs
    if best_pairs is None:
        raise RuntimeError("no label-preserving total tracking permutation exists")
    return TrackedTransition(
        source=source.core,
        target=target.core,
        components={
            ENTITY_TYPE: PartialBijection(
                source.core.relational.carriers[ENTITY_TYPE],
                target.core.relational.carriers[ENTITY_TYPE],
                best_pairs,
            )
        },
    )


def decode_ablation_predictions(
    model: TrainableStructuralJEPA,
    split_name: str,
) -> tuple[AblationPrediction, ...]:
    """Project soft heads to valid states and label-preserving tracking."""

    if split_name not in SPLIT_NAMES:
        raise ValueError(f"unknown split: {split_name!r}")
    split = model.dataset.splits[split_name]
    forward = model.forward_split(split)
    predictions: list[AblationPrediction] = []
    for index, case in enumerate(split.cases):
        candidate_index = int(forward.nearest_candidate_indices[index])
        code = model.dataset.candidate_codes[candidate_index]
        target = model.dataset.benchmark.states[code]
        relation = forward.probabilities["relation"][index]
        simplicial = forward.probabilities["simplicial"][index]
        bridge_values = np.asarray(
            [
                simplicial[simplex_index] - relation[relation_index]
                for simplex_index, relation_index in model.dataset.bridge_links
            ]
        )
        candidate_features = model.dataset.candidate_features[candidate_index]
        correction = float(
            np.sqrt(np.mean((forward.soft_state[index] - candidate_features) ** 2))
        )
        tracking = _decode_tracking(
            case.example.source,
            target,
            forward.probabilities["tracking"][index],
        )
        predictions.append(
            AblationPrediction(
                target_code=code,
                target=target,
                tracking=tracking,
                latent=forward.predicted_latent[index],
                soft_bridge_defect=float(np.sqrt(np.mean(bridge_values**2))),
                soft_validity_defect=float(forward.nearest_candidate_scores[index]),
                projection_correction=correction,
            )
        )
    return tuple(predictions)
