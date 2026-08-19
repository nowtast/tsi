"""Observation-level input path for v2 noise and cardinality regimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .paper3_learned_observation import SetObservation
from .paper3_learned_v2_generator import V2TransitionCase
from .paper3_multiworld import MultiworldStateCode, StructuredAction


OBSERVED_INPUT_WIDTH = 31
PIXEL_SIZE = 16
REGIMES = ("exact_state", "noisy_recovered_structure", "pixel_object_observation")


@dataclass(frozen=True)
class PixelTransitionCase:
    partition: str
    graph_variant: str
    image: tuple[tuple[float, ...], ...]
    action: StructuredAction
    source_code: MultiworldStateCode
    target_code: MultiworldStateCode
    intervention: bool

    @property
    def input_key(self) -> tuple[MultiworldStateCode, tuple[int, ...]]:
        return self.source_code, self.action.components


@dataclass(frozen=True)
class ObservedTransitionCase:
    partition: str
    graph_variant: str
    observation: SetObservation
    action: StructuredAction
    source_code: MultiworldStateCode
    target_code: MultiworldStateCode
    intervention: bool

    @classmethod
    def from_v2_case(
        cls,
        case: V2TransitionCase,
        observation: SetObservation,
    ) -> "ObservedTransitionCase":
        return cls(
            partition=case.partition,
            graph_variant=case.graph_variant,
            observation=observation,
            action=case.action,
            source_code=case.source_code,
            target_code=case.target_code,
            intervention=case.intervention,
        )

    @property
    def input_key(self) -> tuple[MultiworldStateCode, tuple[int, ...]]:
        return self.source_code, self.action.components


def _state_entity_features(code: MultiworldStateCode, count: int) -> np.ndarray:
    values = np.asarray(code.as_tuple(), dtype=np.float64)
    normalized = values / np.asarray((2.0, 2.0, 2.0, 3.0, 2.0))
    base = normalized[:4]
    return np.repeat(base[np.newaxis, :], count, axis=0)


def _state_pair_features(code: MultiworldStateCode, count: int) -> np.ndarray:
    values = np.asarray(code.as_tuple(), dtype=np.float64)
    normalized = values / np.asarray((2.0, 2.0, 2.0, 3.0, 2.0))
    pairs = np.zeros((count, count, 2), dtype=np.float64)
    for source in range(count):
        for target in range(count):
            pairs[source, target] = (
                normalized[4],
                normalized[(source + 2 * target + 1) % len(normalized)],
            )
    return pairs


def observation_for_state(
    code: MultiworldStateCode,
    *,
    entity_count: int,
    regime: str = "exact_state",
) -> SetObservation:
    if regime not in REGIMES:
        raise ValueError("unknown observation regime")
    if type(entity_count) is not int or not 2 <= entity_count <= 4:
        raise ValueError("entity_count must be in [2, 4]")
    return _observation_from_arrays(
        regime,
        _state_entity_features(code, entity_count),
        _state_pair_features(code, entity_count),
    )


def _observation_from_arrays(
    regime: str,
    entity_features: np.ndarray,
    pair_features: np.ndarray,
) -> SetObservation:
    from .paper3_learned_observation import observation_from_arrays

    return observation_from_arrays(regime, entity_features, pair_features)


def _rasterize_observation(observation: SetObservation) -> np.ndarray:
    image = np.zeros((PIXEL_SIZE, PIXEL_SIZE), dtype=np.float64)
    order_value = float(observation.pair_features[:, :, 0].mean())
    values = (*observation.entities[0].features, order_value)
    starts = (0, 3, 6, 9, 12)
    for feature_index, value in enumerate(values):
        width = 4 if feature_index == 4 else 3
        x = int(np.clip(starts[feature_index] + round(value * (width - 1)), 0, PIXEL_SIZE - 1))
        y = int(np.clip(round(value * (PIXEL_SIZE - 1)), 0, PIXEL_SIZE - 1))
        image[y, x] = 1.0
    return image


def pixel_case_from_v2_case(
    case: V2TransitionCase,
    *,
    entity_count: int,
    noise: float = 0.0,
    seed: int = 0,
) -> PixelTransitionCase:
    observation = observation_for_state(
        case.source_code,
        entity_count=entity_count,
        regime="pixel_object_observation",
    )
    if noise > 0.0:
        observation = observation.with_gaussian_noise(noise, seed=seed)
    image = _rasterize_observation(observation)
    return PixelTransitionCase(
        partition=case.partition,
        graph_variant=case.graph_variant,
        image=tuple(tuple(float(value) for value in row) for row in image),
        action=case.action,
        source_code=case.source_code,
        target_code=case.target_code,
        intervention=case.intervention,
    )


def encode_pixel_cases(
    cases: Sequence[PixelTransitionCase],
) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("pixel case encoding requires cases")
    rows: list[np.ndarray] = []
    deltas: list[tuple[int, ...]] = []
    widths = (3, 3, 3, 4, 3)
    starts = (0, 3, 6, 9, 12)
    for case in cases:
        image = np.asarray(case.image, dtype=np.float64)
        if image.shape != (PIXEL_SIZE, PIXEL_SIZE):
            raise ValueError("pixel image has an invalid shape")
        source_features: list[float] = []
        for start, width in zip(starts, widths, strict=True):
            region = image[:, start : start + width]
            row_bins = region.reshape(4, 4, width).sum(axis=2)
            source_features.extend(row_bins.sum(axis=1)[:width])
        action_features = np.zeros(15, dtype=np.float64)
        for layer, value in enumerate(case.action.components):
            action_features[layer * 3 + value] = 1.0
        features = np.concatenate(
            (np.asarray(source_features, dtype=np.float64), action_features)
        )
        if len(features) != OBSERVED_INPUT_WIDTH:
            raise RuntimeError("pixel encoder width changed")
        rows.append(features)
        deltas.append(
            tuple(
                (target - source) % (4 if layer == 3 else 3)
                for layer, (source, target) in enumerate(
                    zip(case.source_code.as_tuple(), case.target_code.as_tuple(), strict=True)
                )
            )
        )
    return np.asarray(rows, dtype=np.float64), np.asarray(deltas, dtype=np.int64)


def observed_case_from_v2_case(
    case: V2TransitionCase,
    *,
    entity_count: int,
    regime: str = "exact_state",
    noise: float = 0.0,
    seed: int = 0,
) -> ObservedTransitionCase:
    if regime == "pixel_object_observation":
        return pixel_case_from_v2_case(
            case,
            entity_count=entity_count,
            noise=noise,
            seed=seed,
        )
    observation = observation_for_state(
        case.source_code,
        entity_count=entity_count,
        regime=regime,
    )
    if noise > 0.0:
        observation = observation.with_gaussian_noise(noise, seed=seed)
    return ObservedTransitionCase.from_v2_case(case, observation)


def build_observed_partitions(
    partitions: dict[str, Sequence[V2TransitionCase]],
    *,
    entity_count: int,
    regime: str = "exact_state",
    noise: float = 0.0,
    seed: int = 0,
) -> dict[str, tuple[ObservedTransitionCase, ...]]:
    if noise < 0.0:
        raise ValueError("noise must be nonnegative")
    return {
        partition: tuple(
            observed_case_from_v2_case(
                case,
                entity_count=entity_count,
                regime=regime,
                noise=noise,
                seed=seed + index,
            )
            for index, case in enumerate(cases)
        )
        for partition, cases in partitions.items()
    }


def encode_observed_cases(
    cases: Sequence[ObservedTransitionCase],
) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("observed case encoding requires cases")
    rows: list[np.ndarray] = []
    deltas: list[tuple[int, ...]] = []
    for case in cases:
        entity = case.observation.entity_features
        pair = case.observation.pair_features.reshape(-1, 2)
        entity_summary = np.concatenate(
            (entity.mean(axis=0), entity.std(axis=0), entity.min(axis=0), entity.max(axis=0))
        )
        pair_summary = np.concatenate((pair.mean(axis=0), pair.std(axis=0)))
        del pair_summary
        source_features = entity_summary
        action_features = np.zeros(15, dtype=np.float64)
        for layer, value in enumerate(case.action.components):
            action_features[layer * 3 + value] = 1.0
        row = np.concatenate((source_features, action_features))
        if len(row) != OBSERVED_INPUT_WIDTH:
            raise RuntimeError("observed encoder width changed")
        rows.append(row)
        deltas.append(
            tuple(
                (target - source) % (4 if layer == 3 else 3)
                for layer, (source, target) in enumerate(
                    zip(case.source_code.as_tuple(), case.target_code.as_tuple(), strict=True)
                )
            )
        )
    return np.asarray(rows, dtype=np.float64), np.asarray(deltas, dtype=np.int64)


def corrupt_pixel_case(
    case: PixelTransitionCase,
    *,
    gaussian_noise: float = 0.0,
    dropout_probability: float = 0.0,
    quantization_levels: int | None = None,
    seed: int = 0,
) -> PixelTransitionCase:
    if gaussian_noise < 0.0:
        raise ValueError("gaussian_noise must be nonnegative")
    if not 0.0 <= dropout_probability < 1.0:
        raise ValueError("dropout_probability must lie in [0, 1)")
    if quantization_levels is not None and quantization_levels < 2:
        raise ValueError("quantization_levels must be at least 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    image = np.asarray(case.image, dtype=np.float64)
    if gaussian_noise:
        image = image + rng.normal(0.0, gaussian_noise, image.shape)
    if dropout_probability:
        image = image * (rng.random(image.shape) >= dropout_probability)
    image = np.clip(image, 0.0, 1.0)
    if quantization_levels is not None:
        image = np.round(image * (quantization_levels - 1)) / (quantization_levels - 1)
    return PixelTransitionCase(
        partition=case.partition,
        graph_variant=case.graph_variant,
        image=tuple(tuple(float(value) for value in row) for row in image),
        action=case.action,
        source_code=case.source_code,
        target_code=case.target_code,
        intervention=case.intervention,
    )


def pixel_feature_leakage_audit(
    cases: Sequence[PixelTransitionCase],
) -> dict[str, object]:
    if not cases:
        raise ValueError("leakage audit requires cases")
    features, _deltas = encode_pixel_cases(cases)
    unique_fraction = float(len({tuple(row.round(12)) for row in features}) / len(features))
    centered = features - features.mean(axis=0, keepdims=True)
    rank = int(np.linalg.matrix_rank(centered))
    same_action = {}
    for case, row in zip(cases, features, strict=True):
        same_action.setdefault(case.action.components, []).append(row)
    within_action_distances: list[float] = []
    for rows in same_action.values():
        if len(rows) > 1:
            values = np.asarray(rows)
            distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
            within_action_distances.extend(distances[np.triu_indices(len(values), k=1)])
    return {
        "case_count": len(cases),
        "feature_width": int(features.shape[1]),
        "feature_rank": rank,
        "unique_feature_fraction": unique_fraction,
        "minimum_same_action_pair_distance": (
            float(min(within_action_distances)) if within_action_distances else None
        ),
        "warning": (
            "high uniqueness may permit source-state lookup; this is a diagnostic, "
            "not proof of information leakage"
            if unique_fraction > 0.95
            else "no high-uniqueness warning"
        ),
    }
