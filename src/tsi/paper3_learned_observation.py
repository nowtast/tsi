"""Permutation-safe observation primitives for P3-5A development pilots.

The observation carries sample-local keys only for evaluation and relation
bookkeeping.  Model-facing feature accessors intentionally omit those keys, so
an implementation cannot use a persistent entity index as a shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

import numpy as np


FEATURE_WIDTH = 4
PAIR_FEATURE_WIDTH = 2
MIN_ENTITY_COUNT = 2
MAX_ENTITY_COUNT = 4


def _finite_tuple(values: Sequence[float], *, width: int, name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != width:
        raise ValueError(f"{name} must have width {width}")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class EntityObservation:
    """One object observation with a non-semantic sample-local key."""

    key: int
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.key) is not int or self.key < 0:
            raise ValueError("entity observation keys must be nonnegative integers")
        object.__setattr__(
            self,
            "features",
            _finite_tuple(self.features, width=FEATURE_WIDTH, name="entity features"),
        )


@dataclass(frozen=True)
class PairObservation:
    """A directed pair feature; endpoints are not exposed to model features."""

    source_key: int
    target_key: int
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.source_key) is not int or type(self.target_key) is not int:
            raise ValueError("pair keys must be integers")
        if self.source_key < 0 or self.target_key < 0:
            raise ValueError("pair keys must be nonnegative")
        object.__setattr__(
            self,
            "features",
            _finite_tuple(self.features, width=PAIR_FEATURE_WIDTH, name="pair features"),
        )


@dataclass(frozen=True)
class SetObservation:
    """An unordered object set plus directed pair observations."""

    regime: str
    entities: tuple[EntityObservation, ...]
    pairs: tuple[PairObservation, ...]

    def __post_init__(self) -> None:
        if not self.regime:
            raise ValueError("observation regime must be nonempty")
        if not MIN_ENTITY_COUNT <= len(self.entities) <= MAX_ENTITY_COUNT:
            raise ValueError("entity count is outside the P3-5A development range")
        keys = tuple(entity.key for entity in self.entities)
        if len(set(keys)) != len(keys):
            raise ValueError("entity keys must be unique within one observation")
        key_set = set(keys)
        for pair in self.pairs:
            if pair.source_key not in key_set or pair.target_key not in key_set:
                raise ValueError("pair endpoint is absent from the entity set")
        pair_keys = tuple((pair.source_key, pair.target_key) for pair in self.pairs)
        if len(set(pair_keys)) != len(pair_keys):
            raise ValueError("directed pair keys must be unique")

    @property
    def entity_count(self) -> int:
        return len(self.entities)

    @property
    def entity_features(self) -> np.ndarray:
        """Return model-facing features without sample-local entity keys."""

        return np.asarray([entity.features for entity in self.entities], dtype=np.float64)

    @property
    def pair_features(self) -> np.ndarray:
        """Return the pair matrix in the current presentation order."""

        values = {
            (pair.source_key, pair.target_key): pair.features for pair in self.pairs
        }
        return np.asarray(
            [
                [values[(source.key, target.key)] for target in self.entities]
                for source in self.entities
            ],
            dtype=np.float64,
        )

    def permute(self, presentation_order: Sequence[int]) -> "SetObservation":
        """Change presentation order while preserving the represented structure."""

        order = tuple(presentation_order)
        if tuple(sorted(order)) != tuple(range(len(self.entities))):
            raise ValueError("presentation_order must be a permutation of positions")
        entities = tuple(self.entities[index] for index in order)
        return SetObservation(self.regime, entities, self.pairs)

    def with_gaussian_noise(
        self,
        standard_deviation: float,
        *,
        seed: int,
    ) -> "SetObservation":
        """Add prescribed feature noise without changing keys or cardinality."""

        if standard_deviation <= 0.0 or not isfinite(standard_deviation):
            raise ValueError("standard_deviation must be finite and positive")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        entities = tuple(
            EntityObservation(
                entity.key,
                tuple(
                    np.asarray(entity.features)
                    + rng.normal(0.0, standard_deviation, FEATURE_WIDTH)
                ),
            )
            for entity in self.entities
        )
        pairs = tuple(
            PairObservation(
                pair.source_key,
                pair.target_key,
                tuple(
                    np.asarray(pair.features)
                    + rng.normal(0.0, standard_deviation, PAIR_FEATURE_WIDTH)
                ),
            )
            for pair in self.pairs
        )
        return SetObservation(self.regime, entities, pairs)


def observation_from_arrays(
    regime: str,
    entity_features: np.ndarray,
    pair_features: np.ndarray,
) -> SetObservation:
    """Build a keyed observation while keeping keys out of model features."""

    entities_array = np.asarray(entity_features, dtype=np.float64)
    pairs_array = np.asarray(pair_features, dtype=np.float64)
    if entities_array.ndim != 2 or entities_array.shape[1] != FEATURE_WIDTH:
        raise ValueError("entity_features must have shape (n, 4)")
    if pairs_array.ndim != 3 or pairs_array.shape[2] != PAIR_FEATURE_WIDTH:
        raise ValueError("pair_features must have shape (n, n, 2)")
    if pairs_array.shape[:2] != (len(entities_array), len(entities_array)):
        raise ValueError("pair_features must have one matrix entry per entity pair")
    entities = tuple(
        EntityObservation(index, tuple(row))
        for index, row in enumerate(entities_array)
    )
    pairs = tuple(
        PairObservation(source, target, tuple(pairs_array[source, target]))
        for source in range(len(entities))
        for target in range(len(entities))
    )
    return SetObservation(regime, entities, pairs)
