"""Finite reference implementation for the geometry developed in Paper 2B.

The correspondence search is exact and deliberately exponential. It is intended
for theorem checks and small synthetic fixtures, not large-scale training.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import inf, isclose, isfinite
from typing import Hashable, Iterator, Mapping


Entity = Hashable
Label = Hashable
Correspondence = frozenset[tuple[int, int]]
_METRIC_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FiniteMetricState:
    """A nonempty finite metric carrier with one label per entity."""

    entities: tuple[Entity, ...]
    distances: tuple[tuple[float, ...], ...]
    labels: tuple[Label, ...]

    def __post_init__(self) -> None:
        entities = tuple(self.entities)
        labels = tuple(self.labels)
        distances = tuple(tuple(float(value) for value in row) for row in self.distances)
        object.__setattr__(self, "entities", entities)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "distances", distances)

        size = len(entities)
        if size == 0:
            raise ValueError("a finite metric state must be nonempty")
        if len(set(entities)) != size:
            raise ValueError("entities must be unique")
        if len(labels) != size:
            raise ValueError("labels must have one entry per entity")
        if len(distances) != size or any(len(row) != size for row in distances):
            raise ValueError("distance matrix must be square and match the entity count")

        for i in range(size):
            for j in range(size):
                value = distances[i][j]
                if not isfinite(value) or value < 0:
                    raise ValueError("distances must be finite and nonnegative")
                if i == j and not isclose(value, 0.0, abs_tol=_METRIC_TOLERANCE):
                    raise ValueError("distance matrix must have a zero diagonal")
                if i != j and value <= 0:
                    raise ValueError("distinct entities must have positive distance")
                if not isclose(
                    value,
                    distances[j][i],
                    rel_tol=_METRIC_TOLERANCE,
                    abs_tol=_METRIC_TOLERANCE,
                ):
                    raise ValueError("distance matrix must be symmetric")

        for i in range(size):
            for j in range(size):
                for k in range(size):
                    if distances[i][k] > distances[i][j] + distances[j][k] + _METRIC_TOLERANCE:
                        raise ValueError("distance matrix violates the triangle inequality")

    @property
    def diameter(self) -> float:
        """Return the maximum pairwise distance."""

        return max(max(row) for row in self.distances)


def _nonempty_subsets(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    )


def label_compatible_correspondences(
    left: FiniteMetricState,
    right: FiniteMetricState,
    *,
    max_compatible_pairs: int = 20,
) -> Iterator[Correspondence]:
    """Enumerate every label-compatible correspondence between two small states."""

    allowed_by_left = tuple(
        tuple(j for j, right_label in enumerate(right.labels) if right_label == left_label)
        for left_label in left.labels
    )
    if any(not allowed for allowed in allowed_by_left):
        return

    covered_right = {
        j
        for allowed in allowed_by_left
        for j in allowed
    }
    if len(covered_right) != len(right.entities):
        return

    compatible_pair_count = sum(len(allowed) for allowed in allowed_by_left)
    if compatible_pair_count > max_compatible_pairs:
        raise ValueError(
            "exact correspondence enumeration is restricted to small states; "
            f"found {compatible_pair_count} compatible pairs"
        )

    choices = tuple(_nonempty_subsets(allowed) for allowed in allowed_by_left)
    required_right = set(range(len(right.entities)))
    for selected_by_left in product(*choices):
        relation = frozenset(
            (i, j)
            for i, selected in enumerate(selected_by_left)
            for j in selected
        )
        if {j for _, j in relation} == required_right:
            yield relation


def correspondence_distortion(
    left: FiniteMetricState,
    right: FiniteMetricState,
    correspondence: Correspondence,
) -> float:
    """Compute distortion after validating correspondence and label coverage."""

    left_covered = {i for i, _ in correspondence}
    right_covered = {j for _, j in correspondence}
    if left_covered != set(range(len(left.entities))):
        raise ValueError("correspondence does not cover the left carrier")
    if right_covered != set(range(len(right.entities))):
        raise ValueError("correspondence does not cover the right carrier")
    if any(left.labels[i] != right.labels[j] for i, j in correspondence):
        raise ValueError("correspondence is not label-compatible")

    return max(
        abs(left.distances[i][k] - right.distances[j][ell])
        for i, j in correspondence
        for k, ell in correspondence
    )


def geometric_discrepancy(
    left: FiniteMetricState,
    right: FiniteMetricState,
    *,
    max_compatible_pairs: int = 20,
) -> float:
    """Compute Paper 2B's exact label-compatible discrepancy ``Delta_g``."""

    best = inf
    for correspondence in label_compatible_correspondences(
        left,
        right,
        max_compatible_pairs=max_compatible_pairs,
    ):
        best = min(best, correspondence_distortion(left, right, correspondence))
    return best


def find_label_preserving_isometry(
    left: FiniteMetricState,
    right: FiniteMetricState,
) -> Mapping[Entity, Entity] | None:
    """Return a label-preserving isometry when one exists."""

    size = len(left.entities)
    if size != len(right.entities):
        return None

    for candidate in permutations(range(size)):
        if any(left.labels[i] != right.labels[candidate[i]] for i in range(size)):
            continue
        if all(
            isclose(
                left.distances[i][j],
                right.distances[candidate[i]][candidate[j]],
                rel_tol=_METRIC_TOLERANCE,
                abs_tol=_METRIC_TOLERANCE,
            )
            for i in range(size)
            for j in range(size)
        ):
            return {
                left.entities[i]: right.entities[candidate[i]]
                for i in range(size)
            }
    return None
