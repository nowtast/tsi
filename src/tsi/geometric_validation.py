"""Finite audits for Paper 2B ambient and metric-measure geometry."""

from __future__ import annotations

from math import inf, isclose, isfinite, sqrt
from typing import Hashable, Mapping, Sequence

from .geometric import FiniteMetricState


Point = tuple[float, ...]
Matrix = tuple[tuple[float, ...], ...]
Coupling = tuple[tuple[float, ...], ...]
_TOLERANCE = 1e-9


def determinant(matrix: Matrix) -> float:
    """Return a determinant by finite Laplace expansion."""

    size = len(matrix)
    if any(len(row) != size for row in matrix):
        raise ValueError("a determinant requires a square matrix")
    if size == 0:
        return 1.0
    if size == 1:
        return float(matrix[0][0])
    return sum(
        ((-1.0) ** column)
        * matrix[0][column]
        * determinant(
            tuple(
                tuple(row[index] for index in range(size) if index != column)
                for row in matrix[1:]
            )
        )
        for column in range(size)
    )


def is_orthogonal(matrix: Matrix) -> bool:
    size = len(matrix)
    if size == 0 or any(len(row) != size for row in matrix):
        return False
    return all(
        isclose(
            sum(matrix[k][i] * matrix[k][j] for k in range(size)),
            1.0 if i == j else 0.0,
            rel_tol=_TOLERANCE,
            abs_tol=_TOLERANCE,
        )
        for i in range(size)
        for j in range(size)
    )


def is_special_orthogonal(matrix: Matrix) -> bool:
    return is_orthogonal(matrix) and isclose(
        determinant(matrix),
        1.0,
        rel_tol=_TOLERANCE,
        abs_tol=_TOLERANCE,
    )


def apply_rigid_motion(
    points: Mapping[Hashable, Sequence[float]],
    orthogonal: Matrix,
    translation: Sequence[float],
) -> Mapping[Hashable, Point]:
    """Apply ``x -> R x + t`` after checking ``R`` is orthogonal."""

    if not is_orthogonal(orthogonal):
        raise ValueError("the linear part must be orthogonal")
    size = len(orthogonal)
    shift = tuple(float(value) for value in translation)
    if len(shift) != size:
        raise ValueError("translation dimension does not match the matrix")
    normalized = {
        entity: tuple(float(value) for value in point)
        for entity, point in points.items()
    }
    if any(len(point) != size for point in normalized.values()):
        raise ValueError("point dimension does not match the matrix")
    return {
        entity: tuple(
            sum(orthogonal[row][column] * point[column] for column in range(size))
            + shift[row]
            for row in range(size)
        )
        for entity, point in normalized.items()
    }


def ambient_alignment_error(
    source: Mapping[Hashable, Sequence[float]],
    target: Mapping[Hashable, Sequence[float]],
    orthogonal: Matrix,
    translation: Sequence[float],
) -> float:
    """Evaluate the maximum aligned Euclidean point error."""

    if set(source) != set(target):
        raise ValueError("ambient alignment requires the same aligned entities")
    transformed = apply_rigid_motion(source, orthogonal, translation)
    return max(
        sqrt(
            sum(
                (transformed[entity][axis] - float(target[entity][axis])) ** 2
                for axis in range(len(transformed[entity]))
            )
        )
        for entity in source
    )


def pairwise_distances(
    points: Mapping[Hashable, Sequence[float]],
) -> Mapping[tuple[Hashable, Hashable], float]:
    normalized = {
        entity: tuple(float(value) for value in point)
        for entity, point in points.items()
    }
    dimensions = {len(point) for point in normalized.values()}
    if len(dimensions) != 1:
        raise ValueError("all ambient points must have the same dimension")
    return {
        (left, right): sqrt(
            sum(
                (normalized[left][axis] - normalized[right][axis]) ** 2
                for axis in range(next(iter(dimensions)))
            )
        )
        for left in normalized
        for right in normalized
    }


def signed_area_2d(
    first: Sequence[float],
    second: Sequence[float],
    third: Sequence[float],
) -> float:
    points = tuple(tuple(float(value) for value in point) for point in (first, second, third))
    if any(len(point) != 2 for point in points):
        raise ValueError("signed area requires three points in R^2")
    return 0.5 * (
        (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
        - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0])
    )


def validate_probability(
    mass: Sequence[float],
    *,
    full_support: bool,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in mass)
    if not normalized:
        raise ValueError("a probability vector must be nonempty")
    if any(not isfinite(value) or value < 0 for value in normalized):
        raise ValueError("probability masses must be finite and nonnegative")
    if not isclose(sum(normalized), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("probability masses must sum to one")
    if full_support and any(value <= 0 for value in normalized):
        raise ValueError("full support requires every mass to be positive")
    return normalized


def validate_coupling(
    left: FiniteMetricState,
    right: FiniteMetricState,
    left_mass: Sequence[float],
    right_mass: Sequence[float],
    coupling: Coupling,
    *,
    full_support: bool = True,
) -> Coupling:
    """Validate marginals, nonnegativity, and hard label compatibility."""

    left_probability = validate_probability(left_mass, full_support=full_support)
    right_probability = validate_probability(right_mass, full_support=full_support)
    if len(left_probability) != len(left.entities):
        raise ValueError("left probability size does not match its carrier")
    if len(right_probability) != len(right.entities):
        raise ValueError("right probability size does not match its carrier")
    normalized = tuple(tuple(float(value) for value in row) for row in coupling)
    if len(normalized) != len(left.entities) or any(
        len(row) != len(right.entities) for row in normalized
    ):
        raise ValueError("coupling shape does not match the two carriers")
    if any(not isfinite(value) or value < 0 for row in normalized for value in row):
        raise ValueError("coupling entries must be finite and nonnegative")
    for i, expected in enumerate(left_probability):
        if not isclose(sum(normalized[i]), expected, abs_tol=_TOLERANCE):
            raise ValueError("coupling left marginal is incorrect")
    for j, expected in enumerate(right_probability):
        if not isclose(
            sum(normalized[i][j] for i in range(len(left.entities))),
            expected,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError("coupling right marginal is incorrect")
    if any(
        normalized[i][j] > _TOLERANCE and left.labels[i] != right.labels[j]
        for i in range(len(left.entities))
        for j in range(len(right.entities))
    ):
        raise ValueError("positive coupling mass must preserve labels")
    return normalized


def coupling_distortion(
    left: FiniteMetricState,
    right: FiniteMetricState,
    left_mass: Sequence[float],
    right_mass: Sequence[float],
    coupling: Coupling,
    *,
    p: float = 2.0,
    full_support: bool = True,
) -> float:
    """Evaluate the finite hard-label Gromov-Wasserstein distortion objective."""

    exponent = float(p)
    if not isfinite(exponent) or exponent < 1:
        raise ValueError("p must be finite and at least one")
    normalized = validate_coupling(
        left,
        right,
        left_mass,
        right_mass,
        coupling,
        full_support=full_support,
    )
    total = sum(
        abs(left.distances[i][k] - right.distances[j][ell]) ** exponent
        * normalized[i][j]
        * normalized[k][ell]
        for i in range(len(left.entities))
        for j in range(len(right.entities))
        for k in range(len(left.entities))
        for ell in range(len(right.entities))
    )
    return total ** (1.0 / exponent)


def zero_coupling_bijection(
    left: FiniteMetricState,
    right: FiniteMetricState,
    left_mass: Sequence[float],
    right_mass: Sequence[float],
    coupling: Coupling,
    *,
    p: float = 2.0,
) -> Mapping[Hashable, Hashable] | None:
    """Recover the exact measure-preserving isometry certified by a zero coupling."""

    normalized = validate_coupling(
        left,
        right,
        left_mass,
        right_mass,
        coupling,
        full_support=True,
    )
    if not isclose(
        coupling_distortion(
            left,
            right,
            left_mass,
            right_mass,
            normalized,
            p=p,
            full_support=True,
        ),
        0.0,
        abs_tol=_TOLERANCE,
    ):
        return None
    support = {
        (i, j)
        for i in range(len(left.entities))
        for j in range(len(right.entities))
        if normalized[i][j] > _TOLERANCE
    }
    by_left = {
        i: tuple(j for left_index, j in support if left_index == i)
        for i in range(len(left.entities))
    }
    if any(len(targets) != 1 for targets in by_left.values()):
        return None
    index_mapping = {i: targets[0] for i, targets in by_left.items()}
    if len(set(index_mapping.values())) != len(right.entities):
        return None
    return {
        left.entities[i]: right.entities[j]
        for i, j in index_mapping.items()
    }
