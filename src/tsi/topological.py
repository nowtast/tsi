"""Exact finite audits for simplicial homology and one-parameter filtrations."""

from __future__ import annotations

from itertools import permutations
from math import inf, isclose, isfinite
from typing import Hashable, Mapping


Vertex = Hashable
Simplex = frozenset[Vertex]
Complex = frozenset[Simplex]
Barcode = tuple[tuple[float, float], ...]
_TOLERANCE = 1e-9


def validate_complex(complex_: Complex) -> Complex:
    """Normalize and validate a finite abstract simplicial complex."""

    normalized = frozenset(frozenset(simplex) for simplex in complex_)
    if frozenset() not in normalized:
        raise ValueError("a simplicial complex must contain the empty simplex")
    for simplex in normalized:
        for vertex in simplex:
            if frozenset((vertex,)) not in normalized:
                raise ValueError("every used vertex must occur as a singleton")
        members = tuple(simplex)
        for mask in range(1 << len(members)):
            face = frozenset(
                members[index]
                for index in range(len(members))
                if mask & (1 << index)
            )
            if face not in normalized:
                raise ValueError("a simplicial complex must be downward closed")
    return normalized


def simplices_of_dimension(complex_: Complex, dimension: int) -> tuple[Simplex, ...]:
    normalized = validate_complex(complex_)
    if dimension < 0:
        return (frozenset(),) if dimension == -1 else ()
    return tuple(
        sorted(
            (simplex for simplex in normalized if len(simplex) == dimension + 1),
            key=lambda simplex: tuple(sorted(map(repr, simplex))),
        )
    )


def gf2_rank(rows: tuple[tuple[int, ...], ...]) -> int:
    """Return matrix rank over the field with two elements."""

    if not rows:
        return 0
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix rows must have equal length")
    matrix = [
        sum((value % 2) << column for column, value in enumerate(row))
        for row in rows
    ]
    rank = 0
    for column in range(width):
        pivot = next(
            (index for index in range(rank, len(matrix)) if matrix[index] >> column & 1),
            None,
        )
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        for index in range(len(matrix)):
            if index != rank and matrix[index] >> column & 1:
                matrix[index] ^= matrix[rank]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def boundary_rank(complex_: Complex, dimension: int) -> int:
    """Return the rank of the simplicial boundary ``C_k -> C_{k-1}`` over F2."""

    if dimension <= 0:
        return 0
    columns = simplices_of_dimension(complex_, dimension)
    rows = simplices_of_dimension(complex_, dimension - 1)
    row_index = {simplex: index for index, simplex in enumerate(rows)}
    matrix = [[0 for _ in columns] for _ in rows]
    for column, simplex in enumerate(columns):
        for vertex in simplex:
            face = frozenset(set(simplex) - {vertex})
            matrix[row_index[face]][column] = 1
    return gf2_rank(tuple(tuple(row) for row in matrix))


def betti_numbers(
    complex_: Complex,
    *,
    max_dimension: int | None = None,
) -> tuple[int, ...]:
    """Return the zero-padded Betti vector over F2 through ``max_dimension``."""

    normalized = validate_complex(complex_)
    actual_dimension = max((len(simplex) - 1 for simplex in normalized), default=-1)
    maximum = actual_dimension if max_dimension is None else max_dimension
    if maximum < 0:
        return ()
    counts = {
        dimension: len(simplices_of_dimension(normalized, dimension))
        for dimension in range(maximum + 2)
    }
    ranks = {
        dimension: boundary_rank(normalized, dimension)
        for dimension in range(1, maximum + 2)
    }
    return tuple(
        counts[dimension]
        - ranks.get(dimension, 0)
        - ranks.get(dimension + 1, 0)
        for dimension in range(maximum + 1)
    )


def validate_filtration(
    complex_: Complex,
    values: Mapping[Simplex, float],
) -> Mapping[Simplex, float]:
    """Validate a finite real-valued filtration function."""

    normalized = validate_complex(complex_)
    filtration = {frozenset(simplex): float(value) for simplex, value in values.items()}
    if set(filtration) != set(normalized):
        raise ValueError("a filtration value is required for every simplex")
    if any(not isfinite(value) for value in filtration.values()):
        raise ValueError("filtration values must be finite")
    for simplex in normalized:
        for face in normalized:
            if face.issubset(simplex) and filtration[face] > filtration[simplex]:
                raise ValueError("filtration values must be monotone under inclusion")
    return filtration


def sublevel_complex(
    complex_: Complex,
    values: Mapping[Simplex, float],
    threshold: float,
) -> Complex:
    filtration = validate_filtration(complex_, values)
    return frozenset(
        simplex for simplex, value in filtration.items() if value <= threshold
    )


def filtration_interleaving_audit(
    complex_: Complex,
    left: Mapping[Simplex, float],
    right: Mapping[Simplex, float],
) -> tuple[float, bool]:
    """Return the sup distance and verify both simplex-level shift inclusions."""

    left_values = validate_filtration(complex_, left)
    right_values = validate_filtration(complex_, right)
    delta = max(
        abs(left_values[simplex] - right_values[simplex])
        for simplex in left_values
    )
    left_to_right = all(
        right_values[simplex] <= left_values[simplex] + delta + _TOLERANCE
        for simplex in left_values
    )
    right_to_left = all(
        left_values[simplex] <= right_values[simplex] + delta + _TOLERANCE
        for simplex in left_values
    )
    return delta, left_to_right and right_to_left


def zero_dimensional_barcode(
    complex_: Complex,
    values: Mapping[Simplex, float],
) -> Barcode:
    """Compute the ordinary H0 barcode by the elder rule."""

    normalized = validate_complex(complex_)
    filtration = validate_filtration(normalized, values)
    vertices = simplices_of_dimension(normalized, 0)
    edges = simplices_of_dimension(normalized, 1)
    vertex_of = {next(iter(simplex)): simplex for simplex in vertices}
    events = sorted(
        (
            *((filtration[vertex], 0, vertex) for vertex in vertices),
            *((filtration[edge], 1, edge) for edge in edges),
        ),
        key=lambda item: (
            item[0],
            item[1],
            tuple(sorted(map(repr, item[2]))),
        ),
    )
    parent: dict[Vertex, Vertex] = {}
    birth: dict[Vertex, float] = {}
    bars: list[tuple[float, float]] = []

    def find(value: Vertex) -> Vertex:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for time, dimension, simplex in events:
        if dimension == 0:
            value = next(iter(simplex))
            parent[value] = value
            birth[value] = filtration[vertex_of[value]]
            continue
        first, second = tuple(simplex)
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            continue
        first_key = (birth[first_root], repr(first_root))
        second_key = (birth[second_root], repr(second_root))
        elder, younger = (
            (first_root, second_root)
            if first_key <= second_key
            else (second_root, first_root)
        )
        bars.append((birth[younger], time))
        parent[younger] = elder

    roots = {find(value) for value in parent}
    bars.extend((birth[root], inf) for root in roots)
    return tuple(sorted(bars, key=lambda bar: (bar[0], bar[1])))


def _point_cost(left: tuple[float, float], right: tuple[float, float]) -> float:
    left_essential = not isfinite(left[1])
    right_essential = not isfinite(right[1])
    if left_essential != right_essential:
        return inf
    if left_essential:
        return abs(left[0] - right[0])
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _diagonal_cost(point: tuple[float, float]) -> float:
    if not isfinite(point[1]):
        return inf
    return (point[1] - point[0]) / 2.0


def bottleneck_distance(left: Barcode, right: Barcode) -> float:
    """Compute exact bottleneck distance for small barcodes by full matching."""

    left = tuple((float(birth), float(death)) for birth, death in left)
    right = tuple((float(birth), float(death)) for birth, death in right)
    size = len(left) + len(right)
    costs = [[0.0 for _ in range(size)] for _ in range(size)]
    for i, left_point in enumerate(left):
        for j, right_point in enumerate(right):
            costs[i][j] = _point_cost(left_point, right_point)
        for j in range(len(right), size):
            costs[i][j] = _diagonal_cost(left_point)
    for i, right_diagonal in enumerate(right, start=len(left)):
        for j, right_point in enumerate(right):
            costs[i][j] = _diagonal_cost(right_point)
        for j in range(len(right), size):
            costs[i][j] = 0.0
    return min(
        max(costs[row][column] for row, column in enumerate(matching))
        for matching in permutations(range(size))
    )
