"""Exact finite audits for TSI Extension 2B-X1.

The module separates four claims: finite metric spaces are not geodesic,
positive weighted graphs have compact geodesic metric realizations, weighted
graph isomorphisms induce realization isometries, and lazy Ollivier--Ricci
curvature is invariant and quantitatively stable.  The transport solver is an
exact finite reference routine up to floating-point arithmetic; it is intended
for theorem checks on small graphs, not large-scale optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations
from math import inf, isclose, isfinite
from types import MappingProxyType
from typing import Hashable, Mapping, TypeAlias

from .geometric import FiniteMetricState


Vertex = Hashable
Edge: TypeAlias = frozenset[Vertex]
Measure: TypeAlias = Mapping[Vertex, float]
_TOLERANCE = 1e-9


def _normalize_edge_key(key: object) -> Edge:
    if isinstance(key, frozenset):
        edge = key
    elif isinstance(key, tuple) and len(key) == 2:
        edge = frozenset(key)
    else:
        raise ValueError("each edge key must be a two-element tuple or frozenset")
    if len(edge) != 2:
        raise ValueError("loops and degenerate edges are not allowed")
    return edge


@dataclass(frozen=True)
class PositiveWeightedGraph:
    """A connected finite simple graph with strictly positive edge lengths."""

    vertices: tuple[Vertex, ...]
    edge_lengths: Mapping[object, float]
    _vertex_index: Mapping[Vertex, int] = field(init=False, repr=False)
    _adjacency: Mapping[Vertex, Mapping[Vertex, float]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        vertices = tuple(self.vertices)
        if len(vertices) < 2:
            raise ValueError("the curvature domain requires at least two vertices")
        try:
            unique_count = len(set(vertices))
        except TypeError as error:
            raise ValueError("vertices must be hashable") from error
        if unique_count != len(vertices):
            raise ValueError("vertices must be unique")

        vertex_set = set(vertices)
        normalized: dict[Edge, float] = {}
        for raw_edge, raw_length in self.edge_lengths.items():
            edge = _normalize_edge_key(raw_edge)
            if not edge <= vertex_set:
                raise ValueError("every edge endpoint must belong to the carrier")
            if edge in normalized:
                raise ValueError("the edge mapping contains a duplicate undirected edge")
            length = float(raw_length)
            if not isfinite(length) or length <= 0:
                raise ValueError("edge lengths must be finite and strictly positive")
            normalized[edge] = length
        if not normalized:
            raise ValueError("a connected nontrivial graph must have an edge")

        adjacency: dict[Vertex, dict[Vertex, float]] = {
            vertex: {} for vertex in vertices
        }
        for edge, length in normalized.items():
            left, right = tuple(edge)
            adjacency[left][right] = length
            adjacency[right][left] = length

        reached = {vertices[0]}
        frontier = [vertices[0]]
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    frontier.append(neighbor)
        if reached != vertex_set:
            raise ValueError("the weighted graph must be connected")

        index = {vertex: position for position, vertex in enumerate(vertices)}
        frozen_adjacency = {
            vertex: MappingProxyType(dict(neighbors))
            for vertex, neighbors in adjacency.items()
        }
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(
            self,
            "edge_lengths",
            MappingProxyType(dict(normalized)),
        )
        object.__setattr__(self, "_vertex_index", MappingProxyType(index))
        object.__setattr__(
            self,
            "_adjacency",
            MappingProxyType(frozen_adjacency),
        )

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self.edge_lengths)

    @property
    def adjacency(self) -> Mapping[Vertex, Mapping[Vertex, float]]:
        return self._adjacency

    def canonical_endpoints(self, edge: object) -> tuple[Vertex, Vertex]:
        normalized = _normalize_edge_key(edge)
        if normalized not in self.edge_lengths:
            raise ValueError("the requested edge is not in the graph")
        left, right = tuple(normalized)
        if self._vertex_index[left] < self._vertex_index[right]:
            return left, right
        return right, left

    def edge_length(self, edge: object) -> float:
        normalized = _normalize_edge_key(edge)
        try:
            return self.edge_lengths[normalized]
        except KeyError as error:
            raise ValueError("the requested edge is not in the graph") from error

    def degree(self, vertex: Vertex) -> int:
        try:
            return len(self.adjacency[vertex])
        except KeyError as error:
            raise ValueError("the vertex is not in the graph") from error


def shortest_vertex_path(
    graph: PositiveWeightedGraph,
    source: Vertex,
    target: Vertex,
) -> tuple[Vertex, ...]:
    """Return a minimum-length simple vertex path by Dijkstra's algorithm."""

    if source not in graph.adjacency or target not in graph.adjacency:
        raise ValueError("source and target must be graph vertices")
    if source == target:
        return (source,)

    distances: dict[Vertex, float] = {vertex: inf for vertex in graph.vertices}
    predecessors: dict[Vertex, Vertex] = {}
    distances[source] = 0.0
    queue: list[tuple[float, int, Vertex]] = [(0.0, 0, source)]
    serial = 1
    settled: set[Vertex] = set()

    while queue:
        distance, _, current = heappop(queue)
        if current in settled:
            continue
        settled.add(current)
        if current == target:
            break
        for neighbor, length in graph.adjacency[current].items():
            candidate = distance + length
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                predecessors[neighbor] = current
                heappush(queue, (candidate, serial, neighbor))
                serial += 1

    if target not in predecessors:
        raise RuntimeError("connectedness validation and shortest-path search disagree")
    reversed_path = [target]
    while reversed_path[-1] != source:
        reversed_path.append(predecessors[reversed_path[-1]])
    return tuple(reversed(reversed_path))


def vertex_path_length(
    graph: PositiveWeightedGraph,
    path: tuple[Vertex, ...],
) -> float:
    if not path:
        raise ValueError("a vertex path must be nonempty")
    if any(vertex not in graph.adjacency for vertex in path):
        raise ValueError("a vertex path must stay in the graph")
    total = 0.0
    for left, right in zip(path, path[1:]):
        edge = frozenset((left, right))
        if edge not in graph.edge_lengths:
            raise ValueError("consecutive path vertices must be adjacent")
        total += graph.edge_lengths[edge]
    return total


def vertex_distance(
    graph: PositiveWeightedGraph,
    source: Vertex,
    target: Vertex,
) -> float:
    return vertex_path_length(graph, shortest_vertex_path(graph, source, target))


def all_pairs_vertex_distances(
    graph: PositiveWeightedGraph,
) -> Mapping[tuple[Vertex, Vertex], float]:
    return MappingProxyType(
        {
            (source, target): vertex_distance(graph, source, target)
            for source in graph.vertices
            for target in graph.vertices
        }
    )


@dataclass(frozen=True)
class VertexPoint:
    vertex: Vertex


@dataclass(frozen=True)
class InteriorPoint:
    """An edge-interior point, measured from the graph's canonical endpoint."""

    edge: Edge
    offset: float


RealizationPoint: TypeAlias = VertexPoint | InteriorPoint


def interior_point(
    graph: PositiveWeightedGraph,
    left: Vertex,
    right: Vertex,
    offset_from_left: float,
) -> RealizationPoint:
    """Construct a canonical realization point on one closed edge interval."""

    edge = frozenset((left, right))
    length = graph.edge_length(edge)
    offset = float(offset_from_left)
    if not isfinite(offset) or offset < 0 or offset > length:
        raise ValueError("the edge coordinate must lie in the closed edge interval")
    canonical_left, canonical_right = graph.canonical_endpoints(edge)
    canonical_offset = offset if left == canonical_left else length - offset
    if canonical_offset == 0.0:
        return VertexPoint(canonical_left)
    if canonical_offset == length:
        return VertexPoint(canonical_right)
    return InteriorPoint(edge, canonical_offset)


def _validate_point(
    graph: PositiveWeightedGraph,
    point: RealizationPoint,
) -> None:
    if isinstance(point, VertexPoint):
        if point.vertex not in graph.adjacency:
            raise ValueError("a realization vertex point must belong to the graph")
        return
    if not isinstance(point, InteriorPoint):
        raise TypeError("a realization point must be a VertexPoint or InteriorPoint")
    length = graph.edge_length(point.edge)
    if (
        not isfinite(point.offset)
        or point.offset <= 0
        or point.offset >= length
    ):
        raise ValueError("an interior coordinate must be strictly inside its edge")


def _endpoint_attachments(
    graph: PositiveWeightedGraph,
    point: RealizationPoint,
) -> tuple[tuple[Vertex, float], ...]:
    _validate_point(graph, point)
    if isinstance(point, VertexPoint):
        return ((point.vertex, 0.0),)
    left, right = graph.canonical_endpoints(point.edge)
    length = graph.edge_length(point.edge)
    return ((left, point.offset), (right, length - point.offset))


@dataclass(frozen=True)
class RealizationGeodesicWitness:
    """Finite data specifying a shortest path in the graph realization."""

    start: RealizationPoint
    end: RealizationPoint
    length: float
    vertex_path: tuple[Vertex, ...]
    start_attachment: Vertex | None
    end_attachment: Vertex | None
    direct_same_edge: bool


def shortest_realization_path(
    graph: PositiveWeightedGraph,
    start: RealizationPoint,
    end: RealizationPoint,
) -> RealizationGeodesicWitness:
    """Return an attained shortest-path witness between arbitrary realization points."""

    start_attachments = _endpoint_attachments(graph, start)
    end_attachments = _endpoint_attachments(graph, end)
    candidates: list[RealizationGeodesicWitness] = []

    if isinstance(start, InteriorPoint) and isinstance(end, InteriorPoint):
        if start.edge == end.edge:
            candidates.append(
                RealizationGeodesicWitness(
                    start=start,
                    end=end,
                    length=abs(start.offset - end.offset),
                    vertex_path=(),
                    start_attachment=None,
                    end_attachment=None,
                    direct_same_edge=True,
                )
            )
    elif start == end:
        candidates.append(
            RealizationGeodesicWitness(
                start=start,
                end=end,
                length=0.0,
                vertex_path=(),
                start_attachment=None,
                end_attachment=None,
                direct_same_edge=True,
            )
        )

    for start_vertex, start_cost in start_attachments:
        for end_vertex, end_cost in end_attachments:
            path = shortest_vertex_path(graph, start_vertex, end_vertex)
            length = start_cost + vertex_path_length(graph, path) + end_cost
            candidates.append(
                RealizationGeodesicWitness(
                    start=start,
                    end=end,
                    length=length,
                    vertex_path=path,
                    start_attachment=start_vertex,
                    end_attachment=end_vertex,
                    direct_same_edge=False,
                )
            )
    return min(candidates, key=lambda candidate: candidate.length)


def realization_distance(
    graph: PositiveWeightedGraph,
    start: RealizationPoint,
    end: RealizationPoint,
) -> float:
    return shortest_realization_path(graph, start, end).length


@dataclass(frozen=True)
class FiniteGeodesicObstruction:
    """A missing radial parameter required by any putative geodesic segment."""

    start: Vertex
    end: Vertex
    endpoint_distance: float
    missing_parameter: float
    radial_distances: tuple[float, ...]


def finite_metric_geodesic_obstruction(
    state: FiniteMetricState,
    start: Vertex,
    end: Vertex,
) -> FiniteGeodesicObstruction:
    """Construct a parameter that no geodesic from ``start`` to ``end`` can realize."""

    try:
        start_index = state.entities.index(start)
        end_index = state.entities.index(end)
    except ValueError as error:
        raise ValueError("start and end must be entities of the metric state") from error
    if start_index == end_index:
        raise ValueError("the obstruction requires distinct endpoints")

    endpoint_distance = state.distances[start_index][end_index]
    radial = tuple(
        sorted(
            {
                distance
                for distance in state.distances[start_index]
                if 0.0 <= distance <= endpoint_distance
            }
        )
    )
    levels = tuple(sorted(set(radial) | {0.0, endpoint_distance}))
    for left, right in zip(levels, levels[1:]):
        if right > left:
            missing = (left + right) / 2.0
            return FiniteGeodesicObstruction(
                start=start,
                end=end,
                endpoint_distance=endpoint_distance,
                missing_parameter=missing,
                radial_distances=radial,
            )
    raise RuntimeError("a finite set of radial distances cannot fill a nonzero interval")


def complete_metric_realization_graph(
    state: FiniteMetricState,
) -> PositiveWeightedGraph:
    """Return the canonical complete weighted graph preserving ``state`` exactly."""

    if len(state.entities) < 2:
        raise ValueError("a nontrivial finite metric state is required")
    index = {entity: position for position, entity in enumerate(state.entities)}
    edge_lengths = {
        (left, right): state.distances[index[left]][index[right]]
        for left, right in combinations(state.entities, 2)
    }
    return PositiveWeightedGraph(state.entities, edge_lengths)


def is_length_preserving_graph_isomorphism(
    source: PositiveWeightedGraph,
    target: PositiveWeightedGraph,
    vertex_map: Mapping[Vertex, Vertex],
) -> bool:
    mapping = dict(vertex_map)
    if set(mapping) != set(source.vertices):
        return False
    if set(mapping.values()) != set(target.vertices):
        return False
    if len(set(mapping.values())) != len(mapping):
        return False

    image_edges = {
        frozenset((mapping[left], mapping[right]))
        for edge in source.edges
        for left, right in (source.canonical_endpoints(edge),)
    }
    if image_edges != set(target.edges):
        return False
    return all(
        source.edge_lengths[edge]
        == target.edge_lengths[
            frozenset(mapping[vertex] for vertex in edge)
        ]
        for edge in source.edges
    )


def induced_realization_map(
    source: PositiveWeightedGraph,
    target: PositiveWeightedGraph,
    vertex_map: Mapping[Vertex, Vertex],
    point: RealizationPoint,
) -> RealizationPoint:
    """Extend a weighted graph isomorphism linearly over every edge."""

    if not is_length_preserving_graph_isomorphism(source, target, vertex_map):
        raise ValueError("the vertex map must be a length-preserving graph isomorphism")
    mapping = dict(vertex_map)
    _validate_point(source, point)
    if isinstance(point, VertexPoint):
        return VertexPoint(mapping[point.vertex])
    left, right = source.canonical_endpoints(point.edge)
    return interior_point(
        target,
        mapping[left],
        mapping[right],
        point.offset,
    )


def is_reduced_graph(graph: PositiveWeightedGraph) -> bool:
    """Return whether the chosen vertex set has no metrically invisible degree-two vertex."""

    return all(graph.degree(vertex) != 2 for vertex in graph.vertices)


def lazy_neighbor_measure(
    graph: PositiveWeightedGraph,
    vertex: Vertex,
    *,
    idleness: float = 0.5,
) -> Measure:
    """Return alpha delta_x plus the uniform neighbor measure of mass 1-alpha."""

    alpha = float(idleness)
    if not isfinite(alpha) or alpha < 0 or alpha > 1:
        raise ValueError("idleness must lie in the closed unit interval")
    if vertex not in graph.adjacency:
        raise ValueError("the measure center must be a graph vertex")
    degree = graph.degree(vertex)
    neighbor_mass = (1.0 - alpha) / degree
    measure = {candidate: 0.0 for candidate in graph.vertices}
    measure[vertex] += alpha
    for neighbor in graph.adjacency[vertex]:
        measure[neighbor] += neighbor_mass
    return MappingProxyType(measure)


def _validate_probability_measure(
    graph: PositiveWeightedGraph,
    measure: Measure,
) -> dict[Vertex, float]:
    if any(vertex not in graph.adjacency for vertex in measure):
        raise ValueError("a transport measure must be supported on graph vertices")
    normalized = {vertex: float(measure.get(vertex, 0.0)) for vertex in graph.vertices}
    if any(not isfinite(mass) or mass < 0 for mass in normalized.values()):
        raise ValueError("transport masses must be finite and nonnegative")
    if not isclose(sum(normalized.values()), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("each transport measure must have total mass one")
    return normalized


@dataclass
class _ResidualEdge:
    target: int
    reverse: int
    capacity: float
    cost: float


def _add_residual_edge(
    network: list[list[_ResidualEdge]],
    source: int,
    target: int,
    capacity: float,
    cost: float,
) -> None:
    forward = _ResidualEdge(target, len(network[target]), capacity, cost)
    reverse = _ResidualEdge(source, len(network[source]), 0.0, -cost)
    network[source].append(forward)
    network[target].append(reverse)


def wasserstein_1(
    graph: PositiveWeightedGraph,
    left_measure: Measure,
    right_measure: Measure,
) -> float:
    """Compute finite ``W_1`` by a min-cost transport flow."""

    left = _validate_probability_measure(graph, left_measure)
    right = _validate_probability_measure(graph, right_measure)
    left_support = tuple(vertex for vertex in graph.vertices if left[vertex] > _TOLERANCE)
    right_support = tuple(vertex for vertex in graph.vertices if right[vertex] > _TOLERANCE)
    distances = all_pairs_vertex_distances(graph)

    source = 0
    left_offset = 1
    right_offset = left_offset + len(left_support)
    sink = right_offset + len(right_support)
    network: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]

    for index, vertex in enumerate(left_support):
        _add_residual_edge(
            network,
            source,
            left_offset + index,
            left[vertex],
            0.0,
        )
    for left_index, left_vertex in enumerate(left_support):
        for right_index, right_vertex in enumerate(right_support):
            _add_residual_edge(
                network,
                left_offset + left_index,
                right_offset + right_index,
                1.0,
                distances[(left_vertex, right_vertex)],
            )
    for index, vertex in enumerate(right_support):
        _add_residual_edge(
            network,
            right_offset + index,
            sink,
            right[vertex],
            0.0,
        )

    flow = 0.0
    total_cost = 0.0
    node_count = len(network)
    while flow < 1.0 - _TOLERANCE:
        best = [inf] * node_count
        predecessor: list[tuple[int, int] | None] = [None] * node_count
        best[source] = 0.0
        for _ in range(node_count - 1):
            changed = False
            for node, edges in enumerate(network):
                if best[node] == inf:
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= _TOLERANCE:
                        continue
                    candidate = best[node] + edge.cost
                    if candidate < best[edge.target]:
                        best[edge.target] = candidate
                        predecessor[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if predecessor[sink] is None:
            raise RuntimeError("the finite transport network has no augmenting path")

        augmentation = 1.0 - flow
        current = sink
        while current != source:
            previous, edge_index = predecessor[current]  # type: ignore[misc]
            augmentation = min(
                augmentation,
                network[previous][edge_index].capacity,
            )
            current = previous

        current = sink
        while current != source:
            previous, edge_index = predecessor[current]  # type: ignore[misc]
            edge = network[previous][edge_index]
            reverse_index = edge.reverse
            edge.capacity -= augmentation
            network[current][reverse_index].capacity += augmentation
            current = previous
        flow += augmentation
        total_cost += augmentation * best[sink]

    return total_cost


def ollivier_ricci_curvature(
    graph: PositiveWeightedGraph,
    left: Vertex,
    right: Vertex,
    *,
    idleness: float = 0.5,
) -> float:
    """Return uniform-neighbor lazy Ollivier--Ricci curvature on one edge."""

    edge = frozenset((left, right))
    if edge not in graph.edge_lengths:
        raise ValueError("curvature is defined here only on graph edges")
    left_measure = lazy_neighbor_measure(graph, left, idleness=idleness)
    right_measure = lazy_neighbor_measure(graph, right, idleness=idleness)
    transport = wasserstein_1(graph, left_measure, right_measure)
    return 1.0 - transport / vertex_distance(graph, left, right)


def curvature_profile(
    graph: PositiveWeightedGraph,
    *,
    idleness: float = 0.5,
) -> Mapping[Edge, float]:
    return MappingProxyType(
        {
            edge: ollivier_ricci_curvature(
                graph,
                *graph.canonical_endpoints(edge),
                idleness=idleness,
            )
            for edge in graph.edges
        }
    )


@dataclass(frozen=True)
class CurvatureIsomorphismAudit:
    maximum_error: float
    holds: bool


def curvature_isomorphism_audit(
    source: PositiveWeightedGraph,
    target: PositiveWeightedGraph,
    vertex_map: Mapping[Vertex, Vertex],
    *,
    idleness: float = 0.5,
) -> CurvatureIsomorphismAudit:
    """Audit curvature invariance under a length-preserving graph isomorphism."""

    if not is_length_preserving_graph_isomorphism(source, target, vertex_map):
        raise ValueError("curvature invariance requires a weighted graph isomorphism")
    mapping = dict(vertex_map)
    source_profile = curvature_profile(source, idleness=idleness)
    target_profile = curvature_profile(target, idleness=idleness)
    maximum_error = max(
        abs(
            source_profile[edge]
            - target_profile[frozenset(mapping[vertex] for vertex in edge)]
        )
        for edge in source.edges
    )
    return CurvatureIsomorphismAudit(
        maximum_error=maximum_error,
        holds=maximum_error <= _TOLERANCE,
    )


@dataclass(frozen=True)
class CurvaturePerturbationAudit:
    edge_length_sup_error: float
    path_metric_bound: float
    path_metric_sup_error: float
    minimum_separation: float
    maximum_diameter: float
    curvature_sup_error: float
    curvature_bound: float
    path_metric_bound_holds: bool
    curvature_bound_holds: bool


def curvature_perturbation_audit(
    source: PositiveWeightedGraph,
    perturbed: PositiveWeightedGraph,
    *,
    idleness: float = 0.5,
) -> CurvaturePerturbationAudit:
    """Audit the explicit finite curvature bound on a fixed graph."""

    if set(source.vertices) != set(perturbed.vertices):
        raise ValueError("a length perturbation must keep the vertex carrier fixed")
    if set(source.edges) != set(perturbed.edges):
        raise ValueError("a length perturbation must keep the edge set fixed")

    eta = max(
        abs(source.edge_lengths[edge] - perturbed.edge_lengths[edge])
        for edge in source.edges
    )
    path_bound = (len(source.vertices) - 1) * eta
    source_distances = all_pairs_vertex_distances(source)
    perturbed_distances = all_pairs_vertex_distances(perturbed)
    distinct_pairs = tuple(
        (left, right)
        for left in source.vertices
        for right in source.vertices
        if left != right
    )
    path_error = max(
        abs(
            source_distances[(left, right)]
            - perturbed_distances[(left, right)]
        )
        for left, right in distinct_pairs
    )
    minimum_separation = min(
        min(
            source_distances[(left, right)],
            perturbed_distances[(left, right)],
        )
        for left, right in distinct_pairs
    )
    source_diameter = max(source_distances.values())
    perturbed_diameter = max(perturbed_distances.values())
    maximum_diameter = max(source_diameter, perturbed_diameter)

    source_curvature = curvature_profile(source, idleness=idleness)
    perturbed_curvature = curvature_profile(perturbed, idleness=idleness)
    curvature_error = max(
        abs(source_curvature[edge] - perturbed_curvature[edge])
        for edge in source.edges
    )
    curvature_bound = path_bound * (
        1.0 / minimum_separation
        + maximum_diameter / minimum_separation**2
    )
    return CurvaturePerturbationAudit(
        edge_length_sup_error=eta,
        path_metric_bound=path_bound,
        path_metric_sup_error=path_error,
        minimum_separation=minimum_separation,
        maximum_diameter=maximum_diameter,
        curvature_sup_error=curvature_error,
        curvature_bound=curvature_bound,
        path_metric_bound_holds=path_error <= path_bound + _TOLERANCE,
        curvature_bound_holds=curvature_error <= curvature_bound + _TOLERANCE,
    )
