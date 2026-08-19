"""Exact finite audits for TSI Extension 2A-X2.

This module keeps five claims separate: label-induced subcomplexes, labeled
filtrations, aligned simplicial maps, contiguity, and persistence stability.
The algorithms are theorem-audit tools for small finite complexes, not scalable
topological-data-analysis software.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from types import MappingProxyType
from typing import Hashable, Mapping

from .topological import (
    Barcode,
    Complex,
    Simplex,
    betti_numbers,
    bottleneck_distance,
    filtration_interleaving_audit,
    simplices_of_dimension,
    sublevel_complex,
    validate_complex,
    validate_filtration,
    zero_dimensional_barcode,
)


Vertex = Hashable
Label = Hashable
VertexMap = Mapping[Vertex, Vertex]
Chain = frozenset[Simplex]
_TOLERANCE = 1e-9


def _vertices(complex_: Complex) -> frozenset[Vertex]:
    return frozenset(
        next(iter(simplex))
        for simplex in validate_complex(complex_)
        if len(simplex) == 1
    )


@dataclass(frozen=True)
class LabeledSimplicialComplex:
    """A nonempty finite abstract simplicial complex with vertex labels."""

    complex: Complex
    labels: Mapping[Vertex, Label]

    def __post_init__(self) -> None:
        normalized = validate_complex(self.complex)
        vertices = _vertices(normalized)
        labels = dict(self.labels)
        if not vertices:
            raise ValueError("a labeled complex must have at least one vertex")
        if set(labels) != set(vertices):
            raise ValueError("labels must be specified exactly on the vertex set")
        for label in labels.values():
            try:
                hash(label)
            except TypeError as error:
                raise ValueError("labels must be hashable") from error
        object.__setattr__(self, "complex", normalized)
        object.__setattr__(self, "labels", MappingProxyType(labels))

    @property
    def vertices(self) -> frozenset[Vertex]:
        return _vertices(self.complex)

    @property
    def label_set(self) -> frozenset[Label]:
        return frozenset(self.labels.values())


def induced_label_subcomplex(
    state: LabeledSimplicialComplex,
    allowed_labels: frozenset[Label],
) -> Complex:
    """Return the full subcomplex induced by vertices with allowed labels."""

    allowed = frozenset(allowed_labels)
    return validate_complex(
        frozenset(
            simplex
            for simplex in state.complex
            if all(state.labels[vertex] in allowed for vertex in simplex)
        )
    )


def label_filtration(
    state: LabeledSimplicialComplex,
    label_values: Mapping[Label, float],
) -> Mapping[Simplex, float]:
    """Return the max-label filtration, with the empty face born first."""

    values = {
        label: float(label_values[label])
        for label in state.label_set
        if label in label_values
    }
    if set(values) != set(state.label_set):
        missing = state.label_set - set(values)
        raise ValueError(f"filtration values are missing for labels: {missing!r}")
    if any(not isfinite(value) for value in values.values()):
        raise ValueError("label filtration values must be finite")
    minimum = min(values.values())
    filtration = {
        simplex: (
            minimum
            if not simplex
            else max(values[state.labels[vertex]] for vertex in simplex)
        )
        for simplex in state.complex
    }
    return MappingProxyType(dict(validate_filtration(state.complex, filtration)))


def validate_simplicial_map(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
) -> Mapping[Vertex, Vertex]:
    """Validate a total vertex map whose simplex images lie in the target."""

    mapping = dict(vertex_map)
    if set(mapping) != set(source.vertices):
        raise ValueError("a simplicial map must be defined exactly on source vertices")
    if any(image not in target.vertices for image in mapping.values()):
        raise ValueError("a simplicial map must land in target vertices")
    for simplex in source.complex:
        image = frozenset(mapping[vertex] for vertex in simplex)
        if image not in target.complex:
            raise ValueError("the image of every simplex must be a target simplex")
    return MappingProxyType(mapping)


def simplex_image(simplex: Simplex, vertex_map: VertexMap) -> Simplex:
    return frozenset(vertex_map[vertex] for vertex in simplex)


def is_simplicial_isomorphism(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
) -> bool:
    try:
        mapping = validate_simplicial_map(source, target, vertex_map)
    except ValueError:
        return False
    if len(set(mapping.values())) != len(mapping):
        return False
    if set(mapping.values()) != set(target.vertices):
        return False
    return frozenset(
        simplex_image(simplex, mapping) for simplex in source.complex
    ) == target.complex


def is_label_preserving(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
) -> bool:
    try:
        mapping = validate_simplicial_map(source, target, vertex_map)
    except ValueError:
        return False
    return all(
        source.labels[vertex] == target.labels[image]
        for vertex, image in mapping.items()
    )


def is_label_preserving_isomorphism(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
) -> bool:
    return is_simplicial_isomorphism(
        source, target, vertex_map
    ) and is_label_preserving(source, target, vertex_map)


def label_stratum_preserved(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
    allowed_labels: frozenset[Label],
) -> bool:
    """Audit exact preservation of one label-induced full subcomplex."""

    if not is_label_preserving_isomorphism(source, target, vertex_map):
        return False
    mapping = dict(vertex_map)
    source_stratum = induced_label_subcomplex(source, allowed_labels)
    target_stratum = induced_label_subcomplex(target, allowed_labels)
    image = frozenset(simplex_image(simplex, mapping) for simplex in source_stratum)
    return image == target_stratum


def label_filtration_preserved(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
    label_values: Mapping[Label, float],
) -> bool:
    """Audit equality of filtration values under a labeled isomorphism."""

    if not is_label_preserving_isomorphism(source, target, vertex_map):
        return False
    mapping = dict(vertex_map)
    source_values = label_filtration(source, label_values)
    target_values = label_filtration(target, label_values)
    return all(
        isclose(
            source_values[simplex],
            target_values[simplex_image(simplex, mapping)],
            abs_tol=_TOLERANCE,
        )
        for simplex in source.complex
    )


def is_filtered_simplicial_map(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    vertex_map: VertexMap,
    source_values: Mapping[Simplex, float],
    target_values: Mapping[Simplex, float],
) -> bool:
    """Return whether a simplicial map sends each sublevel into the same sublevel."""

    try:
        mapping = validate_simplicial_map(source, target, vertex_map)
        source_filtration = validate_filtration(source.complex, source_values)
        target_filtration = validate_filtration(target.complex, target_values)
    except ValueError:
        return False
    return all(
        target_filtration[simplex_image(simplex, mapping)]
        <= source_filtration[simplex] + _TOLERANCE
        for simplex in source.complex
    )


def commuting_square_holds(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    aligned_source: LabeledSimplicialComplex,
    aligned_target: LabeledSimplicialComplex,
    transition: VertexMap,
    aligned_transition: VertexMap,
    source_alignment: VertexMap,
    target_alignment: VertexMap,
) -> bool:
    """Audit beta o f = f' o alpha on vertices."""

    try:
        f = validate_simplicial_map(source, target, transition)
        f_prime = validate_simplicial_map(
            aligned_source,
            aligned_target,
            aligned_transition,
        )
        alpha = validate_simplicial_map(source, aligned_source, source_alignment)
        beta = validate_simplicial_map(target, aligned_target, target_alignment)
    except ValueError:
        return False
    return all(beta[f[vertex]] == f_prime[alpha[vertex]] for vertex in source.vertices)


def _chain_image_of_simplex(
    simplex: Simplex,
    vertex_map: VertexMap,
) -> Chain:
    image = simplex_image(simplex, vertex_map)
    if len(image) != len(simplex):
        return frozenset()
    return frozenset((image,))


def commuting_chain_maps_hold(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    aligned_source: LabeledSimplicialComplex,
    aligned_target: LabeledSimplicialComplex,
    transition: VertexMap,
    aligned_transition: VertexMap,
    source_alignment: VertexMap,
    target_alignment: VertexMap,
) -> bool:
    """Audit beta_# f_# = f'_# alpha_# over F2 in every dimension."""

    if not commuting_square_holds(
        source,
        target,
        aligned_source,
        aligned_target,
        transition,
        aligned_transition,
        source_alignment,
        target_alignment,
    ):
        return False
    f = dict(transition)
    f_prime = dict(aligned_transition)
    alpha = dict(source_alignment)
    beta = dict(target_alignment)
    maximum = max(len(simplex) - 1 for simplex in source.complex)
    for dimension in range(maximum + 1):
        for simplex in simplices_of_dimension(source.complex, dimension):
            left_first = _chain_image_of_simplex(simplex, f)
            left = frozenset(
                image
                for middle in left_first
                for image in _chain_image_of_simplex(middle, beta)
            )
            right_first = _chain_image_of_simplex(simplex, alpha)
            right = frozenset(
                image
                for middle in right_first
                for image in _chain_image_of_simplex(middle, f_prime)
            )
            if left != right:
                return False
    return True


def are_contiguous(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    left_map: VertexMap,
    right_map: VertexMap,
) -> bool:
    """Return whether two simplicial maps are contiguous."""

    try:
        left = validate_simplicial_map(source, target, left_map)
        right = validate_simplicial_map(source, target, right_map)
    except ValueError:
        return False
    return all(
        simplex_image(simplex, left).union(simplex_image(simplex, right))
        in target.complex
        for simplex in source.complex
    )


def _add_chains(*chains: Chain) -> Chain:
    result: set[Simplex] = set()
    for chain in chains:
        result.symmetric_difference_update(chain)
    return frozenset(result)


def _boundary_chain(chain: Chain) -> Chain:
    boundary: set[Simplex] = set()
    for simplex in chain:
        if len(simplex) <= 1:
            continue
        for vertex in simplex:
            face = frozenset(set(simplex) - {vertex})
            if face in boundary:
                boundary.remove(face)
            else:
                boundary.add(face)
    return frozenset(boundary)


def _prism_on_simplex(
    simplex: Simplex,
    left_map: VertexMap,
    right_map: VertexMap,
) -> Chain:
    ordered = tuple(sorted(simplex, key=repr))
    dimension = len(ordered) - 1
    result: set[Simplex] = set()
    for index in range(dimension + 1):
        vertices = (
            *(left_map[ordered[position]] for position in range(index + 1)),
            *(
                right_map[ordered[position]]
                for position in range(index, dimension + 1)
            ),
        )
        term = frozenset(vertices)
        if len(term) != dimension + 2:
            continue
        if term in result:
            result.remove(term)
        else:
            result.add(term)
    return frozenset(result)


def contiguity_chain_homotopy_audit(
    source: LabeledSimplicialComplex,
    target: LabeledSimplicialComplex,
    left_map: VertexMap,
    right_map: VertexMap,
) -> bool:
    """Verify dP + Pd = f_# + g_# over F2 on every basis simplex."""

    if not are_contiguous(source, target, left_map, right_map):
        return False
    left = dict(left_map)
    right = dict(right_map)
    maximum = max(len(simplex) - 1 for simplex in source.complex)
    for dimension in range(maximum + 1):
        for simplex in simplices_of_dimension(source.complex, dimension):
            prism = _prism_on_simplex(simplex, left, right)
            if any(term not in target.complex for term in prism):
                return False
            prism_boundary = _boundary_chain(prism)
            source_boundary = _boundary_chain(frozenset((simplex,)))
            boundary_prism = _add_chains(
                *(
                    _prism_on_simplex(face, left, right)
                    for face in source_boundary
                )
            )
            expected = _add_chains(
                _chain_image_of_simplex(simplex, left),
                _chain_image_of_simplex(simplex, right),
            )
            if _add_chains(prism_boundary, boundary_prism) != expected:
                return False
    return True


@dataclass(frozen=True)
class LabelStabilityAudit:
    """Finite H0 witness for the label-filtration stability theorem."""

    epsilon: float
    filtration_sup_distance: float
    interleaving_holds: bool
    left_barcode: Barcode
    right_barcode: Barcode
    h0_bottleneck: float
    bound_holds: bool


def label_filtration_stability_audit(
    state: LabeledSimplicialComplex,
    left_values: Mapping[Label, float],
    right_values: Mapping[Label, float],
) -> LabelStabilityAudit:
    """Audit the common-scale sup-norm and H0 bottleneck bounds."""

    left = label_filtration(state, left_values)
    right = label_filtration(state, right_values)
    epsilon = max(
        abs(float(left_values[label]) - float(right_values[label]))
        for label in state.label_set
    )
    filtration_sup, interleaving = filtration_interleaving_audit(
        state.complex,
        left,
        right,
    )
    left_barcode = zero_dimensional_barcode(state.complex, left)
    right_barcode = zero_dimensional_barcode(state.complex, right)
    distance = bottleneck_distance(left_barcode, right_barcode)
    return LabelStabilityAudit(
        epsilon=epsilon,
        filtration_sup_distance=filtration_sup,
        interleaving_holds=interleaving,
        left_barcode=left_barcode,
        right_barcode=right_barcode,
        h0_bottleneck=distance,
        bound_holds=(
            filtration_sup <= epsilon + _TOLERANCE
            and distance <= epsilon + _TOLERANCE
        ),
    )


def filtration_betti_signature(
    state: LabeledSimplicialComplex,
    label_values: Mapping[Label, float],
    *,
    max_dimension: int,
) -> tuple[tuple[float, tuple[int, ...]], ...]:
    """Return Betti vectors at every critical value; this is not complete."""

    if max_dimension < 0:
        raise ValueError("max_dimension must be nonnegative")
    filtration = label_filtration(state, label_values)
    thresholds = tuple(sorted(set(filtration.values())))
    return tuple(
        (
            threshold,
            betti_numbers(
                sublevel_complex(state.complex, filtration, threshold),
                max_dimension=max_dimension,
            ),
        )
        for threshold in thresholds
    )

