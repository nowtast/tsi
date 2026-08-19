"""Pareto geometry for finite coherent correspondences.

The Stage 2-I0 discrepancy scalarizes five component distortions before
minimizing over typed correspondences.  This module retains the finite set of
attainable distortion vectors, its Pareto frontier, and the gap between a
single common alignment and independently optimized layer alignments.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite
from typing import Iterable, Sequence

from .coherent import (
    CoherentStructuralState,
    CorrespondenceCosts,
    correspondence_costs,
    typed_correspondences,
)


LAYER_NAMES = ("label", "simplicial", "metric", "relation", "order")
_LAYER_COUNT = len(LAYER_NAMES)


@dataclass(frozen=True, order=True)
class LayerDistortionVector:
    """A nonnegative distortion vector in the fixed I0 layer order."""

    label: float
    simplicial: float
    metric: float
    relation: float
    order: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if not isfinite(value) or value < 0:
                raise ValueError("layer distortions must be finite and nonnegative")
            object.__setattr__(self, field.name, value)

    @classmethod
    def zero(cls) -> "LayerDistortionVector":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_costs(cls, costs: CorrespondenceCosts) -> "LayerDistortionVector":
        return cls(
            label=costs.label,
            simplicial=costs.simplicial,
            metric=costs.metric,
            relation=costs.relation,
            order=costs.order,
        )

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(getattr(self, name) for name in LAYER_NAMES)

    def componentwise_leq(self, other: "LayerDistortionVector") -> bool:
        return all(left <= right for left, right in zip(self.values, other.values))

    def strictly_dominates(self, other: "LayerDistortionVector") -> bool:
        return self != other and self.componentwise_leq(other)

    def __add__(self, other: "LayerDistortionVector") -> "LayerDistortionVector":
        return LayerDistortionVector(
            *(left + right for left, right in zip(self.values, other.values))
        )

    def scalarize(self, weights: Sequence[float]) -> float:
        validated = _positive_weights(weights)
        return sum(
            weight * value
            for weight, value in zip(validated, self.values, strict=True)
        )

    def as_dict(self) -> dict[str, float]:
        return dict(zip(LAYER_NAMES, self.values, strict=True))


def _positive_weights(weights: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in weights)
    if len(normalized) != _LAYER_COUNT:
        raise ValueError(f"exactly {_LAYER_COUNT} layer weights are required")
    if any(not isfinite(value) or value <= 0 for value in normalized):
        raise ValueError("all layer weights must be finite and strictly positive")
    return normalized


def signature_weights(
    state: CoherentStructuralState,
) -> tuple[float, float, float, float, float]:
    signature = state.signature
    return (
        signature.label_weight,
        signature.simplicial_weight,
        signature.metric_weight,
        signature.relation_weight,
        signature.order_weight,
    )


def pareto_minima(
    vectors: Iterable[LayerDistortionVector],
) -> tuple[LayerDistortionVector, ...]:
    """Return the distinct coordinatewise minimal vectors."""

    unique = tuple(sorted(set(vectors)))
    if not unique:
        raise ValueError("a Pareto frontier requires at least one vector")
    return tuple(
        vector
        for vector in unique
        if not any(candidate.strictly_dominates(vector) for candidate in unique)
    )


@dataclass(frozen=True)
class CorrespondenceSpectrum:
    """Finite attainable vectors and their Pareto-minimal upper-set generators."""

    attainable: tuple[LayerDistortionVector, ...]
    pareto: tuple[LayerDistortionVector, ...]
    ideal: LayerDistortionVector
    correspondence_count: int

    def __post_init__(self) -> None:
        if not self.attainable:
            raise ValueError("the attainable spectrum must be nonempty")
        if self.attainable != tuple(sorted(set(self.attainable))):
            raise ValueError("attainable vectors must be distinct and sorted")
        if (
            not isinstance(self.correspondence_count, int)
            or isinstance(self.correspondence_count, bool)
        ):
            raise TypeError("correspondence count must be an integer")
        if self.correspondence_count < len(self.attainable):
            raise ValueError("correspondence count cannot be smaller than vector count")
        if self.pareto != pareto_minima(self.attainable):
            raise ValueError("stored Pareto frontier is not the attainable minimum set")
        expected_ideal = LayerDistortionVector(
            *(min(vector.values[index] for vector in self.attainable) for index in range(_LAYER_COUNT))
        )
        if self.ideal != expected_ideal:
            raise ValueError("stored ideal vector is not the componentwise infimum")

    @classmethod
    def from_vectors(
        cls,
        vectors: Iterable[LayerDistortionVector],
        *,
        correspondence_count: int | None = None,
    ) -> "CorrespondenceSpectrum":
        supplied = tuple(vectors)
        if not supplied:
            raise ValueError("the attainable spectrum must be nonempty")
        attainable = tuple(sorted(set(supplied)))
        ideal = LayerDistortionVector(
            *(min(vector.values[index] for vector in attainable) for index in range(_LAYER_COUNT))
        )
        return cls(
            attainable=attainable,
            pareto=pareto_minima(attainable),
            ideal=ideal,
            correspondence_count=(
                len(supplied) if correspondence_count is None else correspondence_count
            ),
        )

    @property
    def ideal_is_attainable(self) -> bool:
        return self.ideal in self.attainable

    @property
    def has_zero(self) -> bool:
        return LayerDistortionVector.zero() in self.attainable

    def upper_contains(self, vector: LayerDistortionVector) -> bool:
        """Return whether the upper attainable set contains ``vector``."""

        return any(point.componentwise_leq(vector) for point in self.pareto)

    def scalarized_value(self, weights: Sequence[float]) -> float:
        validated = _positive_weights(weights)
        return min(vector.scalarize(validated) for vector in self.pareto)

    def as_dict(self) -> dict[str, object]:
        return {
            "correspondence_count": self.correspondence_count,
            "attainable_vector_count": len(self.attainable),
            "pareto_vector_count": len(self.pareto),
            "attainable": [vector.as_dict() for vector in self.attainable],
            "pareto": [vector.as_dict() for vector in self.pareto],
            "ideal": self.ideal.as_dict(),
            "ideal_is_attainable": self.ideal_is_attainable,
            "has_zero": self.has_zero,
        }


def coherent_correspondence_spectrum(
    left: CoherentStructuralState,
    right: CoherentStructuralState,
    *,
    max_correspondences: int = 100_000,
) -> CorrespondenceSpectrum:
    """Enumerate the exact finite distortion spectrum for two coherent states."""

    vectors: list[LayerDistortionVector] = []
    correspondence_count = 0
    for correspondence in typed_correspondences(
        left,
        right,
        max_correspondences=max_correspondences,
    ):
        correspondence_count += 1
        vectors.append(
            LayerDistortionVector.from_costs(
                correspondence_costs(correspondence, left, right)
            )
        )
    return CorrespondenceSpectrum.from_vectors(
        vectors,
        correspondence_count=correspondence_count,
    )


@dataclass(frozen=True)
class AlignmentFrustration:
    """The common-alignment cost above the independent layerwise ideal."""

    joint_cost: float
    independent_lower_bound: float
    gap: float
    ideal: LayerDistortionVector
    ideal_is_attainable: bool

    def __post_init__(self) -> None:
        for value in (self.joint_cost, self.independent_lower_bound, self.gap):
            if not isfinite(value) or value < 0:
                raise ValueError("alignment-frustration values must be nonnegative")
        if self.joint_cost != self.independent_lower_bound + self.gap:
            raise ValueError("joint cost must equal lower bound plus frustration")
        if self.ideal_is_attainable != (self.gap == 0.0):
            raise ValueError("zero frustration must exactly detect ideal attainment")

    @property
    def is_zero(self) -> bool:
        return self.gap == 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "joint_cost": self.joint_cost,
            "independent_lower_bound": self.independent_lower_bound,
            "gap": self.gap,
            "ideal": self.ideal.as_dict(),
            "ideal_is_attainable": self.ideal_is_attainable,
        }


def alignment_frustration(
    spectrum: CorrespondenceSpectrum,
    weights: Sequence[float],
) -> AlignmentFrustration:
    """Return the exact positive-weight common-versus-independent gap."""

    validated = _positive_weights(weights)
    independent = spectrum.ideal.scalarize(validated)
    gap = min(
        sum(
            weight * (value - ideal)
            for weight, value, ideal in zip(
                validated,
                vector.values,
                spectrum.ideal.values,
                strict=True,
            )
        )
        for vector in spectrum.attainable
    )
    joint = independent + gap
    return AlignmentFrustration(
        joint_cost=joint,
        independent_lower_bound=independent,
        gap=gap,
        ideal=spectrum.ideal,
        ideal_is_attainable=spectrum.ideal_is_attainable,
    )


@dataclass(frozen=True)
class ParetoTriangleAudit:
    """Finite audit of the Pareto--Minkowski triangle inclusion."""

    tested_frontier_pairs: int
    violations: tuple[
        tuple[LayerDistortionVector, LayerDistortionVector], ...
    ]

    @property
    def passed(self) -> bool:
        return not self.violations


def audit_pareto_triangle(
    first: CorrespondenceSpectrum,
    second: CorrespondenceSpectrum,
    direct: CorrespondenceSpectrum,
) -> ParetoTriangleAudit:
    """Check ``upper(first) + upper(second)`` is contained in ``upper(direct)``."""

    violations = tuple(
        (left, right)
        for left in first.pareto
        for right in second.pareto
        if not direct.upper_contains(left + right)
    )
    return ParetoTriangleAudit(
        tested_frontier_pairs=len(first.pareto) * len(second.pareto),
        violations=violations,
    )
