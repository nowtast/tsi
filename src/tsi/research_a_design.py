"""Unfrozen selector implementations for the prospective Research A design."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from .paper34_resolution_benchmark import (
    Graph,
    ResolutionModel,
    TransitionCase,
    WorldSpec,
    generic_features,
    head_value,
)
from .paper34_resolution_contract import HEAD_FAMILIES, LAYER_COUNT, STATE_CARDINALITY


def _best_nonzero_coefficient(rows: Iterable[tuple[int, int]]) -> tuple[int, int]:
    materialized = tuple(rows)
    coefficient = min(
        range(1, STATE_CARDINALITY),
        key=lambda value: (
            sum(
                (value * feature) % STATE_CARDINALITY != observed
                for feature, observed in materialized
            ),
            value,
        ),
    )
    errors = sum(
        (coefficient * feature) % STATE_CARDINALITY != observed
        for feature, observed in materialized
    )
    return coefficient, errors


def _increments(cases: Sequence[TransitionCase]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            (case.observed[layer] - case.source[layer]) % STATE_CARDINALITY
            for layer in range(LAYER_COUNT)
        )
        for case in cases
    )


def fit_typed_structured(
    cases: Sequence[TransitionCase], graph: Graph, *, name: str = "typed_structured"
) -> ResolutionModel:
    """Fit five direct cells and two typed edge cells with a supplied graph."""

    increments = _increments(cases)
    multipliers = []
    for layer in range(LAYER_COUNT):
        coefficient, _ = _best_nonzero_coefficient(
            (case.action[layer], delta[layer])
            for case, delta in zip(cases, increments, strict=True)
            if case.action[layer] != 0
        )
        multipliers.append(coefficient)

    target, sources = graph
    families = []
    coefficients = []
    for source_index in sources:
        best: tuple[int, int, int, str] | None = None
        for family_index, family in enumerate(HEAD_FAMILIES):
            coefficient, errors = _best_nonzero_coefficient(
                (
                    head_value(
                        family,
                        case.source,
                        case.action,
                        source_index,
                        target,
                    ),
                    delta[target],
                )
                for case, delta in zip(cases, increments, strict=True)
                if case.action[source_index] != 0
            )
            candidate = (errors, family_index, coefficient, family)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("typed edge search produced no candidate")
        _, _, coefficient, family = best
        families.append(family)
        coefficients.append(coefficient)

    return ResolutionModel(
        name=name,
        graph=graph,
        families=tuple(families),  # type: ignore[arg-type]
        multipliers=tuple(multipliers),
        coefficients=tuple(coefficients),
    )


def fit_isomorphic_generic(
    cases: Sequence[TransitionCase],
    graph: Graph,
    *,
    name: str = "generic_isomorphic",
) -> ResolutionModel:
    """Fit the typed function class through independent generic coordinates."""

    features = tuple(generic_features(case.source, case.action, graph) for case in cases)
    increments = _increments(cases)
    terms = []
    for layer in range(LAYER_COUNT):
        coefficient, _ = _best_nonzero_coefficient(
            (row[layer], delta[layer])
            for row, delta in zip(features, increments, strict=True)
            if row[layer] != 0
        )
        terms.append((layer, layer, coefficient))

    target, sources = graph
    for edge, source_index in enumerate(sources):
        best: tuple[int, int, int] | None = None
        for family_index in range(len(HEAD_FAMILIES)):
            feature_index = LAYER_COUNT + edge * len(HEAD_FAMILIES) + family_index
            coefficient, errors = _best_nonzero_coefficient(
                (row[feature_index], delta[target])
                for case, row, delta in zip(
                    cases, features, increments, strict=True
                )
                if case.action[source_index] != 0
            )
            candidate = (errors, family_index, coefficient)
            if best is None or candidate < best:
                best = candidate
        if best is None:
            raise RuntimeError("isomorphic generic edge search produced no candidate")
        _, family_index, coefficient = best
        feature_index = LAYER_COUNT + edge * len(HEAD_FAMILIES) + family_index
        terms.append((target, feature_index, coefficient))

    return ResolutionModel(name, graph, None, None, None, tuple(terms))


def fit_unstructured_generic(
    cases: Sequence[TransitionCase],
    graph: Graph,
    *,
    budget: int = 7,
    name: str = "generic_unstructured_greedy",
) -> ResolutionModel:
    """Vectorized equivalent of the declared 55-position greedy selector."""

    x = np.asarray(
        [generic_features(case.source, case.action, graph) for case in cases],
        dtype=np.int64,
    )
    y = np.asarray(_increments(cases), dtype=np.int64)
    prediction = np.zeros_like(y)
    available = np.ones((LAYER_COUNT, x.shape[1]), dtype=bool)
    coefficients = np.arange(1, STATE_CARDINALITY, dtype=np.int64)
    selected = []
    for _ in range(min(budget, available.size)):
        current_errors = np.sum(prediction != y, axis=0)
        scores = np.empty(
            (LAYER_COUNT, x.shape[1], len(coefficients)), dtype=np.int64
        )
        for output in range(LAYER_COUNT):
            candidates = (
                prediction[:, output, None, None]
                + x[:, :, None] * coefficients[None, None, :]
            ) % STATE_CARDINALITY
            errors = np.sum(candidates != y[:, output, None, None], axis=0)
            scores[output] = errors - current_errors[output]
        scores[~available] = np.iinfo(scores.dtype).max
        flat_index = int(np.argmin(scores))
        output, feature, coefficient_offset = np.unravel_index(
            flat_index, scores.shape
        )
        coefficient = int(coefficients[coefficient_offset])
        prediction[:, output] = (
            prediction[:, output] + coefficient * x[:, feature]
        ) % STATE_CARDINALITY
        available[output, feature] = False
        selected.append((int(output), int(feature), coefficient))
    return ResolutionModel(name, graph, None, None, None, tuple(selected))


def generic_true_terms(spec: WorldSpec) -> tuple[tuple[int, int, int], ...]:
    """Return the exact seven generic moves for a generating world."""

    target, _ = spec.graph
    terms = [
        (layer, layer, int(spec.multipliers[layer]))
        for layer in range(LAYER_COUNT)
    ]
    for edge, (family, coefficient) in enumerate(
        zip(spec.families, spec.coefficients, strict=True)
    ):
        feature = (
            LAYER_COUNT
            + edge * len(HEAD_FAMILIES)
            + HEAD_FAMILIES.index(family)
        )
        terms.append((target, feature, int(coefficient)))
    return tuple(terms)


def exact_generic_support_recovered(model: ResolutionModel, spec: WorldSpec) -> bool:
    """Check exact output-feature-coefficient recovery, ignoring term order."""

    return set(model.generic_terms) == set(generic_true_terms(spec))


def isomorphic_prediction_audit(
    typed: ResolutionModel,
    generic: ResolutionModel,
    cases: Sequence[TransitionCase],
) -> dict[str, int | bool]:
    """Verify the notation-control invariant on a supplied case set."""

    mismatches = sum(
        typed.predict(case.source, case.action)
        != generic.predict(case.source, case.action)
        for case in cases
    )
    return {
        "case_count": len(cases),
        "prediction_mismatch_count": mismatches,
        "passed": mismatches == 0,
    }
