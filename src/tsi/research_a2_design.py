"""Unfrozen A2 selectors built without changing frozen A1 sources."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .paper34_resolution_benchmark import Graph, TransitionCase
from .paper34_resolution_contract import LAYER_COUNT, STATE_CARDINALITY
from .research_a_design import _best_nonzero_coefficient, _increments
from .research_a2_features import (
    TYPED_FAMILY_CATALOG,
    a2_head_value,
    augmented_generic_features,
    catalog_generic_features,
)


@dataclass(frozen=True)
class A2ResolutionModel:
    name: str
    graph: Graph
    families: tuple[str, str] | None = None
    multipliers: tuple[int, ...] | None = None
    coefficients: tuple[int, ...] | None = None
    generic_terms: tuple[tuple[int, int, int], ...] = ()
    generic_family_catalog: tuple[str, ...] | None = None
    width_position_count: int | None = None

    @property
    def active_parameter_count(self) -> int:
        if self.generic_terms:
            return len(self.generic_terms)
        return 0 if self.multipliers is None else len(self.multipliers) + 2

    def predict(self, source: Sequence[int], action: Sequence[int]) -> tuple[int, ...]:
        if self.generic_terms:
            if self.width_position_count is not None:
                features = augmented_generic_features(
                    source, action, self.graph, self.width_position_count
                )
            elif self.generic_family_catalog is not None:
                features = catalog_generic_features(
                    source, action, self.graph, self.generic_family_catalog
                )
            else:
                raise RuntimeError("generic A2 model has no feature specification")
            delta = [0] * LAYER_COUNT
            for output, feature, coefficient in self.generic_terms:
                delta[output] += int(coefficient) * int(features[feature])
            return tuple(
                (int(source[index]) + delta[index]) % STATE_CARDINALITY
                for index in range(LAYER_COUNT)
            )
        if (
            self.families is None
            or self.multipliers is None
            or self.coefficients is None
        ):
            raise RuntimeError("incomplete factorized A2 model")
        target, sources = self.graph
        delta = [
            int(self.multipliers[layer]) * int(action[layer])
            for layer in range(LAYER_COUNT)
        ]
        for edge, source_index in enumerate(sources):
            delta[target] += int(self.coefficients[edge]) * a2_head_value(
                self.families[edge], source, action, source_index, target
            )
        return tuple(
            (int(source[layer]) + delta[layer]) % STATE_CARDINALITY
            for layer in range(LAYER_COUNT)
        )


def _fit_greedy_matrix(
    cases: Sequence[TransitionCase],
    graph: Graph,
    x: np.ndarray,
    *,
    budget: int,
    name: str,
    family_catalog: tuple[str, ...] | None = None,
    position_count: int | None = None,
) -> A2ResolutionModel:
    y = np.asarray(_increments(cases), dtype=np.int64)
    prediction = np.zeros_like(y)
    available = np.ones((LAYER_COUNT, x.shape[1]), dtype=bool)
    coefficients = np.arange(1, STATE_CARDINALITY, dtype=np.int64)
    selected = []
    for _ in range(min(budget, available.size)):
        current_errors = np.sum(prediction != y, axis=0)
        scores = np.empty((LAYER_COUNT, x.shape[1], len(coefficients)), dtype=np.int64)
        for output in range(LAYER_COUNT):
            candidates = (
                prediction[:, output, None, None]
                + x[:, :, None] * coefficients[None, None, :]
            ) % STATE_CARDINALITY
            errors = np.sum(candidates != y[:, output, None, None], axis=0)
            scores[output] = errors - current_errors[output]
        scores[~available] = np.iinfo(scores.dtype).max
        output, feature, coefficient_offset = np.unravel_index(
            int(np.argmin(scores)), scores.shape
        )
        coefficient = int(coefficients[coefficient_offset])
        prediction[:, output] = (
            prediction[:, output] + coefficient * x[:, feature]
        ) % STATE_CARDINALITY
        available[output, feature] = False
        selected.append((int(output), int(feature), coefficient))
    return A2ResolutionModel(
        name=name,
        graph=graph,
        generic_terms=tuple(selected),
        generic_family_catalog=family_catalog,
        width_position_count=position_count,
    )


def fit_width_generic(
    cases: Sequence[TransitionCase],
    graph: Graph,
    position_count: int,
    *,
    budget: int = 7,
    name: str | None = None,
) -> A2ResolutionModel:
    """Fit seven greedy moves over a collision-audited A2 dictionary."""

    x = np.asarray(
        [
            augmented_generic_features(case.source, case.action, graph, position_count)
            for case in cases
        ],
        dtype=np.int64,
    )
    if x.shape[1] * LAYER_COUNT != position_count:
        raise RuntimeError("A2 dictionary width changed")
    return _fit_greedy_matrix(
        cases,
        graph,
        x,
        budget=budget,
        name=name or f"generic_width_{position_count}",
        position_count=position_count,
    )


def fit_typed_catalog(
    cases: Sequence[TransitionCase],
    graph: Graph,
    *,
    family_catalog: Sequence[str] = TYPED_FAMILY_CATALOG,
    name: str = "typed_catalog",
) -> A2ResolutionModel:
    """Fit direct cells and two edge cells from an explicit typed catalog."""

    catalog = tuple(family_catalog)
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
    edge_coefficients = []
    for source_index in sources:
        best: tuple[int, int, int, str] | None = None
        for family_index, family in enumerate(catalog):
            coefficient, errors = _best_nonzero_coefficient(
                (
                    a2_head_value(
                        family, case.source, case.action, source_index, target
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
            raise RuntimeError("typed A2 edge search produced no candidate")
        _, _, coefficient, family = best
        families.append(family)
        edge_coefficients.append(coefficient)
    return A2ResolutionModel(
        name=name,
        graph=graph,
        families=tuple(families),  # type: ignore[arg-type]
        multipliers=tuple(multipliers),
        coefficients=tuple(edge_coefficients),
    )


def fit_catalog_generic(
    cases: Sequence[TransitionCase],
    graph: Graph,
    *,
    family_catalog: Sequence[str],
    budget: int = 7,
    name: str = "generic_catalog_greedy",
) -> A2ResolutionModel:
    """Fit the seven-move generic selector over an explicit head catalog."""

    catalog = tuple(family_catalog)
    x = np.asarray(
        [
            catalog_generic_features(case.source, case.action, graph, catalog)
            for case in cases
        ],
        dtype=np.int64,
    )
    return _fit_greedy_matrix(
        cases,
        graph,
        x,
        budget=budget,
        name=name,
        family_catalog=catalog,
    )


def generic_true_terms_for_catalog(
    spec: object, family_catalog: Sequence[str]
) -> tuple[tuple[int, int, int], ...] | None:
    """Map a world to its seven generic terms, or None if unrepresentable."""

    catalog = tuple(family_catalog)
    graph = getattr(spec, "graph")
    families = tuple(getattr(spec, "families"))
    multipliers = tuple(getattr(spec, "multipliers"))
    coefficients = tuple(getattr(spec, "coefficients"))
    if any(family not in catalog for family in families):
        return None
    target, _ = graph
    terms = [(layer, layer, int(multipliers[layer])) for layer in range(LAYER_COUNT)]
    for edge, (family, coefficient) in enumerate(
        zip(families, coefficients, strict=True)
    ):
        feature = LAYER_COUNT + edge * len(catalog) + catalog.index(family)
        terms.append((target, feature, int(coefficient)))
    return tuple(terms)


def exact_catalog_support_recovered(
    model: A2ResolutionModel, spec: object, family_catalog: Sequence[str]
) -> bool:
    truth = generic_true_terms_for_catalog(spec, family_catalog)
    return truth is not None and set(model.generic_terms) == set(truth)


def typed_parameters_recovered(model: A2ResolutionModel, spec: object) -> bool:
    return bool(
        model.families == getattr(spec, "families")
        and model.multipliers == getattr(spec, "multipliers")
        and model.coefficients == getattr(spec, "coefficients")
    )
