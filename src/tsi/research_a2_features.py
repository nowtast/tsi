"""Collision-audited generic feature libraries for Research A2 width tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

from .paper34_resolution_benchmark import (
    GRAPH_MANIFEST,
    Graph,
    generic_features,
    head_value,
)
from .paper34_resolution_contract import HEAD_FAMILIES, LAYER_COUNT, STATE_CARDINALITY


WIDTH_POSITION_COUNTS = (55, 100, 300)
WIDTH_FEATURE_COUNTS = tuple(value // LAYER_COUNT for value in WIDTH_POSITION_COUNTS)
CUBIC_FAMILY = "cubic_target"
TYPED_FAMILY_CATALOG = tuple(HEAD_FAMILIES)
ALTERNATIVE_FAMILY_CATALOG = (
    "linear_target",
    CUBIC_FAMILY,
    "source_target",
)


def a2_head_value(
    family: str,
    source: Sequence[int],
    action: Sequence[int],
    source_index: int,
    target: int,
) -> int:
    """Evaluate an A1 head or the prespecified cubic A2 alternative."""

    if family == CUBIC_FAMILY:
        return int(action[source_index]) * (1 + int(source[target])) ** 3
    return head_value(family, source, action, source_index, target)


def catalog_generic_features(
    source: Sequence[int],
    action: Sequence[int],
    graph: Graph,
    family_catalog: Sequence[str],
) -> tuple[int, ...]:
    """Build a generic dictionary from an explicit, ordered family catalog."""

    target, sources = graph
    values = [int(value) for value in action]
    for source_index in sources:
        values.extend(
            a2_head_value(family, source, action, source_index, target)
            for family in family_catalog
        )
    return tuple(value % STATE_CARDINALITY for value in values)


@dataclass(frozen=True)
class NuisanceFeature:
    action_coordinate: int
    state_coordinate: int
    degree: int

    def evaluate(self, source: Sequence[int], action: Sequence[int]) -> int:
        return (
            int(action[self.action_coordinate])
            * int(source[self.state_coordinate]) ** self.degree
        ) % STATE_CARDINALITY


NUISANCE_FEATURE_ORDER = (
    "state_coordinate",
    "degree",
    "action_coordinate",
)
ALL_NUISANCE_FEATURES = tuple(
    NuisanceFeature(action, state, degree)
    for state in range(LAYER_COUNT)
    for degree in (1, 2)
    for action in range(LAYER_COUNT)
)
NUISANCE_FEATURES = ALL_NUISANCE_FEATURES[:49]
EXCLUDED_NUISANCE_FEATURES = ALL_NUISANCE_FEATURES[49:]


def augmented_generic_features(
    source: Sequence[int],
    action: Sequence[int],
    graph: Graph,
    position_count: int,
) -> tuple[int, ...]:
    """Return exactly position_count / 5 features, retaining A1 first."""

    if position_count not in WIDTH_POSITION_COUNTS:
        raise ValueError(f"unsupported A2 position count: {position_count}")
    feature_count = position_count // LAYER_COUNT
    base = generic_features(source, action, graph)
    extra_count = feature_count - len(base)
    return base + tuple(
        descriptor.evaluate(source, action)
        for descriptor in NUISANCE_FEATURES[:extra_count]
    )


def _projective_signature(values: Sequence[int]) -> bytes:
    first = next((value for value in values if value), None)
    if first is None:
        raise ValueError("zero feature has no projective signature")
    inverse = pow(int(first), -1, STATE_CARDINALITY)
    return bytes((int(value) * inverse) % STATE_CARDINALITY for value in values)


def audit_width_feature_libraries() -> dict[str, object]:
    """Exhaust all states and graphs for zero or scalar-duplicate features."""

    states = tuple(product(range(STATE_CARDINALITY), repeat=LAYER_COUNT))
    nuisance_signatures = {
        descriptor: _projective_signature(
            [
                int(source[descriptor.state_coordinate]) ** descriptor.degree
                % STATE_CARDINALITY
                for source in states
            ]
        )
        for descriptor in NUISANCE_FEATURES
    }
    width_audits = []
    errors = []
    for position_count, feature_count in zip(
        WIDTH_POSITION_COUNTS, WIDTH_FEATURE_COUNTS, strict=True
    ):
        minimum_unique = feature_count
        for graph in GRAPH_MANIFEST:
            target, sources = graph
            keys = []
            direct_signature = _projective_signature([1] * len(states))
            keys.extend((layer, direct_signature) for layer in range(LAYER_COUNT))
            for source_index in sources:
                for family in HEAD_FAMILIES:
                    unit_values = []
                    for source in states:
                        action = [0] * LAYER_COUNT
                        action[source_index] = 1
                        unit_values.append(
                            head_value(family, source, action, source_index, target)
                            % STATE_CARDINALITY
                        )
                    keys.append((source_index, _projective_signature(unit_values)))
            extra_count = feature_count - 11
            keys.extend(
                (
                    descriptor.action_coordinate,
                    nuisance_signatures[descriptor],
                )
                for descriptor in NUISANCE_FEATURES[:extra_count]
            )
            unique_count = len(set(keys))
            minimum_unique = min(minimum_unique, unique_count)
            if len(keys) != feature_count or unique_count != feature_count:
                errors.append(
                    {
                        "position_count": position_count,
                        "graph": [graph[0], list(graph[1])],
                        "feature_count": len(keys),
                        "unique_projective_count": unique_count,
                    }
                )
        width_audits.append(
            {
                "position_count": position_count,
                "feature_count": feature_count,
                "nuisance_feature_count": feature_count - 11,
                "minimum_unique_projective_count_over_30_graphs": minimum_unique,
            }
        )
    return {
        "state_count": len(states),
        "graph_count": len(GRAPH_MANIFEST),
        "nuisance_feature_order": list(NUISANCE_FEATURE_ORDER),
        "nuisance_feature_pool_count": len(ALL_NUISANCE_FEATURES),
        "nuisance_features": [asdict(item) for item in NUISANCE_FEATURES],
        "excluded_nuisance_features": [
            asdict(item) for item in EXCLUDED_NUISANCE_FEATURES
        ],
        "widths": width_audits,
        "errors": errors,
        "passed": not errors,
    }


def audit_fourth_family_separation() -> dict[str, object]:
    """Exhaustively audit projective separation of cubic from all A1 heads."""

    actions = (0, 1, 0, 0, 0)
    graph = (0, (1, 2))
    source_index = 1
    rows = tuple(
        (target_state, source_state)
        for target_state in range(STATE_CARDINALITY)
        for source_state in range(STATE_CARDINALITY)
    )
    comparisons = []
    errors = []
    for family in TYPED_FAMILY_CATALOG:
        minimum = len(rows)
        minimizers = []
        for cubic_coefficient in range(1, STATE_CARDINALITY):
            for other_coefficient in range(1, STATE_CARDINALITY):
                mismatches = 0
                for target_state, source_state in rows:
                    source = [0] * LAYER_COUNT
                    source[0] = target_state
                    source[1] = source_state
                    cubic = cubic_coefficient * a2_head_value(
                        CUBIC_FAMILY, source, actions, source_index, graph[0]
                    )
                    other = other_coefficient * a2_head_value(
                        family, source, actions, source_index, graph[0]
                    )
                    mismatches += cubic % STATE_CARDINALITY != other % STATE_CARDINALITY
                if mismatches < minimum:
                    minimum = mismatches
                    minimizers = [[cubic_coefficient, other_coefficient]]
                elif mismatches == minimum:
                    minimizers.append([cubic_coefficient, other_coefficient])
        expected_minimum = {
            "linear_target": 28,
            "quadratic_target": 35,
            "source_target": 42,
        }[family]
        if minimum != expected_minimum:
            errors.append(
                f"{family}: expected {expected_minimum} minimum disagreements, got {minimum}"
            )
        comparisons.append(
            {
                "family": family,
                "row_count": len(rows),
                "minimum_disagreement_count_over_nonzero_scalings": minimum,
                "minimum_disagreement_fraction": minimum / len(rows),
                "minimizing_coefficient_pairs": minimizers,
            }
        )
    reverse_minimum = len(rows)
    reverse_minimizers = []
    for direct_coefficient in range(STATE_CARDINALITY):
        for linear_coefficient in range(STATE_CARDINALITY):
            for cubic_coefficient in range(STATE_CARDINALITY):
                for source_coefficient in range(STATE_CARDINALITY):
                    mismatches = 0
                    for target_state, source_state in rows:
                        source = [0] * LAYER_COUNT
                        source[0] = target_state
                        source[1] = source_state
                        truth = a2_head_value(
                            "quadratic_target",
                            source,
                            actions,
                            source_index,
                            graph[0],
                        )
                        prediction = (
                            direct_coefficient
                            + linear_coefficient
                            * a2_head_value(
                                "linear_target",
                                source,
                                actions,
                                source_index,
                                graph[0],
                            )
                            + cubic_coefficient
                            * a2_head_value(
                                CUBIC_FAMILY,
                                source,
                                actions,
                                source_index,
                                graph[0],
                            )
                            + source_coefficient
                            * a2_head_value(
                                "source_target",
                                source,
                                actions,
                                source_index,
                                graph[0],
                            )
                        ) % STATE_CARDINALITY
                        mismatches += prediction != truth % STATE_CARDINALITY
                    coefficients = [
                        direct_coefficient,
                        linear_coefficient,
                        cubic_coefficient,
                        source_coefficient,
                    ]
                    if mismatches < reverse_minimum:
                        reverse_minimum = mismatches
                        reverse_minimizers = [coefficients]
                    elif mismatches == reverse_minimum:
                        reverse_minimizers.append(coefficients)
    if reverse_minimum != 28:
        errors.append(
            "quadratic_target: expected 28 minimum disagreements from the "
            f"alternative generic span, got {reverse_minimum}"
        )
    return {
        "field": "Z_7",
        "fourth_family": CUBIC_FAMILY,
        "comparison_domain": "all 49 ordered (target_state, source_state) pairs",
        "comparisons": comparisons,
        "reverse_generic_span_audit": {
            "truth": "quadratic_target",
            "candidate_span": [
                "direct_action",
                "linear_target",
                "cubic_target",
                "source_target",
            ],
            "coefficient_vectors_exhausted": STATE_CARDINALITY**4,
            "minimum_disagreement_count": reverse_minimum,
            "minimum_disagreement_fraction": reverse_minimum / len(rows),
            "minimizing_coefficient_vectors": reverse_minimizers,
        },
        "errors": errors,
        "passed": not errors,
    }


def audit_misspecification_catalogs() -> dict[str, object]:
    """Verify equal catalog width and the intended one-family substitution."""

    typed = set(TYPED_FAMILY_CATALOG)
    alternative = set(ALTERNATIVE_FAMILY_CATALOG)
    errors = []
    if len(TYPED_FAMILY_CATALOG) != len(ALTERNATIVE_FAMILY_CATALOG):
        errors.append("family catalogs have unequal widths")
    if typed - alternative != {"quadratic_target"}:
        errors.append("alternative catalog did not remove only quadratic_target")
    if alternative - typed != {CUBIC_FAMILY}:
        errors.append("alternative catalog did not add only cubic_target")
    return {
        "typed_catalog": list(TYPED_FAMILY_CATALOG),
        "alternative_catalog": list(ALTERNATIVE_FAMILY_CATALOG),
        "features_per_catalog": LAYER_COUNT + 2 * len(TYPED_FAMILY_CATALOG),
        "output_feature_positions": LAYER_COUNT
        * (LAYER_COUNT + 2 * len(TYPED_FAMILY_CATALOG)),
        "errors": errors,
        "passed": not errors,
    }
