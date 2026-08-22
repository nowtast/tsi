"""World populations and stochastic cases for Research A2."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .paper34_resolution_benchmark import GRAPH_MANIFEST, Graph, TransitionCase
from .paper34_resolution_contract import LAYER_COUNT, STATE_CARDINALITY
from .research_a2_features import (
    ALTERNATIVE_FAMILY_CATALOG,
    CUBIC_FAMILY,
    TYPED_FAMILY_CATALOG,
    a2_head_value,
)


MATCHED = "matched"
TYPED_MISSPECIFIED = "typed_misspecified"
GENERIC_MISSPECIFIED = "generic_misspecified"
MISSPECIFICATION_CONDITIONS = (
    MATCHED,
    TYPED_MISSPECIFIED,
    GENERIC_MISSPECIFIED,
)


@dataclass(frozen=True)
class A2WorldSpec:
    world_index: int
    graph: Graph
    families: tuple[str, str]
    multipliers: tuple[int, ...]
    coefficients: tuple[int, int]
    condition: str = MATCHED


def family_pairs_for_condition(condition: str) -> tuple[tuple[str, str], ...]:
    if condition == MATCHED:
        return tuple(product(TYPED_FAMILY_CATALOG, repeat=2))
    if condition == TYPED_MISSPECIFIED:
        catalog = ALTERNATIVE_FAMILY_CATALOG
        special = CUBIC_FAMILY
    elif condition == GENERIC_MISSPECIFIED:
        catalog = TYPED_FAMILY_CATALOG
        special = "quadratic_target"
    else:
        raise ValueError(f"unknown A2 condition: {condition}")
    return tuple(pair for pair in product(catalog, repeat=2) if special in pair)


def balanced_world_specs(
    world_count: int,
    condition: str,
    rng: np.random.Generator,
) -> tuple[A2WorldSpec, ...]:
    """Balance graphs and eligible family pairs as evenly as integer counts allow."""

    if world_count <= 0:
        raise ValueError("world_count must be positive")
    pairs = family_pairs_for_condition(condition)
    graphs = [
        GRAPH_MANIFEST[index % len(GRAPH_MANIFEST)] for index in range(world_count)
    ]
    families = [pairs[index % len(pairs)] for index in range(world_count)]
    rng.shuffle(graphs)
    rng.shuffle(families)
    specs = []
    for world_index, (graph, family_pair) in enumerate(
        zip(graphs, families, strict=True)
    ):
        specs.append(
            A2WorldSpec(
                world_index=world_index,
                graph=graph,
                families=family_pair,
                multipliers=tuple(
                    int(value) for value in rng.integers(1, 4, LAYER_COUNT)
                ),
                coefficients=tuple(int(value) for value in rng.integers(1, 4, 2)),
                condition=condition,
            )
        )
    return tuple(specs)


def paired_misspecification_specs(
    world_count: int, rng: np.random.Generator
) -> tuple[tuple[A2WorldSpec, ...], tuple[A2WorldSpec, ...]]:
    """Create aligned cubic and quadratic populations for the directional audit."""

    cubic_specs = balanced_world_specs(world_count, TYPED_MISSPECIFIED, rng)
    quadratic_specs = tuple(
        A2WorldSpec(
            world_index=spec.world_index,
            graph=spec.graph,
            families=tuple(
                "quadratic_target" if family == CUBIC_FAMILY else family
                for family in spec.families
            ),  # type: ignore[arg-type]
            multipliers=spec.multipliers,
            coefficients=spec.coefficients,
            condition=GENERIC_MISSPECIFIED,
        )
        for spec in cubic_specs
    )
    return cubic_specs, quadratic_specs


def deterministic_a2_successor(
    source: Sequence[int], action: Sequence[int], spec: A2WorldSpec
) -> tuple[int, ...]:
    target, graph_sources = spec.graph
    delta = [
        int(spec.multipliers[index]) * int(action[index])
        for index in range(LAYER_COUNT)
    ]
    for edge, source_index in enumerate(graph_sources):
        delta[target] += int(spec.coefficients[edge]) * a2_head_value(
            spec.families[edge], source, action, source_index, target
        )
    return tuple(
        (int(source[index]) + delta[index]) % STATE_CARDINALITY
        for index in range(LAYER_COUNT)
    )


def _noisy_observation(
    center: Sequence[int], probability: float, rng: np.random.Generator
) -> tuple[int, ...]:
    observed = []
    for value in center:
        if rng.random() < probability:
            shift = int(rng.integers(1, STATE_CARDINALITY))
            observed.append((int(value) + shift) % STATE_CARDINALITY)
        else:
            observed.append(int(value))
    return tuple(observed)


def _primitive_action(rng: np.random.Generator) -> tuple[int, ...]:
    action = [0] * LAYER_COUNT
    action[int(rng.integers(0, LAYER_COUNT))] = int(rng.integers(1, 3))
    return tuple(action)


def _composition_action(
    spec: A2WorldSpec, index: int, rng: np.random.Generator
) -> tuple[tuple[int, ...], str]:
    target, sources = spec.graph
    available = tuple(layer for layer in range(LAYER_COUNT) if layer != target)
    stratum = index % 3
    if stratum == 0:
        pair = sources
        name = "both_true_mechanisms"
    elif stratum == 1:
        true_source = sources[index % 2]
        distractors = tuple(layer for layer in available if layer not in sources)
        pair = (true_source, distractors[int(rng.integers(0, len(distractors)))])
        name = "one_true_mechanism"
    else:
        distractors = tuple(layer for layer in available if layer not in sources)
        pair = (distractors[0], distractors[1])
        name = "distractor_composition"
    action = [0] * LAYER_COUNT
    for layer in pair:
        action[layer] = int(rng.integers(1, 3))
    return tuple(action), name


def generate_a2_cases(
    spec: A2WorldSpec,
    count: int,
    rng: np.random.Generator,
    *,
    composition: bool,
    noise_probability: float,
) -> tuple[TransitionCase, ...]:
    if count <= 0:
        raise ValueError("case count must be positive")
    if not 0.0 <= noise_probability <= 1.0:
        raise ValueError("noise probability must lie in [0, 1]")
    cases = []
    for index in range(count):
        source = tuple(
            int(value) for value in rng.integers(0, STATE_CARDINALITY, LAYER_COUNT)
        )
        if composition:
            action, stratum = _composition_action(spec, index, rng)
        else:
            action, stratum = _primitive_action(rng), "primitive"
        center = deterministic_a2_successor(source, action, spec)
        cases.append(
            TransitionCase(
                source=source,
                action=action,
                observed=_noisy_observation(center, noise_probability, rng),
                center=center,
                composition_stratum=stratum,
            )
        )
    return tuple(cases)


def generate_coupled_noise_cases(
    spec: A2WorldSpec,
    count: int,
    rng: np.random.Generator,
    noise_probabilities: Sequence[float],
) -> dict[float, tuple[TransitionCase, ...]]:
    """Generate primitive streams with nested corruption masks across noise levels."""

    levels = tuple(float(value) for value in noise_probabilities)
    if not levels or levels != tuple(sorted(set(levels))):
        raise ValueError("noise probabilities must be unique and increasing")
    if levels[0] < 0.0 or levels[-1] > 1.0:
        raise ValueError("noise probabilities must lie in [0, 1]")
    streams: dict[float, list[TransitionCase]] = {level: [] for level in levels}
    for _ in range(count):
        source = tuple(
            int(value) for value in rng.integers(0, STATE_CARDINALITY, LAYER_COUNT)
        )
        action = _primitive_action(rng)
        center = deterministic_a2_successor(source, action, spec)
        uniforms = rng.random(LAYER_COUNT)
        shifts = rng.integers(1, STATE_CARDINALITY, LAYER_COUNT)
        for level in levels:
            observed = tuple(
                (int(center[layer]) + int(shifts[layer])) % STATE_CARDINALITY
                if uniforms[layer] < level
                else int(center[layer])
                for layer in range(LAYER_COUNT)
            )
            streams[level].append(
                TransitionCase(
                    source=source,
                    action=action,
                    observed=observed,
                    center=center,
                    composition_stratum="primitive",
                )
            )
    return {level: tuple(cases) for level, cases in streams.items()}
