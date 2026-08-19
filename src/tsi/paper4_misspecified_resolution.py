"""Outside-model-family composition stress test for Paper 4."""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np

from .paper34_resolution_benchmark import (
    TransitionCase,
    center_accuracy,
    coordinate_nll,
    fit_factorized,
    generate_cases,
    learn_factorized,
    wrong_graph,
    world_spec,
)
from .paper34_resolution_contract import (
    OOD_CASES_PER_WORLD,
    OOD_NOISE_PROBABILITY,
    SELECTION_CASES_PER_WORLD,
    STATE_CARDINALITY,
    TRAIN_CASES_PER_WORLD,
    TRAIN_NOISE_PROBABILITY,
)
from .paper4_misspecified_contract import FAMILYWISE_ALPHA, GRAPH_NLL_SESOI


def _stress_cases(spec, cases: tuple[TransitionCase, ...]) -> tuple[TransitionCase, ...]:
    target, sources = spec.graph
    context = next(layer for layer in range(5) if layer != target and layer not in sources)
    coefficient = 1 + spec.world_index % 3
    stressed = []
    for case in cases:
        synergy = (
            coefficient
            * case.action[sources[0]]
            * case.action[sources[1]]
            * (1 + case.source[context])
        ) % STATE_CARDINALITY
        center = list(case.center)
        observed = list(case.observed)
        noise_shift = (observed[target] - center[target]) % STATE_CARDINALITY
        center[target] = (center[target] + synergy) % STATE_CARDINALITY
        observed[target] = (center[target] + noise_shift) % STATE_CARDINALITY
        stressed.append(
            TransitionCase(
                case.source,
                case.action,
                tuple(observed),
                tuple(center),
                case.composition_stratum,
            )
        )
    return tuple(stressed)


def run_stress_world(world_index: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    spec = world_spec(world_index, rng)
    train = generate_cases(spec, TRAIN_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
    selection = generate_cases(spec, SELECTION_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
    ordinary_test = generate_cases(spec, OOD_CASES_PER_WORLD, rng, composition=True, noise_probability=OOD_NOISE_PROBABILITY)
    test = _stress_cases(spec, ordinary_test)
    learned = learn_factorized(train, selection)
    wrong = fit_factorized(train, wrong_graph(spec.graph), spec.families, name="wrong_graph_correct_head")
    learned_nll = coordinate_nll(learned, test)
    wrong_nll = coordinate_nll(wrong, test)
    both = tuple(case for case in test if case.composition_stratum == "both_true_mechanisms")
    return {
        "world_index": world_index,
        "graph_exact": learned.graph == spec.graph,
        "head_exact": learned.families == spec.families,
        "learned_nll": learned_nll,
        "wrong_nll": wrong_nll,
        "graph_nll_effect": wrong_nll - learned_nll,
        "learned_center_accuracy": center_accuracy(learned, test),
        "learned_both_mechanisms_center_accuracy": center_accuracy(learned, both),
        "all_primitive_training_synergies_zero": all(
            case.action[spec.graph[1][0]] * case.action[spec.graph[1][1]] == 0
            for case in (*train, *selection)
        ),
    }


def summarize_stress(rows: tuple[dict[str, object], ...]) -> dict[str, object]:
    effects = np.asarray([row["graph_nll_effect"] for row in rows], dtype=float)
    mean = float(np.mean(effects))
    sd = float(np.std(effects, ddof=1))
    se = sd / sqrt(len(effects))
    critical = NormalDist().inv_cdf(1.0 - FAMILYWISE_ALPHA / 2.0)
    lower, upper = mean - critical * se, mean + critical * se
    return {
        "world_count": len(rows),
        "graph_and_head_identification_rate": float(np.mean([row["graph_exact"] and row["head_exact"] for row in rows])),
        "graph_nll_effect": {"mean": mean, "world_sd": sd, "lower_95": lower, "upper_95": upper},
        "worlds_with_nonexact_learned_center": sum(float(row["learned_center_accuracy"]) < 1.0 for row in rows),
        "mean_both_mechanisms_center_accuracy": float(np.mean([row["learned_both_mechanisms_center_accuracy"] for row in rows])),
        "training_synergy_zero_in_all_worlds": all(row["all_primitive_training_synergies_zero"] for row in rows),
        "gates": {
            "outside_model_family_nonexact_in_every_world": all(float(row["learned_center_accuracy"]) < 1.0 for row in rows),
            "learned_graph_effect_positive": lower > 0.0 and mean >= GRAPH_NLL_SESOI,
            "train_only_identification_exact": all(row["graph_exact"] and row["head_exact"] for row in rows),
            "synergy_dormant_during_training": all(row["all_primitive_training_synergies_zero"] for row in rows),
        },
    }
