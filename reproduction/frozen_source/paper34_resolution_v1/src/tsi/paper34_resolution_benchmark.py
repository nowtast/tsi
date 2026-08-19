"""Prospective stochastic benchmark for the Paper 3/4 resolution study."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import log
from typing import Iterable, Sequence

import numpy as np

from .paper34_resolution_contract import (
    CRITERION_PREFIX,
    HEAD_FAMILIES,
    LAYER_COUNT,
    OOD_CASES_PER_WORLD,
    OOD_NOISE_PROBABILITY,
    ROLLOUT_HORIZON,
    ROLLOUTS_PER_WORLD,
    SELECTION_CASES_PER_WORLD,
    STATE_CARDINALITY,
    TRAIN_CASES_PER_WORLD,
    TRAIN_NOISE_PROBABILITY,
)


Graph = tuple[int, tuple[int, int]]


def graph_manifest() -> tuple[Graph, ...]:
    return tuple(
        (target, sources)
        for target in range(LAYER_COUNT)
        for sources in combinations(
            tuple(index for index in range(LAYER_COUNT) if index != target), 2
        )
    )


GRAPH_MANIFEST = graph_manifest()


@dataclass(frozen=True)
class WorldSpec:
    world_index: int
    graph: Graph
    families: tuple[str, str]
    multipliers: tuple[int, ...]
    coefficients: tuple[int, int]


@dataclass(frozen=True)
class TransitionCase:
    source: tuple[int, ...]
    action: tuple[int, ...]
    observed: tuple[int, ...]
    center: tuple[int, ...]
    composition_stratum: str


@dataclass(frozen=True)
class ResolutionModel:
    name: str
    graph: Graph
    families: tuple[str, str] | None
    multipliers: tuple[int, ...] | None
    coefficients: tuple[int, ...] | None
    generic_terms: tuple[tuple[int, int, int], ...] = ()

    @property
    def active_parameter_count(self) -> int:
        if self.generic_terms:
            return len(self.generic_terms)
        return 0 if self.multipliers is None else len(self.multipliers) + 2

    def predict(self, source: Sequence[int], action: Sequence[int]) -> tuple[int, ...]:
        if self.generic_terms:
            features = generic_features(source, action, self.graph)
            delta = [0] * LAYER_COUNT
            for output, feature, coefficient in self.generic_terms:
                delta[output] += coefficient * features[feature]
            return tuple(
                (int(source[index]) + delta[index]) % STATE_CARDINALITY
                for index in range(LAYER_COUNT)
            )
        if self.families is None or self.multipliers is None or self.coefficients is None:
            raise RuntimeError("incomplete factorized model")
        return deterministic_successor(
            source,
            action,
            self.graph,
            self.families,
            self.multipliers,
            self.coefficients,
        )


def head_value(
    family: str, source: Sequence[int], action: Sequence[int], source_index: int, target: int
) -> int:
    active = int(action[source_index])
    if family == "linear_target":
        return active * (1 + int(source[target]))
    if family == "quadratic_target":
        return active * (1 + int(source[target])) ** 2
    if family == "source_target":
        return active * (1 + int(source[source_index]) + int(source[target]))
    raise ValueError(f"unknown head family: {family}")


def deterministic_successor(
    source: Sequence[int],
    action: Sequence[int],
    graph: Graph,
    families: Sequence[str],
    multipliers: Sequence[int],
    coefficients: Sequence[int],
) -> tuple[int, ...]:
    target, graph_sources = graph
    delta = [
        int(multipliers[index]) * int(action[index])
        for index in range(LAYER_COUNT)
    ]
    for edge, source_index in enumerate(graph_sources):
        delta[target] += int(coefficients[edge]) * head_value(
            str(families[edge]), source, action, source_index, target
        )
    return tuple(
        (int(source[index]) + delta[index]) % STATE_CARDINALITY
        for index in range(LAYER_COUNT)
    )


def world_spec(world_index: int, rng: np.random.Generator) -> WorldSpec:
    graph = GRAPH_MANIFEST[int(rng.integers(0, len(GRAPH_MANIFEST)))]
    # One third of worlds instantiate the original linear family.  The other
    # two thirds cycle over all eight ordered pairs outside that family.
    if world_index % 3 == 0:
        families = ("linear_target", "linear_target")
    else:
        outside_pairs = tuple(
            (first, second)
            for first in HEAD_FAMILIES
            for second in HEAD_FAMILIES
            if (first, second) != ("linear_target", "linear_target")
        )
        block = world_index // 3
        offset = 0 if world_index % 3 == 1 else 1
        families = outside_pairs[(2 * block + offset) % len(outside_pairs)]
    return WorldSpec(
        world_index=world_index,
        graph=graph,
        families=families,
        multipliers=tuple(int(value) for value in rng.integers(1, 4, LAYER_COUNT)),
        coefficients=tuple(int(value) for value in rng.integers(1, 4, 2)),
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


def _composition_action(spec: WorldSpec, index: int, rng: np.random.Generator) -> tuple[tuple[int, ...], str]:
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


def generate_cases(
    spec: WorldSpec,
    count: int,
    rng: np.random.Generator,
    *,
    composition: bool,
    noise_probability: float,
) -> tuple[TransitionCase, ...]:
    cases = []
    for index in range(count):
        source = tuple(
            int(value) for value in rng.integers(0, STATE_CARDINALITY, LAYER_COUNT)
        )
        if composition:
            action, stratum = _composition_action(spec, index, rng)
        else:
            action, stratum = _primitive_action(rng), "primitive"
        center = deterministic_successor(
            source,
            action,
            spec.graph,
            spec.families,
            spec.multipliers,
            spec.coefficients,
        )
        cases.append(
            TransitionCase(
                source,
                action,
                _noisy_observation(center, noise_probability, rng),
                center,
                stratum,
            )
        )
    return tuple(cases)


def _mode_parameter(values: Iterable[tuple[int, int]], candidates: range = range(STATE_CARDINALITY)) -> int:
    rows = tuple(values)
    return min(
        candidates,
        key=lambda coefficient: (
            sum((coefficient * feature) % STATE_CARDINALITY != observed for feature, observed in rows),
            coefficient,
        ),
    )


def estimate_multipliers(cases: Sequence[TransitionCase]) -> tuple[int, ...]:
    estimates = []
    for layer in range(LAYER_COUNT):
        rows = []
        for case in cases:
            if case.action[layer] and sum(value != 0 for value in case.action) == 1:
                delta = (case.observed[layer] - case.source[layer]) % STATE_CARDINALITY
                rows.append((case.action[layer], delta))
        estimates.append(_mode_parameter(rows))
    return tuple(estimates)


def fit_factorized(
    cases: Sequence[TransitionCase],
    graph: Graph,
    families: tuple[str, str],
    *,
    name: str,
    multipliers: tuple[int, ...] | None = None,
) -> ResolutionModel:
    fitted_multipliers = multipliers or estimate_multipliers(cases)
    target, sources = graph
    coefficients = []
    for source_index, family in zip(sources, families, strict=True):
        rows = []
        for case in cases:
            if case.action[source_index] and sum(value != 0 for value in case.action) == 1:
                direct = fitted_multipliers[target] * case.action[target]
                residual = (
                    case.observed[target] - case.source[target] - direct
                ) % STATE_CARDINALITY
                rows.append((head_value(family, case.source, case.action, source_index, target), residual))
        coefficients.append(_mode_parameter(rows))
    return ResolutionModel(
        name=name,
        graph=graph,
        families=families,
        multipliers=fitted_multipliers,
        coefficients=tuple(coefficients),
    )


def coordinate_nll(
    model: ResolutionModel,
    cases: Sequence[TransitionCase],
    noise_probability: float = OOD_NOISE_PROBABILITY,
) -> float:
    matched = log(1.0 - noise_probability)
    mismatched = log(noise_probability / (STATE_CARDINALITY - 1))
    total = 0.0
    count = 0
    for case in cases:
        prediction = model.predict(case.source, case.action)
        for predicted, observed in zip(prediction, case.observed, strict=True):
            total -= matched if predicted == observed else mismatched
            count += 1
    return total / count


def center_accuracy(model: ResolutionModel, cases: Sequence[TransitionCase]) -> float:
    return float(
        np.mean(
            [model.predict(case.source, case.action) == case.center for case in cases]
        )
    )


def learn_factorized(
    train: Sequence[TransitionCase], selection: Sequence[TransitionCase]
) -> ResolutionModel:
    multipliers = estimate_multipliers(train)
    best: tuple[float, Graph, tuple[str, str], ResolutionModel] | None = None
    for graph in GRAPH_MANIFEST:
        for first in HEAD_FAMILIES:
            for second in HEAD_FAMILIES:
                families = (first, second)
                model = fit_factorized(
                    train,
                    graph,
                    families,
                    name="learned_factorized",
                    multipliers=multipliers,
                )
                key = (coordinate_nll(model, selection, TRAIN_NOISE_PROBABILITY), graph, families, model)
                if best is None or key[:3] < best[:3]:
                    best = key
    if best is None:
        raise RuntimeError("factorized candidate search produced no model")
    return best[3]


def wrong_graph(graph: Graph) -> Graph:
    return GRAPH_MANIFEST[(GRAPH_MANIFEST.index(graph) + 11) % len(GRAPH_MANIFEST)]


def wrong_families(families: tuple[str, str]) -> tuple[str, str]:
    return tuple(
        HEAD_FAMILIES[(HEAD_FAMILIES.index(family) + 1) % len(HEAD_FAMILIES)]
        for family in families
    )  # type: ignore[return-value]


def generic_features(source: Sequence[int], action: Sequence[int], graph: Graph) -> tuple[int, ...]:
    target, sources = graph
    values = [int(value) for value in action]
    for source_index in sources:
        values.extend(
            head_value(family, source, action, source_index, target)
            for family in HEAD_FAMILIES
        )
    return tuple(value % STATE_CARDINALITY for value in values)


def fit_generic_sparse(
    cases: Sequence[TransitionCase], graph: Graph, budget: int, *, name: str
) -> ResolutionModel:
    x = np.asarray([generic_features(case.source, case.action, graph) for case in cases], dtype=np.int64)
    y = np.asarray(
        [
            [
                (case.observed[layer] - case.source[layer]) % STATE_CARDINALITY
                for layer in range(LAYER_COUNT)
            ]
            for case in cases
        ],
        dtype=np.int64,
    )
    prediction = np.zeros_like(y)
    selected: list[tuple[int, int, int]] = []
    available = {(output, feature) for output in range(LAYER_COUNT) for feature in range(x.shape[1])}
    for _ in range(min(budget, len(available))):
        current_errors = np.sum(prediction != y, axis=0)
        best = None
        for output, feature in sorted(available):
            for coefficient in range(1, STATE_CARDINALITY):
                candidate = (prediction[:, output] + coefficient * x[:, feature]) % STATE_CARDINALITY
                errors = int(np.sum(candidate != y[:, output]))
                improvement = int(current_errors[output]) - errors
                key = (-improvement, output, feature, coefficient)
                if best is None or key < best[0]:
                    best = (key, output, feature, coefficient, candidate)
        if best is None:
            break
        _key, output, feature, coefficient, candidate = best
        prediction[:, output] = candidate
        available.remove((output, feature))
        selected.append((output, feature, coefficient))
    return ResolutionModel(name, graph, None, None, None, tuple(selected))


def fit_generic_dense(
    cases: Sequence[TransitionCase], graph: Graph, *, name: str
) -> ResolutionModel:
    """Fit all 55 graph-conditioned modular coefficients by coordinate descent."""
    x = np.asarray([generic_features(case.source, case.action, graph) for case in cases], dtype=np.int64)
    y = np.asarray(
        [
            [
                (case.observed[layer] - case.source[layer]) % STATE_CARDINALITY
                for layer in range(LAYER_COUNT)
            ]
            for case in cases
        ],
        dtype=np.int64,
    )
    coefficients = np.zeros((LAYER_COUNT, x.shape[1]), dtype=np.int64)
    prediction = np.zeros_like(y)
    for _ in range(8):
        changed = False
        for output in range(LAYER_COUNT):
            for feature in range(x.shape[1]):
                old = int(coefficients[output, feature])
                residual = (prediction[:, output] - old * x[:, feature]) % STATE_CARDINALITY
                candidate = min(
                    range(STATE_CARDINALITY),
                    key=lambda value: (
                        int(np.sum((residual + value * x[:, feature]) % STATE_CARDINALITY != y[:, output])),
                        value,
                    ),
                )
                if candidate != old:
                    coefficients[output, feature] = candidate
                    prediction[:, output] = (residual + candidate * x[:, feature]) % STATE_CARDINALITY
                    changed = True
        if not changed:
            break
    terms = tuple(
        (output, feature, int(coefficients[output, feature]))
        for output in range(LAYER_COUNT)
        for feature in range(x.shape[1])
    )
    return ResolutionModel(name, graph, None, None, None, terms)


def _rollout_action(spec: WorldSpec, index: int, rng: np.random.Generator) -> tuple[int, ...]:
    action, _ = _composition_action(spec, index, rng)
    return action


def rollout_records(
    spec: WorldSpec,
    learned: ResolutionModel,
    wrong: ResolutionModel,
    rng: np.random.Generator,
) -> tuple[dict[str, float], ...]:
    records = []
    learned_target, learned_sources = learned.graph
    wrong_target, wrong_sources = wrong.graph
    for rollout_index in range(ROLLOUTS_PER_WORLD):
        initial = tuple(int(value) for value in rng.integers(0, STATE_CARDINALITY, LAYER_COUNT))
        actual = initial
        learned_open = initial
        wrong_open = initial
        correct_prefix = 0.0
        wrong_prefix = 0.0
        learned_hamming = 0.0
        wrong_hamming = 0.0
        for step in range(ROLLOUT_HORIZON):
            action = _rollout_action(spec, rollout_index + step, rng)
            center = deterministic_successor(
                actual, action, spec.graph, spec.families, spec.multipliers, spec.coefficients
            )
            observed = _noisy_observation(center, OOD_NOISE_PROBABILITY, rng)
            teacher_prediction = learned.predict(actual, action)
            if step < CRITERION_PREFIX:
                if any(action[source] for source in learned_sources):
                    correct_prefix += float(teacher_prediction[learned_target] != observed[learned_target])
                if any(action[source] for source in wrong_sources):
                    wrong_prefix += float(teacher_prediction[wrong_target] != observed[wrong_target])
            learned_open = learned.predict(learned_open, action)
            wrong_open = wrong.predict(wrong_open, action)
            actual = observed
            learned_hamming += sum(a != b for a, b in zip(learned_open, actual, strict=True)) / LAYER_COUNT
            wrong_hamming += sum(a != b for a, b in zip(wrong_open, actual, strict=True)) / LAYER_COUNT
        records.append(
            {
                "correct_score": correct_prefix / CRITERION_PREFIX,
                "wrong_score": wrong_prefix / CRITERION_PREFIX,
                "terminal_failure": float(learned_open[learned_target] != actual[learned_target]),
                "learned_hamming_auc": learned_hamming / ROLLOUT_HORIZON,
                "wrong_hamming_auc": wrong_hamming / ROLLOUT_HORIZON,
            }
        )
    return tuple(records)


def fit_logistic_calibrator(scores: Sequence[float], outcomes: Sequence[float]) -> tuple[float, float]:
    x = np.column_stack((np.ones(len(scores)), np.asarray(scores, dtype=float)))
    y = np.asarray(outcomes, dtype=float)
    coefficients = np.zeros(2, dtype=float)
    for _ in range(100):
        linear = np.clip(x @ coefficients, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.maximum(probability * (1.0 - probability), 1.0e-8)
        hessian = x.T @ (weights[:, None] * x) + np.eye(2) * 1.0e-6
        step = np.linalg.solve(hessian, x.T @ (y - probability))
        coefficients += step
        if float(np.max(np.abs(step))) < 1.0e-10:
            break
    return float(coefficients[0]), float(coefficients[1])


def calibrated_brier(score: Sequence[float], outcome: Sequence[float], calibration: tuple[float, float]) -> float:
    values = np.asarray(score, dtype=float)
    target = np.asarray(outcome, dtype=float)
    probability = 1.0 / (1.0 + np.exp(-np.clip(calibration[0] + calibration[1] * values, -30.0, 30.0)))
    return float(np.mean((probability - target) ** 2))


def run_world(world_index: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    spec = world_spec(world_index, rng)
    train = generate_cases(spec, TRAIN_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
    selection = generate_cases(spec, SELECTION_CASES_PER_WORLD, rng, composition=False, noise_probability=TRAIN_NOISE_PROBABILITY)
    test = generate_cases(spec, OOD_CASES_PER_WORLD, rng, composition=True, noise_probability=OOD_NOISE_PROBABILITY)
    learned = learn_factorized(train, selection)
    graph_wrong = wrong_graph(spec.graph)
    family_wrong = wrong_families(spec.families)
    models = {
        "learned_factorized": learned,
        "correct_graph_correct_head": fit_factorized(train, spec.graph, spec.families, name="correct_graph_correct_head"),
        "wrong_graph_correct_head": fit_factorized(train, graph_wrong, spec.families, name="wrong_graph_correct_head"),
        "correct_graph_wrong_head": fit_factorized(train, spec.graph, family_wrong, name="correct_graph_wrong_head"),
        "wrong_graph_wrong_head": fit_factorized(train, graph_wrong, family_wrong, name="wrong_graph_wrong_head"),
        "correct_graph_generic_7": fit_generic_sparse(train, spec.graph, 7, name="correct_graph_generic_7"),
        "wrong_graph_generic_7": fit_generic_sparse(train, graph_wrong, 7, name="wrong_graph_generic_7"),
        "correct_graph_generic_55": fit_generic_dense(train, spec.graph, name="correct_graph_generic_55"),
        "wrong_graph_generic_55": fit_generic_dense(train, graph_wrong, name="wrong_graph_generic_55"),
    }
    metrics = {
        name: {
            "composition_nll": coordinate_nll(model, test),
            "noiseless_center_accuracy": center_accuracy(model, test),
            "active_parameter_count": model.active_parameter_count,
        }
        for name, model in models.items()
    }
    rollouts = rollout_records(spec, learned, models["wrong_graph_correct_head"], rng)
    return {
        "world_index": world_index,
        "seed": seed,
        "true_graph": [spec.graph[0], list(spec.graph[1])],
        "true_families": list(spec.families),
        "learned_graph": [learned.graph[0], list(learned.graph[1])],
        "learned_families": list(learned.families or ()),
        "graph_exact": learned.graph == spec.graph,
        "head_exact": learned.families == spec.families,
        "outside_original_linear_family": any(family != "linear_target" for family in spec.families),
        "metrics": metrics,
        "rollouts": list(rollouts),
        "case_counts": {"train": len(train), "selection": len(selection), "ood": len(test)},
    }
