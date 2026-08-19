#!/usr/bin/env python3
"""Reproducible implementation audits for TSI Extension 2B-X2."""

from __future__ import annotations

import json
import random
from pathlib import Path

from tsi.attribute_geometry import (
    FiniteAttributedMetricMeasureState,
    FiniteAttributedMetricState,
    attribute_aware_discrepancy,
    coupling_perturbation_audit,
    empirical_mass,
    find_attribute_preserving_isometry,
    finite_support_tv_radius,
    total_variation,
)


SEED = 20_260_727
TRIALS = 2_000
SAMPLE_SIZE = 200
FAILURE_PROBABILITY = 0.05


def attribute_distance(left: int, right: int) -> float:
    return float(abs(left - right))


def states() -> tuple[FiniteAttributedMetricState, FiniteAttributedMetricState]:
    left = FiniteAttributedMetricState(
        ("x0", "x1", "x2"),
        (
            (0.0, 1.0, 3.0),
            (1.0, 0.0, 2.0),
            (3.0, 2.0, 0.0),
        ),
        ("anchor", "part", "part"),
        (0, 1, 2),
    )
    right = FiniteAttributedMetricState(
        ("y2", "y0", "y1"),
        (
            (0.0, 3.0, 2.0),
            (3.0, 0.0, 1.0),
            (2.0, 1.0, 0.0),
        ),
        ("part", "anchor", "part"),
        (2, 0, 1),
    )
    return left, right


def draw(
    rng: random.Random,
    carrier: tuple[str, ...],
    mass: tuple[float, ...],
    size: int,
) -> tuple[str, ...]:
    return tuple(rng.choices(carrier, weights=mass, k=size))


def exact_audit() -> dict[str, object]:
    left, right = states()
    discrepancy = attribute_aware_discrepancy(
        left,
        right,
        attribute_distance,
    )
    isometry = find_attribute_preserving_isometry(
        left,
        right,
        attribute_distance,
    )
    return {
        "attribute_discrepancy": discrepancy,
        "isometry": dict(isometry) if isometry is not None else None,
        "zero_iff_witness_found": discrepancy == 0.0 and isometry is not None,
    }


def sampling_audit(rng: random.Random) -> dict[str, object]:
    left, right = states()
    left_mass = (0.5, 0.3, 0.2)
    right_mass = (0.2, 0.5, 0.3)
    allocated_failure = FAILURE_PROBABILITY / 2.0
    left_radius = finite_support_tv_radius(
        len(left.entities),
        SAMPLE_SIZE,
        allocated_failure,
    )
    right_radius = finite_support_tv_radius(
        len(right.entities),
        SAMPLE_SIZE,
        allocated_failure,
    )
    joint_violations = 0
    maximum_tv_sum = 0.0
    for _ in range(TRIALS):
        left_empirical = empirical_mass(
            left,
            draw(rng, left.entities, left_mass, SAMPLE_SIZE),
        )
        right_empirical = empirical_mass(
            right,
            draw(rng, right.entities, right_mass, SAMPLE_SIZE),
        )
        left_tv = total_variation(left_empirical, left_mass)
        right_tv = total_variation(right_empirical, right_mass)
        maximum_tv_sum = max(maximum_tv_sum, left_tv + right_tv)
        if left_tv > left_radius or right_tv > right_radius:
            joint_violations += 1
    return {
        "trials": TRIALS,
        "sample_size_per_state": SAMPLE_SIZE,
        "declared_failure_probability": FAILURE_PROBABILITY,
        "source_tv_radius": left_radius,
        "target_tv_radius": right_radius,
        "joint_radius_violations": joint_violations,
        "observed_joint_violation_rate": joint_violations / TRIALS,
        "maximum_observed_tv_sum": maximum_tv_sum,
    }


def perturbation_audit(rng: random.Random) -> dict[str, object]:
    left_state, right_state = states()
    left = FiniteAttributedMetricMeasureState(left_state, (0.5, 0.3, 0.2))
    right = FiniteAttributedMetricMeasureState(right_state, (0.2, 0.5, 0.3))
    coupling = tuple(
        tuple(left_mass * right_mass for right_mass in right.mass)
        for left_mass in left.mass
    )
    violations = 0
    maximum_coupling_slack = 0.0
    maximum_objective_slack = 0.0
    for _ in range(500):
        new_left = empirical_mass(
            left_state,
            draw(rng, left_state.entities, left.mass, 80),
        )
        new_right = empirical_mass(
            right_state,
            draw(rng, right_state.entities, right.mass, 80),
        )
        audit = coupling_perturbation_audit(
            left,
            right,
            coupling,
            new_left,
            new_right,
            attribute_distance,
            p=2.0,
        )
        coupling_slack = (
            audit.coupling_tv - audit.source_tv - audit.target_tv
        )
        objective_slack = audit.objective_difference - audit.objective_bound
        maximum_coupling_slack = max(maximum_coupling_slack, coupling_slack)
        maximum_objective_slack = max(maximum_objective_slack, objective_slack)
        if coupling_slack > 1e-9 or objective_slack > 1e-9:
            violations += 1
    return {
        "trials": 500,
        "violations": violations,
        "maximum_coupling_tv_bound_violation": maximum_coupling_slack,
        "maximum_objective_bound_violation": maximum_objective_slack,
    }


def support_miss_audit(rng: random.Random) -> dict[str, object]:
    rare_mass = 0.02
    size = 50
    trials = 5_000
    misses = 0
    for _ in range(trials):
        if all(
            observation == "common"
            for observation in rng.choices(
                ("common", "rare"),
                weights=(1.0 - rare_mass, rare_mass),
                k=size,
            )
        ):
            misses += 1
    theoretical = (1.0 - rare_mass) ** size
    return {
        "rare_population_mass": rare_mass,
        "sample_size": size,
        "trials": trials,
        "observed_support_miss_rate": misses / trials,
        "exact_support_miss_probability": theoretical,
        "absolute_simulation_error": abs(misses / trials - theoretical),
    }


def main() -> None:
    rng = random.Random(SEED)
    result = {
        "status": "completed",
        "seed": SEED,
        "interpretation": (
            "These simulations audit the executable finite formulas. They do "
            "not prove the concentration theorem and do not validate recovery "
            "of a noisy carrier, metric, label, or attribute map."
        ),
        "exact": exact_audit(),
        "sampling": sampling_audit(rng),
        "coupling_perturbation": perturbation_audit(rng),
        "support_recovery_counterexample": support_miss_audit(rng),
    }
    output = Path(__file__).with_name("results.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

