"""Finite-sample envelopes for the Research A support-recovery design.

The bounds apply to the graph-conditioned modulo-7 primitive-intervention
family. They are sufficient upper bounds, not estimates of an empirical
crossing point.
"""

from __future__ import annotations

from math import ceil, exp, log
from itertools import combinations


STATE_CARDINALITY = 7
TRUE_TERM_COUNT = 7
STRUCTURED_SUPPORT_COUNT = 9
GENERIC_FEATURE_COUNT = 11
OUTPUT_COUNT = 5
NONZERO_COEFFICIENT_COUNT = STATE_CARDINALITY - 1
STRUCTURED_CLASS_COUNT = (
    STRUCTURED_SUPPORT_COUNT * NONZERO_COEFFICIENT_COUNT**TRUE_TERM_COUNT
)
STRUCTURED_DIRECT_COMPARISON_COUNT = 5 * (
    NONZERO_COEFFICIENT_COUNT - 1
)
STRUCTURED_EDGE_COMPARISON_COUNT = 2 * (
    3 * NONZERO_COEFFICIENT_COUNT - 1
)
MINIMUM_INFORMATIVE_ROW_PROBABILITY = 6.0 / 35.0
GENERIC_CANDIDATE_MOVE_COUNT = (
    OUTPUT_COUNT * GENERIC_FEATURE_COUNT * NONZERO_COEFFICIENT_COUNT
)
GENERIC_ADAPTIVE_COMPARISON_COUNT = (
    (2**TRUE_TERM_COUNT - 1) * GENERIC_CANDIDATE_MOVE_COUNT
)
HEAD_FAMILIES = ("linear_target", "quadratic_target", "source_target")


def _head_basis(family: str, source_state: int, target_state: int) -> int:
    if family == "linear_target":
        return (1 + target_state) % STATE_CARDINALITY
    if family == "quadratic_target":
        return (1 + target_state) ** 2 % STATE_CARDINALITY
    if family == "source_target":
        return (1 + source_state + target_state) % STATE_CARDINALITY
    raise ValueError(f"unknown head family: {family}")


def minimum_scaled_head_disagreement() -> float:
    """Exhaust the finite field to audit the 5/7 family-separation constant."""

    fractions = []
    states = tuple(
        (source, target)
        for source in range(STATE_CARDINALITY)
        for target in range(STATE_CARDINALITY)
    )
    for first, second in combinations(HEAD_FAMILIES, 2):
        for first_coefficient in range(1, STATE_CARDINALITY):
            for second_coefficient in range(1, STATE_CARDINALITY):
                disagreements = sum(
                    (
                        first_coefficient * _head_basis(first, source, target)
                        - second_coefficient * _head_basis(second, source, target)
                    )
                    % STATE_CARDINALITY
                    != 0
                    for source, target in states
                )
                fractions.append(disagreements / len(states))
    return min(fractions)


def maximum_false_useful_agreement() -> float:
    """Audit the 2/7 upper bound for one false feature imitating a true cell."""

    states = tuple(
        (source, target)
        for source in range(STATE_CARDINALITY)
        for target in range(STATE_CARDINALITY)
    )
    bases = ("direct",) + HEAD_FAMILIES

    def value(basis: str, source: int, target: int) -> int:
        return 1 if basis == "direct" else _head_basis(basis, source, target)

    maximum = 0.0
    for true_basis in bases:
        for true_coefficient in range(1, STATE_CARDINALITY):
            true_values = tuple(
                true_coefficient * value(true_basis, source, target)
                % STATE_CARDINALITY
                for source, target in states
            )
            for false_basis in bases:
                for false_coefficient in range(1, STATE_CARDINALITY):
                    false_values = tuple(
                        false_coefficient * value(false_basis, source, target)
                        % STATE_CARDINALITY
                        for source, target in states
                    )
                    if false_values == true_values:
                        continue
                    useful = sum(
                        truth != 0 and candidate == truth
                        for truth, candidate in zip(
                            true_values, false_values, strict=True
                        )
                    )
                    maximum = max(maximum, useful / len(states))
    return maximum


def audit_research_a_constants() -> dict[str, float | int | bool]:
    """Return the exhaustive finite-population audit used by the proof."""

    minimum_disagreement = minimum_scaled_head_disagreement()
    maximum_agreement = maximum_false_useful_agreement()
    return {
        "minimum_scaled_head_disagreement": minimum_disagreement,
        "maximum_false_useful_agreement": maximum_agreement,
        "generic_candidate_move_count": GENERIC_CANDIDATE_MOVE_COUNT,
        "generic_adaptive_comparison_count": GENERIC_ADAPTIVE_COMPARISON_COUNT,
        "passed": (
            minimum_disagreement >= 5.0 / 7.0
            and maximum_agreement <= 2.0 / 7.0
            and GENERIC_ADAPTIVE_COMPARISON_COUNT == 41910
        ),
    }


def qary_mode_gap(noise_probability: float) -> float:
    """Return P(center)-P(a fixed wrong value) for symmetric modulo-7 noise."""

    if not 0.0 <= noise_probability < 6.0 / 7.0:
        raise ValueError("noise_probability must lie in [0, 6/7)")
    return 1.0 - noise_probability - noise_probability / 6.0


def typed_recovery_failure_bound(sample_size: int, noise_probability: float) -> float:
    """Bound failure of seven known-support modal coefficient estimates.

    Edge rows are informative with probability at least (1/5)(6/7)=6/35.
    A Chernoff lower-tail event controls informative-row coverage, followed by
    Hoeffding and a union bound over seven coefficients and six wrong values.
    """

    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    gap = qary_mode_gap(noise_probability)
    probability = MINIMUM_INFORMATIVE_ROW_PROBABILITY
    coverage_failure = TRUE_TERM_COUNT * exp(-sample_size * probability / 8.0)
    modal_failure = (
        TRUE_TERM_COUNT
        * NONZERO_COEFFICIENT_COUNT
        * exp(-sample_size * probability * gap * gap / 4.0)
    )
    return min(1.0, coverage_failure + modal_failure)


def generic_population_improvement_margin(noise_probability: float) -> float:
    """Minimum true-versus-false greedy improvement gap per random row."""

    return 4.0 * qary_mode_gap(noise_probability) / 35.0


def generic_greedy_failure_bound(sample_size: int, noise_probability: float) -> float:
    """Uniform bound for seven-step greedy recovery under the margin lemma.

    The union covers every proper subset of the seven true disjoint support
    cells and every output-feature-coefficient move. Per-row differences of two
    empirical improvements lie in [-2, 2].
    """

    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    margin = generic_population_improvement_margin(noise_probability)
    bound = GENERIC_ADAPTIVE_COMPARISON_COUNT * exp(
        -sample_size * margin * margin / 8.0
    )
    return min(1.0, bound)


def typed_sufficient_sample_size(noise_probability: float, failure_probability: float) -> int:
    """Closed-form sufficient n making the typed union bound at most delta."""

    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    gap = qary_mode_gap(noise_probability)
    probability = MINIMUM_INFORMATIVE_ROW_PROBABILITY
    coverage = (8.0 / probability) * log(14.0 / failure_probability)
    modal = (4.0 / (probability * gap * gap)) * log(
        84.0 / failure_probability
    )
    return ceil(max(coverage, modal))


def structured_erm_failure_bound(sample_size: int, noise_probability: float) -> float:
    """Bound exact recovery by factorized ERM over supports and coefficients.

    Direct cells differ on at least 1/5 of rows. Every wrong family-coefficient
    choice for an edge differs on at least 1/7. The factorized objective permits
    local comparisons rather than a union over all 9 * 6**7 joint functions.
    """

    if sample_size < 0:
        raise ValueError("sample_size must be nonnegative")
    gap = qary_mode_gap(noise_probability)
    direct = STRUCTURED_DIRECT_COMPARISON_COUNT * exp(
        -sample_size * gap * gap / 50.0
    )
    edge = STRUCTURED_EDGE_COMPARISON_COUNT * exp(
        -sample_size * gap * gap / 98.0
    )
    return min(1.0, direct + edge)


def structured_erm_sufficient_sample_size(
    noise_probability: float, failure_probability: float
) -> int:
    """Closed-form sufficient n for full typed support-and-coefficient ERM."""

    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    gap = qary_mode_gap(noise_probability)
    direct = 50.0 * log(
        2.0 * STRUCTURED_DIRECT_COMPARISON_COUNT / failure_probability
    ) / (gap * gap)
    edge = 98.0 * log(
        2.0 * STRUCTURED_EDGE_COMPARISON_COUNT / failure_probability
    ) / (gap * gap)
    return ceil(
        max(direct, edge)
    )


def generic_sufficient_sample_size(noise_probability: float, failure_probability: float) -> int:
    """Closed-form sufficient n for the conservative greedy union bound."""

    if not 0.0 < failure_probability < 1.0:
        raise ValueError("failure_probability must lie in (0, 1)")
    margin = generic_population_improvement_margin(noise_probability)
    return ceil(
        8.0
        * log(GENERIC_ADAPTIVE_COMPARISON_COUNT / failure_probability)
        / (margin * margin)
    )
