"""Prospective design contract for resolving the Paper 3/4 review findings.

This module contains design constants only.  Confirmatory random seeds and
results deliberately live outside the source tree and must be created after a
source freeze has been recorded.
"""

from __future__ import annotations

from hashlib import sha256
import json


RESOLUTION_CONTRACT_ID = "TSI-P34-REVIEW-RESOLUTION-v1"
STATE_CARDINALITY = 7
LAYER_COUNT = 5
HEAD_FAMILIES = ("linear_target", "quadratic_target", "source_target")

DEVELOPMENT_WORLD_COUNT = 24
CONFIRMATORY_WORLD_COUNT = 120
TRAIN_CASES_PER_WORLD = 800
SELECTION_CASES_PER_WORLD = 400
OOD_CASES_PER_WORLD = 1_200
ROLLOUTS_PER_WORLD = 160
ROLLOUT_HORIZON = 6
CRITERION_PREFIX = 3

TRAIN_NOISE_PROBABILITY = 0.08
OOD_NOISE_PROBABILITY = 0.12
FAMILYWISE_ALPHA = 0.05
PRIMARY_INFERENTIAL_EFFECT_COUNT = 8

# Post-freeze reporting clarification.  The v1 payload froze only the divisor
# above, not these names; keeping this tuple outside contract_payload() preserves
# the historical contract digest rather than presenting this list as preregistered.
PRIMARY_INFERENTIAL_QUANTITIES = (
    "routing_identification_rate",
    "learned_routing_nll",
    "factorized_graph_nll",
    "generic_graph_nll",
    "large_generic_graph_nll",
    "rollout_hamming",
    "criterion_brier",
    "outside_family_nll_difference",
)

# These are prospective practical-effect thresholds, not confidence limits.
ROUTING_IDENTIFICATION_RATE_MINIMUM = 0.90
LEARNED_ROUTING_NLL_SESOI = 0.04
FACTORIZED_GRAPH_NLL_SESOI = 0.04
GENERIC_GRAPH_NLL_SESOI = 0.015
DENSE_GRAPH_NLL_SESOI = 0.01
MATCHED_HEAD_EQUIVALENCE_MARGIN = 0.01
ROLLOUT_HAMMING_SESOI = 0.025
CRITERION_BRIER_SESOI = 0.005
OUTSIDE_FAMILY_NONINFERIORITY_MARGIN = 0.03

PRIMARY_GATES = (
    "learned_graph_and_head_identification",
    "learned_vs_wrong_graph_composition_nll",
    "factorized_graph_effect",
    "generic_head_graph_effect",
    "dense_head_graph_effect",
    "matched_head_predictive_equivalence",
    "learned_vs_wrong_graph_rollout_hamming",
    "correct_vs_wrong_routing_criterion_brier",
    "outside_original_linear_family_noninferiority",
)

NONNEGOTIABLE_POLICIES = (
    "All graph and head-family choices are made from train and routing-selection cases only.",
    "The OOD partition contains two-coordinate actions and is never used for fitting or selection.",
    "At least two causal edges are simultaneously active in the designated composition stratum.",
    "Two of the three generator head families are outside the original linear TSI transition equation.",
    "Graph and head are crossed as a complete two-by-two design for factorized and generic heads.",
    "The parameter-matched generic head has exactly seven active modular coefficients, equal to the factorized head.",
    "Larger 55-parameter generic heads are reported as conservative sensitivity controls, not capacity-matched controls.",
    "Observed stochastic transitions define the inferential outcomes; noiseless centers are audit outcomes only.",
    "World is the independent unit; cases and rollouts are nested observations.",
    "Development data may calibrate the criterion map and power calculation but may not enter confirmation.",
    "The contract and implementation hashes are frozen before the confirmatory root seed is generated.",
    "A failed primary gate is reported as a resolved negative result and cannot be repaired within this cohort.",
)


def contract_payload() -> dict[str, object]:
    return {
        "identifier": RESOLUTION_CONTRACT_ID,
        "state_cardinality": STATE_CARDINALITY,
        "layer_count": LAYER_COUNT,
        "head_families": list(HEAD_FAMILIES),
        "sample_sizes": {
            "development_worlds": DEVELOPMENT_WORLD_COUNT,
            "confirmatory_worlds": CONFIRMATORY_WORLD_COUNT,
            "train_cases_per_world": TRAIN_CASES_PER_WORLD,
            "selection_cases_per_world": SELECTION_CASES_PER_WORLD,
            "ood_cases_per_world": OOD_CASES_PER_WORLD,
            "rollouts_per_world": ROLLOUTS_PER_WORLD,
            "rollout_horizon": ROLLOUT_HORIZON,
            "criterion_prefix": CRITERION_PREFIX,
        },
        "noise": {
            "train": TRAIN_NOISE_PROBABILITY,
            "ood": OOD_NOISE_PROBABILITY,
        },
        "familywise_alpha": FAMILYWISE_ALPHA,
        "primary_inferential_effect_count": PRIMARY_INFERENTIAL_EFFECT_COUNT,
        "multiplicity_rule": "bonferroni_simultaneous_two_sided_intervals",
        "thresholds": {
            "routing_identification_rate_minimum": ROUTING_IDENTIFICATION_RATE_MINIMUM,
            "learned_routing_nll_sesoi": LEARNED_ROUTING_NLL_SESOI,
            "factorized_graph_nll_sesoi": FACTORIZED_GRAPH_NLL_SESOI,
            "generic_graph_nll_sesoi": GENERIC_GRAPH_NLL_SESOI,
            "dense_graph_nll_sesoi": DENSE_GRAPH_NLL_SESOI,
            "matched_head_equivalence_margin": MATCHED_HEAD_EQUIVALENCE_MARGIN,
            "rollout_hamming_sesoi": ROLLOUT_HAMMING_SESOI,
            "criterion_brier_sesoi": CRITERION_BRIER_SESOI,
            "outside_family_noninferiority_margin": OUTSIDE_FAMILY_NONINFERIORITY_MARGIN,
        },
        "primary_gates": list(PRIMARY_GATES),
        "policies": list(NONNEGOTIABLE_POLICIES),
    }


def contract_digest() -> str:
    return sha256(
        json.dumps(contract_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_contract() -> dict[str, object]:
    errors: list[str] = []
    if len(HEAD_FAMILIES) < 3:
        errors.append("at least three head families are required")
    if CONFIRMATORY_WORLD_COUNT < 100:
        errors.append("the confirmatory cohort must contain at least 100 worlds")
    if CRITERION_PREFIX >= ROLLOUT_HORIZON:
        errors.append("criterion observations must precede the terminal outcome")
    if not 0.0 < TRAIN_NOISE_PROBABILITY < OOD_NOISE_PROBABILITY < 0.5:
        errors.append("the stochastic OOD shift is invalid")
    if len(PRIMARY_GATES) != 9:
        errors.append("the nine review-resolution gates changed")
    if len(PRIMARY_INFERENTIAL_QUANTITIES) != PRIMARY_INFERENTIAL_EFFECT_COUNT:
        errors.append("the named inferential quantities do not match the frozen divisor")
    return {
        "identifier": RESOLUTION_CONTRACT_ID,
        "contract_digest": contract_digest(),
        "errors": errors,
        "passed": not errors,
    }
