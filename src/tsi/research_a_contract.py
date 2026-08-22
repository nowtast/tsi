"""Frozen-design candidate for Research A1 matched-dictionary efficiency."""

from __future__ import annotations

from hashlib import sha256
import json


CONTRACT_ID = "TSI-RESEARCH-A1-MATCHED-SAMPLE-EFFICIENCY-v1"
STATUS = "freeze_candidate_no_confirmatory_seed"
WORLD_COUNT = 126
PRIMARY_SAMPLE_SIZES = (5, 10, 15, 20, 25, 30, 40, 50)
TEST_CASE_COUNT = 1200
TRAIN_NOISE = 0.08
OOD_NOISE = 0.12
FAMILYWISE_ALPHA = 0.05
NLL_SESOI = 0.01
NLL_EQUIVALENCE_MARGIN = 0.01
RECOVERY_SESOI = 0.10
RECOVERY_EQUIVALENCE_MARGIN = 0.05
PRIMARY_ENDPOINTS = tuple(
    [f"generic_minus_typed_composition_nll_n{size}" for size in PRIMARY_SAMPLE_SIZES]
    + [f"typed_minus_generic_exact_recovery_n{size}" for size in PRIMARY_SAMPLE_SIZES]
)

REVIEW_SAFEGUARDS = (
    "arm_specific_estimands_no_cross_arm_average",
    "all_stochastic_multiplicity_members_named",
    "sesoi_applied_to_point_estimate_and_interval_sign_frozen",
    "prospective_power_uses_development_only",
    "misspecification_excluded_from_matched_primary_estimand",
    "world_level_distributions_and_quantiles_reported",
    "cleanroom_scope_named_before_confirmation",
    "world_derivation_and_balance_audited",
    "printed_digests_are_actual_file_sha256_values",
    "external_replay_requested_and_not_called_independent_until_completed",
)

POLICIES = (
    "The graph assigned to a world is supplied identically to every arm.",
    "No arm observes the generating head families or coefficients.",
    "All sample sizes are nested prefixes of one training stream per world.",
    "The held-out composition set is never used for fitting or selection.",
    "Typed and isomorphic generic arms enumerate the same functions and must agree exactly.",
    "The unstructured arm takes exactly seven greedy moves over 55 positions and six nonzero coefficients.",
    "World is the independent unit; rows within a world are nested observations.",
    "The 16 explicitly named paired world-level quantities form the Bonferroni family.",
    "A joint advantage requires both lower bounds above zero and both point estimates at or above their SESOI.",
    "Joint equivalence requires both simultaneous intervals inside their frozen margins.",
    "A transition is an interval from the largest joint-advantage n to the smallest later joint-equivalence n; no point n-star is estimated.",
    "Candidate-width, misspecification, and noise-boundary studies are separate robustness estimands and cannot repair A1.",
    "No confirmatory seed is generated until source freeze and a public seed commitment precedes one-shot execution.",
)


def contract_payload() -> dict[str, object]:
    return {
        "contract_id": CONTRACT_ID,
        "status": STATUS,
        "population": {
            "state": "uniform Z_7^5",
            "graph": "common-target two-source graph, supplied to all arms",
            "graph_manifest_size": 30,
            "family_pairs": "nine ordered pairs, exactly 14 worlds per pair",
            "generator_coefficients": [1, 2, 3],
            "selector_nonzero_coefficients": [1, 2, 3, 4, 5, 6],
            "train_action": "uniform primitive coordinate, magnitude 1 or 2",
            "train_noise": TRAIN_NOISE,
            "ood_noise": OOD_NOISE,
        },
        "sample_sizes": {
            "world_count": WORLD_COUNT,
            "train_prefixes": list(PRIMARY_SAMPLE_SIZES),
            "test_cases_per_world": TEST_CASE_COUNT,
        },
        "arms": [
            "typed_structured_search",
            "generic_isomorphic_search",
            "generic_unstructured_greedy_55_positions_7_moves",
        ],
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "multiplicity": {
            "method": "bonferroni_simultaneous_two_sided_normal_intervals",
            "familywise_alpha": FAMILYWISE_ALPHA,
            "divisor": len(PRIMARY_ENDPOINTS),
        },
        "thresholds": {
            "nll_sesoi": NLL_SESOI,
            "nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
            "recovery_sesoi": RECOVERY_SESOI,
            "recovery_equivalence_margin": RECOVERY_EQUIVALENCE_MARGIN,
        },
        "review_safeguards": list(REVIEW_SAFEGUARDS),
        "policies": list(POLICIES),
    }


def contract_digest() -> str:
    return sha256(
        json.dumps(contract_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_contract() -> dict[str, object]:
    errors = []
    if WORLD_COUNT % 9:
        errors.append("world count does not balance the nine family pairs")
    if len(PRIMARY_ENDPOINTS) != 16 or len(set(PRIMARY_ENDPOINTS)) != 16:
        errors.append("the 16 primary multiplicity members are not unique")
    if len(REVIEW_SAFEGUARDS) != 10:
        errors.append("the ten review safeguards are incomplete")
    if PRIMARY_SAMPLE_SIZES != tuple(sorted(set(PRIMARY_SAMPLE_SIZES))):
        errors.append("the primary sample-size grid is not strictly increasing")
    return {
        "contract_id": CONTRACT_ID,
        "contract_digest": contract_digest(),
        "errors": errors,
        "passed": not errors,
    }
