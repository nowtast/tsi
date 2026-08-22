"""Numeric draft contract for Research A2; deliberately not frozen or seeded."""

from __future__ import annotations

from hashlib import sha256
import json

from .research_a2_development import (
    MISSPECIFICATION_SAMPLE_SIZES,
    NOISE_PROBABILITIES,
    NOISE_SAMPLE_SIZES,
    WIDTH_SAMPLE_SIZES,
)
from .research_a2_features import WIDTH_POSITION_COUNTS
from .research_a2_power import (
    FAMILYWISE_ALPHA,
    NLL_EQUIVALENCE_MARGIN,
    NLL_SESOI,
    RECOVERY_EQUIVALENCE_MARGIN,
    RECOVERY_SESOI,
    SCOPE_ACCURACY_EQUIVALENCE_MARGIN,
    SCOPE_ACCURACY_SESOI,
    SCOPE_NLL_EQUIVALENCE_MARGIN,
    SCOPE_NLL_SESOI,
    SCOPE_SAMPLE_SIZE,
)


CONTRACT_ID = "TSI-RESEARCH-A2-ROBUSTNESS-AND-SCOPE-v1"
STATUS = "numeric_draft_ready_for_review_not_frozen_no_confirmatory_seed"
WORLD_COUNT_PER_AXIS_OR_SCOPE_CONDITION = 135
TEST_CASE_COUNT = 1200
TRAIN_NOISE_FOR_WIDTH_AND_SCOPE = 0.08
OOD_NOISE = 0.12
GENERIC_MOVE_BUDGET = 7

WIDTH_ENDPOINTS = tuple(
    endpoint
    for width in WIDTH_POSITION_COUNTS
    for size in WIDTH_SAMPLE_SIZES
    for endpoint in (
        f"width_{width}_n{size}_generic_minus_typed_composition_nll",
        f"width_{width}_n{size}_typed_minus_generic_exact_recovery",
    )
)
NOISE_ENDPOINTS = tuple(
    endpoint
    for probability in NOISE_PROBABILITIES
    for size in NOISE_SAMPLE_SIZES
    for endpoint in (
        f"noise_{probability}_n{size}_generic_minus_typed_composition_nll",
        f"noise_{probability}_n{size}_typed_minus_generic_exact_recovery",
    )
)
SCOPE_ENDPOINTS = (
    "matched_generic_minus_typed_composition_nll",
    "matched_typed_minus_generic_center_accuracy",
    "typed_misspecified_generic_minus_typed_composition_nll",
    "typed_misspecified_typed_minus_generic_center_accuracy",
    "generic_misspecified_generic_minus_typed_composition_nll",
    "generic_misspecified_typed_minus_generic_center_accuracy",
)

POLICIES = (
    "A2 is independent of A1 and cannot alter or repair the A1 estimand.",
    "World is the independent unit; cases are nested observations.",
    "Every arm in a condition receives identical graph, raw state-action rows, and held-out rows.",
    "Width dictionaries retain the exact A1 eleven features first and add only collision-audited deterministic observable features.",
    "The typed model and its three-family search are unchanged throughout the width and noise axes.",
    "Noise masks are coupled and nested within world across the four prespecified probabilities.",
    "Held-out composition cases remain at noise probability 0.12 and are never used for fitting or selection.",
    "The width, noise, and scope endpoints are three separate Bonferroni families, each at familywise alpha 0.05.",
    "Width robustness requires at least one joint SESOI-qualified advantage at every declared width.",
    "Noise robustness requires at least one joint SESOI-qualified advantage on the declared probability-by-n grid; no advantage is required at p=0.80.",
    "The cubic and quadratic misspecification conditions share graph, coefficients, and raw random streams in aligned world pairs.",
    "The scope gate requires matched equivalence and both prespecified directional reversals at n=320.",
    "Misspecification is a scope and falsification audit only and cannot rescue a width or noise efficiency failure.",
    "No confirmatory seed may be generated until review, source freeze, and public recording of the freeze digest are complete.",
    "A failed endpoint or gate is reported without within-cohort repair or threshold revision.",
)


def contract_payload() -> dict[str, object]:
    return {
        "contract_id": CONTRACT_ID,
        "status": STATUS,
        "population": {
            "state": "uniform Z_7^5",
            "graph": "common-target two-source graph supplied to both arms",
            "matched_family_pairs": "nine ordered pairs, 15 worlds each",
            "special_family_pairs": "five ordered pairs containing the special family, 27 worlds each",
            "generator_coefficients": [1, 2, 3],
            "selector_nonzero_coefficients": [1, 2, 3, 4, 5, 6],
            "primitive_action_magnitudes": [1, 2],
        },
        "sample_sizes": {
            "world_count_per_axis_or_scope_condition": WORLD_COUNT_PER_AXIS_OR_SCOPE_CONDITION,
            "candidate_width_train_prefixes": list(WIDTH_SAMPLE_SIZES),
            "training_noise_train_prefixes": list(NOISE_SAMPLE_SIZES),
            "scope_development_grid": list(MISSPECIFICATION_SAMPLE_SIZES),
            "scope_confirmatory_prefix": SCOPE_SAMPLE_SIZE,
            "test_cases_per_world": TEST_CASE_COUNT,
        },
        "candidate_width": {
            "generic_output_feature_positions": list(WIDTH_POSITION_COUNTS),
            "generic_move_budget": GENERIC_MOVE_BUDGET,
            "typed_class_unchanged": True,
            "true_support_in_every_dictionary": True,
        },
        "noise": {
            "width_and_scope_train": TRAIN_NOISE_FOR_WIDTH_AND_SCOPE,
            "noise_axis_train_probabilities": list(NOISE_PROBABILITIES),
            "held_out": OOD_NOISE,
            "unique_mode_boundary": 6 / 7,
            "coupling": "same clean rows, shifts, and uniforms; thresholded nested masks",
        },
        "misspecification": {
            "typed_catalog": [
                "linear_target",
                "quadratic_target",
                "source_target",
            ],
            "alternative_generic_catalog": [
                "linear_target",
                "cubic_target",
                "source_target",
            ],
            "position_count_each_catalog": 55,
            "conditions": [
                "matched",
                "typed_misspecified",
                "generic_misspecified",
            ],
        },
        "multiplicity": {
            "method": "separate_bonferroni_simultaneous_two_sided_normal_intervals",
            "familywise_alpha_per_family": FAMILYWISE_ALPHA,
            "width_endpoints": list(WIDTH_ENDPOINTS),
            "noise_endpoints": list(NOISE_ENDPOINTS),
            "scope_endpoints": list(SCOPE_ENDPOINTS),
        },
        "thresholds": {
            "efficiency_nll_sesoi": NLL_SESOI,
            "efficiency_recovery_sesoi": RECOVERY_SESOI,
            "efficiency_nll_equivalence_margin": NLL_EQUIVALENCE_MARGIN,
            "efficiency_recovery_equivalence_margin": RECOVERY_EQUIVALENCE_MARGIN,
            "scope_nll_sesoi": SCOPE_NLL_SESOI,
            "scope_accuracy_sesoi": SCOPE_ACCURACY_SESOI,
            "scope_nll_equivalence_margin": SCOPE_NLL_EQUIVALENCE_MARGIN,
            "scope_accuracy_equivalence_margin": SCOPE_ACCURACY_EQUIVALENCE_MARGIN,
        },
        "policies": list(POLICIES),
    }


def contract_digest() -> str:
    return sha256(
        json.dumps(contract_payload(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def audit_contract() -> dict[str, object]:
    errors = []
    worlds = WORLD_COUNT_PER_AXIS_OR_SCOPE_CONDITION
    if worlds < 126 or worlds % 9 or worlds % 5:
        errors.append("world count violates the pre-A1 minimum or stratum balance")
    if len(WIDTH_ENDPOINTS) != 36 or len(set(WIDTH_ENDPOINTS)) != 36:
        errors.append("width multiplicity family must contain 36 unique endpoints")
    if len(NOISE_ENDPOINTS) != 48 or len(set(NOISE_ENDPOINTS)) != 48:
        errors.append("noise multiplicity family must contain 48 unique endpoints")
    if len(SCOPE_ENDPOINTS) != 6 or len(set(SCOPE_ENDPOINTS)) != 6:
        errors.append("scope multiplicity family must contain six unique endpoints")
    if max(NOISE_PROBABILITIES) >= 6 / 7:
        errors.append("a noise level reaches or exceeds the unique-mode boundary")
    if SCOPE_SAMPLE_SIZE != 320:
        errors.append("scope confirmatory prefix changed")
    if len(POLICIES) != 15:
        errors.append("the 15 A2 safeguards are incomplete")
    return {
        "contract_id": CONTRACT_ID,
        "status": STATUS,
        "contract_digest": contract_digest(),
        "confirmatory_seed_created": False,
        "errors": errors,
        "passed": not errors,
    }
