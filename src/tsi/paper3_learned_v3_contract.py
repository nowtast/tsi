"""Frozen design contract for P3-5A-v3 held-out mechanism generalization."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json


P3_LEARNED_V3_CONTRACT_ID = "P3-5A-LEARNED-v3"
PARENT_V2_CONTRACT_ID = "P3-5A-LEARNED-v2"
SPLITS = ("train", "routing_selection", "calibration", "downstream_evaluation", "test")
GRAPH_VARIANTS = (
    "bridge_topology_to_relation",
    "context_order_to_metric",
    "independent_relation",
    "wrong_direction_negative_control",
)
OBSERVATION_REGIMES = ("exact_state", "gaussian_0.25", "dropout_0.25", "boundary_dropout_0.50")
MODEL_CONTROLS = (
    "mechanism_conditioned_neural",
    "no_signature_neural",
    "shuffled_signature_neural",
    "dense_matched_neural",
    "exact_structural_diagnostic",
    "wrong_direction_control",
)
MECHANISM_HYPOTHESIS_COUNT = 192
TRAIN_COMBINATION_COUNT = 96
VALIDATION_COMBINATION_COUNT = 48
TEST_COMBINATION_COUNT = 48
OPTIMIZER_SEEDS = (0, 1, 2)
FAMILYWISE_ALPHA = 0.05
PRIMARY_DEGRADATION_MAXIMUM = 0.15
PRIMARY_SESOI = 0.05
PRIMARY_COVERAGE = 0.90
PRIMARY_SELECTIVE_ACCURACY = 1.0


def _all_mechanism_combinations() -> tuple[tuple[tuple[int, int, int, int, int], int, int], ...]:
    return tuple(
        (multipliers, bridge, context)
        for multipliers in product((1, 2), (1, 2), (1, 2), (1, 2, 3), (1, 2))
        for bridge in (1, 3)
        for context in (1, 2)
    )


def mechanism_combinations() -> tuple[tuple[tuple[int, int, int, int, int], int, int], ...]:
    return _all_mechanism_combinations()


def _combination_order() -> tuple[int, ...]:
    seed = f"{P3_LEARNED_V3_CONTRACT_ID}:mechanism-split".encode()
    return tuple(
        sorted(
            range(MECHANISM_HYPOTHESIS_COUNT),
            key=lambda index: sha256(seed + str(index).encode()).digest(),
        )
    )


_ORDER = _combination_order()
TRAIN_COMBINATION_INDICES = frozenset(_ORDER[:TRAIN_COMBINATION_COUNT])
VALIDATION_COMBINATION_INDICES = frozenset(_ORDER[TRAIN_COMBINATION_COUNT:TRAIN_COMBINATION_COUNT + VALIDATION_COMBINATION_COUNT])
TEST_COMBINATION_INDICES = frozenset(_ORDER[TRAIN_COMBINATION_COUNT + VALIDATION_COMBINATION_COUNT:])


def mechanism_split_for_combination(index: int) -> str:
    if index in TRAIN_COMBINATION_INDICES:
        return "train"
    if index in VALIDATION_COMBINATION_INDICES:
        return "validation"
    if index in TEST_COMBINATION_INDICES:
        return "test"
    raise ValueError("unknown mechanism combination index")


@dataclass(frozen=True)
class V3StatisticalPlan:
    independent_unit: str = "world_graph_mechanism_combination"
    nested_replicate: str = "optimizer_seed"
    split_order: tuple[str, ...] = SPLITS
    aggregation: str = "intersection_union_positive_graph_mechanism_strata"
    calibration_partition: str = "calibration"
    boundary_policy: str = "hard_corruption_is_negative_control_unless_recoverability_gate_passes"

    def as_dict(self) -> dict[str, object]:
        return {
            "independent_unit": self.independent_unit,
            "nested_replicate": self.nested_replicate,
            "split_order": list(self.split_order),
            "aggregation": self.aggregation,
            "calibration_partition": self.calibration_partition,
            "boundary_policy": self.boundary_policy,
            "optimizer_seeds": list(OPTIMIZER_SEEDS),
            "familywise_alpha": FAMILYWISE_ALPHA,
        }


FROZEN_V3_STATISTICAL_PLAN = V3StatisticalPlan()


def _payload() -> dict[str, object]:
    return {
        "identifier": P3_LEARNED_V3_CONTRACT_ID,
        "parent": PARENT_V2_CONTRACT_ID,
        "splits": list(SPLITS),
        "graph_variants": list(GRAPH_VARIANTS),
        "observation_regimes": list(OBSERVATION_REGIMES),
        "model_controls": list(MODEL_CONTROLS),
        "mechanism_hypothesis_count": MECHANISM_HYPOTHESIS_COUNT,
        "identification_target": "graph_conditional_active_mechanism_signature",
        "inactive_parameter_policy": "inactive_coefficients_are_not_required_to_be_identified",
        "train_combination_indices": sorted(TRAIN_COMBINATION_INDICES),
        "validation_combination_indices": sorted(VALIDATION_COMBINATION_INDICES),
        "test_combination_indices": sorted(TEST_COMBINATION_INDICES),
        "thresholds": {
            "primary_degradation_maximum": PRIMARY_DEGRADATION_MAXIMUM,
            "primary_sesoi": PRIMARY_SESOI,
            "primary_coverage": PRIMARY_COVERAGE,
            "primary_selective_accuracy": PRIMARY_SELECTIVE_ACCURACY,
        },
        "statistical_plan": FROZEN_V3_STATISTICAL_PLAN.as_dict(),
    }


def learned_v3_contract_digest() -> str:
    return sha256(json.dumps(_payload(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_learned_v3_contract() -> dict[str, object]:
    errors = []
    if len(SPLITS) != 5 or len(set(SPLITS)) != 5:
        errors.append("v3 requires five unique splits")
    if len(TRAIN_COMBINATION_INDICES & TEST_COMBINATION_INDICES):
        errors.append("train and test mechanism combinations overlap")
    if len(VALIDATION_COMBINATION_INDICES & TEST_COMBINATION_INDICES):
        errors.append("validation and test mechanism combinations overlap")
    if len(TRAIN_COMBINATION_INDICES | VALIDATION_COMBINATION_INDICES | TEST_COMBINATION_INDICES) != MECHANISM_HYPOTHESIS_COUNT:
        errors.append("mechanism combinations do not exhaust the public hypothesis class")
    if PRIMARY_COVERAGE < 0.90 or PRIMARY_SELECTIVE_ACCURACY != 1.0:
        errors.append("primary selective rule is not frozen")
    return {
        "identifier": P3_LEARNED_V3_CONTRACT_ID,
        "contract_digest": learned_v3_contract_digest(),
        "split_count": len(SPLITS),
        "mechanism_hypothesis_count": MECHANISM_HYPOTHESIS_COUNT,
        "identification_target": "graph_conditional_active_mechanism_signature",
        "inactive_parameter_policy": "inactive_coefficients_are_not_required_to_be_identified",
        "train_combination_count": len(TRAIN_COMBINATION_INDICES),
        "validation_combination_count": len(VALIDATION_COMBINATION_INDICES),
        "test_combination_count": len(TEST_COMBINATION_INDICES),
        "errors": errors,
        "passed": not errors,
    }
