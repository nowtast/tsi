"""Identifiability limits and selective acceptance for the pixel carrier."""

from __future__ import annotations

from collections import defaultdict

from .paper3_learned_v2_generator import V2TransitionCase
from .paper3_learned_v2_observation import pixel_case_from_v2_case
from .paper3_multiworld import (
    PRIMITIVE_ACTIONS,
    all_multiworld_state_codes,
)
from .paper3_learned_v2_mechanism import (
    DenoisedMechanismConditionedStructuredHead,
    evaluate_selective_prediction,
)


def pixel_carrier_collision_audit() -> dict[str, object]:
    """Compute clean-carrier equivalence classes over the full finite state carrier."""
    representative = {}
    classes = defaultdict(set)
    for source in all_multiworld_state_codes():
        case = V2TransitionCase(
            partition="train",
            graph_variant="bridge_topology_to_relation",
            source_code=source,
            action=PRIMITIVE_ACTIONS[0],
            target_code=source,
            intervention=False,
        )
        pixel = pixel_case_from_v2_case(case, entity_count=3)
        key = pixel.image
        classes[key].add(source.as_tuple())
        representative.setdefault(key, pixel)
    sizes = [len(values) for values in classes.values()]
    ambiguous = [size for size in sizes if size > 1]
    return {
        "state_count": len(all_multiworld_state_codes()),
        "unique_clean_images": len(classes),
        "ambiguous_image_classes": len(ambiguous),
        "ambiguous_state_count": sum(ambiguous),
        "maximum_class_size": max(sizes),
        "collision_free_fraction": sum(size == 1 for size in sizes) / len(sizes),
        "irreducible_uniform_source_accuracy_bound": sum(1.0 / size for size in sizes) / len(sizes),
    }


def build_selective_prediction_report(training_cases, evaluation_cases) -> dict[str, object]:
    head = DenoisedMechanismConditionedStructuredHead.fit(training_cases)
    thresholds = []
    for threshold in (0.50, 0.70, 0.80, 0.90, 0.95):
        thresholds.append({
            "confidence_threshold": threshold,
            **evaluate_selective_prediction(
                head, evaluation_cases, confidence_threshold=threshold
            ),
        })
    return {
        "head": "soft_prototype_mechanism_conditioned",
        "carrier_audit": pixel_carrier_collision_audit(),
        "thresholds": thresholds,
        "rule": "promote only if coverage >= 0.90 and selective exact accuracy == 1.0",
        "promotion_passed": any(
            row["coverage"] >= 0.90
            and row["selective_exact_accuracy"] == 1.0
            and row["acceptance_gate_passed"]
            for row in thresholds
        ),
    }
