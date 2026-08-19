"""Training-only mechanism identification for the v2 structural learner.

The identifier treats graph labels, world indices, and mechanism slots as
unavailable. It searches the public finite transition-law hypothesis class using
only primitive training transitions, then applies the identified law to held-out
unseen-action cases.
"""

from __future__ import annotations

from dataclasses import dataclass

from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_learned_v2_generator import (
    GRAPH_VARIANT_MANIFEST,
    V2GraphVariant,
    V2TransitionCase,
    _successor,
    mechanism_parameter_candidates,
)
from .paper3_multiworld import MultiworldStateCode, WorldMechanism


@dataclass(frozen=True)
class ObservableMechanismSignature:
    graph_variant: str
    layer_multipliers: tuple[int, int, int, int, int]
    bridge_coefficient: int
    context_coefficient: int
    matched_training_cases: int
    candidate_count: int

    @property
    def mechanism_tuple(self) -> tuple[tuple[int, int, int, int, int], int, int]:
        return (self.layer_multipliers, self.bridge_coefficient, self.context_coefficient)

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_variant": self.graph_variant,
            "layer_multipliers": list(self.layer_multipliers),
            "bridge_coefficient": self.bridge_coefficient,
            "context_coefficient": self.context_coefficient,
            "matched_training_cases": self.matched_training_cases,
            "candidate_count": self.candidate_count,
        }


def _dummy_mechanism(
    multipliers: tuple[int, int, int, int, int],
    bridge: int,
    context: int,
) -> WorldMechanism:
    return WorldMechanism(
        family=WorldFamily.CONTEXT_DEPENDENT,
        cohort=BenchmarkSplit.DEVELOPMENT,
        world_index=0,
        layer_multipliers=multipliers,
        bridge_coefficient=bridge,
        context_coefficient=context,
        root_commitment="observable-training-only",
        mechanism_digest="observable-training-only",
    )


def _candidate_matches(
    cases: tuple[V2TransitionCase, ...],
    graph: V2GraphVariant,
    mechanism_tuple: tuple[tuple[int, int, int, int, int], int, int],
) -> bool:
    mechanism = _dummy_mechanism(*mechanism_tuple)
    return all(
        _successor(case.source_code, case.action, mechanism, graph) == case.target_code
        for case in cases
    )


def identify_observable_mechanism(
    training_cases: tuple[V2TransitionCase, ...] | list[V2TransitionCase],
) -> ObservableMechanismSignature:
    """Identify graph and mechanism using only primitive training transitions."""
    cases = tuple(training_cases)
    if not cases:
        raise ValueError("mechanism identification requires training cases")
    if any(case.intervention for case in cases):
        raise ValueError("mechanism identification cannot use intervention cases")
    candidates = []
    for graph in GRAPH_VARIANT_MANIFEST:
        for mechanism_tuple in mechanism_parameter_candidates():
            if _candidate_matches(cases, graph, mechanism_tuple):
                candidates.append((graph, mechanism_tuple))
    if not candidates:
        raise ValueError("training transitions do not identify a public mechanism")
    graph, (multipliers, bridge, context) = candidates[0]
    return ObservableMechanismSignature(
        graph_variant=graph.identifier,
        layer_multipliers=multipliers,
        bridge_coefficient=bridge,
        context_coefficient=context,
        matched_training_cases=len(cases),
        candidate_count=len(candidates),
    )


def predict_target_code(
    source_code: MultiworldStateCode,
    action,
    signature: ObservableMechanismSignature,
) -> MultiworldStateCode:
    graph = next(
        graph for graph in GRAPH_VARIANT_MANIFEST
        if graph.identifier == signature.graph_variant
    )
    return _successor(
        source_code,
        action,
        _dummy_mechanism(signature.layer_multipliers, signature.bridge_coefficient, signature.context_coefficient),
        graph,
    )


def evaluate_identified_signature(
    training_cases: tuple[V2TransitionCase, ...] | list[V2TransitionCase],
    evaluation_cases: tuple[V2TransitionCase, ...] | list[V2TransitionCase],
) -> dict[str, object]:
    signature = identify_observable_mechanism(training_cases)
    correct = sum(
        predict_target_code(case.source_code, case.action, signature) == case.target_code
        for case in evaluation_cases
    )
    total = len(evaluation_cases)
    return {
        "signature": signature.as_dict(),
        "evaluation_count": total,
        "exact_accuracy": float(correct / total) if total else float("nan"),
        "all_exact": bool(total and correct == total),
    }


def run_balanced_mechanism_conditioned_validation(
    *,
    worlds: int = 8,
    world_start: int = 40,
    mechanism_slots: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[dict[str, object], ...]:
    """Evaluate identification on a fresh factorial panel without world labels."""
    if worlds <= 0 or world_start < 0:
        raise ValueError("worlds must be positive and world_start nonnegative")
    if not mechanism_slots or len(set(mechanism_slots)) != len(mechanism_slots):
        raise ValueError("mechanism_slots must be nonempty and unique")
    from .paper3_learned_v2_generator import build_balanced_v2_world_dataset

    results: list[dict[str, object]] = []
    for world_index in range(world_start, world_start + worlds):
        for mechanism_slot in mechanism_slots:
            dataset = build_balanced_v2_world_dataset(world_index, mechanism_slot)
            result = evaluate_identified_signature(
                dataset.partitions["train"],
                dataset.partitions["test"],
            )
            results.append({
                "world_index": world_index,
                "mechanism_slot": mechanism_slot,
                "graph_variant": dataset.graph.identifier,
                **result,
            })
    return tuple(results)


def _decode_pixel_source_code(case) -> MultiworldStateCode:
    """Decode the raster coordinate, without consulting ``case.source_code``."""
    image = __import__("numpy").asarray(case.image, dtype=float)
    values = []
    for start, width, maximum in zip(
        (0, 3, 6, 9, 12), (3, 3, 3, 3, 4), (2, 2, 2, 3, 2), strict=True
    ):
        region = image[:, start:start + width]
        if not __import__("numpy").any(region > 0.0):
            row = 0
        else:
            row = int(__import__("numpy").unravel_index(__import__("numpy").argmax(region), region.shape)[0])
        values.append(int(round(row * maximum / 15.0)))
    return MultiworldStateCode(*values)


def identify_observable_mechanism_from_pixel(training_cases) -> ObservableMechanismSignature:
    """Identify a mechanism from raster inputs and observed transition outcomes."""
    from .paper3_learned_v2_generator import V2TransitionCase

    proxy_cases = tuple(
        V2TransitionCase(
            partition=case.partition,
            graph_variant="wrong_direction_negative_control",
            source_code=_decode_pixel_source_code(case),
            action=case.action,
            target_code=case.target_code,
            intervention=case.intervention,
        )
        for case in training_cases
    )
    return identify_observable_mechanism(proxy_cases)


def evaluate_pixel_identified_signature(training_cases, evaluation_cases) -> dict[str, object]:
    signature = identify_observable_mechanism_from_pixel(training_cases)
    correct = 0
    for case in evaluation_cases:
        source = _decode_pixel_source_code(case)
        if predict_target_code(source, case.action, signature) == case.target_code:
            correct += 1
    total = len(evaluation_cases)
    return {
        "signature": signature.as_dict(),
        "evaluation_count": total,
        "exact_accuracy": float(correct / total) if total else float("nan"),
        "all_exact": bool(total and correct == total),
    }


def run_balanced_pixel_mechanism_conditioned_validation(
    *,
    worlds: int = 8,
    world_start: int = 40,
    mechanism_slots: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[dict[str, object], ...]:
    from .paper3_learned_v2_generator import build_balanced_v2_world_dataset
    from .paper3_learned_v2_observation import build_observed_partitions

    results: list[dict[str, object]] = []
    for world_index in range(world_start, world_start + worlds):
        for mechanism_slot in mechanism_slots:
            dataset = build_balanced_v2_world_dataset(world_index, mechanism_slot)
            observed = build_observed_partitions(
                dict(dataset.partitions),
                entity_count=3,
                regime="pixel_object_observation",
                seed=10_000 + world_index,
            )
            result = evaluate_pixel_identified_signature(
                observed["train"], observed["test"]
            )
            results.append({
                "world_index": world_index,
                "mechanism_slot": mechanism_slot,
                "graph_variant": dataset.graph.identifier,
                **result,
            })
    return tuple(results)


@dataclass(frozen=True)
class MechanismConditionedStructuredHead:
    """A non-oracle structured head conditioned on a training-only signature."""

    signature: ObservableMechanismSignature

    @classmethod
    def fit(cls, training_cases) -> "MechanismConditionedStructuredHead":
        if training_cases and hasattr(training_cases[0], "image"):
            signature = identify_observable_mechanism_from_pixel(training_cases)
        else:
            signature = identify_observable_mechanism(training_cases)
        return cls(signature)

    def predict_target(self, case) -> MultiworldStateCode:
        source = (
            _decode_pixel_source_code(case)
            if hasattr(case, "image")
            else case.source_code
        )
        return predict_target_code(source, case.action, self.signature)

    def predict_delta(self, case) -> tuple[int, ...]:
        source = (
            _decode_pixel_source_code(case)
            if hasattr(case, "image")
            else case.source_code
        )
        target = self.predict_target(case)
        return tuple(
            (predicted - observed) % (4 if index == 3 else 3)
            for index, (predicted, observed) in enumerate(
                zip(target.as_tuple(), source.as_tuple(), strict=True)
            )
        )

    def mean_logloss(self, cases, *, smoothing: float = 1.0e-6) -> float:
        if not cases:
            raise ValueError("structured head evaluation requires cases")
        if not 0.0 < smoothing < 1.0:
            raise ValueError("smoothing must lie in (0, 1)")
        total = 0.0
        count = 0
        for case in cases:
            truth_source = case.source_code
            predicted = self.predict_delta(case)
            observed = tuple(
                (target - value) % (4 if index == 3 else 3)
                for index, (value, target) in enumerate(
                    zip(truth_source.as_tuple(), case.target_code.as_tuple(), strict=True)
                )
            )
            for index, (guess, truth) in enumerate(zip(predicted, observed, strict=True)):
                cardinality = 4 if index == 3 else 3
                probability = 1.0 - smoothing if guess == truth else smoothing / (cardinality - 1)
                total -= __import__("math").log(probability)
                count += 1
        return total / count


def evaluate_conditioned_pixel_head(training_cases, clean_cases, corrupted_cases) -> dict[str, object]:
    head = MechanismConditionedStructuredHead.fit(training_cases)
    return {
        "signature": head.signature.as_dict(),
        "clean_logloss": head.mean_logloss(clean_cases),
        "corrupted_logloss": head.mean_logloss(corrupted_cases),
        "logloss_degradation": head.mean_logloss(corrupted_cases) - head.mean_logloss(clean_cases),
    }


@dataclass(frozen=True)
class PixelSourcePrototype:
    image: tuple[tuple[float, ...], ...]
    source_code: MultiworldStateCode


def fit_pixel_source_prototypes(
    training_cases,
    *,
    corruption_profile: str | None = None,
    augmentation_count: int = 3,
) -> tuple[PixelSourcePrototype, ...]:
    """Build clean and corruption-matched prototypes from training images only."""
    from .paper3_learned_v2_observation import corrupt_pixel_case

    profiles = {
        "gaussian_0.25": {"gaussian_noise": 0.25},
        "gaussian_0.50": {"gaussian_noise": 0.50},
        "dropout_0.25": {"dropout_probability": 0.25},
        "dropout_0.50_quantized": {"dropout_probability": 0.50, "quantization_levels": 4},
        "near_blank_0.99": {"dropout_probability": 0.99, "quantization_levels": 2},
    }
    if corruption_profile not in (None, *profiles):
        raise ValueError("unknown corruption profile")
    if type(augmentation_count) is not int or augmentation_count < 0:
        raise ValueError("augmentation_count must be a nonnegative integer")
    prototypes: list[PixelSourcePrototype] = []
    seen: set[tuple[tuple[float, ...], ...]] = set()
    unique_cases = {}
    for case in training_cases:
        if not hasattr(case, "image"):
            raise ValueError("pixel source prototypes require pixel cases")
        key = tuple(tuple(float(value) for value in row) for row in case.image)
        unique_cases.setdefault(key, case)
    for key, case in unique_cases.items():
        source = _decode_pixel_source_code(case)
        candidates = [case]
        if corruption_profile is not None:
            candidates.extend(
                corrupt_pixel_case(
                    case, seed=30_000 + index, **profiles[corruption_profile]
                )
                for index in range(augmentation_count)
            )
        for candidate in candidates:
            image_key = tuple(tuple(float(value) for value in row) for row in candidate.image)
            if image_key not in seen:
                seen.add(image_key)
                prototypes.append(PixelSourcePrototype(image_key, source))
    if not prototypes:
        raise ValueError("pixel source prototypes require training cases")
    return tuple(prototypes)


def _dropout_log_likelihood(image, prototypes, dropout_probability: float):
    import numpy as np

    clean = prototypes > 0.5
    observed = image > 0.5
    epsilon = 1.0e-9
    log_keep = np.log(max(1.0 - dropout_probability, epsilon))
    log_drop = np.log(max(dropout_probability, epsilon))
    # A clean one is retained or dropped; a clean zero should remain zero.
    scores = np.where(
        clean & observed,
        log_keep,
        np.where(clean & ~observed, log_drop, np.where(~clean & ~observed, 0.0, np.log(epsilon))),
    )
    return np.sum(scores, axis=(1, 2))


@dataclass(frozen=True)
class DenoisedMechanismConditionedStructuredHead:
    """Mechanism-conditioned head with training-image nearest-prototype denoising."""

    signature: ObservableMechanismSignature
    prototypes: tuple[PixelSourcePrototype, ...]
    corruption_profile: str | None = None

    @classmethod
    def fit(
        cls,
        training_cases,
        *,
        prototype_cases=None,
        corruption_profile: str | None = None,
    ) -> "DenoisedMechanismConditionedStructuredHead":
        return cls(
            signature=identify_observable_mechanism_from_pixel(training_cases),
            prototypes=fit_pixel_source_prototypes(
                training_cases if prototype_cases is None else prototype_cases,
                corruption_profile=corruption_profile,
            ),
            corruption_profile=corruption_profile,
        )

    def denoised_source(self, case) -> MultiworldStateCode:
        import numpy as np

        image = np.asarray(case.image, dtype=float)
        distances = [
            float(np.linalg.norm(image - np.asarray(prototype.image, dtype=float)))
            for prototype in self.prototypes
        ]
        return self.prototypes[int(np.argmin(distances))].source_code

    def _prototype_weights(self, case, temperature: float) -> tuple[tuple[PixelSourcePrototype, float], ...]:
        import numpy as np

        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        image = np.asarray(case.image, dtype=float)
        prototype_images = np.asarray([prototype.image for prototype in self.prototypes], dtype=float)
        profile = self.corruption_profile
        if profile is not None and profile.startswith("gaussian_"):
            sigma = float(profile.rsplit("_", 1)[1])
            scores = -np.sum((prototype_images - image) ** 2, axis=(1, 2)) / (2.0 * sigma * sigma)
        elif profile == "dropout_0.25":
            scores = _dropout_log_likelihood(image, prototype_images, 0.25)
        elif profile == "dropout_0.50_quantized":
            scores = _dropout_log_likelihood(image, prototype_images, 0.50)
        elif profile == "near_blank_0.99":
            scores = _dropout_log_likelihood(image, prototype_images, 0.99)
        else:
            distances = np.asarray([
                float(np.linalg.norm(image - prototype_image))
                for prototype_image in prototype_images
            ])
            scores = -distances
        weights = np.exp((scores - np.max(scores)) / temperature)
        weights /= np.sum(weights)
        return tuple(zip(self.prototypes, (float(value) for value in weights), strict=True))

    def predict_delta(self, case) -> tuple[int, ...]:
        source = self.denoised_source(case)
        target = predict_target_code(source, case.action, self.signature)
        return tuple(
            (target_value - source_value) % (4 if index == 3 else 3)
            for index, (source_value, target_value) in enumerate(
                zip(source.as_tuple(), target.as_tuple(), strict=True)
            )
        )

    def delta_probabilities(self, case, *, temperature: float = 0.5) -> tuple[tuple[float, ...], ...]:
        cardinalities = (3, 3, 3, 4, 3)
        probabilities = [
            [0.0 for _ in range(cardinality)] for cardinality in cardinalities
        ]
        for prototype, weight in self._prototype_weights(case, temperature):
            target = predict_target_code(prototype.source_code, case.action, self.signature)
            delta = tuple(
                (target_value - source_value) % (4 if index == 3 else 3)
                for index, (source_value, target_value) in enumerate(
                    zip(prototype.source_code.as_tuple(), target.as_tuple(), strict=True)
                )
            )
            for index, value in enumerate(delta):
                probabilities[index][value] += weight
        return tuple(tuple(row) for row in probabilities)

    def mean_logloss(self, cases, *, smoothing: float = 1.0e-6) -> float:
        if not cases:
            raise ValueError("denoised head evaluation requires cases")
        total = 0.0
        count = 0
        for case in cases:
            truth_source = case.source_code
            observed = tuple(
                (target - value) % (4 if index == 3 else 3)
                for index, (value, target) in enumerate(
                    zip(truth_source.as_tuple(), case.target_code.as_tuple(), strict=True)
                )
            )
            probabilities = self.delta_probabilities(case)
            for index, truth in enumerate(observed):
                probability = max(probabilities[index][truth], smoothing)
                total -= __import__("math").log(probability)
                count += 1
        return total / count


def evaluate_denoised_pixel_head(training_cases, clean_cases, corrupted_cases) -> dict[str, object]:
    head = DenoisedMechanismConditionedStructuredHead.fit(training_cases)
    clean = head.mean_logloss(clean_cases)
    corrupted = head.mean_logloss(corrupted_cases)
    return {
        "signature": head.signature.as_dict(),
        "prototype_count": len(head.prototypes),
        "clean_logloss": clean,
        "corrupted_logloss": corrupted,
        "logloss_degradation": corrupted - clean,
    }


def evaluate_selective_prediction(
    head: DenoisedMechanismConditionedStructuredHead,
    cases,
    *,
    confidence_threshold: float = 0.8,
) -> dict[str, float | int | bool]:
    """Evaluate abstention using only prediction confidence for acceptance."""
    if not cases:
        raise ValueError("selective evaluation requires cases")
    if not 0.0 < confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must lie in (0, 1]")
    accepted = 0
    correct = 0
    logloss = 0.0
    import math
    for case in cases:
        probabilities = head.delta_probabilities(case)
        confidence = min(max(row) for row in probabilities)
        truth = tuple(
            (target - source) % (4 if index == 3 else 3)
            for index, (source, target) in enumerate(
                zip(case.source_code.as_tuple(), case.target_code.as_tuple(), strict=True)
            )
        )
        if confidence < confidence_threshold:
            continue
        accepted += 1
        correct += all(
            int(max(range(len(row)), key=row.__getitem__)) == truth[index]
            for index, row in enumerate(probabilities)
        )
        logloss -= sum(math.log(max(row[value], 1.0e-12)) for row, value in zip(probabilities, truth, strict=True))
    return {
        "total": len(cases),
        "accepted": accepted,
        "coverage": accepted / len(cases),
        "selective_exact_accuracy": correct / accepted if accepted else float("nan"),
        "selective_mean_logloss": logloss / (accepted * 5) if accepted else float("nan"),
        "acceptance_gate_passed": bool(
            accepted / len(cases) >= 0.90 and accepted and correct == accepted
        ),
    }
