"""Finite audits for the P3-3A multi-world generator contract."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from .coherent import bridge_defects
from .paper3_independence_contract import BenchmarkSplit, WorldFamily
from .paper3_multiworld import (
    DEVELOPMENT_WORLDS_PER_FAMILY,
    LAYER_ORDER,
    PRIMITIVE_ACTIONS,
    VALIDATION_WORLDS_PER_FAMILY,
    MultiworldStateCode,
    build_multiworld_state,
    build_world_dataset,
    build_world_mechanism,
    development_validation_world_manifest,
    multiworld_generator_digest,
    successor_code,
)


REQUIRED_OOD_SLICES = (
    "unseen_recombination",
    "unseen_structural_mode",
    "unseen_mechanism_parameter",
    "unseen_action_composition",
    "bridge_consistent_shift",
    "bridge_violating_control",
)


@dataclass(frozen=True)
class MultiworldGeneratorAudit:
    generator_digest: str
    state_count: int
    manifest_world_count: int
    cohort_family_counts: Mapping[str, int]
    active_signature_counts: Mapping[str, int]
    representative_partition_counts: Mapping[str, Mapping[str, int]]
    representative_ood_counts: Mapping[str, Mapping[str, int]]
    independent_relation_witnesses: int
    dependency_witnesses: Mapping[str, bool]
    sealed_test_worlds_materialized: int
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict[str, object]:
        return {
            "generator_digest": self.generator_digest,
            "state_count": self.state_count,
            "manifest_world_count": self.manifest_world_count,
            "cohort_family_counts": dict(self.cohort_family_counts),
            "active_signature_counts": dict(self.active_signature_counts),
            "representative_partition_counts": {
                family: dict(counts)
                for family, counts in self.representative_partition_counts.items()
            },
            "representative_ood_counts": {
                family: dict(counts)
                for family, counts in self.representative_ood_counts.items()
            },
            "independent_relation_witnesses": (self.independent_relation_witnesses),
            "dependency_witnesses": dict(self.dependency_witnesses),
            "sealed_test_worlds_materialized": (self.sealed_test_worlds_materialized),
            "errors": list(self.errors),
            "passed": self.passed,
        }


def _coordinate_value(code: MultiworldStateCode, layer: str) -> int:
    if layer == "label":
        return code.label_phase
    if layer == "topology":
        return code.topology_mode
    if layer == "metric":
        return code.metric_mode
    if layer == "relation":
        return code.influence_mode
    if layer == "order":
        return code.order_mode
    raise ValueError(f"unknown layer: {layer}")


def _audit_train_marginal_support(
    errors: list[str],
    family: WorldFamily,
    train_cases,
) -> None:
    for action in PRIMITIVE_ACTIONS:
        action_cases = tuple(case for case in train_cases if case.action == action)
        for layer in LAYER_ORDER:
            expected = set(range(4 if layer == "relation" else 3))
            observed = {
                _coordinate_value(case.source_code, layer) for case in action_cases
            }
            if observed != expected:
                errors.append(
                    f"{family.value} train support misses "
                    f"{layer}/{action.name}: {sorted(expected - observed)}"
                )


def _audit_partition_independence(
    errors: list[str],
    family: WorldFamily,
    dataset,
) -> None:
    train_keys = {case.input_key for case in dataset.partitions["train"]}
    validation_keys = {case.input_key for case in dataset.partitions["validation"]}
    ood_keys = {case.input_key for case in dataset.partitions["ood"]}
    if train_keys.intersection(validation_keys):
        errors.append(f"{family.value} train and validation inputs overlap")
    if train_keys.intersection(ood_keys):
        errors.append(f"{family.value} train and OOD inputs overlap")

    observed_slices = set(dataset.ood_by_slice)
    missing_slices = set(REQUIRED_OOD_SLICES).difference(observed_slices)
    if missing_slices:
        errors.append(f"{family.value} misses OOD slices: {sorted(missing_slices)}")

    consistent = dataset.ood_by_slice.get("bridge_consistent_shift", ())
    violating = dataset.ood_by_slice.get("bridge_violating_control", ())
    consistent_targets = {(case.input_key, case.target_code) for case in consistent}
    violating_targets = {(case.input_key, case.target_code) for case in violating}
    if len(consistent) != len(violating):
        errors.append(f"{family.value} bridge controls are not paired")
    if consistent_targets.intersection(violating_targets):
        errors.append(f"{family.value} bridge controls have identical targets")
    if any(not case.follows_declared_mechanism for case in consistent):
        errors.append(f"{family.value} consistent controls are mislabeled")
    if any(case.follows_declared_mechanism for case in violating):
        errors.append(f"{family.value} violating controls are mislabeled")


def _independent_relation_witness_count() -> int:
    same_topology_left = build_multiworld_state(MultiworldStateCode(0, 0, 0, 0, 0))
    same_topology_right = build_multiworld_state(MultiworldStateCode(0, 0, 0, 1, 0))
    same_relation_left = build_multiworld_state(MultiworldStateCode(0, 0, 0, 2, 0))
    same_relation_right = build_multiworld_state(MultiworldStateCode(0, 1, 0, 2, 0))
    witnesses = 0
    if (
        same_topology_left.core.simplices == same_topology_right.core.simplices
        and same_topology_left.core.relational.generators["influences"]
        != same_topology_right.core.relational.generators["influences"]
    ):
        witnesses += 1
    if (
        same_relation_left.core.relational.generators["influences"]
        == same_relation_right.core.relational.generators["influences"]
        and same_relation_left.core.simplices != same_relation_right.core.simplices
    ):
        witnesses += 1
    return witnesses


def _dependency_witnesses() -> Mapping[str, bool]:
    source_topology_zero = MultiworldStateCode(0, 0, 0, 0, 0)
    source_topology_two = MultiworldStateCode(0, 2, 0, 0, 0)
    source_order_zero = MultiworldStateCode(0, 0, 0, 0, 0)
    source_order_one = MultiworldStateCode(0, 0, 0, 0, 1)
    topology_action = next(
        action for action in PRIMITIVE_ACTIONS if action.name == "topology_step"
    )
    metric_action = next(
        action for action in PRIMITIVE_ACTIONS if action.name == "metric_step"
    )

    separable = build_world_mechanism(
        WorldFamily.SEPARABLE,
        BenchmarkSplit.DEVELOPMENT,
        0,
    )
    bridge = build_world_mechanism(
        WorldFamily.BRIDGE_COUPLED,
        BenchmarkSplit.DEVELOPMENT,
        0,
    )
    context = build_world_mechanism(
        WorldFamily.CONTEXT_DEPENDENT,
        BenchmarkSplit.DEVELOPMENT,
        0,
    )

    separable_left = successor_code(
        source_topology_zero,
        topology_action,
        separable,
    )
    separable_right = successor_code(
        source_topology_two,
        topology_action,
        separable,
    )
    bridge_left = successor_code(
        source_topology_zero,
        topology_action,
        bridge,
    )
    bridge_right = successor_code(
        source_topology_two,
        topology_action,
        bridge,
    )
    context_left = successor_code(
        source_order_zero,
        metric_action,
        context,
    )
    context_right = successor_code(
        source_order_one,
        metric_action,
        context,
    )
    return MappingProxyType(
        {
            "separable_excludes_topology_to_relation": (
                separable_left.influence_mode == separable_right.influence_mode
            ),
            "bridge_coupled_requires_topology_to_relation": (
                bridge_left.influence_mode != bridge_right.influence_mode
            ),
            "context_requires_order_to_metric": (
                context_left.metric_mode != context_right.metric_mode
            ),
        }
    )


def audit_multiworld_generator() -> MultiworldGeneratorAudit:
    """Exhaustively audit public worlds without constructing sealed-test data."""

    errors: list[str] = []
    manifest = development_validation_world_manifest()
    identifiers = tuple(mechanism.identifier for mechanism in manifest)
    mechanism_digests = tuple(mechanism.mechanism_digest for mechanism in manifest)
    if len(identifiers) != len(set(identifiers)):
        errors.append("world identifiers are not unique")
    if len(mechanism_digests) != len(set(mechanism_digests)):
        errors.append("world mechanism digests are not unique")
    active_signature_counts: dict[str, int] = {}
    for family in WorldFamily:
        family_mechanisms = tuple(
            mechanism for mechanism in manifest if mechanism.family is family
        )
        active_signatures = tuple(
            mechanism.active_parameter_signature for mechanism in family_mechanisms
        )
        active_signature_counts[family.value] = len(set(active_signatures))
        if len(active_signatures) != len(set(active_signatures)):
            errors.append(f"{family.value} active mechanism signatures are not unique")
    if any(mechanism.cohort is BenchmarkSplit.SEALED_TEST for mechanism in manifest):
        errors.append("sealed-test worlds were materialized")

    cohort_family_counts: dict[str, int] = {}
    for cohort in (BenchmarkSplit.DEVELOPMENT, BenchmarkSplit.VALIDATION):
        expected = (
            DEVELOPMENT_WORLDS_PER_FAMILY
            if cohort is BenchmarkSplit.DEVELOPMENT
            else VALIDATION_WORLDS_PER_FAMILY
        )
        for family in WorldFamily:
            key = f"{cohort.value}:{family.value}"
            count = sum(
                mechanism.cohort is cohort and mechanism.family is family
                for mechanism in manifest
            )
            cohort_family_counts[key] = count
            if count != expected:
                errors.append(f"{key} has {count} worlds instead of {expected}")

    state_codes = tuple(
        MultiworldStateCode(label, topology, metric, influence, order)
        for label in range(3)
        for topology in range(3)
        for metric in range(3)
        for influence in range(4)
        for order in range(3)
    )
    for code in state_codes:
        state = build_multiworld_state(code)
        defects = bridge_defects(state.core, state.order, state.signature)
        if any(value != 0.0 for value in defects.values()):
            errors.append(f"state {code.as_tuple()} violates a static bridge")
            break

    representative_partition_counts: dict[str, Mapping[str, int]] = {}
    representative_ood_counts: dict[str, Mapping[str, int]] = {}
    for family in WorldFamily:
        mechanism = build_world_mechanism(
            family,
            BenchmarkSplit.DEVELOPMENT,
            0,
        )
        dataset = build_world_dataset(mechanism)
        _audit_train_marginal_support(
            errors,
            family,
            dataset.partitions["train"],
        )
        _audit_partition_independence(errors, family, dataset)
        representative_partition_counts[family.value] = MappingProxyType(
            {partition: len(cases) for partition, cases in dataset.partitions.items()}
        )
        representative_ood_counts[family.value] = MappingProxyType(
            {
                slice_name: len(cases)
                for slice_name, cases in dataset.ood_by_slice.items()
            }
        )

    witnesses = _independent_relation_witness_count()
    if witnesses != 2:
        errors.append("independent relation nonredundancy witnesses are missing")
    dependency_witnesses = _dependency_witnesses()
    for name, passed in dependency_witnesses.items():
        if not passed:
            errors.append(f"dependency witness failed: {name}")

    return MultiworldGeneratorAudit(
        generator_digest=multiworld_generator_digest(),
        state_count=len(state_codes),
        manifest_world_count=len(manifest),
        cohort_family_counts=MappingProxyType(cohort_family_counts),
        active_signature_counts=MappingProxyType(active_signature_counts),
        representative_partition_counts=MappingProxyType(
            representative_partition_counts
        ),
        representative_ood_counts=MappingProxyType(representative_ood_counts),
        independent_relation_witnesses=witnesses,
        dependency_witnesses=dependency_witnesses,
        sealed_test_worlds_materialized=0,
        errors=tuple(errors),
    )
