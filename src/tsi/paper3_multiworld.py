"""Multi-world exact-state generator for the TSI P3-3 preregistration.

The generator keeps the fixed three-entity P3-I0 carrier while removing two
P3-2R shortcuts:

* ``influences`` is independent of the topology-induced ``adjacent`` relation;
* transition mechanisms vary across independent worlds and include declared
  separable, bridge-coupled, and context-dependent dependency families.

Only development and validation world roots are materialized here. Sealed-test
world generation requires a separately committed seed and is deliberately not
called by this module's preregistration audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from types import MappingProxyType
from typing import Hashable, Mapping

from .coherent import BridgeSpec, CoherenceSignature, CoherentStructuralState
from .dynamical import (
    IntegratedStructuralState,
    PartialBijection,
    TrackedTransition,
)
from .order_topology import FinitePreorder
from .paper3_evidence_contract import EVIDENCE_CONTRACT_ID
from .paper3_independence_contract import (
    BenchmarkSplit,
    P3_INDEPENDENCE_CONTRACT_ID,
    WorldFamily,
)
from .relational import (
    ArrowSpec,
    FiniteRelation,
    FiniteRelationAssignment,
    FiniteRelationalSchema,
)


P3_MULTIWORLD_GENERATOR_ID = "P3-3A-MULTIWORLD-v1"
ENTITY_TYPE = "entity"
ENTITY_IDS = (0, 1, 2)
BASE_LABELS = ("red", "red", "blue")
LAYER_ORDER = ("label", "topology", "metric", "relation", "order")
PARTITION_ORDER = ("train", "validation", "ood")
DEVELOPMENT_WORLDS_PER_FAMILY = 24
VALIDATION_WORLDS_PER_FAMILY = 12
DEVELOPMENT_ROOT = "tsi:p3-3a:development:2026-07-29:v1"
VALIDATION_ROOT = "tsi:p3-3a:validation:2026-07-29:v1"


P3_MULTIWORLD_SCHEMA = FiniteRelationalSchema(
    objects=(ENTITY_TYPE,),
    arrows=(
        ArrowSpec("adjacent", ENTITY_TYPE, ENTITY_TYPE),
        ArrowSpec("influences", ENTITY_TYPE, ENTITY_TYPE),
    ),
)


P3_MULTIWORLD_SIGNATURE = CoherenceSignature(
    metric_scale=5.0,
    label_weight=0.2,
    simplicial_weight=0.2,
    metric_weight=0.2,
    relation_weight=0.2,
    order_weight=0.2,
    bridges=(BridgeSpec("adjacent", "adjacency"),),
)


@dataclass(frozen=True, order=True)
class MultiworldStateCode:
    """Five independent finite coordinates for one coherent structural state."""

    label_phase: int
    topology_mode: int
    metric_mode: int
    influence_mode: int
    order_mode: int

    def __post_init__(self) -> None:
        for name in ("label_phase", "topology_mode", "metric_mode", "order_mode"):
            value = getattr(self, name)
            if type(value) is not int or value not in (0, 1, 2):
                raise ValueError(f"{name} must be one of 0, 1, or 2")
        if type(self.influence_mode) is not int or self.influence_mode not in range(4):
            raise ValueError("influence_mode must be one of 0, 1, 2, or 3")

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.label_phase,
            self.topology_mode,
            self.metric_mode,
            self.influence_mode,
            self.order_mode,
        )


def all_multiworld_state_codes() -> tuple[MultiworldStateCode, ...]:
    """Enumerate the complete 324-state finite carrier-independent family."""

    return tuple(
        MultiworldStateCode(label, topology, metric, influence, order)
        for label, topology, metric, influence, order in product(
            range(3),
            range(3),
            range(3),
            range(4),
            range(3),
        )
    )


def _topology_edges(mode: int) -> tuple[tuple[int, int], ...]:
    if mode == 0:
        return ((0, 1), (1, 2))
    if mode == 1:
        return ((0, 1), (0, 2), (1, 2))
    return ((0, 2),)


def _metric_coordinates(mode: int) -> tuple[float, float, float]:
    if mode == 0:
        return (0.0, 1.0, 3.0)
    if mode == 1:
        return (0.0, 2.0, 3.0)
    return (0.0, 1.0, 5.0)


def _influence_pairs(mode: int) -> frozenset[tuple[int, int]]:
    if mode == 0:
        return frozenset(((0, 1), (1, 2), (2, 0)))
    if mode == 1:
        return frozenset(((1, 0), (2, 1), (0, 2)))
    if mode == 2:
        return frozenset(((0, 1), (0, 2)))
    return frozenset(((0, 1), (1, 2)))


def _order_relation(
    tagged_entities: tuple[tuple[Hashable, Hashable], ...],
    mode: int,
) -> frozenset[tuple[tuple[Hashable, Hashable], tuple[Hashable, Hashable]]]:
    if mode == 0:
        return frozenset((entity, entity) for entity in tagged_entities)
    positions = {entity: index for index, entity in enumerate(tagged_entities)}
    if mode == 1:
        return frozenset(
            (left, right)
            for left in tagged_entities
            for right in tagged_entities
            if positions[left] <= positions[right]
        )
    return frozenset(
        (left, right)
        for left in tagged_entities
        for right in tagged_entities
        if positions[left] >= positions[right]
    )


def build_multiworld_state(
    code: MultiworldStateCode,
) -> CoherentStructuralState:
    """Construct one exact state without consulting a global state codebook."""

    labels = tuple(
        BASE_LABELS[(identifier - code.label_phase) % len(ENTITY_IDS)]
        for identifier in ENTITY_IDS
    )
    tagged = tuple((ENTITY_TYPE, identifier) for identifier in ENTITY_IDS)
    simplices: set[frozenset[tuple[Hashable, Hashable]]] = {frozenset()}
    simplices.update(frozenset((entity,)) for entity in tagged)
    topology_edges = _topology_edges(code.topology_mode)
    simplices.update(
        frozenset(((ENTITY_TYPE, left), (ENTITY_TYPE, right)))
        for left, right in topology_edges
    )

    coordinates = _metric_coordinates(code.metric_mode)
    distances = tuple(
        tuple(abs(coordinates[left] - coordinates[right]) for right in ENTITY_IDS)
        for left in ENTITY_IDS
    )
    adjacent = frozenset(
        pair for edge in topology_edges for pair in (edge, tuple(reversed(edge)))
    )
    relational = FiniteRelationAssignment(
        schema=P3_MULTIWORLD_SCHEMA,
        carriers={ENTITY_TYPE: ENTITY_IDS},
        labels={ENTITY_TYPE: labels},
        generators={
            "adjacent": FiniteRelation(ENTITY_IDS, ENTITY_IDS, adjacent),
            "influences": FiniteRelation(
                ENTITY_IDS,
                ENTITY_IDS,
                _influence_pairs(code.influence_mode),
            ),
        },
    )
    core = IntegratedStructuralState(
        relational=relational,
        simplices=frozenset(simplices),
        distances=distances,
    )
    order = FinitePreorder(
        core.tagged_entities,
        _order_relation(core.tagged_entities, code.order_mode),
        core.tagged_labels,
    )
    return CoherentStructuralState(
        core=core,
        order=order,
        signature=P3_MULTIWORLD_SIGNATURE,
    )


@dataclass(frozen=True)
class StructuredAction:
    """A named intervention with an explicit five-layer action vector."""

    name: str
    components: tuple[int, int, int, int, int]
    role: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("action name must be nonempty")
        if len(self.components) != len(LAYER_ORDER):
            raise ValueError("action components must follow the five-layer order")
        if any(
            type(value) is not int or value not in (0, 1, 2)
            for value in self.components
        ):
            raise ValueError("action components must be ternary integers")

    @property
    def mapping(self) -> Mapping[str, int]:
        return MappingProxyType(dict(zip(LAYER_ORDER, self.components, strict=True)))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "components": list(self.components),
            "role": self.role,
        }


PRIMITIVE_ACTIONS = (
    StructuredAction("hold", (0, 0, 0, 0, 0), "training_primitive"),
    StructuredAction("label_step", (1, 0, 0, 0, 0), "training_primitive"),
    StructuredAction("topology_step", (0, 1, 0, 0, 0), "training_primitive"),
    StructuredAction("metric_step", (0, 0, 1, 0, 0), "training_primitive"),
    StructuredAction("relation_step", (0, 0, 0, 1, 0), "training_primitive"),
    StructuredAction("order_step", (0, 0, 0, 0, 1), "training_primitive"),
)

MECHANISM_PARAMETER_ACTIONS = (
    StructuredAction(
        "double_metric",
        (0, 0, 2, 0, 0),
        "unseen_mechanism_parameter",
    ),
    StructuredAction(
        "double_relation",
        (0, 0, 0, 2, 0),
        "unseen_mechanism_parameter",
    ),
)

COMPOSED_ACTIONS = (
    StructuredAction(
        "label_topology",
        (1, 1, 0, 0, 0),
        "unseen_action_composition",
    ),
    StructuredAction(
        "metric_order",
        (0, 0, 1, 0, 1),
        "unseen_action_composition",
    ),
)

BRIDGE_PROBE_ACTION = StructuredAction(
    "bridge_probe",
    (0, 1, 0, 0, 0),
    "bridge_consistent_shift",
)

ALL_ACTIONS = (
    *PRIMITIVE_ACTIONS,
    *MECHANISM_PARAMETER_ACTIONS,
    *COMPOSED_ACTIONS,
    BRIDGE_PROBE_ACTION,
)


@dataclass(frozen=True)
class WorldMechanism:
    """One independent transition mechanism with a declared dependency graph."""

    family: WorldFamily
    cohort: BenchmarkSplit
    world_index: int
    layer_multipliers: tuple[int, int, int, int, int]
    bridge_coefficient: int
    context_coefficient: int
    root_commitment: str
    mechanism_digest: str

    def __post_init__(self) -> None:
        if type(self.world_index) is not int or self.world_index < 0:
            raise ValueError("world_index must be a nonnegative integer")
        if len(self.layer_multipliers) != len(LAYER_ORDER):
            raise ValueError("one multiplier is required per layer")
        for index, multiplier in enumerate(self.layer_multipliers):
            modulus = 4 if index == 3 else 3
            if multiplier not in range(1, modulus):
                raise ValueError("layer multipliers must be invertible and nonzero")
        if self.bridge_coefficient not in (1, 3):
            raise ValueError("bridge coefficient must be odd modulo four")
        if self.context_coefficient not in (1, 2):
            raise ValueError("context coefficient must be nonzero modulo three")

    @property
    def identifier(self) -> str:
        return f"{self.cohort.value}:{self.family.value}:{self.world_index:03d}"

    @property
    def dependency_graph(self) -> tuple[tuple[str, str], ...]:
        dependencies = [(layer, layer) for layer in LAYER_ORDER]
        if self.family in (
            WorldFamily.BRIDGE_COUPLED,
            WorldFamily.CONTEXT_DEPENDENT,
        ):
            dependencies.append(("topology", "relation"))
        if self.family is WorldFamily.CONTEXT_DEPENDENT:
            dependencies.append(("order", "metric"))
        return tuple(dependencies)

    @property
    def active_parameter_signature(self) -> tuple[object, ...]:
        """Return only parameters that affect this family's transition law."""

        if self.family is WorldFamily.SEPARABLE:
            return (self.layer_multipliers,)
        if self.family is WorldFamily.BRIDGE_COUPLED:
            return (self.layer_multipliers, self.bridge_coefficient)
        return (
            self.layer_multipliers,
            self.bridge_coefficient,
            self.context_coefficient,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "family": self.family.value,
            "cohort": self.cohort.value,
            "world_index": self.world_index,
            "layer_multipliers": list(self.layer_multipliers),
            "bridge_coefficient": self.bridge_coefficient,
            "context_coefficient": self.context_coefficient,
            "active_parameter_signature": [
                list(value) if isinstance(value, tuple) else value
                for value in self.active_parameter_signature
            ],
            "dependency_graph": [list(edge) for edge in self.dependency_graph],
            "root_commitment": self.root_commitment,
            "mechanism_digest": self.mechanism_digest,
        }


def _root_for_cohort(cohort: BenchmarkSplit) -> str:
    if cohort is BenchmarkSplit.DEVELOPMENT:
        return DEVELOPMENT_ROOT
    if cohort is BenchmarkSplit.VALIDATION:
        return VALIDATION_ROOT
    raise ValueError("sealed-test roots are unavailable before reveal")


def _ranked_active_parameters(
    family: WorldFamily,
) -> tuple[tuple[tuple[int, int, int, int, int], int, int], ...]:
    layer_multipliers = tuple(
        (label, topology, metric, relation, order)
        for label, topology, metric, relation, order in product(
            (1, 2),
            (1, 2),
            (1, 2),
            (1, 2, 3),
            (1, 2),
        )
    )
    bridge_values = (1,) if family is WorldFamily.SEPARABLE else (1, 3)
    context_values = (1, 2) if family is WorldFamily.CONTEXT_DEPENDENT else (1,)
    candidates = tuple(
        (multipliers, bridge, context)
        for multipliers in layer_multipliers
        for bridge in bridge_values
        for context in context_values
    )
    ranking_seed = (
        f"{P3_MULTIWORLD_GENERATOR_ID}:active-parameters:{family.value}"
    ).encode("utf-8")
    return tuple(
        sorted(
            candidates,
            key=lambda candidate: sha256(
                ranking_seed
                + json.dumps(candidate, separators=(",", ":")).encode("utf-8")
            ).digest(),
        )
    )


def build_world_mechanism(
    family: WorldFamily,
    cohort: BenchmarkSplit,
    world_index: int,
) -> WorldMechanism:
    """Derive one development/validation mechanism from a public root."""

    if type(world_index) is not int or world_index < 0:
        raise ValueError("world_index must be a nonnegative integer")
    root = _root_for_cohort(cohort)
    identifier = f"{root}:{family.value}:{world_index}"
    manifest_index = world_index + (
        DEVELOPMENT_WORLDS_PER_FAMILY if cohort is BenchmarkSplit.VALIDATION else 0
    )
    candidates = _ranked_active_parameters(family)
    if manifest_index >= len(candidates):
        raise ValueError("world index exceeds unique active parameter supply")
    multipliers, bridge, context = candidates[manifest_index]
    payload = {
        "identifier": identifier,
        "multipliers": multipliers,
        "bridge": bridge,
        "context": context,
    }
    mechanism_digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return WorldMechanism(
        family=family,
        cohort=cohort,
        world_index=world_index,
        layer_multipliers=multipliers,
        bridge_coefficient=bridge,
        context_coefficient=context,
        root_commitment=sha256(root.encode("utf-8")).hexdigest(),
        mechanism_digest=mechanism_digest,
    )


def successor_code(
    source: MultiworldStateCode,
    action: StructuredAction,
    mechanism: WorldMechanism,
) -> MultiworldStateCode:
    """Apply the world's declared transition dependencies exactly."""

    action_map = action.mapping
    multipliers = dict(zip(LAYER_ORDER, mechanism.layer_multipliers, strict=True))
    label_delta = multipliers["label"] * action_map["label"]
    topology_delta = multipliers["topology"] * action_map["topology"]
    metric_delta = multipliers["metric"] * action_map["metric"]
    relation_delta = multipliers["relation"] * action_map["relation"]
    order_delta = multipliers["order"] * action_map["order"]

    if mechanism.family in (
        WorldFamily.BRIDGE_COUPLED,
        WorldFamily.CONTEXT_DEPENDENT,
    ):
        relation_delta += (
            mechanism.bridge_coefficient
            * action_map["topology"]
            * (1 + source.topology_mode)
        )
    if mechanism.family is WorldFamily.CONTEXT_DEPENDENT:
        metric_delta += (
            mechanism.context_coefficient * action_map["metric"] * source.order_mode
        )

    return MultiworldStateCode(
        label_phase=(source.label_phase + label_delta) % 3,
        topology_mode=(source.topology_mode + topology_delta) % 3,
        metric_mode=(source.metric_mode + metric_delta) % 3,
        influence_mode=(source.influence_mode + relation_delta) % 4,
        order_mode=(source.order_mode + order_delta) % 3,
    )


def violating_bridge_successor_code(
    source: MultiworldStateCode,
    action: StructuredAction,
    mechanism: WorldMechanism,
) -> MultiworldStateCode:
    """Return the negative-control target with the bridge direction reversed."""

    correct = successor_code(source, action, mechanism)
    action_topology = action.mapping["topology"]
    if mechanism.family is WorldFamily.SEPARABLE:
        # Inject a spurious topology-to-relation path as a null-family control.
        bridge_term = action_topology * (1 + source.topology_mode)
        influence_mode = (correct.influence_mode + bridge_term) % 4
    else:
        bridge_term = (
            mechanism.bridge_coefficient * action_topology * (1 + source.topology_mode)
        )
        influence_mode = (correct.influence_mode - 2 * bridge_term) % 4
    return MultiworldStateCode(
        label_phase=correct.label_phase,
        topology_mode=correct.topology_mode,
        metric_mode=correct.metric_mode,
        influence_mode=influence_mode,
        order_mode=correct.order_mode,
    )


def build_multiworld_tracking(
    source: CoherentStructuralState,
    target: CoherentStructuralState,
    action: StructuredAction,
    mechanism: WorldMechanism,
) -> TrackedTransition:
    """Build the exact total label-preserving tracking map."""

    label_delta = (mechanism.layer_multipliers[0] * action.mapping["label"]) % len(
        ENTITY_IDS
    )
    pairs = frozenset(
        (identifier, (identifier + label_delta) % len(ENTITY_IDS))
        for identifier in ENTITY_IDS
    )
    return TrackedTransition(
        source=source.core,
        target=target.core,
        components={
            ENTITY_TYPE: PartialBijection(
                source.core.relational.carriers[ENTITY_TYPE],
                target.core.relational.carriers[ENTITY_TYPE],
                pairs,
            )
        },
    )


@dataclass(frozen=True)
class GeneratedTransitionCase:
    """One exact transition or an explicitly marked negative control."""

    partition: str
    ood_slice: str | None
    source_code: MultiworldStateCode
    action: StructuredAction
    target_code: MultiworldStateCode
    follows_declared_mechanism: bool

    def __post_init__(self) -> None:
        if self.partition not in PARTITION_ORDER:
            raise ValueError("unknown transition partition")
        if self.partition == "ood" and self.ood_slice is None:
            raise ValueError("OOD cases require a named slice")
        if self.partition != "ood" and self.ood_slice is not None:
            raise ValueError("non-OOD cases cannot have an OOD slice")

    @property
    def input_key(self) -> tuple[MultiworldStateCode, tuple[int, ...]]:
        return self.source_code, self.action.components


@dataclass(frozen=True)
class GeneratedWorldDataset:
    mechanism: WorldMechanism
    partitions: Mapping[str, tuple[GeneratedTransitionCase, ...]]
    digest: str

    def __post_init__(self) -> None:
        if tuple(self.partitions) != PARTITION_ORDER:
            raise ValueError("world partitions must use the frozen order")
        object.__setattr__(
            self,
            "partitions",
            MappingProxyType(
                {
                    partition: tuple(cases)
                    for partition, cases in self.partitions.items()
                }
            ),
        )

    @property
    def ood_by_slice(self) -> Mapping[str, tuple[GeneratedTransitionCase, ...]]:
        grouped: dict[str, list[GeneratedTransitionCase]] = {}
        for case in self.partitions["ood"]:
            assert case.ood_slice is not None
            grouped.setdefault(case.ood_slice, []).append(case)
        return MappingProxyType(
            {name: tuple(cases) for name, cases in sorted(grouped.items())}
        )


def _source_residue(code: MultiworldStateCode) -> int:
    return (
        code.label_phase
        + 2 * code.topology_mode
        + 3 * code.metric_mode
        + 5 * code.order_mode
        + code.influence_mode
    ) % 7


def _is_unseen_structural_mode(code: MultiworldStateCode) -> bool:
    return (
        code.topology_mode == 2
        and code.order_mode == 2
        and code.influence_mode in (2, 3)
    )


def _case_payload(case: GeneratedTransitionCase) -> dict[str, object]:
    return {
        "partition": case.partition,
        "ood_slice": case.ood_slice,
        "source": list(case.source_code.as_tuple()),
        "action": case.action.as_dict(),
        "target": list(case.target_code.as_tuple()),
        "follows_declared_mechanism": case.follows_declared_mechanism,
    }


def build_world_dataset(mechanism: WorldMechanism) -> GeneratedWorldDataset:
    """Build frozen train/validation/OOD cases for one independent world."""

    partitions: dict[str, list[GeneratedTransitionCase]] = {
        partition: [] for partition in PARTITION_ORDER
    }
    codes = all_multiworld_state_codes()

    for source in codes:
        structural_holdout = _is_unseen_structural_mode(source)
        residue = _source_residue(source)
        for action in PRIMITIVE_ACTIONS:
            target = successor_code(source, action, mechanism)
            if structural_holdout:
                partition = "ood"
                ood_slice = "unseen_structural_mode"
            elif residue == 6:
                partition = "ood"
                ood_slice = "unseen_recombination"
            elif residue == 5:
                partition = "validation"
                ood_slice = None
            else:
                partition = "train"
                ood_slice = None
            partitions[partition].append(
                GeneratedTransitionCase(
                    partition=partition,
                    ood_slice=ood_slice,
                    source_code=source,
                    action=action,
                    target_code=target,
                    follows_declared_mechanism=True,
                )
            )

        if residue == 0 and not structural_holdout:
            for action in MECHANISM_PARAMETER_ACTIONS:
                partitions["ood"].append(
                    GeneratedTransitionCase(
                        partition="ood",
                        ood_slice="unseen_mechanism_parameter",
                        source_code=source,
                        action=action,
                        target_code=successor_code(source, action, mechanism),
                        follows_declared_mechanism=True,
                    )
                )
        if residue == 1 and not structural_holdout:
            for action in COMPOSED_ACTIONS:
                partitions["ood"].append(
                    GeneratedTransitionCase(
                        partition="ood",
                        ood_slice="unseen_action_composition",
                        source_code=source,
                        action=action,
                        target_code=successor_code(source, action, mechanism),
                        follows_declared_mechanism=True,
                    )
                )
        if (residue == 6 or structural_holdout) and source.topology_mode in (0, 2):
            correct = successor_code(source, BRIDGE_PROBE_ACTION, mechanism)
            violating = violating_bridge_successor_code(
                source,
                BRIDGE_PROBE_ACTION,
                mechanism,
            )
            partitions["ood"].extend(
                (
                    GeneratedTransitionCase(
                        partition="ood",
                        ood_slice="bridge_consistent_shift",
                        source_code=source,
                        action=BRIDGE_PROBE_ACTION,
                        target_code=correct,
                        follows_declared_mechanism=True,
                    ),
                    GeneratedTransitionCase(
                        partition="ood",
                        ood_slice="bridge_violating_control",
                        source_code=source,
                        action=BRIDGE_PROBE_ACTION,
                        target_code=violating,
                        follows_declared_mechanism=False,
                    ),
                )
            )

    frozen_partitions = {name: tuple(cases) for name, cases in partitions.items()}
    payload = {
        "generator": P3_MULTIWORLD_GENERATOR_ID,
        "mechanism": mechanism.as_dict(),
        "partitions": {
            name: [_case_payload(case) for case in cases]
            for name, cases in frozen_partitions.items()
        },
    }
    digest = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return GeneratedWorldDataset(
        mechanism=mechanism,
        partitions=frozen_partitions,
        digest=digest,
    )


def development_validation_world_manifest() -> tuple[WorldMechanism, ...]:
    """Return the frozen public P3-3A world manifest without test worlds."""

    manifest: list[WorldMechanism] = []
    for cohort, count in (
        (BenchmarkSplit.DEVELOPMENT, DEVELOPMENT_WORLDS_PER_FAMILY),
        (BenchmarkSplit.VALIDATION, VALIDATION_WORLDS_PER_FAMILY),
    ):
        for family in WorldFamily:
            manifest.extend(
                build_world_mechanism(family, cohort, index) for index in range(count)
            )
    return tuple(manifest)


def multiworld_generator_digest() -> str:
    """Digest the semantic generator and public world-manifest contract."""

    payload = {
        "identifier": P3_MULTIWORLD_GENERATOR_ID,
        "parent_contract": P3_INDEPENDENCE_CONTRACT_ID,
        "evidence_contract": EVIDENCE_CONTRACT_ID,
        "state_count": len(all_multiworld_state_codes()),
        "layer_order": list(LAYER_ORDER),
        "schema_arrows": [
            {
                "name": arrow.name,
                "source": arrow.source,
                "target": arrow.target,
            }
            for arrow in P3_MULTIWORLD_SCHEMA.arrows
        ],
        "bridges": [
            {
                "arrow": bridge.arrow,
                "kind": bridge.kind,
                "threshold": bridge.threshold,
            }
            for bridge in P3_MULTIWORLD_SIGNATURE.bridges
        ],
        "actions": [action.as_dict() for action in ALL_ACTIONS],
        "manifest": [
            mechanism.as_dict() for mechanism in development_validation_world_manifest()
        ],
        "sealed_test_materialized": False,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
