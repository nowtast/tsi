"""Graph-conditioned factorization of the v3 primitive transition law."""

from __future__ import annotations

from dataclasses import dataclass

from .paper3_learned_v2_generator import GRAPH_VARIANT_MANIFEST
from .paper3_learned_v3_generator import V3TransitionCase, V3WorldDataset
from .paper3_learned_v2_mechanism import ObservableMechanismSignature


@dataclass(frozen=True)
class FactorizedSignature:
    graph_variant: str
    layer_multipliers: tuple[int, int, int, int, int]
    bridge_coefficient: int
    context_coefficient: int
    active_parameter: str
    matched_case_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "graph_variant": self.graph_variant,
            "layer_multipliers": list(self.layer_multipliers),
            "bridge_coefficient": self.bridge_coefficient,
            "context_coefficient": self.context_coefficient,
            "active_parameter": self.active_parameter,
            "matched_case_count": self.matched_case_count,
        }


def _delta(case: V3TransitionCase, layer: int) -> int:
    cardinality = 4 if layer == 3 else 3
    source = case.source_code.as_tuple()[layer]
    target = case.target_code.as_tuple()[layer]
    return (target - source) % cardinality


def _single_layer_cases(cases: tuple[V3TransitionCase, ...], layer: int, graph):
    layer_names = ("label", "topology", "metric", "relation", "order")
    target_name = layer_names[layer]
    source_name = next(
        (
            source
            for source, target in graph.source_cross_edges
            if target == target_name
        ),
        None,
    )
    source_index = layer_names.index(source_name) if source_name else None
    return tuple(
        case
        for case in cases
        if case.action.components[layer] == 1
        and sum(case.action.components) == 1
        and (source_index is None or case.source_code.as_tuple()[source_index] == 0)
    )


def _unique_parameter(
    candidates: tuple[int, ...], equations: list[tuple[int, int, int]]
) -> int:
    matches = tuple(
        candidate
        for candidate in candidates
        if all(
            (coefficient * value) % modulus == observed
            for value, observed, modulus in equations
            for coefficient in (candidate,)
        )
    )
    if len(matches) != 1:
        raise ValueError(f"factorized parameter is not uniquely identified: {matches}")
    return matches[0]


def factorize_training_signature(
    cases: tuple[V3TransitionCase, ...] | list[V3TransitionCase], graph_variant: str
) -> FactorizedSignature:
    """Estimate each parameter from its own primitive-action equation."""
    cases = tuple(cases)
    if not cases:
        raise ValueError("factorization requires primitive training cases")
    graph = next(
        item for item in GRAPH_VARIANT_MANIFEST if item.identifier == graph_variant
    )
    multipliers = []
    for layer in range(5):
        layer_cases = _single_layer_cases(cases, layer, graph)
        modulus = 4 if layer == 3 else 3
        equations = [(1, _delta(case, layer), modulus) for case in layer_cases]
        multipliers.append(
            _unique_parameter((1, 2, 3) if layer == 3 else (1, 2), equations)
        )

    bridge_equations = []
    context_equations = []
    layer_names = ("label", "topology", "metric", "relation", "order")
    source_name, target_name = graph.source_cross_edges[0]
    source_index = layer_names.index(source_name)
    target_index = layer_names.index(target_name)
    for case in cases:
        if (
            case.action.components[source_index] == 1
            and sum(case.action.components) == 1
        ):
            target_value = case.source_code.as_tuple()[target_index]
            equation = (
                (1 + target_value),
                _delta(case, target_index),
                4 if target_index == 3 else 3,
            )
            if target_name == "metric":
                context_equations.append(equation)
            else:
                bridge_equations.append(equation)
    bridge = (
        _unique_parameter((1, 3), bridge_equations) if target_name != "metric" else 1
    )
    context = (
        _unique_parameter((1, 2), context_equations) if target_name == "metric" else 1
    )
    active_parameter = (
        "context_coefficient"
        if graph_variant == "context_order_to_metric"
        else "bridge_coefficient"
    )
    return FactorizedSignature(
        graph_variant, tuple(multipliers), bridge, context, active_parameter, len(cases)
    )


def _as_observable(signature: FactorizedSignature) -> ObservableMechanismSignature:
    return ObservableMechanismSignature(
        graph_variant=signature.graph_variant,
        layer_multipliers=signature.layer_multipliers,
        bridge_coefficient=signature.bridge_coefficient,
        context_coefficient=signature.context_coefficient,
        matched_training_cases=signature.matched_case_count,
        candidate_count=1,
    )


@dataclass(frozen=True)
class FactorizedHeadTrace:
    training_world_count: int
    training_case_count: int
    active_signature_count: int


@dataclass(frozen=True)
class GraphConditionedFactorizedHead:
    trace: FactorizedHeadTrace

    @classmethod
    def fit(
        cls, datasets: tuple[V3WorldDataset, ...] | list[V3WorldDataset]
    ) -> "GraphConditionedFactorizedHead":
        if not datasets:
            raise ValueError("factorized head requires training datasets")
        signatures = [
            factorize_training_signature(
                dataset.partitions["train"], dataset.graph.identifier
            )
            for dataset in datasets
        ]
        return cls(
            FactorizedHeadTrace(
                len(datasets),
                sum(len(dataset.partitions["train"]) for dataset in datasets),
                len({signature.as_dict().__repr__() for signature in signatures}),
            )
        )

    def evaluate(
        self, dataset: V3WorldDataset, *, partition: str = "test"
    ) -> dict[str, object]:
        signature = factorize_training_signature(
            dataset.partitions["train"], dataset.graph.identifier
        )
        observable = _as_observable(signature)
        from .paper3_learned_v2_mechanism import predict_target_code

        cases = dataset.partitions[partition]
        correct = sum(
            predict_target_code(case.source_code, case.action, observable)
            == case.target_code
            for case in cases
        )
        return {
            "world_index": dataset.world_index,
            "mechanism_combination_index": dataset.mechanism_combination_index,
            "graph_variant": dataset.graph.identifier,
            "partition": partition,
            "case_count": len(cases),
            "exact_accuracy": correct / len(cases) if cases else float("nan"),
            "factorized_signature": signature.as_dict(),
        }
