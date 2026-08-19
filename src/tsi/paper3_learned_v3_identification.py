"""Training-only identification audit for every v3 mechanism combination."""

from __future__ import annotations

from .paper3_learned_v2_mechanism import identify_observable_mechanism
from .paper3_learned_v3_contract import mechanism_split_for_combination
from .paper3_learned_v3_generator import build_v3_world_dataset


def identify_v3_training_mechanism(dataset) -> dict[str, object]:
    """Identify graph and mechanism from the v3 train partition only."""
    signature = identify_observable_mechanism(dataset.partitions["train"])
    expected = dataset.mechanism
    if dataset.graph.identifier == "context_order_to_metric":
        expected_signature = (expected.layer_multipliers, expected.context_coefficient)
        identified_signature = (signature.layer_multipliers, signature.context_coefficient)
    else:
        expected_signature = (expected.layer_multipliers, expected.bridge_coefficient)
        identified_signature = (signature.layer_multipliers, signature.bridge_coefficient)
    return {
        "world_index": dataset.world_index,
        "graph_variant": dataset.graph.identifier,
        "mechanism_combination_index": dataset.mechanism_combination_index,
        "mechanism_split": mechanism_split_for_combination(dataset.mechanism_combination_index),
        "signature": signature.as_dict(),
        "expected_graph_variant": dataset.graph.identifier,
        "expected_mechanism_tuple": [list(expected.layer_multipliers), expected.bridge_coefficient, expected.context_coefficient],
        "graph_exact": signature.graph_variant == dataset.graph.identifier,
        "active_mechanism_exact": identified_signature == expected_signature,
        "inactive_parameter_ambiguity_expected": signature.candidate_count == 2,
        "candidate_count": signature.candidate_count,
        "training_case_count": len(dataset.partitions["train"]),
    }


def run_v3_identification_audit(
    *,
    combination_indices: tuple[int, ...] | None = None,
    graph_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[dict[str, object], ...]:
    if combination_indices is None:
        from .paper3_learned_v3_contract import MECHANISM_HYPOTHESIS_COUNT

        combination_indices = tuple(range(MECHANISM_HYPOTHESIS_COUNT))
    if not combination_indices or not graph_indices:
        raise ValueError("identification audit requires combinations and graphs")
    results = []
    for combination_index in combination_indices:
        for graph_index in graph_indices:
            dataset = build_v3_world_dataset(
                world_index=combination_index * len(graph_indices) + graph_index,
                mechanism_combination_index=combination_index,
                graph_index=graph_index,
            )
            results.append(identify_v3_training_mechanism(dataset))
    return tuple(results)
