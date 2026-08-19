"""Frozen comparative-validation contract for Paper 4."""

from __future__ import annotations

from dataclasses import dataclass


PAPER4_CONTRACT_ID = "TSI-PAPER4-COMPARATIVE-v3"
MODEL_FAMILIES = (
    "diagonal_trainable",
    "dense_polynomial_trainable",
    "unstructured_lookup",
    "wrong_routed_factorized",
    "tsi_graph_discovered_factorized",
)
SEEDS = (0, 1, 2, 3, 4)
PRIMARY_METRIC = "intervention_exact_accuracy"
PRIMARY_CONTRAST = "tsi_graph_discovered_factorized - dense_polynomial_trainable"
COMPUTE_BUDGET = {
    "feature_evaluations_per_case": 36,
    "fit_passes": 1,
    "prediction_passes": 1,
    "seed_count": len(SEEDS),
}


@dataclass(frozen=True)
class Paper4Contract:
    identifier: str = PAPER4_CONTRACT_ID
    model_families: tuple[str, ...] = MODEL_FAMILIES
    seeds: tuple[int, ...] = SEEDS
    primary_metric: str = PRIMARY_METRIC
    primary_contrast: str = PRIMARY_CONTRAST

    def as_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "model_families": list(self.model_families),
            "seeds": list(self.seeds),
            "primary_metric": self.primary_metric,
            "primary_contrast": self.primary_contrast,
            "compute_budget": COMPUTE_BUDGET,
            "independent_unit": "graph_mechanism_cell",
            "nested_unit": "bootstrap_seed",
            "test_includes_interventions": True,
        }


FROZEN_PAPER4_CONTRACT = Paper4Contract()
