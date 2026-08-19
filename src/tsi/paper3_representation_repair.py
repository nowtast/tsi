"""Validation-only representation repairs for TSI Paper 3 gate ``P3-2R``.

The official P3-2 result remains frozen.  This module changes only the
architecture used on the train and validation residues.  It constrains the
encoder and structural heads to declared layer factors, then optionally
constrains each action operator to preserve those factors.

The masks define a lower-dimensional constrained parameter space.  Masked
coordinates are fixed at zero and are not counted as active parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .paper3_objective_ablation import (
    FeatureSlices,
    ForwardPass,
    NumericSplit,
    ObjectiveCondition,
    P3AblationDataset,
    P3AblationSpec,
    TrainableStructuralJEPA,
    TrainingSnapshot,
)


P3_REPAIR_BENCHMARK_ID = "P3-2R-REPRESENTATION-v1"
REPAIR_ALLOWED_EVALUATION_SPLITS = ("train", "validation")
DEFAULT_REPAIR_SEEDS = (
    20_260_733,
    20_260_734,
    20_260_735,
    20_260_736,
    20_260_737,
)


class RepairVariant(str, Enum):
    """Cumulative, validation-only P3-2R interventions."""

    REFERENCE = "reference"
    LAYER_ROUTED = "layer_routed"
    FACTORIZED_ACTION = "factorized_action"


@dataclass(frozen=True)
class LatentFactorLayout:
    """Fixed allocation of the 16 latent coordinates to five state layers."""

    label: slice
    simplicial: slice
    metric: slice
    relation: slice
    order: slice
    dimension: int

    @classmethod
    def default(cls, latent_dimension: int) -> LatentFactorLayout:
        if latent_dimension != 16:
            raise ValueError("P3-2R-v1 requires latent dimension 16")
        return cls(
            label=slice(0, 4),
            simplicial=slice(4, 7),
            metric=slice(7, 10),
            relation=slice(10, 13),
            order=slice(13, 16),
            dimension=latent_dimension,
        )

    def items(self) -> tuple[tuple[str, slice], ...]:
        return (
            ("label", self.label),
            ("simplicial", self.simplicial),
            ("metric", self.metric),
            ("relation", self.relation),
            ("order", self.order),
        )


def _feature_slice(slices: FeatureSlices, name: str) -> slice:
    return getattr(slices, name)


def _encoder_mask(
    dataset: P3AblationDataset,
    factors: LatentFactorLayout,
) -> np.ndarray:
    mask = np.zeros(
        (dataset.input_dimension, factors.dimension),
        dtype=np.float64,
    )
    active_original = dataset.active_coordinates
    for name, latent_slice in factors.items():
        block = _feature_slice(dataset.slices, name)
        rows = np.flatnonzero(
            (active_original >= block.start) & (active_original < block.stop)
        )
        mask[rows, latent_slice] = 1.0
    if np.any(np.sum(mask, axis=1) == 0.0):
        raise RuntimeError("every active input coordinate needs one latent factor")
    return mask


def _head_mask(
    shape: tuple[int, ...],
    latent_slice: slice,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float64)
    mask[latent_slice, :] = 1.0
    return mask


def _block_action_mask(
    shape: tuple[int, ...],
    factors: LatentFactorLayout,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.float64)
    for _, latent_slice in factors.items():
        mask[:, latent_slice, latent_slice] = 1.0
    return mask


class RepairStructuralJEPA(TrainableStructuralJEPA):
    """Full-objective JEPA restricted to a declared architecture subspace."""

    def __init__(
        self,
        dataset: P3AblationDataset,
        variant: RepairVariant | str,
        seed: int,
        spec: P3AblationSpec,
    ) -> None:
        try:
            self.variant = (
                variant
                if isinstance(variant, RepairVariant)
                else RepairVariant(variant)
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"unknown P3-2R variant: {variant!r}") from error
        super().__init__(
            dataset,
            ObjectiveCondition.FULL,
            seed,
            spec,
        )
        self.factors = LatentFactorLayout.default(spec.latent_dimension)
        self.parameter_masks = self._build_parameter_masks()
        for name, mask in self.parameter_masks.items():
            self.parameters[name] *= mask
        self.target_weight *= self.parameter_masks["encoder_weight"]

    def _build_parameter_masks(self) -> Mapping[str, np.ndarray]:
        masks = {name: np.ones_like(value) for name, value in self.parameters.items()}
        if self.variant is RepairVariant.REFERENCE:
            return MappingProxyType(masks)

        masks["encoder_weight"] = _encoder_mask(self.dataset, self.factors)
        for name, latent_slice in self.factors.items():
            masks[f"{name}_weight"] = _head_mask(
                self.parameters[f"{name}_weight"].shape,
                latent_slice,
            )
        if self.variant is RepairVariant.FACTORIZED_ACTION:
            masks["predictor_weight"] = _block_action_mask(
                self.parameters["predictor_weight"].shape,
                self.factors,
            )
        return MappingProxyType(masks)

    @property
    def active_parameter_count(self) -> int:
        """Return the dimension of the constrained optimized parameter space."""

        return sum(
            int(np.count_nonzero(self.parameter_masks[name]))
            for name in self.parameters
        )

    @property
    def inactive_parameter_count(self) -> int:
        return self.parameter_count - self.active_parameter_count

    def mask_invariant_holds(self) -> bool:
        """Return whether every structurally inactive coordinate is exactly zero."""

        return bool(
            all(
                np.all(self.parameters[name][mask == 0.0] == 0.0)
                for name, mask in self.parameter_masks.items()
            )
            and np.all(
                self.target_weight[self.parameter_masks["encoder_weight"] == 0.0] == 0.0
            )
        )

    def _losses_and_gradients(
        self,
        split: NumericSplit,
        *,
        gradients: bool,
    ) -> tuple[
        TrainingSnapshot,
        Mapping[str, np.ndarray] | None,
        ForwardPass,
    ]:
        snapshot, parameter_gradients, forward = super()._losses_and_gradients(
            split,
            gradients=gradients,
        )
        if parameter_gradients is None:
            return snapshot, None, forward
        constrained = {
            name: parameter_gradients[name] * self.parameter_masks[name]
            for name in self.parameters
        }
        return snapshot, MappingProxyType(constrained), forward


def build_repair_model(
    dataset: P3AblationDataset,
    variant: RepairVariant | str,
    seed: int,
    spec: P3AblationSpec,
) -> RepairStructuralJEPA:
    """Construct one full-objective P3-2R model without touching test data."""

    return RepairStructuralJEPA(dataset, variant, seed, spec)
