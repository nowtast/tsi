"""Trainable capacity-matched routing models for the P3-3 exact-state gate.

Each control receives the same 31-dimensional exact-state/action input and the
same 420 trainable coefficients. A frozen mask determines which source and
action coordinates can influence each target layer. Fixed random nonlinear
features make interactions expressible without counting nontrainable routing
features as learned capacity; their distribution and count are paired across
controls and optimizer seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
from typing import Sequence

import numpy as np

from .paper3_multiworld import GeneratedTransitionCase, LAYER_ORDER, MultiworldStateCode
from .paper3_routing_controls import (
    FULL_INPUT_WIDTH,
    OUTPUT_LOGIT_COUNT,
    TRAINING_MINIBATCH_SIZE,
    TRAINABLE_RANDOM_FEATURE_COEFFICIENTS,
    TRAINING_UPDATES,
    TRANSITION_ACTIVE_PARAMETER_BUDGET,
    RoutingControlManifest,
    routing_control_digest,
)


P3_ROUTING_MODEL_ID = "P3-3A-MASKED-RANDOM-FEATURE-DELTA-v1"
LAYER_CARDINALITIES = (3, 3, 3, 4, 3)
MAX_FEATURES_PER_LOGIT = 26
MINIBATCH_SIZE = TRAINING_MINIBATCH_SIZE
LEARNING_RATE = 0.03
ADAM_BETA_1 = 0.9
ADAM_BETA_2 = 0.999
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 1.0e-4


def _slices(widths: Sequence[int], start: int = 0) -> tuple[slice, ...]:
    result: list[slice] = []
    cursor = start
    for width in widths:
        result.append(slice(cursor, cursor + width))
        cursor += width
    return tuple(result)


SOURCE_FEATURE_SLICES = _slices(LAYER_CARDINALITIES)
ACTION_FEATURE_SLICES = _slices((3,) * len(LAYER_ORDER), start=16)
if ACTION_FEATURE_SLICES[-1].stop != FULL_INPUT_WIDTH:
    raise RuntimeError("the routing model input layout does not have width 31")

LOGIT_OFFSETS = tuple(
    sum(LAYER_CARDINALITIES[:index]) for index in range(len(LAYER_ORDER))
)
LOGIT_TO_LAYER = tuple(
    layer_index
    for layer_index, cardinality in enumerate(LAYER_CARDINALITIES)
    for _ in range(cardinality)
)
FEATURES_PER_LOGIT = tuple(
    TRAINABLE_RANDOM_FEATURE_COEFFICIENTS // OUTPUT_LOGIT_COUNT
    + (index < TRAINABLE_RANDOM_FEATURE_COEFFICIENTS % OUTPUT_LOGIT_COUNT)
    for index in range(OUTPUT_LOGIT_COUNT)
)
if sum(FEATURES_PER_LOGIT) + OUTPUT_LOGIT_COUNT != (TRANSITION_ACTIVE_PARAMETER_BUDGET):
    raise RuntimeError("the frozen trainable parameter budget does not close")


def encode_cases(
    cases: Sequence[GeneratedTransitionCase],
) -> tuple[np.ndarray, np.ndarray]:
    """Encode exact local coordinates and target deltas without a state codebook."""

    inputs = np.zeros((len(cases), FULL_INPUT_WIDTH), dtype=np.float64)
    deltas = np.zeros((len(cases), len(LAYER_ORDER)), dtype=np.int64)
    for row, case in enumerate(cases):
        source = case.source_code.as_tuple()
        target = case.target_code.as_tuple()
        for layer, (value, feature_slice) in enumerate(
            zip(source, SOURCE_FEATURE_SLICES, strict=True)
        ):
            inputs[row, feature_slice.start + value] = 1.0
            deltas[row, layer] = (target[layer] - value) % LAYER_CARDINALITIES[layer]
        for value, feature_slice in zip(
            case.action.components,
            ACTION_FEATURE_SLICES,
            strict=True,
        ):
            inputs[row, feature_slice.start + value] = 1.0
    return inputs, deltas


def routing_input_masks(manifest: RoutingControlManifest) -> np.ndarray:
    """Return one frozen 31-coordinate mask for each predicted target layer."""

    masks = np.zeros((len(LAYER_ORDER), FULL_INPUT_WIDTH), dtype=np.float64)
    layer_index = {name: index for index, name in enumerate(LAYER_ORDER)}
    for source, target in manifest.source_edges:
        masks[layer_index[target], SOURCE_FEATURE_SLICES[layer_index[source]]] = 1.0
    for source, target in manifest.action_edges:
        masks[layer_index[target], ACTION_FEATURE_SLICES[layer_index[source]]] = 1.0
    if np.any(np.sum(masks, axis=1) == 0.0):
        raise ValueError("every target layer needs at least one routed input")
    return masks


def _seed_from(*parts: object) -> int:
    payload = ":".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "little")


@dataclass(frozen=True)
class RoutingTrainingTrace:
    initial_nll: float
    final_nll: float
    update_count: int
    minibatch_size: int
    finite: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "initial_nll": self.initial_nll,
            "final_nll": self.final_nll,
            "update_count": self.update_count,
            "minibatch_size": self.minibatch_size,
            "finite": self.finite,
        }


class MaskedRandomFeatureBasis:
    """Paired nontrainable feature bank whose masks implement causal routing."""

    def __init__(
        self,
        manifest: RoutingControlManifest,
        optimizer_seed: int,
    ) -> None:
        if type(optimizer_seed) is not int or optimizer_seed < 0:
            raise ValueError("optimizer_seed must be a nonnegative integer")
        self.manifest = manifest
        self.optimizer_seed = optimizer_seed
        self.input_masks = routing_input_masks(manifest)
        self.valid_features = np.zeros(
            (OUTPUT_LOGIT_COUNT, MAX_FEATURES_PER_LOGIT),
            dtype=bool,
        )
        self.weights = np.zeros(
            (OUTPUT_LOGIT_COUNT, MAX_FEATURES_PER_LOGIT, FULL_INPUT_WIDTH),
            dtype=np.float64,
        )
        self.biases = np.zeros(
            (OUTPUT_LOGIT_COUNT, MAX_FEATURES_PER_LOGIT),
            dtype=np.float64,
        )
        for logit, count in enumerate(FEATURES_PER_LOGIT):
            layer = LOGIT_TO_LAYER[logit]
            mask = self.input_masks[layer]
            active_width = int(np.sum(mask))
            rng = np.random.default_rng(
                _seed_from(
                    P3_ROUTING_MODEL_ID,
                    manifest.family.value,
                    optimizer_seed,
                    logit,
                )
            )
            self.valid_features[logit, :count] = True
            self.weights[logit, :count] = (
                rng.normal(size=(count, FULL_INPUT_WIDTH))
                * mask[np.newaxis, :]
                / sqrt(active_width)
            )
            self.biases[logit, :count] = rng.normal(scale=0.5, size=count)

    def transform_inputs(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FULL_INPUT_WIDTH:
            raise ValueError("routing inputs must have shape (n, 31)")
        if not np.all(np.isfinite(values)):
            raise ValueError("routing inputs must be finite")
        features = np.tanh(
            np.einsum("ni,cki->nck", values, self.weights, optimize=True)
            + self.biases[np.newaxis, :, :]
        )
        features[:, ~self.valid_features] = 0.0
        return features

    def transform_cases(
        self,
        cases: Sequence[GeneratedTransitionCase],
    ) -> tuple[np.ndarray, np.ndarray]:
        inputs, deltas = encode_cases(cases)
        return self.transform_inputs(inputs), deltas


class TrainableRoutingModel:
    """Five delta classifiers trained with exact manual NumPy Adam updates."""

    def __init__(
        self,
        manifest: RoutingControlManifest,
        optimizer_seed: int,
    ) -> None:
        self.manifest = manifest
        self.optimizer_seed = optimizer_seed
        self.basis = MaskedRandomFeatureBasis(manifest, optimizer_seed)
        self.coefficients = np.zeros(
            (OUTPUT_LOGIT_COUNT, MAX_FEATURES_PER_LOGIT),
            dtype=np.float64,
        )
        self.logit_biases = np.zeros(OUTPUT_LOGIT_COUNT, dtype=np.float64)
        self.trace: RoutingTrainingTrace | None = None

    @property
    def parameter_count(self) -> int:
        return int(np.sum(self.basis.valid_features)) + len(self.logit_biases)

    def _logits(self, features: np.ndarray) -> np.ndarray:
        return (
            np.einsum(
                "nck,ck->nc",
                features,
                self.coefficients,
                optimize=True,
            )
            + self.logit_biases[np.newaxis, :]
        )

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def _nll(self, features: np.ndarray, deltas: np.ndarray) -> float:
        logits = self._logits(features)
        losses: list[np.ndarray] = []
        rows = np.arange(len(features))
        for layer, (offset, cardinality) in enumerate(
            zip(LOGIT_OFFSETS, LAYER_CARDINALITIES, strict=True)
        ):
            probabilities = self._softmax(logits[:, offset : offset + cardinality])
            losses.append(
                -np.log(
                    np.maximum(
                        probabilities[rows, deltas[:, layer]],
                        np.finfo(np.float64).tiny,
                    )
                )
            )
        return float(np.mean(np.column_stack(losses)))

    def fit_precomputed(
        self,
        features: np.ndarray,
        deltas: np.ndarray,
        *,
        updates: int = TRAINING_UPDATES,
    ) -> RoutingTrainingTrace:
        values = np.asarray(features, dtype=np.float64)
        targets = np.asarray(deltas, dtype=np.int64)
        if values.shape[1:] != (
            OUTPUT_LOGIT_COUNT,
            MAX_FEATURES_PER_LOGIT,
        ):
            raise ValueError("precomputed feature shape is inconsistent")
        if targets.shape != (len(values), len(LAYER_ORDER)):
            raise ValueError("delta target shape is inconsistent")
        if type(updates) is not int or updates <= 0:
            raise ValueError("updates must be a positive integer")
        if len(values) == 0:
            raise ValueError("training requires at least one example")

        initial_nll = self._nll(values, targets)
        coefficient_m = np.zeros_like(self.coefficients)
        coefficient_v = np.zeros_like(self.coefficients)
        bias_m = np.zeros_like(self.logit_biases)
        bias_v = np.zeros_like(self.logit_biases)
        rng = np.random.default_rng(
            _seed_from(
                P3_ROUTING_MODEL_ID,
                self.manifest.family.value,
                self.optimizer_seed,
                "minibatches",
            )
        )
        order = rng.permutation(len(values))
        cursor = 0

        for update in range(1, updates + 1):
            if cursor + MINIBATCH_SIZE > len(values):
                order = rng.permutation(len(values))
                cursor = 0
            indices = order[cursor : cursor + MINIBATCH_SIZE]
            cursor += MINIBATCH_SIZE
            batch_features = values[indices]
            batch_targets = targets[indices]
            logits = self._logits(batch_features)
            residuals = np.zeros_like(logits)
            rows = np.arange(len(indices))
            for layer, (offset, cardinality) in enumerate(
                zip(LOGIT_OFFSETS, LAYER_CARDINALITIES, strict=True)
            ):
                probabilities = self._softmax(logits[:, offset : offset + cardinality])
                probabilities[
                    rows,
                    batch_targets[:, layer],
                ] -= 1.0
                residuals[:, offset : offset + cardinality] = probabilities / len(
                    indices
                )

            coefficient_gradient = np.einsum(
                "nck,nc->ck",
                batch_features,
                residuals,
                optimize=True,
            )
            coefficient_gradient += WEIGHT_DECAY * self.coefficients
            coefficient_gradient[~self.basis.valid_features] = 0.0
            bias_gradient = np.sum(residuals, axis=0)

            coefficient_m = (
                ADAM_BETA_1 * coefficient_m + (1.0 - ADAM_BETA_1) * coefficient_gradient
            )
            coefficient_v = (
                ADAM_BETA_2 * coefficient_v
                + (1.0 - ADAM_BETA_2) * coefficient_gradient**2
            )
            bias_m = ADAM_BETA_1 * bias_m + (1.0 - ADAM_BETA_1) * bias_gradient
            bias_v = ADAM_BETA_2 * bias_v + (1.0 - ADAM_BETA_2) * bias_gradient**2
            coefficient_m_hat = coefficient_m / (1.0 - ADAM_BETA_1**update)
            coefficient_v_hat = coefficient_v / (1.0 - ADAM_BETA_2**update)
            bias_m_hat = bias_m / (1.0 - ADAM_BETA_1**update)
            bias_v_hat = bias_v / (1.0 - ADAM_BETA_2**update)
            self.coefficients -= (
                LEARNING_RATE
                * coefficient_m_hat
                / (np.sqrt(coefficient_v_hat) + ADAM_EPSILON)
            )
            self.logit_biases -= (
                LEARNING_RATE * bias_m_hat / (np.sqrt(bias_v_hat) + ADAM_EPSILON)
            )
            self.coefficients[~self.basis.valid_features] = 0.0

        final_nll = self._nll(values, targets)
        finite = bool(
            np.isfinite(initial_nll)
            and np.isfinite(final_nll)
            and np.all(np.isfinite(self.coefficients))
            and np.all(np.isfinite(self.logit_biases))
        )
        self.trace = RoutingTrainingTrace(
            initial_nll=initial_nll,
            final_nll=final_nll,
            update_count=updates,
            minibatch_size=MINIBATCH_SIZE,
            finite=finite,
        )
        return self.trace

    def fit(
        self,
        cases: Sequence[GeneratedTransitionCase],
        *,
        updates: int = TRAINING_UPDATES,
    ) -> RoutingTrainingTrace:
        features, deltas = self.basis.transform_cases(cases)
        return self.fit_precomputed(features, deltas, updates=updates)

    def predict_deltas_precomputed(self, features: np.ndarray) -> np.ndarray:
        logits = self._logits(np.asarray(features, dtype=np.float64))
        predictions = np.zeros((len(features), len(LAYER_ORDER)), dtype=np.int64)
        for layer, (offset, cardinality) in enumerate(
            zip(LOGIT_OFFSETS, LAYER_CARDINALITIES, strict=True)
        ):
            predictions[:, layer] = np.argmax(
                logits[:, offset : offset + cardinality],
                axis=1,
            )
        return predictions

    def predict_codes_precomputed(
        self,
        cases: Sequence[GeneratedTransitionCase],
        features: np.ndarray,
    ) -> tuple[MultiworldStateCode, ...]:
        deltas = self.predict_deltas_precomputed(features)
        predictions: list[MultiworldStateCode] = []
        for case, row in zip(cases, deltas, strict=True):
            source = case.source_code.as_tuple()
            target = tuple(
                (source[layer] + int(row[layer])) % LAYER_CARDINALITIES[layer]
                for layer in range(len(LAYER_ORDER))
            )
            predictions.append(MultiworldStateCode(*target))
        return tuple(predictions)

    def parameter_digest(self) -> str:
        digest = sha256()
        digest.update(self.manifest.identifier.encode("utf-8"))
        digest.update(str(self.optimizer_seed).encode("ascii"))
        digest.update(self.coefficients.tobytes())
        digest.update(self.logit_biases.tobytes())
        return digest.hexdigest()


def routing_model_digest() -> str:
    payload = {
        "identifier": P3_ROUTING_MODEL_ID,
        "routing_control_digest": routing_control_digest(),
        "layer_cardinalities": list(LAYER_CARDINALITIES),
        "source_feature_slices": [
            [feature_slice.start, feature_slice.stop]
            for feature_slice in SOURCE_FEATURE_SLICES
        ],
        "action_feature_slices": [
            [feature_slice.start, feature_slice.stop]
            for feature_slice in ACTION_FEATURE_SLICES
        ],
        "features_per_logit": list(FEATURES_PER_LOGIT),
        "trainable_parameter_count": TRANSITION_ACTIVE_PARAMETER_BUDGET,
        "optimizer": {
            "name": "adam",
            "learning_rate": LEARNING_RATE,
            "beta_1": ADAM_BETA_1,
            "beta_2": ADAM_BETA_2,
            "epsilon": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "minibatch_size": MINIBATCH_SIZE,
            "updates": TRAINING_UPDATES,
        },
        "prediction": "five_modular_delta_softmax_heads",
        "global_target_state_candidates": 0,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
