"""Jointly learned soft edge-gate routing model for P3-5A-v2."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from .paper3_independence_contract import WorldFamily
from .paper3_learned_v2_generator import V2TransitionCase
from .paper3_multiworld import LAYER_ORDER, MultiworldStateCode
from .paper3_routing_controls import (
    TRAINING_MINIBATCH_SIZE,
    TRAINING_UPDATES,
    routing_control_manifests,
)
from .paper3_routing_model import (
    ACTION_FEATURE_SLICES,
    ADAM_BETA_1,
    ADAM_BETA_2,
    ADAM_EPSILON,
    FULL_INPUT_WIDTH,
    LEARNING_RATE,
    LOGIT_OFFSETS,
    LOGIT_TO_LAYER,
    MAX_FEATURES_PER_LOGIT,
    OUTPUT_LOGIT_COUNT,
    SOURCE_FEATURE_SLICES,
    WEIGHT_DECAY,
    MaskedRandomFeatureBasis,
    encode_cases,
)


P3_LEARNED_V2_MODEL_ID = "P3-5A-V2-JOINT-SOFT-EDGE-GATE-v1"
SPARSITY_PENALTY = 0.001
INITIAL_GATE_LOGIT = -1.0
EDGE_COUNT = len(LAYER_ORDER) * 2


def _seed_from(*parts: object) -> int:
    return int.from_bytes(
        sha256(":".join(str(part) for part in parts).encode("utf-8")).digest()[:8],
        "little",
    )


def _dense_manifest(family: WorldFamily):
    return next(
        manifest
        for manifest in routing_control_manifests(family)
        if manifest.identifier == "dense_active_matched"
    )


def _edge_coordinate_index() -> np.ndarray:
    result = np.zeros((len(LAYER_ORDER), FULL_INPUT_WIDTH), dtype=np.int64)
    for target in range(len(LAYER_ORDER)):
        for source in range(len(LAYER_ORDER)):
            result[target, SOURCE_FEATURE_SLICES[source]] = source
            result[target, ACTION_FEATURE_SLICES[source]] = len(LAYER_ORDER) + source
    return result


EDGE_COORDINATE_INDEX = _edge_coordinate_index()


def _encode_model_cases(cases: Sequence[V2TransitionCase]) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("model encoding requires cases")
    from .paper3_learned_v2_observation import (
        ObservedTransitionCase,
        PixelTransitionCase,
        encode_observed_cases,
        encode_pixel_cases,
    )

    if isinstance(cases[0], ObservedTransitionCase):
        return encode_observed_cases(cases)  # type: ignore[arg-type]
    if isinstance(cases[0], PixelTransitionCase):
        return encode_pixel_cases(cases)  # type: ignore[arg-type]
    return encode_cases(cases)


@dataclass(frozen=True)
class JointGateTrainingTrace:
    initial_nll: float
    final_nll: float
    final_sparsity_penalty: float
    update_count: int
    minibatch_size: int
    finite: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "initial_nll": self.initial_nll,
            "final_nll": self.final_nll,
            "final_sparsity_penalty": self.final_sparsity_penalty,
            "update_count": self.update_count,
            "minibatch_size": self.minibatch_size,
            "finite": self.finite,
        }


@dataclass(frozen=True)
class GateThresholdSelection:
    selected_threshold: float
    selected_model: "JointGateRoutingModel"
    candidate_scores: tuple[tuple[float, float, int], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_threshold": self.selected_threshold,
            "candidate_scores": [
                {
                    "threshold": threshold,
                    "selection_nll": score,
                    "edge_count": edge_count,
                }
                for threshold, score, edge_count in self.candidate_scores
            ],
        }


class JointGateRoutingModel:
    """A dense candidate model with differentiable source/action edge gates."""

    def __init__(self, family: WorldFamily, optimizer_seed: int) -> None:
        if type(optimizer_seed) is not int or optimizer_seed < 0:
            raise ValueError("optimizer_seed must be a nonnegative integer")
        self.family = family
        self.optimizer_seed = optimizer_seed
        basis_seed = _seed_from(P3_LEARNED_V2_MODEL_ID, family.value, "fixed_basis")
        self.basis = MaskedRandomFeatureBasis(_dense_manifest(family), basis_seed)
        self.coefficients = np.zeros(
            (OUTPUT_LOGIT_COUNT, MAX_FEATURES_PER_LOGIT), dtype=np.float64
        )
        self.logit_biases = np.zeros(OUTPUT_LOGIT_COUNT, dtype=np.float64)
        self.gate_logits = np.full(
            (len(LAYER_ORDER), EDGE_COUNT), INITIAL_GATE_LOGIT, dtype=np.float64
        )
        self.trace: JointGateTrainingTrace | None = None

    @property
    def parameter_count(self) -> int:
        return int(np.sum(self.basis.valid_features)) + len(self.logit_biases) + self.gate_logits.size

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / np.sum(exponentials, axis=1, keepdims=True)

    def gate_values(self) -> np.ndarray:
        return self._sigmoid(self.gate_logits)

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != FULL_INPUT_WIDTH:
            raise ValueError("joint gate inputs must have shape (n, 31)")
        gates = self.gate_values()
        target_gates = gates[np.arange(len(LAYER_ORDER))[:, np.newaxis], EDGE_COORDINATE_INDEX]
        logit_gates = target_gates[np.asarray(LOGIT_TO_LAYER)]
        gated_inputs = values[:, np.newaxis, :] * logit_gates[np.newaxis, :, :]
        features = np.tanh(
            np.einsum(
                "nci,cki->nck",
                gated_inputs,
                self.basis.weights,
                optimize=True,
            )
            + self.basis.biases[np.newaxis, :, :]
        )
        features[:, ~self.basis.valid_features] = 0.0
        return features

    def _logits(self, features: np.ndarray) -> np.ndarray:
        return (
            np.einsum("nck,ck->nc", features, self.coefficients, optimize=True)
            + self.logit_biases[np.newaxis, :]
        )

    def _nll(self, features: np.ndarray, deltas: np.ndarray) -> float:
        logits = self._logits(features)
        rows = np.arange(len(features))
        losses: list[np.ndarray] = []
        for layer, cardinality in enumerate((3, 3, 3, 4, 3)):
            offset = LOGIT_OFFSETS[layer]
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

    def _loss(self, features: np.ndarray, deltas: np.ndarray) -> float:
        return self._nll(features, deltas) + SPARSITY_PENALTY * float(np.sum(self.gate_values()))

    def fit(
        self,
        cases: Sequence[V2TransitionCase],
        *,
        updates: int = TRAINING_UPDATES,
        fixed_gate_mask: np.ndarray | None = None,
    ) -> JointGateTrainingTrace:
        if not cases:
            raise ValueError("joint gate training requires cases")
        if type(updates) is not int or updates <= 0:
            raise ValueError("updates must be positive")
        if fixed_gate_mask is not None:
            fixed_gate_mask = np.asarray(fixed_gate_mask, dtype=bool)
            if fixed_gate_mask.shape != self.gate_logits.shape:
                raise ValueError("fixed_gate_mask has an invalid shape")
            self.gate_logits[fixed_gate_mask] = 20.0
            self.gate_logits[~fixed_gate_mask] = -20.0
        inputs, deltas = _encode_model_cases(cases)
        features = self._features(inputs)
        initial_nll = self._nll(features, deltas)
        coefficient_m = np.zeros_like(self.coefficients)
        coefficient_v = np.zeros_like(self.coefficients)
        bias_m = np.zeros_like(self.logit_biases)
        bias_v = np.zeros_like(self.logit_biases)
        gate_m = np.zeros_like(self.gate_logits)
        gate_v = np.zeros_like(self.gate_logits)
        rng = np.random.default_rng(_seed_from(P3_LEARNED_V2_MODEL_ID, self.optimizer_seed, "minibatches"))
        order = rng.permutation(len(inputs))
        cursor = 0
        for update in range(1, updates + 1):
            if cursor + TRAINING_MINIBATCH_SIZE > len(inputs):
                order = rng.permutation(len(inputs))
                cursor = 0
            indices = order[cursor : cursor + TRAINING_MINIBATCH_SIZE]
            cursor += TRAINING_MINIBATCH_SIZE
            batch_inputs = inputs[indices]
            batch_deltas = deltas[indices]
            batch_features = self._features(batch_inputs)
            logits = self._logits(batch_features)
            residuals = np.zeros_like(logits)
            rows = np.arange(len(indices))
            for layer, cardinality in enumerate((3, 3, 3, 4, 3)):
                offset = LOGIT_OFFSETS[layer]
                probabilities = self._softmax(logits[:, offset : offset + cardinality])
                probabilities[rows, batch_deltas[:, layer]] -= 1.0
                residuals[:, offset : offset + cardinality] = probabilities / len(indices)
            coefficient_gradient = np.einsum("nck,nc->ck", batch_features, residuals, optimize=True)
            coefficient_gradient += WEIGHT_DECAY * self.coefficients
            coefficient_gradient[~self.basis.valid_features] = 0.0
            bias_gradient = np.sum(residuals, axis=0)
            feature_gradient = residuals[:, :, np.newaxis] * self.coefficients[np.newaxis, :, :]
            preactivation_gradient = feature_gradient * (1.0 - batch_features**2)
            gated_input_gradient = np.einsum(
                "nck,cki->nci",
                preactivation_gradient,
                self.basis.weights,
                optimize=True,
            )
            gates = self.gate_values()
            gate_gradient = np.zeros_like(self.gate_logits)
            logit_layers = np.asarray(LOGIT_TO_LAYER)
            for target in range(len(LAYER_ORDER)):
                layer_logits = np.flatnonzero(logit_layers == target)
                layer_gradient = np.sum(gated_input_gradient[:, layer_logits, :], axis=1)
                for edge in range(EDGE_COUNT):
                    coordinates = EDGE_COORDINATE_INDEX[target] == edge
                    gate_gradient[target, edge] = np.sum(
                        layer_gradient[:, coordinates]
                        * batch_inputs[:, coordinates]
                        * gates[target, edge]
                        * (1.0 - gates[target, edge])
                    ) / len(indices)
            gate_gradient += SPARSITY_PENALTY * gates * (1.0 - gates)
            if fixed_gate_mask is not None:
                gate_gradient[fixed_gate_mask] = 0.0
                gate_gradient[~fixed_gate_mask] = 0.0
            coefficient_m = ADAM_BETA_1 * coefficient_m + (1.0 - ADAM_BETA_1) * coefficient_gradient
            coefficient_v = ADAM_BETA_2 * coefficient_v + (1.0 - ADAM_BETA_2) * coefficient_gradient**2
            bias_m = ADAM_BETA_1 * bias_m + (1.0 - ADAM_BETA_1) * bias_gradient
            bias_v = ADAM_BETA_2 * bias_v + (1.0 - ADAM_BETA_2) * bias_gradient**2
            gate_m = ADAM_BETA_1 * gate_m + (1.0 - ADAM_BETA_1) * gate_gradient
            gate_v = ADAM_BETA_2 * gate_v + (1.0 - ADAM_BETA_2) * gate_gradient**2
            coefficient_m_hat = coefficient_m / (1.0 - ADAM_BETA_1**update)
            coefficient_v_hat = coefficient_v / (1.0 - ADAM_BETA_2**update)
            bias_m_hat = bias_m / (1.0 - ADAM_BETA_1**update)
            bias_v_hat = bias_v / (1.0 - ADAM_BETA_2**update)
            gate_m_hat = gate_m / (1.0 - ADAM_BETA_1**update)
            gate_v_hat = gate_v / (1.0 - ADAM_BETA_2**update)
            self.coefficients -= LEARNING_RATE * coefficient_m_hat / (np.sqrt(coefficient_v_hat) + ADAM_EPSILON)
            self.logit_biases -= LEARNING_RATE * bias_m_hat / (np.sqrt(bias_v_hat) + ADAM_EPSILON)
            self.gate_logits -= LEARNING_RATE * gate_m_hat / (np.sqrt(gate_v_hat) + ADAM_EPSILON)
            if fixed_gate_mask is not None:
                self.gate_logits[fixed_gate_mask] = 20.0
                self.gate_logits[~fixed_gate_mask] = -20.0
            self.coefficients[~self.basis.valid_features] = 0.0
        final_features = self._features(inputs)
        final_nll = self._nll(final_features, deltas)
        final_sparsity = SPARSITY_PENALTY * float(np.sum(self.gate_values()))
        finite = bool(
            np.isfinite(initial_nll)
            and np.isfinite(final_nll)
            and np.isfinite(final_sparsity)
            and np.all(np.isfinite(self.coefficients))
            and np.all(np.isfinite(self.logit_biases))
            and np.all(np.isfinite(self.gate_logits))
        )
        self.trace = JointGateTrainingTrace(
            initial_nll=initial_nll,
            final_nll=final_nll,
            final_sparsity_penalty=final_sparsity,
            update_count=updates,
            minibatch_size=TRAINING_MINIBATCH_SIZE,
            finite=finite,
        )
        return self.trace

    def predict_codes(self, cases: Sequence[V2TransitionCase]) -> tuple[MultiworldStateCode, ...]:
        inputs, _deltas = encode_cases(cases)
        predictions = np.zeros((len(cases), len(LAYER_ORDER)), dtype=np.int64)
        logits = self._logits(self._features(inputs))
        for layer, cardinality in enumerate((3, 3, 3, 4, 3)):
            offset = LOGIT_OFFSETS[layer]
            predictions[:, layer] = np.argmax(logits[:, offset : offset + cardinality], axis=1)
        result: list[MultiworldStateCode] = []
        for case, row in zip(cases, predictions, strict=True):
            source = case.source_code.as_tuple()
            result.append(
                MultiworldStateCode(
                    *((source[layer] + int(row[layer])) % (4 if layer == 3 else 3) for layer in range(len(LAYER_ORDER)))
                )
            )
        return tuple(result)

    def inferred_edges(self, threshold: float = 0.5) -> tuple[tuple[tuple[str, str], ...], tuple[tuple[str, str], ...]]:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        gates = self.gate_values()
        source = tuple(
            (LAYER_ORDER[edge], LAYER_ORDER[target])
            for target in range(len(LAYER_ORDER))
            for edge in range(len(LAYER_ORDER))
            if gates[target, edge] >= threshold
        )
        action = tuple(
            (LAYER_ORDER[edge - len(LAYER_ORDER)], LAYER_ORDER[target])
            for target in range(len(LAYER_ORDER))
            for edge in range(len(LAYER_ORDER), EDGE_COUNT)
            if gates[target, edge] >= threshold
        )
        return source, action


    def edge_mask(self, threshold: float) -> np.ndarray:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie in (0, 1)")
        return self.gate_values() >= threshold

    def _frozen_refit(
        self,
        cases: Sequence[V2TransitionCase],
        edge_mask: np.ndarray,
        *,
        updates: int,
    ) -> "JointGateRoutingModel":
        refit = JointGateRoutingModel(self.family, self.optimizer_seed)
        refit.fit(cases, updates=updates, fixed_gate_mask=edge_mask)
        return refit

    def select_threshold(
        self,
        train_cases: Sequence[V2TransitionCase],
        selection_cases: Sequence[V2TransitionCase],
        *,
        thresholds: Sequence[float] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        updates: int = TRAINING_UPDATES,
        refit_updates: int = TRAINING_UPDATES,
    ) -> GateThresholdSelection:
        if not train_cases or not selection_cases:
            raise ValueError("threshold selection requires train and routing-selection cases")
        candidates = tuple(float(value) for value in thresholds)
        if not candidates or any(not 0.0 < value < 1.0 for value in candidates):
            raise ValueError("threshold candidates must lie in (0, 1)")
        if tuple(sorted(set(candidates))) != candidates:
            raise ValueError("threshold candidates must be strictly increasing")
        scores: list[tuple[float, float, int]] = []
        inputs, deltas = _encode_model_cases(selection_cases)
        for threshold in candidates:
            edge_mask = self.edge_mask(threshold)
            candidate = self._frozen_refit(train_cases, edge_mask, updates=updates)
            score = candidate._nll(candidate._features(inputs), deltas)
            scores.append((threshold, score, int(np.sum(edge_mask))))
        selected_threshold, _score, _edge_count = min(
            scores,
            key=lambda row: (row[1], row[2], row[0]),
        )
        selected_mask = self.edge_mask(selected_threshold)
        selected_model = self._frozen_refit(
            tuple(train_cases) + tuple(selection_cases),
            selected_mask,
            updates=refit_updates,
        )
        return GateThresholdSelection(
            selected_threshold=selected_threshold,
            selected_model=selected_model,
            candidate_scores=tuple(scores),
        )
