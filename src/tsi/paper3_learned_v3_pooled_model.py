"""Pooled neural mechanism-conditioned learner for P3-5A-v3."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from .paper3_learned_v2_mechanism import identify_observable_mechanism
from .paper3_learned_v3_generator import V3TransitionCase, V3WorldDataset
from .paper3_multiworld import LAYER_ORDER
from .paper3_routing_model import LOGIT_OFFSETS


INPUT_WIDTH = 42  # 31 local state/action features plus 11 observable signature features.
HIDDEN_WIDTH = 128
OUTPUT_WIDTH = 16
CARDINALITIES = (3, 3, 3, 4, 3)
LEARNING_RATE = 0.01
BATCH_SIZE = 256


def _seed_from(*parts: object) -> int:
    return int.from_bytes(sha256(":".join(map(str, parts)).encode()).digest()[:8], "little")


def signature_features(signature) -> np.ndarray:
    graph_names = (
        "bridge_topology_to_relation",
        "context_order_to_metric",
        "independent_relation",
        "wrong_direction_negative_control",
    )
    result = np.zeros(11, dtype=np.float64)
    result[graph_names.index(signature.graph_variant)] = 1.0
    result[4:9] = np.asarray(signature.layer_multipliers, dtype=np.float64) / 3.0
    result[9] = signature.bridge_coefficient / 3.0
    result[10] = signature.context_coefficient / 2.0
    return result


def encode_v3_cases(cases: Sequence[V3TransitionCase], signature) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("v3 encoding requires cases")
    inputs = np.zeros((len(cases), INPUT_WIDTH), dtype=np.float64)
    deltas = np.zeros((len(cases), len(LAYER_ORDER)), dtype=np.int64)
    signature_row = signature_features(signature)
    for row, case in enumerate(cases):
        source = case.source_code.as_tuple()
        target = case.target_code.as_tuple()
        for layer, value in enumerate(source):
            width = 4 if layer == 3 else 3
            offset = sum((4 if index == 3 else 3) for index in range(layer))
            inputs[row, offset + value] = 1.0
            deltas[row, layer] = (target[layer] - value) % width
        for layer, value in enumerate(case.action.components):
            offset = 16 + layer * 3
            inputs[row, offset + value] = 1.0
        inputs[row, 31:] = signature_row
    return inputs, deltas


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


@dataclass(frozen=True)
class PooledTrainingTrace:
    initial_loss: float
    final_loss: float
    updates: int
    training_case_count: int
    world_count: int
    finite: bool


class PooledMechanismConditionedModel:
    """A pooled one-hidden-layer neural delta predictor with observable conditioning."""

    def __init__(self, *, seed: int = 0, use_signature: bool = True) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.seed = seed
        self.use_signature = use_signature
        rng = np.random.default_rng(_seed_from("P3-5A-v3-pooled", seed))
        self.hidden_weights = rng.normal(scale=1.0 / np.sqrt(INPUT_WIDTH), size=(INPUT_WIDTH, HIDDEN_WIDTH))
        self.hidden_bias = rng.normal(scale=0.1, size=HIDDEN_WIDTH)
        self.output_weights = np.zeros((HIDDEN_WIDTH, OUTPUT_WIDTH), dtype=np.float64)
        self.output_bias = np.zeros(OUTPUT_WIDTH, dtype=np.float64)
        self.trace: PooledTrainingTrace | None = None

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float64).copy()
        if values.ndim != 2 or values.shape[1] != INPUT_WIDTH:
            raise ValueError("v3 inputs must have shape (n, 42)")
        if not self.use_signature:
            values[:, 31:] = 0.0
        return np.tanh(values @ self.hidden_weights + self.hidden_bias)

    def _logits(self, inputs: np.ndarray) -> np.ndarray:
        return self._features(inputs) @ self.output_weights + self.output_bias

    @staticmethod
    def _loss(logits: np.ndarray, deltas: np.ndarray) -> float:
        probabilities = _softmax(logits)
        rows = np.arange(len(deltas))
        losses = []
        for layer, cardinality in enumerate(CARDINALITIES):
            offset = LOGIT_OFFSETS[layer]
            losses.append(-np.log(np.maximum(probabilities[rows, offset + deltas[:, layer]], 1.0e-12)))
        return float(np.mean(np.column_stack(losses)))

    def fit(
        self,
        datasets: Sequence[V3WorldDataset],
        *,
        updates: int = 1000,
    ) -> PooledTrainingTrace:
        if not datasets or any(not dataset.partitions["train"] for dataset in datasets):
            raise ValueError("pooled fitting requires nonempty training worlds")
        if type(updates) is not int or updates <= 0:
            raise ValueError("updates must be positive")
        encoded = []
        for dataset in datasets:
            signature = identify_observable_mechanism(dataset.partitions["train"])
            encoded.append(encode_v3_cases(dataset.partitions["train"], signature))
        inputs = np.concatenate([item[0] for item in encoded], axis=0)
        deltas = np.concatenate([item[1] for item in encoded], axis=0)
        rng = np.random.default_rng(_seed_from("P3-5A-v3-pooled-batches", self.seed))
        m_w = np.zeros_like(self.output_weights)
        v_w = np.zeros_like(self.output_weights)
        m_b = np.zeros_like(self.output_bias)
        v_b = np.zeros_like(self.output_bias)
        m_h = np.zeros_like(self.hidden_weights)
        v_h = np.zeros_like(self.hidden_weights)
        m_hb = np.zeros_like(self.hidden_bias)
        v_hb = np.zeros_like(self.hidden_bias)
        order = rng.permutation(len(inputs))
        initial = self._loss(self._logits(inputs[: min(4096, len(inputs))]), deltas[: min(4096, len(inputs))])
        cursor = 0
        for update in range(1, updates + 1):
            if cursor + BATCH_SIZE > len(inputs):
                order = rng.permutation(len(inputs))
                cursor = 0
            indices = order[cursor:cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            batch_inputs = inputs[indices]
            batch_deltas = deltas[indices]
            hidden = self._features(batch_inputs)
            logits = hidden @ self.output_weights + self.output_bias
            residual = _softmax(logits)
            rows = np.arange(len(indices))
            for layer, cardinality in enumerate(CARDINALITIES):
                residual[rows, LOGIT_OFFSETS[layer] + batch_deltas[:, layer]] -= 1.0
            residual /= len(indices)
            grad_w = hidden.T @ residual
            grad_b = residual.sum(axis=0)
            hidden_gradient = (residual @ self.output_weights.T) * (1.0 - hidden * hidden)
            grad_h = batch_inputs.T @ hidden_gradient
            grad_hb = hidden_gradient.sum(axis=0)
            m_w = 0.9 * m_w + 0.1 * grad_w
            v_w = 0.999 * v_w + 0.001 * grad_w * grad_w
            m_b = 0.9 * m_b + 0.1 * grad_b
            v_b = 0.999 * v_b + 0.001 * grad_b * grad_b
            m_h = 0.9 * m_h + 0.1 * grad_h
            v_h = 0.999 * v_h + 0.001 * grad_h * grad_h
            m_hb = 0.9 * m_hb + 0.1 * grad_hb
            v_hb = 0.999 * v_hb + 0.001 * grad_hb * grad_hb
            self.output_weights -= LEARNING_RATE * m_w / (np.sqrt(v_w) + 1.0e-8)
            self.output_bias -= LEARNING_RATE * m_b / (np.sqrt(v_b) + 1.0e-8)
            self.hidden_weights -= LEARNING_RATE * m_h / (np.sqrt(v_h) + 1.0e-8)
            self.hidden_bias -= LEARNING_RATE * m_hb / (np.sqrt(v_hb) + 1.0e-8)
        final = self._loss(self._logits(inputs[: min(4096, len(inputs))]), deltas[: min(4096, len(inputs))])
        finite = bool(np.isfinite(initial) and np.isfinite(final) and np.all(np.isfinite(self.output_weights)))
        self.trace = PooledTrainingTrace(initial, final, updates, len(inputs), len(datasets), finite)
        return self.trace

    def evaluate(self, dataset: V3WorldDataset, *, partition: str = "test") -> dict[str, object]:
        signature = identify_observable_mechanism(dataset.partitions["train"])
        cases = dataset.partitions[partition]
        inputs, deltas = encode_v3_cases(cases, signature)
        logits = self._logits(inputs)
        probabilities = _softmax(logits)
        correct_layers = []
        for layer, cardinality in enumerate(CARDINALITIES):
            offset = LOGIT_OFFSETS[layer]
            correct_layers.append(np.argmax(probabilities[:, offset:offset + cardinality], axis=1) == deltas[:, layer])
        exact = np.all(np.column_stack(correct_layers), axis=1)
        return {
            "world_index": dataset.world_index,
            "mechanism_combination_index": dataset.mechanism_combination_index,
            "graph_variant": dataset.graph.identifier,
            "partition": partition,
            "case_count": len(cases),
            "layer_accuracy": [float(np.mean(values)) for values in correct_layers],
            "exact_accuracy": float(np.mean(exact)),
            "finite": bool(np.all(np.isfinite(logits))),
        }
