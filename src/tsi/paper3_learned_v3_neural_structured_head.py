"""Neural structured head using explicit transition sufficient statistics."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from .paper3_learned_v2_mechanism import identify_observable_mechanism
from .paper3_learned_v3_generator import V3WorldDataset, V3TransitionCase
from .paper3_learned_v3_pooled_model import (
    CARDINALITIES,
    HIDDEN_WIDTH,
    OUTPUT_WIDTH,
)
from .paper3_routing_model import LOGIT_OFFSETS


INPUT_WIDTH = 77
BATCH_SIZE = 256
LEARNING_RATE = 0.01


def _block_softmax(logits: np.ndarray) -> np.ndarray:
    probabilities = np.zeros_like(logits)
    for layer, cardinality in enumerate(CARDINALITIES):
        offset = LOGIT_OFFSETS[layer]
        block = logits[:, offset : offset + cardinality]
        shifted = block - np.max(block, axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        probabilities[:, offset : offset + cardinality] = exponentiated / np.sum(
            exponentiated, axis=1, keepdims=True
        )
    return probabilities


def _seed_from(*parts: object) -> int:
    return int.from_bytes(
        sha256(":".join(map(str, parts)).encode()).digest()[:8], "little"
    )


def _signature_values(signature) -> np.ndarray:
    names = (
        "bridge_topology_to_relation",
        "context_order_to_metric",
        "independent_relation",
        "wrong_direction_negative_control",
    )
    values = np.zeros(11, dtype=np.float64)
    values[names.index(signature.graph_variant)] = 1.0
    values[4:9] = np.asarray(signature.layer_multipliers, dtype=np.float64) / 3.0
    values[9] = signature.bridge_coefficient / 3.0
    values[10] = signature.context_coefficient / 2.0
    return values


def encode_structured_cases(
    cases: Sequence[V3TransitionCase], signature, *, shuffle_signature: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    if not cases:
        raise ValueError("structured encoding requires cases")
    rows = []
    deltas = []
    sig = _signature_values(signature)
    if shuffle_signature:
        sig = np.roll(sig, 1)
    for case in cases:
        source = np.asarray(case.source_code.as_tuple(), dtype=np.float64) / np.asarray(
            (2, 2, 2, 3, 2), dtype=np.float64
        )
        action = np.asarray(case.action.components, dtype=np.float64) / 3.0
        one_hot = np.zeros(31, dtype=np.float64)
        for index, value in enumerate(case.source_code.as_tuple()):
            offset = sum(4 if i == 3 else 3 for i in range(index))
            one_hot[offset + value] = 1.0
        for index, value in enumerate(case.action.components):
            one_hot[16 + index * 3 + value] = 1.0
        interaction = np.outer(source, action).reshape(-1)
        rows.append(np.concatenate((one_hot, source, action, interaction, sig)))
        deltas.append(
            tuple(
                (target - value) % (4 if index == 3 else 3)
                for index, (value, target) in enumerate(
                    zip(
                        case.source_code.as_tuple(),
                        case.target_code.as_tuple(),
                        strict=True,
                    )
                )
            )
        )
    return np.asarray(rows), np.asarray(deltas, dtype=np.int64)


@dataclass(frozen=True)
class NeuralStructuredTrace:
    initial_loss: float
    final_loss: float
    updates: int
    training_case_count: int
    finite: bool


class NeuralStructuredTransitionHead:
    def __init__(
        self,
        *,
        seed: int = 0,
        use_signature: bool = True,
        shuffle_signature: bool = False,
    ) -> None:
        rng = np.random.default_rng(_seed_from("P3-5A-v3-neural-structured", seed))
        self.seed = seed
        self.use_signature = use_signature
        self.shuffle_signature = shuffle_signature
        self.hidden_weights = rng.normal(
            scale=1.0 / np.sqrt(INPUT_WIDTH), size=(INPUT_WIDTH, HIDDEN_WIDTH)
        )
        self.hidden_bias = rng.normal(scale=0.1, size=HIDDEN_WIDTH)
        self.output_weights = np.zeros((HIDDEN_WIDTH, OUTPUT_WIDTH))
        self.output_bias = np.zeros(OUTPUT_WIDTH)
        self.trace: NeuralStructuredTrace | None = None

    def _features(self, inputs: np.ndarray) -> np.ndarray:
        values = inputs.copy()
        if not self.use_signature:
            values[:, -11:] = 0.0
        return np.tanh(values @ self.hidden_weights + self.hidden_bias)

    def _logits(self, inputs: np.ndarray) -> np.ndarray:
        return self._features(inputs) @ self.output_weights + self.output_bias

    @staticmethod
    def _loss(logits, deltas) -> float:
        probabilities = _block_softmax(logits)
        rows = np.arange(len(deltas))
        losses = []
        for layer, cardinality in enumerate(CARDINALITIES):
            offset = LOGIT_OFFSETS[layer]
            losses.append(
                -np.log(
                    np.maximum(probabilities[rows, offset + deltas[:, layer]], 1e-12)
                )
            )
        return float(np.mean(np.column_stack(losses)))

    def fit(
        self, datasets: Sequence[V3WorldDataset], *, updates: int = 1000
    ) -> NeuralStructuredTrace:
        encoded = []
        for dataset in datasets:
            signature = identify_observable_mechanism(dataset.partitions["train"])
            encoded.append(
                encode_structured_cases(
                    dataset.partitions["train"],
                    signature,
                    shuffle_signature=self.shuffle_signature,
                )
            )
        inputs = np.concatenate([x[0] for x in encoded])
        deltas = np.concatenate([x[1] for x in encoded])
        rng = np.random.default_rng(_seed_from("batches", self.seed))
        order = rng.permutation(len(inputs))
        cursor = 0
        mw = np.zeros_like(self.output_weights)
        vw = np.zeros_like(self.output_weights)
        mb = np.zeros_like(self.output_bias)
        vb = np.zeros_like(self.output_bias)
        mh = np.zeros_like(self.hidden_weights)
        vh = np.zeros_like(self.hidden_weights)
        mhb = np.zeros_like(self.hidden_bias)
        vhb = np.zeros_like(self.hidden_bias)
        initial = self._loss(self._logits(inputs[:4096]), deltas[:4096])
        for update in range(1, updates + 1):
            if cursor + BATCH_SIZE > len(inputs):
                order = rng.permutation(len(inputs))
                cursor = 0
            idx = order[cursor : cursor + BATCH_SIZE]
            cursor += BATCH_SIZE
            batch = inputs[idx]
            target = deltas[idx]
            hidden = self._features(batch)
            logits = hidden @ self.output_weights + self.output_bias
            residual = _block_softmax(logits)
            rows = np.arange(len(idx))
            for layer, cardinality in enumerate(CARDINALITIES):
                residual[rows, LOGIT_OFFSETS[layer] + target[:, layer]] -= 1
            residual /= len(idx)
            gw = hidden.T @ residual
            gb = residual.sum(axis=0)
            gh = (residual @ self.output_weights.T) * (1 - hidden * hidden)
            g_h = batch.T @ gh
            g_hb = gh.sum(axis=0)
            mw = 0.9 * mw + 0.1 * gw
            vw = 0.999 * vw + 0.001 * gw * gw
            mb = 0.9 * mb + 0.1 * gb
            vb = 0.999 * vb + 0.001 * gb * gb
            mh = 0.9 * mh + 0.1 * g_h
            vh = 0.999 * vh + 0.001 * g_h * g_h
            mhb = 0.9 * mhb + 0.1 * g_hb
            vhb = 0.999 * vhb + 0.001 * g_hb * g_hb
            self.output_weights -= LEARNING_RATE * mw / (np.sqrt(vw) + 1e-8)
            self.output_bias -= LEARNING_RATE * mb / (np.sqrt(vb) + 1e-8)
            self.hidden_weights -= LEARNING_RATE * mh / (np.sqrt(vh) + 1e-8)
            self.hidden_bias -= LEARNING_RATE * mhb / (np.sqrt(vhb) + 1e-8)
        final = self._loss(self._logits(inputs[:4096]), deltas[:4096])
        finite = bool(np.isfinite(final) and np.all(np.isfinite(self.hidden_weights)))
        self.trace = NeuralStructuredTrace(initial, final, updates, len(inputs), finite)
        return self.trace

    def evaluate(
        self, dataset: V3WorldDataset, *, partition: str = "test"
    ) -> dict[str, object]:
        signature = identify_observable_mechanism(dataset.partitions["train"])
        inputs, deltas = encode_structured_cases(
            dataset.partitions[partition],
            signature,
            shuffle_signature=self.shuffle_signature,
        )
        logits = self._logits(inputs)
        probabilities = _block_softmax(logits)
        layer_accuracy = []
        for layer, cardinality in enumerate(CARDINALITIES):
            offset = LOGIT_OFFSETS[layer]
            layer_accuracy.append(
                float(
                    np.mean(
                        np.argmax(
                            probabilities[:, offset : offset + cardinality], axis=1
                        )
                        == deltas[:, layer]
                    )
                )
            )
        return {
            "graph_variant": dataset.graph.identifier,
            "mechanism_combination_index": dataset.mechanism_combination_index,
            "exact_accuracy": float(
                np.mean(
                    np.all(
                        np.column_stack(
                            [
                                np.argmax(
                                    probabilities[
                                        :, LOGIT_OFFSETS[i] : LOGIT_OFFSETS[i] + c
                                    ],
                                    axis=1,
                                )
                                == deltas[:, i]
                                for i, c in enumerate(CARDINALITIES)
                            ]
                        ),
                        axis=1,
                    )
                )
            ),
            "layer_accuracy": layer_accuracy,
            "finite": bool(np.all(np.isfinite(logits))),
        }
