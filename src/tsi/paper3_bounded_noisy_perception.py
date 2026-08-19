"""Bounded noisy perception with a declared majority-recovery condition."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from .paper3_replication_family import ReplicationCase, ReplicationDataset
from .paper3_replication_factorized import evaluate, factorize


REPETITIONS = 3
MAX_CORRUPTED_REPETITIONS = 1


def _corrupt(value: int, modulus: int, offset: int) -> int:
    return (value + offset) % modulus


def noisy_views(case: ReplicationCase) -> tuple[ReplicationCase, ...]:
    views = []
    for repetition in range(REPETITIONS):
        source = tuple(
            _corrupt(value, 3 if index in (1, 3) else 4, repetition + 1)
            if repetition == 0
            else value
            for index, value in enumerate(case.source)
        )
        target = tuple(
            _corrupt(value, 3 if index in (1, 3) else 4, repetition + 1)
            if repetition == 0
            else value
            for index, value in enumerate(case.target)
        )
        views.append(replace(case, source=source, target=target))
    return tuple(views)


def majority_decode(views: tuple[ReplicationCase, ...]) -> ReplicationCase:
    if len(views) != REPETITIONS:
        raise ValueError("majority decoder requires the declared repetition count")
    source = tuple(
        Counter(view.source[index] for view in views).most_common(1)[0][0]
        for index in range(5)
    )
    target = tuple(
        Counter(view.target[index] for view in views).most_common(1)[0][0]
        for index in range(5)
    )
    return replace(views[0], source=source, target=target)


def denoise_dataset(dataset: ReplicationDataset) -> ReplicationDataset:
    partitions = {
        name: tuple(majority_decode(noisy_views(case)) for case in cases)
        for name, cases in dataset.partitions.items()
    }
    return replace(dataset, partitions=partitions)


def evaluate_bounded_noise(dataset: ReplicationDataset) -> dict[str, object]:
    noisy = denoise_dataset(dataset)
    signature = factorize(noisy)
    return {
        "repetitions": REPETITIONS,
        "max_corrupted_repetitions": MAX_CORRUPTED_REPETITIONS,
        "parameter": signature,
        "exact_accuracy": evaluate(noisy, signature)["exact_accuracy"],
    }
