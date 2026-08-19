"""Exact finite bridge identifiability and coherence-repair primitives.

The module has two deliberately separate domains:

* local channel repair reconciles one observed generator relation with the
  relation induced by another structural layer;
* a finite bridge observation code studies identifiability and robust repair
  over a declared finite family of structural hypotheses.

The implementation is an exhaustive theorem oracle, not a scalable decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import inf, isfinite
from types import MappingProxyType
from typing import Hashable, Iterable, Mapping, Sequence


Cell = Hashable
Bit = int
PartialBit = int | None
CellWeights = float | Mapping[Cell, float]
ProbeWeights = float | Mapping[str, float]


def _normalize_cells(cells: Sequence[Cell]) -> tuple[Cell, ...]:
    normalized = tuple(cells)
    if not normalized:
        raise ValueError("the bridge cell universe must be nonempty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("bridge cells must be unique")
    return normalized


def _normalize_binary(value: object, *, name: str) -> Bit:
    if value not in (0, 1, False, True):
        raise ValueError(f"{name} must contain only binary values")
    return int(value)


def _validate_relation(
    cells: tuple[Cell, ...],
    relation: Iterable[Cell],
    *,
    name: str,
) -> frozenset[Cell]:
    normalized = frozenset(relation)
    unknown = normalized - frozenset(cells)
    if unknown:
        raise ValueError(f"{name} contains cells outside the bridge universe")
    return normalized


def _positive_cell_weights(
    cells: tuple[Cell, ...],
    weights: CellWeights,
    *,
    name: str,
) -> tuple[float, ...]:
    if isinstance(weights, Mapping):
        if set(weights) != set(cells):
            raise ValueError(f"{name} must assign one weight to every bridge cell")
        values = tuple(float(weights[cell]) for cell in cells)
    else:
        values = tuple(float(weights) for _ in cells)
    if any(not isfinite(value) or value <= 0 for value in values):
        raise ValueError(f"{name} must be finite and strictly positive")
    return values


def relation_word(
    cells: Sequence[Cell],
    relation: Iterable[Cell],
) -> tuple[Bit, ...]:
    """Encode a finite relation as a binary word in a fixed cell order."""

    normalized_cells = _normalize_cells(cells)
    normalized_relation = _validate_relation(
        normalized_cells,
        relation,
        name="relation",
    )
    return tuple(int(cell in normalized_relation) for cell in normalized_cells)


def normalized_bridge_defect(
    cells: Sequence[Cell],
    relation: Iterable[Cell],
    induced_relation: Iterable[Cell],
) -> float:
    """Return the normalized symmetric-difference bridge defect."""

    normalized_cells = _normalize_cells(cells)
    observed = _validate_relation(
        normalized_cells,
        relation,
        name="relation",
    )
    induced = _validate_relation(
        normalized_cells,
        induced_relation,
        name="induced_relation",
    )
    return len(observed.symmetric_difference(induced)) / len(normalized_cells)


@dataclass(frozen=True)
class OneSidedBridgeRepair:
    """Unique repair when the inducing structural layer is frozen."""

    consensus: frozenset[Cell]
    relation_flips: frozenset[Cell]
    cost: float
    normalized_defect: float


def one_sided_relation_repair(
    cells: Sequence[Cell],
    relation: Iterable[Cell],
    induced_relation: Iterable[Cell],
    *,
    relation_weights: CellWeights = 1.0,
) -> OneSidedBridgeRepair:
    """Replace the mutable relation by the frozen induced relation."""

    normalized_cells = _normalize_cells(cells)
    observed = _validate_relation(
        normalized_cells,
        relation,
        name="relation",
    )
    induced = _validate_relation(
        normalized_cells,
        induced_relation,
        name="induced_relation",
    )
    weights = _positive_cell_weights(
        normalized_cells,
        relation_weights,
        name="relation_weights",
    )
    flips = observed.symmetric_difference(induced)
    cost = sum(
        weight
        for cell, weight in zip(normalized_cells, weights, strict=True)
        if cell in flips
    )
    return OneSidedBridgeRepair(
        consensus=induced,
        relation_flips=flips,
        cost=cost,
        normalized_defect=len(flips) / len(normalized_cells),
    )


@dataclass(frozen=True)
class JointBridgeRepair:
    """One minimum-cost coherent consensus of two binary bridge channels."""

    consensus: frozenset[Cell]
    relation_flips: frozenset[Cell]
    induced_flips: frozenset[Cell]
    cost: float


def joint_bridge_repairs(
    cells: Sequence[Cell],
    relation: Iterable[Cell],
    induced_relation: Iterable[Cell],
    *,
    relation_weights: CellWeights = 1.0,
    induced_weights: CellWeights = 1.0,
    max_repairs: int = 100_000,
) -> tuple[JointBridgeRepair, ...]:
    """Return every weighted-Hamming minimizer on the coherent diagonal.

    At a conflicting cell, the consensus keeps the channel whose mutation
    weight is larger. Equal weights leave two independent minimizing choices.
    """

    if max_repairs <= 0:
        raise ValueError("max_repairs must be positive")
    normalized_cells = _normalize_cells(cells)
    observed = _validate_relation(
        normalized_cells,
        relation,
        name="relation",
    )
    induced = _validate_relation(
        normalized_cells,
        induced_relation,
        name="induced_relation",
    )
    relation_costs = _positive_cell_weights(
        normalized_cells,
        relation_weights,
        name="relation_weights",
    )
    induced_costs = _positive_cell_weights(
        normalized_cells,
        induced_weights,
        name="induced_weights",
    )

    choices: list[tuple[Bit, ...]] = []
    tied_conflicts = 0
    for cell, relation_cost, induced_cost in zip(
        normalized_cells,
        relation_costs,
        induced_costs,
        strict=True,
    ):
        relation_bit = int(cell in observed)
        induced_bit = int(cell in induced)
        if relation_bit == induced_bit:
            choices.append((relation_bit,))
        elif relation_cost > induced_cost:
            choices.append((relation_bit,))
        elif induced_cost > relation_cost:
            choices.append((induced_bit,))
        else:
            choices.append((relation_bit, induced_bit))
            tied_conflicts += 1

    repair_count = 1 << tied_conflicts
    if repair_count > max_repairs:
        raise ValueError(
            "exact joint repair enumeration is restricted to small ambiguity; "
            f"found {repair_count} minimizers"
        )

    repairs: list[JointBridgeRepair] = []
    for word in product(*choices):
        consensus = frozenset(
            cell
            for cell, bit in zip(normalized_cells, word, strict=True)
            if bit
        )
        relation_flips = consensus.symmetric_difference(observed)
        induced_flips = consensus.symmetric_difference(induced)
        cost = sum(
            relation_cost
            for cell, relation_cost in zip(
                normalized_cells,
                relation_costs,
                strict=True,
            )
            if cell in relation_flips
        ) + sum(
            induced_cost
            for cell, induced_cost in zip(
                normalized_cells,
                induced_costs,
                strict=True,
            )
            if cell in induced_flips
        )
        repairs.append(
            JointBridgeRepair(
                consensus=consensus,
                relation_flips=relation_flips,
                induced_flips=induced_flips,
                cost=cost,
            )
        )
    repairs.sort(key=lambda repair: relation_word(normalized_cells, repair.consensus))
    return tuple(repairs)


@dataclass(frozen=True)
class BinaryObservation:
    """A binary probe word with ``None`` denoting an erasure."""

    values: tuple[PartialBit, ...]

    def __post_init__(self) -> None:
        normalized: list[PartialBit] = []
        for value in self.values:
            normalized.append(
                None
                if value is None
                else _normalize_binary(value, name="observation")
            )
        object.__setattr__(self, "values", tuple(normalized))

    @property
    def observed_indices(self) -> tuple[int, ...]:
        return tuple(
            index for index, value in enumerate(self.values) if value is not None
        )


@dataclass(frozen=True)
class CodeRepairResult:
    """Nearest finite bridge-code repair and its strict second-best margin."""

    cost: float
    candidates: tuple[str, ...]
    margin: float
    candidate_costs: tuple[tuple[str, float], ...]

    @property
    def is_unique(self) -> bool:
        return len(self.candidates) == 1

    @property
    def stability_radius(self) -> float:
        """Perturbation budgets strictly below this preserve a unique repair."""

        return self.margin / 2


@dataclass(frozen=True)
class FiniteBridgeCode:
    """Binary observations of a finite, anchored structural hypothesis class."""

    probes: tuple[str, ...]
    codewords: tuple[tuple[str, tuple[Bit, ...]], ...]

    def __post_init__(self) -> None:
        probes = tuple(self.probes)
        if not probes:
            raise ValueError("a bridge code needs at least one probe")
        if len(set(probes)) != len(probes):
            raise ValueError("probe names must be unique")
        if not self.codewords:
            raise ValueError("a bridge code needs at least one candidate")

        names: set[str] = set()
        normalized_rows: list[tuple[str, tuple[Bit, ...]]] = []
        for name, raw_word in self.codewords:
            if not isinstance(name, str) or not name:
                raise ValueError("candidate names must be nonempty strings")
            if name in names:
                raise ValueError("candidate names must be unique")
            names.add(name)
            if len(raw_word) != len(probes):
                raise ValueError("every codeword must have one bit per probe")
            word = tuple(
                _normalize_binary(value, name=f"codeword {name!r}")
                for value in raw_word
            )
            normalized_rows.append((name, word))

        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "codewords", tuple(normalized_rows))

    @property
    def mapping(self) -> Mapping[str, tuple[Bit, ...]]:
        return MappingProxyType(dict(self.codewords))

    def _probe_indices(
        self,
        selected: Iterable[str] | None,
    ) -> tuple[int, ...]:
        if selected is None:
            return tuple(range(len(self.probes)))
        names = tuple(selected)
        if len(set(names)) != len(names):
            raise ValueError("selected probes must be unique")
        index = {name: position for position, name in enumerate(self.probes)}
        unknown = set(names) - set(index)
        if unknown:
            raise ValueError(f"unknown probes: {sorted(unknown)!r}")
        return tuple(index[name] for name in names)

    def _probe_weights(
        self,
        indices: tuple[int, ...],
        weights: ProbeWeights,
    ) -> tuple[float, ...]:
        if isinstance(weights, Mapping):
            unknown = set(weights) - set(self.probes)
            if unknown:
                raise ValueError(f"weights reference unknown probes: {sorted(unknown)!r}")
            missing = {self.probes[index] for index in indices} - set(weights)
            if missing:
                raise ValueError(
                    f"weights are missing selected probes: {sorted(missing)!r}"
                )
            values = tuple(float(weights[self.probes[index]]) for index in indices)
        else:
            values = tuple(float(weights) for _ in indices)
        if any(not isfinite(value) or value <= 0 for value in values):
            raise ValueError("probe weights must be finite and strictly positive")
        return values

    def fibers(
        self,
        selected: Iterable[str] | None = None,
    ) -> tuple[tuple[tuple[Bit, ...], tuple[str, ...]], ...]:
        """Return candidate fibers of the restricted observation map."""

        indices = self._probe_indices(selected)
        groups: dict[tuple[Bit, ...], list[str]] = {}
        for name, word in self.codewords:
            signature = tuple(word[index] for index in indices)
            groups.setdefault(signature, []).append(name)
        return tuple(
            (signature, tuple(names))
            for signature, names in sorted(groups.items())
        )

    def is_identifiable(
        self,
        selected: Iterable[str] | None = None,
    ) -> bool:
        """Return whether the selected probes separate every candidate pair."""

        return all(len(names) == 1 for _, names in self.fibers(selected))

    def pairwise_difference_supports(
        self,
    ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Return the probe set separating each unordered candidate pair."""

        supports: list[tuple[str, str, tuple[str, ...]]] = []
        for (left_name, left), (right_name, right) in combinations(
            self.codewords,
            2,
        ):
            support = tuple(
                probe
                for probe, left_bit, right_bit in zip(
                    self.probes,
                    left,
                    right,
                    strict=True,
                )
                if left_bit != right_bit
            )
            supports.append((left_name, right_name, support))
        return tuple(supports)

    def minimum_identifying_probe_sets(self) -> tuple[tuple[str, ...], ...]:
        """Enumerate all identifying probe sets of minimum cardinality."""

        for size in range(len(self.probes) + 1):
            identifying = tuple(
                selected
                for selected in combinations(self.probes, size)
                if self.is_identifiable(selected)
            )
            if identifying:
                return identifying
        return ()

    @property
    def information_lower_bound(self) -> int:
        """Return ``ceil(log2(number of candidates))`` without rounding error."""

        return (len(self.codewords) - 1).bit_length()

    def minimum_distance(
        self,
        selected: Iterable[str] | None = None,
        *,
        weights: ProbeWeights = 1.0,
    ) -> float:
        """Return the minimum weighted Hamming distance between candidates."""

        indices = self._probe_indices(selected)
        selected_weights = self._probe_weights(indices, weights)
        if len(self.codewords) == 1:
            return inf
        distances = []
        for (_, left), (_, right) in combinations(self.codewords, 2):
            distances.append(
                sum(
                    weight
                    for index, weight in zip(
                        indices,
                        selected_weights,
                        strict=True,
                    )
                    if left[index] != right[index]
                )
            )
        return min(distances)

    def error_erasure_identifiable(
        self,
        max_errors: int,
        max_erasures: int,
        selected: Iterable[str] | None = None,
    ) -> bool:
        """Return the exact Hamming criterion ``d_min > 2t+s``."""

        if (
            not isinstance(max_errors, int)
            or isinstance(max_errors, bool)
            or max_errors < 0
        ):
            raise ValueError("max_errors must be a nonnegative integer")
        if (
            not isinstance(max_erasures, int)
            or isinstance(max_erasures, bool)
            or max_erasures < 0
        ):
            raise ValueError("max_erasures must be a nonnegative integer")
        return self.minimum_distance(selected) > 2 * max_errors + max_erasures

    def _normalize_observation(
        self,
        observation: BinaryObservation | Mapping[str, PartialBit],
    ) -> BinaryObservation:
        if isinstance(observation, BinaryObservation):
            if len(observation.values) != len(self.probes):
                raise ValueError("observation length does not match the bridge code")
            return observation
        unknown = set(observation) - set(self.probes)
        if unknown:
            raise ValueError(f"observation references unknown probes: {sorted(unknown)!r}")
        return BinaryObservation(
            tuple(observation.get(probe) for probe in self.probes)
        )

    def nearest_repair(
        self,
        observation: BinaryObservation | Mapping[str, PartialBit],
        *,
        weights: ProbeWeights = 1.0,
    ) -> CodeRepairResult:
        """Return all minimum weighted-Hamming structural hypotheses."""

        normalized = self._normalize_observation(observation)
        indices = normalized.observed_indices
        selected_weights = self._probe_weights(indices, weights)
        costs: list[tuple[str, float]] = []
        for name, word in self.codewords:
            cost = sum(
                weight
                for index, weight in zip(
                    indices,
                    selected_weights,
                    strict=True,
                )
                if word[index] != normalized.values[index]
            )
            costs.append((name, cost))
        best = min(cost for _, cost in costs)
        candidates = tuple(name for name, cost in costs if cost == best)
        if len(candidates) > 1:
            margin = 0.0
        else:
            second_best = min(
                (cost for _, cost in costs if cost > best),
                default=inf,
            )
            margin = second_best - best
        return CodeRepairResult(
            cost=best,
            candidates=candidates,
            margin=margin,
            candidate_costs=tuple(costs),
        )

    def observation_distance(
        self,
        left: BinaryObservation | Mapping[str, PartialBit],
        right: BinaryObservation | Mapping[str, PartialBit],
        *,
        weights: ProbeWeights = 1.0,
    ) -> float:
        """Return weighted Hamming perturbation for one fixed erasure pattern."""

        normalized_left = self._normalize_observation(left)
        normalized_right = self._normalize_observation(right)
        if normalized_left.observed_indices != normalized_right.observed_indices:
            raise ValueError("observations must have the same erasure pattern")
        indices = normalized_left.observed_indices
        selected_weights = self._probe_weights(indices, weights)
        return sum(
            weight
            for index, weight in zip(indices, selected_weights, strict=True)
            if normalized_left.values[index] != normalized_right.values[index]
        )


def threshold_profile(
    distance: float,
    thresholds: Sequence[float],
) -> tuple[Bit, ...]:
    """Encode one nonnegative distance by closed threshold predicates."""

    value = float(distance)
    if not isfinite(value) or value < 0:
        raise ValueError("distance must be finite and nonnegative")
    normalized_thresholds = tuple(float(threshold) for threshold in thresholds)
    if any(
        not isfinite(threshold) or threshold < 0
        for threshold in normalized_thresholds
    ):
        raise ValueError("thresholds must be finite and nonnegative")
    if len(set(normalized_thresholds)) != len(normalized_thresholds):
        raise ValueError("thresholds must be unique")
    return tuple(int(value <= threshold) for threshold in normalized_thresholds)


def thresholds_separate_distance_alphabet(
    distances: Sequence[float],
    thresholds: Sequence[float],
) -> bool:
    """Return whether threshold profiles identify every allowed distance."""

    normalized_distances = tuple(float(distance) for distance in distances)
    if not normalized_distances:
        raise ValueError("the distance alphabet must be nonempty")
    if any(
        not isfinite(distance) or distance < 0
        for distance in normalized_distances
    ):
        raise ValueError("distance alphabet values must be finite and nonnegative")
    if len(set(normalized_distances)) != len(normalized_distances):
        raise ValueError("distance alphabet values must be unique")
    profiles = {
        threshold_profile(distance, thresholds)
        for distance in normalized_distances
    }
    return len(profiles) == len(normalized_distances)
