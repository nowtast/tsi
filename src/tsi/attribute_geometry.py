"""Exact finite and statistical audits for TSI Extension 2B-X2.

The exact correspondence search is exponential.  The coupling routines
evaluate declared plans and construct perturbation witnesses; they do not
pretend to solve the general nonconvex fused optimization problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import inf, isclose, isfinite, log, sqrt
from typing import Callable, Hashable, Iterator, Mapping, Sequence

from .geometric import Correspondence, FiniteMetricState


Attribute = Hashable
AttributeDistance = Callable[[Attribute, Attribute], float]
Coupling = tuple[tuple[float, ...], ...]
_TOLERANCE = 1e-9


@dataclass(frozen=True)
class FiniteAttributedMetricState:
    """A finite metric state with one point in a common attribute space per entity."""

    entities: tuple[Hashable, ...]
    distances: tuple[tuple[float, ...], ...]
    labels: tuple[Hashable, ...]
    attributes: tuple[Attribute, ...]

    def __post_init__(self) -> None:
        metric = FiniteMetricState(self.entities, self.distances, self.labels)
        attributes = tuple(self.attributes)
        if len(attributes) != len(metric.entities):
            raise ValueError("attributes must have one entry per entity")
        for attribute in attributes:
            try:
                hash(attribute)
            except TypeError as error:
                raise ValueError("attributes must be hashable metric-space points") from error
        object.__setattr__(self, "entities", metric.entities)
        object.__setattr__(self, "distances", metric.distances)
        object.__setattr__(self, "labels", metric.labels)
        object.__setattr__(self, "attributes", attributes)

    @property
    def metric_state(self) -> FiniteMetricState:
        return FiniteMetricState(self.entities, self.distances, self.labels)

    @property
    def diameter(self) -> float:
        return max(max(row) for row in self.distances)


@dataclass(frozen=True)
class AttributeCorrespondenceAudit:
    """The three component distortions and their weighted maximum."""

    discrepancy: float
    metric_distortion: float
    label_distortion: float
    attribute_distortion: float
    correspondence: Correspondence


@dataclass(frozen=True)
class FiniteAttributedMetricMeasureState:
    """An attributed metric state carrying a probability vector on its full carrier."""

    state: FiniteAttributedMetricState
    mass: tuple[float, ...]

    def __post_init__(self) -> None:
        mass = validate_probability(self.mass)
        if len(mass) != len(self.state.entities):
            raise ValueError("probability size does not match the carrier")
        object.__setattr__(self, "mass", mass)

    @property
    def full_support(self) -> bool:
        return all(value > 0.0 for value in self.mass)


@dataclass(frozen=True)
class FusedCouplingAudit:
    """Decomposition of the fused coupling power objective."""

    power_value: float
    discrepancy: float
    structural_power: float
    label_power: float
    attribute_power: float
    coupling: Coupling


@dataclass(frozen=True)
class FusedSamplingBound:
    """A two-sample high-probability error bound on the fused power optimum."""

    confidence: float
    source_tv_radius: float
    target_tv_radius: float
    coupling_lipschitz_constant: float
    statistical_power_error: float


@dataclass(frozen=True)
class CouplingPerturbationAudit:
    """Executable witness for the deterministic coupling sensitivity lemma."""

    perturbed_coupling: Coupling
    source_tv: float
    target_tv: float
    coupling_tv: float
    objective_difference: float
    objective_bound: float


def _validate_positive_weight(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive")
    return normalized


def _attribute_distance_table(
    states: Sequence[FiniteAttributedMetricState],
    attribute_distance: AttributeDistance,
) -> Mapping[tuple[Attribute, Attribute], float]:
    values = tuple(dict.fromkeys(
        attribute
        for state in states
        for attribute in state.attributes
    ))
    table: dict[tuple[Attribute, Attribute], float] = {}
    for first in values:
        for second in values:
            distance = float(attribute_distance(first, second))
            if not isfinite(distance) or distance < 0.0:
                raise ValueError("attribute distances must be finite and nonnegative")
            table[first, second] = distance

    for first in values:
        for second in values:
            distance = table[first, second]
            if first == second and not isclose(distance, 0.0, abs_tol=_TOLERANCE):
                raise ValueError("the attribute metric must have a zero diagonal")
            if first != second and distance <= 0.0:
                raise ValueError("the attribute metric must separate distinct attributes")
            if not isclose(
                distance,
                table[second, first],
                rel_tol=_TOLERANCE,
                abs_tol=_TOLERANCE,
            ):
                raise ValueError("the attribute metric must be symmetric")
            for third in values:
                if (
                    table[first, third]
                    > distance + table[second, third] + _TOLERANCE
                ):
                    raise ValueError("the attribute metric violates the triangle inequality")
    return table


def validate_attribute_metric(
    states: Sequence[FiniteAttributedMetricState],
    attribute_distance: AttributeDistance,
) -> None:
    """Validate the metric axioms on the finite set of attributes used by states."""

    if not states:
        raise ValueError("at least one attributed state is required")
    _attribute_distance_table(states, attribute_distance)


def _nonempty_subsets(values: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(1, len(values) + 1)
        for subset in combinations(values, size)
    )


def all_correspondences(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    *,
    max_pairs: int = 20,
) -> Iterator[Correspondence]:
    """Enumerate every carrier-covering relation between two small finite states."""

    pair_count = len(left.entities) * len(right.entities)
    if pair_count > max_pairs:
        raise ValueError(
            "exact correspondence enumeration is restricted to small states; "
            f"found {pair_count} possible pairs"
        )
    right_indices = tuple(range(len(right.entities)))
    choices = _nonempty_subsets(right_indices)
    required_right = set(right_indices)
    for selected_by_left in product(choices, repeat=len(left.entities)):
        relation = frozenset(
            (i, j)
            for i, selected in enumerate(selected_by_left)
            for j in selected
        )
        if {j for _, j in relation} == required_right:
            yield relation


def _validate_correspondence(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    correspondence: Correspondence,
) -> None:
    valid_left = set(range(len(left.entities)))
    valid_right = set(range(len(right.entities)))
    if any(i not in valid_left or j not in valid_right for i, j in correspondence):
        raise ValueError("correspondence contains an out-of-range index")
    if {i for i, _ in correspondence} != valid_left:
        raise ValueError("correspondence does not cover the left carrier")
    if {j for _, j in correspondence} != valid_right:
        raise ValueError("correspondence does not cover the right carrier")


def attribute_correspondence_audit(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    correspondence: Correspondence,
    attribute_distance: AttributeDistance,
    *,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> AttributeCorrespondenceAudit:
    """Evaluate metric, discrete-label, and metric-attribute distortions."""

    label_scale = _validate_positive_weight(label_weight, "label_weight")
    attribute_scale = _validate_positive_weight(attribute_weight, "attribute_weight")
    _validate_correspondence(left, right, correspondence)
    attribute_table = _attribute_distance_table((left, right), attribute_distance)

    metric_distortion = max(
        abs(left.distances[i][k] - right.distances[j][ell])
        for i, j in correspondence
        for k, ell in correspondence
    )
    label_distortion = max(
        float(left.labels[i] != right.labels[j])
        for i, j in correspondence
    )
    attribute_distortion = max(
        attribute_table[left.attributes[i], right.attributes[j]]
        for i, j in correspondence
    )
    return AttributeCorrespondenceAudit(
        discrepancy=max(
            metric_distortion,
            label_scale * label_distortion,
            attribute_scale * attribute_distortion,
        ),
        metric_distortion=metric_distortion,
        label_distortion=label_distortion,
        attribute_distortion=attribute_distortion,
        correspondence=correspondence,
    )


def optimal_attribute_correspondence(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    attribute_distance: AttributeDistance,
    *,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
    max_pairs: int = 20,
) -> AttributeCorrespondenceAudit:
    """Return a minimizing finite correspondence and its exact audit."""

    best: AttributeCorrespondenceAudit | None = None
    for correspondence in all_correspondences(left, right, max_pairs=max_pairs):
        audit = attribute_correspondence_audit(
            left,
            right,
            correspondence,
            attribute_distance,
            label_weight=label_weight,
            attribute_weight=attribute_weight,
        )
        if best is None or audit.discrepancy < best.discrepancy:
            best = audit
    if best is None:
        raise RuntimeError("nonempty finite carriers must admit a correspondence")
    return best


def attribute_aware_discrepancy(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    attribute_distance: AttributeDistance,
    *,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
    max_pairs: int = 20,
) -> float:
    return optimal_attribute_correspondence(
        left,
        right,
        attribute_distance,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
        max_pairs=max_pairs,
    ).discrepancy


def compose_correspondences(
    first: Correspondence,
    second: Correspondence,
) -> Correspondence:
    """Compose relations represented by source-target index pairs."""

    return frozenset(
        (left, right)
        for left, middle in first
        for second_middle, right in second
        if middle == second_middle
    )


def find_attribute_preserving_isometry(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    attribute_distance: AttributeDistance,
) -> Mapping[Hashable, Hashable] | None:
    """Find a label-, metric-, and attribute-preserving bijection when one exists."""

    _attribute_distance_table((left, right), attribute_distance)
    size = len(left.entities)
    if size != len(right.entities):
        return None
    for candidate in permutations(range(size)):
        if any(left.labels[i] != right.labels[candidate[i]] for i in range(size)):
            continue
        if any(
            not isclose(
                attribute_distance(
                    left.attributes[i],
                    right.attributes[candidate[i]],
                ),
                0.0,
                abs_tol=_TOLERANCE,
            )
            for i in range(size)
        ):
            continue
        if all(
            isclose(
                left.distances[i][j],
                right.distances[candidate[i]][candidate[j]],
                rel_tol=_TOLERANCE,
                abs_tol=_TOLERANCE,
            )
            for i in range(size)
            for j in range(size)
        ):
            return {
                left.entities[i]: right.entities[candidate[i]]
                for i in range(size)
            }
    return None


def validate_probability(mass: Sequence[float]) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in mass)
    if not normalized:
        raise ValueError("a probability vector must be nonempty")
    if any(not isfinite(value) or value < 0.0 for value in normalized):
        raise ValueError("probability masses must be finite and nonnegative")
    if not isclose(sum(normalized), 1.0, abs_tol=_TOLERANCE):
        raise ValueError("probability masses must sum to one")
    return normalized


def validate_fused_coupling(
    left: FiniteAttributedMetricMeasureState,
    right: FiniteAttributedMetricMeasureState,
    coupling: Sequence[Sequence[float]],
) -> Coupling:
    """Validate an unconstrained coupling; labels enter the objective as a cost."""

    normalized = tuple(tuple(float(value) for value in row) for row in coupling)
    if len(normalized) != len(left.state.entities) or any(
        len(row) != len(right.state.entities) for row in normalized
    ):
        raise ValueError("coupling shape does not match the two carriers")
    if any(not isfinite(value) or value < 0.0 for row in normalized for value in row):
        raise ValueError("coupling entries must be finite and nonnegative")
    for i, expected in enumerate(left.mass):
        if not isclose(sum(normalized[i]), expected, abs_tol=_TOLERANCE):
            raise ValueError("coupling left marginal is incorrect")
    for j, expected in enumerate(right.mass):
        if not isclose(
            sum(normalized[i][j] for i in range(len(left.state.entities))),
            expected,
            abs_tol=_TOLERANCE,
        ):
            raise ValueError("coupling right marginal is incorrect")
    return normalized


def fused_coupling_audit(
    left: FiniteAttributedMetricMeasureState,
    right: FiniteAttributedMetricMeasureState,
    coupling: Sequence[Sequence[float]],
    attribute_distance: AttributeDistance,
    *,
    p: float = 1.0,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> FusedCouplingAudit:
    """Evaluate the finite fused Gromov-Wasserstein-type power objective."""

    exponent = float(p)
    if not isfinite(exponent) or exponent < 1.0:
        raise ValueError("p must be finite and at least one")
    label_scale = _validate_positive_weight(label_weight, "label_weight")
    attribute_scale = _validate_positive_weight(attribute_weight, "attribute_weight")
    normalized = validate_fused_coupling(left, right, coupling)
    attribute_table = _attribute_distance_table(
        (left.state, right.state),
        attribute_distance,
    )
    left_size = len(left.state.entities)
    right_size = len(right.state.entities)

    structural_power = sum(
        abs(
            left.state.distances[i][k]
            - right.state.distances[j][ell]
        ) ** exponent
        * normalized[i][j]
        * normalized[k][ell]
        for i in range(left_size)
        for j in range(right_size)
        for k in range(left_size)
        for ell in range(right_size)
    )
    label_power = label_scale ** exponent * sum(
        float(left.state.labels[i] != right.state.labels[j])
        * normalized[i][j]
        for i in range(left_size)
        for j in range(right_size)
    )
    attribute_power = attribute_scale ** exponent * sum(
        attribute_table[
            left.state.attributes[i],
            right.state.attributes[j],
        ] ** exponent
        * normalized[i][j]
        for i in range(left_size)
        for j in range(right_size)
    )
    power_value = structural_power + label_power + attribute_power
    return FusedCouplingAudit(
        power_value=power_value,
        discrepancy=power_value ** (1.0 / exponent),
        structural_power=structural_power,
        label_power=label_power,
        attribute_power=attribute_power,
        coupling=normalized,
    )


def zero_fused_coupling_isometry(
    left: FiniteAttributedMetricMeasureState,
    right: FiniteAttributedMetricMeasureState,
    coupling: Sequence[Sequence[float]],
    attribute_distance: AttributeDistance,
    *,
    p: float = 1.0,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> Mapping[Hashable, Hashable] | None:
    """Recover the exact measure-preserving attributed isometry at zero cost."""

    if not left.full_support or not right.full_support:
        raise ValueError("zero-cost exactness requires full support on both carriers")
    audit = fused_coupling_audit(
        left,
        right,
        coupling,
        attribute_distance,
        p=p,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
    )
    if not isclose(audit.power_value, 0.0, abs_tol=_TOLERANCE):
        return None

    support = {
        (i, j)
        for i, row in enumerate(audit.coupling)
        for j, value in enumerate(row)
        if value > 0.0
    }
    by_left = {
        i: tuple(j for source, j in support if source == i)
        for i in range(len(left.state.entities))
    }
    if any(len(targets) != 1 for targets in by_left.values()):
        return None
    index_mapping = {i: targets[0] for i, targets in by_left.items()}
    if set(index_mapping.values()) != set(range(len(right.state.entities))):
        return None

    for i, j in index_mapping.items():
        if left.state.labels[i] != right.state.labels[j]:
            return None
        if not isclose(
            attribute_distance(left.state.attributes[i], right.state.attributes[j]),
            0.0,
            abs_tol=_TOLERANCE,
        ):
            return None
        if not isclose(left.mass[i], right.mass[j], abs_tol=_TOLERANCE):
            return None
    if any(
        not isclose(
            left.state.distances[i][k],
            right.state.distances[index_mapping[i]][index_mapping[k]],
            rel_tol=_TOLERANCE,
            abs_tol=_TOLERANCE,
        )
        for i in index_mapping
        for k in index_mapping
    ):
        return None
    return {
        left.state.entities[i]: right.state.entities[j]
        for i, j in index_mapping.items()
    }


def total_variation(first: Sequence[float], second: Sequence[float]) -> float:
    first_probability = validate_probability(first)
    second_probability = validate_probability(second)
    if len(first_probability) != len(second_probability):
        raise ValueError("total variation requires a common finite carrier")
    return 0.5 * sum(
        abs(left - right)
        for left, right in zip(first_probability, second_probability)
    )


def empirical_mass(
    state: FiniteAttributedMetricState,
    samples: Sequence[Hashable],
) -> tuple[float, ...]:
    """Return empirical frequencies on the declared carrier, retaining zero cells."""

    observations = tuple(samples)
    if not observations:
        raise ValueError("at least one sample is required")
    indices = {entity: index for index, entity in enumerate(state.entities)}
    counts = [0] * len(state.entities)
    for observation in observations:
        if observation not in indices:
            raise ValueError("sample is outside the declared carrier")
        counts[indices[observation]] += 1
    size = len(observations)
    return tuple(count / size for count in counts)


def finite_support_tv_radius(
    support_size: int,
    sample_size: int,
    failure_probability: float,
) -> float:
    """Coordinatewise Hoeffding plus a union bound, capped by TV <= 1."""

    if support_size <= 0:
        raise ValueError("support_size must be positive")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    failure = float(failure_probability)
    if not isfinite(failure) or not 0.0 < failure < 1.0:
        raise ValueError("failure_probability must lie strictly between zero and one")
    radius = 0.5 * support_size * sqrt(
        log(2.0 * support_size / failure) / (2.0 * sample_size)
    )
    return min(1.0, radius)


def fused_cost_envelope(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    attribute_distance: AttributeDistance,
    *,
    p: float = 1.0,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> float:
    """Return ``2 D^p + w_L^p + w_A^p A^p`` for the sensitivity theorem."""

    exponent = float(p)
    if not isfinite(exponent) or exponent < 1.0:
        raise ValueError("p must be finite and at least one")
    label_scale = _validate_positive_weight(label_weight, "label_weight")
    attribute_scale = _validate_positive_weight(attribute_weight, "attribute_weight")
    table = _attribute_distance_table((left, right), attribute_distance)
    diameter = max(left.diameter, right.diameter)
    attribute_bound = max(
        table[left_attribute, right_attribute]
        for left_attribute in left.attributes
        for right_attribute in right.attributes
    )
    return (
        2.0 * diameter ** exponent
        + label_scale ** exponent
        + attribute_scale ** exponent * attribute_bound ** exponent
    )


def fused_sampling_bound(
    left: FiniteAttributedMetricState,
    right: FiniteAttributedMetricState,
    attribute_distance: AttributeDistance,
    source_sample_size: int,
    target_sample_size: int,
    failure_probability: float,
    *,
    p: float = 1.0,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> FusedSamplingBound:
    """Return the two-sample plug-in bound with a half-delta allocation."""

    failure = float(failure_probability)
    if not isfinite(failure) or not 0.0 < failure < 1.0:
        raise ValueError("failure_probability must lie strictly between zero and one")
    source_radius = finite_support_tv_radius(
        len(left.entities),
        source_sample_size,
        failure / 2.0,
    )
    target_radius = finite_support_tv_radius(
        len(right.entities),
        target_sample_size,
        failure / 2.0,
    )
    constant = fused_cost_envelope(
        left,
        right,
        attribute_distance,
        p=p,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
    )
    return FusedSamplingBound(
        confidence=1.0 - failure,
        source_tv_radius=source_radius,
        target_tv_radius=target_radius,
        coupling_lipschitz_constant=constant,
        statistical_power_error=constant * (source_radius + target_radius),
    )


def _maximal_coupling_kernel(
    source: Sequence[float],
    target: Sequence[float],
) -> tuple[tuple[float, ...], ...]:
    """Build a Markov kernel whose induced self-carrier coupling is maximal."""

    source_probability = validate_probability(source)
    target_probability = validate_probability(target)
    if len(source_probability) != len(target_probability):
        raise ValueError("a relabeling kernel requires a common carrier")
    size = len(source_probability)
    joint = [[0.0 for _ in range(size)] for _ in range(size)]
    source_residual = list(source_probability)
    target_residual = list(target_probability)
    for index in range(size):
        diagonal = min(source_residual[index], target_residual[index])
        joint[index][index] = diagonal
        source_residual[index] -= diagonal
        target_residual[index] -= diagonal

    source_index = 0
    target_index = 0
    while source_index < size and target_index < size:
        while source_index < size and source_residual[source_index] <= _TOLERANCE:
            source_index += 1
        while target_index < size and target_residual[target_index] <= _TOLERANCE:
            target_index += 1
        if source_index == size or target_index == size:
            break
        moved = min(source_residual[source_index], target_residual[target_index])
        joint[source_index][target_index] += moved
        source_residual[source_index] -= moved
        target_residual[target_index] -= moved

    kernel: list[tuple[float, ...]] = []
    for index, mass in enumerate(source_probability):
        if mass > 0.0:
            kernel.append(tuple(value / mass for value in joint[index]))
        else:
            kernel.append(tuple(target_probability))
    return tuple(kernel)


def perturb_coupling(
    left: FiniteAttributedMetricMeasureState,
    right: FiniteAttributedMetricMeasureState,
    coupling: Sequence[Sequence[float]],
    new_left_mass: Sequence[float],
    new_right_mass: Sequence[float],
) -> Coupling:
    """Push a coupling through maximal same-carrier kernels to new marginals."""

    normalized = validate_fused_coupling(left, right, coupling)
    target_left = validate_probability(new_left_mass)
    target_right = validate_probability(new_right_mass)
    if len(target_left) != len(left.mass) or len(target_right) != len(right.mass):
        raise ValueError("new marginals must use the declared carriers")
    left_kernel = _maximal_coupling_kernel(left.mass, target_left)
    right_kernel = _maximal_coupling_kernel(right.mass, target_right)
    perturbed = tuple(
        tuple(
            sum(
                normalized[i][j]
                * left_kernel[i][new_i]
                * right_kernel[j][new_j]
                for i in range(len(left.mass))
                for j in range(len(right.mass))
            )
            for new_j in range(len(target_right))
        )
        for new_i in range(len(target_left))
    )
    new_left = FiniteAttributedMetricMeasureState(left.state, target_left)
    new_right = FiniteAttributedMetricMeasureState(right.state, target_right)
    return validate_fused_coupling(new_left, new_right, perturbed)


def coupling_total_variation(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    first_rows = tuple(tuple(float(value) for value in row) for row in first)
    second_rows = tuple(tuple(float(value) for value in row) for row in second)
    if len(first_rows) != len(second_rows) or any(
        len(first_row) != len(second_row)
        for first_row, second_row in zip(first_rows, second_rows)
    ):
        raise ValueError("coupling total variation requires equal matrix shapes")
    return 0.5 * sum(
        abs(first_value - second_value)
        for first_row, second_row in zip(first_rows, second_rows)
        for first_value, second_value in zip(first_row, second_row)
    )


def coupling_perturbation_audit(
    left: FiniteAttributedMetricMeasureState,
    right: FiniteAttributedMetricMeasureState,
    coupling: Sequence[Sequence[float]],
    new_left_mass: Sequence[float],
    new_right_mass: Sequence[float],
    attribute_distance: AttributeDistance,
    *,
    p: float = 1.0,
    label_weight: float = 1.0,
    attribute_weight: float = 1.0,
) -> CouplingPerturbationAudit:
    """Construct and numerically audit the deterministic sensitivity witness."""

    original = fused_coupling_audit(
        left,
        right,
        coupling,
        attribute_distance,
        p=p,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
    )
    perturbed = perturb_coupling(
        left,
        right,
        original.coupling,
        new_left_mass,
        new_right_mass,
    )
    target_left = FiniteAttributedMetricMeasureState(
        left.state,
        tuple(float(value) for value in new_left_mass),
    )
    target_right = FiniteAttributedMetricMeasureState(
        right.state,
        tuple(float(value) for value in new_right_mass),
    )
    changed = fused_coupling_audit(
        target_left,
        target_right,
        perturbed,
        attribute_distance,
        p=p,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
    )
    source_tv = total_variation(left.mass, target_left.mass)
    target_tv = total_variation(right.mass, target_right.mass)
    constant = fused_cost_envelope(
        left.state,
        right.state,
        attribute_distance,
        p=p,
        label_weight=label_weight,
        attribute_weight=attribute_weight,
    )
    return CouplingPerturbationAudit(
        perturbed_coupling=perturbed,
        source_tv=source_tv,
        target_tv=target_tv,
        coupling_tv=coupling_total_variation(original.coupling, perturbed),
        objective_difference=abs(original.power_value - changed.power_value),
        objective_bound=constant * (source_tv + target_tv),
    )


def combined_power_error_bound(
    statistical_power_error: float,
    optimization_suboptimality: float,
) -> float:
    """Combine value-estimation error and feasible-solver suboptimality."""

    statistical = float(statistical_power_error)
    optimization = float(optimization_suboptimality)
    if (
        not isfinite(statistical)
        or not isfinite(optimization)
        or statistical < 0.0
        or optimization < 0.0
    ):
        raise ValueError("error terms must be finite and nonnegative")
    return statistical + optimization

