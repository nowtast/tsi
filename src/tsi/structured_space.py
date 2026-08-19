"""Minimal interfaces for structured imagination spaces.

The first implementation goal is conceptual clarity rather than performance.
These classes provide a typed place to collect the objects, relations, and
transitions that later experiments will make concrete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence


ObjectId = Hashable
RelationId = Hashable


@dataclass(frozen=True)
class StructuralState:
    """A structured latent state with objects, relations, and metadata."""

    objects: frozenset[ObjectId]
    relations: Mapping[RelationId, tuple[ObjectId, ...]] = field(default_factory=dict)
    attributes: Mapping[ObjectId, Mapping[str, float]] = field(default_factory=dict)


@dataclass
class StructuralImaginationSpace:
    """Container for states and transitions in a structured latent space."""

    states: list[StructuralState] = field(default_factory=list)
    transitions: list[tuple[int, int, str]] = field(default_factory=list)

    def add_state(self, state: StructuralState) -> int:
        self.states.append(state)
        return len(self.states) - 1

    def add_transition(self, source: int, target: int, label: str) -> None:
        self._check_state_index(source)
        self._check_state_index(target)
        self.transitions.append((source, target, label))

    def neighbors(self, state_index: int) -> Sequence[int]:
        self._check_state_index(state_index)
        return tuple(target for source, target, _ in self.transitions if source == state_index)

    def _check_state_index(self, index: int) -> None:
        if index < 0 or index >= len(self.states):
            raise IndexError(f"state index out of range: {index}")
