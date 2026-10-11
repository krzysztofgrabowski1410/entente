"""Immutable game state: units and their placement (docs/adjudication.tex Sec. 1).

Occupancy is a **multiset of units per province** -- the representation is a
canonically sorted tuple, which (a) allows several units in one province on
maps whose capacity data permits it, and (b) makes equal states compare equal
regardless of construction order. The ``|occupants(p)| <= cap(p)`` invariant
is checked against a :class:`MapGraph` by :func:`check_state`, not assumed by
any code path (capacity 1 is *data* on classic maps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entente.engine.map.types import Location, UnitKind

if TYPE_CHECKING:
    from collections.abc import Iterable

    from entente.engine.map.graph import MapGraph


@dataclass(frozen=True, slots=True)
class Unit:
    """A unit on the board.

    ``strength`` is the unit's own base strength as written in its rulebook
    (Beleriand: 1, 2 or 3 -- "Each Unit has its own Strength"); how it enters
    the strength pipeline is ruleset policy (Sec. 2: the standard ruleset ignores
    it, ``base == 1``). Two units equal in all fields are fungible -- the
    engine may bind orders to either occurrence.
    """

    power: str
    kind: UnitKind
    location: Location
    strength: int = 1

    def __str__(self) -> str:
        prefix = str(self.strength) if self.strength != 1 else ""
        return f"{prefix}{self.kind.value} {self.location}"


def _unit_sort_key(unit: Unit) -> tuple[str, str, str, str, int]:
    """Canonical sort key of a unit: province, coast (``""`` if none), power, kind, strength."""
    return (
        unit.location.province,
        unit.location.coast or "",
        unit.power,
        unit.kind.value,
        unit.strength,
    )


@dataclass(frozen=True, slots=True)
class State:
    """An immutable position: map reference, phase tag, canonically sorted units."""

    map_name: str
    phase: str
    units: tuple[Unit, ...]

    @classmethod
    def of(cls, map_name: str, phase: str, units: Iterable[Unit]) -> State:
        """Build a state with canonical unit ordering (determinism: Sec. 6 Phase 0)."""
        return cls(map_name=map_name, phase=phase, units=tuple(sorted(units, key=_unit_sort_key)))

    def occupants(self, province: str) -> tuple[Unit, ...]:
        """Return the multiset of units in ``province``, in canonical order."""
        return tuple(u for u in self.units if u.location.province == province)


def check_state(graph: MapGraph, state: State) -> None:
    """Validate placement legality and the occupancy-capacity invariant.

    Raises ``ValueError`` on: a unit on terrain its kind cannot occupy (or on a
    missing/undeclared coast), or more occupants in a province than its
    capacity (Sec. 1). Called on adjudicator input and asserted on its output.
    """
    counts: dict[str, int] = {}
    for unit in state.units:
        if not graph.is_legal_location(unit.kind, unit.location):
            raise ValueError(f"illegal placement: {unit.power} {unit}")
        counts[unit.location.province] = counts.get(unit.location.province, 0) + 1
    for province, count in counts.items():
        capacity = graph.capacity(province)
        if capacity is not None and count > capacity:
            raise ValueError(f"occupancy {count} exceeds capacity {capacity} in {province}")
