"""Adjustment-phase adjudication -- docs/adjudication.tex Sec. 9, DATC 6.I/6.J.

Winter brings each power's unit count in line with its supply-center count.
A power owning more centers than units may build the difference at its *owned
home* supply centers (6.I.5/6.I.6) or decline builds (4.D.7 choice A); a
power owning fewer must remove the difference, any shortfall covered by the
civil-disorder rule. Adjustments are order-sheet business, not contests -- the
strength calculus plays no role.

The DATC choices this engine makes (the 4.D rows):

- **4.D.4/4.D.5/4.D.6 choice B** -- orders are processed top to bottom and the
  *first* legal orders consume the quota; later ones fail (6.I.1, 6.J.1).
  An illegal order consumes nothing (6.B.14: a failed build leaves the right
  available to a later order). A province accepts one build (6.I.7); a unit
  disbands once (6.J.2).
- **4.B.7** -- a fleet built in a split-coast province must name a declared
  coast (6.B.14); generally a build must name a location the unit kind could
  legally occupy (armies drop written coasts; a fleet on an inland province
  fails, 4.C.4 choice A / 6.I.2).
- **4.D.8, 2023 wording** -- civil-disorder removal (and the shortfall
  of an incomplete removal sheet, 6.J.2) takes the unit farthest from the
  power's nearest *owned* supply center, distance counted over absolute
  province adjacency -- the union of army and fleet adjacency projected to
  provinces, identical for both unit kinds (6.J.8/6.J.11) -- with ties removing
  fleets before armies, then alphabetically by full province *name* as
  printed on the map (6.J.4/6.J.6/6.J.7). A power owning no centers removes
  everything (its quota); unreachable centers rank as infinitely far.
- Orders in the wrong direction (a build while removing, a disband while
  entitled to build, anything at quota 0) fail; unfilled build quota is an
  implicit waive; unfilled removal quota triggers the 4.D.8 rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import OrderResolution, Resolution
from entente.engine.map.types import Location, UnitKind
from entente.engine.orders import (
    Build,
    Disband,
    Waive,
    format_order,
    parse_adjustment_order,
)
from entente.engine.state import State, Unit, check_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import AnyOrder
    from entente.engine.rules.protocol import Ruleset


@dataclass(frozen=True, slots=True)
class AdjustmentResult:
    """Adjustment output.

    Per-order resolutions in sheet order (powers sorted), the post-winter
    state, the units placed and removed by order, and the removals the
    civil-disorder rule made automatically.
    """

    resolutions: tuple[OrderResolution, ...]
    new_state: State
    built: tuple[Unit, ...]
    disbanded: tuple[Unit, ...]
    civil_disorder: tuple[Unit, ...]

    def resolution_of(self, text: str) -> Resolution:
        """Look up a written adjustment order's outcome; ``text`` is normalized first."""
        canonical = format_order(parse_adjustment_order(text))
        for entry in self.resolutions:
            if entry.text == canonical:
                return entry.resolution
        raise KeyError(f"no resolution recorded for {canonical!r}")


def owned_centers(ownership: Mapping[str, str], power: str) -> frozenset[str]:
    """Return the supply-center provinces ``power`` currently owns."""
    return frozenset(center for center, owner in ownership.items() if owner == power)


def build_quota(graph: MapGraph, state: State, ownership: Mapping[str, str], power: str) -> int:
    """Return owned centers minus units: positive = builds allowed, negative = removals due.

    Centers are counted per ``supply_center_count`` -- variant maps may stack
    several in one province (Beleriand's Angband holds 7).
    """
    units = sum(1 for unit in state.units if unit.power == power)
    centers = sum(
        graph.province(center).supply_center_count for center in owned_centers(ownership, power)
    )
    return centers - units


def _build_location(graph: MapGraph, build: Build) -> Location | None:
    """Return the location a build would occupy, or None if the kind cannot stand there.

    4.B.7: declared-coast provinces need the coast named.
    """
    if build.location.province not in graph.provinces:
        return None
    location = build.location
    if build.unit_kind is UnitKind.ARMY:
        location = Location(location.province)
    return location if graph.is_legal_location(build.unit_kind, location) else None


def _has_build_room(graph: MapGraph, state: State, province: str) -> bool:
    """Whether a build could stand in ``province``, i.e. free capacity remains.

    The classic "home center must be vacant" is the capacity-1 reading;
    unbounded provinces always have room.
    """
    capacity = graph.capacity(province)
    return capacity is None or len(state.occupants(province)) < capacity


def legal_builds(
    graph: MapGraph, state: State, ownership: Mapping[str, str], power: str
) -> tuple[Build, ...]:
    """Return every individually legal build for ``power``.

    One per unit kind and coast at each *owned, non-full, home* supply center
    (Sec. 9).
    """
    if build_quota(graph, state, ownership, power) <= 0:
        return ()
    owned = owned_centers(ownership, power)
    options: list[Build] = []
    for province in sorted(graph.home_centers.get(power, frozenset())):
        if province not in owned or not _has_build_room(graph, state, province):
            continue
        for kind in (UnitKind.ARMY, UnitKind.FLEET):
            for location in graph.locations_for(kind, province):
                options.append(Build(kind, location))
    return tuple(options)


def _absolute_adjacency(graph: MapGraph) -> dict[str, set[str]]:
    """Return the province-level union of army and fleet adjacency.

    The 4.D.8 "absolute adjacent provinces" metric graph, identical for both
    kinds.
    """
    union: dict[str, set[str]] = {key: set() for key in graph.provinces}
    for origin, neighbors in graph.army_adjacency.items():
        union[origin].update(neighbors)
    for location, reachable in graph.fleet_adjacency.items():
        union[location.province].update(loc.province for loc in reachable)
    return union


def _distances_to_owned(graph: MapGraph, owned: frozenset[str]) -> dict[str, int]:
    """Return the BFS distance of every province to the nearest owned center.

    Over the absolute adjacency graph; unreachable provinces are absent.
    """
    adjacency = _absolute_adjacency(graph)
    distances = dict.fromkeys(owned, 0)
    frontier = sorted(owned)
    while frontier:
        next_frontier: list[str] = []
        for province in frontier:
            for neighbor in sorted(adjacency[province]):
                if neighbor not in distances:
                    distances[neighbor] = distances[province] + 1
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return distances


def civil_disorder_removals(
    graph: MapGraph,
    ownership: Mapping[str, str],
    units: Sequence[Unit],
    count: int,
) -> tuple[Unit, ...]:
    """Return the ``count`` units the 4.D.8 rule removes from ``units`` (one power's).

    Farthest from the nearest owned center first, fleets before armies on
    ties, then alphabetical by province name.
    """
    if count <= 0 or not units:
        return ()
    power = units[0].power
    distances = _distances_to_owned(graph, owned_centers(ownership, power))
    unreachable = len(graph.provinces) + 1  # exceeds any BFS distance

    def priority(unit: Unit) -> tuple[int, int, str]:
        distance = distances.get(unit.location.province, unreachable)
        kind_rank = 0 if unit.kind is UnitKind.FLEET else 1
        return (-distance, kind_rank, graph.province(unit.location.province).name)

    return tuple(sorted(units, key=priority)[:count])


def adjudicate_adjustments(
    ruleset: Ruleset,
    state: State,
    ownership: Mapping[str, str],
    sheets: Mapping[str, Sequence[AnyOrder]],
    *,
    graph: MapGraph,
) -> AdjustmentResult:
    """Adjudicate one winter adjustment phase (Sec. 9) as a pure function.

    Of the post-Autumn state, the center ownership, and the adjustment sheets.
    """
    del ruleset  # Sec. 9: standard adjustment reads map data only; variant build rules are L2
    powers = sorted({unit.power for unit in state.units} | set(sheets) | set(ownership.values()))

    resolutions: list[OrderResolution] = []
    built: list[Unit] = []
    disbanded: list[Unit] = []
    civil: list[Unit] = []

    removed_slots: set[int] = set()
    for power in powers:
        quota = build_quota(graph, state, ownership, power)
        sheet = sheets.get(power, ())
        remaining = abs(quota)
        used_provinces: set[str] = set()
        seen: set[str] = set()
        for order in sheet:
            text = format_order(order)
            if not isinstance(order, Waive):
                # Duplicate unit-referencing lines are one order (6.I.7/6.J.2,
                # first occurrence reported); waives are not idempotent -- each
                # declines one build -- so every waive line is processed.
                if text in seen:
                    continue
                seen.add(text)
            resolution = Resolution.FAILS
            if quota > 0:
                resolution = _resolve_entitled(
                    graph,
                    state,
                    ownership,
                    power,
                    order,
                    remaining,
                    used_provinces,
                    built,
                )
                if resolution is Resolution.SUCCEEDS:
                    remaining -= 1
            elif quota < 0:
                resolution = _resolve_removal(
                    state, power, order, remaining, removed_slots, disbanded
                )
                if resolution is Resolution.SUCCEEDS:
                    remaining -= 1
            resolutions.append(OrderResolution(power=power, text=text, resolution=resolution))
        if quota < 0 and remaining > 0:
            # Unit occurrences are removed by slot, not by value -- equal units
            # are fungible and a multiset state may hold duplicates (Sec. 1).
            leftovers = [
                (slot, u)
                for slot, u in enumerate(state.units)
                if u.power == power and slot not in removed_slots
            ]
            automatic = civil_disorder_removals(
                graph, ownership, [u for _, u in leftovers], remaining
            )
            pending = list(automatic)
            for slot, unit in leftovers:
                if unit in pending:
                    pending.remove(unit)
                    removed_slots.add(slot)
            civil.extend(automatic)

    surviving = [unit for slot, unit in enumerate(state.units) if slot not in removed_slots]
    surviving.extend(built)
    new_state = State.of(state.map_name, state.phase, surviving)
    check_state(graph, new_state)
    return AdjustmentResult(
        resolutions=tuple(resolutions),
        new_state=new_state,
        built=tuple(built),
        disbanded=tuple(disbanded),
        civil_disorder=tuple(civil),
    )


def _resolve_entitled(
    graph: MapGraph,
    state: State,
    ownership: Mapping[str, str],
    power: str,
    order: AnyOrder,
    remaining: int,
    used_provinces: set[str],
    built: list[Unit],
) -> Resolution:
    """Resolve one order of a build-entitled power.

    Builds and waives consume the quota top-to-bottom (4.D.4/4.D.5/4.D.7);
    anything else fails.
    """
    if isinstance(order, Waive):
        return Resolution.SUCCEEDS if remaining > 0 else Resolution.FAILS
    if not isinstance(order, Build) or remaining <= 0:
        return Resolution.FAILS
    location = _build_location(graph, order)
    if location is None:
        return Resolution.FAILS
    province = location.province
    if (
        province in used_provinces
        or province not in graph.home_centers.get(power, frozenset())
        or ownership.get(province) != power
        or state.occupants(province)
    ):
        return Resolution.FAILS
    used_provinces.add(province)
    built.append(Unit(power, order.unit_kind, location))
    return Resolution.SUCCEEDS


def _resolve_removal(
    state: State,
    power: str,
    order: AnyOrder,
    remaining: int,
    removed_slots: set[int],
    disbanded: list[Unit],
) -> Resolution:
    """Resolve one order of a power that must remove units.

    Disbands consume the quota top-to-bottom (4.D.6); anything else fails.
    Occurrences are consumed by slot so duplicate orders on a stacked province
    remove one unit each (Sec. 1 multiset occupancy).

    The written base strength is exact, an unwritten one meaning base strength 1
    (the rulebook reads a bare "A" as "1A"), so ``Disband A NOG`` never removes a
    lone 2A. There is no advisory province-only fallback here, and on the
    standard column every unit has base strength 1, so the base-strength test is
    invisible there.
    Written KIND is not read.
    """
    if not isinstance(order, Disband) or remaining <= 0:
        return Resolution.FAILS
    slot = next(
        (
            i
            for i, u in enumerate(state.units)
            if u.power == power
            and u.location.province == order.location.province
            and (order.strength or 1) == u.strength
            and i not in removed_slots
        ),
        None,
    )
    if slot is None:
        return Resolution.FAILS
    removed_slots.add(slot)
    disbanded.append(state.units[slot])
    return Resolution.SUCCEEDS
