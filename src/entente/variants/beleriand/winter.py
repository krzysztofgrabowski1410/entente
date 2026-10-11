"""Beleriand Layer-2 winter adjudication.

Replaces the standard adjustment routine (docs/adjudication.tex Sec. 9) for
Beleriand games: the game hooks dispatch here, and standard games keep
``adjudicator.builds`` unchanged. Everything the rulebook layers onto winter
lives here:

- the Morgoth-unit/Glaurung supply exemption (*"Morgoth and Glaurung Units
  don't need Supply Centres"*) -- they never count toward quota and are never
  removable;
- Eriador's off-board center counter feeding Men's quota and Eriador build
  eligibility (rulebook "Eriador"/"Migration");
- elven supply-delayed centers (entered only in Autumn; rulebook "Gaining
  Ownership of Supply Centres"): counted for maintenance, not for builds;
- build sites: Morgoth at Angband/Anfauglith plus four-year converted
  devastated homes, Men at any owned center (and Eriador while the counter
  is positive), everyone else at owned home centers; a province accepts as
  many builds per winter as its supply-center weight (Angband's seven),
  capacity permitting, so **duplicate build lines are individual orders
  here** (the multi-center province needs them); duplicate disband lines
  likewise remove one twin each, and a line repeated with no unit left for it
  is one order (DATC 6.J.2);
- fleet builds gated on *usable* fleet-building knowledge (the Naugrim hold
  but never use it; rulebook "Dwarves (Naugrim)") and coastal terrain;
- double builds (``Build 2A NAR BOUND EGL``) for elven powers except
  Ossiriand, consuming one free point of a double-eligible center (owned the
  whole preceding year, no foreign entry) -- exclusivity is per supply point,
  not per province, so a two-Centre province (Talath Dimen, Taur-im-Duinath,
  South Ossiriand) can carry two live doubles at once (rulebook "Provinces
  with two Supply Centres");
- civil-disorder removals over the non-special units only, under the
  rulebook's OWN 4.D.8 distance law rather than the standard one every other
  map uses: distance runs to the power's nearest OWNED Supply Centre and is
  measured per unit kind -- armies over any territory, fleets over sea and
  coastal lanes only -- so a power holding no coastal Centre loses its fleets
  first; ties break on territory name, with no blanket fleet priority; a
  defeated Morgoth's sheet is discarded outright (civil disorder -- rulebook
  "Morgoth Unit").
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, TypeVar

from entente.engine.adjudicator.builds import AdjustmentResult, owned_centers
from entente.engine.adjudicator.common import OrderResolution, Resolution
from entente.engine.map.types import Location, Terrain, UnitKind
from entente.engine.orders import Build, Disband, Waive, format_order
from entente.engine.state import State, Unit, check_state

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence

    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import AnyOrder
    from entente.variants.beleriand.layer2 import BeleriandHooks

_MORGOTH = "MORGOTH"
_MEN = "MEN"
_NAUGRIM = "NAUGRIM"
_OSSIRIAND = "OSSIRIAND"
_ERIADOR = "ERI"
_ELVISH = frozenset(
    {"DORIATH", "FEANORIANS", "FINGOLFIN", "FINROD", "GONDOLIN", "OSSIRIAND", "TELERI"}
)


def usable_fleet_knowledge(hooks: BeleriandHooks, power: str) -> bool:
    """Whether ``power`` may *use* the fleet-building knowledge.

    It must hold the knowledge and be neither Naugrim (who hold but never
    use it) nor Morgoth (who never holds it).
    """
    return power in hooks.fleet_knowledge and power not in (_NAUGRIM, _MORGOTH)


def center_weight(hooks: BeleriandHooks, power: str, ownership: Mapping[str, str]) -> int:
    """Owned supply weight: map centers plus Men's off-board Eriador counter."""
    weight = sum(
        hooks.graph.province(center).supply_center_count
        for center in owned_centers(ownership, power)
    )
    if power == _MEN:
        weight += hooks.eriador_centers
    return weight


def delayed_weight(hooks: BeleriandHooks, power: str, ownership: Mapping[str, str]) -> int:
    """Return the weight of ``power``'s supply-delayed centers.

    Elven gains entered only in Autumn: maintained, but not yet supplying a
    build.
    """
    return sum(
        hooks.graph.province(center).supply_center_count
        for center, until in hooks.delayed_supply.items()
        if ownership.get(center) == power and until > hooks.autumn_count
    )


def special_slots(hooks: BeleriandHooks, state: State) -> frozenset[int]:
    """Return the state-unit indexes occupied by the Morgoth unit and Glaurung.

    Both are exempt from quota, removal orders and civil disorder: the
    rulebook says they need no supply centres.
    """
    slots: set[int] = set()
    for entry in hooks.ledger.board():
        if entry.special is None:
            continue
        for index, unit in enumerate(state.units):
            if (
                index not in slots
                and unit.power == entry.power
                and unit.kind is entry.kind
                and unit.strength == entry.strength
                and unit.location == entry.location
            ):
                slots.add(index)
                break
    return frozenset(slots)


def unit_count(state: State, power: str, exempt: frozenset[int]) -> int:
    """Count ``power``'s units, the exempt (special) slots excluded."""
    return sum(
        1 for index, unit in enumerate(state.units) if unit.power == power and index not in exempt
    )


def gross_quota(
    hooks: BeleriandHooks, power: str, state: State, ownership: Mapping[str, str]
) -> int:
    """Return owned centers minus non-special units, the number ``Game.build_quota`` reports.

    Builds are additionally capped by the eligible weight, enforced during
    adjudication.
    """
    exempt = special_slots(hooks, state)
    return center_weight(hooks, power, ownership) - unit_count(state, power, exempt)


def _build_allowance(
    hooks: BeleriandHooks, power: str, state: State, ownership: Mapping[str, str]
) -> int:
    """Return the builds ``power`` may make: eligible weight minus non-special units."""
    exempt = special_slots(hooks, state)
    eligible = center_weight(hooks, power, ownership) - delayed_weight(hooks, power, ownership)
    return eligible - unit_count(state, power, exempt)


def _province_budget(hooks: BeleriandHooks, power: str, province: str) -> int:
    """Return the builds a province accepts per winter: its supply weight.

    Eriador's budget is the off-board counter, for Men only.
    """
    if province == _ERIADOR:
        return hooks.eriador_centers if power == _MEN else 0
    return hooks.graph.province(province).supply_center_count


def _build_sites(hooks: BeleriandHooks, power: str, ownership: Mapping[str, str]) -> set[str]:
    """Return the provinces ``power`` may build in this winter."""
    if power == _MORGOTH:
        homes = set(hooks.graph.home_centers.get(power, frozenset())) | hooks.converted_homes
        return {p for p in homes if ownership.get(p) == power}
    if power == _MEN:
        sites = set(owned_centers(ownership, power))
        if hooks.eriador_centers > 0:
            sites.add(_ERIADOR)
        return sites
    return {
        p for p in hooks.graph.home_centers.get(power, frozenset()) if ownership.get(p) == power
    }


def _bindable_centers(
    hooks: BeleriandHooks,
    power: str,
    ownership: Mapping[str, str],
    bindings_taken: Mapping[str, int] | None = None,
) -> list[str]:
    """Return the double-eligible centers ``power`` could still bind a new double to.

    Eligible at the last autumn boundary, still owned, with a remaining slot
    after this winter's own new bindings are subtracted.

    ``hooks.double_eligible[p]`` is already net of every living unit's point
    (the ``spent_supply`` that ``_reconcile_spent`` counts), so no ledger scan is
    needed here -- only ``bindings_taken`` (this winter's not-yet-committed
    binds; the ledger gains no entry until ``post_adjustment``) still needs
    subtracting. Exclusivity is PER SUPPLY POINT, not per province: a
    two-Centre province offers two slots, a weight-1 province one.
    """
    taken = bindings_taken or {}
    return sorted(
        p
        for p, points in hooks.double_eligible.items()
        if ownership.get(p) == power and points - taken.get(p, 0) > 0
    )


def _resolve_build(
    hooks: BeleriandHooks,
    power: str,
    order: Build,
    state: State,
    ownership: Mapping[str, str],
    remaining: int,
    builds_at: dict[str, int],
    bindings_taken: Counter[str],
    built: list[Unit],
) -> Resolution:
    """Resolve one Build order against the quota, the sites and the double rules.

    Every test is a refusal in order: quota spent, a special designator, an
    illegal location, a province the power may not build in, a strength
    other than 1 or 2, a fleet without usable knowledge, a double by a power
    that may not build one or without a bindable center, a binding on a
    plain unit, the province's per-winter budget, and its capacity. A
    success records the build and its binding.
    """
    if remaining <= 0:
        return Resolution.FAILS
    if order.special is not None:
        return Resolution.FAILS  # no special is buildable: (M)/(G) is invalid in winter
    location = order.location
    if order.unit_kind is UnitKind.ARMY:
        location = Location(location.province)
    province = location.province
    if province not in hooks.graph.provinces or not hooks.graph.is_legal_location(
        order.unit_kind, location
    ):
        return Resolution.FAILS
    if province not in _build_sites(hooks, power, ownership):
        return Resolution.FAILS
    strength = order.strength or 1
    if strength not in (1, 2):
        return Resolution.FAILS
    if order.unit_kind is UnitKind.FLEET and not usable_fleet_knowledge(hooks, power):
        return Resolution.FAILS
    bound_to: str | None = None
    if strength == 2:
        if power not in _ELVISH or power == _OSSIRIAND:
            return Resolution.FAILS
        options = _bindable_centers(hooks, power, ownership, bindings_taken)
        if order.bound_to is not None:
            if order.bound_to not in options:
                return Resolution.FAILS
            bound_to = order.bound_to
        elif options:
            bound_to = options[0]
        else:
            return Resolution.FAILS
    elif order.bound_to is not None:
        return Resolution.FAILS  # binding is double-build notation only
    if builds_at.get(province, 0) >= _province_budget(hooks, power, province):
        return Resolution.FAILS
    capacity = hooks.graph.capacity(province)
    if capacity is not None:
        occupied = len(state.occupants(province)) + builds_at.get(province, 0)
        if occupied >= capacity:
            return Resolution.FAILS
    builds_at[province] = builds_at.get(province, 0) + 1
    if bound_to is not None:
        bindings_taken[bound_to] += 1
    unit = Unit(power, order.unit_kind, location, strength)
    built.append(unit)
    hooks.stash_binding(unit, bound_to)
    return Resolution.SUCCEEDS


def _disband_target(
    power: str, order: Disband, state: State, removed: set[int], exempt: frozenset[int]
) -> int | None:
    """Return the first unremoved, non-special unit a Disband order describes, or None."""
    return next(
        (
            i
            for i, u in enumerate(state.units)
            if u.power == power
            and u.location.province == order.location.province
            # bare means base strength 1, never "any base strength" ("A is same as 1A"): with a
            # 2A standing alone, `Disband A X` removes nothing rather than
            # quietly removing the double the player did not name
            and (order.strength or 1) == u.strength
            and (order.unit_kind is u.kind)
            and i not in removed
            and i not in exempt
        ),
        None,
    )


def _resolve_disband(
    power: str,
    order: Disband,
    state: State,
    remaining: int,
    removed: set[int],
    exempt: frozenset[int],
    disbanded: list[Unit],
) -> Resolution:
    """Resolve one Disband order: the first matching, unremoved, non-special unit goes."""
    if remaining <= 0:
        return Resolution.FAILS
    if order.special is not None:
        return Resolution.FAILS  # specials are unremovable: (M)/(G) is invalid in winter
    slot = _disband_target(power, order, state, removed, exempt)
    if slot is None:
        return Resolution.FAILS
    removed.add(slot)
    disbanded.append(state.units[slot])
    return Resolution.SUCCEEDS


# -- the rulebook's own civil-disorder removals (rulebook 4.D.8) -----------------

_Node = TypeVar("_Node")


def _nearest_steps(
    adjacency: Mapping[_Node, Collection[_Node]], seeds: Collection[_Node]
) -> dict[_Node, int]:
    """Return the BFS layer index of every node reachable from ``seeds``.

    Unreachable nodes are absent. Layer numbers are invariant under neighbour
    iteration order, so the node type need not be orderable (fleet nodes are
    unordered locations).
    """
    distances: dict[_Node, int] = dict.fromkeys(seeds, 0)
    frontier = list(distances)
    while frontier:
        next_frontier: list[_Node] = []
        for node in frontier:
            for neighbor in adjacency.get(node, ()):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return distances


def _army_steps(graph: MapGraph, centers: Collection[str]) -> dict[str, int]:
    """Return each province's steps to the nearest owned Supply Centre for an ARMY.

    4.D.8 allows armies *"to Move over any territory"*, so the metric graph is
    the province projection of army adjacency united with fleet adjacency --
    seas included.
    """
    union: dict[str, set[str]] = {key: set() for key in graph.provinces}
    for origin, neighbors in graph.army_adjacency.items():
        union[origin].update(neighbors)
    for location, reachable in graph.fleet_adjacency.items():
        union[location.province].update(loc.province for loc in reachable)
    return _nearest_steps(union, centers)


def _fleet_steps(graph: MapGraph, centers: Collection[str]) -> dict[Location, int]:
    """Return each fleet location's steps to the nearest owned Supply Centre for a FLEET.

    Measured over the fleet lanes alone (4.D.8: fleets *"only Move over places
    where fleets may Move"*). An inland centre has no fleet location and so
    seeds nothing, which is what makes the owned seed bite: a power holding
    only inland centres has no coastal one to measure to, every one of its
    fleets is infinitely far, and they come off before any army.
    """
    seeds = [location for center in sorted(centers) for location in graph.fleet_locations(center)]
    return _nearest_steps(graph.fleet_adjacency, seeds)


def rulebook_civil_disorder_removals(
    graph: MapGraph, ownership: Mapping[str, str], units: Sequence[Unit], count: int
) -> tuple[Unit, ...]:
    """Return the ``count`` units Beleriand's own 4.D.8 removes from ``units``.

    ``units`` are one power's, the exempt specials already filtered out; the
    farthest from that power's nearest OWNED Supply Centre go first, ties
    alphabetical by territory name.

    Beleriand keeps the rulebook's law, not the standard one that
    :func:`entente.engine.adjudicator.builds.civil_disorder_removals`
    implements for every other map (DATC 4.D.8, certified by DATC 6.J). Both
    count to an owned Centre; what differs is that distance here is measured
    per unit KIND rather than by one adjacency union for both, and that an
    equal-distance tie is settled by territory name alone, with no blanket
    fleet priority. The rulebook's "Paradoxes resolve" 4.D.8: *"in civil
    disorder Units will be removed by distance from the nearest owned Supply
    Centre, allowing armies to Move over any territory, and fleets to only
    Move over places where fleets may Move. If two Units are at equal
    distance alphabetical order of the territories is used."*

    Fleets do come off first in one case, and it falls out of the metric
    rather than being a separate clause: a fleet counts only sea and coastal
    lanes, so a power that owns no coastal Centre has nothing for its fleets
    to measure to -- every one of them is infinitely far and outranks any
    army, however far inland (*"If a Player owns no coastal Supply Centre,
    then no fleet of that Player can reach an owned Supply Centre, and all
    their fleets are removed before their armies"*). Such fleets are ranked
    AMONG THEMSELVES on the full graph -- the army metric, *"measured as if
    it could Move over any territory"* -- farthest first, so the ordering
    stays a distance rather than falling to the alphabet. Both are gated on
    the power owning NO coastal Centre: while one exists the ordinary
    sea-and-coast rule applies to every fleet, stranded or not.

    Distances read STATIC geometry only. The Girdle, devastation, the Gondolin
    shield and one-way entry bans never enter them: the metric is the
    idealized "may Move" reach of the sentence, not a legality check on this
    position.
    """
    if count <= 0 or not units:
        return ()
    centers = owned_centers(ownership, units[0].power)
    army_steps = _army_steps(graph, centers)
    fleet_steps = _fleet_steps(graph, centers)
    # Exceeds any BFS depth on EITHER metric graph: the army BFS spans at most
    # every province, the fleet BFS at most every fleet location (a map with
    # named coasts has more of those than provinces).
    unreachable = len(graph.provinces) + len(graph.fleet_adjacency) + 1

    # Whether the power owns a Centre a fleet could reach at all. This is a
    # POWER-level test, not a per-fleet one: while one owned coastal Centre
    # exists the ordinary sea-and-coast rule governs every fleet, including any
    # the lanes happen to strand.
    owns_coastal = any(graph.fleet_locations(center) for center in centers)

    def priority(unit: Unit) -> tuple[int, int, str]:
        name = graph.province(unit.location.province).name
        if unit.kind is UnitKind.ARMY:
            return (0, -army_steps.get(unit.location.province, unreachable), name)
        if not owns_coastal:
            # No owned coastal Centre: no fleet can reach one, so they all come
            # off ahead of the armies, ranked against each other on the full
            # graph rather than by the alphabet.
            return (-1, -army_steps.get(unit.location.province, unreachable), name)
        return (0, -fleet_steps.get(unit.location, unreachable), name)

    # The top ``count`` of ONE sorted list: the metric is static, so removing one
    # unit and recomputing would name the same units in the same order. Units
    # that tie on both keys -- an Area holds several -- keep their state order,
    # the sort being stable; equal units are fungible anyway (Sec. 1).
    return tuple(sorted(units, key=priority)[:count])


def adjudicate_winter(
    hooks: BeleriandHooks,
    state: State,
    ownership: Mapping[str, str],
    sheets: Mapping[str, Sequence[AnyOrder]],
) -> AdjustmentResult:
    """Adjudicate the full Beleriand winter.

    Same output contract as
    :func:`entente.engine.adjudicator.builds.adjudicate_adjustments`.
    """
    graph = hooks.graph
    exempt = special_slots(hooks, state)
    powers = sorted({unit.power for unit in state.units} | set(sheets) | set(ownership.values()))

    resolutions: list[OrderResolution] = []
    built: list[Unit] = []
    disbanded: list[Unit] = []
    civil: list[Unit] = []
    removed: set[int] = set()

    for power in powers:
        sheet: Sequence[AnyOrder] = sheets.get(power, ())
        if power == _MORGOTH and hooks.morgoth_defeated:
            for order in sheet:  # civil disorder discards a defeated Morgoth's sheet
                resolutions.append(OrderResolution(power, format_order(order), Resolution.FAILS))
            sheet = ()
        allowance = _build_allowance(hooks, power, state, ownership)
        quota = gross_quota(hooks, power, state, ownership)
        remaining = allowance if quota >= 0 else -quota
        builds_at: dict[str, int] = {}
        bindings_taken: Counter[str] = Counter()
        seen_disbands: set[str] = set()
        for order in sheet:
            text = format_order(order)
            if isinstance(order, Disband):
                # Twins written alike are removed one per line; a line repeated with
                # no unit left for it is the same order again and is passed over
                # (DATC 6.J.2), civil disorder taking what the quota still owes.
                if (
                    text in seen_disbands
                    and _disband_target(power, order, state, removed, exempt) is None
                ):
                    continue
                seen_disbands.add(text)
            resolution = Resolution.FAILS
            if isinstance(order, Waive):
                if quota >= 0 and remaining > 0:
                    resolution = Resolution.SUCCEEDS
                    remaining -= 1
            elif quota >= 0 and isinstance(order, Build):
                resolution = _resolve_build(
                    hooks,
                    power,
                    order,
                    state,
                    ownership,
                    remaining,
                    builds_at,
                    bindings_taken,
                    built,
                )
                if resolution is Resolution.SUCCEEDS:
                    remaining -= 1
            elif quota < 0 and isinstance(order, Disband):
                resolution = _resolve_disband(
                    power, order, state, remaining, removed, exempt, disbanded
                )
                if resolution is Resolution.SUCCEEDS:
                    remaining -= 1
            resolutions.append(OrderResolution(power=power, text=text, resolution=resolution))
        if quota < 0 and remaining > 0:
            leftovers = [
                (slot, u)
                for slot, u in enumerate(state.units)
                if u.power == power and slot not in removed and slot not in exempt
            ]
            automatic = rulebook_civil_disorder_removals(
                graph, ownership, [u for _, u in leftovers], remaining
            )
            pending = list(automatic)
            for slot, unit in leftovers:
                if unit in pending:
                    pending.remove(unit)
                    removed.add(slot)
            civil.extend(automatic)

    surviving = [unit for slot, unit in enumerate(state.units) if slot not in removed]
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


def winter_decisions(hooks: BeleriandHooks, state: State, ownership: Mapping[str, str]) -> bool:
    """Whether some power faces a winter choice under the variant rules.

    A choice is a build, a waive or a site to pick, or which of several units
    to remove.
    """
    exempt = special_slots(hooks, state)
    for power in sorted({unit.power for unit in state.units} | set(ownership.values())):
        if power == _MORGOTH and hooks.morgoth_defeated:
            continue
        quota = gross_quota(hooks, power, state, ownership)
        if quota >= 0:
            if _build_allowance(hooks, power, state, ownership) > 0 and legal_winter_orders(
                hooks, power, state, ownership
            ):
                return True
        elif unit_count(state, power, exempt) > -quota:
            return True
    return False


def winter_effects(hooks: BeleriandHooks, state: State, ownership: Mapping[str, str]) -> bool:
    """Whether an empty-sheet winter changes anything.

    Removals always execute, ordered or by civil disorder.
    """
    return any(
        gross_quota(hooks, power, state, ownership) < 0
        for power in {unit.power for unit in state.units}
    )


def legal_winter_orders(
    hooks: BeleriandHooks, power: str, state: State, ownership: Mapping[str, str]
) -> tuple[AnyOrder, ...]:
    """Return every individually legal winter order for ``power``."""
    if power == _MORGOTH and hooks.morgoth_defeated:
        return ()
    exempt = special_slots(hooks, state)
    quota = gross_quota(hooks, power, state, ownership)
    orders: list[AnyOrder] = []
    if quota >= 0:
        if _build_allowance(hooks, power, state, ownership) <= 0:
            return ()
        fleet_ok = usable_fleet_knowledge(hooks, power)
        bindable = _bindable_centers(hooks, power, ownership)
        doubles_ok = power in _ELVISH and power != _OSSIRIAND and bindable
        for province in sorted(_build_sites(hooks, power, ownership)):
            if _province_budget(hooks, power, province) <= 0:
                continue
            capacity = hooks.graph.capacity(province)
            if capacity is not None and len(state.occupants(province)) >= capacity:
                continue
            kinds = [UnitKind.ARMY]
            if fleet_ok and hooks.graph.terrain(province) is Terrain.COASTAL:
                kinds.append(UnitKind.FLEET)
            for kind in kinds:
                for location in hooks.graph.locations_for(kind, province):
                    orders.append(Build(kind, location))
                    if doubles_ok:
                        for center in bindable:
                            orders.append(Build(kind, location, 2, center))
        if orders:
            orders.append(Waive())
    elif quota < 0:
        for index, unit in enumerate(state.units):
            if unit.power == power and index not in exempt:
                orders.append(Disband(unit.kind, unit.location, unit.strength or None))
    return tuple(orders)
