"""Retreat-phase adjudication -- docs/adjudication.tex Sec. 9, DATC section 6.H.

The retreat phase is a mini-adjudication under the same generalized entry
rule as movement (Sec. 4.3), with two structural simplifications (Sec. 9):
every retreat has strength 0 (the standard rules have no strengths; the
Beleriand rulebook says "Retreat Orders always have 0 Strength") and there are
no supports, convoys or holds -- the only valid retreat orders are a plain
move and ``Disband`` (DATC 6.H.1-6.H.4). With all
candidates in one strength class, greedy class admission degenerates to: the
retreats into a province are all admitted iff they all fit its free capacity,
else they all bounce -- at capacity 1 exactly the classic rule that contested
retreats disband every contestant (6.H.7/6.H.8) and that an occupied province
admits nobody. No standard special-casing: capacity is map data throughout.

Forbidden destinations are static per-province facts of the movement phase
(6.H.13/6.H.14 -- nothing ordered *during* the retreat can open or close
them; 6.H.15/6.H.17 -- coasts never help, the ban is on the province):

- the unit's own province (no retreat in place) and, for a convoyed order,
  anywhere: ``via convoy`` is invalid during retreats (6.H.3);
- the dislodger's province of origin -- unless the dislodger arrived by
  convoy (DATC 4.A.5 choice B; 6.H.5/6.H.10 forbid, 6.H.11/6.H.12 allow);
- provinces embattled by a movement-phase standoff, as recorded in
  ``PhaseResult.embattled`` (6.H.6; a lone beaten entrant does not embattle,
  6.H.9, nor does a failed convoyed move, 6.F.7);
- provinces the unit may not enter at all (``Ruleset.entry_permission`` --
  Beleriand's Angband/Eriador, Sec. 4.3; vacuous on classic maps).

An invalid retreat order exerts nothing -- it cannot bounce anyone (6.H.10) --
and its unit disbands, as does every unordered dislodged unit; of several
orders to one dislodged unit the first valid one wins (the 4.D.3/4.E.1
reading, mirroring movement Phase 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import Dislodgement, OrderResolution, Resolution
from entente.engine.map.types import Location, UnitKind
from entente.engine.orders import (
    Build,
    Disband,
    Move,
    Order,
    Waive,
    format_order,
    parse_retreat_order,
)
from entente.engine.state import State, Unit, check_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.adjudicator.common import PhaseResult
    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import AnyOrder
    from entente.engine.rules.protocol import Ruleset


@dataclass(frozen=True, slots=True)
class RetreatResult:
    """Retreat adjudication output.

    Per-order resolutions in sheet order (powers sorted), the state with
    successful retreats placed, the dislodged units that disbanded (by
    choice, default, or contest), and the dislodgements that retreated, each
    with the location it reached, in the movement result's order. A row's
    text says what was written, not which unit it bound -- an invalid line, a
    second order for a unit, or a line naming a unit still on the board binds
    none -- so ``retreated`` is what a caller tracking the units themselves
    reads (the Beleriand game layer's ledger).
    """

    resolutions: tuple[OrderResolution, ...]
    new_state: State
    disbanded: tuple[Unit, ...]
    retreated: tuple[tuple[Dislodgement, Location], ...] = ()

    def resolution_of(self, text: str) -> Resolution:
        """Look up a written retreat order's outcome; ``text`` is normalized first."""
        canonical = format_order(parse_retreat_order(text))
        for entry in self.resolutions:
            if entry.text == canonical:
                return entry.resolution
        raise KeyError(f"no resolution recorded for {canonical!r}")


def _written(strength: int) -> int | None:
    """Return the base strength an order WRITES for a unit of ``strength``.

    The canonical grammar has no ``1A``, so base strength 1 is spelled with no prefix
    at all.
    """
    return strength if strength != 1 else None


def _resolve_destination(graph: MapGraph, unit: Unit, move: Move) -> Location | None:
    """Return a retreat move's destination under the Sec. 1 reachability rules, or None.

    No convoy option exists: armies drop written coasts and use province
    adjacency; a fleet moving to a split-coast province must name a reachable
    coast, an omitted coast is filled in iff exactly one is reachable (mirrors
    movement, DATC 6.B.1-3).
    """
    if move.via_convoy:
        return None  # no convoy during retreat (6.H.3, 6.H.13)
    destination = move.destination
    if unit.kind is UnitKind.ARMY:
        target = Location(destination.province)
        return target if graph.army_reachable(unit.location.province, target.province) else None
    if destination.coast is None and graph.coasts.get(destination.province):
        options = graph.fleet_destinations(unit.location, destination.province)
        return options[0] if len(options) == 1 else None
    return destination if graph.fleet_reachable(unit.location, destination) else None


def _valid_retreat(
    ruleset: Ruleset,
    graph: MapGraph,
    movement: PhaseResult,
    dislodgement: Dislodgement,
    move: Move,
) -> Move | None:
    """Return the effective retreat move (destination coast resolved), or None.

    The move names the RETREATING UNIT's base strength: retreats fight at strength 0,
    but a dislodged ``2A`` and a dislodged ``A`` are still different units, so
    one order must not be able to denote either. Strength 1 stays prefixless
    -- the canonical grammar has no ``1A`` -- which keeps the standard column
    byte-identical by construction.
    """
    unit = dislodgement.unit
    destination = _resolve_destination(graph, unit, move)
    if destination is None or destination.province == unit.location.province:
        return None
    if destination.province == dislodgement.dislodger_origin and not dislodgement.by_convoy:
        return None  # 4.A.5-b: open only when the dislodger came by convoy
    if destination.province in movement.embattled:
        return None
    if not ruleset.entry_permission(unit, graph.province(destination.province)):
        return None
    return Move(unit.kind, unit.location, destination, strength=_written(unit.strength))


def legal_retreats(
    ruleset: Ruleset,
    graph: MapGraph,
    movement: PhaseResult,
    dislodgement: Dislodgement,
) -> tuple[Move, ...]:
    """Return every retreat move of a dislodged unit that could succeed if uncontested.

    Valid per the Sec. 9 forbidden-destination rules *and* aimed at free
    capacity (strength-0 retreats can never displace, so an already-full
    province is a dead option -- a unit with an empty tuple here disbands with
    no decision, "destroyed immediately", 6.H.16).
    """
    unit = dislodgement.unit
    if unit.kind is UnitKind.ARMY:
        reachable = graph.army_adjacency.get(unit.location.province, frozenset())
        neighbors = [Location(province) for province in sorted(reachable)]
    else:
        neighbors = sorted(
            graph.fleet_adjacency.get(unit.location, frozenset()),
            key=lambda loc: (loc.province, loc.coast or ""),
        )
    options: list[Move] = []
    seen: set[Location] = set()
    for destination in neighbors:
        candidate = Move(unit.kind, unit.location, destination)
        move = _valid_retreat(ruleset, graph, movement, dislodgement, candidate)
        if move is None or move.destination in seen:
            continue
        capacity = graph.capacity(move.destination.province)
        occupants = movement.new_state.occupants(move.destination.province)
        if capacity is not None and capacity - len(occupants) < 1:
            continue
        seen.add(move.destination)
        options.append(move)
    return tuple(options)


def _fitting_dislodged_units(
    dislodgements: Sequence[Dislodgement],
    power: str,
    order: Order | Disband | Build,
    strict: bool,
) -> list[int]:
    """Return the indices of the dislodged own units a retreat order's spec describes.

    Retreat orders resolve against *dislodged own units*, by province first --
    written kind, strength, designator and coast narrow the match when they
    can, else they are advisory (mirrors movement Phase 0 claim binding;
    indices keep equal dislodgements distinct -- multiset Areas may shed twins).

    In the exact-match layer an unwritten prefix means base strength 1, never
    "any base strength" (the rulebook reads a bare "A" as "1A"), and the
    designator matches exactly both ways, as in the movement phase's reference
    matching: a written ``(M)``/``(G)`` fits only the dislodgement recorded
    under that designator, an unwritten one only a dislodgement recorded under
    none -- so of two specials dislodged from one Area, ``3A(M)`` binds the
    Morgoth unit's dislodgement and is valid or not for its dislodger alone.
    ``strict`` is the ruleset's ``strict_designation`` and governs the
    province-only fallback exactly as it does for support references: without
    it the fallback stands (the DATC reading), with it a spec describing no
    dislodged unit binds nothing -- otherwise the fallback would hand a bare
    ``A NEB - GEL`` the very 2A the exact layer just refused. Standard is
    byte-identical either way: its binding is advisory, and at capacity 1 the
    province-only fallback binds the one dislodged unit of the province
    whatever the spec writes.
    """
    own = [
        i
        for i, d in enumerate(dislodgements)
        if d.unit.power == power and d.unit.location.province == order.location.province
    ]
    exact = [
        i
        for i in own
        if dislodgements[i].unit.kind is order.unit_kind
        and (order.strength or 1) == dislodgements[i].unit.strength
        and order.special == dislodgements[i].special
    ]
    if exact or strict:
        return exact
    return own


def _bind(
    row: int,
    options: Mapping[int, Mapping[int, Move | Disband]],
    holder: dict[int, int],
    visited: set[int],
) -> bool:
    """Give retreat order ``row`` a unit it is valid for, moving earlier orders along if need be.

    One augmenting step of a bipartite matching between a power's retreat
    orders and its dislodged units: an earlier order keeps a unit, though
    perhaps another one it is also valid for, so taking the orders in sheet
    order lets as many succeed as can, earlier lines first.
    """
    for slot in sorted(options[row]):
        if slot in visited:
            continue
        visited.add(slot)
        if slot not in holder or _bind(holder[slot], options, holder, visited):
            holder[slot] = row
            return True
    return False


@dataclass(frozen=True, slots=True)
class _Row:
    """A retreat order that reaches the binding: its power, its place in that sheet, its text."""

    power: str
    index: int
    text: str


def _bind_sheets(
    ruleset: Ruleset,
    graph: MapGraph,
    movement: PhaseResult,
    sheets: Mapping[str, Sequence[AnyOrder]],
) -> tuple[list[_Row], dict[int, tuple[int, Move | Disband]]]:
    """Bind the retreat orders to dislodged units: the Phase 0 analogue of the retreat phase.

    Returns the rows -- every order in sheet order (powers sorted), but a line
    repeated with no unit left for it -- and, per row that binds a unit, the
    unit's index in ``movement.dislodgements`` and the order it executes.

    Each order binds a dislodged unit occurrence (by index -- equal
    dislodgements stay distinct) that its spec describes and it is valid for,
    the first valid order of a unit counting. Twins that cannot be told apart
    take identically worded lines one each, and each line goes to a twin it is
    valid for wherever one is free: the rulebook distributes a specification
    that fits only twins, and the one-order-per-unit rule does not apply in an
    Area (its notes to 4.C.5 and 4.D.3).
    """
    dislodgements = movement.dislodgements
    rows: list[_Row] = []
    bound: dict[int, tuple[int, Move | Disband]] = {}
    for power in sorted(sheets):
        seen: set[str] = set()
        options: dict[int, dict[int, Move | Disband]] = {}
        holder: dict[int, int] = {}
        for index, order in enumerate(sheets[power]):
            text = format_order(order)
            fitting = (
                []
                if isinstance(order, Waive)
                else _fitting_dislodged_units(
                    dislodgements, power, order, ruleset.strict_designation
                )
            )
            if text in seen and all(slot in holder for slot in fitting):
                continue  # a repeated line with no unit left for it is one order (DATC 4.D.3)
            seen.add(text)
            row = len(rows)
            valid_for: dict[int, Move | Disband] = {}
            for slot in fitting:
                dislodgement = dislodgements[slot]
                valid: Move | Disband | None = None
                if isinstance(order, Disband):
                    valid = Disband(dislodgement.unit.kind, dislodgement.unit.location)
                elif isinstance(order, Move):
                    valid = _valid_retreat(ruleset, graph, movement, dislodgement, order)
                if valid is not None:
                    valid_for[slot] = valid
            options[row] = valid_for
            _bind(row, options, holder, set())
            rows.append(_Row(power, index, text))
        for slot, row in holder.items():
            bound[row] = (slot, options[row][slot])
    return rows, bound


def bind_retreat_orders(
    ruleset: Ruleset,
    movement: PhaseResult,
    sheets: Mapping[str, Sequence[AnyOrder]],
    *,
    graph: MapGraph,
) -> dict[tuple[str, int], tuple[Dislodgement, Move | Disband]]:
    """Return the dislodged unit each retreat order binds and the order it executes.

    Keyed by (power, index in that power's sheet). An order that binds no unit
    -- one that describes none, is valid for none it describes, or is a
    second order for a unit -- is absent. This is the binding of
    :func:`adjudicate_retreats` alone, before the entry contest, for a caller
    that must know which line counts for which unit: the Beleriand game layer
    spends an entry credential only for such a line.
    """
    rows, bound = _bind_sheets(ruleset, graph, movement, sheets)
    return {
        (rows[row].power, rows[row].index): (movement.dislodgements[slot], order)
        for row, (slot, order) in bound.items()
    }


def adjudicate_retreats(
    ruleset: Ruleset,
    movement: PhaseResult,
    sheets: Mapping[str, Sequence[AnyOrder]],
    *,
    graph: MapGraph,
) -> RetreatResult:
    """Adjudicate one retreat phase (Sec. 9) as a pure function of the movement result.

    Deterministic; the ruleset enters through entry permission and the
    capacity data consumed by the shared entry rule (strength plays no role:
    every retreat has strength 0).
    """
    dislodgements = movement.dislodgements
    rows, bound = _bind_sheets(ruleset, graph, movement, sheets)
    chosen: dict[int, Move | Disband] = dict(bound.values())

    # The entry contest (Sec. 4.3 at strength 0): per province, the one strength
    # class of candidates is admitted whole iff it fits the free capacity.
    candidates: dict[str, list[int]] = {}
    for slot, order in chosen.items():
        if isinstance(order, Move):
            candidates.setdefault(order.destination.province, []).append(slot)

    admitted: set[int] = set()
    for province, entrants in candidates.items():
        capacity = graph.capacity(province)
        free = None if capacity is None else capacity - len(movement.new_state.occupants(province))
        if free is None or len(entrants) <= free:
            admitted.update(entrants)

    surviving = list(movement.new_state.units)
    disbanded: list[Unit] = []
    retreated: list[tuple[Dislodgement, Location]] = []
    outcome_by_slot: dict[int, bool] = {}
    for slot, record in enumerate(dislodgements):
        choice = chosen.get(slot)
        if isinstance(choice, Move) and slot in admitted:
            surviving.append(
                Unit(
                    record.unit.power,
                    record.unit.kind,
                    choice.destination,
                    record.unit.strength,
                )
            )
            retreated.append((record, choice.destination))
            outcome_by_slot[slot] = True
            continue
        disbanded.append(record.unit)
        outcome_by_slot[slot] = isinstance(choice, Disband)

    final: list[OrderResolution] = []
    for row, entry in enumerate(rows):
        claimant = bound.get(row)
        succeeds = claimant is not None and outcome_by_slot[claimant[0]]
        resolution = Resolution.SUCCEEDS if succeeds else Resolution.FAILS
        final.append(OrderResolution(power=entry.power, text=entry.text, resolution=resolution))

    new_state = State.of(movement.new_state.map_name, movement.new_state.phase, surviving)
    check_state(graph, new_state)
    disbanded.sort(key=lambda u: (u.location.province, u.location.coast or "", u.power))
    return RetreatResult(
        resolutions=tuple(final),
        new_state=new_state,
        disbanded=tuple(disbanded),
        retreated=tuple(retreated),
    )
