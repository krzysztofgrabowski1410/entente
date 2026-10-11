"""Index every order the engine can emit on a map, stably and by derivation.

The policy head decodes over this vocabulary, so the index of an order must be
the same for every model and every checkpoint built against the same map.

The vocabulary is *derived*, never hand-listed. Its entries are exactly the
canonical strings (:func:`entente.engine.orders.format_order`) the engine's own
legal-order enumeration (:meth:`entente.engine.game.Game.legal_orders`) produces,
collected by driving that enumeration over hypothetical boards:

- **singles** -- one unit of every kind on every location it may occupy: holds
  and overland moves;
- **pairs** -- every ordered pair of placements on otherwise-empty boards:
  supports (of holds and of the other unit's moves) and single-hop convoys;
- **convoy geometry** -- the pair pass under-covers convoys, because a two-unit
  board realizes only single-sea chains. Multi-sea via-convoy moves, the convoy
  orders that carry them (any sea fleet may convoy any army under the standard
  rules -- :meth:`StandardRuleset.convoy_capability`, so a route needs no fleet
  *on* it), and supports of those moves are enumerated directly from sea
  connectivity (:func:`entente.engine.adjudicator.common.sea_chain_exists`, the
  same predicate the engine's ``potential_chain`` uses), building real order
  objects and formatting them so the strings match the engine exactly;
- **phase-terminal orders** -- disband (per placement), build (per home-center
  location), and waive.

Because the result is a strict *superset* of every legal order, the per-unit
legality mask (:mod:`entente.nn.heads`) equals ``game.legal_orders`` exactly --
the invariant ``tests/nn/test_order_vocab.py`` and the masking test pin. On the
standard map this is 36,343 entries, which that same test pins. Off-chain convoy
orders dominate the count: this engine permits a sea fleet to convoy an army it
does not lie on the route of, so estimates drawn from DipNet's data source
(around 13k) are a lower bound on the vocabulary, not a target for it. The size
is reported by the enumeration, never fixed in advance.

Index stability: entries are sorted by their canonical string, so an order's
index never depends on enumeration or dict-iteration order. This module imports
no torch -- the vocabulary is pure engine data, cached per (map, ruleset).

Scope: CLASSIC boards only, and :func:`_reject_uncoverable` enforces it rather
than assuming it. The three passes above quantify over one- and two-unit boards
of units of base strength 1 on capacity-1 provinces and read ``legal_orders`` alone, so a
map with Areas, units above base strength 1 or variant order types would come out silently
INCOMPLETE -- the opposite of the superset guarantee. Such a (map, ruleset)
raises instead. This is the vocabulary-level statement of the learned seats
being standard-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

from entente.engine.adjudicator.common import sea_chain_exists
from entente.engine.game import Game
from entente.engine.map import Location, MapGraph, Terrain, UnitKind, load_map
from entente.engine.orders import (
    AnyOrder,
    Build,
    Convoy,
    Disband,
    Move,
    Support,
    Waive,
    format_order,
    parse_unit_spec,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS, STANDARD, Ruleset
from entente.engine.state import State, Unit

_MOVEMENT = Phase(1901, Season.SPRING, PhaseKind.MOVEMENT)
_WAIVE = format_order(Waive())


@dataclass(frozen=True, slots=True)
class OrderVocabulary:
    """An immutable, index-stable order vocabulary for one (map, ruleset).

    ``entries`` is the sorted tuple of canonical order strings; ``index`` maps a
    string (or an order object, via :func:`format_order`) to its position. The
    two are consistent by construction -- ``entries[index(s)] == s``.
    """

    map_name: str
    ruleset_name: str
    entries: tuple[str, ...]
    _index: dict[str, int] = field(repr=False)

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, order: AnyOrder | str) -> bool:
        return self._key(order) in self._index

    @staticmethod
    def _key(order: AnyOrder | str) -> str:
        return order if isinstance(order, str) else format_order(order)

    def index(self, order: AnyOrder | str) -> int:
        """Give the vocabulary index of an order.

        Raises :class:`KeyError` if the order is absent. Since the vocabulary is
        a superset of the legal orders, an absent order means the builder missed
        one and the mask would be wrong, so this fails loudly rather than
        returning a stand-in index.
        """
        return self._index[self._key(order)]

    def get(self, order: AnyOrder | str) -> int | None:
        """Give the index of an order, or ``None`` if it is not in the vocabulary."""
        return self._index.get(self._key(order))

    def order(self, idx: int) -> str:
        """Give the canonical string at ``idx``, inverting :meth:`index`."""
        return self.entries[idx]


def _placements(graph: MapGraph) -> list[tuple[UnitKind, Location]]:
    """List every ``(kind, location)`` a single unit could occupy, in a fixed order.

    These placements are the atoms both enumeration passes quantify over, so
    their order fixes the enumeration's own determinism.
    """
    out: list[tuple[UnitKind, Location]] = []
    for province in sorted(graph.provinces):
        for kind in (UnitKind.ARMY, UnitKind.FLEET):
            out.extend((kind, loc) for loc in graph.locations_for(kind, province))
    return out


def _legal_strings(graph: MapGraph, ruleset: Ruleset, units: list[Unit], power: str) -> set[str]:
    """Enumerate every order ``power`` may submit on a board holding exactly ``units``.

    The strings come from the engine's own enumeration, verbatim, so the
    vocabulary cannot drift from what the engine will accept.
    """
    state = State.of(graph.name, _MOVEMENT.tag, units)
    game = Game(ruleset, graph, state, {}, _MOVEMENT)
    return {format_order(order) for order in game.legal_orders(power)}


def _convoy_pairs(graph: MapGraph) -> list[tuple[str, str]]:
    """Pair the coasts joined by *some* chain of seas.

    Every sea is convoy-capable under the standard rules, so this is the maximal
    set ``potential_chain`` can accept and therefore every via-convoy move the
    map admits.
    """
    seas = {p for p in graph.provinces if graph.terrain(p) is Terrain.SEA}
    coastal = [p for p in sorted(graph.provinces) if graph.terrain(p) is Terrain.COASTAL]
    return [
        (origin, dest)
        for origin in coastal
        for dest in coastal
        if origin != dest
        and sea_chain_exists(graph, seas, graph.sea_neighbors(origin), graph.sea_neighbors(dest))
    ]


def _supports_province(graph: MapGraph, kind: UnitKind, loc: Location, province: str) -> bool:
    """Say whether a ``kind`` unit at ``loc`` could support an action into ``province``.

    This mirrors ``Game._support_reachable``: an army reaches by province
    adjacency to a non-sea province, a fleet by a reachable location of it.
    """
    if kind is UnitKind.ARMY:
        return graph.terrain(province) is not Terrain.SEA and graph.army_reachable(
            loc.province, province
        )
    return bool(graph.fleet_destinations(loc, province))


def _convoy_strings(graph: MapGraph) -> set[str]:
    """Enumerate every convoy-related order the map admits.

    That is the via-convoy moves, the convoy orders any sea fleet can give for
    them, and the supports of those moves. Each is built as a real order object
    and formatted, so the strings match the engine's own exactly.
    """
    placements = _placements(graph)
    sea_fleets = [
        loc
        for p in sorted(graph.provinces)
        if graph.terrain(p) is Terrain.SEA
        for loc in graph.fleet_locations(p)
    ]
    out: set[str] = set()
    for origin, dest in _convoy_pairs(graph):
        cargo = Move(UnitKind.ARMY, Location(origin), Location(dest), via_convoy=True)
        out.add(format_order(cargo))
        for fleet in sea_fleets:
            out.add(format_order(Convoy(UnitKind.FLEET, fleet, cargo)))
        supported = Move(UnitKind.ARMY, Location(origin), Location(dest))
        for kind, loc in placements:
            if loc.province != origin and _supports_province(graph, kind, loc, dest):
                out.add(format_order(Support(kind, loc, supported)))
    return out


def _terminal_strings(graph: MapGraph) -> set[str]:
    """Enumerate the retreat- and adjustment-phase orders.

    Those are a disband per placement, since any unit can be dislodged or
    removed, a build per home-center location, and the waive.
    """
    out = {_WAIVE}
    for kind, loc in _placements(graph):
        out.add(format_order(Disband(kind, loc)))
    for centers in graph.home_centers.values():
        for center in centers:
            for kind in (UnitKind.ARMY, UnitKind.FLEET):
                for loc in graph.locations_for(kind, center):
                    out.add(format_order(Build(kind, loc)))
    return out


def _build_entries(graph: MapGraph, ruleset: Ruleset) -> tuple[str, ...]:
    """Run the three enumeration passes and return their union, sorted.

    Two nominal powers are enough for every order form: one owner covers the
    singles, and an ordered pair of placements covers supports and single-hop
    convoys from both sides.
    """
    powers = sorted(set(graph.home_centers) | set(graph.starting_units))
    if len(powers) < 2:  # pragma: no cover - every real map has >= 2 powers
        raise ValueError(f"map {graph.name!r} has too few powers to enumerate orders")
    first, second = powers[0], powers[1]
    placements = _placements(graph)

    strings: set[str] = set()
    for kind, loc in placements:
        strings |= _legal_strings(graph, ruleset, [Unit(first, kind, loc)], first)
    for kind_a, loc_a in placements:
        for kind_b, loc_b in placements:
            if loc_a.province == loc_b.province:
                continue  # capacity: two units never share a province on classic maps
            units = [Unit(first, kind_a, loc_a), Unit(second, kind_b, loc_b)]
            strings |= _legal_strings(graph, ruleset, units, first)
    strings |= _convoy_strings(graph)
    strings |= _terminal_strings(graph)
    return tuple(sorted(strings))


_WAIVE_KEY = "<waive>"


def legal_order_indices(game: Game, power: str, vocab: OrderVocabulary) -> dict[str, list[int]]:
    """Group ``power``'s legal orders this phase by the ordering unit, as vocabulary indices.

    A group is keyed by the unit's ``str(location)``, and waives share a sentinel
    key. The union over the groups equals ``{vocab.index(o) for o in
    game.legal_orders(power)}`` exactly, which is the per-unit legality mask the
    policy head (:mod:`entente.nn.heads`) applies: the vocabulary is a superset
    of every legal order, so grouping the engine's own enumeration reproduces the
    legal set rather than approximating it. Indices within a group are sorted, so
    a mask does not depend on enumeration order. This function imports no torch,
    which lets the masking-exactness test run without the ``ml`` group installed.
    """
    groups: dict[str, list[int]] = {}
    for order in game.legal_orders(power):
        key = _WAIVE_KEY if isinstance(order, Waive) else str(order.location)
        groups.setdefault(key, []).append(vocab.index(format_order(order)))
    return {key: sorted(indices) for key, indices in groups.items()}


def _reject_uncoverable(graph: MapGraph, ruleset: Ruleset) -> None:
    """Refuse a (map, ruleset) whose orders this builder cannot enumerate.

    The construction rests on three classic-map facts and silently returns an
    INCOMPLETE vocabulary where any of them fails -- breaking the superset
    guarantee :meth:`OrderVocabulary.index` depends on, and which that method
    exists to fail loudly about. Beleriand breaks all three, which is why the
    learned seats are standard-only; this puts the refusal at the source, so a
    vocabulary either covers its board or raises.

    - **Base strengths.** Boards are built from ``Unit(power, kind, loc)`` at
      base strength 1, and ``format_order`` writes a prefix only above base
      strength 1, so no entry this builder can produce ever carries one. The
      orders of a map with stronger units would be missing wholesale --
      Beleriand's OPENING alone has 232 of them.
    - **Capacity.** The pair pass skips same-province placements ("two units
      never share a province on classic maps"), so an Area's co-located orders,
      Area-mate supports above all, are never enumerated.
    - **Variant orders.** :func:`_legal_strings` reads ``legal_orders`` only,
      never ``legal_layer2_orders``, so Dislodge/Bar/Transform/mask orders get
      no entries at all.
    """
    reasons: list[str] = []
    if any(graph.capacity(province) != 1 for province in graph.provinces):
        reasons.append("provinces of capacity != 1 (Areas): co-located orders are not enumerated")
    stronger = sorted(
        {
            spec
            for specs in graph.starting_units.values()
            for spec in specs
            if parse_unit_spec(spec)[2] != 1
        }
    )
    if stronger:
        reasons.append(
            f"units above base strength 1 ({', '.join(stronger[:3])}): "
            "no entry can name a base strength"
        )
    if len(ruleset.order_registry) > len(STANDARD.order_registry):
        reasons.append("variant order types: _legal_strings reads legal_orders only")
    if reasons:
        raise ValueError(
            f"cannot build an order vocabulary for map {graph.name!r} with ruleset "
            f"{ruleset.name!r}: " + "; ".join(reasons) + ". The superset guarantee would be "
            "false, so this raises rather than return a vocabulary that silently cannot "
            "express legal orders."
        )


@cache
def build_vocabulary(map_name: str = "standard", ruleset_name: str = "standard") -> OrderVocabulary:
    """Build the order vocabulary for a (map, ruleset), once, and cache it.

    The same map and ruleset always yield the same entries in the same order, so
    a checkpoint's head indices stay meaningful across processes;
    ``tests/nn/test_order_vocab.py`` pins both the ordering and a content
    snapshot. Raises :class:`ValueError` for a board this construction cannot
    cover -- see :func:`_reject_uncoverable`.
    """
    graph = load_map(map_name)
    ruleset = RULESETS[ruleset_name]
    _reject_uncoverable(graph, ruleset)
    entries = _build_entries(graph, ruleset)
    return OrderVocabulary(
        map_name=map_name,
        ruleset_name=ruleset_name,
        entries=entries,
        _index={entry: i for i, entry in enumerate(entries)},
    )
