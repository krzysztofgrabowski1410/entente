"""Pin the convoy toll to the seas a convoy's own orders reach the landing coast through.

The rulebook tolls a convoyed army's disembarking: *"if the crossing from the
last Sea of the Convoy into the landing Province is marked, the Move is
halved"*. The last sea belongs to the convoy, not to the coast, so the engine
reads the move's **ordered disembark seas** -- the seas touching the landing
province that a chain of fleets, each ordered to convoy exactly this move,
reaches from the origin coast -- and halves the move iff that set is non-empty
and every sea in it is marked into the landing province. A fleet ordered to
convoy from a sea no such chain runs through opens no beach, and a sea without
a convoying fleet is never read.

The reading is static: it looks at what was ordered, never at which fleets
survive the phase. That is a choice and not an oversight: a toll on the
surviving route would make a strength depend on a dislodgement that depends on
that strength, and it would change no verdict on any shipped map, since none
has a mixed coast. Four layers are held here, on the synthetic two-beach map
(:mod:`tests.engine.two_beach`) because no shipped coast is mixed: the verdicts
under both engines, the table the hook is fed from, the sea walk behind the
table, and the guard that keeps the shipped maps inside the region where every
reading of the rule gives one answer (docs/adjudication.tex Sec. 2).
"""

from __future__ import annotations

from itertools import combinations

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, Resolution, adjudicate
from entente.engine.adjudicator.common import (
    build_tables,
    phase0,
    sea_chain_ends,
    sea_chain_exists,
)
from entente.engine.map import Location, MapGraph, load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit
from tests.engine.two_beach import (
    CONVOYED_MOVE,
    TWO_BEACH,
    crossing,
    two_beach_sheets,
    two_beach_state,
)

BELERIAND = RULESETS["beleriand"]
SEAS = ("SEA", "SEB", "SEC", "SED", "SEX", "SEY")

CROSSINGS = [
    # The fleet in SED was ordered to convoy, but with SEB empty no chain runs
    # through it: the convoy can only put the army ashore over the cliff.
    pytest.param(("SEA", "SEC", "SED"), False, frozenset({"SEC"}), False, id="T1-idle-beach-fleet"),
    pytest.param(("SEA", "SEC"), False, frozenset({"SEC"}), False, id="T2-cliff-route-alone"),
    pytest.param(
        ("SEA", "SEC", "SEB", "SED"), False, frozenset({"SEC", "SED"}), True, id="T3-both-routes"
    ),
    # The chosen limit of a static toll. Both routes were ordered, so both seas
    # count and the army lands whole -- although the SED fleet is dislodged
    # this very phase and only the cliff route is left to sail. The rulebook's
    # letter reads the last sea of the route that survives: it halves the
    # move, which then ties the hold of 1 and FAILS with only the SED fleet
    # dislodged. The engine leaves that reading out on purpose: a toll that
    # reads which convoying fleets survive makes a strength depend on a
    # dislodgement that depends on that strength, and it buys no verdict on
    # any shipped map -- this row is unreachable wherever no coast is mixed,
    # which test_no_shipped_coast_is_mixed guards.
    pytest.param(
        ("SEA", "SEC", "SEB", "SED"),
        True,
        frozenset({"SEC", "SED"}),
        True,
        id="T4-beach-route-disrupted",
    ),
    pytest.param(("SEB", "SED"), False, frozenset({"SED"}), True, id="T5-beach-route-alone"),
]


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize(("convoying", "attacked", "seas", "lands"), CROSSINGS)
def test_the_toll_follows_the_ordered_chains(
    convoying: tuple[str, ...],
    attacked: bool,
    seas: frozenset[str],
    lands: bool,
    engine: EngineName,
) -> None:
    state, sheets = crossing(convoying, beach_fleet_attacked=attacked)
    result = adjudicate(BELERIAND, state, sheets, graph=TWO_BEACH, engine=engine)
    expected = Resolution.SUCCEEDS if lands else Resolution.FAILS
    assert result.resolution_of(CONVOYED_MOVE) is expected
    dislodged = {d.unit.location.province for d in result.dislodgements}
    assert dislodged == ({"LND"} if lands else set()) | ({"SED"} if attacked else set())
    assert len(result.dislodgements) == len(dislodged)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_the_idle_fleet_carries_nothing(engine: EngineName) -> None:
    """A fleet ordered to convoy from a sea off every chain fails its own order too."""
    state, sheets = crossing(("SEA", "SEC", "SED"))
    result = adjudicate(BELERIAND, state, sheets, graph=TWO_BEACH, engine=engine)
    assert result.resolution_of("2F SEA C 2A ORI - LND") is Resolution.SUCCEEDS
    assert result.resolution_of("2F SEC C 2A ORI - LND") is Resolution.SUCCEEDS
    assert result.resolution_of("2F SED C 2A ORI - LND") is Resolution.FAILS


@pytest.mark.parametrize(("convoying", "attacked", "seas", "lands"), CROSSINGS)
def test_tables_expose_the_ordered_disembark_seas(
    convoying: tuple[str, ...], attacked: bool, seas: frozenset[str], lands: bool
) -> None:
    state, sheets = crossing(convoying, beach_fleet_attacked=attacked)
    records = phase0(TWO_BEACH, BELERIAND, state, sheets).records
    tables = build_tables(TWO_BEACH, BELERIAND, records)
    army = next(r.index for r in records if r.unit.location.province == "ORI")
    # Only the convoyed move has an entry: BLU's fleet move in the disrupted
    # row walks its own edge and is tolled there.
    assert tables.disembark_seas_of == {army: seas}
    # The base is read off the set: halved where every ordered beach is marked.
    marked = all(TWO_BEACH.is_mountain_crossing(sea, "LND") for sea in seas)
    assert tables.move_base_of[army] == (1 if marked else 2)
    assert tables.cutval_of[army] == tables.move_base_of[army]


@pytest.mark.parametrize(
    "convoying",
    [
        pytest.param((), id="no-fleet"),
        pytest.param(("SEA",), id="one-sea-short-of-the-coast"),
        pytest.param(("SEC", "SED"), id="landing-seas-only"),
        pytest.param(("SEA", "SED", "SEX"), id="both-ends-and-no-link"),
    ],
)
def test_a_convoyed_move_without_an_ordered_chain_has_no_disembark_sea(
    convoying: tuple[str, ...],
) -> None:
    """Fleets that link no origin sea to a landing sea leave the set empty.

    Such a move can never be viable, so nothing ever reads its strength; the
    whole base is recorded all the same, and both engines fail the move.
    """
    state, sheets = crossing(convoying)
    records = phase0(TWO_BEACH, BELERIAND, state, sheets).records
    tables = build_tables(TWO_BEACH, BELERIAND, records)
    army = next(r.index for r in records if r.unit.location.province == "ORI")
    assert tables.disembark_seas_of == {army: frozenset()}
    assert tables.move_base_of[army] == 2
    for engine in sorted(ENGINES):
        result = adjudicate(BELERIAND, state, sheets, graph=TWO_BEACH, engine=engine)
        assert result.resolution_of(CONVOYED_MOVE) is Resolution.FAILS
        assert not result.dislodgements


def test_only_fleets_convoying_this_move_open_a_beach() -> None:
    """A fleet in a landing sea that holds, or convoys another army, is not read."""
    state = two_beach_state(
        {"RED": ["2A ORI", "2F SEA", "2F SEC", "2F SEB", "2F SED"], "BLU": ["2A LND"]}
    )
    sheets = two_beach_sheets(
        {
            "RED": [
                CONVOYED_MOVE,
                "2F SEA C 2A ORI - LND",
                "2F SEC C 2A ORI - LND",
                "2F SEB C 2A LND - ORI",
                "2F SED H",
            ],
            "BLU": ["2A LND - ORI via convoy"],
        }
    )
    records = phase0(TWO_BEACH, BELERIAND, state, sheets).records
    tables = build_tables(TWO_BEACH, BELERIAND, records)
    by_origin = {r.unit.location.province: r.index for r in records}
    assert tables.disembark_seas_of == {
        by_origin["ORI"]: frozenset({"SEC"}),
        by_origin["LND"]: frozenset(),
    }


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_fleet_too_weak_to_carry_the_army_opens_no_beach(engine: EngineName) -> None:
    """Single fleets on the free route cannot carry a Double Army, so the cliff alone counts.

    Their convoy orders are invalid and become holds, which leaves the beach
    route unordered however many fleets stand on it.
    """
    state = two_beach_state(
        {"RED": ["2A ORI", "2F SEA", "2F SEC", "F SEB", "F SED"], "BLU": ["A LND"]}
    )
    sheets = two_beach_sheets(
        {
            "RED": [
                CONVOYED_MOVE,
                "2F SEA C 2A ORI - LND",
                "2F SEC C 2A ORI - LND",
                "F SEB C 2A ORI - LND",
                "F SED C 2A ORI - LND",
            ],
            "BLU": ["A LND H"],
        }
    )
    records = phase0(TWO_BEACH, BELERIAND, state, sheets).records
    tables = build_tables(TWO_BEACH, BELERIAND, records)
    army = next(r.index for r in records if r.unit.location.province == "ORI")
    assert tables.disembark_seas_of == {army: frozenset({"SEC"})}
    result = adjudicate(BELERIAND, state, sheets, graph=TWO_BEACH, engine=engine)
    assert result.resolution_of(CONVOYED_MOVE) is Resolution.FAILS
    assert not result.dislodgements


def test_the_standard_ruleset_gets_the_seas_and_ignores_them() -> None:
    """The table is calculus data, computed for every ruleset; the toll is Beleriand's."""
    state = two_beach_state({"RED": ["A ORI", "F SEA", "F SEC"], "BLU": ["A LND"]})
    sheets = two_beach_sheets(
        {"RED": ["A ORI - LND via convoy", "F SEA C A ORI - LND", "F SEC C A ORI - LND"]}
    )
    records = phase0(TWO_BEACH, STANDARD, state, sheets).records
    tables = build_tables(TWO_BEACH, STANDARD, records)
    army = next(r.index for r in records if r.unit.location.province == "ORI")
    assert tables.disembark_seas_of == {army: frozenset({"SEC"})}
    assert tables.move_base_of[army] == 1


def test_no_disembark_sea_is_recorded_on_a_board_without_convoys() -> None:
    graph = load_map("beleriand")
    units = [
        Unit("FINROD", *parse_unit_spec("2A VIN")),
        Unit("FINGOLFIN", *parse_unit_spec("A NEV")),
    ]
    state = State.of("beleriand", "S25M", units)
    sheets = {"FINROD": [parse_order("2A VIN - NEV")]}
    tables = build_tables(graph, BELERIAND, phase0(graph, BELERIAND, state, sheets).records)
    assert tables.disembark_seas_of == {}


def _reached(seas: frozenset[str], start: frozenset[str]) -> frozenset[str]:
    """Close ``start`` under fleet adjacency inside ``seas``: the oracle for the walk."""
    reached = set(seas & start)
    grew = True
    while grew:
        grew = False
        for sea in seas - reached:
            if any(TWO_BEACH.fleet_reachable(Location(r), Location(sea)) for r in reached):
                reached.add(sea)
                grew = True
    return frozenset(reached)


def test_the_walk_returns_every_goal_sea_a_chain_ends_in() -> None:
    """Check the exhaustive walk against a plain closure, over every set of fleets."""
    anchors = TWO_BEACH.sea_neighbors("ORI")
    goals = TWO_BEACH.sea_neighbors("LND")
    assert (anchors, goals) == (frozenset({"SEA", "SEB"}), frozenset({"SEC", "SED"}))
    for size in range(len(SEAS) + 1):
        for subset in combinations(SEAS, size):
            seas = set(subset)
            ends = sea_chain_ends(TWO_BEACH, seas, anchors, goals)
            assert ends == _reached(frozenset(seas), anchors) & goals
            # The boolean form is the same walk stopped at its first goal.
            assert sea_chain_exists(TWO_BEACH, seas, anchors, goals) is bool(ends)


def test_a_sea_that_is_anchor_and_goal_is_a_one_fleet_chain() -> None:
    graph = load_map("beleriand")
    nbe = frozenset({"NBE"})
    assert graph.sea_neighbors("VIN") & graph.sea_neighbors("NEV") == nbe
    assert (
        sea_chain_ends(graph, {"NBE"}, graph.sea_neighbors("VIN"), graph.sea_neighbors("NEV"))
        == nbe
    )


def _mixed_coasts(graph: MapGraph) -> dict[str, dict[str, list[str]]]:
    """Return every province with one sea marked into it and another free, with both lists."""
    mixed = {}
    for key in sorted(graph.provinces):
        seas = graph.sea_neighbors(key)
        marked = {sea for sea in seas if graph.is_mountain_crossing(sea, key)}
        if marked and marked != seas:
            mixed[key] = {"marked": sorted(marked), "free": sorted(seas - marked)}
    return mixed


def test_the_two_beach_landing_is_a_mixed_coast() -> None:
    """The guard's own control: the map built to be mixed is reported as mixed."""
    assert _mixed_coasts(TWO_BEACH) == {"LND": {"marked": ["SEC"], "free": ["SED"]}}


@pytest.mark.parametrize("map_name", ["standard", "beleriand"])
def test_no_shipped_coast_is_mixed(map_name: str) -> None:
    """Every sea touching a shipped province is marked into it, or none is.

    On such a map every ordered chain, every surviving chain and the route
    actually sailed end in seas of one toll status, so the ordered-chain
    reading, a surviving-route reading and the rulebook's letter agree on every
    position. That agreement is why the engine reads only what was ordered: a
    toll on the surviving route would cost a strength that depends on the
    resolution, and here it would change nothing.
    """
    mixed = _mixed_coasts(load_map(map_name))
    assert not mixed, (
        f"{map_name} has a mixed coast {mixed}: a landing province with one sea marked into "
        "it and another free. The convoy toll reads the chains a convoy was ORDERED to "
        "sail, which equals the rulebook's letter (the last sea of the route actually "
        "sailed) only while no coast is mixed. A mixed coast makes the disrupted-route "
        "position reachable -- two routes ordered, the free one's last fleet dislodged, the "
        "army landing whole over the marked crossing (the T4 row of "
        "test_the_toll_follows_the_ordered_chains). A toll on the SURVIVING route was left "
        "out on purpose: it makes a strength depend on a dislodgement that depends on that "
        "strength, and it changed no verdict while no shipped coast was mixed. This map "
        "takes that reason away, so the choice goes back to the rulebook's author before "
        "the map ships."
    )
