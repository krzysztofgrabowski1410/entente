"""Pin the retreat-phase corners the DATC 6.H cases leave open.

The capacity tests run on a synthetic multi-capacity map, because the classic
board has none. What they establish is that retreats go through the generalised
entry rule at strength 0 (docs/adjudication.tex Sec. 4.3): a capacity-2 province
admits two retreats whole and bounces three, which no special case written
around capacity 1 could produce.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.common import Dislodgement, PhaseResult, Resolution
from entente.engine.adjudicator.retreats import adjudicate_retreats, legal_retreats
from entente.engine.map import Location, UnitKind, build_graph, load_map
from entente.engine.orders import format_order, parse_retreat_order
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

MINI_DATA = {
    "name": "mini",
    "default_capacity": 1,
    "provinces": {
        "AAA": {"name": "Aaa", "terrain": "inland", "capacity": 2},
        "BBB": {"name": "Bbb", "terrain": "inland"},
        "CCC": {"name": "Ccc", "terrain": "inland"},
        "DDD": {"name": "Ddd", "terrain": "inland"},
        "EEE": {"name": "Eee", "terrain": "inland"},
    },
    "army_adjacency": {
        "AAA": ["BBB", "CCC", "DDD", "EEE"],
        "BBB": ["AAA", "CCC", "DDD", "EEE"],
        "CCC": ["AAA", "BBB", "DDD", "EEE"],
        "DDD": ["AAA", "BBB", "CCC", "EEE"],
        "EEE": ["AAA", "BBB", "CCC", "DDD"],
    },
    "fleet_adjacency": {},
}

MINI = build_graph(MINI_DATA)


def mini_movement(
    stayers: list[Unit], dislodgements: list[Dislodgement], embattled: frozenset[str] = frozenset()
) -> PhaseResult:
    """Fabricate a movement result on the mini map, since it is plain data."""
    return PhaseResult(
        resolutions=(),
        new_state=State.of("mini", "S1901M", stayers),
        dislodgements=tuple(dislodgements),
        embattled=embattled,
    )


def army(power: str, province: str) -> Unit:
    """Build one army of ``power`` standing in ``province``."""
    return Unit(power, UnitKind.ARMY, Location(province))


def dislodged(power: str, province: str, origin: str, by_convoy: bool = False) -> Dislodgement:
    """Record an army of ``power`` dislodged out of ``province`` from ``origin``."""
    return Dislodgement(unit=army(power, province), dislodger_origin=origin, by_convoy=by_convoy)


class TestGeneralizedCapacity:
    """Retreats admitted by the generalised entry rule at strength 0."""

    def test_two_retreats_fit_a_capacity_two_province(self) -> None:
        movement = mini_movement(
            [army("BLUE", "BBB"), army("BLUE", "CCC")],
            [dislodged("RED", "BBB", "CCC"), dislodged("RED", "CCC", "DDD")],
        )
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {"RED": [parse_retreat_order("A BBB - AAA"), parse_retreat_order("A CCC - AAA")]},
            graph=MINI,
        )
        assert result.resolution_of("A BBB - AAA") is Resolution.SUCCEEDS
        assert result.resolution_of("A CCC - AAA") is Resolution.SUCCEEDS
        assert result.disbanded == ()
        assert len(result.new_state.occupants("AAA")) == 2

    def test_three_retreats_overflow_capacity_two_and_all_disband(self) -> None:
        movement = mini_movement(
            [army("BLUE", "BBB"), army("BLUE", "CCC"), army("BLUE", "DDD")],
            [
                dislodged("RED", "BBB", "CCC"),
                dislodged("RED", "CCC", "DDD"),
                dislodged("RED", "DDD", "EEE"),
            ],
        )
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {
                "RED": [
                    parse_retreat_order("A BBB - AAA"),
                    parse_retreat_order("A CCC - AAA"),
                    parse_retreat_order("A DDD - AAA"),
                ]
            },
            graph=MINI,
        )
        assert all(entry.resolution is Resolution.FAILS for entry in result.resolutions)
        assert len(result.disbanded) == 3
        assert result.new_state.occupants("AAA") == ()

    def test_partially_occupied_capacity_two_bounces_a_pair(self) -> None:
        # One slot free: the two-candidate class does not fit whole, so the
        # tie-overflow rule bounces both (Sec. 4.3 greedy class admission).
        movement = mini_movement(
            [army("GREEN", "AAA"), army("BLUE", "BBB"), army("BLUE", "CCC")],
            [dislodged("RED", "BBB", "CCC"), dislodged("RED", "CCC", "DDD")],
        )
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {"RED": [parse_retreat_order("A BBB - AAA"), parse_retreat_order("A CCC - AAA")]},
            graph=MINI,
        )
        assert all(entry.resolution is Resolution.FAILS for entry in result.resolutions)
        assert len(result.disbanded) == 2

    def test_legal_retreats_include_multi_capacity_options(self) -> None:
        movement = mini_movement(
            [army("GREEN", "AAA"), army("BLUE", "BBB")],
            [dislodged("RED", "BBB", "CCC")],
        )
        options = legal_retreats(STANDARD, MINI, movement, movement.dislodgements[0])
        # AAA has a free slot (capacity 2, one occupant); CCC is the
        # dislodger's origin; DDD and EEE are open.
        assert [move.destination.province for move in options] == ["AAA", "DDD", "EEE"]


class TestOrderHandling:
    """How a retreat sheet's rows are bound, ordered and refused."""

    def test_first_valid_order_wins_and_later_ones_fail(self) -> None:
        movement = mini_movement(
            [army("BLUE", "BBB")],
            [dislodged("RED", "BBB", "CCC")],
        )
        sheets = {
            "RED": [
                parse_retreat_order("A BBB - CCC"),  # dislodger origin: invalid
                parse_retreat_order("A BBB - DDD"),  # first valid: executes
                parse_retreat_order("A BBB - EEE"),  # never reached
            ]
        }
        result = adjudicate_retreats(STANDARD, movement, sheets, graph=MINI)
        assert result.resolution_of("A BBB - CCC") is Resolution.FAILS
        assert result.resolution_of("A BBB - DDD") is Resolution.SUCCEEDS
        assert result.resolution_of("A BBB - EEE") is Resolution.FAILS
        assert army("RED", "DDD") in result.new_state.units

    def test_orders_for_non_dislodged_units_fail_and_change_nothing(self) -> None:
        movement = mini_movement([army("BLUE", "BBB")], [])
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {"BLUE": [parse_retreat_order("Disband A BBB")]},
            graph=MINI,
        )
        assert result.resolution_of("Disband A BBB") is Resolution.FAILS
        assert result.new_state == movement.new_state
        assert result.disbanded == ()

    def test_unordered_dislodged_units_disband(self) -> None:
        movement = mini_movement([army("BLUE", "BBB")], [dislodged("RED", "BBB", "CCC")])
        result = adjudicate_retreats(STANDARD, movement, {}, graph=MINI)
        assert result.disbanded == (army("RED", "BBB"),)

    def test_resolution_of_unknown_order_raises(self) -> None:
        movement = mini_movement([army("BLUE", "BBB")], [dislodged("RED", "BBB", "CCC")])
        result = adjudicate_retreats(STANDARD, movement, {}, graph=MINI)
        with pytest.raises(KeyError):
            result.resolution_of("A BBB - DDD")

    def test_duplicate_retreat_text_reports_once(self) -> None:
        movement = mini_movement([army("BLUE", "BBB")], [dislodged("RED", "BBB", "CCC")])
        sheets = {"RED": [parse_retreat_order("A BBB - DDD"), parse_retreat_order("A BBB - DDD")]}
        result = adjudicate_retreats(STANDARD, movement, sheets, graph=MINI)
        assert len(result.resolutions) == 1
        assert result.resolutions[0].resolution is Resolution.SUCCEEDS


class TestCoastHandling:
    """Retreats to and from the split-coast provinces."""

    def test_fleet_retreat_fills_single_reachable_coast(self) -> None:
        graph = load_map("standard")
        fleet = Unit("RUSSIA", UnitKind.FLEET, Location("BOT"))
        movement = PhaseResult(
            resolutions=(),
            new_state=State.of(
                "standard", "F1901M", [Unit("GERMANY", UnitKind.FLEET, Location("BOT"))]
            ),
            dislodgements=(Dislodgement(unit=fleet, dislodger_origin="BAL", by_convoy=False),),
            embattled=frozenset(),
        )
        # From the Gulf of Bothnia only St Petersburg's south coast is
        # reachable: the written coastless retreat resolves to it (mirrors
        # movement coast resolution, DATC 6.B.2).
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {"RUSSIA": [parse_retreat_order("F BOT - STP")]},
            graph=graph,
        )
        assert result.resolution_of("F BOT - STP") is Resolution.SUCCEEDS
        assert Unit("RUSSIA", UnitKind.FLEET, Location("STP", "SC")) in result.new_state.units

    def test_legal_retreats_enumerate_exact_coasts(self) -> None:
        graph = load_map("standard")
        fleet = Unit("RUSSIA", UnitKind.FLEET, Location("BOT"))
        movement = PhaseResult(
            resolutions=(),
            new_state=State.of(
                "standard", "F1901M", [Unit("GERMANY", UnitKind.FLEET, Location("BOT"))]
            ),
            dislodgements=(Dislodgement(unit=fleet, dislodger_origin="BAL", by_convoy=False),),
            embattled=frozenset({"SWE"}),
        )
        options = legal_retreats(STANDARD, graph, movement, movement.dislodgements[0])
        destinations = {str(move.destination) for move in options}
        # Baltic is the dislodger origin, Sweden embattled; Finland, Livonia
        # and STP/SC remain.
        assert destinations == {"FIN", "LVN", "STP/SC"}


class TestRetreatOrdersNameTheirBaseStrength:
    """Write a dislodged unit's own base strength into its retreat order.

    A dislodged ``2A`` and a dislodged ``A`` are different units, so their
    retreat orders must be different text. Retreats fight at strength 0
    (docs/adjudication.tex Sec. 4.3), which makes the base strength irrelevant to the
    contest, but not to the binding: two units dislodged into one province from
    different origins
    have DIFFERENT legal destinations (4.A.5 bars each from its own dislodger's
    origin), and the flat union offered both units the union of the two sets, so
    a player could draft a retreat that was illegal for the unit it would bind
    and lose it.
    """

    def _mixed(self):
        """Dislodge two RED armies out of AAA from two different origins.

        The 1A is dislodged by an attacker from BBB and the 2A by one from
        CCC. Both attackers then stand in AAA, which has capacity 2, so both
        origins are free ground: the only thing keeping BBB off the 1A's list
        is the rule barring a unit from its own dislodger's origin.
        """
        strong = Unit("RED", UnitKind.ARMY, Location("AAA"), strength=2)
        weak = Unit("RED", UnitKind.ARMY, Location("AAA"))
        return mini_movement(
            [army("BLUE", "AAA"), army("BLUE", "AAA")],
            [
                Dislodgement(unit=weak, dislodger_origin="BBB", by_convoy=False),
                Dislodgement(unit=strong, dislodger_origin="CCC", by_convoy=False),
            ],
        )

    def test_each_base_strength_writes_its_own_prefix(self) -> None:
        movement = self._mixed()
        weak, strong = movement.dislodgements
        weak_texts = {format_order(m) for m in legal_retreats(STANDARD, MINI, movement, weak)}
        strong_texts = {format_order(m) for m in legal_retreats(STANDARD, MINI, movement, strong)}
        assert "A AAA - DDD" in weak_texts
        assert "2A AAA - DDD" in strong_texts
        # base strength 1 stays prefixless -- the canonical grammar has no `1A`
        assert not any(t.startswith("1A") for t in weak_texts | strong_texts)
        assert weak_texts.isdisjoint(strong_texts)

    def test_neither_unit_is_offered_the_other_s_forbidden_origin(self) -> None:
        """Offer each dislodged unit only the destinations legal for itself.

        Each unit is barred from its own dislodger's origin (DATC 4.A.5), so
        the two have different legal sets. Offering their union would let a
        player draft ``A AAA - CCC``, which binds the 1A as the first unclaimed
        own occurrence and then fails -- destroying the unit through a row the
        client had shown as legal.
        """
        movement = self._mixed()
        weak, strong = movement.dislodgements
        weak_dests = {str(m.destination) for m in legal_retreats(STANDARD, MINI, movement, weak)}
        strong_dests = {
            str(m.destination) for m in legal_retreats(STANDARD, MINI, movement, strong)
        }
        assert "BBB" not in weak_dests and "CCC" in weak_dests
        assert "CCC" not in strong_dests and "BBB" in strong_dests

    def test_a_written_base_strength_binds_that_unit_and_the_other_survives(self) -> None:
        """Bind a retreat order written ``2A`` to the 2A, past a disbanding twin.

        ``2A AAA - DDD`` moves the strength-2 unit, and the 1A, disbanded by
        its own row, does not consume the 2A's order on the way.
        """
        movement = self._mixed()
        result = adjudicate_retreats(
            STANDARD,
            movement,
            {"RED": [parse_retreat_order("2A AAA - DDD"), parse_retreat_order("Disband A AAA")]},
            graph=MINI,
        )
        assert result.resolution_of("2A AAA - DDD") is Resolution.SUCCEEDS
        assert Unit("RED", UnitKind.ARMY, Location("DDD"), strength=2) in result.new_state.units
        assert not [u for u in result.new_state.units if u.power == "RED" and u.strength == 1]
