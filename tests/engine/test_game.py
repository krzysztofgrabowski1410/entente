"""Pin the game facade: its public contract and how it drives the phase machine.

:class:`Game` is the only surface a caller needs for a whole game, so what it
promises -- submit, adjudicate, read the record, ask for legal orders -- is
checked here against the API contract (docs/adjudication.tex Sec. 4.3).
"""

from __future__ import annotations

import pytest

from entente.engine.game import Game, GameError
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import Build, Disband, Hold, Move, Support, Waive, parse_order
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def scripted_dislodgement_game() -> Game:
    """Script two phases that end in a live retreat decision.

    Germany walks into Burgundy, and France throws it out with support, which
    leaves the game waiting on a retreat.
    """
    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("A MUN - BUR")])
    game.adjudicate()
    assert game.phase == "F1901M"
    game.submit("FRANCE", [parse_order("A PAR - BUR"), parse_order("A MAR S A PAR - BUR")])
    game.adjudicate()
    return game


class TestNewGame:
    """What :meth:`Game.new` sets up before any order is written."""

    def test_start_position(self) -> None:
        game = Game.new("standard", "standard")
        assert game.phase == "S1901M"
        assert game.phase_kind is PhaseKind.MOVEMENT
        assert len(game.state.units) == 22
        assert len(game.powers) == 7
        assert game.supply_count("RUSSIA") == 4
        assert sum(game.supply_count(p) for p in game.powers) == 22
        assert game.winner is None and not game.is_over

    def test_unknown_ruleset_raises(self) -> None:
        with pytest.raises(GameError, match="unknown ruleset"):
            Game.new("no-such-rules", "standard")

    def test_start_year_parameter(self) -> None:
        game = Game.new(STANDARD, "standard", start_year=1899)
        assert game.phase == "S1899M"
        assert game.start_year == 1899


class TestSubmission:
    """What the facade accepts, refuses and stores as a pending sheet."""

    def test_unknown_power_raises(self) -> None:
        game = Game.new("standard", "standard")
        with pytest.raises(GameError, match="unknown power"):
            game.submit("ATLANTIS", [])
        with pytest.raises(GameError, match="unknown power"):
            game.legal_orders("ATLANTIS")

    def test_wrong_phase_order_kind_raises(self) -> None:
        game = Game.new("standard", "standard")
        with pytest.raises(GameError, match="not a movement-phase order"):
            game.submit("FRANCE", [Build(UnitKind.ARMY, Location("PAR"))])
        with pytest.raises(GameError, match="not a movement-phase order"):
            game.submit("FRANCE", [Waive()])

    def test_submit_replaces_the_whole_sheet(self) -> None:
        # DATC 4.D.1 choice C: only the latest order set counts.
        game = Game.new("standard", "standard")
        game.submit("FRANCE", [parse_order("A PAR - BUR")])
        game.submit("FRANCE", [parse_order("A MAR - SPA")])
        assert game.pending_sheets == {"FRANCE": ("A MAR - SPA",)}
        record = game.adjudicate()
        assert record.sheets == {"FRANCE": ("A MAR - SPA",)}
        placed = {str(u) for u in game.state.units if u.power == "FRANCE"}
        assert placed == {"A PAR", "A SPA", "F BRE"}


class TestRetreatFlow:
    """How a dislodgement drives the game into and out of a retreat phase."""

    def test_dislodgement_opens_a_retreat_phase(self) -> None:
        game = scripted_dislodgement_game()
        assert game.phase == "F1901R"
        assert game.phase_kind is PhaseKind.RETREAT
        dislodgements = game.pending_dislodgements
        assert [str(d.unit) for d in dislodgements] == ["A BUR"]
        assert dislodgements[0].dislodger_origin == "PAR"
        options = game.legal_orders("GERMANY")
        assert Disband(UnitKind.ARMY, Location("BUR")) in options
        # Paris is the dislodger's origin (overland): not among the moves.
        assert all(not (isinstance(o, Move) and o.destination.province == "PAR") for o in options)
        assert game.legal_orders("FRANCE") == ()

    def test_retreat_resolves_and_winter_is_skipped_when_balanced(self) -> None:
        game = scripted_dislodgement_game()
        game.submit("GERMANY", [Move(UnitKind.ARMY, Location("BUR"), Location("MUN"))])
        record = game.adjudicate()
        assert record.phase == "F1901R"
        assert Unit("GERMANY", UnitKind.ARMY, Location("MUN")) in game.state.units
        # Every power holds exactly its home centers: no winter decisions.
        assert game.phase == "S1902M"

    def test_hold_orders_are_accepted_but_fail_in_retreats(self) -> None:
        game = scripted_dislodgement_game()
        game.submit("GERMANY", [Hold(UnitKind.ARMY, Location("BUR"))])
        record = game.adjudicate()
        assert record.disbanded == (Unit("GERMANY", UnitKind.ARMY, Location("BUR")),)


class TestAutoPhases:
    """Which phases the facade skips when nothing is left for them to do."""

    def test_forced_removal_executes_automatically(self) -> None:
        # France holds one army on non-center soil and owns nothing: the
        # winter removal is forced, executes with empty sheets, and is
        # recorded as auto.
        state = State.of(
            "standard",
            "F1901M",
            [
                Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
                Unit("GERMANY", UnitKind.ARMY, Location("MUN")),
            ],
        )
        game = Game(
            STANDARD, GRAPH, state, {"MUN": "GERMANY"}, Phase(1901, Season.FALL, PhaseKind.MOVEMENT)
        )
        game.adjudicate()
        assert game.phase == "S1902M"
        auto = [record for record in game.history if record.auto]
        assert [record.phase for record in auto] == ["W1901A"]
        assert auto[0].civil_disorder == (Unit("FRANCE", UnitKind.ARMY, Location("BUR")),)
        assert {u.power for u in game.state.units} == {"GERMANY"}

    def test_trapped_dislodgement_disbands_automatically(self) -> None:
        # The 6.H.16 shape through the Game facade: the retreat stage has
        # effects but no decisions, so it auto-executes.
        state = State.of(
            "standard",
            "S1901M",
            [
                Unit("ENGLAND", UnitKind.FLEET, Location("POR")),
                Unit("FRANCE", UnitKind.FLEET, Location("SPA", "SC")),
                Unit("FRANCE", UnitKind.FLEET, Location("MAO")),
            ],
        )
        game = Game(
            STANDARD,
            GRAPH,
            state,
            {"POR": "ENGLAND", "SPA": "FRANCE", "BRE": "FRANCE"},
            Phase(1901, Season.SPRING, PhaseKind.MOVEMENT),
        )
        game.submit(
            "FRANCE", [parse_order("F SPA/SC - POR"), parse_order("F MAO S F SPA/SC - POR")]
        )
        game.adjudicate()
        auto = [record for record in game.history if record.auto]
        assert [record.phase for record in auto] == ["S1901R"]
        assert auto[0].disbanded == (Unit("ENGLAND", UnitKind.FLEET, Location("POR")),)
        assert game.phase == "F1901M"


class TestEndings:
    """How a game finishes, and what it reports once it has."""

    def test_majority_owner_wins_at_the_fall_count(self) -> None:
        centers = [key for key, p in sorted(GRAPH.provinces.items()) if p.is_supply_center]
        # France owns everything except Munich and (until the Fall count
        # claims it) Paris; the win is detected at the ownership update.
        ownership = {center: "FRANCE" for center in centers if center not in ("MUN", "PAR")}
        ownership["MUN"] = "GERMANY"
        state = State.of(
            "standard",
            "F1901M",
            [
                Unit("FRANCE", UnitKind.ARMY, Location("PAR")),
                Unit("GERMANY", UnitKind.ARMY, Location("MUN")),
            ],
        )
        game = Game(STANDARD, GRAPH, state, ownership, Phase(1901, Season.FALL, PhaseKind.MOVEMENT))
        game.adjudicate()
        assert game.winner == "FRANCE"
        assert game.supply_count("FRANCE") >= 18
        assert game.is_over
        with pytest.raises(GameError, match="over"):
            game.adjudicate()
        with pytest.raises(GameError, match="over"):
            game.submit("FRANCE", [])
        assert game.legal_orders("FRANCE") == ()

    def test_empty_board_without_builds_ends_the_game(self) -> None:
        state = State.of("standard", "F1901M", [Unit("FRANCE", UnitKind.ARMY, Location("BUR"))])
        game = Game(STANDARD, GRAPH, state, {}, Phase(1901, Season.FALL, PhaseKind.MOVEMENT))
        game.adjudicate()
        assert game.state.units == ()
        assert game.is_over
        assert game.winner is None


class TestLegalOrders:
    """What the facade offers as the legal orders of a position."""

    def test_start_position_orders_are_submittable_and_valid(self) -> None:
        game = Game.new("standard", "standard")
        for power in game.powers:
            orders = game.legal_orders(power)
            per_unit = {order.location for order in orders if not isinstance(order, Waive)}
            own_units = {u.location for u in game.state.units if u.power == power}
            assert per_unit == own_units
            game.submit(power, list(orders))

    def test_adjustment_orders_shape(self) -> None:
        state = State.of(
            "standard",
            "W1901A",
            [
                Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
                Unit("FRANCE", UnitKind.ARMY, Location("PIC")),
                Unit("GERMANY", UnitKind.ARMY, Location("MUN")),
            ],
        )
        ownership = {"PAR": "FRANCE", "MUN": "GERMANY", "KIE": "GERMANY", "BER": "GERMANY"}
        game = Game(
            STANDARD, GRAPH, state, ownership, Phase(1901, Season.WINTER, PhaseKind.ADJUSTMENT)
        )
        france = game.legal_orders("FRANCE")
        assert all(isinstance(order, Disband) for order in france)
        assert len(france) == 2
        germany = game.legal_orders("GERMANY")
        assert any(isinstance(order, Build) for order in germany)
        assert any(isinstance(order, Waive) for order in germany)
        assert game.build_quota("GERMANY") == 2
        assert game.build_quota("FRANCE") == -1

    def test_twin_stack_offers_same_area_supports(self) -> None:
        # A unit may support the order of a co-located Area-mate, so with a
        # fungible twin in the same Area the menu offers a support of the
        # twin's move out. The twin's value equals the supporter's, but a
        # subject reference never denotes the supporter itself, so the value
        # denotes the other occurrence. A lone unit has no Area-mate to name,
        # so nothing standing in its own province is offered as a subject --
        # what it may back from inside the Area are the orders aimed at that
        # Area, whose subjects stand outside it.
        graph = load_map("beleriand")
        phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
        twins = State.of(
            "beleriand",
            "S25M",
            [
                Unit("MORGOTH", UnitKind.ARMY, Location("ANF")),
                Unit("MORGOTH", UnitKind.ARMY, Location("ANF")),
            ],
        )
        game = Game(RULESETS["beleriand"], graph, twins, {}, phase)
        assert any(
            isinstance(order, Support)
            and isinstance(order.subject, Move)
            and order.subject.location.province == "ANF"
            and order.subject.destination.province == "LOT"
            for order in game.legal_orders("MORGOTH")
        )
        alone = State.of(
            "beleriand",
            "S25M",
            [
                Unit("MORGOTH", UnitKind.ARMY, Location("ANF")),
                Unit("DORIATH", UnitKind.ARMY, Location("LOT")),
            ],
        )
        game_alone = Game(RULESETS["beleriand"], graph, alone, {}, phase)
        menu = game_alone.legal_orders("MORGOTH")
        assert not any(
            isinstance(order, Support) and order.subject.location.province == "ANF"
            for order in menu
        )
        assert any(
            isinstance(order, Support)
            and isinstance(order.subject, Move)
            and order.subject.location.province == "LOT"
            and order.subject.destination.province == "ANF"
            for order in menu
        )
