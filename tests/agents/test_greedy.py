"""GreedyAgent: determinism, sheet well-formedness, and the scoring primitives.

The first half mirrors RandomAgent's tests. The second checks the heuristic's
scoring primitives in isolation on hand-built positions with contrived
``values``: a real board position, so legality is genuine, but scores chosen so
the *intended* winner is unambiguous. Using the true diffused values instead
would mean hand-predicting a one-hop average over the full 75-province standard
map, and a test that fragile pins arithmetic rather than behaviour.
"""

from __future__ import annotations

from entente.agents import Agent, GreedyAgent
from entente.agents.greedy import (
    DEFENSE_WEIGHT,
    DISBAND_SCORE,
    RISK_WEIGHT,
    _diffused_values,
    _group_by_unit,
)
from entente.engine.game import Game
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import Build, Disband, Hold, Move, Support, Waive, parse_order
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def _movement_game(units: list[Unit], ownership: dict[str, str] | None = None) -> Game:
    state = State.of("standard", "S1901M", units)
    return Game(
        STANDARD, GRAPH, state, ownership or {}, Phase(1901, Season.SPRING, PhaseKind.MOVEMENT)
    )


def play(seed: int, steps: int) -> Game:
    game = Game.new("standard", "standard")
    agent = GreedyAgent(seed)
    for _ in range(steps):
        if game.is_over:
            break
        for power in game.powers:
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
    return game


# -- protocol, determinism, sheet shape (mirrors test_random_agent.py) ---------


def test_satisfies_the_agent_protocol() -> None:
    assert isinstance(GreedyAgent(0), Agent)


def test_identical_games_are_reproduced_exactly() -> None:
    first, second = play(0, 8), play(0, 8)
    assert first.state == second.state
    assert [r.sheets for r in first.history] == [r.sheets for r in second.history]


def test_different_seeds_produce_the_same_sheet() -> None:
    """Pin that there is no randomness anywhere in this agent.

    ``seed`` exists only so the factory signature can stay
    ``Callable[[int], Agent]``, uniform across every agent.
    """
    game = Game.new("standard", "standard")
    sheets = {seed: tuple(GreedyAgent(seed).orders(game, "FRANCE")) for seed in range(5)}
    assert len(set(sheets.values())) == 1


def test_sheets_do_not_depend_on_call_order() -> None:
    game = Game.new("standard", "standard")
    agent = GreedyAgent(0)
    forward = {power: tuple(agent.orders(game, power)) for power in game.powers}
    backward = {power: tuple(agent.orders(game, power)) for power in reversed(game.powers)}
    assert forward == backward


def test_movement_sheets_order_every_unit_once() -> None:
    game = Game.new("standard", "standard")
    sheet = GreedyAgent(0).orders(game, "RUSSIA")
    ordered = [order.location for order in sheet if not isinstance(order, Waive)]
    own = [u.location for u in game.state.units if u.power == "RUSSIA"]
    assert sorted(ordered, key=str) == sorted(own, key=str)


def test_adjustment_sheets_fit_the_quota() -> None:
    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("F KIE - HOL")])
    game.adjudicate()  # S1901M
    game.adjudicate()  # F1901M (all hold); ownership update gives Germany 4
    assert game.phase == "W1901A"
    assert game.build_quota("GERMANY") == 1

    sheet = GreedyAgent(0).orders(game, "GERMANY")

    builds = [o for o in sheet if isinstance(o, Build)]
    waives = [o for o in sheet if isinstance(o, Waive)]
    assert len(builds) + len(waives) == 1
    provinces = [o.location.province for o in builds]
    assert len(provinces) == len(set(provinces))


def test_full_game_runs_to_completion_without_error() -> None:
    for seed in range(3):
        game = play(seed, 60)
        assert len(game.history) > 0


# -- scoring primitives, isolated with contrived values ------------------------


def test_prefers_a_high_value_reachable_destination_over_holding() -> None:
    game = _movement_game(
        [Unit("FRANCE", UnitKind.ARMY, Location("PIC"))],
        {"PAR": "FRANCE", "BRE": "FRANCE", "MAR": "FRANCE"},
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["BEL"] = 100.0  # an irresistible, unambiguous target

    order, score = GreedyAgent(0)._best_independent_action(
        game, "FRANCE", values, game.legal_orders("FRANCE")
    )

    assert order == Move(UnitKind.ARMY, Location("PIC"), Location("BEL"))
    assert score == 100.0


def test_penalizes_a_destination_with_an_adjacent_enemy_unit() -> None:
    # BEL is adjacent to both PIC (the mover's origin) and HOL (the enemy).
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("PIC")),
            Unit("GERMANY", UnitKind.ARMY, Location("HOL")),
        ]
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["BEL"] = 1.0

    order, score = GreedyAgent(0)._best_independent_action(
        game, "FRANCE", values, game.legal_orders("FRANCE")
    )

    assert order == Move(UnitKind.ARMY, Location("PIC"), Location("BEL"))
    assert score == 1.0 - RISK_WEIGHT


def test_defends_an_owned_single_defender_center_under_threat() -> None:
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("PAR")),
            Unit("GERMANY", UnitKind.ARMY, Location("PIC")),  # adjacent to PAR
        ],
        {"PAR": "FRANCE"},
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)  # every move is worthless; only defense matters

    order, score = GreedyAgent(0)._best_independent_action(
        game, "FRANCE", values, game.legal_orders("FRANCE")
    )

    assert order == Hold(UnitKind.ARMY, Location("PAR"))
    assert score == DEFENSE_WEIGHT  # one threat, no other defender


def test_no_defense_bonus_for_a_center_that_is_not_under_threat() -> None:
    game = _movement_game([Unit("FRANCE", UnitKind.ARMY, Location("PAR"))], {"PAR": "FRANCE"})
    values = dict.fromkeys(GRAPH.provinces, 0.0)

    order, score = GreedyAgent(0)._best_independent_action(
        game, "FRANCE", values, game.legal_orders("FRANCE")
    )

    assert order == Hold(UnitKind.ARMY, Location("PAR"))
    assert score == 0.0


def test_never_considers_a_via_convoy_move_in_the_independent_pass() -> None:
    """A lone army cannot arrange its own convoy.

    Even where a convoy route to a high-value destination exists, the
    independent pass must not pick it: nothing has ordered the fleet to carry.
    """
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("PIC")),
            Unit("FRANCE", UnitKind.FLEET, Location("ENG")),  # capable of convoying PIC
        ]
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["WAL"] = 100.0  # only reachable from PIC via convoy through ENG
    pic_options = [o for o in game.legal_orders("FRANCE") if str(o.location) == "PIC"]
    assert any(
        isinstance(o, Move) and o.via_convoy and o.destination.province == "WAL"
        for o in pic_options
    )  # the tempting convoy route really is on offer

    order, _ = GreedyAgent(0)._best_independent_action(game, "FRANCE", values, pic_options)

    assert not (isinstance(order, Move) and order.via_convoy)


# -- support coordination (pass 2) ---------------------------------------------


def test_redirects_a_weak_unit_to_support_a_contested_friendly_move() -> None:
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
            Unit("FRANCE", UnitKind.ARMY, Location("RUH")),
            Unit("GERMANY", UnitKind.ARMY, Location("TYR")),  # adjacent to MUN: contests it
        ]
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["MUN"] = 1.0
    by_unit = _group_by_unit(game.legal_orders("FRANCE"))
    bur_move = Move(UnitKind.ARMY, Location("BUR"), Location("MUN"))
    chosen: dict[str, object] = {"BUR": bur_move, "RUH": Hold(UnitKind.ARMY, Location("RUH"))}
    scores = {"BUR": 0.6, "RUH": 0.0}  # RUH's own hold is worthless; BUR's attack is not

    GreedyAgent(0)._coordinate_supports(game, "FRANCE", values, by_unit, chosen, scores)  # type: ignore[arg-type]

    assert chosen["BUR"] == bur_move  # 0.6 already beats supporting RUH's (worthless) hold
    assert chosen["RUH"] == Support(UnitKind.ARMY, Location("RUH"), bur_move)


def test_does_not_redirect_toward_an_uncontested_target() -> None:
    """With no German unit adjacent, MUN is uncontested and support buys nothing.

    RUH should keep its own action.
    """
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
            Unit("FRANCE", UnitKind.ARMY, Location("RUH")),
        ]
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["MUN"] = 1.0
    by_unit = _group_by_unit(game.legal_orders("FRANCE"))
    bur_move = Move(UnitKind.ARMY, Location("BUR"), Location("MUN"))
    ruh_hold = Hold(UnitKind.ARMY, Location("RUH"))
    chosen: dict[str, object] = {"BUR": bur_move, "RUH": ruh_hold}
    scores = {"BUR": 0.6, "RUH": 0.0}

    GreedyAgent(0)._coordinate_supports(game, "FRANCE", values, by_unit, chosen, scores)  # type: ignore[arg-type]

    assert chosen["RUH"] == ruh_hold


def test_does_not_redirect_when_the_units_own_action_already_scores_higher() -> None:
    game = _movement_game(
        [
            Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
            Unit("FRANCE", UnitKind.ARMY, Location("RUH")),
            Unit("GERMANY", UnitKind.ARMY, Location("TYR")),
        ]
    )
    values = dict.fromkeys(GRAPH.provinces, 0.0)
    values["MUN"] = 1.0
    by_unit = _group_by_unit(game.legal_orders("FRANCE"))
    bur_move = Move(UnitKind.ARMY, Location("BUR"), Location("MUN"))
    ruh_move = Move(UnitKind.ARMY, Location("RUH"), Location("KIE"))
    chosen: dict[str, object] = {"BUR": bur_move, "RUH": ruh_move}
    scores = {"BUR": 0.6, "RUH": 5.0}  # RUH's own move is already excellent

    GreedyAgent(0)._coordinate_supports(game, "FRANCE", values, by_unit, chosen, scores)  # type: ignore[arg-type]

    assert chosen["RUH"] == ruh_move


# -- retreats -------------------------------------------------------------------


def test_retreats_rather_than_disbanding_when_a_retreat_is_legal() -> None:
    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("A MUN - BUR")])
    game.adjudicate()  # S1901M
    game.submit("FRANCE", [parse_order("A PAR - BUR"), parse_order("A MAR S A PAR - BUR")])
    game.adjudicate()  # F1901M: Germany's Munich army is dislodged from Burgundy
    assert game.phase == "F1901R"

    sheet = GreedyAgent(0).orders(game, "GERMANY")

    assert len(sheet) == 1
    assert isinstance(sheet[0], Move)  # never Disband while a retreat square is open


def test_disband_score_is_below_any_real_destination() -> None:
    assert DISBAND_SCORE < 0.0


# -- forced removal ---------------------------------------------------------------


def test_removal_disbands_the_lowest_value_units_first() -> None:
    # France owns only PAR and MAR (2 centers) but has 3 units on the board
    # -- PAR, MAR, and BUR -- forcing one removal. Values here come from the
    # real diffusion rule, not a contrived override (this is checking the
    # ranking end-to-end); empirically (see the module docstring's
    # diffusion rule) MAR < PAR < BUR for this exact position, so MAR must
    # be the one disbanded.
    ownership = {"PAR": "FRANCE", "MAR": "FRANCE"}
    units = [
        Unit("FRANCE", UnitKind.ARMY, Location("PAR")),
        Unit("FRANCE", UnitKind.ARMY, Location("MAR")),
        Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
    ]
    state = State.of("standard", "W1901A", units)
    game = Game(STANDARD, GRAPH, state, ownership, Phase(1901, Season.WINTER, PhaseKind.ADJUSTMENT))
    assert game.build_quota("FRANCE") == -1
    values = _diffused_values(GRAPH, ownership, "FRANCE")
    assert values["MAR"] < values["PAR"] < values["BUR"]  # the ranking this test relies on

    sheet = GreedyAgent(0).orders(game, "FRANCE")

    assert len(sheet) == 1
    assert isinstance(sheet[0], Disband)
    assert sheet[0].location.province == "MAR"
