"""RandomAgent tests: the determinism contract and sheet well-formedness."""

from __future__ import annotations

from entente.agents import Agent, RandomAgent
from entente.engine.game import Game
from entente.engine.orders import Build, Disband, Waive


def play(seed: int, steps: int) -> Game:
    game = Game.new("standard", "standard")
    agent = RandomAgent(seed)
    for _ in range(steps):
        if game.is_over:
            break
        for power in game.powers:
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
    return game


def test_satisfies_the_agent_protocol() -> None:
    assert isinstance(RandomAgent(0), Agent)


def test_identical_seeds_reproduce_identical_games() -> None:
    first, second = play(7, 6), play(7, 6)
    assert first.state == second.state
    assert first.ownership == second.ownership
    assert [record.phase for record in first.history] == [record.phase for record in second.history]
    assert [record.sheets for record in first.history] == [
        record.sheets for record in second.history
    ]


def test_sheets_do_not_depend_on_call_order() -> None:
    game = Game.new("standard", "standard")
    agent = RandomAgent(3)
    forward = {power: tuple(agent.orders(game, power)) for power in game.powers}
    backward = {power: tuple(agent.orders(game, power)) for power in reversed(game.powers)}
    assert forward == backward


def test_different_seeds_diverge() -> None:
    game = Game.new("standard", "standard")
    sheets = {seed: tuple(RandomAgent(seed).orders(game, "FRANCE")) for seed in range(8)}
    assert len(set(sheets.values())) > 1


def test_movement_sheets_order_every_unit_once() -> None:
    game = Game.new("standard", "standard")
    sheet = RandomAgent(11).orders(game, "RUSSIA")
    ordered = [order.location for order in sheet if not isinstance(order, Waive)]
    own = [u.location for u in game.state.units if u.power == "RUSSIA"]
    assert sorted(ordered, key=str) == sorted(own, key=str)


def test_adjustment_sheets_fit_the_quota() -> None:
    # Drive a real game into a winter with builds by scripting a capture:
    # Germany sails to Holland and owns it at the Fall count.
    from entente.engine.orders import parse_order

    game = Game.new("standard", "standard")
    game.submit("GERMANY", [parse_order("F KIE - HOL")])
    game.adjudicate()  # S1901M
    game.adjudicate()  # F1901M (all hold); ownership update gives Germany 4
    assert game.phase == "W1901A"
    assert game.build_quota("GERMANY") == 1
    for seed in range(10):
        sheet = RandomAgent(seed).orders(game, "GERMANY")
        builds = [order for order in sheet if isinstance(order, Build)]
        waives = [order for order in sheet if isinstance(order, Waive)]
        disbands = [order for order in sheet if isinstance(order, Disband)]
        assert len(builds) + len(waives) == 1
        assert not disbands
        provinces = [order.location.province for order in builds]
        assert len(provinces) == len(set(provinces))
