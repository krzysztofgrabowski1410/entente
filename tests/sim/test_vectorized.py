"""entente.sim.vectorized: lockstep batching against the plain single-game loop.

The PolicyCallback contract belongs here too: what a callback receives, and what
the runner does with what it returns.
"""

from __future__ import annotations

import pytest

from entente.agents.random_agent import RandomAgent
from entente.engine.game import Game
from entente.engine.orders import AnyOrder
from entente.sim.vectorized import DecisionRequest, LockstepRunner, PolicyCallback, new_games


def _agent_policy(base_seed: int) -> PolicyCallback:
    """Give each game index its own RandomAgent stream.

    Batchmates must not affect each other's outcome, so no two indices may share
    a seed.
    """

    def policy(requests: list[DecisionRequest]) -> list[list[AnyOrder]]:
        return [
            list(RandomAgent(base_seed + request.index).orders(request.game, request.power))
            for request in requests
        ]

    return policy


def _play_individually(seed: int, *, max_year: int) -> Game:
    game = Game.new("standard", "standard")
    agent = RandomAgent(seed)
    while not game.is_over and game.year <= max_year:
        for power in game.powers:
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
    return game


def test_lockstep_matches_individually_run_games() -> None:
    """Running M games together must be identical to running each one alone.

    Lockstep batching is a scheduling change, never a rules change.
    """
    base_seed = 100
    max_year = 1902
    games = new_games(3)
    runner = LockstepRunner(games, _agent_policy(base_seed), max_year=max_year)
    runner.run()

    for index, game in enumerate(runner.games):
        expected = _play_individually(base_seed + index, max_year=max_year)
        assert game.state == expected.state
        assert game.ownership == expected.ownership
        assert [r.phase for r in game.history] == [r.phase for r in expected.history]
        assert [r.sheets for r in game.history] == [r.sheets for r in expected.history]


def test_step_returns_false_once_every_game_is_done() -> None:
    games = new_games(2)
    runner = LockstepRunner(games, _agent_policy(1), max_year=1901)
    ticks = 0
    while runner.step():
        ticks += 1
        assert ticks < 100  # guard against a genuine infinite loop
    assert runner.step() is False
    assert all(game.is_over or game.year > 1901 for game in runner.games)


def test_run_reports_the_tick_count_and_is_idempotent_once_done() -> None:
    games = new_games(2)
    runner = LockstepRunner(games, _agent_policy(2), max_year=1901)
    ticks = runner.run()
    assert ticks > 0
    assert runner.run() == 0  # already exhausted, no-op


def test_run_respects_max_ticks() -> None:
    games = new_games(2)
    runner = LockstepRunner(games, _agent_policy(3), max_year=1910)
    ticks = runner.run(max_ticks=1)
    assert ticks == 1
    assert all(len(game.history) > 0 for game in runner.games)


def test_decision_requests_carry_precomputed_legal_orders() -> None:
    checked: list[DecisionRequest] = []

    def recording_policy(requests: list[DecisionRequest]) -> list[list[AnyOrder]]:
        for request in requests:
            assert request.legal_orders == request.game.legal_orders(request.power)
            checked.append(request)
        return [list(RandomAgent(9).orders(r.game, r.power)) for r in requests]

    games = new_games(1)
    LockstepRunner(games, recording_policy, max_year=1901).step()

    assert checked  # the policy actually ran and the check above executed


def test_policy_sheet_count_mismatch_raises() -> None:
    def broken_policy(requests: list[DecisionRequest]) -> list[list[AnyOrder]]:
        return []  # wrong length whenever there is at least one request

    runner = LockstepRunner(new_games(1), broken_policy)
    with pytest.raises(ValueError, match="policy returned"):
        runner.step()


def test_new_games_builds_independent_fresh_games() -> None:
    games = new_games(3)
    assert len(games) == 3
    assert all(game.phase == "S1901M" for game in games)
    assert games[0] is not games[1]
