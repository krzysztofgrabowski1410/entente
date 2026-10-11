"""Measure adjudicator throughput, resolver against enumerator.

The design target is at least a thousand movement adjudications per second per
core, which is what makes self-play and search affordable.

Run: ``uv run pytest benchmarks/bench_adjudicator.py --benchmark-only``.
``testpaths`` excludes ``benchmarks/`` from a plain ``uv run pytest``, so this
never slows the main gate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from entente.agents.random_agent import RandomAgent
from entente.engine.adjudicator import EngineName, adjudicate
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.phase import PhaseKind
from entente.engine.rules import STANDARD

if TYPE_CHECKING:
    from entente.engine.orders import Order
    from entente.engine.state import State

GRAPH = load_map("standard")
ENGINES: tuple[EngineName, ...] = ("resolver", "enumerator")


def _random_movement_position(
    seed: int, *, min_phases: int
) -> tuple[State, dict[str, list[Order]]]:
    """Build a random but legal movement-phase position, deterministic in ``seed``.

    The game is played ``min_phases`` phases from the start, then as many more
    as it takes to land back on a movement phase; retreats and adjustments are
    skipped as positions to return, not as phases played, so the sheet that
    comes back is always Hold, Move, Support or Convoy. Timing this rather
    than an all-holds sheet exercises the bounces, supports and convoys that
    real self-play spends its time on.
    """
    game = Game.new("standard", "standard")
    agent = RandomAgent(seed)
    phases = 0
    while phases < min_phases or game.phase_kind is not PhaseKind.MOVEMENT:
        for power in game.powers:
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
        phases += 1
    sheets = {power: list(agent.orders(game, power)) for power in game.powers}
    return game.state, sheets


POSITIONS: dict[str, tuple[State, dict[str, list[Order]]]] = {
    "opening": _random_movement_position(seed=12345, min_phases=0),
    "midgame": _random_movement_position(seed=12345, min_phases=6),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("position", sorted(POSITIONS))
def test_bench_adjudication(benchmark: Any, position: str, engine: EngineName) -> None:
    """Time one movement adjudication of ``position`` under ``engine``."""
    state, sheets = POSITIONS[position]
    benchmark.group = f"adjudicate/{position}"
    benchmark.extra_info["engine"] = engine
    benchmark(lambda: adjudicate(STANDARD, state, sheets, graph=GRAPH, engine=engine))
