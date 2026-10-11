"""Generate Paquette-schema games from engine self-play.

When the human dataset is unavailable, the pipeline still needs games in the
exact schema :mod:`entente.data.paquette` reads. This module plays the engine
and serializes each game to that schema, so the reader -> preprocess -> dataset
path is exercised end to end and the committed test fixture is real facts of
play (generated, not human). The default ``random`` self-play gives diverse,
physically-consistent games (each phase's orders produce the next state);
swapping ``agent`` to ``greedy`` yields a greedy teacher, which is the fallback
if the human dump ever becomes unavailable. These are *not* human games and no
cloning-quality claim rests on them: they verify plumbing and back the smoke
overfit, while real behavioral cloning trains on the Paquette dump (see
docs/setup.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from entente.agents.registry import build_agent
from entente.engine.game import Game
from entente.engine.orders import format_order

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


def _units_by_power(game: Game) -> dict[str, list[str]]:
    units: dict[str, list[str]] = {power: [] for power in game.powers}
    for unit in game.state.units:
        units.setdefault(unit.power, []).append(str(unit))
    return units


def _centers_by_power(game: Game) -> dict[str, list[str]]:
    centers: dict[str, list[str]] = {power: [] for power in game.powers}
    for province, owner in sorted(game.ownership.items()):
        centers.setdefault(owner, []).append(province)
    return centers


def synthesize_game(
    seed: int,
    *,
    map_name: str = "standard",
    ruleset: str = "standard",
    agent: str = "random",
    teacher: str | None = None,
    max_year: int = 1905,
) -> dict[str, Any]:
    """Play one game and return it in the Paquette saved-game schema.

    Each decision phase records the pre-adjudication board (``units``,
    ``centers``) and the orders every power submitted; a terminal ``COMPLETED``
    phase records the final board (its ``centers`` are the value target's
    source). Powers are seeded distinctly for varied but reproducible play.

    ``teacher`` decouples the *recorded* policy from the *played* one: with
    ``agent="random", teacher="greedy"``, ``agent`` drives the trajectory (so
    boards are diverse) while the recorded orders -- the behavioral-cloning labels
    -- come from the deterministic ``teacher`` (so board->order is learnable).
    That pairing is what makes a self-play corpus trainable at all. When
    ``teacher`` is ``None`` the recorded and played orders coincide, giving
    physically consistent games.
    """
    game = Game.new(ruleset, map_name)
    players = {power: build_agent(agent, seed * 7 + i) for i, power in enumerate(game.powers)}
    teachers = (
        players
        if teacher is None
        else {power: build_agent(teacher, seed * 7 + i) for i, power in enumerate(game.powers)}
    )
    phases: list[dict[str, Any]] = []
    while not game.is_over and game.year <= max_year:
        record: dict[str, Any] = {
            "name": game.phase,
            "state": {"units": _units_by_power(game), "centers": _centers_by_power(game)},
            "orders": {},
        }
        for power in game.powers:
            record["orders"][power] = [
                format_order(order) for order in teachers[power].orders(game, power)
            ]
            played = players[power].orders(game, power)
            if played:
                game.submit(power, list(played))
        phases.append(record)
        game.adjudicate()
    phases.append(
        {
            "name": "COMPLETED",
            "state": {"units": _units_by_power(game), "centers": _centers_by_power(game)},
            "orders": {},
        }
    )
    return {"id": f"synthetic-{map_name}-{agent}-{seed}", "map": map_name, "phases": phases}


def synthesize_games(seeds: Iterable[int], **kwargs: Any) -> Iterator[dict[str, Any]]:
    """Play a game per seed (see :func:`synthesize_game` for keyword options)."""
    for seed in seeds:
        yield synthesize_game(seed, **kwargs)


def write_jsonl(games: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write games as JSONL (one game per line). Returns the count written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for game in games:
            handle.write(json.dumps(game))
            handle.write("\n")
            count += 1
    return count
