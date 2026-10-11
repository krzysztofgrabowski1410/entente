"""Regenerate the committed README demo GIF (docs/media/demo.gif).

A seeded greedy self-play game on the standard map, rendered one frame per
executed phase by entente.viz (movement frames carry outcome-colored order
arrows; the closing frame holds). Deterministic: same seed, same bytes --
tests/viz/test_gif.py pins the encoder determinism this relies on.

Run:  uv run --group viz python scripts/make_demo_gif.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from entente.agents.registry import build_agent
from entente.engine.game import Game
from entente.viz.gif import game_frames, write_gif


def main() -> None:
    """Play a seeded game and write its frames out as the demo GIF."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--agent", default="greedy", help="registry agent name for all powers")
    parser.add_argument("--max-year", type=int, default=1907)
    parser.add_argument("--size", type=int, default=560, help="frame edge in pixels")
    parser.add_argument("--duration-ms", type=int, default=900, help="per-frame delay")
    parser.add_argument("--out", type=Path, default=Path("docs/media/demo.gif"))
    args = parser.parse_args()

    game = Game.new("standard", "standard")
    agents = {power: build_agent(args.agent, seed=args.seed) for power in game.powers}
    while not game.is_over and game.year <= args.max_year:
        for power, agent in agents.items():
            game.submit(power, list(agent.orders(game, power)))
        game.adjudicate()

    frames = game_frames(game, size=args.size)
    out = write_gif(frames, args.out, duration_ms=args.duration_ms)
    print(
        f"{out} -- {len(frames)} frames ({len(game.history)} phases, "
        f"final year {game.year}), {out.stat().st_size / 1024:.0f} KiB"
    )


if __name__ == "__main__":
    main()
