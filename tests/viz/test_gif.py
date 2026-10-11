"""GIF rasterizer tests, which need Pillow from the ``viz`` dependency group.

They skip in the default environment, exactly like the ``ml``-marked torch
suites.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PIL = pytest.importorskip("PIL", reason="viz dependency group not installed")

from PIL import Image  # noqa: E402

from entente.agents.registry import build_agent  # noqa: E402
from entente.engine.game import Game  # noqa: E402
from entente.viz.gif import game_frames, scene_to_image, write_gif  # noqa: E402
from entente.viz.svg import build_board_scene  # noqa: E402


def _tiny_game(phases: int = 3) -> Game:
    game = Game.new("standard", "standard")
    agents = {power: build_agent("random", seed=7) for power in game.powers}
    for _ in range(phases):
        if game.is_over:
            break
        for power, agent in agents.items():
            game.submit(power, list(agent.orders(game, power)))
        game.adjudicate()
    return game


def test_scene_rasterizes_at_requested_size() -> None:
    game = Game.new("standard", "standard")
    scene = build_board_scene(game.graph, game.state, game.ownership, caption="S1901M")
    image = scene_to_image(scene, size=256)
    assert image.size == (256, 256)
    # something was actually drawn: not a blank white canvas
    assert len(image.getcolors(maxcolors=1 << 20)) > 3


def test_game_frames_cover_every_phase_plus_bookends() -> None:
    game = _tiny_game(phases=2)
    frames = game_frames(game, size=192)
    assert len(frames) == len(game.history) + 2  # start + records + result
    assert all(f.size == (192, 192) for f in frames)


def test_write_gif_roundtrip(tmp_path: Path) -> None:
    game = _tiny_game(phases=2)
    frames = game_frames(game, size=160)
    out = write_gif(frames, tmp_path / "demo.gif", duration_ms=200)
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == len(frames)
        assert gif.info.get("loop", None) == 0


def test_write_gif_is_deterministic(tmp_path: Path) -> None:
    """Two identical renders must encode byte for byte the same.

    That is what makes the committed demo artifact reproducible from its script
    and seed alone.
    """
    game = _tiny_game(phases=1)
    frames = game_frames(game, size=128)
    a = write_gif(frames, tmp_path / "a.gif")
    b = write_gif(frames, tmp_path / "b.gif")
    assert a.read_bytes() == b.read_bytes()


def test_empty_frames_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_gif([], tmp_path / "x.gif")
