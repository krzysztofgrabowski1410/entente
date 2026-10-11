"""Animated-GIF rasterizer for :mod:`entente.viz.svg` scenes.

Maps the exact scene primitives the SVG serializer draws onto a Pillow canvas
-- one geometry, two backends -- and assembles per-phase frames of a played
:class:`~entente.engine.game.Game` into a small looping GIF
(``scripts/make_demo_gif.py`` regenerates the committed ``docs/media/`` demo).

Requires Pillow from the ``viz`` dependency group; the default environment
never imports this module (tests are ``importorskip``-guarded, mirroring the
``ml`` group's treatment).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from entente.engine.game import Game
from entente.engine.state import State
from entente.viz.svg import (
    CANVAS,
    Circle,
    Line,
    Polygon,
    Scene,
    Text,
    build_board_scene,
    build_record_scene,
)

__all__ = ["game_frames", "scene_to_image", "write_gif"]


def _font(size: float) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a scalable default font, or the bitmap fallback on an older Pillow.

    Which one it is never matters to a test: the geometry checks depend on no
    glyph metric.
    """
    try:
        return ImageFont.load_default(size=max(size, 6.0))
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _dashed(
    draw: ImageDraw.ImageDraw,
    a: tuple[float, float],
    b: tuple[float, float],
    dash: tuple[float, ...],
    color: str,
    width: int,
) -> None:
    """Draw a dashed segment by hand, since a Pillow line has no dash pattern."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx / length, dy / length
    pattern = list(dash) or [length]
    pos, index, pen_down = 0.0, 0, True
    while pos < length:
        step = min(pattern[index % len(pattern)], length - pos)
        if pen_down:
            draw.line(
                [
                    (a[0] + ux * pos, a[1] + uy * pos),
                    (a[0] + ux * (pos + step), a[1] + uy * (pos + step)),
                ],
                fill=color,
                width=width,
            )
        pos += step
        index += 1
        pen_down = not pen_down


def scene_to_image(scene: Scene, *, size: int = 720, supersample: int = 2) -> Image.Image:
    """Rasterize one scene into an image of ``size`` pixels square.

    It is drawn at ``supersample`` times the target size and then downscaled,
    which is the whole antialiasing strategy: Pillow has none of its own.
    """
    s = size * supersample / CANVAS
    image = Image.new("RGB", (size * supersample, size * supersample), "#ffffff")
    draw = ImageDraw.Draw(image)
    for el in scene:
        if isinstance(el, Line):
            width = max(1, round(el.width * s))
            a, b = (el.x1 * s, el.y1 * s), (el.x2 * s, el.y2 * s)
            if el.dash:
                _dashed(draw, a, b, tuple(d * s for d in el.dash), el.color, width)
            else:
                draw.line([a, b], fill=el.color, width=width)
        elif isinstance(el, Circle):
            box = (
                el.cx * s - el.r * s,
                el.cy * s - el.r * s,
                el.cx * s + el.r * s,
                el.cy * s + el.r * s,
            )
            draw.ellipse(
                box,
                fill=el.fill,
                outline=el.stroke,
                width=max(1, round(el.stroke_width * s)) if el.stroke else 0,
            )
        elif isinstance(el, Polygon):
            draw.polygon(
                [(x * s, y * s) for x, y in el.points],
                fill=el.fill,
                outline=el.stroke,
                width=max(1, round(el.stroke_width * s)) if el.stroke else 0,
            )
        else:
            _draw_text(draw, el, s)
    if supersample > 1:
        image = image.resize((size, size), Image.LANCZOS)
    return image


def _draw_text(draw: ImageDraw.ImageDraw, el: Text, s: float) -> None:
    """Draw one text element, anchored the way the SVG serializer anchors it."""
    font = _font(el.size * s)
    # Manual anchoring via measured bounds: works for bitmap and FreeType
    # fonts alike (Pillow's anchor= needs FreeType). el.y is the text
    # *baseline* in SVG; approximate it by aligning the glyph box bottom.
    bbox = draw.textbbox((0, 0), el.text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = el.x * s - {"start": 0.0, "middle": w / 2.0, "end": float(w)}[el.anchor]
    y = el.y * s - h
    draw.text((x - bbox[0], y - bbox[1]), el.text, fill=el.color, font=font)
    if el.bold:  # cheap emphasis: 1px double-strike, font weights need real faces
        draw.text((x - bbox[0] + 1, y - bbox[1]), el.text, fill=el.color, font=font)


def game_frames(game: Game, *, size: int = 720) -> list[Image.Image]:
    """Render one frame per executed phase of a finished or stopped game.

    The sequence is the opening board, then each :class:`PhaseRecord` drawn as
    its pre-phase state with outcome-coloured order arrows, then the final
    position.
    """
    if not game.history:
        raise ValueError("game has no executed phases to render")
    graph = game.graph
    # The opening position is reconstructible by construction: histories are
    # Game.new-rooted (the save format's own scope), so a fresh Game *is* frame 0.
    fresh = Game.new(game.ruleset, graph.name, start_year=game.start_year)
    initial_state, initial_ownership = fresh.state, dict(fresh.ownership)
    frames = [
        scene_to_image(
            build_board_scene(
                graph, initial_state, initial_ownership, caption=f"{initial_state.phase} -- start"
            ),
            size=size,
        )
    ]
    prev_state: State = initial_state
    prev_ownership: Mapping[str, str] = initial_ownership
    for record in game.history:
        scene = build_record_scene(graph, record, prev_state, prev_ownership)
        frames.append(scene_to_image(scene, size=size))
        prev_state, prev_ownership = record.state_after, record.ownership_after
    last = game.history[-1]
    caption = f"{last.phase} -- result" + (f" * winner: {game.winner}" if game.winner else "")
    frames.append(
        scene_to_image(
            build_board_scene(graph, last.state_after, last.ownership_after, caption=caption),
            size=size,
        )
    )
    return frames


def write_gif(
    frames: Sequence[Image.Image],
    path: str | Path,
    *,
    duration_ms: int = 900,
    hold_last_ms: int = 2500,
) -> Path:
    """Assemble frames into a looping palette GIF, returning the written path."""
    if not frames:
        raise ValueError("no frames to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    durations = [duration_ms] * (len(quantized) - 1) + [hold_last_ms]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return path
