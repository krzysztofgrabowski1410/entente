"""Schematic SVG board renderer, driven by map data alone.

The board is drawn as a node diagram: one marker per province at a schematic
centroid stored in the map's JSON data file (``"centroids"``), adjacency as a
light underlay, then supply centers, units, and per-order move, support and
convoy arrows on top. Like every other figure in a map file, the centroids are
hand-assigned from the real-world geography of the named places rather than
converted from another engine, and the integrity tests check they are one per
province, in bounds and separated.

This is *not* a territorial map, by design: the data holds no province
outlines, and a node schematic is enough for the demo artifact this feeds.

Rendering is split in two so the GIF rasterizer shares every drawing decision:
``build_board_scene`` produces a plain list of geometric primitives (the
"scene"), and ``scene_to_svg`` serializes one scene to an SVG string.
:mod:`entente.viz.gif` maps the same primitives onto a Pillow canvas instead.
Everything here is stdlib-only and works in the default environment.

Order arrows cover the standard order union (Move / Support / Convoy; Holds
draw nothing). Variant-specific order types (Beleriand's Dislodge/Bar) render
as plain unit markers only -- the shipped centroid data is standard-map, so a
variant frame renderer would have nothing to draw on today.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files

from entente.engine.adjudicator.common import Resolution
from entente.engine.game import PhaseRecord
from entente.engine.map import data_packages
from entente.engine.map.graph import MapGraph
from entente.engine.map.types import Terrain
from entente.engine.orders import (
    Convoy,
    GameOrder,
    Move,
    Support,
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
)
from entente.engine.state import State

CANVAS = 100.0
"""Centroid coordinate space; ``scene_to_svg``/``scene_to_image`` scale it."""

# Classic board-inspired power colors, keyed by the standard map's power names;
# unknown powers (variant maps) get evenly spaced hues from _fallback_color.
_STANDARD_COLORS = {
    "AUSTRIA": "#c0392b",
    "ENGLAND": "#1f3d7a",
    "FRANCE": "#5b9bd5",
    "GERMANY": "#4d4d4d",
    "ITALY": "#2e8b57",
    "RUSSIA": "#7d5ba6",
    "TURKEY": "#c9a227",
}
_NEUTRAL_SC = "#f5f5f0"
_SEA_FILL = "#e8f1f8"
_SEA_EDGE = "#b9d3e6"
_LAND_FILL = "#f1ecdf"
_LAND_EDGE = "#cbc3ad"
_ADJACENCY = "#dddad2"
_OK = "#1a7f37"
_FAIL = "#c0392b"
_INK = "#3a3a3a"


@dataclass(frozen=True, slots=True)
class Line:
    """Straight segment; ``dash`` is an on/off pattern in canvas units."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    color: str
    dash: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class Circle:
    """Filled disc, optionally outlined."""

    cx: float
    cy: float
    r: float
    fill: str | None
    stroke: str | None = None
    stroke_width: float = 0.0
    tag: str = ""
    """Free-form marker (e.g. the owning power) exposed as ``data-tag`` in the
    SVG so tests can count semantic elements instead of parsing geometry."""


@dataclass(frozen=True, slots=True)
class Polygon:
    """Closed shape through ``points``, in canvas units."""

    points: tuple[tuple[float, float], ...]
    fill: str | None
    stroke: str | None = None
    stroke_width: float = 0.0
    tag: str = ""


@dataclass(frozen=True, slots=True)
class Text:
    """A run of text; ``y`` is its baseline, as in SVG."""

    x: float
    y: float
    text: str
    size: float
    color: str
    anchor: str = "middle"  # "start" | "middle" | "end"
    bold: bool = False


Element = Line | Circle | Polygon | Text
Scene = tuple[Element, ...]


def load_centroids(map_name: str) -> dict[str, tuple[float, float]]:
    """Read the ``centroids`` block of a packaged map's data file.

    Raises ``FileNotFoundError`` for an unknown map and ``KeyError`` for a map
    that ships no centroids (only ``standard`` does today).
    """
    for package in data_packages():
        resource = files(package).joinpath("data").joinpath(f"{map_name}.json")
        if not resource.is_file():
            continue
        data = json.loads(resource.read_text(encoding="utf-8"))
        if "centroids" not in data:
            raise KeyError(f"map {map_name!r} has no centroids block")
        return {key: (float(x), float(y)) for key, (x, y) in data["centroids"].items()}
    raise FileNotFoundError(f"no packaged map named {map_name!r}")


def power_colors(powers: Sequence[str]) -> dict[str, str]:
    """Build a stable power to hex-colour table.

    A power the classic palette names keeps its colour; every other power, which
    means every variant map's, gets one of a set of evenly spaced hues, assigned
    in sorted order so a given board always renders the same way.
    """
    colors: dict[str, str] = {}
    unknown = [p for p in sorted(powers) if p not in _STANDARD_COLORS]
    for power in sorted(powers):
        if power in _STANDARD_COLORS:
            colors[power] = _STANDARD_COLORS[power]
        else:
            colors[power] = _fallback_color(unknown.index(power), len(unknown))
    return colors


def _fallback_color(index: int, total: int) -> str:
    hue = (index / max(total, 1)) * 360.0
    return _hsl_hex(hue, 0.55, 0.42)


def _hsl_hex(hue: float, saturation: float, lightness: float) -> str:
    # colorsys is hls-ordered; keep the call site in the conventional hsl shape.
    import colorsys

    r, g, b = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
    return f"#{round(r * 255):02x}{round(g * 255):02x}{round(b * 255):02x}"


def _province_adjacency(graph: MapGraph) -> set[frozenset[str]]:
    """Province-level union of army and fleet adjacency (underlay edges)."""
    edges: set[frozenset[str]] = set()
    for origin, neighbors in graph.army_adjacency.items():
        edges.update(frozenset((origin, n)) for n in neighbors)
    for loc, fleet_neighbors in graph.fleet_adjacency.items():
        for fleet_neighbor in fleet_neighbors:
            if loc.province != fleet_neighbor.province:
                edges.add(frozenset((loc.province, fleet_neighbor.province)))
    return edges


def _shrink(
    a: tuple[float, float], b: tuple[float, float], by_a: float, by_b: float
) -> tuple[float, float, float, float]:
    """Pull the endpoints of ``a->b`` in by the marker radii.

    An arrow then stops at the markers it joins instead of piercing them.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    return (a[0] + ux * by_a, a[1] + uy * by_a, b[0] - ux * by_b, b[1] - uy * by_b)


def _arrow_head(
    x1: float, y1: float, x2: float, y2: float, size: float, color: str, tag: str
) -> Polygon:
    """Build the head of an arrow running ``(x1, y1) -> (x2, y2)``."""
    angle = math.atan2(y2 - y1, x2 - x1)
    left = angle + math.radians(150.0)
    right = angle - math.radians(150.0)
    return Polygon(
        points=(
            (x2, y2),
            (x2 + size * math.cos(left), y2 + size * math.sin(left)),
            (x2 + size * math.cos(right), y2 + size * math.sin(right)),
        ),
        fill=color,
        tag=tag,
    )


def _unit_marker(
    x: float, y: float, kind_value: str, strength: int, color: str, power: str
) -> list[Element]:
    """Build one unit's marker: a disc for an army, a triangle for a fleet.

    Any other unit kind a variant ships renders as a disc. A strength other than
    1 is printed on the marker.
    """
    marker: list[Element] = []
    if kind_value == "F":
        r = 2.2
        marker.append(
            Polygon(
                points=(
                    (x, y - r),
                    (x + r * math.sin(math.radians(120)), y - r * math.cos(math.radians(120))),
                    (x + r * math.sin(math.radians(240)), y - r * math.cos(math.radians(240))),
                ),
                fill=color,
                stroke="#ffffff",
                stroke_width=0.3,
                tag=f"unit:{power}",
            )
        )
    else:
        marker.append(
            Circle(x, y, 1.8, fill=color, stroke="#ffffff", stroke_width=0.3, tag=f"unit:{power}")
        )
    if strength != 1:
        marker.append(Text(x, y + 0.1, str(strength), size=2.4, color="#ffffff", bold=True))
    return marker


def build_board_scene(
    graph: MapGraph,
    state: State,
    ownership: Mapping[str, str],
    *,
    caption: str = "",
    orders: Sequence[tuple[str, str, bool]] = (),
    dislodged: Sequence[tuple[str, str]] = (),
    centroids: Mapping[str, tuple[float, float]] | None = None,
) -> Scene:
    """Build the full board as drawing primitives.

    ``orders`` rows are ``(power, canonical order text, succeeded)`` -- exactly
    what a :class:`PhaseRecord`'s resolutions carry; ``dislodged`` rows are
    ``(power, province)`` marks. ``centroids`` defaults to the map's own.
    """
    cent = dict(load_centroids(graph.name) if centroids is None else centroids)
    missing = set(graph.provinces) - set(cent)
    if missing:
        raise KeyError(f"centroids missing for provinces: {sorted(missing)}")
    colors = power_colors(tuple(graph.starting_units))

    scene: list[Element] = []
    # 1 -- adjacency underlay.
    for edge in sorted(_province_adjacency(graph), key=sorted):
        a, b = sorted(edge)
        scene.append(Line(*cent[a], *cent[b], width=0.18, color=_ADJACENCY))
    # 2 -- province markers + supply centers + labels.
    for key in sorted(graph.provinces):
        province = graph.provinces[key]
        x, y = cent[key]
        if province.terrain is Terrain.SEA:
            scene.append(Circle(x, y, 1.15, fill=_SEA_FILL, stroke=_SEA_EDGE, stroke_width=0.25))
        elif province.supply_center_count:
            owner = ownership.get(key)
            fill = colors.get(owner, _NEUTRAL_SC) if owner else _NEUTRAL_SC
            scene.append(
                Circle(
                    x, y, 1.5, fill=fill, stroke=_INK, stroke_width=0.35, tag=f"sc:{owner or '-'}"
                )
            )
            scene.append(Circle(x, y, 0.45, fill="#ffffff"))
        else:
            scene.append(Circle(x, y, 1.0, fill=_LAND_FILL, stroke=_LAND_EDGE, stroke_width=0.25))
        scene.append(Text(x, y - 2.0, key, size=1.7, color="#8a8578"))
    # 3 -- order arrows (under the units, so markers stay crisp).
    for power, text, succeeded in orders:
        scene.extend(_order_elements(_parse_any(text), succeeded, cent, colors.get(power, _INK)))
    # 4 -- units, fanned out when a province holds several (capacity > 1 maps).
    by_province: dict[str, list[int]] = {}
    for index, unit in enumerate(state.units):
        by_province.setdefault(unit.location.province, []).append(index)
    for province_key, indexes in by_province.items():
        x, y = cent[province_key]
        fan = 1.9 if len(indexes) > 1 else 0.0
        for slot, index in enumerate(indexes):
            unit = state.units[index]
            angle = 2.0 * math.pi * slot / len(indexes)
            ux, uy = x + fan * math.cos(angle), y + fan * math.sin(angle)
            scene.extend(
                _unit_marker(ux, uy, unit.kind.value, unit.strength, colors[unit.power], unit.power)
            )
    # 5 -- dislodgement marks.
    for _power, province_key in dislodged:
        x, y = cent[province_key]
        for sign in (1.0, -1.0):
            scene.append(
                Line(x - 1.2, y - sign * 1.2, x + 1.2, y + sign * 1.2, width=0.55, color=_FAIL)
            )
    # 6 -- caption + legend.
    if caption:
        scene.append(Text(2.0, 3.2, caption, size=2.6, color=_INK, anchor="start", bold=True))
    x_cursor = 2.0
    for power in sorted(colors):
        scene.append(Circle(x_cursor, 98.0, 0.9, fill=colors[power]))
        scene.append(
            Text(x_cursor + 1.6, 98.6, power.title(), size=1.7, color=_INK, anchor="start")
        )
        x_cursor += 2.4 + 1.05 * len(power)
    return tuple(scene)


def _parse_any(text: str) -> GameOrder:
    """Parse an order text from *any* phase (records mix all three grammars)."""
    first = text.split(maxsplit=1)[0].upper() if text.strip() else ""
    if first in ("BUILD", "WAIVE"):
        return parse_adjustment_order(text)
    if first == "DISBAND":
        return parse_retreat_order(text)
    return parse_order(text)


def _order_elements(
    order: GameOrder, succeeded: bool, cent: Mapping[str, tuple[float, float]], color: str
) -> list[Element]:
    """Build the arrows for one parsed order.

    A move gets a solid arrow in its outcome colour, dashed when it goes by
    convoy; a support gets a thin power-coloured dash ending in a dot at the
    supported *target*; a convoy gets a dotted sea-to-destination hint. Anything
    else -- a hold, a variant order type -- draws nothing.
    """
    outcome = _OK if succeeded else _FAIL
    tag = f"order:{'ok' if succeeded else 'fail'}"
    if isinstance(order, Move):
        a = cent[order.location.province]
        b = cent[order.destination.province]
        x1, y1, x2, y2 = _shrink(a, b, 2.3, 2.6)
        dash = (1.4, 0.8) if order.via_convoy else ()
        return [
            Line(x1, y1, x2, y2, width=0.55, color=outcome, dash=dash),
            _arrow_head(x1, y1, x2, y2, 1.5, outcome, tag),
        ]
    if isinstance(order, Support):
        a = cent[order.location.province]
        subject = order.subject
        target = (
            subject.destination.province if isinstance(subject, Move) else subject.location.province
        )
        b = cent[target]
        x1, y1, x2, y2 = _shrink(a, b, 2.3, 2.9)
        return [
            Line(x1, y1, x2, y2, width=0.3, color=color, dash=(0.9, 0.9)),
            Circle(x2, y2, 0.5, fill=color if succeeded else _FAIL, tag="order:support"),
        ]
    if isinstance(order, Convoy):
        a = cent[order.location.province]
        b = cent[order.cargo.destination.province]
        x1, y1, x2, y2 = _shrink(a, b, 2.3, 2.9)
        return [Line(x1, y1, x2, y2, width=0.25, color=_SEA_EDGE, dash=(0.5, 1.1))]
    return []


def render_board(
    graph: MapGraph,
    state: State,
    ownership: Mapping[str, str],
    *,
    caption: str = "",
    orders: Sequence[tuple[str, str, bool]] = (),
    dislodged: Sequence[tuple[str, str]] = (),
    size: int = 720,
) -> str:
    """Render one board frame as an SVG string, through :func:`build_board_scene`."""
    scene = build_board_scene(
        graph, state, ownership, caption=caption, orders=orders, dislodged=dislodged
    )
    return scene_to_svg(scene, size=size)


def build_record_scene(
    graph: MapGraph,
    record: PhaseRecord,
    prev_state: State,
    prev_ownership: Mapping[str, str],
) -> Scene:
    """Build one executed phase as a scene.

    It is the *pre-phase* board, with the record's orders drawn in outcome
    colours and its dislodgements marked, which is the frame both the demo
    animation and a replay viewer want.
    """
    orders = [(r.power, r.text, r.resolution is Resolution.SUCCEEDS) for r in record.resolutions]
    dislodged = [(d.unit.power, d.unit.location.province) for d in record.dislodgements]
    return build_board_scene(
        graph,
        prev_state,
        prev_ownership,
        caption=record.phase,
        orders=orders,
        dislodged=dislodged,
    )


def render_record(
    graph: MapGraph,
    record: PhaseRecord,
    prev_state: State,
    prev_ownership: Mapping[str, str],
    *,
    size: int = 720,
) -> str:
    """SVG form of :func:`build_record_scene`."""
    return scene_to_svg(build_record_scene(graph, record, prev_state, prev_ownership), size=size)


def scene_to_svg(scene: Scene, *, size: int = 720) -> str:
    """Serialize a scene to a standalone SVG document string."""
    s = size / CANVAS
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">',
        f'<rect width="{size}" height="{size}" fill="#ffffff"/>',
    ]
    for el in scene:
        if isinstance(el, Line):
            dash = (
                f' stroke-dasharray="{",".join(f"{d * s:.2f}" for d in el.dash)}"'
                if el.dash
                else ""
            )
            parts.append(
                f'<line x1="{el.x1 * s:.2f}" y1="{el.y1 * s:.2f}" x2="{el.x2 * s:.2f}" '
                f'y2="{el.y2 * s:.2f}" stroke="{el.color}" stroke-width="{el.width * s:.2f}"'
                f' stroke-linecap="round"{dash}/>'
            )
        elif isinstance(el, Circle):
            stroke = (
                f' stroke="{el.stroke}" stroke-width="{el.stroke_width * s:.2f}"'
                if el.stroke
                else ""
            )
            tag = f' data-tag="{el.tag}"' if el.tag else ""
            parts.append(
                f'<circle cx="{el.cx * s:.2f}" cy="{el.cy * s:.2f}" r="{el.r * s:.2f}" '
                f'fill="{el.fill or "none"}"{stroke}{tag}/>'
            )
        elif isinstance(el, Polygon):
            points = " ".join(f"{x * s:.2f},{y * s:.2f}" for x, y in el.points)
            stroke = (
                f' stroke="{el.stroke}" stroke-width="{el.stroke_width * s:.2f}"'
                if el.stroke
                else ""
            )
            tag = f' data-tag="{el.tag}"' if el.tag else ""
            parts.append(f'<polygon points="{points}" fill="{el.fill or "none"}"{stroke}{tag}/>')
        else:
            anchor = {"start": "start", "middle": "middle", "end": "end"}[el.anchor]
            weight = ' font-weight="bold"' if el.bold else ""
            parts.append(
                f'<text x="{el.x * s:.2f}" y="{el.y * s:.2f}" font-size="{el.size * s:.2f}" '
                f'fill="{el.color}" text-anchor="{anchor}" '
                f'font-family="Verdana, DejaVu Sans, sans-serif"{weight}>{_escape(el.text)}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
