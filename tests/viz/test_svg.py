"""Centroid-data integrity and the SVG renderer (stdlib only, so always run).

The centroid block in ``standard.json`` is hand-assigned from the real geography
of the named places, like every other figure in a map file. Errors in
hand-transcribed data are a *testing* problem, so the layout invariants live
here: completeness, bounds, separation, and respect for local adjacency. A gross
misplacement makes some non-neighbour the nearest marker, which is exactly what
the last test catches.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import pytest

from entente.engine.game import Game
from entente.engine.map.loader import load_map
from entente.viz.svg import (
    build_board_scene,
    load_centroids,
    power_colors,
    render_board,
    render_record,
    scene_to_svg,
)

GRAPH = load_map("standard")
CENTROIDS = load_centroids("standard")


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TestCentroidIntegrity:
    """The hand-assigned centroid layout, checked as data."""

    def test_exactly_one_centroid_per_province(self) -> None:
        assert set(CENTROIDS) == set(GRAPH.provinces)

    def test_all_in_canvas_bounds(self) -> None:
        for key, (x, y) in CENTROIDS.items():
            assert 0.0 <= x <= 100.0 and 0.0 <= y <= 100.0, key

    def test_min_pairwise_separation(self) -> None:
        # Markers have radius <= 2.6 canvas units; 3.0 keeps every pair legible.
        keys = sorted(CENTROIDS)
        for i, a in enumerate(keys):
            for b in keys[i + 1 :]:
                assert _dist(CENTROIDS[a], CENTROIDS[b]) >= 3.0, (a, b)

    def test_nearest_marker_is_a_map_neighbour(self) -> None:
        """Every province's nearest marker must be a real adjacency neighbour.

        Army or fleet, at province level: a marker whose nearest neighbour is
        somewhere it does not border is misplaced.
        """
        neighbours: dict[str, set[str]] = {key: set() for key in GRAPH.provinces}
        for origin, adjacent in GRAPH.army_adjacency.items():
            neighbours[origin].update(adjacent)
        for origin, adjacent in GRAPH.fleet_adjacency.items():
            for loc in adjacent:
                if loc.province != origin.province:
                    neighbours[origin.province].add(loc.province)
                    neighbours[loc.province].add(origin.province)
        for key in GRAPH.provinces:
            nearest = min(
                (other for other in GRAPH.provinces if other != key),
                key=lambda other: _dist(CENTROIDS[key], CENTROIDS[other]),
            )
            assert nearest in neighbours[key], (key, nearest)

    def test_variant_map_without_centroids_raises(self) -> None:
        with pytest.raises(KeyError):
            load_centroids("beleriand")

    def test_unknown_map_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_centroids("atlantis")


class TestRenderer:
    """The SVG the renderer emits, read back as markup."""

    def test_opening_board_is_wellformed_svg_with_all_units(self) -> None:
        game = Game.new("standard", "standard")
        svg = render_board(game.graph, game.state, game.ownership, caption="S1901M")
        root = ET.fromstring(svg)
        unit_tags = [el for el in root.iter() if el.get("data-tag", "").startswith("unit:")]
        assert len(unit_tags) == len(game.state.units) == 22
        # every power's color appears on at least one unit marker
        colors = power_colors(game.powers)
        painted = {el.get("fill") for el in unit_tags}
        assert set(colors.values()) <= painted

    def test_owned_and_neutral_supply_centers_marked(self) -> None:
        game = Game.new("standard", "standard")
        svg = render_board(game.graph, game.state, game.ownership)
        root = ET.fromstring(svg)
        sc_tags = [
            el.get("data-tag") for el in root.iter() if el.get("data-tag", "").startswith("sc:")
        ]
        total_scs = sum(1 for p in game.graph.provinces.values() if p.supply_center_count)
        assert len(sc_tags) == total_scs == 34
        assert "sc:-" in sc_tags  # neutrals exist at game start

    def test_record_frame_draws_outcome_arrows(self) -> None:
        game = Game.new("standard", "standard")
        prev_state, prev_ownership = game.state, dict(game.ownership)
        record = game.adjudicate()  # all-holds movement phase (no sheets submitted)
        svg = render_record(game.graph, record, prev_state, prev_ownership)
        ET.fromstring(svg)  # well-formed
        # A second phase with a real move draws a green arrow.
        prev_state, prev_ownership = game.state, dict(game.ownership)
        # F1901M: a guaranteed-legal unopposed French move.
        from entente.engine.orders import Move, parse_order

        move = next(
            o for o in game.legal_orders("FRANCE") if isinstance(o, Move) and not o.via_convoy
        )
        game.submit("FRANCE", [move])
        record = game.adjudicate()
        svg = render_record(game.graph, record, prev_state, prev_ownership)
        root = ET.fromstring(svg)
        ok_arrows = [el for el in root.iter() if el.get("data-tag") == "order:ok"]
        assert len(ok_arrows) >= 1
        assert parse_order(record.resolutions[0].text)  # canonical text round-trips

    def test_scene_rejects_map_without_centroids(self) -> None:
        game = Game.new("beleriand", "beleriand")
        with pytest.raises(KeyError):
            build_board_scene(game.graph, game.state, game.ownership)

    def test_multi_occupancy_fans_units_apart(self) -> None:
        """Capacity>1 provinces (variant maps) draw every co-occupant."""
        game = Game.new("beleriand", "beleriand")
        centroids = dict.fromkeys(game.graph.provinces, (50.0, 50.0))
        # spread synthetic centroids on a grid to satisfy the renderer's needs
        for i, key in enumerate(sorted(game.graph.provinces)):
            centroids[key] = (5.0 + (i % 10) * 10.0, 5.0 + (i // 10) * 10.0)
        scene = build_board_scene(game.graph, game.state, game.ownership, centroids=centroids)
        svg = scene_to_svg(scene)
        root = ET.fromstring(svg)
        unit_tags = [el for el in root.iter() if el.get("data-tag", "").startswith("unit:")]
        assert len(unit_tags) == len(game.state.units)

    def test_caption_and_legend_render(self) -> None:
        game = Game.new("standard", "standard")
        svg = render_board(game.graph, game.state, game.ownership, caption="Hello <caption>")
        assert "Hello &lt;caption&gt;" in svg  # escaped
        for power in game.powers:
            assert power.title() in svg
