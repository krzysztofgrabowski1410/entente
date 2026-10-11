"""Board featurization + adjacency, derived from map data (``ml`` group)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from entente.engine.game import Game
from entente.engine.map import Location
from entente.nn.features import (
    REL_ADJACENT,
    REL_SELF,
    adjacency_relations,
    board_vocab,
    encode_board,
    encode_game,
    feature_dim,
    feature_layout,
)

pytestmark = pytest.mark.ml


def test_board_vocab_is_derived_from_the_map() -> None:
    standard = board_vocab("standard")
    assert standard.n_locations == 81  # 75 provinces + 6 split-coast fleet tokens
    assert standard.n_powers == 7
    assert list(standard.locations) == sorted(
        standard.locations, key=lambda x: (x.province, x.coast or "")
    )
    assert Location("STP", "NC") in standard.locations
    # a different map yields a different coordinate system -- nothing is hardcoded
    beleriand = board_vocab("beleriand")
    assert beleriand.n_locations != standard.n_locations
    assert beleriand.n_powers != standard.n_powers


def test_feature_dim_tracks_layout_and_power_count() -> None:
    layout = feature_layout("standard")
    assert feature_dim("standard") == layout["_total"] == 2 + 7 + 3 + 1 + 8 + 3 + 1 + 7


def test_encode_start_position_has_one_unit_per_starting_unit() -> None:
    game = Game.new("standard", "standard")
    feats = encode_game(game)
    layout = feature_layout("standard")
    assert feats.shape == (81, feature_dim("standard"))
    kind_block = feats[:, layout["kind"] : layout["kind"] + 2]
    assert int((kind_block.sum(dim=1) > 0).sum()) == len(game.state.units)
    # every token carries exactly one terrain and one sc-owner (incl. neutral) bit
    assert torch.all(feats[:, layout["terrain"] : layout["terrain"] + 3].sum(dim=1) == 1)
    assert torch.all(feats[:, layout["sc_owner"] : layout["sc_owner"] + 8].sum(dim=1) == 1)


def test_encode_marks_unit_owner_and_home_center_owner() -> None:
    game = Game.new("standard", "standard")
    feats = encode_game(game)
    vocab = board_vocab("standard")
    layout = feature_layout("standard")
    par = vocab.location_index(Location("PAR"))
    france = vocab.power_index("FRANCE")
    assert feats[par, layout["kind"] + 0] == 1.0  # army present
    assert feats[par, layout["owner"] + france] == 1.0  # owned by France
    assert feats[par, layout["sc"]] == 1.0  # PAR is a supply center
    assert feats[par, layout["sc_owner"] + france] == 1.0  # France's home center


def test_encode_phase_metadata() -> None:
    game = Game.new("standard", "standard")  # S1901M
    feats = encode_game(game)
    layout = feature_layout("standard")
    phase = layout["phase"]
    assert feats[0, phase + 0] == 1.0  # spring
    assert feats[0, phase + 3] == 1.0  # movement
    assert feats[0, phase + 6] == 0.0  # year offset 0


def test_empty_board_encodes_without_units() -> None:
    graph = Game.new("standard", "standard").graph
    feats = encode_board(graph, units=(), ownership={}, phase_tag="F1903M")
    layout = feature_layout("standard")
    assert torch.all(feats[:, layout["kind"] : layout["kind"] + 2] == 0.0)
    assert feats[0, layout["phase"] + 1] == 1.0  # fall


def test_adjacency_is_symmetric_and_reflects_the_map() -> None:
    rel = adjacency_relations("standard")
    assert rel.shape == (81, 81)
    assert torch.equal(rel, rel.T)
    assert torch.all(rel.diag() == REL_SELF)
    vocab = board_vocab("standard")
    par = vocab.location_index(Location("PAR"))
    bur = vocab.location_index(Location("BUR"))
    assert rel[par, bur] == REL_ADJACENT  # Paris borders Burgundy
    assert rel[par, vocab.location_index(Location("MOS"))] != REL_ADJACENT
