"""Turn a position into the location-token tensor and adjacency matrix the encoder reads.

``L`` (81 on the standard map) and the power set are **derived from the map**,
never hardcoded: a location token is any place a unit may stand -- one per
province for armies/single-coast fleets, one extra per named coast (St
Petersburg, Spain, Bulgaria contribute their split-coast fleet locations). The
:class:`BoardVocab` fixes their order (sorted), so token *i* means the same place
in the features, the adjacency matrix, and any checkpoint trained on them.

Feature channels are power-*absolute* (owner one-hots by identity), so the board
is encoded once and the policy decoder is told which power to decode -- DipNet's
scheme. Everything here is a plain function of a position; a :class:`Game`
convenience wrapper (:func:`encode_game`) and a units+ownership core
(:func:`encode_board`) share it, so the live agent and the offline dataset build
identical tensors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

import torch

from entente.engine.map import Location, MapGraph, Terrain, UnitKind, load_map
from entente.engine.state import Unit

if TYPE_CHECKING:
    from entente.engine.game import Game

# Adjacency relation categories for the learned attention bias (encoder.py holds
# the learned per-head table indexed by these). Disjoint by construction: two
# tokens of one province are never "movement-adjacent" (adjacency joins distinct
# provinces), so SAME_PROVINCE and ADJACENT never collide.
REL_SELF = 0
REL_SAME_PROVINCE = 1
REL_ADJACENT = 2
REL_NONE = 3
N_RELATIONS = 4


@dataclass(frozen=True, slots=True)
class BoardVocab:
    """Hold a map's ordered location tokens and power list as one coordinate system.

    Features, the adjacency matrix and the policy/value heads all index by this
    order, so token *i* means the same place everywhere.
    """

    map_name: str
    locations: tuple[Location, ...]
    powers: tuple[str, ...]
    _loc_index: dict[Location, int] = field(repr=False)
    _power_index: dict[str, int] = field(repr=False)

    @property
    def n_locations(self) -> int:
        """Count the location tokens, which is the sequence length ``L``."""
        return len(self.locations)

    @property
    def n_powers(self) -> int:
        """Count the powers the map plays with."""
        return len(self.powers)

    def location_index(self, location: Location) -> int:
        """Give the token column of a location."""
        return self._loc_index[location]

    def power_index(self, power: str) -> int:
        """Give the one-hot column of a power."""
        return self._power_index[power]


@cache
def board_vocab(map_name: str = "standard") -> BoardVocab:
    """Build (and cache) the location/power coordinate system for a map."""
    graph = load_map(map_name)
    locations: set[Location] = set()
    for province in graph.provinces:
        for kind in (UnitKind.ARMY, UnitKind.FLEET):
            locations.update(graph.locations_for(kind, province))
    ordered = tuple(sorted(locations, key=lambda loc: (loc.province, loc.coast or "")))
    powers = tuple(sorted(set(graph.home_centers) | set(graph.starting_units)))
    return BoardVocab(
        map_name=map_name,
        locations=ordered,
        powers=powers,
        _loc_index={loc: i for i, loc in enumerate(ordered)},
        _power_index={p: i for i, p in enumerate(powers)},
    )


# Feature layout (documented so encoder.py's input dim is derived, not guessed).
# Blocks, in order:
#   unit kind        2   is_army, is_fleet (occupant of this token)
#   unit owner       P   one-hot over powers (zero if empty)
#   dislodged        3   is_dislodged, dislodged_is_army, dislodged_is_fleet
#   supply center    1   is_supply_center (province-level)
#   sc owner         P+1 one-hot over powers + neutral
#   terrain          3   inland, coastal, sea
#   token kind       1   is_coast_token (a named-coast token vs a province token)
#   season           3   spring, fall, winter (broadcast)
#   phase kind       3   movement, retreat, adjustment (broadcast)
#   year             1   (year - start) / 20, clamped (broadcast)
_PHASE_WIDTH = 3 + 3 + 1
_SEASONS = ("S", "F", "W")
_PHASE_KINDS = ("M", "R", "A")
_TERRAIN_INDEX = {Terrain.INLAND: 0, Terrain.COASTAL: 1, Terrain.SEA: 2}


def feature_layout(map_name: str = "standard") -> dict[str, int]:
    """Give the start column of each feature block named in the layout above.

    Two block widths depend on the power count, so this and :func:`feature_dim`
    both derive from the map and nothing downstream hardcodes an offset.
    """
    n_powers = board_vocab(map_name).n_powers
    offsets: dict[str, int] = {}
    cursor = 0
    for name, width in (
        ("kind", 2),
        ("owner", n_powers),
        ("dislodged", 3),
        ("sc", 1),
        ("sc_owner", n_powers + 1),
        ("terrain", 3),
        ("token", 1),
        ("phase", _PHASE_WIDTH),
    ):
        offsets[name] = cursor
        cursor += width
    offsets["_total"] = cursor
    return offsets


def feature_dim(map_name: str = "standard") -> int:
    """Give the per-token feature width ``F`` for a map, which is the encoder's input dim."""
    return feature_layout(map_name)["_total"]


@cache
def adjacency_relations(map_name: str = "standard") -> torch.Tensor:
    """Classify every ordered token pair into one of the ``REL_*`` categories.

    The categories are self, same-province, movement-adjacent (army or fleet)
    and none; the matrix is symmetric. The encoder maps each category to a
    learned per-head bias on the attention logits.
    """
    graph = load_map(map_name)
    vocab = board_vocab(map_name)
    n = vocab.n_locations
    rel = torch.full((n, n), REL_NONE, dtype=torch.long)
    # movement adjacency: armies join province (coast=None) tokens, fleets join
    # exact (coasted) location tokens. Army edges originate only from an army
    # token, so the relation stays symmetric (a fleet coast token is not an army
    # origin -- it would otherwise radiate an edge the province token never mirrors).
    for i, loc_i in enumerate(vocab.locations):
        if loc_i.coast is None and graph.terrain(loc_i.province) is not Terrain.SEA:
            for province_j in graph.army_adjacency.get(loc_i.province, frozenset()):
                rel[i, vocab.location_index(Location(province_j))] = REL_ADJACENT
        for loc_j in graph.fleet_adjacency.get(loc_i, frozenset()):
            rel[i, vocab.location_index(loc_j)] = REL_ADJACENT
    # co-located tokens of a split-coast province (overwrites nothing adjacent --
    # adjacency joins distinct provinces)
    by_province: dict[str, list[int]] = {}
    for i, loc in enumerate(vocab.locations):
        by_province.setdefault(loc.province, []).append(i)
    for tokens in by_province.values():
        for i in tokens:
            for j in tokens:
                if i != j:
                    rel[i, j] = REL_SAME_PROVINCE
    for i in range(n):
        rel[i, i] = REL_SELF
    return rel


def encode_board(
    graph: MapGraph,
    *,
    units: tuple[Unit, ...],
    ownership: dict[str, str],
    phase_tag: str,
    dislodged: tuple[Unit, ...] = (),
    start_year: int = 1901,
) -> torch.Tensor:
    """Build the ``(L, F)`` feature tensor for one position.

    ``phase_tag`` is an engine phase tag (``"S1901M"``); season, phase kind and
    year are read from it. ``dislodged`` are the units awaiting retreat, empty
    outside retreat phases. Army location tokens use ``coast=None``; fleet
    tokens use the unit's actual (possibly coasted) location.
    """
    vocab = board_vocab(graph.name)
    n_powers = vocab.n_powers
    layout = feature_layout(graph.name)
    feats = torch.zeros((vocab.n_locations, layout["_total"]), dtype=torch.float32)

    c_kind = layout["kind"]
    c_owner = layout["owner"]
    c_disl = layout["dislodged"]
    c_sc = layout["sc"]
    c_scowner = layout["sc_owner"]
    c_terrain = layout["terrain"]
    c_token = layout["token"]
    c_phase = layout["phase"]

    for i, loc in enumerate(vocab.locations):
        province = graph.province(loc.province)
        feats[i, c_terrain + _TERRAIN_INDEX[province.terrain]] = 1.0
        if loc.coast is not None:
            feats[i, c_token] = 1.0
        if province.is_supply_center:
            feats[i, c_sc] = 1.0
        owner = ownership.get(loc.province)
        if owner is not None and owner in vocab._power_index:
            feats[i, c_scowner + vocab.power_index(owner)] = 1.0
        else:
            feats[i, c_scowner + n_powers] = 1.0  # neutral slot

    for unit in units:
        idx = vocab._loc_index.get(_unit_token(unit))
        if idx is None:  # pragma: no cover - engine states are always well-placed
            continue
        feats[idx, c_kind + (0 if unit.kind is UnitKind.ARMY else 1)] = 1.0
        if unit.power in vocab._power_index:
            feats[idx, c_owner + vocab.power_index(unit.power)] = 1.0

    for unit in dislodged:
        idx = vocab._loc_index.get(_unit_token(unit))
        if idx is None:  # pragma: no cover
            continue
        feats[idx, c_disl] = 1.0
        feats[idx, c_disl + (1 if unit.kind is UnitKind.ARMY else 2)] = 1.0

    _fill_phase(feats, c_phase, phase_tag, start_year)
    return feats


def _unit_token(unit: Unit) -> Location:
    """Give the location token a unit occupies.

    Armies index by province with the coast dropped; fleets index by their
    actual location with the coast kept.
    """
    if unit.kind is UnitKind.ARMY:
        return Location(unit.location.province)
    return unit.location


def _fill_phase(feats: torch.Tensor, c_phase: int, phase_tag: str, start_year: int) -> None:
    """Broadcast the season, phase kind and scaled year of ``phase_tag`` to every token.

    The year enters as ``(year - start_year) / 20`` clamped to ``[0, 1]``, so a
    game running past twenty years saturates rather than pushing the feature off
    the scale the other channels share.
    """
    season = phase_tag[0]
    kind = phase_tag[-1]
    year = int(phase_tag[1:-1])
    if season in _SEASONS:
        feats[:, c_phase + _SEASONS.index(season)] = 1.0
    if kind in _PHASE_KINDS:
        feats[:, c_phase + 3 + _PHASE_KINDS.index(kind)] = 1.0
    feats[:, c_phase + 6] = max(0.0, min(1.0, (year - start_year) / 20.0))


def encode_game(game: Game) -> torch.Tensor:
    """Encode the current position of a :class:`~entente.engine.game.Game`.

    This is the live agent's path into :func:`encode_board`, which the offline
    dataset calls directly from stored positions.
    """
    return encode_board(
        game.graph,
        units=game.state.units,
        ownership=dict(game.ownership),
        phase_tag=game.phase,
        dislodged=tuple(d.unit for d in game.pending_dislodgements),
        start_year=game.start_year,
    )
