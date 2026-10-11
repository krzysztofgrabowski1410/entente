"""Stream preprocessed Parquet shards as training examples.

Each shard row carries a movement board plus one power's orders (see
:mod:`entente.data.preprocess`); this module rebuilds, per row, the
:func:`~entente.nn.features.encode_board` feature tensor, the per-unit legal
masks (:func:`~entente.nn.order_vocab.legal_order_indices`, engine-exact), the
teacher-forcing labels, and the value target. Shards are read lazily and cached a
few at a time, so a full-dataset pass never holds everything in memory.

Splits are by *game*, not by row: phases of one game are near-duplicates of each
other, so a by-row split would leak a training game into validation and report a
held-out top-1 the model has effectively already seen. Labels default an
unordered unit to a hold, and fall back to a hold if a recorded order is not a
legal in-vocabulary order for that unit (rare cross-engine notation drift), so a
malformed label never crashes a batch.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, Sampler

from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.orders import (
    Hold,
    OrderSyntaxError,
    format_order,
    parse_order,
    parse_unit_spec,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.nn.features import _unit_token, board_vocab, encode_board
from entente.nn.order_vocab import build_vocabulary, legal_order_indices

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from entente.training.config import DataConfig


@dataclass(slots=True)
class TrainingExample:
    """One (movement phase, power) training item, ready for the model."""

    features: torch.Tensor  # (L, F)
    slot_locations: list[int]  # encoder-token index per unit slot (decode order)
    legal_ids: list[list[int]]  # legal vocabulary indices per slot
    label_ids: list[int]  # true order index per slot (teacher forcing)
    value_target: torch.Tensor  # (n_powers,) SoS shares


@dataclass(slots=True)
class Batch:
    """Hold one collated batch.

    The features stack into a tensor for the encoder, while the per-example
    decode structure stays ragged as parallel lists, since boards carry
    different numbers of units.
    """

    features: torch.Tensor  # (B, L, F)
    value_targets: torch.Tensor  # (B, n_powers)
    slot_locations: list[list[int]]
    legal_ids: list[list[list[int]]]
    label_ids: list[list[int]]

    def to(self, device: str | torch.device) -> Batch:
        """Move the batch's tensors to ``device`` in place and return the batch."""
        self.features = self.features.to(device)
        self.value_targets = self.value_targets.to(device)
        return self


def collate(examples: Sequence[TrainingExample]) -> Batch:
    """Stack features/value targets; keep the ragged decode structure as lists."""
    return Batch(
        features=torch.stack([ex.features for ex in examples]),
        value_targets=torch.stack([ex.value_target for ex in examples]),
        slot_locations=[ex.slot_locations for ex in examples],
        legal_ids=[ex.legal_ids for ex in examples],
        label_ids=[ex.label_ids for ex in examples],
    )


def units_from_specs(units_by_power: Mapping[str, Sequence[str]]) -> tuple[Unit, ...]:
    """Build engine units from a shard row's power-to-spec-list mapping.

    A spec with the dump's ``*`` prefix is a dislodged, retreat-pending unit --
    transient state a movement board does not contain, the dislodging occupant
    being listed separately -- so it is dropped rather than parsed. These are
    rare enough to hide for a whole run and frequent enough to appear in a
    full-size one (19 rows in 1.22M), which is why every shard-row
    reconstruction goes through this helper: training here, and the sweep's
    agreement scoring in
    :func:`entente.agents.pikl_agent.movement_game_from_row`. Anything else
    unparseable still raises loudly.
    """
    return tuple(
        Unit(owner, *parse_unit_spec(spec))
        for owner, specs in units_by_power.items()
        for spec in specs
        if not spec.startswith("*")
    )


def _shard_paths(shards_dir: str | Path) -> list[Path]:
    return sorted(Path(shards_dir).glob("shard_*.parquet"))


def game_split(
    shards_dir: str | Path, val_fraction: float, seed: int = 0
) -> tuple[set[str], set[str]]:
    """Partition game ids into train and validation sets.

    The choice is a seeded hash of the id, so the split is stable across runs
    and no game has phases on both sides of the boundary.
    """
    ids: set[str] = set()
    for path in _shard_paths(shards_dir):
        ids.update(pq.read_table(path, columns=["game_id"])["game_id"].to_pylist())
    val: set[str] = set()
    train: set[str] = set()
    for game_id in sorted(ids):
        digest = hashlib.sha256(f"{seed}:{game_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        (val if fraction < val_fraction else train).add(game_id)
    return train, val


class ShardDataset(Dataset[TrainingExample]):
    """Serve training examples from Parquet shards, optionally restricted to a split.

    A split is given as a set of game ids. Reconstruction from a row is
    deterministic, and a small shard cache keeps memory bounded whatever the
    dataset's size.
    """

    def __init__(
        self,
        shards_dir: str | Path,
        *,
        map_name: str = "standard",
        ruleset: str = "standard",
        game_ids: set[str] | None = None,
        cache_shards: int = 3,
    ) -> None:
        self.map_name = map_name
        self.graph = load_map(map_name)
        self.ruleset = RULESETS[ruleset]
        self.vocab = build_vocabulary(map_name, ruleset)
        self.board = board_vocab(map_name)
        self._paths = _shard_paths(shards_dir)
        self._cache_shards = cache_shards
        self._cache: OrderedDict[int, list[dict[str, object]]] = OrderedDict()
        self._index: list[tuple[int, int]] = []
        for shard_i, path in enumerate(self._paths):
            game_col = pq.read_table(path, columns=["game_id"])["game_id"].to_pylist()
            for row_i, game_id in enumerate(game_col):
                if game_ids is None or game_id in game_ids:
                    self._index.append((shard_i, row_i))

    def __len__(self) -> int:
        return len(self._index)

    def _shard(self, shard_i: int) -> list[dict[str, object]]:
        """Read one shard's rows, through a least-recently-used cache of whole shards."""
        cached = self._cache.get(shard_i)
        if cached is not None:
            self._cache.move_to_end(shard_i)
            return cached
        rows = pq.read_table(self._paths[shard_i]).to_pylist()
        self._cache[shard_i] = rows
        if len(self._cache) > self._cache_shards:
            self._cache.popitem(last=False)
        return rows

    def __getitem__(self, idx: int) -> TrainingExample:
        shard_i, row_i = self._index[idx]
        row = self._shard(shard_i)[row_i]
        return self._build_example(row)

    def _build_example(self, row: dict[str, object]) -> TrainingExample:
        power = str(row["power"])
        phase_tag = str(row["phase"])
        units_by_power = json.loads(str(row["units_json"]))
        centers_by_power = json.loads(str(row["centers_json"]))
        order_strings = json.loads(str(row["orders_json"]))

        units = units_from_specs(units_by_power)
        ownership = {
            province: owner
            for owner, provinces in centers_by_power.items()
            for province in provinces
        }
        features = encode_board(self.graph, units=units, ownership=ownership, phase_tag=phase_tag)
        game = self._movement_game(units, ownership, phase_tag)
        groups = legal_order_indices(game, power, self.vocab)

        my_units = sorted((u for u in units if u.power == power), key=lambda u: str(u.location))
        chosen = self._orders_by_location(order_strings)
        slot_locations, legal_ids, label_ids = [], [], []
        for unit in my_units:
            key = str(unit.location)
            legal = groups.get(key)
            if not legal:  # a unit with no enumerable order (should not happen in movement)
                continue
            slot_locations.append(self.board.location_index(_unit_token(unit)))
            legal_ids.append(legal)
            label_ids.append(self._label_for(unit, chosen.get(key), set(legal)))

        value_list = cast("list[float]", row["value_target"])
        value = torch.tensor([float(v) for v in value_list], dtype=torch.float32)
        return TrainingExample(features, slot_locations, legal_ids, label_ids, value)

    def _movement_game(
        self, units: tuple[Unit, ...], ownership: dict[str, str], phase_tag: str
    ) -> Game:
        # season does not affect movement legality; use spring so any movement
        # phase rebuilds a valid legal-order enumeration.
        year = int(phase_tag[1:-1])
        phase = Phase(year, Season.SPRING, PhaseKind.MOVEMENT)
        state = State.of(self.map_name, phase.tag, units)
        return Game(self.ruleset, self.graph, state, ownership, phase)

    @staticmethod
    def _orders_by_location(order_strings: list[str]) -> dict[str, object]:
        parsed: dict[str, object] = {}
        for text in order_strings:
            try:
                order = parse_order(text)
            except OrderSyntaxError:
                # tolerate rare cross-engine notation drift in dataset rows;
                # catch only the parser's own error, not every Exception --
                # a genuine parser/engine bug must surface, not degrade a row
                # to a silent hold label.
                continue
            parsed[str(order.location)] = order
        return parsed

    def _label_for(self, unit: Unit, order: object, legal: set[int]) -> int:
        """Give a unit's teacher-forcing label.

        That is its recorded order when the order is legal and in the
        vocabulary, and otherwise a hold, which always is both.
        """
        if order is not None:
            idx = self.vocab.get(format_order(order))  # type: ignore[arg-type]
            if idx is not None and idx in legal:
                return idx
        return self.vocab.index(format_order(Hold(unit.kind, unit.location)))


class ShardLocalShuffle(Sampler[int]):
    """Shuffle a :class:`ShardDataset` at two levels: shards, then rows within them.

    Shards come in random order and rows are drawn randomly from a rolling
    window of ``interleave`` open shards.

    A fully random sampler defeats the dataset's small shard cache: at 25 shards
    with 3 cached, about 88% of rows re-read and re-convert a whole 50k-row
    parquet shard, measured at 487 ms/row against 9.6 ms/row shard-local, which
    starves the GPU to roughly one step per minute. This sampler
    keeps per-worker cache hits near 100% (at most ``interleave`` shards are
    ever open at once) while every epoch still permutes shards *and* rows --
    the standard two-level shuffle for sharded datasets.

    Deterministic given ``seed``: epoch *k*'s permutation depends only on
    ``(seed, k)``, so a re-run replays the identical index stream.
    """

    def __init__(self, dataset: ShardDataset, *, seed: int = 0, interleave: int = 3) -> None:
        by_shard: dict[int, list[int]] = {}
        for position, (shard_i, _row_i) in enumerate(dataset._index):
            by_shard.setdefault(shard_i, []).append(position)
        self._groups = list(by_shard.values())
        self._count = len(dataset)
        self._seed = seed
        self._interleave = max(1, interleave)
        self._epoch = 0

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator().manual_seed(self._seed * 1_000_003 + self._epoch)
        self._epoch += 1
        shard_order: list[int] = torch.randperm(len(self._groups), generator=generator).tolist()
        open_buckets: list[list[int]] = []
        while shard_order or open_buckets:
            while len(open_buckets) < self._interleave and shard_order:
                group = self._groups[shard_order.pop(0)]
                order = torch.randperm(len(group), generator=generator).tolist()
                open_buckets.append([group[j] for j in order])
            pick = int(torch.randint(len(open_buckets), (), generator=generator))
            bucket = open_buckets[pick]
            yield bucket.pop()
            if not bucket:
                open_buckets.pop(pick)


def load_bc_datasets(cfg: DataConfig) -> tuple[ShardDataset, ShardDataset]:
    """Build (train, val) datasets from a :class:`DataConfig`, split by game."""
    train_ids, val_ids = game_split(cfg.shards_dir, cfg.val_fraction)
    train = ShardDataset(cfg.shards_dir, game_ids=train_ids)
    val = ShardDataset(cfg.shards_dir, game_ids=val_ids)
    return train, val
