"""Pin the shard dataset's reconstruction, its labels and its game-level split.

Needs the ``ml`` group for torch and pyarrow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")

from entente.data.dataset import ShardDataset, ShardLocalShuffle, collate, game_split
from entente.data.paquette import PaquetteJSONLSource
from entente.data.preprocess import preprocess

pytestmark = pytest.mark.ml

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_games.jsonl"


@pytest.fixture(scope="module")
def shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("shards")
    preprocess(PaquetteJSONLSource(FIXTURE), out, shard_size=200)
    return out


@pytest.fixture(scope="module")
def many_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shard the same fixture games small, so several shards exist.

    That is the geometry the shard-local sampler is about.
    """
    out = tmp_path_factory.mktemp("many_shards")
    preprocess(PaquetteJSONLSource(FIXTURE), out, shard_size=40)
    return out


def test_shard_local_shuffle_is_a_permutation_with_bounded_open_shards(
    many_shards: Path,
) -> None:
    ds = ShardDataset(many_shards)
    assert len({shard for shard, _ in ds._index}) >= 4  # the fixture really is multi-shard
    sampler = ShardLocalShuffle(ds, seed=0, interleave=2)
    order = list(sampler)
    assert sorted(order) == list(range(len(ds)))  # every row exactly once

    # locality invariant: at most `interleave` shards are ever open at once --
    # this is what keeps the per-worker shard cache hot.
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    for position, index in enumerate(order):
        shard = ds._index[index][0]
        first_seen.setdefault(shard, position)
        last_seen[shard] = position
    for position in range(len(order)):
        open_now = sum(1 for s in first_seen if first_seen[s] <= position <= last_seen[s])
        assert open_now <= 2

    # it actually shuffles: not the sequential order, and rows within a shard move
    assert order != list(range(len(ds)))


def test_shard_local_shuffle_is_seed_deterministic_and_epoch_varying(
    many_shards: Path,
) -> None:
    ds = ShardDataset(many_shards)
    a, b = ShardLocalShuffle(ds, seed=7), ShardLocalShuffle(ds, seed=7)
    a_epoch0, a_epoch1 = list(a), list(a)
    assert a_epoch0 == list(b)  # same seed, same epoch -> identical stream
    assert a_epoch0 != a_epoch1  # epochs re-permute
    assert sorted(a_epoch1) == list(range(len(ds)))
    assert list(ShardLocalShuffle(ds, seed=8)) != a_epoch0  # seed matters


def test_dislodged_marker_units_are_dropped_not_parsed(tmp_path: Path) -> None:
    """A '*'-prefixed unit spec is dropped, not parsed.

    The dump marks a dislodged, retreat-pending unit that way. They are rare
    enough to hide for a whole run -- 19 rows in 1.22M -- and reaching the
    parser crashes training mid-run, so reconstruction drops the transient and
    keeps the rest of the board.
    """
    from entente.data.dataset import units_from_specs
    from entente.data.paquette import GameRecord, PhaseRecord

    units = units_from_specs({"TURKEY": ["*A GRE", "A CON"]})
    assert [str(unit.location) for unit in units] == ["CON"]

    movement = PhaseRecord(
        name="F1909M",
        units={"TURKEY": ["*A GRE", "A CON"], "AUSTRIA": ["A BUD"]},
        centers={"TURKEY": ["CON", "ANK"], "AUSTRIA": ["BUD"]},
        orders={"TURKEY": ["A CON - BUL"], "AUSTRIA": []},
    )
    terminal = PhaseRecord(name="COMPLETED", units={}, centers={"TURKEY": ["CON"]}, orders={})
    record = GameRecord("g-star", "standard", (movement, terminal))
    out = tmp_path / "star_shards"
    preprocess([record], out)
    ds = ShardDataset(out)
    assert len(ds) == 2  # one row per power with units -- the starred one still counts
    for i in range(len(ds)):
        example = ds[i]  # must not raise
        assert len(example.slot_locations) == 1  # exactly the non-starred unit


def test_dataset_reconstructs_examples(shards: Path) -> None:
    ds = ShardDataset(shards)
    assert len(ds) > 0
    example = ds[0]
    assert example.features.shape == (81, 32)
    assert example.value_target.shape == (7,)
    assert example.value_target.sum() == pytest.approx(1.0, abs=1e-5)
    assert len(example.slot_locations) == len(example.legal_ids) == len(example.label_ids)


def test_every_label_is_legal_for_its_slot(shards: Path) -> None:
    ds = ShardDataset(shards)
    for i in range(min(len(ds), 40)):
        example = ds[i]
        for label, legal in zip(example.label_ids, example.legal_ids, strict=True):
            assert label in legal


def test_collate_stacks_features_and_keeps_ragged_decode(shards: Path) -> None:
    ds = ShardDataset(shards)
    batch = collate([ds[i] for i in range(min(5, len(ds)))])
    assert batch.features.shape[1:] == (81, 32)
    assert batch.value_targets.shape[1] == 7
    assert len(batch.slot_locations) == batch.features.shape[0]


def test_game_split_is_disjoint_and_covers_every_game(shards: Path) -> None:
    train, val = game_split(shards, 0.34, seed=1)
    assert train.isdisjoint(val)
    full = ShardDataset(shards)
    train_ds = ShardDataset(shards, game_ids=train)
    val_ds = ShardDataset(shards, game_ids=val)
    # partition by game partitions the rows exactly (no leakage, no loss)
    assert len(train_ds) + len(val_ds) == len(full)
    assert len(val_ds) > 0


def test_split_is_deterministic(shards: Path) -> None:
    assert game_split(shards, 0.5, seed=7) == game_split(shards, 0.5, seed=7)


def test_shard_cache_bounds_memory(shards: Path) -> None:
    ds = ShardDataset(shards, cache_shards=1)
    _ = ds[0]
    _ = ds[len(ds) - 1]
    assert len(ds._cache) <= 1


def test_labels_map_back_to_canonical_order_strings(shards: Path) -> None:
    ds = ShardDataset(shards)
    example = ds[0]
    labels = [ds.vocab.order(i) for i in example.label_ids]
    assert labels  # non-empty decode
    assert all(isinstance(order, str) for order in labels)


def test_some_recorded_orders_become_non_hold_labels(shards: Path) -> None:
    # Guards the hold-fallback: if the vocab lookup systematically failed,
    # every label would silently degrade to a hold and the top-1 metric would
    # measure holds while every other test still passes. At least
    # one recorded non-hold order must survive as its own label.
    ds = ShardDataset(shards)
    non_hold = 0
    for i in range(len(ds)):
        for label in (ds.vocab.order(j) for j in ds[i].label_ids):
            if not label.endswith(" H"):
                non_hold += 1
    assert non_hold > 0, "every label degraded to a hold -- the vocab/label path is broken"


def test_orders_by_location_catches_only_syntax_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    # The row-parsing fallback tolerates cross-engine notation drift but must
    # catch ONLY the parser's own OrderSyntaxError -- a genuine parser/engine
    # bug (any other exception) must surface, not degrade a row to a silent
    # hold.
    from entente.data import dataset as dataset_module

    parsed = ShardDataset._orders_by_location(["A PAR - BUR", "not a valid order @#$"])
    assert set(parsed) == {"PAR"}  # the junk line is skipped, the valid one kept

    def _boom(_text: str) -> object:
        raise RuntimeError("a real parser bug, not notation drift")

    monkeypatch.setattr(dataset_module, "parse_order", _boom)
    with pytest.raises(RuntimeError, match="real parser bug"):
        ShardDataset._orders_by_location(["A PAR - BUR"])
