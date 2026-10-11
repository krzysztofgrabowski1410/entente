"""Filter Paquette games and reshape them into sharded Parquet training rows.

One row per (movement phase, power-with-units): the pre-adjudication board (units
and centers, as JSON, enough to rebuild the position and its legal orders) plus
that power's submitted orders (the labels) and the game's final Sum-of-Squares
shares (the value target, constant across a game's rows). Movement phases only --
that is where the "per-unit top-1" metric lives, and where a static board fully
determines the legal orders (retreat/adjustment need prior-adjudication context a
saved state does not carry); the value head still sees a board per movement
phase. Features and legal masks are rebuilt in :mod:`entente.data.dataset` from
these columns, so the shards stay small and torch-free.

pyarrow (``ml`` group) writes the shards; everything else is stdlib + engine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq

from entente.data.paquette import GameRecord
from entente.engine.map import load_map
from entente.eval.scoring import sos_scores

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

_COMPLETED_MARKER = "COMPLETED"

_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("phase", pa.string()),
        ("power", pa.string()),
        ("units_json", pa.string()),
        ("centers_json", pa.string()),
        ("orders_json", pa.string()),
        ("value_target", pa.list_(pa.float32())),
    ]
)


@dataclass(frozen=True, slots=True)
class PreprocessStats:
    """What a preprocessing pass produced -- reported, never silently dropped."""

    games_seen: int
    games_kept: int
    rows: int
    shards: int
    skipped_non_standard: int
    skipped_incomplete: int
    skipped_window: int = 0  # filter-passing games skipped by `skip_games` (holdout offset)


def map_powers(map_name: str) -> tuple[str, ...]:
    """List a map's powers in sorted order.

    The value-target vector is indexed by this order, which is the same one
    :func:`entente.nn.features.board_vocab` gives the feature tensor.
    """
    graph = load_map(map_name)
    return tuple(sorted(set(graph.home_centers) | set(graph.starting_units)))


def _majority(map_name: str) -> int:
    """Give the supply-center count a solo win takes on this map."""
    graph = load_map(map_name)
    total = sum(p.supply_center_count for p in graph.provinces.values())
    return total // 2 + 1


def is_completed(game: GameRecord, majority: int) -> bool:
    """Say whether a game finished.

    A game counts as completed when it ends with the dump's terminal marker or
    when a power reached a supply-center majority. Games stopped mid-play with
    neither are abandoned, and their final centers would be a meaningless value
    target, so they are dropped.
    """
    if game.phases and game.phases[-1].name == _COMPLETED_MARKER:
        return True
    final = game.final_centers
    return any(len(centers) >= majority for centers in final.values())


def final_value_target(game: GameRecord, powers: tuple[str, ...], majority: int) -> list[float]:
    """Compute the game's final Sum-of-Squares shares, one per power in ``powers`` order.

    This is the value head's soft target; a solo winner takes the whole share.
    """
    final = game.final_centers
    counts = {power: len(final.get(power, [])) for power in powers}
    winner = next((power for power, count in counts.items() if count >= majority), None)
    shares = sos_scores(counts, winner)
    return [float(shares[power]) for power in powers]


def game_to_rows(
    game: GameRecord, powers: tuple[str, ...], majority: int
) -> Iterator[dict[str, object]]:
    """Per-(movement-phase, power) rows for one completed game."""
    value = final_value_target(game, powers, majority)
    for phase in game.phases:
        if not phase.is_movement:
            continue
        for power in powers:
            if not phase.units.get(power):
                continue  # eliminated power: nothing to order this phase
            yield {
                "game_id": game.game_id,
                "phase": phase.name,
                "power": power,
                "units_json": json.dumps(phase.units, sort_keys=True),
                "centers_json": json.dumps(phase.centers, sort_keys=True),
                "orders_json": json.dumps(phase.orders.get(power, [])),
                "value_target": value,
            }


def preprocess(
    source: Iterable[GameRecord],
    out_dir: str | Path,
    *,
    map_name: str = "standard",
    shard_size: int = 50_000,
    max_games: int | None = None,
    skip_games: int = 0,
    require_completed: bool = True,
) -> PreprocessStats:
    """Filter ``source`` to standard-map completed games and shard their movement rows.

    Shards are written to ``out_dir`` as ``shard_00000.parquet`` and on;
    ``shard_size`` is rows per shard and ``max_games`` caps the kept games, which
    is how a subset run is cut. ``skip_games`` skips the first N *filter-passing*
    games before any is written, counted on the same basis as ``max_games`` over
    the same deterministic stream, so two touching windows partition the kept
    games exactly: a train set is ``max_games=N`` and a disjoint holdout is
    ``skip_games=N, max_games=M``. Returns counts, including how many games were
    dropped and for which reason.
    """
    if skip_games < 0:
        raise ValueError(f"skip_games must be non-negative, got {skip_games}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    powers = map_powers(map_name)
    majority = _majority(map_name)

    buffer: list[dict[str, object]] = []
    seen = kept = rows = shards = 0
    non_standard = incomplete = window_skipped = 0

    def flush() -> None:
        nonlocal shards
        if not buffer:
            return
        table = pa.Table.from_pylist(buffer, schema=_SCHEMA)
        pq.write_table(table, out / f"shard_{shards:05d}.parquet")
        shards += 1
        buffer.clear()

    for game in source:
        seen += 1
        if game.map_name != map_name:
            non_standard += 1
            continue
        if require_completed and not is_completed(game, majority):
            incomplete += 1
            continue
        if window_skipped < skip_games:
            window_skipped += 1
            continue
        kept += 1
        for row in game_to_rows(game, powers, majority):
            buffer.append(row)
            rows += 1
            if len(buffer) >= shard_size:
                flush()
        if max_games is not None and kept >= max_games:
            break
    flush()
    return PreprocessStats(seen, kept, rows, shards, non_standard, incomplete, window_skipped)
