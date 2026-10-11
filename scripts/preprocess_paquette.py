"""CLI: preprocess the Paquette dump (or a synthetic dump) into Parquet shards.

    uv run --group ml python scripts/preprocess_paquette.py \
        --input data/standard_no_press.jsonl.zip --out data/processed --max-games 20000

With no ``--input``, generates a synthetic dump on the fly (engine self-play;
the labels are the played ``--synthetic-agent``'s own orders, random by
default) -- useful for a dry run of the whole pipeline without
the 2.24 GB download. See docs/setup.md for the real-data commands.

``--skip-games`` offsets the *kept* (filter-passing) games before any is
written, counted on the same basis as ``--max-games`` over the same
deterministic stream, so a train-disjoint holdout is expressible from the same
dump: train = the first 20,000 kept games, holdout = exactly the next 2,000:

    ... --max-games 20000                     --out data/processed          # train
    ... --skip-games 20000 --max-games 2000   --out data/processed_holdout  # holdout

Zip inputs read only the ``--zip-member`` member (default
``standard_no_press.jsonl`` -- the real dataset zip also carries variant-map and
press splits, which a training subset must not silently mix in; pass
``--zip-member ""`` to concatenate every member).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from entente.data.paquette import GameRecord, PaquetteJSONLSource, game_from_json
from entente.data.preprocess import preprocess
from entente.data.synthetic import synthesize_games


def _synthetic_source(count: int, agent: str, max_year: int) -> list[GameRecord]:
    return [
        game_from_json(game)
        for game in synthesize_games(range(count), agent=agent, max_year=max_year)
    ]


def main(argv: list[str] | None = None) -> int:
    """Preprocess a dump (or generated games) into Parquet shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Paquette .jsonl or .zip (omit for synthetic)")
    parser.add_argument("--out", default="data/processed", help="output shard directory")
    parser.add_argument("--shard-size", type=int, default=50_000, help="rows per Parquet shard")
    parser.add_argument("--max-games", type=int, default=None, help="cap kept games (subset)")
    parser.add_argument(
        "--skip-games",
        type=int,
        default=0,
        help="skip the first N kept (filter-passing) games before writing any -- "
        "with --max-games this selects a window of the kept stream, so a "
        "train-disjoint holdout is expressible (train: --max-games 20000; "
        "holdout: --skip-games 20000 --max-games 2000)",
    )
    parser.add_argument(
        "--zip-member",
        default="standard_no_press.jsonl",
        help="for .zip inputs, read only this member ('' = concatenate all "
        ".jsonl members; the dataset zip also carries variant-map and press "
        "splits)",
    )
    parser.add_argument("--map", default="standard", help="map to keep (others are dropped)")
    parser.add_argument("--synthetic-games", type=int, default=200, help="synthetic game count")
    parser.add_argument("--synthetic-agent", default="random", help="synthetic self-play agent")
    parser.add_argument("--synthetic-max-year", type=int, default=1905)
    args = parser.parse_args(argv)

    if args.input:
        member = args.zip_member or None
        source: object = PaquetteJSONLSource(args.input, member=member)
        member_note = f" [{member}]" if member and str(args.input).endswith(".zip") else ""
        window = f" (skipping the first {args.skip_games} kept games)" if args.skip_games else ""
        print(f"reading {args.input}{member_note}{window}")
    else:
        source = _synthetic_source(
            args.synthetic_games, args.synthetic_agent, args.synthetic_max_year
        )
        print(f"synthesizing {args.synthetic_games} {args.synthetic_agent} self-play games")

    stats = preprocess(
        source,  # type: ignore[arg-type]
        args.out,
        map_name=args.map,
        shard_size=args.shard_size,
        max_games=args.max_games,
        skip_games=args.skip_games,
    )
    window_note = f", window-skipped={stats.skipped_window}" if stats.skipped_window else ""
    print(
        f"games seen={stats.games_seen} kept={stats.games_kept} "
        f"(dropped non-standard={stats.skipped_non_standard}, "
        f"incomplete={stats.skipped_incomplete}{window_note}); "
        f"rows={stats.rows} across {stats.shards} shard(s) -> {Path(args.out)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
