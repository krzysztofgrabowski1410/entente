"""Pin preprocessing: the filters, the row shape, the value targets and the schema.

A row is one movement phase for one power. Needs the ``ml`` group for pyarrow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

from entente.data.paquette import (
    GameRecord,
    PaquetteJSONLSource,
    PhaseRecord,
)
from entente.data.preprocess import (
    final_value_target,
    is_completed,
    map_powers,
    preprocess,
)
from entente.data.synthetic import synthesize_games

pytestmark = pytest.mark.ml

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_games.jsonl"
POWERS = map_powers("standard")


def _phase(name: str, centers: dict[str, list[str]]) -> PhaseRecord:
    return PhaseRecord(name=name, units={}, centers=centers, orders={})


def test_preprocess_fixture_produces_rows(tmp_path: Path) -> None:
    stats = preprocess(PaquetteJSONLSource(FIXTURE), tmp_path, shard_size=200)
    assert stats.games_seen == 6
    assert stats.games_kept == 6
    assert stats.rows > 0
    assert stats.shards >= 1
    # rows carry the documented schema
    table = pq.read_table(next(tmp_path.glob("shard_*.parquet")))
    assert set(table.column_names) == {
        "game_id",
        "phase",
        "power",
        "units_json",
        "centers_json",
        "orders_json",
        "value_target",
    }
    # every row is a movement phase and its value target is a 7-vector
    assert all(name.endswith("M") for name in table["phase"].to_pylist())
    assert all(len(v) == 7 for v in table["value_target"].to_pylist())


def test_filter_drops_non_standard_and_incomplete(tmp_path: Path) -> None:
    completed = GameRecord(
        "g-ok",
        "standard",
        (_phase("S1901M", {"FRANCE": ["PAR"]}), _phase("COMPLETED", {"FRANCE": ["PAR"]})),
    )
    non_standard = GameRecord("g-bel", "beleriand", (_phase("S1901M", {}), _phase("COMPLETED", {})))
    incomplete = GameRecord("g-aband", "standard", (_phase("S1901M", {"FRANCE": ["PAR"]}),))
    stats = preprocess([completed, non_standard, incomplete], tmp_path)
    assert stats.games_kept == 1
    assert stats.skipped_non_standard == 1
    assert stats.skipped_incomplete == 1


def test_is_completed_recognizes_marker_and_solo() -> None:
    solo = GameRecord(
        "s", "standard", (_phase("F1905M", {"AUSTRIA": [f"C{i}" for i in range(18)]}),)
    )
    marker = GameRecord("m", "standard", (_phase("S1901M", {}), _phase("COMPLETED", {})))
    abandoned = GameRecord("a", "standard", (_phase("S1901M", {"FRANCE": ["PAR"]}),))
    assert is_completed(solo, 18)
    assert is_completed(marker, 18)
    assert not is_completed(abandoned, 18)


def test_value_target_is_a_solo_indicator_or_normalized_shares() -> None:
    solo = GameRecord(
        "s", "standard", (_phase("COMPLETED", {"AUSTRIA": [f"C{i}" for i in range(18)]}),)
    )
    target = final_value_target(solo, POWERS, 18)
    assert target[POWERS.index("AUSTRIA")] == 1.0
    assert sum(target) == pytest.approx(1.0)

    draw = GameRecord(
        "d", "standard", (_phase("COMPLETED", {"FRANCE": ["A", "B"], "ITALY": ["C"]}),)
    )
    shares = final_value_target(draw, POWERS, 18)
    assert sum(shares) == pytest.approx(1.0)
    assert shares[POWERS.index("FRANCE")] > shares[POWERS.index("ITALY")]  # 4/5 vs 1/5


def test_preprocess_100_games_schema(tmp_path: Path) -> None:
    games = list(synthesize_games(range(100), agent="random", max_year=1902))
    from entente.data.paquette import game_from_json

    stats = preprocess([game_from_json(g) for g in games], tmp_path, shard_size=5000)
    assert stats.games_kept == 100
    assert stats.rows > 100
    for shard in tmp_path.glob("shard_*.parquet"):
        table = pq.read_table(shard)
        assert all(len(v) == 7 for v in table["value_target"].to_pylist())


def test_max_games_caps_kept_games(tmp_path: Path) -> None:
    stats = preprocess(PaquetteJSONLSource(FIXTURE), tmp_path, max_games=2)
    assert stats.games_kept == 2


def test_skip_games_partitions_the_kept_stream(tmp_path: Path) -> None:
    """skip_games counts *filter-passing* games, the same basis as max_games.

    Touching windows therefore partition the kept games even when dropped games
    are interleaved among them.
    """
    from entente.data.paquette import game_from_json

    games = [game_from_json(g) for g in synthesize_games(range(6), max_year=1901)]
    non_standard = GameRecord("g-bel", "beleriand", (_phase("S1901M", {}), _phase("COMPLETED", {})))
    stream = [*games[:2], non_standard, *games[2:]]  # a drop inside the train window

    train = preprocess(stream, tmp_path / "train", max_games=4)
    holdout = preprocess(stream, tmp_path / "holdout", skip_games=4, max_games=2)
    assert train.games_kept == 4
    assert holdout.games_kept == 2
    assert holdout.skipped_window == 4
    assert holdout.skipped_non_standard == 1  # scanning the skip span still reports drops

    def game_ids(shards: Path) -> set[str]:
        ids: set[str] = set()
        for shard in shards.glob("shard_*.parquet"):
            ids.update(pq.read_table(shard)["game_id"].to_pylist())
        return ids

    train_ids, holdout_ids = game_ids(tmp_path / "train"), game_ids(tmp_path / "holdout")
    assert train_ids == {g.game_id for g in games[:4]}
    assert holdout_ids == {g.game_id for g in games[4:6]}  # exactly the next kept games


def _run_cli(repo: Path, *args: str) -> str:
    import subprocess
    import sys

    script = repo / "scripts" / "preprocess_paquette.py"
    result = subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo,
    )
    return result.stdout


def test_cli_skip_games_yields_train_disjoint_holdout(tmp_path: Path) -> None:
    """The committed holdout recipe runs end to end and stays train-disjoint.

    Two CLI runs over one dump produce kept-basis windows sharing no game_id,
    with a dropped game inside the train window to prove the counting basis.
    """
    from entente.data.synthetic import write_jsonl

    repo = Path(__file__).resolve().parents[2]
    raws = list(synthesize_games(range(8), max_year=1901))
    variant = dict(raws[0], id="variant-never-kept", map="ancmed")
    dump = tmp_path / "dump.jsonl"
    write_jsonl([*raws[:2], variant, *raws[2:]], dump)

    _run_cli(repo, "--input", str(dump), "--out", str(tmp_path / "train"), "--max-games", "5")
    _run_cli(
        repo,
        "--input",
        str(dump),
        "--out",
        str(tmp_path / "holdout"),
        "--skip-games",
        "5",
        "--max-games",
        "2",
    )

    def game_ids(shards: Path) -> set[str]:
        ids: set[str] = set()
        for shard in shards.glob("shard_*.parquet"):
            ids.update(pq.read_table(shard)["game_id"].to_pylist())
        return ids

    train_ids, holdout_ids = game_ids(tmp_path / "train"), game_ids(tmp_path / "holdout")
    assert train_ids == {str(r["id"]) for r in raws[:5]}  # first 5 KEPT, variant excluded
    assert holdout_ids == {str(r["id"]) for r in raws[5:7]}  # exactly the next 2 kept
    assert not train_ids & holdout_ids
    assert "variant-never-kept" not in (train_ids | holdout_ids)


def test_cli_zip_reads_only_the_no_press_member_by_default(tmp_path: Path) -> None:
    """The default --zip-member confines a training window to the no-press split.

    The dataset zip interleaves five splits and other_maps.jsonl sorts first,
    so without the default a training window reads the wrong map's games and
    the filter keeps nothing.
    """
    import zipfile

    from entente.data.synthetic import write_jsonl

    repo = Path(__file__).resolve().parents[2]
    no_press = tmp_path / "standard_no_press.jsonl"
    other = tmp_path / "other_maps.jsonl"
    raws = list(synthesize_games(range(4), max_year=1901))
    write_jsonl(raws, no_press)
    write_jsonl([dict(raws[0], id="variant", map="ancmed")], other)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(other, arcname="other_maps.jsonl")  # sorts before the no-press member
        zf.write(no_press, arcname="standard_no_press.jsonl")

    stdout = _run_cli(
        repo, "--input", str(archive), "--out", str(tmp_path / "out"), "--max-games", "3"
    )
    assert "[standard_no_press.jsonl]" in stdout
    assert "games seen=3 kept=3" in stdout  # the other member was never even read

    ids: set[str] = set()
    for shard in (tmp_path / "out").glob("shard_*.parquet"):
        ids.update(pq.read_table(shard)["game_id"].to_pylist())
    assert ids == {str(r["id"]) for r in raws[:3]}
