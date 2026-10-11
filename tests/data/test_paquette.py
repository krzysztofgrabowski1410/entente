"""Pin the Paquette reader's schema, its tolerance of bad lines and its zip support.

The schema is checked over at least 100 games. Torch-free, so these run in the
default environment.

The committed fixture (``fixtures/synthetic_games.jsonl``) is engine-generated
facts of play, not human games; the >=100-game schema check
generates more on the fly so the reader is exercised at the acceptance scale
without shipping a large file.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from entente.data.paquette import (
    GameSource,
    PaquetteJSONLSource,
    game_from_json,
    iter_game_records,
)
from entente.data.synthetic import synthesize_games, write_jsonl

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_games.jsonl"


def test_reads_the_committed_fixture() -> None:
    source = PaquetteJSONLSource(FIXTURE)
    games = list(source)
    assert len(games) == 6
    assert not source.skipped
    assert isinstance(source, GameSource)
    first = games[0]
    assert first.map_name == "standard"
    assert first.phases[0].name == "S1901M"
    assert first.phases[0].is_movement
    assert set(first.phases[0].units) == {
        "AUSTRIA",
        "ENGLAND",
        "FRANCE",
        "GERMANY",
        "ITALY",
        "RUSSIA",
        "TURKEY",
    }


def test_schema_holds_over_100_games() -> None:
    powers = {"AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY"}
    count = 0
    for raw in synthesize_games(range(100), agent="random", max_year=1902):
        game = game_from_json(raw)
        count += 1
        assert game.map_name == "standard"
        assert game.phases[-1].name == "COMPLETED"
        assert game.final_centers  # a value target can be computed
        movement = [p for p in game.phases if p.is_movement]
        assert movement, "every game has at least one movement phase"
        for phase in movement:
            assert set(phase.units) <= powers
            assert set(phase.orders) <= powers
            # a moving power's order list references only its own units' locations
            for owner, order_list in phase.orders.items():
                assert isinstance(order_list, list)
                assert owner in powers
    assert count == 100


def test_malformed_lines_are_skipped_not_raised(tmp_path: Path) -> None:
    path = tmp_path / "mixed.jsonl"
    good = next(iter(synthesize_games([0], max_year=1901)))
    import json

    path.write_text(
        json.dumps(good)
        + "\n"  # valid
        + "not json at all\n"  # malformed
        + json.dumps({"map": "standard"})
        + "\n",  # missing "phases"
        encoding="utf-8",
    )
    source = PaquetteJSONLSource(path)
    games = list(source)
    assert len(games) == 1
    assert len(source.skipped) == 2


def test_limit_caps_games(tmp_path: Path) -> None:
    path = tmp_path / "many.jsonl"
    write_jsonl(synthesize_games(range(5), max_year=1901), path)
    assert len(list(PaquetteJSONLSource(path, limit=3))) == 3


def test_skip_and_limit_partition_the_stream(tmp_path: Path) -> None:
    """Touching (skip, limit) windows partition the parsed stream exactly.

    This is what makes a holdout provably disjoint from its training set. Both
    bounds count successfully parsed records only, so a malformed line advances
    neither and cannot shift one window against the other.
    """
    import json

    import pytest

    path = tmp_path / "windows.jsonl"
    lines = [json.dumps(raw) for raw in synthesize_games(range(6), max_year=1901)]
    lines.insert(2, "corrupt, not json")  # inside the train window
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    whole = list(PaquetteJSONLSource(path))
    assert len(whole) == 6
    assert whole[3] != whole[4]  # records differ across the window boundary

    train = list(PaquetteJSONLSource(path, limit=4))
    holdout_source = PaquetteJSONLSource(path, skip=4, limit=2)
    holdout = list(holdout_source)
    assert train == whole[:4]
    assert holdout == whole[4:6]  # exactly the next games: disjoint, gap-free
    assert len(holdout_source.skipped) == 1  # scanning the skip span still reports bad lines
    assert list(PaquetteJSONLSource(path, skip=4)) == whole[4:]  # open-ended tail
    assert list(PaquetteJSONLSource(path, skip=0)) == whole  # the default is a no-op

    with pytest.raises(ValueError):
        PaquetteJSONLSource(path, skip=-1)


def test_reads_from_a_zip(tmp_path: Path) -> None:
    jsonl = tmp_path / "games.jsonl"
    write_jsonl(synthesize_games(range(3), max_year=1901), jsonl)
    archive = tmp_path / "games.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(jsonl, arcname="inner/games.jsonl")
    assert len(list(PaquetteJSONLSource(archive))) == 3


def test_zip_member_selects_one_split(tmp_path: Path) -> None:
    """`member` reads exactly the named split from a multi-split zip.

    The real dataset zip carries five. A wrong name must raise rather than
    yield a silently empty stream.
    """
    import pytest

    no_press = tmp_path / "standard_no_press.jsonl"
    other = tmp_path / "other_maps.jsonl"  # sorts FIRST -- the production trap
    write_jsonl(synthesize_games(range(3), max_year=1901), no_press)
    write_jsonl(synthesize_games(range(3, 5), max_year=1901), other)
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(other, arcname="other_maps.jsonl")
        zf.write(no_press, arcname="standard_no_press.jsonl")

    assert len(list(PaquetteJSONLSource(archive))) == 5  # default: all members
    picked = list(PaquetteJSONLSource(archive, member="standard_no_press.jsonl"))
    assert len(picked) == 3
    with pytest.raises(FileNotFoundError):
        list(PaquetteJSONLSource(archive, member="missing.jsonl"))


def test_real_dump_null_lists_parse_as_empty(tmp_path: Path) -> None:
    """A power's ``null`` units, centers or orders parse as empty, not as malformed.

    The dump writes ``null`` for a civil-disordered or eliminated power and in
    terminal snapshots. Treating it as malformed discards around 80% of the
    dump, biased toward exactly the completed games this pipeline keeps.
    """
    import json

    game = {
        "id": "real-shape",
        "map": "standard",
        "phases": [
            {
                "name": "S1901M",
                "state": {
                    "units": {"FRANCE": ["A PAR"], "GERMANY": None},
                    "centers": {"FRANCE": ["PAR"], "GERMANY": None},
                },
                "orders": {"FRANCE": ["A PAR - BUR"], "GERMANY": None},
            },
            {
                "name": "COMPLETED",
                "state": {
                    "units": None,
                    "centers": {"FRANCE": ["PAR", "MAR", "BRE"], "GERMANY": []},
                },
                "orders": {"FRANCE": None, "GERMANY": None},
            },
        ],
    }
    path = tmp_path / "real.jsonl"
    path.write_text(json.dumps(game) + "\n", encoding="utf-8")
    source = PaquetteJSONLSource(path)
    games = list(source)
    assert not source.skipped  # the whole point: this is data, not corruption
    assert len(games) == 1
    parsed = games[0]
    assert parsed.phases[0].units == {"FRANCE": ["A PAR"], "GERMANY": []}
    assert parsed.phases[0].orders == {"FRANCE": ["A PAR - BUR"], "GERMANY": []}
    assert parsed.phases[-1].name == "COMPLETED"
    assert parsed.final_centers["FRANCE"] == ["PAR", "MAR", "BRE"]


def test_iter_game_records_parses_a_stream() -> None:
    import json

    raws = list(synthesize_games(range(2), max_year=1901))
    lines = iter([json.dumps(raws[0]), "", json.dumps(raws[1])])
    assert len(list(iter_game_records(lines))) == 2
