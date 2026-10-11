"""tournament.py: the job-seed encoding, then the seating structure of a sweep.

The structural checks over small ``one_vs_six`` and ``round_robin`` sweeps
assert that every seat is visited the right number of times by the right agent
name. The games are real, but tiny -- a low year cap and few seeds -- so the
suite stays fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.eval.tournament import (
    POWERS,
    _decode_seed,
    _encode_seed,
    one_vs_six,
    read_seated_jsonl,
    round_robin,
    write_seated_jsonl,
)


@pytest.mark.parametrize(
    ("modulus", "index", "seed"),
    [
        (7, 0, 0),
        (7, 6, 0),  # index at the top of its range
        (7, 3, 41),
        (2, 0, 5),
        (2, 1, 5),
    ],
)
def test_seed_encoding_round_trips(modulus: int, index: int, seed: int) -> None:
    job_seed = _encode_seed(index, seed, modulus)
    assert _decode_seed(job_seed, modulus) == (index, seed)


def test_encode_seed_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _encode_seed(0, -1, 7)


def test_encode_seed_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="index"):
        _encode_seed(7, 0, 7)


def test_one_vs_six_runs_seven_times_len_seeds_games() -> None:
    seeds = [0, 1]
    results = one_vs_six("greedy", "random", seeds, max_year=1901, max_workers=2)
    assert len(results) == 7 * len(seeds)


def test_one_vs_six_puts_agent_a_in_every_seat_exactly_len_seeds_times() -> None:
    seeds = [0, 1, 2]
    results = one_vs_six("greedy", "random", seeds, max_year=1901, max_workers=2)

    seat_counts = dict.fromkeys(POWERS, 0)
    for seated in results:
        for name in seated.seats.values():
            assert name in ("greedy", "random")
        greedy_seats = [power for power, name in seated.seats.items() if name == "greedy"]
        assert len(greedy_seats) == 1  # exactly one seat for agent_a
        assert len({name for name in seated.seats.values() if name == "random"}) == 1
        seat_counts[greedy_seats[0]] += 1

    assert all(count == len(seeds) for count in seat_counts.values())


def test_one_vs_six_seats_match_the_games_actual_supply_counts() -> None:
    results = one_vs_six("greedy", "random", [0], max_year=1901, max_workers=1)
    for seated in results:
        assert set(seated.outcome.supply_counts) == set(POWERS)
        assert set(seated.seats) == set(POWERS)


def test_round_robin_runs_len_agents_times_len_seeds_games() -> None:
    agents = ["greedy", "random"]
    seeds = [0, 1, 2]
    results = round_robin(agents, seeds, max_year=1901, max_workers=2)
    assert len(results) == len(agents) * len(seeds)


def test_round_robin_every_agent_plays_every_seat_at_least_once() -> None:
    agents = ["greedy", "random"]
    results = round_robin(agents, [0, 1], max_year=1901, max_workers=2)

    played_by: dict[str, set[str]] = {name: set() for name in agents}
    for seated in results:
        for power, name in seated.seats.items():
            played_by[name].add(power)

    for name in agents:
        # 2 agents x 2 cyclic rotations cover all seven powers per agent; the
        # weaker "at least one seat" form could not fail even if rotation broke.
        assert played_by[name] == set(POWERS)


def test_non_standard_map_or_ruleset_is_rejected_before_any_game() -> None:
    # POWERS names the standard seats, so another map would seat nobody at
    # all: the harness must refuse rather than burn a whole run first.
    with pytest.raises(ValueError, match="standard"):
        one_vs_six("greedy", "random", [0], map_name="beleriand", max_year=1901)
    with pytest.raises(ValueError, match="standard"):
        round_robin(["greedy", "random"], [0], ruleset="beleriand", max_year=1901)


def test_round_robin_rejects_empty_agent_list() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        round_robin([], [0])


def test_one_vs_six_unknown_agent_raises_key_error() -> None:
    with pytest.raises(KeyError):
        one_vs_six("nonexistent", "random", [0], max_year=1901)


def test_seated_jsonl_round_trip(tmp_path: Path) -> None:
    results = one_vs_six("greedy", "random", [0, 1], max_year=1901, max_workers=2)
    path = tmp_path / "results.jsonl"

    count = write_seated_jsonl(results, path)

    assert count == len(results)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(results)
    assert list(read_seated_jsonl(path)) == results


def test_read_seated_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    results = one_vs_six("greedy", "random", [0], max_year=1901, max_workers=1)
    path = tmp_path / "with_blanks.jsonl"
    write_seated_jsonl(results, path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert list(read_seated_jsonl(path)) == results
