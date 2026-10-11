"""entente.sim.parallel: chunking, worker warmup, JSONL round trip, and agreement.

The agreement here is that ``run_parallel`` produces exactly what
``run_sequential`` does; the full property over generated jobs lives in
tests/properties/test_sim_properties.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from entente.agents.random_agent import RandomAgent
from entente.sim import parallel
from entente.sim.parallel import SimJob, read_jsonl, run_parallel, run_sequential, write_jsonl


def test_run_sequential_yields_in_submission_order() -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent, max_year=1901) for seed in (5, 1, 3)]
    outcomes = list(run_sequential(jobs))
    assert [o.seed for o in outcomes] == [5, 1, 3]


def test_repeat_run_is_deterministic() -> None:
    job = SimJob(seed=42, agents=RandomAgent, max_year=1902)
    first = next(iter(run_sequential([job])))
    second = next(iter(run_sequential([job])))
    assert first == second


def test_different_seeds_diverge() -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent, max_year=1902) for seed in range(6)]
    outcomes = list(run_sequential(jobs))
    fingerprints = {(o.final_phase, tuple(sorted(o.supply_counts.items()))) for o in outcomes}
    assert len(fingerprints) > 1


def test_run_parallel_matches_run_sequential() -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent, max_year=1901) for seed in range(1, 5)]
    sequential = {o.seed: o for o in run_sequential(jobs)}
    parallel_outcomes = {o.seed: o for o in run_parallel(jobs, max_workers=2, chunk_size=2)}
    assert sequential == parallel_outcomes


def test_run_parallel_empty_jobs_yields_nothing() -> None:
    assert list(run_parallel([])) == []


def test_heterogeneous_agent_map_only_drives_named_powers() -> None:
    """A power missing from the agent map submits nothing, every phase.

    The engine resolves that with its phase default, a hold, and the game still
    runs to completion without error.
    """
    job = SimJob(seed=1, agents={"FRANCE": RandomAgent}, max_year=1901)
    outcome = next(iter(run_sequential([job])))
    assert set(outcome.supply_counts) == {
        "AUSTRIA",
        "ENGLAND",
        "FRANCE",
        "GERMANY",
        "ITALY",
        "RUSSIA",
        "TURKEY",
    }
    assert sum(outcome.supply_counts.values()) >= 22  # home centers are never unowned


def test_jsonl_round_trip(tmp_path: Path) -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent, max_year=1901) for seed in range(3)]
    outcomes = list(run_sequential(jobs))
    path = tmp_path / "results.jsonl"

    count = write_jsonl(outcomes, path)

    assert count == len(outcomes)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(outcomes)
    assert all(line == line.strip() for line in lines)  # one compact record per line
    assert list(read_jsonl(path)) == outcomes


def test_read_jsonl_skips_blank_lines(tmp_path: Path) -> None:
    outcome = next(iter(run_sequential([SimJob(seed=1, agents=RandomAgent, max_year=1901)])))
    path = tmp_path / "with_blanks.jsonl"
    write_jsonl([outcome], path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert list(read_jsonl(path)) == [outcome]


def test_chunked_splits_into_expected_groups() -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent) for seed in range(5)]
    chunks = list(parallel._chunked(jobs, 2))
    assert [len(chunk) for chunk in chunks] == [2, 2, 1]
    assert [job.seed for chunk in chunks for job in chunk] == [0, 1, 2, 3, 4]


def test_init_worker_rejects_unknown_ruleset() -> None:
    with pytest.raises(ValueError, match="unknown ruleset"):
        parallel._init_worker("not-a-real-ruleset", "standard")
