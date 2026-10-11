"""Measure self-play throughput, one core against eight workers.

The design targets are at least five full random games per second per core,
and at least thirty per second across eight workers, which is the rate a
training run needs its data at.

Run: ``uv run pytest benchmarks/bench_selfplay.py --benchmark-only``.
``testpaths`` excludes ``benchmarks/`` from a plain ``uv run pytest``.

The eight-worker case batches many games into each round
(``PARALLEL_BATCH``), so it pays ``ProcessPoolExecutor`` spin-up once per
round rather than once per game. The rate is therefore ``PARALLEL_BATCH /
benchmark.stats.mean``, not ``benchmark.stats.ops``, which would count
batches per second.
"""

from __future__ import annotations

import itertools
from typing import Any

from entente.agents.random_agent import RandomAgent
from entente.sim.parallel import DEFAULT_CHUNK_SIZE, SimJob, run_parallel, run_sequential

BENCH_MAX_YEAR = 1905
"""Cap games at a few years deep: long enough to be representative of real
self-play cost, short enough that the suite finishes promptly."""

WORKERS = 8

PARALLEL_BATCH = DEFAULT_CHUNK_SIZE * WORKERS
"""Games per 8-worker round -- must span **at least** ``WORKERS`` chunks at
the default chunk size, or every game lands in one chunk and only one
worker ever runs; the rest sit idle, pure spin-up overhead with zero
throughput contribution."""


def test_bench_selfplay_single_core(benchmark: Any) -> None:
    """Time one full random game played to completion on a single core."""
    seeds = itertools.count()

    def play_one() -> None:
        job = SimJob(seed=next(seeds), agents=RandomAgent, max_year=BENCH_MAX_YEAR)
        list(run_sequential([job]))

    benchmark(play_one)


def test_bench_selfplay_8_workers(benchmark: Any) -> None:
    """Time a batch of full random games played across eight worker processes."""
    seeds = itertools.count()

    def play_batch() -> None:
        jobs = [
            SimJob(seed=next(seeds), agents=RandomAgent, max_year=BENCH_MAX_YEAR)
            for _ in range(PARALLEL_BATCH)
        ]
        list(run_parallel(jobs, max_workers=WORKERS))

    benchmark.extra_info["batch_size"] = PARALLEL_BATCH
    benchmark.extra_info["workers"] = WORKERS
    benchmark.pedantic(play_batch, rounds=3, iterations=1)
