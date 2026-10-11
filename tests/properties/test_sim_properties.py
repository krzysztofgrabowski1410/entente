"""Assert ``run_parallel`` and ``run_sequential`` agree on identical seed sets.

Running games in parallel must not change what they produce: a seed decides a
game, and how many processes ran it is not part of the game. ``tests/sim``
carries the fixed-seed check of the same claim.

``ProcessPoolExecutor`` start-up (spawn: a fresh
interpreter per worker) is inherently expensive next to hypothesis's usual
per-example budget, so this keeps the example count and game length small --
it is the multiprocessing wiring under test, not the adjudicator (that
differential is test_differential.py).
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from entente.agents.random_agent import RandomAgent
from entente.sim.parallel import SimJob, run_parallel, run_sequential

seed_sets = st.lists(
    st.integers(min_value=0, max_value=2**31 - 1), min_size=1, max_size=5, unique=True
)


@settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seeds=seed_sets)
def test_parallel_matches_sequential_for_identical_seed_sets(seeds: list[int]) -> None:
    jobs = [SimJob(seed=seed, agents=RandomAgent, max_year=1901) for seed in seeds]
    sequential = {outcome.seed: outcome for outcome in run_sequential(jobs)}
    parallel_outcomes = {
        outcome.seed: outcome for outcome in run_parallel(jobs, max_workers=2, chunk_size=2)
    }
    assert sequential == parallel_outcomes
