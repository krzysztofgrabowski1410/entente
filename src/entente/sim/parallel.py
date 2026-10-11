"""Multi-process game rollouts across a spawn-safe process pool.

Engine-only rollouts (CPU-bound agents: random, greedy, search -- no GPU
policy in the loop; that is :mod:`entente.sim.vectorized`) parallelize
cleanly across processes. Windows' default (and only) start method is
*spawn* -- a fresh interpreter per worker, nothing inherited from the parent
by copy-on-write -- so this module follows the spawn-safe discipline
everywhere, including on platforms where fork is available, to keep
behaviour identical across the CI matrix -- one mechanism, never a
platform-conditional one:

- every function submitted to the pool is a **module-level** function,
  picklable by qualified name (no closures, lambdas, or bound methods);
- the map and ruleset are loaded **once per worker** by the pool
  ``initializer``, not per job -- ``load_map`` is itself process-local
  (``functools.cache``), and a spawned worker starts with that cache cold;
- jobs are dispatched in **chunks** (~50-100 games) so 1000s of short games
  do not pay one IPC round trip each.

Agent factories (``AgentFactory``) carried on a :class:`SimJob` are pickled
by reference too, so they must be module-level classes or functions (e.g.
``RandomAgent``, not a lambda wrapping it).
"""

from __future__ import annotations

import json
import multiprocessing
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from entente.agents.base import Agent
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.rules import RULESETS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

AgentFactory = Callable[[int], Agent]
"""Builds one seeded agent instance, e.g. ``RandomAgent`` itself (its
``__init__(self, seed: int)`` already matches this shape)."""

DEFAULT_CHUNK_SIZE = 75
"""Games per pool task -- amortizes per-task dispatch overhead without leaving
workers idle near the end of a run."""


@dataclass(frozen=True, slots=True)
class GameOutcome:
    """One finished (or year-capped) game -- the JSONL record unit."""

    seed: int
    ruleset: str
    map_name: str
    max_year: int
    num_phases: int
    final_phase: str
    winner: str | None
    supply_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class SimJob:
    """One game to run: a seed plus its agent(s).

    ``agents`` is either a single factory applied to every power (self-play,
    the common case -- matches ``entente play``'s ``--agents`` flag) or a
    per-power mapping for heterogeneous matchups (e.g. a 1-vs-6 tournament).
    A power missing from a mapping gets no agent and always submits nothing,
    which the engine resolves with the phase defaults.
    """

    seed: int
    agents: AgentFactory | Mapping[str, AgentFactory]
    max_year: int = 1910


def _resolve_agents(
    game: Game, agents: AgentFactory | Mapping[str, AgentFactory], seed: int
) -> dict[str, Agent]:
    """Build one seeded agent per power, from a single factory or a per-power map."""
    if isinstance(agents, Mapping):
        return {power: factory(seed) for power, factory in agents.items()}
    return {power: agents(seed) for power in game.powers}


def _play(
    seed: int,
    agents: AgentFactory | Mapping[str, AgentFactory],
    *,
    ruleset: str,
    map_name: str,
    max_year: int,
) -> GameOutcome:
    """Run one game to completion or to the year cap.

    This is ``entente play``'s loop, generalized to per-power agents.
    """
    game = Game.new(ruleset, map_name)
    resolved = _resolve_agents(game, agents, seed)
    while not game.is_over and game.year <= max_year:
        for power in game.powers:
            agent = resolved.get(power)
            if agent is None:
                continue
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
    return GameOutcome(
        seed=seed,
        ruleset=ruleset,
        map_name=map_name,
        max_year=max_year,
        num_phases=len(game.history),
        final_phase=game.phase,
        winner=game.winner,
        supply_counts={power: game.supply_count(power) for power in game.powers},
    )


def run_sequential(
    jobs: Iterable[SimJob], *, ruleset: str = "standard", map_name: str = "standard"
) -> Iterator[GameOutcome]:
    """Run ``jobs`` one at a time in this process, in submission order.

    This is the correctness baseline :func:`run_parallel` is checked against.
    """
    for job in jobs:
        yield _play(job.seed, job.agents, ruleset=ruleset, map_name=map_name, max_year=job.max_year)


# -- ProcessPoolExecutor runner -----------------------------------------------

_worker_ruleset: str | None = None
_worker_map: str | None = None


def _init_worker(ruleset: str, map_name: str) -> None:
    """Warm ``load_map``'s cache once per worker process, as the pool initializer.

    It also fails fast on an unknown ruleset or map, before any job reaches it.
    """
    global _worker_ruleset, _worker_map
    if ruleset not in RULESETS:
        raise ValueError(f"unknown ruleset {ruleset!r}")
    load_map(map_name)
    _worker_ruleset = ruleset
    _worker_map = map_name


def _run_chunk(chunk: Sequence[SimJob]) -> list[GameOutcome]:
    """Pool task: run every job in ``chunk`` sequentially inside one worker."""
    assert _worker_ruleset is not None  # _init_worker always runs first
    assert _worker_map is not None
    return [
        _play(
            job.seed,
            job.agents,
            ruleset=_worker_ruleset,
            map_name=_worker_map,
            max_year=job.max_year,
        )
        for job in chunk
    ]


def _chunked(jobs: Sequence[SimJob], size: int) -> Iterator[Sequence[SimJob]]:
    """Cut ``jobs`` into consecutive slices of at most ``size`` entries."""
    for start in range(0, len(jobs), size):
        yield jobs[start : start + size]


def run_parallel(
    jobs: Sequence[SimJob],
    *,
    ruleset: str = "standard",
    map_name: str = "standard",
    max_workers: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterator[GameOutcome]:
    """Run ``jobs`` across a spawn-context :class:`ProcessPoolExecutor`, in chunks.

    Every job shares ``ruleset`` and ``map_name``, and those load once per
    worker rather than once per job.

    Results arrive in submission order (``Executor.map`` yields chunks in
    input order however their workers interleave), and every outcome carries
    its ``GameOutcome.seed``, so downstream code may key by seed instead of
    position -- the determinism check in
    ``tests/properties/test_sim_properties.py`` compares the two runners
    exactly that way.
    """
    job_list = list(jobs)
    if not job_list:
        return
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(ruleset, map_name),
    ) as pool:
        for outcomes in pool.map(_run_chunk, _chunked(job_list, chunk_size)):
            yield from outcomes


# -- JSONL I/O -----------------------------------------------------------------


def write_jsonl(outcomes: Iterable[GameOutcome], path: str | Path) -> int:
    """Write ``outcomes`` as JSONL: one compact record per line, keys sorted.

    Returns the number of records written.
    """
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for outcome in outcomes:
            handle.write(json.dumps(asdict(outcome), sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> Iterator[GameOutcome]:
    """Read back what :func:`write_jsonl` wrote."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data: dict[str, Any] = json.loads(line)
            yield GameOutcome(**data)
