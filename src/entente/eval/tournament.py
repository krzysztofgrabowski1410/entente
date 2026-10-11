"""Tournament formats, run over :mod:`entente.sim.parallel`.

Two are supported. **1-vs-6** puts one named agent against six of another,
rotated across every seat, so a result is not confounded by any single power's
structural advantage or disadvantage -- England is a harder seat than average
whatever policy sits in it. **Round-robin** has every named agent take every
seat across as many rotations as there are agents. Both use a fixed seed set,
reused across every seat rotation, so a head-to-head comparison holds starting
conditions constant.

Every submitted job gets a unique job seed derived from ``(rotation index,
logical seed)`` (:func:`_encode_seed`/:func:`_decode_seed`) because a
:class:`GameOutcome` carries only its ``seed`` -- with several rotations
sharing the same *logical* seed, the job seed is what identifies which
rotation an outcome belongs to.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from entente.agents.registry import REGISTRY
from entente.sim.parallel import GameOutcome, SimJob, run_parallel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

POWERS: tuple[str, ...] = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")


def _require_standard(ruleset: str, map_name: str) -> None:
    """Refuse anything but the standard ruleset and map.

    :data:`POWERS` fixes the standard map's seven powers, and seat rotation, the
    agent maps and the scoring keys all read them. Another map would seat
    nobody at all, because :func:`entente.sim.parallel.run_parallel` skips
    unmatched power names, and the run would burn its full budget before scoring
    failed -- so the refusal happens here, before a single game starts.
    """
    if ruleset != "standard" or map_name != "standard":
        raise ValueError(
            "the tournament harness supports ruleset='standard'/map_name='standard' only "
            f"(got ruleset={ruleset!r}, map_name={map_name!r}): POWERS pins the standard seats"
        )


@dataclass(frozen=True, slots=True)
class SeatedOutcome:
    """One finished game plus which agent *name* played which power.

    A plain :class:`~entente.sim.parallel.GameOutcome` does not carry the
    seating, and every downstream scoring function needs it.
    """

    outcome: GameOutcome
    seats: dict[str, str]  # power -> agent name


def _encode_seed(index: int, seed: int, modulus: int) -> int:
    """Fold a rotation index and a logical seed into one unique job seed."""
    if seed < 0:
        raise ValueError("seeds must be non-negative for the job-seed encoding to be unique")
    if not 0 <= index < modulus:
        raise ValueError(f"index must be in [0, {modulus})")
    return seed * modulus + index


def _decode_seed(job_seed: int, modulus: int) -> tuple[int, int]:
    """Split a job seed back into ``(index, original_seed)``."""
    return job_seed % modulus, job_seed // modulus


def one_vs_six(
    agent_a: str,
    agent_b: str,
    seeds: Sequence[int],
    *,
    ruleset: str = "standard",
    map_name: str = "standard",
    max_year: int = 1910,
    max_workers: int | None = None,
) -> list[SeatedOutcome]:
    """Run ``agent_a`` through each of the 7 seats against six of ``agent_b``.

    That is ``7 * len(seeds)`` games in total.
    """
    _require_standard(ruleset, map_name)
    factory_a = REGISTRY[agent_a]
    factory_b = REGISTRY[agent_b]
    modulus = len(POWERS)
    jobs = [
        SimJob(
            seed=_encode_seed(seat_index, seed, modulus),
            agents={power: (factory_a if power == seat else factory_b) for power in POWERS},
            max_year=max_year,
        )
        for seat_index, seat in enumerate(POWERS)
        for seed in seeds
    ]
    outcomes = run_parallel(jobs, ruleset=ruleset, map_name=map_name, max_workers=max_workers)
    seated = []
    for outcome in outcomes:
        seat_index, _ = _decode_seed(outcome.seed, modulus)
        seat = POWERS[seat_index]
        seats = {power: (agent_a if power == seat else agent_b) for power in POWERS}
        seated.append(SeatedOutcome(outcome=outcome, seats=seats))
    return seated


def round_robin(
    agent_names: Sequence[str],
    seeds: Sequence[int],
    *,
    ruleset: str = "standard",
    map_name: str = "standard",
    max_year: int = 1910,
    max_workers: int | None = None,
) -> list[SeatedOutcome]:
    """Rotate every named agent through every seat, once per seed.

    There are ``len(agent_names)`` cyclic rotations of :data:`POWERS`: in
    rotation ``r``, power index ``i`` is played by
    ``agent_names[(i + r) % len(agent_names)]``. That comes to
    ``len(agent_names) * len(seeds)`` games.

    Seven is prime, so seat coverage across agents is exactly even only when
    ``len(agent_names)`` is 1 or 7. Any other count still rotates every agent
    through every seat at least once, just not perfectly uniformly -- one name
    may play a given seat once more or once less than another over a full sweep,
    which this baseline harness accepts.
    """
    _require_standard(ruleset, map_name)
    if not agent_names:
        raise ValueError("agent_names must be non-empty")
    factories = [REGISTRY[name] for name in agent_names]
    modulus = len(agent_names)
    jobs = [
        SimJob(
            seed=_encode_seed(rotation, seed, modulus),
            agents={power: factories[(i + rotation) % modulus] for i, power in enumerate(POWERS)},
            max_year=max_year,
        )
        for rotation in range(modulus)
        for seed in seeds
    ]
    outcomes = run_parallel(jobs, ruleset=ruleset, map_name=map_name, max_workers=max_workers)
    seated = []
    for outcome in outcomes:
        rotation, _ = _decode_seed(outcome.seed, modulus)
        seats = {power: agent_names[(i + rotation) % modulus] for i, power in enumerate(POWERS)}
        seated.append(SeatedOutcome(outcome=outcome, seats=seats))
    return seated


# -- JSONL I/O (mirrors entente.sim.parallel.write_jsonl/read_jsonl, plus seats) --


def write_seated_jsonl(results: Iterable[SeatedOutcome], path: str | Path) -> int:
    """Write ``results`` as JSONL: one compact, sorted-key record per line.

    Each record carries every :class:`~entente.sim.parallel.GameOutcome` field
    plus ``seats``, so a committed results file records who played whom without
    needing the tournament config alongside it. Returns the number of records
    written.
    """
    count = 0
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for seated in results:
            record = {**asdict(seated.outcome), "seats": seated.seats}
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def read_seated_jsonl(path: str | Path) -> Iterator[SeatedOutcome]:
    """Read back what :func:`write_seated_jsonl` wrote."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            seats = data.pop("seats")
            yield SeatedOutcome(outcome=GameOutcome(**data), seats=seats)
