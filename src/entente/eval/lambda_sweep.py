"""Records and aggregation for the piKL lambda sweep (docs/pikl.tex Sec. 7).

Torch-free: the sweep script writes one JSONL record per game, and this module
owns the schema, the round-trip, the per-lambda aggregation with bootstrap
confidence intervals, and the markdown table -- so every reported number
regenerates from the committed JSONL alone, with nothing typed by hand.

``lam = inf`` (the anchor end of the grid) is serialized as the string
``"inf"`` -- strict-JSON-safe, unlike Python's default ``Infinity`` literal.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from entente.eval.stats import bootstrap_ci

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class SweepRecord:
    """One game of piKL(lam) in ``seat`` vs. six anchor opponents.

    ``anchor_agreement`` is the per-unit rate at which piKL's movement orders
    matched the anchor's greedy decode across the game (the humanlikeness
    proxy that needs no dataset); ``human_agreement`` is the per-unit top-1
    rate against held-out human orders, and is ``None`` when the run supplied no
    ``agreement_shards``. A run without shards records the absence rather than
    an estimate: a number is measured or it is not there.
    """

    lam: float
    seat: str
    seed: int
    sos: float
    supply_centers: int
    survived: bool
    anchor_agreement: float | None
    human_agreement: float | None


def _lam_to_json(lam: float) -> float | str:
    """Render ``lam`` for strict JSON, spelling infinity as the string ``"inf"``."""
    return "inf" if math.isinf(lam) else lam


def write_sweep_jsonl(records: Iterable[SweepRecord], path: str | Path) -> int:
    """One sorted-key JSON object per line; returns the record count."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = asdict(record)
            payload["lam"] = _lam_to_json(record.lam)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def append_sweep_jsonl(records: Iterable[SweepRecord], path: str | Path) -> int:
    """Append ``records`` to ``path``, creating it if absent.

    The line format is :func:`write_sweep_jsonl`'s. This is the durable write
    the sweep performs after each completed ``(lambda, seat)`` batch, so a
    killed run keeps every finished seat: a full-scale sweep accumulates around
    twenty hours of search before it would otherwise write anything, and losing
    that to one interruption is not acceptable.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("a", encoding="utf-8") as handle:
        for record in records:
            payload = asdict(record)
            payload["lam"] = _lam_to_json(record.lam)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count


def read_sweep_jsonl(path: str | Path) -> Iterator[SweepRecord]:
    """Read back what :func:`write_sweep_jsonl` wrote, one record per line."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            data["lam"] = float(data["lam"])
            yield SweepRecord(**data)


def completed_seats(records: Iterable[SweepRecord], games_per_seat: int) -> set[tuple[float, str]]:
    """Find the ``(lam, seat)`` pairs already carrying a full ``games_per_seat`` batch.

    This pair is the sweep's resume unit. A seat written only partially, by a
    run killed mid-append, has too few records and is excluded, so it re-runs
    whole.
    """
    counts: dict[tuple[float, str], int] = {}
    for record in records:
        key = (record.lam, record.seat)
        counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n >= games_per_seat}


def keep_completed_seats(records: Sequence[SweepRecord], games_per_seat: int) -> list[SweepRecord]:
    """Restrict ``records`` to fully completed ``(lam, seat)`` seats.

    Dropping the partial tail of an interrupted run is what lets a resume
    neither double-count a seat nor trust a half-written one.
    """
    done = completed_seats(records, games_per_seat)
    return [r for r in records if (r.lam, r.seat) in done]


@dataclass(frozen=True, slots=True)
class LambdaPoint:
    """One point of the strength/agreement tradeoff curve."""

    lam: float
    games: int
    mean_sos: float
    sos_ci: tuple[float, float]
    survival: float
    anchor_agreement: float | None
    human_agreement: float | None


def aggregate(records: Sequence[SweepRecord]) -> list[LambdaPoint]:
    """Aggregate the records into per-lambda curve points, ascending (inf last)."""
    by_lam: dict[float, list[SweepRecord]] = {}
    for record in records:
        by_lam.setdefault(record.lam, []).append(record)
    points = []
    for lam in sorted(by_lam):
        group = by_lam[lam]
        sos = [r.sos for r in group]
        anchor = [r.anchor_agreement for r in group if r.anchor_agreement is not None]
        human = [r.human_agreement for r in group if r.human_agreement is not None]
        points.append(
            LambdaPoint(
                lam=lam,
                games=len(group),
                mean_sos=sum(sos) / len(sos),
                sos_ci=bootstrap_ci(sos),
                survival=sum(r.survived for r in group) / len(group),
                anchor_agreement=sum(anchor) / len(anchor) if anchor else None,
                human_agreement=sum(human) / len(human) if human else None,
            )
        )
    return points


def _fmt_lam(lam: float) -> str:
    """Render a lambda for the table, spelling infinity as ``inf``."""
    return "inf" if math.isinf(lam) else f"{lam:g}"


def _fmt_opt(value: float | None) -> str:
    """Render an optional rate, showing a missing measurement as ``--``."""
    return "--" if value is None else f"{value:.3f}"


def render_table(points: Sequence[LambdaPoint]) -> str:
    """Render the sweep's results table, in the shape the tournament tables use.

    Every column comes out of the committed JSONL through :func:`aggregate`.
    """
    lines = [
        "| lambda | Games | Mean SoS | 95% bootstrap CI | Survival | Anchor agr. | Human agr. |",
        "|---:|---:|---:|:---|---:|---:|---:|",
    ]
    for p in points:
        lines.append(
            f"| {_fmt_lam(p.lam)} | {p.games} | {p.mean_sos:.4f} "
            f"| [{p.sos_ci[0]:.4f}, {p.sos_ci[1]:.4f}] | {p.survival:.3f} "
            f"| {_fmt_opt(p.anchor_agreement)} | {_fmt_opt(p.human_agreement)} |"
        )
    return "\n".join(lines)


__all__ = [
    "LambdaPoint",
    "SweepRecord",
    "aggregate",
    "append_sweep_jsonl",
    "completed_seats",
    "keep_completed_seats",
    "read_sweep_jsonl",
    "render_table",
    "write_sweep_jsonl",
]
