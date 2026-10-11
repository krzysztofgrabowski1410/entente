"""Press-vs-no-press evaluation records and their aggregation.

Torch-free, and with **no** import of ``entente.negotiation``. The dependency
runs strictly one way: ``entente.negotiation.press_eval``, which produces these
records, reaches into this module, and nothing here reaches back, which is what
keeps ``entente.eval`` importable in the core install with no language-model
stack present. It is the same split :mod:`entente.eval.lambda_sweep` already has
with the piKL sweep -- the schema and the aggregation are core, the game-running
harness is not.

One record per full self-play game, reporting the designated **subject** seat's
own outcome only. :mod:`entente.negotiation.press_eval`'s module docstring
explains why: averaging every seat's share under a fully symmetric press toggle
is a tautology fixed at 1/7, so one rotating subject seat -- mirroring
:mod:`entente.eval.lambda_sweep`'s one-record-per-seat convention -- is what
makes a press-vs-no-press delta a real, checkable quantity. ``sos_delta`` is the
headline number: the subject's mean SoS under press minus under no-press, with a
permutation-test p-value, both computed straight from the JSONL at whatever
sample size was actually run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from entente.eval.stats import bootstrap_ci, permutation_test

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


@dataclass(frozen=True, slots=True)
class PressRecord:
    """The subject seat's outcome in one full self-play game, under one condition.

    The condition is ``press``, on or off. ``seat`` is the subject for that
    game, which is constant unless the run rotates seats.
    """

    press: bool
    seat: str
    seed: int
    sos: float
    supply_centers: int
    survived: bool


def write_press_jsonl(records: Iterable[PressRecord], path: str | Path) -> int:
    """Write one sorted-key JSON object per line, returning the record count.

    The format is :func:`entente.eval.lambda_sweep.write_sweep_jsonl`'s.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
            count += 1
    return count


def read_press_jsonl(path: str | Path) -> Iterator[PressRecord]:
    """Read back what :func:`write_press_jsonl` wrote, one record per line."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield PressRecord(**json.loads(line))


@dataclass(frozen=True, slots=True)
class PressSummary:
    """One condition's aggregate over every game it produced.

    Games are pooled across subject seats when the run rotates them.
    """

    press: bool
    games: int
    mean_sos: float
    sos_ci: tuple[float, float]
    survival: float


def summarize(records: Sequence[PressRecord]) -> dict[bool, PressSummary]:
    """Summarize each condition, keyed by ``press``.

    A condition with no records is absent from the mapping rather than present
    with zeros: absence is the honest record of a run that did not happen.
    """
    out: dict[bool, PressSummary] = {}
    for press in (False, True):
        group = [r for r in records if r.press is press]
        if not group:
            continue
        sos = [r.sos for r in group]
        out[press] = PressSummary(
            press=press,
            games=len(group),
            mean_sos=sum(sos) / len(sos),
            sos_ci=bootstrap_ci(sos),
            survival=sum(r.survived for r in group) / len(group),
        )
    return out


def sos_delta(records: Sequence[PressRecord]) -> tuple[float, float]:
    """Compute ``(delta, permutation p-value)`` for press against no-press mean SoS.

    This is the headline number of the whole press evaluation. A positive
    ``delta`` means press *helped* the subject; a null or negative delta is a
    result, not a failure, and is reported exactly as computed either way.
    """
    press_scores = [r.sos for r in records if r.press]
    no_press_scores = [r.sos for r in records if not r.press]
    if not press_scores or not no_press_scores:
        # match the module's empty-input policy (summarize documents absence;
        # a headline delta over one condition is meaningless) and the error
        # shape permutation_test would raise a line later anyway.
        raise ValueError("sos_delta needs records under both conditions (press and no-press)")
    delta = (sum(press_scores) / len(press_scores)) - (sum(no_press_scores) / len(no_press_scores))
    p_value = permutation_test(press_scores, no_press_scores)
    return delta, p_value


def render_table(records: Sequence[PressRecord]) -> str:
    """Render the press results table: every column comes from the JSONL."""
    summaries = summarize(records)
    lines = [
        "| Condition | Games | Mean SoS (subject) | 95% bootstrap CI | Survival |",
        "|:--|--:|--:|:--|--:|",
    ]
    for press in (False, True):
        summary = summaries.get(press)
        if summary is None:
            continue
        label = "press" if press else "no press"
        lines.append(
            f"| {label} | {summary.games} | {summary.mean_sos:.4f} "
            f"| [{summary.sos_ci[0]:.4f}, {summary.sos_ci[1]:.4f}] | {summary.survival:.3f} |"
        )
    return "\n".join(lines)


__all__ = [
    "PressRecord",
    "PressSummary",
    "read_press_jsonl",
    "render_table",
    "sos_delta",
    "summarize",
    "write_press_jsonl",
]
