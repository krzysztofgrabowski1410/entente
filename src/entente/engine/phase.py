"""The season/phase machine (docs/adjudication.tex Sec. 9).

A game year is the fixed cycle ``S<y>M -> S<y>R -> F<y>M -> F<y>R -> W<y>A``,
then the next year. Phase *tags* (``"S1901M"``) are the exchange format used
in state snapshots, test cases and serialization.

Not every phase in the cycle plays out. The game loop distinguishes three
cases, computed here from pure phase data:

- a phase with **decisions** waits for order sheets (movement with any unit
  on the board; a retreat where some dislodged unit has a real retreat
  option; an adjustment where some power chooses what to build or which of
  several units to remove);
- a phase with **effects only** executes automatically with empty sheets
  (forced disbands of trapped dislodged units -- "destroyed immediately",
  DATC 6.H.16; forced or civil-disorder removals);
- a phase with neither is **skipped** without a history record.

The Autumn supply-center ownership update is tied to the *season boundary*,
not to the winter phase: it runs after the Fall retreat stage settles (even
when that stage was skipped), through the ruleset's
``update_supply_ownership`` hook (Sec. 9; standard = occupants claim centers).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from entente.engine.adjudicator.builds import build_quota, legal_builds
from entente.engine.adjudicator.retreats import legal_retreats

if TYPE_CHECKING:
    from collections.abc import Mapping

    from entente.engine.adjudicator.common import PhaseResult
    from entente.engine.map.graph import MapGraph
    from entente.engine.rules.protocol import Ruleset
    from entente.engine.state import State


class Season(Enum):
    """The three phase seasons; values are the tag letters."""

    SPRING = "S"
    FALL = "F"
    WINTER = "W"


class PhaseKind(Enum):
    """What is being adjudicated; values are the tag letters."""

    MOVEMENT = "M"
    RETREAT = "R"
    ADJUSTMENT = "A"


_CYCLE: tuple[tuple[Season, PhaseKind], ...] = (
    (Season.SPRING, PhaseKind.MOVEMENT),
    (Season.SPRING, PhaseKind.RETREAT),
    (Season.FALL, PhaseKind.MOVEMENT),
    (Season.FALL, PhaseKind.RETREAT),
    (Season.WINTER, PhaseKind.ADJUSTMENT),
)


@dataclass(frozen=True, slots=True)
class Phase:
    """One point of the phase cycle: year, season, kind.

    ``year_step`` is the direction the year counter moves at each cycle
    wrap: ``+1`` for calendar rulesets (standard), ``-1`` for countdown
    rulesets (Beleriand numbers its years 25 down to 0 -- rulebook "Valar's
    intervention"). Tags carry no direction; only the game loop constructs
    descending phases.
    """

    year: int
    season: Season
    kind: PhaseKind
    year_step: int = 1

    def __post_init__(self) -> None:
        if (self.season, self.kind) not in _CYCLE:
            raise ValueError(f"no such phase: {self.season.name} {self.kind.name}")
        if self.year_step not in (1, -1):
            raise ValueError(f"year_step must be +1 or -1, got {self.year_step}")

    @property
    def tag(self) -> str:
        """The canonical tag, e.g. ``"S1901M"``, ``"F1903R"``, ``"W1901A"``."""
        return f"{self.season.value}{self.year}{self.kind.value}"

    def __str__(self) -> str:
        return self.tag

    @classmethod
    def parse(cls, tag: str) -> Phase:
        """Parse a tag back into a :class:`Phase`; raises ``ValueError`` on malformed tags."""
        if len(tag) < 3:
            raise ValueError(f"malformed phase tag: {tag!r}")
        season = Season(tag[0].upper())
        kind = PhaseKind(tag[-1].upper())
        return cls(year=int(tag[1:-1]), season=season, kind=kind)

    def successor(self) -> Phase:
        """Return the next phase of the full cycle (skipping is the game loop's call)."""
        position = _CYCLE.index((self.season, self.kind))
        if position + 1 < len(_CYCLE):
            season, kind = _CYCLE[position + 1]
            return Phase(self.year, season, kind, self.year_step)
        season, kind = _CYCLE[0]
        return Phase(self.year + self.year_step, season, kind, self.year_step)


def retreat_has_decisions(ruleset: Ruleset, graph: MapGraph, movement: PhaseResult) -> bool:
    """Whether any dislodged unit has an actual retreat option.

    Otherwise every outcome is forced and the phase can auto-execute or be
    skipped.
    """
    return any(legal_retreats(ruleset, graph, movement, d) for d in movement.dislodgements)


def retreat_has_effects(movement: PhaseResult) -> bool:
    """Whether the retreat stage changes anything at all: any dislodgement must at least disband."""
    return bool(movement.dislodgements)


def adjustment_has_decisions(graph: MapGraph, state: State, ownership: Mapping[str, str]) -> bool:
    """Whether some power faces an adjustment choice.

    A build-entitled power with at least one legal build site (build vs.
    waive, kind, site), or a removing power with more units than forced
    removals (which ones).
    """
    for power in sorted({unit.power for unit in state.units} | set(ownership.values())):
        quota = build_quota(graph, state, ownership, power)
        if quota > 0 and legal_builds(graph, state, ownership, power):
            return True
        if quota < 0:
            units = sum(1 for unit in state.units if unit.power == power)
            if units > -quota:
                return True
    return False


def adjustment_has_effects(graph: MapGraph, state: State, ownership: Mapping[str, str]) -> bool:
    """Whether the adjustment stage changes anything with empty sheets.

    Removals always execute (by order or civil disorder); unexercised builds
    do not.
    """
    return any(
        build_quota(graph, state, ownership, power) < 0
        for power in {unit.power for unit in state.units}
    )
