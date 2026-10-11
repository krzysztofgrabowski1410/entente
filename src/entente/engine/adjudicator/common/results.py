"""Result and record types: what an adjudication *is*, before any of it runs.

The output contract of docs/adjudication.tex Sec. 5.1 (:class:`PhaseResult` and
the rows it carries), the Phase-0 record the rest of the package resolves over
(:class:`OrderRecord`, :class:`Phase0`), and the internal-invariant exception.
Pure data with no adjudication logic, so every other module in the package may
import it and none of them are imported back: this is the package's leaf.

This module changes only when the *result schema* -- what a caller is promised
-- changes, which is independent of every rule that computes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from entente.engine.orders import format_order, parse_order

if TYPE_CHECKING:
    from entente.engine.orders import Order
    from entente.engine.state import State, Unit


_UNORDERED_SHEET_POS = 10**9
"""Sheet position of units the power left unordered.

They sort after every written order, ties broken by canonical record index (the
Sec. 4.4 victim-selection key).
"""


class AdjudicationError(Exception):
    """An internal invariant of the calculus failed: the spec, not the input, is at fault."""


class Resolution(Enum):
    """Final status of a written order (Sec. 5.1)."""

    SUCCEEDS = "succeeds"
    FAILS = "fails"


@dataclass(frozen=True, slots=True)
class OrderResolution:
    """One written order's outcome, in canonical notation.

    ``reason`` is an optional human sentence naming the rule that voided the
    order. It is *explanatory only* -- nothing in the calculus reads it, and a
    row's ``resolution`` is what it would have been with the field absent, so
    attaching one never perturbs adjudication. Only failures a player cannot
    diagnose from the board carry one; an ordinary bounce explains itself and
    leaves it None.
    """

    power: str
    text: str
    resolution: Resolution
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class Dislodgement:
    """A dislodged unit plus what the retreat phase needs to know about its dislodger.

    The dislodger's province of origin and whether it arrived by convoy (DATC
    4.A.5 choice B reads both; Sec. 9). ``dislodger_origin`` is None for a
    dislodgement with no attacking unit at all -- Beleriand's devastation
    expels every co-located unit except Morgoth's own and the mask carriers --
    and the retreat validator's origin-province ban (Sec. 9) is a plain
    equality test against a province name, so it never fires on a None
    origin, exactly the "no dislodger" reading.

    ``special`` is the designator the dislodged unit's effective order writes
    (Sec. 1: the adjudicator knows a special only by the designator its own
    order writes), so a retreat order that writes a designator binds only the
    dislodgement of the special it names, and is valid or not for that
    special's own dislodger (Sec. 9; the Morgoth unit and Glaurung are never
    twins, Sec. 4.4). None for a unit whose order writes none -- in a
    Beleriand game every unit but the two specials, whose Holds the game layer
    writes when they have no order, and every unit a devastation expels. A
    standard order may still write one (Sec. 1's advisory designator); the
    record keeps it, and the advisory retreat binding gives the province's one
    dislodged unit its order whatever either side writes.
    """

    unit: Unit
    dislodger_origin: str | None
    by_convoy: bool
    special: str | None = None


@dataclass(frozen=True, slots=True)
class PhaseResult:
    """The output of one movement-phase adjudication.

    Per-order resolutions in sheet order (powers sorted), the post-movement
    state (dislodged units are off-board awaiting the retreat phase; the phase
    tag is owned by the game loop), dislodgement records, and the provinces
    embattled by a standoff (Sec. 5.1).

    ``embattled`` is the retreat-phase input "left contested by a standoff"
    (Sec. 9): provinces where no entry succeeded although two or more
    attackers contested entry *live* -- existing as attackers (Sec. 3.1.4, so a
    non-viable or szykmaned convoyed move never embattles its landing
    province, DATC 6.F.7) and not prevent-nulled (4.A.7 choice B; a single
    beaten head-to-head entrant leaves the province open, DATC 6.H.9). Live
    Dislodge attempts count, since they contest their Area, so a standoff of
    dislodgers embattles it; a Dislodge whose unit is dislodged by an attack
    from that Area is prevent-nulled there as a Move is (BATC.G.6). Computed
    here because retreat options are determined by movement-phase data alone
    (DATC 6.H.13).
    """

    resolutions: tuple[OrderResolution, ...]
    new_state: State
    dislodgements: tuple[Dislodgement, ...]
    embattled: frozenset[str]

    def resolution_of(self, text: str) -> Resolution:
        """Look up a written order's outcome; ``text`` is normalized first."""
        canonical = format_order(parse_order(text))
        for entry in self.resolutions:
            if entry.text == canonical:
                return entry.resolution
        raise KeyError(f"no resolution recorded for {canonical!r}")


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """One unit occurrence's effective order after Phase 0.

    ``written`` is the submitted order (None for units left unordered);
    ``effective`` is what adjudication executes -- the written order if valid,
    else Hold (Sec. 1). ``index`` is the record's position in the canonical
    ordering (by unit location), the determinism anchor for Phase 1 and all
    residue tie-breaks. ``sheet_pos`` is the position of the winning claim in
    its power's submitted sheet (unordered units sort last) -- the Sec. 4.4
    victim-selection key.
    """

    index: int
    power: str
    unit: Unit
    written: Order | None
    effective: Order
    valid: bool
    sheet_pos: int = _UNORDERED_SHEET_POS


@dataclass(frozen=True, slots=True)
class Phase0:
    """Phase 0 output: the canonical records plus the rejected written orders.

    A written order is rejected when it names no matchable own unit or when an
    earlier order in the sheet already claimed the unit (the first legal order
    wins, DATC 4.D.3); the rejects are kept, in submission order, so every
    written row can be reported as FAILS.

    ``illegal`` holds the indices of the records whose every written order was
    illegal (Sec. 1): none of them is the unit's order, and ``written`` keeps
    the first only so that its row is reported failing. The adjudicator reads
    such a record as it reads any invalid one -- a Hold -- and never reads this
    set; a game layer that resolves orders of its own after the adjudicator
    reads it to tell an ignored order from a legal one that proved invalid,
    which keeps its unit.
    """

    records: tuple[OrderRecord, ...]
    rejected: tuple[tuple[str, Order], ...]
    illegal: frozenset[int] = frozenset()
