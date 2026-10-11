"""Result assembly: a settled labeling becomes a :class:`PhaseResult`.

The last step both engines share: read each written order's outcome off the
derived predicates, build the successor state, record dislodgements with what
the retreat phase needs, and compute the embattled provinces. Nothing here
decides anything -- by the time :func:`assemble` runs, every relocating status
is final; it only *reports* (docs/adjudication.tex Sec. 5.1).

Kept apart from ``selection`` because this module changes only with the output
schema and the FAILS-reporting rules (which written row maps to which record),
never with the calculus.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.adjudicator.common.phase1 import entry_province
from entente.engine.adjudicator.common.results import (
    AdjudicationError,
    Dislodgement,
    OrderResolution,
    PhaseResult,
    Resolution,
)
from entente.engine.map.types import Location
from entente.engine.orders import Move, format_order
from entente.engine.rules.strength import is_relocating
from entente.engine.state import State, Unit, check_state

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.adjudicator.common.evaluator import Evaluator
    from entente.engine.adjudicator.common.results import OrderRecord, Phase0
    from entente.engine.map.graph import MapGraph
    from entente.engine.orders import Order


def assemble(
    graph: MapGraph,
    state: State,
    sheets: Mapping[str, Sequence[Order]],
    parsed: Phase0,
    evaluator: Evaluator,
) -> PhaseResult:
    """Build the :class:`PhaseResult` from a fully decided labeling.

    Per-written-order resolutions come off the derived predicates; the
    successor state and the dislodgement records carry what the retreat phase
    needs (Sec. 5.1).
    """
    records = parsed.records

    def final_status(record: OrderRecord) -> Resolution:
        if not record.valid:
            return Resolution.FAILS
        return Resolution.SUCCEEDS if evaluator.derived_status(record.index) else Resolution.FAILS

    # Every written order gets a row, in sheet order -- textual duplicates
    # included: the k-th duplicate binds the k-th record claiming that text
    # (claim order = recorded sheet_pos), so two fungible units under identical
    # orders report their own outcomes. A duplicate beyond the number of
    # claiming records bound no unit and FAILS (the 4.D.3 redundant-order shape).
    resolutions: list[OrderResolution] = []
    consumed: dict[tuple[str, str], int] = {}
    for power in sorted(sheets):
        for order in sheets[power]:
            text = format_order(order)
            occurrence = consumed.get((power, text), 0)
            consumed[(power, text)] = occurrence + 1
            matches = sorted(
                (c for c in records if c.power == power and c.written == order),
                key=lambda c: c.sheet_pos,
            )
            record = matches[occurrence] if occurrence < len(matches) else None
            resolution = final_status(record) if record is not None else Resolution.FAILS
            resolutions.append(OrderResolution(power=power, text=text, resolution=resolution))

    dislodgements: list[Dislodgement] = []
    surviving: list[Unit] = []
    for record in records:
        order = record.effective
        if is_relocating(order) and evaluator.succeeded(record.index):
            target = entry_province(order)
            assert target is not None
            landing = order.destination if isinstance(order, Move) else Location(target)
            surviving.append(Unit(record.power, record.unit.kind, landing, record.unit.strength))
            continue
        if evaluator.unit_dislodged(record.index):
            province = record.unit.location.province
            entrants = [
                d
                for d in evaluator.tables.dislodges_on.get(record.index, ())
                if evaluator.succeeded(d)
            ]
            if not entrants:
                walker = evaluator._stack_dislodger(record.index)
                if walker is not None:
                    entrants = [walker]
            if not entrants:
                entrants = [
                    d
                    for d in evaluator.tables.into.get(province, ())
                    if d not in evaluator.szykmaned
                    and isinstance(evaluator.records[d].effective, Move)
                    and evaluator.succeeded(d)
                    and evaluator.records[d].power != record.power
                ]
            if len(entrants) != 1:  # pragma: no cover - single-dislodger invariant (L3/Sec. 4.2)
                raise AdjudicationError(f"expected one dislodger in {province}, got {entrants}")
            dislodger = evaluator.records[entrants[0]]
            dislodgements.append(
                Dislodgement(
                    unit=record.unit,
                    dislodger_origin=dislodger.unit.location.province,
                    by_convoy=evaluator.is_convoyed(entrants[0]),
                    special=record.effective.special,
                )
            )
            continue
        surviving.append(record.unit)

    new_state = State.of(state.map_name, state.phase, surviving)
    check_state(graph, new_state)
    dislodgements.sort(
        key=lambda d: (
            d.unit.location.province,
            d.unit.location.coast or "",
            d.unit.power,
            d.unit.kind.value,
            d.unit.strength,
        )
    )

    embattled: set[str] = set()
    for province, movers in evaluator.tables.into.items():
        live = [
            mover
            for mover in movers
            if mover not in evaluator.szykmaned
            and evaluator.exists_as_attacker(mover)
            and not evaluator.beaten_in_clash(mover)
            and not evaluator.prevent_nulled(mover, province)
        ]
        if len(live) >= 2 and not any(evaluator.succeeded(mover) for mover in live):
            embattled.add(province)

    return PhaseResult(
        resolutions=tuple(resolutions),
        new_state=new_state,
        dislodgements=tuple(dislodgements),
        embattled=frozenset(embattled),
    )
