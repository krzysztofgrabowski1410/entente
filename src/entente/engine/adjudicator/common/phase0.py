"""Phase 0: order sheets become canonical records (docs/adjudication.tex Sec. 1, Sec. 6).

Sheets stay *ordered*; orders bind to the power's own unit occurrences in sheet
order (multiset occupancy makes binding claim-based: written kind/strength
narrow the match, and the sheet position of the winning claim is recorded --
Beleriand's Sec. 4.4 selection reads it); of several orders to one unit the
*first* legal one wins, illegal ones are ignored (DATC 4.D.3/4.E.1); orders
naming no matchable own unit are dropped; well-formed but invalid orders
(terrain, adjacency, entry permission, unmatched support/convoy subjects,
unreachable support targets, out-of-registry types) become Hold. Invalidity
here is static -- it never depends on resolution. A Dislodge whose *victim
reference* resolves to no unit stays valid and simply fails (Sec. 3.4):
mis-targeting is resolution semantics, not order illegality.

This module changes only with the DATC and rulebook order-binding law, which
no later stage shares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from entente.engine.adjudicator.common.geometry import potential_chain
from entente.engine.adjudicator.common.matching import (
    _ref_special_ok,
    _spec_matches_unit,
    _unit_key,
)
from entente.engine.adjudicator.common.results import (
    _UNORDERED_SHEET_POS,
    OrderRecord,
    Phase0,
)
from entente.engine.map.types import Location, Terrain, UnitKind
from entente.engine.orders import (
    RELOCATING_TYPES,
    Bar,
    Convoy,
    Dislodge,
    Hold,
    Move,
    Order,
    Support,
    kind_reaches_province,
    static_move_legal,
)
from entente.engine.rules.strength import refs_compatible, support_matches
from entente.engine.state import State, Unit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from entente.engine.map.graph import MapGraph
    from entente.engine.rules.protocol import Ruleset


class _Phase0State:
    """Working state of Phase 0: canonical unit slots and claim bookkeeping."""

    def __init__(self, graph: MapGraph, ruleset: Ruleset, state: State) -> None:
        self.graph = graph
        self.ruleset = ruleset
        self.units: list[Unit] = sorted(state.units, key=_unit_key)
        self.claimed: dict[int, tuple[str, Order, int]] = {}
        """slot -> (power, written order, sheet position)."""
        self.fallback: dict[int, tuple[str, Order, int]] = {}
        """slot -> the first illegal claim, the record's order should no legal order claim it."""
        self.sea_fleets: dict[str, list[int]] = {}
        """sea province -> the slots of the fleets standing in it."""
        for index, unit in enumerate(self.units):
            if unit.kind is UnitKind.FLEET and graph.terrain(unit.location.province) is Terrain.SEA:
                self.sea_fleets.setdefault(unit.location.province, []).append(index)

    def capable_seas(self, cargo: Unit, without: int | None = None) -> frozenset[str]:
        """Sea provinces holding at least one fleet able to convoy ``cargo`` (Sec. 3.5).

        ``without`` is the slot of one fleet to leave out (by slot: twin fleets are equal
        values, and the twin must stay).
        """
        return frozenset(
            sea
            for sea, fleets in self.sea_fleets.items()
            if any(
                self.ruleset.convoy_capability(self.units[fleet], cargo)
                for fleet in fleets
                if fleet != without
            )
        )

    def claim_slot(self, power: str, order: Order) -> int | None:
        """Bind an order to the first unclaimed own unit occurrence it can describe.

        Exact spec matches (kind + written strength) come first, then the
        province-only reading (written kind and strength are advisory here --
        DATC 4.C.2 choice b). Under strict binding (4.C.2 choice C) the
        province-only reading survives only for a spec that describes no unit
        there at all, which :meth:`designation_contradicts` then marks
        invalid, so no mismatched occupant is ever *executed*; a spec that does
        describe units there, all of which already have their orders, binds
        nothing -- "A" never stands for an Army of another strength. The
        advisory branch is untouched, which keeps the standard column
        byte-identical by construction.

        A written ``(M)``/``(G)`` designator names one unit: where an earlier
        legal order of the sheet already claimed an occurrence in the
        province under the same designator, the special it names has its
        order, and this one binds nothing -- it is passed over like any second
        order for one unit, and the other special's own order or Hold claims
        the other occurrence (docs/adjudication.tex Sec. 1, Sec. 6). A
        standard province holds one unit, so there a second order for it
        binds nothing with or without this test, and the standard column is
        untouched by it as well.

        An illegal order holds the occurrence it binds only as a fallback
        (``fallback``), which a later legal order for its unit takes over. To
        an order that writes a designator, a fallback writing another one, or
        none, names another unit, so the order passes over its occurrence
        while another one is free: a special whose only order is illegal
        keeps its record and holds under its designator wherever its line
        stands on the sheet (Sec. 1). Where every free occurrence is held so,
        the order binds as before; a province where the power has one unit --
        every standard province -- is therefore untouched by this test too.
        """
        province = order.location.province
        if order.special is not None and any(
            self.designator(i) == order.special
            for i, unit in enumerate(self.units)
            if unit.power == power and unit.location.province == province
        ):
            return None  # the special this designator names already has its order
        own = [
            i
            for i, unit in enumerate(self.units)
            if unit.power == power and unit.location.province == province and i not in self.claimed
        ]
        if order.special is not None:
            # a stable sort: an occurrence another unit's illegal order holds goes last
            fallback = self.fallback
            own.sort(key=lambda i: i in fallback and fallback[i][1].special != order.special)
        for i in own:
            if _spec_matches_unit(order, self.units[i]):
                return i
        if self.ruleset.strict_designation and not self.designation_contradicts(power, order):
            return None  # every unit the spec describes already has its order
        return own[0] if own else None

    def designation_contradicts(self, power: str, order: Order) -> bool:
        """Whether the written kind or strength prefix matches no own unit in the named province.

        The rulebook's strict reading of 4.C.2 (choice C): a designation that
        contradicts every candidate makes the order invalid (4.E.1 Hold)
        rather than advisorily bound.

        Always ``False`` when the ruleset keeps the advisory default
        (``strict_designation`` off), so the standard column never sees this
        gate and stays byte-identical. Under strict binding it fires only on a
        genuine contradiction: a spec that matches some occurrence (exact, bare,
        a duplicate whose twin is already claimed, or a 3A(M)/3A(G) designator
        whose value stands there) is never contradicted -- those bind
        advisorily. Checks *all* own occupants (claimed included), so a spec
        whose units all have their orders is no contradiction:
        :meth:`claim_slot` binds it to nothing instead.
        """
        if not self.ruleset.strict_designation:
            return False
        province = order.location.province
        return not any(
            _spec_matches_unit(order, unit)
            for unit in self.units
            if unit.power == power and unit.location.province == province
        )

    def designator(self, slot: int) -> str | None:
        """Return the designator the occurrence's order writes, or None if it has no order.

        Every effective form of an order keeps the designator it writes -- the
        validated order and the Hold an invalid one degrades to alike -- so this
        is the designator of the record Phase 0 makes for the occurrence, known
        from the claims alone.
        """
        entry = self.claimed.get(slot)
        return entry[1].special if entry is not None else None


def _validate_move(
    graph: MapGraph,
    ruleset: Ruleset,
    unit: Unit,
    move: Move,
    capable_seas: frozenset[str],
) -> Move | None:
    """Check a move's legality and return its effective form, or None if invalid.

    The effective move has its destination coast resolved for fleets and its
    convoy attempt flagged (Sec. 1). Armies ignore written destination coasts.
    An unflagged army move to a non-adjacent coastal province becomes a convoy
    *attempt* iff capable fleets on the board could form a chain (see
    :func:`potential_chain`); adjacent unflagged moves stay overland, because
    the written form decides the route (DATC 4.A.3; Sec. 3.5). A fleet moving
    to a split-coast province must name a reachable coast; if no coast is
    written and exactly one is reachable, it is filled in (DATC 6.B.1-3).
    Entry permission (Sec. 4.3: Angband/Eriador) is a static gate.
    """
    move = Move(
        unit_kind=unit.kind,
        location=unit.location,
        destination=move.destination,
        via_convoy=move.via_convoy,
        strength=move.strength,
        special=move.special,
    )
    destination = move.destination.province
    if destination not in graph.provinces:
        return None
    if not ruleset.entry_permission(unit, graph.province(destination)):
        return None
    if unit.kind is UnitKind.ARMY:
        move = Move(
            move.unit_kind,
            move.location,
            Location(destination),
            via_convoy=move.via_convoy,
            strength=move.strength,
            special=move.special,
        )
        if static_move_legal(graph, move):
            return move
        origin = move.location.province
        if (
            not move.via_convoy
            and origin != destination
            and graph.terrain(origin) is Terrain.COASTAL
            and graph.terrain(destination) is Terrain.COASTAL
            and not graph.army_reachable(origin, destination)
            and potential_chain(graph, capable_seas, origin, destination)
        ):
            return Move(
                move.unit_kind,
                move.location,
                move.destination,
                via_convoy=True,
                strength=move.strength,
                special=move.special,
            )
        return None
    if move.via_convoy:
        return None  # fleets are never convoyed
    target = move.destination
    if target.coast is None and graph.coasts.get(target.province):
        options = graph.fleet_destinations(unit.location, target.province)
        if len(options) != 1:
            return None
        move = Move(
            move.unit_kind,
            move.location,
            options[0],
            via_convoy=False,
            strength=move.strength,
            special=move.special,
        )
    return move if static_move_legal(graph, move) else None


def _kind_can_reach(graph: MapGraph, unit: Unit, province: str) -> bool:
    """Whether ``unit`` could relocate into ``province`` by its own adjacency.

    Province-level: a Dislodge names an Area, a support target a province. The
    geometry is :func:`entente.engine.orders.kind_reaches_province`, shared
    with :func:`static_move_legal`'s overland army branch; this wrapper only
    unpacks a :class:`Unit`, a board type the syntax-only ``orders`` module
    must not depend on.
    """
    return kind_reaches_province(graph, unit.kind, unit.location, province)


def _validate_dislodge(
    graph: MapGraph, ruleset: Ruleset, unit: Unit, dislodge: Dislodge
) -> Dislodge | None:
    """Check a Dislodge's legality and return its effective form, or None if invalid.

    The victim's province must be a multi-occupancy Area the dislodger can
    reach and enter, and the victim must be foreign: a power can never
    dislodge its own unit (Sec. 1, Sec. 4.2). Whether the victim *reference*
    resolves to a unit is resolution semantics -- a mis-target stays valid and
    fails, delivering no cut (Sec. 3.4).
    """
    province = dislodge.victim.location.province
    if province not in graph.provinces:
        return None
    if graph.capacity(province) == 1:
        return None  # targeted displacement is an Area order (rulebook "Areas")
    if province == unit.location.province:
        return None  # the dislodger attacks from an adjacent province
    if dislodge.victim.power == unit.power:
        return None  # Sec. 4.2: no targeted self-dislodgement
    if not _kind_can_reach(graph, unit, province):
        return None
    if not ruleset.entry_permission(unit, graph.province(province)):
        return None
    return Dislodge(
        unit.kind,
        unit.location,
        dislodge.victim,
        dislodge.victim_order,
        dislodge.strength,
        dislodge.special,
    )


def _validate_bar(graph: MapGraph, unit: Unit, bar: Bar) -> Bar | None:
    """Check a Bar's legality and return its effective form, or None if invalid.

    The barring unit stands in a multi-occupancy Area, names its own Area as
    the barred destination, and the barred move is geometrically possible for
    the named unit's kind (Sec. 1, Sec. 4.2).
    """
    own_province = unit.location.province
    if graph.capacity(own_province) == 1:
        return None  # bars are Area orders (rulebook "Barring order")
    if bar.destination.province != own_province:
        return None  # the barred move targets the barring unit's own Area
    victim_province = bar.victim.location.province
    if victim_province not in graph.provinces or victim_province == own_province:
        return None
    if bar.victim.kind is UnitKind.ARMY:
        if not graph.army_reachable(victim_province, own_province):
            return None
    else:
        reachable = any(
            graph.fleet_destinations(loc, own_province)
            for loc in graph.fleet_locations(victim_province)
        )
        if not reachable:
            return None
    return Bar(
        unit.kind, unit.location, bar.victim, Location(own_province), bar.strength, bar.special
    )


def _resolve_subject_slot(
    work: _Phase0State,
    location: Location,
    spec_kind: UnitKind,
    spec_strength: int | None,
    spec_power: str | None,
    exclude_slot: int,
    prefer: Callable[[int], bool] | None = None,
    *,
    fits: Callable[[int], bool] | None = None,
) -> int | None:
    """Resolve a support or convoy subject reference to a unit occurrence.

    Resolution is by province first: written kind, strength and owner narrow
    the match when they *can* (DATC 6.D.35 marks a support naming the wrong
    kind as given; 6.B.16/17 do the same for coasts -- the advisory reading).
    ``exclude_slot`` is the referring unit's own occurrence: a unit is never
    its own subject, so in a multi-unit Area the reference binds a co-located
    unit and the referrer itself neither captures it nor makes it ambiguous
    (the rulebook's "Areas" section). ``fits`` narrows the occurrences as the
    owner does: a support's subject has to write the designator its
    reference writes, while a convoy's cargo is bound without one. Among the
    remaining candidates, non-fungible ambiguity voids the reference (the
    rulebook's 4.C.5/4.C.6); fungible duplicates resolve to an occurrence
    satisfying ``prefer`` when one exists -- a support's subject is a whole
    order, so its reference binds to a twin that actually received that
    order -- else to the canonically first occurrence.
    """
    at_province = [
        i
        for i, unit in enumerate(work.units)
        if unit.location.province == location.province and i != exclude_slot
    ]
    if spec_power is not None:
        at_province = [i for i in at_province if work.units[i].power == spec_power]
    if fits is not None:
        at_province = [i for i in at_province if fits(i)]
    exact = [
        i
        for i in at_province
        if work.units[i].kind is spec_kind and work.units[i].strength == (spec_strength or 1)
    ]
    # Under strict designation the written kind and base strength bind here too, so a
    # reference describing no occurrence resolves to nothing rather than falling
    # back to the province (the reference-level counterpart of
    # designation_contradicts). _subject_record replays this exact choice.
    candidates = exact if work.ruleset.strict_designation else (exact or at_province)
    if not candidates:
        return None
    fungible = {
        (work.units[i].power, work.units[i].kind, work.units[i].strength) for i in candidates
    }
    if len(fungible) > 1:
        return None  # ambiguous reference (4.C.5/4.C.6): invalid, becomes Hold
    if prefer is not None:
        preferred = [i for i in candidates if prefer(i)]
        if preferred:
            return preferred[0]
    return candidates[0]


def _validate_support(
    work: _Phase0State,
    slot: int,
    support: Support,
    relocations: Mapping[int, Move | Dislodge | None],
    bars: Mapping[int, Bar | None],
) -> Support | None:
    """Check a support's legality and return its effective form, or None if invalid.

    The rule (Sec. 1) has three parts. The subject unit exists and is another
    unit: the supporter's own occurrence is excluded from reference resolution,
    so in a multi-unit Area a support may name a co-located unit's order. The
    target province is reachable by the supporter, or IS the supporter's own
    Area -- any subject shape, the whole set of orders aimed at that Area. The
    unit the subject reference describes, its designator matched exactly
    (:func:`_ref_special_ok`), actually received the subject order:
    support-hold matches any stationary order; support-move matches the
    subject's effective move by destination province with any written
    destination coast agreeing (DATC 6.B.9); support-dislodge/support-bar
    match the actual variant order with a compatible victim (the
    resolved-victim agreement of Sec. 3.4 is re-checked by the tables). A
    support that matches no order is invalid and becomes a Hold, so no support
    stands that lends nothing. A support for a convoyed move is illegal when
    the fleets on the board could not carry the army without the supporter
    itself -- a fleet cannot convoy and support at the same time (6.D.31); a
    second fleet in the supporter's own sea may complete the chain.
    """
    graph = work.graph
    unit = work.units[slot]
    subject = support.subject

    def received_subject(occurrence: int) -> bool:
        """Whether one occurrence's effective order is the subject order (the Sec. 1 match).

        Doubles as the fungible tie-break: the reference binds to a twin that
        actually received the subject order.
        """
        relocation = relocations.get(occurrence)
        if isinstance(subject, Move):
            if not isinstance(relocation, Move):
                return False
            if relocation.destination.province != subject.destination.province:
                return False
            written_coast = subject.destination.coast
            return written_coast is None or written_coast == relocation.destination.coast
        if isinstance(subject, Dislodge):
            return isinstance(relocation, Dislodge) and refs_compatible(
                subject.victim, relocation.victim
            )
        if isinstance(subject, Bar):
            actual_bar = bars.get(occurrence)
            return actual_bar is not None and support_matches(support, actual_bar)
        return relocation is None  # Hold subject: any stationary order matches

    subject_slot = _resolve_subject_slot(
        work,
        subject.location,
        subject.unit_kind,
        subject.strength,
        support.subject_power,
        slot,
        received_subject,
        fits=lambda occurrence: _ref_special_ok(subject.special, work.designator(occurrence)),
    )
    if subject_slot is None or not received_subject(subject_slot):
        return None
    subject_unit = work.units[subject_slot]

    if isinstance(subject, Move):
        subject_relocation = relocations.get(subject_slot)
        assert isinstance(subject_relocation, Move)  # received_subject
        if subject_relocation.via_convoy and not potential_chain(
            graph,
            work.capable_seas(subject_unit, without=slot),
            subject_relocation.location.province,
            subject_relocation.destination.province,
        ):
            return None
    if not _supporter_reaches(graph, unit, _support_target(support)):
        return None
    return Support(
        unit.kind, unit.location, subject, support.strength, support.subject_power, support.special
    )


def _support_target(support: Support) -> str:
    """Return the province a support's written subject acts on (Sec. 1: its target)."""
    subject = support.subject
    if isinstance(subject, Move):
        return subject.destination.province
    if isinstance(subject, Dislodge):
        return subject.victim.location.province
    if isinstance(subject, Bar):
        return subject.destination.province
    return subject.location.province


def _supporter_reaches(graph: MapGraph, unit: Unit, target: str) -> bool:
    """Whether ``unit`` may support an order aimed at ``target`` (the Sec. 1 reach rule)."""
    if _kind_can_reach(graph, unit, target):
        return True
    # A unit standing in an Area backs every order aimed at that Area. It
    # never reaches its own province, so the reach rule alone would refuse
    # all of them, yet the rulebook grants the whole set: a co-located Bar
    # or hold-defence, a Move entering the Area, a Dislodge against a unit
    # inside it. Outside an Area the classic rule survives untouched, which
    # is what the capacity test says: at capacity 1 there is no Area to
    # stand in, and the supporter would be backing an attack on the
    # province it holds alone.
    own_province = unit.location.province
    return target == own_province and graph.capacity(own_province) != 1


def _support_board_legal(work: _Phase0State, slot: int, support: Support) -> bool:
    """Whether a support is possible on the board as it stands, reading no other order.

    Its supporter reaches the target, and a unit its subject reference
    describes stands there. A support that fails this is illegal and ignored,
    so a later order for the same unit is used (DATC 4.E.1); whether the
    subject received the subject order is read from the other orders in
    Pass B, and a support that fails only that keeps its unit and holds.
    """
    subject = support.subject
    if not _supporter_reaches(work.graph, work.units[slot], _support_target(support)):
        return False
    return (
        _resolve_subject_slot(
            work,
            subject.location,
            subject.unit_kind,
            subject.strength,
            support.subject_power,
            slot,
        )
        is not None
    )


def _convoy_board_legal(work: _Phase0State, slot: int, convoy: Convoy) -> bool:
    """Whether a convoy is possible on the board as it stands, reading no other order.

    Its unit is a fleet in a sea province, and the cargo reference describes an
    army that fleet can carry (Beleriand: a Double Army needs Double Fleets).
    As for a support, a convoy that fails this is ignored, while one whose
    cargo's own order does not match it is invalid in Pass B and holds.
    """
    graph = work.graph
    unit = work.units[slot]
    if unit.kind is not UnitKind.FLEET or graph.terrain(unit.location.province) is not Terrain.SEA:
        return False
    cargo = convoy.cargo
    cargo_slot = _resolve_subject_slot(
        work, cargo.location, cargo.unit_kind, cargo.strength, None, slot
    )
    if cargo_slot is None:
        return False
    cargo_unit = work.units[cargo_slot]
    return cargo_unit.kind is UnitKind.ARMY and work.ruleset.convoy_capability(unit, cargo_unit)


def _validate_convoy(
    work: _Phase0State,
    slot: int,
    convoy: Convoy,
    relocations: Mapping[int, Move | Dislodge | None],
) -> Convoy | None:
    """Check a convoy's legality and return its effective form, or None if invalid.

    A fleet in a sea province, capable of carrying the cargo (Beleriand: a
    Double Army needs Double Fleets), carrying an existing army's effective
    convoyed move -- explicitly flagged, or a non-adjacent attempt per
    :func:`_validate_move`; an adjacent unflagged move is overland even if
    fleets offer, because the written form decides the route (DATC 4.A.3).
    Sec. 1 and Sec. 3.5 state the rule.
    """
    unit = work.units[slot]
    if not _convoy_board_legal(work, slot, convoy):
        return None
    cargo = convoy.cargo
    cargo_slot = _resolve_subject_slot(
        work, cargo.location, cargo.unit_kind, cargo.strength, None, slot
    )
    assert cargo_slot is not None  # _convoy_board_legal resolved it
    cargo_move = relocations.get(cargo_slot)
    if (
        not isinstance(cargo_move, Move)
        or not cargo_move.via_convoy
        or cargo_move.destination.province != cargo.destination.province
    ):
        return None
    return Convoy(unit.kind, unit.location, cargo_move, convoy.strength, convoy.special)


def phase0(
    graph: MapGraph,
    ruleset: Ruleset,
    state: State,
    sheets: Mapping[str, Sequence[Order]],
) -> Phase0:
    """Run Phase 0: sheets become canonical order records, invalid orders Hold (Sec. 1, Sec. 6)."""
    registry = ruleset.order_registry
    work = _Phase0State(graph, ruleset, state)
    validated: dict[tuple[int, Order], Move | Dislodge | Bar | None] = {}

    def validate_static(slot: int, order: Move | Dislodge | Bar) -> Move | Dislodge | Bar | None:
        """Validate one of the three normalizing order shapes, once per (occurrence, order).

        The two stages below want different halves of one pure call. The claim
        stage needs only the verdict, and asks it of every order competing for
        an occurrence, since all but the first legal one are discarded (4.D.3);
        Pass A needs the validator's rewritten order -- a fleet's filled-in
        destination coast, an army move promoted to a convoy attempt, a Bar's
        canonical own-Area destination -- and asks it of the winner alone.
        Without memoization the winner would be validated twice, and
        ``_validate_move`` rebuilds the convoy-capable sea set from scratch on
        every call (Sec. 3.5, ``_Phase0State.capable_seas``). Memoizing on the
        frozen order value serves both readings and pays for the work once;
        two equal orders claiming one occurrence validate identically by
        construction.
        """
        key = (slot, order)
        if key not in validated:
            unit = work.units[slot]
            result: Move | Dislodge | Bar | None
            if isinstance(order, Move):
                result = _validate_move(graph, ruleset, unit, order, work.capable_seas(unit))
            elif isinstance(order, Dislodge):
                result = _validate_dislodge(graph, ruleset, unit, order)
            else:
                result = _validate_bar(graph, unit, order)
            validated[key] = result
        return validated[key]

    def claim_stage_legal(slot: int, order: Order) -> bool:
        """Whether an order is legal enough to claim its unit occurrence.

        DATC 4.E.1: illegal orders are ignored entirely, legality being decided
        on the board alone. A support or a convoy is checked here for what the
        board decides (reach, the subject or cargo standing there, the fleet at
        sea, the cargo an army it can carry); if pass B later invalidates the
        winner on the other orders, the unit holds -- a later legal order to
        the same unit does not win the claim instead.

        A strict-binding designation contradiction (4.C.2 choice C) is illegal
        here too, so a correctly-designated order to the same occurrence wins
        the claim over the sloppy one -- the rulebook's *"if an Order with
        correct designation exists, then the correct Order is valid and takes
        place"* (4.D.3). Advisory rulesets never see this branch.
        """
        if work.designation_contradicts(work.units[slot].power, order):
            return False
        if type(order) not in registry:
            return False
        if isinstance(order, (Move, Dislodge, Bar)):
            return validate_static(slot, order) is not None
        if isinstance(order, Support):
            return _support_board_legal(work, slot, order)
        if isinstance(order, Convoy):
            return _convoy_board_legal(work, slot, order)
        return True

    # Claims: bind each power's sheet, in order, to unit occurrences. Of
    # several orders claiming one occurrence the first legal one wins
    # (4.D.3/4.E.1); a fully illegal claim set still records its first claim
    # for FAILS reporting, and the record is marked illegal. That first claim
    # waits in ``work.fallback``, where claim_slot reads its designator.
    rejected: list[tuple[str, Order]] = []
    illegal: set[int] = set()
    for power in sorted(sheets):
        for position, order in enumerate(sheets[power]):
            slot = work.claim_slot(power, order)
            if slot is None:
                rejected.append((power, order))
                continue
            if claim_stage_legal(slot, order):
                work.claimed[slot] = (power, order, position)
            elif slot not in work.fallback:
                work.fallback[slot] = (power, order, position)
            else:
                rejected.append((power, order))
    for slot, claim in work.fallback.items():
        if slot in work.claimed:
            rejected.append((claim[0], claim[1]))
        else:
            work.claimed[slot] = claim
            illegal.add(slot)

    # Pass A: relocations and bars. Every occurrence's effective relocating
    # order (or None) and validated Bar must be known before supports/convoys
    # can be matched (Sec. 1: matching is static; a support-of-bar matches only a
    # *valid* bar). An order whose designation contradicts its unit is neither:
    # Pass B makes it a Hold, so a support of the move it wrote matches nothing,
    # and a support-hold matches the Hold the unit keeps.
    relocations: dict[int, Move | Dislodge | None] = {}
    bars: dict[int, Bar | None] = {}
    for slot in range(len(work.units)):
        entry = work.claimed.get(slot)
        effective_relocation: Move | Dislodge | None = None
        effective_bar: Bar | None = None
        if (
            entry is not None
            and type(entry[1]) in registry
            and not work.designation_contradicts(entry[0], entry[1])
        ):
            written_order = entry[1]
            checked = (
                validate_static(slot, written_order)
                if isinstance(written_order, (Move, Dislodge, Bar))
                else None
            )
            # An invalid order of any of the three shapes leaves both None,
            # which is what the Pass B branches read as "degrade to Hold".
            if isinstance(checked, Bar):
                effective_bar = checked
            elif checked is not None:
                effective_relocation = checked
        relocations[slot] = effective_relocation
        bars[slot] = effective_bar

    # Pass B: full validation.
    records: list[OrderRecord] = []
    for slot, unit in enumerate(work.units):
        entry = work.claimed.get(slot)
        written = entry[1] if entry is not None else None
        sheet_pos = entry[2] if entry is not None else _UNORDERED_SHEET_POS
        effective: Order = Hold(unit.kind, unit.location)
        valid = written is None or type(written) in registry
        if valid and written is not None and work.designation_contradicts(unit.power, written):
            # Strict 4.C.2 choice C: the written designation matches no own unit
            # in the province, so the order degrades to Hold (4.E.1). No-op for
            # advisory rulesets. A fallback claim that lost to a correctly
            # designated order never reaches Pass B, so this only fires when the
            # contradiction is the occurrence's effective order.
            valid = False
        if isinstance(written, Hold):
            # The designator survives normalization: it is the record's
            # fungibility signature (an unordered unit stays bare).
            effective = Hold(unit.kind, unit.location, written.strength, written.special)
        elif isinstance(written, RELOCATING_TYPES):
            relocation = relocations[slot]
            if relocation is None:
                valid = False
            else:
                effective = relocation
        elif isinstance(written, Bar) and valid:
            bar = bars[slot]
            if bar is None:
                valid = False
            else:
                effective = bar
        elif isinstance(written, Support) and valid:
            support = _validate_support(work, slot, written, relocations, bars)
            if support is None:
                valid = False
            else:
                effective = support
        elif isinstance(written, Convoy) and valid:
            convoy = _validate_convoy(work, slot, written, relocations)
            if convoy is None:
                valid = False
            else:
                effective = convoy
        if not valid:
            # An invalid order degrades to a Hold that keeps the written
            # designator: the designator is the unit's signature and not the
            # order's, so a special whose order failed validation is still told
            # apart from the other special (Sec. 4.2). Every other unit writes
            # none and degrades to a bare Hold.
            effective = Hold(
                unit.kind, unit.location, None, written.special if written is not None else None
            )
        records.append(
            OrderRecord(
                index=slot,
                power=unit.power,
                unit=unit,
                written=written,
                effective=effective,
                valid=valid,
                sheet_pos=sheet_pos,
            )
        )
    return Phase0(records=tuple(records), rejected=tuple(rejected), illegal=frozenset(illegal))
