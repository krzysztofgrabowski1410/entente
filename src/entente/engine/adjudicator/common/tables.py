"""Static resolution tables: everything about a phase that does not depend on a labeling.

:func:`build_tables` precomputes once what the evaluator would otherwise
recompute per query -- who attacks what province, which supports attach to
which order, the Sec. 3.4 victim binding, the twin walk groups, the seas each
ordered convoy can disembark from, and every order's statically modified base
(Sec. 2 ``mods``, which by definition never depend on resolution). Both engines
consume the same :class:`Tables`, so the differential test compares labeling
searches rather than bookkeeping (docs/adjudication.tex Sec. 3, Sec. 4).

:func:`residue_adjacency` sits here rather than with Phase 1 because the extra
edges it adds are a *table* fact (walk-group membership). Which twin each
unnamed attack takes from the top of the attacked player's list (twins rule 6)
is not: it depends on the labeling, so it lives with the evaluator
(``Evaluator._walk_engagement``), and the Beleriand identity ledger reads it
there (``evaluator.final_walk_victims``).

This module changes only when the resolution stages need something else
precomputed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common.geometry import sea_chain_ends
from entente.engine.adjudicator.common.matching import _ref_special_ok, ref_matches_unit
from entente.engine.adjudicator.common.phase1 import dependency_graph, entry_province
from entente.engine.map.types import Location, UnitKind
from entente.engine.orders import (
    Bar,
    Convoy,
    Dislodge,
    Hold,
    Move,
    Order,
    Support,
    format_order,
)
from entente.engine.rules.protocol import StrengthRole
from entente.engine.rules.strength import (
    cut_value,
    is_relocating,
    refs_compatible,
    support_matches,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.engine.adjudicator.common.results import OrderRecord
    from entente.engine.map.graph import MapGraph
    from entente.engine.rules.protocol import Ruleset


@dataclass(frozen=True, slots=True)
class WalkGroup:
    """One twin stack under unnamed Dislodge attacks (the rulebook's twins rule 6).

    ``pool`` holds the stack's member records in the attacked player's list
    order (sheet position, then canonical index -- "from the top of the
    list"); ``attacks`` the unnamed Dislodge records engaging the stack, in
    attacker order (canonical power, sheet, index); ``power`` the attacked
    power. Victim assignment is labeling-dependent (winners take twins top
    to bottom; losers consume nothing) and lives on the Evaluator
    (``_walk_engagement``) -- the group only fixes the static participants.
    """

    pool: tuple[int, ...]
    attacks: tuple[int, ...]
    power: str


@dataclass(frozen=True, slots=True)
class Tables:
    """Static (labeling-independent) resolution tables over the records.

    ``moves`` lists every *relocating* record (Move and Dislodge); ``into``
    buckets them by contested province (Sec. 4.3). ``victim_of`` is the Sec. 3.4
    victim selection (None = mis-target: the attack fails and cuts nothing);
    ``dislodges_on`` its inverse. ``bars_on`` maps a relocating record to the
    Bar records that name it; ``barred_move_of`` is the inverse. The
    ``*_base_of`` maps carry each record's modified base strength per Sec. 2
    (role modifiers are static). ``protected_record_of``/
    ``protected_province_of`` realize Sec. 3.1 condition 3 for the two scopes.

    ``disembark_seas_of`` maps each convoyed Move to its *ordered disembark
    seas* (Sec. 2, the convoy toll): the seas touching the landing province
    that a chain of ``convoys_for`` fleets reaches from a sea touching the
    origin -- the last seas of the chains the convoy was ordered to form. It
    is read off the written orders alone: no fleet's fate enters, so a chain
    counts whether or not it survives the phase, which keeps the base it
    decides static. An empty set means no chain was ordered at all; such a
    Move is never viable, so nothing ever reads its strength.

    Twin-stack machinery (the rulebook's "Indistinguishable Units (twins)"
    section; every map is empty at capacity 1, so the standard column never
    sees any of it): ``wave_of`` maps each member of an identical-move wave
    of twins to its champion (the canonically first member, where Phase 0
    already concentrates every support -- rule 1); ``wave_members`` the
    inverse. ``stack_members_of`` maps each member of a >=2 twin class in a
    multi-capacity province to the class's sheet-ordered member tuple;
    ``wall_supports_of`` the stack-wide Support Hold wall those members defend
    with (rule 5). ``walk_groups``/``walk_group_of``/``stack_group_of`` wire
    unnamed Dislodge attacks and stack members to the walk in which such
    attacks take twins from the top of the attacked player's list (twins
    rule 6); ``cut_target_of`` assigns each such attack the stack support it
    cuts (the attacked player's own list, first listed downward, named cuts
    consumed first -- the same rule). ``converted_bars`` are
    same-power duplicate Bars against one wave, changed into Hold Orders
    (rule 3) after their supports were redirected to the standing Bar.

    ``bar_supports`` holds every support attached to a Bar record -- to the
    standing one, where a duplicate was converted -- whether or not that Bar
    is bound to an entrant. It is the one place both engines learn that a
    support matches a Bar, and they need to know it because such a support is
    not lost when its giver is dislodged: "in the same moment in which that
    Unit is Dislodged, it Supports the Bar, so both happen". It is cut as any
    other support, so a Bar's strength reads the order sheets and the convoy
    viabilities and no dislodgement at all.
    """

    moves: tuple[int, ...]
    into: Mapping[str, tuple[int, ...]]
    records_at: Mapping[str, tuple[int, ...]]
    supports_of: Mapping[int, tuple[int, ...]]
    hold_supports_of: Mapping[int, tuple[int, ...]]
    convoys_for: Mapping[int, tuple[int, ...]]
    disembark_seas_of: Mapping[int, frozenset[str]]
    counter_of: Mapping[int, int]
    cutval_of: Mapping[int, Fraction]
    move_record_of_cargo: Mapping[int, int]
    move_base_of: Mapping[int, Fraction]
    support_base_of: Mapping[int, Fraction]
    hold_base_of: Mapping[int, Fraction]
    victim_of: Mapping[int, int | None]
    dislodges_on: Mapping[int, tuple[int, ...]]
    bars_on: Mapping[int, tuple[int, ...]]
    barred_move_of: Mapping[int, int | None]
    protected_record_of: Mapping[int, int | None]
    protected_province_of: Mapping[int, str | None]
    wave_of: Mapping[int, int]
    wave_members: Mapping[int, tuple[int, ...]]
    stack_members_of: Mapping[int, tuple[int, ...]]
    wall_supports_of: Mapping[int, tuple[int, ...]]
    walk_groups: tuple[WalkGroup, ...]
    walk_group_of: Mapping[int, int]
    stack_group_of: Mapping[int, int]
    cut_target_of: Mapping[int, int | None]
    converted_bars: frozenset[int]
    bar_supports: frozenset[int]


def find_subject_record(records: Sequence[OrderRecord], location: Location) -> OrderRecord | None:
    """Return the record a subject reference resolves to by province, or None.

    Kind and coast are advisory (DATC 6.D.35, 6.B.16/6.B.17 -- mirrors Phase 0
    resolution).
    """
    for record in records:
        if record.unit.location.province == location.province:
            return record
    return None


def _subject_record(
    records: Sequence[OrderRecord], support: Support, supporter: int, strict: bool
) -> OrderRecord | None:
    """Return the record a validated support's subject binds to, or None.

    Phase 0 (:func:`_validate_support` via :func:`_resolve_subject_slot`)
    keeps a support valid only when the unit its subject reference describes
    received the subject order, and this function binds that unit's record:
    the same candidate filters -- the owner, the designator matched exactly
    (:func:`_ref_special_ok`), the kind and base strength under strict
    designation, the exclusion of the supporter's own occurrence
    (``supporter`` is the supporting record's index) -- and the same
    tie-break, a record whose effective order matches the subject winning over
    the canonically first, read off the same orders, since Pass A of Phase 0
    holds every relocation and Bar as Pass B leaves it. So a valid support
    always binds here: no support stands that matches no order. The two
    resolutions share their predicates and may never disagree.

    ``strict`` is the ruleset's :attr:`strict_designation`, and it must be the
    same value :func:`_resolve_subject_slot` saw: on that knob the two
    resolutions replay one another and may never disagree. Under it a written kind or base
    strength that describes no candidate binds NOTHING, so ``S MORGOTH A DNN`` matches no
    order when only a ``2A`` stands at DNN and Phase 0 makes it a Hold -- the
    reference-level counterpart of :meth:`designation_contradicts`, the
    rulebook's "A is the same as 1A" applied to the units an order NAMES as
    well as to its own spec. Without it the DATC province-only fallback stands.
    """
    subject = support.subject
    candidates = [
        r
        for r in records
        if r.unit.location.province == subject.location.province
        and r.index != supporter
        and _ref_special_ok(subject.special, r.effective.special, strict)
    ]
    if support.subject_power is not None:
        candidates = [r for r in candidates if r.power == support.subject_power]
    exact = [
        r
        for r in candidates
        if r.unit.kind is subject.unit_kind and r.unit.strength == (subject.strength or 1)
    ]
    pool = exact if strict else (exact or candidates)
    return next((r for r in pool if support_matches(support, r.effective)), None)


def _order_equivalent(written: Order, actual: Order) -> bool:
    """Whether a written victim-order specification describes the actual effective order.

    Sec. 3.4 "naming its action": same action type and same semantic targets;
    unit-spec fields and route flags are not compared, save that a named
    support names its subject's unit by the province it stands in -- the
    rulebook's "A(NEB) sup A(Gel)-Tha" finds no support given to an army
    moving from elsewhere.
    """
    if isinstance(written, Hold):
        return isinstance(actual, Hold)
    if isinstance(written, Move):
        return (
            isinstance(actual, Move) and written.destination.province == actual.destination.province
        )
    if isinstance(written, Dislodge):
        return isinstance(actual, Dislodge) and refs_compatible(written.victim, actual.victim)
    if isinstance(written, Bar):
        return (
            isinstance(actual, Bar)
            and refs_compatible(written.victim, actual.victim)
            and written.destination.province == actual.destination.province
        )
    if isinstance(written, Support):
        if not isinstance(actual, Support):
            return False
        if written.subject.location.province != actual.subject.location.province:
            return False
        return _order_equivalent(written.subject, actual.subject)
    if isinstance(written, Convoy):
        return (
            isinstance(actual, Convoy)
            and written.cargo.location.province == actual.cargo.location.province
            and written.cargo.destination.province == actual.cargo.destination.province
        )
    return False


def _resolve_victims(
    records: Sequence[OrderRecord],
) -> tuple[dict[int, int | None], list[WalkGroup]]:
    """Select the Sec. 3.4 victim of every Dislodge record.

    Attacks naming the victim's action bind to the first unit (victim-power
    sheet order) whose *effective* order matches -- or fail outright when no
    unit has that order (the rulebook's explicit failure clause); they do
    not claim exclusively, so several action-naming attacks may engage one
    victim (Sec. 4.2 rival dislodgers). An attack without an action takes the
    one unit its reference fits, and several such attacks on that unit are
    its rival dislodgers; only a reference matching no unit at all fails
    (None).

    Twin stacks (the rulebook's twins rule 6): a reference fixes the power,
    kind, base strength, designator and province of every unit it fits, so
    where it fits two or more they are the twins of one stack. An unnamed
    attack on a stack is NOT statically assigned -- rule 6 gives victims to
    *winners*, top of the list first, so the assignment is
    labeling-dependent. Such attacks are grouped into a :class:`WalkGroup` per
    stack and resolved by ``Evaluator._walk_engagement``. Named attacks keep
    their static binding: "a Dislodge which names an action takes the twin so
    named".
    """
    dislodges = [r for r in records if isinstance(r.effective, Dislodge) and r.valid]
    victim_of: dict[int, int | None] = {}
    groups: list[WalkGroup] = []
    by_group: dict[tuple[str, str], list[OrderRecord]] = {}
    for record in dislodges:
        order = record.effective
        assert isinstance(order, Dislodge)
        key = (order.victim.location.province, order.victim.power)
        by_group.setdefault(key, []).append(record)

    for (province, victim_power), attacks in sorted(by_group.items()):
        pool = sorted(
            (
                r
                for r in records
                if r.power == victim_power and r.unit.location.province == province
            ),
            key=lambda r: (r.sheet_pos, r.index),
        )
        attacks = sorted(attacks, key=lambda r: (r.power, r.sheet_pos, r.index))
        sequenced: list[OrderRecord] = []
        walked: dict[tuple[UnitKind, int, str | None], list[int]] = {}
        walked_pools: dict[tuple[UnitKind, int, str | None], list[int]] = {}
        for attack in attacks:
            order = attack.effective
            assert isinstance(order, Dislodge)
            if order.victim_order is None:
                sequenced.append(attack)
                continue
            bound: int | None = None
            victim_ref = order.victim
            named_pool = [
                c
                for c in pool
                if ref_matches_unit(victim_ref, c.unit)
                and _ref_special_ok(victim_ref.special, c.effective.special)
            ]
            for candidate in named_pool:
                if _order_equivalent(order.victim_order, candidate.effective):
                    bound = candidate.index
                    break
            victim_of[attack.index] = bound
        for attack in sequenced:
            order = attack.effective
            assert isinstance(order, Dislodge)
            victim_ref = order.victim
            compatible = [
                c
                for c in pool
                if ref_matches_unit(victim_ref, c.unit)
                and _ref_special_ok(victim_ref.special, c.effective.special)
            ]
            if len(compatible) >= 2:
                # Every unit the reference fits has its kind, base strength and
                # designator, so together they are one stack, in list order.
                cls = (victim_ref.kind, victim_ref.strength or 1, victim_ref.special)
                walked.setdefault(cls, []).append(attack.index)
                walked_pools[cls] = [c.index for c in compatible]
                continue
            victim_of[attack.index] = compatible[0].index if compatible else None
        for cls in sorted(walked, key=lambda c: (c[0].value, c[1], c[2] or "")):
            groups.append(
                WalkGroup(
                    pool=tuple(walked_pools[cls]),
                    attacks=tuple(walked[cls]),
                    power=victim_power,
                )
            )
    return victim_of, groups


def build_tables(graph: MapGraph, ruleset: Ruleset, records: Sequence[OrderRecord]) -> Tables:
    """Precompute every labeling-independent table of the phase from the Phase 0 records."""
    moves: list[int] = []
    into: dict[str, list[int]] = {}
    records_at: dict[str, list[int]] = {}
    supports_of: dict[int, list[int]] = {}
    hold_supports_of: dict[int, list[int]] = {}
    convoys_for: dict[int, list[int]] = {}
    disembark_seas_of: dict[int, frozenset[str]] = {}
    counter_of: dict[int, int] = {}
    move_record_of_cargo: dict[int, int] = {}
    move_base_of: dict[int, Fraction] = {}
    support_base_of: dict[int, Fraction] = {}
    hold_base_of: dict[int, Fraction] = {}
    bars_on: dict[int, list[int]] = {}
    barred_move_of: dict[int, int | None] = {}
    bar_supports: set[int] = set()
    protected_record_of: dict[int, int | None] = {}
    protected_province_of: dict[int, str | None] = {}

    victim_of, walk_groups_list = _resolve_victims(records)
    dislodges_on: dict[int, list[int]] = {}
    for index, victim in sorted(victim_of.items()):
        if victim is not None:
            dislodges_on.setdefault(victim, []).append(index)

    for record in records:
        records_at.setdefault(record.unit.location.province, []).append(record.index)
        order = record.effective
        target = entry_province(order)
        if target is not None:
            moves.append(record.index)
            into.setdefault(target, []).append(record.index)
        else:
            hold_base_of[record.index] = ruleset.modified_base(
                StrengthRole.HOLD, record.unit, order, graph
            )

    # Identical Moves of twins are a wave (twins rule 2): valid overland Move
    # records of one power, of one unit value, standing in one multi-capacity
    # province, with byte-identical effective orders, share one fate. The
    # champion is the canonically first member -- exactly the occurrence every
    # subject reference already binds (rule 1: supports concentrate). Convoyed
    # moves are excluded (no coastal Area exists; per-unit semantics kept).
    # Empty at capacity 1 by construction: no province there holds two units.
    wave_of: dict[int, int] = {}
    wave_members: dict[int, tuple[int, ...]] = {}
    wave_buckets: dict[tuple[str, str, int, str], list[int]] = {}
    for record in records:
        order = record.effective
        if not isinstance(order, Move) or not record.valid or order.via_convoy:
            continue
        origin = record.unit.location.province
        if graph.capacity(origin) == 1:
            continue
        key = (record.power, record.unit.kind.value, record.unit.strength, format_order(order))
        wave_buckets.setdefault(key, []).append(record.index)
    for members in wave_buckets.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        champion = members[0]
        wave_members[champion] = tuple(members)
        for member in members:
            wave_of[member] = champion

    # Bars bind to the entry they guard against: same province, compatible
    # unit reference, first matching ENTRANT -- plain Move or Dislodge alike
    # (a Bar guards the door against the unit it names whatever that unit's
    # entry order is; the unit spec suffices, the entry target being the
    # barring unit's own Area) -- in the victim power's sheet order
    # (Sec. 4.4). For a wave the first matching mover IS the champion, so one
    # binding gates the whole wave (twins rule 3).
    for record in records:
        order = record.effective
        if not isinstance(order, Bar) or not record.valid:
            continue
        bar_victim = order.victim
        candidates = sorted(
            (
                r
                for r in records
                if r.power == bar_victim.power
                and r.unit.location.province == bar_victim.location.province
                and is_relocating(r.effective)
                and entry_province(r.effective) == order.destination.province
                and ref_matches_unit(bar_victim, r.unit)
                and _ref_special_ok(bar_victim.special, r.effective.special)
            ),
            key=lambda r: (r.sheet_pos, r.index),
        )
        bound = candidates[0].index if candidates else None
        barred_move_of[record.index] = bound
        if bound is not None:
            bars_on.setdefault(bound, []).append(record.index)

    # Additional Barring Orders of the same Power against the same barred
    # group do nothing additional and are changed into Hold Orders -- twins
    # rule 3, which applies to EVERY shared target because a second Bar
    # against the same group never does anything and is therefore always
    # changed into a Hold: per (power, bound entrant) -- wave champion, lone
    # mover or dislodger alike -- the Bar of the strongest barring unit
    # stands (the strongest twin meets the strongest opponent, ties by sheet
    # order -- twins rule 1) and every other one is converted -- unbound,
    # resolution FAILS, its unit defending as a holding unit with the stack's
    # Support Hold wall (twins rule 5). The standing bar inherits the
    # converted bars' supports below, which is exactly what makes the
    # duplicate matter.
    converted_bars: set[int] = set()
    standing_of: dict[int, int] = {}
    bar_owners: dict[tuple[str, int], list[int]] = {}
    for bar_index, bound in barred_move_of.items():
        if bound is None:
            continue
        bar_owners.setdefault((records[bar_index].power, bound), []).append(bar_index)
    for (_, bound), bar_group in sorted(bar_owners.items()):
        if len(bar_group) < 2:
            continue
        standing = min(
            bar_group,
            key=lambda b: (
                -ruleset.base_strength(records[b].unit),
                records[b].sheet_pos,
                b,
            ),
        )
        for bar_index in bar_group:
            if bar_index == standing:
                continue
            converted_bars.add(bar_index)
            standing_of[bar_index] = standing
            barred_move_of[bar_index] = None
            bars_on[bound].remove(bar_index)

    for record in records:
        order = record.effective
        if isinstance(order, Support) and record.valid:
            support_base_of[record.index] = ruleset.modified_base(
                StrengthRole.SUPPORT, record.unit, order, graph
            )
            subject = order.subject
            target_record = _subject_record(
                records, order, record.index, ruleset.strict_designation
            )
            # Phase 0 voids a support that matches no order, so a valid one
            # always binds a record whose effective order it matches.
            assert target_record is not None
            if isinstance(subject, Dislodge):
                actual = target_record.effective
                assert isinstance(actual, Dislodge)
                supports_of.setdefault(target_record.index, []).append(record.index)
                protected_record_of[record.index] = victim_of.get(target_record.index)
                protected_province_of[record.index] = None
            elif isinstance(subject, Bar):
                # All of that Power's Supports for that Bar attach to the one
                # Bar that stands (twins rule 3) -- a support bound to a
                # converted duplicate is redirected.
                target_index = standing_of.get(target_record.index, target_record.index)
                supports_of.setdefault(target_index, []).append(record.index)
                bar_supports.add(record.index)
                protected_record_of[record.index] = barred_move_of.get(target_index)
                protected_province_of[record.index] = None
            elif isinstance(subject, Move):
                supports_of.setdefault(target_record.index, []).append(record.index)
                destination = subject.destination.province
                protected_province_of[record.index] = destination
                occupants = [r.index for r in records if r.unit.location.province == destination]
                protected_record_of[record.index] = (
                    occupants[0]
                    if len(occupants) == 1 and graph.capacity(destination) == 1
                    else None
                )
            else:
                hold_supports_of.setdefault(target_record.index, []).append(record.index)
                protected_record_of[record.index] = None
                protected_province_of[record.index] = None
        elif isinstance(order, Convoy) and record.valid:
            cargo = order.cargo
            target_record = find_subject_record(records, cargo.location)
            if target_record is not None and isinstance(target_record.effective, Move):
                convoys_for.setdefault(target_record.index, []).append(record.index)
                move_record_of_cargo[record.index] = target_record.index

    # A convoyed Move is tolled on the seas its own convoy orders reach the
    # landing coast through, so the relocating bases wait until the convoys
    # are bound. The walk offers every sea holding a fleet ordered to convoy
    # exactly this Move and asks no fleet's fate: the base has to be the same
    # under every labeling.
    for index in moves:
        record = records[index]
        order = record.effective
        disembark: frozenset[str] | None = None
        if isinstance(order, Move) and order.via_convoy:
            disembark = sea_chain_ends(
                graph,
                {records[c].unit.location.province for c in convoys_for.get(index, ())},
                graph.sea_neighbors(order.location.province),
                graph.sea_neighbors(order.destination.province),
            )
            disembark_seas_of[index] = disembark
        move_base_of[index] = ruleset.modified_base(
            StrengthRole.MOVE, record.unit, order, graph, disembark=disembark
        )

    # Twin stacks and the Support Hold wall (twins rule 5): every >=2 class
    # of same-power, same-value units in a multi-capacity province is a
    # stack, and every Support Hold bound to ANY member defends every member
    # ("the defence of every twin in the stack ... is its own Strength plus
    # ALL such Support Holds"). Members ordered by the attacked player's
    # sheet, top down -- the list unnamed attacks take twins from (twins
    # rule 6).
    stack_members_of: dict[int, tuple[int, ...]] = {}
    wall_supports_of: dict[int, tuple[int, ...]] = {}
    stack_buckets: dict[tuple[str, str, str, int, str | None], list[int]] = {}
    for record in records:
        province = record.unit.location.province
        if graph.capacity(province) == 1:
            continue
        # The order's special designator is part of the fungibility signature
        # -- a signed unit (Glaurung, the Morgoth unit) is never anyone's twin,
        # so it joins no stack and pools no wall. Standard orders never carry
        # one.
        stack_key = (
            province,
            record.power,
            record.unit.kind.value,
            record.unit.strength,
            record.effective.special,
        )
        stack_buckets.setdefault(stack_key, []).append(record.index)
    for members in stack_buckets.values():
        if len(members) < 2:
            continue
        ordered = tuple(sorted(members, key=lambda i: (records[i].sheet_pos, i)))
        wall = sorted({s for member in ordered for s in hold_supports_of.get(member, ())})
        for member in ordered:
            stack_members_of[member] = ordered
            wall_supports_of[member] = tuple(wall)

    # Rule-6 walk wiring plus the cut list "on its own list": each unnamed
    # attack on a stack cuts one of the stack's written Supports, first
    # listed downward -- after named attacks have consumed the supports they
    # name (their cut stays victim-coupled), and skipping a support the
    # per-unit exception protects from this attacker.
    walk_group_of: dict[int, int] = {}
    stack_group_of: dict[int, int] = {}
    cut_target_of: dict[int, int | None] = {}
    for position, group in enumerate(walk_groups_list):
        for attack in group.attacks:
            walk_group_of[attack] = position
        for member in group.pool:
            stack_group_of[member] = position
        support_list = [
            member
            for member in group.pool
            if isinstance(records[member].effective, Support) and records[member].valid
        ]
        consumed: set[int] = {
            member
            for member in group.pool
            if dislodges_on.get(member) and isinstance(records[member].effective, Support)
        }
        for attack in group.attacks:
            cut_support: int | None = None
            for supporter in support_list:
                if supporter in consumed:
                    continue
                if protected_record_of.get(supporter) == attack:
                    continue
                cut_support = supporter
                break
            cut_target_of[attack] = cut_support
            if cut_support is not None:
                consumed.add(cut_support)

    for index in moves:
        order = records[index].effective
        if not isinstance(order, Move):
            continue
        for rival in into.get(order.location.province, ()):
            counter = records[rival].effective
            if not isinstance(counter, Move):
                continue
            if counter.location.province == order.destination.province:
                counter_of[index] = rival
                break

    cutval_of: dict[int, Fraction] = {}
    for index in moves:
        cutval_of[index] = cut_value(  # Sec. 3.2: supports at pre-cut, pre-void modified values
            move_base_of[index],
            (support_base_of[s] for s in supports_of.get(index, ()) if s in support_base_of),
        )

    return Tables(
        moves=tuple(moves),
        into={p: tuple(v) for p, v in into.items()},
        records_at={p: tuple(v) for p, v in records_at.items()},
        supports_of={k: tuple(v) for k, v in supports_of.items()},
        hold_supports_of={k: tuple(v) for k, v in hold_supports_of.items()},
        convoys_for={k: tuple(v) for k, v in convoys_for.items()},
        disembark_seas_of=disembark_seas_of,
        counter_of=counter_of,
        cutval_of=cutval_of,
        move_record_of_cargo=move_record_of_cargo,
        move_base_of=move_base_of,
        support_base_of=support_base_of,
        hold_base_of=hold_base_of,
        victim_of=victim_of,
        dislodges_on={k: tuple(v) for k, v in dislodges_on.items()},
        bars_on={k: tuple(v) for k, v in bars_on.items()},
        barred_move_of=barred_move_of,
        protected_record_of=protected_record_of,
        protected_province_of=protected_province_of,
        wave_of=wave_of,
        wave_members=wave_members,
        stack_members_of=stack_members_of,
        wall_supports_of=wall_supports_of,
        walk_groups=tuple(walk_groups_list),
        walk_group_of=walk_group_of,
        stack_group_of=stack_group_of,
        cut_target_of=cut_target_of,
        converted_bars=frozenset(converted_bars),
        bar_supports=frozenset(bar_supports),
    )


def residue_adjacency(
    records: Sequence[OrderRecord], tables: Tables
) -> tuple[tuple[int, ...], ...]:
    """Return ``dependency_graph`` plus the twin walk-stack coupling the resolver needs.

    The staged resolver's *induced* residue decomposition needs the extra
    edges (Sec. 6; Lemma L1). A stack's top-of-the-list victim walk (twins
    rule 6; ``Evaluator._walk_engagement``/``_walk_taken``) is one function of
    the whole group: it reads ``remains``/``succeeded`` of every pool member
    and every unnamed attack together. ``dependency_graph``
    already binds a walk group into a single *full-graph* SCC -- each pool
    member reaches the shared attack through ``into[origin]`` (own-dislodge
    prevent-nulling) and the attack reaches every pool member through
    ``at_province`` -- so the enumerator, which condenses the full graph
    (``csp.py``), always resolves a walk group as one unit and never reads an
    undecided member out of turn.

    The resolver instead condenses the graph *induced on the still-undecided
    orders* (``_Resolver._residue_components``): the smaller residue is the
    point. But that induced restriction drops the coupling when the bridging
    attack has already decided -- pool members bound for different
    destinations, whose only link ran through that attack, fall into separate
    residue sub-components, and the crisp Eval of one then reads an undecided
    member of another (the Lemma L1 violation). A direct clique over each
    group's pool and attacks restores the coupling *within* the undecided
    subgraph, so the resolver groups a walk group exactly as the enumerator
    does. Empty at capacity 1 (no walk groups exist there), so the standard
    map -- and every DATC outcome -- is byte-identical to ``dependency_graph``.
    """
    edges = [set(deps) for deps in dependency_graph(records)]
    for group in tables.walk_groups:
        members = sorted(set(group.pool) | set(group.attacks))
        for member in members:
            edges[member].update(other for other in members if other != member)
    return tuple(tuple(sorted(deps)) for deps in edges)
