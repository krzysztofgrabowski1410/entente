"""The Beleriand ruleset, Layer 1: the variant's movement calculus as data.

This is the Beleriand column of docs/adjudication.tex Sec. 7, instantiated from
the variant's primary rulebook, ``BeleriandInstruction.docx``.

Layer 1 covers the movement/retreat adjudication axes only: unit strengths
1/2/3 (rulebook "Strength and cutting Support"), the strength-valued one-shot
cut model (Sec. 3.2 -- the same arithmetic the standard ruleset runs at base == 1),
mountain-barrier halving -- a convoyed army paying at the seas its own convoy
can put it ashore from -- and fortress defense (Sec. 2), Areas with capacity 6 and
greedy strength-class admission (Sec. 4.3), targeted Dislodge and Bar orders
(Sec. 4.2), per-unit cut exception via the pseudo-province reduction (Sec. 3.1/Sec. 4.2),
sheet-sequence cut targeting (Sec. 4.4), Double-Fleet convoy capability (Sec. 3.5),
Angband/Eriador entry permission (Sec. 4.3), and the rulebook's own 4.A
paradox choices as the backup policy. Everything in Layer 2 -- knowledge
facts, masks, Ulmo timers, timed events, Transform/Devastate/Make-Masks,
migration, victory conditions, elven ownership, double-unit building -- is
implemented nowhere here.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from entente.engine.orders import Bar, Convoy, Dislodge, Hold, Move, Support, format_order
from entente.engine.rules import RULESETS
from entente.engine.rules.protocol import (
    AdmissionPolicy,
    BackupAction,
    CutExceptionScope,
    ResidueClass,
    StrengthRole,
)
from entente.engine.rules.strength import is_relocating

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from entente.engine.map.graph import MapGraph
    from entente.engine.map.types import Province
    from entente.engine.orders import Order
    from entente.engine.state import State, Unit

_HALF = Fraction(1, 2)
_MORGOTH = "MORGOTH"
_ANGBAND_REENTRY_STRENGTH = 3
"""Only Morgoth's strength-3 units -- the Morgoth unit itself, and Glaurung
when Layer 2 adds him -- may enter Angband from the board (rulebook
"Angband": *"Only Glaurung and Morgoth's personal Unit may enter (or return
to) Angband"*). Eriador has no such exception."""


def _crossing_target(order: Order) -> str | None:
    """Return the province an order projects strength *into* across an edge, or None.

    A move's destination, a Dislodge's victim Area, a support's subject
    target (the supported action's destination, the victim's Area, or the
    barred Area). Stationary holds project nothing across edges. This is the
    ``target`` half of the directional ``origin``->``target`` mountain-crossing
    check (``graph.is_mountain_crossing``). The ``origin`` half is the unit's
    own province -- except for a convoyed move, which never walks out of its
    origin: its crossings start at the seas its convoy can put it ashore from,
    the ``disembark`` set handed to :meth:`BeleriandRuleset.modified_base`.
    """
    if isinstance(order, Move):
        return order.destination.province
    if isinstance(order, Dislodge):
        return order.victim.location.province
    if isinstance(order, Support):
        subject = order.subject
        if isinstance(subject, Move):
            return subject.destination.province
        if isinstance(subject, Dislodge):
            return subject.victim.location.province
        if isinstance(subject, Bar):
            return subject.destination.province
        return subject.location.province  # support-hold: strength flows to the unit
    return None


@dataclass(frozen=True, slots=True)
class BeleriandRuleset:
    """Beleriand Layer 1 as calculus parameters (docs/adjudication.tex Sec. 7)."""

    @property
    def name(self) -> str:
        """The registry key of the ruleset, the map and the game hooks."""
        return "beleriand"

    @property
    def power_symmetric(self) -> bool:
        """False: the powers are asymmetric by design.

        Victim selection among twins (Sec. 4.4) reads order-sheet sequence, so relabeling
        powers permutes sheet interleavings and may change outcomes.
        """
        return False

    @property
    def order_registry(self) -> frozenset[type]:
        """Sec. 1/Sec. 7: the core registry plus targeted Dislodge and Bar."""
        # Spelled out on purpose, NOT frozenset(ADJUDICATION_ORDER_TYPES):
        # the registry is Beleriand's law -- it must not widen silently if the
        # engine grammar ever grows a seventh order type.
        return frozenset({Hold, Move, Support, Convoy, Dislodge, Bar})

    @property
    def strict_designation(self) -> bool:
        """True: the rulebook's "Paradoxes resolve" takes 4.C.2 choice C.

        A written kind or strength that contradicts every unit of the ordering
        power in the named province invalidates the order (4.E.1 Hold), never
        advisorily binding a mismatched occupant. Exact and bare designations,
        duplicate-text k-th binding, and the 3A(M)/3A(G) designators are
        unaffected -- they still match a unit, so the strict gate never fires
        on them.
        """
        return True

    @property
    def admission_policy(self) -> AdmissionPolicy:
        """Sec. 4.3: the same greedy descending-class rule as the standard map.

        Capacity-6 Areas make it visible (rulebook "Stopping a Unit from
        entering an Area").
        """
        return AdmissionPolicy.GREEDY_DESCENDING_CLASSES

    @property
    def cut_exception_scope(self) -> CutExceptionScope:
        """Sec. 3.1.3, Beleriand column: protection is per *unit*.

        The support is not cut by the specific unit its subject is directed
        against; a different unit attacking from the same Area does cut
        (Sec. 4.2 derives this from the pseudo-province reduction).
        """
        return CutExceptionScope.DIRECTED_UNIT

    def base_strength(self, unit: Unit) -> Fraction:
        """Return the unit's own base strength as an exact rational (Sec. 2).

        Base strengths are 1, 2 (Double Units) or 3 (Morgoth; Glaurung in Layer 2).
        """
        return Fraction(unit.strength)

    def modified_base(
        self,
        role: StrengthRole,
        unit: Unit,
        order: Order,
        graph: MapGraph,
        *,
        disembark: frozenset[str] | None = None,
    ) -> Fraction:
        """Return the role-modified strength: mountain halving and the fortress bonus (Sec. 2).

        Crossing or supporting through a mountain barrier halves the acting
        strength (rulebook "Mountain Barriers"). The barrier is a directional
        ``origin``->``target`` crossing (``graph.is_mountain_crossing``);
        retention is unaffected -- the engine uses the unmodified base for a
        failed mover's defense.

        A convoyed move never walks its origin edge: its toll is the
        *disembark* crossing (*"A Convoyed Army does not pay for the Province
        it left - it pays for its disembarking: if the crossing from the last
        Sea of the Convoy into the landing Province is marked, the Move is
        halved"*). The last sea belongs to the convoy, not to the coast, so
        the question goes to ``disembark`` -- the seas touching the landing
        province that the fleets ordered to convoy exactly this move link to
        the origin coast -- and the move is halved iff that set is non-empty
        and every sea in it is penalized into the landing province. A fleet
        ordered to convoy from a sea no chain runs through is not in the set,
        and neither is a sea without a convoying fleet: neither spares the
        army the toll nor imposes it. An empty set means no chain was ordered,
        so the move is never viable and its strength is never read; it keeps
        the whole base. A convoyed move arriving without the set raises,
        because the coast alone cannot say which sea the convoy ends in. The
        set is read off the written orders, so a chain that was ordered and
        is disrupted this phase still counts; the rulebook's letter differs
        from that only on a coast touching both a marked and a free sea, and
        no shipped coast does.

        A fortress adds 1 to hold strength iff the unit's order is not
        relocating (rulebook "Fortresses": a failed Move or Dislodge forfeits
        the bonus; the bonus never helps Move, Support, Dislodge or Bar).
        Bars take no modifiers (*"Barring Orders are never halved"*).
        """
        base = self.base_strength(unit)
        if role is StrengthRole.HOLD:
            if graph.province(unit.location.province).fortress and not is_relocating(order):
                return base + 1
            return base
        target = _crossing_target(order)
        if target is None:
            return base
        if isinstance(order, Move) and order.via_convoy:
            if disembark is None:
                raise ValueError(
                    f"convoyed move {format_order(order)!r} needs its ordered disembark seas: "
                    "the toll is read off the convoy's own chains, never off the landing coast"
                )
            if disembark and all(graph.is_mountain_crossing(sea, target) for sea in disembark):
                return base * _HALF
            return base
        if graph.is_mountain_crossing(unit.location.province, target):
            return base * _HALF
        return base

    def cut_aggregation(self, cutvals: Iterable[Fraction]) -> Fraction:
        """Sum the cut values: the support is diminished by the *combined* strength (Sec. 3.2).

        The rulebook's "Strength and cutting Support" section.
        """
        return sum(cutvals, Fraction(0))

    def edge_contest(
        self, move_a: Move, move_b: Move, cap_a: int | None, cap_b: int | None
    ) -> bool:
        """Whether two opposing moves are a head-to-head: the standard predicate (Sec. 4.1).

        Areas (capacity 6) are not displacement-governed, so cross-Area
        opposing moves are two independent entries, never a head-to-head
        (BATC.B.7).
        """
        return cap_a == 1 and cap_b == 1 and not move_a.via_convoy and not move_b.via_convoy

    def implicit_displacement(self, capacity: int | None) -> bool:
        """Whether entry displaces the occupant: only at capacity 1 (Sec. 4.2, Sec. 7).

        Ordinary provinces and off-map boxes behave classically; in Areas only
        a targeted Dislodge frees a slot.
        """
        return capacity == 1

    def convoy_capability(self, fleet: Unit, cargo: Unit) -> bool:
        """Whether ``fleet`` may carry ``cargo`` (Sec. 3.5, rulebook "Fleets").

        Convoying a Double Army needs a Double Fleet -- every chain fleet must
        match the army's strength. And no other power's fleet may carry
        Morgoth's units at all (*"No Morgoth's Unit may be on the Sea or
        Convoyed by other Powers"* -- a static movement rule, not an Ulmo
        timer). The sea half of that sentence is unreachable in Layer 1:
        Morgoth has no coastal home and cannot build or transform fleets, so a
        Morgoth-owned convoy stays permitted-but-unreachable here.
        """
        if cargo.power == _MORGOTH and fleet.power != cargo.power:
            return False
        return fleet.strength >= cargo.strength

    def entry_permission(self, unit: Unit, province: Province) -> bool:
        """Whether ``unit`` may enter ``province``: the one-way off-map boxes (Sec. 4.3).

        The rulebook's "Provinces off the map": Angband and Eriador admit only
        their own power's units, and entry from the board is forbidden
        outright -- *"the passage is one-way: Units may leave Angband, but no
        Unit may enter it from the board - not even a Morgoth's Unit that has
        never left it."* The sole exception is the rulebook's own: Morgoth's
        strength-3 specials (the Morgoth unit and Glaurung) may always
        (re)enter Angband; Eriador grants none. The ban is exact -- there is
        no per-unit provenance to keep, and Layer 2 adds none.
        """
        if province.restricted_to is not None and unit.power != province.restricted_to:
            return False
        if province.one_way:
            return province.restricted_to == _MORGOTH and unit.strength >= _ANGBAND_REENTRY_STRENGTH
        return True

    def backup_policy(self, residue: ResidueClass) -> BackupAction:
        """Return the backup action for a residue class (Sec. 5.2).

        The rulebook's own 4.A list ("Paradoxes resolve"): 4.A.1 choice B is
        the viability definition, 4.A.2 choice F is Szykman for convoy
        paradoxes; move cycles all succeed (standard rules apply where the
        rulebook is silent).
        """
        if residue is ResidueClass.PURE_CYCLE:
            return BackupAction.ALL_MOVES_SUCCEED
        return BackupAction.SZYKMAN

    def update_supply_ownership(
        self, graph: MapGraph, state: State, ownership: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Return the Autumn ownership update under the rulebook's Area-control rule (Sec. 9).

        "Control of an Area containing a Supply Centre": a power controls an
        occupied province iff it is the only power present or has at least two
        more units there than any other power; sole occupants of capacity-1
        provinces satisfy this trivially (the standard rule). Elven two-step
        ownership timing is Layer 2's.
        """
        updated = dict(ownership)
        provinces = {unit.location.province for unit in state.units}
        for province in sorted(provinces):
            if not graph.province(province).is_supply_center:
                continue
            counts: dict[str, int] = {}
            for unit in state.units:
                if unit.location.province == province:
                    counts[unit.power] = counts.get(unit.power, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if len(ranked) == 1 or ranked[0][1] >= ranked[1][1] + 2:
                updated[province] = ranked[0][0]
        return updated


BELERIAND = BeleriandRuleset()

RULESETS.setdefault(BELERIAND.name, BELERIAND)
