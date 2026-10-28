"""Beleriand Layer 2: the variant's game layer behind the unchanged ``Game`` API.

Everything the rulebook adds beyond the movement calculus lives here as
engine-level state and events, so the adjudicator stays a pure function of
the board and the sheets: transferable knowledge facts (the rulebook's
"Fleets" and "Doriath" sections), Dwarven masks ("Dwarven Masks"), Transform /
Devastate / Make-Masks / Sail-West as end-of-phase steps, devastation
passability and the four-year home conversion ("Glaurung Unit and
Devastation"), Ulmo's-wrath seeded timers ("Ulmo's wrath"), Glaurung's
arrival, Men's migration and Eriador's schedule ("Migration", "Eriador"),
elven two-step ownership and double-unit binding ("Gaining Ownership of
Supply Centres", "Double Units"), the Teleri fleet-disorder rule ("Teleri"),
the Gondolin shield ("Gondolin"), the elven and Feanorian support bans
("Elven Powers", "Feanorian Powers"), the one-way Angband / Eriador entry ban
("Provinces off the map"), and the per-power victory conditions with the
Valar countdown ("Valar's intervention", "Victory").

The adjudicator is **consumed, never modified**: movement outcomes are read
back through its public record (``phase0``, ``build_tables``, the
per-written-order resolutions under the ``assemble`` duplicate-text
discipline, and -- where unnamed Dislodges walk a twin stack -- the settled
labeling of ``resolver.settle``, whose rule-6 engagement says which twin fell)
to drive the identity ledger. Dynamic rule data reaches
adjudication only through :class:`DynamicBeleriandRuleset`, a per-game
wrapper implementing the frozen ``Ruleset`` protocol.

Layer-2 main orders (Transform, Devastate, Make-Masks, Sail-West) bind only
to units that received no adjudication order this phase: a unit already
acting cannot also perform an end-of-phase main step. Mask-transfer
directives ride alongside any main order (rulebook: *"a Unit may Move and
then pass a mask"*).

Randomness: one ``random.Random(seed)`` stream, consumed in a fixed order --
the Ulmo dice at construction (one d6 per listed coastal province, canonical
order), then one draw per Teleri disorder pick -- so a game is deterministic
given its seed, which the determinism property test checks.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from entente.engine.adjudicator.common import (
    Dislodgement,
    Resolution,
    build_tables,
    entry_province,
    final_walk_victims,
    phase0,
    ref_matches_unit,
)
from entente.engine.adjudicator.resolver import settle
from entente.engine.adjudicator.retreats import bind_retreat_orders
from entente.engine.game_hooks import (
    GAME_HOOKS,
    GameHooks,
    GameResult,
    MovementPlan,
    OathState,
    SpecialUnit,
)
from entente.engine.map.types import Location, Terrain, UnitKind
from entente.engine.orders import (
    ADJUDICATION_ORDER_TYPES,
    RELOCATING_TYPES,
    AcceptMask,
    Devastate,
    Dislodge,
    GiveMask,
    Hold,
    MakeMasks,
    Move,
    SailWest,
    Support,
    Transform,
    UnitRef,
    format_order,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.state import State, Unit
from entente.variants.beleriand.ledger import (
    GLAURUNG,
    MORGOTH_UNIT,
    SPECIAL_DESIGNATOR,
    Ledger,
    LedgerUnit,
)
from entente.variants.beleriand.winter import (
    adjudicate_winter,
    center_weight,
    gross_quota,
    legal_winter_orders,
    usable_fleet_knowledge,
    winter_decisions,
    winter_effects,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from fractions import Fraction

    from entente.engine.adjudicator.builds import AdjustmentResult
    from entente.engine.adjudicator.common import (
        OrderRecord,
        OrderResolution,
        PhaseResult,
        Tables,
    )
    from entente.engine.adjudicator.retreats import RetreatResult
    from entente.engine.map.graph import MapGraph
    from entente.engine.map.types import Province
    from entente.engine.orders import AnyOrder, GameOrder, Order
    from entente.engine.rules.protocol import (
        AdmissionPolicy,
        BackupAction,
        CutExceptionScope,
        ResidueClass,
        Ruleset,
        StrengthRole,
    )

MORGOTH = "MORGOTH"
MEN = "MEN"
TELERI = "TELERI"
NAUGRIM = "NAUGRIM"
DORIATH = "DORIATH"
FEANORIANS = "FEANORIANS"
GONDOLIN_POWER = "GONDOLIN"
ELVISH = frozenset(
    {DORIATH, FEANORIANS, "FINGOLFIN", "FINROD", GONDOLIN_POWER, "OSSIRIAND", TELERI}
)

ANGBAND = "ANG"
ERIADOR = "ERI"
ANFAUGLITH = "ANF"
GONDOLIN = "GON"
GIRDLE = frozenset({"FOR", "FON"})
DWARF_CITIES = frozenset({"BEL", "DOL", "NOG"})
WEST_SEAS = frozenset({"NBE", "SBE"})

ULMO_PROVINCES: tuple[str, ...] = (
    "ARV",
    "BNI",
    "BRI",
    "CBB",
    "EGL",
    "ELO",
    "LAM",
    "MSI",
    "NEV",
    "NFA",
    "NLA",
    "SFA",
    "SWB",
    "VIN",
    "WTD",
)
"""The rulebook's "Ulmo's wrath" coastal list, transcribed verbatim (fifteen
provinces): *"Northern Lammoth, Lammoth, Ered Lomin, Nevrast, Vinyamar,
Northern Falas, Brithombar, Barad Nimras, Eglarest, Southern Falas, South
West Beleriand, Arvenien, Mouths of Sirion, Western Taur-im-Duinath, Coasts
of the Bay of Balar"*. The map's one other coastal province, the Isle of
Balar, is not in the list and so stays outside the wrath."""

_DIE_TO_DELAY = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
"""d6 -> delay in Diplomacy phases (rulebook "Ulmo's wrath": 1-2 -> the
following phase, 3-4 -> after a whole year, 5-6 -> after three phases)."""

FLEET_FACT = "fleet-building"
GIRDLE_FACT = "girdle-permission"

BAN_ELVES_SUPPORT_MORGOTH = "No elven power can support Morgoth."
BAN_MORGOTH_SUPPORTS_ELVES = "Morgoth cannot support an elven power."
BAN_FEANORIAN_MUTUAL = "No other elven power may support or be supported by the Feanorians."
BAN_ANFAUGLITH = "A power with a unit in Anfauglith may not be supported by the Feanorians."
"""The four support bans, said in the rulebook's own words for the player who
watched a support die. Each mirrors the governing sentence, cut to the
direction that fired: *"None of the elven power can support or be supported
by Morgoth."* splits into the first two, *"No other elven Powers may Support
or be Supported by Feanorian Powers."* is the third whichever side wrote the
support, and *"Any Power with Unit in Anfauglith may not be Supported by
Feanorian Powers."* is the fourth. They are
:meth:`BeleriandHooks._support_forbidden`'s return values -- shown to the
player, never read by the calculus."""

FINAL_YEAR = 0
"""The countdown ends after Autumn of Year 0 (rulebook "Valar's
intervention"); the year is absolute, whatever year a game starts in."""

VICTORY_WEIGHT = 40
"""Morgoth's base victory threshold in weighted supply centers (rulebook
"Victory": *"Victory for Morgoth is when he Owns at least forty Supply
Centres"*) -- 40 of the board's 53, from a home endowment of 8. Every Oath he
has broken adds one for the rest of the game (the "Morgoth" section's
"Victory"), so the live threshold is :attr:`BeleriandHooks.victory_bar`, not
this constant."""

_YEAR_BASE = 25
"""Ordinal reference for the Ulmo timers: the rulebook calendar's first
year. Ordinals are relative, so any fixed base works for any start year."""


def _ordinal(year: int, season: Season) -> int:
    """Return the movement-phase ordinal of a season (retreats share their season's)."""
    return 2 * (_YEAR_BASE - year) + (0 if season is Season.SPRING else 1)


def _needs_glaurung(entry: LedgerUnit) -> bool:
    return entry.special == GLAURUNG


def _needs_mask(entry: LedgerUnit) -> bool:
    return entry.masks > 0


_DESIGNATOR_TAG = {"M": MORGOTH_UNIT, "G": GLAURUNG}
"""Order-notation designator letter -> ledger identity tag: ``3A(M)`` names
the Morgoth unit, ``3A(G)`` Glaurung. The parser admits only these two
letters, so a lookup here never misses for a validated designator."""


def _needs_special(tag: str) -> Callable[[LedgerUnit], bool]:
    """Return the binding demand that narrows to the identity tagged ``tag``.

    A written designator adds this ``special == tag`` predicate so the named
    special, not a fungible twin, binds the order's record.
    """

    def need(entry: LedgerUnit) -> bool:
        return entry.special == tag

    return need


def _and_(
    first: Callable[[LedgerUnit], bool], second: Callable[[LedgerUnit], bool]
) -> Callable[[LedgerUnit], bool]:
    """Return the conjunction of two binding demands."""

    def need(entry: LedgerUnit) -> bool:
        return first(entry) and second(entry)

    return need


_DesignatorCheck = tuple[str | None, UnitKind, Location, int | None, str | None, bool]
"""One written spec or reference for the designator law: owner, kind, location,
strength, designator, and whether it names a unit dislodged from there."""


def _reference_checks(order: object, checks: list[_DesignatorCheck]) -> None:
    """Append a check for every unit reference written inside ``order`` to ``checks``.

    The references are a Dislodge's or a Bar's victim, a mask transfer's
    receiver or giver, a support's subject and the references inside it, a
    convoy's cargo, and the references inside a Dislodge's victim action. A
    victim action's own spec is its victim reference by construction, so only
    what it names in turn is added. Every reference names a board unit.
    """
    for ref in (
        getattr(order, "victim", None),
        getattr(order, "receiver", None),
        getattr(order, "giver", None),
    ):
        if ref is not None:
            checks.append((ref.power, ref.kind, ref.location, ref.strength, ref.special, False))
    subject = getattr(order, "subject", None)
    if subject is not None:
        owner = getattr(order, "subject_power", None)
        checks.append(
            (owner, subject.unit_kind, subject.location, subject.strength, subject.special, False)
        )
        _reference_checks(subject, checks)
    cargo = getattr(order, "cargo", None)
    if cargo is not None:  # a Move, which names no other unit
        checks.append((None, cargo.unit_kind, cargo.location, cargo.strength, cargo.special, False))
    action = getattr(order, "victim_order", None)
    if action is not None:
        _reference_checks(action, checks)


# -- the dynamic ruleset wrapper -------------------------------------------------------


@dataclass(frozen=True)
class DynamicBeleriandRuleset:
    """A per-game delegate implementing the frozen ``Ruleset`` protocol.

    Only two hooks read game state: ``entry_permission`` (Girdle,
    devastation, the Gondolin shield, the one-way Angband/Eriador ban) and
    ``update_supply_ownership`` (elven two-step, devastated capture). The
    calculus axes delegate to the static Layer-1 ruleset untouched.
    """

    base: Ruleset
    hooks: BeleriandHooks

    @property
    def name(self) -> str:
        """The static ruleset's name; the hooks register under the same key."""
        return self.base.name

    @property
    def power_symmetric(self) -> bool:
        """Whether the static ruleset is power-symmetric (Beleriand is not)."""
        return self.base.power_symmetric

    @property
    def order_registry(self) -> frozenset[type]:
        """The static ruleset's order registry; Layer-2 orders never enter it."""
        return self.base.order_registry

    @property
    def strict_designation(self) -> bool:
        """The static ruleset's designation strictness (Beleriand is strict)."""
        return self.base.strict_designation

    @property
    def admission_policy(self) -> AdmissionPolicy:
        """The static ruleset's Area admission policy."""
        return self.base.admission_policy

    @property
    def cut_exception_scope(self) -> CutExceptionScope:
        """The static ruleset's support-cut exception scope."""
        return self.base.cut_exception_scope

    def base_strength(self, unit: Unit) -> Fraction:
        """Delegate the unit's base strength to the static ruleset."""
        return self.base.base_strength(unit)

    def modified_base(
        self,
        role: StrengthRole,
        unit: Unit,
        order: Order,
        graph: MapGraph,
        *,
        disembark: frozenset[str] | None = None,
    ) -> Fraction:
        """Delegate the role-modified strength, disembark seas included, to the static ruleset."""
        return self.base.modified_base(role, unit, order, graph, disembark=disembark)

    def cut_aggregation(self, cutvals: object) -> Fraction:
        """Delegate the cut aggregation to the static ruleset."""
        return self.base.cut_aggregation(cutvals)  # type: ignore[arg-type]

    def edge_contest(
        self, move_a: Move, move_b: Move, cap_a: int | None, cap_b: int | None
    ) -> bool:
        """Delegate the head-to-head predicate to the static ruleset."""
        return self.base.edge_contest(move_a, move_b, cap_a, cap_b)

    def implicit_displacement(self, capacity: int | None) -> bool:
        """Delegate the implicit-displacement predicate to the static ruleset."""
        return self.base.implicit_displacement(capacity)

    def convoy_capability(self, fleet: Unit, cargo: Unit) -> bool:
        """Delegate the convoy-capability predicate to the static ruleset."""
        return self.base.convoy_capability(fleet, cargo)

    def backup_policy(self, residue: ResidueClass) -> BackupAction:
        """Delegate the backup policy to the static ruleset."""
        return self.base.backup_policy(residue)

    def entry_permission(self, unit: Unit, province: Province) -> bool:
        """Whether ``unit`` may enter ``province`` under the static and the dynamic gates.

        The dynamic gates read game state: devastated ground admits Morgoth's
        units and mask carriers only, the Girdle admits the powers holding its
        permission, and Gondolin admits nobody but its own power while the
        shield stands. The test is value-level: a credential held by *any*
        co-located twin answers yes, and the claim discipline of
        :meth:`BeleriandHooks._claim` (of :meth:`BeleriandHooks._claim_retreat`
        in the retreats) keeps that exact by spending one credentialed
        identity per gated order.

        One-way entry is a flat ban with a single exception: only a Morgoth
        strength-3 special may enter Angband, and nothing enters Eriador from
        the board. The rule keeps no per-unit provenance, so this matches the
        static Layer-1 gate exactly.
        """
        hooks = self.hooks
        key = province.key
        if key in hooks.devastated and unit.power != MORGOTH and not hooks.mask_available(unit):
            return False
        if key in GIRDLE and unit.power not in hooks.girdle:
            return False
        if key == GONDOLIN and hooks.gondolin_shield and unit.power != GONDOLIN_POWER:
            return False
        if province.restricted_to is not None and unit.power != province.restricted_to:
            return False
        if province.one_way:
            return key == ANGBAND and unit.power == MORGOTH and unit.strength >= 3
        return True

    def update_supply_ownership(
        self, graph: MapGraph, state: State, ownership: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Apply the Autumn ownership update with the variant's gates.

        Layer-1 Area control decides who dominates a centre; the Layer-2 gates
        add the elven two-step timing and the devastated-centre rule. Side
        effects (the capture stash, the supply delays) are the hooks' boundary
        bookkeeping: this method is the variant's own contract point, called
        exactly once per Autumn.
        """
        return self.hooks.autumn_ownership(graph, state, ownership)


# -- the hooks -------------------------------------------------------------------------


class BeleriandHooks(GameHooks):
    """One game's Layer-2 state machine, registered under ``"beleriand"``.

    ``graph`` is the variant map and ``seed`` feeds the game's one random
    stream (``None`` reads as 0). The state it owns, all of it public by rule
    and written into the snapshot: the Ulmo dice (``ulmo_delays``) and the
    armed timers (``ulmo_armed``, movement ordinals); the identity
    ``ledger``; the two knowledge pools (``fleet_knowledge``, ``girdle``);
    the devastation record (``devastated``, ``devastation_years``,
    ``converted_homes``); Men's off-board counter (``eriador_centers``); the
    ownership calendar (``autumn_count``, ``delayed_supply``, ``owner_since``,
    ``entered_this_year``, ``double_eligible``, ``spent_supply``,
    ``spring_control``); the Gondolin shield and the double it guards; the
    victory flags (``morgoth_defeated``, ``sail_west_pending``,
    ``sail_west_ever``, ``glaurung_arrived``, ``result``); and the Oath of the
    Second (``oath``, ``oath_year``, ``oaths_broken``, ``oath_stabbed``).
    The underscored fields are per-phase scratch: the wrapped ruleset, the
    event buffer, the pending build bindings, the Teleri disorder pick, the
    Autumn capture stash, the neighbour cache, the winter's ownership and the
    designated Holds injected for the specials that wrote no valid order.
    """

    def __init__(self, graph: MapGraph, seed: int | None = None) -> None:
        super().__init__(graph, 0 if seed is None else seed)
        self.rng = random.Random(self.seed)
        self.ulmo_delays: dict[str, int] = {
            province: _DIE_TO_DELAY[self.rng.randint(1, 6)] for province in sorted(ULMO_PROVINCES)
        }
        self.ulmo_armed: dict[str, int] = {}
        self.ledger = Ledger()
        self.fleet_knowledge: set[str] = {TELERI}
        self.girdle: set[str] = {DORIATH}
        self.devastated: set[str] = set()
        self.devastation_years: dict[str, int] = {}
        self.converted_homes: set[str] = set()
        self.eriador_centers = 0
        self.autumn_count = 0
        self.delayed_supply: dict[str, int] = {}
        self.owner_since: dict[str, int] = {}
        self.entered_this_year: dict[str, set[str]] = {}
        self.double_eligible: dict[str, int] = {}
        self.spent_supply: dict[str, int] = {}
        self.spring_control: dict[str, tuple[str, int | None]] = {}
        self.gondolin_shield = False
        self.gondolin_double_uid: int | None = None
        self.morgoth_defeated = False
        self.sail_west_pending = False
        self.sail_west_ever = False
        self.glaurung_arrived = False
        self.result: GameResult | None = None
        # The Oath of the Second: the standing Oath and the Year it was sworn,
        # the permanent price of every Oath broken so far, and the current
        # Oath's free-break exemption (per-Oath, cleared by the next swearing).
        # All four are public by rule and ride the snapshot.
        self.oath: str | None = None
        self.oath_year: int | None = None
        self.oaths_broken = 0
        self.oath_stabbed = False
        self._wrapped: DynamicBeleriandRuleset | None = None
        self._events: list[str] = []
        self._pending_bindings: list[tuple[Unit, str | None]] = []
        self._teleri_locked: tuple[tuple[str, str | None, str, int], int] | None = None
        self._captures: dict[str, tuple[str, str]] = {}
        self._neighbor_cache: dict[str, set[str]] = {}
        self._winter_ownership: dict[str, str] = {}
        self._injected: dict[tuple[str, int], str] = {}

    # -- construction -----------------------------------------------------------

    def initial_phase(self, start_year: int | None) -> Phase:
        """Return Spring of Year 25 (or an explicit test year), counting down."""
        year = start_year if start_year is not None else _YEAR_BASE
        return Phase(year, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)

    def wrap_ruleset(self, base: Ruleset) -> Ruleset:
        """Wrap the static ruleset so its two state-reading hooks consult this game."""
        self._wrapped = DynamicBeleriandRuleset(base, self)
        return self._wrapped

    def register_start(self, state: State, ownership: Mapping[str, str]) -> None:
        """Seed the ledger, arm the Gondolin shield and charge the opening supply."""
        self.ledger.seed(state)
        for entry in self.ledger.alive():
            if (
                entry.power == GONDOLIN_POWER
                and entry.strength == 2
                and entry.location.province == GONDOLIN
            ):
                self.gondolin_double_uid = entry.uid
                self.gondolin_shield = True
                break
        for center in ownership:
            self.owner_since.setdefault(center, 0)
        # A starting unit is supplied by the centre it starts on (the rulebook:
        # "the Units a Player starts the game with are supplied by their Home
        # Supply Centres in the same way").
        for entry in self.ledger.alive():
            if entry.bound_to is None and entry.special is None:
                entry.supplied_by = entry.location.province
        self._reconcile_spent(state, ownership)

    # -- small queries ------------------------------------------------------------

    def _ids_matching(self, unit: Unit, *, limbo: bool | None = None) -> list[LedgerUnit]:
        """Return the alive identities of ``unit``'s fungibility class.

        ``limbo`` narrows to board entries (False) or dislodged ones (True) at
        the unit's recorded position; None takes both.
        """
        found: list[LedgerUnit] = []
        for entry in self.ledger.alive():
            if limbo is not None and entry.limbo is not limbo:
                continue
            if (
                entry.power == unit.power
                and entry.kind is unit.kind
                and entry.strength == unit.strength
                and entry.location == unit.location
            ):
                found.append(entry)
        return found

    def mask_available(self, unit: Unit) -> bool:
        """Whether any identity of ``unit``'s class carries a mask."""
        return any(entry.masks > 0 for entry in self._ids_matching(unit))

    def special_designators(self, unit: Unit, *, limbo: bool = False) -> tuple[str, ...]:
        """Return the designators of the live specials of ``unit``'s value, in ledger order.

        The Morgoth unit precedes Glaurung, who is created later, so an
        enumeration that hands the k-th occurrence of a value the k-th letter
        writes both of them wherever they stand together.
        """
        return tuple(
            SPECIAL_DESIGNATOR[entry.special]
            for entry in self.ledger.alive()
            if entry.special is not None
            and entry.limbo is limbo
            and entry.power == unit.power
            and entry.kind is unit.kind
            and entry.strength == unit.strength
            and entry.location == unit.location
        )

    def special_units(self) -> tuple[SpecialUnit, ...]:
        """Return the living specials, in ledger order, each with its designator."""
        return tuple(
            SpecialUnit(
                entry.special,
                SPECIAL_DESIGNATOR[entry.special],
                Unit(entry.power, entry.kind, entry.location, entry.strength),
                limbo=entry.limbo,
            )
            for entry in self.ledger.alive()
            if entry.special is not None
        )

    # -- special-unit designators ---------------------------------------------------

    def _special_at(
        self,
        power: str,
        designator: str,
        kind: UnitKind,
        location: Location,
        strength: int | None,
        *,
        limbo: bool,
    ) -> bool:
        """Whether the ``(M)``/``(G)`` designator names a live special at the written spec.

        The special must stand on the board (``limbo=False``) or in limbo
        (``limbo=True``) at the spec's province. The designator names one of
        MORGOTH's two ``3A`` specials only; a wrong power, or an absent or
        mis-placed special, is a false designation and the order becomes
        invalid. An unwritten strength reads as base strength 1, as in Phase-0 spec
        matching.
        """
        if power != MORGOTH:
            return False
        tag = _DESIGNATOR_TAG.get(designator)
        if tag is None:  # pragma: no cover - the grammar admits only M/G
            return False
        return any(
            entry.special == tag
            and entry.limbo is limbo
            and entry.kind is kind
            and entry.location.province == location.province
            # bare means base strength 1 here too; the grammar's designated refs
            # (`3A(M)`, `3A(G)`) always write one, so this is only a guard
            and (strength or 1) == entry.strength
            for entry in self.ledger.alive()
        )

    def _special_described(
        self,
        power: str | None,
        kind: UnitKind,
        location: Location,
        strength: int | None,
        *,
        limbo: bool,
    ) -> bool:
        """Whether a spec or reference written without a designator describes a live special.

        It does when a special of the ledger stands at it -- on the board, or
        with ``limbo`` dislodged from that position -- with the written kind,
        province and strength (an unwritten strength reads as base strength 1) and,
        where an owner is written, that owner. A support's subject and a
        convoy's cargo write no owner and describe the special all the same:
        only MORGOTH has one.
        """
        return any(
            entry.special is not None
            and entry.limbo is limbo
            and (power is None or entry.power == power)
            and entry.kind is kind
            and entry.location.province == location.province
            and (strength or 1) == entry.strength
            for entry in self.ledger.alive()
        )

    def _spec_obeys_designator_law(
        self,
        power: str | None,
        kind: UnitKind,
        location: Location,
        strength: int | None,
        special: str | None,
        *,
        limbo: bool,
    ) -> bool:
        """Whether one written spec or reference names a special iff it writes a designator.

        A written designator must name a live special standing at the spec; a
        spec without one must describe no special. A subject or cargo spec
        that writes a designator but no owner names MORGOTH's unit, the only
        power that has one.
        """
        if special is not None:
            owner = MORGOTH if power is None else power
            return self._special_at(owner, special, kind, location, strength, limbo=limbo)
        return not self._special_described(power, kind, location, strength, limbo=limbo)

    def _designations_valid(self, power: str, order: GameOrder, *, limbo: bool = False) -> bool:
        """Whether every unit spec and unit reference of ``order`` obeys the designator law.

        The rulebook's Signature rule: a special unit -- the Morgoth unit or
        Glaurung -- is named only with its designator, ``3A(M)`` or ``3A(G)``,
        and a ``3A`` written without one fits no unit; a designator in turn
        names only the live special standing at the written spec. The law
        holds at every position of an order: the ordering unit's own spec, a
        Dislodge's or a Bar's victim, a support's subject and -- when that
        subject is a Dislodge or a Bar -- the subject's victim, a convoy's
        cargo, a mask transfer's receiver or giver, and every reference
        written inside a Dislodge's victim action, the subject's included. A
        victim action's own spec is the victim reference's, designator and
        all, by construction, so it is not checked twice. A
        spec or reference describes a special when a live special of the
        ledger stands at it (owner, kind, province and strength, an unwritten
        strength reading as base strength 1); written bare it is invalid, and so is a
        designator naming no special standing there or a special of another
        power. One fault voids the whole order: it is dropped and reported
        failing, and the unit takes its next order or holds, as for an
        illegal order (docs/adjudication.tex Sec. 1).

        The own spec is checked against the board, or with ``limbo`` against
        the units dislodged from it, which is what a retreat order names;
        every reference names a board unit. The adjudicator sees the
        designators the sheet writes and reads them as the units' signatures
        (Sec. 4.2); which occurrence is which special stays the ledger's
        business.
        """
        checks: list[_DesignatorCheck] = []
        kind = getattr(order, "unit_kind", None)
        location = getattr(order, "location", None)
        if kind is not None and location is not None:  # every order but Waive names a unit
            strength = getattr(order, "strength", None)
            checks.append((power, kind, location, strength, getattr(order, "special", None), limbo))
        _reference_checks(order, checks)
        return all(
            self._spec_obeys_designator_law(
                owner, kind, location, strength, special, limbo=at_limbo
            )
            for owner, kind, location, strength, special, at_limbo in checks
        )

    def _absolute_neighbors(self, province: str) -> set[str]:
        """Return every province one map edge away from ``province`` for either kind.

        Army adjacency union province-level fleet reach, memoized on the
        immutable graph; the returned set is shared and must be treated as
        read-only (all callers only test membership or build a fresh set).
        """
        cached = self._neighbor_cache.get(province)
        if cached is not None:
            return cached
        neighbors = set(self.graph.army_adjacency.get(province, frozenset()))
        for location in self.graph.fleet_locations(province):
            for reachable in self.graph.fleet_adjacency.get(location, frozenset()):
                neighbors.add(reachable.province)
        self._neighbor_cache[province] = neighbors
        return neighbors

    def _emit(self, line: str) -> None:
        self._events.append(line)

    def drain_events(self) -> tuple[str, ...]:
        """Return the buffered event lines and clear the buffer."""
        events = tuple(self._events)
        self._events = []
        return events

    # -- knowledge grants -----------------------------------------------------------

    def _holds_fact(self, power: str, fact: str) -> bool:
        pool = self.fleet_knowledge if fact == FLEET_FACT else self.girdle
        return power in pool

    def _voluntary_source_ok(self, receiver: str, source: str) -> bool:
        """Whether ``source`` may voluntarily hand a fact to ``receiver``.

        The Feanorian source rule, scoped to the two voluntary channels: the
        written grant and mask acceptance. The rulebook's "Doriath" and
        "Feanorian Powers" sections: *"No elven Power may give a written
        permission to enter the Girdle to Feanorian Powers."* and *"They must
        obtain this knowledge from a non-elvish Power."* -- a rule about who
        may hand the knowledge over. The non-elvish powers are Morgoth, Men
        and the Naugrim, and ``{MEN, NAUGRIM}`` is that set's reachable
        remainder on the two channels this guards, not a narrowing of it:
        Morgoth never holds the fleet fact to give, never gives the Girdle
        fact to an elvish power (:meth:`validate_grant`'s own clause, checked
        first), and can neither give nor accept masks, so he cannot stand on
        the source side here at all. The two conditions coincide, and the
        enumerated one is written for the reader who has to check it.

        Being attacked is neither a permission given nor knowledge obtained
        from anyone, so ``_grant_girdle``'s attack channel does not consult
        this -- which is also why no channel is left where "non-elvish" and
        ``{MEN, NAUGRIM}`` could part.
        """
        return receiver != FEANORIANS or source in (MEN, NAUGRIM)

    def validate_grant(self, giver: str, receiver: str, fact: str) -> str | None:
        """Return why ``giver`` may not hand ``fact`` to ``receiver`` in writing, or None.

        The rulebook's "Doriath" section: *"Morgoth can only learn how to
        pass through Melian's Girdle via being attacked from its inside or
        from a non-elvish Power (via written statement), and cannot pass the
        knowledge to any elven powers (via written statement)."* Morgoth
        therefore receives the Girdle fact by written grant only from a
        non-elvish giver. The non-elvish powers are Morgoth, Men and the
        Naugrim; Morgoth is the receiver in this clause, so he is out as his
        own source (the self-grant refusal above states it), leaving ``{MEN,
        NAUGRIM}`` as the rule's reachable content rather than a narrowing of
        it -- the same reading the Feanorian source rule takes
        (:meth:`_voluntary_source_ok`). The other channels are their own: the
        attack-grant is unconditional (see ``_grant_girdle``), the mask
        channel is shut for Morgoth, and he never grants either fact to an
        elvish power by written statement. "Feanorian Powers" and "Dwarves
        (Naugrim)": the Feanorians obtain either fact only from Men or the
        Naugrim, which is one of the two channels that rule reaches
        (:meth:`_voluntary_source_ok`).
        """
        if fact not in (FLEET_FACT, GIRDLE_FACT):
            return f"unknown fact {fact!r}"
        if giver == receiver:
            return "a power cannot grant to itself"
        if not self._holds_fact(giver, fact):
            return f"{giver} does not hold {fact}"
        if self._holds_fact(receiver, fact):
            return f"{receiver} already holds {fact}"
        if fact == FLEET_FACT and receiver == MORGOTH:
            return "Morgoth can never gain the fleet-building knowledge"
        if fact == GIRDLE_FACT and receiver == MORGOTH and giver not in (MEN, NAUGRIM):
            return (
                "Morgoth learns the Girdle passage from a non-elvish power "
                "(Men or the Naugrim) or by being attacked from inside it"
            )
        if giver == MORGOTH and receiver in ELVISH:
            return "Morgoth passes no knowledge to elvish powers"
        if not self._voluntary_source_ok(receiver, giver):
            return "the Feanorians obtain knowledge only from Men or the Naugrim"
        return None

    def apply_grant(self, giver: str, receiver: str, fact: str) -> None:
        """Record a written grant of ``fact`` to ``receiver`` and log it."""
        pool = self.fleet_knowledge if fact == FLEET_FACT else self.girdle
        pool.add(receiver)
        self._emit(f"{giver} grants {fact} to {receiver}")

    def _grant_girdle(self, power: str, how: str) -> None:
        """Grant ``power`` the Girdle permission automatically, by attack or by mask.

        Unconditional in both receiver and source: every restriction lives in
        the channel that owns it. The attack channel has none -- *"the owner
        of the Attacked Unit gains ability to pass the Girdle of Melian"*
        names neither party, so a Feanorian victim is compensated like any
        other and Morgoth may stand at either end of it. The mask channel
        screens its source through :meth:`_voluntary_source_ok` before
        calling here, as :meth:`validate_grant` does for written permissions.
        Every gain emits an event.
        """
        if power in self.girdle:
            return
        self.girdle.add(power)
        self._emit(f"{power} gains the Girdle of Melian permission ({how})")

    # -- phase start ------------------------------------------------------------------

    def on_phase_start(self, phase: Phase, state: State) -> State:
        """Run the phase-start events: Ulmo's wrath, the shield check, the Teleri pick."""
        self._teleri_locked = None
        state = self._ulmo_strike(phase, state)
        self._refresh_gondolin_shield(state)
        self._pick_teleri_disorder(state)
        return state

    def _ulmo_strike(self, phase: Phase, state: State) -> State:
        """Annihilate the Morgoth units whose coastal timers fire this phase.

        The rulebook's "Ulmo's wrath" section: the wrath falls on the
        province, before the orders are placed; a unit that moved out before
        it falls escapes it, and the timer is spent either way.
        """
        now = _ordinal(phase.year, phase.season)
        units = list(state.units)
        for province in sorted(self.ulmo_armed):
            if self.ulmo_armed[province] > now:
                continue
            del self.ulmo_armed[province]
            victim = next(
                (u for u in units if u.location.province == province and u.power == MORGOTH),
                None,
            )
            if victim is None:
                continue  # the unit left before the wrath fell
            units.remove(victim)
            entries = self._ids_matching(victim, limbo=False)
            entry = entries[0] if entries else None
            # the identity's spec, so a special is named with its designator
            named = entry.spec() if entry is not None else str(victim)
            self._emit(f"Ulmo's wrath annihilates MORGOTH {named} in {province}")
            if entry is not None:
                self._release_older_centre(entry)
                self.ledger.kill(entry.uid)
                if entry.special == MORGOTH_UNIT:
                    self._mark_morgoth_defeated("the Morgoth unit is annihilated by Ulmo")
        if len(units) != len(state.units):
            return State.of(state.map_name, state.phase, units)
        return state

    def _mark_morgoth_defeated(self, why: str, *, civil_disorder: bool = True) -> None:
        """Record Morgoth's defeat once and emit it; later calls change nothing."""
        if not self.morgoth_defeated:
            self.morgoth_defeated = True
            # The "his Units stand in Civil Disorder" consequence (rulebook
            # "Morgoth Unit") is a mid-game effect on the phases that follow;
            # a defeat declared AS the Valar end the game (``_intervene``) has
            # no such phases, so callers there pass ``civil_disorder=False`` to
            # avoid emitting a misleading post-game consequence.
            consequence = "; his units stand in civil disorder" if civil_disorder else ""
            self._emit(f"MORGOTH is defeated ({why}){consequence}")

    def _refresh_gondolin_shield(self, state: State) -> None:
        """Lift the shield for Men adjacency, and as a backstop when the double is gone.

        :meth:`_lift_gondolin_shield_on_order` covers every ordered departure
        (a written Move or Dislodge lifts the shield in
        ``preprocess_movement``, ahead of this method's run on the amended
        post-resolution state); this still catches a double that leaves GON
        by some other means, such as a Layer-2 relocation this hook does not
        itself gate on a Move or Dislodge order.
        """
        if not self.gondolin_shield:
            return
        entry = (
            self.ledger.units.get(self.gondolin_double_uid)
            if self.gondolin_double_uid is not None
            else None
        )
        if entry is None or entry.limbo or entry.location.province != GONDOLIN:
            self.gondolin_shield = False
            self._emit("GONDOLIN's shield is lifted (the double army left Gondolin)")
            return
        adjacent = self._absolute_neighbors(GONDOLIN)
        if any(u.power == MEN and u.location.province in adjacent for u in state.units):
            self.gondolin_shield = False
            self._emit("GONDOLIN's shield is lifted (a mannish unit stands adjacent)")

    def _pick_teleri_disorder(self, state: State) -> None:
        """Draw the Teleri army that stands in disorder this phase, if any.

        The rulebook's "Teleri" section: *"If Teleri has more Armies than
        Fleets, then one of Armies, chosen at random, will stand in disorder,
        i. e. will have forced an order to Hold"*, chosen before the orders are
        placed and anew each phase. The pick is recorded as its unit value
        plus its occurrence index among equal armies in state order, so
        :meth:`_void_disorder` voids every order written for exactly that unit
        among fungible twins: the twin of that index in Phase 0's binding of
        the sheet.
        """
        armies = [u for u in state.units if u.power == TELERI and u.kind is UnitKind.ARMY]
        fleets = sum(1 for u in state.units if u.power == TELERI and u.kind is UnitKind.FLEET)
        if not armies or len(armies) <= fleets:
            return
        pick = self.rng.randrange(len(armies))
        chosen = armies[pick]
        value = (
            chosen.location.province,
            chosen.location.coast,
            chosen.kind.value,
            chosen.strength,
        )
        occurrence = sum(
            1
            for u in armies[:pick]
            if (u.location.province, u.location.coast, u.kind.value, u.strength) == value
        )
        self._teleri_locked = (value, occurrence)
        self._emit(f"TELERI {chosen} stands in disorder (more armies than fleets)")

    # -- movement preprocessing --------------------------------------------------------

    def _order_unit_value(self, order: GameOrder) -> tuple[str, str | None, str, int] | None:
        """Return the (province, coast, kind, strength) of an order's unit, or None."""
        kind = getattr(order, "unit_kind", None)
        location = getattr(order, "location", None)
        if kind is None or location is None:
            return None
        strength = getattr(order, "strength", None) or 1
        return (location.province, location.coast, kind.value, strength)

    def _subject_owner(self, power: str, order: Support, state: State) -> str | None:
        """Return the supported subject's owner, or None when it is ambiguous.

        The written ``subject_power`` wins; otherwise it is the unique power
        occupying the subject's province with a unit matching the subject
        spec. A lone occurrence equal to the supporter is discounted (a unit
        is never its own subject). An ambiguous subject is Phase 0's
        invalid-reference problem, not a support-legality one, so the caller
        treats it as allowed.
        """
        if order.subject_power is not None:
            return order.subject_power
        subject = order.subject
        # an unwritten prefix names base strength 1, never every base strength
        # ("A is same as 1A"): a bare subject over a mixed Area must not widen the
        # candidate set to the owner of a unit of another base strength and turn
        # a legal support into a banned one, or the reverse
        candidates = [
            u
            for u in state.occupants(subject.location.province)
            if u.kind is subject.unit_kind and (subject.strength or 1) == u.strength
        ]
        if subject.location.province == order.location.province:
            supporter = Unit(power, order.unit_kind, order.location, order.strength or 1)
            if supporter in candidates:
                candidates.remove(supporter)
        matches = {u.power for u in candidates}
        if len(matches) != 1:
            return None
        return next(iter(matches))

    def _support_forbidden(self, power: str, order: Support, state: State) -> str | None:
        """Return the sentence of the ban that voids ``order``, or None when it stands.

        Support legality under the rulebook's "Elven Powers" and "Feanorian
        Powers" sections, whose governing sentences read:

            "None of the elven power can support or be supported by Morgoth."
            "Any Power with Unit in Anfauglith may not be Supported by
             Feanorian Powers."

        Four preprocess-level bans (invalid -> FAILS + effective Hold), in the
        order tested below:
        (1) no elvish power (Feanorians included) may support a MORGOTH unit;
        (2) MORGOTH may not support any elvish unit -- MEN and NAUGRIM subjects
            stay legal; (3) the Feanorian<->other-elvish mutual ban, with no
            Anfauglith exception in either direction;
        (4) a power keeping any unit in Anfauglith is unsupportable by the
            Feanorians power-wide: any unit of that owner, standing anywhere, is
            unsupportable while any of the owner's units occupies ANF (order-time
            occupancy, read from the start-of-phase ``state``).

        Self-support is exempt (owner == supporter short-circuits first, even
        when the Feanorians themselves hold a unit in Anfauglith); MEN and
        NAUGRIM supporters are unrestricted in every direction; convoys are
        untouched (the clauses name Support only).

        The sentence is presentation only: it is shown to the player whose
        support died and never read by the calculus.
        """
        if power not in ELVISH and power != MORGOTH:
            return None
        owner = self._subject_owner(power, order, state)
        if owner is None or owner == power:
            return None
        if power == MORGOTH:  # (2) Morgoth supports no elvish unit
            return BAN_MORGOTH_SUPPORTS_ELVES if owner in ELVISH else None
        if owner == MORGOTH:
            # (1) no elvish power supports Morgoth (Feanorians included)
            return BAN_ELVES_SUPPORT_MORGOTH
        if power == FEANORIANS:
            if owner in ELVISH:
                # (3) Feanorian <-> other-elvish, no Anfauglith exception
                return BAN_FEANORIAN_MUTUAL
            holds_anfauglith = any(  # (4) unsupportable power-wide while it holds ANF
                unit.power == owner and unit.location.province == ANFAUGLITH for unit in state.units
            )
            return BAN_ANFAUGLITH if holds_anfauglith else None
        # reverse Feanorian<->other-elvish ban, unchanged
        return BAN_FEANORIAN_MUTUAL if owner == FEANORIANS else None

    def _entry_need(self, power: str, destination: str) -> Callable[[LedgerUnit], bool] | None:
        """Return the identity credential a relocation or retreat into ``destination`` needs.

        None means power-level rules suffice and the wrapper settles them.
        One-way entry (Angband/Eriador) carries no credential: the value-level
        gate admits only a MORGOTH strength-3 unit into Angband and nobody into
        Eriador, and Phase 0 downgrades any illegal move to a Hold before
        binding, as the retreat binding passes over an invalid retreat.
        Movement records and dislodgements alike are told apart by strength,
        so a surviving 3A->Angband move or retreat binds a 3A identity without
        one.
        """
        if destination in self.devastated and power != MORGOTH:
            return _needs_mask
        return None

    def _lift_gondolin_shield_on_order(self, sheets: Mapping[str, Sequence[GameOrder]]) -> None:
        """Lift the shield on the double's *written* Move or Dislodge, whatever its outcome.

        The rulebook's "Gondolin" section: *"Until 2A(Gon) is Ordered an Order
        that changes its location (whether or not the Order succeeds) ... no
        other Power may Attack Gondolin Province."* An Order that changes its
        location is a Move or Dislodge naming the shield double's own unit
        spec exactly -- Support, Hold, MakeMasks and mask directives do not
        lift it, and the written order counts even if Phase 0 later
        invalidates it (the same written-order-not-outcome principle as
        :meth:`_mark_oath_stab`).

        Runs before any per-order filtering, from :meth:`preprocess_movement`
        -- ahead of :meth:`adjudicate`'s own Phase 0 pass -- so a same-phase
        attack into GON sees the lift already applied to
        ``DynamicBeleriandRuleset.entry_permission`` (orders are revealed
        simultaneously: the shield is down for the whole phase the
        qualifying order is written in, not just the next one).
        """
        if not self.gondolin_shield or self.gondolin_double_uid is None:
            return
        entry = self.ledger.units.get(self.gondolin_double_uid)
        if entry is None:
            return
        target = (entry.location.province, entry.location.coast, entry.kind.value, entry.strength)
        for order in sheets.get(GONDOLIN_POWER, ()):
            if isinstance(order, RELOCATING_TYPES) and self._order_unit_value(order) == target:
                self.gondolin_shield = False
                self._emit("GONDOLIN's shield is lifted (the double army ordered out of Gondolin)")
                return

    def preprocess_movement(
        self, phase: Phase, state: State, sheets: Mapping[str, Sequence[GameOrder]]
    ) -> MovementPlan:
        """Split the sheets into what the adjudicator sees and what the hooks resolve.

        Per sheet: a defeated Morgoth's orders are discarded (civil disorder);
        an order that names a special without its designator, or with a false
        one, is invalid; a banned support is voided with the ban's sentence as
        its reason; a relocation into gated ground that Phase 0 keeps must
        claim a credentialed identity (:meth:`_void_uncredentialed`); last, the
        Teleri army in disorder loses every order written for it
        (:meth:`_void_disorder`). Layer-2 orders are deferred to
        :meth:`postprocess_movement`; everything else is adjudicated, and every
        special left without a valid order of its own is given a designated
        Hold to hold under.
        """
        del phase
        self._lift_gondolin_shield_on_order(sheets)
        self._injected = {}
        plan = MovementPlan(adjudicated={})
        for power, sheet in sheets.items():
            kept: list[Order] = []
            positions: list[int] = []  # the sheet index of each kept order
            for index, order in enumerate(sheet):
                if power == MORGOTH and self.morgoth_defeated:
                    plan.invalid[(power, index)] = order
                    continue
                if not self._designations_valid(power, order):
                    plan.invalid[(power, index)] = order  # a special named wrongly: dropped
                    continue
                if isinstance(order, ADJUDICATION_ORDER_TYPES):
                    ban = (
                        self._support_forbidden(power, order, state)
                        if isinstance(order, Support)
                        else None
                    )
                    if ban is not None:
                        plan.invalid[(power, index)] = order
                        plan.reasons[(power, index)] = ban
                        continue
                    kept.append(order)
                    positions.append(index)
                else:
                    plan.deferred[(power, index)] = order
            self._void_uncredentialed(state, power, kept, positions, plan)
            if power == TELERI and self._teleri_locked is not None:
                self._void_disorder(state, kept, positions, plan)
            plan.adjudicated[power] = kept
        self._inject_special_holds(plan.adjudicated)
        return plan

    def _void_uncredentialed(
        self,
        state: State,
        power: str,
        kept: list[Order],
        positions: list[int],
        plan: MovementPlan,
    ) -> None:
        """Void each relocation into gated ground that Phase 0 keeps and no credential covers.

        A relocation into devastated ground needs an identity of its unit's
        class that carries a mask (:meth:`_entry_need`). The credentials go
        to the lines Phase 0 keeps, in sheet order: a line Phase 0 ignores as
        illegal, or passes over because its unit already has its order, claims
        none, so it never spends the mask of a legal line below it (the
        rulebook's 4.E.1: *"if the same Unit is given another, legal Order,
        that Order is executed"*). A kept line left without a
        credential is voided like an illegal order -- removed from ``kept``
        and ``positions`` into ``plan.invalid`` -- and its unit takes its next
        legal line, which Phase 0 then keeps and which claims in turn; every
        round voids one line, so the rounds end. A power's binding reads its
        own sheet and the board alone, so Phase 0 runs on that sheet only.
        The Teleri army in disorder claims none either: it holds, and
        :meth:`_void_disorder` voids its line on this same binding, so the
        line never spends the mask a twin's line needs.
        """

        def need_of(order: Order) -> Callable[[LedgerUnit], bool] | None:
            target = entry_province(order)
            return None if target is None else self._entry_need(power, target)

        if not any(need_of(order) is not None for order in kept):
            return
        while True:
            records = phase0(self.graph, self._require_wrapped(), state, {power: kept}).records
            disordered = (
                self._disorder_twins(records)[1].index
                if power == TELERI and self._teleri_locked is not None
                else None
            )
            claimed: set[int] = set()  # identities consumed by credential claims
            voided: int | None = None
            for record in sorted(records, key=lambda r: r.sheet_pos):
                written = record.written
                if written is None or not record.valid:  # unordered, or ignored as illegal
                    continue
                if record.index == disordered:  # holds: its line is void in _void_disorder
                    continue
                need = need_of(written)
                if need is not None and not self._claim(power, written, need, claimed, limbo=False):
                    voided = record.sheet_pos
                    break
            if voided is None:
                return
            plan.invalid[(power, positions.pop(voided))] = kept.pop(voided)

    def _disorder_twins(
        self, records: Sequence[OrderRecord]
    ) -> tuple[list[OrderRecord], OrderRecord]:
        """Return the Teleri army in disorder's twin records, and the record of the army itself.

        The twins are the TELERI records of the pick's unit value, in record
        order; the army in disorder is the one of the pick's occurrence index.
        """
        assert self._teleri_locked is not None  # the callers check the pick
        value, occurrence = self._teleri_locked
        twins = [
            record
            for record in records
            if record.power == TELERI
            and (record.unit.location.province, record.unit.location.coast) == value[:2]
            and (record.unit.kind.value, record.unit.strength) == value[2:]
        ]
        return twins, twins[occurrence]

    def _void_disorder(
        self,
        state: State,
        kept: list[Order],
        positions: list[int],
        plan: MovementPlan,
    ) -> None:
        """Void every order written for the Teleri army in disorder, and no order of its twins.

        The rulebook's "Teleri": the army *"will have forced an order to
        Hold"*. Which twin an order is written for is Phase 0's binding of the
        sheet the adjudicator will see -- the orders voided above are gone, an
        illegal one is ignored, and of several orders for one unit the first
        legal one counts -- and the army in disorder is the twin of the pick's
        occurrence index in that binding. Every other twin keeps the order
        Phase 0 gives it there, and the Layer-2 order
        :meth:`_bind_deferred_mains` gives it on that binding. Every other
        order naming the value is void: the army in disorder's own, and each
        one bound to no twin, since such an order could take the voided one's
        place. The kept orders bind the other twins again in the same order,
        so one twin is left with no legal order and holds.
        """
        assert self._teleri_locked is not None  # the caller checks the pick
        value = self._teleri_locked[0]
        deferred = sorted(
            (key, order)
            for key, order in plan.deferred.items()
            if key[0] == TELERI and self._order_unit_value(order) == value
        )
        if not deferred and not any(order.location.province == value[0] for order in kept):
            return  # no order binds a twin of the army in disorder
        parsed = phase0(self.graph, self._require_wrapped(), state, {TELERI: kept})
        twins, picked = self._disorder_twins(parsed.records)
        others = {record.index for record in twins if record is not picked}
        keep = {
            record.sheet_pos
            for record in twins
            if record is not picked and record.written is not None
        }
        for line in range(len(kept) - 1, -1, -1):
            own = picked.written is not None and picked.sheet_pos == line
            if line not in keep and (own or self._order_unit_value(kept[line]) == value):
                plan.invalid[(TELERI, positions.pop(line))] = kept.pop(line)
        bound = self._bind_deferred_mains(parsed.records, parsed.illegal, deferred, {})
        for key, _ in deferred:
            if bound.get(key) not in others:
                plan.invalid[key] = plan.deferred.pop(key)

    def _inject_special_holds(self, adjudicated: dict[str, list[Order]]) -> None:
        """Give every special without a valid order of its own a designated Hold to hold under.

        The adjudicator reads a unit's signature off its effective order, so
        a special that wrote nothing, or whose order the checks above threw
        out, would reach it bare -- and two bare specials in one Area would be
        twins: one wall, one wave, one list for rule 6 to walk. The Morgoth
        unit and Glaurung are never twins, so each of them holds under its
        own designator instead. The Hold is appended to the sheet the
        adjudicator sees and claims the special's record there; its row is
        dropped again by :meth:`merge_resolutions`, so a player who wrote
        nothing sees nothing and one whose order was invalid still sees that
        order failing. A Layer-2 main order binds such a record as an
        unordered one (:meth:`_bind_deferred_mains`): the unit did not act.
        """
        for entry in self.ledger.board():
            if entry.special is None:
                continue
            letter = SPECIAL_DESIGNATOR[entry.special]
            sheet = adjudicated.setdefault(entry.power, [])
            if any(getattr(order, "special", None) == letter for order in sheet):
                continue
            hold = Hold(
                entry.kind, entry.location, entry.strength if entry.strength != 1 else None, letter
            )
            self._injected[(entry.power, len(sheet))] = format_order(hold)
            sheet.append(hold)

    def _claim(
        self,
        power: str,
        order: GameOrder,
        need: Callable[[LedgerUnit], bool],
        claimed: set[int],
        *,
        limbo: bool,
    ) -> bool:
        """Consume one credentialed identity of the ordering unit's class for a gated move.

        Claims are granted in sheet order while unclaimed satisfying
        identities remain: different gates draw from the same twins, so a
        mask-holder spent on one claim cannot also cover another.
        """
        kind = getattr(order, "unit_kind", None)
        location = getattr(order, "location", None)
        if kind is None or location is None:  # pragma: no cover - relocations carry units
            return False
        probe = Unit(power, kind, location, getattr(order, "strength", None) or 1)
        for entry in self._ids_matching(probe, limbo=limbo):
            if entry.uid not in claimed and need(entry):
                claimed.add(entry.uid)
                return True
        return False

    def _claim_retreat(
        self,
        dislodgement: Dislodgement,
        need: Callable[[LedgerUnit], bool],
        claimed: set[int],
    ) -> bool:
        """Consume one credentialed limbo identity of a dislodgement's class for a gated retreat.

        The class is the dislodged unit's own -- power, origin, kind, base
        strength and the designator its record carries (:meth:`_limbo_class`)
        -- since that is the unit the adjudicator bound the retreat to; a
        stray line naming another unit can claim nothing from it. Twins are
        walked in ascending uid order, as :meth:`Ledger.assign` walks them in
        :meth:`post_retreat`.
        """
        for entry in self._limbo_class(dislodgement):
            if entry.uid not in claimed and need(entry):
                claimed.add(entry.uid)
                return True
        return False

    # -- movement postprocessing (the end-of-phase steps) -----------------------------

    def postprocess_movement(
        self,
        phase: Phase,
        pre_state: State,
        plan: MovementPlan,
        result: PhaseResult,
    ) -> tuple[State, Mapping[tuple[str, int], Resolution], tuple[Dislodgement, ...]]:
        """Replay the movement outcome onto the ledger and resolve the deferred orders.

        The end-of-phase steps run in rulebook order: transforms, devastation
        (which can itself dislodge), mask making, mask passes, sailings West;
        then the Girdle attack-grants, the Oath stab check and the entry
        tracking for double eligibility. Returns the amended state, the
        outcomes of the deferred orders keyed like the movement plan, and the extra
        dislodgements devastation caused, which the game merges into the
        phase result before the retreat stage.
        """
        ruleset = self._require_wrapped()
        parsed = phase0(self.graph, ruleset, pre_state, plan.adjudicated)
        records = parsed.records
        tables = build_tables(self.graph, ruleset, records)
        succeeded = self._record_statuses(records, plan.adjudicated, result)
        # The twin an unnamed attack on a stack took is the rule-6 engagement
        # under the final labeling. The rows carry the statuses but not the
        # Szykman marks the attack strengths read, so the labeling is settled
        # again -- the same pure function of these records the adjudication
        # ran -- and the engagement is read off it; a phase without a walk
        # group has none to read.
        walk_victims = (
            final_walk_victims(settle(self.graph, ruleset, records, tables))
            if tables.walk_groups
            else {}
        )
        dislodged = self._dislodged_records(records, tables, succeeded, result, walk_victims)
        victim_demands = self._designated_victim_demands(records, tables, succeeded, walk_victims)
        deferred = sorted(plan.deferred.items())
        outcomes: dict[tuple[str, int], Resolution] = {}

        main_binding = self._bind_deferred_mains(records, parsed.illegal, deferred, outcomes)
        fates = self._fates(records, succeeded, dislodged)
        record_uid = self._bind_identities(records, fates, deferred, main_binding, victim_demands)
        self._apply_flow(phase, records, fates, record_uid)

        attacks, supported = self._attack_maps(records, tables)
        units = list(result.new_state.units)

        self._resolve_transforms(
            deferred, main_binding, record_uid, dislodged, attacks, units, outcomes
        )
        # Devastate can itself dislodge co-located units: it mutates
        # ``dislodged`` in place BEFORE the mask steps below run, so a
        # devastation-fallen unit is already "dislodged" as far as
        # _resolve_masks_made/_resolve_mask_passes are concerned -- it can
        # neither make, give, nor accept a mask that phase.
        expelled = self._resolve_devastates(
            deferred, main_binding, record_uid, dislodged, units, outcomes
        )
        self._resolve_masks_made(deferred, main_binding, record_uid, dislodged, outcomes)
        self._resolve_mask_passes(
            deferred, main_binding, record_uid, dislodged, attacks, supported, outcomes
        )
        self._resolve_sails(deferred, main_binding, record_uid, dislodged, units, outcomes)
        self._girdle_attack_grants(records, succeeded)
        self._mark_oath_stab(records, pre_state)
        self._track_entries(records, fates)

        amended = State.of(result.new_state.map_name, result.new_state.phase, units)
        self._refresh_gondolin_shield(amended)
        return amended, outcomes, tuple(expelled)

    def _require_wrapped(self) -> Ruleset:
        if self._wrapped is None:  # pragma: no cover - Game always wraps first
            raise RuntimeError("hooks used before wrap_ruleset")
        return self._wrapped

    def merge_resolutions(
        self,
        sheets: Mapping[str, Sequence[GameOrder]],
        plan: MovementPlan,
        adjudicated: tuple[OrderResolution, ...],
        outcomes: Mapping[tuple[str, int], Resolution],
    ) -> tuple[OrderResolution, ...]:
        """Reassemble the rows over the submitted sheets, minus the Holds the layer injected.

        The designated Holds of :meth:`_inject_special_holds` are the
        adjudicator's business only: nobody wrote them, so nobody reads a row
        for them. Each sits at the end of its power's sheet, so its row is the
        last one of that power and text; a written row of the same text -- a
        defeated Morgoth's own designated Hold, thrown out with his sheet --
        keeps its earlier place and its failure.
        """
        merged = list(super().merge_resolutions(sheets, plan, adjudicated, outcomes))
        for (power, _), text in self._injected.items():
            for position in range(len(merged) - 1, -1, -1):
                row = merged[position]
                if row.power == power and row.text == text:
                    del merged[position]
                    break
        return tuple(merged)

    def _record_statuses(
        self,
        records: Sequence[OrderRecord],
        sheets: Mapping[str, Sequence[Order]],
        result: PhaseResult,
    ) -> dict[int, bool]:
        """Return, per record index, whether its written order succeeded.

        Rows align by the ``assemble`` duplicate-text discipline: the k-th
        duplicate row binds the k-th claiming record in sheet_pos order.
        """
        del sheets
        rows: dict[tuple[str, str, int], Resolution] = {}
        counters: dict[tuple[str, str], int] = {}
        for row in result.resolutions:
            k = counters.get((row.power, row.text), 0)
            counters[(row.power, row.text)] = k + 1
            rows[(row.power, row.text, k)] = row.resolution
        grouped: dict[tuple[str, str], list[OrderRecord]] = {}
        for record in records:
            if record.written is None:
                continue
            grouped.setdefault((record.power, format_order(record.written)), []).append(record)
        succeeded: dict[int, bool] = {}
        for (power, text), members in grouped.items():
            members.sort(key=lambda r: r.sheet_pos)
            for k, record in enumerate(members):
                verdict = rows.get((power, text, k))
                if verdict is not None:
                    succeeded[record.index] = verdict is Resolution.SUCCEEDS
        return succeeded

    def _dislodged_records(
        self,
        records: Sequence[OrderRecord],
        tables: Tables,
        succeeded: dict[int, bool],
        result: PhaseResult,
        walk_victims: Mapping[int, int],
    ) -> set[int]:
        """Bind the phase's dislodgements to record occurrences.

        Victim bindings first (successful targeted Dislodges, walk victims
        included -- the rule-6 assignment the adjudicator actually took;
        docs/adjudication.tex Sec. 3.4), then the unique stationary occupant,
        then ascending record index.
        """

        def relocated(record: OrderRecord) -> bool:
            return isinstance(record.effective, RELOCATING_TYPES) and succeeded.get(
                record.index, False
            )

        named: dict[tuple[str, str, str, int], list[int]] = {}
        for record in records:
            if isinstance(record.effective, Dislodge) and succeeded.get(record.index, False):
                victim_index = tables.victim_of.get(record.index)
                if victim_index is None:
                    victim_index = walk_victims.get(record.index)
                if victim_index is not None and not relocated(records[victim_index]):
                    victim = records[victim_index]
                    key = (
                        victim.unit.location.province,
                        victim.power,
                        victim.unit.kind.value,
                        victim.unit.strength,
                    )
                    named.setdefault(key, []).append(victim_index)
        chosen: set[int] = set()
        for dislodgement in result.dislodgements:
            unit = dislodgement.unit
            key = (unit.location.province, unit.power, unit.kind.value, unit.strength)
            candidates = named.get(key, [])
            picked = next((i for i in sorted(candidates) if i not in chosen), None)
            if picked is None:
                picked = next(
                    (
                        r.index
                        for r in records
                        if r.index not in chosen and r.unit == unit and not relocated(r)
                    ),
                    None,
                )
            if picked is not None:
                chosen.add(picked)
        return chosen

    def _designated_victim_demands(
        self,
        records: Sequence[OrderRecord],
        tables: Tables,
        succeeded: dict[int, bool],
        walk_victims: Mapping[int, int],
    ) -> dict[int, Callable[[LedgerUnit], bool]]:
        """Return the designator demands successful designated Dislodges place on victims.

        The named special is the identity that goes to limbo, though the
        value-level victim selection (docs/adjudication.tex Sec. 3.4) and the
        rest of adjudication never saw the designator. A walked attack's
        victim comes from the rule-6 assignment (the specials are fungible
        twins at the engine level; the demand makes the fallen record the
        named identity). Keyed by victim record index for
        :meth:`_bind_identities`.
        """
        demands: dict[int, Callable[[LedgerUnit], bool]] = {}
        for record in records:
            written = record.written
            if not isinstance(written, Dislodge) or written.victim.special is None:
                continue
            if not succeeded.get(record.index, False):
                continue
            victim_index = tables.victim_of.get(record.index)
            if victim_index is None:
                victim_index = walk_victims.get(record.index)
            if victim_index is not None:
                demands[victim_index] = _needs_special(_DESIGNATOR_TAG[written.victim.special])
        return demands

    def _fates(
        self,
        records: Sequence[OrderRecord],
        succeeded: dict[int, bool],
        dislodged: set[int],
    ) -> dict[int, tuple[str, Location]]:
        """Return each record's fate ("stay", "move" or "limbo") and its end location."""
        fates: dict[int, tuple[str, Location]] = {}
        for record in records:
            if record.index in dislodged:
                fates[record.index] = ("limbo", record.unit.location)
            elif isinstance(record.effective, RELOCATING_TYPES) and succeeded.get(
                record.index, False
            ):
                if isinstance(record.effective, Move):
                    landing = record.effective.destination
                else:
                    target = entry_province(record.effective)
                    assert target is not None
                    landing = Location(target)
                fates[record.index] = ("move", landing)
            else:
                fates[record.index] = ("stay", record.unit.location)
        return fates

    def _bind_deferred_mains(
        self,
        records: Sequence[OrderRecord],
        illegal: Collection[int],
        deferred: list[tuple[tuple[str, int], GameOrder]],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> dict[tuple[str, int], int]:
        """Bind the Layer-2 orders to record occurrences.

        Main orders (Transform/Devastate/MakeMasks/SailWest) claim *unordered*
        records, since a unit with an adjudication order cannot also
        main-act; transfer directives (GiveMask/AcceptMask) ride any record.
        Claims walk (power, sheet index) order against record-index order.

        A record is unordered when its unit wrote no adjudication order, or
        when every adjudication order written for it was illegal (``illegal``,
        Phase 0's record indices): an illegal order is ignored, so the unit
        takes its Layer-2 main order instead (the rulebook's "Invalid
        Orders": *"if the same Unit is given another, valid Order, that Order
        is executed"*). A legal order that proves invalid keeps its unit, which
        holds, as it does for the adjudicator. A special's record is
        unordered too when its written order is the
        designated Hold the layer injected for it (:meth:`_inject_special_holds`):
        the unit wrote no adjudication order, so its Devastate may still bind.
        A Layer-2 order binds only a record whose written order carries the
        designator it writes -- ``3A(G) ANF DEV`` takes Glaurung's record and
        never the Morgoth unit's beside it -- and a plain unit's order, which
        writes none, binds only a record that writes none.
        """
        unordered: dict[tuple[str, str | None, str, str, int], list[int]] = {}
        anyrec: dict[tuple[str, str | None, str, str, int], list[int]] = {}
        for record in records:
            key = (
                record.unit.location.province,
                record.unit.location.coast,
                record.power,
                record.unit.kind.value,
                record.unit.strength,
            )
            anyrec.setdefault(key, []).append(record.index)
            if (
                record.written is None
                or record.index in illegal
                or (record.power, record.sheet_pos) in self._injected
            ):
                unordered.setdefault(key, []).append(record.index)
        binding: dict[tuple[str, int], int] = {}
        taken_main: set[int] = set()
        taken_directive: dict[str, set[int]] = {"give": set(), "accept": set()}
        for (power, index), order in deferred:
            value = self._order_unit_value(order)
            if value is None:  # pragma: no cover - all variant orders carry a unit
                continue
            key = (value[0], value[1], power, value[2], value[3])
            designator = getattr(order, "special", None)

            def fits(record_index: int, designator: str | None = designator) -> bool:
                return getattr(records[record_index].written, "special", None) == designator

            if isinstance(order, (GiveMask, AcceptMask)):
                # one give and one accept may ride the same unit's phase, but
                # two gives (or two accepts) need two distinct occurrences
                used = taken_directive["give" if isinstance(order, GiveMask) else "accept"]
                pool = anyrec.get(key, [])
                chosen = next((i for i in pool if i not in used and fits(i)), None)
                if chosen is None:
                    outcomes[(power, index)] = Resolution.FAILS
                    continue
                used.add(chosen)
                binding[(power, index)] = chosen
            else:
                pool = unordered.get(key, [])
                chosen = next((i for i in pool if i not in taken_main and fits(i)), None)
                if chosen is None:
                    outcomes[(power, index)] = Resolution.FAILS
                    continue
                taken_main.add(chosen)
                binding[(power, index)] = chosen
        return binding

    def _bind_identities(
        self,
        records: Sequence[OrderRecord],
        fates: dict[int, tuple[str, Location]],
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        victim_demands: dict[int, Callable[[LedgerUnit], bool]],
    ) -> dict[int, int]:
        """Bind each record index to a ledger uid, credentialed demands first.

        Per fungibility group, records demanding a credential -- a gated
        destination, Glaurung for a Devastate, a mask to give, or the special
        a written ``(M)``/``(G)`` designator names -- claim the lowest
        satisfying uid first. ``victim_demands`` carries the designator a
        successful designated Dislodge places on its victim record, so the
        named special is the identity that goes to limbo. A record whose
        ``SAILS WEST`` will leave the board binds, among its twins, an
        identity with the fewest masks (:meth:`_resolve_sails`).
        """
        demands: dict[int, Callable[[LedgerUnit], bool]] = {}

        def demand(index: int, need: Callable[[LedgerUnit], bool]) -> None:
            prior = demands.get(index)
            demands[index] = need if prior is None else _and_(prior, need)

        for record in records:
            fate, landing = fates[record.index]
            if fate == "move":
                need = self._entry_need(record.power, landing.province)
                if need is not None:
                    demand(record.index, need)
            written = record.written
            own = getattr(written, "special", None) if written is not None else None
            if own is not None:  # a designated own-unit order
                demand(record.index, _needs_special(_DESIGNATOR_TAG[own]))
        for (power, index), order in deferred:
            bound = main_binding.get((power, index))
            if bound is None:
                continue
            if isinstance(order, Devastate):
                demand(bound, _needs_glaurung)
            elif isinstance(order, GiveMask):
                demand(bound, _needs_mask)
            designator = getattr(order, "special", None)
            if designator is not None:  # a designated Layer-2 main/directive
                demand(bound, _needs_special(_DESIGNATOR_TAG[designator]))
        for index, need in victim_demands.items():
            demand(index, need)
        sailing = {
            main_binding[(power, index)]
            for (power, index), order in deferred
            if isinstance(order, SailWest)
            and (power, index) in main_binding
            and self._sails_west(power, order, fates[main_binding[(power, index)]][0] == "limbo")
        }
        groups = self.ledger.board_groups()
        binding: dict[int, int] = {}
        members: dict[tuple[str, str | None, str, str, int], list[OrderRecord]] = {}
        for record in records:
            key = (
                record.unit.location.province,
                record.unit.location.coast,
                record.power,
                record.unit.kind.value,
                record.unit.strength,
            )
            members.setdefault(key, []).append(record)
        for key, recs in members.items():
            uids = groups.get(key, [])
            slot_demands = [demands.get(r.index) for r in recs]
            leaving = {position for position, r in enumerate(recs) if r.index in sailing}
            chosen = self.ledger.assign(uids, slot_demands, fewest_masks=leaving)
            for record, uid in zip(recs, chosen, strict=True):
                if uid is not None:
                    binding[record.index] = uid
        return binding

    @staticmethod
    def _sails_west(power: str, order: SailWest, dislodged: bool) -> bool:
        """Whether a ``SAILS WEST`` order takes its fleet off the board.

        The rulebook's "Sail West Order" admits a mannish fleet in the West
        seas that is not dislodged.
        """
        return (
            power == MEN
            and not dislodged
            and order.unit_kind is UnitKind.FLEET
            and order.location.province in WEST_SEAS
        )

    def _apply_flow(
        self,
        phase: Phase,
        records: Sequence[OrderRecord],
        fates: dict[int, tuple[str, Location]],
        record_uid: dict[int, int],
    ) -> None:
        """Move each bound identity to its fate: limbo, a new location, or unchanged."""
        for record in records:
            uid = record_uid.get(record.index)
            if uid is None:  # pragma: no cover - ledger covers every unit
                continue
            entry = self.ledger.units[uid]
            fate, landing = fates[record.index]
            if fate == "limbo":
                entry.limbo = True
            elif fate == "move":
                entry.location = landing
                self._arm_ulmo(entry, phase.year, phase.season)

    def _arm_ulmo(self, entry: LedgerUnit, year: int, season: Season) -> None:
        """Arm the coast's timer when a Morgoth unit arrives in a listed province."""
        province = entry.location.province
        if entry.power == MORGOTH and province in self.ulmo_delays:
            self.ulmo_armed[province] = _ordinal(year, season) + self.ulmo_delays[province]

    def _attack_maps(
        self,
        records: Sequence[OrderRecord],
        tables: Tables,
    ) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
        """Return who attacked whom this phase, and who supported such an attack.

        ``attacks[victim record]`` holds the attacker records and
        ``supported[victim]`` the records supporting such an attack; every
        Phase-0-valid order counts, success not required, because the
        transform and mask rules turn on being attacked, not on the outcome.
        """
        occupants: dict[str, list[int]] = {}
        for record in records:
            occupants.setdefault(record.unit.location.province, []).append(record.index)

        def move_victims(destination: str) -> list[int]:
            if self.graph.capacity(destination) == 1:
                return occupants.get(destination, [])
            return []

        attacks: dict[int, set[int]] = {}
        supported: dict[int, set[int]] = {}
        for record in records:
            if not record.valid or record.written is None:
                continue
            effective = record.effective
            if isinstance(effective, Move):
                for victim in move_victims(effective.destination.province):
                    if records[victim].power != record.power:
                        attacks.setdefault(victim, set()).add(record.index)
            elif isinstance(effective, Dislodge):
                position = tables.walk_group_of.get(record.index)
                if position is not None:
                    # An unnamed walk attack menaces the whole twin stack
                    # (rule 6 assigns its victim only at resolution) -- the
                    # threat-based cancel maps register every member, the
                    # same convention the ref-matching support branch below
                    # already uses.
                    for member_index in tables.walk_groups[position].pool:
                        attacks.setdefault(member_index, set()).add(record.index)
                    continue
                victim_index = tables.victim_of.get(record.index)
                if victim_index is not None:
                    attacks.setdefault(victim_index, set()).add(record.index)
            elif isinstance(effective, Support) and isinstance(effective.subject, RELOCATING_TYPES):
                subject = effective.subject
                if isinstance(subject, Move):
                    victims = [
                        v
                        for v in move_victims(subject.destination.province)
                        if records[v].power != record.power
                    ]
                else:
                    victims = [
                        v
                        for v in occupants.get(subject.victim.location.province, [])
                        if ref_matches_unit(subject.victim, records[v].unit)
                    ]
                for victim in victims:
                    supported.setdefault(victim, set()).add(record.index)
        return attacks, supported

    # -- the deferred-order resolutions ---------------------------------------------

    def _resolve_transforms(
        self,
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        record_uid: dict[int, int],
        dislodged: set[int],
        attacks: dict[int, set[int]],
        units: list[Unit],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> None:
        """Resolve the Transform orders at the end of the phase.

        The rulebook's "Fleets" section: a power that may use the
        fleet-building knowledge converts a unit standing in a coastal
        province to the other kind, strength carried over, unless the unit
        was attacked this phase (success not required) or dislodged.
        """
        for (power, index), order in deferred:
            if not isinstance(order, Transform):
                continue
            record_index = main_binding.get((power, index))
            outcomes[(power, index)] = Resolution.FAILS
            if record_index is None or record_index in dislodged:
                continue
            uid = record_uid.get(record_index)
            if uid is None:
                continue
            entry = self.ledger.units[uid]
            if (
                not usable_fleet_knowledge(self, power)
                or order.target_kind is entry.kind
                or self.graph.terrain(entry.location.province) is not Terrain.COASTAL
                or attacks.get(record_index)
            ):
                continue
            old = Unit(power, entry.kind, entry.location, entry.strength)
            if old not in units:  # pragma: no cover - flow invariant
                continue
            units.remove(old)
            entry.kind = order.target_kind
            units.append(Unit(power, entry.kind, entry.location, entry.strength))
            outcomes[(power, index)] = Resolution.SUCCEEDS
            self._emit(f"{power} {old} transforms to {entry.kind.value}")

    def _resolve_devastates(
        self,
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        record_uid: dict[int, int],
        dislodged: set[int],
        units: list[Unit],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> list[Dislodgement]:
        """Resolve the Devastate orders: Glaurung, undislodged, devastates his province.

        Devastation is permanent; a repeat on devastated ground succeeds and
        changes nothing. The units it expels are returned so the game can
        merge them into the phase's dislodgements.
        """
        expelled: list[Dislodgement] = []
        for (power, index), order in deferred:
            if not isinstance(order, Devastate):
                continue
            record_index = main_binding.get((power, index))
            outcomes[(power, index)] = Resolution.FAILS
            if power != MORGOTH or record_index is None or record_index in dislodged:
                continue
            uid = record_uid.get(record_index)
            if uid is None or self.ledger.units[uid].special != GLAURUNG:
                continue
            province = order.location.province
            if province not in self.devastated:
                self.devastated.add(province)
                self.devastation_years.setdefault(province, 0)
                self._emit(f"Glaurung devastates {province}")
                expelled.extend(self._expel_devastated(province, record_uid, dislodged, units))
            outcomes[(power, index)] = Resolution.SUCCEEDS
        return expelled

    def _expel_devastated(
        self,
        province: str,
        record_uid: dict[int, int],
        dislodged: set[int],
        units: list[Unit],
    ) -> list[Dislodgement]:
        """Dislodge every unmasked foreign unit standing in a newly devastated province.

        The rulebook's "Glaurung Unit and Devastation" section: *"the Units
        standing in it -- except Morgoth's own and Units equipped with Dwarven
        Masks -- are Dislodged and must Retreat"*. Runs once, the phase
        ``province`` first joins :attr:`devastated` (a re-Devastate of
        devastated ground expels nobody, matching the caller's own guard).
        Reads the ledger AFTER :meth:`_apply_flow`, so a unit that
        successfully moved into the Area this same phase is already at its
        landing spot here and falls with the rest.

        The exemption is per-identity (``entry.masks > 0``), not per class --
        of two otherwise-identical twins, a masked one saves only itself, so
        an unmasked twin still falls beside it (the ledger's claim
        discipline). A unit that still holds its mask at this exact moment
        stands even if it gives the mask away later this same phase (the
        expulsion is a one-time effect of the Devastate order resolving, not
        re-checked); a unit with none yet does not gain one ahead of this step,
        since mask-making resolves after devastation (see the caller).

        Each fallen record index joins ``dislodged`` in place -- the caller's
        shared set, also read by the mask-resolution steps that run after
        this one -- so a devastation-dislodged unit can neither make, give,
        nor accept a mask this phase, exactly like a combat-dislodged one.
        """
        uid_to_record = {uid: index for index, uid in record_uid.items()}
        fallen: list[Dislodgement] = []
        for entry in self.ledger.board():
            if entry.location.province != province or entry.power == MORGOTH or entry.masks > 0:
                continue
            unit = Unit(entry.power, entry.kind, entry.location, entry.strength)
            if unit not in units:  # pragma: no cover - flow invariant
                continue
            units.remove(unit)
            entry.limbo = True
            record_index = uid_to_record.get(entry.uid)
            if record_index is not None:
                dislodged.add(record_index)
            fallen.append(Dislodgement(unit=unit, dislodger_origin=None, by_convoy=False))
            self._emit(f"Glaurung's devastation dislodges {entry.power} {unit}")
        return fallen

    def _resolve_masks_made(
        self,
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        record_uid: dict[int, int],
        dislodged: set[int],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> None:
        """Resolve the MakeMasks orders: a Naugrim unit in a dwarf city makes two.

        The rulebook's "Dwarven Masks" section: *"When a dwarven Unit is in
        Belegost, Mount Dolmed or Nogrod it may make Dwarven Masks; only the
        Naugrim may make masks."* Masks are the dwarven diplomatic card, so
        every other power's ``MakeMasks`` FAILS wherever it stands. That
        covers *"Morgoth units can never acquire masks and therefore they can
        never order accept masks or give masks."* as well: MORGOTH is refused
        here as one non-dwarf among many, and standing in Belegost / Mount
        Dolmed / Nogrod does not matter. Only making is dwarven -- a made mask
        passes to any non-MORGOTH power (``_resolve_mask_passes``).
        """
        for (power, index), order in deferred:
            if not isinstance(order, MakeMasks):
                continue
            record_index = main_binding.get((power, index))
            outcomes[(power, index)] = Resolution.FAILS
            if power != NAUGRIM:
                continue
            if record_index is None or record_index in dislodged:
                continue
            if order.location.province not in DWARF_CITIES:
                continue
            uid = record_uid.get(record_index)
            if uid is None:
                continue
            entry = self.ledger.units[uid]
            entry.masks += 2
            outcomes[(power, index)] = Resolution.SUCCEEDS
            self._emit(f"{power} {entry.spec()} makes 2 masks")

    def _resolve_mask_passes(
        self,
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        record_uid: dict[int, int],
        dislodged: set[int],
        attacks: dict[int, set[int]],
        supported: dict[int, set[int]],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> None:
        """Match GIVES/ACCEPTS halves at end-of-phase positions and execute the transfers.

        The rulebook's "Dwarven Masks" section governs the pairing: givers
        pair with the first cross-matching acceptor in (power, sheet) order,
        the two must stand in the same Area or in adjacent provinces, neither
        may be dislodged, and neither may have attacked or supported an
        attack on the other this phase. A pair whose giver or acceptor is
        MORGOTH never matches -- both halves keep FAILS and no mask count
        changes -- and that guard reads unit identity, not the live mask
        count, so a loaded save is never trusted to have kept Morgoth's count
        at zero.
        """
        gives = [(key, order) for key, order in deferred if isinstance(order, GiveMask)]
        accepts = [(key, order) for key, order in deferred if isinstance(order, AcceptMask)]

        def conflict(a: int, b: int) -> bool:
            return (
                b in attacks.get(a, set())
                or a in attacks.get(b, set())
                or b in supported.get(a, set())
                or a in supported.get(b, set())
            )

        def bound_entry(key: tuple[str, int]) -> tuple[int, LedgerUnit] | None:
            record_index = main_binding.get(key)
            if record_index is None or record_index in dislodged:
                return None
            uid = record_uid.get(record_index)
            if uid is None:
                return None
            return record_index, self.ledger.units[uid]

        used_accepts: set[tuple[str, int]] = set()
        for accept_key, _ in accepts:
            outcomes.setdefault(accept_key, Resolution.FAILS)
        for give_key, give in gives:
            outcomes[give_key] = Resolution.FAILS
            giver_bound = bound_entry(give_key)
            if giver_bound is None:
                continue
            giver_record, giver = giver_bound
            if giver.masks < 1:
                continue
            for accept_key, accept in accepts:
                if accept_key in used_accepts:
                    continue
                acceptor_bound = bound_entry(accept_key)
                if acceptor_bound is None:
                    continue
                acceptor_record, acceptor = acceptor_bound
                if not self._refs_cross_match(give, accept, giver, acceptor):
                    continue
                if MORGOTH in (giver.power, acceptor.power):
                    continue
                if conflict(giver_record, acceptor_record):
                    continue
                if not self._mask_geometry_ok(giver, acceptor):
                    continue
                if not self._mask_girdle_ok(giver, acceptor):
                    continue
                giver.masks -= 1
                acceptor.masks += 1
                used_accepts.add(accept_key)
                outcomes[give_key] = Resolution.SUCCEEDS
                outcomes[accept_key] = Resolution.SUCCEEDS
                self._emit(
                    f"{giver.power} {giver.spec()} passes a mask"
                    f" to {acceptor.power} {acceptor.spec()}"
                )
                break

    def _refs_cross_match(
        self, give: GiveMask, accept: AcceptMask, giver: LedgerUnit, acceptor: LedgerUnit
    ) -> bool:
        """Whether the give's receiver ref and the accept's giver ref name each other's unit."""
        receiver_ref = give.receiver
        giver_ref = accept.giver
        return (
            receiver_ref.power == acceptor.power
            and receiver_ref.kind is acceptor.kind
            and (receiver_ref.strength or 1) == acceptor.strength
            and receiver_ref.location.province == acceptor.location.province
            and giver_ref.power == giver.power
            and giver_ref.kind is giver.kind
            and (giver_ref.strength or 1) == giver.strength
            and giver_ref.location.province == giver.location.province
        )

    def _mask_geometry_ok(self, giver: LedgerUnit, acceptor: LedgerUnit) -> bool:
        return self._mask_geometry_provinces_ok(giver.location.province, acceptor.location.province)

    def _mask_geometry_provinces_ok(self, giver_province: str, acceptor_province: str) -> bool:
        """Whether a mask may pass from ``giver_province`` to ``acceptor_province``.

        The acceptor's province is a (per-kind) neighbour of the giver's, or
        the same province -- the giver is the FIRST argument, matching the
        pass direction. This is the exact predicate :meth:`_mask_geometry_ok`
        applies to resolved end positions; :meth:`_transfer_orders` shares it,
        so an enumerated directive can never name an end pair the
        resolution's geometry would reject (the direction-consistency
        invariant -- one-way map edges make the reversed acceptor->giver test
        differ, so the direction is standardized here).
        """
        if giver_province == acceptor_province:
            return True
        return acceptor_province in self._absolute_neighbors(giver_province)

    def _mask_girdle_ok(self, giver: LedgerUnit, acceptor: LedgerUnit) -> bool:
        """Whether a mask pass across the Girdle boundary is permitted, granting as it goes.

        A mask crossing the Girdle boundary carries the permission to the
        acceptor -- the second voluntary channel, so the Feanorian source
        rule applies here and is enforced here. A Feanorian acceptor taking
        an elvish giver's mask gains nothing and the pass then fails on the
        membership test below (BATC-L2.M.11).
        """
        a_in = giver.location.province in GIRDLE
        b_in = acceptor.location.province in GIRDLE
        if a_in == b_in:
            return True
        if acceptor.power not in self.girdle and self._voluntary_source_ok(
            acceptor.power, giver.power
        ):
            self._grant_girdle(acceptor.power, "mask acceptance")
        return giver.power in self.girdle and acceptor.power in self.girdle

    def _resolve_sails(
        self,
        deferred: list[tuple[tuple[str, int], GameOrder]],
        main_binding: dict[tuple[str, int], int],
        record_uid: dict[int, int],
        dislodged: set[int],
        units: list[Unit],
        outcomes: dict[tuple[str, int], Resolution],
    ) -> None:
        """Resolve the SailWest orders: a mannish fleet in the West seas leaves the board.

        The rulebook's "Sail West Order": the fleet is removed and the Valar
        intervene after the season; a dislodged fleet fails to sail.
        """
        for (power, index), order in deferred:
            if not isinstance(order, SailWest):
                continue
            record_index = main_binding.get((power, index))
            outcomes[(power, index)] = Resolution.FAILS
            if record_index is None or not self._sails_west(
                power, order, record_index in dislodged
            ):
                continue
            uid = record_uid.get(record_index)
            if uid is None:
                continue
            entry = self.ledger.units[uid]
            unit = Unit(power, entry.kind, entry.location, entry.strength)
            if unit not in units:  # pragma: no cover - flow invariant
                continue
            units.remove(unit)
            self._release_older_centre(entry)
            self.ledger.kill(uid)
            self.sail_west_pending = True
            self.sail_west_ever = True
            outcomes[(power, index)] = Resolution.SUCCEEDS
            self._emit(f"MEN {unit} sails into the West -- the Valar are summoned")

    def _girdle_attack_grants(
        self,
        records: Sequence[OrderRecord],
        succeeded: Mapping[int, bool],
    ) -> None:
        """Grant the Girdle permission to every power a Girdle sortie attacked this phase.

        Any valid relocating order out of FoR/FoN grants the attacked unit's
        owner the permission: the attempt is the attack, so a bounce grants
        too. The rulebook's "Doriath" section: *"If a Unit in FoR or FoN is
        Ordered to Move into a Province which is not an Area and in which
        another Power's Unit stands, or to Dislodge a Unit in an adjacent
        Area, then the owner of the Attacked Unit gains ability to pass the
        Girdle of Melian - whether or not the Order succeeds. If the Attacked
        Unit successfully Moves away in the same Phase, it was not Attacked,
        and its owner gains nothing."* Each clause is a line below:

        * *"a Province which is not an Area"* is the capacity-1 test -- every
          Area on this board has capacity 6 and every other on-board province
          capacity 1 -- so entering an Area beside a foreign unit is
          coexistence, not an attack.
        * *"or to Dislodge"* always attacks; a Support never does.
        * *"whether or not the Order succeeds"* is why a bounce grants too.
        * *"successfully Moves away ... gains nothing"* is the :func:`remains`
          filter, the same guard :meth:`_dislodged_records` and
          :meth:`_designated_victim_demands` apply.

        The question this asks is **power-level and positional**, not a unit
        binding: *was the attacked province still occupied by that power once
        the phase resolved?* The book conditions the grant on a province *"in
        which another Power's Unit stands"* and pays *"the owner of the
        Attacked Unit"*, and both halves follow from occupancy alone. A stack
        of identical twins has an owner even when the adjudicator declines to
        name one of them, so a Dislodge reads the owner and province its own
        order names. That is why no victim binding appears here:
        ``Tables.victim_of`` and the rule-6 walk assignment answer *which
        unit*, a harder question this grant never has to ask, and asking it
        would make a failed attack on two identical armies pay less than the
        same attack on one.
        """

        def remains(index: int) -> bool:
            record = records[index]
            return not (
                isinstance(record.effective, RELOCATING_TYPES) and succeeded.get(index, False)
            )

        occupants: dict[str, list[int]] = {}
        for record in records:
            occupants.setdefault(record.unit.location.province, []).append(record.index)
        for record in records:
            if not record.valid or record.written is None:
                continue
            if record.unit.location.province not in GIRDLE:
                continue
            effective = record.effective
            victims: list[int] = []
            if isinstance(effective, Move):
                if self.graph.capacity(effective.destination.province) == 1:
                    victims = occupants.get(effective.destination.province, [])
            elif isinstance(effective, Dislodge):
                victim_ref = effective.victim
                victims = [
                    index
                    for index in occupants.get(victim_ref.location.province, [])
                    if records[index].power == victim_ref.power
                ]
            for victim in victims:
                if not remains(victim):
                    continue
                owner = records[victim].power
                if owner != record.power:
                    self._grant_girdle(owner, "attacked from the Girdle")

    def _track_entries(
        self, records: Sequence[OrderRecord], fates: dict[int, tuple[str, Location]]
    ) -> None:
        """Note every power that moved into a province this year, for double eligibility."""
        for record in records:
            fate, landing = fates[record.index]
            if fate == "move":
                self.entered_this_year.setdefault(landing.province, set()).add(record.power)

    # -- retreats ---------------------------------------------------------------------

    def preprocess_retreat(
        self,
        phase: Phase,
        sheets: Mapping[str, Sequence[AnyOrder]],
        movement: PhaseResult,
    ) -> tuple[Mapping[str, Sequence[AnyOrder]], list[tuple[str, AnyOrder]]]:
        """Gate the retreat sheets: Morgoth's defeat, the designators, the entry credentials.

        A defeated Morgoth's sheet is discarded. A retreat order for a
        dislodged special must write its designator, and the designator must
        name the special dislodged from the order's origin (it stands in
        limbo there); written bare, or with a false designator, the retreat is
        invalid, and with no valid retreat the unit disbands -- the Morgoth
        unit's death being Morgoth's defeat. A retreat into gated ground that
        no credential covers is invalid too (:meth:`_withhold_uncredentialed`).
        The withheld orders come back in sheet order.
        """
        del phase
        kept: dict[str, list[tuple[int, AnyOrder]]] = {}
        invalid: list[tuple[str, int, AnyOrder]] = []
        for power, sheet in sheets.items():
            lines: list[tuple[int, AnyOrder]] = []
            for index, order in enumerate(sheet):
                defeated = power == MORGOTH and self.morgoth_defeated
                if defeated or not self._designations_valid(power, order, limbo=True):
                    invalid.append((power, index, order))
                else:
                    lines.append((index, order))
            kept[power] = lines
        invalid.extend(self._withhold_uncredentialed(movement, kept))
        invalid.sort(key=lambda line: (line[0], line[1]))
        gated = {power: [order for _, order in lines] for power, lines in kept.items()}
        return gated, [(power, order) for power, _, order in invalid]

    def _withhold_uncredentialed(
        self, movement: PhaseResult, kept: dict[str, list[tuple[int, AnyOrder]]]
    ) -> list[tuple[str, int, AnyOrder]]:
        """Withhold each retreat into gated ground left without a credential; return them.

        The value-level gate (``entry_permission``) lets a retreat into
        devastated ground through when any identity of the unit's class
        carries a mask, but one mask covers one retreat. The credentials are
        spent in sheet order by the lines the adjudicator binds
        (``bind_retreat_orders``), each from the class of the unit its line
        binds (:meth:`_claim_retreat`), so an invalid line, a second order for
        a unit or a line naming another unit never spends the credential of
        the line that counts. The first bound line that finds no credential
        left is invalid: it is withheld from ``kept`` and the sheets are bound
        again, since its unit may now take a later line. Whether a line binds
        depends on the lines above it alone, so the claims above a withheld
        line stand.
        """
        withheld: list[tuple[str, int, AnyOrder]] = []
        while True:
            sheets = {power: [order for _, order in lines] for power, lines in kept.items()}
            bound = bind_retreat_orders(self._require_wrapped(), movement, sheets, graph=self.graph)
            culprit: tuple[str, int] | None = None
            for power in sorted(sheets):
                claimed: set[int] = set()
                for position in range(len(sheets[power])):
                    binding = bound.get((power, position))
                    if binding is None:
                        continue
                    dislodgement, retreat = binding
                    if not isinstance(retreat, Move):
                        continue
                    need = self._entry_need(power, retreat.destination.province)
                    if need is not None and not self._claim_retreat(dislodgement, need, claimed):
                        culprit = (power, position)
                        break
                if culprit is not None:
                    break
            if culprit is None:
                return withheld
            power, position = culprit
            index, order = kept[power].pop(position)
            withheld.append((power, index, order))

    def _limbo_class(self, dislodgement: Dislodgement) -> list[LedgerUnit]:
        """Return the limbo identities a dislodgement may stand for, in ascending uid order.

        They are the identities of the dislodged unit's class -- power, origin,
        kind and base strength -- that carry the designator the dislodgement
        records: the named special for a designated record, the untagged
        identities for any other. Within a class the identities are twins the
        board cannot tell apart.
        """
        unit = dislodgement.unit
        return [
            entry
            for entry in self.ledger.limbo()
            if entry.power == unit.power
            and entry.kind is unit.kind
            and entry.strength == unit.strength
            and entry.location == unit.location
            and (SPECIAL_DESIGNATOR[entry.special] if entry.special is not None else None)
            == dislodgement.special
        ]

    def post_retreat(self, phase: Phase, result: RetreatResult) -> None:
        """Move the identities of the units that retreated; every other limbo identity disbands.

        A retreat moves the identity of the very unit the adjudicator bound it
        to (``result.retreated``): an identity of that dislodgement's class
        (:meth:`_limbo_class`), and for a retreat into gated ground one that
        holds the credential, claimed first; the rest of a class goes in
        ascending uid order, except that the identities carrying the most masks go
        first (:meth:`Ledger.assign`), so the twin left to die is one with the
        fewest. The rows are not read: a
        row's text says what was written, and an invalid line, a second order
        for one unit or a line for a unit still on the board binds no unit at
        all, so binding identities to rows would kill the identity of a unit
        that retreated on a later line. Every other limbo identity disbands
        -- its unit was ordered to, had no valid order, or bounced -- and the
        Morgoth unit's death is Morgoth's defeat. It disbands while the
        identities that retreat still stand in limbo beside it, so a death
        among twins frees the centre owned longest (the rulebook's "Double
        Units": *"of two twins, the point of the Supply Centre owned
        longer"*, :meth:`_release_older_centre`) and the twin that retreats,
        or that never left the board, keeps the newer one -- unless the twins
        carry different numbers of masks, which tells them apart: each then
        keeps its own centre.
        """
        classes: dict[tuple[Unit, str | None], list[tuple[Dislodgement, Location]]] = {}
        for dislodgement, landing in result.retreated:
            key = (dislodgement.unit, dislodgement.special)
            classes.setdefault(key, []).append((dislodgement, landing))
        moves: dict[int, Location] = {}
        for retreats in classes.values():
            uids = [entry.uid for entry in self._limbo_class(retreats[0][0])]
            demands = [
                self._entry_need(dislodgement.unit.power, landing.province)
                for dislodgement, landing in retreats
            ]
            chosen = self.ledger.assign(uids, demands, masks_first=True)
            for (_dislodgement, landing), uid in zip(retreats, chosen, strict=True):
                if uid is None:  # pragma: no cover - the ledger holds every dislodged unit
                    continue
                moves[uid] = landing
        for entry in list(self.ledger.limbo()):
            if entry.uid not in moves:
                self._kill_entry(entry, "disbands during retreats")
        for uid, landing in moves.items():
            entry = self.ledger.units[uid]
            entry.limbo = False
            entry.location = landing
            self._arm_ulmo(entry, phase.year, phase.season)
            self.entered_this_year.setdefault(landing.province, set()).add(entry.power)

    def _kill_entry(self, entry: LedgerUnit, why: str) -> None:
        """Drop an identity; the Morgoth unit's death is Morgoth's defeat."""
        self._release_older_centre(entry)
        self.ledger.kill(entry.uid)
        if entry.special == MORGOTH_UNIT:
            self._mark_morgoth_defeated(f"the Morgoth unit {why}")

    # -- adjustments --------------------------------------------------------------------

    def stash_binding(self, unit: Unit, bound_to: str | None) -> None:
        """Remember a build's binding; the winter routine calls this per build, in order."""
        self._pending_bindings.append((unit, bound_to))

    def adjudicate_adjustments(
        self,
        phase: Phase,
        state: State,
        ownership: Mapping[str, str],
        sheets: Mapping[str, Sequence[AnyOrder]],
    ) -> AdjustmentResult | None:
        """Adjudicate the Beleriand winter in place of the standard adjustment routine."""
        del phase
        self._pending_bindings = []
        self._winter_ownership = dict(ownership)
        return adjudicate_winter(self, state, ownership, sheets)

    def post_adjustment(self, phase: Phase, result: AdjustmentResult) -> None:
        """Replay the winter onto the ledger: removals, builds with their bindings, supply."""
        removed = list(result.disbanded) + list(result.civil_disorder)
        for unit in removed:
            matches = self._ids_matching(unit, limbo=False)
            plain = [e for e in matches if e.special is None]
            # twins that carry masks are kept: the removal takes one with the fewest
            target = (
                min(plain, key=lambda e: (e.masks, e.uid))
                if plain
                else (matches[0] if matches else None)
            )
            if target is not None:
                self._kill_entry(target, "is removed in the winter")
        prefer: dict[int, str] = {}
        for position, unit in enumerate(result.built):
            bound_to = None
            if position < len(self._pending_bindings):
                stashed_unit, bound_to = self._pending_bindings[position]
                assert stashed_unit == unit
            entry = self.ledger.create(unit, bound_to=bound_to)
            if unit.power == MEN:
                # Men build at any owned centre and take the point of the one
                # they are built in when it is free
                prefer[entry.uid] = unit.location.province
            # a winter build "arrives" for the wrath after the year's last
            # Diplomacy phase (the rulebook: a unit built in a coastal
            # province is treated as if it had moved in after the Autumn),
            # so the Fall ordinal anchors its timer
            self._arm_ulmo(entry, phase.year, Season.FALL)
        self._pending_bindings = []
        # bind this winter's builds at once, so a center spent on a plain
        # army is already unavailable to a double next year, and move the
        # units whose centre was lost, the disbands having freed points
        self._reconcile_spent(result.new_state, self._winter_ownership, prefer)

    def adjustment_has_decisions(self, state: State, ownership: Mapping[str, str]) -> bool | None:
        """Whether some power faces a winter choice under the variant rules."""
        return winter_decisions(self, state, ownership)

    def adjustment_has_effects(self, state: State, ownership: Mapping[str, str]) -> bool | None:
        """Whether an empty-sheet winter changes anything under the variant rules."""
        return winter_effects(self, state, ownership)

    def legal_adjustment_orders(
        self, power: str, state: State, ownership: Mapping[str, str]
    ) -> tuple[AnyOrder, ...] | None:
        """Return every individually legal winter order for ``power``."""
        return legal_winter_orders(self, power, state, ownership)

    def build_quota(self, power: str, state: State, ownership: Mapping[str, str]) -> int | None:
        """Return ``power``'s winter quota: owned weight minus its non-special units."""
        return gross_quota(self, power, state, ownership)

    # -- season boundaries and victory -----------------------------------------------------

    def autumn_ownership(
        self, graph: MapGraph, state: State, ownership: Mapping[str, str]
    ) -> Mapping[str, str]:
        """Return the Autumn ownership update, with the variant's capture gates.

        Area control decides who dominates a centre (the rulebook's "Control
        of an Area containing a Supply Centre"); a devastated centre changes
        hands only to Morgoth or to a mask carrier's power; an elven power
        that did not hold the centre since Spring owns it but waits a year
        for its supply. Captures and new supply delays are stashed for the
        boundary hook.
        """
        updated = dict(ownership)
        self._captures = {}
        provinces = {unit.location.province for unit in state.units}
        for province in sorted(provinces):
            if not graph.province(province).is_supply_center:
                continue
            counts: dict[str, int] = {}
            for unit in state.units:
                if unit.location.province == province:
                    counts[unit.power] = counts.get(unit.power, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if not (len(ranked) == 1 or ranked[0][1] >= ranked[1][1] + 2):
                continue
            dominant = ranked[0][0]
            old = updated.get(province)
            if old == dominant:
                continue
            if province in self.devastated:
                if dominant != MORGOTH and not self._masked_presence(dominant, province):
                    continue
            elif dominant in ELVISH and not self._spring_held(dominant, province):
                self.delayed_supply[province] = self.autumn_count + 2
            updated[province] = dominant
            self._captures[province] = (old if old is not None else "", dominant)
            self.owner_since[province] = self.autumn_count + 1
            # a captured center stops supplying the old owner's armies, which
            # move to another centre of theirs at the boundary, and arrives
            # unspent for the new owner
        return updated

    def _masked_presence(self, power: str, province: str) -> bool:
        """Whether ``power`` has a mask-carrying unit standing in ``province``."""
        return any(
            entry.power == power and entry.masks > 0 and not entry.limbo
            for entry in self.ledger.alive()
            if entry.location.province == province
        )

    def _spring_held(self, power: str, province: str) -> bool:
        """Whether ``power`` held ``province`` since the Spring boundary.

        The elven two-step's first step: in capacity-1 provinces the very
        unit that stood there in Spring must still stand there
        (ledger-checked; the rulebook: *"the Unit that entered it in Spring
        has to stay there also for Autumn"*).
        """
        snapshot = self.spring_control.get(province)
        if snapshot is None or snapshot[0] != power:
            return False
        if self.graph.capacity(province) != 1:
            return True
        uid = snapshot[1]
        if uid is None or uid not in self.ledger.units:
            return False
        entry = self.ledger.units[uid]
        return not entry.limbo and entry.location.province == province

    def on_season_end(
        self, phase: Phase, state: State, ownership: Mapping[str, str]
    ) -> tuple[State, Mapping[str, str], bool]:
        """Run the season-boundary bookkeeping and the three game-ending checks.

        Spring: snapshot the control for the elven two-step, then intervene
        if a fleet sailed. Autumn (ownership already updated by the
        wrapper): disband doubles whose centre was captured, tick the
        devastation counters, charge the supply, compute double eligibility,
        run the migration schedule and Glaurung's arrival, then the win check
        and the year-0 countdown.
        """
        if phase.season is Season.SPRING:
            self._snapshot_spring(state)
            if self.sail_west_pending:
                self.sail_west_pending = False
                self._intervene(ownership)
                return state, ownership, True
            return state, ownership, False
        # Autumn boundary: ownership was updated by the wrapper just before.
        self.autumn_count += 1
        state = self._double_disbands(state, ownership)
        self._devastation_ticks(ownership)
        self._reconcile_spent(state, ownership)
        self._compute_double_eligibility(ownership)
        state = self._migration_and_arrivals(phase, state)
        self.entered_this_year = {}
        self.spring_control = {}
        over = self._autumn_victory(phase, ownership)
        return state, ownership, over

    def _snapshot_spring(self, state: State) -> None:
        """Record who controls each centre at the Spring boundary, and by which unit."""
        self.spring_control = {}
        provinces = {u.location.province for u in state.units}
        for province in sorted(provinces):
            if not self.graph.province(province).is_supply_center:
                continue
            counts: dict[str, int] = {}
            for unit in state.units:
                if unit.location.province == province:
                    counts[unit.power] = counts.get(unit.power, 0) + 1
            ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            if not (len(ranked) == 1 or ranked[0][1] >= ranked[1][1] + 2):
                continue
            power = ranked[0][0]
            uid: int | None = None
            if self.graph.capacity(province) == 1:
                occupant = state.occupants(province)[0]
                entries = self._ids_matching(occupant, limbo=False)
                uid = entries[0].uid if entries else None
            self.spring_control[province] = (power, uid)

    def _double_disbands(self, state: State, ownership: Mapping[str, str]) -> State:
        """Disband every double bound to a centre that changed hands this Autumn."""
        units = list(state.units)
        changed = False
        for province, (_, new_owner) in sorted(self._captures.items()):
            for entry in list(self.ledger.alive()):
                if entry.bound_to == province and entry.power != new_owner:
                    unit = Unit(entry.power, entry.kind, entry.location, entry.strength)
                    if unit in units:
                        units.remove(unit)
                        changed = True
                    # the binding names this very double, so no twin trades its
                    # centre with it (:meth:`_release_older_centre`); a special
                    # is never bound, so this death is never Morgoth's defeat
                    self.ledger.kill(entry.uid)
                    self._emit(
                        f"{entry.power} {entry.spec()} disbands"
                        f" ({province} was captured by {new_owner})"
                    )
        self._captures = {}
        if changed:
            return State.of(state.map_name, state.phase, units)
        return state

    def _devastation_ticks(self, ownership: Mapping[str, str]) -> None:
        """Advance the four-year conversion counters of devastated Morgoth-owned centres.

        The rulebook: a province devastated and owned for four consecutive
        years becomes a Morgoth home centre for good; losing either resets
        the count.
        """
        for province in sorted(self.devastated):
            if not self.graph.province(province).is_supply_center:
                continue
            if province in self.converted_homes:
                continue
            if ownership.get(province) == MORGOTH:
                self.devastation_years[province] = self.devastation_years.get(province, 0) + 1
                if self.devastation_years[province] >= 4:
                    self.converted_homes.add(province)
                    self._emit(
                        f"{province} becomes a Morgoth home center"
                        " (devastated and owned four years)"
                    )
            else:
                self.devastation_years[province] = 0

    def _compute_double_eligibility(self, ownership: Mapping[str, str]) -> None:
        """Compute the centres that may bind a double next winter, by free point count.

        A centre qualifies when it was owned since the previous Autumn boundary
        at the latest and entered by no other power this year, and offers as many
        free supply points as remain -- a centre that already supplies standing
        units or doubles has spent itself on them, one point per unit (see
        ``_reconcile_spent``). A two-Centre province (Talath Dimen,
        Taur-im-Duinath, South Ossiriand) can offer up to two points; every
        weight-1 province offers at most one.
        """
        self.double_eligible = {}
        for province, owner in ownership.items():
            if not self.graph.province(province).is_supply_center:
                continue
            if self.owner_since.get(province, 0) > self.autumn_count - 1:
                continue
            entrants = self.entered_this_year.get(province, set())
            if entrants - {owner}:
                continue
            points = min(self._free_supply(province), self._weight(province))
            if points <= 0:
                continue
            self.double_eligible[province] = points

    def _free_supply(self, province: str) -> int:
        """Supply points of ``province`` not yet spent on a standing unit."""
        weight = self.graph.province(province).supply_center_count
        return weight - self.spent_supply.get(province, 0)

    def _reconcile_spent(
        self,
        state: State,
        ownership: Mapping[str, str],
        prefer: Mapping[int, str] | None = None,
    ) -> None:
        """Bind every standing unit to a point of a supply centre its power owns.

        The rulebook's "Double Units" section: *"Every Supply Centre supplies one
        Unit at a time"*, and *"when the Unit a Supply Centre supplies is gone, the
        Supply Centre may supply another"*. So each identity remembers the centre
        that supplies it (``LedgerUnit.supplied_by``): a double the centre it is
        bound to, a starting unit the centre it starts on, and a single unit
        that has none -- a new build, or one whose centre was lost -- the
        newest centre its power owns with a point to spare, ties by map code,
        so that the centres waiting out their year for a double keep their
        wait. ``prefer`` names, by uid, a centre to try first: the one a Men
        unit was built in. Units that no centre can take stand unattributed
        until a point frees, and the specials need no supply at all. The
        points spent per centre (``spent_supply``) are counted from the
        bindings, so a centre is free for a double exactly when no living unit
        holds its point.

        Called at the start, at every Autumn boundary and after every winter.
        """
        del state  # the ledger is synced to the board; the state is kept for the callers
        owned: dict[str, list[str]] = {}
        for province, owner in ownership.items():
            if self.graph.province(province).is_supply_center:
                owned.setdefault(owner, []).append(province)
        used: Counter[str] = Counter()
        waiting: list[LedgerUnit] = []
        for entry in self.ledger.alive():
            if entry.special is not None:
                entry.supplied_by = None
                continue
            centre = entry.bound_to if entry.bound_to is not None else entry.supplied_by
            if (
                centre is not None
                and ownership.get(centre) == entry.power
                and used[centre] < self._weight(centre)
            ):
                entry.supplied_by = centre
                used[centre] += 1
            else:
                entry.supplied_by = None
                waiting.append(entry)
        for entry in waiting:
            if entry.bound_to is not None:
                continue  # a double off its own centre is the capture rule's business
            wish = (prefer or {}).get(entry.uid)
            centre = self._newest_free_centre(owned.get(entry.power, ()), used, wish)
            if centre is not None:
                entry.supplied_by = centre
                used[centre] += 1
        self.spent_supply = {province: count for province, count in sorted(used.items()) if count}

    def _newest_free_centre(
        self, centres: Sequence[str], used: Counter[str], wish: str | None
    ) -> str | None:
        """Return ``wish`` if it has a point to spare, else the newest centre that has one."""
        free = [p for p in centres if used[p] < self._weight(p)]
        if wish is not None and wish in free:
            return wish
        if not free:
            return None
        return min(free, key=lambda p: (-self.owner_since.get(p, 0), p))

    def _release_older_centre(self, entry: LedgerUnit) -> None:
        """Before ``entry`` dies, let it carry the oldest centre among its identical twins.

        Twins cannot be told apart, so which of them a death removes is the
        ledger's choice -- unless they carry different numbers of Dwarven
        masks, which a player can see: such twins trade no centre, and each
        keeps its own. Among twins with equally many masks the centre it
        frees is the one held longest (the rulebook's "Double Units": *"of
        two twins, the point of the Supply Centre owned longer"*), so that a
        centre waiting for a double is the one set free. A double trades with
        a twin double only, and the two trade their bindings with the point,
        so the double that dies is the one bound to the centre held longest.
        A double disbanded because its own centre was captured is no such
        choice: :meth:`_double_disbands` kills it without this trade.

        The twins of an identity are the other identities of its
        :attr:`LedgerUnit.group` -- the same power, kind, base strength,
        province and coast -- where the location of a dislodged identity
        (``limbo``) is the place it stood when it was dislodged. A dislodged
        identity that dies therefore trades with every twin of its class,
        the twin that still stands on the board included; an identity that
        dies on the board trades with the twins on the board only.
        """
        if entry.special is not None or entry.supplied_by is None:
            return
        double = entry.bound_to is not None
        twins = [
            other
            for other in self.ledger.alive()
            if other.uid != entry.uid
            and other.special is None
            and (other.bound_to is not None) is double
            and other.supplied_by is not None
            and other.group == entry.group
            and other.masks == entry.masks
            and (entry.limbo or not other.limbo)
        ]
        if not twins:
            return
        oldest = min(
            twins, key=lambda e: (self.owner_since.get(e.supplied_by or "", 0), e.supplied_by)
        )
        here = (self.owner_since.get(entry.supplied_by, 0), entry.supplied_by)
        there = (self.owner_since.get(oldest.supplied_by or "", 0), oldest.supplied_by)
        if there < here:
            entry.supplied_by, oldest.supplied_by = oldest.supplied_by, entry.supplied_by
            if double:
                entry.bound_to, oldest.bound_to = oldest.bound_to, entry.bound_to

    def _weight(self, province: str) -> int:
        return self.graph.province(province).supply_center_count

    def _migration_and_arrivals(self, phase: Phase, state: State) -> State:
        """Apply the year's timed arrivals after Autumn: migration and Glaurung.

        The rulebook's "Migration" and "Eriador" sections: after Autumn 25
        and again after Autumn 20 Men gain two off-board centres and two
        armies in Eriador; after each Autumn from 10 down to 7 they lose one
        of those centres. "Glaurung Unit and Devastation": Glaurung appears
        in Angband after Autumn 21, available from Spring 20.
        """
        year = phase.year
        units = list(state.units)
        changed = False
        if year in (25, 20):
            self.eriador_centers += 2
            for _ in range(2):
                unit = Unit(MEN, UnitKind.ARMY, Location(ERIADOR))
                units.append(unit)
                self.ledger.create(unit)
            changed = True
            self._emit("MEN receive 2 off-board supply centers and 2 armies in Eriador (migration)")
        if year in (10, 9, 8, 7) and self.eriador_centers > 0:
            self.eriador_centers -= 1
            self._emit("Eriador loses one off-board supply center (migration)")
        if year == 21 and not self.glaurung_arrived:
            self.glaurung_arrived = True
            unit = Unit(MORGOTH, UnitKind.ARMY, Location(ANGBAND), 3)
            units.append(unit)
            self.ledger.create(unit, special=GLAURUNG)
            changed = True
            self._emit("Glaurung arrives in Angband (available Spring 20)")
        if changed:
            return State.of(state.map_name, state.phase, units)
        return state

    def _autumn_victory(self, phase: Phase, ownership: Mapping[str, str]) -> bool:
        """Check the Autumn boundary's three endings, in this order and no other.

        A sailing ends the game first; then Morgoth's win; then the year-0
        countdown. The win check stays AHEAD of the countdown, which is what
        gives a broken Oath its sharpest tooth: a Morgoth standing on exactly
        forty in the final Autumn who has broken one Oath fails the raised bar
        here and falls straight through to the Valar, so breaking can lose him
        the game outright -- and an intervention places nobody second, so the
        Power he betrayed gets nothing either. Reordering these two would
        quietly delete that consequence.
        """
        if self.sail_west_pending:
            self.sail_west_pending = False
            self._intervene(ownership)
            return True
        if not self.morgoth_defeated and not self.sail_west_ever:
            weight = center_weight(self, MORGOTH, ownership)
            bar = self.victory_bar
            if weight >= bar:
                scores = self._scores(ownership)
                self.result = GameResult(
                    winners=(MORGOTH,),
                    reason=f"Morgoth owns {weight} supply centers (victory at {bar})",
                    scores=scores,
                    # "If an Oath stands when the game ends, the Power named by
                    # it gets second place" -- bound here, at the win, not by a
                    # later act: the game is over and nothing can break it now.
                    second=self.oath,
                )
                self._emit(self.result.reason)
                if self.oath is not None:
                    self._emit(
                        f"the Oath of the Second sworn to {self.oath} in Year "
                        f"{self.oath_year} binds -- {self.oath} takes second place"
                    )
                return True
        if phase.year == FINAL_YEAR:
            self._intervene(ownership)
            return True
        return False

    def _intervene(self, ownership: Mapping[str, str]) -> None:
        """End the game by the Valar's intervention and rank everyone but Morgoth."""
        # Every Valar intervention defeats Morgoth (rulebook "Defeat": "Defeat
        # for Morgoth occurs if Morgoth 3A(M) is eliminated or if the Valar
        # intervene") -- both the mannish-fleet sail and the plain year-0
        # countdown, not only the sail. The flag gates only in-game movement
        # and adjustment steps, none of which run once the game is over, so
        # this is snapshot correctness with no play effect; the moot civil
        # disorder line is suppressed (game over -- no phases follow).
        why = "the mannish fleet reached Valinor" if self.sail_west_ever else "the Valar intervene"
        self._mark_morgoth_defeated(why, civil_disorder=False)
        scores = self._scores(ownership)
        winners = self._victors(scores, ownership)
        self.result = GameResult(
            winners=winners,
            reason=(
                "the Valar intervene; victory by centers gained over the home "
                "endowment, ties broken by total centers owned"
            ),
            scores=scores,
        )
        self._emit(
            "The Valar intervene -- the game ends"
            + (f"; {', '.join(winners)} win(s)" if winners else "")
        )

    def _victors(self, scores: Mapping[str, int], ownership: Mapping[str, str]) -> tuple[str, ...]:
        """Return the intervention's winners: greatest net gain, then greatest weight.

        The rulebook's "Victory" section: the greatest net gain over the home
        endowment wins; a tie there is broken by the greatest absolute owned
        supply weight (``center_weight`` -- Men's off-board counter included,
        per *"Men can count their off board Supply Centres"*); *"Powers equal
        on both counts share the Victory"*. Morgoth is excluded by his own
        clause.
        """
        contenders = [power for power in scores if power != MORGOTH]
        if not contenders:
            return ()
        best_net = max(scores[power] for power in contenders)
        leaders = [power for power in contenders if scores[power] == best_net]
        best_weight = max(center_weight(self, power, ownership) for power in leaders)
        return tuple(sorted(p for p in leaders if center_weight(self, p, ownership) == best_weight))

    def _scores(self, ownership: Mapping[str, str]) -> dict[str, int]:
        """Return each power's net gain: owned weight minus its home endowment."""
        scores: dict[str, int] = {}
        for power in sorted(self.graph.home_centers):
            home = sum(
                self.graph.province(center).supply_center_count
                for center in self.graph.home_centers[power]
            )
            scores[power] = center_weight(self, power, ownership) - home
        return scores

    @property
    def decides_victory(self) -> bool:
        """Beleriand decides its own game end; the classic centre count is off."""
        return True

    def game_result(self) -> GameResult | None:
        """Return the standing game result, or None while the game runs."""
        return self.result

    def validate_second(self, power: str) -> str | None:
        """Return why ``power`` may not be designated second, or None.

        Second place is Morgoth's alone to bestow, and only on a Morgoth
        victory (rulebook "Victory": *"If Morgoth wins, he may choose one
        Player to get second place. The rest of the Players are
        unplaced."*). Refused unless the standing result is a solo Morgoth
        win, refused for Morgoth himself, and refused once already given --
        naming any other power (surviving or eliminated) is fine. The
        ``Game`` facade has already checked the game is over and the power
        exists.

        One refusal more: an Oath standing at the win has already placed
        second (rulebook: *"If Morgoth wins and an Oath stands when the game
        ends, the Power named by it gets second place"*), and with the game
        over no way to break it remains -- the placement is final and this
        channel reports why by name. With no Oath standing the choice is
        untouched: the Oath is an option Morgoth takes, never a constraint he
        is born under.
        """
        result = self.result
        if result is None or result.winners != (MORGOTH,):
            return "second place is only designated after a Morgoth win"
        if power == MORGOTH:
            return "Morgoth cannot be designated his own second"
        if self.oath is not None:
            return f"second place is bound by the Oath sworn to {self.oath}"
        if result.second is not None:
            return f"second place is already {result.second}"
        return None

    def designate_second(self, power: str) -> None:
        """Record ``power`` as second; the ``Game`` facade validates first."""
        result = self.result
        if result is None:  # pragma: no cover - the Game facade validates first
            raise RuntimeError("second place designated with no game result")
        self.result = replace(result, second=power)

    # -- the Oath of the Second (rulebook "Morgoth", "Victory") ----------------------------

    @property
    def victory_bar(self) -> int:
        """The weighted supply-center count Morgoth must own to win *now*.

        Forty, then *"forty-one after the first, forty-two after the second,
        and so on"* (rulebook). Cumulative and permanent, never escalating:
        at this altitude a flat centre is already a heavy price, and an
        escalating one would make a second break impossible rather than
        expensive, which is exactly the negotiating value the Oath exists
        for.
        """
        return VICTORY_WEIGHT + self.oaths_broken

    def validate_swear(self, power: str, pending: Collection[str]) -> str | None:
        """Return why Morgoth may not swear an Oath naming ``power`` now, or None.

        Exactly three refusals, and no more. Morgoth may not name himself; he
        *"may stand under only one Oath at a time"*, so swapping is
        break-then-swear at one price and visibly; and both verbs are refused
        once he has placed his orders for the phase -- the book's own gate:
        *"An Oath is sworn or broken during a Phase, before Morgoth has placed
        his Orders for that Phase, and takes effect at once."* An Oath
        announced simultaneously with orders would reach a victim already
        committed and unable to react, leaving the price as the only
        protection.

        Naming an eliminated power is LEGAL: *"An Oath binds whether or not the
        Power named by it still has Units on the map"*.
        """
        if power == MORGOTH:
            return "Morgoth cannot swear the Oath of the Second to himself"
        if self.oath is not None:
            return f"Morgoth already stands under an Oath to {self.oath}; it must be broken first"
        if MORGOTH in pending:
            return "an Oath is sworn before Morgoth places his orders for the season"
        return None

    def apply_swear(self, power: str, year: int) -> None:
        """Swear the Oath to ``power`` in ``year``; the free-break exemption resets."""
        self.oath = power
        self.oath_year = year
        self.oath_stabbed = False  # the exemption is per-Oath
        self._emit(f"Morgoth swears the Oath of the Second to {power} (Year {year})")

    def validate_break(self, pending: Collection[str]) -> str | None:
        """Return why Morgoth may not break the Oath now, or None.

        Nothing to break, or his orders for the phase are already placed --
        the book gates both verbs in one sentence (quoted in
        :meth:`validate_swear`).
        """
        if self.oath is None:
            return "Morgoth stands under no Oath of the Second"
        if MORGOTH in pending:
            return "an Oath is broken before Morgoth places his orders for the season"
        return None

    def apply_break(self, year: int) -> None:
        """Break the standing Oath and charge its price, unless the sworn power attacked.

        The rulebook: *"An Oath may be broken at no price if the Power named
        by it has Attacked Morgoth since the Oath was sworn, i. e. has
        Ordered a Move into a Province which is not an Area and in which a
        Unit of Morgoth stands, or a Dislodge of a Unit of Morgoth, or has
        Supported such an Order - whether or not the Order succeeded."*
        :meth:`_mark_oath_stab` is where that is read.

        The flag is per-Oath and cleared by the next swearing, so it is left
        standing here: with no Oath to break it can grant nothing.
        """
        sworn = self.oath
        free = self.oath_stabbed
        self.oath = None
        self.oath_year = None
        if not free:
            self.oaths_broken += 1
        note = (
            f"at no price ({sworn} had attacked him)"
            if free
            else f"victory now needs {self.victory_bar} supply centers"
        )
        self._emit(f"Morgoth breaks the Oath of the Second to {sworn} (Year {year}) -- {note}")

    def oath_state(self) -> OathState:
        """Return the Oath surface as one value for the game and the snapshot."""
        return OathState(
            power=self.oath,
            year=self.oath_year,
            broken=self.oaths_broken,
            stabbed=self.oath_stabbed,
        )

    def validate_oath_state(self, state: OathState) -> str | None:
        """Return why a loaded Oath state is refused, or None.

        The loader's door (see :meth:`entente.engine.game.Game.restore_oath_state`):
        Morgoth may never be the sworn power, and the facade has already
        checked the power exists and the count is sane.
        """
        if state.power == MORGOTH:
            return "Morgoth cannot swear the Oath of the Second to himself"
        return None

    def apply_oath_state(self, state: OathState) -> None:
        """Restore the Oath surface from a loaded save."""
        self.oath = state.power
        self.oath_year = state.year
        self.oaths_broken = state.broken
        self.oath_stabbed = state.stabbed

    def _mark_oath_stab(self, records: Sequence[OrderRecord], pre_state: State) -> None:
        """Record whether the sworn power wrote an attack on Morgoth this phase.

        The rulebook's "Morgoth" section, "Victory":

            *"An Oath may be broken at no price if the Power named by it has
            Attacked Morgoth since the Oath was sworn, i. e. has Ordered a Move
            into a Province which is not an Area and in which a Unit of Morgoth
            stands, or a Dislodge of a Unit of Morgoth, or has Supported such an
            Order - whether or not the Order succeeded."*

        So a Move stabs only where the destination's capacity is 1 -- the same
        implicit-dislodge test :meth:`_girdle_attack_grants` applies, and for
        the same reason: entering an Area beside a Morgoth army is coexistence,
        not an attack (BATC-L2.V.12).

        Two differences from the Girdle grant are intended. Supports count,
        because the rulebook says *"or has Supported such an Order"*. And
        outcome is irrelevant on **both** branches -- *"whether or not the
        Order succeeded"* -- so nothing here consults resolutions, and no
        :func:`remains`-style filter asks whether the victim was still there
        afterwards; a bounced Move and a failed Dislodge were both attacks.

        Occupancy is read from the pre-movement board, which is the position
        the sworn power was looking at when it wrote the order. Angband is the
        only non-Area province the map leaves without capacity 1 (it is
        unbounded), and it admits no foreign unit at all, so no valid Move ever
        lands there to tell the literal capacity test apart from the rulebook's
        "is not an Area".
        """
        if self.oath is None or self.oath_stabbed:
            return
        morgoth_held = {unit.location.province for unit in pre_state.units if unit.power == MORGOTH}
        for record in records:
            if record.power != self.oath or not record.valid or record.written is None:
                continue
            order = record.effective
            if isinstance(order, Support):
                order = order.subject
            if isinstance(order, Move):
                target = order.destination.province
                if self.graph.capacity(target) != 1:
                    continue
            elif isinstance(order, Dislodge):
                target = order.victim.location.province
            else:
                continue
            if target in morgoth_held:
                self.oath_stabbed = True
                self._emit(
                    f"{self.oath} attacks Morgoth in {target} -- "
                    "the Oath of the Second may now be broken at no price"
                )
                return

    # -- enumeration --------------------------------------------------------------------

    def legal_extra_movement_orders(self, power: str, state: State) -> tuple[GameOrder, ...]:
        """Return every Layer-2 order ``power`` may write this phase.

        Glaurung's Devastate writes his designator, as every order of a
        special must; no other Layer-2 order can name a special -- Morgoth
        never uses the fleet knowledge, so his units never transform, and he
        neither makes nor passes masks -- so the rest of the list stays
        value-level, one row per unit value.
        """
        orders: list[GameOrder] = []
        fleet_ok = usable_fleet_knowledge(self, power)
        for unit in state.units:
            if unit.power != power:
                continue
            province = unit.location.province
            strength = unit.strength if unit.strength != 1 else None
            if fleet_ok and self.graph.terrain(province) is Terrain.COASTAL:
                target = UnitKind.FLEET if unit.kind is UnitKind.ARMY else UnitKind.ARMY
                orders.append(Transform(unit.kind, unit.location, target, strength))
            if province in DWARF_CITIES and power == NAUGRIM:
                orders.append(MakeMasks(unit.kind, unit.location, strength))
            if power == MEN and unit.kind is UnitKind.FLEET and province in WEST_SEAS:
                orders.append(SailWest(unit.kind, unit.location, strength))
        orders.extend(self._transfer_orders(power, state))
        glaurung = self.ledger.special(GLAURUNG)
        if power == MORGOTH and glaurung is not None and not glaurung.limbo:
            orders.append(
                Devastate(UnitKind.ARMY, glaurung.location, 3, SPECIAL_DESIGNATOR[GLAURUNG])
            )
        return tuple(orders)

    def _anticipated_ends(self, kind: UnitKind, province: str) -> set[str]:
        """Return the provinces a unit of ``kind`` in ``province`` could end the phase in.

        Its current province plus every province one relocating order could
        carry it to -- per-kind movement adjacency, which also covers the
        variant's dislodger-advance (a successful Dislodge lands the mover in
        the adjacent victim's province, always a movement neighbour).

        Convoy landings are NOT anticipated: they depend on the phase's
        actual convoying fleets, so deriving them here would duplicate the
        adjudicator's convoy-path logic -- and the hooks stay a *consumer* of
        the frozen calculus, never a reimplementation. The pass RESOLUTION
        still handles a convoyed giver (a convoy move lands the ledger like
        any move), so a directly-built convoy-then-pass resolves
        (BATC-L2.M.20); it is simply absent from the offered mask list.
        """
        ends = {province}
        if kind is UnitKind.ARMY:
            ends |= set(self.graph.army_adjacency.get(province, frozenset()))
        else:
            for location in self.graph.fleet_locations(province):
                for reachable in self.graph.fleet_adjacency.get(location, frozenset()):
                    ends.add(reachable.province)
        return ends

    def _transfer_orders(self, power: str, state: State) -> list[GameOrder]:
        """Enumerate the mask directives over both counterparts' anticipated end positions.

        The rulebook's "Dwarven Masks" section: *"a Unit may Move and then
        pass a mask"*, and its own examples name the counterpart at its
        post-move spot -- ``A(NOs)->NEB (gives mask to Doriath A(FoR))`` /
        ``A(FSi)->FoR (accepts the dwarven mask from Ossiriand A(NEB))``.

        The RESOLUTION reads the units' post-move positions
        (:meth:`_refs_cross_match` / :meth:`_mask_geometry_ok`), so a directive's
        COUNTERPART ref names that unit at any province it could reach this phase
        (:meth:`_anticipated_ends`). The giver's/acceptor's OWN spec stays at its
        current sheet location, because :meth:`_bind_deferred_mains` binds a
        directive to the record occupying that slot (main-order-and-directive
        ride the same unit) -- an own end would never bind.

        Both sides may move at once: the rule restricts neither counterpart.
        So the pairing is the full cross product, geometry-filtered: the give
        side names a receiver R at every ``r in ends(R)`` for which SOME giver
        end ``g`` clears the mask geometry; the accept side names a giver at
        every ``g in ends(giver)`` for which SOME acceptor end clears it.
        Geometry is standardized on the resolution direction
        (:meth:`_mask_geometry_provinces_ok`, giver->acceptor), because with the
        map's one-way edges the reversed test can disagree. The single-moving-
        side pairs and the stationary both-current pairs are a strict subset.
        The list can be long; a client filters it by the drafted end position,
        so its length costs the enumeration nothing. Geometry is re-checked on
        the ACTUAL end positions at resolution (bounce semantics): a directive
        is offered on the possibility of reaching the pairing; if the mover
        bounces, the pass FAILS on the true ledger positions
        (BATC-L2.M.17/M.22). Deduped by canonical text.

        Convoy landings are NOT anticipated (fleet-position dependent;
        :meth:`_anticipated_ends`), so a convoy-then-pass resolves by hand yet
        is absent here (BATC-L2.M.20) -- a known scope bound.

        Morgoth never acquires masks, so neither directive is offered when he
        is the ordering power, and no GiveMask row ever names a MORGOTH-owned
        unit as the receiver.
        """
        orders: list[GameOrder] = []
        if power == MORGOTH:
            return orders
        holders = [entry for entry in self.ledger.board() if entry.masks > 0]
        if not holders:
            return orders
        ends_cache: dict[tuple[UnitKind, str], set[str]] = {}

        def ends(kind: UnitKind, province: str) -> set[str]:
            key = (kind, province)
            if key not in ends_cache:
                ends_cache[key] = self._anticipated_ends(kind, province)
            return ends_cache[key]

        seen_gives: set[str] = set()
        for entry in holders:
            if entry.power != power:
                continue
            giver_ends = ends(entry.kind, entry.location.province)
            for unit in state.units:
                if unit.power == power and unit.location == entry.location:
                    continue
                if unit.power == MORGOTH:
                    continue
                # every receiver end reachable by SOME giver end (giver->acceptor)
                targets = {
                    r
                    for r in ends(unit.kind, unit.location.province)
                    if any(self._mask_geometry_provinces_ok(g, r) for g in giver_ends)
                }
                for target in sorted(targets):
                    order: GameOrder = GiveMask(
                        entry.kind,
                        entry.location,
                        UnitRef(
                            unit.power,
                            unit.kind,
                            Location(target),
                            unit.strength if unit.strength != 1 else None,
                        ),
                        entry.strength if entry.strength != 1 else None,
                    )
                    text = format_order(order)
                    if text not in seen_gives:
                        seen_gives.add(text)
                        orders.append(order)
        seen_accepts: set[str] = set()
        for unit in state.units:
            if unit.power != power:
                continue
            acceptor_ends = ends(unit.kind, unit.location.province)
            for entry in holders:
                if entry.power == power and entry.location == unit.location:
                    continue
                if entry.power == MORGOTH:  # holders exclude Morgoth; kept explicit
                    continue
                # every giver end that reaches SOME acceptor end (giver->acceptor)
                sources = {
                    g
                    for g in ends(entry.kind, entry.location.province)
                    if any(self._mask_geometry_provinces_ok(g, a) for a in acceptor_ends)
                }
                for source in sorted(sources):
                    order = AcceptMask(
                        unit.kind,
                        unit.location,
                        UnitRef(
                            entry.power,
                            entry.kind,
                            Location(source),
                            entry.strength if entry.strength != 1 else None,
                        ),
                        unit.strength if unit.strength != 1 else None,
                    )
                    text = format_order(order)
                    if text not in seen_accepts:
                        seen_accepts.add(text)
                        orders.append(order)
        return orders

    # -- serialization fragment --------------------------------------------------------------

    def snapshot(self) -> dict[str, object] | None:
        """Return the Layer-2 fragment a save carries and a reload verifies against.

        Everything is sorted so the fragment is canonical; the Oath keys are
        written only once someone has sworn.
        """
        masks: dict[str, int] = {}
        raw_bindings: dict[str, list[str]] = {}
        specials: dict[str, dict[str, object]] = {}
        for entry in self.ledger.alive():
            if entry.masks:
                key = f"{entry.power} {entry.spec()}"
                masks[key] = masks.get(key, 0) + entry.masks
            if entry.bound_to is not None:
                raw_bindings.setdefault(entry.bound_to, []).append(entry.power)
            if entry.special is not None:  # the Morgoth unit / Glaurung, keyed by tag
                specials[entry.special] = {
                    "province": entry.location.province,
                    "limbo": entry.limbo,
                }
        # A two-Centre province may carry two live doubles: the value is the
        # plain power string when exactly one binding holds the key (every
        # weight-1 province, and a two-Centre province with one double) and a
        # sorted list of power names only when more than one does, so the
        # fragment takes the list shape only where a second binding exists.
        bindings: dict[str, object] = {
            province: powers[0] if len(powers) == 1 else sorted(powers)
            for province, powers in raw_bindings.items()
        }
        fragment: dict[str, object] = {
            "fleet_knowledge": sorted(self.fleet_knowledge),
            "girdle": sorted(self.girdle),
            "devastated": sorted(self.devastated),
            "converted_homes": sorted(self.converted_homes),
            "devastation_years": dict(sorted(self.devastation_years.items())),
            "eriador_centers": self.eriador_centers,
            "ulmo_delays": dict(sorted(self.ulmo_delays.items())),
            "ulmo_armed": dict(sorted(self.ulmo_armed.items())),
            "masks": dict(sorted(masks.items())),
            "double_bindings": dict(sorted(bindings.items())),
            "specials": dict(sorted(specials.items())),
            "delayed_supply": dict(sorted(self.delayed_supply.items())),
            "spent_supply": dict(sorted(self.spent_supply.items())),
            "morgoth_defeated": self.morgoth_defeated,
            "second": self.result.second if self.result is not None else None,
            "gondolin_shield": self.gondolin_shield,
            "glaurung_arrived": self.glaurung_arrived,
            "sail_west": self.sail_west_ever,
            "autumn_count": self.autumn_count,
        }
        # The Oath rides the fragment beside "second" and "sail_west" --
        # verification-only, like its neighbours -- but only once someone has
        # sworn one. A game that never touched the channel produces the same
        # fragment as a game with no Oath surface at all, so its saves stay
        # byte-identical. Absent means "no Oath was ever involved", which is
        # exactly the default state.
        state = self.oath_state()
        if not state.is_default:
            fragment["oath"] = state.power
            fragment["oath_year"] = state.year
            fragment["oaths_broken"] = state.broken
            fragment["oath_stabbed"] = state.stabbed
        return fragment


def _factory(graph: MapGraph, seed: int | None) -> BeleriandHooks:
    """Build one game's hooks; the registry calls this per game."""
    return BeleriandHooks(graph, seed)


GAME_HOOKS.setdefault("beleriand", _factory)
