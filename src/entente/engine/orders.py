"""Order types and notation parser/formatter, all phases (docs/adjudication.tex Sec. 1, Sec. 9).

The movement registry: Hold, Move (optionally ``via convoy``), Support (whose
subject is a *whole order* -- hold, move, or a variant order), Convoy, and the
Beleriand variant orders Dislodge (targeted displacement) and Bar (targeted
entry denial). Which types a ruleset accepts is registry data (Sec. 1/Sec. 7); orders
outside the registry become Hold in Phase 0. The retreat phase (Sec. 9) reuses
Move and adds Disband; the adjustment phase uses Build, Disband and Waive.

Notation (canonical form emitted by :func:`format_order`; parsers are
case-insensitive and tolerant of an omitted trailing ``H``)::

    A PAR H                      hold
    A PAR - BUR                  move (movement or retreat phase)
    A LON - HOL via convoy       move, convoy route requested (Sec. 3.5, DATC 4.A.3)
    A SMY S F ANK                support-hold (subject: the unit stays)
    F CON S F BLA - ANK          support-move
    F NTH C A LON - HOL          convoy
    2A GEL - THA                 move by a strength-2 unit (Beleriand)
    3A(M) ANF - LOT              the Morgoth unit specifically (Beleriand designator)
    3A(G) ANF DEV                Glaurung specifically (Beleriand designator)
    A FOR DIS MORGOTH 3A(M) ANF  dislodge the *named* special (designated victim)
    A FOR DIS MORGOTH A NEB      dislodge the named victim (Beleriand, Sec. 4.2)
    A EST DIS MORGOTH A NEB S A GEL - THA
                                 dislodge, victim disambiguated by its order (Sec. 3.4)
    A NEB BAR DORIATH A FOR - NEB
                                 bar the named unit's move into this Area
    A EST S 3A NEB BAR DORIATH 2A FOR - NEB
                                 support a bar (subjects are whole orders)
    A EST S MORGOTH A NEB        support-hold, subject owner written (Areas may
                                 need the owner to disambiguate -- rulebook
                                 "Order specifications")
    Disband A PAR                retreat-phase disband / adjustment removal
    Build F STP/NC               adjustment build
    Build 2A NAR BOUND EGL       double-unit build, bound to its earning center
                                 (Beleriand Layer 2)
    Waive                        adjustment: decline one build (DATC 4.D.7)

Layer-2 movement orders (resolved by the game hooks, never by the
adjudicator; rulebook "Other abbreviations")::

    A NFA TFM F                  transform to fleet ("tfm to F")
    F SFA TFM A                  transform to army
    3A HIT DEV                   Glaurung devastates his province ("dev")
    A DOL MSK                    make Dwarven masks ("msk")
    A NOS GIVES MASK TO DORIATH A FOR
                                 the giver's transfer directive (end of phase)
    A FOR ACCEPTS MASK FROM OSSIRIAND A NOS
                                 the matching acceptance (both must be ordered)
    F NBE SAILS WEST             the mannish fleet leaves for Valinor

A unit spec may carry a strength prefix (``2A``, ``3A``, ``2F`` -- the
Beleriand rulebook's ``XA(Province)`` form): it is written only when the
strength is not 1 and is part of unit *identification* (Sec. 1: sheets bind
orders to units; in multi-unit Areas the prefix disambiguates). ``strength``
is ``None`` on orders written without a prefix.

A unit spec may additionally carry a **special-unit designator** immediately
after the strength+kind token: ``3A(M)`` (the Morgoth unit) or ``3A(G)``
(Glaurung), the rulebook's own notation for MORGOTH's two named ``3A``
specials (parsed case-insensitively; ``format_order`` emits canonical upper).
It rides ``special`` (``"M"``/``"G"``/``None``) on every unit-bearing order
and on :class:`UnitRef`. The rulebook requires it: a special is named only
with its designator, in its own orders and wherever another order names it,
and a ``3A`` written without one fits no unit -- so the Beleriand game layer
rejects an order that names a special bare, or names with a designator a
special that is not there, and every list the engine offers writes the
designators. Phase 0's validity rules and the strength arithmetic never
read ``special``; the adjudicator reads it as the unit's signature, so the
two specials are never twins, and its victim selection and reference
matching match a designator exactly both ways: a written one fits only a
unit whose order writes the same, an unwritten one only a unit whose order
writes none (``_ref_special_ok``). Plain notation never carries one, so the
conformance suites never meet it.

Two normalizations keep notation round-trips exact: a Support subject move is
stored with ``via_convoy=False`` (support matching ignores the route flag),
and a Convoy cargo is stored with ``via_convoy=True`` (a convoy carries a
convoyed move by definition).

Only *syntax* lives here. Semantic validity (terrain, adjacency, matching,
victim resolution -- spec Phase 0, "invalid orders become Hold"; Sec. 9
retreat/build legality) is decided by the adjudicator against a
:class:`MapGraph`; :func:`static_move_legal` is the per-order piece it
consumes.

Each phase has its own parser (:func:`parse_order`,
:func:`parse_retreat_order`, :func:`parse_adjustment_order`) so sheet types
stay narrow; :func:`format_order` covers the full union.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Union, assert_never

from entente.engine.map.types import Location, Terrain, UnitKind

if TYPE_CHECKING:
    from entente.engine.map.graph import MapGraph


class OrderSyntaxError(ValueError):
    """Raised for text that is not well-formed order notation.

    Distinct from semantic invalidity: unparseable text is an interface error
    and raises; a well-formed but illegal order becomes a Hold in Phase 0.
    """


@dataclass(frozen=True, slots=True)
class UnitRef:
    """A written reference to another power's unit (Dislodge/Bar victims).

    The rulebook requires the owner ("must have specified whose Unit"); kind
    and strength narrow the match among a power's units in an Area. Whether
    the reference *resolves* is semantic (Sec. 3.4: a mis-specified victim makes
    the attack fail), decided by the adjudicator. ``special`` is the
    ``(M)``/``(G)`` designator naming one of MORGOTH's specials as the victim
    -- required whenever the victim is one, since a bare ``3A`` names no
    special (the adjudicator's victim binding filters on it; the Beleriand
    game layer validates it).
    """

    power: str
    kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Hold:
    """``u H`` -- stationary (Sec. 1)."""

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Move:
    """``u - p`` (optionally ``via convoy``) -- relocating; contests entry at ``p`` (Sec. 1)."""

    unit_kind: UnitKind
    location: Location
    destination: Location
    via_convoy: bool = False
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Dislodge:
    """``u DIS <power> <victim>[ <victim's action>]`` -- relocating; a targeted displacement.

    Targeted displacement plus entry into the victim's Area (Sec. 1,
    Sec. 4.2). ``victim_order``, when written, names the victim's action and
    narrows the Sec. 3.4 selection to the unit that actually received that
    order.
    """

    unit_kind: UnitKind
    location: Location
    victim: UnitRef
    victim_order: Order | None = None
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Bar:
    """``u BAR <power> <victim> - p`` -- stationary; a targeted entry denial.

    Sec. 1, Sec. 4.2 "Bar contests". ``destination`` is the written target
    Area -- validly always the barring unit's own province (rulebook: "the
    Area in which the Unit ordered to Bar stands").
    """

    unit_kind: UnitKind
    location: Location
    victim: UnitRef
    destination: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Support:
    """``u S [<power> ]<subject order>`` -- stationary; adds strength to its subject.

    Sec. 1, Sec. 3. ``subject_power`` is the written owner of the subject unit, when
    disambiguation requires one (Areas).
    """

    unit_kind: UnitKind
    location: Location
    subject: SupportableOrder
    strength: int | None = None
    subject_power: str | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Convoy:
    """``u C <move>`` -- stationary; carries a convoyed move's path (Sec. 1, Sec. 3.5)."""

    unit_kind: UnitKind
    location: Location
    cargo: Move
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Disband:
    """``Disband u`` -- remove the unit.

    A retreat-phase alternative to retreating, or an adjustment-phase removal
    (Sec. 9).
    """

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Build:
    """``Build u [BOUND p]`` -- place a new unit at an owned home supply center (Sec. 9).

    ``bound_to`` is the Beleriand double-unit binding center: written when a
    double build names the center that earned it; None elsewhere (and for
    every classic build).
    """

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    bound_to: str | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Waive:
    """``Waive`` -- decline one build the power is entitled to (DATC 4.D.7 choice A)."""


@dataclass(frozen=True, slots=True)
class Transform:
    """``u TFM A|F`` -- Beleriand Layer 2: convert between army and fleet.

    Legal while standing in a coastal province (rulebook "Fleets").
    Stationary; never reaches the adjudicator -- the game hooks extract it and
    apply the conversion at end of phase unless the unit was attacked.
    """

    unit_kind: UnitKind
    location: Location
    target_kind: UnitKind
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class Devastate:
    """``u DEV`` -- Beleriand Layer 2: Glaurung devastates the province he stands in.

    The devastation is permanent (rulebook "Glaurung Unit and Devastation").
    Stationary end-of-phase step.
    """

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class MakeMasks:
    """``u MSK`` -- Beleriand Layer 2: make Dwarven masks in Belegost, Mount Dolmed or Nogrod.

    The rulebook's "Dwarven Masks" section; a stationary end-of-phase step.
    The order TEXT is power-blind like every other order shape; who may
    actually make masks is the variant's business -- only the Naugrim make
    masks, and any other power's MSK FAILS.
    """

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class GiveMask:
    """``u GIVES MASK TO <power> <unit>`` -- Beleriand Layer 2: the giver's half of a mask transfer.

    Executed after every main order at end of phase (rulebook "Dwarven
    Masks"). ``receiver`` names the accepting unit at its end-of-phase
    position (the rulebook's own examples reference post-move locations).
    """

    unit_kind: UnitKind
    location: Location
    receiver: UnitRef
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class AcceptMask:
    """``u ACCEPTS MASK FROM <power> <unit>`` -- the receiving half of a mask transfer.

    Both halves must be ordered (rulebook: "A Unit which is receiving a mask
    must order the reception").
    """

    unit_kind: UnitKind
    location: Location
    giver: UnitRef
    strength: int | None = None
    special: str | None = None


@dataclass(frozen=True, slots=True)
class SailWest:
    """``F <sea> SAILS WEST`` -- Beleriand Layer 2: a mannish fleet leaves for Valinor.

    Legal from North or South Belegaer; it schedules the Valar's intervention
    for the end of the season (rulebook "Valar's intervention").
    """

    unit_kind: UnitKind
    location: Location
    strength: int | None = None
    special: str | None = None


Order = Union[Hold, Move, Support, Convoy, Dislodge, Bar]  # noqa: UP007 - runtime union (3.11)
SupportableOrder = Union[Hold, Move, Dislodge, Bar]  # noqa: UP007
"""What a support's subject may be (Sec. 1 "subjects are whole orders", Sec. 3).

Every :data:`Order` member except Support and Convoy, which
:func:`_parse_subject` rejects at parse time. ``Dislodge.victim_order`` stays
the full :data:`Order` -- a written victim tail may name any order the victim
wrote.
"""
Layer2Order = Union[  # noqa: UP007
    Transform, Devastate, MakeMasks, GiveMask, AcceptMask, SailWest
]
"""Layer-2 movement-phase orders: resolved by the game hooks, never by the adjudicator.

They pass the phase gate for every ruleset, and every adjudicator-facing
union below excludes them. The axis this union encodes is who resolves the
order: Dislodge and Bar are *also* variant orders, yet they belong to the
adjudicator grammar.
"""
MovementOrder = Union[Order, Layer2Order]  # noqa: UP007
RetreatOrder = Union[Move, Disband]  # noqa: UP007
AdjustmentOrder = Union[Build, Disband, Waive]  # noqa: UP007
AnyOrder = Union[Order, Disband, Build, Waive]  # noqa: UP007
GameOrder = Union[AnyOrder, Layer2Order]  # noqa: UP007
"""The full submittable union -- what the ``Game`` facade accepts; the hooks
strip the Layer-2 members before anything reaches an adjudicator."""
RELOCATING_TYPES: tuple[type[Move | Dislodge], ...] = (Move, Dislodge)
"""Order types whose unit attacks/enters (Sec. 1 'effect class: relocating')."""
STATIONARY_TYPES: tuple[type[Hold | Support | Convoy | Bar], ...] = (Hold, Support, Convoy, Bar)
"""Order types whose unit stays put (Sec. 1 'effect class: stationary') -- the
complement of :data:`RELOCATING_TYPES` within the adjudicator's grammar, and
the set a support-hold may cover (Sec. 3, ``strength.support_matches``).

Spelled out rather than derived so both effect classes read the same way at
their definition site; the assertion below is what keeps them a partition."""
ADJUDICATION_ORDER_TYPES: tuple[type[Order], ...] = (Hold, Move, Support, Convoy, Dislodge, Bar)
"""Runtime tuple of :data:`Order` members -- the adjudicator's grammar. The
member type is written out so a positive ``isinstance`` against the tuple
narrows to :data:`Order` (mypy narrows the match branch through a variable
tuple; the else branch it narrows only through a literal one)."""
LAYER2_ORDER_TYPES: tuple[type[Layer2Order], ...] = (
    Transform,
    Devastate,
    MakeMasks,
    GiveMask,
    AcceptMask,
    SailWest,
)
"""Runtime tuple of :data:`Layer2Order` members (isinstance checks)."""

# Sec. 1's two effect classes partition the adjudicator's grammar, enforced at
# import because both predicates over them are *membership* tests, not
# exhaustive matches: a seventh order type nobody classified would make
# is_relocating and is_stationary BOTH answer False, and Sec. 3 support-hold
# matching would quietly stop covering it -- a wrong answer, not a crash. Fail
# the import instead, where the omission is unmissable.
if set(RELOCATING_TYPES) | set(STATIONARY_TYPES) != set(ADJUDICATION_ORDER_TYPES) or set(
    RELOCATING_TYPES
) & set(STATIONARY_TYPES):
    raise AssertionError(  # pragma: no cover - import-time partition guard
        "RELOCATING_TYPES and STATIONARY_TYPES must partition ADJUDICATION_ORDER_TYPES"
    )

_KIND_LETTERS = "".join(kind.value for kind in UnitKind)
"""The grammar's unit-kind letters, in :class:`UnitKind` declaration order.

Single source of truth for the kind group below: adding an enum member extends
the parser for free. (A registered kind table would replace the enum only once
a variant needs a third kind; until then the enum *is* the registry.)
"""

if any(len(kind.value) != 1 for kind in UnitKind):
    raise AssertionError(  # pragma: no cover - import-time grammar guard
        f"unit-kind letters must be single characters, got {_KIND_LETTERS!r}"
    )

_KIND_TOKEN = re.compile(rf"^([2-9])?([{re.escape(_KIND_LETTERS)}])(?:\(([MG])\))?$")
"""``[strength]LETTER[(designator)]``.

The ``(M)``/``(G)`` designator group is engine-wide rather than
variant-supplied: :func:`parse_order` takes no ruleset, so one canonical text
has one canonical parse everywhere, and ``Ruleset.order_registry`` -- not the
grammar -- is where a ruleset narrows what it accepts (the same superset
discipline Dislodge and Bar follow).
"""


def parse_location(text: str) -> Location:
    """Parse ``"STP/SC"`` into a :class:`Location`; an empty province is a syntax error."""
    province, _, coast = text.partition("/")
    if not province:
        raise OrderSyntaxError(f"empty location in {text!r}")
    return Location(province, coast or None)


def _match_kind(token: str) -> tuple[UnitKind, int | None, str | None] | None:
    """Return (kind, strength, designator) for a unit-kind token, or None for any other token.

    ``"2A"`` -> (ARMY, 2, None); ``"3A(M)"`` -> (ARMY, 3, "M"); ``"F"`` ->
    (FLEET, None, None). The ``(M)``/``(G)`` group is the Beleriand
    special-unit designator (case folded by the tokenizer).
    """
    match = _KIND_TOKEN.match(token)
    if match is None:
        return None
    strength = int(match.group(1)) if match.group(1) else None
    return UnitKind(match.group(2)), strength, match.group(3)


def _parse_kind(token: str, context: str) -> tuple[UnitKind, int | None, str | None]:
    """Parse a unit-kind token, or raise :class:`OrderSyntaxError` quoting ``context``."""
    parsed = _match_kind(token)
    if parsed is None:
        raise OrderSyntaxError(f"unknown unit kind {token!r} in {context!r}")
    return parsed


def parse_unit_spec(text: str) -> tuple[UnitKind, Location, int]:
    """Parse ``"F STP/SC"`` -> (kind, location, strength); ``"2A GEL"`` -> strength 2.

    The tuple unpacks positionally into :class:`entente.engine.state.Unit`
    after its power. An unwritten strength prefix means strength 1.
    """
    tokens = text.upper().split()
    if len(tokens) != 2:
        raise OrderSyntaxError(f"unit spec must be 'KIND LOCATION': {text!r}")
    kind, strength, special = _parse_kind(tokens[0], text)
    if special is not None:  # a board unit carries no designator (identity is the ledger's)
        raise OrderSyntaxError(f"a unit spec carries no special-unit designator: {text!r}")
    return kind, parse_location(tokens[1]), strength or 1


class _Tokens:
    """A one-token-lookahead cursor over an order's whitespace-separated words.

    Every phase parser in this module is a recursive-descent walk over one of
    these: :meth:`next` consumes, :meth:`peek` looks ahead without consuming
    (the ``via convoy`` and subject-owner branches need it), :meth:`done` and
    :meth:`expect_end` close the walk. Running out of tokens and leaving tokens
    over are both :class:`OrderSyntaxError`, so no caller has to check.

    Upper-casing happens once here, at construction -- this is where the module
    docstring's "parsers are case-insensitive" promise is actually kept, and
    why the designator group of :data:`_KIND_TOKEN` may match ``[MG]`` alone.
    ``text`` is retained verbatim for error messages, so a report quotes what
    the player wrote rather than the folded form.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = text.upper().split()
        self.pos = 0

    def next(self) -> str:
        if self.pos >= len(self.tokens):
            raise OrderSyntaxError(f"unexpected end of order: {self.text!r}")
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def done(self) -> bool:
        return self.pos >= len(self.tokens)

    def expect_end(self) -> None:
        if not self.done():
            raise OrderSyntaxError(f"trailing tokens in order: {self.text!r}")


def _parse_via_convoy(tokens: _Tokens) -> bool:
    if tokens.peek() == "VIA":
        tokens.next()
        if tokens.next() != "CONVOY":
            raise OrderSyntaxError(f"expected 'via convoy' in {tokens.text!r}")
        return True
    return False


def _parse_unit_ref(tokens: _Tokens) -> UnitRef:
    """Parse a victim reference: ``POWER [X]KIND LOCATION`` (owner mandatory)."""
    power = tokens.next()
    if _match_kind(power) is not None:
        raise OrderSyntaxError(f"victim reference must name its owner in {tokens.text!r}")
    kind, strength, special = _parse_kind(tokens.next(), tokens.text)
    location = parse_location(tokens.next())
    return UnitRef(power=power, kind=kind, location=location, strength=strength, special=special)


_LAYER2_ACTIONS = frozenset({"TFM", "DEV", "MSK", "GIVES", "ACCEPTS", "SAILS"})


def _parse_layer2_action(
    kind: UnitKind,
    location: Location,
    strength: int | None,
    special: str | None,
    action: str,
    tokens: _Tokens,
) -> Layer2Order:
    """Parse a Layer-2 action; the grammar is reachable only from top-level orders.

    A variant action can neither be a support's subject nor name a Dislodge
    victim's order (victims doing Layer-2 actions are named by
    owner/kind/strength/province instead). ``special`` is the ordering unit's
    optional designator (e.g. ``3A(G) ANF DEV``); it rides the resulting order
    for Layer 2 to consume.
    """
    if action == "TFM":
        target, target_strength, target_special = _parse_kind(tokens.next(), tokens.text)
        if target_strength is not None or target_special is not None:
            raise OrderSyntaxError(
                f"transform target carries no prefix or designator (it is kept): {tokens.text!r}"
            )
        return Transform(kind, location, target, strength, special)
    if action == "DEV":
        return Devastate(kind, location, strength, special)
    if action == "MSK":
        return MakeMasks(kind, location, strength, special)
    if action == "GIVES":
        if tokens.next() != "MASK" or tokens.next() != "TO":
            raise OrderSyntaxError(f"expected 'gives mask to' in {tokens.text!r}")
        return GiveMask(kind, location, _parse_unit_ref(tokens), strength, special)
    if action == "ACCEPTS":
        if tokens.next() != "MASK" or tokens.next() != "FROM":
            raise OrderSyntaxError(f"expected 'accepts mask from' in {tokens.text!r}")
        return AcceptMask(kind, location, _parse_unit_ref(tokens), strength, special)
    if tokens.next() != "WEST":  # action == "SAILS", the only remaining member
        raise OrderSyntaxError(f"expected 'sails west' in {tokens.text!r}")
    return SailWest(kind, location, strength, special)


def _parse_action(
    kind: UnitKind, location: Location, strength: int | None, special: str | None, tokens: _Tokens
) -> Order:
    """Parse the action grammar shared by top-level orders, support subjects and victim tails.

    An exhausted stream (or ``H``) is a Hold. ``special`` is the ordering
    unit's optional designator, threaded onto the order it produces (each
    nested unit -- convoy cargo, victim-order tail -- carries its own).
    """
    if tokens.done():
        return Hold(kind, location, strength, special)
    action = tokens.next()
    if action == "H":
        return Hold(kind, location, strength, special)
    if action == "-":
        destination = parse_location(tokens.next())
        return Move(kind, location, destination, _parse_via_convoy(tokens), strength, special)
    if action == "S":
        subject_power, subject = _parse_subject(tokens)
        return Support(kind, location, subject, strength, subject_power, special)
    if action == "C":
        cargo_kind, cargo_strength, cargo_special = _parse_kind(tokens.next(), tokens.text)
        cargo_location = parse_location(tokens.next())
        cargo = _parse_action(cargo_kind, cargo_location, cargo_strength, cargo_special, tokens)
        if not isinstance(cargo, Move):
            raise OrderSyntaxError(f"convoy requires a move as cargo: {tokens.text!r}")
        return Convoy(kind, location, replace(cargo, via_convoy=True), strength, special)
    if action == "DIS":
        victim = _parse_unit_ref(tokens)
        victim_order: Order | None = None
        if not tokens.done():  # a written victim action narrows Sec. 3.4 selection
            victim_order = _parse_action(
                victim.kind, victim.location, victim.strength, victim.special, tokens
            )
        return Dislodge(kind, location, victim, victim_order, strength, special)
    if action == "BAR":
        victim = _parse_unit_ref(tokens)
        if tokens.next() != "-":
            raise OrderSyntaxError(f"bar requires the victim's move: {tokens.text!r}")
        destination = parse_location(tokens.next())
        return Bar(kind, location, victim, destination, strength, special)
    raise OrderSyntaxError(f"unknown order action {action!r} in {tokens.text!r}")


def _parse_subject(tokens: _Tokens) -> tuple[str | None, SupportableOrder]:
    """Parse a support's subject: ``[POWER] [X]KIND LOCATION [action...]``.

    The leading owner is written only when Areas make the bare unit spec
    ambiguous (rulebook "Order specifications"); any token that is not a
    unit-kind token is read as the owner.
    """
    subject_power: str | None = None
    first = tokens.next()
    if _match_kind(first) is None:
        subject_power = first
        first = tokens.next()
    kind, strength, special = _parse_kind(first, tokens.text)
    location = parse_location(tokens.next())
    subject = _parse_action(kind, location, strength, special, tokens)
    if isinstance(subject, (Convoy, Support)):
        raise OrderSyntaxError(
            f"a support's subject cannot be {type(subject).__name__}: {tokens.text!r}"
        )
    if isinstance(subject, Move):
        subject = replace(subject, via_convoy=False)  # route flag is not part of matching
    return subject_power, subject


def parse_order(text: str) -> MovementOrder:
    """Parse one movement-phase order: the adjudication grammar plus the Layer-2 actions.

    Raises :class:`OrderSyntaxError` on text that is not well-formed.
    """
    tokens = _Tokens(text)
    kind, strength, special = _parse_kind(tokens.next(), text)
    location = parse_location(tokens.next())
    order: MovementOrder
    if tokens.peek() in _LAYER2_ACTIONS:
        order = _parse_layer2_action(kind, location, strength, special, tokens.next(), tokens)
    else:
        order = _parse_action(kind, location, strength, special, tokens)
    tokens.expect_end()
    return order


def parse_retreat_order(text: str) -> Order | Disband:
    """Parse one retreat-phase order: ``Disband u`` or any movement-grammar order.

    Permissive by design: only moves and disbands are *valid* retreat
    orders (:data:`RetreatOrder`), but supports, convoys and holds submitted
    during a retreat phase are well-formed game orders that the retreat
    adjudicator must see and reject (DATC 6.H.1-6.H.4 report them as
    illegal); unparseable text stays an interface error.
    """
    tokens = _Tokens(text)
    if tokens.next() == "DISBAND":
        kind, strength, special = _parse_kind(tokens.next(), text)
        location = parse_location(tokens.next())
        tokens.expect_end()
        return Disband(kind, location, strength, special)
    order = parse_order(text)
    if isinstance(order, ADJUDICATION_ORDER_TYPES):
        return order
    raise OrderSyntaxError(f"Layer-2 orders are movement-phase only: {text!r}")


def parse_adjustment_order(text: str) -> AdjustmentOrder:
    """Parse one adjustment-phase order: ``Build u``, ``Disband u`` or ``Waive`` (Sec. 9)."""
    tokens = _Tokens(text)
    first = tokens.next()
    if first == "WAIVE":
        tokens.expect_end()
        return Waive()
    if first not in ("BUILD", "DISBAND"):
        raise OrderSyntaxError(f"adjustment orders are builds, disbands or waives: {text!r}")
    kind, strength, special = _parse_kind(tokens.next(), text)
    location = parse_location(tokens.next())
    if first == "BUILD":
        bound_to: str | None = None
        if tokens.peek() == "BOUND":  # Beleriand double-unit binding
            tokens.next()
            bound_to = parse_location(tokens.next()).province
        tokens.expect_end()
        return Build(kind, location, strength, bound_to, special)
    tokens.expect_end()
    return Disband(kind, location, strength, special)


def _unit_text(
    kind: UnitKind, location: Location, strength: int | None, special: str | None = None
) -> str:
    prefix = str(strength) if strength is not None and strength != 1 else ""
    designator = f"({special})" if special is not None else ""
    return f"{prefix}{kind.value}{designator} {location}"


def _ref_text(ref: UnitRef) -> str:
    return f"{ref.power} {_unit_text(ref.kind, ref.location, ref.strength, ref.special)}"


def format_order(order: GameOrder) -> str:
    """Canonical text of an order; the phase parsers invert it exactly."""
    if isinstance(order, Waive):
        return "Waive"
    unit = _unit_text(order.unit_kind, order.location, order.strength, order.special)
    if isinstance(order, Hold):
        return f"{unit} H"
    if isinstance(order, Move):
        suffix = " via convoy" if order.via_convoy else ""
        return f"{unit} - {order.destination}{suffix}"
    if isinstance(order, Dislodge):
        tail = ""
        if order.victim_order is not None:
            victim_text = format_order(order.victim_order)
            _, _, action = victim_text.partition(" ")
            _, _, action = action.partition(" ")
            tail = f" {action}"
        return f"{unit} DIS {_ref_text(order.victim)}{tail}"
    if isinstance(order, Bar):
        return f"{unit} BAR {_ref_text(order.victim)} - {order.destination}"
    if isinstance(order, Disband):
        return f"Disband {unit}"
    if isinstance(order, Build):
        suffix = f" BOUND {order.bound_to}" if order.bound_to is not None else ""
        return f"Build {unit}{suffix}"
    if isinstance(order, Transform):
        return f"{unit} TFM {order.target_kind.value}"
    if isinstance(order, Devastate):
        return f"{unit} DEV"
    if isinstance(order, MakeMasks):
        return f"{unit} MSK"
    if isinstance(order, GiveMask):
        return f"{unit} GIVES MASK TO {_ref_text(order.receiver)}"
    if isinstance(order, AcceptMask):
        return f"{unit} ACCEPTS MASK FROM {_ref_text(order.giver)}"
    if isinstance(order, SailWest):
        return f"{unit} SAILS WEST"
    if isinstance(order, Support):
        subject = order.subject
        owner = f"{order.subject_power} " if order.subject_power is not None else ""
        subject_text = format_order(subject)
        if isinstance(subject, Hold):
            subject_text = subject_text[: -len(" H")]
        return f"{unit} S {owner}{subject_text}"
    if isinstance(order, Convoy):
        return f"{unit} C {format_order(order.cargo).replace(' via convoy', '')}"
    assert_never(order)


def kind_reaches_province(graph: MapGraph, kind: UnitKind, origin: Location, province: str) -> bool:
    """Whether a unit of ``kind`` at ``origin`` could reach ``province`` by its own adjacency.

    Province-level and coast-blind: the shared geometry behind two callers
    whose destination *models* differ.
    :func:`static_move_legal`'s overland army branch: armies carry no coast, so
    province-level is already their model. The adjudicator's Dislodge and
    support-target validation: a Dislodge names a victim's Area rather than a
    Location and is never convoyed (Sec. 4.2/Sec. 3.5), so there is no coast to
    resolve and no route branch -- a fleet asks only whether *some* fleet
    location of ``province`` is reachable. A Move's fleet branch needs the exact
    written location instead and does not come through here.
    """
    if kind is UnitKind.ARMY:
        if graph.terrain(province) is Terrain.SEA:
            return False
        return graph.army_reachable(origin.province, province)
    return bool(graph.fleet_destinations(origin, province))


def static_move_legal(graph: MapGraph, move: Move) -> bool:
    """Per-order legality of a move's *written* path (Sec. 1 Phase 0 input).

    Overland: destination reachable by the unit's kind adjacency (armies by
    province via :func:`kind_reaches_province`, fleets by exact location
    including coast). Via convoy: the unit is an army and both endpoints are
    coastal -- chain existence is resolution data (Sec. 3.5), not static
    legality.
    """
    origin, destination = move.location, move.destination
    if destination.province == origin.province:
        return False
    if move.via_convoy:
        return (
            move.unit_kind is UnitKind.ARMY
            and graph.terrain(origin.province) is Terrain.COASTAL
            and graph.terrain(destination.province) is Terrain.COASTAL
        )
    if move.unit_kind is UnitKind.ARMY:
        return kind_reaches_province(graph, UnitKind.ARMY, origin, destination.province)
    return graph.fleet_reachable(origin, destination)
