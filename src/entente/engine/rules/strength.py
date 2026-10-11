"""The strength pipeline: spec sections 2-3 of docs/adjudication.tex, in exact Fractions.

Pure arithmetic and predicates. Everything resolution-dependent -- which
attackers *exist* (convoy viability, Sec. 3.1.4), which supports are *delivered*
(a dislodged giver delivers nothing, unless what it backs is a Bar, Sec. 3.3),
prevent-nulling (Sec. 4.2) -- is decided by the adjudicator and passed in; this
module never sees a labeling. Strength *modifiers* (Sec. 2) are likewise
applied upstream: the adjudicator precomputes each order's modified base via
``Ruleset.modified_base`` (they are static per order sheet), and the arithmetic
here consumes those values. That split is what lets both implementations
(enumerator and staged resolver) share one arithmetic core.

No floats anywhere: every value is a :class:`fractions.Fraction`.
"""

from __future__ import annotations

from fractions import Fraction
from typing import TYPE_CHECKING

from entente.engine.orders import (
    RELOCATING_TYPES,
    STATIONARY_TYPES,
    Bar,
    Dislodge,
    Hold,
    Move,
    Order,
    Support,
    UnitRef,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

ZERO = Fraction(0)


def is_relocating(order: Order) -> bool:
    """Whether the order makes its unit an attacker (Sec. 1): Move and Dislodge."""
    return isinstance(order, RELOCATING_TYPES)


def is_stationary(order: Order) -> bool:
    """Whether the order keeps its unit in place (Sec. 1): Hold, Support, Convoy and Bar.

    The exact complement of :func:`is_relocating` over the adjudicator's
    grammar -- ``orders`` asserts the partition at import, so neither
    predicate can silently answer False for an unclassified type.
    """
    return isinstance(order, STATIONARY_TYPES)


def refs_compatible(written: UnitRef, actual: UnitRef) -> bool:
    """Whether two written victim references describe the same unit.

    Owner and province must agree; kind is always written and must agree; an
    unwritten strength prefix means strength 1 (the rulebook reads a bare "A"
    as "1A": the bare token is the single-strength unit's own name, not a
    wildcard; on the standard map every unit is strength 1, so the reading is
    observably identical there). The designator must agree too: ``3A(M)`` and
    ``3A(G)`` name different units, so a support written for a Dislodge or a
    Bar against one of them names an order nobody gave when the subject's
    order names the other, and matches nothing. Standard orders write none.
    """
    if written.power != actual.power or written.kind is not actual.kind:
        return False
    if written.location.province != actual.location.province:
        return False
    if written.special != actual.special:
        return False
    return (written.strength or 1) == (actual.strength or 1)


def support_matches(support: Support, subject_order: Order | None) -> bool:
    """Sec. 1: a support matches only if some unit actually received the subject order.

    ``subject_order`` is the order actually held by the unit at the support's
    subject location (None if there is no such unit). Support-hold matches any
    *stationary* order -- supporting a unit that itself supports, convoys or
    bars is legal; a support-hold on a unit ordered to relocate is void.
    Support-move matches on destination *province*; a destination coast, when
    written, must agree (DATC 6.B.9), while the route flag and the departure
    coast are not part of matching (6.B.17).
    A support-dislodge/support-bar matches an actual Dislodge/Bar whose victim
    reference describes the same unit (the resolved victims must also agree --
    checked by the adjudicator's tables, Sec. 3.4); a support-bar must also name
    the same barred destination.
    """
    if subject_order is None:
        return False
    subject = support.subject
    if isinstance(subject, Hold):
        return is_stationary(subject_order)
    if isinstance(subject, Dislodge):
        return isinstance(subject_order, Dislodge) and refs_compatible(
            subject.victim, subject_order.victim
        )
    if isinstance(subject, Bar):
        return (
            isinstance(subject_order, Bar)
            and refs_compatible(subject.victim, subject_order.victim)
            and subject.destination.province == subject_order.destination.province
        )
    if not isinstance(subject_order, Move):
        return False
    if subject_order.destination.province != subject.destination.province:
        return False
    written_coast = subject.destination.coast
    return written_coast is None or written_coast == subject_order.destination.coast


def cut_value(attacker_base: Fraction, support_bases: Iterable[Fraction]) -> Fraction:
    """Return the Sec. 3.2 ``cutval``: the attacker's modified base plus its supports' bases.

    The supports count at full PRE-cut, PRE-void value: every matched support
    of the attacker counts even if it is itself cut or its unit is later
    dislodged -- the one-shot model's closed-form guarantee.
    """
    total = attacker_base
    for base in support_bases:
        total += base
    return total


def effective_support(support_base: Fraction, cut: Fraction) -> Fraction:
    """Return the Sec. 3.2 ``eff``: the support's modified base minus its cut, floored at 0."""
    return max(ZERO, support_base - cut)


def pipeline_value(base: Fraction, delivered_effs: Iterable[Fraction]) -> Fraction:
    """Return the Sec. 2 ``value``: base plus the sum of *delivered* support effectivenesses.

    The caller applies the role filters of Sec. 3.3 (e.g. dropping the defender's
    own power's supports in the attack role) before summing.
    """
    total = base
    for eff in delivered_effs:
        total += eff
    return total
