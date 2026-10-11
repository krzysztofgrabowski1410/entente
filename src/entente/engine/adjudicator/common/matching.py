"""Reference matching: whether written text describes *this* unit or order.

The signature law in one place: a written designator is part of a unit's
fungibility signature, and an unwritten strength prefix means strength 1 (the
rulebook reads a bare ``A`` as ``1A``). A designator is matched exactly both
ways: ``3A(M)`` fits only a unit whose order writes ``(M)``, and a bare ``3A``
only one whose order writes none. Every binding site in the package asks
these same questions -- Phase 0 when an order claims a unit occurrence and when
a support resolves its subject, the static tables when a Dislodge victim, a
Bar victim or a support subject binds to a record -- so the predicates live
below both rather than inside either (docs/adjudication.tex Sec. 1, Sec. 3.4).

``_unit_key`` sits here too: it is the canonical unit ordering the whole
package's determinism rests on, and it is the sort form of the same fields the
predicates compare. Note it spells a missing coast ``""`` rather than ``None``:
tuple comparison needs a totally ordered element, and ``None < str`` raises.
``None`` stays the domain value everywhere else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from entente.engine.orders import Order, UnitRef
    from entente.engine.state import Unit


def _unit_key(unit: Unit) -> tuple[str, str, str, str, int]:
    """Canonical sort key of a unit: province, coast (``""`` if none), power, kind, strength."""
    return (
        unit.location.province,
        unit.location.coast or "",
        unit.power,
        unit.kind.value,
        unit.strength,
    )


def ref_matches_unit(ref: UnitRef, unit: Unit) -> bool:
    """Whether a written victim reference describes ``unit``.

    Owner, kind and province must agree; an unwritten strength prefix means
    strength 1, because the rulebook treats a designation without a prefix as
    base strength 1 ("A" is the same as "1A") -- observably identical on the standard map,
    where every unit is strength 1 (Sec. 3.4).
    """
    return (
        ref.power == unit.power
        and ref.kind is unit.kind
        and ref.location.province == unit.location.province
        and (ref.strength or 1) == unit.strength
    )


def _ref_special_ok(ref_special: str | None, record_special: str | None) -> bool:
    """Whether a reference's designator fits a record's effective one: equal, or neither written.

    On the board both specials are a plain ``3A``, so the adjudicator knows a
    special only by the designator its own order writes: a reference written
    ``3A(M)`` fits only a record whose order writes ``(M)``, and a bare
    ``3A`` only a record whose order writes none.

    In a game that is every special: the Beleriand game layer puts each
    special's designator on its effective order -- a special with no order or
    with an invalidated one included -- and refuses every order whose spec or
    reference describes a special without one, so a designated reference
    always meets a designated record and a bare one an ordinary unit. Run
    without the game layer, a ``3A`` whose order writes no designator, or
    that has no order, is an unnamed ``3A``, which only a bare reference fits.
    """
    return ref_special == record_special


def _spec_matches_unit(order: Order, unit: Unit) -> bool:
    """Whether an order's own unit spec fits ``unit`` exactly.

    The written kind must agree and so must the strength prefix, an unwritten
    one meaning strength 1 (the rulebook reads a bare "A" as "1A"). The
    advisory province-only fallback and the strict 4.C.2 choice C gate both
    sit at the call sites.
    """
    if order.unit_kind is not unit.kind:
        return False
    return (order.strength or 1) == unit.strength
