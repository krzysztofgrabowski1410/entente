"""Read an incoming message back to an intent and reweight candidates by it.

:func:`parse_message` reads the fixed grammar :class:`entente.negotiation
.generate.StubProvider` emits -- a deliberately constrained template, not
general natural-language understanding: this prototype parses a small proposal
grammar and does not attempt to read arbitrary press (see the Limitations
section of negotiation/README.md). :func:`reweight_from_intent` is the
negotiation-specific half of the reweighting hook: it turns a parsed
:class:`~entente.negotiation.intents.Intent` into the opaque numeric bias
vector that the game-agnostic :func:`entente.search.pikl.reweight_anchor`
needs, by matching the intent's proposed action
against a power's own :class:`~entente.search.candidates.CandidateSet` --
piKL's core algorithm never sees an ``Intent`` or an ``Order``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from entente.engine.orders import parse_order
from entente.negotiation.intents import Intent
from entente.search.pikl import reweight_anchor

if TYPE_CHECKING:
    from entente.search.candidates import CandidateSet

DEFAULT_BOOST = 4.0
"""Multiplicative tilt applied to a candidate matching the proposed action;
every other candidate keeps weight 1.0 (see :func:`reweight_from_intent`)."""

_MESSAGE_RE = re.compile(
    r"^PROPOSAL from (?P<proposer>\S+) to (?P<ally>\S+) for (?P<phase>\S+): "
    r"(?P=proposer) orders \[(?P<proposer_action>[^\]]*)\]; "
    r"(?P=ally) orders \[(?P<ally_action>[^\]]*)\]\."
    r"(?:\s+(?P<rationale>.+))?$"
)


class MessageParseError(ValueError):
    """Raised for text that is not well-formed proposal notation.

    Distinct from a semantically odd intent (e.g. naming a power not in the
    game) -- that is left to the caller, which has the game context this
    module deliberately does not import.
    """


def _split_action(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_message(text: str) -> Intent:
    """Parse one proposal message into an :class:`Intent`.

    Raises :class:`MessageParseError` if ``text`` does not match the fixed
    grammar :class:`~entente.negotiation.generate.StubProvider` emits.
    """
    match = _MESSAGE_RE.match(text.strip())
    if match is None:
        raise MessageParseError(f"text is not a well-formed proposal: {text!r}")
    return Intent(
        proposer=match["proposer"],
        ally=match["ally"],
        phase=match["phase"],
        proposer_action=tuple(parse_order(t) for t in _split_action(match["proposer_action"])),
        ally_action=tuple(parse_order(t) for t in _split_action(match["ally_action"])),
        rationale=match["rationale"] or "",
    )


def reweight_from_intent(
    candidates: CandidateSet, intent: Intent, *, boost: float = DEFAULT_BOOST
) -> tuple[float, ...]:
    """Bias ``candidates.anchor`` toward the order-set ``intent`` proposes.

    The bias applies to ``candidates.power``.

    Matches ``intent.action_for(candidates.power)`` against
    ``candidates.order_sets`` as an unordered set of orders (a candidate is a
    one-order-per-unit sheet; sequence order is not meaningful -- mirrors the
    dedup key in :func:`entente.search.candidates.collect_candidates`), boosts
    an exact match by ``boost``, and renormalizes via
    :func:`entente.search.pikl.reweight_anchor`. An intent can only sharpen
    belief in a candidate piKL was already considering, never inject a new
    one: if ``candidates.power`` is not a party to ``intent``, or no
    candidate order-set matches the proposed action, the anchor is returned
    unchanged.
    """
    target = intent.action_for(candidates.power)
    if target is None:
        return candidates.anchor
    target_set = frozenset(target)
    bias = tuple(
        boost if frozenset(order_set) == target_set else 1.0 for order_set in candidates.order_sets
    )
    return tuple(reweight_anchor(candidates.anchor, bias))


__all__ = ["DEFAULT_BOOST", "MessageParseError", "parse_message", "reweight_from_intent"]
