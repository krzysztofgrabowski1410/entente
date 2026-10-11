"""Define the serializable schema a negotiation proposal is carried in.

An :class:`Intent` grounds a proposal in piKL's own candidate output
(docs/pikl.tex Sec. 5): the proposed action for each named power is meant to be one
of the order-sets that power's :class:`entente.search.candidates.CandidateSet`
is already considering for the current phase, not an arbitrary order -- that is
what lets :func:`entente.negotiation.parse.reweight_from_intent` match an
intent straight against a candidate set. Orders serialize as their canonical
text (:func:`entente.engine.orders.format_order`/:func:`~entente.engine.orders
.parse_order`), reusing the engine's own convention
(:mod:`entente.engine.serialization`) instead of inventing a second one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from entente.engine.orders import format_order, parse_order

if TYPE_CHECKING:
    from collections.abc import Mapping

    from entente.engine.orders import GameOrder


@dataclass(frozen=True, slots=True)
class Intent:
    """Carry one proposal between two powers for a named phase.

    ``proposer`` commits to ``proposer_action`` and asks ``ally`` to play
    ``ally_action``.

    ``rationale`` is free text carried through for message generation only
    (:mod:`entente.negotiation.generate`) -- it is never parsed back and never
    feeds :func:`entente.search.pikl.reweight_anchor`.
    """

    proposer: str
    ally: str
    phase: str
    proposer_action: tuple[GameOrder, ...]
    ally_action: tuple[GameOrder, ...]
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.proposer == self.ally:
            raise ValueError("proposer and ally must be different powers")
        if not self.phase:
            raise ValueError("phase must be non-empty")
        if not self.proposer_action:
            raise ValueError("proposer_action must be non-empty")
        if not self.ally_action:
            raise ValueError("ally_action must be non-empty")

    def action_for(self, power: str) -> tuple[GameOrder, ...] | None:
        """Give the proposed action for ``power``, or ``None`` if it is not a party."""
        if power == self.proposer:
            return self.proposer_action
        if power == self.ally:
            return self.ally_action
        return None

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON representation; orders as canonical text."""
        return {
            "proposer": self.proposer,
            "ally": self.ally,
            "phase": self.phase,
            "proposer_action": [format_order(order) for order in self.proposer_action],
            "ally_action": [format_order(order) for order in self.ally_action],
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Intent:
        """Rebuild an intent from the mapping :meth:`to_dict` produces."""
        return cls(
            proposer=str(data["proposer"]),
            ally=str(data["ally"]),
            phase=str(data["phase"]),
            proposer_action=tuple(parse_order(text) for text in data["proposer_action"]),
            ally_action=tuple(parse_order(text) for text in data["ally_action"]),
            rationale=str(data.get("rationale", "")),
        )

    def to_json(self) -> str:
        """Serialize to JSON with sorted keys, so a transcript is diffable."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Intent:
        """Rebuild an intent from the JSON :meth:`to_json` produces."""
        data: Mapping[str, Any] = json.loads(text)
        return cls.from_dict(data)


__all__ = ["Intent"]
