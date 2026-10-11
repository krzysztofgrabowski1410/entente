"""The Agent protocol -- the seam between game loop and policy.

An agent is anything that produces an *ordered* sheet for one power in the
current phase of a game. The loop (CLI, tournaments, self-play) calls
``orders`` once per power per decision phase; returning an empty sequence
submits nothing, which the engine resolves with the phase defaults (units
hold / dislodged units disband / builds waived, removals by civil disorder).

Agents read the :class:`~entente.engine.game.Game` surface only -- they never
see adjudicator internals, and any randomness must be their own, seeded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.game import Game
    from entente.engine.orders import GameOrder


@runtime_checkable
class Agent(Protocol):
    """A per-power order policy."""

    def orders(self, game: Game, power: str) -> Sequence[GameOrder]:
        """Return the sheet ``power`` submits for ``game``'s current phase."""
        ...
