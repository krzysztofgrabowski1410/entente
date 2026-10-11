"""Single-process lockstep stepper, for a neural-net policy in the loop.

Multiprocessing (:mod:`entente.sim.parallel`) does not suit a neural-net
policy: dispatching one game per process serializes GPU access instead of
sharing it. Here, ``M`` games advance **in lockstep**, one phase at a time;
every tick, every still-running game's pending (power, legal-order) decisions
are collected into a single flat batch and handed to one policy callback,
which returns one order sheet per request. A GPU policy can then run one
batched forward pass per tick instead of ``M`` sequential ones.

This module defines only the batching *interface* and the bookkeeping loop --
**no torch import here**. ``PolicyCallback`` is plain ``Callable``; anything
conforming to it works, from a one-line random choice to a batched network
forward pass.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from entente.engine.game import Game
from entente.engine.orders import AnyOrder


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """One (game, power) pending an order sheet this tick.

    ``index`` is the request's position in the game list the
    :class:`LockstepRunner` was built with -- batched callbacks return orders
    aligned to *this* index (list position), not to ``index`` itself, so a
    policy that groups requests by game must key on ``index`` explicitly.
    ``legal_orders`` is precomputed because every consumer needs it -- random
    choice, heuristic scoring, engine-exact legality masking -- so no callback
    has to recompute what the engine already knows.
    """

    index: int
    power: str
    game: Game
    legal_orders: tuple[AnyOrder, ...]


PolicyCallback = Callable[[Sequence[DecisionRequest]], Sequence[Sequence[AnyOrder]]]
"""Given every pending decision this tick, return one order sheet per
request, aligned by list position (``sheets[i]`` answers ``requests[i]``) --
the same convention a batched tensor forward pass naturally produces."""


def new_games(count: int, *, ruleset: str = "standard", map_name: str = "standard") -> list[Game]:
    """Build ``count`` fresh games, ready for :class:`LockstepRunner`."""
    return [Game.new(ruleset, map_name) for _ in range(count)]


class LockstepRunner:
    """Step ``M`` games forward together, one policy call per tick.

    Every pending decision across every active game goes into a single
    :data:`PolicyCallback` call.

    ``step()`` is the primitive a training loop drives directly (to interleave
    trajectory collection, value targets, and so on between ticks); ``run()``
    is a convenience wrapper for the common "play everything out" case.
    """

    def __init__(
        self, games: Sequence[Game], policy: PolicyCallback, *, max_year: int | None = None
    ) -> None:
        self._games = list(games)
        self._policy = policy
        self._max_year = max_year

    @property
    def games(self) -> tuple[Game, ...]:
        """Return the games being stepped, in the order they were given."""
        return tuple(self._games)

    def _pending_indices(self) -> list[int]:
        """List the games still to be stepped: not over, and within the year cap."""
        return [
            i
            for i, game in enumerate(self._games)
            if not game.is_over and (self._max_year is None or game.year <= self._max_year)
        ]

    def step(self) -> bool:
        """Advance every active game by exactly one phase.

        Returns ``False``, having done nothing, once every game is over or past
        the year cap.

        A game with no legal-order decisions pending is never "active" here:
        ``Game.adjudicate()`` already auto-executes or skips such phases
        internally, so any non-over game always has at least one power with
        a real decision when control reaches this method.
        """
        active = self._pending_indices()
        if not active:
            return False
        requests: list[DecisionRequest] = []
        for index in active:
            game = self._games[index]
            for power in game.powers:
                legal = game.legal_orders(power)
                if legal:
                    requests.append(DecisionRequest(index, power, game, legal))
        sheets = self._policy(requests)
        if len(sheets) != len(requests):
            raise ValueError(f"policy returned {len(sheets)} sheets for {len(requests)} requests")
        for request, sheet in zip(requests, sheets, strict=True):
            if sheet:
                request.game.submit(request.power, list(sheet))
        for index in active:
            self._games[index].adjudicate()
        return True

    def run(self, *, max_ticks: int | None = None) -> int:
        """Step until every game is done, or until ``max_ticks`` is reached.

        Returns the number of ticks actually executed.
        """
        ticks = 0
        while (max_ticks is None or ticks < max_ticks) and self.step():
            ticks += 1
        return ticks
