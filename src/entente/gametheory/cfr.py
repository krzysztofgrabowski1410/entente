"""Counterfactual regret minimization (CFR) over an extensive-form game tree.

Implemented from Zinkevich et al., *Regret Minimization in Games with
Incomplete Information* (NeurIPS 2007) -- the original algorithm and its
convergence proof (average strategy -> Nash equilibrium in two-player zero-sum
extensive-form games, via a per-information-set regret bound) -- and from
Neller & Lanctot's *An Introduction to Counterfactual Regret Minimization*
(2013, Sec. 3-Sec. 5), which works the recursive tree-walk formulation used
here and its Monte Carlo variant.

:func:`vanilla_cfr` runs one :class:`~entente.gametheory.regret.RegretMatcher`
per information set, updated once per iteration by a full traversal of the
game tree (every chance outcome enumerated exactly -- "vanilla", no
sampling). :func:`chance_sampled_mccfr` is the same tree walk with the
chance node (in :mod:`entente.gametheory.kuhn`, the card deal) sampled once
per iteration instead of enumerated -- Neller & Lanctot Sec. 5's chance-sampled
MCCFR -- trading exactness for a cheaper iteration; both converge to the same
equilibrium value.

Both functions traverse any ``GameState`` implementation structurally (no
import of :mod:`entente.gametheory.kuhn` here) -- the CFR recursion itself is
game-agnostic; only the deal enumeration/sampling at the root is Kuhn's
"one chance event up front" shape.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Protocol

from entente.gametheory.regret import RegretMatcher

if TYPE_CHECKING:
    from collections.abc import Sequence


class GameState(Protocol):
    """The per-node interface CFR needs from a game.

    The game must be two-player, zero-sum and extensive-form, with at most one
    chance move, already resolved by the caller before the walk begins. The
    protocol is structural, so any object of this shape works, for instance
    :class:`entente.gametheory.kuhn.KuhnState`.
    """

    @property
    def is_terminal(self) -> bool:
        """Report whether the game has ended at this node."""
        ...

    @property
    def player(self) -> int:
        """0 or 1: whose decision this node is (undefined if terminal)."""
        ...

    @property
    def legal_actions(self) -> tuple[str, ...]:
        """Return the actions available at this node, empty when terminal."""
        ...

    def information_set(self) -> str:
        """Return the acting player's information set, which regret is tracked per.

        It must collapse exactly the histories that player cannot distinguish --
        their own private information plus the public history -- and nothing
        else: collapse too much and regret is shared between distinguishable
        situations, too little and it never accumulates.
        """
        ...

    def after(self, action: str) -> GameState:
        """Return the node reached by taking ``action`` here."""
        ...

    def payoff(self, player: int) -> float:
        """Terminal utility to ``player`` (undefined if not terminal)."""
        ...


def _walk(
    state: GameState,
    reach0: float,
    reach1: float,
    chance_reach: float,
    regrets: dict[str, RegretMatcher],
    *,
    plus: bool,
) -> float:
    """Walk the tree once from ``state``, running one CFR iteration on the way.

    Every information set's regret matcher the walk passes through is updated
    exactly once. The return is the expected utility to player 0 from ``state``
    onward, under the strategies the regret matchers currently imply.

    ``reach0``/``reach1`` are each player's own reach probability to
    ``state`` (product of their own action probabilities only); combined
    with ``chance_reach`` (constant across one deal's walk -- Kuhn's only
    chance event is the deal, resolved by the caller before the first call)
    they give the counterfactual reach `chance_reach * reach_{-i}` Zinkevich
    et al. weight each information set's regret update by, and the plain
    `reach_i` the average-strategy accumulation is weighted by.
    """
    if state.is_terminal:
        return state.payoff(0)
    player = state.player
    infoset = state.information_set()
    actions = state.legal_actions
    matcher = regrets.get(infoset)
    if matcher is None:
        matcher = RegretMatcher(len(actions), plus=plus)
        regrets[infoset] = matcher
    strategy = matcher.strategy

    action_value_p0 = []
    for probability, action in zip(strategy, actions, strict=True):
        if player == 0:
            child_value = _walk(
                state.after(action), reach0 * probability, reach1, chance_reach, regrets, plus=plus
            )
        else:
            child_value = _walk(
                state.after(action), reach0, reach1 * probability, chance_reach, regrets, plus=plus
            )
        action_value_p0.append(child_value)
    node_value_p0 = sum(p * v for p, v in zip(strategy, action_value_p0, strict=True))

    sign = 1.0 if player == 0 else -1.0
    own_reach = reach0 if player == 0 else reach1
    opponent_reach = chance_reach * (reach1 if player == 0 else reach0)
    actor_utility = [sign * opponent_reach * v for v in action_value_p0]
    matcher.observe_utility(actor_utility, weight=own_reach)
    return node_value_p0


def _average_strategies(regrets: dict[str, RegretMatcher]) -> dict[str, list[float]]:
    """Collect every visited information set's average strategy."""
    return {infoset: matcher.average_strategy for infoset, matcher in regrets.items()}


def vanilla_cfr(
    root_states: Sequence[GameState], iterations: int, *, plus: bool = False
) -> tuple[dict[str, list[float]], list[float]]:
    """Run exact CFR: every iteration walks the full tree once per chance outcome.

    ``root_states`` holds one entry per chance outcome (for instance
    :func:`entente.gametheory.kuhn.all_deals`), each weighted by its uniform
    chance probability ``1 / len(root_states)``.

    ``plus=True`` runs RM+ (:class:`entente.gametheory.regret.RegretMatcher`)
    at every information set instead of vanilla regret matching.

    Returns ``(average_strategy, value_per_iteration)``: the first is every
    visited information set's average strategy (Zinkevich et al. Theorem 3 --
    this, not the final iteration's strategy, is what converges); the
    second is player 0's expected value under *that iteration's* current
    strategies, one entry per iteration, for plotting convergence.
    """
    if not root_states:
        raise ValueError("root_states must be non-empty")
    regrets: dict[str, RegretMatcher] = {}
    chance_reach = 1.0 / len(root_states)
    values: list[float] = []
    for _ in range(iterations):
        total = 0.0
        for root in root_states:
            total += chance_reach * _walk(root, 1.0, 1.0, chance_reach, regrets, plus=plus)
        values.append(total)
    return _average_strategies(regrets), values


def chance_sampled_mccfr(
    root_states: Sequence[GameState], iterations: int, *, plus: bool = False, seed: int = 0
) -> tuple[dict[str, list[float]], list[float]]:
    """Run chance-sampled MCCFR (Neller & Lanctot Sec. 5).

    Each iteration samples one entry of ``root_states`` uniformly instead of
    summing over all of them, then walks the rest of the tree exactly as
    :func:`vanilla_cfr` does. The sampled walk uses ``chance_reach=1.0`` rather
    than ``1/len`` because the 1-in-``len`` sampling probability already carries
    chance's weight *in expectation*; that is what makes the estimator unbiased
    without multiplying it back in.

    Same return shape as :func:`vanilla_cfr`.
    """
    if not root_states:
        raise ValueError("root_states must be non-empty")
    rng = random.Random(seed)
    regrets: dict[str, RegretMatcher] = {}
    values: list[float] = []
    for _ in range(iterations):
        root = rng.choice(root_states)
        values.append(_walk(root, 1.0, 1.0, 1.0, regrets, plus=plus))
    return _average_strategies(regrets), values
