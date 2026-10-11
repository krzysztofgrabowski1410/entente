"""Regret-based strategy iteration.

Implemented from Neller & Lanctot, *An Introduction to Counterfactual Regret
Minimization* (2013, Sec. 2 "Regret Matching", Sec. 3 "Hedge") and Zinkevich et al.,
*Regret Minimization in Games with Incomplete Information* (NeurIPS 2007),
which grounds the regret-matching update in Blackwell approachability and
proves that the *average* strategy of a no-regret learner converges to a
(coarse-)correlated equilibrium under self-play, and to a Nash equilibrium in
two-player zero-sum games.

Two learners are implemented, both against the same tiny interface (an
observed per-action utility vector in, a strategy out):

- :class:`RegretMatcher` -- cumulative-regret matching; ``plus=True`` selects
  RM+ (regret clipped to zero every iteration, Tammelin et al. 2015 as
  surveyed by Neller & Lanctot), which converges faster in practice and is
  the local update :mod:`entente.gametheory.cfr` runs at every information
  set.
- :class:`Hedge` -- multiplicative-weights (Freund & Schapire 1997), the
  learner :mod:`entente.search.pikl` anchors to a base policy via a KL
  penalty; unanchored Hedge is its ``lambda -> 0`` primitive.

Both accumulate a running *average* strategy (reach-weighted for
:class:`RegretMatcher` so :mod:`entente.gametheory.cfr` can weight
per-information-set contributions by the acting player's own reach
probability, per Zinkevich et al.'s definition of the average strategy).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence


def normalize(weights: Sequence[float]) -> list[float]:
    """Rescale non-negative ``weights`` to sum to one.

    A vector whose weights are all numerically zero normalizes to the uniform
    distribution, which is the standard regret-matching answer to "no positive
    regret yet" (Neller & Lanctot Sec. 2, initial iterations).
    """
    total = sum(weights)
    count = len(weights)
    if count == 0:
        raise ValueError("cannot normalize an empty weight vector")
    if total <= 0.0:
        return [1.0 / count] * count
    return [w / total for w in weights]


class Learner(Protocol):
    """The interface every driver of a learner uses.

    :func:`entente.gametheory.normal_form.self_play` and
    :mod:`entente.gametheory.cfr` both work through it: read the current
    strategy, report what utility each action would have earned, get back what
    was actually played this round.
    """

    @property
    def strategy(self) -> list[float]:
        """Return the strategy this iteration plays."""
        ...

    def observe_utility(self, utility: Sequence[float], *, weight: float = 1.0) -> list[float]:
        """Fold one round's per-action ``utility`` in, returning the strategy played."""
        ...

    @property
    def average_strategy(self) -> list[float]:
        """Return the mean of every iteration's strategy so far."""
        ...


class RegretMatcher:
    """Regret matching over a fixed action set (Neller & Lanctot Sec. 2).

    ``plus=True`` selects RM+ (Sec. 2, "Regret Matching+"), which clips
    cumulative regret at zero every iteration.

    Each call to :meth:`observe_utility` is one iteration: the caller reports
    the utility *each action would have earned* had it been played alone
    (e.g. the counterfactual value at a CFR information set, or a
    normal-form action's payoff against the opponent's current strategy).
    The learner accumulates regret for not having played the better actions,
    and next iteration's :attr:`strategy` shifts proportionally to positive
    cumulative regret.
    """

    def __init__(self, num_actions: int, *, plus: bool = False) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        self.num_actions = num_actions
        self.plus = plus
        self._cumulative_regret = [0.0] * num_actions
        self._strategy_sum = [0.0] * num_actions

    @property
    def strategy(self) -> list[float]:
        """Return this iteration's strategy: positive cumulative regret, normalized.

        It is uniform for as long as every action's regret is still at or below
        zero.
        """
        return normalize([max(r, 0.0) for r in self._cumulative_regret])

    def observe_utility(self, utility: Sequence[float], *, weight: float = 1.0) -> list[float]:
        """Update regrets from ``utility`` and fold this iteration into the average.

        ``utility`` holds one entry per action: this iteration's payoff had that
        action been played alone. ``weight`` scales the strategy's contribution
        to the average -- a reach probability for CFR, and 1.0 for a plain
        repeated game, where every iteration counts equally.

        Returns the strategy that was in effect for this iteration (read
        *before* the update below mutates it), so callers never need to
        cache :attr:`strategy` themselves to stay consistent.
        """
        if len(utility) != self.num_actions:
            raise ValueError(f"expected {self.num_actions} utilities, got {len(utility)}")
        strategy = self.strategy
        played_utility = sum(p * u for p, u in zip(strategy, utility, strict=True))
        for action in range(self.num_actions):
            regret = self._cumulative_regret[action] + utility[action] - played_utility
            self._cumulative_regret[action] = max(regret, 0.0) if self.plus else regret
            self._strategy_sum[action] += weight * strategy[action]
        return strategy

    @property
    def average_strategy(self) -> list[float]:
        """Return the reach-weighted mean of every iteration's :attr:`strategy`.

        This mean, not the latest and typically oscillating :attr:`strategy`, is
        the quantity Zinkevich et al.'s Theorem 3 shows converges to a Nash
        equilibrium strategy in two-player zero-sum self-play.
        """
        return normalize(self._strategy_sum)


class Hedge:
    """Multiplicative-weights, or Hedge (Freund & Schapire 1997).

    Surveyed in Neller & Lanctot Sec. 3: weights grow or shrink by
    ``exp(eta * utility)`` each iteration, and the average strategy has external
    regret ``O(sqrt(T log n))``, matching regret matching's asymptotic guarantee
    through a different, multiplicative, rate-``eta`` update rule.

    :mod:`entente.search.pikl` KL-anchors this exact update to a fixed base
    policy, and plain :class:`Hedge` is the unanchored ``lambda -> 0`` special
    case. That is why it lives alongside :class:`RegretMatcher` here rather than
    being folded into it.
    """

    def __init__(self, num_actions: int, *, eta: float) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if eta <= 0.0:
            raise ValueError("eta must be > 0")
        self.num_actions = num_actions
        self.eta = eta
        self._weights = [1.0] * num_actions
        self._strategy_sum = [0.0] * num_actions

    @property
    def strategy(self) -> list[float]:
        """Return this iteration's strategy: the current weights, normalized."""
        return normalize(self._weights)

    def observe_utility(self, utility: Sequence[float], *, weight: float = 1.0) -> list[float]:
        """Update weights from this iteration's per-action ``utility``.

        Utilities are re-centered on their max before exponentiating (a
        shared per-round rescaling of every weight by ``exp(-eta * max)``,
        which cancels out of the normalized :attr:`strategy` and therefore
        does not change the algorithm -- only keeps ``exp`` in a numerically
        safe range for large ``eta`` or utility magnitudes).
        """
        if len(utility) != self.num_actions:
            raise ValueError(f"expected {self.num_actions} utilities, got {len(utility)}")
        strategy = self.strategy
        for action in range(self.num_actions):
            self._strategy_sum[action] += weight * strategy[action]
        top = max(utility)
        self._weights = [
            w * math.exp(self.eta * (u - top)) for w, u in zip(self._weights, utility, strict=True)
        ]
        return strategy

    @property
    def average_strategy(self) -> list[float]:
        """Return the mean of every iteration's strategy, which is what converges."""
        return normalize(self._strategy_sum)
