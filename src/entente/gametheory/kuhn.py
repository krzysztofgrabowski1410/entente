"""Kuhn poker: a solved game small enough to check a CFR implementation against.

It is the worked exercise game of Neller & Lanctot's *An Introduction to
Counterfactual Regret Minimization* (2013, Sec. 3.5); the game itself is due to
Kuhn (1950).

Three cards {J, Q, K}; two players each ante 1 chip and are dealt one
private card (the third stays hidden); Player 0 acts first. Each decision is
PASS (check, or fold if facing a bet) or BET (bet, or call if facing a bet);
there is no raise, so a betting round is at most two actions. A pass facing
a bet forfeits the pot to the bettor; two passes or a call go to a showdown,
which the higher card wins. This is small enough for :mod:`entente.gametheory.cfr`
to solve exactly by full-tree traversal, with a known equilibrium value
(-1/18 to Player 0) to check convergence against.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

CARDS: tuple[str, ...] = ("J", "Q", "K")
_RANK = {card: rank for rank, card in enumerate(CARDS)}

PASS = "p"
BET = "b"
ACTIONS: tuple[str, ...] = (PASS, BET)

_TERMINAL_HISTORIES = frozenset({"pp", "pbp", "pbb", "bp", "bb"})
_ANTE = 1.0
"""Chips each player puts in before cards are dealt (Kuhn's rules)."""


@dataclass(frozen=True, slots=True)
class Deal:
    """One chance outcome: ``player_card[i]`` is the card dealt to player ``i``.

    The third card is dealt to no one, because Kuhn poker is heads-up.
    """

    player_card: tuple[str, str]


def all_deals() -> tuple[Deal, ...]:
    """Return the 6 equally likely deals, which vanilla CFR sums over exactly.

    They are the ordered draws of 2 cards from 3, without replacement.
    """
    return tuple(Deal(pair) for pair in permutations(CARDS, 2))


def initial_states() -> tuple[KuhnState, ...]:
    """Return one root, empty-history :class:`KuhnState` per deal.

    These are the ``root_states`` :mod:`entente.gametheory.cfr` and
    :mod:`entente.gametheory.exploitability` traverse.
    """
    return tuple(KuhnState(deal) for deal in all_deals())


@dataclass(frozen=True, slots=True)
class KuhnState:
    """One node of the game tree: a fixed deal plus the betting history so far.

    The history is a string over :data:`ACTIONS`, such as ``"pb"``. This
    satisfies the ``GameState`` shape :mod:`entente.gametheory.cfr` traverses
    generically -- the typing is structural, so nothing there imports this
    module.
    """

    deal: Deal
    history: str = ""

    @property
    def is_terminal(self) -> bool:
        """Report whether the betting is over and the hand is decided."""
        return self.history in _TERMINAL_HISTORIES

    @property
    def player(self) -> int:
        """Return whose turn it is.

        Players alternate strictly, Player 0 first, so the acting player is the
        history length's parity.
        """
        return len(self.history) % 2

    @property
    def legal_actions(self) -> tuple[str, ...]:
        """Return the actions available here: both, or none once terminal."""
        return () if self.is_terminal else ACTIONS

    def information_set(self) -> str:
        """Return the acting player's own card plus the public history.

        That is everything they actually observe, and never the opponent's
        hidden card.
        """
        return self.deal.player_card[self.player] + self.history

    def after(self, action: str) -> KuhnState:
        """Return the node reached by appending ``action`` to the history."""
        if self.is_terminal:
            raise ValueError(f"no actions from a terminal history {self.history!r}")
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        return KuhnState(self.deal, self.history + action)

    def payoff(self, player: int) -> float:
        """Return the terminal payoff to ``player``, in chips won or lost this hand.

        The game is zero-sum, so ``payoff(1) == -payoff(0)``. A fold forfeits
        only the antes, one chip; a showdown -- ``"pp"``, or a called bet --
        additionally moves whatever was bet and called.
        """
        if not self.is_terminal:
            raise ValueError(f"history {self.history!r} is not terminal")
        if self.history in ("bp", "pbp"):
            winner = 0 if self.history == "bp" else 1
            magnitude = _ANTE
        else:
            winner = 0 if self._higher_card(0, 1) else 1
            magnitude = _ANTE if self.history == "pp" else _ANTE + 1.0
        return magnitude if winner == player else -magnitude

    def _higher_card(self, a: int, b: int) -> bool:
        return _RANK[self.deal.player_card[a]] > _RANK[self.deal.player_card[b]]
