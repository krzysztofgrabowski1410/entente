"""Two-player normal-form (matrix) games.

The setting Neller & Lanctot's *An Introduction to Counterfactual Regret
Minimization* (2013) opens with (Sec. 2, the Rock-Paper-Scissors running
example) before extending regret matching to extensive-form trees: each
player picks one action from a fixed set; payoffs come from a matrix indexed
by the joint action. :func:`self_play` runs two learners
(:class:`~entente.gametheory.regret.Learner`) against each other --
:mod:`entente.gametheory.regret`'s convergence guarantee (average strategy
-> Nash equilibrium in two-player zero-sum self-play) is exercised here on
the simplest possible game before :mod:`entente.gametheory.kuhn` /
:mod:`entente.gametheory.cfr` extend it to a tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.gametheory.regret import Learner


@dataclass(frozen=True, slots=True)
class MatrixGame:
    """A two-player game given by a payoff matrix for each player.

    ``row_payoffs[i][j]`` is paid to the row player, and ``col_payoffs[i][j]``
    to the column player, when row plays action ``i`` and column plays action
    ``j``. General-sum by construction; :meth:`zero_sum` builds the common
    special case where ``col_payoffs = -row_payoffs``.
    """

    row_actions: tuple[str, ...]
    col_actions: tuple[str, ...]
    row_payoffs: tuple[tuple[float, ...], ...]
    col_payoffs: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Reject an empty action set or a payoff matrix of the wrong shape."""
        rows, cols = len(self.row_actions), len(self.col_actions)
        if rows == 0 or cols == 0:
            raise ValueError("a matrix game needs at least one action per player")
        for name, matrix in (("row_payoffs", self.row_payoffs), ("col_payoffs", self.col_payoffs)):
            if len(matrix) != rows or any(len(row) != cols for row in matrix):
                raise ValueError(f"{name} must be {rows}x{cols}")

    @classmethod
    def zero_sum(
        cls,
        row_actions: Sequence[str],
        col_actions: Sequence[str],
        row_payoffs: Sequence[Sequence[float]],
    ) -> MatrixGame:
        """Build a zero-sum game from the row player's payoff matrix alone."""
        col_payoffs = tuple(tuple(-value for value in row) for row in row_payoffs)
        return cls(
            tuple(row_actions), tuple(col_actions), tuple(map(tuple, row_payoffs)), col_payoffs
        )

    def row_utility_vector(self, col_strategy: Sequence[float]) -> list[float]:
        """Return each row action's expected payoff against a fixed ``col_strategy``."""
        if len(col_strategy) != len(self.col_actions):
            raise ValueError("col_strategy length must match col_actions")
        return [
            sum(p * v for p, v in zip(col_strategy, row, strict=True)) for row in self.row_payoffs
        ]

    def col_utility_vector(self, row_strategy: Sequence[float]) -> list[float]:
        """Return each column action's expected payoff against a fixed ``row_strategy``."""
        if len(row_strategy) != len(self.row_actions):
            raise ValueError("row_strategy length must match row_actions")
        return [
            sum(row_strategy[i] * self.col_payoffs[i][j] for i in range(len(self.row_actions)))
            for j in range(len(self.col_actions))
        ]

    def expected_payoffs(
        self, row_strategy: Sequence[float], col_strategy: Sequence[float]
    ) -> tuple[float, float]:
        """Return ``(row player's value, column player's value)``.

        Both strategies are read as independent mixed strategies.
        """
        row_value = sum(
            p * v for p, v in zip(row_strategy, self.row_utility_vector(col_strategy), strict=True)
        )
        col_value = sum(
            p * v for p, v in zip(col_strategy, self.col_utility_vector(row_strategy), strict=True)
        )
        return row_value, col_value


def rock_paper_scissors(win: float = 1.0) -> MatrixGame:
    """Build standard Rock-Paper-Scissors, symmetric and zero-sum.

    Its unique Nash equilibrium is uniform over all three actions for both
    players, which is what makes it a convergence check (Neller & Lanctot
    Sec. 2's running example).
    """
    actions = ("ROCK", "PAPER", "SCISSORS")
    row_payoffs = (
        (0.0, -win, win),
        (win, 0.0, -win),
        (-win, win, 0.0),
    )
    return MatrixGame.zero_sum(actions, actions, row_payoffs)


def matching_pennies() -> MatrixGame:
    """Build Matching Pennies: row wins by matching, column wins by mismatching.

    It is the minimal two-action zero-sum example with a known, non-degenerate
    mixed equilibrium: both players uniform over ``{HEADS, TAILS}``, and a game
    value of 0.
    """
    actions = ("HEADS", "TAILS")
    row_payoffs = ((1.0, -1.0), (-1.0, 1.0))
    return MatrixGame.zero_sum(actions, actions, row_payoffs)


def self_play(
    game: MatrixGame, row_learner: Learner, col_learner: Learner, iterations: int
) -> None:
    """Run two learners against each other in ``game`` for ``iterations`` rounds.

    The rounds are simultaneous: each learner observes the utility vector its
    actions would have earned against the other's *current* strategy that round
    (Neller & Lanctot Sec. 2's self-play protocol). Learners accumulate their
    own average strategy in place, so read ``row_learner.average_strategy`` and
    ``col_learner.average_strategy`` afterwards.
    """
    for _ in range(iterations):
        row_strategy = row_learner.strategy
        col_strategy = col_learner.strategy
        row_learner.observe_utility(game.row_utility_vector(col_strategy))
        col_learner.observe_utility(game.col_utility_vector(row_strategy))
