"""normal_form.py: MatrixGame construction, validation and utility-vector math.

Every expected number is computed by hand on Rock-Paper-Scissors.
"""

from __future__ import annotations

import pytest

from entente.gametheory.normal_form import MatrixGame, matching_pennies, rock_paper_scissors


def test_zero_sum_negates_row_payoffs_for_the_column_player() -> None:
    game = rock_paper_scissors()
    assert game.col_payoffs == tuple(tuple(-v for v in row) for row in game.row_payoffs)


def test_rejects_mismatched_payoff_shape() -> None:
    with pytest.raises(ValueError, match="2x2"):
        MatrixGame(("A", "B"), ("X", "Y"), ((0.0, 0.0), (0.0, 0.0)), ((0.0,), (0.0, 0.0)))


def test_rejects_empty_action_set() -> None:
    with pytest.raises(ValueError, match="at least one action"):
        MatrixGame((), (), (), ())


def test_row_utility_vector_against_uniform_rps() -> None:
    game = rock_paper_scissors()
    # against a uniform opponent, every action's expected payoff is 0
    assert game.row_utility_vector([1 / 3, 1 / 3, 1 / 3]) == pytest.approx([0.0, 0.0, 0.0])


def test_row_utility_vector_against_pure_rock() -> None:
    game = rock_paper_scissors()
    # ROCK ties rock (0), PAPER beats rock (+1), SCISSORS loses to rock (-1)
    assert game.row_utility_vector([1.0, 0.0, 0.0]) == pytest.approx([0.0, 1.0, -1.0])


def test_row_utility_vector_rejects_wrong_length_col_strategy() -> None:
    with pytest.raises(ValueError, match="col_actions"):
        rock_paper_scissors().row_utility_vector([1.0, 0.0])


def test_col_utility_vector_rejects_wrong_length_row_strategy() -> None:
    with pytest.raises(ValueError, match="row_actions"):
        rock_paper_scissors().col_utility_vector([1.0, 0.0])


def test_expected_payoffs_are_zero_sum() -> None:
    game = rock_paper_scissors()
    row_value, col_value = game.expected_payoffs([0.5, 0.3, 0.2], [0.1, 0.6, 0.3])
    assert row_value == pytest.approx(-col_value)


def test_matching_pennies_uniform_equilibrium_has_zero_value() -> None:
    game = matching_pennies()
    row_value, col_value = game.expected_payoffs([0.5, 0.5], [0.5, 0.5])
    assert row_value == pytest.approx(0.0)
    assert col_value == pytest.approx(0.0)


def test_matching_pennies_row_exploits_a_biased_column() -> None:
    game = matching_pennies()
    # column favors HEADS -> row should prefer HEADS too (to match)
    utility = game.row_utility_vector([0.9, 0.1])
    assert utility[0] > utility[1]
