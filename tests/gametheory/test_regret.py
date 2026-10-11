"""regret.py: hand-computed updates first, then convergence to known equilibria.

The single- and two-step updates are worked by hand, including where regret
matching and RM+ diverge and how Hedge's weights move. The acceptance checks
that follow are quantitative: both learners' average strategies must converge to
the known equilibria of Rock-Paper-Scissors and Matching Pennies in self-play.
"""

from __future__ import annotations

import math

import pytest

from entente.gametheory.normal_form import matching_pennies, rock_paper_scissors, self_play
from entente.gametheory.regret import Hedge, RegretMatcher, normalize

UNIFORM_2 = [0.5, 0.5]
UNIFORM_3 = [1 / 3, 1 / 3, 1 / 3]


def test_normalize_falls_back_to_uniform_on_zero_weights() -> None:
    assert normalize([0.0, 0.0, 0.0]) == pytest.approx(UNIFORM_3)


def test_normalize_rescales_to_sum_one() -> None:
    assert normalize([1.0, 3.0]) == pytest.approx([0.25, 0.75])


def test_normalize_rejects_an_empty_weight_vector() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize([])


def test_regret_matcher_rejects_empty_action_set() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        RegretMatcher(0)


def test_regret_matcher_first_strategy_is_uniform() -> None:
    matcher = RegretMatcher(3)
    assert matcher.strategy == pytest.approx(UNIFORM_3)


def test_regret_matcher_hand_computed_single_step() -> None:
    """One step from uniform must move all the weight onto the only positive regret.

    The uniform first strategy plays each action with weight 1/3, so
    ``played_utility = mean([1, 0, -1]) = 0``; regret then becomes the utility
    itself, ``[1, 0, -1]``, and only action 0 is positive.
    """
    matcher = RegretMatcher(3)
    played = matcher.observe_utility([1.0, 0.0, -1.0])
    assert played == pytest.approx(UNIFORM_3)
    assert matcher.strategy == pytest.approx([1.0, 0.0, 0.0])


def test_regret_matcher_plus_diverges_from_plain_after_negative_regret() -> None:
    """Same utility sequence fed to a plain matcher and an RM+ one.

    Step 1 (uniform play, utility [-1, 1]): both accumulate regret [-1, 1];
    ``strategy`` itself clips negative regret when read, so both report
    [0, 1] after this step regardless of ``plus`` -- the stored cumulative
    regret hasn't visibly diverged yet.

    Step 2 (play [0, 1], utility [1, -1]): played_utility = -1, so the
    regret increment is [2, 0]. Plain adds it to the *unclipped* [-1, 1],
    landing on [1, 1] -> strategy [0.5, 0.5]. RM+ adds it to its
    *already-clipped* [0, 1], landing on [2, 1] -> strategy [2/3, 1/3]. The
    two algorithms now visibly disagree.
    """
    plain = RegretMatcher(2)
    plus = RegretMatcher(2, plus=True)
    for matcher in (plain, plus):
        matcher.observe_utility([-1.0, 1.0])
    assert plain.strategy == pytest.approx([0.0, 1.0])
    assert plus.strategy == pytest.approx([0.0, 1.0])

    for matcher in (plain, plus):
        matcher.observe_utility([1.0, -1.0])
    assert plain.strategy == pytest.approx([0.5, 0.5])
    assert plus.strategy == pytest.approx([2 / 3, 1 / 3])


def test_observe_utility_rejects_wrong_length() -> None:
    matcher = RegretMatcher(3)
    with pytest.raises(ValueError, match="expected 3"):
        matcher.observe_utility([1.0, 0.0])


@pytest.mark.parametrize("plus", [False, True])
def test_regret_matching_self_play_converges_to_uniform_on_rps(plus: bool) -> None:
    game = rock_paper_scissors()
    row = RegretMatcher(3, plus=plus)
    col = RegretMatcher(3, plus=plus)

    self_play(game, row, col, iterations=10_000)

    for average in (row.average_strategy, col.average_strategy):
        assert max(abs(p - u) for p, u in zip(average, UNIFORM_3, strict=True)) < 1e-2


def test_hedge_rejects_non_positive_eta() -> None:
    with pytest.raises(ValueError, match="eta"):
        Hedge(2, eta=0.0)


def test_hedge_rejects_empty_action_set() -> None:
    with pytest.raises(ValueError, match="num_actions"):
        Hedge(0, eta=0.5)


def test_hedge_observe_utility_rejects_wrong_length() -> None:
    learner = Hedge(3, eta=0.5)
    with pytest.raises(ValueError, match="expected 3"):
        learner.observe_utility([1.0, 0.0])


def test_hedge_hand_computed_single_step() -> None:
    """One Hedge step from uniform must scale the weights by ``exp(eta * utility)``.

    The first strategy is uniform, weights ``[1, 1]``. The utility ``[1, 0]``
    re-centers on its maximum -- 1 subtracted from both entries -- so the update
    is ``exp(eta * [0, -1]) = [1, exp(-eta)]``, which is proportional to
    ``[exp(eta), 1]``; that is the form the normalized assertion uses.
    """
    learner = Hedge(2, eta=0.5)
    played = learner.observe_utility([1.0, 0.0])
    assert played == pytest.approx(UNIFORM_2)
    expected = normalize([math.exp(0.5), 1.0])
    assert learner.strategy == pytest.approx(expected)


def test_hedge_self_play_converges_to_matching_pennies_equilibrium() -> None:
    game = matching_pennies()
    row = Hedge(2, eta=0.3)
    col = Hedge(2, eta=0.3)

    self_play(game, row, col, iterations=10_000)

    for average in (row.average_strategy, col.average_strategy):
        assert max(abs(p - u) for p, u in zip(average, UNIFORM_2, strict=True)) < 1e-2
