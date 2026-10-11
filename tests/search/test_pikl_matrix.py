"""piKL on normal-form games whose equilibria are known (docs/pikl.tex Sec. 8).

Covered: the ``lam -> 0`` equilibrium limit, the ``lam -> inf`` anchor limit
with Theorem 1's KL bound, the anchored-quantal-response fixed point, and
agreement between the :func:`run_pikl` orchestrator and the independently
written ``self_play`` driver.

Bound bookkeeping (Jacob et al. 2022, Theorems 1-2): with reward bound D and
beta = max_a log(1/tau(a)), the mean policies form a (lam*beta)-approximate
Nash equilibrium (two-player zero-sum) and satisfy
KL(mean || tau) <= (regret/T + D)/lam. Tolerances below state the theory term
plus an explicit finite-T allowance.
"""

from __future__ import annotations

import math
import random

import pytest

from entente.gametheory.normal_form import (
    MatrixGame,
    matching_pennies,
    rock_paper_scissors,
    self_play,
)
from entente.search.pikl import (
    AnchoredHedge,
    PiklPlayer,
    ProfileUtilityFn,
    anchored_quantal_response,
    run_pikl,
)


def _kl(p: list[float], q: list[float] | tuple[float, ...]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0.0)


def _nash_conv(game: MatrixGame, row: list[float], col: list[float]) -> float:
    """NashConv = sum of both players' best-response gains; 0 at equilibrium."""
    row_value, col_value = game.expected_payoffs(row, col)
    br_row = max(game.row_utility_vector(col))
    br_col = max(game.col_utility_vector(row))
    return (br_row - row_value) + (br_col - col_value)


def _matrix_oracle(game: MatrixGame) -> ProfileUtilityFn:
    """Build a joint-profile oracle over a MatrixGame.

    It is the exact stand-in for the Diplomacy value net: exact, so a
    discrepancy is the search's and not the estimator's (docs/pikl.tex Sec. 8).
    """

    def evaluate(profiles: list[tuple[int, ...]]) -> list[tuple[float, float]]:
        return [(game.row_payoffs[i][j], game.col_payoffs[i][j]) for i, j in profiles]

    return evaluate  # type: ignore[return-value]


class TestOrchestratorAgreesWithSelfPlay:
    """Two independently written loops over one update rule must agree exactly."""

    def test_expected_mode_equals_self_play(self) -> None:
        """``run_pikl`` in expected mode and ``self_play`` must agree bit for bit.

        They drive the same update with the same numbers by two separately
        written routes, so anything short of floating-point identity is a bug in
        one of them.
        """
        game = rock_paper_scissors()
        anchor_row = (0.6, 0.3, 0.1)
        anchor_col = (0.2, 0.5, 0.3)
        lam = 0.4
        result = run_pikl(
            [PiklPlayer(anchor_row, lam), PiklPlayer(anchor_col, lam)],
            _matrix_oracle(game),
            iterations=200,
            mode="expected",
            eta=0.7,
        )
        row = AnchoredHedge(anchor_row, lam=lam, eta=0.7)
        col = AnchoredHedge(anchor_col, lam=lam, eta=0.7)
        self_play(game, row, col, 200)
        assert result.mean_strategies[0] == pytest.approx(row.average_strategy, abs=1e-12)
        assert result.mean_strategies[1] == pytest.approx(col.average_strategy, abs=1e-12)
        assert result.profile_evaluations == 9  # 3x3, each profile scored once

    def test_sampled_mode_is_deterministic_given_seed(self) -> None:
        game = matching_pennies()
        players = [PiklPlayer((0.5, 0.5), 0.1), PiklPlayer((0.5, 0.5), 0.1)]
        a = run_pikl(players, _matrix_oracle(game), iterations=50, rng=random.Random(7))
        b = run_pikl(players, _matrix_oracle(game), iterations=50, rng=random.Random(7))
        assert a == b
        assert a.profile_evaluations <= 4  # caching: only 4 joint profiles exist


class TestLambdaSmallApproachesEquilibrium:
    """The weak-anchor limit, where play must reach the game's own equilibrium."""

    def test_rps_biased_anchor(self) -> None:
        """Rock-Paper-Scissors, whose unique equilibrium is uniform.

        At ``lam = 0.01`` with ``beta = ln(10)`` the theory gap is ``lam*beta``,
        about 0.023; the assertion adds 0.05 for finite T (T = 4000).
        """
        game = rock_paper_scissors()
        anchor = (0.6, 0.3, 0.1)
        row = AnchoredHedge(anchor, lam=0.01)
        col = AnchoredHedge(anchor, lam=0.01)
        self_play(game, row, col, 4000)
        row_mean, col_mean = row.average_strategy, col.average_strategy
        assert _nash_conv(game, row_mean, col_mean) < 0.01 * math.log(10.0) + 0.05
        for mean in (row_mean, col_mean):
            assert max(abs(p - 1.0 / 3.0) for p in mean) < 0.05
        # ... and the anchor did NOT capture play: the mean sits a large KL
        # away from the biased anchor (KL(uniform || anchor) ~ 0.24).
        assert _kl(row_mean, anchor) > 0.15
        assert max(abs(p - a) for p, a in zip(row_mean, anchor, strict=True)) > 0.15

    def test_sampled_mode_converges_too(self) -> None:
        """The Diplomacy code path -- sampled opponents, cached oracle -- converges too.

        The check is statistical, so it runs from a fixed seed.
        """
        game = matching_pennies()
        result = run_pikl(
            [PiklPlayer((0.7, 0.3), 0.02), PiklPlayer((0.4, 0.6), 0.02)],
            _matrix_oracle(game),
            iterations=6000,
            rng=random.Random(11),
        )
        row = list(result.mean_strategies[0])
        col = list(result.mean_strategies[1])
        assert _nash_conv(game, row, col) < 0.12
        assert result.profile_evaluations == 4


class TestLambdaLargeStaysNearAnchor:
    """The strong-anchor limit, where play must stay inside Theorem 1's KL ball."""

    def test_kl_bound_matching_pennies(self) -> None:
        """Theorem 1's bound, ``KL(mean || tau) <= (regret/T + D)/lam``, must hold.

        Payoffs lie in [-1, 1], so D = 2 bounds the reward range and at
        ``lam = 25`` the asymptotic bound is 0.08. Both that and the visible
        fact that play *is* the anchor are asserted.
        """
        game = matching_pennies()
        anchor_row = (0.3, 0.7)
        anchor_col = (0.6, 0.4)
        lam = 25.0
        row = AnchoredHedge(anchor_row, lam=lam)
        col = AnchoredHedge(anchor_col, lam=lam)
        self_play(game, row, col, 400)
        assert _kl(row.average_strategy, anchor_row) < 2.0 / lam
        assert _kl(col.average_strategy, anchor_col) < 2.0 / lam
        assert row.average_strategy == pytest.approx(anchor_row, abs=0.02)
        assert col.average_strategy == pytest.approx(anchor_col, abs=0.02)

    def test_kl_to_anchor_decreases_along_the_lambda_grid(self) -> None:
        """The tradeoff curve's monotone half: larger lam means smaller KL.

        Game, anchors and T are all held fixed, so lambda is the only thing
        moving.
        """
        game = rock_paper_scissors()
        anchor = (0.6, 0.3, 0.1)
        kls = []
        for lam in (0.1, 0.3, 1.0, 3.0, 10.0):
            row = AnchoredHedge(anchor, lam=lam)
            col = AnchoredHedge(anchor, lam=lam)
            self_play(game, row, col, 1500)
            kls.append(_kl(row.average_strategy, anchor))
        for tighter, looser in zip(kls[1:], kls[:-1], strict=True):
            assert tighter <= looser + 1e-6
        assert kls[-1] < 0.02  # lam = 10 is essentially anchored
        assert kls[0] > 0.1  # lam = 0.1 is essentially unanchored


class TestQuantalResponseFixedPoint:
    """Converged play must be the fixed point the theory names."""

    def test_mean_play_is_an_anchored_quantal_response(self) -> None:
        """At convergence ``pi(a)`` is proportional to ``tau(a)*exp(Qbar(a)/lam)``.

        The mean strategies are checked against the anchored quantal response of
        *independently computed* expected utilities -- MatrixGame arithmetic,
        never the learner's own state (docs/pikl.tex Sec. 3).
        """
        game = MatrixGame.zero_sum(("A", "B"), ("L", "R"), ((0.9, 0.2), (0.1, 0.8)))
        lam = 0.7
        row = AnchoredHedge((0.5, 0.5), lam=lam)
        col = AnchoredHedge((0.5, 0.5), lam=lam)
        self_play(game, row, col, 6000)
        row_mean, col_mean = row.average_strategy, col.average_strategy
        aqr_row = anchored_quantal_response(game.row_utility_vector(col_mean), (0.5, 0.5), lam)
        aqr_col = anchored_quantal_response(game.col_utility_vector(row_mean), (0.5, 0.5), lam)
        assert row_mean == pytest.approx(aqr_row, abs=0.02)
        assert col_mean == pytest.approx(aqr_col, abs=0.02)


class TestRunPiklValidation:
    """The orchestrator's argument guards."""

    def test_rejects_bad_arguments(self) -> None:
        oracle = _matrix_oracle(matching_pennies())
        players = [PiklPlayer((0.5, 0.5), 0.1), PiklPlayer((0.5, 0.5), 0.1)]
        with pytest.raises(ValueError, match="iterations"):
            run_pikl(players, oracle, iterations=0)
        with pytest.raises(ValueError, match="players"):
            run_pikl([], oracle, iterations=1)
        with pytest.raises(ValueError, match="max_expected_profiles"):
            run_pikl(players, oracle, iterations=1, mode="expected", max_expected_profiles=3)

    def test_rejects_misbehaving_oracle(self) -> None:
        players = [PiklPlayer((0.5, 0.5), 0.1), PiklPlayer((0.5, 0.5), 0.1)]

        def short(profiles: list[tuple[int, ...]]) -> list[tuple[float, float]]:
            return [(0.0, 0.0)] * (len(profiles) - 1)

        with pytest.raises(ValueError, match="utility_fn returned"):
            run_pikl(players, short, iterations=1, mode="expected")  # type: ignore[arg-type]

        def narrow(profiles: list[tuple[int, ...]]) -> list[tuple[float]]:
            return [(0.0,) for _ in profiles]

        with pytest.raises(ValueError, match="must score all 2 players"):
            run_pikl(players, narrow, iterations=1, mode="expected")  # type: ignore[arg-type]
