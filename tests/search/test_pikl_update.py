"""Hand-computed pins of the piKL-Hedge update rule (docs/pikl.tex Sec. 8 item 1).

Every expectation here is derived by hand from Jacob et al. 2022, Algorithm 1
(eq. 6) -- transcribed in the comments -- and evaluated inline with ``math.exp``
/ ``math.log`` only. Nothing below calls the implementation to produce an
expected value, so a sign/indexing/denominator slip in the implementation
cannot cancel out of the test.
"""

from __future__ import annotations

import math

import pytest

from entente.gametheory.regret import Hedge
from entente.search.pikl import (
    AnchoredHedge,
    anchored_quantal_response,
)


def _norm2(a: float, b: float) -> tuple[float, float]:
    return a / (a + b), b / (a + b)


class TestHandComputedConstantEta:
    """Two iterations of the constant-rate schedule, worked out by hand.

    Constant ``eta = 1``, ``lam = 1/2``, anchor ``tau = (0.8, 0.2)``, and the
    fixed utility stream ``u1 = (0.9, 0.1)``, ``u2 = (0.2, 0.4)``.

    Eq. (6) at iteration t (1-indexed), constant eta:

        pi_t(a) proportional to exp[ (eta*CV_{t-1}(a) + t*lam*eta*log tau(a)) / (1 + t*lam*eta) ]

    t = 1:  CV_0 = 0, pull = 1*(1/2)*1 = 1/2, denom = 3/2
            pi_1(a) proportional to exp[ (1/2)*log tau(a) / (3/2) ] = tau(a)^(1/3)
    t = 2:  CV_1 = u1, pull = 2*(1/2)*1 = 1, denom = 2
            pi_2(a) proportional to exp[ (CV_1(a) + log tau(a)) / 2 ]
                    = tau(a)^(1/2) * exp(CV_1(a)/2)

    The mean strategy is (pi_1 + pi_2)/2 -- uniform weights, paper eq. (7).
    """

    def test_two_iterations(self) -> None:
        learner = AnchoredHedge((0.8, 0.2), lam=0.5, eta=1.0)
        u1 = (0.9, 0.1)
        u2 = (0.2, 0.4)

        played_1 = learner.observe_utility(u1)
        played_2 = learner.observe_utility(u2)

        # t = 1: pi_1 proportional to tau^(1/3)
        pi_1 = _norm2(0.8 ** (1.0 / 3.0), 0.2 ** (1.0 / 3.0))
        # t = 2: pi_2 proportional to sqrt(tau) * exp(u1/2)  (CV_1 = u1)
        pi_2 = _norm2(
            math.sqrt(0.8) * math.exp(0.9 / 2.0),
            math.sqrt(0.2) * math.exp(0.1 / 2.0),
        )
        assert played_1 == pytest.approx(pi_1, abs=1e-12)
        assert played_2 == pytest.approx(pi_2, abs=1e-12)
        mean = [(a + b) / 2.0 for a, b in zip(pi_1, pi_2, strict=True)]
        assert learner.average_strategy == pytest.approx(mean, abs=1e-12)
        # CV sums the raw utility vectors, nothing else.
        assert learner.cumulative_values == pytest.approx([1.1, 0.5], abs=1e-12)
        assert learner.iterations == 2

    def test_third_iterate_uses_cv2(self) -> None:
        """The *next* strategy must read ``CV_2 = u1 + u2`` at ``t = 3``.

        That gives ``pull = 3/2`` and ``denom = 5/2``, so ``pi_3(a)`` is
        proportional to ``exp[(CV_2(a) + (3/2)*log tau(a)) / (5/2)]``.
        """
        learner = AnchoredHedge((0.8, 0.2), lam=0.5, eta=1.0)
        learner.observe_utility((0.9, 0.1))
        learner.observe_utility((0.2, 0.4))
        pi_3 = _norm2(
            math.exp((1.1 + 1.5 * math.log(0.8)) / 2.5),
            math.exp((0.5 + 1.5 * math.log(0.2)) / 2.5),
        )
        assert learner.strategy == pytest.approx(pi_3, abs=1e-12)


class TestHandComputedAdaptiveEta:
    """Three iterations of the adaptive schedule, worked out by hand.

    The schedule is ``eta_t = c/(sigma_t*sqrt(t))`` with ``c = 2``, ``lam = 1``,
    ``tau = (0.5, 0.5)``, and utilities ``u1 = (0.9, 0.1)``,
    ``u2 = (0.2, 0.4)``, ``u3 = (0.5, 0.5)``.

    The sigma reading this pins (docs/pikl.tex Sec. 7): the population standard
    deviation of the past per-iteration expected utilities ``<pi_s, u_s>``,
    with ``sigma := 1`` while fewer than two iterations exist, or the spread is
    zero.

    t = 1: sigma = 1 (no history), eta_1 = 2. pull = 1*1*2 = 2, denom = 3:
           pi_1(a) proportional to exp[2*log(1/2)/3] -- equal scores, so pi_1 = (1/2, 1/2).
    t = 2: sigma = 1 (one sample), eta_2 = 2/sqrt(2) = sqrt(2).
           pull = 2*1*sqrt(2), denom = 1 + 2*sqrt(2):
           pi_2(a) proportional to exp[(sqrt(2)*u1(a) + 2*sqrt(2)*log(1/2)) / (1 + 2*sqrt(2))]
    t = 3: v1 = <pi_1, u1> = 0.5, v2 = <pi_2, u2>,
           sigma = pstdev([v1, v2]) = |v2 - v1|/2, eta_3 = 2/(sigma*sqrt(3)).
           pull = 3*eta_3, denom = 1 + 3*eta_3:
           pi_3(a) proportional to exp[(eta_3*CV_2(a) + 3*eta_3*log(1/2)) / (1 + 3*eta_3)]
    """

    def test_three_iterations(self) -> None:
        learner = AnchoredHedge((0.5, 0.5), lam=1.0, eta=None, adaptive_scale=2.0)
        u1, u2, u3 = (0.9, 0.1), (0.2, 0.4), (0.5, 0.5)

        played_1 = learner.observe_utility(u1)
        assert played_1 == pytest.approx((0.5, 0.5), abs=1e-12)

        eta_2 = 2.0 / math.sqrt(2.0)
        denom_2 = 1.0 + 2.0 * eta_2
        pi_2 = _norm2(
            math.exp((eta_2 * 0.9 + 2.0 * eta_2 * math.log(0.5)) / denom_2),
            math.exp((eta_2 * 0.1 + 2.0 * eta_2 * math.log(0.5)) / denom_2),
        )
        played_2 = learner.observe_utility(u2)
        assert played_2 == pytest.approx(pi_2, abs=1e-12)

        v1 = 0.5 * 0.9 + 0.5 * 0.1
        v2 = pi_2[0] * 0.2 + pi_2[1] * 0.4
        sigma = abs(v2 - v1) / 2.0  # population stdev of two samples
        eta_3 = 2.0 / (sigma * math.sqrt(3.0))
        denom_3 = 1.0 + 3.0 * eta_3
        cv2 = (0.9 + 0.2, 0.1 + 0.4)
        pi_3 = _norm2(
            math.exp((eta_3 * cv2[0] + 3.0 * eta_3 * math.log(0.5)) / denom_3),
            math.exp((eta_3 * cv2[1] + 3.0 * eta_3 * math.log(0.5)) / denom_3),
        )
        played_3 = learner.observe_utility(u3)
        assert played_3 == pytest.approx(pi_3, abs=1e-12)

        mean = [(a + b + c) / 3.0 for a, b, c in zip(played_1, pi_2, pi_3, strict=True)]
        assert learner.average_strategy == pytest.approx(mean, abs=1e-12)


class TestLimits:
    """The two limits of the anchor strength, each against an independent answer."""

    def test_lambda_zero_is_exactly_hedge(self) -> None:
        """``lam = 0`` must reproduce ``gametheory.Hedge`` step for step.

        The two are independent implementations of one special case
        (docs/pikl.tex Sec. 4), and at this limit the anchor must make no
        difference at all.
        """
        anchored = AnchoredHedge((0.7, 0.2, 0.1), lam=0.0, eta=0.5)
        plain = Hedge(3, eta=0.5)
        stream = [
            (0.9, 0.1, 0.4),
            (0.2, 0.8, 0.3),
            (0.5, 0.5, 0.9),
            (0.1, 0.3, 0.2),
            (0.7, 0.6, 0.5),
        ]
        for utility in stream:
            a = anchored.observe_utility(utility)
            b = plain.observe_utility(utility)
            assert a == pytest.approx(b, abs=1e-12)
        assert anchored.average_strategy == pytest.approx(plain.average_strategy, abs=1e-12)

    def test_lambda_infinity_is_exactly_the_anchor(self) -> None:
        anchor = (0.6, 0.3, 0.1)
        learner = AnchoredHedge(anchor, lam=math.inf, eta=1.0)
        for utility in [(5.0, -3.0, 0.0), (0.0, 100.0, -7.0)]:
            assert learner.observe_utility(utility) == pytest.approx(anchor, abs=1e-15)
        assert learner.average_strategy == pytest.approx(anchor, abs=1e-15)
        kl = sum(p * math.log(p / q) for p, q in zip(learner.average_strategy, anchor, strict=True))
        assert kl == pytest.approx(0.0, abs=1e-12)

    def test_large_lambda_first_iterate_tempers_toward_anchor(self) -> None:
        """At ``t = 1`` the iterate is ``tau^(pull/(1+pull))``, renormalized.

        That is a *tempered* anchor, approaching ``tau`` as lambda grows, and
        checking it by hand covers the finite-lambda anchoring path rather than
        only the infinite limit.
        """
        anchor = (0.8, 0.2)
        learner = AnchoredHedge(anchor, lam=4.0, eta=1.0)
        pull = 1.0 * 4.0 * 1.0
        expected = _norm2(0.8 ** (pull / (1.0 + pull)), 0.2 ** (pull / (1.0 + pull)))
        assert learner.strategy == pytest.approx(expected, abs=1e-12)


class TestValidation:
    """What the learner refuses at construction and at each update."""

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="at least one action"):
            AnchoredHedge((), lam=1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            AnchoredHedge((0.5, 0.0), lam=1.0)
        with pytest.raises(ValueError, match="lam"):
            AnchoredHedge((0.5, 0.5), lam=-0.1)
        with pytest.raises(ValueError, match="lam"):
            AnchoredHedge((0.5, 0.5), lam=math.nan)
        with pytest.raises(ValueError, match="eta"):
            AnchoredHedge((0.5, 0.5), lam=1.0, eta=0.0)
        with pytest.raises(ValueError, match="adaptive_scale"):
            AnchoredHedge((0.5, 0.5), lam=1.0, adaptive_scale=0.0)
        learner = AnchoredHedge((0.5, 0.5), lam=1.0, eta=1.0)
        with pytest.raises(ValueError, match="expected 2 utilities"):
            learner.observe_utility((1.0, 2.0, 3.0))

    def test_anchor_renormalized(self) -> None:
        learner = AnchoredHedge((4.0, 1.0), lam=math.inf)
        assert learner.strategy == pytest.approx((0.8, 0.2), abs=1e-12)


class TestSampleIndex:
    """Drawing an index, including the floating-point shortfall at the tail."""

    def test_cumulative_walk_and_fp_fallback(self) -> None:
        import random as _random

        from entente.search.pikl import sample_index

        rng = _random.Random(0)
        counts = [0, 0]
        for _ in range(2000):
            counts[sample_index([0.25, 0.75], rng)] += 1
        assert 400 < counts[0] < 600  # ~= 0.25 of 2000, seeded
        # degenerate all-zero mass: the walk never crosses, last index absorbs
        assert sample_index([0.0, 0.0, 0.0], _random.Random(1)) == 2


class TestAnchoredQuantalResponse:
    """The fixed point's closed form, against a softmax computed by hand."""

    def test_matches_hand_softmax(self) -> None:
        """``AQR(u, tau, lam)`` must be proportional to ``tau(a)*exp(u(a)/lam)``."""
        got = anchored_quantal_response((0.6, 0.2), (0.3, 0.7), 2.0)
        w0 = 0.3 * math.exp(0.6 / 2.0)
        w1 = 0.7 * math.exp(0.2 / 2.0)
        assert got == pytest.approx((w0 / (w0 + w1), w1 / (w0 + w1)), abs=1e-12)

    def test_rejects_bad_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive and finite"):
            anchored_quantal_response((0.5,), (1.0,), 0.0)
        with pytest.raises(ValueError, match="positive and finite"):
            anchored_quantal_response((0.5,), (1.0,), math.inf)
        with pytest.raises(ValueError, match="equal length"):
            anchored_quantal_response((0.5, 0.5), (1.0,), 1.0)
        with pytest.raises(ValueError, match="strictly positive"):
            anchored_quantal_response((0.5,), (0.0,), 1.0)
