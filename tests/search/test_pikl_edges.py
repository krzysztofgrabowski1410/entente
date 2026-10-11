"""Two latent edges of :mod:`entente.search.pikl` that broke its own contract.

Both sit at the far end of the float range, where no committed configuration
goes, so both stayed silent: a band of huge *finite* ``lam`` whose anchor pull
does not overflow while its product with ``log tau`` does (the iterate turned
into a wrong strategy or into NaN), and a ``reweight_anchor`` bias of zero that
rounded a tiny anchor entry to exactly zero (an anchor ``AnchoredHedge``
refuses).

The invariants under protection are the ones docs/pikl.tex states: a finite
``lam`` so large that the exponent overflows plays the anchor, as ``lam = inf``
does (Sec. 4), and the output of the reweighting is again a strictly positive
anchor, and the anchor itself under an all-zero bias (Sec. 5).
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest

from entente.search.pikl import AnchoredHedge, PiklPlayer, reweight_anchor, run_pikl


def _assert_plays_anchor(got: list[float], anchor: tuple[float, ...]) -> None:
    assert all(math.isfinite(p) for p in got), got
    assert got == pytest.approx(anchor, rel=1e-9, abs=0.0)


class TestHugeFiniteLambdaBand:
    """``t * lam * eta`` is finite, but ``t * lam * eta * log tau`` overflows."""

    @pytest.mark.parametrize(
        ("anchor", "lam"),
        [
            ((0.99, 0.01), 1e308),
            ((0.5, 0.3, 0.2), 1.5e308),
            ((0.25, 0.25, 0.25, 0.25), 1.6e308),
        ],
    )
    def test_first_iterate_is_the_anchor(self, anchor: tuple[float, ...], lam: float) -> None:
        """Every score overflows or some do; either way the limit is the anchor.

        Before the repair the first anchor lost its second candidate (the
        product with ``log 0.01`` overflowed to ``-inf``, probability 0), and
        the uniform one turned into NaN (every score ``-inf``, then
        ``exp(-inf - -inf)``).
        """
        learner = AnchoredHedge(anchor, lam=lam, eta=1.0)
        _assert_plays_anchor(learner.strategy, anchor)

    def test_just_below_the_band_is_unchanged(self) -> None:
        """A pull whose products still fit a double is computed, not short-cut."""
        anchor = (0.99, 0.01)
        learner = AnchoredHedge(anchor, lam=3.8e307, eta=1.0)
        _assert_plays_anchor(learner.strategy, anchor)

    @pytest.mark.parametrize("lam", [1e290, 1e300, 1e305, 2.6e305, 1e306, 1e307, 1e308, 1.7e308])
    def test_no_nan_anywhere_on_the_way_to_the_overflow(self, lam: float) -> None:
        """The strategy is continuous in lam up to the overflow, then the anchor."""
        anchor = (0.7, 0.2, 0.09, 0.01)
        learner = AnchoredHedge(anchor, lam=lam, eta=1.0)
        for utility in ((1.0, 0.0, 0.5, 0.25), (0.0, 1.0, 0.5, 0.75)):
            _assert_plays_anchor(learner.strategy, anchor)
            learner.observe_utility(utility)
        _assert_plays_anchor(learner.average_strategy, anchor)

    def test_adaptive_schedule_survives_the_band(self) -> None:
        """A NaN strategy must never reach ``pstdev``, which would crash on the third step."""
        anchor = (0.25, 0.25, 0.25, 0.25)
        learner = AnchoredHedge(anchor, lam=5e307)
        for step in range(6):
            _assert_plays_anchor(learner.strategy, anchor)
            learner.observe_utility((0.1 * step, 1.0, 0.0, 0.5))
        _assert_plays_anchor(learner.average_strategy, anchor)

    def test_a_single_candidate_stays_certain(self) -> None:
        """``log tau = 0``: no product to overflow, and ``inf * 0`` must not be formed."""
        for lam in (1e308, math.inf):
            learner = AnchoredHedge((1.0,), lam=lam, eta=1.0)
            learner.observe_utility((0.5,))
            learner.observe_utility((0.5,))
            assert learner.strategy == [1.0]

    def test_run_pikl_mean_is_the_anchor(self) -> None:
        """The orchestrator's mean, the object the guarantees attach to, is finite."""
        anchor = (0.25, 0.25, 0.25, 0.25)
        players = [PiklPlayer(anchor=anchor, lam=1.6e308)] * 2

        def oracle(profiles: object) -> list[tuple[float, float]]:
            return [(0.1 * p[0], 0.1 * p[1]) for p in profiles]  # type: ignore[attr-defined]

        for mode in ("sampled", "expected"):
            result = run_pikl(
                players,
                oracle,
                iterations=8,
                mode=mode,  # type: ignore[arg-type]
                eta=1.0,
                rng=random.Random(3),
            )
            for mean in result.mean_strategies:
                _assert_plays_anchor(list(mean), anchor)


class TestReweightAnchorTinyEntries:
    """A bias of zero keeps the anchor strictly positive, whatever the entry sizes."""

    @pytest.mark.parametrize("tiny", [1e-24, 2.4e-24, 1e-100, 1e-305, 5e-324])
    @pytest.mark.parametrize("bias", [(1.0, 0.0), (0.0, 0.0), (4.0, 0.0)])
    def test_result_is_strictly_positive_and_a_valid_anchor(
        self, tiny: float, bias: tuple[float, float]
    ) -> None:
        """``tiny * 1e-300`` is 0.0 in a double; the result must not contain it."""
        result = reweight_anchor((1.0, tiny), bias)
        assert all(p > 0.0 for p in result)
        assert math.isclose(sum(result), 1.0)
        AnchoredHedge(result, lam=1.0)  # must not raise
        AnchoredHedge(result, lam=math.inf)  # must not raise

    def test_a_tiny_positive_bias_cannot_zero_an_entry_either(self) -> None:
        """The product ``anchor * bias`` underflows for a positive bias as well."""
        result = reweight_anchor((1.0, 1e-200), (1.0, 1e-200))
        assert all(p > 0.0 for p in result)

    def test_a_huge_boost_cannot_zero_the_others(self) -> None:
        """Dividing by a total of 1e100 must not round the floor to exactly 0."""
        result = reweight_anchor((0.5, 0.5), (1e100, 0.0))
        assert all(p > 0.0 for p in result)

    @pytest.mark.parametrize(("boost", "floored"), [(1e10, 1e-210), (1e20, 1e-220), (1e25, 1e-225)])
    def test_a_huge_boost_on_a_small_mass_keeps_the_floor_exact(
        self, boost: float, floored: float
    ) -> None:
        """The zero-bias weight ``1e-300 / boost`` must not be formed as a float.

        The boosted candidate holds ``1e-100``, so the floored one ends at
        ``1e-300 / (1e-100 * boost)``, a normal float -- but the weight
        ``1e-300 / boost`` alone is subnormal (``1e-310``, ``1e-320``) or
        zero (``1e-325``), and dividing the biases by the largest first lost
        its digits, down to ``5e-324`` for a boost of ``1e25``.
        """
        result = reweight_anchor((1e-100, 1.0), (boost, 0.0))
        assert result[1] == pytest.approx(floored, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("seed", range(4))
    def test_agrees_with_exact_arithmetic_to_two_ulps(self, seed: int) -> None:
        """Anchor entries down to subnormal, biases from 0 to 1e300, against Fractions.

        The reference is the exact tilt with the floor, rounded once to a
        float and clamped at the smallest positive float.
        """
        rng = random.Random(seed)
        for _ in range(250):
            size = rng.randint(1, 6)
            anchor = [10.0 ** rng.uniform(-320.0, 0.0) for _ in range(size)]
            bias = [
                0.0 if rng.random() < 0.3 else 10.0 ** rng.uniform(-310.0, 300.0) for _ in anchor
            ]
            exact = [
                Fraction(a) * Fraction(max(b, 1e-300)) for a, b in zip(anchor, bias, strict=True)
            ]
            total = sum(exact)
            expected = [max(float(x / total), math.ulp(0.0)) for x in exact]
            result = reweight_anchor(anchor, bias)
            for got, want in zip(result, expected, strict=True):
                assert abs(got - want) <= 2 * math.ulp(want), (anchor, bias, result, expected)

    @pytest.mark.parametrize("tiny", [1e-15, 1e-20, 3e-24, 1e-100, 1e-300])
    def test_all_zero_bias_gives_the_anchor_back_unchanged(self, tiny: float) -> None:
        """Every candidate at the floor is a common factor, which cancels exactly.

        Multiplying a small entry by the floor first lost its low bits in the
        denormal range (a relative error of 1.5e-9 at 1e-15), and 3e-24 came
        back as 4.9e-24.
        """
        anchor = (0.6, 0.4 - tiny, tiny)
        result = reweight_anchor(anchor, (0.0, 0.0, 0.0))
        assert result == pytest.approx(anchor, rel=1e-12, abs=0.0)

    @pytest.mark.parametrize("anchor", [(0.5, 0.3, 0.2), (0.5, 0.5, 5e-308)])
    def test_a_power_of_two_bias_is_bit_exact(self, anchor: tuple[float, float, float]) -> None:
        """The production bias (4.0 or 1.0) gives the same bits as the plain formula.

        The second anchor holds an entry whose quarter is subnormal: scaling it
        by ``1 / 4`` before the division would lose its low bits.
        """
        bias = (4.0, 1.0, 1.0)
        tilted = [a * b for a, b in zip(anchor, bias, strict=True)]
        total = sum(tilted)
        assert reweight_anchor(anchor, bias) == [w / total for w in tilted]
