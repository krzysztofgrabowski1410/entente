"""Numerical robustness of :mod:`entente.search` at the edges of the float range.

Each case is one way arithmetic corrupts a search silently rather than loudly:
a NaN slips past a ``<= 0.0`` guard, ``exp`` underflows to exactly zero past
about -745 nats, and ``t * lam * eta`` overflows to infinity for a huge finite
``lam``. None of those raises on its own; each one poisons a strategy.

The invariants under protection: anchors stay strictly positive, non-finite
numbers are rejected loudly at every seam, and the ``lam -> inf`` family of
limits keeps returning the anchor exactly.
"""

from __future__ import annotations

import math
import random

import pytest

from entente.engine.game import Game
from entente.search.candidates import CandidatePolicy, ScoredOrderSet, collect_candidates
from entente.search.pikl import (
    AnchoredHedge,
    PiklPlayer,
    anchored_quantal_response,
    reweight_anchor,
    run_pikl,
)

NAN = float("nan")
INF = float("inf")


class TestAnchorValidation:
    """The anchor a learner will accept, and the values it must refuse."""

    @pytest.mark.parametrize("bad", [NAN, INF, 0.0, -0.5])
    def test_constructor_rejects_non_positive_or_non_finite_anchor(self, bad: float) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            AnchoredHedge((bad, 0.5), lam=1.0)

    @pytest.mark.parametrize("bad", [NAN, INF, 0.0, -1.0])
    def test_constant_eta_must_be_positive_finite(self, bad: float) -> None:
        with pytest.raises(ValueError, match="eta"):
            AnchoredHedge((0.5, 0.5), lam=1.0, eta=bad)

    def test_quantal_response_rejects_nan_anchor(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            anchored_quantal_response((1.0, 0.0), (NAN, 0.5), lam=1.0)


class TestUtilityValidation:
    """A non-finite utility must be caught before it can enter the state."""

    def test_observe_utility_rejects_nan(self) -> None:
        learner = AnchoredHedge((0.6, 0.4), lam=1.0, eta=0.5)
        with pytest.raises(ValueError, match="finite"):
            learner.observe_utility((NAN, 1.0))

    def test_run_pikl_rejects_non_finite_oracle_values(self) -> None:
        players = [PiklPlayer(anchor=(0.5, 0.5), lam=0.3)] * 2

        def bad_oracle(profiles: object) -> list[tuple[float, float]]:
            return [(NAN, 1.0) for _ in profiles]  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="non-finite"):
            run_pikl(players, bad_oracle, iterations=3, eta=0.5)


class TestHugeLambda:
    """A lambda large enough to overflow the anchor pull must still play the anchor."""

    def test_huge_finite_lam_behaves_as_the_infinite_limit(self) -> None:
        """An overflowing ``t * lam * eta`` must give the anchor, never a NaN strategy.

        The mathematical limit of a huge pull is the ``lam = inf`` answer, so
        that is what an overflow has to resolve to.
        """
        anchor = (0.8, 0.2)
        learner = AnchoredHedge(anchor, lam=1e308, eta=1.0)
        learner.observe_utility((1.0, 0.0))
        learner.observe_utility((1.0, 0.0))
        for got, want in zip(learner.strategy, anchor, strict=True):
            assert math.isclose(got, want, rel_tol=1e-12)
        assert all(math.isfinite(p) for p in learner.average_strategy)


class TestReweightAnchor:
    """The reweighting hook's output must remain a usable anchor, always."""

    def test_rejects_non_finite_bias(self) -> None:
        for bad in (NAN, INF):
            with pytest.raises(ValueError, match="finite"):
                reweight_anchor((0.6, 0.4), (bad, 1.0))

    def test_rejects_non_positive_anchor(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            reweight_anchor((0.0, 1.0), (1.0, 1.0))

    def test_zero_bias_keeps_output_strictly_positive(self) -> None:
        """A zeroed candidate is floored, never eliminated.

        ``AnchoredHedge`` takes the log of every anchor entry, so it must keep
        accepting the reweighted anchor -- including at the ``lam = inf`` limit.
        """
        reweighted = reweight_anchor((0.6, 0.4), (0.0, 1.0))
        assert all(p > 0.0 for p in reweighted)
        assert math.isclose(sum(reweighted), 1.0)
        pinned = AnchoredHedge(reweighted, lam=INF)
        assert pinned.strategy == pytest.approx(reweighted)

    def test_all_zero_bias_renormalizes_back_to_the_anchor(self) -> None:
        anchor = (0.6, 0.4)
        assert reweight_anchor(anchor, (0.0, 0.0)) == pytest.approx(anchor)


class _LopsidedPolicy(CandidatePolicy):
    """Two legal hold-sheets whose scores differ by more than exp underflow."""

    def __init__(self, game: Game, power: str, gap: float) -> None:
        from entente.engine.orders import Hold, Move

        legal = game.legal_orders(power)
        holds = tuple(o for o in legal if isinstance(o, Hold))
        move = next(o for o in legal if isinstance(o, Move) and not o.via_convoy)
        alternative = (
            *(h for h in holds if h.location.province != move.location.province),
            move,
        )
        self._sets = [
            ScoredOrderSet(orders=holds, log_prob=0.0),
            ScoredOrderSet(orders=alternative, log_prob=-gap),
        ]

    def sample_order_sets(
        self,
        game: Game,
        power: str,
        count: int,
        *,
        include_greedy: bool = True,
        rng: object = None,
    ) -> list[ScoredOrderSet]:
        return [self._sets[i % 2] for i in range(count)]


class TestCandidateAnchorUnderflow:
    """A log-probability gap wide enough to underflow ``exp`` to zero."""

    def test_huge_log_prob_gap_yields_strictly_positive_anchor(self) -> None:
        game = Game.new("standard", "standard")
        policy = _LopsidedPolicy(game, "FRANCE", gap=800.0)
        candidates = collect_candidates(policy, game, "FRANCE", k=2, rng=random.Random(0))
        assert all(p > 0.0 for p in candidates.anchor)
        AnchoredHedge(candidates.anchor, lam=1.0)  # must not raise

    def test_non_finite_log_prob_rejected_loudly(self) -> None:
        game = Game.new("standard", "standard")
        policy = _LopsidedPolicy(game, "FRANCE", gap=800.0)
        policy._sets[1] = ScoredOrderSet(orders=policy._sets[1].orders, log_prob=-INF)
        with pytest.raises(ValueError, match="non-finite"):
            collect_candidates(policy, game, "FRANCE", k=2, rng=random.Random(0))
