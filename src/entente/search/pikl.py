"""piKL-Hedge: no-regret learning KL-anchored to a fixed policy.

Implements docs/pikl.tex Sec. 3-Sec. 4, restated from Jacob et al., *Modeling Strong
and Human-Like Gameplay with KL-Regularized Search* (ICML 2022,
arXiv:2112.07544), Algorithm 1: each player keeps cumulative action values
``CV`` and plays, at iteration ``t`` (1-indexed) with learning rate ``eta_t``
and anchor strength ``lam``,

    pi_t(a)  proportional to  exp[ (eta_t*CV(a) + t*lam*eta_t*log tau(a)) / (1 + t*lam*eta_t) ]

then adds the utility every one of its actions would have earned against the
other players' joint action into ``CV``. The returned policy is the **mean**
of the iterates (paper eq. 7) -- the object the paper's guarantees speak about:
``KL(mean || tau) <= (regret/T + D)/lam`` (their Theorem 1) and, in two-player
zero-sum games, mean play is a ``max_i(lam_i*beta_i)``-approximate Nash
equilibrium with ``beta_i = max_a log(1/tau_i(a))`` (their Theorem 2).

Limits: ``lam = 0`` is exactly plain :class:`entente.gametheory.regret.Hedge`
(the softmax cancels the action-independent baseline between cumulative value
and cumulative regret); ``lam = inf`` plays the anchor itself at every
iteration, and so does a finite ``lam`` so large that the scores would
overflow a double (the limit is the same). ``lam`` is scale-coupled to the
utilities (paper Appendix B); the Diplomacy path feeds Sum-of-Squares shares in
``[0, 1]``.

:class:`AnchoredHedge` satisfies :class:`entente.gametheory.regret.Learner`,
so :func:`entente.gametheory.normal_form.self_play` can drive it unchanged --
the validation suite (tests/search/) exploits that to check the update against
hand-computed iterates, known matrix-game equilibria, and Kuhn poker before
any Diplomacy use. :func:`run_pikl` is the n-player orchestrator over a joint
utility oracle, in the paper's sampled-opponents mode or an exact
expected-update mode for small games.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from statistics import pstdev
from typing import TYPE_CHECKING, Literal, TypeAlias

from entente.gametheory.regret import normalize

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

ADAPTIVE_ETA_SCALE = 10.0 / 3.0
"""Default scale ``c`` of the adaptive learning rate -- the value Jacob et al.
report works well for no-press Diplomacy (their Appendix H)."""

_SIGMA_FLOOR = 1e-12
"""Below this, the utility spread is treated as zero and the adaptive rate
falls back to ``c / sqrt(t)`` (see :meth:`AnchoredHedge._eta`)."""
_BIAS_FLOOR = 1e-300
"""A bias below this, zero included, counts as this in :func:`reweight_anchor`:
a zero-bias candidate keeps this fraction of its anchor mass -- the output must
stay a *strictly positive* distribution (an AnchoredHedge precondition;
KL(pi || tau) needs tau > 0)."""
_SMALLEST_MASS = math.ulp(0.0)
"""The smallest positive float, ``5e-324``. Where the floor fraction of a tiny
anchor entry is not representable, :func:`reweight_anchor` gives that
entry this mass instead of letting it round to zero."""


def _softmax(scores: Sequence[float]) -> list[float]:
    """Return a max-subtracted softmax of ``scores``.

    Subtracting the maximum keeps the exponent finite even for the linearly
    growing ``lam = 0`` case (docs/pikl.tex Sec. 3).
    """
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    return normalize(weights)


class AnchoredHedge:
    """One player's piKL-Hedge state (docs/pikl.tex Sec. 3; paper Algorithm 1).

    ``anchor`` must be strictly positive (its log enters the update; a
    zero-probability candidate could never be played anyway) and is
    renormalized on ingestion. ``lam`` is the anchor strength in utility
    units; ``lam = 0`` recovers plain Hedge and ``lam = math.inf`` plays the
    anchor exactly. ``eta`` fixes a constant learning rate; ``None`` selects
    the paper's adaptive schedule ``eta_t = c / (sigma_t * sqrt(t))`` where
    ``sigma_t`` is the standard deviation across past iterations of the
    utility the player's strategy earned. The paper states that schedule in
    one sentence (Appendix H); the reading taken here is: population standard
    deviation over the per-iteration expected utilities ``<pi_s, u_s>``,
    ``s < t``, with ``sigma_t := 1`` while fewer than two iterations exist or
    the spread is numerically zero.

    Satisfies the :class:`entente.gametheory.regret.Learner` protocol:
    :attr:`strategy` is this iteration's policy, :meth:`observe_utility`
    consumes the per-action utility vector (against the opponents' current
    play -- sampled or expected, the caller's choice), :attr:`average_strategy`
    is the mean iterate the guarantees attach to.
    """

    def __init__(
        self,
        anchor: Sequence[float],
        *,
        lam: float,
        eta: float | None = None,
        adaptive_scale: float = ADAPTIVE_ETA_SCALE,
    ) -> None:
        if len(anchor) == 0:
            raise ValueError("anchor must have at least one action")
        if any(not math.isfinite(p) or p <= 0.0 for p in anchor):
            raise ValueError("anchor probabilities must be strictly positive (and finite)")
        if math.isnan(lam) or lam < 0.0:
            raise ValueError("lam must be >= 0 (inf allowed)")
        if eta is not None and (not math.isfinite(eta) or eta <= 0.0):
            raise ValueError("eta must be > 0 and finite when constant")
        if not math.isfinite(adaptive_scale) or adaptive_scale <= 0.0:
            raise ValueError("adaptive_scale must be > 0 and finite")
        self.anchor = normalize(anchor)
        self.lam = lam
        self.num_actions = len(self.anchor)
        self._log_anchor = [math.log(p) for p in self.anchor]
        # The paper's beta, max_a log(1/tau(a)): the largest |log tau(a)| a pull is multiplied by.
        self._beta = -min(self._log_anchor)
        self._constant_eta = eta
        self._adaptive_scale = adaptive_scale
        self._cv = [0.0] * self.num_actions
        self._strategy_sum = [0.0] * self.num_actions
        self._experienced: list[float] = []
        self._iterations = 0

    @property
    def iterations(self) -> int:
        """Completed :meth:`observe_utility` updates."""
        return self._iterations

    @property
    def cumulative_values(self) -> list[float]:
        """``CV`` -- the summed per-action utilities observed so far."""
        return list(self._cv)

    def _eta(self, t: int) -> float:
        """Learning rate for 1-indexed iteration ``t`` (class docstring)."""
        if self._constant_eta is not None:
            return self._constant_eta
        sigma = pstdev(self._experienced) if len(self._experienced) >= 2 else 0.0
        if sigma <= _SIGMA_FLOOR:
            sigma = 1.0
        return self._adaptive_scale / (sigma * math.sqrt(t))

    @property
    def strategy(self) -> list[float]:
        """Return this iteration's policy, paper eq. (6) at ``t = iterations + 1``.

        It is built from ``CV^(t-1)``, and is the anchor itself when
        ``lam = inf`` -- and when a finite ``lam`` is so large that the scores
        would overflow a double, the same limit.
        """
        if math.isinf(self.lam):
            return list(self.anchor)
        t = self._iterations + 1
        eta_t = self._eta(t)
        pull = t * self.lam * eta_t
        if math.isinf(pull) or math.isinf(pull * self._beta):
            # Huge-but-finite lam (or many iterations of it) overflows the
            # anchor weight, or -- a little below that -- its product with
            # log tau(a) for the least likely candidate. Either way the
            # mathematical limit is the lam = inf answer: the anchor weight
            # pull / (1 + pull) is 1 to the last bit, and the CV term is
            # divided by a pull of at least 1e305. Return it rather than let
            # inf/inf, or a score that overflowed to -inf, turn the strategy
            # into NaN or a candidate into probability 0.
            return list(self.anchor)
        denom = 1.0 + pull
        scores = [
            (eta_t * cv + pull * log_tau) / denom
            for cv, log_tau in zip(self._cv, self._log_anchor, strict=True)
        ]
        return _softmax(scores)

    def observe_utility(self, utility: Sequence[float], *, weight: float = 1.0) -> list[float]:
        """Run one iteration: fold this round's per-action ``utility`` into the state.

        ``utility`` is what each action would have earned against the other
        players' play; it enters both ``CV`` and the running mean policy. The
        return is the strategy that was in effect, read before the update, which
        is what the :class:`~entente.gametheory.regret.Learner` contract asks
        for.
        """
        if len(utility) != self.num_actions:
            raise ValueError(f"expected {self.num_actions} utilities, got {len(utility)}")
        if any(not math.isfinite(u) for u in utility):
            # A single NaN here would poison CV (and the adaptive-rate stdev)
            # permanently and *silently* -- softmax never raises on NaN.
            raise ValueError("utilities must be finite")
        strategy = self.strategy
        for action in range(self.num_actions):
            self._strategy_sum[action] += weight * strategy[action]
            self._cv[action] += utility[action]
        self._experienced.append(sum(p * u for p, u in zip(strategy, utility, strict=True)))
        self._iterations += 1
        return strategy

    @property
    def average_strategy(self) -> list[float]:
        """Return the mean iterate (paper eq. 7).

        This is what a piKL agent actually plays, and what Theorems 1 and 2
        bound.
        """
        return normalize(self._strategy_sum)


ProfileUtilityFn: TypeAlias = "Callable[[Sequence[tuple[int, ...]]], Sequence[Sequence[float]]]"
"""Batch utility oracle: joint candidate-index profiles in, per profile the
utility of **every** player (same player order as the ``players`` argument of
:func:`run_pikl`) out. One call per iteration in sampled mode -- the batching
seam GPU value functions need (docs/pikl.tex Sec. 6)."""


@dataclass(frozen=True, slots=True)
class PiklPlayer:
    """One player's subgame inputs: anchor over its candidates + strength."""

    anchor: tuple[float, ...]
    lam: float


@dataclass(frozen=True, slots=True)
class PiklResult:
    """Output of :func:`run_pikl` (per player, indexed like ``players``)."""

    mean_strategies: tuple[tuple[float, ...], ...]
    final_strategies: tuple[tuple[float, ...], ...]
    mean_utilities: tuple[tuple[float, ...], ...]
    iterations: int
    profile_evaluations: int


def sample_index(distribution: Sequence[float], rng: random.Random) -> int:
    """Draw an index from a normalized distribution.

    The walk is cumulative, and the final index absorbs any floating-point
    shortfall so that a distribution summing to slightly under one still always
    yields an index.
    """
    roll = rng.random()
    acc = 0.0
    for index, p in enumerate(distribution):
        acc += p
        if roll < acc:
            return index
    return len(distribution) - 1


def run_pikl(
    players: Sequence[PiklPlayer],
    utility_fn: ProfileUtilityFn,
    *,
    iterations: int,
    mode: Literal["sampled", "expected"] = "sampled",
    eta: float | None = None,
    adaptive_scale: float = ADAPTIVE_ETA_SCALE,
    rng: random.Random | None = None,
    max_expected_profiles: int = 200_000,
) -> PiklResult:
    """Run simultaneous piKL-Hedge for every player against a joint utility oracle.

    The result carries the mean strategies, which are the object the paper's
    guarantees speak about (docs/pikl.tex Sec. 3).

    ``mode="sampled"`` is the paper's Algorithm 1 (and the only tractable
    choice for Diplomacy): each iteration samples one joint profile from the
    current strategies, then every player scores each of its candidates
    against the others' sampled slice -- all one-deviation profiles of one
    iteration go to ``utility_fn`` in a single deduplicated batch, and
    evaluations are cached across iterations (repeat profiles are free).

    ``mode="expected"`` evaluates the full joint product once up front and
    feeds every player its exact expected utility vector each iteration --
    deterministic (no ``rng``), exponential in the player count, guarded by
    ``max_expected_profiles``; meant for the validation games.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if not players:
        raise ValueError("players must be non-empty")
    learners = [
        AnchoredHedge(p.anchor, lam=p.lam, eta=eta, adaptive_scale=adaptive_scale) for p in players
    ]
    counts = [learner.num_actions for learner in learners]
    cache: dict[tuple[int, ...], tuple[float, ...]] = {}

    def evaluate(profiles: Sequence[tuple[int, ...]]) -> None:
        missing = [p for p in dict.fromkeys(profiles) if p not in cache]
        if missing:
            values = utility_fn(missing)
            if len(values) != len(missing):
                raise ValueError(
                    f"utility_fn returned {len(values)} results for {len(missing)} profiles"
                )
            for profile, value in zip(missing, values, strict=True):
                if len(value) != len(players):
                    raise ValueError(
                        f"utility_fn must score all {len(players)} players, got {len(value)}"
                    )
                if any(not math.isfinite(v) for v in value):
                    # Fail at the oracle seam (e.g. a NaN value-net forward)
                    # before the bad number enters the cache and the learners.
                    raise ValueError(f"utility_fn returned non-finite value for {profile}")
                cache[profile] = tuple(value)

    if mode == "expected":
        total = math.prod(counts)
        if total > max_expected_profiles:
            raise ValueError(
                f"expected mode would enumerate {total} profiles "
                f"(> max_expected_profiles={max_expected_profiles}); use sampled mode"
            )
        table = [tuple(p) for p in product(*(range(k) for k in counts))]
        evaluate(table)
        for _ in range(iterations):
            strategies = [learner.strategy for learner in learners]
            for i, learner in enumerate(learners):
                util = [0.0] * counts[i]
                for profile in table:
                    weight = 1.0
                    for j, strategy in enumerate(strategies):
                        if j != i:
                            weight *= strategy[profile[j]]
                    util[profile[i]] += weight * cache[profile][i]
                learner.observe_utility(util)
    else:
        generator = rng if rng is not None else random.Random(0)
        for _ in range(iterations):
            strategies = [learner.strategy for learner in learners]
            base = tuple(sample_index(strategy, generator) for strategy in strategies)
            wanted: list[tuple[int, ...]] = []
            for i in range(len(learners)):
                for action in range(counts[i]):
                    wanted.append((*base[:i], action, *base[i + 1 :]))
            evaluate(wanted)
            for i, learner in enumerate(learners):
                util = [
                    cache[(*base[:i], action, *base[i + 1 :])][i] for action in range(counts[i])
                ]
                learner.observe_utility(util)

    return PiklResult(
        mean_strategies=tuple(tuple(learner.average_strategy) for learner in learners),
        final_strategies=tuple(tuple(learner.strategy) for learner in learners),
        mean_utilities=tuple(
            tuple(cv / iterations for cv in learner.cumulative_values) for learner in learners
        ),
        iterations=iterations,
        profile_evaluations=len(cache),
    )


def reweight_anchor(anchor: Sequence[float], bias: Sequence[float]) -> list[float]:
    """Tilt an anchor distribution by an external per-candidate ``bias``, and renormalize.

    This is the one public seam for conditioning a piKL subgame on outside
    information, and it is all ``negotiation/`` (or any other caller outside
    this module) gets: multiply each candidate's anchor mass by its ``bias``
    weight and renormalize. Pass ``bias`` of all ``1.0`` for a no-op; a value
    ``> 1`` favors a candidate, ``< 1`` disfavors it. Keeping the seam this
    narrow is what leaves the search itself game-agnostic -- ``bias`` is opaque
    numbers, never orders or intents, so callers translate their own domain
    semantics ("this candidate matches a proposed action") into a bias vector
    before calling.

    The result is always a valid piKL anchor -- strictly positive, summing to
    one -- because :class:`AnchoredHedge` (the only intended consumer) rejects
    anything else. A bias below ``_BIAS_FLOOR``, ``0.0`` included, counts as
    ``_BIAS_FLOOR``: it reduces a candidate to a vanishing floor of its anchor
    mass rather than exactly zero (KL(pi || tau) needs tau > 0 on the whole
    candidate set), and an all-zero bias renormalizes back to the anchor
    unchanged. Both hold whatever the size of the anchor entries and of the
    biases. Each tilted mass ``a * w / max(w)`` is kept as a mantissa and a
    binary exponent and becomes a float only once it is scaled against the
    largest of them, so no intermediate product under- or overflows (the floor
    fraction of a tiny entry, or the weight of a zero bias next to a huge one,
    would otherwise lose its low bits in the subnormal range or vanish), and an
    all-zero bias multiplies every mass by exactly one. A mass still too small
    to be a float next to the largest -- beside a bias of 1, the floor fraction
    of an entry below about ``2.5e-24`` -- ends at the smallest positive float,
    ``5e-324``, not at zero.
    """
    if len(bias) != len(anchor):
        raise ValueError("bias must have the same length as anchor")
    if any(not math.isfinite(a) or a <= 0.0 for a in anchor):
        raise ValueError("anchor probabilities must be strictly positive (and finite)")
    if any(not math.isfinite(b) or b < 0.0 for b in bias):
        raise ValueError("bias must be non-negative and finite")
    weights = [max(b, _BIAS_FLOOR) for b in bias]
    top_mantissa, top_exponent = math.frexp(max(weights, default=1.0))
    parts: list[tuple[float, int]] = []
    for a, w in zip(anchor, weights, strict=True):
        a_mantissa, a_exponent = math.frexp(a)
        w_mantissa, w_exponent = math.frexp(w)
        # a * (w / top) as mantissa * 2**exponent; the mantissa lies in (0.25, 2).
        parts.append(
            (a_mantissa * (w_mantissa / top_mantissa), a_exponent + w_exponent - top_exponent)
        )
    # Scale so that the largest exponent becomes 1 (the largest mass lands in
    # (0.5, 4)): nothing overflows, and the scaling rounds a mass only where it
    # is subnormal next to the largest -- where its share is subnormal anyway.
    shift = 1 - max((exponent for _, exponent in parts), default=0)
    tilted = [math.ldexp(mantissa, exponent + shift) for mantissa, exponent in parts]
    return [max(p, _SMALLEST_MASS) for p in normalize(tilted)]


def anchored_quantal_response(
    expected_utilities: Sequence[float], anchor: Sequence[float], lam: float
) -> list[float]:
    """Compute the fixed point piKL converges toward (docs/pikl.tex Sec. 3).

    That fixed point is the anchor reweighted by ``exp(utility / lam)``, which
    is the softmax of ``utility/lam + log(anchor)``; with a uniform anchor it is
    the logit quantal response at temperature ``lam``. Computing it here
    independently is what lets the validation suite check a converged strategy
    against a target it did not come from. ``lam`` must be positive and finite,
    because both limits are handled by the caller.
    """
    if not math.isfinite(lam) or lam <= 0.0:
        raise ValueError("lam must be positive and finite")
    if len(expected_utilities) != len(anchor):
        raise ValueError("expected_utilities and anchor must have equal length")
    if any(not math.isfinite(p) or p <= 0.0 for p in anchor):
        raise ValueError("anchor probabilities must be strictly positive (and finite)")
    norm_anchor = normalize(anchor)
    scores = [u / lam + math.log(p) for u, p in zip(expected_utilities, norm_anchor, strict=True)]
    return _softmax(scores)


__all__ = [
    "ADAPTIVE_ETA_SCALE",
    "AnchoredHedge",
    "PiklPlayer",
    "PiklResult",
    "ProfileUtilityFn",
    "anchored_quantal_response",
    "reweight_anchor",
    "run_pikl",
    "sample_index",
]
