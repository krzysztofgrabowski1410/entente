"""Top-k candidate order-set selection from a policy (docs/pikl.tex Sec. 5).

The subgame piKL solves is spanned by a small candidate set per power, drawn
from the anchor policy (Jacob et al. 2022 Sec. 4.3.1 sample "up to 30 of the most
likely actions" from the imitation policy; the k used here is 8-16).
:func:`collect_candidates` is the policy-agnostic half: it takes whatever a
:class:`CandidatePolicy` sampled, deduplicates complete order-sets, keeps the
``k`` most likely by temperature-1 log-probability (always retaining the
greedy decode when requested), re-checks legality against the engine, and
renormalizes the anchor over the survivors. The torch-backed policy lives in
:mod:`entente.agents.pikl_agent`; tests drive this module with a fake.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from entente.engine.orders import Waive, format_order

_MIN_LOG_WEIGHT = -700.0
"""Floor for a kept candidate's log-weight relative to the best one:
``exp(-700)`` is the last comfortably-normal double, so every anchor entry
stays strictly positive no matter how lopsided the kept set's scores are."""

if TYPE_CHECKING:
    import random
    from collections.abc import Sequence

    from entente.engine.game import Game
    from entente.engine.orders import AnyOrder


@dataclass(frozen=True, slots=True)
class ScoredOrderSet:
    """One sampled complete order-set with its anchor-policy score.

    ``log_prob`` is the temperature-1 log-probability of the whole set under
    the anchor policy (docs/pikl.tex Sec. 5: sampling may be tempered or nucleus,
    the anchor is always scored untempered).
    """

    orders: tuple[AnyOrder, ...]
    log_prob: float


class CandidatePolicy(Protocol):
    """The sampling surface :func:`collect_candidates` needs.

    An implementation returns exactly ``count`` scored order-sets for ``power``
    in ``game``; when ``include_greedy`` the first entry is the greedy (argmax)
    decode and the rest are samples. Duplicates are expected -- deduplication is
    the caller's job. Implementations own their batching (encode once, decode
    ``count`` times).
    """

    def sample_order_sets(
        self,
        game: Game,
        power: str,
        count: int,
        *,
        include_greedy: bool,
        rng: random.Random,
    ) -> Sequence[ScoredOrderSet]:
        """Draw ``count`` scored order-sets for ``power`` in ``game``'s position."""
        ...


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """One power's axis of the subgame piKL solves.

    It carries the distinct order-sets, their raw log-probabilities, and the
    anchor distribution renormalized over exactly those survivors
    (docs/pikl.tex Sec. 5).
    """

    power: str
    order_sets: tuple[tuple[AnyOrder, ...], ...]
    log_probs: tuple[float, ...]
    anchor: tuple[float, ...]


def _canonical(orders: Sequence[AnyOrder]) -> tuple[str, ...]:
    """Return the dedup key for an order-set: its canonical order texts, sorted.

    A movement sheet holds one order per unit, so the sorted multiset of texts
    identifies the sheet.
    """
    return tuple(sorted(format_order(order) for order in orders))


def _validate_legal(game: Game, power: str, orders: Sequence[AnyOrder]) -> None:
    """Defensive re-check: every order legal, one order per acting unit.

    Samplers that mask against ``game.legal_orders`` satisfy this by
    construction, and the BC decoder does: its masking-exactness test pins the
    decoder's mask to the engine's own legal set. A violation here is therefore
    an upstream bug worth failing loudly on, not a position to search from.
    """
    legal = set(game.legal_orders(power))
    seen_locations: set[str] = set()
    for order in orders:
        if order not in legal:
            raise ValueError(f"candidate order {format_order(order)!r} is not legal for {power}")
        if isinstance(order, Waive):  # location-less; never legal in movement
            raise ValueError(f"candidate order sets are per-unit; got a waive for {power}")
        location = str(order.location)
        if location in seen_locations:
            raise ValueError(f"candidate set orders {location} twice for {power}")
        seen_locations.add(location)


def collect_candidates(
    policy: CandidatePolicy,
    game: Game,
    power: str,
    *,
    k: int,
    attempts: int | None = None,
    include_greedy: bool = True,
    rng: random.Random,
) -> CandidateSet:
    """Sample from ``policy``, deduplicate, rank by anchor score and renormalize.

    ``attempts`` is the total number of draws requested from ``policy``
    (default ``4 * k`` -- sampling with replacement needs headroom to find
    ``k`` distinct sets); fewer than ``k`` distinct results is fine (forced
    positions may not have k plausible order-sets). Ranking is by anchor
    log-probability, descending (ties broken canonically); the greedy decode,
    when requested, is always retained even if ``k`` higher-probability
    samples exist, so the anchor's mode is in every subgame.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    total = attempts if attempts is not None else 4 * k
    if total < k:
        raise ValueError("attempts must be >= k")
    draws = policy.sample_order_sets(game, power, total, include_greedy=include_greedy, rng=rng)
    if len(draws) != total:
        raise ValueError(f"policy returned {len(draws)} order sets for count={total}")

    distinct: dict[tuple[str, ...], ScoredOrderSet] = {}
    greedy_key: tuple[str, ...] | None = None
    for position, scored in enumerate(draws):
        key = _canonical(scored.orders)
        if position == 0 and include_greedy:
            greedy_key = key
        if key not in distinct:
            distinct[key] = scored

    ranked = sorted(distinct.items(), key=lambda item: (-item[1].log_prob, item[0]))
    kept = ranked[:k]
    if greedy_key is not None and all(key != greedy_key for key, _ in kept):
        kept = [*kept[: k - 1], (greedy_key, distinct[greedy_key])]

    for _, scored in kept:
        _validate_legal(game, power, scored.orders)

    log_probs = tuple(scored.log_prob for _, scored in kept)
    if any(not math.isfinite(lp) for lp in log_probs):
        # A NaN/inf score would flow straight into the anchor and silently
        # corrupt AnchoredHedge; -inf ("impossible candidate") should have
        # been dropped by the policy, not kept as a candidate.
        raise ValueError(f"policy returned non-finite log-probability for {power}")
    top = max(log_probs)
    # Clamp the spread: exp underflows to exactly 0.0 beyond ~-745, and the
    # anchor must stay strictly positive (an AnchoredHedge precondition).
    weights = [math.exp(max(lp - top, _MIN_LOG_WEIGHT)) for lp in log_probs]
    norm = sum(weights)
    anchor = tuple(w / norm for w in weights)
    return CandidateSet(
        power=power,
        order_sets=tuple(scored.orders for _, scored in kept),
        log_probs=log_probs,
        anchor=anchor,
    )


__all__ = ["CandidatePolicy", "CandidateSet", "ScoredOrderSet", "collect_candidates"]
