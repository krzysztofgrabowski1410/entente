"""collect_candidates: dedup, ranking, greedy retention, renormalization, legality.

A scripted fake policy drives the module over a *real* engine game
(docs/pikl.tex Sec. 5), so the legality re-check runs against genuine
``legal_orders`` output rather than against a fixture that could agree with a
wrong answer.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pytest

from entente.engine.game import Game
from entente.engine.orders import AnyOrder, Hold, Move
from entente.search.candidates import (
    CandidateSet,
    ScoredOrderSet,
    collect_candidates,
)

POWER = "FRANCE"


@pytest.fixture(scope="module")
def game() -> Game:
    return Game.new("standard", "standard")


def _holds(game: Game, power: str) -> tuple[AnyOrder, ...]:
    return tuple(
        Hold(unit.kind, unit.location)
        for unit in sorted(
            (u for u in game.state.units if u.power == power),
            key=lambda u: str(u.location),
        )
    )


def _one_move_set(game: Game, power: str) -> tuple[AnyOrder, ...]:
    """Build the hold sheet with the first unit's hold swapped for a legal move."""
    holds = _holds(game, power)
    first = holds[0]
    move = next(
        order
        for order in game.legal_orders(power)
        if isinstance(order, Move) and order.location == first.location and not order.via_convoy
    )
    return (move, *holds[1:])


@dataclass
class ScriptedPolicy:
    """A candidate policy that returns exactly the draws it was handed."""

    draws: list[ScoredOrderSet]

    def sample_order_sets(
        self,
        game: Game,
        power: str,
        count: int,
        *,
        include_greedy: bool,
        rng: random.Random,
    ) -> list[ScoredOrderSet]:
        assert count == len(self.draws), "test script must match requested count"
        return self.draws


def test_dedup_and_topk_by_log_prob(game: Game) -> None:
    holds = _holds(game, POWER)
    moved = _one_move_set(game, POWER)
    draws = [
        ScoredOrderSet(holds, -1.0),  # greedy (first)
        ScoredOrderSet(moved, -0.5),  # more likely than the greedy decode
        ScoredOrderSet(holds, -1.0),  # duplicate -- must collapse
        ScoredOrderSet(moved, -0.5),
    ]
    result = collect_candidates(
        ScriptedPolicy(draws), game, POWER, k=2, attempts=4, rng=random.Random(0)
    )
    assert isinstance(result, CandidateSet)
    assert result.power == POWER
    assert result.order_sets == (moved, holds)  # ranked by log-prob, descending
    assert result.log_probs == (-0.5, -1.0)
    w0, w1 = math.exp(-0.5), math.exp(-1.0)
    assert result.anchor == pytest.approx((w0 / (w0 + w1), w1 / (w0 + w1)), abs=1e-12)
    assert sum(result.anchor) == pytest.approx(1.0, abs=1e-12)


def test_greedy_is_retained_even_when_outranked(game: Game) -> None:
    holds = _holds(game, POWER)
    moved = _one_move_set(game, POWER)
    # A third distinct set: swap a *different* unit's hold for a move.
    second_move = next(
        order
        for order in game.legal_orders(POWER)
        if isinstance(order, Move) and order.location == holds[1].location and not order.via_convoy
    )
    third = (holds[0], second_move, *holds[2:])

    draws = [
        ScoredOrderSet(holds, -9.0),  # greedy decode, scored worst on purpose
        ScoredOrderSet(moved, -0.5),
        ScoredOrderSet(third, -0.7),
    ]
    result = collect_candidates(
        ScriptedPolicy(draws), game, POWER, k=2, attempts=3, rng=random.Random(0)
    )
    # top-k alone would keep (moved, third); greedy replaces the last slot.
    assert result.order_sets == (moved, holds)

    unpinned = collect_candidates(
        ScriptedPolicy(draws),
        game,
        POWER,
        k=2,
        attempts=3,
        include_greedy=False,
        rng=random.Random(0),
    )
    assert unpinned.order_sets == (moved, third)


def test_fewer_distinct_than_k_is_fine(game: Game) -> None:
    holds = _holds(game, POWER)
    draws = [ScoredOrderSet(holds, -1.0)] * 5
    result = collect_candidates(
        ScriptedPolicy(draws), game, POWER, k=4, attempts=5, rng=random.Random(0)
    )
    assert result.order_sets == (holds,)
    assert result.anchor == (1.0,)


def test_empty_order_sets_collapse_to_one_candidate(game: Game) -> None:
    draws = [ScoredOrderSet((), 0.0)] * 3
    result = collect_candidates(
        ScriptedPolicy(draws), game, POWER, k=2, attempts=3, rng=random.Random(0)
    )
    assert result.order_sets == ((),)
    assert result.anchor == (1.0,)


def test_rejects_illegal_and_malformed_candidates(game: Game) -> None:
    holds = _holds(game, POWER)
    foreign = _holds(game, "GERMANY")
    with pytest.raises(ValueError, match="not legal for FRANCE"):
        collect_candidates(
            ScriptedPolicy([ScoredOrderSet(foreign, -1.0)]),
            game,
            POWER,
            k=1,
            attempts=1,
            include_greedy=False,
            rng=random.Random(0),
        )
    doubled = (holds[0], *holds)
    with pytest.raises(ValueError, match="twice"):
        collect_candidates(
            ScriptedPolicy([ScoredOrderSet(doubled, -1.0)]),
            game,
            POWER,
            k=1,
            attempts=1,
            include_greedy=False,
            rng=random.Random(0),
        )


def test_rejects_waive_in_candidate_sets(game: Game) -> None:
    """On an adjustment board a Waive is legal, yet never a per-unit candidate.

    The guard has to name that, rather than report the sheet as illegal.
    """
    from entente.engine.phase import Phase, PhaseKind, Season
    from entente.engine.state import State

    # drop one French unit: its home center becomes vacant and buildable,
    # so the quota is positive AND a build (hence Waive) is enumerable.
    survivors = [u for u in game.state.units if u.power != POWER]
    survivors += sorted(
        (u for u in game.state.units if u.power == POWER), key=lambda u: str(u.location)
    )[1:]
    winter = Game(
        game.ruleset,
        game.graph,
        State.of("standard", "W1901A", survivors),
        {**game.ownership, "SPA": POWER},
        Phase(1901, Season.WINTER, PhaseKind.ADJUSTMENT),
    )
    waive = next(o for o in winter.legal_orders(POWER) if type(o).__name__ == "Waive")
    with pytest.raises(ValueError, match="per-unit"):
        collect_candidates(
            ScriptedPolicy([ScoredOrderSet((waive,), -1.0)]),
            winter,
            POWER,
            k=1,
            attempts=1,
            include_greedy=False,
            rng=random.Random(0),
        )


def test_rejects_bad_parameters(game: Game) -> None:
    holds = _holds(game, POWER)
    policy = ScriptedPolicy([ScoredOrderSet(holds, -1.0)])
    with pytest.raises(ValueError, match="k must be"):
        collect_candidates(policy, game, POWER, k=0, attempts=1, rng=random.Random(0))
    with pytest.raises(ValueError, match="attempts"):
        collect_candidates(policy, game, POWER, k=2, attempts=1, rng=random.Random(0))

    class Short:
        def sample_order_sets(
            self,
            game: Game,
            power: str,
            count: int,
            *,
            include_greedy: bool,
            rng: random.Random,
        ) -> list[ScoredOrderSet]:
            return []

    with pytest.raises(ValueError, match="policy returned"):
        collect_candidates(Short(), game, POWER, k=1, attempts=1, rng=random.Random(0))


def test_default_attempts_is_4k(game: Game) -> None:
    holds = _holds(game, POWER)
    seen: list[int] = []

    class Counting:
        def sample_order_sets(
            self,
            game: Game,
            power: str,
            count: int,
            *,
            include_greedy: bool,
            rng: random.Random,
        ) -> list[ScoredOrderSet]:
            seen.append(count)
            return [ScoredOrderSet(holds, -1.0)] * count

    collect_candidates(Counting(), game, POWER, k=3, rng=random.Random(0))
    assert seen == [12]
