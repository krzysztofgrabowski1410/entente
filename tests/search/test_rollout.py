"""branch_game and profile_utility_fn, against the real engine and exact values.

Covered (docs/pikl.tex Sec. 6): the original game is never mutated, successor
phases are right, the 1-ply utility signal is genuine -- a fall-phase
supply-centre capture moves the share -- and the batching contract holds.
"""

from __future__ import annotations

import pytest

from entente.engine.game import Game
from entente.engine.orders import AnyOrder, Move
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.state import State
from entente.search.rollout import branch_game, profile_utility_fn, supply_share_value

FRANCE = "FRANCE"


@pytest.fixture()
def spring() -> Game:
    return Game.new("standard", "standard")


@pytest.fixture()
def fall() -> Game:
    game = Game.new("standard", "standard")
    game.adjudicate()  # all-hold spring; retreats skip -> F1901M
    assert game.phase == "F1901M"
    return game


def _some_move(game: Game, power: str) -> Move:
    return next(
        order
        for order in game.legal_orders(power)
        if isinstance(order, Move) and not order.via_convoy
    )


def _capture_move(game: Game, power: str) -> Move:
    """Find a legal move into an unowned supply centre.

    France has one in F1901M, into Spain or Portugal.
    """
    owned = set(game.ownership)
    for order in game.legal_orders(power):
        if (
            isinstance(order, Move)
            and not order.via_convoy
            and order.destination.province not in owned
            and game.graph.province(order.destination.province).supply_center_count > 0
        ):
            return order
    raise AssertionError("no capture move available -- fixture assumption broken")


class TestBranchGame:
    """Branching a position: the successor, and the original left untouched."""

    def test_successor_without_mutating_original(self, spring: Game) -> None:
        move = _some_move(spring, FRANCE)
        units_before = tuple(spring.state.units)
        branch = branch_game(spring, {FRANCE: (move,)})

        assert spring.phase == "S1901M"  # original untouched
        assert tuple(spring.state.units) == units_before
        assert spring.pending_sheets == {}  # branch's submit left no trace
        assert branch.phase == "F1901M"  # spring retreats auto-skip
        destination = move.destination.province
        assert any(
            u.power == FRANCE and u.location.province == destination for u in branch.state.units
        )
        # the original still plays on normally after branching
        spring.adjudicate()
        assert spring.phase == "F1901M"

    def test_defaults_are_holds_for_absent_powers(self, spring: Game) -> None:
        branch = branch_game(spring, {})
        assert {(u.power, str(u.location)) for u in branch.state.units} == {
            (u.power, str(u.location)) for u in spring.state.units
        }

    def test_fall_branch_crosses_the_ownership_update(self, fall: Game) -> None:
        move = _capture_move(fall, FRANCE)
        branch = branch_game(fall, {FRANCE: (move,)})
        assert branch.ownership[move.destination.province] == FRANCE
        assert fall.ownership.get(move.destination.province) is None

    def test_rejects_non_movement_phase(self, spring: Game) -> None:
        winter = Game(
            spring.ruleset,
            spring.graph,
            spring.state,
            spring.ownership,
            Phase(1901, Season.WINTER, PhaseKind.ADJUSTMENT),
        )
        with pytest.raises(ValueError, match="movement"):
            branch_game(winter, {})

    def test_rejects_finished_game(self, spring: Game) -> None:
        empty = Game(
            spring.ruleset,
            spring.graph,
            State.of("standard", "S1901M", []),
            {},
            Phase.parse("S1901M"),
        )
        empty.adjudicate()  # no units, nothing can change -> game over
        assert empty.is_over
        with pytest.raises(ValueError, match="finished"):
            branch_game(empty, {})


class TestSupplyShareValue:
    """The torch-free exact value function used as the oracle here."""

    def test_initial_shares(self, spring: Game) -> None:
        [scores] = supply_share_value([spring])
        total = sum(spring.supply_count(p) for p in spring.powers)
        assert scores[FRANCE] == pytest.approx(spring.supply_count(FRANCE) / total)
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_uniform_when_nothing_owned(self, spring: Game) -> None:
        bare = Game(spring.ruleset, spring.graph, spring.state, {}, Phase.parse("S1901M"))
        [scores] = supply_share_value([bare])
        assert list(scores.values()) == pytest.approx(
            [1.0 / len(spring.powers)] * len(spring.powers)
        )


class TestProfileUtilityFn:
    """The oracle the search calls: its signal, its ordering and its batching."""

    def test_capture_shows_up_in_the_utilities(self, fall: Game) -> None:
        """The 1-ply signal, end to end and in exact numbers.

        Holding keeps France at its spring share; the capture branch raises it.
        The oracle is exact, so the two utilities can be asserted outright
        rather than merely compared.
        """
        holds: tuple[AnyOrder, ...] = ()
        capture = (_capture_move(fall, FRANCE),)
        calls: list[int] = []

        def counting_value(games: list[Game]) -> list[dict[str, float]]:
            calls.append(len(games))
            return supply_share_value(games)

        oracle = profile_utility_fn(fall, [FRANCE], [[holds, capture]], counting_value)
        utilities = oracle([(0,), (1,)])
        assert calls == [2]  # one batched value call for the whole batch

        base_total = sum(fall.supply_count(p) for p in fall.powers)
        base_share = fall.supply_count(FRANCE) / base_total
        assert utilities[0] == (pytest.approx(base_share),)
        assert utilities[1] == (pytest.approx((fall.supply_count(FRANCE) + 1) / (base_total + 1)),)
        assert utilities[1][0] > utilities[0][0]
        assert fall.phase == "F1901M"  # original never mutated

    def test_multi_power_profile_indexing(self, fall: Game) -> None:
        france = ((), (_capture_move(fall, FRANCE),))
        germany = ((),)
        oracle = profile_utility_fn(
            fall, [FRANCE, "GERMANY"], [france, germany], supply_share_value
        )
        [only] = oracle([(1, 0)])
        assert len(only) == 2  # utilities in (FRANCE, GERMANY) order
        assert only[0] > only[1]  # France just captured; Germany did not

    def test_contract_violations(self, fall: Game) -> None:
        with pytest.raises(ValueError, match="parallel"):
            profile_utility_fn(fall, [FRANCE], [], supply_share_value)
        with pytest.raises(ValueError, match="distinct"):
            profile_utility_fn(fall, [FRANCE, FRANCE], [[()], [()]], supply_share_value)
        oracle = profile_utility_fn(fall, [FRANCE], [[()]], supply_share_value)
        with pytest.raises(ValueError, match="does not index"):
            oracle([(0, 0)])

        def short_value(games: list[Game]) -> list[dict[str, float]]:
            return []

        bad = profile_utility_fn(fall, [FRANCE], [[()]], short_value)
        with pytest.raises(ValueError, match="value_fn returned"):
            bad([(0,)])
