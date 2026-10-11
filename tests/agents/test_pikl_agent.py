"""piKL agent wiring over a tiny random-weight checkpoint (``ml`` group).

Covered: candidate sampling from the BC policy, value scoring, full-search
order selection, the lockstep tournament policy, and the human-agreement path
over synthetic shards.

Random weights are a valid anchor for a *wiring* check, because every assertion
here is about legality, determinism, shapes and plumbing, and never about play
quality. The quality claims live in tests/search/, where exact oracles exist
(docs/pikl.tex Sec. 8).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
import torch

from entente.agents.greedy import GreedyAgent
from entente.agents.pikl_agent import (
    AnchorAgreement,
    BCCandidatePolicy,
    PiklAgent,
    human_agreement,
    movement_game_from_row,
    pikl_lockstep_policy,
    value_scores,
)
from entente.engine.game import Game
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.state import State
from entente.nn.model import BCModel
from entente.search.candidates import collect_candidates
from entente.search.config import PiklAgentConfig, PiklSearchConfig
from entente.sim.vectorized import LockstepRunner, new_games

pytestmark = pytest.mark.ml

TINY_SEARCH = PiklSearchConfig(k=2, attempts=4, iterations=3, lam=1.0, temperature=1.0)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = BCModel(d_model=32, n_layers=1, n_heads=4)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    model.save(path)
    return path


@pytest.fixture(scope="module")
def agent(checkpoint: Path) -> PiklAgent:
    return PiklAgent(PiklAgentConfig(checkpoint=str(checkpoint), search=TINY_SEARCH))


class TestBCCandidatePolicy:
    """Sampling candidate order-sets out of the BC policy."""

    def test_returns_scored_legal_sets(self, agent: PiklAgent) -> None:
        game = Game.new("standard", "standard")
        policy = BCCandidatePolicy(agent.model, device=agent.device, temperature=0.5, top_p=0.95)
        draws = policy.sample_order_sets(
            game, "FRANCE", 4, include_greedy=True, rng=random.Random(0)
        )
        assert len(draws) == 4
        legal = set(game.legal_orders("FRANCE"))
        units = sum(1 for u in game.state.units if u.power == "FRANCE")
        for scored in draws:
            assert len(scored.orders) == units
            assert all(order in legal for order in scored.orders)
            assert scored.log_prob < 0.0
        # first draw is the greedy decode: deterministic across calls
        again = policy.sample_order_sets(
            game, "FRANCE", 4, include_greedy=True, rng=random.Random(99)
        )
        assert draws[0] == again[0]

    def test_collect_candidates_integration(self, agent: PiklAgent) -> None:
        game = Game.new("standard", "standard")
        policy = BCCandidatePolicy(agent.model, device=agent.device, temperature=1.0, top_p=1.0)
        candidates = collect_candidates(
            policy, game, "TURKEY", k=3, attempts=6, rng=random.Random(1)
        )
        assert 1 <= len(candidates.order_sets) <= 3
        assert sum(candidates.anchor) == pytest.approx(1.0)


class TestValueScores:
    """Scoring positions: the value head for live boards, the truth for decided ones."""

    def test_live_states_softmax_over_powers(self, agent: PiklAgent) -> None:
        games = [Game.new("standard", "standard") for _ in range(2)]
        scores = value_scores(agent.model, agent.device, games)
        assert len(scores) == 2
        for row in scores:
            assert set(row) == set(games[0].powers)
            assert sum(row.values()) == pytest.approx(1.0, abs=1e-5)

    def test_finished_games_get_exact_sos(self, agent: PiklAgent) -> None:
        """A solo must be scored exactly, not by the value net.

        France on 18 centres at the Fall boundary takes 1.0 and everyone else
        takes 0.0.
        """
        template = Game.new("standard", "standard")
        centers = sorted(
            p for p, prov in template.graph.provinces.items() if prov.supply_center_count > 0
        )
        solo = Game(
            template.ruleset,
            template.graph,
            State.of("standard", "F1901M", []),
            dict.fromkeys(centers[:18], "FRANCE"),
            Phase.parse("F1901M"),
        )
        solo.adjudicate()  # fall boundary: ownership check crowns the winner
        assert solo.is_over and solo.winner == "FRANCE"
        [scores] = value_scores(agent.model, agent.device, [solo])
        assert scores["FRANCE"] == pytest.approx(1.0)
        assert scores["GERMANY"] == pytest.approx(0.0)


class TestPiklAgent:
    """The complete player: legal, deterministic movement out of a full search."""

    def test_movement_orders_are_legal_and_deterministic(self, checkpoint: Path) -> None:
        game = Game.new("standard", "standard")
        config = PiklAgentConfig(checkpoint=str(checkpoint), search=TINY_SEARCH)
        first = PiklAgent(config).orders(game, "FRANCE")
        second = PiklAgent(config).orders(game, "FRANCE")
        assert list(first) == list(second)  # replay contract: seed + phase + power
        legal = set(game.legal_orders("FRANCE"))
        units = sum(1 for u in game.state.units if u.power == "FRANCE")
        assert len(first) == units
        assert all(order in legal for order in first)
        assert game.phase == "S1901M"  # the search never mutates the live game

    def test_non_movement_delegates_to_fallback(self, agent: PiklAgent) -> None:
        template = Game.new("standard", "standard")
        winter = Game(
            template.ruleset,
            template.graph,
            template.state,
            {**template.ownership, "SPA": "FRANCE"},  # quota +1: a real decision
            Phase(1901, Season.WINTER, PhaseKind.ADJUSTMENT),
        )
        assert winter.phase_kind is PhaseKind.ADJUSTMENT
        expected = GreedyAgent(0).orders(winter, "FRANCE")
        assert list(agent.orders(winter, "FRANCE")) == list(expected)

    def test_lambda_inf_plays_a_candidate_from_the_anchor(self, checkpoint: Path) -> None:
        game = Game.new("standard", "standard")
        config = PiklAgentConfig(
            checkpoint=str(checkpoint),
            search=TINY_SEARCH.model_copy(update={"lam": float("inf"), "play": "argmax"}),
        )
        orders = PiklAgent(config).orders(game, "ENGLAND")
        legal = set(game.legal_orders("ENGLAND"))
        assert orders and all(order in legal for order in orders)


class TestLockstepPolicy:
    """Routing the searched seat one way and every other request the other."""

    def test_routes_pikl_seat_and_batches_the_rest(self, agent: PiklAgent) -> None:
        games = new_games(2)
        agreement = AnchorAgreement()
        policy = pikl_lockstep_policy(
            agent,
            pikl_power="FRANCE",
            generator=torch.Generator().manual_seed(0),
            fallback=GreedyAgent(0),
            agreement=agreement,
        )
        runner = LockstepRunner(games, policy)
        assert runner.step()  # S1901M for both games, all powers
        for game in games:
            assert game.phase != "S1901M"
            rate = agreement.rate_for(game)
            assert rate is not None and 0.0 <= rate <= 1.0
        assert agreement.rate_for(Game.new("standard", "standard")) is None


class TestHumanAgreementPath:
    """The shard to game to search to per-unit-match path, end to end."""

    def test_wiring_over_synthetic_shards(self, agent: PiklAgent, tmp_path: Path) -> None:
        """Synthetic greedy-teacher shards stand in for human data that is absent.

        What this proves is the shard to game to search to per-unit-match
        plumbing. It is not the human metric, and no number from it is reported
        as one: that measurement waits on the real dataset.
        """
        from entente.data.paquette import game_from_json
        from entente.data.preprocess import preprocess
        from entente.data.synthetic import synthesize_games

        games = [
            game_from_json(g)
            for g in synthesize_games(range(2), agent="random", teacher="greedy", max_year=1902)
        ]
        shards = tmp_path / "shards"
        preprocess(games, shards, shard_size=10_000)
        rate, units = human_agreement(agent, shards, rows=3, seed=0)
        assert units > 0
        assert rate is not None and 0.0 <= rate <= 1.0
        with pytest.raises(FileNotFoundError):
            human_agreement(agent, tmp_path / "nowhere", rows=1)

    def test_movement_game_from_row_drops_dislodged_markers(self) -> None:
        """A dislodged marker in a dump row is a transient the reconstruction drops.

        ``'*A GRE'`` means a unit dislodged and pending retreat, which is not a
        position a movement phase can be rebuilt from. The agreement path shares
        the training set's reconstruction, so it must drop the marker and keep
        the rest of the board.
        """
        template = Game.new("standard", "standard")
        game = movement_game_from_row(
            template,
            {"TURKEY": ["*A GRE", "A CON"], "AUSTRIA": ["A BUD"]},
            {"TURKEY": ["CON", "ANK"], "AUSTRIA": ["BUD"]},
            "F1909M",
        )
        locations = {str(unit.location) for unit in game.state.units}
        assert locations == {"CON", "BUD"}  # GRE transient dropped, board intact
