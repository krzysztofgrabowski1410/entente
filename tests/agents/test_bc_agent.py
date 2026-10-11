"""BCAgent, the batched policy and the BC-vs-greedy evaluation (``ml`` group).

An *untrained* checkpoint suffices: legality comes from the engine-exact mask,
not the weights, so these test the machinery (legal play, batching, the
tournament harness) without a training run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from entente.agents.bc_agent import BCAgent, bc_lockstep_policy, bc_vs_greedy
from entente.engine.game import Game
from entente.engine.orders import format_order
from entente.nn.model import BCModel
from entente.sim.vectorized import LockstepRunner, new_games

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    # Seeded: the module's tests share these game trajectories, and
    # test_agent_falls_back_off_movement needs one that reaches a retreat or
    # adjustment phase by 1902, which this seed's argmax play does on every run.
    path = tmp_path_factory.mktemp("ckpt") / "bc.pt"
    torch.manual_seed(0)
    BCModel(d_model=32, n_layers=1, n_heads=2).save(path)
    return str(path)


def test_agent_submits_only_legal_orders(checkpoint: str) -> None:
    agent = BCAgent(checkpoint)
    game = Game.new("standard", "standard")
    for _ in range(4):  # a few phases, incl. non-movement (fallback path)
        for power in game.powers:
            legal = {format_order(o) for o in game.legal_orders(power)}
            sheet = agent.orders(game, power)
            assert all(format_order(o) in legal for o in sheet)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()


def test_agent_falls_back_off_movement(checkpoint: str) -> None:
    # a retreat/adjustment phase uses the greedy fallback; the agent still plays.
    agent = BCAgent(checkpoint, fallback="greedy")
    game = Game.new("standard", "standard")
    played_non_movement = False
    while not game.is_over and game.year <= 1902:
        for power in game.powers:
            if game.phase[-1] != "M":
                played_non_movement = True
            sheet = agent.orders(game, power)
            if sheet:
                game.submit(power, list(sheet))
        game.adjudicate()
    assert played_non_movement


def test_lockstep_policy_batches_and_runs(checkpoint: str) -> None:
    model = BCModel.load(checkpoint)
    policy = bc_lockstep_policy(model, bc_power="FRANCE")
    games = new_games(3)
    runner = LockstepRunner(games, policy, max_year=1902)
    ticks = runner.run()
    assert ticks > 0
    assert all(g.year >= 1901 for g in games)


def test_bc_vs_greedy_runs_and_has_variance(checkpoint: str) -> None:
    result = bc_vs_greedy(checkpoint, [0, 1, 2], max_year=1902, seed=1)
    assert result.games == 21  # 7 seats * 3 seeds
    assert 0.0 <= result.bc_mean_sos <= 1.0
    assert 0.0 <= result.greedy_mean_sos <= 1.0
    lo, hi = result.bc_ci
    assert lo <= result.bc_mean_sos <= hi
    assert hi > lo  # sampling gives a non-degenerate interval


def test_bc_vs_greedy_writes_regenerable_seated_jsonl(checkpoint: str, tmp_path: Path) -> None:
    """``output=`` leaves a tournament-schema artifact the aggregate recomputes from.

    Every reported number has to come back out of the committed file, so the
    printed aggregate must be reproducible from the JSONL alone.
    """
    from statistics import fmean

    from entente.eval.scoring import per_agent_sos
    from entente.eval.tournament import read_seated_jsonl

    out = tmp_path / "bc_vs_greedy.jsonl"
    result = bc_vs_greedy(checkpoint, [0, 1], max_year=1902, seed=1, output=out)
    seated = list(read_seated_jsonl(out))
    assert len(seated) == result.games == 14  # 7 seats * 2 games
    for record in seated:
        names = list(record.seats.values())
        assert names.count("bc") == 1 and names.count("greedy") == 6
    sos = per_agent_sos(seated)
    assert len(sos["bc"]) == 14
    assert len(sos["greedy"]) == 84
    assert fmean(sos["bc"]) == pytest.approx(result.bc_mean_sos)
    assert fmean(sos["greedy"]) == pytest.approx(result.greedy_mean_sos)


def test_sampling_differs_from_argmax(checkpoint: str) -> None:
    model = BCModel.load(checkpoint)
    game = Game.new("standard", "standard")
    from entente.nn.features import _unit_token, encode_game
    from entente.nn.order_vocab import legal_order_indices

    enc = model.encode(encode_game(game).unsqueeze(0))[0]
    units = sorted(
        (u for u in game.state.units if u.power == "RUSSIA"), key=lambda u: str(u.location)
    )
    groups = legal_order_indices(game, "RUSSIA", model.vocab)
    slots = [model.board.location_index(_unit_token(u)) for u in units]
    legal = [groups[str(u.location)] for u in units]
    gen = torch.Generator().manual_seed(0)
    argmax = model.policy.decode(enc[slots], legal)
    samples = [model.policy.decode(enc[slots], legal, sample=True, generator=gen) for _ in range(8)]
    assert all(all(pick in cand for pick, cand in zip(s, legal, strict=True)) for s in samples)
    assert any(s != argmax for s in samples)  # sampling can diverge from the mode
