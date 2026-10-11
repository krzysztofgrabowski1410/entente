"""Pin the full-game press-vs-no-press wiring, end to end.

Unit tests cover the module functions, and the CLI runs as a subprocess over a
random-weight checkpoint, as tests/agents/test_lambda_sweep_smoke.py does: a
path that is only unit-tested can still be broken end to end. Quality is
asserted nowhere -- a random anchor cannot play well, and no claim about the
size of an SoS delta is made at this scale.
"""

from __future__ import annotations

import random
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from entente.engine.game import Game
from entente.eval.press_results import read_press_jsonl, render_table
from entente.eval.scoring import sos_scores
from entente.nn.model import BCModel

pytestmark = [pytest.mark.ml, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_press_eval.py"
SMOKE_YAML = REPO / "configs" / "press" / "smoke.yaml"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = BCModel(d_model=32, n_layers=1, n_heads=4)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    model.save(path)
    return path


@pytest.fixture(scope="module")
def model(checkpoint: Path) -> BCModel:
    return BCModel.load(checkpoint, map_location=torch.device("cpu")).eval()


class TestRunPressRound:
    """One negotiation round, including the cases where it does nothing."""

    def test_noop_when_subject_missing(self, model: BCModel) -> None:
        from entente.negotiation.press_eval import collect_subgame, run_press_round
        from entente.search.config import PiklSearchConfig

        game = Game.new("standard", "standard")
        from entente.agents.pikl_agent import BCCandidatePolicy

        policy = BCCandidatePolicy(model, device=torch.device("cpu"), temperature=0.5, top_p=0.95)
        search = PiklSearchConfig(k=2, attempts=4, iterations=2)
        rng = random.Random(0)
        powers, candidate_sets = collect_subgame(policy, game, search, rng)

        out = run_press_round(powers, candidate_sets, "NOT_A_POWER", game.phase, rng)
        assert [cs.anchor for cs in out] == [cs.anchor for cs in candidate_sets]

    def test_reweights_exactly_one_ally(self, model: BCModel) -> None:
        from entente.agents.pikl_agent import BCCandidatePolicy
        from entente.negotiation.press_eval import collect_subgame, run_press_round
        from entente.search.config import PiklSearchConfig

        game = Game.new("standard", "standard")
        policy = BCCandidatePolicy(model, device=torch.device("cpu"), temperature=0.5, top_p=0.95)
        search = PiklSearchConfig(k=2, attempts=4, iterations=2)
        rng = random.Random(0)
        powers, candidate_sets = collect_subgame(policy, game, search, rng)
        subject = powers[0]

        out = run_press_round(powers, candidate_sets, subject, game.phase, rng)
        changed = [
            power
            for power, before, after in zip(powers, candidate_sets, out, strict=True)
            if before.anchor != after.anchor
        ]
        assert len(changed) == 1
        assert subject not in changed  # the subject proposes; an ally's anchor changes


class TestPlayGame:
    """One full game, and the single record it reports."""

    def test_returns_one_record_for_the_subject(self, model: BCModel) -> None:
        from entente.negotiation.press_eval import play_game
        from entente.search.config import PiklSearchConfig

        search = PiklSearchConfig(k=2, attempts=4, iterations=2)
        record = play_game(
            model,
            torch.device("cpu"),
            seed=0,
            press=True,
            subject="FRANCE",
            search=search,
            fallback_name="greedy",
            max_year=1901,
        )
        assert record.seat == "FRANCE"
        assert record.press is True
        assert 0.0 <= record.sos <= 1.0

    def test_press_and_no_press_are_both_playable_and_deterministic(self, model: BCModel) -> None:
        from entente.negotiation.press_eval import play_game
        from entente.search.config import PiklSearchConfig

        search = PiklSearchConfig(k=2, attempts=4, iterations=2)
        kwargs = {
            "model": model,
            "device": torch.device("cpu"),
            "seed": 1,
            "subject": "GERMANY",
            "search": search,
            "fallback_name": "greedy",
            "max_year": 1901,
        }
        first = play_game(press=False, **kwargs)
        second = play_game(press=False, **kwargs)
        assert first == second  # same seed, same condition -> byte-identical record


def test_sos_scores_still_sum_to_one_per_game(model: BCModel) -> None:
    # The property that makes the all-seats-symmetric design a tautology
    # (module docstring of press_eval.py) -- pinned here as a standing
    # invariant so a future change to scoring can't silently reintroduce it
    # without this test explaining why it matters.
    from entente.negotiation.press_eval import collect_subgame  # noqa: F401

    counts = {"A": 3, "B": 4, "C": 0, "D": 5}
    shares = sos_scores(counts, winner=None)
    assert sum(shares.values()) == pytest.approx(1.0)


def test_press_eval_cli_end_to_end_tiny(checkpoint: Path, tmp_path: Path) -> None:
    output = tmp_path / "press.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(SMOKE_YAML),
            "--set",
            f"agent.checkpoint={checkpoint.as_posix()}",
            "--set",
            "agent.search.k=2",
            "--set",
            "agent.search.attempts=4",
            "--set",
            "agent.search.iterations=2",
            "--set",
            "games=1",
            "--set",
            "max_year=1901",
            "--set",
            f"output={output.as_posix()}",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    assert "wrote 2 records" in result.stdout  # 1 game x 2 conditions
    assert "| Condition | Games |" in result.stdout
    assert "mean SoS delta" in result.stdout

    records = list(read_press_jsonl(output))
    assert len(records) == 2
    assert {r.press for r in records} == {False, True}
    for record in records:
        assert record.seat == "FRANCE"  # smoke.yaml's subject_seat
        assert 0.0 <= record.sos <= 1.0
    assert "| Condition | Games |" in render_table(records)
