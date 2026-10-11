r"""Pin that every text-artifact writer emits LF regardless of platform.

Committed artifacts (results JSONL, transcripts, fixtures, saves) are LF in
git, and Python's text mode translates ``"\n"`` to ``os.linesep`` unless
``newline="\n"`` is passed, so a writer that leaves the translation to the
platform rewrites a committed file with CRLF on every Windows run and dirties
the tree. Each test here writes one artifact through its real writer and
checks the bytes for a carriage return.
"""

from __future__ import annotations

from pathlib import Path

from entente.cli import main
from entente.data.synthetic import write_jsonl as write_games_jsonl
from entente.eval.lambda_sweep import SweepRecord, write_sweep_jsonl
from entente.eval.press_results import PressRecord, write_press_jsonl
from entente.eval.tournament import SeatedOutcome, write_seated_jsonl
from entente.negotiation.cli import _write_json
from entente.negotiation.data.instructions import InstructionExample, write_instruction_jsonl
from entente.sim.parallel import GameOutcome
from entente.sim.parallel import write_jsonl as write_outcomes_jsonl
from entente.training.tracking import MetricTracker

_OUTCOME = GameOutcome(
    seed=0,
    ruleset="standard",
    map_name="standard",
    max_year=1901,
    num_phases=1,
    final_phase="F1901M",
    winner=None,
    supply_counts={"FRANCE": 3},
)


def _assert_lf_only(path: Path) -> None:
    data = path.read_bytes()
    assert data, f"{path.name} is empty"
    assert b"\r" not in data, f"{path.name} contains CR bytes (platform newline leak)"


def test_tournament_jsonl_is_lf(tmp_path: Path) -> None:
    path = tmp_path / "seated.jsonl"
    write_seated_jsonl([SeatedOutcome(outcome=_OUTCOME, seats={"FRANCE": "greedy"})], path)
    _assert_lf_only(path)


def test_sim_outcomes_jsonl_is_lf(tmp_path: Path) -> None:
    path = tmp_path / "outcomes.jsonl"
    write_outcomes_jsonl([_OUTCOME], path)
    _assert_lf_only(path)


def test_lambda_sweep_jsonl_is_lf(tmp_path: Path) -> None:
    record = SweepRecord(
        lam=1.0,
        seat="FRANCE",
        seed=0,
        sos=0.25,
        supply_centers=5,
        survived=True,
        anchor_agreement=None,
        human_agreement=None,
    )
    path = tmp_path / "sweep.jsonl"
    write_sweep_jsonl([record], path)
    _assert_lf_only(path)


def test_press_jsonl_is_lf(tmp_path: Path) -> None:
    record = PressRecord(
        press=True, seat="FRANCE", seed=0, sos=0.25, supply_centers=5, survived=True
    )
    path = tmp_path / "press.jsonl"
    write_press_jsonl([record], path)
    _assert_lf_only(path)


def test_synthetic_games_jsonl_is_lf(tmp_path: Path) -> None:
    path = tmp_path / "games.jsonl"
    write_games_jsonl([{"id": "synthetic-0", "map": "standard", "phases": []}], path)
    _assert_lf_only(path)


def test_instruction_jsonl_is_lf(tmp_path: Path) -> None:
    example = InstructionExample(
        prompt="p", completion="c", truthful=True, game_id=1, message_index=0
    )
    path = tmp_path / "instructions.jsonl"
    write_instruction_jsonl([example], path)
    _assert_lf_only(path)


def test_metric_tracker_jsonl_is_lf(tmp_path: Path) -> None:
    with MetricTracker(tmp_path, "run") as tracker:
        tracker.log(1, {"loss": 1.0})
    _assert_lf_only(tmp_path / "run.metrics.jsonl")


def test_transcript_json_is_lf(tmp_path: Path) -> None:
    path = tmp_path / "transcript.json"
    _write_json(path, {"scenario": "s1", "messages": []})
    _assert_lf_only(path)


def test_cli_game_save_is_lf(tmp_path: Path) -> None:
    out = tmp_path / "game.json"
    code = main(
        ["play", "--agents", "random", "--seed", "3", "--max-year", "1901", "--out", str(out)]
    )
    assert code == 0
    _assert_lf_only(out)
