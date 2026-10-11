"""End-to-end lambda-sweep wiring, on a tiny CPU config (docs/pikl.tex Sec. 8).

The real CLI -- the committed sweep YAML plus dotted overrides -- runs as a
subprocess over a random-weight checkpoint, and its JSONL round-trips into the
aggregate table, so the whole path is exercised rather than mocked.

Quality is asserted nowhere. A random anchor cannot play well, and a smoke-scale
run is evidence that the wiring works and nothing more; the quality claims live
in tests/search/, against exact oracles.
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from entente.eval.lambda_sweep import aggregate, read_sweep_jsonl, render_table
from entente.nn.model import BCModel

pytestmark = [pytest.mark.ml, pytest.mark.slow]

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "run_lambda_sweep.py"
SWEEP_YAML = REPO / "configs" / "pikl" / "lambda_sweep.yaml"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    model = BCModel(d_model=32, n_layers=1, n_heads=4)
    path = tmp_path_factory.mktemp("ckpt") / "tiny.pt"
    model.save(path)
    return path


def test_sweep_cli_end_to_end_tiny(checkpoint: Path, tmp_path: Path) -> None:
    output = tmp_path / "sweep.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(SWEEP_YAML),
            "--set",
            f"agent.checkpoint={checkpoint.as_posix()}",
            "--set",
            "agent.search.k=2",
            "--set",
            "agent.search.attempts=4",
            "--set",
            "agent.search.iterations=3",
            "--set",
            "games=1",
            "--set",
            "rotate_seats=false",
            "--set",
            "max_year=1901",
            "--set",
            "lambdas=[0.0, .inf]",
            "--set",
            f"output={output.as_posix()}",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    assert "2 records in" in result.stdout
    assert "| lambda | Games |" in result.stdout  # the regenerable table printed

    records = list(read_sweep_jsonl(output))
    assert len(records) == 2  # one game per lambda, one seat
    for record in records:
        assert record.seat == "FRANCE"
        assert 0.0 <= record.sos <= 1.0
        assert record.anchor_agreement is None or 0.0 <= record.anchor_agreement <= 1.0
        assert record.human_agreement is None  # no shards configured

    points = aggregate(records)
    assert [p.lam for p in points] == [0.0, math.inf]
    assert "| inf |" in render_table(points)


def _sweep_argv(checkpoint: Path, output: Path, games: int) -> list[str]:
    return [
        sys.executable, str(SCRIPT), "--config", str(SWEEP_YAML),
        "--set", f"agent.checkpoint={checkpoint.as_posix()}",
        "--set", "agent.search.k=2", "--set", "agent.search.attempts=4",
        "--set", "agent.search.iterations=3", "--set", f"games={games}",
        "--set", "rotate_seats=true", "--set", "max_year=1901",
        "--set", "lambdas=[0.0, .inf]", "--set", f"output={output.as_posix()}",
    ]  # fmt: skip


def test_sweep_resumes_and_matches_an_uninterrupted_run(checkpoint: Path, tmp_path: Path) -> None:
    """A kill and restart must land exactly where a clean run lands.

    The output is pre-seeded with a genuinely complete seat plus a half-written
    one; the CLI then skips the complete seat, discards the partial one,
    finishes the rest, and ends identical to an uninterrupted run of the same
    command.
    """
    from dataclasses import replace

    from entente.eval.lambda_sweep import append_sweep_jsonl

    clean = tmp_path / "clean.jsonl"
    subprocess.run(_sweep_argv(checkpoint, clean, 2), check=True, cwd=REPO, capture_output=True)
    clean_records = sorted(read_sweep_jsonl(clean), key=lambda r: (r.lam, r.seat, r.seed))
    assert len(clean_records) == 28  # 2 lambdas x 7 rotated seats x 2 games

    resumed = tmp_path / "resumed.jsonl"
    # a genuinely complete FRANCE/0.0 seat (both games, copied verbatim)...
    append_sweep_jsonl([r for r in clean_records if r.lam == 0.0 and r.seat == "FRANCE"], resumed)
    # ...and a HALF-written ITALY/inf seat (1 of 2 games, bogus sos): resume must
    # drop it and re-run the seat, not keep the poisoned partial.
    italy = next(r for r in clean_records if r.lam == math.inf and r.seat == "ITALY")
    append_sweep_jsonl([replace(italy, sos=0.999)], resumed)

    proc = subprocess.run(
        _sweep_argv(checkpoint, resumed, 2), check=True, cwd=REPO, capture_output=True, text=True
    )
    assert "already complete, skipping" in proc.stdout  # FRANCE/0.0 skipped
    assert "dropping 1 record" in proc.stdout  # the poisoned ITALY partial discarded

    resumed_records = sorted(read_sweep_jsonl(resumed), key=lambda r: (r.lam, r.seat, r.seed))
    assert resumed_records == clean_records  # identical; no 0.999 survivor, no double-count
