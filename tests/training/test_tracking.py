"""Pin metric tracking: the JSONL sink is always written, the hosted one is opt-in.

Torch-free, so these run in the default environment.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from entente.training.tracking import MetricTracker


def test_writes_jsonl_and_no_wandb_without_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with MetricTracker(tmp_path, "run", config={"lr": 0.1}) as tracker:
        assert tracker._wandb is None
        tracker.log(1, {"loss": 2.0, "top1": 0.1})
        tracker.log(2, {"loss": 1.5, "top1": 0.3})

    lines = [json.loads(line) for line in tracker.path.read_text().splitlines()]
    assert [row["step"] for row in lines] == [1, 2]
    assert lines[0]["loss"] == 2.0
    assert lines[1]["top1"] == 0.3


def test_metrics_path_is_named_after_the_run(tmp_path: Path) -> None:
    tracker = MetricTracker(tmp_path, "bc_smoke")
    assert tracker.path == tmp_path / "bc_smoke.metrics.jsonl"
    tracker.close()


class _WandbBoom:
    """A stand-in wandb module whose every mirroring call fails."""

    def log(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("wandb died mid-run")

    def finish(self) -> None:
        raise RuntimeError("wandb died at shutdown")


def test_wandb_log_failure_mid_run_degrades_to_jsonl(tmp_path: Path, monkeypatch) -> None:
    # The module's promise covers the whole run, not only init: a mid-run wandb
    # error must drop the mirror, never the run.
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with MetricTracker(tmp_path, "run") as tracker:
        tracker._wandb = _WandbBoom()
        tracker.log(1, {"loss": 1.0})  # must not raise
        assert tracker._wandb is None  # mirror dropped, JSONL continues
        tracker.log(2, {"loss": 0.5})

    lines = tracker.path.read_text().splitlines()
    assert len(lines) == 2


def test_wandb_finish_failure_degrades(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    tracker = MetricTracker(tmp_path, "run")
    tracker.log(1, {"loss": 1.0})
    tracker._wandb = _WandbBoom()
    tracker.close()  # must not raise
    assert tracker.path.read_text().strip()


@pytest.mark.ml
def test_wandb_failure_degrades_to_jsonl(tmp_path: Path, monkeypatch) -> None:
    # With a key set, wandb is attempted; if init fails, the run must still write
    # JSONL and carry on (tracking never breaks training). `importorskip` (not a bare
    # import) so the plain `uv sync` gate SKIPS this -- the ml marker only categorizes,
    # it does not skip; every sibling torch test guards the same way.
    wandb = pytest.importorskip("wandb")

    monkeypatch.setenv("WANDB_API_KEY", "fake-key")

    def _boom(**_: object) -> object:
        raise RuntimeError("no network")

    monkeypatch.setattr(wandb, "init", _boom)
    with MetricTracker(tmp_path, "run") as tracker:
        assert tracker._wandb is None
        tracker.log(1, {"loss": 1.0})
    assert tracker.path.read_text().strip()
