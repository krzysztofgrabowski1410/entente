"""Overfit a few greedy-teacher games on CPU and assert the loop actually learns.

Loss must fall and per-unit top-1 must rise past a sanity bar. Slow and
ml-marked. This is a plumbing proof that the gradients reach the weights, not
the held-out quality run, which needs a GPU and the base config (docs/setup.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
import torch

from entente.data.paquette import game_from_json
from entente.data.preprocess import preprocess
from entente.data.synthetic import synthesize_games

pytestmark = [pytest.mark.ml, pytest.mark.slow]

SMOKE_CONFIG = "src/entente/training/configs/bc/smoke.yaml"


@pytest.fixture(scope="module")
def overfit_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("smoke_proc")
    games = [
        game_from_json(g)
        for g in synthesize_games(range(30), agent="random", teacher="greedy", max_year=1903)
    ]
    preprocess(games, out, shard_size=100_000)
    return out


def test_smoke_overfit_learns(overfit_shards: Path, tmp_path: Path) -> None:
    from entente.data.dataset import ShardDataset
    from entente.nn.model import BCModel
    from entente.training.bc import evaluate_top1, train
    from entente.training.config import BCConfig, load_config

    dataset = ShardDataset(overfit_shards)
    assert len(dataset) > 100
    cfg = load_config(
        BCConfig,
        SMOKE_CONFIG,
        overrides=[f"out_dir={tmp_path / 'out'}", "optim.max_steps=220", "log_every=20"],
    )
    result = train(cfg, train_ds=dataset, val_ds=dataset)

    losses = [r["loss"] for r in _read_jsonl(result.metrics_path) if "loss" in r]
    assert len(losses) >= 3
    assert losses[-1] < 0.75 * losses[0], f"loss did not fall: {losses[0]:.2f} -> {losses[-1]:.2f}"

    model = BCModel.load(result.checkpoint_path)
    top1, units = evaluate_top1(model, dataset, torch.device("cpu"))
    assert units > 0
    assert top1 > 0.45, f"train top-1 too low after overfit: {top1:.3f}"


def _read_jsonl(path: Path) -> list[dict[str, float]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
