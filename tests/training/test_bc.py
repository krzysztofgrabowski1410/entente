"""Pin the training loop's pieces: schedule, losses, device and a short full run.

That the loop actually learns is checked by the slow smoke test instead
(``ml`` group).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")
import torch

from entente.data.dataset import ShardDataset
from entente.data.paquette import game_from_json
from entente.data.preprocess import preprocess
from entente.data.synthetic import synthesize_games
from entente.nn.model import BCModel
from entente.training.bc import (
    _cosine_warmup,
    _soft_cross_entropy,
    evaluate_top1,
    resolve_device,
    train,
)
from entente.training.config import BCConfig, load_config

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def tiny_shards(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("bc_shards")
    games = [
        game_from_json(g)
        for g in synthesize_games(range(8), agent="random", teacher="greedy", max_year=1902)
    ]
    preprocess(games, out, shard_size=100_000)
    return out


def test_resolve_device_cpu() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda"}


def test_cosine_warmup_shape() -> None:
    assert _cosine_warmup(0, 10, 100) < _cosine_warmup(9, 10, 100)  # ramps up
    assert _cosine_warmup(10, 10, 100) == pytest.approx(1.0)  # peak at warmup end
    assert _cosine_warmup(100, 10, 100) == pytest.approx(0.0, abs=1e-6)  # cosine floor


def test_soft_cross_entropy_matches_uniform() -> None:
    logits = torch.zeros(1, 7)
    target = torch.full((1, 7), 1 / 7)
    assert float(_soft_cross_entropy(logits, target)) == pytest.approx(math.log(7), abs=1e-5)


def test_evaluate_top1_returns_a_fraction(tiny_shards: Path) -> None:
    dataset = ShardDataset(tiny_shards)
    model = BCModel(d_model=32, n_layers=1, n_heads=2)
    top1, units = evaluate_top1(model, dataset, torch.device("cpu"), max_examples=8)
    assert units > 0
    assert 0.0 <= top1 <= 1.0


def test_evaluate_top1_empty_dataset() -> None:
    dataset = ShardDataset.__new__(ShardDataset)
    dataset._index = []  # type: ignore[attr-defined]
    model = BCModel(d_model=32, n_layers=1, n_heads=2)
    assert evaluate_top1(model, dataset, torch.device("cpu")) == (0.0, 0)


def test_cli_eval_only_evaluates_without_training(tiny_shards: Path, tmp_path: Path) -> None:
    """--eval-only finishes a run whose training already completed.

    This is the recovery path: the steps ran and the checkpoint saved, then the
    final evaluation failed. It must load ``out_dir/run_name.pt`` and print the
    acceptance line without training a single step, so a long run is never
    repeated to recover one number.
    """
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    torch.manual_seed(0)
    BCModel(d_model=32, n_layers=1, n_heads=2).save(tmp_path / "bc_smoke.pt")
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "train_bc.py"),
            "--config",
            "src/entente/training/configs/bc/smoke.yaml",
            "--set",
            f"out_dir={tmp_path.as_posix()}",
            "--set",
            f"data.shards_dir={tiny_shards.as_posix()}",
            "--eval-only",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo,
    )
    assert "held-out per-unit top-1=" in result.stdout
    assert "steps=" not in result.stdout  # no training happened
    assert not (tmp_path / "bc_smoke.metrics.jsonl").exists()  # really no training


def test_train_raises_on_empty_dataset(tmp_path: Path) -> None:
    # an empty DataLoader yields no batches, so the training while-loop would
    # busy-spin forever; the guard must fail loudly instead.
    empty = ShardDataset.__new__(ShardDataset)
    empty._index = []  # type: ignore[attr-defined]
    cfg = load_config(
        BCConfig,
        "src/entente/training/configs/bc/smoke.yaml",
        overrides=[f"out_dir={tmp_path}", "optim.max_steps=4"],
    )
    with pytest.raises(ValueError, match="empty"):
        train(cfg, train_ds=empty, val_ds=empty)


def test_grad_clip_zero_does_not_freeze_gradients(tiny_shards: Path, tmp_path: Path) -> None:
    # grad_clip=0.0 must SKIP clipping, not scale every gradient to ~0. A run
    # with grad_clip=0 must therefore match one with a clip so large it never
    # triggers (both are effectively unclipped); before the fix the former
    # froze at the init weights and diverged.
    def run(grad_clip: float, name: str) -> BCModel:
        dataset = ShardDataset(tiny_shards)
        cfg = load_config(
            BCConfig,
            "src/entente/training/configs/bc/smoke.yaml",
            overrides=[
                f"out_dir={tmp_path / name}",
                "optim.max_steps=10",
                "log_every=10",
                "data.shuffle=false",
                f"optim.grad_clip={grad_clip}",
            ],
        )
        result = train(cfg, train_ds=dataset, val_ds=dataset)
        return BCModel.load(result.checkpoint_path)

    unclipped = run(1e9, "huge")  # a clip norm this large never actually clips
    zero = run(0.0, "zero")
    for (name_a, pa), (_, pb) in zip(
        unclipped.named_parameters(), zero.named_parameters(), strict=True
    ):
        assert torch.allclose(pa, pb, rtol=1e-4, atol=1e-6), f"grad_clip=0 diverged at {name_a}"


def test_short_train_writes_checkpoint_and_metrics(tiny_shards: Path, tmp_path: Path) -> None:
    dataset = ShardDataset(tiny_shards)
    cfg = load_config(
        BCConfig,
        "src/entente/training/configs/bc/smoke.yaml",
        overrides=[f"out_dir={tmp_path}", "optim.max_steps=6", "log_every=2", "eval_every=4"],
    )
    result = train(cfg, train_ds=dataset, val_ds=dataset)
    assert result.steps == 6
    assert result.checkpoint_path.exists()
    assert result.metrics_path.exists()
    # the checkpoint reloads
    BCModel.load(result.checkpoint_path)
