"""Train the policy and value heads by behavioral cloning on human games.

One step: encode a batch of boards once, decode every board's units under teacher
forcing (policy cross-entropy + top-1), and predict the 7-power Sum-of-Squares
share (value cross-entropy against the soft final-outcome target). AdamW with a
linear-warmup cosine schedule; autocast on CUDA -- bf16 where the card supports
it natively (Ampere and later), fp16 with gradient scaling on older cards;
CPU runs fp32. Shuffling goes through
:class:`~entente.data.dataset.ShardLocalShuffle` (row-random order across shard
boundaries thrashes the shard cache -- measured 51x per-row slowdown).
The held-out metric is per-unit top-1 on a game-disjoint validation split.

Both the teacher-forced loss and evaluation run the policy head's *batched*
paths (:meth:`~entente.nn.heads.PolicyHead.teacher_forced_batched` /
:meth:`~entente.nn.heads.PolicyHead.decode_batched`): under teacher forcing the
AR inputs are known up front, and at decode time boards are independent, so
slot positions advance in lockstep across the whole batch. The per-board loop
variants are the reference implementations, and
tests/nn/test_batched_equivalence.py pins the batched paths equal to them; the
batching is what makes a full-size run finish at all, since the loop measures
under 2 steps/min on an RTX 2060 with the GPU saturated by per-slot call
overhead. In-loop validation snapshots use a fixed
2048-row probe; the acceptance top-1 is the *full* held-out evaluation
scripts/train_bc.py runs after training.
"""

from __future__ import annotations

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from entente.data.dataset import (
    Batch,
    ShardDataset,
    ShardLocalShuffle,
    collate,
    load_bc_datasets,
)
from entente.nn.model import BCModel
from entente.training.config import BCConfig
from entente.training.tracking import MetricTracker

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Summary of a finished run (the full curve is in the JSONL metrics file)."""

    steps: int
    final_loss: float
    train_top1: float
    val_top1: float | None
    checkpoint_path: Path
    metrics_path: Path


def resolve_device(spec: str) -> torch.device:
    """``"auto"`` -> CUDA if available else CPU; anything else is taken verbatim."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against soft targets (the SoS share distribution)."""
    return -(targets * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


def _cosine_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _forward_policy(
    model: BCModel, enc: torch.Tensor, batch: Batch
) -> tuple[torch.Tensor, int, int]:
    """Score one encoded batch's policy under teacher forcing, in a single pass.

    Returns the loss summed over slots, the top-1-correct count and the slot
    count.
    """
    encs = [enc[b, batch.slot_locations[b]] for b in range(enc.shape[0])]
    return model.policy.teacher_forced_batched(encs, batch.legal_ids, batch.label_ids)


@torch.no_grad()
def evaluate_top1(
    model: BCModel,
    dataset: ShardDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
    max_examples: int | None = None,
) -> tuple[float, int]:
    """Measure greedy per-unit top-1 accuracy over ``dataset``.

    This is the acceptance metric, taken on the held-out split. Returns
    ``(top1, unit_count)``, or ``(0.0, 0)`` for an empty dataset.
    """
    if len(dataset) == 0:
        return 0.0, 0
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate)
    correct = total = seen = 0
    for batch in loader:
        enc = model.encode(batch.features.to(device))
        encs = [enc[b, batch.slot_locations[b]] for b in range(enc.shape[0])]
        decoded = model.policy.decode_batched(encs, batch.legal_ids)
        for labels, chosen in zip(batch.label_ids, decoded, strict=True):
            if not labels:
                continue
            correct += sum(int(c == label) for c, label in zip(chosen, labels, strict=True))
            total += len(labels)
        seen += batch.features.shape[0]
        if max_examples is not None and seen >= max_examples:
            break
    return (correct / total if total else 0.0), total


def _batches(loader: DataLoader[object], device: torch.device) -> Iterator[Batch]:
    for batch in loader:
        yield batch.to(device)


def train(
    cfg: BCConfig,
    *,
    train_ds: ShardDataset | None = None,
    val_ds: ShardDataset | None = None,
) -> TrainResult:
    """Run behavioral cloning as ``cfg`` describes it.

    Datasets are built from ``cfg.data`` unless a caller supplies them, which is
    how the tests run tiny ones. Writes a JSONL metrics stream and a checkpoint
    under ``cfg.out_dir``, and returns a summary of the run.
    """
    _seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    if train_ds is None or val_ds is None:
        train_ds, val_ds = load_bc_datasets(cfg.data)
    if len(train_ds) == 0:
        # an empty DataLoader yields no batches, so the `while step < max_steps`
        # loop below would busy-spin forever without advancing `step`; a
        # misspelled/empty `data.shards_dir` produces exactly this, so fail
        # loudly instead of hanging.
        raise ValueError(
            f"training dataset is empty (shards_dir={cfg.data.shards_dir!r}); "
            "check the path or preprocess data first"
        )

    model = BCModel(
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        ff_mult=cfg.model.ff_mult,
        dropout=cfg.model.dropout,
        independent_heads=cfg.model.independent_heads,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: _cosine_warmup(step, cfg.optim.warmup_steps, cfg.optim.max_steps)
    )
    # bf16 autocast needs Ampere or later; an older CUDA card runs fp16 autocast
    # with loss scaling instead -- native on Turing tensor cores, and half the
    # activation memory of the fp32 alternative. The scaler is a
    # pass-through everywhere else (bf16 needs no scaling; CPU runs fp32).
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    # shuffle via the shard-local sampler, not DataLoader(shuffle=True): random
    # row order across shard boundaries thrashes the dataset's shard cache
    # (measured 51x slower per row on a full-size shard set -- it starved the GPU).
    sampler = ShardLocalShuffle(train_ds, seed=cfg.seed) if cfg.data.shuffle else None
    loader: DataLoader[object] = DataLoader(
        train_ds,
        batch_size=cfg.optim.batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=cfg.data.num_workers,
        persistent_workers=cfg.data.num_workers > 0,
        drop_last=False,
    )

    tracker = MetricTracker(cfg.out_dir, cfg.run_name, config=cfg.model_dump())
    step = 0
    last_loss = last_top1 = 0.0
    val_top1: float | None = None
    try:
        while step < cfg.optim.max_steps:
            for batch in _batches(loader, device):
                model.train()
                amp = torch.autocast("cuda", dtype=amp_dtype) if use_amp else nullcontext()
                with amp:
                    enc = model.encode(batch.features)
                    policy_sum, correct, slots = _forward_policy(model, enc, batch)
                    policy_loss = policy_sum / max(slots, 1)
                    value_loss = _soft_cross_entropy(model.value(enc), batch.value_targets)
                    loss = policy_loss + cfg.optim.value_loss_weight * value_loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                if cfg.optim.grad_clip > 0.0:
                    # clip_grad_norm_(..., 0.0) scales every gradient to ~0, so
                    # the common "0 = off" convention would silently freeze the
                    # run; skip the call instead.
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                scaler.step(opt)
                scaler.update()
                scheduler.step()
                step += 1
                last_loss = float(loss.item())
                last_top1 = correct / max(slots, 1)

                if step % cfg.log_every == 0:
                    tracker.log(
                        step,
                        {
                            "loss": last_loss,
                            "policy_loss": float(policy_loss.item()),
                            "value_loss": float(value_loss.item()),
                            "train_top1": last_top1,
                            "lr": scheduler.get_last_lr()[0],
                        },
                    )
                if step % cfg.eval_every == 0:
                    # fixed probe: a training-curve snapshot, NOT the acceptance
                    # metric (train_bc.py evaluates the full split once at the end)
                    val_top1, _ = evaluate_top1(model, val_ds, device, max_examples=2048)
                    tracker.log(step, {"val_top1": val_top1})
                if step >= cfg.optim.max_steps:
                    break
    finally:
        tracker.close()

    checkpoint = Path(cfg.out_dir) / f"{cfg.run_name}.pt"
    model.save(checkpoint)
    return TrainResult(step, last_loss, last_top1, val_top1, checkpoint, tracker.path)
