"""CLI: train the behavioral-cloning model from a config.

    uv run --group ml python scripts/train_bc.py \
        --config src/entente/training/configs/bc/base.yaml \
        --set data.shards_dir=data/processed

Trains per the config, prints held-out per-unit top-1 -- the acceptance metric,
with a target of at least 50% -- and -- with ``--eval-vs-greedy`` -- runs a BC-vs-greedy 1-vs-6
tournament and prints BC's mean SoS with a bootstrap CI. Full training is the
WSL2/CUDA path (docs/setup.md); on a CPU box this runs but will not reach the
target within a reasonable budget.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from entente.agents.bc_agent import bc_vs_greedy
from entente.data.dataset import load_bc_datasets
from entente.nn.model import BCModel
from entente.training.bc import evaluate_top1, resolve_device, train
from entente.training.config import BCConfig, load_config


def main(argv: list[str] | None = None) -> int:
    """Train from the config, then print the held-out acceptance metric."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a BC YAML config")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="dotted config override (repeatable)"
    )
    parser.add_argument(
        "--eval-vs-greedy", action="store_true", help="also run a BC-vs-greedy tournament"
    )
    parser.add_argument("--eval-seeds", type=int, default=30, help="seeds/seat for the tournament")
    parser.add_argument(
        "--eval-jsonl",
        metavar="PATH",
        help="write the tournament's per-game seated JSONL here (the committable "
        "artifact the result tables regenerate from)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="skip training and evaluate the config's existing checkpoint "
        "(out_dir/run_name.pt) -- for finishing a run whose training already "
        "completed",
    )
    args = parser.parse_args(argv)
    if args.eval_jsonl and not args.eval_vs_greedy:
        parser.error("--eval-jsonl requires --eval-vs-greedy")

    cfg = load_config(BCConfig, args.config, overrides=args.set or ())
    device = resolve_device(cfg.device)
    print(f"cuda_available={torch.cuda.is_available()} device={device}")

    if args.eval_only:
        checkpoint_path = Path(cfg.out_dir) / f"{cfg.run_name}.pt"
        if not checkpoint_path.exists():
            parser.error(f"--eval-only: no checkpoint at {checkpoint_path}")
        prefix = ""
    else:
        result = train(cfg)
        checkpoint_path = result.checkpoint_path
        prefix = f"steps={result.steps} final_loss={result.final_loss:.3f} "
    model = BCModel.load(str(checkpoint_path), map_location=device)
    _, val_ds = load_bc_datasets(cfg.data)
    top1, units = evaluate_top1(model, val_ds, device)
    print(
        f"{prefix}held-out per-unit top-1={top1:.3f} over {units} units (target >=0.50); "
        f"checkpoint={checkpoint_path}"
    )

    if args.eval_vs_greedy:
        res = bc_vs_greedy(
            str(checkpoint_path),
            list(range(args.eval_seeds)),
            device=str(device),
            output=args.eval_jsonl,
        )
        print(
            f"BC-vs-greedy: games={res.games} bc_mean_sos={res.bc_mean_sos:.4f} "
            f"CI=({res.bc_ci[0]:.4f}, {res.bc_ci[1]:.4f}) greedy_mean_sos={res.greedy_mean_sos:.4f}"
        )
        if args.eval_jsonl:
            print(f"wrote per-game seated JSONL -> {args.eval_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
