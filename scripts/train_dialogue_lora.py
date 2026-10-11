r"""CLI: QLoRA/LoRA fine-tune a dialogue model from a config.

    uv run --group finetune python scripts/train_dialogue_lora.py \
        --config src/entente/negotiation/finetune/configs/smoke.yaml

Smoke config trains a plain LoRA adapter over a ~112k-parameter public test
model on CPU in well under a minute -- proves the loop and the
:class:`~entente.negotiation.generate.LocalLoraProvider` round-trip. The base
config (``configs/base.yaml``) targets the real ~3B model in 4-bit and needs
WSL2/CUDA (see docs/setup.md).
"""

from __future__ import annotations

import argparse
import sys

import torch

from entente.negotiation.finetune.config import DialogueLoraConfig
from entente.negotiation.finetune.train import resolve_device, train
from entente.training.config import load_config


def main(argv: list[str] | None = None) -> int:
    """Fine-tune from the config and write the adapter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a dialogue-LoRA YAML config")
    parser.add_argument(
        "--set", action="append", metavar="KEY=VALUE", help="dotted config override (repeatable)"
    )
    args = parser.parse_args(argv)

    cfg = load_config(DialogueLoraConfig, args.config, overrides=args.set or ())
    device = resolve_device(cfg.device)
    print(f"cuda_available={torch.cuda.is_available()} device={device} base_model={cfg.base_model}")

    result = train(cfg)
    print(
        f"steps={result.steps} final_loss={result.final_loss:.4f} "
        f"adapter={result.adapter_dir} metrics={result.metrics_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
