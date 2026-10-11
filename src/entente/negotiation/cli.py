"""Wire ``entente negotiate-demo`` into the CLI without a static import.

This lives inside :mod:`entente.negotiation`, not :mod:`entente.cli`, so the
one-way dependency rule holds: :mod:`entente.cli` reaches :func:`register`
through ``importlib.import_module`` -- a string, not a static ``import`` or
``from`` statement -- precisely so that deleting ``negotiation/`` leaves
:mod:`entente.cli` with nothing dangling to import.
``tests/negotiation/test_isolation.py`` greps the source tree for static
imports of this package; it does not (and should not) forbid this module
from existing, only forbid anything *outside* ``negotiation/`` from
importing it directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from entente.negotiation.generate import AnthropicProvider, LocalLoraProvider, StubProvider
from entente.negotiation.parse import MessageParseError, parse_message
from entente.negotiation.scenario import load_scenario, run_predictions

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_TRANSCRIPTS_DIR = Path("src/entente/negotiation/transcripts")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def _cmd_negotiate_demo(args: argparse.Namespace) -> int:
    path = Path(args.scenario)
    try:
        scenario = load_scenario(path)
    except OSError as error:
        print(f"cannot read {path}: {error}", file=sys.stderr)
        return 1
    except (KeyError, ValueError) as error:
        print(f"cannot load {path}: {error}", file=sys.stderr)
        return 1

    try:
        intent = parse_message(scenario.message)
    except MessageParseError as error:
        print(f"cannot parse scenario message: {error}", file=sys.stderr)
        return 1

    result = run_predictions(scenario)
    baseline_p = result.baseline.mean_strategies[0][scenario.true_move_index]
    conditioned_p = result.conditioned.mean_strategies[0][scenario.true_move_index]
    stub_message = StubProvider().generate(intent)

    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_TRANSCRIPTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript: dict[str, Any] = {
        "scenario": scenario.name,
        "self_power": scenario.self_power,
        "proposer": scenario.proposer,
        "ally": scenario.ally,
        "phase": scenario.phase,
        "incoming_message": scenario.message,
        "parsed_intent": intent.to_dict(),
        "regenerated_message": {"provider": "stub", "text": stub_message},
        "prediction": {
            "true_move_index": scenario.true_move_index,
            "baseline_mean_strategy": list(result.baseline.mean_strategies[0]),
            "conditioned_mean_strategy": list(result.conditioned.mean_strategies[0]),
            "baseline_probability_of_true_move": baseline_p,
            "conditioned_probability_of_true_move": conditioned_p,
        },
        "note": scenario.note,
    }
    stub_path = out_dir / f"{scenario.name}.json"
    _write_json(stub_path, transcript)
    print(f"wrote {stub_path}")
    print(
        f"P(true move of {scenario.proposer}) baseline={baseline_p:.4f} "
        f"conditioned={conditioned_p:.4f} delta={conditioned_p - baseline_p:+.4f}"
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            provider = AnthropicProvider()
            real_message = provider.generate(intent)
        except RuntimeError as error:
            print(f"ANTHROPIC_API_KEY is set but generation failed: {error}", file=sys.stderr)
            return 1
        real_transcript = {
            "scenario": scenario.name,
            "provider": "anthropic",
            "model": provider.model,
            "incoming_message": scenario.message,
            "parsed_intent": intent.to_dict(),
            "regenerated_message": real_message,
        }
        real_path = out_dir / f"{scenario.name}.anthropic.json"
        _write_json(real_path, real_transcript)
        print(
            f"wrote {real_path} (real-model transcript, provider=anthropic, model={provider.model})"
        )
    else:
        print("ANTHROPIC_API_KEY not set; skipping the real-model transcript (stub only).")

    # LocalLoraProvider is demonstrated here: its message is generated and shown,
    # never parsed back. The reweighting loop stays StubProvider-only, exactly as
    # AnthropicProvider's output is treated above, so no measured result depends
    # on a model's prose.
    lora_dir = os.environ.get("ENTENTE_DIALOGUE_LORA_DIR")
    if lora_dir:
        try:
            lora_provider = LocalLoraProvider(lora_dir)
            lora_message = lora_provider.generate(intent)
        except RuntimeError as error:
            print(
                f"ENTENTE_DIALOGUE_LORA_DIR is set but generation failed: {error}",
                file=sys.stderr,
            )
            return 1
        lora_transcript = {
            "scenario": scenario.name,
            "provider": "lora",
            "base_model": lora_provider.base_model,
            "adapter_dir": lora_provider.adapter_dir,
            "incoming_message": scenario.message,
            "parsed_intent": intent.to_dict(),
            "regenerated_message": lora_message,
        }
        lora_path = out_dir / f"{scenario.name}.lora.json"
        _write_json(lora_path, lora_transcript)
        print(
            f"wrote {lora_path} (real-model transcript, provider=lora, "
            f"base_model={lora_provider.base_model})"
        )
    else:
        print("ENTENTE_DIALOGUE_LORA_DIR not set; skipping the LoRA transcript (stub only).")

    return 0


def register(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add ``negotiate-demo`` to an existing ``entente`` subparser group."""
    parser = commands.add_parser(
        "negotiate-demo",
        help="run a scripted negotiation scenario and write a transcript (Phase 4 prototype)",
    )
    parser.add_argument("scenario", help="path to a negotiation scenario JSON file")
    parser.add_argument(
        "--out-dir",
        help=f"directory for the written transcript(s) (default: {DEFAULT_TRANSCRIPTS_DIR})",
    )
    parser.set_defaults(func=_cmd_negotiate_demo)


__all__ = ["register"]
