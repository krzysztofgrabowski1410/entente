"""Probe whether parsed-intent conditioning predicts the opponent's next move better.

The comparison is against the same subgame with no message at all.

    uv run python scripts/probe_negotiation.py [--out docs/results/negotiation_probe.jsonl]

For each committed scenario (``src/entente/negotiation/scenarios/*.json``),
runs piKL twice over the ``{proposer, ally}`` subgame
(:mod:`entente.negotiation.scenario`) -- once with the proposer's raw anchor,
once with it reweighted by parsing the scenario's scripted incoming message --
and reads off the probability each run assigns to the proposer's true
(scripted) next move. Honest, small scale by construction (candidates and
utility are hand-authored, not BC/torch-derived -- see negotiation/README.md
Sec. Limitations): 3 scripted scenarios, no claim beyond what they show.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from entente.negotiation.scenario import load_scenario, run_predictions

DEFAULT_SCENARIOS_DIR = Path("src/entente/negotiation/scenarios")
DEFAULT_OUTPUT = Path("docs/results/negotiation_probe.jsonl")


def probe_scenario(path: Path) -> dict[str, Any]:
    """Run one scenario both ways and record what each run gave the true move."""
    scenario = load_scenario(path)
    result = run_predictions(scenario)
    baseline_p = result.baseline.mean_strategies[0][scenario.true_move_index]
    conditioned_p = result.conditioned.mean_strategies[0][scenario.true_move_index]
    return {
        "scenario": scenario.name,
        "proposer": scenario.proposer,
        "ally": scenario.ally,
        "true_move_index": scenario.true_move_index,
        "baseline_probability_of_true_move": baseline_p,
        "conditioned_probability_of_true_move": conditioned_p,
        "delta": conditioned_p - baseline_p,
        "note": scenario.note,
    }


def render_table(records: list[dict[str, Any]]) -> str:
    """Format the probe records as a fixed-width table for the terminal."""
    header = (
        f"{'scenario':10s} {'proposer':10s} {'baseline':>10s} {'conditioned':>12s} {'delta':>8s}"
    )
    lines = [header]
    for record in records:
        lines.append(
            f"{record['scenario']:10s} {record['proposer']:10s} "
            f"{record['baseline_probability_of_true_move']:10.4f} "
            f"{record['conditioned_probability_of_true_move']:12.4f} "
            f"{record['delta']:+8.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Probe every committed scenario and write the records as JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios-dir",
        default=str(DEFAULT_SCENARIOS_DIR),
        help=f"directory of scenario JSON files (default: {DEFAULT_SCENARIOS_DIR})",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUTPUT), help=f"JSONL output path (default: {DEFAULT_OUTPUT})"
    )
    args = parser.parse_args(argv)

    scenario_dir = Path(args.scenarios_dir)
    paths = sorted(scenario_dir.glob("*.json"))
    if not paths:
        print(f"no scenario JSON files found under {scenario_dir}", file=sys.stderr)
        return 1

    records = [probe_scenario(path) for path in paths]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(render_table(records))
    mean_delta = sum(record["delta"] for record in records) / len(records)
    print(f"\nmean delta over {len(records)} scenarios: {mean_delta:+.4f}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
