r"""Run the full-game press-vs-no-press evaluation and write the JSONL.

    uv run --group ml python scripts/run_press_eval.py \
        --config configs/press/smoke.yaml [--set key=value ...]

Seven piKL agents sharing one BC checkpoint play full self-play games twice
per seed -- once with a mechanical `StubProvider`-only negotiation round each
movement phase, once without -- and every seat's outcome becomes one
:class:`entente.eval.press_results.PressRecord`. The JSONL is the artifact;
the table and the press-vs-no-press SoS delta regenerate from
it alone via ``entente.eval.press_results.render_table``/``sos_delta``.
"""

from __future__ import annotations

import argparse
import sys

from entente.eval.press_results import render_table, sos_delta, write_press_jsonl
from entente.negotiation.press_eval import PressEvalConfig, run_press_eval
from entente.training.config import load_config


def main(argv: list[str] | None = None) -> int:
    """Run the press evaluation named by the config and write its JSONL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="PressEvalConfig YAML")
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        default=None,
        metavar="KEY=VALUE",
        help="dotted config override, repeatable (e.g. --set games=20)",
    )
    args = parser.parse_args(argv)

    cfg = load_config(PressEvalConfig, args.config, overrides=args.overrides or ())
    records = run_press_eval(cfg)
    count = write_press_jsonl(records, cfg.output)
    print(f"wrote {count} records to {cfg.output}")
    print(render_table(records))
    delta, p_value = sos_delta(records)
    print(f"press - no_press mean SoS delta = {delta:+.4f} (permutation p={p_value:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
