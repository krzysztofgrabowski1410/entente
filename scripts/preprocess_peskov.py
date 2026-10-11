r"""Convert one split of the Peskov "It Takes Two to Lie" corpus for fine-tuning.

The output is stance-conditioned instruction-tuning pairs.

    uv run python scripts/preprocess_peskov.py \
        --input data/peskov/train.jsonl --out data/peskov_instructions/train.jsonl

Run once per split (``train``/``validation``/``test``); docs/setup.md,
"Negotiation dialogue data", carries the download commands and the CC-BY 4.0
attribution note. No network access and no ``ml``/``finetune``
group needed; this is stdlib + :mod:`entente.negotiation.data`.
"""

from __future__ import annotations

import argparse
import sys

from entente.negotiation.data.instructions import to_instruction_examples, write_instruction_jsonl
from entente.negotiation.data.peskov import PeskovJSONLSource


def main(argv: list[str] | None = None) -> int:
    """Convert one corpus split into instruction-tuning pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="a Peskov split .jsonl (train/val/test)")
    parser.add_argument("--out", required=True, help="output instruction-pairs .jsonl path")
    parser.add_argument("--limit", type=int, default=None, help="cap conversations read (debug)")
    args = parser.parse_args(argv)

    source = PeskovJSONLSource(args.input, limit=args.limit)
    count = write_instruction_jsonl(to_instruction_examples(source), args.out)
    print(f"read {args.input}: {count} instruction pairs -> {args.out}")
    if source.skipped:
        print(f"skipped {len(source.skipped)} malformed line(s): {source.skipped[:3]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
