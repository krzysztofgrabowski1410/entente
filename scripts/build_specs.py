r"""Build the typeset documents, LaTeX to PDF: the two specifications.

``latexmk`` drives ``pdflatex`` to a fixpoint for each document. latexmk exits 0 when a
reference or a label is left unresolved, so the log is read afterwards and an undefined
reference, an undefined citation or a multiply defined label fails the build: a broken
cross-reference never ships inside a committed PDF. The auxiliary files are removed after
each build unless ``--keep-aux`` is given, so the tree holds only the ``.tex`` and the
``.pdf``.

Needs ``latexmk`` and ``pdflatex`` on PATH (MiKTeX or TeX Live); the test suite does not,
so CI never runs this.

    uv run python scripts/build_specs.py                  # every document
    uv run python scripts/build_specs.py docs/pikl.tex    # one of them
    uv run python scripts/build_specs.py --keep-aux       # leave the .log for inspection
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    ROOT / "docs" / "adjudication.tex",
    ROOT / "docs" / "pikl.tex",
)

# What a finished latexmk run still leaves in the log when a reference did not resolve.
_UNRESOLVED = re.compile(
    r"There were undefined references"
    r"|Reference `[^']*' on page \d+ undefined"
    r"|Citation `[^']*' on page \d+ undefined"
    r"|There were multiply-defined labels"
)


def build(tex: Path, keep_aux: bool) -> int:
    """Compile one document; non-zero on a TeX error or on an unresolved reference."""
    command = [
        "latexmk",
        "-pdf",
        "-cd",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-file-line-error",
        str(tex),
    ]
    code = subprocess.run(command, check=False).returncode
    log = tex.with_suffix(".log")
    if code == 0 and log.exists():
        problem = _UNRESOLVED.search(log.read_text(encoding="utf-8", errors="replace"))
        if problem:
            print(f"{tex.relative_to(ROOT)}: {problem.group(0)}", file=sys.stderr)
            code = 1
    if not keep_aux:
        subprocess.run(
            ["latexmk", "-c", "-cd", str(tex)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return code


def main(argv: list[str] | None = None) -> int:
    """Build the documents named on the command line, or all of them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("documents", nargs="*", help="the .tex files to build (default: all)")
    parser.add_argument("--keep-aux", action="store_true", help="keep latexmk's auxiliary files")
    args = parser.parse_args(argv)
    documents = [Path(d).resolve() for d in args.documents] or list(DOCUMENTS)
    worst = 0
    for tex in documents:
        code = build(tex, args.keep_aux)
        status = "built" if code == 0 else f"FAILED ({code})"
        print(f"{tex.relative_to(ROOT)}: {status}")
        worst = max(worst, code)
    return worst


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
