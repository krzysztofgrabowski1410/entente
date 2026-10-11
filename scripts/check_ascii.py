"""Enforce the keyboard-ASCII source policy.

Source text in this repository is keyboard-typeable ASCII (bytes 0x09, 0x0A,
0x0D, 0x20-0x7E): anyone on a US keyboard can read, grep and retype any source
file without an input-method detour, and a diff never hinges on an invisible
smart quote. Documentation is source too, so the docs and the LaTeX sources of
the specifications follow the same rule as the code; LaTeX is math markup, so
the PDFs still show real notation.

Scans the tracked tree, minus the exemptions below, for bytes outside
printable ASCII + tab/newline/carriage-return, and reports every offender as
``path:line:col: U+XXXX 'glyph'``. Exit status is 1 if any non-exempt file
carries a non-ASCII character, 0 otherwise. ``tests/test_ascii_policy.py``
runs this over the real tree so the policy cannot silently rot.

Each exemption has a reason:

- binary and generated artifacts (pictures, PDFs, the rulebook ``.docx``,
  ``uv.lock``): their bytes are not typed by hand, so the fix belongs in their
  source;
- fixtures that mirror an external corpus verbatim, and the committed result
  JSONL: changing a byte would make them stop mirroring their source;
- conformance data whose bytes the tests assert (the DATC and BATC cases, the
  map data): a byte change there is a conformance event that goes through
  those tests, never a blind sweep.

One rule cannot be written as a path: a cited proper name keeps its
diacritics, since a fact is not ours to respell; where the format offers a
character escape, the escape keeps the file ASCII.

Usage::

    uv run python scripts/check_ascii.py            # scan the whole tracked tree
    uv run python scripts/check_ascii.py path ...   # scan just these paths
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Suffixes that are binary or generated artifacts (bytes are not typed source).
_EXEMPT_SUFFIXES = frozenset({".png", ".gif", ".pdf", ".svg", ".docx", ".lock"})

# Directory prefixes (repo-relative, trailing slash) exempted wholesale.
_EXEMPT_PREFIXES = (
    ".git/",
    "tests/datc/cases/",  # hand-encoded conformance data whose bytes the tests assert
    "tests/batc/cases/",  # conformance data
    "tests/batc/cases_l2/",  # conformance data
    "tests/negotiation/fixtures/",  # external-corpus fixtures (verbatim)
    "tests/data/fixtures/",  # data fixtures
    "docs/results/",  # committed result JSONL (data, not prose)
    "src/entente/engine/map/data/",  # map data (conformance-visible)
    "src/entente/variants/beleriand/data/",  # map data
)


def is_exempt(rel_path: str) -> bool:
    """Whether a repo-relative path is exempt from the ASCII policy."""
    posix = rel_path.replace("\\", "/")
    if Path(posix).suffix.lower() in _EXEMPT_SUFFIXES:
        return True
    return any(posix.startswith(prefix) for prefix in _EXEMPT_PREFIXES)


def _tracked_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def scan_file(path: Path) -> list[tuple[int, int, str]]:
    """Return ``(line, col, char)`` for every character above ``0x7E`` in ``path``.

    Undecodable (binary) files report nothing -- they are caught by the suffix
    exemption or are genuinely binary; this scanner is for text sources.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    offenders: list[tuple[int, int, str]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for col, char in enumerate(line, start=1):
            if ord(char) > 0x7E and char not in "\t":
                offenders.append((line_no, col, char))
    return offenders


def find_violations(paths: list[str] | None = None) -> dict[str, list[tuple[int, int, str]]]:
    """Return the offenders per repo-relative path over the non-exempt tracked set.

    The scan covers ``paths`` when given and every tracked file otherwise; a file
    with no offender has no entry.
    """
    candidates = paths if paths is not None else _tracked_files()
    violations: dict[str, list[tuple[int, int, str]]] = {}
    for rel in candidates:
        if is_exempt(rel):
            continue
        offenders = scan_file(REPO_ROOT / rel)
        if offenders:
            violations[rel] = offenders
    return violations


def main(argv: list[str]) -> int:
    """Scan the tracked tree (or the given paths) and return the process exit status."""
    paths = argv[1:] or None
    violations = find_violations(paths)
    if not violations:
        print("ascii policy: clean")
        return 0
    total = 0
    for rel in sorted(violations):
        for line, col, char in violations[rel]:
            # !a keeps this printable on a non-UTF-8 console (e.g. Windows
            # cp1250) -- ironic to crash the ASCII checker on its own output.
            print(f"{rel}:{line}:{col}: U+{ord(char):04X} {char!a}")
            total += 1
    summary = f"ascii policy: {total} non-ASCII characters in {len(violations)} files"
    print(f"\n{summary}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
