r"""The typeset specifications keep their section numbers, and every citation of one resolves.

``docs/adjudication.tex`` and ``docs/pikl.tex`` are cited by section number from module
docstrings, tests and configuration comments ("Sec. 4.2"), so the numbering is a contract,
pinned three ways here: the numbering each document computes from its headings
equals a frozen table; the ids the code cites literally (T1-T4, L1-L4, R1-R8, S1-S3) are
anchored exactly once; and every "Sec. N" in the tree names a section that exists.

The parser relies on two source conventions: a heading's ``\label{sec:...}`` stands on the
heading's own line, and the ids are anchored as ``\label{thm:T1}``, ``\label{lem:L1}``,
``\repairmark{R1}`` and ``\stagemark{S1}{...}``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "adjudication.tex": ROOT / "docs" / "adjudication.tex",
    "pikl.tex": ROOT / "docs" / "pikl.tex",
}

# (number, or None for an unnumbered heading; title stem). A stem is a prefix, so a heading's
# tail may be reworded while a renumbering, a reordering or a renamed lead fails.
ADJUDICATION = (
    ("1", "State, orders"),
    ("2", "The strength pipeline"),
    ("3", "Support cutting"),
    ("3.1", "Qualifying cutters"),
    ("3.2", "One-shot"),
    ("3.3", "Dislodgement voids"),
    ("3.4", "Cut-target selection"),
    ("3.5", "Convoy viability"),
    ("4", "Contests"),
    ("4.1", "Edge"),
    ("4.2", "Displacement contests"),
    ("4.3", "Entry"),
    ("4.4", "Twins"),
    ("5", "Semantics"),
    ("5.1", "Definition"),
    ("5.2", "The two circularity sources"),
    ("6", "The algorithm"),
    (None, "Phase 0"),
    (None, "Phase 1"),
    (None, "Phases 2"),
    (None, "The enumerator"),
    (None, "Repairs"),
    ("7", "Ruleset parameterization"),
    ("8", "Theorems"),
    ("8.1", "T1"),
    ("8.2", "T2"),
    ("8.3", "T3"),
    ("8.4", "T4"),
    ("9", "Retreats and builds"),
    ("10", "Implementation mapping"),
)
PIKL = (
    ("1", "Why anchored search"),
    ("2", "Setting"),
    ("3", "The anchored-hedge update"),
    ("4", "Limits and guarantees"),
    ("5", "Candidate sampling"),
    ("6", "1-ply utility"),
    ("7", "Hyperparameters"),
    ("8", "Validation"),
    ("9", "Module map"),
)
ANCHORS = {
    r"\label{thm:%s}": ["T1", "T2", "T3", "T4"],
    r"\label{lem:%s}": ["L1", "L2", "L3", "L4"],
    r"\repairmark{%s}": [f"R{i}" for i in range(1, 9)],
    r"\stagemark{%s}": ["S1", "S2", "S3"],
}

_HEADING = re.compile(r"^\\(section|subsection)(\*?)\{")
_LABEL = re.compile(r"\\label\{sec:([^}]*)\}")
_TEXORPDF = re.compile(r"\\texorpdfstring\{((?:[^{}]|\{[^{}]*\})*)\}\{(?:[^{}]|\{[^{}]*\})*\}")
_MARKUP = re.compile(r"\\[A-Za-z]+\*?|[{}]|\\&")


def _source(path: Path) -> list[str]:
    """Read the document's lines with comments removed; a spec source is keyboard ASCII."""
    text = path.read_text(encoding="ascii")
    return [re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines()]


def _braced(line: str, start: int) -> str:
    """Return the text inside the brace group that opens at ``line[start] == '{'``."""
    depth = 0
    for i in range(start, len(line)):
        if line[i] == "{":
            depth += 1
        elif line[i] == "}":
            depth -= 1
            if depth == 0:
                return line[start + 1 : i]
    raise ValueError(f"unbalanced heading: {line}")


def headings(lines: list[str]) -> list[tuple[str | None, str, str | None]]:
    """Every numbered heading and every starred subsection: (number, title, label)."""
    out: list[tuple[str | None, str, str | None]] = []
    section = subsection = 0
    for line in lines:
        head = _HEADING.match(line)
        if head is None:
            continue
        kind, star = head.group(1), head.group(2)
        if kind == "section" and star:
            continue  # front and back matter carry no number and no contract
        raw_title = _braced(line, head.end() - 1)
        title = _TEXORPDF.sub(r"\1", raw_title)
        title = " ".join(_MARKUP.sub("", title).split())
        if star:
            number = None
        elif kind == "section":
            section, subsection = section + 1, 0
            number = str(section)
        else:
            subsection += 1
            number = f"{section}.{subsection}"
        label = _LABEL.search(line)
        out.append((number, title, label.group(1) if label else None))
    return out


@pytest.mark.parametrize(
    ("name", "frozen"), [("adjudication.tex", ADJUDICATION), ("pikl.tex", PIKL)]
)
def test_numbering_is_frozen(name: str, frozen: tuple[tuple[str | None, str], ...]) -> None:
    got = headings(_source(SPECS[name]))
    assert len(got) == len(frozen), [(n, t) for n, t, _ in got]
    for (number, title, label), (want_number, stem) in zip(got, frozen, strict=True):
        assert number == want_number, (name, number, title)
        assert title.startswith(stem), (name, number, title, stem)
        assert label is not None, (name, number, title)
        if number is not None:
            assert label == number, (name, number, label)


def test_cited_ids_are_anchored_once() -> None:
    text = "\n".join(_source(SPECS["adjudication.tex"]))
    for pattern, ids in ANCHORS.items():
        for label in ids:
            needle = pattern % label
            assert text.count(needle) == 1, (needle, text.count(needle))


# --- every "Sec. N" in the tree names a real section ----------------------------------------

SWEEP = ("src", "tests", "configs", "scripts", "docs", "README.md")
SUFFIXES = {".py", ".md", ".tex", ".yaml", ".yml", ".toml"}
_OWNER = re.compile(r"[\w./\\-]+\.(?:md|tex|typ|docx)|\b(?:their|paper|report|rulebook)\b")
_CITE = re.compile(r"\bSec\.[ ~]?(\d+(?:\.\d+)*)")


def _swept_files() -> list[Path]:
    files: list[Path] = []
    for top in SWEEP:
        path = ROOT / top
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        for f in candidates:
            if f.suffix not in SUFFIXES or f.name in SPECS:
                continue
            if any(part in {".venv", "__pycache__"} for part in f.parts):
                continue
            files.append(f)
    return files


def _unresolved(rel: str, line: str, numbers: dict[str, set[str]]) -> list[str]:
    """List the citations on ``line`` that name no existing section."""
    owners = [(m.start(), m.group(0)) for m in _OWNER.finditer(line)]
    bad = []
    for cite in _CITE.finditer(line):
        owner = next((name for pos, name in reversed(owners) if pos < cite.start()), None)
        number = cite.group(1)
        if owner is None:
            # The specs number two levels deep, so a bare three-level number is either an
            # item of an enumerated list ("Sec. 3.1.4" is item 4 of Sec. 3.1), which the
            # list keeps by construction, or another document's section (a paper's); a
            # number starting with 0 is another document's as well.
            if number.startswith("0") or number.count(".") > 1:
                continue
            known = set().union(*numbers.values())
        else:
            spec = next((s for s in SPECS if owner.endswith(s)), None)
            if spec is None:
                continue  # another document's numbering
            known = numbers[spec]
        if number not in known:
            bad.append(f"{rel}: Sec. {number} (cited as {owner or 'bare'})")
    return bad


def test_every_section_citation_resolves() -> None:
    numbers = {
        name: {h[0] for h in headings(_source(path)) if h[0] is not None}
        for name, path in SPECS.items()
    }
    bad: list[str] = []
    for f in _swept_files():
        rel = f.relative_to(ROOT).as_posix()
        for lineno, line in enumerate(
            f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            bad.extend(f"{item} at line {lineno}" for item in _unresolved(rel, line, numbers))
    assert not bad, "\n".join(bad)
