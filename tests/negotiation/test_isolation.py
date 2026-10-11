"""Pin that `negotiation/` stays deletable by construction.

Nothing outside the package may import it, so `git rm -r
src/entente/negotiation` plus dropping the `llm` group never leaves a dangling
import. `tests/oracle/test_isolation.py` makes the same kind of check in the
opposite direction.

`entente.cli` reaches `entente.negotiation.cli` through `importlib` (a
string, not a static import) precisely so it does not trip this check -- see
that module's docstring.
"""

from __future__ import annotations

import re
from pathlib import Path

_IMPORT_PATTERN = re.compile(
    r"^\s*(import entente\.negotiation\b|from entente\.negotiation\b)", re.MULTILINE
)


def test_the_guard_itself_catches_a_docstring_prefixed_import() -> None:
    """The pattern must match imports on any line, not only at byte 0.

    Without ``re.MULTILINE`` both checks below go vacuous on any module that
    opens with a docstring, which is every module in the tree -- a guard that
    passes because it inspects nothing.
    """
    sample = '"""doc."""\nfrom entente.negotiation.parse import parse_message\n'
    assert _IMPORT_PATTERN.search(sample)
    assert _IMPORT_PATTERN.search('"""doc."""\nimport entente.negotiation\n')
    assert not _IMPORT_PATTERN.search('"""doc."""\nimport entente.engine.game\n')


def test_negotiation_not_imported_outside_itself() -> None:
    src = Path(__file__).resolve().parents[2] / "src" / "entente"
    offenders = [
        path
        for path in src.rglob("*.py")
        if "negotiation" not in path.relative_to(src).parts
        and _IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"non-negotiation code imports entente.negotiation: {offenders}"


def test_cli_reaches_negotiation_only_through_importlib() -> None:
    """The CLI's one route into the package is the importlib call, and nothing else.

    The check above would catch a static import too, but only as a bare
    violation. Naming the escape hatch here means an edit that replaces it
    fails against a test that says what the hatch was for.
    """
    cli_source = (Path(__file__).resolve().parents[2] / "src" / "entente" / "cli.py").read_text(
        encoding="utf-8"
    )
    assert 'importlib.import_module("entente.negotiation' in cli_source
    assert not _IMPORT_PATTERN.search(cli_source)
