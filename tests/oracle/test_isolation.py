"""Prove the AGPL ``diplomacy`` package stays a test-time oracle only.

Nothing under ``src/`` may import it, because shipping code that does would
put this project under that package's licence. The check runs
unconditionally, with no dependency on the ``oracle`` group being installed,
and lives in its own module for that reason: a failed
``pytest.importorskip`` aborts a whole module's import, so keeping it beside
the differential itself would take this check down with it.
"""

from __future__ import annotations

import re
from pathlib import Path

_IMPORT_PATTERN = re.compile(r"^\s*(import diplomacy\b|from diplomacy\b)", re.MULTILINE)


def test_oracle_package_not_imported_by_src() -> None:
    src = Path(__file__).resolve().parents[2] / "src"
    offenders = [
        path
        for path in src.rglob("*.py")
        if _IMPORT_PATTERN.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, f"src/ imports the oracle package: {offenders}"
