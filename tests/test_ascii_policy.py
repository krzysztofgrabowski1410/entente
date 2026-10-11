"""Enforce the keyboard-ASCII source policy over the real tree.

Every non-exempt tracked file must be pure ASCII, so that source stays
readable and diffable in any terminal and no character depends on an editor's
encoding. Running the checker from the suite is what keeps the policy from
rotting into a document nobody applies.

The exemption list lives in ``scripts/check_ascii.py``, whose docstring
gives the reason for each category.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CHECKER = Path(__file__).resolve().parents[1] / "scripts" / "check_ascii.py"
_spec = importlib.util.spec_from_file_location("check_ascii", _CHECKER)
assert _spec is not None and _spec.loader is not None
check_ascii = importlib.util.module_from_spec(_spec)
sys.modules["check_ascii"] = check_ascii
_spec.loader.exec_module(check_ascii)


def test_tree_is_keyboard_ascii() -> None:
    violations = check_ascii.find_violations()
    if violations:
        report = "\n".join(
            f"{rel}:{line}:{col}: U+{ord(ch):04X} {ch!a}"
            for rel, offenders in sorted(violations.items())
            for line, col, ch in offenders[:5]
        )
        raise AssertionError(
            "non-ASCII characters in non-exempt source (scripts/check_ascii.py):\n" + report
        )


def test_checker_flags_a_planted_non_ascii_file(tmp_path: Path) -> None:
    # The guard must actually catch something -- a checker that can never fail
    # is worthless. Plant a non-ASCII byte via a \u escape (so this test file
    # itself stays ASCII and the sweep cannot defang the planted char) and
    # confirm scan_file reports it.
    planted = tmp_path / "planted.py"
    planted.write_text(f"x = 1  # em dash {chr(0x2014)} here\n", encoding="utf-8")
    offenders = check_ascii.scan_file(planted)
    assert offenders and offenders[0][2] == chr(0x2014)


def test_exemptions_cover_the_expected_categories() -> None:
    # Spot-check the exemption predicate so a careless edit to the list is
    # visible (the reasons are in the checker's docstring).
    assert check_ascii.is_exempt("tests/datc/cases/6a.yaml")  # conformance data
    assert check_ascii.is_exempt("uv.lock")  # tool-generated
    assert check_ascii.is_exempt("docs/media/demo.gif")  # binary artifact
    # documentation is source and is checked, code most of all
    assert not check_ascii.is_exempt("src/entente/engine/game.py")
    assert not check_ascii.is_exempt("docs/setup.md")
    assert not check_ascii.is_exempt("README.md")
    assert not check_ascii.is_exempt("docs/adjudication.tex")  # the spec source is checked
    assert check_ascii.is_exempt("docs/adjudication.pdf")  # its build artifact is not
