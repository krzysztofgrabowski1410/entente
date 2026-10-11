"""Check scripts/datc_badge.py, which turns a junit XML report into shields.io badge fields.

Invoked as a subprocess against small fixture reports, matching exactly how
the nightly/CI step calls it -- a black-box check of the actual CLI contract,
not its internals.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "datc_badge.py"


def _run(report_path: Path) -> dict[str, str]:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(report_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if line)


def _write_report(
    path: Path, *, tests: int, failures: int = 0, errors: int = 0, skipped: int = 0
) -> None:
    path.write_text(
        f'<testsuite tests="{tests}" failures="{failures}" '
        f'errors="{errors}" skipped="{skipped}"></testsuite>',
        encoding="utf-8",
    )


def test_all_passing_is_brightgreen(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    _write_report(report, tests=352)
    assert _run(report) == {"message": "352/352", "color": "brightgreen"}


def test_skipped_cases_are_not_counted_as_passed(tmp_path: Path) -> None:
    # A skipped case is not a pass: a skip inside `-m datc` must not inflate a green N/N badge.
    report = tmp_path / "results.xml"
    _write_report(report, tests=352, skipped=2)
    assert _run(report) == {"message": "350/352", "color": "red"}


def test_failures_are_red(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    _write_report(report, tests=352, failures=3)
    assert _run(report) == {"message": "349/352", "color": "red"}


def test_errors_count_as_not_passed(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    _write_report(report, tests=10, errors=1)
    assert _run(report) == {"message": "9/10", "color": "red"}


def test_testsuites_wrapper_sums_across_suites(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    report.write_text(
        "<testsuites>"
        '<testsuite tests="100" failures="0" errors="0"></testsuite>'
        '<testsuite tests="76" failures="0" errors="0"></testsuite>'
        "</testsuites>",
        encoding="utf-8",
    )
    assert _run(report) == {"message": "176/176", "color": "brightgreen"}


def test_missing_report_is_grey_unknown(tmp_path: Path) -> None:
    assert _run(tmp_path / "does_not_exist.xml") == {"message": "unknown", "color": "lightgrey"}


def test_malformed_report_is_grey_unknown(tmp_path: Path) -> None:
    report = tmp_path / "results.xml"
    report.write_text("not xml at all", encoding="utf-8")
    assert _run(report) == {"message": "unknown", "color": "lightgrey"}
