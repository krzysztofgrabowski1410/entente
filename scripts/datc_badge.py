"""Compute the DATC shields.io badge fields from a pytest junit-xml report.

This is the counting half of the badge job; the CI step that follows feeds the
output into schneegans/dynamic-badges-action. The job fails below 100%, so a
badge cannot go stale while claiming to be green.

Usage: ``uv run python scripts/datc_badge.py <junitxml-path> >> "$GITHUB_OUTPUT"``

Prints ``message=<passed>/<total>`` and ``color=<...>`` in GitHub Actions'
``key=value`` step-output format. On a missing or unparseable report, emits
a grey "unknown" badge instead of raising -- a badge push that quietly stays
stale beats one that takes the whole nightly job down.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _counts(report_path: Path) -> tuple[int, int]:
    """Count ``(passed, total)`` in a junit-xml report.

    Every ``<testsuite>`` is summed, since pytest may write several under a
    ``<testsuites>`` root.

    A **skipped** case is not a passed case: it is counted out of ``passed``
    (but stays in ``total``), so a skip inside ``-m datc`` shows as ``N-1/N``
    red rather than inflating a green ``N/N`` badge.
    """
    root = ET.parse(report_path).getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = sum(int(suite.get("tests", 0)) for suite in suites)
    not_passed = sum(
        int(suite.get("failures", 0)) + int(suite.get("errors", 0)) + int(suite.get("skipped", 0))
        for suite in suites
    )
    return total - not_passed, total


def badge_fields(report_path: Path) -> tuple[str, str]:
    """(message, color) for the shields.io endpoint schema."""
    try:
        passed, total = _counts(report_path)
    except (OSError, ET.ParseError):
        return "unknown", "lightgrey"
    color = "brightgreen" if total > 0 and passed == total else "red"
    return f"{passed}/{total}", color


def main(argv: list[str]) -> int:
    """Print the badge fields for the report named on the command line."""
    report_path = Path(argv[1]) if len(argv) > 1 else Path("datc-results.xml")
    message, color = badge_fields(report_path)
    print(f"message={message}")
    print(f"color={color}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
