"""Loader for adjudicator conformance cases (DATC / BATC) encoded as YAML.

Cases are hand-encoded from primary specification documents -- the DATC
(https://webdiplomacy.net/datc.php) for the standard ruleset and the Beleriand
rulebook for BATC -- never copied from other engines' test suites. One YAML
file per DATC section keeps case ids traceable to the document
(``pytest -k 6.A`` runs one section).

Schema per file::

    section: <human-readable section title>
    ruleset: standard | beleriand
    cases:
      - id: 6.A.1                    # spec-traceable id
        title: <short description>
        phase: S1901M
        units: {POWER: ["F NTH", ...]}
        orders: {POWER: ["F NTH - PIC", ...]}   # ordered lists -- sequence matters
        expect:
          resolutions: {"F NTH - PIC": succeeds | fails}
          units: {POWER: [...]}      # optional: final unit placement
          dislodged: [...]           # optional

Two case shapes extend this for the phases after movement (6.H, and
6.I/6.J):

- **retreat cases** add ``retreats: {POWER: [...]}`` (possibly empty -- the
  stage still runs, e.g. a unit destroyed with no options, 6.H.16). The
  movement keys keep their meaning; ``expect.retreat_resolutions`` asserts
  the retreat orders, ``expect.disbanded`` the removed units, and
  ``expect.units`` the *post-retreat* placement.
- **adjustment cases** add ``centers: {POWER: ["KIE", ...]}`` (supply-center
  ownership); ``orders`` then holds build/disband/waive notation (may be
  empty for pure civil-disorder cases, 6.J.3+). ``expect.civil_disorder``
  asserts the automatic removals, ``expect.units`` the post-winter placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CASES_DIR = Path(__file__).parent / "cases"

_VALID_RESOLUTIONS = {"succeeds", "fails"}

# Every key the runner (tests/datc/runner.py check_case) actually reads from
# ``expect``. A misspelt key such as ``dislodgd`` or ``retreat_resolution``
# would otherwise load cleanly and its assertion would simply never fire, which
# is the one failure mode a harness certifying data cannot tolerate: a case
# that passes because it asserts nothing.
_VALID_EXPECT_KEYS = frozenset(
    {"resolutions", "retreat_resolutions", "civil_disorder", "units", "dislodged", "disbanded"}
)


@dataclass(frozen=True)
class ConformanceCase:
    """One adjudication scenario with its expected outcome."""

    id: str
    title: str
    section: str
    ruleset: str
    phase: str
    units: dict[str, list[str]]
    orders: dict[str, list[str]]
    expect: dict[str, Any]
    retreats: dict[str, list[str]] | None = None
    centers: dict[str, list[str]] | None = None

    @property
    def pytest_id(self) -> str:
        """Return the node id pytest shows for this case, id first so it sorts."""
        return f"{self.id}-{self.title.lower().replace(' ', '_')}"


def _validate(case: ConformanceCase, path: Path) -> None:
    context = f"{path.name}:{case.id}"
    if case.retreats is not None and case.centers is not None:
        raise ValueError(f"{context}: a case is a retreat case or an adjustment case, not both")
    # Orders from a power without units are allowed: the engine drops orders
    # naming no own unit, and DATC 6.A.6 certifies exactly that behavior.
    # Adjustment cases may have no orders at all (pure civil disorder, 6.J.3+).
    if not case.orders and case.centers is None:
        raise ValueError(f"{context}: case has no orders")
    resolutions = dict(case.expect.get("resolutions", {}))
    resolutions.update(case.expect.get("retreat_resolutions", {}))
    if case.orders and not case.expect.get("resolutions"):
        raise ValueError(f"{context}: expect.resolutions is required")
    if not resolutions and not case.expect.get("civil_disorder"):
        raise ValueError(f"{context}: case asserts no resolutions and no civil disorder")
    bad = {order: res for order, res in resolutions.items() if res not in _VALID_RESOLUTIONS}
    if bad:
        raise ValueError(f"{context}: invalid resolution values: {bad}")
    if case.retreats is None and case.expect.get("retreat_resolutions"):
        raise ValueError(f"{context}: retreat_resolutions without a retreats stage")
    unknown_keys = sorted(set(case.expect) - _VALID_EXPECT_KEYS)
    if unknown_keys:
        raise ValueError(
            f"{context}: unknown expect keys {unknown_keys} "
            f"(known: {sorted(_VALID_EXPECT_KEYS)}) -- a typo would silently drop the assertion"
        )
    if case.retreats is not None and not (
        case.expect.get("retreat_resolutions")
        or case.expect.get("disbanded")
        or case.expect.get("units")
    ):
        raise ValueError(
            f"{context}: a retreats stage that asserts nothing about the retreat outcome "
            "(needs retreat_resolutions, disbanded, or units)"
        )


def load_file(path: Path) -> list[ConformanceCase]:
    """Load and validate every case in one section file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases = []
    for raw in data["cases"]:
        case = ConformanceCase(
            id=str(raw["id"]),
            title=raw["title"],
            section=data["section"],
            ruleset=data["ruleset"],
            phase=raw["phase"],
            units=raw["units"],
            orders=raw.get("orders", {}),
            expect=raw["expect"],
            retreats=raw.get("retreats"),
            centers=raw.get("centers"),
        )
        _validate(case, path)
        cases.append(case)
    return cases


def load_all(directory: Path = CASES_DIR) -> list[ConformanceCase]:
    """Load every section file in ``directory``, in file order, ids unique."""
    cases = [case for path in sorted(directory.glob("*.yaml")) for case in load_file(path)]
    # A duplicated id would mask a dropped sibling: the section-count guard
    # counts by prefix, and pytest de-duplicates colliding parametrize ids
    # without saying so, which leaves both checks green over a missing case.
    seen: set[str] = set()
    duplicates: set[str] = set()
    for case in cases:
        if case.id in seen:
            duplicates.add(case.id)
        seen.add(case.id)
    if duplicates:
        raise ValueError(f"duplicate conformance case ids in {directory}: {sorted(duplicates)}")
    return cases
