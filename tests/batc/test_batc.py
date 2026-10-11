"""BATC -- the Beleriand Adjudicator Test Cases, an original artifact.

Hand-derived from ``BeleriandInstruction.docx`` and never from another
engine's tests: the rulebook's worked example is BATC.A.1, and the sections
walk the Layer-1 axes -- strength-valued one-shot cutting, Area capacity
admission with tie-overflow, targeted Dislodge/Bar contests with
sheet-sequence victim selection, mountain halving in exact Fractions,
fortress defense, Double-Fleet convoys, and Area-capacity retreats. Every
case runs under BOTH engines through the exact executor the DATC uses
(:mod:`tests.datc.runner`): for Beleriand this differential is the only
correctness oracle in existence, and the shared code path is what "no
standard special-casing" means operationally.
"""

from pathlib import Path

import pytest

from entente.engine.adjudicator import ENGINES, EngineName
from tests.datc.loader import ConformanceCase, load_all
from tests.datc.runner import check_case

CASES_DIR = Path(__file__).parent / "cases"

CASES = load_all(CASES_DIR)


@pytest.mark.batc
def test_the_suite_is_beleriand_with_the_expected_case_counts() -> None:
    """Pin the exact per-section counts, so a hand-dropped case cannot pass silently."""
    # Exact counts, mirroring the DATC loader guard: a loose >= floor would
    # let a dropped case through, and the guard carries the batc mark so
    # `pytest -m batc` runs it.
    by_section: dict[str, int] = {}
    for case in CASES:
        prefix = case.id.rsplit(".", 1)[0]
        by_section[prefix] = by_section.get(prefix, 0) + 1
    assert by_section == {
        "BATC.A": 7,
        "BATC.B": 27,
        "BATC.C": 23,
        "BATC.D": 14,
        "BATC.E": 8,
        "BATC.F": 13,
        "BATC.G": 6,
        "BATC.H": 8,
        "BATC.I": 9,
        "BATC.J": 12,
        "BATC.K": 22,
    }
    assert all(case.ruleset == "beleriand" for case in CASES)


@pytest.mark.batc
@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("case", CASES, ids=[case.pytest_id for case in CASES])
def test_batc_case(case: ConformanceCase, engine: EngineName) -> None:
    """Adjudicate one case under one engine and check every expectation."""
    check_case(case, engine)
