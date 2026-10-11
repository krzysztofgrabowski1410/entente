"""DATC conformance suite (data-driven) -- live against BOTH engines.

Every case runs through the generalized core: ``adjudicate`` with the standard
ruleset supplied as pure data, so no code path is specific to the standard
board. Each runs under both the staged resolver and the stable-labelling
enumerator, which makes the suite the empirical certificate that the general
core reproduces the classic rules, and, case by case, the sharpest form of the
resolver-against-enumerator differential. Cases are hand-encoded from the DATC
v3.3 document, and where it allows more than one outcome the case states the
choice this engine makes.

Retreat cases (6.H) adjudicate their movement phase under the selected
engine, then the retreat stage on its PhaseResult -- dislodger origins,
came-by-convoy flags and embattled provinces all cross that seam, so the
suite also certifies the movement->retreat contract under both engines.
Adjustment cases (6.I/6.J, 6.B.14) have a single implementation; they run
once per engine anyway, which keeps the suite's shape uniform.

The executor lives in :mod:`tests.datc.runner`, shared verbatim with the
BATC suite (``tests/batc/``) -- one code path certifies both rulesets.
"""

import pytest

from entente.engine.adjudicator import ENGINES, EngineName
from tests.datc.loader import ConformanceCase, load_all
from tests.datc.runner import check_case

CASES = load_all()


@pytest.mark.datc
@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("case", CASES, ids=[case.pytest_id for case in CASES])
def test_datc_case(case: ConformanceCase, engine: EngineName) -> None:
    check_case(case, engine)
