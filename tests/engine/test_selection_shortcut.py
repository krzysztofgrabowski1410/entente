"""The over-the-bound shortcut of the shared selection agrees with the enumeration.

Above ``selection.MAX_ENUMERATED`` undecided orders the selection does not
enumerate: with no convoy tracked it commits the all-succeed completion when
that is stable (docs/adjudication.tex Sec. 6; the argument is the docstring of
``selection._all_succeed_is_stable``). Setting the bound to 0 sends every
residue through that shortcut, so wherever it commits at all the verdicts must
be the ones of the exact enumeration. Where it does not, the selection raises
(a residue the shortcut cannot settle is not compared). The corpus is every
DATC and BATC case under both engines, and the positions of the differential
sampler (``tests/properties``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings

from entente.engine.adjudicator import ENGINES, EngineName, PhaseResult, adjudicate
from entente.engine.adjudicator.common import AdjudicationError, selection
from entente.engine.orders import Order
from entente.engine.state import State
from entente.variants.beleriand import BELERIAND
from tests.datc.loader import ConformanceCase, load_all
from tests.datc.runner import run_case
from tests.properties.strategies import BELERIAND_GRAPH, beleriand_position_and_sheets

TESTS = Path(__file__).parent.parent
CASES = [
    case
    for case in (*load_all(TESTS / "datc" / "cases"), *load_all(TESTS / "batc" / "cases"))
    if case.centers is None  # a movement phase, not a build or disband adjustment
]


def observables(result: PhaseResult) -> tuple[object, ...]:
    """Return the four things an adjudication reports, as one comparable tuple."""
    return (result.resolutions, result.dislodgements, result.new_state, result.embattled)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_the_shortcut_agrees_with_the_enumeration_on_the_case_corpus(
    engine: EngineName, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = {}
    for case in CASES:
        exact[case.id] = observables(run_case(case, engine))
    monkeypatch.setattr(selection, "MAX_ENUMERATED", 0)
    answered = 0
    for case in CASES:
        try:
            shortcut = observables(run_case(case, engine))
        except AdjudicationError:
            continue
        answered += 1
        assert shortcut == exact[case.id], case.id
    assert answered >= 50  # the comparison is not vacuous
    print(f"{engine}: {answered} of {len(CASES)} cases answered at bound 0")


def adjudicate_both_ways(
    state: State, sheets: dict[str, list[Order]], engine: EngineName
) -> tuple[tuple[object, ...], tuple[object, ...] | None]:
    """Adjudicate under the exact bound and under 0; the second is None where it raises."""
    exact = observables(adjudicate(BELERIAND, state, sheets, graph=BELERIAND_GRAPH, engine=engine))
    saved = selection.MAX_ENUMERATED
    selection.MAX_ENUMERATED = 0
    try:
        shortcut = observables(
            adjudicate(BELERIAND, state, sheets, graph=BELERIAND_GRAPH, engine=engine)
        )
    except AdjudicationError:
        return exact, None
    finally:
        selection.MAX_ENUMERATED = saved
    return exact, shortcut


@pytest.mark.parametrize("engine", sorted(ENGINES))
@settings(
    max_examples=300,
    deadline=None,
    derandomize=True,
    suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
)
@given(beleriand_position_and_sheets())
def test_the_shortcut_agrees_with_the_enumeration_on_sampled_positions(
    engine: EngineName, pos: tuple[State, dict[str, list[Order]]]
) -> None:
    exact, shortcut = adjudicate_both_ways(*pos, engine)
    if shortcut is not None:
        assert shortcut == exact


def test_the_case_corpus_is_loaded() -> None:
    assert all(isinstance(case, ConformanceCase) for case in CASES)
    assert len(CASES) > 300  # DATC and BATC movement cases
