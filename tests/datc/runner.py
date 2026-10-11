"""Shared executor for YAML conformance cases (DATC and BATC).

One code path runs every case through the generalized core under a selected
engine -- the ruleset (and its same-named map) comes from the case file, so
the suites differ only in their case directories. This sameness is the point:
BATC certifying the Beleriand column through the identical machinery that
DATC certifies the standard column is what makes the generalisation claim
checkable rather than asserted (docs/adjudication.tex Sec. 8.1).
"""

from __future__ import annotations

from entente.engine.adjudicator import EngineName, PhaseResult, adjudicate
from entente.engine.adjudicator.builds import AdjustmentResult, adjudicate_adjustments
from entente.engine.adjudicator.retreats import RetreatResult, adjudicate_retreats
from entente.engine.map import load_map
from entente.engine.orders import (
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
    parse_unit_spec,
)
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from tests.datc.loader import ConformanceCase


def case_state(case: ConformanceCase) -> State:
    """Build the opening state a case describes, before any order is read."""
    units = [
        Unit(power, *parse_unit_spec(spec)) for power, specs in case.units.items() for spec in specs
    ]
    return State.of(case.ruleset, case.phase, units)


def run_case(case: ConformanceCase, engine: EngineName = "resolver") -> PhaseResult:
    """Adjudicate a case's movement phase under ``engine``."""
    ruleset = RULESETS[case.ruleset]
    graph = load_map(case.ruleset)
    sheets = {power: [parse_order(text) for text in sheet] for power, sheet in case.orders.items()}
    return adjudicate(ruleset, case_state(case), sheets, graph=graph, engine=engine)


def run_retreat_stage(case: ConformanceCase, movement: PhaseResult) -> RetreatResult:
    """Adjudicate a case's retreat stage on the movement result it produced.

    The stage reads dislodger origins, came-by-convoy flags and embattled
    provinces off ``movement``, so it must be given that phase's own result
    rather than a rebuilt one.
    """
    ruleset = RULESETS[case.ruleset]
    graph = load_map(case.ruleset)
    assert case.retreats is not None
    sheets = {
        power: [parse_retreat_order(text) for text in sheet]
        for power, sheet in case.retreats.items()
    }
    return adjudicate_retreats(ruleset, movement, sheets, graph=graph)


def run_adjustment_case(case: ConformanceCase) -> AdjustmentResult:
    """Adjudicate an adjustment case against the ownership its file declares."""
    ruleset = RULESETS[case.ruleset]
    graph = load_map(case.ruleset)
    assert case.centers is not None
    ownership = {center: power for power, centers in case.centers.items() for center in centers}
    sheets = {
        power: [parse_adjustment_order(text) for text in sheet]
        for power, sheet in case.orders.items()
    }
    return adjudicate_adjustments(ruleset, case_state(case), ownership, sheets, graph=graph)


def assert_units(expected: dict[str, list[str]], state: State) -> None:
    """Assert each named power's units, comparing the two sides order-blind."""
    for power, specs in expected.items():
        placed = sorted(str(u) for u in state.units if u.power == power)
        assert placed == sorted(specs), power


def check_case(case: ConformanceCase, engine: EngineName) -> None:
    """Run one conformance case under ``engine`` and assert every expectation."""
    if case.centers is not None:
        adjustment = run_adjustment_case(case)
        actual = {
            text: adjustment.resolution_of(text).value
            for text in case.expect.get("resolutions", {})
        }
        assert actual == case.expect.get("resolutions", {})
        expected_civil = case.expect.get("civil_disorder")
        if expected_civil is not None:
            removed = {f"{u.power} {u}" for u in adjustment.civil_disorder}
            assert removed == set(expected_civil)
        expected_units = case.expect.get("units")
        if expected_units is not None:
            assert_units(expected_units, adjustment.new_state)
        return

    result = run_case(case, engine)

    actual = {text: result.resolution_of(text).value for text in case.expect["resolutions"]}
    assert actual == case.expect["resolutions"]

    expected_dislodged = case.expect.get("dislodged")
    if expected_dislodged is not None:
        actual_dislodged = {f"{d.unit.power} {d.unit}" for d in result.dislodgements}
        assert actual_dislodged == set(expected_dislodged)

    final_state = result.new_state
    if case.retreats is not None:
        retreat = run_retreat_stage(case, result)
        final_state = retreat.new_state
        expected_retreats = case.expect.get("retreat_resolutions", {})
        actual_retreats = {text: retreat.resolution_of(text).value for text in expected_retreats}
        assert actual_retreats == expected_retreats
        expected_disbanded = case.expect.get("disbanded")
        if expected_disbanded is not None:
            actual_disbanded = {f"{u.power} {u}" for u in retreat.disbanded}
            assert actual_disbanded == set(expected_disbanded)

    expected_units = case.expect.get("units")
    if expected_units is not None:
        assert_units(expected_units, final_state)
