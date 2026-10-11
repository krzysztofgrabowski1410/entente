"""Anchor both adjudication engines on eleven hand-coded correctness positions.

These are the shapes where two labelling searches could disagree: paradoxes,
corridor ties and cuts by an undisputed convoy. Every position is parametrized
over both engines, and its expected outcome is fixed by the adjudication
specification rather than by either implementation
(docs/adjudication.tex Sec. 3-Sec. 6).

Four of the anchors are stated with armies supporting into sea provinces, and
one with a support from a non-adjacent province. The classic legality rule --
a supporter must be able to move into the province it supports into -- turns
those into holds, so each is written here with the minimal legal supporter
instead, which leaves every expected outcome unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from entente.engine.adjudicator import ENGINES, EngineName, PhaseResult, Resolution, adjudicate
from entente.engine.map import load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

S, F = Resolution.SUCCEEDS, Resolution.FAILS

RunFn = Callable[[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]]], PhaseResult]


@pytest.fixture(name="run", params=sorted(ENGINES))
def run_fixture(request: pytest.FixtureRequest) -> RunFn:
    """Adjudicate a position under one engine, parametrized over both."""
    engine: EngineName = request.param

    def run(units: Mapping[str, Sequence[str]], orders: Mapping[str, Sequence[str]]) -> PhaseResult:
        state = State.of(
            "standard",
            "S1901M",
            [
                Unit(power, *parse_unit_spec(spec))
                for power, specs in units.items()
                for spec in specs
            ],
        )
        sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
        return adjudicate(STANDARD, state, sheets, graph=load_map("standard"), engine=engine)

    return run


def check(result: PhaseResult, expected: Mapping[str, Resolution]) -> None:
    """Assert the named orders resolved exactly as ``expected`` says."""
    actual = {text: result.resolution_of(text) for text in expected}
    assert actual == dict(expected)


def dislodged(result: PhaseResult) -> set[tuple[str, str, str, bool]]:
    """Summarise a result's dislodgements as comparable tuples."""
    return {
        (d.unit.power, str(d.unit), d.dislodger_origin, d.by_convoy) for d in result.dislodgements
    }


def units_in(result: PhaseResult, province: str) -> set[str]:
    """Name the units standing in ``province`` after the phase."""
    return {f"{u.power} {u}" for u in result.new_state.occupants(province)}


def test_anchor_1_cut_exception(run: RunFn) -> None:
    result = run(
        {"RUSSIA": ["F CON", "F BLA"], "TURKEY": ["F ANK"]},
        {"RUSSIA": ["F CON S F BLA - ANK", "F BLA - ANK"], "TURKEY": ["F ANK - CON"]},
    )
    check(
        result,
        {"F CON S F BLA - ANK": S, "F BLA - ANK": S, "F ANK - CON": F},
    )
    assert dislodged(result) == {("TURKEY", "F ANK", "BLA", False)}
    assert units_in(result, "ANK") == {"RUSSIA F ANK"}
    assert units_in(result, "CON") == {"RUSSIA F CON"}


def test_anchor_2_dislodgement_voids_the_protected_support(run: RunFn) -> None:
    result = run(
        {"RUSSIA": ["F CON", "F BLA"], "TURKEY": ["F ANK", "A SMY", "A ARM"]},
        {
            "RUSSIA": ["F CON S F BLA - ANK", "F BLA - ANK"],
            "TURKEY": ["F ANK - CON", "A SMY S F ANK - CON", "A ARM - ANK"],
        },
    )
    check(
        result,
        {
            "F ANK - CON": S,
            "A SMY S F ANK - CON": S,
            "F CON S F BLA - ANK": F,
            "F BLA - ANK": F,
            "A ARM - ANK": F,
        },
    )
    assert dislodged(result) == {("RUSSIA", "F CON", "ANK", False)}
    assert units_in(result, "ANK") == set()
    assert units_in(result, "CON") == {"TURKEY F CON"}


def test_anchor_3_double_head_to_head_tie(run: RunFn) -> None:
    # Adapted: the validation record's A WAR S A PRU - BER is illegal (Warsaw
    # is not adjacent to Berlin, so the support would void and the corridor
    # becomes 2v1); the Russian supporter stands in Kiel instead. 2v2 tie,
    # both moves fail, nobody dislodged -- outcomes unchanged.
    result = run(
        {"GERMANY": ["A BER", "A SIL"], "RUSSIA": ["A PRU", "A KIE"]},
        {
            "GERMANY": ["A BER - PRU", "A SIL S A BER - PRU"],
            "RUSSIA": ["A PRU - BER", "A KIE S A PRU - BER"],
        },
    )
    check(
        result,
        {
            "A BER - PRU": F,
            "A PRU - BER": F,
            "A SIL S A BER - PRU": S,
            "A KIE S A PRU - BER": S,
        },
    )
    assert dislodged(result) == set()
    assert units_in(result, "BER") == {"GERMANY A BER"}
    assert units_in(result, "PRU") == {"RUSSIA A PRU"}


def test_anchor_4_beleaguered_garrison(run: RunFn) -> None:
    result = run(
        {"AUSTRIA": ["A VIE"], "ITALY": ["A TYR", "A BOH"], "RUSSIA": ["A BUD", "A GAL"]},
        {
            "AUSTRIA": ["A VIE H"],
            "ITALY": ["A TYR - VIE", "A BOH S A TYR - VIE"],
            "RUSSIA": ["A BUD - VIE", "A GAL S A BUD - VIE"],
        },
    )
    check(result, {"A VIE H": S, "A TYR - VIE": F, "A BUD - VIE": F})
    assert dislodged(result) == set()
    assert units_in(result, "VIE") == {"AUSTRIA A VIE"}


def test_anchor_5_rotation_succeeds(run: RunFn) -> None:
    result = run(
        {"TURKEY": ["A ANK", "A CON", "A SMY"]},
        {"TURKEY": ["A ANK - CON", "A CON - SMY", "A SMY - ANK"]},
    )
    check(result, {"A ANK - CON": S, "A CON - SMY": S, "A SMY - ANK": S})
    assert dislodged(result) == set()
    assert units_in(result, "CON") == {"TURKEY A CON"}
    assert units_in(result, "SMY") == {"TURKEY A SMY"}
    assert units_in(result, "ANK") == {"TURKEY A ANK"}


def test_anchor_6_disrupted_rotation_all_fail(run: RunFn) -> None:
    result = run(
        {"TURKEY": ["A ANK", "A CON", "A SMY"], "RUSSIA": ["A BUL"]},
        {
            "TURKEY": ["A ANK - CON", "A CON - SMY", "A SMY - ANK"],
            "RUSSIA": ["A BUL - CON"],
        },
    )
    check(
        result,
        {"A ANK - CON": F, "A CON - SMY": F, "A SMY - ANK": F, "A BUL - CON": F},
    )
    assert dislodged(result) == set()
    assert units_in(result, "CON") == {"TURKEY A CON"}


def test_anchor_7_illegal_swap_bounces_both(run: RunFn) -> None:
    result = run(
        {"TURKEY": ["A ANK", "A CON"]},
        {"TURKEY": ["A ANK - CON", "A CON - ANK"]},
    )
    check(result, {"A ANK - CON": F, "A CON - ANK": F})
    assert dislodged(result) == set()
    assert units_in(result, "ANK") == {"TURKEY A ANK"}
    assert units_in(result, "CON") == {"TURKEY A CON"}


def test_anchor_8_undisputed_convoys_cut_lands(run: RunFn) -> None:
    # Adapted: the validation record's A HOL S F DEN - SKA is illegal (army
    # into sea); the supporter becomes F NWY, cut by a convoyed A YOR - NWY
    # through the undisputed North Sea. Structure and outcomes unchanged:
    # the undisputed convoy's cut lands, DEN-SKA fails 1v1, Skagerrak survives,
    # the convoyed move bounces but cuts.
    result = run(
        {"ENGLAND": ["F NTH", "A YOR"], "GERMANY": ["F DEN", "F NWY"], "RUSSIA": ["F SKA"]},
        {
            "ENGLAND": ["F NTH C A YOR - NWY", "A YOR - NWY via convoy"],
            "GERMANY": ["F DEN - SKA", "F NWY S F DEN - SKA"],
            "RUSSIA": ["F SKA H"],
        },
    )
    check(
        result,
        {
            "F NWY S F DEN - SKA": F,
            "F DEN - SKA": F,
            "F SKA H": S,
            "A YOR - NWY via convoy": F,
            "F NTH C A YOR - NWY": S,
        },
    )
    assert dislodged(result) == set()
    assert units_in(result, "SKA") == {"RUSSIA F SKA"}
    assert units_in(result, "NWY") == {"GERMANY F NWY"}


def test_anchor_9_self_protecting_convoy_paradox_szykman(run: RunFn) -> None:
    # Adapted: A BEL -> F BEL (legal supporter into the North Sea).
    result = run(
        {"ENGLAND": ["F NTH", "A LON"], "GERMANY": ["F SKA", "F BEL"]},
        {
            "ENGLAND": ["F NTH C A LON - BEL", "A LON - BEL via convoy"],
            "GERMANY": ["F SKA - NTH", "F BEL S F SKA - NTH"],
        },
    )
    check(
        result,
        {
            "A LON - BEL via convoy": F,
            "F BEL S F SKA - NTH": S,
            "F SKA - NTH": S,
            "F NTH C A LON - BEL": F,
        },
    )
    assert dislodged(result) == {("ENGLAND", "F NTH", "SKA", False)}
    assert units_in(result, "NTH") == {"GERMANY F NTH"}
    assert units_in(result, "LON") == {"ENGLAND A LON"}


def test_anchor_10_zero_labeling_paradox_szykman(run: RunFn) -> None:
    # Adapted: FRA A BEL -> FRA F BEL (legal support-hold into the North Sea).
    result = run(
        {"ENGLAND": ["F NTH", "A LON"], "FRANCE": ["F BEL"], "GERMANY": ["F SKA", "F HEL"]},
        {
            "ENGLAND": ["F NTH C A LON - BEL", "A LON - BEL via convoy"],
            "FRANCE": ["F BEL S F NTH"],
            "GERMANY": ["F SKA - NTH", "F HEL S F SKA - NTH"],
        },
    )
    check(
        result,
        {
            "A LON - BEL via convoy": F,
            "F BEL S F NTH": S,
            "F SKA - NTH": F,
            "F HEL S F SKA - NTH": S,
            "F NTH C A LON - BEL": F,  # szykmaned cargo chose no route (Sec. 5.1)
        },
    )
    assert dislodged(result) == set()
    assert units_in(result, "NTH") == {"ENGLAND F NTH"}


def test_anchor_11_convoy_protecting_another_convoy_no_paradox(run: RunFn) -> None:
    # Adapted: ENG A SPA -> ENG F SPA/NC (legal supporter into the MAO).
    result = run(
        {
            "FRANCE": ["F LYO", "A MAR", "F MAO", "A BRE"],
            "ENGLAND": ["F IRI", "F SPA/NC"],
        },
        {
            "FRANCE": [
                "F LYO C A MAR - SPA",
                "A MAR - SPA via convoy",
                "F MAO C A BRE - POR",
                "A BRE - POR via convoy",
            ],
            "ENGLAND": ["F IRI - MAO", "F SPA/NC S F IRI - MAO"],
        },
    )
    check(
        result,
        {
            "F SPA/NC S F IRI - MAO": F,
            "F IRI - MAO": F,
            "A BRE - POR via convoy": S,
            "A MAR - SPA via convoy": F,
            "F MAO C A BRE - POR": S,
            "F LYO C A MAR - SPA": S,  # route used; the carried attack bounced (Sec. 5.1)
        },
    )
    assert dislodged(result) == set()
    assert units_in(result, "MAO") == {"FRANCE F MAO"}
    assert units_in(result, "POR") == {"FRANCE A POR"}
    assert units_in(result, "MAR") == {"FRANCE A MAR"}
