"""Pin what a support of a Bar is: cut as any other support, never lost with its giver.

The rulebook ("Barring order"): "A Support given to a Barring Order is cut as
any other Support, but it is not lost when the Unit giving it is Dislodged in
the same Phase: in the same moment in which that Unit is Dislodged, it Supports
the Bar, so both happen." The calculus states it in one place
(docs/adjudication.tex Sec. 2): such a support delivers its effectiveness, and
whether its giver is dislodged is not read for it. So a Bar's strength is a
function of the order sheets and the convoy viabilities alone, and a Dislodge
cannot weaken the Bar that gates it by taking one of the Bar's backers.

Everything else about those supports stays as it was, and that is pinned here
beside the new law: every attacker but the unit the Bar names still cuts them,
and every other kind of support is still lost with its giver -- even one that
nothing cut. Both engines are checked wherever a verdict is asserted
(docs/adjudication.tex Sec. 3.1, Sec. 3.3, Sec. 4.2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, Resolution, adjudicate, resolver
from entente.engine.adjudicator.common import Evaluator, build_tables, phase0
from entente.engine.map import load_map
from entente.engine.orders import format_order, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit

if TYPE_CHECKING:
    from entente.engine.adjudicator import PhaseResult
    from entente.engine.orders import Order

BELERIAND = RULESETS["beleriand"]
GRAPH = load_map("beleriand")

ENGINE_IDS = sorted(ENGINES)

Units = list[tuple[str, str]]
Sheets = dict[str, list[str]]

DISLODGE = "2A DNN DIS MORGOTH A ANF"
BAR = "2A ANF BAR FINROD 2A DNN - ANF"
BACKER = "A ANF S 2A ANF BAR FINROD 2A DNN - ANF"

# The victim of the Dislodge backs the Bar that gates it: attack 2 + 1 against
# a Bar of 2 + 1.
GUARDED_UNITS: Units = [
    ("FINROD", "2A DNN"),
    ("FINROD", "A LAD"),
    ("MORGOTH", "A ANF"),
    ("MORGOTH", "2A ANF"),
]
GUARDED_SHEETS: Sheets = {
    "FINROD": [DISLODGE, "A LAD S 2A DNN DIS MORGOTH A ANF"],
    "MORGOTH": [BAR, BACKER],
}


def _state(units: Units) -> State:
    return State.of("beleriand", "S25M", [Unit(power, *parse_unit_spec(s)) for power, s in units])


def _sheets(sheets: Sheets) -> dict[str, list[Order]]:
    return {power: [parse_order(text) for text in sheet] for power, sheet in sheets.items()}


def _resolve(units: Units, sheets: Sheets, engine: EngineName) -> PhaseResult:
    return adjudicate(BELERIAND, _state(units), _sheets(sheets), graph=GRAPH, engine=engine)


def _evaluator(units: Units, sheets: Sheets) -> Evaluator:
    parsed = phase0(GRAPH, BELERIAND, _state(units), _sheets(sheets))
    tables = build_tables(GRAPH, BELERIAND, parsed.records)
    return Evaluator(GRAPH, BELERIAND, parsed.records, tables)


def _record(evaluator: Evaluator, text: str) -> int:
    return next(r.index for r in evaluator.records if format_order(r.effective) == text)


def _dislodged(result: PhaseResult) -> list[tuple[str, int, str]]:
    return sorted(
        (d.unit.power, d.unit.strength, d.unit.location.province) for d in result.dislodgements
    )


def test_a_bar_never_asks_whether_its_backer_is_dislodged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The question is answered by the very Dislodge the Bar gates, so the Bar's
    # strength may not depend on the answer -- and the surest way not to depend
    # on it is not to ask. The value is one number under both verdicts.
    evaluator = _evaluator(GUARDED_UNITS, GUARDED_SHEETS)
    dislodge, bar, backer = (_record(evaluator, text) for text in (DISLODGE, BAR, BACKER))
    assert evaluator.tables.bar_supports == {backer}
    asked: list[int] = []
    original = Evaluator.unit_dislodged

    def recording(self: Evaluator, index: int) -> bool:
        asked.append(index)
        return original(self, index)

    monkeypatch.setattr(Evaluator, "unit_dislodged", recording)
    for verdict in (False, True):
        evaluator.status[dislodge] = verdict
        assert evaluator.bar_value(bar) == 3
        assert evaluator.bar_stops(dislodge)
    assert backer not in asked
    # The backer's own fate is still a fact of the labeling: it falls if the
    # Dislodge is given SUCCEEDS, and its support delivers all the same.
    evaluator.status[dislodge] = True
    assert evaluator.unit_dislodged(backer)
    assert evaluator.delivered_eff(backer) == evaluator.eff(backer) == 1
    assert evaluator.derived_status(backer)


def test_the_resolver_decides_a_dislodge_gated_by_its_own_victim_without_the_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The backer's interval closes as soon as its cutters are decided, though
    # its own dislodgement is still open, so the Bar gate is a forced fact and
    # propagation alone fails the Dislodge: nothing is left for the selection.
    def unreachable(*_args: object) -> None:
        raise AssertionError("the residue selection was reached")

    monkeypatch.setattr(resolver, "resolve_residue", unreachable)
    result = resolver.adjudicate(
        BELERIAND, _state(GUARDED_UNITS), _sheets(GUARDED_SHEETS), graph=GRAPH
    )
    assert result.resolution_of(DISLODGE) is Resolution.FAILS
    assert result.resolution_of(BAR) is Resolution.SUCCEEDS
    assert result.resolution_of(BACKER) is Resolution.SUCCEEDS
    assert not result.dislodgements


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_a_backer_that_falls_has_still_delivered(engine: EngineName) -> None:
    # One more supporter carries the attack to 4 against the Bar's 3 and the
    # victim's 1: the Dislodge goes in and the backer falls. Its support was
    # not cut and it counted -- the Bar stood at 3, not at 2 -- so its row
    # succeeds, as a Bar's own row would if its unit fell.
    units = [*GUARDED_UNITS, ("FINROD", "A RIV")]
    sheets = {**GUARDED_SHEETS, "FINROD": [*GUARDED_SHEETS["FINROD"], "A RIV S " + DISLODGE]}
    result = _resolve(units, sheets, engine)
    assert result.resolution_of(DISLODGE) is Resolution.SUCCEEDS
    assert result.resolution_of(BAR) is Resolution.FAILS
    assert result.resolution_of(BACKER) is Resolution.SUCCEEDS
    assert _dislodged(result) == [("MORGOTH", 1, "ANF")]


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_a_fully_cut_support_of_a_bar_still_fails(engine: EngineName) -> None:
    # A second attacker on the backer is not the unit the Bar names, so it cuts
    # the support, 1 - 1 = 0. The Bar is back at 2 and the attack of 3 passes:
    # the classic counter to a supported defence is untouched.
    units = [*GUARDED_UNITS, ("FINROD", "A EIT")]
    sheets = {**GUARDED_SHEETS, "FINROD": [*GUARDED_SHEETS["FINROD"], "A EIT DIS MORGOTH A ANF"]}
    evaluator = _evaluator(units, sheets)
    assert evaluator.eff(_record(evaluator, BACKER)) == 0
    result = _resolve(units, sheets, engine)
    assert result.resolution_of(BACKER) is Resolution.FAILS
    assert result.resolution_of(BAR) is Resolution.FAILS
    assert result.resolution_of(DISLODGE) is Resolution.SUCCEEDS
    assert result.resolution_of("A EIT DIS MORGOTH A ANF") is Resolution.FAILS
    assert _dislodged(result) == [("MORGOTH", 1, "ANF")]


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_a_support_hold_from_the_same_backer_is_still_lost(engine: EngineName) -> None:
    # The same backer, the same Dislodge, the same attack of 4 -- but the
    # backer supports the barring unit to hold instead of backing its Bar. A
    # support-hold is directed against nobody, so the Dislodge cuts it, and its
    # giver falls: the row fails where the Bar's backer succeeded.
    units = [*GUARDED_UNITS, ("FINROD", "A RIV")]
    sheets = {
        "FINROD": [*GUARDED_SHEETS["FINROD"], "A RIV S " + DISLODGE],
        "MORGOTH": [BAR, "A ANF S 2A ANF"],
    }
    result = _resolve(units, sheets, engine)
    assert result.resolution_of(DISLODGE) is Resolution.SUCCEEDS
    assert result.resolution_of("A ANF S 2A ANF") is Resolution.FAILS
    assert _dislodged(result) == [("MORGOTH", 1, "ANF")]


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_an_uncut_support_of_a_move_is_still_lost_with_its_giver(engine: EngineName) -> None:
    # The sharpest contrast: the backer supports a Move against the very unit
    # that dislodges it. That unit cannot cut the support -- it is the one the
    # support is directed against -- so the support survives its cut whole, and
    # is lost all the same when its giver falls. Only a Bar's backer is spared.
    units = GUARDED_UNITS
    sheets = {
        "FINROD": GUARDED_SHEETS["FINROD"],
        "MORGOTH": ["2A ANF - DNN", "A ANF S 2A ANF - DNN"],
    }
    evaluator = _evaluator(units, sheets)
    dislodge = _record(evaluator, DISLODGE)
    support = _record(evaluator, "A ANF S 2A ANF - DNN")
    assert not evaluator.tables.bar_supports
    evaluator.status[dislodge] = True
    evaluator.status[_record(evaluator, "2A ANF - DNN")] = True
    assert evaluator.eff(support) == 1
    assert evaluator.delivered_eff(support) == 0
    result = _resolve(units, sheets, engine)
    assert result.resolution_of(DISLODGE) is Resolution.SUCCEEDS
    assert result.resolution_of("A ANF S 2A ANF - DNN") is Resolution.FAILS
    assert _dislodged(result) == [("MORGOTH", 1, "ANF")]


TWIN_BAR = "A ANF BAR FINROD 2A DNN - ANF"
TWIN_BACKER = "A ANF S A ANF BAR FINROD 2A DNN - ANF"


@pytest.mark.parametrize("engine", ENGINE_IDS)
@pytest.mark.parametrize(
    "morgoth",
    [
        [TWIN_BAR, TWIN_BACKER, TWIN_BACKER],
        [TWIN_BACKER, TWIN_BAR, TWIN_BACKER],
        [TWIN_BACKER, TWIN_BACKER, TWIN_BAR],
    ],
    ids=["bar-first", "bar-second", "bar-last"],
)
def test_the_list_order_of_a_twin_stack_does_not_decide_its_bar(
    morgoth: list[str], engine: EngineName
) -> None:
    # An unnamed Dislodge meets the twin listed first. Whether that twin bars
    # or backs, the Bar is 1 + 1 + 1: the attack of 3 is stopped, and a barred
    # attack takes no twin.
    units = [("FINROD", "2A DNN"), ("FINROD", "A LAD"), *[("MORGOTH", "A ANF")] * 3]
    sheets = {"FINROD": GUARDED_SHEETS["FINROD"], "MORGOTH": morgoth}
    result = _resolve(units, sheets, engine)
    assert result.resolution_of(DISLODGE) is Resolution.FAILS
    assert {row.resolution for row in result.resolutions if row.power == "MORGOTH"} == {
        Resolution.SUCCEEDS
    }
    assert not result.dislodgements


def test_the_table_lists_every_support_attached_to_a_bar() -> None:
    # Attached means: to the Bar that stands, after a duplicate's supports have
    # been moved over to it -- and also to a Bar that is bound to no entrant,
    # because what such a support is does not depend on what the Bar meets.
    units = [
        ("MORGOTH", "2A NEB"),
        ("MORGOTH", "A NEB"),
        ("MORGOTH", "A THA"),
        ("DORIATH", "A FOR"),
        ("DORIATH", "A EST"),
    ]
    named = "A NEB BAR DORIATH A FOR - NEB"
    stronger = "2A NEB BAR DORIATH A FOR - NEB"
    written = "A THA S MORGOTH " + named
    morgoth = [stronger, named, written]
    # Both Bars meet the Move: the weaker one is changed into a Hold and the
    # support written for it backs the Bar that stands.
    met = _evaluator(units, {"MORGOTH": morgoth, "DORIATH": ["A FOR - NEB", "A EST H"]})
    support = _record(met, written)
    assert met.tables.converted_bars == {_record(met, named)}
    assert met.tables.supports_of[_record(met, stronger)] == (support,)
    assert met.tables.bar_supports == {support}
    # Nobody comes: neither Bar is bound, none is changed, and the support
    # stays with the Bar it names -- a support of a Bar like any other.
    idle = _evaluator(units, {"MORGOTH": morgoth, "DORIATH": ["A FOR H", "A EST H"]})
    support = _record(idle, written)
    assert not idle.tables.converted_bars
    assert idle.tables.barred_move_of[_record(idle, named)] is None
    assert idle.tables.supports_of[_record(idle, named)] == (support,)
    assert idle.tables.bar_supports == {support}
