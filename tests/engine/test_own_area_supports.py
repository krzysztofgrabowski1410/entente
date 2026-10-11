"""Pin the calculus around supports aimed at the supporter's own Area.

The rulebook lets a unit standing in an Area back every order aimed at that
Area, the two offensive shapes included: a Move entering it, and a Dislodge
against a unit standing in it. Both subjects are ordered from outside, so
neither is an Area-mate and the supporter reaches neither target.

Widening which supports exist changes nothing about what a support *does*, and
that is what this module proves rather than argues: ``supports_of`` binds by
subject, cutters bind by victim or target, the role filter binds by power, and
none of the three reads where the supporter stands. The mountain modifier
cannot bite either, because the crossing a support is charged for runs from the
supporter's province to the target -- here the same province, which is no edge.

Every position is checked on both engines (docs/adjudication.tex Sec. 1, Sec.
3.2-3.4, Sec. 4.3).
"""

from __future__ import annotations

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, Resolution, adjudicate
from entente.engine.adjudicator.common import phase0
from entente.engine.map import load_map
from entente.engine.orders import format_order, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit

BELERIAND = RULESETS["beleriand"]
GRAPH = load_map("beleriand")

ENGINE_IDS = sorted(ENGINES)


def _state(units: list[tuple[str, str]]) -> State:
    return State.of("beleriand", "S25M", [Unit(power, *parse_unit_spec(s)) for power, s in units])


def _sheets(sheets: dict[str, list[str]]) -> dict[str, list[object]]:
    return {power: [parse_order(text) for text in sheet] for power, sheet in sheets.items()}


def _resolve(units, sheets, engine: EngineName):
    return adjudicate(BELERIAND, _state(units), _sheets(sheets), graph=GRAPH, engine=engine)


def _verdicts(units, sheets, engine: EngineName) -> dict[str, Resolution]:
    return {entry.text: entry.resolution for entry in _resolve(units, sheets, engine).resolutions}


def _validity(units, sheets) -> dict[str, bool]:
    parsed = phase0(GRAPH, BELERIAND, _state(units), _sheets(sheets))  # type: ignore[arg-type]
    return {format_order(record.written): record.valid for record in parsed.records}


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_role_filter_drops_an_inside_support_by_the_victims_own_power(engine: EngineName) -> None:
    # A unit may write a support of the Dislodge aimed at itself. Phase 0
    # takes it -- the subject is another unit's order, and the target is the
    # supporter's own Area -- and the Sec. 3.3 role filter then leaves it out
    # of the attack, so the verdict is the one it would have without it.
    units = [("MEN", "A NEB"), ("DORIATH", "A FOR")]
    sheets = {
        "DORIATH": ["A FOR DIS MEN A NEB"],
        "MEN": ["A NEB S DORIATH A FOR DIS MEN A NEB"],
    }
    assert all(_validity(units, sheets).values())  # valid, not void
    result = _resolve(units, sheets, engine)
    assert result.resolution_of("A FOR DIS MEN A NEB") is Resolution.FAILS
    assert not result.dislodgements

    # the same support from an Area-mate of the victim's own power adds as little
    own_power = [("MEN", "A NEB"), ("MEN", "A NEB"), ("DORIATH", "A FOR")]
    mate_sheets = {
        "DORIATH": ["A FOR DIS MEN A NEB"],
        "MEN": ["A NEB S DORIATH A FOR DIS MEN A NEB", "A NEB H"],
    }
    assert not _resolve(own_power, mate_sheets, engine).dislodgements

    # the control: a third power standing in the Area delivers it
    third = [("MEN", "A NEB"), ("NAUGRIM", "A NEB"), ("DORIATH", "A FOR")]
    third_sheets = {
        "DORIATH": ["A FOR DIS MEN A NEB"],
        "NAUGRIM": ["A NEB S DORIATH A FOR DIS MEN A NEB"],
        "MEN": ["A NEB H"],
    }
    dislodged = _resolve(third, third_sheets, engine).dislodgements
    assert [(d.unit.power, d.unit.location.province) for d in dislodged] == [("MEN", "NEB")]


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_an_attack_from_a_third_province_cuts_an_inside_support(engine: EngineName) -> None:
    # Sec. 3.2 reads the supporter's province, not its subject's: a Dislodge
    # naming the inside supporter, ordered from a province adjacent to the
    # Area, cuts the support numerically. Uncut the Move's prevent is 2 and
    # beats the Bar; cut it is 1 and does not.
    units = [("DORIATH", "A FOR"), ("DORIATH", "A NEB"), ("MORGOTH", "A NEB"), ("MORGOTH", "A THA")]
    entry = {
        "DORIATH": ["A FOR - NEB", "A NEB S DORIATH A FOR - NEB"],
        "MORGOTH": ["A NEB BAR DORIATH A FOR - NEB", "A THA H"],
    }
    assert _verdicts(units, entry, engine)["A FOR - NEB"] is Resolution.SUCCEEDS

    cut = {
        "DORIATH": ["A FOR - NEB", "A NEB S DORIATH A FOR - NEB"],
        "MORGOTH": ["A NEB BAR DORIATH A FOR - NEB", "A THA DIS DORIATH A NEB"],
    }
    verdicts = _verdicts(units, cut, engine)
    assert verdicts["A NEB S DORIATH A FOR - NEB"] is Resolution.FAILS
    assert verdicts["A FOR - NEB"] is Resolution.FAILS
    assert verdicts["A NEB BAR DORIATH A FOR - NEB"] is Resolution.SUCCEEDS


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_the_own_list_rule_cuts_an_inside_support_written_first(engine: EngineName) -> None:
    # Sec. 4.4: an unnamed attack on a twin stack cuts the stack's written
    # supports on their own list, first listed downward. An inside support
    # written first is cut exactly like any other -- the rule reads sheet
    # order, never the support's shape.
    units = [
        ("DORIATH", "A NEB"),
        ("DORIATH", "A NEB"),
        ("DORIATH", "A FOR"),
        ("DORIATH", "A EST"),
        ("MORGOTH", "A THA"),
        ("MORGOTH", "A NEB"),
    ]
    sheets = {
        "DORIATH": [
            "A NEB S DORIATH A FOR - NEB",
            "A NEB S DORIATH A EST - GEL",
            "A FOR - NEB",
            "A EST - GEL",
        ],
        "MORGOTH": ["A THA DIS DORIATH A NEB", "A NEB BAR DORIATH A FOR - NEB"],
    }
    verdicts = _verdicts(units, sheets, engine)
    assert verdicts["A NEB S DORIATH A FOR - NEB"] is Resolution.FAILS  # first listed, cut
    assert verdicts["A NEB S DORIATH A EST - GEL"] is Resolution.SUCCEEDS  # second, untouched
    assert verdicts["A FOR - NEB"] is Resolution.FAILS  # its prevent fell to the Bar's 1


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_a_dislodged_inside_supporter_delivers_nothing(engine: EngineName) -> None:
    # Sec. 3.3 dislodgement void: the support dies with its giver, wherever the
    # giver stood. A Double Army out of THA takes the supporter off the board
    # and the Move it was backing falls to the Bar.
    units = [
        ("DORIATH", "A FOR"),
        ("DORIATH", "A NEB"),
        ("MORGOTH", "A NEB"),
        ("MORGOTH", "2A THA"),
    ]
    sheets = {
        "DORIATH": ["A FOR - NEB", "A NEB S DORIATH A FOR - NEB"],
        "MORGOTH": ["A NEB BAR DORIATH A FOR - NEB", "2A THA DIS DORIATH A NEB"],
    }
    result = _resolve(units, sheets, engine)
    assert result.resolution_of("A NEB S DORIATH A FOR - NEB") is Resolution.FAILS
    assert result.resolution_of("A FOR - NEB") is Resolution.FAILS
    assert [(d.unit.power, d.unit.location.province) for d in result.dislodgements] == [
        ("DORIATH", "NEB")
    ]


def test_the_cut_exception_is_vacuous_for_a_dislodge_supported_from_inside() -> None:
    # A support of a Dislodge against V is protected from V alone. For an
    # inside support V stands in the supporter's own Area, and from there V
    # can reach
    # the supporter by no legal order at all: a Dislodge is never written from
    # inside its victim's Area, and a Move never targets its own province. The
    # exception therefore never fires on this shape.
    units = [("MORGOTH", "A NEB"), ("DORIATH", "A NEB"), ("DORIATH", "A FOR")]
    sheets = {
        "MORGOTH": ["A NEB DIS DORIATH A NEB"],
        "DORIATH": [
            "A FOR DIS MORGOTH A NEB",
            "A NEB S DORIATH A FOR DIS MORGOTH A NEB",
        ],
    }
    validity = _validity(units, sheets)
    assert validity["A NEB S DORIATH A FOR DIS MORGOTH A NEB"]
    assert not validity["A NEB DIS DORIATH A NEB"]  # V cannot attack its own Area-mate


@pytest.mark.parametrize("engine", ENGINE_IDS)
def test_an_inside_support_ranks_its_entrant_in_the_admission_ladder(engine: EngineName) -> None:
    # Sec. 4.3: a support aimed at the Area raises its entrant's prevent, which
    # is what the admission ladder ranks by. NEB holds six; five stayers leave
    # one slot and two rivals want it, so without a support their equal class
    # does not fit and both bounce. The inside support splits the class and the
    # backed rival takes the slot.
    units = [
        ("DORIATH", "A NEB"),
        ("MEN", "A NEB"),
        ("MEN", "A NEB"),
        ("NAUGRIM", "A NEB"),
        ("NAUGRIM", "A NEB"),
        ("DORIATH", "A FOR"),
        ("MORGOTH", "A EST"),
    ]
    stayers = {"MEN": ["A NEB H", "A NEB H"], "NAUGRIM": ["A NEB H", "A NEB H"]}

    tie = {"DORIATH": ["A NEB H", "A FOR - NEB"], "MORGOTH": ["A EST - NEB"], **stayers}
    tied = _verdicts(units, tie, engine)
    assert tied["A FOR - NEB"] is Resolution.FAILS
    assert tied["A EST - NEB"] is Resolution.FAILS

    backed = {
        "DORIATH": ["A NEB S DORIATH A FOR - NEB", "A FOR - NEB"],
        "MORGOTH": ["A EST - NEB"],
        **stayers,
    }
    verdicts = _verdicts(units, backed, engine)
    assert verdicts["A FOR - NEB"] is Resolution.SUCCEEDS
    assert verdicts["A EST - NEB"] is Resolution.FAILS


def test_no_mountain_barrier_can_charge_an_inside_support() -> None:
    # The crossing a support is charged for runs from the supporter's province
    # to the target. For these shapes the two are the same province, which is
    # no edge of the graph -- and on the shipped board no mountain edge ends in
    # an Area at all, so the modifier has nothing to bite even in principle.
    assert not any(GRAPH.is_mountain_crossing(name, name) for name in GRAPH.provinces)
    areas = {name for name in GRAPH.provinces if GRAPH.capacity(name) != 1}
    assert not [edge for edge in GRAPH.mountain_edges if edge[1] in areas]
