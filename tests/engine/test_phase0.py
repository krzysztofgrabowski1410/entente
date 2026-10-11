"""Certify what Phase 0 does with duplicate and illegal orders.

The DATC 4.D.3 and 4.E.1 rows, on their 2025 readings: of several orders
written to one unit the first legal one is used and illegal ones are ignored
outright -- legality read on the board alone, so a support or a convoy the
board forbids is ignored too -- and a support or convoy claim that pass-B
matching later invalidates on the other orders leaves the unit holding, with
no later order resurrected in its place. Phase 0
is where a sheet stops being text and becomes bound orders, so these readings
are checked at that boundary rather than through their effect on a board.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

from entente.engine.adjudicator import ENGINES, EngineName, Resolution, adjudicate
from entente.engine.adjudicator.common import phase0
from entente.engine.map import load_map
from entente.engine.orders import Hold, Move, Support, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def make_state(units: Mapping[str, Sequence[str]]) -> State:
    """Place written unit specs on the standard board at a movement phase."""
    return State.of(
        "standard",
        "S1901M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def sheets_of(orders: Mapping[str, Sequence[str]]) -> dict[str, list[object]]:
    """Parse written order sheets, keeping each power's sheet in its order."""
    return {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}


def record_for(state: State, orders: Mapping[str, Sequence[str]], province: str):  # type: ignore[no-untyped-def]
    """Run Phase 0 and return the bound record for the unit in ``province``."""
    parsed = phase0(GRAPH, STANDARD, state, sheets_of(orders))  # type: ignore[arg-type]
    for record in parsed.records:
        if record.unit.location.province == province:
            return record, parsed
    raise AssertionError(f"no record at {province}")


def test_first_legal_order_wins_among_duplicates() -> None:
    # 4.D.3 (2025EE): both orders legal -> the first is used.
    state = make_state({"FRANCE": ["A PAR"]})
    record, parsed = record_for(state, {"FRANCE": ["A PAR - BUR", "A PAR - PIC"]}, province="PAR")
    assert isinstance(record.effective, Move)
    assert record.effective.destination.province == "BUR"
    assert [order for _, order in parsed.rejected] == [parse_order("A PAR - PIC")]


def test_illegal_duplicate_is_ignored_not_first() -> None:
    # 4.E.1 + 4.D.3: an illegal first order is ignored entirely; the first
    # *legal* order (here the second) is used.
    state = make_state({"FRANCE": ["A PAR"]})
    record, _ = record_for(state, {"FRANCE": ["A PAR - LON", "A PAR - PIC"]}, province="PAR")
    assert record.valid
    assert isinstance(record.effective, Move)
    assert record.effective.destination.province == "PIC"


def test_all_illegal_orders_leave_the_unit_holding() -> None:
    state = make_state({"FRANCE": ["A PAR"]})
    record, _ = record_for(state, {"FRANCE": ["A PAR - LON", "A PAR - NAF"]}, province="PAR")
    assert not record.valid
    assert isinstance(record.effective, Hold)
    # End-to-end: the recorded (first) illegal order resolves FAILS.
    result = adjudicate(
        STANDARD, state, sheets_of({"FRANCE": ["A PAR - LON", "A PAR - NAF"]}), graph=GRAPH
    )
    assert result.resolution_of("A PAR - LON") is Resolution.FAILS
    assert result.resolution_of("A PAR - NAF") is Resolution.FAILS


def test_pass_b_invalidated_support_is_not_resurrected() -> None:
    # A support claim that pass-B matching invalidates, because the subject
    # never moved, leaves the unit holding: the later legal move written to the
    # same unit is not resurrected in its place.
    state = make_state({"FRANCE": ["A PAR", "A MAR"]})
    record, _ = record_for(
        state,
        {"FRANCE": ["A PAR S A MAR - BUR", "A PAR - PIC", "A MAR H"]},
        province="PAR",
    )
    assert not record.valid
    assert isinstance(record.effective, Hold)
    assert isinstance(record.written, Support)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_support_the_board_forbids_is_ignored_for_the_next_order(engine: EngineName) -> None:
    # 4.E.1 and the note to 6.D.31: legality is decided on the board alone and
    # an illegal order is ignored, so the unit's next order is used. MUN does
    # not reach MAR, whatever the army in BUR is ordered.
    state = make_state({"GERMANY": ["A MUN"], "FRANCE": ["A BUR"]})
    orders = {"GERMANY": ["A MUN S A BUR - MAR", "A MUN - RUH"]}
    result = adjudicate(STANDARD, state, sheets_of(orders), graph=GRAPH, engine=engine)
    assert result.resolution_of("A MUN S A BUR - MAR") is Resolution.FAILS
    assert result.resolution_of("A MUN - RUH") is Resolution.SUCCEEDS


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_convoy_the_board_forbids_is_ignored_for_the_next_order(engine: EngineName) -> None:
    # The cargo is a fleet, which no convoy carries, so the convoy is illegal
    # and the fleet's next order is used.
    state = make_state({"ENGLAND": ["F NTH", "F LON"]})
    orders = {"ENGLAND": ["F NTH C F LON - BEL", "F NTH - HEL"]}
    result = adjudicate(STANDARD, state, sheets_of(orders), graph=GRAPH, engine=engine)
    assert result.resolution_of("F NTH C F LON - BEL") is Resolution.FAILS
    assert result.resolution_of("F NTH - HEL") is Resolution.SUCCEEDS


def test_orders_naming_no_own_unit_are_dropped() -> None:
    # DATC 6.A.6 behavior at the phase0 level: another power's unit cannot be
    # claimed; the order is rejected, the unit's own (absent) order defaults
    # to Hold with valid=True (unordered units are hold-supportable, 4.E.1).
    state = make_state({"FRANCE": ["A PAR"], "GERMANY": ["A MUN"]})
    record, parsed = record_for(state, {"GERMANY": ["A PAR - BUR", "A MUN H"]}, province="PAR")
    assert record.written is None
    assert record.valid
    assert isinstance(record.effective, Hold)
    assert ("GERMANY", parse_order("A PAR - BUR")) in parsed.rejected


def test_a_stacked_twin_can_support_its_area_mates_move() -> None:
    # A subject reference never denotes the supporter itself, so with fungible
    # twins in a multi-unit Area the bare "A ANF" binds the other occurrence
    # and the support is valid. Reading the reference by unit value alone
    # cannot express that, because the two values are equal.
    state = State.of(
        "beleriand",
        "S25M",
        [Unit("MORGOTH", *parse_unit_spec("A ANF")), Unit("MORGOTH", *parse_unit_spec("A ANF"))],
    )
    sheets = sheets_of({"MORGOTH": ["A ANF - LOT", "A ANF S A ANF - LOT"]})
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], state, sheets)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid
    assert isinstance(supports[0].effective, Support)


def test_a_lone_unit_still_cannot_support_itself() -> None:
    # The exclusion must not loosen classic maps: a unit is never its own
    # subject, so with no co-located unit the reference of "A PAR S A PAR -
    # BUR" resolves to nothing and the order becomes Hold.
    state = make_state({"FRANCE": ["A PAR"]})
    record, _ = record_for(state, {"FRANCE": ["A PAR S A PAR - BUR"]}, province="PAR")
    assert not record.valid
    assert isinstance(record.effective, Hold)


def test_area_mates_support_co_located_defensive_orders() -> None:
    # An Area-mate may support a co-located order: a support-hold of an
    # Area-mate, and a support of a co-located unit's Bar, are both valid even
    # though the supporter cannot move into its own province, because those two
    # defensive shapes target exactly there.
    state = State.of(
        "beleriand",
        "S25M",
        [
            Unit("MORGOTH", *parse_unit_spec("A ANF")),
            Unit("MORGOTH", *parse_unit_spec("A ANF")),
            Unit("DORIATH", *parse_unit_spec("A DNN")),
        ],
    )
    sheets = sheets_of(
        {
            "MORGOTH": ["A ANF H", "A ANF S A ANF"],
            "DORIATH": [],
        }
    )
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], state, sheets)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid  # support-hold of the Area-mate delivers

    bar_sheets = sheets_of(
        {
            "MORGOTH": [
                "A ANF BAR DORIATH A DNN - ANF",
                "A ANF S A ANF BAR DORIATH A DNN - ANF",
            ],
            "DORIATH": ["A DNN - ANF"],
        }
    )
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], state, bar_sheets)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid  # support of the co-located Bar delivers


def test_units_in_an_area_support_orders_aimed_at_that_area() -> None:
    # A unit standing in an Area backs everything aimed at it, the two
    # offensive shapes included: a Move entering the Area, and a Dislodge
    # against a unit standing in it. Neither subject is co-located with the
    # supporter -- both are ordered from outside -- and the supporter reaches
    # neither target, since a unit never reaches its own province. Unlike the
    # defensive carve-out, neither shape needs an Area-mate: the supporter
    # here stands alone.
    state = State.of(
        "beleriand",
        "S25M",
        [
            Unit("MORGOTH", *parse_unit_spec("A ANF")),
            Unit("DORIATH", *parse_unit_spec("A DNN")),
        ],
    )
    into_own = sheets_of(
        {
            "MORGOTH": ["A ANF S DORIATH A DNN - ANF"],
            "DORIATH": ["A DNN - ANF"],
        }
    )
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], state, into_own)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid  # a Move entering the supporter's own Area

    # the second shape needs the supporter standing beside the victim
    beside_victim = State.of(
        "beleriand",
        "S25M",
        [
            Unit("MORGOTH", *parse_unit_spec("A ANF")),
            Unit("DORIATH", *parse_unit_spec("A ANF")),
            Unit("DORIATH", *parse_unit_spec("A DNN")),
        ],
    )
    dislodge_in_own = sheets_of(
        {
            "DORIATH": ["A DNN DIS MORGOTH A ANF", "A ANF S DORIATH A DNN DIS MORGOTH A ANF"],
            "MORGOTH": ["A ANF H"],
        }
    )
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], beside_victim, dislodge_in_own)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid  # a Dislodge against a unit in the supporter's own Area


def test_capacity_one_provinces_keep_the_classic_reach_rule() -> None:
    # Outside an Area the classic rule survives: a lone unit cannot back an
    # attack on the square it stands on. This is DATC 6.D.34 written from the
    # supporter's side -- the trap the Area carve-out would otherwise spring on
    # the standard map, where every province holds one unit.
    state = make_state({"FRANCE": ["A PAR", "A BUR"]})
    record, _ = record_for(
        state, {"FRANCE": ["A BUR - PAR", "A PAR S A BUR - PAR"]}, province="PAR"
    )
    assert not record.valid
    assert isinstance(record.effective, Hold)


def test_fungible_reference_prefers_the_twin_that_received_the_order() -> None:
    # With three fungible twins -- a holder, a mover and a supporter -- the
    # subject reference binds the twin that actually received the subject
    # order, wherever it sits on the sheet. Binding the canonically first
    # occurrence instead would let a hold written first capture the reference
    # and void the support.
    state = State.of(
        "beleriand",
        "S25M",
        [Unit("MORGOTH", *parse_unit_spec("A ANF")) for _ in range(3)],
    )
    sheets = sheets_of({"MORGOTH": ["A ANF H", "A ANF - LOT", "A ANF S A ANF - LOT"]})
    parsed = phase0(load_map("beleriand"), RULESETS["beleriand"], state, sheets)  # type: ignore[arg-type]
    supports = [r for r in parsed.records if isinstance(r.written, Support)]
    assert len(supports) == 1
    assert supports[0].valid
    assert isinstance(supports[0].effective, Support)
