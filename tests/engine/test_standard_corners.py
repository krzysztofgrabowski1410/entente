"""The standard map ignores what only Beleriand reads: designators and a Disband's strength.

``Ruleset.strict_designation`` is False for the standard ruleset, so a written base
strength is advisory in every phase (a Disband included) and the special-unit designators
``(M)``/``(G)`` mean nothing: an order that carries one is adjudicated exactly as the same
order without it (docs/adjudication.tex Sec. 1, Sec. 7, Sec. 9).
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator import Resolution, adjudicate
from entente.engine.adjudicator.builds import adjudicate_adjustments
from entente.engine.adjudicator.retreats import adjudicate_retreats
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import parse_adjustment_order, parse_order, parse_retreat_order
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def _army(power: str, province: str) -> Unit:
    return Unit(power, UnitKind.ARMY, Location(province))


def _fleet(power: str, province: str) -> Unit:
    return Unit(power, UnitKind.FLEET, Location(province))


def _move(units: list[Unit], sheets: dict[str, list[str]], engine: str = "resolver"):  # type: ignore[no-untyped-def]
    state = State.of("standard", "S1901M", units)
    parsed = {p: [parse_order(t) for t in sheet] for p, sheet in sheets.items()}
    return adjudicate(STANDARD, state, parsed, graph=GRAPH, engine=engine)  # type: ignore[arg-type]


def _verdicts(result) -> list[Resolution]:  # type: ignore[no-untyped-def]
    return [entry.resolution for entry in result.resolutions]


# -- movement: a designator is ignored ---------------------------------------


@pytest.mark.parametrize("engine", ["resolver", "enumerator"])
@pytest.mark.parametrize("designator", ["(M)", "(G)"])
def test_a_designator_on_the_mover_does_not_unmatch_its_support(
    engine: str, designator: str
) -> None:
    units = [_army("FRANCE", "PAR"), _army("FRANCE", "MAR"), _army("GERMANY", "BUR")]
    plain = _move(
        units,
        {
            "FRANCE": ["A PAR - BUR", "A MAR S A PAR - BUR"],
            "GERMANY": ["A BUR H"],
        },
        engine,
    )
    marked = _move(
        units,
        {
            "FRANCE": [f"A{designator} PAR - BUR", "A MAR S A PAR - BUR"],
            "GERMANY": ["A BUR H"],
        },
        engine,
    )
    assert _verdicts(plain) == [Resolution.SUCCEEDS, Resolution.SUCCEEDS, Resolution.FAILS]
    assert _verdicts(marked) == _verdicts(plain)
    assert sorted(str(u) for u in marked.new_state.units) == sorted(
        str(u) for u in plain.new_state.units
    )
    assert len(marked.dislodgements) == len(plain.dislodgements) == 1


@pytest.mark.parametrize("engine", ["resolver", "enumerator"])
def test_a_designator_on_the_supports_subject_is_ignored(engine: str) -> None:
    units = [_army("FRANCE", "PAR"), _army("FRANCE", "MAR"), _army("GERMANY", "BUR")]
    result = _move(
        units,
        {
            "FRANCE": ["A PAR - BUR", "A MAR S A(M) PAR - BUR"],
            "GERMANY": ["A BUR H"],
        },
        engine,
    )
    assert _verdicts(result) == [Resolution.SUCCEEDS, Resolution.SUCCEEDS, Resolution.FAILS]
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", ["resolver", "enumerator"])
def test_designators_on_both_sides_of_a_support_are_ignored(engine: str) -> None:
    units = [_army("FRANCE", "PAR"), _army("FRANCE", "MAR"), _army("GERMANY", "BUR")]
    result = _move(
        units,
        {
            "FRANCE": ["A(G) PAR - BUR", "A(M) MAR S A(M) PAR - BUR"],
            "GERMANY": ["A(G) BUR H"],
        },
        engine,
    )
    assert _verdicts(result) == [Resolution.SUCCEEDS, Resolution.SUCCEEDS, Resolution.FAILS]


@pytest.mark.parametrize("engine", ["resolver", "enumerator"])
def test_a_designated_fleet_supports_and_is_supported_alike(engine: str) -> None:
    units = [_fleet("ENGLAND", "NTH"), _fleet("ENGLAND", "NWG"), _fleet("RUSSIA", "NWY")]
    sheets = {
        "ENGLAND": ["F(M) NTH - NWY", "F NWG S F NTH - NWY"],
        "RUSSIA": ["F NWY H"],
    }
    plain = _move(
        units,
        {"ENGLAND": ["F NTH - NWY", "F NWG S F NTH - NWY"], "RUSSIA": ["F NWY H"]},
        engine,
    )
    marked = _move(units, sheets, engine)
    assert _verdicts(marked) == _verdicts(plain)


# -- retreats ------------------------------------------------------------------


@pytest.mark.parametrize("designator", ["", "(M)", "(G)"])
def test_a_retreat_with_or_without_a_designator_binds_the_same(designator: str) -> None:
    units = [_army("FRANCE", "PAR"), _army("FRANCE", "MAR"), _army("GERMANY", "BUR")]
    movement = _move(
        units,
        {"FRANCE": ["A PAR - BUR", "A MAR S A PAR - BUR"], "GERMANY": ["A(M) BUR H"]},
    )
    assert len(movement.dislodgements) == 1
    order = parse_retreat_order(f"A{designator} BUR - MUN")
    result = adjudicate_retreats(STANDARD, movement, {"GERMANY": [order]}, graph=GRAPH)
    assert [e.resolution for e in result.resolutions] == [Resolution.SUCCEEDS]
    assert sorted(str(u) for u in result.new_state.units) == ["A BUR", "A MAR", "A MUN"] or any(
        u.location.province == "MUN" for u in result.new_state.units
    )


# -- adjustments: a Disband's written strength is advisory ----------------------


def _adjust(orders: list[str]):  # type: ignore[no-untyped-def]
    state = State.of(
        "standard",
        "W1901A",
        [_army("FRANCE", "PAR"), _army("FRANCE", "MAR"), _fleet("FRANCE", "MAO")],
    )
    ownership = {"PAR": "FRANCE", "MAR": "FRANCE"}
    sheets = {"FRANCE": [parse_adjustment_order(t) for t in orders]}
    return adjudicate_adjustments(STANDARD, state, ownership, sheets, graph=GRAPH)


@pytest.mark.parametrize("text", ["Disband A PAR", "Disband 2A PAR", "Disband 3A PAR"])
def test_a_disband_ignores_the_written_strength(text: str) -> None:
    result = _adjust([text])
    assert result.resolution_of(text) is Resolution.SUCCEEDS
    assert [str(u) for u in result.disbanded] == ["A PAR"]
    assert result.civil_disorder == ()


def test_a_disband_of_the_wrong_strength_and_kind_still_removes_the_unit_there() -> None:
    result = _adjust(["Disband 2F MAR"])
    assert [str(u) for u in result.disbanded] == ["A MAR"]
    assert result.civil_disorder == ()
