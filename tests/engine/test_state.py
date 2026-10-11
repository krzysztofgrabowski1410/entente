"""State: canonical ordering, multiset occupancy, capacity/placement checks."""

import pytest

from entente.engine.map import Location, UnitKind, load_map
from entente.engine.state import State, Unit, check_state

ARMY, FLEET = UnitKind.ARMY, UnitKind.FLEET


def test_units_sort_canonically_regardless_of_input_order() -> None:
    a = Unit("FRANCE", ARMY, Location("PAR"))
    b = Unit("GERMANY", ARMY, Location("MUN"))
    assert State.of("standard", "S1901M", [a, b]) == State.of("standard", "S1901M", [b, a])


def test_occupants_is_a_per_province_multiset() -> None:
    unit = Unit("FRANCE", ARMY, Location("PAR"))
    state = State.of("standard", "S1901M", [unit, Unit("FRANCE", FLEET, Location("BRE"))])
    assert state.occupants("PAR") == (unit,)
    assert state.occupants("MUN") == ()


def test_check_state_accepts_the_standard_start() -> None:
    graph = load_map("standard")
    units = [
        Unit(power, UnitKind(spec.split()[0]), _loc(spec.split()[1]))
        for power, specs in graph.starting_units.items()
        for spec in specs
    ]
    check_state(graph, State.of("standard", "S1901M", units))


def _loc(text: str) -> Location:
    province, _, coast = text.partition("/")
    return Location(province, coast or None)


def test_check_state_rejects_capacity_violation() -> None:
    graph = load_map("standard")
    two = [Unit("FRANCE", ARMY, Location("PAR")), Unit("GERMANY", ARMY, Location("PAR"))]
    with pytest.raises(ValueError, match="capacity"):
        check_state(graph, State.of("standard", "S1901M", two))


def test_check_state_rejects_illegal_terrain_and_missing_coast() -> None:
    graph = load_map("standard")
    with pytest.raises(ValueError, match="illegal placement"):
        check_state(graph, State.of("standard", "S1901M", [Unit("X", ARMY, Location("NTH"))]))
    with pytest.raises(ValueError, match="illegal placement"):
        check_state(graph, State.of("standard", "S1901M", [Unit("X", FLEET, Location("STP"))]))
    check_state(graph, State.of("standard", "S1901M", [Unit("X", FLEET, Location("STP", "SC"))]))
