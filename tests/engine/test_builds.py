"""Adjustment-phase unit tests -- the corners DATC 6.I/6.J does not pin."""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.builds import (
    adjudicate_adjustments,
    build_quota,
    civil_disorder_removals,
    legal_builds,
)
from entente.engine.adjudicator.common import Resolution
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import Build, parse_adjustment_order
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def state_of(*units: Unit) -> State:
    """Place ``units`` on the standard board at an adjustment phase."""
    return State.of("standard", "W1901A", units)


def army(power: str, province: str) -> Unit:
    """Build one army of ``power`` standing in ``province``."""
    return Unit(power, UnitKind.ARMY, Location(province))


def fleet(power: str, province: str, coast: str | None = None) -> Unit:
    """Build one fleet of ``power``, optionally on a named coast."""
    return Unit(power, UnitKind.FLEET, Location(province, coast))


def run(state: State, ownership: dict[str, str], sheets: dict[str, list[str]]):
    """Adjudicate written adjustment sheets against ``ownership``."""
    parsed = {
        power: [parse_adjustment_order(text) for text in sheet] for power, sheet in sheets.items()
    }
    return adjudicate_adjustments(STANDARD, state, ownership, parsed, graph=GRAPH)


class TestBuildsAndWaives:
    """Builds and waives against the quota, the home centres and the map."""

    def test_waives_consume_quota_and_are_not_deduplicated(self) -> None:
        state = state_of()
        ownership = {"MOS": "RUSSIA", "STP": "RUSSIA"}
        result = run(state, ownership, {"RUSSIA": ["Waive", "Waive", "Build A MOS"]})
        waives = [entry for entry in result.resolutions if entry.text == "Waive"]
        assert [entry.resolution for entry in waives] == [Resolution.SUCCEEDS, Resolution.SUCCEEDS]
        # Both waives consumed the quota of two: the build comes too late.
        assert result.resolution_of("Build A MOS") is Resolution.FAILS
        assert result.built == ()

    def test_waive_without_entitlement_fails(self) -> None:
        state = state_of(army("RUSSIA", "MOS"))
        result = run(state, {"MOS": "RUSSIA"}, {"RUSSIA": ["Waive"]})
        assert result.resolution_of("Waive") is Resolution.FAILS

    def test_disband_while_entitled_to_build_fails(self) -> None:
        state = state_of(army("RUSSIA", "WAR"))
        ownership = {"MOS": "RUSSIA", "WAR": "RUSSIA"}
        result = run(state, ownership, {"RUSSIA": ["Disband A WAR"]})
        assert result.resolution_of("Disband A WAR") is Resolution.FAILS
        assert army("RUSSIA", "WAR") in result.new_state.units

    def test_build_while_removing_fails(self) -> None:
        state = state_of(army("RUSSIA", "MOS"), army("RUSSIA", "WAR"))
        ownership = {"MOS": "RUSSIA"}
        result = run(state, ownership, {"RUSSIA": ["Build A MOS"]})
        assert result.resolution_of("Build A MOS") is Resolution.FAILS
        # The unanswered removal falls to civil disorder: Warsaw (distance 1)
        # is farther than Moscow (0).
        assert result.civil_disorder == (army("RUSSIA", "WAR"),)

    def test_anything_at_quota_zero_fails(self) -> None:
        state = state_of(army("RUSSIA", "MOS"))
        ownership = {"MOS": "RUSSIA"}
        result = run(state, ownership, {"RUSSIA": ["Build A MOS", "Waive", "Disband A MOS"]})
        assert all(entry.resolution is Resolution.FAILS for entry in result.resolutions)

    def test_build_on_unknown_province_fails(self) -> None:
        state = state_of()
        result = run(state, {"MOS": "RUSSIA"}, {"RUSSIA": ["Build A XXX"]})
        assert result.resolution_of("Build A XXX") is Resolution.FAILS

    def test_army_build_drops_written_coast(self) -> None:
        state = state_of()
        ownership = {"STP": "RUSSIA", "MOS": "RUSSIA"}
        result = run(state, ownership, {"RUSSIA": ["Build A STP/NC"]})
        assert result.resolution_of("Build A STP/NC") is Resolution.SUCCEEDS
        assert army("RUSSIA", "STP") in result.new_state.units

    def test_legal_builds_enumerate_kinds_and_coasts(self) -> None:
        state = state_of()
        ownership = {"STP": "RUSSIA", "MOS": "RUSSIA", "WAR": "GERMANY"}
        options = legal_builds(GRAPH, state, ownership, "RUSSIA")
        # Moscow: army only (inland). St Petersburg: army plus one fleet per
        # coast. Warsaw is Russian home soil but German-owned; SEV unowned.
        assert options == (
            Build(UnitKind.ARMY, Location("MOS")),
            Build(UnitKind.ARMY, Location("STP")),
            Build(UnitKind.FLEET, Location("STP", "NC")),
            Build(UnitKind.FLEET, Location("STP", "SC")),
        )

    def test_legal_builds_empty_without_entitlement(self) -> None:
        state = state_of(army("RUSSIA", "MOS"))
        assert legal_builds(GRAPH, state, {"MOS": "RUSSIA"}, "RUSSIA") == ()


class TestRemovals:
    """Removals against the quota, including the civil-disorder sweep."""

    def test_disband_resolves_by_province_with_advisory_kind(self) -> None:
        state = state_of(fleet("FRANCE", "BRE"), army("FRANCE", "PAR"))
        ownership = {"PAR": "FRANCE"}
        result = run(state, ownership, {"FRANCE": ["Disband A BRE"]})
        assert result.resolution_of("Disband A BRE") is Resolution.SUCCEEDS
        assert result.disbanded == (fleet("FRANCE", "BRE"),)

    def test_power_owning_nothing_removes_everything(self) -> None:
        state = state_of(fleet("FRANCE", "BRE"), army("FRANCE", "PAR"))
        result = run(state, {}, {})
        assert set(result.civil_disorder) == {fleet("FRANCE", "BRE"), army("FRANCE", "PAR")}
        assert result.new_state.units == ()

    def test_civil_disorder_tiebreak_uses_province_name_not_key(self) -> None:
        # Both fleets are infinitely far (no owned centers). By map *name*
        # "Marseilles" < "Mid-Atlantic Ocean" although by key MAO < MAR -- the
        # 4.D.8 pin orders by the printed name.
        removals = civil_disorder_removals(
            GRAPH, {}, [fleet("FRANCE", "MAO"), fleet("FRANCE", "MAR")], 1
        )
        assert removals == (fleet("FRANCE", "MAR"),)

    def test_civil_disorder_empty_inputs(self) -> None:
        assert civil_disorder_removals(GRAPH, {}, [], 1) == ()
        assert civil_disorder_removals(GRAPH, {}, [army("FRANCE", "PAR")], 0) == ()

    def test_quota_arithmetic(self) -> None:
        state = state_of(army("FRANCE", "PAR"))
        assert build_quota(GRAPH, state, {"PAR": "FRANCE", "BRE": "FRANCE"}, "FRANCE") == 1
        assert build_quota(GRAPH, state, {"PAR": "FRANCE"}, "FRANCE") == 0
        assert build_quota(GRAPH, state, {}, "FRANCE") == -1

    def test_resolution_of_unknown_order_raises(self) -> None:
        result = run(state_of(), {"MOS": "RUSSIA"}, {"RUSSIA": ["Waive"]})
        with pytest.raises(KeyError):
            result.resolution_of("Build A MOS")
