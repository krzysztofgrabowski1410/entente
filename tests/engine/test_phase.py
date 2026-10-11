"""Phase-machine unit tests: tags, the cycle, and the decision/effect cases."""

from __future__ import annotations

import pytest

from entente.engine.adjudicator import adjudicate
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import parse_order
from entente.engine.phase import (
    Phase,
    PhaseKind,
    Season,
    adjustment_has_decisions,
    adjustment_has_effects,
    retreat_has_decisions,
    retreat_has_effects,
)
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


class TestPhaseTags:
    """Parsing, formatting and succession of the season tags."""

    def test_tag_round_trips_the_full_cycle(self) -> None:
        phase = Phase(1901, Season.SPRING, PhaseKind.MOVEMENT)
        tags = []
        for _ in range(10):
            tags.append(phase.tag)
            assert Phase.parse(phase.tag) == phase
            assert str(phase) == phase.tag
            phase = phase.successor()
        assert tags == [
            "S1901M",
            "S1901R",
            "F1901M",
            "F1901R",
            "W1901A",
            "S1902M",
            "S1902R",
            "F1902M",
            "F1902R",
            "W1902A",
        ]

    def test_invalid_combinations_raise(self) -> None:
        with pytest.raises(ValueError, match="no such phase"):
            Phase(1901, Season.WINTER, PhaseKind.MOVEMENT)
        with pytest.raises(ValueError, match="no such phase"):
            Phase(1901, Season.SPRING, PhaseKind.ADJUSTMENT)

    def test_malformed_tags_raise(self) -> None:
        for tag in ("", "S1", "X1901M", "S1901X", "SM"):
            with pytest.raises(ValueError):
                Phase.parse(tag)


def movement_result(units: dict[str, list[str]], orders: dict[str, list[str]]):
    """Adjudicate one standard movement phase from written units and orders."""
    state = State.of(
        "standard",
        "S1901M",
        [
            Unit(power, UnitKind(spec.split()[0]), _loc(spec.split()[1]))
            for power, specs in units.items()
            for spec in specs
        ],
    )
    sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
    return adjudicate(STANDARD, state, sheets, graph=GRAPH)


def _loc(text: str) -> Location:
    province, _, coast = text.partition("/")
    return Location(province, coast or None)


class TestRetreatDecisionsAndEffects:
    """Which movement outcomes open a retreat phase."""

    def test_no_dislodgements_means_no_effects(self) -> None:
        result = movement_result({"FRANCE": ["A PAR"]}, {"FRANCE": ["A PAR - BUR"]})
        assert not retreat_has_effects(result)
        assert not retreat_has_decisions(STANDARD, GRAPH, result)

    def test_trapped_dislodgement_has_effects_but_no_decisions(self) -> None:
        # The 6.H.16 shape: the Portuguese fleet has nowhere to go.
        result = movement_result(
            {"ENGLAND": ["F POR"], "FRANCE": ["F SPA/SC", "F MAO"]},
            {"ENGLAND": ["F POR H"], "FRANCE": ["F SPA/SC - POR", "F MAO S F SPA/SC - POR"]},
        )
        assert retreat_has_effects(result)
        assert not retreat_has_decisions(STANDARD, GRAPH, result)

    def test_open_retreat_is_a_decision(self) -> None:
        result = movement_result(
            {"GERMANY": ["A BUR"], "FRANCE": ["A PAR", "A MAR"]},
            {"FRANCE": ["A PAR - BUR", "A MAR S A PAR - BUR"], "GERMANY": ["A BUR H"]},
        )
        assert retreat_has_effects(result)
        assert retreat_has_decisions(STANDARD, GRAPH, result)


class TestAdjustmentDecisionsAndEffects:
    """Which ownership positions open an adjustment phase."""

    def test_balanced_powers_have_neither(self) -> None:
        state = State.of("standard", "W1901A", [Unit("FRANCE", UnitKind.ARMY, Location("PAR"))])
        ownership = {"PAR": "FRANCE"}
        assert not adjustment_has_decisions(GRAPH, state, ownership)
        assert not adjustment_has_effects(GRAPH, state, ownership)

    def test_build_entitlement_with_a_site_is_a_decision(self) -> None:
        state = State.of("standard", "W1901A", [])
        ownership = {"PAR": "FRANCE"}
        assert adjustment_has_decisions(GRAPH, state, ownership)
        assert not adjustment_has_effects(GRAPH, state, ownership)  # unexercised builds

    def test_entitlement_without_a_site_is_no_decision(self) -> None:
        # France owns two centers but no *vacant owned home* center: quota is
        # positive yet nothing can be built -- implicit waive, no decision.
        state = State.of("standard", "W1901A", [Unit("FRANCE", UnitKind.ARMY, Location("PAR"))])
        ownership = {"PAR": "FRANCE", "SPA": "FRANCE", "POR": "FRANCE"}
        assert not adjustment_has_decisions(GRAPH, state, ownership)

    def test_forced_full_removal_is_effects_only(self) -> None:
        state = State.of("standard", "W1901A", [Unit("FRANCE", UnitKind.ARMY, Location("BUR"))])
        assert not adjustment_has_decisions(GRAPH, state, {})
        assert adjustment_has_effects(GRAPH, state, {})

    def test_partial_removal_is_a_decision(self) -> None:
        state = State.of(
            "standard",
            "W1901A",
            [
                Unit("FRANCE", UnitKind.ARMY, Location("BUR")),
                Unit("FRANCE", UnitKind.ARMY, Location("PIC")),
            ],
        )
        ownership = {"PAR": "FRANCE"}
        assert adjustment_has_decisions(GRAPH, state, ownership)
        assert adjustment_has_effects(GRAPH, state, ownership)
