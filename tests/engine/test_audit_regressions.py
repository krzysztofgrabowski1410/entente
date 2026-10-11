"""Guard the three adjudicator defects the differential check cannot see.

Comparing the two engines against each other catches a disagreement, so it is
blind to anything they get identically wrong, or to a shape that makes both of
them fail:

* the enumerator's totality on a legal over-budget rotation, where the
  differential crashed rather than diverged, so both engines have to agree
  that everything moves;
* duplicate written orders collapsing into one resolution row, which they did
  identically, because they share ``assemble``;
* the resolver dropping a decided mover from a residue component while the
  enumeration's crisp evaluation still reads its open dislodgement fact.

Each is checked here directly. The prevent-nulling half of the same sweep is
visible in play, so it lives in the case files as BATC.C.12 and C.13.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator import adjudicate
from entente.engine.adjudicator.common import Resolution
from entente.engine.map.loader import load_map
from entente.engine.map.types import Location, Terrain, UnitKind
from entente.engine.orders import Move, parse_order
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit

ENGINES = ("resolver", "enumerator")


def _army_cycle(target: int) -> list[str]:
    """Find a simple cycle of at least ``target`` land provinces, deterministically.

    The search is a depth-first walk over the sorted army adjacency, so the
    same cycle comes back every run. A long enough cycle is the shape that
    overflows the enumeration bound while being perfectly legal input.
    """
    graph = load_map("standard")
    land = sorted(p for p in graph.provinces if graph.provinces[p].terrain is not Terrain.SEA)
    adjacency = {
        p: sorted(q for q in graph.army_adjacency.get(p, ()) if q in set(land)) for p in land
    }

    def dfs(path: list[str], seen: set[str]) -> list[str] | None:
        if len(path) >= target and path[0] in adjacency[path[-1]]:
            return list(path)
        if len(path) >= target + 3:
            return None
        for nxt in adjacency[path[-1]]:
            if nxt not in seen:
                path.append(nxt)
                seen.add(nxt)
                if (found := dfs(path, seen)) is not None:
                    return found
                path.pop()
                seen.remove(nxt)
        return None

    for start in land:
        if (found := dfs([start], {start})) is not None:
            return found
    raise AssertionError("standard map lost its long land cycles?")


class TestRotationTotality:
    """Both engines resolve a legal rotation larger than the enumeration bound."""

    def test_legal_rotation_beyond_the_enumeration_bound(self) -> None:
        """Rotate a legal cycle larger than the enumeration bound, on both engines.

        A 21-unit vacancy rotation is legal input, and it exceeds the budget
        the enumerating engine searches within. Totality is part of the
        contract, so neither engine may refuse it: everything moves, under the
        pure-cycle policy (docs/adjudication.tex Sec. 5.2).
        """
        cycle = _army_cycle(21)
        powers = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")
        units = [
            Unit(powers[i % len(powers)], UnitKind.ARMY, Location(p)) for i, p in enumerate(cycle)
        ]
        state = State.of("standard", "S1901M", units)
        sheets: dict[str, list[Move]] = {}
        for i, (province, unit) in enumerate(zip(cycle, units, strict=True)):
            nxt = cycle[(i + 1) % len(cycle)]
            sheets.setdefault(unit.power, []).append(
                Move(UnitKind.ARMY, Location(province), Location(nxt))
            )
        results = {}
        for engine in ENGINES:
            result = adjudicate(STANDARD, state, sheets, engine=engine)
            assert all(r.resolution is Resolution.SUCCEEDS for r in result.resolutions), engine
            results[engine] = (tuple(result.resolutions), result.new_state)
        assert results["resolver"] == results["enumerator"]


class TestDuplicateWrittenOrders:
    """Two identical written orders keep two resolution rows."""

    @pytest.mark.parametrize("engine", ENGINES)
    def test_fungible_twins_report_their_own_rows(self, engine: str) -> None:
        """Report one row per written order over two identical twins.

        Each written row binds its own unit by claim position, so the
        dislodged twin's failure does not swallow the survivor's success, or
        the other way round.
        """
        ruleset = RULESETS["beleriand"]
        units = [
            Unit("MORGOTH", UnitKind.ARMY, Location("MOS")),
            Unit("MORGOTH", UnitKind.ARMY, Location("MOS")),
            Unit("OSSIRIAND", UnitKind.ARMY, Location("SOS"), strength=2),
        ]
        state = State.of("beleriand", "S25M", units)
        sheets = {
            "MORGOTH": [parse_order("A MOS H"), parse_order("A MOS H")],
            "OSSIRIAND": [parse_order("2A SOS DIS MORGOTH A MOS")],
        }
        result = adjudicate(ruleset, state, sheets, engine=engine)
        hold_rows = [r.resolution for r in result.resolutions if r.text == "A MOS H"]
        assert len(hold_rows) == 2
        # Sec. 3.4: the dislodge binds the victim power's first-mentioned unit.
        assert hold_rows == [Resolution.FAILS, Resolution.SUCCEEDS]
        assert len(result.dislodgements) == 1


class TestKnotWithDecidedRival:
    """A decided mover stays in the residue component that reads it."""

    def test_self_protecting_knot_plus_failed_rival(self) -> None:
        """Keep a decided rival in the residue component that still reads it.

        The convoy knot stalls the fixpoint, and the extra ``F BRE - ENG``
        rival is decided as a failure by propagation while its dislodgement
        fact stays open -- a fact the residue enumeration's crisp evaluation
        re-reads. The component must therefore hold both it and what it reads,
        and both engines must agree on the whole result.
        """
        units = [
            Unit("ENGLAND", UnitKind.ARMY, Location("LON")),
            Unit("ENGLAND", UnitKind.FLEET, Location("ENG")),
            Unit("FRANCE", UnitKind.FLEET, Location("IRI")),
            Unit("FRANCE", UnitKind.FLEET, Location("WAL")),
            Unit("GERMANY", UnitKind.FLEET, Location("BRE")),
        ]
        state = State.of("standard", "S1901M", units)
        sheets = {
            "ENGLAND": [
                parse_order("A LON - WAL via convoy"),
                parse_order("F ENG C A LON - WAL"),
            ],
            "FRANCE": [
                parse_order("F IRI - ENG"),
                parse_order("F WAL S F IRI - ENG"),
            ],
            "GERMANY": [parse_order("F BRE - ENG")],
        }
        results = {}
        for engine in ENGINES:
            result = adjudicate(STANDARD, state, sheets, engine=engine)
            results[engine] = (
                tuple(result.resolutions),
                result.new_state,
                tuple(result.dislodgements),
            )
        assert results["resolver"] == results["enumerator"]
