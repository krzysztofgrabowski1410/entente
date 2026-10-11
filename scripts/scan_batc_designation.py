"""Scan the BATC suites for orders that rely on advisory unit-designation binding.

The rulebook treats a written base strength as advisory, so an order whose prefix names
no unit still binds to a unit of that power in the province. This audit reports
where that latitude is actually used, which is what a strict-designation setting
would change. For every case in both BATC suites (L1 ``tests/batc/cases`` and L2
``tests/batc/cases_l2``) and every written order, it replays the
Phase-0 province-only claim binding (``common._Phase0State.claim_slot``, the
byte-identical selection both knob settings share) and compares the order's
written kind/strength prefix against the unit it binds to. It flags two things:

- **fallback** -- the prefix did not exactly match the bound occupant, so the
  binding used the advisory province-only reading (the order "relies on advisory
  binding" in the audit's sense);
- **flip** -- ``designation_contradicts`` is true, i.e. the prefix matches *no*
  unit of that power in the province, so strict binding (4.C.2 choice C)
  invalidates the order (4.E.1 Hold) and the case's outcome would change.

A flip is the only thing that can break a seeded BATC expectation under strict;
a non-flipping fallback (a duplicate whose exact twin was already claimed)
keeps its advisory binding under strict too. The audit's belief is zero of
either; run this and read the summary. Exit code is non-zero iff any flip is
found, so it doubles as a CI-able guard.

Run: ``uv run python scripts/scan_batc_designation.py``
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from entente.engine.adjudicator.common import _Phase0State, _spec_matches_unit
from entente.engine.map import load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit

SUITES = {
    "L1": Path(__file__).resolve().parent.parent / "tests" / "batc" / "cases",
    "L2": Path(__file__).resolve().parent.parent / "tests" / "batc" / "cases_l2",
}


@dataclass(frozen=True)
class _Case:
    """Hold the subset of a YAML conformance case this scan reads.

    The test-only loader is not imported: its schema validation is
    pytest's job, and this scan should keep working on a case file the loader
    would reject.

    Two schemas feed this. The flat L1 movement case (``units``/``orders`` at
    the top) is read verbatim. The L2 game-scenario case (``start.units`` plus
    a ``script`` of phases) is read at its **seeded** movement phase only: the
    start units against the orders of the script entry standing on that same
    phase -- exactly "the seeded unit the order binds to". Later script phases
    act on positions the engine has since evolved, so their binding is verified
    dynamically by the green suite, not statically here (noted in the summary).
    """

    id: str
    ruleset: str
    phase: str
    units: dict[str, list[str]]
    orders: dict[str, list[str]]
    deferred_phases: int = 0  # L2 script phases past the seeded one (suite-verified)


def load_all(directory: Path) -> list[_Case]:
    """Read every case file in a suite directory, under either schema."""
    cases: list[_Case] = []
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        ruleset = data.get("ruleset", "beleriand")  # L2 files omit it -- always beleriand
        for raw in data["cases"]:
            if "start" in raw or "script" in raw:  # L2 game-scenario schema
                start = raw.get("start", {})
                phase = start.get("phase", "S25M")
                units = start.get("units", {})
                script = raw.get("script", [])
                seeded = next((s for s in script if s.get("phase") == phase), None)
                orders = seeded.get("orders", {}) if seeded else {}
                deferred = sum(1 for s in script if s is not seeded)
                cases.append(_Case(str(raw["id"]), ruleset, phase, units, orders, deferred))
            else:  # flat L1 movement/retreat/adjustment schema
                cases.append(
                    _Case(
                        str(raw["id"]),
                        ruleset,
                        raw["phase"],
                        raw.get("units", {}),
                        raw.get("orders", {}),
                    )
                )
    return cases


def _spec(kind: object, strength: int | None) -> str:
    prefix = "" if strength in (None, 1) else str(strength)
    return f"{prefix}{kind.value if hasattr(kind, 'value') else kind}"


@dataclass
class _Tally:
    cases: int = 0
    scanned_orders: int = 0
    skipped_orders: int = 0  # retreat/build/Layer-2 lines the movement binding never sees
    deferred_phases: int = 0  # L2 script phases past the seeded one (suite-verified)


def scan_case(case: _Case, tally: _Tally) -> list[str]:
    """Return one report line per fallback/flip *movement* order in ``case``.

    Only movement-phase cases exercise the ``claim_slot`` designation binding
    the knob governs; retreat (``_resolve_dislodged_unit``) and adjustment
    binding are out of scope (and out of phase). Layer-2 order lines (MSK, TFM,
    GIVES MASK, ...) are extracted by ``preprocess_movement`` before Phase 0, so
    they never reach ``claim_slot``: the movement parser rejects them and they
    are counted as skipped.
    """
    tally.deferred_phases += case.deferred_phases
    if not case.phase.endswith("M"):
        tally.skipped_orders += sum(len(s) for s in case.orders.values())
        return []
    ruleset = RULESETS[case.ruleset]
    graph = load_map(case.ruleset)
    units = [
        Unit(power, *parse_unit_spec(spec)) for power, specs in case.units.items() for spec in specs
    ]
    state = State.of(case.ruleset, case.phase, units)
    work = _Phase0State(graph, ruleset, state)
    lines: list[str] = []
    for power in sorted(case.orders):
        for text in case.orders[power]:
            try:
                order = parse_order(text)
            except Exception:
                tally.skipped_orders += 1
                continue
            tally.scanned_orders += 1
            slot = work.claim_slot(power, order)
            contradicts = work.designation_contradicts(power, order)
            if slot is None:
                # Names no own unit in the province -- dropped, never bound;
                # not an advisory-binding reliance (there is no unit to mis-bind).
                continue
            bound = work.units[slot]
            exact = _spec_matches_unit(order, bound)
            work.claimed[slot] = (power, order, 0)  # mimic the claim loop's occupancy
            if exact and not contradicts:
                continue
            tag = "FLIP" if contradicts else "fallback"
            lines.append(
                f"    [{tag}] {case.id}: {power} order '{_spec(order.unit_kind, order.strength)} "
                f"{order.location.province}...' bound {_spec(bound.kind, bound.strength)} "
                f"{bound.location.province}"
            )
    return lines


def main() -> int:
    """Scan both suites and print every fallback and flip, with a tally."""
    tally = _Tally()
    flips: list[str] = []
    fallbacks: list[str] = []
    for label, directory in SUITES.items():
        cases = load_all(directory)
        print(f"== {label}: {len(cases)} cases from {directory.name} ==")
        for case in cases:
            tally.cases += 1
            for line in scan_case(case, tally):
                (flips if "[FLIP]" in line else fallbacks).append(line)
                print(line)
    print()
    print(
        f"scanned {tally.cases} cases: {tally.scanned_orders} seeded movement orders bound, "
        f"{tally.skipped_orders} non-movement/Layer-2 lines skipped, "
        f"{tally.deferred_phases} evolved L2 script phases deferred to the dynamic suite"
    )
    print(f"advisory-fallback bindings (prefix != bound unit, non-flipping): {len(fallbacks)}")
    print(f"strict flips (prefix matches no unit -> invalid under 4.C.2-C): {len(flips)}")
    if not flips and not fallbacks:
        print(
            "RESULT: no case relies on advisory binding; strict binding is a no-op on the suites."
        )
    elif not flips:
        print("RESULT: fallbacks exist but none flip under strict; suites are unchanged.")
    else:
        print("RESULT: FLIPS FOUND -- these cases must be re-derived under strict binding.")
    return 1 if flips else 0


if __name__ == "__main__":
    sys.exit(main())
