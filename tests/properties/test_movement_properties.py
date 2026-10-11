"""Certify the adjudicator's own invariants over generated positions.

Positions and sheets come from ``tests.properties.strategies``: reachable
positions, sampled by the mostly-legal samplers, since Phase 0 is part of the
system under test and occasionally-invalid orders are wanted. The core
invariants are parameterized by ruleset, so they run over standard and
Beleriand positions alike (docs/adjudication.tex Sec. 7).

Power-permutation equivariance is asserted for the standard ruleset only.
Beleriand declares ``power_symmetric = False``, because its powers differ from
each other and its targeting reads the sheet's sequence
(docs/adjudication.tex Sec. 4.4), so for that column the suite asserts the flag
rather than the property. The engine-against-engine differential and its slow
deep profiles live in ``test_differential``.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from entente.engine.adjudicator import PhaseResult, adjudicate
from entente.engine.map import UnitKind
from entente.engine.map.graph import MapGraph
from entente.engine.orders import Dislodge, Move, Order, format_order, parse_order
from entente.engine.rules import STANDARD, Ruleset
from entente.engine.state import State, Unit, check_state
from entente.variants.beleriand import BELERIAND
from tests.properties.strategies import (
    BELERIAND_GRAPH,
    GRAPH,
    beleriand_position_and_sheets,
    position_and_sheets,
)

SUPPRESS = (HealthCheck.too_slow, HealthCheck.data_too_large)

Position = tuple[State, dict[str, list[Order]]]


def _observables(result: PhaseResult) -> tuple[object, object, object]:
    return (
        {(r.power, r.text): r.resolution for r in result.resolutions},
        result.new_state,
        result.dislodgements,
    )


def check_determinism(ruleset: Ruleset, graph: MapGraph, pos: Position) -> None:
    """Return the same ``PhaseResult`` for the same inputs, byte for byte.

    Sheet-order permutation is asserted for the standard ruleset only: with
    one order per unit the order of the sheet carries no meaning there, while
    Beleriand reads that sequence for victim selection and for claim binding
    in stacked Areas (docs/adjudication.tex Sec. 4.4).
    """
    state, sheets = pos
    first = adjudicate(ruleset, state, sheets, graph=graph)
    second = adjudicate(ruleset, state, sheets, graph=graph)
    assert first == second
    if ruleset.power_symmetric:
        permuted = {power: list(reversed(sheet)) for power, sheet in reversed(list(sheets.items()))}
        third = adjudicate(ruleset, state, permuted, graph=graph)
        assert _observables(third) == _observables(first)


def check_conservation(ruleset: Ruleset, graph: MapGraph, pos: Position) -> None:
    """Conserve every unit across a movement phase, and place them legally.

    A unit stays, moves or is dislodged, and is never duplicated or lost;
    survivors sit where they stood or where their relocation took them; and
    the new occupancy respects placement legality and capacity
    (docs/adjudication.tex Sec. 1). The census runs per ``(power, kind,
    strength)``, because a unit's base strength is part of its identity and
    must be conserved exactly.
    """
    state, sheets = pos
    result = adjudicate(ruleset, state, sheets, graph=graph)

    def census(units: tuple[Unit, ...]) -> dict[tuple[str, UnitKind, int], int]:
        counts: dict[tuple[str, UnitKind, int], int] = {}
        for unit in units:
            key = (unit.power, unit.kind, unit.strength)
            counts[key] = counts.get(key, 0) + 1
        return counts

    before = census(state.units)
    after = census(result.new_state.units)
    for dislodgement in result.dislodgements:
        key = (
            dislodgement.unit.power,
            dislodgement.unit.kind,
            dislodgement.unit.strength,
        )
        after[key] = after.get(key, 0) + 1
    assert after == before

    allowed: dict[str, set[str]] = {}
    for unit in state.units:
        allowed.setdefault(unit.power, set()).add(unit.location.province)
    for power, sheet in sheets.items():
        for order in sheet:
            if isinstance(order, Move):
                allowed.setdefault(power, set()).add(order.destination.province)
            elif isinstance(order, Dislodge):
                allowed.setdefault(power, set()).add(order.victim.location.province)
    for unit in result.new_state.units:
        assert unit.location.province in allowed[unit.power]

    check_state(graph, result.new_state)


@settings(max_examples=60, deadline=None, suppress_health_check=SUPPRESS)
@given(position_and_sheets())
def test_determinism_identical_and_permuted(pos: Position) -> None:
    check_determinism(STANDARD, GRAPH, pos)


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(beleriand_position_and_sheets())
def test_determinism_beleriand(pos: Position) -> None:
    check_determinism(BELERIAND, BELERIAND_GRAPH, pos)


@settings(max_examples=60, deadline=None, suppress_health_check=SUPPRESS)
@given(position_and_sheets())
def test_unit_conservation_and_capacity(pos: Position) -> None:
    check_conservation(STANDARD, GRAPH, pos)


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(beleriand_position_and_sheets())
def test_unit_conservation_and_capacity_beleriand(pos: Position) -> None:
    check_conservation(BELERIAND, BELERIAND_GRAPH, pos)


@settings(max_examples=60, deadline=None, suppress_health_check=SUPPRESS)
@given(position_and_sheets())
def test_order_notation_round_trip(pos: Position) -> None:
    """Sec. 1 notation: parse(format(order)) == order for every generated order."""
    _, sheets = pos
    for sheet in sheets.values():
        for order in sheet:
            assert parse_order(format_order(order)) == order


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(beleriand_position_and_sheets())
def test_order_notation_round_trip_beleriand(pos: Position) -> None:
    """Round-trip the variant grammar's notation through parse and format.

    Strength prefixes, Dislodge, Bar and subject owners all survive exactly
    (docs/adjudication.tex Sec. 1).
    """
    _, sheets = pos
    for sheet in sheets.values():
        for order in sheet:
            assert parse_order(format_order(order)) == order


def _relabeled(pos: Position, mapping: dict[str, str]) -> Position:
    state, sheets = pos
    units = tuple(Unit(mapping[u.power], u.kind, u.location, u.strength) for u in state.units)
    relabeled_state = State.of(state.map_name, state.phase, units)
    relabeled_sheets = {mapping[power]: list(sheet) for power, sheet in sheets.items()}
    return relabeled_state, relabeled_sheets


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(position_and_sheets(), st.randoms(use_true_random=False))
def test_power_permutation_equivariance_standard(pos: Position, rng: object) -> None:
    """Relabel the powers by any permutation and get the relabelled outcome.

    The standard ruleset is power-symmetric, so a permutation changes nothing
    but the names. Beleriand is not, because its powers differ from each other
    and its targeting reads the sheet's sequence; the suite asserts that flag
    below rather than this property.
    """
    assert STANDARD.power_symmetric
    state, sheets = pos
    powers = sorted({u.power for u in state.units} | set(sheets))
    shuffled = list(powers)
    rng.shuffle(shuffled)  # type: ignore[attr-defined]
    mapping = dict(zip(powers, shuffled, strict=True))

    base = adjudicate(STANDARD, state, sheets, graph=GRAPH)
    relabeled_state, relabeled_sheets = _relabeled(pos, mapping)
    relabeled = adjudicate(STANDARD, relabeled_state, relabeled_sheets, graph=GRAPH)

    expected_resolutions = {(mapping[r.power], r.text): r.resolution for r in base.resolutions}
    actual_resolutions = {(r.power, r.text): r.resolution for r in relabeled.resolutions}
    assert actual_resolutions == expected_resolutions

    expected_units = sorted(
        str(Unit(mapping[u.power], u.kind, u.location, u.strength)) + "|" + mapping[u.power]
        for u in base.new_state.units
    )
    actual_units = sorted(str(u) + "|" + u.power for u in relabeled.new_state.units)
    assert actual_units == expected_units

    expected_dislodged = sorted(
        f"{mapping[d.unit.power]} {d.unit} from {d.dislodger_origin}" for d in base.dislodgements
    )
    actual_dislodged = sorted(
        f"{d.unit.power} {d.unit} from {d.dislodger_origin}" for d in relabeled.dislodgements
    )
    assert actual_dislodged == expected_dislodged


def test_beleriand_declares_power_asymmetry() -> None:
    """Read power symmetry off the ruleset, so a variant can opt out of it.

    Beleriand declares itself asymmetric: reading the sheet's sequence makes
    permutation equivariance false in general, and the case files exhibit a
    position where it fails. The flag is what decides which invariants a
    column is held to (docs/adjudication.tex Sec. 7).
    """
    assert not BELERIAND.power_symmetric
    assert STANDARD.power_symmetric
