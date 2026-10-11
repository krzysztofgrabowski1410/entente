"""Check this engine's standard adjudication against an independent one.

The external AGPL ``diplomacy`` package is used strictly as a test-time
oracle, never imported by ``src/``; ``test_isolation.py`` checks that claim
unconditionally, and lives in its own file because a failed
``pytest.importorskip`` aborts a whole module's import and this module skips
that way whenever the ``oracle`` dependency group is not installed, which is
the default local and CI environment.

Scope: one movement phase from the shared S1901M starting position, driven by
``RandomAgent``-sampled sheets -- legal by construction (sampled from
``game.legal_orders``), so any disagreement is a genuine adjudication
difference, not a notation or legality dispute. No unit starts adjacent to an
occupied sea at S1901M (every starting fleet sits in a coastal province, not
a sea one), so ``RandomAgent`` never offers a via-convoy move here -- this
sidesteps convoy-path notation differences between the two engines by
construction, not by filtering. Order notation ("A PAR - BUR",
"A PAR S F BRE - PIC", "A PAR H") and the "S1901M"-style phase tag are the
shared textual conventions both engines independently implement from the
same domain standard (pre-dating both, used by the DATC document itself) --
observed to match at the oracle's public API surface. Nothing here was read
out of that package's source, which is what keeps this project's own
implementation original.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from entente.agents.random_agent import RandomAgent
from entente.engine.game import Game
from entente.engine.orders import format_order, parse_order

if TYPE_CHECKING:
    from collections.abc import Mapping

diplomacy = pytest.importorskip("diplomacy", reason="oracle dependency group not installed")

STANDARD_POWERS = ("AUSTRIA", "ENGLAND", "FRANCE", "GERMANY", "ITALY", "RUSSIA", "TURKEY")


def _sheet_text(game: Game, agent: RandomAgent) -> dict[str, list[str]]:
    return {
        power: [format_order(order) for order in agent.orders(game, power)] for power in game.powers
    }


def _entente_result(sheets_text: Mapping[str, list[str]]) -> dict[str, frozenset[str]]:
    game = Game.new("standard", "standard")
    for power, texts in sheets_text.items():
        if texts:
            game.submit(power, [parse_order(text) for text in texts])
    game.adjudicate()
    by_power: dict[str, set[str]] = {power: set() for power in STANDARD_POWERS}
    for unit in game.state.units:
        by_power[unit.power].add(str(unit))
    return {power: frozenset(units) for power, units in by_power.items()}


def _oracle_result(sheets_text: Mapping[str, list[str]]) -> dict[str, frozenset[str]]:
    game = diplomacy.Game(map_name="standard")
    for power, texts in sheets_text.items():
        if texts:
            game.set_orders(power, texts)
    game.process()
    state = game.get_state()
    units: Mapping[str, list[str]] = state["units"]
    return {power: frozenset(units.get(power, [])) for power in STANDARD_POWERS}


@pytest.mark.oracle
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_movement_matches_external_oracle(seed: int) -> None:
    game = Game.new("standard", "standard")
    sheets_text = _sheet_text(game, RandomAgent(seed))

    entente_result = _entente_result(sheets_text)
    oracle_result = _oracle_result(sheets_text)

    assert entente_result == oracle_result, (
        f"seed={seed}\nsheets={sheets_text}\nentente={entente_result}\noracle={oracle_result}"
    )
