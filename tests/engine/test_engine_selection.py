"""The engine-selection facade: both engines callable, resolver the default."""

from __future__ import annotations

import pytest

from entente.engine.adjudicator import ENGINES, adjudicate
from entente.engine.map import load_map
from entente.engine.map.types import Location, UnitKind
from entente.engine.orders import parse_order
from entente.engine.rules import STANDARD
from entente.engine.state import State, Unit

GRAPH = load_map("standard")


def _tiny_state() -> State:
    return State.of("standard", "S1901M", [Unit("FRANCE", UnitKind.ARMY, Location("PAR"))])


def test_registry_names_both_engines() -> None:
    assert set(ENGINES) == {"resolver", "enumerator"}


def test_default_engine_is_the_resolver() -> None:
    state = _tiny_state()
    sheets = {"FRANCE": [parse_order("A PAR - BUR")]}
    default = adjudicate(STANDARD, state, sheets, graph=GRAPH)
    explicit = adjudicate(STANDARD, state, sheets, graph=GRAPH, engine="resolver")
    assert default == explicit


def test_unknown_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown adjudication engine"):
        adjudicate(STANDARD, _tiny_state(), {}, graph=GRAPH, engine="bogus")  # type: ignore[arg-type]
