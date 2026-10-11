"""The two-beach map: one landing coast that touches a marked sea and a free one.

No shipped coast is mixed -- every sea touching a Beleriand landing coast is
penalized into it, or none is -- so the positions that tell the convoy toll's
readings apart cannot be set up on a shipped map. This synthetic one is the
smallest that can::

    ORI -- SEA -- SEC ==> LND        the cliff route: SEC -> LND is marked
    ORI -- SEB -- SED --- LND        the beach route: SED -> LND is free
                  SED -- SEX, SEY    two seas an attack on the SED fleet starts from

ORI and LND share no land border, so an army crosses by convoy or not at all.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from entente.engine.map import MapGraph, build_graph
from entente.engine.orders import Order, parse_order, parse_unit_spec
from entente.engine.state import State, Unit

TWO_BEACH_DATA = {
    "name": "two-beach",
    "provinces": {
        "ORI": {"name": "Origin", "terrain": "coastal"},
        "LND": {"name": "Landing", "terrain": "coastal"},
        "SEA": {"name": "Cliff Approach", "terrain": "sea"},
        "SEC": {"name": "Cliff Water", "terrain": "sea"},
        "SEB": {"name": "Beach Approach", "terrain": "sea"},
        "SED": {"name": "Beach Water", "terrain": "sea"},
        "SEX": {"name": "Outer Water", "terrain": "sea"},
        "SEY": {"name": "Far Water", "terrain": "sea"},
    },
    "army_adjacency": {"ORI": [], "LND": []},
    "fleet_adjacency": {
        "ORI": ["SEA", "SEB"],
        "SEA": ["ORI", "SEC"],
        "SEC": ["SEA", "LND"],
        "SEB": ["ORI", "SED"],
        "SED": ["SEB", "LND", "SEX", "SEY"],
        "LND": ["SEC", "SED"],
        "SEX": ["SED", "SEY"],
        "SEY": ["SED", "SEX"],
    },
    "mountain_barriers": [["SEC", "LND"]],
}

TWO_BEACH: MapGraph = build_graph(TWO_BEACH_DATA)

CONVOYED_MOVE = "2A ORI - LND via convoy"
"""RED's Double Army crossing to the landing coast: whole it beats BLU's hold of
1 and dislodges, halved it ties that hold and fails."""


def two_beach_state(units: Mapping[str, Sequence[str]]) -> State:
    """Build a two-beach position from unit specs per power."""
    return State.of(
        "two-beach",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def two_beach_sheets(orders: Mapping[str, Sequence[str]]) -> dict[str, list[Order]]:
    """Parse one order sheet per power."""
    return {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}


def crossing(
    convoying: Sequence[str], *, beach_fleet_attacked: bool = False
) -> tuple[State, dict[str, list[Order]]]:
    """Set up RED's crossing with a Double Fleet ordered to convoy it in each named sea.

    BLU holds the landing coast with one army. With ``beach_fleet_attacked``
    BLU also dislodges the SED fleet: a Double Fleet from SEX, supported from
    SEY, attacks at 3 against the convoying fleet's 2.
    """
    units = {
        "RED": ["2A ORI", *(f"2F {sea}" for sea in convoying)],
        "BLU": ["A LND"],
    }
    orders = {
        "RED": [CONVOYED_MOVE, *(f"2F {sea} C 2A ORI - LND" for sea in convoying)],
        "BLU": ["A LND H"],
    }
    if beach_fleet_attacked:
        units["BLU"] += ["2F SEX", "F SEY"]
        orders["BLU"] += ["2F SEX - SED", "F SEY S 2F SEX - SED"]
    return two_beach_state(units), two_beach_sheets(orders)
