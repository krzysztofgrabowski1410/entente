"""Map data loader: JSON -> validated :class:`MapGraph`.

Map data files are hand-transcribed from primary rule documents; this loader
treats transcription errors as a *testing* problem and therefore validates
structure aggressively at load time:

- every adjacency endpoint names a declared province/coast,
- adjacency is symmetric per unit kind, and it always leaves the province:
  no province is army-adjacent to itself and no fleet adjacency joins two
  locations of one province, so a unit never reaches the province it stands in
  (the support reach rule rests on this -- docs/adjudication.tex Sec. 1),
- army adjacency never touches sea provinces; fleet adjacency never touches
  inland provinces and only uses declared coast names,
- split-coast provinces are entered per coast, never as a bare province,
- capacity is data (``default_capacity`` plus per-province overrides) -- the
  engine never assumes capacity 1 (docs/adjudication.tex Sec. 1, Sec. 4.3),
- ``mountain_barriers`` holds ordered ``[from, to]`` pairs -- directed crossings
  naming declared provinces that lie on real adjacency,
- ``supply_center`` is a bool (0/1 centers) or a non-negative int count (variant
  maps may stack several centers in one special province),
- each province's own fields are self-consistent: capacity is ``None``
  (unbounded) or a positive int, named coasts sit on a non-inland province, and
  ``restricted_to`` names a power the map declares.

One shape throughout, so a reader can predict where a rule lives: ``_build_*``
constructs and enforces only what is decidable from its own input;
``_validate_*`` cross-references what is already built. Cross-references need
their referents, so :func:`build_graph` runs the pairs in dependency order --
provinces, then each adjacency relation, then the mountain barriers that name
both.

Map files live in the engine's own data directory or in a variant package's
``data/`` directory (``VariantSpec.data_package``); :func:`load_map` searches
the engine's first, then every registered variant's -- variant maps are data
next to their ruleset, never engine code.
"""

from __future__ import annotations

import json
from functools import cache
from importlib.resources import files
from typing import Any

from entente.engine.map.graph import MapGraph
from entente.engine.map.types import Location, Province, Terrain
from entente.engine.variant_meta import VARIANT_SPECS

_ENGINE_DATA_PACKAGE = "entente.engine.map"


def data_packages() -> tuple[str, ...]:
    """Return the packages whose ``data/`` directory may hold ``<name>.json`` map files.

    The engine's own, then each registered variant's, in registration order
    (:data:`entente.engine.variant_meta.VARIANT_SPECS`).
    """
    packages = [_ENGINE_DATA_PACKAGE]
    for spec in VARIANT_SPECS.values():
        if spec.data_package is not None and spec.data_package not in packages:
            packages.append(spec.data_package)
    return tuple(packages)


def _parse_location(text: str) -> Location:
    province, _, coast = text.partition("/")
    return Location(province, coast or None)


def _build_provinces(data: dict[str, Any]) -> dict[str, Province]:
    default_capacity = data.get("default_capacity", 1)
    provinces: dict[str, Province] = {}
    for key, raw in data["provinces"].items():
        provinces[key] = Province(
            key=key,
            name=raw["name"],
            terrain=Terrain(raw["terrain"]),
            supply_center_count=int(raw.get("supply_center", 0)),
            capacity=raw.get("capacity", default_capacity),
            fortress=raw.get("fortress", False),
            restricted_to=raw.get("restricted_to"),
            one_way=raw.get("one_way", False),
        )
    return provinces


def _validate_provinces(
    provinces: dict[str, Province],
    coasts: dict[str, tuple[str, ...]],
    powers: frozenset[str],
) -> None:
    """Per-province field consistency, checked against the rest of the file.

    ``Terrain`` and the ``name`` key already fail in :func:`_build_provinces`
    (an unknown terrain string is an ``Enum`` ``ValueError``, a missing name a
    ``KeyError``); what is left is the data no constructor can catch. A
    non-positive capacity yields a province no unit may ever enter and
    :func:`entente.engine.state.check_state` would never complain -- it only
    tests ``count > capacity``. A negative center count would silently unmake a
    supply center. Named coasts on an inland province are unreachable, since
    fleet adjacency may not name one. A ``restricted_to`` typo produces a
    province nobody may enter, again silently: the ruleset compares the string
    to a unit's power (Sec. 4.3), so only the map can catch the misspelling.
    """
    for key, province in provinces.items():
        capacity = province.capacity
        # bool is a subclass of int, so ``"capacity": true`` would otherwise
        # read as capacity 1 -- exactly the transcription slip this catches.
        if capacity is not None and (
            isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
        ):
            raise ValueError(f"province {key} declares capacity {capacity!r}: want null or >= 1")
        if province.supply_center_count < 0:
            raise ValueError(
                f"province {key} declares {province.supply_center_count} supply centers"
            )
        if province.terrain is Terrain.INLAND and coasts.get(key):
            raise ValueError(f"inland province {key} declares coasts {coasts[key]}")
        if province.restricted_to is not None and province.restricted_to not in powers:
            raise ValueError(
                f"province {key} is restricted to undeclared power {province.restricted_to!r}"
            )


def _build_mountain_edges(raw_edges: list[list[str]]) -> frozenset[tuple[str, str]]:
    """Normalize the ``mountain_barriers`` list into directed crossings.

    Each entry is an ordered ``[from, to]`` pair of distinct province keys.
    Listing both orders is how a symmetric barrier is expressed; an exact
    duplicate directed entry is a transcription error. Whether the endpoints
    exist and lie on real adjacency is :func:`_validate_mountain_edges`' job --
    it needs the provinces and both adjacency relations, which do not exist
    yet at this point.
    """
    edges: set[tuple[str, str]] = set()
    for pair in raw_edges:
        if len(pair) != 2 or pair[0] == pair[1]:
            raise ValueError(f"mountain barrier must join two distinct provinces: {pair}")
        edge = (pair[0], pair[1])
        if edge in edges:
            raise ValueError(f"duplicate mountain barrier crossing: {list(edge)}")
        edges.add(edge)
    return frozenset(edges)


def _validate_mountain_edges(
    provinces: dict[str, Province],
    army_adjacency: dict[str, frozenset[str]],
    fleet_adjacency: dict[Location, frozenset[Location]],
    edges: frozenset[tuple[str, str]],
) -> None:
    """Check that every crossing names declared provinces that some unit kind finds adjacent.

    A barrier on a non-edge is a transcription error. Iterates sorted, so the
    first failure reported is data-order-independent.
    """
    fleet_pairs = {
        frozenset((origin.province, neighbor.province))
        for origin, neighbors in fleet_adjacency.items()
        for neighbor in neighbors
    }
    for origin, target in sorted(edges):
        for key in (origin, target):
            if key not in provinces:
                raise ValueError(f"mountain barrier names unknown province {key!r}")
        army_linked = target in army_adjacency.get(origin, frozenset())
        if not army_linked and frozenset((origin, target)) not in fleet_pairs:
            raise ValueError(f"mountain barrier on non-adjacent pair: {sorted((origin, target))}")


def _validate_army_adjacency(
    provinces: dict[str, Province], adjacency: dict[str, frozenset[str]]
) -> None:
    unknown = set(adjacency) - set(provinces)
    if unknown:
        raise ValueError(f"army adjacency for unknown provinces: {sorted(unknown)}")
    for origin, neighbors in adjacency.items():
        if provinces[origin].terrain is Terrain.SEA:
            raise ValueError(f"army adjacency declared for sea province {origin}")
        for neighbor in neighbors:
            if neighbor not in provinces:
                raise ValueError(f"army adjacency {origin}-{neighbor} names an unknown province")
            if neighbor == origin:
                raise ValueError(f"army adjacency joins {origin} to itself")
            if provinces[neighbor].terrain is Terrain.SEA:
                raise ValueError(f"army adjacency {origin}-{neighbor} reaches a sea province")
            if origin not in adjacency.get(neighbor, frozenset()):
                raise ValueError(f"army adjacency asymmetric: {origin}->{neighbor}")


def _validate_fleet_adjacency(
    provinces: dict[str, Province],
    coasts: dict[str, tuple[str, ...]],
    adjacency: dict[Location, frozenset[Location]],
) -> None:
    def check_location(loc: Location, context: str) -> None:
        province = provinces.get(loc.province)
        if province is None:
            raise ValueError(f"{context}: unknown province {loc}")
        if province.terrain is Terrain.INLAND:
            raise ValueError(f"{context}: fleet adjacency reaches inland province {loc}")
        named = coasts.get(loc.province, ())
        if named and (loc.coast not in named):
            raise ValueError(f"{context}: {loc} must name one of the declared coasts {named}")
        if not named and loc.coast is not None:
            raise ValueError(f"{context}: {loc} names a coast but {loc.province} declares none")

    for origin, neighbors in adjacency.items():
        check_location(origin, "fleet adjacency key")
        for neighbor in neighbors:
            check_location(neighbor, f"fleet adjacency of {origin}")
            if neighbor.province == origin.province:
                raise ValueError(
                    f"fleet adjacency {origin}-{neighbor} stays inside province {origin.province}"
                )
            if origin not in adjacency.get(neighbor, frozenset()):
                raise ValueError(f"fleet adjacency asymmetric: {origin}->{neighbor}")


def build_graph(data: dict[str, Any]) -> MapGraph:
    """Construct and validate a :class:`MapGraph` from parsed JSON data.

    Build/validate pairs run in dependency order: each ``_validate_*`` fires as
    soon as everything it cross-references exists, and never before.
    """
    starting_units = {
        power: tuple(units) for power, units in data.get("starting_units", {}).items()
    }
    home_centers = {
        power: frozenset(centers) for power, centers in data.get("home_centers", {}).items()
    }

    provinces = _build_provinces(data)
    coasts = {key: tuple(raw.get("coasts", ())) for key, raw in data["provinces"].items()}
    coasts = {key: named for key, named in coasts.items() if named}
    _validate_provinces(provinces, coasts, frozenset(starting_units) | frozenset(home_centers))

    army_adjacency = {
        origin: frozenset(neighbors) for origin, neighbors in data["army_adjacency"].items()
    }
    _validate_army_adjacency(provinces, army_adjacency)

    fleet_adjacency = {
        _parse_location(origin): frozenset(_parse_location(neighbor) for neighbor in neighbors)
        for origin, neighbors in data["fleet_adjacency"].items()
    }
    _validate_fleet_adjacency(provinces, coasts, fleet_adjacency)

    mountain_edges = _build_mountain_edges(data.get("mountain_barriers", []))
    _validate_mountain_edges(provinces, army_adjacency, fleet_adjacency, mountain_edges)

    sea_neighbors: dict[str, frozenset[str]] = {}
    for key, province in provinces.items():
        if province.terrain is Terrain.INLAND:
            continue
        named = coasts.get(key, ())
        own_locations = (
            tuple(Location(key, coast) for coast in named) if named else (Location(key),)
        )
        touching = frozenset(
            neighbor.province
            for loc in own_locations
            for neighbor in fleet_adjacency.get(loc, frozenset())
            if provinces[neighbor.province].terrain is Terrain.SEA
        )
        if touching:
            sea_neighbors[key] = touching

    return MapGraph(
        name=data["name"],
        provinces=provinces,
        coasts=coasts,
        army_adjacency=army_adjacency,
        fleet_adjacency=fleet_adjacency,
        starting_units=starting_units,
        home_centers=home_centers,
        _sea_neighbors=sea_neighbors,
        mountain_edges=mountain_edges,
    )


@cache
def load_map(name: str) -> MapGraph:
    """Load a packaged map by name (e.g. ``"standard"``, ``"beleriand"``).

    Cached; graphs are immutable. Searches the engine data directory, then
    each variant package's.
    """
    for package in data_packages():
        resource = files(package).joinpath("data").joinpath(f"{name}.json")
        if not resource.is_file():
            continue
        data = json.loads(resource.read_text(encoding="utf-8"))
        if data["name"] != name:
            raise ValueError(f"map file {name}.json declares name {data['name']!r}")
        return build_graph(data)
    raise FileNotFoundError(f"no packaged map named {name!r}")
