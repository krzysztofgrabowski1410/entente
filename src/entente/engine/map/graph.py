"""Board graph: provinces, per-unit-kind location adjacency, convoy geometry.

Realizes the map model of docs/adjudication.tex Sec. 1: a finite set of provinces,
a set of :class:`Location` values (province + optional named coast), a terrain
function, and a capacity function. Armies and fleets get *separate* adjacency
relations over locations -- armies move province-to-province (their location
coast is always ``None``); fleets move location-to-location, where split-coast
provinces (Stp/Spa/Bul on the standard map) contribute one location per coast.

The graph is pure data: rulesets never read it directly except through the
adjudicator, and nothing here is standard-map-specific.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from entente.engine.map.types import Location, Province, Terrain, UnitKind


@dataclass(frozen=True, slots=True)
class MapGraph:
    """An immutable, validated board graph (see :mod:`entente.engine.map.loader`)."""

    name: str
    provinces: Mapping[str, Province]
    coasts: Mapping[str, tuple[str, ...]]
    """Named coasts per province key; empty tuple for single-coast/sea/inland."""
    army_adjacency: Mapping[str, frozenset[str]]
    """Province key -> army-reachable province keys."""
    fleet_adjacency: Mapping[Location, frozenset[Location]]
    """Fleet location -> fleet-reachable locations."""
    starting_units: Mapping[str, tuple[str, ...]]
    """Power -> unit specs (e.g. ``"F STP/SC"``, ``"2A GEL"``), as written in
    the map data."""
    home_centers: Mapping[str, frozenset[str]]
    _sea_neighbors: Mapping[str, frozenset[str]] = field(repr=False)
    """Province key -> sea province keys touching any of its fleet locations."""
    mountain_edges: frozenset[tuple[str, str]] = frozenset()
    """Directed ``(origin, target)`` crossings that traverse a mountain barrier
    (docs/adjudication.tex Sec. 2: crossing or supporting through one halves
    strength under rulesets that say so)."""

    def province(self, key: str) -> Province:
        """Return the :class:`Province` with key ``key``."""
        return self.provinces[key]

    def is_mountain_crossing(self, origin: str, target: str) -> bool:
        """Whether crossing from ``origin`` into ``target`` traverses a mountain barrier."""
        return (origin, target) in self.mountain_edges

    def terrain(self, key: str) -> Terrain:
        """Return the terrain of province ``key``."""
        return self.provinces[key].terrain

    def capacity(self, key: str) -> int | None:
        """Return the capacity of province ``key``; ``None`` means unbounded."""
        return self.provinces[key].capacity

    def fleet_locations(self, key: str) -> tuple[Location, ...]:
        """Return every fleet-occupiable location of a province (empty for inland)."""
        if self.provinces[key].terrain is Terrain.INLAND:
            return ()
        named = self.coasts.get(key, ())
        if named:
            return tuple(Location(key, coast) for coast in named)
        return (Location(key),)

    def locations_for(self, kind: UnitKind, key: str) -> tuple[Location, ...]:
        """Return the locations of the province a unit of ``kind`` may legally occupy."""
        terrain = self.provinces[key].terrain
        if kind is UnitKind.ARMY:
            return (Location(key),) if terrain is not Terrain.SEA else ()
        return self.fleet_locations(key)

    def is_legal_location(self, kind: UnitKind, location: Location) -> bool:
        """Whether a unit of ``kind`` may occupy ``location`` at all."""
        return location in self.locations_for(kind, location.province)

    def army_reachable(self, origin: str, destination: str) -> bool:
        """Whether an army may move from ``origin`` to ``destination`` (province keys)."""
        return destination in self.army_adjacency.get(origin, frozenset())

    def fleet_reachable(self, origin: Location, destination: Location) -> bool:
        """Whether a fleet may move from ``origin`` to ``destination`` (exact locations)."""
        return destination in self.fleet_adjacency.get(origin, frozenset())

    def fleet_destinations(self, origin: Location, province: str) -> tuple[Location, ...]:
        """Return the locations of ``province`` a fleet at ``origin`` reaches, sorted by coast."""
        reachable = self.fleet_adjacency.get(origin, frozenset())
        return tuple(
            sorted(
                (loc for loc in reachable if loc.province == province),
                key=lambda loc: loc.coast or "",
            )
        )

    def sea_neighbors(self, key: str) -> frozenset[str]:
        """Return the sea provinces touching any fleet location of ``key`` (Sec. 3.5)."""
        return self._sea_neighbors.get(key, frozenset())
