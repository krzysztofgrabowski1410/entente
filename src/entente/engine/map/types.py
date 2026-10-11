"""Core identity types for the board model.

The load-bearing distinction: a :class:`Province` is a node of the board graph;
a :class:`Location` is a place a unit can occupy -- a province plus an optional
named coast. Provinces with split coasts (e.g. St Petersburg on the standard
map) contain multiple fleet locations but a single army location, so adjacency
is always defined between locations per unit kind, never between raw provinces.
Getting this wrong is the classic engine mistake that DATC section 6.B punishes.

Province capacity generalizes occupancy: classic Diplomacy has capacity 1
everywhere; variant maps may declare multi-unit provinces (capacity ``n``) or
unbounded special provinces (capacity ``None``). The ruleset's entry rule
consumes this value -- the classic bounce rule is the capacity-1 special case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Terrain(Enum):
    """Terrain trichotomy: armies occupy inland+coastal, fleets sea+coastal."""

    INLAND = "inland"
    COASTAL = "coastal"
    SEA = "sea"


class UnitKind(Enum):
    """The two unit kinds. Unit *strength* is a ruleset concern, not a kind."""

    ARMY = "A"
    FLEET = "F"


@dataclass(frozen=True, slots=True)
class Province:
    """A node of the board graph, identified by its canonical short key."""

    key: str
    name: str
    terrain: Terrain
    supply_center_count: int = 0
    """How many supply centers the province holds. Classic maps have 0 or 1;
    variant maps may stack several in one special province (Beleriand's
    Angband holds 7 -- its rulebook, "Provinces off the map")."""
    capacity: int | None = 1
    """Maximum number of units that may occupy the province; ``None`` = unbounded."""
    fortress: bool = False
    """Fortress provinces grant +1 hold strength to a non-relocating occupant
    (Beleriand rulebook, "Fortresses"; consumed by the ruleset's strength
    modifiers -- docs/adjudication.tex Sec. 2)."""
    restricted_to: str | None = None
    """If set, only units of this power may enter (Beleriand's Angband/Eriador
    entry permission -- docs/adjudication.tex Sec. 4.3)."""
    one_way: bool = False
    """Units may leave but not (re-)enter; exceptions are ruleset policy
    (docs/adjudication.tex Sec. 7 ``entry_permission``)."""

    @property
    def is_supply_center(self) -> bool:
        """Whether the province holds at least one supply center."""
        return self.supply_center_count > 0


@dataclass(frozen=True, slots=True)
class Location:
    """A place a unit can occupy: a province key plus an optional named coast."""

    province: str
    coast: str | None = None

    def __str__(self) -> str:
        return self.province if self.coast is None else f"{self.province}/{self.coast}"
