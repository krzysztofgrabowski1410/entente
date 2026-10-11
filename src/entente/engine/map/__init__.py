"""Board model: provinces, locations, adjacency graph, map data loading.

Maps are pure data (JSON under ``data/``), hand-transcribed from primary rule
documents and guarded by integrity tests. Nothing in this package knows about
any specific map.
"""

from entente.engine.map.graph import MapGraph
from entente.engine.map.loader import build_graph, data_packages, load_map
from entente.engine.map.types import Location, Province, Terrain, UnitKind

__all__ = [
    "Location",
    "MapGraph",
    "Province",
    "Terrain",
    "UnitKind",
    "build_graph",
    "data_packages",
    "load_map",
]
