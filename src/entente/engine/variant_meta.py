"""Per-variant metadata registry: everything a shipped variant carries besides its law.

The law registries (:data:`entente.engine.rules.RULESETS`,
:data:`entente.engine.game_hooks.GAME_HOOKS`) know which rules exist; this
module records the rest: where a variant's map data package lives, how its
calendar reads, and which provinces live on their own off-map tiles -- what a
front end would offer and draw. One :class:`VariantSpec` per shipped variant,
keyed by its name -- the standard entry is seeded here, and variant
packages register theirs on import exactly like the law registries
(``entente/__init__`` triggers them). Every consumer (map loading today, and
any board renderer or front end) reads these facts from here rather than
restating them in a table of its own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """One shipped variant: names, calendar presentation, and packaging."""

    name: str
    """Registry key and the variant's public identity (what a front end lists)."""

    ruleset: str
    """The :data:`entente.engine.rules.RULESETS` entry this variant plays
    under. Equal to ``name`` for both shipped variants; a map-only variant
    (new board, unchanged law) reuses ``"standard"`` here."""

    map_name: str
    """The packaged map (:func:`entente.engine.map.load_map`)."""

    start_year: int
    """First game year (standard: 1901; Beleriand: 25, counting down)."""

    year_direction: int
    """``+1`` ascending calendar, ``-1`` descending."""

    caption: str
    """Meta line describing the calendar, for a front end to show."""

    data_package: str | None = None
    """Package whose ``data/`` directory holds the map JSON; ``None`` means
    the engine's own ``entente.engine.map`` (always searched first)."""

    off_board: frozenset[str] = frozenset()
    """Provinces represented by their OWN off-map tile image, not drawn on
    the main board picture (the rulebook's "Provinces off the map")."""


VARIANT_SPECS: dict[str, VariantSpec] = {}
"""Shipped variants by variant name; insertion order is presentation order
(the order a front end lists them in). Variant packages register on import via
``VARIANT_SPECS.setdefault``, exactly like ``RULESETS``/``GAME_HOOKS``."""

VARIANT_SPECS["standard"] = VariantSpec(
    name="standard",
    ruleset="standard",
    map_name="standard",
    start_year=1901,
    year_direction=1,
    caption="counts up from 1901",
)


def off_board_provinces(map_name: str) -> frozenset[str]:
    """Return the province keys with their own off-map tile on ``map_name``.

    The union over the specs playing on that map; empty for fully drawn
    boards.
    """
    keys: frozenset[str] = frozenset()
    for spec in VARIANT_SPECS.values():
        if spec.map_name == map_name:
            keys |= spec.off_board
    return keys
