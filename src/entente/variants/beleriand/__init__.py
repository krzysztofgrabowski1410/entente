"""The Beleriand variant: the map, the Layer-1 ruleset and the Layer-2 game layer.

``data/beleriand.json`` is the map, hand-transcribed from the variant's
rulebook and carrying its provenance; ``ruleset.py`` instantiates the
Beleriand column of docs/adjudication.tex Sec. 7 (Layer 1); ``layer2.py`` +
``ledger.py`` + ``winter.py`` are the engine-level game layer -- knowledge
facts, masks, timers, timed events, per-power victory (Layer 2). Importing
this package registers the ruleset, the game hooks, and the variant spec.
"""

from entente.engine.variant_meta import VARIANT_SPECS, VariantSpec
from entente.variants.beleriand.layer2 import BeleriandHooks
from entente.variants.beleriand.ruleset import BELERIAND, BeleriandRuleset

VARIANT_SPECS.setdefault(
    BELERIAND.name,
    VariantSpec(
        name=BELERIAND.name,
        ruleset=BELERIAND.name,
        map_name="beleriand",
        start_year=25,
        year_direction=-1,
        caption="counts down toward 1 (the Valar intervene)",
        data_package="entente.variants.beleriand",
        # Angband and Eriador are the rulebook's "Provinces off the map":
        # each is its own side tile (img/Morgoth special tile.png,
        # img/ludzieObszarSpecjalny.png), never drawn on the board itself, so
        # the board segmentation skips them.
        off_board=frozenset({"ANG", "ERI"}),
    ),
)

__all__ = ["BELERIAND", "BeleriandHooks", "BeleriandRuleset"]
