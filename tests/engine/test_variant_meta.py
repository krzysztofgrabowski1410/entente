"""Keep every shipped variant described in exactly one place.

A ``VariantSpec`` is a variant's single registration site: the loader's search
path, the presentation metadata, the CLI's choices and the calibration
off-board table are all derived from it, so a new variant is added once and
cannot be half-registered.
"""

from entente.engine.map.loader import data_packages
from entente.engine.variant_meta import VARIANT_SPECS, off_board_provinces


def test_shipped_specs() -> None:
    """Register both shipped variants, standard first, with their calendar facts.

    The order is the presentation order, and each spec carries the calendar
    facts a caller would otherwise have to hardcode: the starting year, which
    way it counts, and the season names.
    """
    assert list(VARIANT_SPECS) == ["standard", "beleriand"]
    standard = VARIANT_SPECS["standard"]
    beleriand = VARIANT_SPECS["beleriand"]
    assert (standard.map_name, standard.start_year, standard.year_direction) == (
        "standard",
        1901,
        1,
    )
    assert (beleriand.map_name, beleriand.start_year, beleriand.year_direction) == (
        "beleriand",
        25,
        -1,
    )
    assert standard.data_package is None
    assert beleriand.data_package == "entente.variants.beleriand"
    # Both shipped variants play under their own ruleset; the field exists so
    # a map-only variant can reuse "standard" without a fake RULESETS entry.
    assert standard.ruleset == standard.name
    assert beleriand.ruleset == beleriand.name


def test_data_packages_search_path() -> None:
    """Search the engine's own data directory first, then each variant package's.

    The search path is derived from the registry rather than written out, so
    registering a variant is what puts its data on the path.
    """
    assert data_packages() == ("entente.engine.map", "entente.variants.beleriand")


def test_off_board_provinces() -> None:
    """Report each variant's off-map tiles, and none for a fully drawn map.

    Angband and Eriador are Beleriand's; the standard board has no such tile.
    """
    assert off_board_provinces("beleriand") == frozenset({"ANG", "ERI"})
    assert off_board_provinces("standard") == frozenset()
