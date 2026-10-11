"""Hold the hand-transcribed standard map to the published board.

Transcription errors are a testing problem: these assertions plus DATC 6.A/6.B
are the guard rails. Counts and split-coast facts come from the published
rulebook map; the classic traps (Stp/Spa/Bul coasts, army-vs-fleet asymmetries)
get explicit spot checks.
"""

from typing import Any

import pytest

from entente.engine.map import Location, Terrain, UnitKind, load_map
from entente.engine.map.loader import build_graph


@pytest.fixture(scope="module")
def graph():  # type: ignore[no-untyped-def]
    """Load the standard map once for the whole module."""
    return load_map("standard")


def test_province_counts(graph) -> None:  # type: ignore[no-untyped-def]
    terrains = [p.terrain for p in graph.provinces.values()]
    assert len(graph.provinces) == 75
    assert terrains.count(Terrain.SEA) == 19
    assert terrains.count(Terrain.INLAND) == 14
    assert terrains.count(Terrain.COASTAL) == 42


def test_supply_center_count(graph) -> None:  # type: ignore[no-untyped-def]
    centers = {key for key, p in graph.provinces.items() if p.is_supply_center}
    assert len(centers) == 34
    home = {c for centers_ in graph.home_centers.values() for c in centers_}
    assert len(home) == 22
    assert home <= centers


def test_capacity_is_one_everywhere_by_data(graph) -> None:  # type: ignore[no-untyped-def]
    assert all(p.capacity == 1 for p in graph.provinces.values())


def test_split_coasts_are_exactly_the_classic_three(graph) -> None:  # type: ignore[no-untyped-def]
    assert graph.coasts == {"BUL": ("EC", "SC"), "SPA": ("NC", "SC"), "STP": ("NC", "SC")}


def test_army_adjacency_covers_all_land_and_is_symmetric(graph) -> None:  # type: ignore[no-untyped-def]
    land = {k for k, p in graph.provinces.items() if p.terrain is not Terrain.SEA}
    assert set(graph.army_adjacency) == land
    for origin, neighbors in graph.army_adjacency.items():
        for neighbor in neighbors:
            assert origin in graph.army_adjacency[neighbor], f"{origin}<->{neighbor}"


def test_fleet_adjacency_is_symmetric_and_touches_no_inland(graph) -> None:  # type: ignore[no-untyped-def]
    for origin, neighbors in graph.fleet_adjacency.items():
        assert graph.terrain(origin.province) is not Terrain.INLAND
        for neighbor in neighbors:
            assert origin in graph.fleet_adjacency[neighbor], f"{origin}<->{neighbor}"


def test_every_sea_and_coastal_province_has_fleet_locations(graph) -> None:  # type: ignore[no-untyped-def]
    for key, province in graph.provinces.items():
        if province.terrain is Terrain.INLAND:
            assert graph.fleet_locations(key) == ()
        else:
            locations = graph.fleet_locations(key)
            assert locations
            for location in locations:
                assert location in graph.fleet_adjacency, f"{location} missing adjacency"


def test_split_coast_provinces_have_no_bare_fleet_location(graph) -> None:  # type: ignore[no-untyped-def]
    for key in ("STP", "SPA", "BUL"):
        assert Location(key) not in graph.fleet_adjacency
        assert not graph.is_legal_location(UnitKind.FLEET, Location(key))
        assert graph.is_legal_location(UnitKind.ARMY, Location(key))


# -- Classic-trap spot checks (DATC 6.B territory) ----------------------------


def test_split_coast_spot_checks(graph) -> None:  # type: ignore[no-untyped-def]
    assert graph.fleet_reachable(Location("STP", "NC"), Location("BAR"))
    assert graph.fleet_reachable(Location("STP", "NC"), Location("NWY"))
    assert not graph.fleet_reachable(Location("STP", "NC"), Location("FIN"))
    assert graph.fleet_reachable(Location("STP", "SC"), Location("BOT"))
    assert graph.fleet_reachable(Location("STP", "SC"), Location("FIN"))
    assert graph.fleet_reachable(Location("STP", "SC"), Location("LVN"))
    assert graph.fleet_reachable(Location("SPA", "NC"), Location("GAS"))
    assert not graph.fleet_reachable(Location("SPA", "NC"), Location("MAR"))
    assert graph.fleet_reachable(Location("SPA", "SC"), Location("MAR"))
    assert not graph.fleet_reachable(Location("SPA", "SC"), Location("GAS"))
    assert graph.fleet_reachable(Location("SPA", "NC"), Location("MAO"))
    assert graph.fleet_reachable(Location("SPA", "SC"), Location("MAO"))
    assert not graph.fleet_reachable(Location("SPA", "NC"), Location("LYO"))
    assert graph.fleet_reachable(Location("BUL", "EC"), Location("BLA"))
    assert graph.fleet_reachable(Location("BUL", "SC"), Location("AEG"))
    assert not graph.fleet_reachable(Location("BUL", "EC"), Location("GRE"))
    assert graph.fleet_reachable(Location("CON"), Location("BUL", "EC"))
    assert graph.fleet_reachable(Location("CON"), Location("BUL", "SC"))


def test_army_vs_fleet_asymmetries(graph) -> None:  # type: ignore[no-untyped-def]
    # Armies cross the Denmark-Sweden sound; no army road Denmark-Norway.
    assert graph.army_reachable("DEN", "SWE")
    assert not graph.army_reachable("DEN", "NWY")
    # No army crossing of the Strait of Gibraltar.
    assert not graph.army_reachable("SPA", "NAF")
    assert graph.army_reachable("NAF", "TUN")
    # Wales-Yorkshire share no coastline; armies pass, fleets do not.
    assert graph.army_reachable("WAL", "YOR")
    assert not graph.fleet_reachable(Location("WAL"), Location("YOR"))
    # Kiel/Constantinople canals: fleets pass through, terrain stays coastal.
    assert graph.fleet_reachable(Location("KIE"), Location("BAL"))
    assert graph.fleet_reachable(Location("KIE"), Location("HEL"))
    assert graph.fleet_reachable(Location("CON"), Location("BLA"))
    assert graph.fleet_reachable(Location("CON"), Location("AEG"))
    # Black Sea and Aegean only connect through Constantinople.
    assert not graph.fleet_reachable(Location("BLA"), Location("AEG"))


def test_convoy_geometry_helpers(graph) -> None:  # type: ignore[no-untyped-def]
    assert graph.sea_neighbors("LON") == frozenset({"ENG", "NTH"})
    assert graph.sea_neighbors("STP") == frozenset({"BAR", "BOT"})
    assert graph.sea_neighbors("MOS") == frozenset()
    assert "NTH" in graph.sea_neighbors("NWY")
    assert graph.fleet_destinations(Location("MAO"), "SPA") == (
        Location("SPA", "NC"),
        Location("SPA", "SC"),
    )
    assert graph.fleet_destinations(Location("GAS"), "SPA") == (Location("SPA", "NC"),)


def test_starting_positions_are_legal_and_complete(graph) -> None:  # type: ignore[no-untyped-def]
    assert len(graph.starting_units) == 7
    total = 0
    for power, specs in graph.starting_units.items():
        homes = graph.home_centers[power]
        for spec in specs:
            kind_text, _, location_text = spec.partition(" ")
            kind = UnitKind(kind_text)
            province, _, coast = location_text.partition("/")
            location = Location(province, coast or None)
            assert graph.is_legal_location(kind, location), spec
            assert province in homes, spec
            total += 1
    assert total == 22


def test_loader_rejects_asymmetric_data() -> None:
    data = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "coastal"},
            "BBB": {"name": "B", "terrain": "coastal"},
        },
        "army_adjacency": {"AAA": ["BBB"], "BBB": []},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="asymmetric"):
        build_graph(data)


def test_loader_rejects_fleet_route_to_inland() -> None:
    data = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "sea"},
            "BBB": {"name": "B", "terrain": "inland"},
        },
        "army_adjacency": {"BBB": []},
        "fleet_adjacency": {"AAA": ["BBB"], "BBB": ["AAA"]},
    }
    with pytest.raises(ValueError, match="inland"):
        build_graph(data)


def test_loader_rejects_undeclared_coast() -> None:
    data = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "sea"},
            "BBB": {"name": "B", "terrain": "coastal", "coasts": ["NC"]},
        },
        "army_adjacency": {"BBB": []},
        "fleet_adjacency": {"AAA": ["BBB/WC"], "BBB/WC": ["AAA"]},
    }
    with pytest.raises(ValueError, match="declared coasts"):
        build_graph(data)


def test_loader_rejects_army_adjacency_touching_sea() -> None:
    data = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "coastal"},
            "SSS": {"name": "S", "terrain": "sea"},
        },
        "army_adjacency": {"AAA": ["SSS"], "SSS": ["AAA"]},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="sea province"):
        build_graph(data)


def test_loader_rejects_adjacency_inside_one_province() -> None:
    """Reject an adjacency that stays inside one province, for armies and for fleets.

    A unit never reaches the province it stands in, and the support reach rule rests on
    that: neither relation may join a province to itself, and a fleet adjacency may not
    join two coasts of one province either.
    """
    army_loop = {
        "name": "broken",
        "provinces": {"AAA": {"name": "A", "terrain": "coastal"}},
        "army_adjacency": {"AAA": ["AAA"]},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="joins AAA to itself"):
        build_graph(army_loop)
    fleet_loop = {
        "name": "broken",
        "provinces": {"SSS": {"name": "S", "terrain": "sea"}},
        "army_adjacency": {},
        "fleet_adjacency": {"SSS": ["SSS"]},
    }
    with pytest.raises(ValueError, match="stays inside province SSS"):
        build_graph(fleet_loop)
    coast_to_coast = {
        "name": "broken",
        "provinces": {"BBB": {"name": "B", "terrain": "coastal", "coasts": ["NC", "SC"]}},
        "army_adjacency": {"BBB": []},
        "fleet_adjacency": {"BBB/NC": ["BBB/SC"], "BBB/SC": ["BBB/NC"]},
    }
    with pytest.raises(ValueError, match="stays inside province BBB"):
        build_graph(coast_to_coast)


def test_loader_rejects_unknown_provinces_and_coast_on_sea() -> None:
    unknown_army = {
        "name": "broken",
        "provinces": {"AAA": {"name": "A", "terrain": "coastal"}},
        "army_adjacency": {"AAA": [], "ZZZ": []},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="unknown provinces"):
        build_graph(unknown_army)
    unknown_fleet = {
        "name": "broken",
        "provinces": {"AAA": {"name": "A", "terrain": "sea"}},
        "army_adjacency": {},
        "fleet_adjacency": {"AAA": ["ZZZ"]},
    }
    with pytest.raises(ValueError, match="unknown province"):
        build_graph(unknown_fleet)
    coast_on_plain = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "sea"},
            "BBB": {"name": "B", "terrain": "coastal"},
        },
        "army_adjacency": {"BBB": []},
        "fleet_adjacency": {"AAA": ["BBB/NC"], "BBB/NC": ["AAA"]},
    }
    with pytest.raises(ValueError, match="declares none"):
        build_graph(coast_on_plain)


def test_loader_rejects_bad_mountain_barriers() -> None:
    base = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "coastal"},
            "BBB": {"name": "B", "terrain": "coastal"},
            "CCC": {"name": "C", "terrain": "coastal"},
        },
        "army_adjacency": {"AAA": ["BBB"], "BBB": ["AAA"], "CCC": []},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="distinct provinces"):
        build_graph({**base, "mountain_barriers": [["AAA", "AAA"]]})
    with pytest.raises(ValueError, match="unknown province"):
        build_graph({**base, "mountain_barriers": [["AAA", "ZZZ"]]})
    with pytest.raises(ValueError, match="non-adjacent"):
        build_graph({**base, "mountain_barriers": [["AAA", "CCC"]]})
    # A single entry is a directed crossing: only the listed order is a barrier.
    graph = build_graph({**base, "mountain_barriers": [["AAA", "BBB"]]})
    assert graph.is_mountain_crossing("AAA", "BBB")
    assert not graph.is_mountain_crossing("BBB", "AAA")
    # Listing both orders is how a symmetric barrier is expressed.
    both = build_graph({**base, "mountain_barriers": [["AAA", "BBB"], ["BBB", "AAA"]]})
    assert both.is_mountain_crossing("AAA", "BBB")
    assert both.is_mountain_crossing("BBB", "AAA")
    # An exact directed duplicate is a transcription error.
    with pytest.raises(ValueError, match="duplicate"):
        build_graph({**base, "mountain_barriers": [["AAA", "BBB"], ["AAA", "BBB"]]})


def test_loader_rejects_inconsistent_province_fields() -> None:
    """Reject a province whose own fields contradict each other, at load time.

    Each of these would otherwise load silently and misbehave somewhere far
    away: a non-positive capacity makes a province no unit can ever enter,
    which ``check_state`` never complains about because it only tests
    ``count > capacity``; a negative centre count unmakes a supply centre;
    coasts on an inland province are unreachable, because fleet adjacency may
    not name one; and a ``restricted_to`` typo yields a province nobody may
    enter, since the ruleset compares that string to a unit's power and only
    the map can catch the misspelling.
    """
    base = {
        "name": "broken",
        "provinces": {
            "AAA": {"name": "A", "terrain": "coastal"},
            "BBB": {"name": "B", "terrain": "inland"},
        },
        "army_adjacency": {"AAA": ["BBB"], "BBB": ["AAA"]},
        "fleet_adjacency": {},
        "starting_units": {"RED": ["A AAA"]},
    }

    def with_province(key: str, **fields: object) -> dict[str, Any]:
        provinces = {k: dict(v) for k, v in base["provinces"].items()}  # type: ignore[union-attr]
        provinces[key].update(fields)
        return {**base, "provinces": provinces}

    build_graph(base)  # the base itself is well-formed

    for bad_capacity in (0, -1, True, "2"):  # bool is an int subclass: `true` is not capacity 1
        with pytest.raises(ValueError, match="capacity"):
            build_graph(with_province("AAA", capacity=bad_capacity))
    build_graph(with_province("AAA", capacity=None))  # unbounded stays legal
    build_graph(with_province("AAA", capacity=3))

    with pytest.raises(ValueError, match="supply centers"):
        build_graph(with_province("AAA", supply_center=-1))

    with pytest.raises(ValueError, match="inland province BBB declares coasts"):
        build_graph(with_province("BBB", coasts=["nc"]))

    with pytest.raises(ValueError, match="undeclared power"):
        build_graph(with_province("AAA", restricted_to="REDD"))
    # A power the map declares -- in either table -- is fine.
    build_graph(with_province("AAA", restricted_to="RED"))
    homes = {**base, "home_centers": {"BLUE": ["AAA"]}}
    build_graph({**homes, "provinces": with_province("AAA", restricted_to="BLUE")["provinces"]})


def test_loader_rejects_unknown_army_adjacency_neighbour() -> None:
    """Name the broken pair when an adjacency lists an unknown neighbour.

    Validating the neighbour, and not only the key it hangs under, is what
    turns a bare ``KeyError`` from a later terrain lookup into a message that
    says which edge is wrong.
    """
    data = {
        "name": "broken",
        "provinces": {"AAA": {"name": "A", "terrain": "coastal"}},
        "army_adjacency": {"AAA": ["ZZZ"]},
        "fleet_adjacency": {},
    }
    with pytest.raises(ValueError, match="AAA-ZZZ names an unknown province"):
        build_graph(data)


def test_load_map_rejects_unknown_names() -> None:
    with pytest.raises(FileNotFoundError, match="no packaged map"):
        load_map("atlantis")


def test_province_accessor_and_names(graph) -> None:  # type: ignore[no-untyped-def]
    assert graph.province("PAR").name == "Paris"
    assert graph.province("STP").is_supply_center
