"""Hold the hand-transcribed Beleriand map to every geometry fact the rulebook states.

The rulebook carries no adjacency table, so the map's edges are a
reconstruction, and the only external ground truth is the text itself. Every
geometry fact the rulebook does state is asserted here verbatim: the
abbreviation table's province count, the 13 Areas, the off-map boxes and the
neighbours they declare, the Angband and Eriador clauses, the six fortresses,
Ulmo's coastal list, the starting armies, and every adjacency a rulebook
example order exercises. The case files are the end-to-end check on top.
"""

import pytest

from entente.engine.map import Terrain, load_map
from entente.engine.orders import parse_unit_spec
from entente.engine.state import Unit
from entente.variants.beleriand.layer2 import ULMO_PROVINCES

# The ten land Areas and three sea Areas (rulebook "Areas").
LAND_AREAS = {"HIT", "ANF", "LOT", "ELA", "TAL", "NEB", "SEB", "MOS", "SOS", "TID"}
SEA_AREAS = {"NBE", "SBE", "BOB"}

# The rulebook's "Ulmo's wrath" section enumerates the coastal provinces the
# wrath can reach. That list governs the wrath, not the terrain flag: the
# island is coastal too, because a fleet reaches it, and the two agree
# everywhere except there.
ULMO_COASTAL = {
    "NLA",
    "LAM",
    "NEV",
    "VIN",
    "NFA",
    "BRI",
    "BNI",
    "EGL",
    "SFA",
    "SWB",
    "ARV",
    "MSI",
    "WTD",
    "CBB",
    "ELO",
}

FORTRESSES = {"TOL", "GON", "MEN", "NAR", "BEL", "NOG"}  # rulebook "Fortresses"


@pytest.fixture(scope="module")
def graph():  # type: ignore[no-untyped-def]
    """Load the Beleriand map once for the whole module."""
    return load_map("beleriand")


def test_province_counts(graph) -> None:  # type: ignore[no-untyped-def]
    # 79 abbreviation-table provinces + 3 off-map boxes + Angband + Eriador.
    assert len(graph.provinces) == 84
    terrains = [p.terrain for p in graph.provinces.values()]
    assert terrains.count(Terrain.SEA) == 3
    assert terrains.count(Terrain.COASTAL) == 16
    assert terrains.count(Terrain.INLAND) == 65


def test_the_thirteen_areas_have_capacity_six(graph) -> None:  # type: ignore[no-untyped-def]
    areas = {key for key, p in graph.provinces.items() if p.capacity == 6}
    assert areas == LAND_AREAS | SEA_AREAS
    assert len(areas) == 13
    for key in SEA_AREAS:
        assert graph.terrain(key) is Terrain.SEA


def test_off_map_boxes(graph) -> None:  # type: ignore[no-untyped-def]
    # Rulebook "Off-map boxes": capacity 1, no supply centers, and exactly
    # the declared neighbor pairs; the box is the ONLY connection of its pair.
    for box, pair in {
        "NLA-HIT": {"NLA", "HIT"},
        "HIT-ANF": {"HIT", "ANF"},
        "LOT-ELA": {"LOT", "ELA"},
    }.items():
        province = graph.province(box)
        assert province.capacity == 1
        assert not province.is_supply_center
        assert graph.army_adjacency[box] == frozenset(pair)
        a, b = sorted(pair)
        assert not graph.army_reachable(a, b), f"{a}-{b} must connect only through {box}"


def test_angband_and_eriador_flags(graph) -> None:  # type: ignore[no-untyped-def]
    angband = graph.province("ANG")
    assert angband.capacity is None
    assert angband.restricted_to == "MORGOTH"
    assert angband.one_way
    assert angband.supply_center_count == 7  # rulebook: seven off-map centers
    assert graph.army_adjacency["ANG"] == frozenset({"ANF"})
    eriador = graph.province("ERI")
    assert eriador.capacity is None
    assert eriador.restricted_to == "MEN"
    assert eriador.one_way
    assert eriador.supply_center_count == 0  # migration centers are Layer 2
    assert graph.army_adjacency["ERI"] == frozenset({"ELA"})


def test_supply_center_counts(graph) -> None:  # type: ignore[no-untyped-def]
    total = sum(p.supply_center_count for p in graph.provinces.values())
    assert total == 53  # 32 on-map home + 14 neutral + Angband's 7
    home = {c for centers in graph.home_centers.values() for c in centers}
    assert len(home) == 32  # 31 on-map + Angband
    for center in home:
        assert graph.province(center).is_supply_center


def test_fortresses_are_exactly_the_rulebook_six(graph) -> None:  # type: ignore[no-untyped-def]
    assert {k for k, p in graph.provinces.items() if p.fortress} == FORTRESSES
    for key in FORTRESSES:
        assert graph.capacity(key) == 1  # no fortress is an Area


def test_coastal_terrain_covers_ulmos_list(graph) -> None:  # type: ignore[no-untyped-def]
    coastal = {k for k, p in graph.provinces.items() if p.terrain is Terrain.COASTAL}
    assert coastal >= ULMO_COASTAL
    assert coastal - ULMO_COASTAL == {"IOB"}  # the island, which the wrath omits
    # Layer 2's dice list is the same rulebook transcription, so tying the two
    # together catches a correction that reaches only one side of the board.
    assert set(ULMO_PROVINCES) == ULMO_COASTAL


def test_army_adjacency_is_symmetric_and_total_on_land(graph) -> None:  # type: ignore[no-untyped-def]
    land = {k for k, p in graph.provinces.items() if p.terrain is not Terrain.SEA}
    assert set(graph.army_adjacency) == land
    for origin, neighbors in graph.army_adjacency.items():
        for neighbor in neighbors:
            assert origin in graph.army_adjacency[neighbor], f"{origin}<->{neighbor}"
    # The island is army-isolated (convoy-only); everything else connects.
    assert graph.army_adjacency["IOB"] == frozenset()
    for key in land - {"IOB"}:
        assert graph.army_adjacency[key], key


def test_fleet_adjacency_is_symmetric_and_stays_off_inland(graph) -> None:  # type: ignore[no-untyped-def]
    for origin, neighbors in graph.fleet_adjacency.items():
        assert graph.terrain(origin.province) is not Terrain.INLAND
        for neighbor in neighbors:
            assert origin in graph.fleet_adjacency[neighbor], f"{origin}<->{neighbor}"
    # No split coasts anywhere (rulebook 4.B.1: no double coastlines).
    assert graph.coasts == {}
    # Every sea/coastal province has a fleet location wired into the graph.
    for key, province in graph.provinces.items():
        if province.terrain is Terrain.INLAND:
            assert graph.fleet_locations(key) == ()
        else:
            for location in graph.fleet_locations(key):
                assert location in graph.fleet_adjacency, f"{location} missing adjacency"


def test_rulebook_example_adjacencies(graph) -> None:  # type: ignore[no-untyped-def]
    """Assert every adjacency a rulebook example order exercises."""
    for a, b in [
        ("GEL", "HIG"),  # A(GEL)-HIG (worked example)
        ("THA", "MMA"),  # A(THA)-MMa
        ("HID", "HIG"),  # A(HID)-HIG
        ("MMA", "HIG"),  # 2A(MMa) sup A(HID)-HIG
        ("NLA", "LAM"),  # 2A(NLa) sup hold A(Lam)
        ("FOR", "NEB"),  # A(FoR) dis Morgoth's A(NEB)
        ("EST", "NEB"),  # A(Est) sup ... dis Morgoth's A(NEB)
        ("NOS", "NEB"),  # A(NOs)->NEB
        ("FSI", "FOR"),  # A(FSi)->FoR
        ("FOB", "TAL"),  # 2A(Tal) bar Doriath's 2A(FoB)-Tal
        ("MSI", "SEB"),  # 2A(MSi) dis Ossiriand's A(SEB)
        ("AND", "SEB"),  # A(And) retreat to SEB
        ("GVA", "ETD"),  # A(GVa)-ETa (the rulebook's Move example)
        ("NEB", "THA"),  # A(Est) dis Morgoth's A(NEB) sup A(Gel)-Tha (action-naming example)
    ]:
        assert graph.army_reachable(a, b), f"{a}-{b}"
    # 2F(BoB) cnv 2A(CBB)-Arv: both endpoints coastal on the Bay of Balar.
    assert "BOB" in graph.sea_neighbors("CBB")
    assert "BOB" in graph.sea_neighbors("ARV")


def test_mountain_barriers_lie_on_range_edges(graph) -> None:  # type: ignore[no-untyped-def]
    # A barrier is directional, so the 78 undirected range pairs the rulebook
    # draws become 84 directed crossings.
    # Barriers apply to fleet crossings too, so "real edge" means army- OR
    # fleet-adjacent (mirrors the loader's own _build_mountain_edges check).
    assert len(graph.mountain_edges) == 84
    fleet_pairs = {
        frozenset((origin.province, neighbor.province))
        for origin, neighbors in graph.fleet_adjacency.items()
        for neighbor in neighbors
    }
    for edge in graph.mountain_edges:
        a, b = sorted(edge)
        assert graph.army_reachable(a, b) or frozenset((a, b)) in fleet_pairs, (
            f"barrier on non-edge {a}-{b}"
        )
    assert graph.is_mountain_crossing("HIT", "NEW")
    assert not graph.is_mountain_crossing("NEW", "HIT")
    assert graph.is_mountain_crossing("NEW", "SEW")
    assert graph.is_mountain_crossing("SEW", "NEW")
    assert not graph.is_mountain_crossing("GEL", "HIG")


def test_starting_positions_are_legal_and_complete(graph) -> None:  # type: ignore[no-untyped-def]
    assert len(graph.starting_units) == 10  # the ten powers
    total = 0
    strengths: dict[str, list[int]] = {}
    for power, specs in graph.starting_units.items():
        homes = graph.home_centers[power]
        for spec in specs:
            kind, location, strength = parse_unit_spec(spec)
            assert graph.is_legal_location(kind, location), spec
            assert location.province in homes, spec
            strengths.setdefault(power, []).append(strength)
            total += 1
    assert total == 40
    # Rulebook starting strengths: 3A(M) for Morgoth; the listed doubles.
    assert sorted(strengths["MORGOTH"]) == [1] * 8 + [3]
    assert sorted(strengths["FINGOLFIN"]) == [1, 1, 2]
    assert sorted(strengths["FINROD"]) == [1, 1, 1, 1, 1, 2]
    assert sorted(strengths["GONDOLIN"]) == [1, 1, 2]
    assert sorted(strengths["DORIATH"]) == [1, 1, 2]
    assert sorted(strengths["TELERI"]) == [1, 1, 2]
    assert sorted(strengths["FEANORIANS"]) == [1, 1, 1, 1, 2, 2]
    assert strengths["MEN"] == [1]
    assert sorted(strengths["NAUGRIM"]) == [1, 1, 1]
    assert sorted(strengths["OSSIRIAND"]) == [1, 1, 1]  # A(TGa), A(SOs), A(SOs)
    # Anfauglith starts exactly full: six armies in a capacity-6 Area.
    assert len([s for s in graph.starting_units["MORGOTH"] if s.endswith("ANF")]) == 6
    # Teleri's fleets stand on coastal home centers.
    assert graph.terrain("EGL") is Terrain.COASTAL
    assert graph.terrain("SFA") is Terrain.COASTAL


def test_unit_count_matches_supply_count_per_power(graph) -> None:  # type: ignore[no-untyped-def]
    """Balance every power's starting units against its home supply weight.

    Morgoth's strength-3 unit is the rulebook's one supply-free exception, so
    it is left out of the count. Layer 1 only records the imbalance; the
    exemption is enforced by the variant layer.
    """
    for power, specs in graph.starting_units.items():
        centers = sum(graph.province(c).supply_center_count for c in graph.home_centers[power])
        units = [Unit(power, *parse_unit_spec(spec)) for spec in specs]
        ordinary = [u for u in units if not (power == "MORGOTH" and u.strength == 3)]
        assert len(ordinary) == centers, power
