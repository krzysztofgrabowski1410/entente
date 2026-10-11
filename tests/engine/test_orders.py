"""Order notation: parsing, canonical formatting, round-trips, static legality."""

import pytest

from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import (
    ADJUDICATION_ORDER_TYPES,
    RELOCATING_TYPES,
    STATIONARY_TYPES,
    Bar,
    Build,
    Convoy,
    Devastate,
    Disband,
    Dislodge,
    Hold,
    Move,
    OrderSyntaxError,
    Support,
    UnitRef,
    format_order,
    kind_reaches_province,
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
    parse_unit_spec,
    static_move_legal,
)
from entente.engine.rules.strength import is_relocating, is_stationary

ARMY, FLEET = UnitKind.ARMY, UnitKind.FLEET


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A PAR H", Hold(ARMY, Location("PAR"))),
        ("A PAR", Hold(ARMY, Location("PAR"))),
        ("a par hold".replace("hold", "h"), Hold(ARMY, Location("PAR"))),
        ("A PAR - BUR", Move(ARMY, Location("PAR"), Location("BUR"))),
        ("F STP/SC - BOT", Move(FLEET, Location("STP", "SC"), Location("BOT"))),
        (
            "A LON - HOL via convoy",
            Move(ARMY, Location("LON"), Location("HOL"), via_convoy=True),
        ),
        ("A SMY S F ANK", Support(ARMY, Location("SMY"), Hold(FLEET, Location("ANK")))),
        ("A SMY S F ANK H", Support(ARMY, Location("SMY"), Hold(FLEET, Location("ANK")))),
        (
            "F CON S F BLA - ANK",
            Support(FLEET, Location("CON"), Move(FLEET, Location("BLA"), Location("ANK"))),
        ),
        (
            "F NTH C A LON - HOL",
            Convoy(
                FLEET,
                Location("NTH"),
                Move(ARMY, Location("LON"), Location("HOL"), via_convoy=True),
            ),
        ),
    ],
)
def test_parse(text: str, expected: object) -> None:
    assert parse_order(text) == expected


def test_support_subject_via_flag_is_normalized_away() -> None:
    parsed = parse_order("F ENG S A LON - BEL via convoy")
    assert parsed == Support(FLEET, Location("ENG"), Move(ARMY, Location("LON"), Location("BEL")))


@pytest.mark.parametrize(
    "text",
    [
        "A PAR H",
        "A PAR - BUR",
        "A LON - HOL via convoy",
        "F MAO - SPA/NC",
        "A SMY S F ANK",
        "F CON S F BLA - ANK",
        "F POR S F MAO - SPA/NC",
        "F NTH C A LON - HOL",
    ],
)
def test_round_trip_is_canonical(text: str) -> None:
    order = parse_order(text)
    assert format_order(order) == text.replace("VIA CONVOY", "via convoy")
    assert parse_order(format_order(order)) == order


@pytest.mark.parametrize(
    "text",
    ["", "A", "Z PAR H", "A PAR X BUR", "A PAR - BUR via land", "F NTH C A LON", "A PAR H extra"],
)
def test_syntax_errors_raise(text: str) -> None:
    with pytest.raises(OrderSyntaxError):
        parse_order(text)


def test_parse_unit_spec() -> None:
    assert parse_unit_spec("F STP/SC") == (FLEET, Location("STP", "SC"), 1)
    assert parse_unit_spec("2A GEL") == (ARMY, Location("GEL"), 2)
    assert parse_unit_spec("3A ANG") == (ARMY, Location("ANG"), 3)
    with pytest.raises(OrderSyntaxError):
        parse_unit_spec("F")
    with pytest.raises(OrderSyntaxError):
        parse_unit_spec("1A PAR")  # a strength-1 prefix is never written


def test_variant_notation_round_trips() -> None:
    for text in [
        "2A GEL - THA",
        "3A ANG - ANF",
        "A FOR DIS MORGOTH A NEB",
        "2A MSI DIS OSSIRIAND A SEB",
        "A EST DIS MORGOTH A NEB S A GEL - THA",
        "A NEB BAR DORIATH A FOR - NEB",
        "2A TAL BAR DORIATH 2A FOB - TAL",
        "A EST S A FOR DIS MORGOTH A NEB",
        "A EST S 3A NEB BAR DORIATH 2A FOR - NEB",
        "A EST S MORGOTH A NEB",
        "2F BOB C 2A CBB - ARV",
    ]:
        order = parse_order(text)
        assert format_order(order) == text, text
        assert parse_order(format_order(order)) == order


def test_designator_parses_onto_the_right_field() -> None:
    # The (M)/(G) special-unit designator rides ``special`` on the ordering
    # unit, and a Dislodge victim's designator rides its ``UnitRef``, so a
    # designated order and a designated victim narrow different sides.
    assert parse_order("3A(M) ANF - LOT") == Move(
        ARMY, Location("ANF"), Location("LOT"), strength=3, special="M"
    )
    assert parse_order("3A(G) ANF DEV") == Devastate(ARMY, Location("ANF"), strength=3, special="G")
    dislodge = parse_order("A FOR DIS MORGOTH 3A(M) ANF")
    assert dislodge == Dislodge(
        ARMY, Location("FOR"), UnitRef("MORGOTH", ARMY, Location("ANF"), 3, "M")
    )
    assert dislodge.special is None  # the dislodger itself carries no designator
    support = parse_order("A EST S MORGOTH 3A(M) ANF")
    assert isinstance(support.subject, Hold) and support.subject.special == "M"


@pytest.mark.parametrize(
    "text",
    [
        "3A(M) ANF - LOT",
        "3A(G) ANF DEV",
        "3A(M) ANF H",
        "A(M) ANF - LOT",  # designator without a strength prefix
        "A FOR DIS MORGOTH 3A(M) ANF",
        "A FOR DIS MORGOTH 3A(G) ANF - NEB",
        "A EST S MORGOTH 3A(M) ANF",
        "2A NAR BAR MORGOTH 3A(G) FOR - NEB",
    ],
)
def test_designator_round_trips(text: str) -> None:
    order = parse_order(text)
    assert format_order(order) == text
    assert parse_order(format_order(order)) == order


def test_designator_is_case_insensitive_and_formats_canonical_upper() -> None:
    assert format_order(parse_order("3a(g) anf dev")) == "3A(G) ANF DEV"
    assert parse_order("3A(m) ANF - LOT").special == "M"


def test_retreat_and_adjustment_designators_round_trip() -> None:
    assert parse_retreat_order("3A(G) ANF - RIV") == Move(
        ARMY, Location("ANF"), Location("RIV"), strength=3, special="G"
    )
    assert parse_retreat_order("Disband 3A(M) ANF") == Disband(
        ARMY, Location("ANF"), 3, special="M"
    )
    assert parse_adjustment_order("Build 3A(M) ANF") == Build(ARMY, Location("ANF"), 3, special="M")
    for text in ("3A(G) ANF - RIV", "Disband 3A(M) ANF"):
        assert format_order(parse_retreat_order(text)) == text
    assert format_order(parse_adjustment_order("Build 3A(M) ANF")) == "Build 3A(M) ANF"


@pytest.mark.parametrize(
    "text",
    [
        "3A(X) ANF H",  # unknown designator letter
        "3A() ANF H",  # empty designator
        "3A(MM) ANF H",  # two letters
        "3A(m ANF H",  # unbalanced parenthesis
        "A NFA TFM F(M)",  # a transform target keeps its kind, takes no designator
    ],
)
def test_junk_designators_raise(text: str) -> None:
    with pytest.raises(OrderSyntaxError):
        parse_order(text)


def test_unit_spec_rejects_a_designator() -> None:
    # a board unit's identity lives in the ledger, never in its written spec
    with pytest.raises(OrderSyntaxError):
        parse_unit_spec("3A(M) ANF")


def test_static_move_legality() -> None:
    graph = load_map("standard")

    def move(text: str) -> Move:
        parsed = parse_order(text)
        assert isinstance(parsed, Move)
        return parsed

    assert static_move_legal(graph, move("A PAR - BUR"))
    assert not static_move_legal(graph, move("F NTH - PIC"))  # DATC 6.A.1
    assert not static_move_legal(graph, move("A LVP - IRI"))  # DATC 6.A.2
    assert not static_move_legal(graph, move("A PAR - PAR"))
    assert static_move_legal(graph, move("F MAO - SPA/NC"))
    assert not static_move_legal(graph, move("F GAS - SPA/SC"))
    assert static_move_legal(graph, move("A LON - HOL via convoy"))
    assert not static_move_legal(graph, move("F LON - HOL via convoy"))  # fleets don't convoy-ride
    assert not static_move_legal(graph, move("A PAR - LON via convoy"))  # inland origin


def test_kind_reaches_province_is_the_shared_province_level_geometry() -> None:
    """Share one province-level geometry between every order that targets an Area.

    ``static_move_legal``'s army branch and the adjudicator's Dislodge and
    support-target checks all ask the same question, so they ask it in one
    place. The predicate is province-level and coast-blind: a fleet asks
    whether some fleet location in the province is reachable, which is what an
    Area-targeting order needs and what an exact Move must not do.
    """
    graph = load_map("standard")
    assert kind_reaches_province(graph, ARMY, Location("PAR"), "BUR")
    assert not kind_reaches_province(graph, ARMY, Location("PAR"), "LON")
    assert not kind_reaches_province(graph, ARMY, Location("LVP"), "IRI")  # armies shun sea
    # Coast-blind where a Move would not be: MAO reaches Spain, naming no coast.
    assert kind_reaches_province(graph, FLEET, Location("MAO"), "SPA")
    assert kind_reaches_province(graph, FLEET, Location("GAS"), "SPA")
    assert not kind_reaches_province(graph, FLEET, Location("GAS"), "MOS")


def test_effect_classes_partition_the_adjudicator_grammar() -> None:
    """Partition every adjudicated order type between the two effect classes.

    Relocating and stationary cover the grammar exactly once each
    (docs/adjudication.tex Sec. 1). ``orders`` asserts it at import; this states
    it where a reader of the grammar will meet it. Both predicates are
    membership tests, so an unclassified new type would make both answer
    False, which is a wrong support-hold match rather than a crash.
    """
    assert set(RELOCATING_TYPES) | set(STATIONARY_TYPES) == set(ADJUDICATION_ORDER_TYPES)
    assert not set(RELOCATING_TYPES) & set(STATIONARY_TYPES)
    for order_type in ADJUDICATION_ORDER_TYPES:
        assert (order_type in RELOCATING_TYPES) != (order_type in STATIONARY_TYPES)
    # ... and the predicates agree with the tuples on real instances.
    victim = UnitRef("MORGOTH", ARMY, Location("NEB"))
    samples = [
        Hold(ARMY, Location("PAR")),
        Move(ARMY, Location("PAR"), Location("BUR")),
        Support(ARMY, Location("SMY"), Hold(FLEET, Location("ANK"))),
        Convoy(FLEET, Location("NTH"), Move(ARMY, Location("LON"), Location("HOL"), True)),
        Dislodge(ARMY, Location("FOR"), victim),
        Bar(ARMY, Location("NEB"), victim, Location("NEB")),
    ]
    assert {type(order) for order in samples} == set(ADJUDICATION_ORDER_TYPES)
    for order in samples:
        assert is_relocating(order) != is_stationary(order)
        assert is_relocating(order) == (type(order) in RELOCATING_TYPES)


def test_kind_token_letters_come_from_the_unit_kind_enum() -> None:
    """Derive the grammar's kind letters from :class:`UnitKind` itself.

    The parser's kind group is generated from the enum rather than written out
    beside it, so adding a unit kind cannot leave the two disagreeing: every
    enum member parses, and nothing outside it does.
    """
    for kind in UnitKind:
        parsed = parse_order(f"{kind.value} PAR H")
        assert isinstance(parsed, Hold)
        assert parsed.unit_kind is kind
    with pytest.raises(OrderSyntaxError):
        parse_order("Q PAR H")
