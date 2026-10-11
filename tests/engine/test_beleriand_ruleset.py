"""Pin the Beleriand column of the ruleset's hook values.

A ruleset is the engine's one place for per-variant behaviour, so every rule
Beleriand changes -- the barrier tolls, the extra order types, what each of
them enumerates -- is readable as a value here. The standard column is checked
alongside wherever a hook must leave it alone (docs/adjudication.tex Sec. 7).
"""

from fractions import Fraction

import pytest

from entente.engine.adjudicator import adjudicate
from entente.engine.game import Game
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import (
    Bar,
    Convoy,
    Dislodge,
    Hold,
    Move,
    Support,
    UnitRef,
    format_order,
    parse_order,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS, AdmissionPolicy, CutExceptionScope, Ruleset, StrengthRole
from entente.engine.rules.standard import STANDARD
from entente.engine.state import State, Unit
from entente.variants.beleriand import BELERIAND
from entente.variants.beleriand.layer2 import BeleriandHooks, DynamicBeleriandRuleset
from tests.engine.two_beach import CONVOYED_MOVE, TWO_BEACH

GRAPH = load_map("beleriand")
ARMY, FLEET = UnitKind.ARMY, UnitKind.FLEET


def test_beleriand_satisfies_the_protocol_and_registers() -> None:
    assert isinstance(BELERIAND, Ruleset)
    assert RULESETS["beleriand"] is BELERIAND
    assert not BELERIAND.power_symmetric
    assert BELERIAND.order_registry == frozenset({Hold, Move, Support, Convoy, Dislodge, Bar})
    assert BELERIAND.cut_exception_scope is CutExceptionScope.DIRECTED_UNIT
    # Both instantiations share the greedy admission rule -- the axis is data.
    assert BELERIAND.admission_policy is AdmissionPolicy.GREEDY_DESCENDING_CLASSES
    assert STANDARD.admission_policy is AdmissionPolicy.GREEDY_DESCENDING_CLASSES


def test_base_strength_is_the_units_own_strength() -> None:
    assert BELERIAND.base_strength(Unit("MORGOTH", ARMY, Location("ANG"), 3)) == Fraction(3)
    assert BELERIAND.base_strength(Unit("TELERI", FLEET, Location("SFA"), 2)) == Fraction(2)
    assert BELERIAND.base_strength(Unit("MEN", ARMY, Location("ELA"))) == Fraction(1)


def test_modified_base_mountains_and_fortresses() -> None:
    unit = Unit("FINGOLFIN", ARMY, Location("DLN"), 2)
    crossing = parse_order("2A DLN - SEW")  # DLN->SEW is a penalized crossing
    flat = parse_order("2A DLN - MIT")  # DLN-MIT carries no barrier either way
    assert BELERIAND.modified_base(StrengthRole.MOVE, unit, crossing, GRAPH) == Fraction(1)
    assert BELERIAND.modified_base(StrengthRole.MOVE, unit, flat, GRAPH) == Fraction(2)
    # The same edge in reverse is free, because a barrier is directional:
    # descending out of the Ered Wethrin costs nothing, so 2 stays 2.
    descending = Unit("FINGOLFIN", ARMY, Location("SEW"), 2)
    assert BELERIAND.modified_base(
        StrengthRole.MOVE, descending, parse_order("2A SEW - DLN"), GRAPH
    ) == Fraction(2)

    # A support pays the toll of the crossing from the supporter's own province
    # to the subject's target: from the range province SEW, supporting outward
    # into DLN is full (1),
    # while the DLN supporter helping a move up into SEW is halved (1/2).
    outward = Unit("FINGOLFIN", ARMY, Location("SEW"))
    assert BELERIAND.modified_base(
        StrengthRole.SUPPORT, outward, parse_order("A SEW S 2A IVR - DLN"), GRAPH
    ) == Fraction(1)
    inward = Unit("FINGOLFIN", ARMY, Location("DLN"))
    assert BELERIAND.modified_base(
        StrengthRole.SUPPORT, inward, parse_order("A DLN S 2A IVR - SEW"), GRAPH
    ) == Fraction(1, 2)

    supporter = Unit("NAUGRIM", ARMY, Location("BEL"))
    through = parse_order("A BEL S MEN 2A ELA - NEL")  # BEL->NEL is penalized
    assert BELERIAND.modified_base(StrengthRole.SUPPORT, supporter, through, GRAPH) == Fraction(
        1, 2
    )
    sup_dis = parse_order("A EST S A FOR DIS MORGOTH A NEB")  # EST-NEB is not
    est = Unit("DORIATH", ARMY, Location("EST"))
    assert BELERIAND.modified_base(StrengthRole.SUPPORT, est, sup_dis, GRAPH) == Fraction(1)
    sup_bar = parse_order("A EST S MORGOTH A NEB BAR DORIATH A FOR - NEB")
    assert BELERIAND.modified_base(StrengthRole.SUPPORT, est, sup_bar, GRAPH) == Fraction(1)
    sup_hold = parse_order("A EST S MORGOTH A NEB")
    assert BELERIAND.modified_base(StrengthRole.SUPPORT, est, sup_hold, GRAPH) == Fraction(1)

    garrison = Unit("FINROD", ARMY, Location("TOL"))
    holding = parse_order("A TOL H")
    moving = parse_order("A TOL - TAL")  # a real, barrier-free edge (TOL-FOB is gone)
    targeting = parse_order("A TOL DIS MORGOTH A TAL")
    assert BELERIAND.modified_base(StrengthRole.HOLD, garrison, holding, GRAPH) == Fraction(2)
    assert BELERIAND.modified_base(StrengthRole.HOLD, garrison, moving, GRAPH) == Fraction(1)
    assert BELERIAND.modified_base(StrengthRole.HOLD, garrison, targeting, GRAPH) == Fraction(1)
    plain = Unit("DORIATH", ARMY, Location("EST"))
    assert BELERIAND.modified_base(StrengthRole.HOLD, plain, parse_order("A EST H"), GRAPH) == (
        Fraction(1)
    )
    # Non-projecting stationary orders take no edge modifier in any role.
    convoying = Unit("TELERI", FLEET, Location("BOB"), 2)
    convoy = parse_order("2F BOB C 2A CBB - ARV")
    assert BELERIAND.modified_base(StrengthRole.MOVE, convoying, convoy, GRAPH) == Fraction(2)


def test_convoyed_moves_toll_the_disembark_crossing() -> None:
    """Toll a convoyed move on the seas its convoy can put it ashore from.

    A convoyed army never walks out of its origin, so the barrier it pays is
    the one between the last convoying sea and the landing province -- and
    which sea that can be is the convoy's to say, not the coast's: the hook is
    handed the seas the move's own convoy orders reach the landing through.
    """
    nbe = frozenset({"NBE"})
    army = Unit("FINROD", ARMY, Location("VIN"), 2)
    by_sea = parse_order("2A VIN - NEV via convoy")  # NBE->NEV is penalized
    overland = parse_order("2A VIN - NEV")  # the VIN-NEV land edge is not
    assert BELERIAND.modified_base(
        StrengthRole.MOVE, army, by_sea, GRAPH, disembark=nbe
    ) == Fraction(1)
    assert BELERIAND.modified_base(StrengthRole.MOVE, army, overland, GRAPH) == Fraction(2)
    # A free beach lands whole: NBE->LAM is not in the table, so the convoy
    # pays nothing regardless of the origin's own edges.
    sailor = Unit("FINROD", ARMY, Location("ELO"), 2)
    free_beach = parse_order("2A ELO - LAM via convoy")
    assert BELERIAND.modified_base(
        StrengthRole.MOVE, sailor, free_beach, GRAPH, disembark=nbe
    ) == Fraction(2)

    # A coast with one marked sea and one free: only the seas handed over are
    # read. Landing over the cliff alone pays; one free beach among the
    # ordered ones spares the army, and the coast's other sea is never asked.
    crosser = Unit("RED", ARMY, Location("ORI"), 2)
    crossing = parse_order(CONVOYED_MOVE)
    for seas, expected in (
        (frozenset({"SEC"}), Fraction(1)),
        (frozenset({"SED"}), Fraction(2)),
        (frozenset({"SEC", "SED"}), Fraction(2)),
        (frozenset(), Fraction(2)),  # no chain ordered: never viable, never read
    ):
        assert (
            BELERIAND.modified_base(StrengthRole.MOVE, crosser, crossing, TWO_BEACH, disembark=seas)
            == expected
        )

    # A convoyed move without its seas is a caller's error: falling back on
    # the landing coast would answer a question the rulebook does not ask.
    with pytest.raises(ValueError, match="ordered disembark seas"):
        BELERIAND.modified_base(StrengthRole.MOVE, army, by_sea, GRAPH)
    # The seas concern convoyed moves alone; every other order ignores them.
    assert BELERIAND.modified_base(
        StrengthRole.MOVE, army, overland, GRAPH, disembark=nbe
    ) == Fraction(2)
    # The standard ruleset is handed the same seas and pays no toll.
    assert STANDARD.modified_base(
        StrengthRole.MOVE, crosser, crossing, TWO_BEACH, disembark=frozenset({"SEC"})
    ) == Fraction(1)


def test_the_game_layer_delegate_hands_the_disembark_seas_through() -> None:
    wrapper = DynamicBeleriandRuleset(BELERIAND, BeleriandHooks(GRAPH, seed=0))
    army = Unit("FINROD", ARMY, Location("VIN"), 2)
    by_sea = parse_order("2A VIN - NEV via convoy")
    assert wrapper.modified_base(
        StrengthRole.MOVE, army, by_sea, GRAPH, disembark=frozenset({"NBE"})
    ) == Fraction(1)
    with pytest.raises(ValueError, match="ordered disembark seas"):
        wrapper.modified_base(StrengthRole.MOVE, army, by_sea, GRAPH)


def test_implicit_displacement_is_capacity_derived() -> None:
    assert BELERIAND.implicit_displacement(1)
    assert not BELERIAND.implicit_displacement(6)
    assert not BELERIAND.implicit_displacement(None)
    assert STANDARD.implicit_displacement(1)


def test_convoy_capability_needs_matching_strength() -> None:
    single = Unit("TELERI", FLEET, Location("SBE"))
    double = Unit("TELERI", FLEET, Location("SBE"), 2)
    army = Unit("TELERI", ARMY, Location("SFA"), 2)
    assert not BELERIAND.convoy_capability(single, army)
    assert BELERIAND.convoy_capability(double, army)
    assert BELERIAND.convoy_capability(single, Unit("TELERI", ARMY, Location("SFA")))


def test_entry_permission_matrix() -> None:
    angband = GRAPH.province("ANG")
    eriador = GRAPH.province("ERI")
    ordinary = GRAPH.province("EST")
    assert BELERIAND.entry_permission(Unit("MORGOTH", ARMY, Location("ANF"), 3), angband)
    assert not BELERIAND.entry_permission(Unit("MORGOTH", ARMY, Location("ANF")), angband)
    assert not BELERIAND.entry_permission(Unit("FINROD", ARMY, Location("ANF")), angband)
    assert not BELERIAND.entry_permission(Unit("MEN", ARMY, Location("ELA")), eriador)
    assert not BELERIAND.entry_permission(Unit("NAUGRIM", ARMY, Location("ELA")), eriador)
    assert BELERIAND.entry_permission(Unit("MEN", ARMY, Location("ELA")), ordinary)


def test_area_control_supply_ownership() -> None:
    """Award an Area's supply centre by the rulebook's two-unit control margin.

    An Area changes hands only for a Power holding *"two more units than any
    other Power"* in it. A sole occupant, and any capacity-1 province, reduce
    to the standard rule, where simply standing there is enough.
    """
    units = [
        # NEB (a supply-center Area): 3 Morgoth vs 1 Fingolfin -- controlled.
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("FINGOLFIN", ARMY, Location("NEB")),
        # ANF: 2 vs 1 -- margin below two, no owner change.
        Unit("DORIATH", ARMY, Location("ANF")),
        Unit("DORIATH", ARMY, Location("ANF")),
        Unit("MORGOTH", ARMY, Location("ANF")),
        # GVA (capacity 1): sole occupant claims it.
        Unit("OSSIRIAND", ARMY, Location("GVA")),
        # MOS (capacity-6 Area, not a supply center) -- irrelevant to ownership.
        Unit("MEN", ARMY, Location("MOS")),
    ]
    state = State.of("beleriand", "F25R", units)
    before = {"NEB": "FINGOLFIN", "ANF": "MORGOTH"}
    after = BELERIAND.update_supply_ownership(GRAPH, state, before)
    assert after["NEB"] == "MORGOTH"
    assert after["ANF"] == "MORGOTH"  # unchanged: margin of one is not control
    assert after["GVA"] == "OSSIRIAND"
    assert "MOS" not in after


def _game(units: list[Unit], phase: str = "S25M") -> Game:
    state = State.of("beleriand", phase, units)
    ownership = {"ANG": "MORGOTH"}
    return Game(BELERIAND, GRAPH, state, ownership, Phase(25, Season.SPRING, PhaseKind.MOVEMENT))


def test_legal_orders_enumerate_dislodges_and_bars() -> None:
    units = [
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB"), 2),
        Unit("TELERI", FLEET, Location("SBE"), 2),
        Unit("FINROD", FLEET, Location("NBE")),
    ]
    game = _game(units)
    doriath = game.legal_orders("DORIATH")
    dislodges = [o for o in doriath if isinstance(o, Dislodge)]
    assert (
        Dislodge(ARMY, Location("FOR"), UnitRef("MORGOTH", ARMY, Location("NEB"), None))
        in dislodges
    )
    assert (
        Dislodge(ARMY, Location("FOR"), UnitRef("MORGOTH", ARMY, Location("NEB"), 2)) in dislodges
    )
    # The Doriath army inside the Area may bar the outside foreign units...
    bars = [o for o in doriath if isinstance(o, Bar)]
    assert any(o.victim.power == "MORGOTH" for o in bars) is False  # Morgoth is inside, not out
    # ...and Morgoth's stack can bar Doriath's approach from FOR.
    morgoth = game.legal_orders("MORGOTH")
    morgoth_bars = [o for o in morgoth if isinstance(o, Bar)]
    assert (
        Bar(ARMY, Location("NEB"), UnitRef("DORIATH", ARMY, Location("FOR"), None), Location("NEB"))
        in morgoth_bars
    )
    # Fleets dislodge in sea Areas.
    finrod = game.legal_orders("FINROD")
    assert Dislodge(FLEET, Location("NBE"), UnitRef("TELERI", FLEET, Location("SBE"), 2)) in [
        o for o in finrod if isinstance(o, Dislodge)
    ]
    # No enumerated move enters Angband or Eriador illegally.
    for power in game.powers:
        for order in game.legal_orders(power):
            if isinstance(order, Move):
                assert order.destination.province not in ("ANG", "ERI")


def test_legal_orders_enumerate_victim_order_dislodges() -> None:
    """Enumerate one Dislodge per action the victim class could be cutting.

    A Dislodge may name the victim's action in order to cut one specific
    support, which is the rulebook's "Cutting Support in an Area". The
    enumeration therefore offers one variant per support the victim class could
    give, on top of the generic unnamed Dislodge, and offers the generic one
    first so a plain board click keeps naming it. The position is the
    rulebook's own example.
    """
    units = [
        Unit("DORIATH", ARMY, Location("EST")),
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("GEL")),
        Unit("FEANORIANS", ARMY, Location("THA")),
    ]
    game = _game(units)
    dislodges = [o for o in game.legal_orders("DORIATH") if isinstance(o, Dislodge)]
    generic = [o for o in dislodges if o.victim_order is None]
    victim_order = [o for o in dislodges if o.victim_order is not None]
    # the generic dislodge is still enumerated first -- the board-click order
    ref = UnitRef("MORGOTH", ARMY, Location("NEB"), None)
    assert Dislodge(ARMY, Location("EST"), ref) in generic
    assert dislodges.index(generic[0]) < dislodges.index(victim_order[0])
    # Every victim-order variant names either the victim class's plain Hold or
    # a Support it could give, with its unit spec normalised to the
    # province-level victim reference so the tail round-trips exactly. The Hold
    # is a real tactical choice and not a filler: a stack's wall excludes its
    # own giver, so a holding twin defends differently from its supporting
    # mates.
    assert victim_order
    for o in victim_order:
        assert isinstance(o.victim_order, (Hold, Support))
        assert o.victim_order.unit_kind is ref.kind
        assert o.victim_order.location == ref.location  # province-level, coast-free
        assert o.victim_order.strength == ref.strength
    hold_tails = [o for o in victim_order if isinstance(o.victim_order, Hold)]
    assert len(hold_tails) == 1  # one victim class -> exactly one Hold tail
    assert format_order(hold_tails[0]) == "A EST DIS MORGOTH A NEB H"
    assert victim_order[0] == hold_tails[0]  # the Hold tail leads the class
    # the rulebook example is offered verbatim in canonical form (BATC.C.10:
    # "A(Est) dis Morgoth's A(NEB) sup A(Gel)-Tha"; GEL uncrowded -> bare owner)
    example = Dislodge(
        ARMY,
        Location("EST"),
        ref,
        Support(ARMY, Location("NEB"), Move(ARMY, Location("GEL"), Location("THA"))),
    )
    assert example in victim_order
    assert format_order(example) == "A EST DIS MORGOTH A NEB S A GEL - THA"
    # and that exact enumerated order cuts the named support through the engine
    # (the bare-owner form BATC.C.10 does not itself submit): NEB's support of
    # GEL-THA is cut, so the unsupported move bounces off the THA defender
    result = adjudicate(
        BELERIAND,
        State.of("beleriand", "S25M", units),
        {
            "DORIATH": [example],
            "MORGOTH": [parse_order("A NEB S MORGOTH A GEL - THA"), parse_order("A GEL - THA")],
        },
        graph=GRAPH,
    )
    assert result.resolution_of("A NEB S MORGOTH A GEL - THA").value == "fails"  # cut
    assert result.resolution_of("A GEL - THA").value == "fails"  # unsupported, bounces


def test_legal_orders_enumerate_dislodge_supports() -> None:
    """Enumerate a support of another unit's Dislodge, aimed at the victim's Area.

    A unit may support a Dislodge, and the support targets the victim's Area,
    which the supporter must reach by its own adjacency the way
    ``common._validate_support`` requires; entry permission is not a
    supporter-side gate. The subject stays generic, because support matching
    compares victim references only (docs/adjudication.tex Sec. 1), so a
    victim-order tail would enumerate a row the adjudicator cannot tell from
    the generic one. The position and sheet are the rulebook's own
    supported-dislodge example.
    """
    units = [
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("EST")),
        Unit("MORGOTH", ARMY, Location("NEB")),
    ]
    game = _game(units)
    doriath = game.legal_orders("DORIATH")
    s_of_dis = [o for o in doriath if isinstance(o, Support) and isinstance(o.subject, Dislodge)]
    example = Support(
        ARMY,
        Location("EST"),
        Dislodge(ARMY, Location("FOR"), UnitRef("MORGOTH", ARMY, Location("NEB"), None)),
    )
    assert example in s_of_dis
    assert format_order(example) == "A EST S A FOR DIS MORGOTH A NEB"  # BATC.C.2 verbatim
    for offered in s_of_dis:
        assert isinstance(offered.subject, Dislodge)
        assert offered.subject.victim_order is None  # generic subjects only
    # the offered pair adjudicates to the rulebook outcome end to end: the
    # supported dislodge succeeds and the victim's slot is freed (BATC.C.2)
    dislodge = next(
        o
        for o in doriath
        if isinstance(o, Dislodge) and o.location.province == "FOR" and o.victim_order is None
    )
    result = adjudicate(
        BELERIAND,
        State.of("beleriand", "S25M", units),
        {"DORIATH": [dislodge, example]},
        graph=GRAPH,
    )
    assert result.resolution_of("A FOR DIS MORGOTH A NEB").value == "succeeds"
    assert result.resolution_of("A EST S A FOR DIS MORGOTH A NEB").value == "succeeds"
    assert [d.unit.power for d in result.dislodgements] == ["MORGOTH"]


def test_legal_orders_enumerate_bar_supports() -> None:
    """Enumerate a support of a Bar, aimed at the barring unit's own Area.

    The rulebook says *"a Unit in adjacent Province can Support a Barring
    Order"*, so this support targets the Area the barring unit stands in, and
    it is written with the bare owner where the Area holds only one candidate.
    The position is the one where a supported Bar holds the line against a
    supported move, with the offered text standing in for the rulebook's sheet.
    """
    units = [
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("THA")),
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("EST")),
    ]
    game = _game(units)
    morgoth = game.legal_orders("MORGOTH")
    s_of_bar = [o for o in morgoth if isinstance(o, Support) and isinstance(o.subject, Bar)]
    example = Support(
        ARMY,
        Location("THA"),
        Bar(
            ARMY, Location("NEB"), UnitRef("DORIATH", ARMY, Location("FOR"), None), Location("NEB")
        ),
    )
    assert example in s_of_bar
    assert format_order(example) == "A THA S A NEB BAR DORIATH A FOR - NEB"
    bar = next(
        o
        for o in morgoth
        if isinstance(o, Bar) and o.victim == UnitRef("DORIATH", ARMY, Location("FOR"), None)
    )
    result = adjudicate(
        BELERIAND,
        State.of("beleriand", "S25M", units),
        {
            "MORGOTH": [bar, example],
            "DORIATH": [parse_order("A FOR - NEB"), parse_order("A EST S A FOR - NEB")],
        },
        graph=GRAPH,
    )
    assert result.resolution_of("A NEB BAR DORIATH A FOR - NEB").value == "succeeds"
    assert result.resolution_of("A THA S A NEB BAR DORIATH A FOR - NEB").value == "succeeds"
    assert result.resolution_of("A FOR - NEB").value == "fails"
    assert not result.dislodgements


def test_targeted_support_reach_gates() -> None:
    """Gate each targeted support on what its own supporter can reach.

    A support-dislodge is offered where the supporter reaches the victim's
    Area, or stands in it. A support-bar is offered from adjacent provinces and
    from Area-mates inside the barring Area itself, because an Area-mate may
    support a co-located order, but never for a unit's own bar. An Area-mate of
    the dislodger is offered the support, since its target lies elsewhere.
    """
    # MEN reaches only FON/FOR, never NEB: hold/move supports enumerate from
    # MEN, support-dislodge rows do not
    units = [
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("MEN")),
        Unit("MORGOTH", ARMY, Location("NEB")),
    ]
    men_orders = [
        o
        for o in _game(units).legal_orders("DORIATH")
        if isinstance(o, Support) and o.location.province == "MEN"
    ]
    assert men_orders
    assert not any(isinstance(o.subject, Dislodge) for o in men_orders)

    # support-bar rows come from adjacent THA (one per barring class) AND from
    # each NEB Area-mate for the other class's bar, never for its own
    units = [
        Unit("MORGOTH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB"), 2),
        Unit("MORGOTH", ARMY, Location("THA")),
        Unit("DORIATH", ARMY, Location("FOR")),
    ]
    bar_supports = [
        o
        for o in _game(units).legal_orders("MORGOTH")
        if isinstance(o, Support) and isinstance(o.subject, Bar)
    ]
    assert bar_supports
    assert {o.location.province for o in bar_supports} == {"THA", "NEB"}
    from_tha = [o for o in bar_supports if o.location.province == "THA"]
    assert {o.subject.strength for o in from_tha} == {None, 2}  # both classes' bars
    from_neb = [o for o in bar_supports if o.location.province == "NEB"]
    # each Area-mate backs the other class's bar, never its own: the single
    # (supporter strength, subject strength) pairs are (1, 2) and (2, 1)
    assert {(o.strength, o.subject.strength) for o in from_neb} == {(None, 2), (2, None)}

    # the co-located dislodger's support IS offered: MOS and SOS are adjacent
    # Areas, so the Area-mate reaches the victim's Area while never reaching
    # its own (crowded subject -> owner and strength prefix written)
    units = [
        Unit("OSSIRIAND", ARMY, Location("MOS")),
        Unit("OSSIRIAND", ARMY, Location("MOS"), 2),
        Unit("MORGOTH", ARMY, Location("SOS")),
    ]
    example = Support(
        ARMY,
        Location("MOS"),
        Dislodge(
            ARMY, Location("MOS"), UnitRef("MORGOTH", ARMY, Location("SOS"), None), strength=2
        ),
        subject_power="OSSIRIAND",
    )
    assert example in _game(units).legal_orders("OSSIRIAND")
    assert format_order(example) == "A MOS S OSSIRIAND 2A MOS DIS MORGOTH A SOS"

    # a unit inside an Area is offered the orders aimed at that Area, whose
    # subjects both stand outside it: the Move entering NEB and the Dislodge
    # against the unit beside it. The supporter reaches neither target.
    units = [
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("NEB")),
        Unit("MORGOTH", ARMY, Location("NEB")),
    ]
    from_neb = {
        format_order(o)
        for o in _game(units).legal_orders("DORIATH")
        if isinstance(o, Support) and o.location.province == "NEB"
    }
    assert "A NEB S A FOR - NEB" in from_neb
    assert "A NEB S A FOR DIS MORGOTH A NEB" in from_neb


def test_standard_supports_stay_hold_move_only() -> None:
    """Keep standard supports to Hold and Move subjects only.

    Neither Dislodge nor Bar is in the standard registry, so the widened
    enumeration walk never runs on that column and its vocabulary cannot
    move. The masking and vocabulary suites check the exact token set; this
    guards the shape at the source.
    """
    game = Game.new("standard", "standard")
    for power in game.powers:
        for order in game.legal_orders(power):
            if isinstance(order, Support):
                assert isinstance(order.subject, (Hold, Move))


def test_enumerated_variant_orders_survive_a_round_trip_and_adjudicate() -> None:
    units = [
        Unit("DORIATH", ARMY, Location("FOR")),
        Unit("DORIATH", ARMY, Location("EST")),
        Unit("MORGOTH", ARMY, Location("NEB")),
    ]
    game = _game(units)
    orders = game.legal_orders("DORIATH")
    dislodge = next(o for o in orders if isinstance(o, Dislodge))
    game.submit("DORIATH", [dislodge])
    record = game.adjudicate()
    assert record.phase == "S25M"


def test_winner_counts_multi_center_provinces() -> None:
    game = _game([Unit("MORGOTH", ARMY, Location("ANF"))])
    assert game.supply_count("MORGOTH") == 7  # Angband alone weighs seven
    assert game.winner is None


@pytest.mark.parametrize("name", ["standard", "beleriand"])
def test_rulesets_share_the_backup_policy_pins(name: str) -> None:
    from entente.engine.rules.protocol import BackupAction, ResidueClass

    ruleset = RULESETS[name]
    assert ruleset.backup_policy(ResidueClass.PURE_CYCLE) is BackupAction.ALL_MOVES_SUCCEED
    assert ruleset.backup_policy(ResidueClass.CONVOY_ENTANGLED) is BackupAction.SZYKMAN
