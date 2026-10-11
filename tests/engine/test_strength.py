"""Strength pipeline arithmetic and the standard ruleset's hook values (spec 2-3, 7)."""

from fractions import Fraction

from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import Bar, Convoy, Dislodge, Hold, Move, Support, parse_order
from entente.engine.rules import CutExceptionScope, Ruleset, StrengthRole
from entente.engine.rules.standard import STANDARD
from entente.engine.rules.strength import (
    cut_value,
    effective_support,
    pipeline_value,
    support_matches,
)
from entente.engine.state import Unit

ARMY, FLEET = UnitKind.ARMY, UnitKind.FLEET


def test_standard_satisfies_the_protocol() -> None:
    assert isinstance(STANDARD, Ruleset)
    assert STANDARD.name == "standard"
    assert STANDARD.power_symmetric
    assert STANDARD.order_registry == frozenset({Hold, Move, Support, Convoy})
    assert STANDARD.cut_exception_scope is CutExceptionScope.DIRECTED_PROVINCE


def test_base_strength_is_one_for_every_unit() -> None:
    assert STANDARD.base_strength(Unit("FRANCE", ARMY, Location("PAR"))) == Fraction(1)
    assert STANDARD.base_strength(Unit("TURKEY", FLEET, Location("ANK"))) == Fraction(1)
    # The unit's own ``strength`` is ignored: base == 1 (Lemma L2/L3 premise).
    assert STANDARD.base_strength(Unit("FRANCE", ARMY, Location("PAR"), 3)) == Fraction(1)


def test_standard_hooks_are_the_degenerate_column() -> None:
    graph = load_map("standard")
    unit = Unit("FRANCE", ARMY, Location("PAR"))
    order = parse_order("A PAR - BUR")
    for role in StrengthRole:
        assert STANDARD.modified_base(role, unit, order, graph) == Fraction(1)
    assert STANDARD.implicit_displacement(1)
    assert STANDARD.convoy_capability(
        Unit("FRANCE", FLEET, Location("ENG")), Unit("FRANCE", ARMY, Location("LON"))
    )
    assert STANDARD.entry_permission(unit, graph.province("BUR"))


def _support(text: str) -> Support:
    order = parse_order(text)
    assert isinstance(order, Support)
    return order


def test_support_matching_rules() -> None:
    support_move = _support("F CON S F BLA - ANK")
    assert support_matches(support_move, parse_order("F BLA - ANK"))
    assert not support_matches(support_move, parse_order("F BLA - RUM"))
    assert not support_matches(support_move, parse_order("F BLA H"))
    assert not support_matches(support_move, None)
    # A written destination coast must agree (DATC 6.B.9); omitted coast and
    # route flag are not part of matching (6.B.7/6.B.8).
    assert not support_matches(_support("F POR S F MAO - SPA/NC"), parse_order("F MAO - SPA/SC"))
    assert support_matches(_support("F POR S F MAO - SPA/NC"), parse_order("F MAO - SPA/NC"))
    assert support_matches(_support("F POR S F MAO - SPA"), parse_order("F MAO - SPA/SC"))
    assert support_matches(_support("F ENG S A LON - BEL"), parse_order("A LON - BEL via convoy"))
    # Support-hold matches any stationary order, never a relocation.
    support_hold = _support("A SMY S F ANK")
    assert support_matches(support_hold, parse_order("F ANK H"))
    assert support_matches(support_hold, parse_order("F ANK C A CON - SEV"))
    assert support_matches(support_hold, parse_order("F ANK S A SMY"))
    assert not support_matches(support_hold, parse_order("F ANK - CON"))


def test_variant_subject_matching() -> None:
    """Match a support to its subject by the victim reference the subject names.

    Support-dislodge and support-bar both compare victim references
    (docs/adjudication.tex Sec. 3.4), while a support-hold matches a Bar,
    because a Bar is stationary.
    """
    sup_dis = _support("A EST S A FOR DIS MORGOTH A NEB")
    actual = parse_order("A FOR DIS MORGOTH A NEB")
    assert isinstance(actual, Dislodge)
    assert support_matches(sup_dis, actual)
    assert not support_matches(sup_dis, parse_order("A FOR DIS OSSIRIAND A NEB"))
    assert not support_matches(sup_dis, parse_order("A FOR - NEB"))
    sup_bar = _support("A EST S 3A NEB BAR DORIATH 2A FOR - NEB")
    bar = parse_order("3A NEB BAR DORIATH 2A FOR - NEB")
    assert isinstance(bar, Bar)
    assert support_matches(sup_bar, bar)
    # The rulebook treats a designation without a prefix as base strength 1 --
    # "A" is the same as "1A" -- so a bare victim reference never describes the
    # 2A. A different owner never matches either.
    assert not support_matches(sup_bar, parse_order("3A NEB BAR DORIATH A FOR - NEB"))
    assert not support_matches(sup_bar, parse_order("3A NEB BAR OSSIRIAND 2A FOR - NEB"))
    assert support_matches(_support("A EST S MORGOTH A NEB"), bar)  # bars are stationary


def test_one_shot_cut_arithmetic_degenerates_to_binary_at_base_one() -> None:
    # Lemma L2: any qualifying cutter fully cuts a base-1 support...
    cutval = cut_value(Fraction(1), [])
    assert cutval == Fraction(1)
    assert effective_support(Fraction(1), cutval) == Fraction(0)
    # ...and cutters count their supports at full pre-cut value (Sec. 3.2).
    assert cut_value(Fraction(1), [Fraction(1)]) == Fraction(2)
    # No cut, full effectiveness; the floor never goes negative.
    assert effective_support(Fraction(1), Fraction(0)) == Fraction(1)
    assert effective_support(Fraction(1), Fraction(5)) == Fraction(0)
    assert STANDARD.cut_aggregation([Fraction(1), Fraction(2)]) == Fraction(3)


def test_strength_valued_cut_arithmetic() -> None:
    """Reproduce the rulebook's worked-example support arithmetic exactly.

    Cuts are counted at pre-cut values, supports aggregate by sum rather than
    by the strongest, and every quantity stays an exact fraction
    (docs/adjudication.tex Sec. 3.2).
    """
    # A(THA)-MMa with A(HIG)'s support: cutval 2 against 2A(MMa)'s support.
    assert cut_value(Fraction(1), [Fraction(1)]) == Fraction(2)
    assert effective_support(Fraction(2), Fraction(2)) == Fraction(0)
    # A(HID)-HIG with 2A(MMa)'s support: cutval 3 against A(HIG)'s support.
    assert cut_value(Fraction(1), [Fraction(2)]) == Fraction(3)
    assert effective_support(Fraction(1), Fraction(3)) == Fraction(0)
    # Mountain-halved supports keep exact rational values.
    assert effective_support(Fraction(3, 2), Fraction(1)) == Fraction(1, 2)


def test_pipeline_value_sums_delivered_supports() -> None:
    assert pipeline_value(Fraction(1), [Fraction(1), Fraction(1)]) == Fraction(3)
    assert pipeline_value(Fraction(1), []) == Fraction(1)
