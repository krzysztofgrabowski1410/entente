"""Pin the gate/champion model for indistinguishable units at the engine level.

The rulebook's "Indistinguishable Units" section makes co-located units of one
power fungible, and the engine resolves them through a gate/champion model. The
case files report only the first duplicate row, so the observables that need
more than that live here, together with the corners where the rules meet:

* the converted second Bar of a same-power pair reports a failure while the
  standing first one succeeds (rule 3: *"changed into Hold Orders"*);
* every row of a barred wave fails, not just the first (rule 2: *"no Unit of
  the wave enters"*);
* an excess winner finds no twin left and fails, while twin attackers resolve
  at their own strength (rule 6);
* admission ranks wave members individually, on the strength of the move
  rather than on pure unit strength: a supported champion outranks its
  supportless siblings and takes a too-small door alone, while a wave whose
  members are all equal still bounces whole and a gate still stops the wave
  as one body;
* the static tables concentrate wave supports on the champion, redirect a
  converted bar's supports to the standing bar, and stay empty on the standard
  map, where every province has capacity 1 -- the by-construction vacuity that
  keeps the DATC results identical.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, adjudicate
from entente.engine.adjudicator.common import build_tables, phase0
from entente.engine.map import load_map
from entente.engine.orders import Order, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS, STANDARD
from entente.engine.state import State, Unit

BELERIAND = RULESETS["beleriand"]
GRAPH_BEL = load_map("beleriand")
GRAPH_STD = load_map("standard")


def _bel_state(units: Mapping[str, Sequence[str]]) -> State:
    return State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def _sheets(orders: Mapping[str, Sequence[str]]) -> dict[str, list[Order]]:
    return {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}


WAVE_UNITS = {
    "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANF"],
    "FEANORIANS": ["A LOT", "A LOT", "A LOT", "A LOT", "A ANF", "A LAD"],
}
WAVE_ORDERS = {
    "MORGOTH": [
        "A ANF - LOT",
        "A ANF - LOT",
        "A ANF S MORGOTH A ANF - LOT",
        "A ANF S MORGOTH A ANF - LOT",
    ],
    "FEANORIANS": [
        "A LOT BAR MORGOTH A ANF - LOT",
        "A LOT BAR MORGOTH A ANF - LOT",
        "A ANF S A LOT BAR MORGOTH A ANF - LOT",
        "A LAD S A LOT BAR MORGOTH A ANF - LOT",
        "A LOT H",
        "A LOT H",
    ],
}


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_converted_duplicate_bar_row_fails(engine: EngineName) -> None:
    # Rule 3: "Additional Barring Orders of the same Power against the same
    # wave do nothing additional and are changed into Hold Orders" -- the
    # k-th duplicate row discipline reports the standing first bar SUCCEEDS
    # (it stopped the wave) and the converted second FAILS.
    result = adjudicate(
        BELERIAND, _bel_state(WAVE_UNITS), _sheets(WAVE_ORDERS), graph=GRAPH_BEL, engine=engine
    )
    bar_text = "A LOT BAR MORGOTH A ANF - LOT"
    bar_rows = [row.resolution.value for row in result.resolutions if row.text == bar_text]
    assert bar_rows == ["succeeds", "fails"]


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_every_row_of_a_barred_wave_fails(engine: EngineName) -> None:
    # Rule 2: "if it does not, no Unit of the wave enters" -- all five rows,
    # not just the first the YAML's resolution_of can see (the pre-redesign
    # engine let rows 2-5 through unbarred).
    units = {
        "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANF", "A ANF"],
        "FEANORIANS": ["A LOT", "A LAD"],
    }
    orders = {
        "MORGOTH": ["A ANF - LOT"] * 5,
        "FEANORIANS": [
            "A LOT BAR MORGOTH A ANF - LOT",
            "A LAD S A LOT BAR MORGOTH A ANF - LOT",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LOT"]
    assert move_rows == ["fails"] * 5


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_excess_winner_finds_no_twin_left(engine: EngineName) -> None:
    # Attacks engage strongest-first, and an equal-strength cohort larger than
    # the remaining twins stands off: the 3A takes the first-listed twin, then
    # the two 2A attacks tie for the one twin left and both fail.
    units = {
        "MORGOTH": ["2A ANF", "2A ANF", "3A LAD"],
        "FEANORIANS": ["A LOT", "A LOT"],
    }
    orders = {
        "MORGOTH": [
            "2A ANF DIS FEANORIANS A LOT",
            "2A ANF DIS FEANORIANS A LOT",
            "3A LAD DIS FEANORIANS A LOT",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    twin_rows = [
        row.resolution.value
        for row in result.resolutions
        if row.text == "2A ANF DIS FEANORIANS A LOT"
    ]
    assert twin_rows == ["fails", "fails"]
    assert result.resolution_of("3A LAD DIS FEANORIANS A LOT").value == "succeeds"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_supported_champion_enters_alone_at_capacity_one(engine: EngineName) -> None:
    # Admission ranks wave members individually, on the strength of each move:
    # the champion at its concentrated prevent (1/2 + 1/2 over the penalised
    # ANF->LAD crossing), the sibling at its own supportless base (1/2). The
    # champion strictly outranks its twin and takes the one slot alone, which
    # is the classic supported-move picture. The first-listed mover is the
    # champion under rule 1's binding, so the first row succeeds.
    units = {"MORGOTH": ["A ANF", "A ANF", "A ANF"]}
    orders = {
        "MORGOTH": [
            "A ANF - LAD",
            "A ANF - LAD",
            "A ANF S MORGOTH A ANF - LAD",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LAD"]
    assert move_rows == ["succeeds", "fails"]
    placed = sorted(u.location.province for u in result.new_state.units)
    assert placed == ["ANF", "ANF", "LAD"]


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_unsupported_wave_still_bounces_at_capacity_one(engine: EngineName) -> None:
    # The ratified sentence stands ("Wave of 2+ of the same strength can't
    # enter capacity-1 province is true"): with no support the members rank
    # equal, form one class of two bodies at a one-slot door, and tie-bounce
    # exactly like the classic self-standoff.
    units = {"MORGOTH": ["A ANF", "A ANF"]}
    orders = {"MORGOTH": ["A ANF - LAD", "A ANF - LAD"]}
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LAD"]
    assert move_rows == ["fails", "fails"]
    assert all(u.location.province == "ANF" for u in result.new_state.units)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_champion_takes_the_last_area_slots_alone(engine: EngineName) -> None:
    # The member-ranked ladder at a multi-slot door: four Feanorian stayers
    # leave Lothlann two free slots; a supported champion (class 2) fits and
    # enters, its two supportless siblings (class {1, 1}, two bodies for the
    # one remaining slot) tie-overflow and bounce -- admission stops there.
    units = {
        "FEANORIANS": ["A LOT", "A LOT", "A LOT", "A LOT"],
        "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANF"],
    }
    orders = {
        "MORGOTH": [
            "A ANF - LOT",
            "A ANF - LOT",
            "A ANF - LOT",
            "A ANF S MORGOTH A ANF - LOT",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LOT"]
    assert move_rows == ["succeeds", "fails", "fails"]
    morgoth_in_lot = [
        u for u in result.new_state.units if u.power == "MORGOTH" and u.location.province == "LOT"
    ]
    assert len(morgoth_in_lot) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_unsupported_wave_still_blocks_a_tight_area_door(engine: EngineName) -> None:
    # Same door, no support: three equal members are one class of three
    # bodies against two free slots -- the class does not wholly fit, so the
    # whole wave bounces (the multi-slot face of the ratified sentence).
    units = {
        "FEANORIANS": ["A LOT", "A LOT", "A LOT", "A LOT"],
        "MORGOTH": ["A ANF", "A ANF", "A ANF"],
    }
    orders = {"MORGOTH": ["A ANF - LOT"] * 3}
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LOT"]
    assert move_rows == ["fails", "fails", "fails"]
    assert not any(
        u.power == "MORGOTH" and u.location.province == "LOT" for u in result.new_state.units
    )


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_stronger_lone_rival_still_beats_the_split_wave(engine: EngineName) -> None:
    # Member ranking never weakens the door contest: the Feanorian 3A ranks
    # 3/2 over the penalized ANF->LAD crossing, strictly above the Morgoth
    # champion's 1 (1/2 + 1/2), so it enters and every wave row fails.
    units = {
        "MORGOTH": ["A ANF", "A ANF", "A ANF"],
        "FEANORIANS": ["3A ANF"],
    }
    orders = {
        "MORGOTH": [
            "A ANF - LAD",
            "A ANF - LAD",
            "A ANF S MORGOTH A ANF - LAD",
        ],
        "FEANORIANS": ["3A ANF - LAD"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A ANF - LAD").value == "succeeds"
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LAD"]
    assert move_rows == ["fails", "fails"]


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_enemy_tie_with_the_champion_bounces_everyone(engine: EngineName) -> None:
    # The champion's concentrated prevent still guards the door for the
    # whole wave: a Feanorian 2A (rank 2 x 1/2 = 1) exactly ties the
    # supported champion (1), so the door stands off -- nobody enters, and
    # the sibling's low rank leaks nothing past the tie.
    units = {
        "MORGOTH": ["A ANF", "A ANF", "A ANF"],
        "FEANORIANS": ["2A ANF"],
    }
    orders = {
        "MORGOTH": [
            "A ANF - LAD",
            "A ANF - LAD",
            "A ANF S MORGOTH A ANF - LAD",
        ],
        "FEANORIANS": ["2A ANF - LAD"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF - LAD").value == "fails"
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LAD"]
    assert move_rows == ["fails", "fails"]
    assert not any(u.location.province == "LAD" for u in result.new_state.units)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_gate_still_stops_the_split_wave_at_one_free_slot(engine: EngineName) -> None:
    # Gate-fate stays wave-wide (rule 3: one Bar gates the whole wave):
    # with five Feanorian stayers Lothlann has one free slot, which the
    # supported champion would now take alone -- but the supported bar ties
    # its prevent (2 <= 2), and a tie favours the Bar, so no member enters.
    units = {
        "FEANORIANS": ["A LOT", "A LOT", "A LOT", "A LOT", "A LOT"],
        "MORGOTH": ["A ANF", "A ANF", "A ANF"],
    }
    orders = {
        "FEANORIANS": [
            "A LOT BAR MORGOTH A ANF - LOT",
            "A LOT S A LOT BAR MORGOTH A ANF - LOT",
        ],
        "MORGOTH": [
            "A ANF - LOT",
            "A ANF - LOT",
            "A ANF S MORGOTH A ANF - LOT",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    move_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - LOT"]
    assert move_rows == ["fails", "fails"]
    assert not any(
        u.power == "MORGOTH" and u.location.province == "LOT" for u in result.new_state.units
    )


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_barred_walk_attack_consumes_no_twin(engine: EngineName) -> None:
    # Bars gate dislodges, and that meets rule 6 here: a walk attack that would
    # beat the wall (2 > 1) but is barred (2 <= gate 2) fails wholly, so the
    # victim stands, the walk consumes no twin, and both twins survive. The bar
    # is the victims' own Area-mate defence, which is what carries the position
    # where support-holds fall short.
    units = {
        "MORGOTH": ["2A ANF"],
        "FEANORIANS": ["A LOT", "A LOT", "A LAD"],
    }
    orders = {
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
        "FEANORIANS": [
            "A LOT BAR MORGOTH 2A ANF - LOT",
            "A LAD S A LOT BAR MORGOTH 2A ANF - LOT",
            "A LOT H",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "fails"
    assert result.resolution_of("A LOT BAR MORGOTH 2A ANF - LOT").value == "succeeds"
    assert not result.dislodgements
    assert len([u for u in result.new_state.units if u.power == "FEANORIANS"]) == 3


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_mixed_class_reference_engages_only_the_primary_stack(engine: EngineName) -> None:
    # A bare-kind victim reference that wildcard-matches several fungibility
    # classes engages the class of the sheet-first match as the stack: the
    # co-located 2A is not a twin of the A pair, and *"a Dislodge which finds
    # no twin left fails"*, so the third winner fails and the 2A stands
    # untouched.
    units = {
        "MORGOTH": ["3A LAD", "2A ANF", "2A ANF"],
        "FEANORIANS": ["A LOT", "A LOT", "2A LOT"],
    }
    orders = {
        "MORGOTH": [
            "3A LAD DIS FEANORIANS A LOT",
            "2A ANF DIS FEANORIANS A LOT",
            "2A ANF DIS FEANORIANS A LOT",
        ],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A LAD DIS FEANORIANS A LOT").value == "succeeds"
    twin_rows = [
        row.resolution.value
        for row in result.resolutions
        if row.text == "2A ANF DIS FEANORIANS A LOT"
    ]
    # The 3A takes the front twin, then the two 2As tie for the one twin left
    # and stand off. That does not touch the scoping point: the co-located 2A
    # defender is not a twin of the A pair and is never a victim candidate.
    assert twin_rows == ["fails", "fails"]
    assert len(result.dislodgements) == 1
    feanorian_left = sorted(str(u) for u in result.new_state.units if u.power == "FEANORIANS")
    assert feanorian_left == ["2A LOT", "A LOT"]


def test_static_tables_concentrate_and_convert() -> None:
    # Rule 1/3 at the tables level: both wave supports sit on the champion
    # (the canonically first mover), both bar supports on the standing bar,
    # the converted duplicate is unbound, and wave_of maps both movers to
    # the champion.
    records = phase0(GRAPH_BEL, BELERIAND, _bel_state(WAVE_UNITS), _sheets(WAVE_ORDERS)).records
    tables = build_tables(GRAPH_BEL, BELERIAND, records)
    movers = sorted(
        r.index
        for r in records
        if r.power == "MORGOTH" and r.effective.__class__.__name__ == "Move"
    )
    champion = movers[0]
    assert tables.wave_of == {movers[0]: champion, movers[1]: champion}
    assert tables.wave_members == {champion: tuple(movers)}
    assert len(tables.supports_of.get(champion, ())) == 2
    assert movers[1] not in tables.supports_of
    bars = sorted(
        r.index
        for r in records
        if r.power == "FEANORIANS" and r.effective.__class__.__name__ == "Bar"
    )
    standing, converted = bars
    assert tables.converted_bars == frozenset({converted})
    assert tables.barred_move_of[standing] == champion
    assert tables.barred_move_of[converted] is None
    assert len(tables.supports_of.get(standing, ())) == 2
    assert tables.bars_on[champion] == (standing,)


def test_standard_map_never_builds_twin_machinery() -> None:
    # Capacity 1 everywhere: no stacks, no waves, no walks, no conversions --
    # the by-construction vacuity behind the DATC byte-identical gate.
    state = State.of(
        "standard",
        "S1901M",
        [
            Unit("FRANCE", *parse_unit_spec("A PAR")),
            Unit("FRANCE", *parse_unit_spec("A MAR")),
            Unit("GERMANY", *parse_unit_spec("A MUN")),
        ],
    )
    orders = {
        "FRANCE": ["A PAR - BUR", "A MAR - BUR"],
        "GERMANY": ["A MUN - BUR"],
    }
    records = phase0(GRAPH_STD, STANDARD, state, _sheets(orders)).records
    tables = build_tables(GRAPH_STD, STANDARD, records)
    assert not tables.wave_of
    assert not tables.wave_members
    assert not tables.stack_members_of
    assert not tables.wall_supports_of
    assert not tables.walk_groups
    assert not tables.converted_bars


# -- wall self-exclusion, the strength-first contest, and signatures --------


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_wall_excludes_the_givers_own_support(engine: EngineName) -> None:
    # A stack member defends at its own strength plus the wall's support-holds
    # minus the ones it gave itself: *"the support doesn't add strength to the
    # unit that does it"*. The giver listed first therefore defends at 1, not
    # 2, so an unnamed strength-2 attack engaging the front twin breaks it.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["2A ANF"],
    }
    orders = {
        "FEANORIANS": ["A LOT S A LOT H", "A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "succeeds"
    assert result.resolution_of("A LOT S A LOT H").value == "fails"  # dislodged giver
    assert result.resolution_of("A LOT H").value == "succeeds"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_unnamed_attack_cuts_the_wall_then_breaks_the_front(engine: EngineName) -> None:
    # An attack on a stack cuts a support on the stack's own list, and that
    # composes with the wall's self-exclusion: the unnamed attack cuts the
    # first-listed written support, which is the co-located giver's, the wall
    # dies, and the front holder falls to a strength-2 attack that a live wall
    # would have tied off.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["2A ANF"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT S A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "succeeds"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_uncuttable_outside_wall_spends_the_weak_attack(engine: EngineName) -> None:
    # An attack that engages a twin is spent on it, win or lose. The wall here
    # is supplied from outside the Area, so the stack attack cannot cut it: the
    # lone strength-2 attack engages the front twin, ties its defence of 2,
    # fails, and is spent, which leaves the twin behind safe.
    units = {
        "FEANORIANS": ["A LOT", "A LOT", "A LAD"],
        "MORGOTH": ["2A ANF"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT H", "A LAD S A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "fails"
    assert not result.dislodgements


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_strength_first_spreading_breaks_both_ranks(engine: EngineName) -> None:
    # Attacks sort strongest-first over the victim's list order: the Naugrim 3
    # breaks the front holder (defence 2), then the Morgoth 2 breaks the giver
    # behind it (defence 1), so both twins fall.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["2A ANF"],
        "NAUGRIM": ["3A MMA"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT S A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
        "NAUGRIM": ["3A MMA DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A MMA DIS FEANORIANS A LOT").value == "succeeds"
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "succeeds"
    assert len(result.dislodgements) == 2
    assert not any(u.power == "FEANORIANS" for u in result.new_state.units)


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_equal_attacks_spread_over_the_list(engine: EngineName) -> None:
    # An equal-strength cohort that fits the remaining twins spreads over them,
    # pairing the attackers' canonical order onto the victims' list order, and
    # it composes with the cut list: the first attack cuts the giver's on-list
    # support, both defences drop to 1, and both twins fall to the pair of 2s.
    # With an uncuttable wall from outside the Area the spread decides alone --
    # :func:`test_uncuttable_outside_wall_spends_the_weak_attack` is that face.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["2A ANF"],
        "NAUGRIM": ["2A MMA"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT S A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT"],
        "NAUGRIM": ["2A MMA DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT").value == "succeeds"
    assert result.resolution_of("2A MMA DIS FEANORIANS A LOT").value == "succeeds"
    assert len(result.dislodgements) == 2


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_tied_excess_cohort_stands_off(engine: EngineName) -> None:
    # A cohort that overflows the twins left over stands off: three winners
    # (5, 4, 4) meet two twins each defending 3. The 5 takes the front twin,
    # then the two 4s overflow the one twin left, stand off and both fail, so
    # one twin survives.
    units = {
        "FEANORIANS": ["A LOT", "A LOT", "2A LOT"],
        "MORGOTH": ["3A ANF", "2A ANF"],
        "NAUGRIM": ["4A MMA"],
        "FINGOLFIN": ["4A AEL"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT H", "2A LOT S A LOT"],
        "MORGOTH": [
            "3A ANF DIS FEANORIANS A LOT",
            "2A ANF S 3A ANF DIS FEANORIANS A LOT",
        ],
        "NAUGRIM": ["4A MMA DIS FEANORIANS A LOT"],
        "FINGOLFIN": ["4A AEL DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A ANF DIS FEANORIANS A LOT").value == "succeeds"
    assert result.resolution_of("4A MMA DIS FEANORIANS A LOT").value == "fails"
    assert result.resolution_of("4A AEL DIS FEANORIANS A LOT").value == "fails"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_lone_victim_tie_still_stands_off(engine: EngineName) -> None:
    # The overflow standoff holds at a single-member class too: two
    # strength-3 attacks on one unit bounce each other off and it stands.
    units = {
        "FEANORIANS": ["A LOT"],
        "MORGOTH": ["3A ANF"],
        "NAUGRIM": ["3A MMA"],
    }
    orders = {
        "FEANORIANS": ["A LOT H"],
        "MORGOTH": ["3A ANF DIS FEANORIANS A LOT"],
        "NAUGRIM": ["3A MMA DIS FEANORIANS A LOT"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A ANF DIS FEANORIANS A LOT").value == "fails"
    assert result.resolution_of("3A MMA DIS FEANORIANS A LOT").value == "fails"
    assert not result.dislodgements


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_named_attack_snipes_the_weak_giver(engine: EngineName) -> None:
    # Writing the victim's distinct order aims past the front rank -- *"a
    # Dislodge which names an action takes the twin so named"* -- and the
    # self-excluded wall does the rest: the strength-2 attack names the giver's
    # support order and breaks its defence of 1, while the front holder, at
    # defence 2, stands.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["2A ANF"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT S A LOT H"],
        "MORGOTH": ["2A ANF DIS FEANORIANS A LOT S A LOT H"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("2A ANF DIS FEANORIANS A LOT S A LOT H").value == "succeeds"
    assert result.resolution_of("A LOT H").value == "succeeds"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_rival_named_attacks_tie_bounces(engine: EngineName) -> None:
    # Several named attacks on the same twin resolve as a lone-victim contest,
    # so an exact tie at the top bounces and the named twin stands.
    units = {
        "FEANORIANS": ["A LOT", "A LOT"],
        "MORGOTH": ["3A ANF"],
        "NAUGRIM": ["3A MMA"],
    }
    orders = {
        "FEANORIANS": ["A LOT H", "A LOT S A LOT H"],
        "MORGOTH": ["3A ANF DIS FEANORIANS A LOT H"],
        "NAUGRIM": ["3A MMA DIS FEANORIANS A LOT H"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("3A ANF DIS FEANORIANS A LOT H").value == "fails"
    assert result.resolution_of("3A MMA DIS FEANORIANS A LOT H").value == "fails"
    assert not result.dislodgements


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_glaurung_shares_no_wall(engine: EngineName) -> None:
    # The designator is part of the fungibility signature, so a signed unit is
    # never a twin of an unsigned one: the support-hold on the plain 3A pools
    # no wall for Glaurung, whose bare defence of 3 falls to a strength-4
    # attack.
    units = {
        "MORGOTH": ["3A ANF", "3A ANF", "2A ANF"],
        "FEANORIANS": ["4A LAD"],
    }
    orders = {
        "MORGOTH": ["3A(G) ANF H", "3A ANF H", "2A ANF S 3A ANF H"],
        "FEANORIANS": ["4A LAD DIS MORGOTH 3A(G) ANF"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("4A LAD DIS MORGOTH 3A(G) ANF").value == "succeeds"
    assert result.resolution_of("3A ANF H").value == "succeeds"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_bare_victim_ref_walks_past_glaurung(engine: EngineName) -> None:
    # A bare reference names the undesignated class only, the way a bare "A"
    # means base strength 1, so a bare "3A" victim reference passes over Glaurung: the
    # walk takes the plain 3A even though Glaurung's row is listed first, and
    # the ``(G)`` row survives. The position is the adjudicator's alone: in a
    # game the Beleriand layer writes every special's designator, so no bare
    # 3A ever stands beside a 3A(G).
    units = {
        "MORGOTH": ["3A ANF", "3A ANF"],
        "FEANORIANS": ["4A LAD"],
    }
    orders = {
        "MORGOTH": ["3A(G) ANF H", "3A ANF H"],
        "FEANORIANS": ["4A LAD DIS MORGOTH 3A ANF"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("4A LAD DIS MORGOTH 3A ANF").value == "succeeds"
    assert result.resolution_of("3A(G) ANF H").value == "succeeds"
    assert result.resolution_of("3A ANF H").value == "fails"
    assert len(result.dislodgements) == 1


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_barred_walk_stack_with_split_doors_resolves(engine: EngineName) -> None:
    # A stack under an unnamed Dislodge is a walk group whose rule-6 contest
    # reads the whole pool as one function, so every order in that pool has to
    # land in one strongly connected component. The full dependency graph binds
    # them, but the resolver condenses the graph induced on the orders still
    # undecided, and there a bridging attack that decides during propagation
    # drops out -- which would leave twins bound for different capacity-1 doors
    # in separate residue sub-components. Evaluating one door's pair would then
    # read the walk engagement over the whole pool, reach the other door's
    # undecided move, and violate the resolver's rule that a crisp read never
    # touches an undecided order. ``residue_adjacency`` couples every walk
    # group's pool and attacks directly, so the induced residue keeps them
    # together and both engines agree.
    #
    # Five ANF twins: one Bars the sole incoming Dislodge (so the Dislodge
    # fails and decides during propagation), the other four split into two
    # capacity-1 bounce pairs (HIT-ANF, EIT). The bar holds, both pairs
    # self-standoff, nothing is dislodged, every twin stays home.
    units = {
        "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANF", "A ANF"],
        "FINROD": ["A DNN"],
    }
    orders = {
        "MORGOTH": [
            "A ANF BAR FINROD A DNN - ANF",
            "A ANF - HIT-ANF",
            "A ANF - EIT",
            "A ANF - HIT-ANF",
            "A ANF - EIT",
        ],
        "FINROD": ["A DNN DIS MORGOTH A ANF"],
    }
    result = adjudicate(
        BELERIAND, _bel_state(units), _sheets(orders), graph=GRAPH_BEL, engine=engine
    )
    assert result.resolution_of("A ANF BAR FINROD A DNN - ANF").value == "succeeds"
    assert result.resolution_of("A DNN DIS MORGOTH A ANF").value == "fails"
    hit_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - HIT-ANF"]
    eit_rows = [row.resolution.value for row in result.resolutions if row.text == "A ANF - EIT"]
    assert hit_rows == ["fails", "fails"]
    assert eit_rows == ["fails", "fails"]
    assert not result.dislodgements
    assert all(u.location.province == "ANF" for u in result.new_state.units if u.power == "MORGOTH")
