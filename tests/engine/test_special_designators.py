"""The special units always carry their designator: the machinery under the law.

The rulebook's Signature paragraph: *"The designator must always be written:
3A(M) fits only Morgoth and 3A(G) only Glaurung, while 3A written without a
designator fits no Unit -- an Order or a reference written so is invalid, like
one with a wrong Unit type. Morgoth and Glaurung are therefore never twins."*
The BATC-L2 section S cases pin that law order by order through the game; this
file pins what carries it. Phase 0 keeps a special's designator on the Hold an
invalid order degrades to and the Move validator keeps it on a valid Move, so
two designated specials share no wall and a bare reference binds neither of
them, on both engines; and a reference to one special is no reference to the
other, so a support or a victim-action tail naming the other special matches
nothing. The game layer hands the adjudicator a designated Hold
for every special without a valid order of its own, drops that row from the
report, and lets a Layer-2 main order bind the record so held; over seeded
positions whose sheets name every 3A bare, every unit of base strength 3
reaches the adjudicator designated, which is why matching a designator
exactly both ways changes nothing in a game. Every list the
engine offers writes the designators -- in the opening, over seeded positions,
in the retreat phase, and for both specials in one Area -- and the standard
opening enumeration is byte-identical. Every phase record keeps the specials
as its phase left them, since a board holds only values and could not say
afterwards which unit was which.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import pytest
from hypothesis import HealthCheck, assume, given, settings

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, adjudicate
from entente.engine.adjudicator.common import Resolution, phase0
from entente.engine.game import Game
from entente.engine.game_hooks import SpecialUnit
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import (
    Hold,
    Move,
    Order,
    format_order,
    parse_order,
    parse_retreat_order,
    parse_unit_spec,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from tests.batc.l2_runner import L2Case, build_game
from tests.properties.strategies import beleriand_position_and_sheets

BELERIAND = RULESETS["beleriand"]
GRAPH = load_map("beleriand")
SPRING = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
SUPPRESS = (HealthCheck.too_slow, HealthCheck.data_too_large)
GLAURUNG_AT_ANF = {"glaurung": "ANF"}
STANDARD_OPENING_DIGEST = "be0ac5faa2775182f9c8aac94e6e6c88bede1024015a8d63af318ee78f83a214"


def _state(units: Mapping[str, Sequence[str]]) -> State:
    return State.of(
        "beleriand",
        SPRING.tag,
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


def _sheets(orders: Mapping[str, Sequence[str]]) -> dict[str, list[Order]]:
    return {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}


def _game(units: Mapping[str, Sequence[str]], l2: Mapping[str, object] | None = None) -> Game:
    """Build a Beleriand game at Spring 25 from hand-placed units and Layer-2 overrides."""
    start: dict[str, object] = {"phase": SPRING.tag, "units": dict(units)}
    if l2:
        start["l2"] = dict(l2)
    case = L2Case(id="probe", title="", section="", seed=0, start=start, script=[], expect={})
    return build_game(case)


def _game_from_state(state: State) -> Game:
    """Wrap a seeded Layer-1 position into a game whose ledger knows the Morgoth unit."""
    hooks = BeleriandHooks(GRAPH, 0)
    present = {unit.power for unit in state.units}
    ownership = {
        center: power
        for power, centers in sorted(GRAPH.home_centers.items())
        if power in present
        for center in centers
    }
    placed = State.of("beleriand", SPRING.tag, state.units)
    hooks.register_start(placed, ownership)
    return Game(BELERIAND, GRAPH, placed, ownership, SPRING, hooks=hooks)


def _hooks(game: Game) -> BeleriandHooks:
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    return hooks


def _rows(game: Game, orders: Mapping[str, Sequence[str]]) -> dict[str, Resolution]:
    """Submit ``orders``, adjudicate, and return each written row's verdict by text."""
    for power, texts in orders.items():
        game.submit(power, _sheets({power: texts})[power])
    record = game.adjudicate()
    return {row.text: row.resolution for row in record.resolutions}


def _four_attack(attacker: str, victim: str) -> list[str]:
    """Return a strength-4 DORIATH Dislodge of ``victim`` in ANF: attacker plus three supports."""
    return [
        f"A {attacker} DIS MORGOTH {victim} ANF",
        *[
            f"A {supporter} S DORIATH A {attacker} DIS MORGOTH {victim} ANF"
            for supporter in ("EIT", "LAD", "DNN")
        ],
    ]


def _digest(game: Game) -> str:
    texts = [
        format_order(order)
        for power in game.powers
        for order in (*game.legal_orders(power), *game.legal_layer2_orders(power))
    ]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


# -- the adjudicator: the designator is the unit's signature --------------------


def test_an_invalid_order_of_a_special_degrades_to_a_designated_hold() -> None:
    # Nogrod is nowhere near Anfauglith, so the Move is invalid and the unit
    # holds; the Hold keeps the (M), because the designator is the unit's
    # signature and not the order's. A unit that wrote nothing stays bare.
    parsed = phase0(
        GRAPH,
        BELERIAND,
        _state({"MORGOTH": ["3A ANF", "A ANF"]}),
        _sheets({"MORGOTH": ["3A(M) ANF - NOG"]}),
    )
    by_unit = {record.unit.strength: record for record in parsed.records}
    assert not by_unit[3].valid
    assert by_unit[3].effective == Hold(UnitKind.ARMY, Location("ANF"), None, "M")
    assert by_unit[1].effective == Hold(UnitKind.ARMY, Location("ANF"))


def test_a_valid_designated_move_keeps_its_designator() -> None:
    (record,) = phase0(
        GRAPH, BELERIAND, _state({"MORGOTH": ["3A ANF"]}), _sheets({"MORGOTH": ["3A(M) ANF - LOT"]})
    ).records
    assert record.valid
    assert isinstance(record.effective, Move)
    assert record.effective.special == "M"


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_two_designated_specials_share_no_wall(engine: EngineName) -> None:
    # Both specials hold under their designators, as the game layer has them
    # hold. The wall MEN raises on the Morgoth unit is his alone: the same
    # strength-4 attack falls on Glaurung's bare 3 and breaks on 3 + 1.
    units = {
        "MORGOTH": ["3A ANF", "3A ANF"],
        "MEN": ["A LOT"],
        "DORIATH": ["A FSE", "A EIT", "A LAD", "A DNN"],
    }
    for victim, verdict, fallen in (
        ("3A(G)", Resolution.SUCCEEDS, 1),
        ("3A(M)", Resolution.FAILS, 0),
    ):
        orders = {
            "MORGOTH": ["3A(M) ANF H", "3A(G) ANF H"],
            "MEN": ["A LOT S MORGOTH 3A(M) ANF"],
            "DORIATH": _four_attack("FSE", victim),
        }
        result = adjudicate(BELERIAND, _state(units), _sheets(orders), graph=GRAPH, engine=engine)
        assert result.resolution_of(f"A FSE DIS MORGOTH {victim} ANF") is verdict
        assert len(result.dislodgements) == fallen


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_bare_bar_reference_binds_no_designated_entrant(engine: EngineName) -> None:
    # The entrant's effective Move carries its (M), so the bare reference of
    # the Bar names nothing and the Move enters; written with the designator
    # the Bar binds it and wins the tie of 3 against 3.
    units = {"MORGOTH": ["3A ANF"], "DORIATH": ["3A LOT"]}
    for reference, move, bar in (
        ("3A", Resolution.SUCCEEDS, Resolution.FAILS),
        ("3A(M)", Resolution.FAILS, Resolution.SUCCEEDS),
    ):
        orders = {
            "DORIATH": [f"3A LOT BAR MORGOTH {reference} ANF - LOT"],
            "MORGOTH": ["3A(M) ANF - LOT"],
        }
        result = adjudicate(BELERIAND, _state(units), _sheets(orders), graph=GRAPH, engine=engine)
        assert result.resolution_of("3A(M) ANF - LOT") is move
        assert result.resolution_of(f"3A LOT BAR MORGOTH {reference} ANF - LOT") is bar


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_support_naming_the_other_special_supports_nothing(engine: EngineName) -> None:
    # 3A(M) fits only Morgoth and 3A(G) only Glaurung: a support written for
    # a Dislodge or a Bar against Glaurung names an order nobody gave when the
    # subject attacked or barred the Morgoth unit, so it matches nothing. The
    # double's attack meets 3 with 2 and fails; the Bar, unsupported, meets a
    # supported entrant at 4 with 3 and falls. With the designators agreeing
    # both supports count.
    units = {"MORGOTH": ["3A ANF", "3A ANF"], "DORIATH": ["2A LOT", "A LAD", "A EIT"]}
    for named, verdict in (("3A(G)", Resolution.FAILS), ("3A(M)", Resolution.SUCCEEDS)):
        orders = {
            "MORGOTH": ["3A(M) ANF H", "3A(G) ANF H"],
            "DORIATH": [
                "2A LOT DIS MORGOTH 3A(M) ANF",
                f"A LAD S DORIATH 2A LOT DIS MORGOTH {named} ANF",
                f"A EIT S DORIATH 2A LOT DIS MORGOTH {named} ANF",
            ],
        }
        result = adjudicate(BELERIAND, _state(units), _sheets(orders), graph=GRAPH, engine=engine)
        assert result.resolution_of("2A LOT DIS MORGOTH 3A(M) ANF") is verdict
        assert result.resolution_of(f"A LAD S DORIATH 2A LOT DIS MORGOTH {named} ANF") is verdict
    bar_units = {"MORGOTH": ["3A ANF", "3A ANF", "A ANF"], "DORIATH": ["3A LOT", "A LAD"]}
    for named, entry in (("3A(G)", Resolution.SUCCEEDS), ("3A(M)", Resolution.FAILS)):
        orders = {
            "MORGOTH": ["3A(M) ANF - LOT", "3A(G) ANF H", "A ANF S MORGOTH 3A(M) ANF - LOT"],
            "DORIATH": [
                "3A LOT BAR MORGOTH 3A(M) ANF - LOT",
                f"A LAD S DORIATH 3A LOT BAR MORGOTH {named} ANF - LOT",
            ],
        }
        result = adjudicate(
            BELERIAND, _state(bar_units), _sheets(orders), graph=GRAPH, engine=engine
        )
        assert result.resolution_of("3A(M) ANF - LOT") is entry


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_victim_action_naming_the_other_special_binds_nothing(engine: EngineName) -> None:
    # A Dislodge that names its victim's action takes the unit that received
    # that action, or fails when none did. The victim attacked the Morgoth
    # unit; an attacker naming a Dislodge of Glaurung as the victim's action
    # names an order nobody gave and fails at 3 against 2, where the same
    # attack naming 3A(M) dislodges the double.
    units = {"MORGOTH": ["3A ANF", "3A ANF"], "DORIATH": ["2A LOT"], "NAUGRIM": ["2A LAD", "A AEL"]}
    for named, verdict, fallen in (
        ("3A(G)", Resolution.FAILS, 0),
        ("3A(M)", Resolution.SUCCEEDS, 1),
    ):
        orders = {
            "MORGOTH": ["3A(M) ANF H", "3A(G) ANF H"],
            "DORIATH": ["2A LOT DIS MORGOTH 3A(M) ANF"],
            "NAUGRIM": [
                f"2A LAD DIS DORIATH 2A LOT DIS MORGOTH {named} ANF",
                "A AEL S NAUGRIM 2A LAD DIS DORIATH 2A LOT",
            ],
        }
        result = adjudicate(BELERIAND, _state(units), _sheets(orders), graph=GRAPH, engine=engine)
        assert result.resolution_of(f"2A LAD DIS DORIATH 2A LOT DIS MORGOTH {named} ANF") is verdict
        assert len(result.dislodgements) == fallen


# -- the game layer: every special reaches the adjudicator designated ---------------


def test_a_special_without_an_order_holds_under_its_designator() -> None:
    game = _game({"MORGOTH": ["3A ANF", "3A ANF"], "MEN": ["A LOT"]}, GLAURUNG_AT_ANF)
    men = _sheets({"MEN": ["A LOT S MORGOTH 3A(M) ANF"]})
    plan = _hooks(game).preprocess_movement(SPRING, game.state, men)
    assert [format_order(o) for o in plan.adjudicated["MORGOTH"]] == ["3A(M) ANF H", "3A(G) ANF H"]
    # nobody wrote those Holds, so nobody reads a row for them
    rows = _rows(game, {"MEN": ["A LOT S MORGOTH 3A(M) ANF"]})
    assert list(rows) == ["A LOT S MORGOTH 3A(M) ANF"]


def test_an_order_the_layer_refuses_still_fails_while_its_unit_holds_designated() -> None:
    # Neither special stands in Lothlann, so both designations are false and
    # both orders fail; each unit holds under its own designator all the
    # same, and the wall on the Morgoth unit does not shield Glaurung.
    game = _game(
        {
            "MORGOTH": ["3A ANF", "3A ANF"],
            "MEN": ["A LOT"],
            "DORIATH": ["A FSE", "A EIT", "A LAD", "A DNN"],
        },
        GLAURUNG_AT_ANF,
    )
    rows = _rows(
        game,
        {
            "MORGOTH": ["3A(M) LOT - EIT", "3A(G) LOT - EIT"],
            "MEN": ["A LOT S MORGOTH 3A(M) ANF"],
            "DORIATH": _four_attack("FSE", "3A(G)"),
        },
    )
    assert rows["3A(M) LOT - EIT"] is Resolution.FAILS
    assert rows["3A(G) LOT - EIT"] is Resolution.FAILS
    assert rows["A FSE DIS MORGOTH 3A(G) ANF"] is Resolution.SUCCEEDS
    assert len(rows) == 7
    specials = _hooks(game).snapshot()["specials"]
    assert specials == {
        "glaurung": {"province": "ANF", "limbo": True},
        "morgoth-unit": {"province": "ANF", "limbo": False},
    }


def test_a_defeated_morgoths_own_designated_hold_keeps_one_failing_row() -> None:
    # Civil disorder throws his sheet out, and the layer's own Hold for the
    # unit carries the same text: the report keeps his row, failing, and not
    # the layer's.
    game = _game({"MORGOTH": ["3A ANF"]}, {"morgoth_defeated": True})
    game.submit("MORGOTH", _sheets({"MORGOTH": ["3A(M) ANF H"]})["MORGOTH"])
    record = game.adjudicate()
    assert [(row.text, row.resolution) for row in record.resolutions] == [
        ("3A(M) ANF H", Resolution.FAILS)
    ]


def test_a_designated_devastate_binds_the_record_held_for_glaurung() -> None:
    # The Morgoth unit walks out and Glaurung, unordered otherwise, holds
    # under the layer's (G) Hold: his Devastate binds that record and not the
    # departing twin's, and the mirror image finds no dragon.
    game = _game({"MORGOTH": ["3A ANF", "3A ANF"]}, GLAURUNG_AT_ANF)
    rows = _rows(game, {"MORGOTH": ["3A(M) ANF - LOT", "3A(G) ANF DEV"]})
    assert rows == {"3A(M) ANF - LOT": Resolution.SUCCEEDS, "3A(G) ANF DEV": Resolution.SUCCEEDS}
    assert "ANF" in _hooks(game).devastated
    mirror = _game({"MORGOTH": ["3A ANF", "3A ANF"]}, GLAURUNG_AT_ANF)
    rows = _rows(mirror, {"MORGOTH": ["3A(G) ANF - LOT", "3A(M) ANF DEV"]})
    assert rows == {"3A(G) ANF - LOT": Resolution.SUCCEEDS, "3A(M) ANF DEV": Resolution.FAILS}
    assert not _hooks(mirror).devastated


def test_a_mask_transfer_or_a_convoy_naming_a_bare_special_is_refused_before_adjudication() -> None:
    # Neither can succeed in play -- Morgoth takes no mask and no fleet carries
    # a strength-3 army -- but the law holds at the reference all the same.
    game = _game(
        {"MORGOTH": ["3A ANF"], "NAUGRIM": ["A LOT"], "MEN": ["F NBE"]},
        {"masks": {"NAUGRIM A LOT": 2}},
    )
    hooks = _hooks(game)
    bare = _sheets(
        {"NAUGRIM": ["A LOT GIVES MASK TO MORGOTH 3A ANF"], "MEN": ["F NBE C 3A ANF - SFA"]}
    )
    plan = hooks.preprocess_movement(SPRING, game.state, bare)
    assert set(plan.invalid) == {("NAUGRIM", 0), ("MEN", 0)}
    designated = _sheets(
        {"NAUGRIM": ["A LOT GIVES MASK TO MORGOTH 3A(M) ANF"], "MEN": ["F NBE C 3A(M) ANF - SFA"]}
    )
    plan = hooks.preprocess_movement(SPRING, game.state, designated)
    assert not plan.invalid
    assert ("NAUGRIM", 0) in plan.deferred
    assert [format_order(o) for o in plan.adjudicated["MEN"]] == ["F NBE C 3A(M) ANF - SFA"]


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(beleriand_position_and_sheets())
def test_every_special_reaches_the_adjudicator_designated(
    pos: tuple[State, dict[str, list[Order]]],
) -> None:
    """The seeded sheets write every 3A bare, and the game layer designates each all the same."""
    state, sheets = pos
    game = _game_from_state(state)
    plan = _hooks(game).preprocess_movement(SPRING, game.state, sheets)
    records = phase0(GRAPH, BELERIAND, game.state, plan.adjudicated).records
    specials = [record for record in records if record.unit.strength == 3]
    assume(specials)
    assert all(record.effective.special in ("M", "G") for record in specials)


# -- the engine writes what it demands --------------------------------------------


def test_the_standard_opening_enumeration_is_unmoved() -> None:
    assert _digest(Game.new("standard", "standard")) == STANDARD_OPENING_DIGEST


def test_the_beleriand_opening_writes_the_morgoth_unit_designated() -> None:
    game = Game.new("beleriand", "beleriand")
    texts = [format_order(o) for o in game.legal_orders("MORGOTH")]
    assert "3A(M) ANG - ANF" in texts
    assert "A ANF S MORGOTH 3A(M) ANG - ANF" in texts
    # every strength-3 unit is a special, so a bare `3A ` names one wrongly
    assert not any("3A " in text for text in texts)


def test_two_specials_in_one_area_are_two_units_with_two_sets_of_orders() -> None:
    game = _game({"MORGOTH": ["3A ANF", "3A ANF", "A ANF"], "MEN": ["A LOT"]}, GLAURUNG_AT_ANF)
    morgoth = [format_order(o) for o in game.legal_orders("MORGOTH")]
    assert "3A(M) ANF - LOT" in morgoth
    assert "3A(G) ANF - LOT" in morgoth
    assert "3A(M) ANF S MORGOTH 3A(G) ANF" in morgoth
    assert "3A(G) ANF S MORGOTH 3A(M) ANF - LOT" in morgoth
    assert "A ANF S MORGOTH 3A(G) ANF" in morgoth
    # a special is never its own subject: its text names that special alone
    assert "3A(M) ANF S MORGOTH 3A(M) ANF" not in morgoth
    men = [format_order(o) for o in game.legal_orders("MEN")]
    assert "A LOT DIS MORGOTH 3A(M) ANF" in men
    assert "A LOT DIS MORGOTH 3A(G) ANF H" in men
    assert "A LOT S MORGOTH 3A(G) ANF" in men
    assert [format_order(o) for o in game.legal_layer2_orders("MORGOTH")] == ["3A(G) ANF DEV"]
    assert not any("3A " in text for text in (*morgoth, *men))


def test_a_dislodged_special_retreats_under_its_designator() -> None:
    game = _game({"MORGOTH": ["3A ANF"], "DORIATH": ["A LOT", "A EIT", "A LAD", "A DNN"]})
    _rows(game, {"DORIATH": _four_attack("LOT", "3A(M)")})
    assert game.phase_kind is PhaseKind.RETREAT
    texts = [format_order(o) for o in game.legal_orders("MORGOTH")]
    assert texts[0] == "Disband 3A(M) ANF"
    assert all(text.startswith(("Disband 3A(M) ANF", "3A(M) ANF - ")) for text in texts)


# -- the records keep the specials ----------------------------------------------------


def test_a_record_keeps_the_specials_as_its_phase_left_them() -> None:
    """Glaurung dislodged beside the Morgoth unit, then disbanded.

    The movement's record has him in limbo where he fell, and the record of
    the retreat that disbanded him has the Morgoth unit alone.
    """
    game = _game(
        {"MORGOTH": ["3A ANF", "3A ANF", "A ANF"], "DORIATH": ["A LOT", "A EIT", "A LAD", "A DNN"]},
        GLAURUNG_AT_ANF,
    )
    game.submit("DORIATH", _sheets({"DORIATH": _four_attack("LOT", "3A(G)")})["DORIATH"])
    movement = game.adjudicate()
    at_anf = Unit("MORGOTH", UnitKind.ARMY, Location("ANF"), 3)
    morgoth_unit = SpecialUnit("morgoth-unit", "M", at_anf)
    assert movement.specials_after == (
        morgoth_unit,
        SpecialUnit("glaurung", "G", at_anf, limbo=True),
    )
    game.submit("MORGOTH", [parse_retreat_order("Disband 3A(G) ANF")])
    retreat = game.adjudicate()
    assert retreat.phase == "S25R"
    assert retreat.specials_after == (morgoth_unit,)


def test_a_standard_record_keeps_no_specials() -> None:
    game = Game.new("standard", "standard")
    assert game.adjudicate().specials_after == ()


@settings(max_examples=40, deadline=None, suppress_health_check=SUPPRESS)
@given(beleriand_position_and_sheets())
def test_no_legal_order_names_a_special_bare(pos: tuple[State, dict[str, list[Order]]]) -> None:
    """Every legal order of every power passes the layer's check, in movement and in retreat."""
    state, sheets = pos
    game = _game_from_state(state)
    hooks = _hooks(game)
    for power in game.powers:
        for order in (*game.legal_orders(power), *game.legal_layer2_orders(power)):
            assert "3A " not in format_order(order)
            assert hooks._designations_valid(power, order)
    for power, sheet in sheets.items():
        game.submit(power, sheet)
    game.adjudicate()
    if game.phase_kind is PhaseKind.RETREAT:
        for power in game.powers:
            for order in game.legal_orders(power):
                assert "3A " not in format_order(order)
                assert hooks._designations_valid(power, order, limbo=True)
