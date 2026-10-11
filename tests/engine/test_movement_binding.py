"""Which written line a unit takes in a Beleriand movement phase, through the game.

The rulebook's 4.E.1, which points to its "Invalid Orders": *"An illegal Order
is ignored: if the same Unit is given another, legal Order, that Order is
executed, and otherwise the Unit holds."* Its Signature paragraph: *"3A
written without a designator fits no Unit -- an Order or a reference written
so is invalid, like one with a wrong Unit type."* The game layer applies both
before and after the adjudicator, and most checks here drive whole phases
through the ``Game`` facade. An illegal line claims nothing: not the unit a
Layer-2 main order needs, so a unit whose only adjudication orders are illegal
makes its masks, and not the mask a later line into devastated ground needs. A
bare special named inside a Dislodge's victim action voids the Dislodge, so
the unit's next line runs. With both specials in one Area a second order
writing one designator is passed over like any second order for one unit, so
the other special keeps its own order or its Hold, and the board and the
ledger agree; and the occurrence an ignored line of one special binds stays
that special's, which holds under its designator wherever the line stands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, adjudicate
from entente.engine.adjudicator.common import Resolution, phase0
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from tests.batc.l2_runner import L2Case, build_game

BELERIAND = RULESETS["beleriand"]
GRAPH = load_map("beleriand")
GLAURUNG_AT_ANF = {"glaurung": "ANF"}


def _game(units: Mapping[str, Sequence[str]], l2: Mapping[str, object] | None = None) -> Game:
    """Build a Beleriand game at Spring 25 from hand-placed units and Layer-2 overrides."""
    start: dict[str, object] = {"phase": "S25M", "units": dict(units)}
    if l2:
        start["l2"] = dict(l2)
    case = L2Case(id="probe", title="", section="", seed=0, start=start, script=[], expect={})
    return build_game(case)


def _hooks(game: Game) -> BeleriandHooks:
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    return hooks


def _rows(game: Game, orders: Mapping[str, Sequence[str]]) -> list[tuple[str, Resolution]]:
    """Submit ``orders``, adjudicate, and return every written row's text and verdict in order."""
    for power, texts in orders.items():
        game.submit(power, [parse_order(text) for text in texts])
    record = game.adjudicate()
    return [(row.text, row.resolution) for row in record.resolutions]


def _where(game: Game, power: str) -> list[str]:
    """Return ``power``'s units on the board as sorted ``<strength><kind> <province>`` texts."""
    return sorted(
        f"{unit.strength}{unit.kind.value} {unit.location.province}"
        for unit in game.state.units
        if unit.power == power
    )


# -- an illegal order leaves its unit unordered for the Layer-2 main orders --------


def _masks_made(orders: Mapping[str, Sequence[str]]) -> tuple[list[tuple[str, Resolution]], bool]:
    """Play ``orders`` on NAUGRIM A DOL beside MORGOTH A NOS; return the rows and the masks made."""
    game = _game({"NAUGRIM": ["A DOL"], "MORGOTH": ["A NOS"]})
    for power, texts in orders.items():
        game.submit(power, [parse_order(text) for text in texts])
    record = game.adjudicate()
    rows = [(row.text, row.resolution) for row in record.resolutions if row.power == "NAUGRIM"]
    return rows, "NAUGRIM A DOL makes 2 masks" in record.events


@pytest.mark.parametrize("sheet", [["A DOL - NEB", "A DOL MSK"], ["A DOL MSK", "A DOL - NEB"]])
def test_an_illegal_order_leaves_its_unit_to_its_layer2_main_order(sheet: list[str]) -> None:
    # Mount Dolmed does not reach NEB: the Move is ignored, so the army is
    # unordered for the adjudicator and makes its masks, whatever the order.
    rows, made = _masks_made({"NAUGRIM": sheet, "MORGOTH": ["A NOS H"]})
    assert dict(rows) == {"A DOL - NEB": Resolution.FAILS, "A DOL MSK": Resolution.SUCCEEDS}
    assert made


def test_a_legal_order_keeps_its_unit_from_its_layer2_main_order() -> None:
    # A legal Move is the unit's order; a legal Support that proves invalid
    # (the subject holds) keeps the unit too: it holds, and MSK fails.
    rows, made = _masks_made({"NAUGRIM": ["A DOL MSK", "A DOL - THA"], "MORGOTH": ["A NOS H"]})
    assert dict(rows) == {"A DOL MSK": Resolution.FAILS, "A DOL - THA": Resolution.SUCCEEDS}
    assert not made
    support = "A DOL S MORGOTH A NOS - THA"
    rows, made = _masks_made({"NAUGRIM": [support, "A DOL MSK"], "MORGOTH": ["A NOS H"]})
    assert dict(rows) == {support: Resolution.FAILS, "A DOL MSK": Resolution.FAILS}
    assert not made


@pytest.mark.parametrize(
    ("move", "devastates"), [("3A(G) ANF - NOG", True), ("3A(G) ANF - LOT", False)]
)
def test_an_illegal_order_of_glaurung_leaves_him_to_his_devastate(
    move: str, devastates: bool
) -> None:
    # Nogrod is out of reach, so Glaurung's Move is ignored and his Devastate
    # binds the record his designated Hold keeps; a legal Move is his order.
    game = _game({"MORGOTH": ["3A ANG", "3A ANF"]}, {"morgoth_unit": "ANG", "glaurung": "ANF"})
    rows = dict(_rows(game, {"MORGOTH": [move, "3A(G) ANF DEV"]}))
    assert rows["3A(G) ANF DEV"] is (Resolution.SUCCEEDS if devastates else Resolution.FAILS)
    assert ("ANF" in _hooks(game).devastated) is devastates


# -- the designator law reaches a Dislodge's victim action ------------------------


@pytest.mark.parametrize(
    ("action", "kept"),
    [
        ("S 3A(M) ANF - LAD", True),
        ("S 3A ANF - LAD", False),  # the Morgoth unit named bare: the Dislodge is dropped
        ("S 2A ANF - LAD", True),  # describes no special: the written strength is not read
    ],
)
def test_a_bare_special_in_a_victim_action_voids_the_dislodge(action: str, kept: bool) -> None:
    game = _game(
        {"MORGOTH": ["3A ANF", "A ANF", "A ANF"], "FINROD": ["A DNN"], "FEANORIANS": ["A LOT"]}
    )
    rows = _rows(
        game,
        {
            "MORGOTH": ["3A(M) ANF - LAD", "A ANF S 3A(M) ANF - LAD", "A ANF H"],
            "FINROD": ["A DNN - LAD"],
            "FEANORIANS": [f"A LOT DIS MORGOTH A ANF {action}", "A LOT - AEL"],
        },
    )
    verdict = dict(rows)
    assert verdict[f"A LOT DIS MORGOTH A ANF {action}"] is Resolution.FAILS
    # the kept Dislodge is the unit's order and its next line is passed over;
    # the dropped one is ignored and the next line runs
    assert verdict["A LOT - AEL"] is (Resolution.FAILS if kept else Resolution.SUCCEEDS)
    assert _where(game, "FEANORIANS") == (["1A LOT"] if kept else ["1A AEL"])


def test_the_law_reads_every_reference_inside_a_victim_action() -> None:
    game = _game({"MORGOTH": ["3A ANF", "A ANF"], "FEANORIANS": ["A LOT", "A AEL"]})
    hooks = _hooks(game)
    for text, valid in (
        ("A LOT DIS MORGOTH A ANF S 3A(M) ANF - LAD", True),
        ("A LOT DIS MORGOTH A ANF S 3A ANF - LAD", False),
        ("A LOT DIS MORGOTH A ANF S MORGOTH 3A ANF", False),
        # a support's subject that is itself a Dislodge with a victim action
        ("A AEL S FEANORIANS A LOT DIS MORGOTH A ANF S 3A(M) ANF - LAD", True),
        ("A AEL S FEANORIANS A LOT DIS MORGOTH A ANF S 3A ANF - LAD", False),
    ):
        assert hooks._designations_valid("FEANORIANS", parse_order(text)) is valid, text


# -- one designator, one order: both specials in one Area ------------------------


def _specials(game: Game) -> dict[str, str]:
    """Return each living special's designator and the province the ledger has it in."""
    return {
        special.designator: special.unit.location.province
        for special in _hooks(game).special_units()
    }


@pytest.mark.parametrize("second", ["3A(M) ANF - LOT", "3A(M) ANF - DNN"])
def test_a_second_order_writing_one_designator_is_passed_over(second: str) -> None:
    # The second (M) order names the Morgoth unit, which already has its
    # order: it binds nothing, and Glaurung holds under the layer's (G) Hold.
    game = _game({"MORGOTH": ["3A ANF", "3A ANF", "A ANG"]}, GLAURUNG_AT_ANF)
    rows = _rows(game, {"MORGOTH": ["3A(M) ANF - LOT", second]})
    assert rows == [("3A(M) ANF - LOT", Resolution.SUCCEEDS), (second, Resolution.FAILS)]
    assert _where(game, "MORGOTH") == ["1A ANG", "3A ANF", "3A LOT"]
    assert _specials(game) == {"M": "LOT", "G": "ANF"}
    # the board and the ledger agree, so Glaurung keeps his orders next phase
    assert game.phase == "F25M"  # nothing was dislodged: no retreat phase
    _rows(game, {"MORGOTH": ["3A(G) ANF - DNN"]})
    assert _where(game, "MORGOTH") == ["1A ANG", "3A DNN", "3A LOT"]
    assert _specials(game) == {"M": "LOT", "G": "DNN"}


def test_an_illegal_first_order_leaves_the_designator_to_the_next() -> None:
    # Nogrod is out of reach, so the first (M) line is ignored and the second
    # one is the Morgoth unit's order; Glaurung holds.
    game = _game({"MORGOTH": ["3A ANF", "3A ANF"]}, GLAURUNG_AT_ANF)
    rows = _rows(game, {"MORGOTH": ["3A(M) ANF - NOG", "3A(M) ANF - LOT"]})
    assert rows == [("3A(M) ANF - NOG", Resolution.FAILS), ("3A(M) ANF - LOT", Resolution.SUCCEEDS)]
    assert _specials(game) == {"M": "LOT", "G": "ANF"}


@pytest.mark.parametrize(
    "sheet",
    [
        ["3A(M) ANF - LOT", "3A(G) ANF - NOG", "3A(G) ANF DEV"],
        ["3A(G) ANF - NOG", "3A(M) ANF - LOT", "3A(G) ANF DEV"],  # the illegal line first
    ],
)
def test_an_illegal_order_of_one_special_keeps_its_occurrence(sheet: list[str]) -> None:
    # Nogrod is out of reach, so Glaurung's Move is ignored and his Devastate
    # is his order, wherever the ignored line stands: the Morgoth unit's Move
    # takes the other occurrence, not the one Glaurung's line holds.
    game = _game({"MORGOTH": ["3A ANF", "3A ANF", "A ANG"]}, GLAURUNG_AT_ANF)
    assert dict(_rows(game, {"MORGOTH": sheet})) == {
        "3A(M) ANF - LOT": Resolution.SUCCEEDS,
        "3A(G) ANF - NOG": Resolution.FAILS,
        "3A(G) ANF DEV": Resolution.SUCCEEDS,
    }
    assert "ANF" in _hooks(game).devastated
    assert _specials(game) == {"M": "LOT", "G": "ANF"}


def test_a_special_whose_only_order_is_illegal_holds_under_its_designator() -> None:
    # The Morgoth unit's only line is ignored and the layer's (G) Hold takes
    # the other occurrence, so the Morgoth unit holds as 3A(M): a support
    # naming it matches.
    game = _game({"MORGOTH": ["3A ANF", "3A ANF", "A ANG"]}, GLAURUNG_AT_ANF)
    rows = dict(_rows(game, {"MORGOTH": ["3A(M) ANF - NOG", "A ANG S 3A(M) ANF"]}))
    assert rows == {"3A(M) ANF - NOG": Resolution.FAILS, "A ANG S 3A(M) ANF": Resolution.SUCCEEDS}
    assert _specials(game) == {"M": "ANF", "G": "ANF"}


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("illegal_first", [True, False])
def test_phase0_keeps_an_illegal_claim_from_the_other_designator(
    engine: EngineName, illegal_first: bool
) -> None:
    # Without the game layer: the (G) Hold binds the occurrence the ignored
    # (M) line does not hold, so each 3A keeps its own designator.
    state = State.of(
        "beleriand",
        "S25M",
        [Unit("MORGOTH", *parse_unit_spec(spec)) for spec in ("3A ANF", "3A ANF")],
    )
    illegal, hold = parse_order("3A(M) ANF - NOG"), parse_order("3A(G) ANF H")
    sheets = {"MORGOTH": [illegal, hold] if illegal_first else [hold, illegal]}
    parsed = phase0(GRAPH, BELERIAND, state, sheets)
    assert {record.written for record in parsed.records} == {illegal, hold}
    assert {record.effective.special for record in parsed.records} == {"M", "G"}
    assert parsed.rejected == ()
    result = adjudicate(BELERIAND, state, sheets, graph=GRAPH, engine=engine)
    assert result.resolution_of("3A(M) ANF - NOG") is Resolution.FAILS
    assert result.resolution_of("3A(G) ANF H") is Resolution.SUCCEEDS


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_phase0_binds_one_occurrence_per_written_designator(engine: EngineName) -> None:
    # Without the game layer: the second (M) order binds nothing, and the
    # other 3A is unordered and holds.
    state = State.of(
        "beleriand",
        "S25M",
        [Unit("MORGOTH", *parse_unit_spec(spec)) for spec in ("3A ANF", "3A ANF")],
    )
    sheets = {"MORGOTH": [parse_order("3A(M) ANF - LOT"), parse_order("3A(M) ANF - DNN")]}
    parsed = phase0(GRAPH, BELERIAND, state, sheets)
    assert [record.written for record in parsed.records] == [sheets["MORGOTH"][0], None]
    assert parsed.rejected == (("MORGOTH", sheets["MORGOTH"][1]),)
    result = adjudicate(BELERIAND, state, sheets, graph=GRAPH, engine=engine)
    assert result.resolution_of("3A(M) ANF - LOT") is Resolution.SUCCEEDS
    assert result.resolution_of("3A(M) ANF - DNN") is Resolution.FAILS
    assert sorted(unit.location.province for unit in result.new_state.units) == ["ANF", "LOT"]


# -- a mask credential goes to the line Phase 0 keeps ------------------------------

DEVASTATED = ["ANF", "DNN", "LAD"]


@pytest.mark.parametrize(
    "sheet",
    [
        ["A LOT - LAD"],
        ["A LOT - DNN", "A LOT - LAD"],  # Lothlann does not reach DNN: ignored
        ["A LOT - LAD", "A LOT - DNN"],
        ["A LOT - NOG", "A LOT - LAD"],  # a stray line into ground no mask gates
    ],
)
def test_an_illegal_line_spends_no_mask(sheet: list[str]) -> None:
    game = _game(
        {"FEANORIANS": ["A LOT"]}, {"devastated": DEVASTATED, "masks": {"FEANORIANS A LOT": 1}}
    )
    rows = dict(_rows(game, {"FEANORIANS": sheet}))
    assert rows["A LOT - LAD"] is Resolution.SUCCEEDS
    assert all(rows[text] is Resolution.FAILS for text in sheet if text != "A LOT - LAD")
    assert _where(game, "FEANORIANS") == ["1A LAD"]


def test_one_mask_takes_one_twin_into_devastated_ground() -> None:
    # Two twins, one mask: the first kept line claims the masked twin, and the
    # second, left without a mask, is voided -- the other twin stays, whether
    # or not an illegal line comes first.
    for prefix in ([], ["A LOT - DNN"]):
        game = _game(
            {"FEANORIANS": ["A LOT", "A LOT"]},
            {"devastated": DEVASTATED, "masks": {"FEANORIANS A LOT": [1, 0]}},
        )
        _rows(game, {"FEANORIANS": [*prefix, "A LOT - ANF", "A LOT - ANF"]})
        assert _where(game, "FEANORIANS") == ["1A ANF", "1A LOT"]
        assert [(e.location.province, e.masks) for e in _hooks(game).ledger.board()] == [
            ("ANF", 1),
            ("LOT", 0),
        ]
