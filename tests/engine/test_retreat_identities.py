"""The game layer follows the unit the retreat adjudicator bound, never a row of the sheet.

docs/adjudication.tex Sec. 9: "of several orders for one dislodged unit the
first valid one counts". The retreat adjudicator binds each order to a
dislodged unit -- the Phase 0 analogue -- and whatever follows the binding
must follow *that* unit.

- The dislodgement record carries the designator of the dislodged unit's own
  order, and a designated retreat binds only that special's dislodgement: with
  both specials dislodged from one Area, each is valid or not for its own
  dislodger, so neither retreats into its own dislodger's origin (DATC 4.A.5
  choice B; the rulebook: "Morgoth and Glaurung are therefore never twins").
- The ledger moves the identities of the units that retreated, keyed by the
  class of the dislodgement each retreat bound -- power, origin, kind, base
  strength and designator -- so an invalid first line kills nobody: the
  Morgoth unit lives where its second line took it, and a mask or a double's
  binding stays with the unit that carried it.
- An entry credential is claimed only by the line the adjudicator binds, from
  that unit's own class, so a stray or invalid line never spends the
  credential of the unit's valid one (the rulebook: an illegal order is
  ignored, "if the same Unit is given another, legal Order, that Order is
  executed").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, adjudicate
from entente.engine.adjudicator.common import Dislodgement, Resolution
from entente.engine.adjudicator.retreats import adjudicate_retreats
from entente.engine.game import Game, PhaseRecord
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import format_order, parse_order, parse_retreat_order, parse_unit_spec
from entente.engine.phase import PhaseKind
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from entente.variants.beleriand.ledger import GLAURUNG, MORGOTH_UNIT
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


def _play(game: Game, orders: Mapping[str, Sequence[str]]) -> PhaseRecord:
    """Submit ``orders`` in the current phase's grammar and adjudicate."""
    parse = parse_retreat_order if game.phase_kind is PhaseKind.RETREAT else parse_order
    for power, texts in orders.items():
        game.submit(power, [parse(text) for text in texts])
    return game.adjudicate()


def _rows(record: PhaseRecord) -> dict[str, Resolution]:
    return {row.text: row.resolution for row in record.resolutions}


def _units(game: Game, power: str) -> list[str]:
    return sorted(str(unit) for unit in game.state.units if unit.power == power)


def _attack(power: str, order: str, supporters: Sequence[str]) -> list[str]:
    """Return ``order`` with a support for it from an army in each of ``supporters``."""
    return [order, *(f"A {supporter} S {power} {order}" for supporter in supporters)]


def _state(units: Mapping[str, Sequence[str]]) -> State:
    return State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )


# The specials dislodged from ANF: by FINROD's 2A from LAD or FINGOLFIN's 2A from
# FSE, each with two supports (strength 4 against 3).
FINROD_ON_M = _attack("FINROD", "2A LAD DIS MORGOTH 3A(M) ANF", ("DNN", "EIT"))
FINROD_ON_G = _attack("FINROD", "2A LAD DIS MORGOTH 3A(G) ANF", ("DNN", "EIT"))
FINGOLFIN_ON_M = _attack("FINGOLFIN", "2A FSE DIS MORGOTH 3A(M) ANF", ("RIV", "NEW"))
BOTH_SPECIALS = {
    "MORGOTH": ["3A ANF", "3A ANF"],
    "FINGOLFIN": ["2A FSE", "A RIV", "A NEW"],
    "FINROD": ["2A LAD", "A DNN", "A EIT"],
}


# -- X2b-7: the dislodgement record carries the designator ----------------------------


@pytest.mark.parametrize("engine", ENGINES)
def test_each_dislodged_special_is_recorded_with_its_designator(engine: EngineName) -> None:
    """The adjudicator records the designator the dislodged unit's own order writes."""
    sheets = {
        "MORGOTH": [parse_order("3A(G) ANF H"), parse_order("3A(M) ANF H")],
        "FINGOLFIN": [parse_order(text) for text in FINGOLFIN_ON_M],
        "FINROD": [parse_order(text) for text in FINROD_ON_G],
    }
    result = adjudicate(BELERIAND, _state(BOTH_SPECIALS), sheets, graph=GRAPH, engine=engine)
    origins = {d.special: d.dislodger_origin for d in result.dislodgements}
    assert origins == {"M": "FSE", "G": "LAD"}


@pytest.mark.parametrize("engine", ENGINES)
def test_a_standard_dislodgement_records_no_designator(engine: EngineName) -> None:
    """A standard order that writes no designator leaves the record the one it always was."""
    state = State.of(
        "standard",
        "S1901M",
        [
            Unit("FRANCE", UnitKind.ARMY, Location("PAR")),
            Unit("FRANCE", UnitKind.ARMY, Location("MAR")),
            Unit("GERMANY", UnitKind.ARMY, Location("BUR")),
        ],
    )
    sheets = {"FRANCE": [parse_order("A PAR - BUR"), parse_order("A MAR S A PAR - BUR")]}
    result = adjudicate(
        RULESETS["standard"], state, sheets, graph=load_map("standard"), engine=engine
    )
    assert result.dislodgements == (
        Dislodgement(Unit("GERMANY", UnitKind.ARMY, Location("BUR")), "PAR", by_convoy=False),
    )
    assert result.dislodgements[0].special is None


def test_a_designated_retreat_is_valid_only_for_its_own_specials_dislodgement() -> None:
    """At the adjudicator ``3A(M) ANF - FSE`` binds the (M) record, dislodged from FSE."""
    sheets = {
        "MORGOTH": [parse_order("3A(M) ANF H"), parse_order("3A(G) ANF H")],
        "FINGOLFIN": [parse_order(text) for text in FINGOLFIN_ON_M],
        "FINROD": [parse_order(text) for text in FINROD_ON_G],
    }
    movement = adjudicate(BELERIAND, _state(BOTH_SPECIALS), sheets, graph=GRAPH)
    retreats = [parse_retreat_order("3A(M) ANF - FSE"), parse_retreat_order("3A(G) ANF - LAD")]
    retreat = adjudicate_retreats(BELERIAND, movement, {"MORGOTH": retreats}, graph=GRAPH)
    assert retreat.resolution_of("3A(M) ANF - FSE") is Resolution.FAILS
    assert retreat.resolution_of("3A(G) ANF - LAD") is Resolution.FAILS
    assert retreat.new_state.units == movement.new_state.units


def _both_specials_dislodged(order_first: str) -> Game:
    """Dislodge the Morgoth unit from FSE and Glaurung from LAD, MORGOTH's Holds in either order."""
    game = _game(BOTH_SPECIALS, GLAURUNG_AT_ANF)
    holds = ["3A(M) ANF H", "3A(G) ANF H"]
    if order_first == "G":
        holds.reverse()
    _play(game, {"MORGOTH": holds, "FINGOLFIN": FINGOLFIN_ON_M, "FINROD": FINROD_ON_G})
    assert game.phase_kind is PhaseKind.RETREAT
    return game


@pytest.mark.parametrize("order_first", ["M", "G"])
def test_a_special_may_not_retreat_into_its_own_dislodgers_origin(order_first: str) -> None:
    game = _both_specials_dislodged(order_first)
    record = _play(game, {"MORGOTH": ["3A(M) ANF - FSE"]})
    assert _rows(record) == {"3A(M) ANF - FSE": Resolution.FAILS}
    assert _hooks(game).morgoth_defeated  # the Morgoth unit had no valid order and disbanded
    assert _units(game, "MORGOTH") == []

    game = _both_specials_dislodged(order_first)
    record = _play(game, {"MORGOTH": ["3A(G) ANF - LAD"]})
    assert _rows(record) == {"3A(G) ANF - LAD": Resolution.FAILS}
    assert _hooks(game).ledger.special(GLAURUNG) is None
    assert _units(game, "MORGOTH") == []


@pytest.mark.parametrize("order_first", ["M", "G"])
def test_each_special_retreats_into_the_other_ones_dislodgers_origin(order_first: str) -> None:
    game = _both_specials_dislodged(order_first)
    record = _play(game, {"MORGOTH": ["3A(M) ANF - LAD", "3A(G) ANF - FSE"]})
    assert set(_rows(record).values()) == {Resolution.SUCCEEDS}
    hooks = _hooks(game)
    morgoth_unit = hooks.ledger.special(MORGOTH_UNIT)
    glaurung = hooks.ledger.special(GLAURUNG)
    assert morgoth_unit is not None and morgoth_unit.location == Location("LAD")
    assert glaurung is not None and glaurung.location == Location("FSE")
    assert not hooks.morgoth_defeated


@pytest.mark.parametrize("order_first", ["M", "G"])
def test_the_offered_retreats_are_each_specials_own(order_first: str) -> None:
    """``legal_orders`` lists each special's retreats from its own dislodgement."""
    game = _both_specials_dislodged(order_first)
    texts = {format_order(order) for order in game.legal_orders("MORGOTH")}
    assert "3A(M) ANF - LAD" in texts and "3A(G) ANF - FSE" in texts
    assert "3A(M) ANF - FSE" not in texts and "3A(G) ANF - LAD" not in texts


# -- X2b-8: the identities follow the units the adjudicator bound -----------------------


def _morgoth_unit_dislodged_from_lad(extra: Mapping[str, Sequence[str]] | None = None) -> Game:
    units: dict[str, list[str]] = {"MORGOTH": ["3A ANF"], "FINROD": ["2A LAD", "A DNN", "A EIT"]}
    for power, specs in (extra or {}).items():
        units.setdefault(power, []).extend(specs)
    return _game(units)


def test_an_invalid_first_line_leaves_the_morgoth_unit_alive_where_it_retreated() -> None:
    """``3A(M) ANF - LAD`` is invalid (its dislodger came from LAD); the next line counts."""
    game = _morgoth_unit_dislodged_from_lad()
    _play(game, {"FINROD": FINROD_ON_M})
    assert game.phase_kind is PhaseKind.RETREAT
    record = _play(game, {"MORGOTH": ["3A(M) ANF - LAD", "3A(M) ANF - LOT"]})
    assert _rows(record) == {
        "3A(M) ANF - LAD": Resolution.FAILS,
        "3A(M) ANF - LOT": Resolution.SUCCEEDS,
    }
    hooks = _hooks(game)
    assert not hooks.morgoth_defeated
    assert not any("defeated" in event for event in record.events)
    assert _units(game, "MORGOTH") == ["3A LOT"]
    morgoth_unit = hooks.ledger.special(MORGOTH_UNIT)
    assert morgoth_unit is not None and not morgoth_unit.limbo
    assert morgoth_unit.location == Location("LOT")


def test_an_invalid_first_line_keeps_the_mask_of_the_unit_that_retreated() -> None:
    """A masked ``A NEB`` dislodged from EST: ``A NEB - EST`` is invalid, ``A NEB - AND`` counts."""
    game = _game(
        {"FEANORIANS": ["A NEB"], "NAUGRIM": ["A EST", "A FOR"]},
        {"masks": {"FEANORIANS A NEB": 1}},
    )
    _play(game, {"NAUGRIM": _attack("NAUGRIM", "A EST DIS FEANORIANS A NEB", ("FOR",))})
    assert game.phase_kind is PhaseKind.RETREAT
    record = _play(game, {"FEANORIANS": ["A NEB - EST", "A NEB - AND"]})
    assert _rows(record) == {"A NEB - EST": Resolution.FAILS, "A NEB - AND": Resolution.SUCCEEDS}
    assert _units(game, "FEANORIANS") == ["A AND"]
    (entry,) = [u for u in _hooks(game).ledger.alive() if u.power == "FEANORIANS"]
    assert entry.location == Location("AND") and not entry.limbo
    assert entry.masks == 1


def test_a_line_for_a_unit_still_on_the_board_takes_no_dislodged_identity() -> None:
    """``A NEB - RAM`` names the A NEB that stayed; the dislodged 2A keeps its binding."""
    game = _game(
        {"FEANORIANS": ["A NEB", "2A NEB"], "NAUGRIM": ["A EST", "A FOR", "A THA"]},
        {"bound_to": {"FEANORIANS 2A NEB": "HEL"}},
    )
    _play(game, {"NAUGRIM": _attack("NAUGRIM", "A EST DIS FEANORIANS 2A NEB", ("FOR", "THA"))})
    assert game.phase_kind is PhaseKind.RETREAT
    record = _play(game, {"FEANORIANS": ["A NEB - RAM", "2A NEB - AND"]})
    assert _rows(record) == {"A NEB - RAM": Resolution.FAILS, "2A NEB - AND": Resolution.SUCCEEDS}
    assert _units(game, "FEANORIANS") == ["2A AND", "A NEB"]
    double = [u for u in _hooks(game).ledger.alive() if u.power == "FEANORIANS" and u.strength == 2]
    assert len(double) == 1
    assert double[0].location == Location("AND") and double[0].bound_to == "HEL"


# -- X2b-9: a credential is claimed only by the line the adjudicator binds ---------------


def test_a_stray_line_into_angband_spends_no_credential_of_the_morgoth_unit() -> None:
    """``A ANF - ANG`` is invalid (a plain army may not enter Angband): ``3A(M)``'s line counts."""
    game = _morgoth_unit_dislodged_from_lad({"MORGOTH": ["A ANF"], "FINGOLFIN": ["2A FSE"]})
    _play(game, {"FINROD": FINROD_ON_M, "FINGOLFIN": ["2A FSE DIS MORGOTH A ANF"]})
    assert len(game.pending_dislodgements) == 2
    record = _play(game, {"MORGOTH": ["A ANF - ANG", "3A(M) ANF - ANG"]})
    assert _rows(record) == {
        "A ANF - ANG": Resolution.FAILS,
        "3A(M) ANF - ANG": Resolution.SUCCEEDS,
    }
    hooks = _hooks(game)
    assert not hooks.morgoth_defeated
    assert _units(game, "MORGOTH") == ["3A ANG"]
    morgoth_unit = hooks.ledger.special(MORGOTH_UNIT)
    assert morgoth_unit is not None and morgoth_unit.location == Location("ANG")


def test_a_stray_line_into_angband_for_no_dislodged_unit_changes_nothing() -> None:
    game = _morgoth_unit_dislodged_from_lad()
    _play(game, {"FINROD": FINROD_ON_M})
    record = _play(game, {"MORGOTH": ["A ANF - ANG", "3A(M) ANF - ANG"]})
    assert _rows(record)["3A(M) ANF - ANG"] is Resolution.SUCCEEDS
    assert not _hooks(game).morgoth_defeated
    assert _units(game, "MORGOTH") == ["3A ANG"]


def _neb_pair_dislodged() -> Game:
    """Dislodge a masked ``A NEB`` and an unmasked ``2A NEB``; AND and RAM are devastated."""
    game = _game(
        {
            "FEANORIANS": ["A NEB", "2A NEB"],
            "NAUGRIM": ["A EST", "A FOR", "2A GEL", "A THA", "A AER"],
        },
        {"masks": {"FEANORIANS A NEB": 1}, "devastated": ["AND", "RAM"]},
    )
    _play(
        game,
        {
            "NAUGRIM": [
                *_attack("NAUGRIM", "A EST DIS FEANORIANS A NEB", ("FOR",)),
                *_attack("NAUGRIM", "2A GEL DIS FEANORIANS 2A NEB", ("THA", "AER")),
            ]
        },
    )
    assert len(game.pending_dislodgements) == 2
    return game


@pytest.mark.parametrize(
    "sheet",
    [["2A NEB - AND", "A NEB - RAM"], ["A NEB - RAM", "2A NEB - AND"]],
    ids=["2A-first", "A-first"],
)
def test_the_mask_of_one_unit_is_not_spent_by_another_units_line(sheet: list[str]) -> None:
    """The unmasked 2A may not enter devastated ground; the masked A's own line counts."""
    game = _neb_pair_dislodged()
    record = _play(game, {"FEANORIANS": sheet})
    assert _rows(record) == {"2A NEB - AND": Resolution.FAILS, "A NEB - RAM": Resolution.SUCCEEDS}
    assert _units(game, "FEANORIANS") == ["A RAM"]
    (entry,) = [u for u in _hooks(game).ledger.alive() if u.power == "FEANORIANS"]
    assert entry.strength == 1 and entry.location == Location("RAM") and entry.masks == 1


def test_an_invalid_line_into_devastated_ground_spends_no_credential() -> None:
    """``A NEB - SEB`` is no edge (SEB devastated too); ``A NEB - AND`` keeps the credential."""
    game = _game(
        {"FEANORIANS": ["A NEB"], "NAUGRIM": ["A EST", "A FOR"]},
        {"masks": {"FEANORIANS A NEB": 1}, "devastated": ["AND", "SEB"]},
    )
    _play(game, {"NAUGRIM": _attack("NAUGRIM", "A EST DIS FEANORIANS A NEB", ("FOR",))})
    record = _play(game, {"FEANORIANS": ["A NEB - SEB", "A NEB - AND"]})
    assert _rows(record) == {"A NEB - SEB": Resolution.FAILS, "A NEB - AND": Resolution.SUCCEEDS}
    assert _units(game, "FEANORIANS") == ["A AND"]


def test_twins_with_one_mask_send_one_twin_into_devastated_ground() -> None:
    """Control: of two lines into devastated ground for twins with one mask, the first counts."""
    game = _game(
        {"FEANORIANS": ["A NEB", "A NEB"], "NAUGRIM": ["A EST", "A FOR", "A GEL", "A THA"]},
        {"masks": {"FEANORIANS A NEB": [1, 0]}, "devastated": ["AND", "RAM"]},
    )
    _play(
        game,
        {
            "NAUGRIM": [
                *_attack("NAUGRIM", "A EST DIS FEANORIANS A NEB", ("FOR",)),
                *_attack("NAUGRIM", "A GEL DIS FEANORIANS A NEB", ("THA",)),
            ]
        },
    )
    assert len(game.pending_dislodgements) == 2
    record = _play(game, {"FEANORIANS": ["A NEB - AND", "A NEB - RAM"]})
    assert _rows(record) == {"A NEB - AND": Resolution.SUCCEEDS, "A NEB - RAM": Resolution.FAILS}
    assert _units(game, "FEANORIANS") == ["A AND"]
    (entry,) = [u for u in _hooks(game).ledger.alive() if u.power == "FEANORIANS"]
    assert entry.location == Location("AND") and entry.masks == 1
