"""Pin the Teleri army in disorder: every order written for it is void.

The rulebook's "Teleri": "If Teleri has more Armies than Fleets, then one of
Armies, chosen at random, will stand in disorder, i. e. will have forced an
order to Hold." So no order written for that army moves it -- not the first
one, and not a later one that the adjudicator would bind to it once the first
is gone. Nor does its line spend the mask a twin's line into devastated
ground needs.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.common import Resolution
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks


def _teleri_spring(armies: list[str], others: dict[str, list[str]] | None = None) -> Game:
    """Open Spring 25 with TELERI's armies and no fleet, on the seed-0 stream.

    With two armies the pick draws ``randrange(2)`` after the fifteen Ulmo dice
    of the construction, which under seed 0 is 0 (BATC-L2.F.5): the first army
    in state order stands in disorder. ``others`` adds other powers' units.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    units = [Unit("TELERI", *parse_unit_spec(spec)) for spec in armies]
    for power, specs in (others or {}).items():
        units.extend(Unit(power, *parse_unit_spec(spec)) for spec in specs)
    state = State.of("beleriand", phase.tag, units)
    ownership = dict.fromkeys(graph.home_centers["TELERI"], "TELERI")
    hooks.register_start(state, ownership)
    return Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)


def test_every_order_written_for_the_army_in_disorder_is_void() -> None:
    """Hold the army in disorder against a second order written for it.

    Seed 0 picks A BRI. Its sheet writes two Moves for it; the second one, a
    duplicate the adjudicator would bind to nothing while the first stands,
    must not take the first one's place: both fail and A BRI stays, while
    A NLA, which is not in disorder, moves.
    """
    game = _teleri_spring(["A BRI", "A NLA"])
    texts = ("A BRI - BNI", "A BRI - NFA", "A NLA - LAM")
    game.submit("TELERI", [parse_order(text) for text in texts])
    record = game.adjudicate()
    assert record.events == ("TELERI A BRI stands in disorder (more armies than fleets)",)
    outcome = {row.text: row.resolution for row in record.resolutions}
    assert outcome == {
        "A BRI - BNI": Resolution.FAILS,
        "A BRI - NFA": Resolution.FAILS,
        "A NLA - LAM": Resolution.SUCCEEDS,
    }
    assert sorted(str(unit) for unit in game.state.units) == ["A BRI", "A LAM"]


def test_the_twin_in_disorder_takes_no_order_past_its_twins_share() -> None:
    """Keep one of two twin armies in place when the sheet writes three Moves for them.

    Seed 0 picks the first of the two A TAL twins, so the first order naming
    them is void. The second goes to the other twin; the third is one order
    more than the twins can take, and it is the disordered twin's as well, so
    one twin moves and the other stays.
    """
    game = _teleri_spring(["A TAL", "A TAL"])
    texts = ("A TAL - DIM", "A TAL - FOB", "A TAL - RON")
    game.submit("TELERI", [parse_order(text) for text in texts])
    record = game.adjudicate()
    assert record.events == ("TELERI A TAL stands in disorder (more armies than fleets)",)
    outcome = {row.text: row.resolution for row in record.resolutions}
    assert outcome == {
        "A TAL - DIM": Resolution.FAILS,
        "A TAL - FOB": Resolution.SUCCEEDS,
        "A TAL - RON": Resolution.FAILS,
    }
    assert sorted(str(unit) for unit in game.state.units) == ["A FOB", "A TAL"]


def test_an_illegal_line_inside_the_twins_share_moves_no_order_past_it() -> None:
    """Give the twin not in disorder the order Phase 0 binds to it, past an illegal line.

    Seed 0 picks the first of the two A TAL twins. Phase 0 binds ``A TAL - DIM``
    to that twin, ignores ``A TAL - DDI`` (Talath Dimen does not reach Dor
    Dinen) and binds ``A TAL - RON`` to the other twin. Only the army in
    disorder loses its order, so the other twin moves to RON.
    """
    game = _teleri_spring(["A TAL", "A TAL"])
    texts = ("A TAL - DIM", "A TAL - DDI", "A TAL - RON")
    game.submit("TELERI", [parse_order(text) for text in texts])
    record = game.adjudicate()
    assert record.events == ("TELERI A TAL stands in disorder (more armies than fleets)",)
    outcome = {row.text: row.resolution for row in record.resolutions}
    assert outcome == {
        "A TAL - DIM": Resolution.FAILS,
        "A TAL - DDI": Resolution.FAILS,
        "A TAL - RON": Resolution.SUCCEEDS,
    }
    assert sorted(str(unit) for unit in game.state.units if unit.power == "TELERI") == [
        "A RON",
        "A TAL",
    ]


def test_a_banned_support_inside_the_twins_share_moves_no_order_past_it() -> None:
    """Give the twin not in disorder its legal line when the game layer voids the one before it.

    Seed 0 picks the first of the two A TAL twins, so ``A TAL - FOB`` is void.
    ``A TAL S MORGOTH A DIM`` is void as well -- no elven power supports
    Morgoth -- so Phase 0 binds ``A TAL - RON`` to the other twin, which moves.
    """
    game = _teleri_spring(["A TAL", "A TAL"], {"MORGOTH": ["A DIM"]})
    texts = ("A TAL - FOB", "A TAL S MORGOTH A DIM", "A TAL - RON")
    game.submit("TELERI", [parse_order(text) for text in texts])
    record = game.adjudicate()
    assert record.events == ("TELERI A TAL stands in disorder (more armies than fleets)",)
    outcome = {row.text: row.resolution for row in record.resolutions}
    assert outcome == {
        "A TAL - FOB": Resolution.FAILS,
        "A TAL S MORGOTH A DIM": Resolution.FAILS,
        "A TAL - RON": Resolution.SUCCEEDS,
    }
    assert sorted(str(unit) for unit in game.state.units if unit.power == "TELERI") == [
        "A RON",
        "A TAL",
    ]


@pytest.mark.parametrize("masked", [0, 1], ids=["first-identity", "second-identity"])
def test_the_army_in_disorder_spends_no_mask_its_twin_needs(masked: int) -> None:
    """Leave the one mask to the twin not in disorder when both lines enter devastated ground.

    Seed 0 picks the first of the two A TAL twins, so ``A TAL - DIM`` is void.
    DIM and RON are devastated and one of the twins' identities carries a
    mask. The army in disorder holds, so its line claims no mask: the other
    twin's ``A TAL - RON`` takes it and moves, whichever identity holds it.
    """
    game = _teleri_spring(["A TAL", "A TAL"])
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    hooks.devastated = {"DIM", "RON"}
    hooks.ledger.board()[masked].masks = 1
    texts = ("A TAL - DIM", "A TAL - RON")
    game.submit("TELERI", [parse_order(text) for text in texts])
    record = game.adjudicate()
    assert record.events == ("TELERI A TAL stands in disorder (more armies than fleets)",)
    outcome = {row.text: row.resolution for row in record.resolutions}
    assert outcome == {"A TAL - DIM": Resolution.FAILS, "A TAL - RON": Resolution.SUCCEEDS}
    assert sorted((e.location.province, e.masks) for e in hooks.ledger.board()) == [
        ("RON", 1),
        ("TAL", 0),
    ]
