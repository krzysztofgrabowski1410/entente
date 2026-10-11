"""Pin the twin the identity ledger sends to limbo to the twin the adjudicator dislodged.

Unnamed Dislodges on a twin stack take twins from the top of the attacked
power's list, strongest first, a cohort of equal attacks spreading over the
twins one per attack (the rulebook's twins rule 6; docs/adjudication.tex
Sec. 4.4). The twins are one value to the adjudicator, so its result says how
many fell, not which; the Beleriand identity ledger has to learn that from the
engagement itself, read under the phase's final labeling. Two positions where
taking the attacks one by one in attacker order, each winner taking the first
twin left, would name the other twin:

* a cohort of two equal attacks spreads over a supported holder and a mover
  that bounced; the attack on the holder fails and the one on the mover
  succeeds, so the mover falls -- the holder's Hold succeeds, and its
  identity, masks and all, stays on the board and passes a mask;
* attacks of strength 2 and 3 on two holders: the stronger takes the top
  twin, so each record falls to the attack the dislodgement record names.

On both, the labeling the ledger settles again is the one either engine
reported, and the twins it names are the twins that labeling dislodged.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pytest

import entente  # noqa: F401 -- wires the variant registry (beleriand into RULESETS)
from entente.engine.adjudicator import ENGINES, EngineName, adjudicate
from entente.engine.adjudicator.common import (
    Resolution,
    assemble,
    build_tables,
    final_walk_victims,
    phase0,
)
from entente.engine.adjudicator.resolver import settle
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.orders import format_order, parse_order, parse_unit_spec
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from tests.batc.l2_runner import L2Case, build_game

BELERIAND = RULESETS["beleriand"]
GRAPH = load_map("beleriand")

# The cohort board: GONDOLIN's and FINROD's 2A Dislodges are one cohort, which
# spreads over the NAUGRIM twins in list order -- FINROD (first in canonical
# order) on the holder, which the 2A's Support Hold lifts to 3, GONDOLIN on the
# mover, which bounced off EIT and defends with its bare 1.
COHORT_UNITS = {
    "NAUGRIM": ["A ANF", "A ANF", "2A ANF"],
    "DORIATH": ["A LOT"],
    "FINGOLFIN": ["A EIT"],
    "FINROD": ["2A DNN"],
    "GONDOLIN": ["2A RIV"],
}
COHORT_ORDERS = {
    "NAUGRIM": [
        "A ANF H",
        "A ANF - EIT",
        "2A ANF S A ANF",
        "A ANF GIVES MASK TO DORIATH A LOT",
    ],
    "DORIATH": ["A LOT ACCEPTS MASK FROM NAUGRIM A ANF"],
    "FINGOLFIN": ["A EIT H"],
    "FINROD": ["2A DNN DIS NAUGRIM A ANF"],
    "GONDOLIN": ["2A RIV DIS NAUGRIM A ANF"],
}


def _game(units: Mapping[str, Sequence[str]], l2: Mapping[str, object]) -> Game:
    """Build a Beleriand game at Spring 25 from hand-placed units and Layer-2 overrides."""
    start: dict[str, object] = {"phase": "S25M", "units": dict(units), "l2": dict(l2)}
    case = L2Case(id="probe", title="", section="", seed=0, start=start, script=[], expect={})
    return build_game(case)


def test_the_twin_the_walk_spared_keeps_its_identity_and_passes_its_mask() -> None:
    # The first-created NAUGRIM twin carries two masks, the other none. The
    # give binds the holder's record and claims the masked identity for it;
    # the holder stands (its Hold succeeds), so the masked identity stays on
    # the board and passes a mask, and the identity without masks -- the
    # mover's -- is the one in limbo.
    game = _game(COHORT_UNITS, {"masks": {"NAUGRIM A ANF": [2, 0]}})
    for power, texts in COHORT_ORDERS.items():
        game.submit(power, [parse_order(text) for text in texts])
    record = game.adjudicate()
    rows = {row.text: row.resolution for row in record.resolutions}
    assert rows["A ANF H"] is Resolution.SUCCEEDS
    assert rows["A ANF - EIT"] is Resolution.FAILS
    assert rows["2A DNN DIS NAUGRIM A ANF"] is Resolution.FAILS
    assert rows["2A RIV DIS NAUGRIM A ANF"] is Resolution.SUCCEEDS
    assert [(str(d.unit), d.dislodger_origin) for d in record.dislodgements] == [("A ANF", "RIV")]
    assert rows["A ANF GIVES MASK TO DORIATH A LOT"] is Resolution.SUCCEEDS
    assert rows["A LOT ACCEPTS MASK FROM NAUGRIM A ANF"] is Resolution.SUCCEEDS
    assert "NAUGRIM A ANF passes a mask to DORIATH A LOT" in record.events
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    limbo = [(e.power, e.spec(), e.masks) for e in hooks.ledger.limbo()]
    assert limbo == [("NAUGRIM", "A ANF", 0)]
    masks = {(e.power, e.spec()): e.masks for e in hooks.ledger.alive() if not e.limbo and e.masks}
    assert masks == {("NAUGRIM", "A ANF"): 1, ("DORIATH", "A LOT"): 1}


# The two boards at the adjudicator, each with the twin every successful walked
# attack took, keyed by the attacker's province: the victim's written order and
# its place on its power's sheet. On the cohort board FINROD's attack fails
# against the holder and takes nobody; on the strength board LAD's 3 takes the
# top twin and DNN's 2 the next.
WALK_BOARDS = {
    "cohort": (
        {
            "MORGOTH": ["A ANF", "A ANF", "2A ANF"],
            "FINGOLFIN": ["A EIT"],
            "FINROD": ["2A DNN"],
            "GONDOLIN": ["2A RIV"],
        },
        {
            "MORGOTH": ["A ANF H", "A ANF - EIT", "2A ANF S A ANF"],
            "FINGOLFIN": ["A EIT H"],
            "FINROD": ["2A DNN DIS MORGOTH A ANF"],
            "GONDOLIN": ["2A RIV DIS MORGOTH A ANF"],
        },
        {"RIV": ("A ANF - EIT", 1)},
    ),
    "strength": (
        {"MORGOTH": ["A ANF", "A ANF"], "FINROD": ["A DNN", "A LAD", "A EIT", "A RIV", "A LOT"]},
        {
            "MORGOTH": ["A ANF H", "A ANF H"],
            "FINROD": [
                "A DNN DIS MORGOTH A ANF",
                "A LAD DIS MORGOTH A ANF",
                "A EIT S A DNN DIS MORGOTH A ANF",
                "A RIV S A LAD DIS MORGOTH A ANF",
                "A LOT S A LAD DIS MORGOTH A ANF",
            ],
        },
        {"LAD": ("A ANF H", 0), "DNN": ("A ANF H", 1)},
    ),
}


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("board", sorted(WALK_BOARDS))
def test_the_walk_the_ledger_reads_is_the_engagement_the_result_reports(
    board: str, engine: EngineName
) -> None:
    # ``settle`` over the phase's own records and tables assembles to the very
    # result either engine reported, and every twin ``final_walk_victims``
    # names is dislodged under that labeling, by an attack from the province
    # the dislodgement record gives.
    units, orders, expected = WALK_BOARDS[board]
    state = State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )
    sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
    result = adjudicate(BELERIAND, state, sheets, graph=GRAPH, engine=engine)
    parsed = phase0(GRAPH, BELERIAND, state, sheets)
    records = parsed.records
    evaluator = settle(GRAPH, BELERIAND, records, build_tables(GRAPH, BELERIAND, records))
    assert assemble(GRAPH, state, sheets, parsed, evaluator) == result
    victims = final_walk_victims(evaluator)
    taken: dict[str, tuple[str, int]] = {}
    for attack, victim in victims.items():
        written = records[victim].written
        assert written is not None
        taken[records[attack].unit.location.province] = (
            format_order(written),
            records[victim].sheet_pos,
        )
        assert evaluator.unit_dislodged(victim)
    assert taken == expected
    assert sorted(taken) == sorted(d.dislodger_origin or "" for d in result.dislodgements)
