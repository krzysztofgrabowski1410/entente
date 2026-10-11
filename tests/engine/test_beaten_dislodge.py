"""Pin that a Dislodge beaten in its clash makes no province embattled.

A Dislodge that loses or ties its head-to-head clash with its victim's order
fails wholly: the clash is the only contest it takes part in (docs/adjudication.tex
Sec. 4.2; the rulebook: "A Dislodge that loses its clash fails, and it does not
stand in the way of the other Units entering the Area"). The BATC cases pin its
part in the entry and in the victim's slot; the case files carry no embattled
provinces, so the third part -- it counts toward no standoff (Sec. 9) -- is
pinned here, beside a control in which the Dislodge fails without a clash and
does embattle the Area. The last test holds the twin boards of BATC.J.11 and
J.12: an unnamed Dislodge meets the twins moving at it before any twin is dealt
(Sec. 4.4, rule 6), so beaten there it acts exactly as the same Dislodge naming
the moving twin's order.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator import EngineName, adjudicate
from entente.engine.map.loader import load_map
from entente.engine.orders import parse_order, parse_unit_spec
from entente.engine.state import State, Unit
from entente.variants.beleriand import BELERIAND

ENGINES: tuple[EngineName, ...] = ("resolver", "enumerator")
GRAPH = load_map("beleriand")


def _adjudicate(
    engine: EngineName, morgoth_order: str, feanorian_orders: list[str]
) -> frozenset[str]:
    """Adjudicate the full-Lothlann position and return its embattled provinces.

    Lothlann holds six FEANORIAN armies, so DORIATH's army from Aeluin bounces
    whatever the Dislodge does; MORGOTH's six armies fill Anfauglith.
    """
    units = {
        "MORGOTH": ["A ANF"] * 6,
        "FEANORIANS": ["A LOT"] * 6,
        "DORIATH": ["A AEL"],
    }
    orders = {
        "MORGOTH": [morgoth_order] + ["A ANF H"] * 5,
        "FEANORIANS": feanorian_orders,
        "DORIATH": ["A AEL - LOT"],
    }
    state = State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )
    sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
    result = adjudicate(BELERIAND, state, sheets, graph=GRAPH, engine=engine)
    assert result.resolution_of("A AEL - LOT").value == "fails"
    return result.embattled


@pytest.mark.parametrize("engine", ENGINES)
def test_a_dislodge_that_ties_its_clash_embattles_nothing(engine: EngineName) -> None:
    # The Dislodge and the army it names clash and tie, 1 against 1: both
    # fail. DORIATH's army bounces off the full Area alone, so Lothlann saw
    # one failed entry, not a standoff.
    embattled = _adjudicate(
        engine,
        "A ANF DIS FEANORIANS A LOT - ANF",
        ["A LOT - ANF"] + ["A LOT H"] * 5,
    )
    assert "LOT" not in embattled


@pytest.mark.parametrize("engine", ENGINES)
def test_a_dislodge_that_fails_without_a_clash_still_embattles_the_area(engine: EngineName) -> None:
    # The control: every army in Lothlann holds, so the Dislodge meets no
    # clash and fails against the stack's defence. It counts as a Move does,
    # and with DORIATH's army it makes two failed entries: Lothlann is
    # embattled.
    embattled = _adjudicate(engine, "A ANF DIS FEANORIANS A LOT", ["A LOT H"] * 6)
    assert "LOT" in embattled


TWIN_UNITS = {"FEANORIANS": ["A LOT", "A LOT", "2A LOT"], "MORGOTH": ["2A ANF", "2A LAD", "2A AEL"]}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "feanorian_orders",
    [
        ["A LOT - ANF", "A LOT - ANF", "2A LOT S A LOT - ANF"],  # BATC.J.11
        ["A LOT - ANF", "A LOT H", "2A LOT S A LOT - ANF"],  # BATC.J.12
    ],
)
def test_a_dislodge_beaten_by_twins_acts_alike_named_or_unnamed(
    engine: EngineName, feanorian_orders: list[str]
) -> None:
    # The Anfauglith Dislodge loses its clash with the supported twin moving
    # into Anfauglith. Written unnamed it names every twin, so it meets that
    # twin before any twin is dealt; written naming that twin's Move it meets
    # it by name. Either way it takes part in nothing else, and the board
    # comes out the same.
    def outcome(anf_order: str) -> tuple[object, object, object]:
        orders = {
            "FEANORIANS": feanorian_orders,
            "MORGOTH": [
                anf_order,
                "2A LAD DIS FEANORIANS A LOT",
                "2A AEL DIS FEANORIANS A LOT",
            ],
        }
        state = State.of(
            "beleriand",
            "S25M",
            [
                Unit(power, *parse_unit_spec(spec))
                for power, specs in TWIN_UNITS.items()
                for spec in specs
            ],
        )
        sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
        result = adjudicate(BELERIAND, state, sheets, graph=GRAPH, engine=engine)
        others = [row.resolution for row in result.resolutions if not row.text.startswith("2A ANF")]
        return others, result.new_state, result.embattled

    assert outcome("2A ANF DIS FEANORIANS A LOT") == outcome("2A ANF DIS FEANORIANS A LOT - ANF")
