"""Make each of the four Beleriand support bans say why it fired.

A support voided by a power restriction is otherwise indistinguishable, on the
board, from one that was merely cut: both resolve as a bare failure. So
``_support_forbidden`` returns the banning rule's own sentence rather than
``True``, ``MovementPlan.reasons`` carries it, and
``GameHooks.merge_resolutions`` copies it onto the resolution row.

The cases drive the public :class:`Game` facade, which is the constructor the
Layer-2 case runner uses too, so they read the sentence where a player meets
it -- on the executed record -- and not only where it is produced. That
attaching a reason moves no outcome is the case suites' business: every
position below is a legality case they already cover, and this file asserts
the reason alone.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.common import Resolution
from entente.engine.orders import parse_order
from entente.variants.beleriand.layer2 import (
    BAN_ANFAUGLITH,
    BAN_ELVES_SUPPORT_MORGOTH,
    BAN_FEANORIAN_MUTUAL,
    BAN_MORGOTH_SUPPORTS_ELVES,
)
from tests.batc.l2_runner import L2Case, build_game

# (label, units, order sheets, the banned support's text, its expected reason).
# One per ban, in the order ``_support_forbidden`` tests them. Ban 3 appears
# twice -- once written by the Feanorians, once written against them -- because
# the rule is mutual and the code reaches it down two different branches.
BANS = [
    pytest.param(
        {"FINROD": ["A LAD"], "MORGOTH": ["A AEL"]},
        {"MORGOTH": ["A AEL H"], "FINROD": ["A LAD S MORGOTH A AEL"]},
        "A LAD S MORGOTH A AEL",
        BAN_ELVES_SUPPORT_MORGOTH,
        id="elves-support-morgoth",
    ),
    pytest.param(
        {"FINROD": ["A LAD"], "MORGOTH": ["A AEL"]},
        {"FINROD": ["A LAD H"], "MORGOTH": ["A AEL S FINROD A LAD"]},
        "A AEL S FINROD A LAD",
        BAN_MORGOTH_SUPPORTS_ELVES,
        id="morgoth-supports-elves",
    ),
    pytest.param(
        {"FEANORIANS": ["A DNN"], "FINROD": ["A LAD"]},
        {"FINROD": ["A LAD H"], "FEANORIANS": ["A DNN S FINROD A LAD"]},
        "A DNN S FINROD A LAD",
        BAN_FEANORIAN_MUTUAL,
        id="feanorians-support-elves",
    ),
    pytest.param(
        {"FEANORIANS": ["A DNN"], "FINROD": ["A LAD"]},
        {"FEANORIANS": ["A DNN H"], "FINROD": ["A LAD S FEANORIANS A DNN"]},
        "A LAD S FEANORIANS A DNN",
        BAN_FEANORIAN_MUTUAL,
        id="elves-support-feanorians",
    ),
    pytest.param(
        # MEN is not elvish, so bans 1-3 all pass it through; only the
        # power-level Anfauglith clause reaches it, and it reaches every unit
        # of the power -- the supported army stands in ELA, not ANF.
        {"FEANORIANS": ["A LAD"], "MEN": ["A ELA", "A ANF"]},
        {"MEN": ["A ELA H", "A ANF H"], "FEANORIANS": ["A LAD S MEN A ELA"]},
        "A LAD S MEN A ELA",
        BAN_ANFAUGLITH,
        id="anfauglith-power-wide",
    ),
]


def _run(units: dict[str, list[str]], orders: dict[str, list[str]]):
    case = L2Case(
        id="P25e",
        title="support ban reasons",
        section="P25e",
        seed=0,
        start={"phase": "S25M", "units": units},
        script=[],
        expect={},
    )
    game = build_game(case)
    for power, texts in sorted(orders.items()):
        game.submit(power, [parse_order(text) for text in texts])
    return game.adjudicate()


@pytest.mark.parametrize("units,orders,text,expected", BANS)
def test_a_banned_support_carries_its_rule(
    units: dict[str, list[str]],
    orders: dict[str, list[str]],
    text: str,
    expected: str,
) -> None:
    """Name the banning rule on the row of every support the bans void.

    The support fails either way; what the row adds is the rule that voided
    it, in the rulebook's own words.
    """
    record = _run(units, orders)
    row = next(r for r in record.resolutions if r.text == text)
    assert row.resolution is Resolution.FAILS
    assert row.reason == expected


def test_an_ordinary_failure_carries_no_reason() -> None:
    """Annotate only the failures the board itself cannot explain.

    A bounce is visible on the map, so its row carries no reason, and neither
    does any order that succeeded.
    """
    record = _run(
        {"FINROD": ["A LAD"], "MORGOTH": ["A AEL"]},
        {"FINROD": ["A LAD - AEL"], "MORGOTH": ["A AEL - LAD"]},
    )
    assert [r.reason for r in record.resolutions] == [None, None]
    assert all(r.resolution is Resolution.FAILS for r in record.resolutions)  # a head-on swap


def test_a_legal_support_is_untouched() -> None:
    """Leave a support none of the four bans reaches entirely alone.

    Men supporting Men is nobody's ban, so the predicate lets it through and
    its row carries no reason.
    """
    record = _run(
        {"MEN": ["A ELA", "A ERI"]},
        {"MEN": ["A ERI H", "A ELA S MEN A ERI"]},
    )
    row = next(r for r in record.resolutions if r.text == "A ELA S MEN A ERI")
    assert row.resolution is Resolution.SUCCEEDS and row.reason is None
