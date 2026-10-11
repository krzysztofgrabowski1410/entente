"""Pin the shared residue machinery on a position that needs the full selection.

``common.resolve_residue`` and ``common.commit_cycle_policy`` decide what
happens to the orders one-sided propagation leaves undecided. The Angband
position here (BATC.B.12) and its variant without a clash each have exactly one
stable labelling, and the selection, handed their orders undecided, must
recompute that labelling exactly (docs/adjudication.tex Sec. 5.1). On the
variant, handing the set straight to the backup policy lands on a point that is
not a stable labelling at all: the overshoot needs Dislodges that clash with
nothing, and in the case file's own position the Dislodge from EIT ties its
clash with the victim and so takes part in no entry, where the policy and the
selection agree. The next tests state that overshoot and the enumeration bound
as contracts of the shared functions themselves; the case files carry the
engine-level outcomes.

The remaining tests hold positions without a convoy. In the first the victim of
a Dislodge backs the Bar that gates that Dislodge, and the rules decide it: a
support of a Bar is not lost with its giver, so the Bar stands at one strength
under both verdicts and only one of them is stable. The others hold the
opposite case -- a position with no stable labelling at all, which nothing but
the backup policy can settle -- and the controls that show which of its parts
the contradiction needs. Another holds the same knot with the victim moving
straight against its attacker, where the head-to-head clash decides it. The
last holds the knot with its Dislodge strong enough to win: one stable
labelling, which the resolver's propagation cannot reach, because nothing
case-splits on the victim's departure -- it hands all four orders to the
selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise, product

import pytest

from entente.engine.adjudicator import ENGINES, EngineName, Resolution
from entente.engine.adjudicator import resolver as resolver_module
from entente.engine.adjudicator.common import (
    AdjudicationError,
    Evaluator,
    build_tables,
    commit_cycle_policy,
    phase0,
    resolve_residue,
    selection,
)
from entente.engine.map import load_map
from entente.engine.orders import (
    RELOCATING_TYPES,
    Order,
    format_order,
    parse_order,
    parse_unit_spec,
)
from entente.engine.state import State, Unit
from entente.variants.beleriand import BELERIAND

GRAPH = load_map("beleriand")

ANG_UNITS: Mapping[str, Sequence[str]] = {
    "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANG", "A ANG", "3A ANG", "A EIT"],
    "GONDOLIN": ["A ANF"],
}
ANG_ORDERS: Mapping[str, Sequence[str]] = {
    "MORGOTH": [
        "A ANF H",
        "A ANF H",
        "A ANF H",
        "A ANG - ANF",
        "A ANG DIS GONDOLIN A ANF",
        "3A(M) ANG - ANF",
        "A EIT DIS GONDOLIN A ANF",
    ],
    "GONDOLIN": ["A ANF - EIT"],
}

# The unique stable labeling (derived in BATC.B.12's comment): the victim
# stays, the rival Dislodges stand each other off, the plain moves enter.
ANG_SELECTION = {
    "A ANF - EIT": False,
    "A ANG - ANF": True,
    "A ANG DIS GONDOLIN A ANF": False,
    "3A(M) ANG - ANF": True,
    "A EIT DIS GONDOLIN A ANF": False,
}

# The variant without a clash: an army holds Eithel Sirion, so the victim's
# Move bounces there, and the second Dislodge comes from the Fen of Serech.
ANG_HELD_UNITS: Mapping[str, Sequence[str]] = {
    "MORGOTH": ["A ANF", "A ANF", "A ANF", "A ANG", "A ANG", "3A ANG", "A EIT", "A FSE"],
    "GONDOLIN": ["A ANF"],
}
ANG_HELD_ORDERS: Mapping[str, Sequence[str]] = {
    "MORGOTH": [
        "A ANF H",
        "A ANF H",
        "A ANF H",
        "A ANG - ANF",
        "A ANG DIS GONDOLIN A ANF",
        "3A(M) ANG - ANF",
        "A EIT H",
        "A FSE DIS GONDOLIN A ANF",
    ],
    "GONDOLIN": ["A ANF - EIT"],
}

# Its unique stable labeling: the victim's Move fails against the holding
# army (1 against 1), so the victim stays; the two Dislodges stand each other
# off in its slot (attack 1 does not exceed prevent 1); the 3A and the plain
# move take the two free slots of Anfauglith.
ANG_HELD_SELECTION = {
    "A ANF - EIT": False,
    "A ANG - ANF": True,
    "A ANG DIS GONDOLIN A ANF": False,
    "3A(M) ANG - ANF": True,
    "A FSE DIS GONDOLIN A ANF": False,
}


def position(
    units: Mapping[str, Sequence[str]], orders: Mapping[str, Sequence[str]]
) -> tuple[State, dict[str, list[Order]]]:
    """Build a Beleriand position and its parsed order sheets."""
    state = State.of(
        "beleriand",
        "S25M",
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )
    sheets = {power: [parse_order(text) for text in sheet] for power, sheet in orders.items()}
    return state, sheets


def fresh_evaluator(
    units: Mapping[str, Sequence[str]], orders: Mapping[str, Sequence[str]]
) -> Evaluator:
    """Build an evaluator over a Beleriand position with the written orders bound."""
    state, sheets = position(units, orders)
    parsed = phase0(GRAPH, BELERIAND, state, sheets)
    tables = build_tables(GRAPH, BELERIAND, parsed.records)
    return Evaluator(GRAPH, BELERIAND, parsed.records, tables)


def move_statuses(evaluator: Evaluator) -> dict[str, bool | None]:
    """Read every move's current label off ``evaluator``, keyed by order text."""
    return {
        format_order(record.effective): evaluator.status[record.index]
        for record in evaluator.records
        if evaluator.is_move(record.index)
    }


def record_of(evaluator: Evaluator, text: str) -> int:
    """Return the index of the first record whose effective order reads ``text``."""
    return next(
        record.index for record in evaluator.records if format_order(record.effective) == text
    )


def stable_labellings(evaluator: Evaluator) -> list[dict[str, bool | None]]:
    """Return every labelling of the relocating orders that ``Eval`` gives back unchanged."""
    moves = evaluator.tables.moves
    stable = []
    for verdicts in product((False, True), repeat=len(moves)):
        for index, verdict in zip(moves, verdicts, strict=True):
            evaluator.status[index] = verdict
        if all(evaluator.eval_move(index) == evaluator.status[index] for index in moves):
            stable.append(move_statuses(evaluator))
    for index in moves:
        evaluator.status[index] = None
    return stable


def test_resolve_residue_commits_the_unique_stable_labeling() -> None:
    evaluator = fresh_evaluator(ANG_UNITS, ANG_ORDERS)
    resolve_residue(evaluator, range(len(evaluator.records)))
    assert move_statuses(evaluator) == ANG_SELECTION


def test_resolve_residue_commits_the_unique_stable_labeling_without_a_clash() -> None:
    evaluator = fresh_evaluator(ANG_HELD_UNITS, ANG_HELD_ORDERS)
    resolve_residue(evaluator, range(len(evaluator.records)))
    assert move_statuses(evaluator) == ANG_HELD_SELECTION


def test_commit_cycle_policy_alone_overshoots_the_selection() -> None:
    # The pre-fix resolver behavior: the backup policy seeded on the whole
    # stalled component. The all-succeed seed reads the victim as departed,
    # the prevent-1 entry class of three bounces over two slots, and the
    # synchronous downgrade drags the genuinely-succeeding "A ANG - ANF"
    # down with the Dislodges -- which is why only resolve_residue may decide
    # what reaches the policy.
    evaluator = fresh_evaluator(ANG_HELD_UNITS, ANG_HELD_ORDERS)
    commit_cycle_policy(evaluator, range(len(evaluator.records)))
    overshoot = move_statuses(evaluator)
    assert overshoot == {**ANG_HELD_SELECTION, "A ANG - ANF": False}
    # The policy's own contract still holds -- every committed SUCCEEDS is
    # eval-consistent under the final assignment...
    for record in evaluator.records:
        if evaluator.is_move(record.index) and evaluator.status[record.index]:
            assert evaluator.eval_move(record.index)
    # ...but the point is not a stable labelling: the bounced move would
    # succeed under it, so L != Eval(L) and it cannot be the selection.
    downgraded = next(
        record.index
        for record in evaluator.records
        if evaluator.is_move(record.index) and format_order(record.effective) == "A ANG - ANF"
    )
    assert evaluator.eval_move(downgraded) and evaluator.status[downgraded] is False


def test_resolve_residue_rejects_an_oversized_residue() -> None:
    # 21 mutual rivals into one Area exceed the 2^20 enumeration bound. The
    # guard is a backstop against a bug upstream, so it fails loudly: a
    # silently truncated search would return a wrong answer as a right one.
    swarm = {"MORGOTH": ["A ANG"] * 21, "GONDOLIN": ["A ANF"]}
    orders = {"MORGOTH": ["A ANG - ANF"] * 21, "GONDOLIN": ["A ANF H"]}
    evaluator = fresh_evaluator(swarm, orders)
    with pytest.raises(AdjudicationError, match="enumeration bound"):
        resolve_residue(evaluator, range(len(evaluator.records)))


# X1b-6: a rotation that a Dislodge closes. MORGOTH's army leaves Anfauglith
# for Dorthonion, FINROD's armies each move into the next province of a ring of
# capacity-1 provinces, and the last of them dislodges the MORGOTH army that
# was leaving -- entering the Area it vacates. Every order succeeds. The
# Dislodge makes the residue undecided as a whole (its victim's departure
# turns on the rotation and the rotation's last step on the Dislodge), so at
# 21 orders the selection is above the bound.
ROTATION = [
    "DNN",
    "AEL",
    "EGO",
    "DDI",
    "FON",
    "DIM",
    "GON",
    "FSE",
    "TOL",
    "SEW",
    "IVR",
    "NWB",
    "NFA",
    "NEV",
    "ELO",
    "CNI",
    "DLN",
    "MIT",
    "NEW",
    "EIT",
]


def rotation(
    armies_in_anf: int, ring: Sequence[str] = ROTATION
) -> tuple[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]], dict[str, bool]]:
    """Return the units, the orders and the all-succeed outcome of a ring closed by a Dislodge."""
    units = {
        "MORGOTH": ["A ANF"] * armies_in_anf,
        "FINROD": [f"A {province}" for province in ring],
    }
    leave = f"A ANF - {ring[0]}"
    orders = {
        "MORGOTH": [leave, *["A ANF H"] * (armies_in_anf - 1)],
        "FINROD": [
            *[f"A {a} - {b}" for a, b in pairwise(ring)],
            f"A {ring[-1]} DIS MORGOTH {leave}",
        ],
    }
    relocating = [leave, *orders["FINROD"]]
    return units, orders, dict.fromkeys(relocating, True)


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("armies_in_anf", [1, 6])
def test_a_rotation_of_21_orders_closed_by_a_dislodge_is_adjudicated(
    armies_in_anf: int, engine: EngineName
) -> None:
    units, orders, outcome = rotation(armies_in_anf)
    assert len(outcome) == 21
    assert engine_verdicts(units, orders, engine) == (outcome, 0)


def test_a_rotation_of_20_orders_closed_by_a_dislodge_is_adjudicated() -> None:
    ring = [province for province in ROTATION if province != "NFA"]
    units, orders, outcome = rotation(1, ring)
    assert len(outcome) == 20
    for engine in sorted(ENGINES):
        assert engine_verdicts(units, orders, engine) == (outcome, 0)


def test_the_all_succeed_completion_is_checked_above_the_bound() -> None:
    # Handed the residue whole, the selection takes the all-succeed completion
    # without enumerating: it is stable, so it is the one the shared policy
    # would reach (docs/adjudication.tex Sec. 6).
    units, orders, outcome = rotation(1)
    evaluator = fresh_evaluator(units, orders)
    assert len(evaluator.tables.moves) > selection.MAX_ENUMERATED
    resolve_residue(evaluator, range(len(evaluator.records)))
    assert move_statuses(evaluator) == outcome


KNOT_UNITS: Mapping[str, Sequence[str]] = {
    "FINROD": ["2A DNN", "A LAD"],
    "MORGOTH": ["A ANF", "2A ANF"],
}
KNOT_ORDERS: Mapping[str, Sequence[str]] = {
    "FINROD": ["2A DNN DIS MORGOTH A ANF", "A LAD S 2A DNN DIS MORGOTH A ANF"],
    "MORGOTH": [
        "A ANF S 2A ANF BAR FINROD 2A DNN - ANF",
        "2A ANF BAR FINROD 2A DNN - ANF",
    ],
}
KNOT_DISLODGE = "2A DNN DIS MORGOTH A ANF"
KNOT_BAR = "2A ANF BAR FINROD 2A DNN - ANF"
KNOT_BACKER = "A ANF S 2A ANF BAR FINROD 2A DNN - ANF"


def test_a_victim_backing_the_bar_against_its_own_dislodger_cannot_weaken_it() -> None:
    # The victim of the Dislodge supports the Bar that gates that Dislodge. The
    # one attack on it comes from the unit the Bar is directed against, so
    # nothing cuts its support; and a support of a Bar is not lost when its
    # giver is dislodged ("in the same moment in which that Unit is Dislodged,
    # it Supports the Bar, so both happen"). Were it lost, the dislodgement
    # would be decided by the Dislodge, the Dislodge by the Bar and the Bar by
    # the support: a circle with two stable verdicts and no convoy in it
    # (docs/adjudication.tex Sec. 3.3 and Sec. 8.4).
    evaluator = fresh_evaluator(KNOT_UNITS, KNOT_ORDERS)
    assert not any(evaluator.is_convoyed(record.index) for record in evaluator.records)
    dislodge = record_of(evaluator, KNOT_DISLODGE)
    bar = record_of(evaluator, KNOT_BAR)
    backer = record_of(evaluator, KNOT_BACKER)
    # The Bar stands at 2 + 1 whatever becomes of its backer, and stops an
    # attack of 2 + 1 because the Bar wins a tie: only FAILS is given back.
    for verdict in (False, True):
        evaluator.status[dislodge] = verdict
        assert evaluator.bar_value(bar) == 3
        assert evaluator.delivered_eff(backer) == evaluator.eff(backer) == 1
        assert evaluator.eval_move(dislodge) is False
    evaluator.status[dislodge] = None
    assert stable_labellings(evaluator) == [{KNOT_DISLODGE: False}]
    # One stable completion is no ambiguity: the selection takes it and the
    # backup policy is never asked.
    resolve_residue(evaluator, range(len(evaluator.records)))
    assert move_statuses(evaluator) == {KNOT_DISLODGE: False}
    assert not evaluator.unit_dislodged(backer)


# A supported Dislodge against a unit that is moving out of a full Area into a
# province whose own unit is moving into the dislodger's province, while a
# stronger plain entrant waits for the slot. If the Dislodge fails, its unit
# stays, the rotation stalls, the victim's Move fails, the victim stays and
# defends with its bare 1, and 1 + 1 beats it: the Dislodge succeeds. If it
# succeeds, the rotation runs, the victim has left, the Dislodge is a plain
# entry, and the one free slot goes to the 3A: the Dislodge fails. No Bar, no
# convoy, no support that is cut or void, and no head-to-head clash: the
# victim moves away from its attacker.
ZERO_UNITS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A DNN"],
    "GONDOLIN": ["A RIV", "A FSE"],
    "MORGOTH": [*["A ANF"] * 6, "3A ANG"],
}
ZERO_ORDERS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A DNN S A RIV DIS MORGOTH A ANF - FSE"],
    "GONDOLIN": ["A RIV DIS MORGOTH A ANF - FSE", "A FSE - RIV"],
    "MORGOTH": ["A ANF - FSE", *["A ANF H"] * 5, "3A(M) ANG - ANF"],
}
ZERO_ENTRANT = "3A(M) ANG - ANF"
NOTHING_MOVES = {
    "A RIV DIS MORGOTH A ANF - FSE": False,
    "A FSE - RIV": False,
    "A ANF - FSE": False,
    ZERO_ENTRANT: False,
}

# The same knot with the dislodger in Dorthonion and the rotation through Ladros.
FREE_UNITS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A RIV"],
    "GONDOLIN": ["A DNN", "A LAD"],
    "MORGOTH": [*["A ANF"] * 6, "3A ANG"],
}
FREE_ORDERS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A RIV S A DNN DIS MORGOTH A ANF - LAD"],
    "GONDOLIN": ["A DNN DIS MORGOTH A ANF - LAD", "A LAD - DNN"],
    "MORGOTH": ["A ANF - LAD", *["A ANF H"] * 5, "3A(M) ANG - ANF"],
}
FREE_NOTHING_MOVES = {
    "A DNN DIS MORGOTH A ANF - LAD": False,
    "A LAD - DNN": False,
    "A ANF - LAD": False,
    ZERO_ENTRANT: False,
}


def engine_verdicts(
    units: Mapping[str, Sequence[str]], orders: Mapping[str, Sequence[str]], engine: EngineName
) -> tuple[dict[str, bool], int]:
    """Adjudicate on one engine; return the relocating verdicts and the dislodgement count."""
    state, sheets = position(units, orders)
    result = ENGINES[engine](BELERIAND, state, sheets, graph=GRAPH)
    relocating = {
        format_order(order)
        for sheet in sheets.values()
        for order in sheet
        if isinstance(order, RELOCATING_TYPES)
    }
    verdicts = {
        row.text: row.resolution is Resolution.SUCCEEDS
        for row in result.resolutions
        if row.text in relocating
    }
    return verdicts, len(result.dislodgements)


@pytest.mark.parametrize(
    ("units", "orders", "outcome"),
    [(ZERO_UNITS, ZERO_ORDERS, NOTHING_MOVES), (FREE_UNITS, FREE_ORDERS, FREE_NOTHING_MOVES)],
    ids=["through-the-fen", "through-ladros"],
)
def test_a_position_without_a_convoy_can_have_no_stable_labelling(
    units: Mapping[str, Sequence[str]],
    orders: Mapping[str, Sequence[str]],
    outcome: Mapping[str, bool],
) -> None:
    evaluator = fresh_evaluator(units, orders)
    assert not any(evaluator.is_convoyed(record.index) for record in evaluator.records)
    assert len(evaluator.tables.moves) == 4
    assert stable_labellings(evaluator) == []
    # With nothing stable to choose from, the all-succeed policy seeds all
    # four and Eval refutes the Dislodge, then the rotation, then the entrant:
    # nothing moves. That is what the policy does, not a verdict the rules
    # reach -- under it the Dislodge would beat the victim that stayed.
    resolve_residue(evaluator, range(len(evaluator.records)))
    assert move_statuses(evaluator) == outcome
    dislodge = next(text for text in outcome if " DIS " in text)
    assert evaluator.eval_move(record_of(evaluator, dislodge))


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_both_engines_leave_the_unstable_position_unmoved(engine: EngineName) -> None:
    assert engine_verdicts(ZERO_UNITS, ZERO_ORDERS, engine) == (NOTHING_MOVES, 0)
    assert engine_verdicts(FREE_UNITS, FREE_ORDERS, engine) == (FREE_NOTHING_MOVES, 0)


# Each control takes one part of the contradiction away and has exactly one
# stable labelling. Without the 3A the rotation goes through -- and "all fail"
# is not stable either, because 1 + 1 beats the bare 1 of a victim that
# stayed. With five armies in the Area there is room for both entrants.
# Without the support the Dislodge never beats anybody, 1 against 1.
CONTROLS: Mapping[str, tuple[Mapping[str, Sequence[str]], Mapping[str, Sequence[str]], bool]] = {
    "no-stronger-entrant": (
        {**ZERO_UNITS, "MORGOTH": ["A ANF"] * 6},
        {**ZERO_ORDERS, "MORGOTH": ["A ANF - FSE", *["A ANF H"] * 5]},
        True,
    ),
    "a-free-slot": (
        {**ZERO_UNITS, "MORGOTH": [*["A ANF"] * 5, "3A ANG"]},
        {**ZERO_ORDERS, "MORGOTH": ["A ANF - FSE", *["A ANF H"] * 4, "3A(M) ANG - ANF"]},
        True,
    ),
    "no-support": (ZERO_UNITS, {**ZERO_ORDERS, "FINROD": ["A DNN H"]}, False),
}


@pytest.mark.parametrize("engine", sorted(ENGINES))
@pytest.mark.parametrize("control", sorted(CONTROLS))
def test_each_part_of_the_unstable_position_is_needed(control: str, engine: EngineName) -> None:
    units, orders, everything_moves = CONTROLS[control]
    evaluator = fresh_evaluator(units, orders)
    stable = stable_labellings(evaluator)
    assert len(stable) == 1
    assert set(stable[0].values()) == {everything_moves}
    verdicts, dislodged = engine_verdicts(units, orders, engine)
    assert verdicts == stable[0]
    assert dislodged == 0


# The knot loses its loop where the victim moves straight against its attacker:
# the Dislodge and that Move clash head-to-head, the stronger Dislodge wins
# whatever the 3A does, and the position has exactly one stable labelling.
CLASH_UNITS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A DNN"],
    "GONDOLIN": ["A RIV"],
    "MORGOTH": [*["A ANF"] * 6, "3A ANG"],
}
CLASH_ORDERS: Mapping[str, Sequence[str]] = {
    "FINROD": ["A DNN S A RIV DIS MORGOTH A ANF - RIV"],
    "GONDOLIN": ["A RIV DIS MORGOTH A ANF - RIV"],
    "MORGOTH": ["A ANF - RIV", *["A ANF H"] * 5, "3A(M) ANG - ANF"],
}


@pytest.mark.parametrize("engine", sorted(ENGINES))
def test_a_victim_moving_against_its_dislodger_meets_it_head_to_head(engine: EngineName) -> None:
    evaluator = fresh_evaluator(CLASH_UNITS, CLASH_ORDERS)
    stable = stable_labellings(evaluator)
    expected = {
        "A RIV DIS MORGOTH A ANF - RIV": True,
        "A ANF - RIV": False,
        ZERO_ENTRANT: False,
    }
    assert stable == [expected]
    assert engine_verdicts(CLASH_UNITS, CLASH_ORDERS, engine) == (expected, 1)


# The knot with its Dislodge also supported by a FINGOLFIN 2A: 1 + 1 + 2 = 4
# beats the 3A's 3 for the slot, so only "the Dislodge succeeds" is consistent.
STALL_UNITS: Mapping[str, Sequence[str]] = {**ZERO_UNITS, "FINGOLFIN": ["2A NEW"]}
STALL_ORDERS: Mapping[str, Sequence[str]] = {
    **ZERO_ORDERS,
    "FINGOLFIN": ["2A NEW S A RIV DIS MORGOTH A ANF - FSE"],
}


def test_the_resolver_can_stall_on_a_position_with_one_stable_labelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stable = stable_labellings(fresh_evaluator(STALL_UNITS, STALL_ORDERS))
    assert len(stable) == 1
    residues: list[set[str]] = []
    real = resolver_module.resolve_residue

    def spy(evaluator: Evaluator, members: Sequence[int]) -> None:
        residues.append(
            {
                format_order(evaluator.records[index].effective)
                for index in members
                if evaluator.is_move(index) and evaluator.status[index] is None
            }
        )
        real(evaluator, members)

    monkeypatch.setattr(resolver_module, "resolve_residue", spy)
    verdicts, _ = engine_verdicts(STALL_UNITS, STALL_ORDERS, "resolver")
    assert residues == [set(NOTHING_MOVES)]
    assert verdicts == stable[0]
