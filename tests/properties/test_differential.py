"""Assert the two adjudication engines agree on every generated position.

Both share Phase 0 and Phase 1, the ``Eval`` step and the assembly of the
result (docs/adjudication.tex Sec. 5.1), so a disagreement here isolates a
defect in one of the two labelling searches themselves: the staged fixpoint's
propagation and residue policy against the stable-labelling enumeration.
Equality is asserted on all four observables -- per-order resolutions,
dislodgement records, the successor state and the embattled provinces.

Four generators feed it, from ``strategies``: the general mostly-legal
sampler, the convoy-rich sampler that concentrates probability on the
entanglement territory of viability knots, convoyed swaps and paradox shapes
(docs/adjudication.tex Sec. 5.2), the Beleriand sampler that stacks Areas
and rolls Dislodge and Bar, and the two-beach sampler that convoys Beleriand
armies onto a coast with one marked and one free sea, where the convoy toll
depends on the chains that were ordered (docs/adjudication.tex Sec. 2). For
the variant this differential is the only correctness oracle that exists -- no
external engine implements those rules -- so it runs at the same budget as the
standard profiles. The fast profiles run in CI; the slow-marked deep profiles
are the local and nightly soak, upwards of 2k examples across the tests.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, example, given, settings

from entente.engine.adjudicator import adjudicate
from entente.engine.map.graph import MapGraph
from entente.engine.orders import Order, parse_order, parse_unit_spec
from entente.engine.rules import STANDARD, Ruleset
from entente.engine.state import State, Unit
from entente.variants.beleriand import BELERIAND
from tests.engine.two_beach import TWO_BEACH, crossing
from tests.properties.strategies import (
    BELERIAND_GRAPH,
    GRAPH,
    beleriand_position_and_sheets,
    convoy_rich_position_and_sheets,
    position_and_sheets,
    two_beach_position_and_sheets,
)

SUPPRESS = (HealthCheck.too_slow, HealthCheck.data_too_large)

Position = tuple[State, dict[str, list[Order]]]


def _position(phase: str, units: dict[str, list[str]], sheets: dict[str, list[str]]) -> Position:
    state = State.of(
        "beleriand",
        phase,
        [Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs],
    )
    return state, {power: [parse_order(text) for text in sheet] for power, sheet in sheets.items()}


# An S25M position the soak caught, kept as a permanent example: FINROD's
# entry into a full ANF fails in every stable labelling and the 3A dislodges
# it, which leaves a live ANF<->EIT vacancy cycle plus "A ANG - ANF" riding
# the freed capacity. Re-seeding a unanimously-failing order as a success
# during the residue commit poisons that cascade and unravels the cycle, so
# the backup policy must be scoped to orders the propagation actually left
# open. The case files carry the trimmed core; this example replays the whole
# caught position through both engines.
#
# The position rides LAD-ANF rather than the AEL-ANF edge it was caught on,
# because that edge is not on the board: FINROD's entrant and holder swap
# posts, LAD <-> AEL, and the three ANF-out movers follow. LAD-ANF carries the
# identical residue topology -- a unanimously failing entrant, a 3A dislodger,
# and a live 2-cycle with a rider -- so the shape is preserved edge for edge.
ANF_RESIDUE_SCOPE = _position(
    "S25M",
    {
        "FINROD": ["A LAD", "A AEL", "2A NAR", "A NWB", "A RON", "A TOL"],
        "MORGOTH": [
            "A ANF",
            "A ANF",
            "A ANF",
            "A ANF",
            "A ANF",
            "3A ANF",
            "A ANG",
            "A ANG",
            "A EIT",
        ],
        "NAUGRIM": ["A BEL", "A DOL", "A NOG"],
        "TELERI": ["A BRI", "F EGL", "2F SFA"],
        "FINGOLFIN": ["2A CNI", "A ELO", "A VIN"],
        "GONDOLIN": ["A DIM", "A DNN", "2A GON"],
        "MEN": ["A ELA"],
        "DORIATH": ["A FON", "2A MEN", "A NEB"],
        "FEANORIANS": ["2A GEL", "A HID", "A HIG", "2A MMA", "A NEL", "A THA"],
        "OSSIRIAND": ["A MOS", "A NOS", "A SOS"],
    },
    {
        "FINROD": ["A LAD - ANF", "A AEL H", "2A NAR S A RON", "A NWB H", "A RON H", "A TOL H"],
        "MORGOTH": [
            "A ANF H",
            "A ANF H",
            "A ANF - EIT",
            "A ANF - LAD",
            "A ANF - LAD",
            "3A ANF - LAD",
            "A ANG H",
            "A ANG - ANF",
            "A EIT - ANF",
        ],
        "NAUGRIM": ["A BEL H", "A DOL H", "A NOG H"],
        "TELERI": ["A BRI H", "F EGL - BNI", "2F SFA H"],
        "FINGOLFIN": ["2A CNI H", "A ELO - CNI", "A VIN H"],
        "GONDOLIN": ["A DIM S 2A GON", "A DNN H", "2A GON H"],
        "MEN": ["A ELA - BEL"],
        "DORIATH": ["A FON - DDI", "2A MEN H", "A NEB H"],
        "FEANORIANS": ["2A GEL H", "A HID H", "A HIG H", "2A MMA S 2A GEL", "A NEL H", "A THA H"],
        "OSSIRIAND": ["A MOS S A NOS", "A NOS H", "A SOS H"],
    },
)


# A second caught S25M position, reproduced by a targeted order fuzz over its
# unit arrangement. GONDOLIN's army leaves a nearly full ANF for EIT
# while two MORGOTH Dislodges name it and two plain ANG moves ride the
# capacity. The component has exactly one stable labelling: the victim's Move
# and the EIT Dislodge clash head-to-head and tie, so both fail and the beaten
# Dislodge takes part in nothing else; the ANG Dislodge fails against the
# victim's hold; the plain moves enter. The resolver's propagation decides it,
# but the enumerator hands the whole component to the selection, which must
# recompute it rather than fall straight through to a backup (the overshoot a
# backup policy makes is shown on a variant without a clash in
# tests/engine/test_residue_selection.py). The case files
# carry the trimmed core; this example replays the caught arrangement whole,
# FINROD's cuttable support-hold, the void Bar and the mismatched
# support-holds included.
#
# As above, FINROD moves AEL -> LAD because the AEL-ANF edge is not on the
# board, and the Bar's named-but-unwritten move follows it. The Bar stays void
# by design and the support-hold stays valid and uncut, so the
# unique-labelling component is untouched.
ANG_RESIDUE_SELECTION = _position(
    "S25M",
    {
        "MORGOTH": [
            "A ANF",
            "A ANF",
            "A ANF",
            "A ANG",
            "A ANG",
            "3A ANG",
            "A EIT",
            "A FSE",
            "A LOT",
        ],
        "GONDOLIN": ["A ANF"],
        "FINROD": ["A LAD"],
    },
    {
        "FINROD": ["A LAD S MORGOTH A ANF"],
        "GONDOLIN": ["A ANF - EIT"],
        "MORGOTH": [
            "A ANF BAR FINROD A LAD - ANF",
            "A ANF H",
            "A ANF S A FSE",
            "A ANG - ANF",
            "A ANG DIS GONDOLIN A ANF",
            "3A ANG - ANF",
            "A EIT DIS GONDOLIN A ANF",
            "A FSE S MORGOTH A ANG - ANF",
            "A LOT H",
        ],
    },
)


def assert_engines_agree(ruleset: Ruleset, graph: MapGraph, pos: Position) -> None:
    """Adjudicate one position under both engines and assert all four observables match."""
    state, sheets = pos
    by_resolver = adjudicate(ruleset, state, sheets, graph=graph, engine="resolver")
    by_enumerator = adjudicate(ruleset, state, sheets, graph=graph, engine="enumerator")
    assert by_resolver.resolutions == by_enumerator.resolutions
    assert by_resolver.dislodgements == by_enumerator.dislodgements
    assert by_resolver.new_state == by_enumerator.new_state
    assert by_resolver.embattled == by_enumerator.embattled


@settings(max_examples=100, deadline=None, suppress_health_check=SUPPRESS)
@given(position_and_sheets())
def test_resolver_equals_enumerator(pos: Position) -> None:
    assert_engines_agree(STANDARD, GRAPH, pos)


@settings(max_examples=100, deadline=None, suppress_health_check=SUPPRESS)
@given(convoy_rich_position_and_sheets())
def test_resolver_equals_enumerator_convoy_rich(pos: Position) -> None:
    assert_engines_agree(STANDARD, GRAPH, pos)


@settings(max_examples=100, deadline=None, suppress_health_check=SUPPRESS)
@example(ANF_RESIDUE_SCOPE)
@example(ANG_RESIDUE_SELECTION)
@given(beleriand_position_and_sheets())
def test_resolver_equals_enumerator_beleriand(pos: Position) -> None:
    assert_engines_agree(BELERIAND, BELERIAND_GRAPH, pos)


@pytest.mark.slow
@settings(suppress_health_check=SUPPRESS)  # max_examples/deadline: active profile (conftest.py)
@given(position_and_sheets())
def test_resolver_equals_enumerator_deep(pos: Position) -> None:
    assert_engines_agree(STANDARD, GRAPH, pos)


@pytest.mark.slow
@settings(suppress_health_check=SUPPRESS)  # max_examples/deadline: active profile (conftest.py)
@given(convoy_rich_position_and_sheets())
def test_resolver_equals_enumerator_convoy_rich_deep(pos: Position) -> None:
    assert_engines_agree(STANDARD, GRAPH, pos)


@pytest.mark.slow
@settings(suppress_health_check=SUPPRESS)  # max_examples/deadline: active profile (conftest.py)
@example(ANF_RESIDUE_SCOPE)  # an explicit example replays the caught position on every run
@example(ANG_RESIDUE_SELECTION)  # likewise, rather than waiting for the generator to redraw it
@given(beleriand_position_and_sheets())
def test_resolver_equals_enumerator_beleriand_deep(pos: Position) -> None:
    assert_engines_agree(BELERIAND, BELERIAND_GRAPH, pos)


# The two-beach crossings the toll's reading is decided on, replayed on every
# run: an idle fleet off the free beach, both routes ordered, and the free
# route disrupted in the same phase.
@settings(max_examples=200, deadline=None, suppress_health_check=SUPPRESS)
@example(crossing(("SEA", "SEC", "SED")))
@example(crossing(("SEA", "SEC", "SEB", "SED")))
@example(crossing(("SEA", "SEC", "SEB", "SED"), beach_fleet_attacked=True))
@given(two_beach_position_and_sheets())
def test_resolver_equals_enumerator_two_beach(pos: Position) -> None:
    assert_engines_agree(BELERIAND, TWO_BEACH, pos)


@pytest.mark.slow
@settings(suppress_health_check=SUPPRESS)  # max_examples/deadline: active profile (conftest.py)
@given(two_beach_position_and_sheets())
def test_resolver_equals_enumerator_two_beach_deep(pos: Position) -> None:
    assert_engines_agree(BELERIAND, TWO_BEACH, pos)
