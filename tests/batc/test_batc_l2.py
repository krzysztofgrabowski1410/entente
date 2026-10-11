"""BATC-L2 -- the Beleriand Layer-2 game-scenario cases, an original artifact.

Hand-derived from ``BeleriandInstruction.docx`` exactly like the movement
BATC, never from another engine's tests, but pitched one level up: these
cases drive the public ``Game`` facade -- end-of-phase steps, seeded timers,
season boundaries, winters, knowledge grants, and the victory layer.
Sections: K knowledge, T transform, M masks, D devastation, U Ulmo's wrath,
G Glaurung & the one-way provinces, E elven ownership & doubles, W winter,
V victory & the Valar, F power restrictions (the elven/Feanorian support
bans, Teleri disorder, the Gondolin shield), S special-unit designators.
"""

from pathlib import Path

import pytest

from tests.batc.l2_runner import L2Case, load_all, run_case

CASES_DIR = Path(__file__).parent / "cases_l2"

CASES = load_all(CASES_DIR)


@pytest.mark.batc
def test_the_l2_suite_covers_every_mechanic() -> None:
    """Pin the exact per-section counts, so a hand-dropped case cannot pass silently."""
    # Exact counts: a loose >= floor would let a dropped case through, and
    # the guard carries the batc mark so `pytest -m batc` runs it.
    by_section: dict[str, int] = {}
    for case in CASES:
        prefix = case.id.rsplit(".", 1)[0]
        by_section[prefix] = by_section.get(prefix, 0) + 1
    assert by_section == {
        # D.1-D.7 devastation and passability; D.8-D.15 devastation dislodges
        # the unmasked: D.8/D.9 a co-located foreign unmasked unit falls and
        # retreats out, D.10 claim discipline (masked twin stands, unmasked
        # twin falls), D.11 Morgoth's own units are exempt, D.12 a unit
        # dislodged elsewhere still cannot retreat into fresh devastation,
        # D.13 the mask-acceptance gate, D.14 two fallen units contest one
        # retreat slot, D.15 a same-phase arrival falls too.
        "BATC-L2.D": 15,
        # E.1-E.11 the elven two-step and the double bindings; E.12-E.14 a
        # supply point funds exactly one unit; E.15-E.17 a two-Centre province
        # carries two Double Units, one per Centre.
        "BATC-L2.E": 17,
        # F.15-F.17: the Gondolin shield falls on the written order, whether
        # or not the move succeeds.
        "BATC-L2.F": 17,
        "BATC-L2.G": 6,
        "BATC-L2.K": 6,
        # M.34/M.35: Men and the Naugrim may hand Morgoth the Girdle passage
        # in writing; every elvish giver is refused.
        "BATC-L2.M": 35,
        # S.33/S.34: a support naming the special that did not make the Move
        # matches no order, and fails.
        "BATC-L2.S": 34,
        "BATC-L2.T": 7,
        "BATC-L2.U": 6,
        # V.7-V.14 the Oath of the Second: V.7/V.8 the bound and the free
        # second place, V.9/V.10 the price on the bar (V.10 the year-0
        # fall-through to the Valar), V.11 the pending-sheet gate, V.12 the
        # priced break for coexistence, V.13 the refusal matrix, V.14 an
        # Oath to a power with no units; V.15/V.16 the free break: the
        # implicit dislodge that earns it, and the supported Dislodge that
        # reaches Morgoth inside an Area.
        "BATC-L2.V": 16,
        # W.5-W.9 the rulebook's own civil-disorder law: W.5 the owned-Centre
        # seed, W.6 the name-only tiebreak (no blanket fleet priority), W.7
        # the two-removal sorted list, W.8 fleets first when no coastal
        # Centre is owned, W.9 an army distance crossing water.
        "BATC-L2.W": 9,
    }


@pytest.mark.batc
@pytest.mark.parametrize("case", CASES, ids=[case.pytest_id for case in CASES])
def test_batc_l2_case(case: L2Case) -> None:
    """Drive one scenario through the game and check its expectations."""
    run_case(case)
