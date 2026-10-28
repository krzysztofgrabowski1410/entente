"""Pin the point two twin doubles free: the double bound to the older centre goes.

The rulebook's "Double Units": "When a Unit is destroyed, its point is freed
(of two twins, the point of the Supply Centre owned longer)". Two twin doubles
cannot be told apart, so a death among them removes the one bound to the
centre owned longer -- unless the death is the capture of a double's own
centre, which names that double. A death in the retreats is such a death too:
of two dislodged twins, the one that retreats keeps the newer centre.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.common import Resolution
from entente.engine.game import Game
from entente.engine.map import Location, UnitKind, load_map
from entente.engine.orders import (
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
    parse_unit_spec,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import BeleriandHooks
from tests.batc.l2_runner import L2Case, build_game


def _twin_doubles(
    phase: Phase, bindings: tuple[str, str], masks: tuple[int, int] = (0, 0)
) -> tuple[BeleriandHooks, State, dict[str, str]]:
    """Return FINROD's two twin doubles in Lothlann, bound in uid order to ``bindings``.

    ``masks`` is the number of Dwarven masks each double carries, in uid order.

    FINROD owns NAR (since the start) and RON (since the second Autumn); each
    double holds the point of the centre it is bound to, as two Builds in
    different winters would leave them, and a third army, A NWB, finds no
    point to spare. The bindings are set by hand.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    double = Unit("FINROD", UnitKind.ARMY, Location("LOT"), 2)
    state = State.of(
        "beleriand", phase.tag, [double, double, Unit("FINROD", UnitKind.ARMY, Location("NWB"))]
    )
    ownership = {"NAR": "FINROD", "RON": "FINROD"}
    hooks.register_start(state, ownership)
    hooks.owner_since["RON"] = 2
    doubles = [entry for entry in hooks.ledger.alive() if entry.strength == 2]
    for entry, centre, carried in zip(doubles, bindings, masks, strict=True):
        entry.bound_to = centre
        entry.masks = carried
    for entry in hooks.ledger.alive():
        entry.supplied_by = None
    hooks._reconcile_spent(state, ownership)
    assert hooks.spent_supply == {"NAR": 1, "RON": 1}
    return hooks, state, ownership


def test_of_two_twin_doubles_the_one_bound_to_the_older_centre_is_disbanded() -> None:
    """Free NAR, the centre owned longer, whichever twin double the ledger created first.

    A winter owes one removal and the sheet writes ``Disband 2A LOT``. The two
    doubles cannot be told apart, so the rulebook's twin clause decides: NAR's
    point is freed, the double left standing is RON's, and the unsupplied
    A NWB takes the freed point.
    """
    graph = load_map("beleriand")
    phase = Phase(24, Season.WINTER, PhaseKind.ADJUSTMENT, year_step=-1)
    for bindings in (("RON", "NAR"), ("NAR", "RON")):
        hooks, state, ownership = _twin_doubles(phase, bindings)
        game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
        assert game.build_quota("FINROD") == -1
        game.submit("FINROD", [parse_adjustment_order("Disband 2A LOT")])
        record = game.adjudicate()
        assert [row.resolution for row in record.resolutions] == [Resolution.SUCCEEDS]
        assert sorted(str(unit) for unit in game.state.units) == ["2A LOT", "A NWB"]
        alive = list(hooks.ledger.alive())
        assert [entry.bound_to for entry in alive if entry.strength == 2] == ["RON"], bindings
        assert [entry.supplied_by for entry in alive if entry.strength == 1] == ["NAR"], bindings


def test_a_captured_centre_disbands_its_own_twin_double_only() -> None:
    """Disband the double bound to a captured centre, never its twin.

    The rulebook: "If such Supply Centre is captured by other Power, then the
    Double Unit associated with it is Disbanded." The binding names the
    double, so the twin clause makes no choice here: when TELERI takes RON,
    RON's double goes and NAR's, the one bound to the centre owned longer,
    stays bound to NAR.
    """
    phase = Phase(24, Season.FALL, PhaseKind.MOVEMENT, year_step=-1)
    for bindings in (("RON", "NAR"), ("NAR", "RON")):
        hooks, state, _ = _twin_doubles(phase, bindings)
        hooks._captures = {"RON": ("FINROD", "TELERI")}
        state = hooks._double_disbands(state, {"NAR": "FINROD", "RON": "TELERI"})
        assert sorted(str(unit) for unit in state.units) == ["2A LOT", "A NWB"]
        alive = list(hooks.ledger.alive())
        assert [entry.bound_to for entry in alive if entry.strength == 2] == ["NAR"], bindings


def _twins_dislodged(
    spec: str, centres: tuple[str, str], masks: tuple[int, int] = (0, 0)
) -> tuple[Game, BeleriandHooks]:
    """Dislodge FINROD's two twin ``spec`` units from Lothlann; they hold ``centres`` in uid order.

    ``masks`` is the number of Dwarven masks each twin carries, in uid order.

    FINROD owns NAR (since the start) and RON (since the second Autumn). A
    double is bound to the centre whose point it holds; a single holds it
    unbound. The Feanorians dislodge both twins, from HIG and from MMA, each
    with one support, so the retreat phase opens with both in limbo; LAD is
    the one free retreat, AEL and NEL hold the supporters.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    attacker = spec.split()[0]
    units = [Unit("FINROD", *parse_unit_spec(spec)) for _ in centres] + [
        Unit("FEANORIANS", *parse_unit_spec(text))
        for text in (f"{attacker} HIG", "A AEL", f"{attacker} MMA", "A NEL")
    ]
    state = State.of("beleriand", phase.tag, units)
    ownership = {"NAR": "FINROD", "RON": "FINROD"}
    ownership.update(dict.fromkeys(graph.home_centers["FEANORIANS"], "FEANORIANS"))
    hooks.register_start(state, ownership)
    hooks.owner_since["RON"] = 2
    twins = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    for entry, centre, carried in zip(twins, centres, masks, strict=True):
        entry.bound_to = centre if entry.strength == 2 else None
        entry.supplied_by = centre
        entry.masks = carried
    hooks._reconcile_spent(state, ownership)
    assert [entry.supplied_by for entry in twins] == list(centres)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
    attacks = [
        f"{attacker} HIG DIS FINROD {spec}",
        f"A AEL S {attacker} HIG DIS FINROD {spec}",
        f"{attacker} MMA DIS FINROD {spec}",
        f"A NEL S {attacker} MMA DIS FINROD {spec}",
    ]
    game.submit("FEANORIANS", [parse_order(text) for text in attacks])
    game.adjudicate()
    assert game.phase_kind is PhaseKind.RETREAT
    assert len(game.pending_dislodgements) == 2
    return game, hooks


@pytest.mark.parametrize("spec", ["2A LOT", "A LOT"], ids=["doubles", "singles"])
@pytest.mark.parametrize("centres", [("NAR", "RON"), ("RON", "NAR")], ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("lad_first", [True, False], ids=["retreat-first", "retreat-second"])
def test_of_two_dislodged_twins_the_one_that_dies_frees_the_older_centre(
    spec: str, centres: tuple[str, str], lad_first: bool
) -> None:
    """Free NAR, the centre owned longer, when one of two dislodged twins retreats and one dies.

    ``- LAD`` succeeds and ``- AEL`` fails (AEL is held), in either order on
    the sheet. The twins cannot be told apart, so whichever identity the
    ledger moves, the one left to die frees NAR and the one in LAD keeps RON
    -- a double its binding as well.
    """
    game, hooks = _twins_dislodged(spec, centres)
    rows = [f"{spec} - LAD", f"{spec} - AEL"]
    if not lad_first:
        rows.reverse()
    game.submit("FINROD", [parse_retreat_order(text) for text in rows])
    record = game.adjudicate()
    assert {row.text: row.resolution for row in record.resolutions} == {
        f"{spec} - LAD": Resolution.SUCCEEDS,
        f"{spec} - AEL": Resolution.FAILS,
    }
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LAD") and not survivor.limbo
    assert survivor.supplied_by == "RON", centres
    assert survivor.bound_to == ("RON" if survivor.strength == 2 else None), centres


# -- twins that carry different numbers of masks are told apart ---------------------------

CENTRES = [("NAR", "RON"), ("RON", "NAR")]
MASKS = [(1, 0), (0, 1)]


@pytest.mark.parametrize("spec", ["2A LOT", "A LOT"], ids=["doubles", "singles"])
@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("masks", MASKS, ids=["mask-1st", "mask-2nd"])
@pytest.mark.parametrize("lad_first", [True, False], ids=["retreat-first", "retreat-second"])
def test_of_two_dislodged_twins_the_masked_one_retreats_and_keeps_its_centre(
    spec: str, centres: tuple[str, str], masks: tuple[int, int], lad_first: bool
) -> None:
    """The twin with the mask retreats whichever uid it has; the one without dies and frees its own.

    ``- LAD`` succeeds and ``- AEL`` fails, in either order on the sheet. The
    twins carry different numbers of masks, so they are told apart: no centre
    is traded between them. The survivor holds the point (and the binding) it
    held before, the dead twin frees its own.
    """
    game, hooks = _twins_dislodged(spec, centres, masks)
    carrier = centres[masks.index(1)]
    rows = [f"{spec} - LAD", f"{spec} - AEL"]
    if not lad_first:
        rows.reverse()
    game.submit("FINROD", [parse_retreat_order(text) for text in rows])
    game.adjudicate()
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LAD") and not survivor.limbo
    assert survivor.masks == 1
    assert survivor.supplied_by == carrier, (centres, masks)
    assert survivor.bound_to == (carrier if survivor.strength == 2 else None), (centres, masks)


@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("masks", MASKS, ids=["mask-1st", "mask-2nd"])
def test_a_mask_decides_which_twin_retreats_into_devastated_ground(
    centres: tuple[str, str], masks: tuple[int, int]
) -> None:
    """Only the carrier may enter devastated LAD; it keeps the mask and its own centre."""
    game, hooks = _twins_dislodged("A LOT", centres, masks)
    hooks.devastated.add("LAD")
    carrier = centres[masks.index(1)]
    game.submit("FINROD", [parse_retreat_order(text) for text in ("A LOT - LAD", "A LOT - AEL")])
    game.adjudicate()
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LAD") and survivor.masks == 1
    assert survivor.supplied_by == carrier, (centres, masks)


@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
def test_of_two_dislodged_twins_with_equal_masks_the_one_that_dies_frees_the_older_centre(
    centres: tuple[str, str],
) -> None:
    """Equal masks leave the twins interchangeable: the survivor keeps RON, NAR is freed."""
    game, hooks = _twins_dislodged("A LOT", centres, (1, 1))
    game.submit("FINROD", [parse_retreat_order(text) for text in ("A LOT - LAD", "A LOT - AEL")])
    game.adjudicate()
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LAD") and survivor.masks == 1
    assert survivor.supplied_by == "RON", centres


# -- a twin dislodged alone is traded with the twin that stays ---------------------------


def _one_twin_dislodged(
    spec: str, centres: tuple[str, str], masks: tuple[int, int] = (0, 0)
) -> tuple[Game, BeleriandHooks]:
    """Dislodge one of FINROD's two twin ``spec`` units from Lothlann; the other stays.

    The twins hold ``centres`` and carry ``masks`` in uid order, as in
    :func:`_twins_dislodged`. One Feanorian Dislodge with one support removes
    a single twin, so the retreat phase opens with one twin in limbo and its
    twin still standing in LOT; AEL holds the supporter, so ``- AEL`` fails.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    attacker = spec.split()[0]
    units = [Unit("FINROD", *parse_unit_spec(spec)) for _ in centres] + [
        Unit("FEANORIANS", *parse_unit_spec(text)) for text in (f"{attacker} HIG", "A AEL")
    ]
    state = State.of("beleriand", phase.tag, units)
    ownership = {"NAR": "FINROD", "RON": "FINROD"}
    ownership.update(dict.fromkeys(graph.home_centers["FEANORIANS"], "FEANORIANS"))
    hooks.register_start(state, ownership)
    hooks.owner_since["RON"] = 2
    twins = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    for entry, centre, carried in zip(twins, centres, masks, strict=True):
        entry.bound_to = centre if entry.strength == 2 else None
        entry.supplied_by = centre
        entry.masks = carried
    hooks._reconcile_spent(state, ownership)
    assert [entry.supplied_by for entry in twins] == list(centres)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
    attacks = [f"{attacker} HIG DIS FINROD {spec}", f"A AEL S {attacker} HIG DIS FINROD {spec}"]
    game.submit("FEANORIANS", [parse_order(text) for text in attacks])
    game.adjudicate()
    assert game.phase_kind is PhaseKind.RETREAT
    assert len(game.pending_dislodgements) == 1
    assert [entry.limbo for entry in twins].count(True) == 1
    return game, hooks


@pytest.mark.parametrize("spec", ["2A LOT", "A LOT"], ids=["doubles", "singles"])
@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("masks", [(0, 0), (1, 1)], ids=["no-masks", "equal-masks"])
def test_of_two_twins_dislodged_alone_the_one_that_dies_frees_the_older_centre(
    spec: str, centres: tuple[str, str], masks: tuple[int, int]
) -> None:
    """Free NAR, the centre owned longer, when the dislodged twin dies and its twin stays.

    The dislodged twin has no way out (``- AEL`` fails), so it dies in the
    retreats while its twin still stands in LOT. The two cannot be told apart
    -- equal masks leave them alike -- so whichever identity the ledger
    dislodged, the one that dies frees NAR and the twin on the board holds RON
    (a double its binding as well).
    """
    game, hooks = _one_twin_dislodged(spec, centres, masks)
    game.submit("FINROD", [parse_retreat_order(f"{spec} - AEL")])
    record = game.adjudicate()
    assert [row.resolution for row in record.resolutions] == [Resolution.FAILS]
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LOT") and not survivor.limbo
    assert survivor.supplied_by == "RON", (centres, masks)
    assert survivor.bound_to == ("RON" if survivor.strength == 2 else None), (centres, masks)


@pytest.mark.parametrize("spec", ["2A LOT", "A LOT"], ids=["doubles", "singles"])
@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("masks", MASKS, ids=["mask-1st", "mask-2nd"])
def test_twins_with_different_masks_trade_no_centre_when_one_is_dislodged_alone(
    spec: str, centres: tuple[str, str], masks: tuple[int, int]
) -> None:
    """The twin that stays keeps the centre it held, whichever twin died.

    The twins carry different numbers of masks, so a player can tell them
    apart: the dead one frees its own centre and trades nothing with the twin
    on the board.
    """
    game, hooks = _one_twin_dislodged(spec, centres, masks)
    held = {
        entry.uid: (entry.supplied_by, entry.bound_to)
        for entry in hooks.ledger.alive()
        if entry.power == "FINROD"
    }
    game.submit("FINROD", [parse_retreat_order(f"{spec} - AEL")])
    game.adjudicate()
    (survivor,) = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert survivor.location == Location("LOT") and not survivor.limbo
    assert (survivor.supplied_by, survivor.bound_to) == held[survivor.uid], (centres, masks)


@pytest.mark.parametrize("centres", CENTRES, ids=["NAR-1st", "RON-1st"])
def test_a_twin_that_retreats_is_not_traded_with_the_twin_that_stays(
    centres: tuple[str, str],
) -> None:
    """A dislodged twin that retreats lives, so nobody dies and no centre moves."""
    game, hooks = _one_twin_dislodged("A LOT", centres)
    game.submit("FINROD", [parse_retreat_order("A LOT - LAD")])
    game.adjudicate()
    alive = [entry for entry in hooks.ledger.alive() if entry.power == "FINROD"]
    assert sorted(entry.location.province for entry in alive) == ["LAD", "LOT"]
    assert sorted(entry.supplied_by or "" for entry in alive) == ["NAR", "RON"]


@pytest.mark.parametrize("bindings", CENTRES, ids=["NAR-1st", "RON-1st"])
@pytest.mark.parametrize("masks", MASKS, ids=["mask-1st", "mask-2nd"])
def test_a_winter_disband_of_two_twin_doubles_removes_the_unmasked_one(
    bindings: tuple[str, str], masks: tuple[int, int]
) -> None:
    """``Disband 2A LOT`` removes the double without a mask; no centre is traded.

    The masked double stays bound to its own centre, and the point the dead
    double held, its own, goes to the unsupplied A NWB.
    """
    graph = load_map("beleriand")
    phase = Phase(24, Season.WINTER, PhaseKind.ADJUSTMENT, year_step=-1)
    hooks, state, ownership = _twin_doubles(phase, bindings, masks)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
    assert game.build_quota("FINROD") == -1
    game.submit("FINROD", [parse_adjustment_order("Disband 2A LOT")])
    game.adjudicate()
    alive = list(hooks.ledger.alive())
    (double,) = [entry for entry in alive if entry.strength == 2]
    assert double.masks == 1
    assert double.bound_to == bindings[masks.index(1)], (bindings, masks)
    (army,) = [entry for entry in alive if entry.strength == 1]
    assert army.supplied_by == bindings[masks.index(0)], (bindings, masks)


@pytest.mark.parametrize("masks", [[1, 0], [0, 1]], ids=["mask-1st", "mask-2nd"])
@pytest.mark.parametrize(
    "sheet",
    [["F NBE SAILS WEST", "F NBE H"], ["F NBE H", "F NBE SAILS WEST"]],
    ids=["sail-first", "sail-second"],
)
def test_of_two_twin_fleets_the_one_that_sails_west_is_the_one_without_a_mask(
    masks: list[int], sheet: list[str]
) -> None:
    """The mannish fleet that sails West leaves the board; the twin that carries a mask stays."""
    case = L2Case(
        id="probe",
        title="",
        section="",
        seed=0,
        start={
            "phase": "S25M",
            "units": {"MEN": ["F NBE", "F NBE"]},
            "l2": {"masks": {"MEN F NBE": masks}},
        },
        script=[],
        expect={},
    )
    game = build_game(case)
    game.submit("MEN", [parse_order(text) for text in sheet])
    game.adjudicate()
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    (fleet,) = [entry for entry in hooks.ledger.alive() if entry.power == "MEN"]
    assert fleet.masks == 1
