"""Pin the engine-side seams a variant plugs into.

A variant reaches the engine through a fixed set of joints: the order grammar
that parses its extra directives, the phase calendar's ability to count years
downwards, the neutral hooks base class whose defaults a classic game keeps,
the knowledge-grant surface, the serialization format's variant edges, and the
identity ledger's primitives. Every joint is exercised here from the engine's
side, so a variant that widens a seam the engine does not offer fails in this
file rather than deep inside a Beleriand scenario.
"""

from __future__ import annotations

import pytest

from entente.engine.adjudicator.common import (
    Dislodgement,
    OrderResolution,
    PhaseResult,
    Resolution,
)
from entente.engine.adjudicator.retreats import RetreatResult, adjudicate_retreats
from entente.engine.game import Game, GameError
from entente.engine.game_hooks import GameHooks, MovementPlan, OathState, hooks_for
from entente.engine.map import Location, Terrain, UnitKind, load_map
from entente.engine.orders import (
    AcceptMask,
    Devastate,
    GiveMask,
    MakeMasks,
    OrderSyntaxError,
    SailWest,
    Transform,
    format_order,
    parse_adjustment_order,
    parse_order,
    parse_retreat_order,
)
from entente.engine.phase import Phase, PhaseKind, Season
from entente.engine.rules import RULESETS
from entente.engine.serialization import (
    SerializationError,
    game_from_json,
    game_from_text,
    game_to_json,
    game_to_text,
)
from entente.engine.state import State, Unit
from entente.variants.beleriand.layer2 import FLEET_FACT, GIRDLE_FACT, BeleriandHooks
from entente.variants.beleriand.ledger import GLAURUNG, MORGOTH_UNIT, Ledger
from entente.variants.beleriand.winter import (
    _army_steps,
    _fleet_steps,
    adjudicate_winter,
    rulebook_civil_disorder_removals,
)

L2_TEXTS = [
    "A NFA TFM F",
    "F SFA TFM A",
    "3A HIT DEV",
    "A DOL MSK",
    "A NOS GIVES MASK TO DORIATH A FOR",
    "A FOR ACCEPTS MASK FROM OSSIRIAND 2A NEB",
    "F NBE SAILS WEST",
]


@pytest.mark.parametrize("text", L2_TEXTS)
def test_l2_order_notation_round_trips(text: str) -> None:
    order = parse_order(text)
    assert isinstance(order, (Transform, Devastate, MakeMasks, GiveMask, AcceptMask, SailWest))
    assert format_order(order) == text
    assert parse_order(format_order(order)) == order


@pytest.mark.parametrize(
    "text",
    [
        "A NFA TFM",  # missing target kind
        "A NFA TFM 2F",  # strength prefix on the target
        "A NOS GIVES MASK DORIATH A FOR",  # missing TO
        "A FOR ACCEPTS MASK OSSIRIAND A NEB",  # missing FROM
        "F NBE SAILS EAST",  # only West exists
        "A EST S A DOL MSK",  # variant actions cannot be support subjects
    ],
)
def test_l2_order_syntax_errors(text: str) -> None:
    with pytest.raises(OrderSyntaxError):
        parse_order(text)


def test_variant_orders_are_movement_phase_only() -> None:
    with pytest.raises(OrderSyntaxError):
        parse_retreat_order("A DOL MSK")


def test_bound_build_notation_round_trips() -> None:
    build = parse_adjustment_order("Build 2A NAR BOUND EGL")
    assert format_order(build) == "Build 2A NAR BOUND EGL"
    plain = parse_adjustment_order("Build A PAR")
    assert format_order(plain) == "Build A PAR"


def test_phase_year_step_descends_and_validates() -> None:
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    tags = []
    for _ in range(6):
        tags.append(phase.tag)
        phase = phase.successor()
    assert tags == ["S25M", "S25R", "F25M", "F25R", "W25A", "S24M"]
    assert Phase.parse("S25M").year_step == 1  # tags carry no direction
    with pytest.raises(ValueError, match="year_step"):
        Phase(1901, Season.SPRING, PhaseKind.MOVEMENT, year_step=2)


def test_base_hooks_invalidate_variant_orders_on_standard_games() -> None:
    game = Game.new("standard", "standard")
    game.submit("FRANCE", [parse_order("A PAR MSK"), parse_order("A MAR H")])
    record = game.adjudicate()
    rows = {row.text: row.resolution.value for row in record.resolutions}
    assert rows["A PAR MSK"] == "fails"
    assert rows["A MAR H"] == "succeeds"
    assert any(u.location.province == "PAR" for u in game.state.units)  # held


def test_base_hooks_surface_defaults() -> None:
    graph = load_map("standard")
    hooks = hooks_for("standard", graph, None)
    assert type(hooks) is GameHooks
    assert hooks.validate_grant("FRANCE", "GERMANY", FLEET_FACT) is not None
    with pytest.raises(NotImplementedError):
        hooks.apply_grant("FRANCE", "GERMANY", FLEET_FACT)
    assert hooks.legal_extra_movement_orders("FRANCE", State.of("standard", "S1901M", [])) == ()
    assert hooks.snapshot() is None
    assert hooks.drain_events() == ()
    assert not hooks.decides_victory
    assert hooks.game_result() is None
    assert hooks.adjustment_has_decisions(State.of("standard", "W1901A", []), {}) is None
    assert hooks.adjustment_has_effects(State.of("standard", "W1901A", []), {}) is None
    assert hooks.build_quota("FRANCE", State.of("standard", "W1901A", []), {}) is None
    assert hooks.legal_adjustment_orders("FRANCE", State.of("standard", "W1901A", []), {}) is None
    assert hooks.validate_second("FRANCE") is not None  # base game has no second place
    with pytest.raises(NotImplementedError):
        hooks.designate_second("FRANCE")
    assert hooks.validate_swear("FRANCE", ()) is not None  # nor an Oath of the Second
    assert hooks.validate_break(()) is not None
    with pytest.raises(NotImplementedError):
        hooks.apply_swear("FRANCE", 1901)
    with pytest.raises(NotImplementedError):
        hooks.apply_break(1901)
    assert hooks.oath_state() == OathState()
    assert hooks.validate_oath_state(OathState()) is None  # the default is not "an Oath"
    hooks.apply_oath_state(OathState())  # ... and installing it is a no-op
    assert hooks.validate_oath_state(OathState(power="FRANCE", year=1901)) is not None


def test_standard_game_has_no_winners_or_second() -> None:
    """Report no winners and no second place on an undecided classic game.

    The win-placement surface is additive, so it is inert for a standard game:
    the readers answer with the empty defaults and designation is refused,
    because a classic game is never a Morgoth win.
    """
    game = Game.new("standard", "standard")
    assert game.winners == () and game.second is None
    with pytest.raises(GameError, match="after the game ends"):
        game.designate_second("FRANCE")


def test_standard_game_refuses_grants() -> None:
    game = Game.new("standard", "standard")
    with pytest.raises(GameError, match="unknown fact"):
        game.grant("FRANCE", "GERMANY", FLEET_FACT)


def test_standard_game_refuses_both_oath_verbs() -> None:
    """Refuse both Oath verbs, and any recorded Oath, on a classic game.

    The Oath of the Second belongs to Beleriand alone, so on a standard game
    the whole surface is inert: both verbs refuse outright, the readers report
    the default state, and the loader's door refuses a recorded Oath rather
    than silently dropping it.
    """
    game = Game.new("standard", "standard")
    assert (game.oath, game.oath_year) == (None, None)
    assert (game.oaths_broken, game.oath_stabbed) == (0, False)
    with pytest.raises(GameError, match="no Oath of the Second"):
        game.swear_oath("FRANCE")
    with pytest.raises(GameError, match="no Oath of the Second"):
        game.break_oath()
    with pytest.raises(GameError, match="no Oath of the Second"):
        game.restore_oath_state("FRANCE", 1901, 0)
    with pytest.raises(GameError, match="no Oath of the Second"):
        game.restore_oath_state(None, None, 2)
    game.restore_oath_state(None, None, 0)  # the default is always installable
    assert game.oath is None and game.oaths_broken == 0
    assert "oath" not in game_to_json(game)  # and nothing reaches the save


def test_grant_validation_matrix() -> None:
    game = Game.new("beleriand", "beleriand", seed=0)
    for giver, receiver, fact, why in [
        ("TELERI", "TELERI", FLEET_FACT, "itself"),
        ("MEN", "NAUGRIM", FLEET_FACT, "does not hold"),
        ("TELERI", "MORGOTH", FLEET_FACT, "never gain"),
        ("TELERI", "FEANORIANS", FLEET_FACT, "only from Men"),
        ("DORIATH", "FEANORIANS", GIRDLE_FACT, "only from Men"),
        ("DORIATH", "MORGOTH", GIRDLE_FACT, "non-elvish power"),
        ("TELERI", "MEN", "sword-forging", "unknown fact"),
    ]:
        with pytest.raises(GameError, match=why):
            game.grant(giver, receiver, fact)
    game.grant("TELERI", "MEN", FLEET_FACT)
    with pytest.raises(GameError, match="already holds"):
        game.grant("TELERI", "MEN", FLEET_FACT)
    game.grant("DORIATH", "MEN", GIRDLE_FACT)
    game.grant("MEN", "FEANORIANS", GIRDLE_FACT)  # Men/Naugrim chain works
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    assert "FEANORIANS" in hooks.girdle

    # The giver axis of the written Girdle grant to Morgoth: exactly the two
    # non-elvish givers, Men and the Naugrim, may pass him the Girdle passage;
    # every elven power is refused, Doriath included, the owner of the Girdle
    # itself. Each giver gets a fresh game because a successful grant is
    # monotone. The fleet fact is swept alongside to show the contrast: the
    # fleet-building ban on Morgoth admits no giver at all.
    for giver in game.powers:
        if giver == "MORGOTH":
            continue  # self-grant, refused one rule earlier
        fresh = Game.new("beleriand", "beleriand", seed=0)
        fresh_hooks = fresh.hooks
        assert isinstance(fresh_hooks, BeleriandHooks)
        fresh_hooks.girdle.add(giver)
        fresh_hooks.fleet_knowledge.add(giver)
        if giver in ("MEN", "NAUGRIM"):
            fresh.grant(giver, "MORGOTH", GIRDLE_FACT)
            assert "MORGOTH" in fresh_hooks.girdle, giver
        else:
            refusal = "learns the Girdle passage from a non-elvish power"
            with pytest.raises(GameError, match=refusal):
                fresh.grant(giver, "MORGOTH", GIRDLE_FACT)
            assert "MORGOTH" not in fresh_hooks.girdle, giver
        with pytest.raises(GameError, match="never gain the fleet-building"):
            fresh.grant(giver, "MORGOTH", FLEET_FACT)
        assert "MORGOTH" not in fresh_hooks.fleet_knowledge, giver


def test_morgoth_onward_girdle_grants_split_by_receiver() -> None:
    """Split Morgoth's onward Girdle grants by the receiver, not by its source.

    Once Morgoth holds the Girdle passage -- forced in here, standing in for
    the attack-grant that normally hands it to him -- his written grants to
    Men and the Naugrim stay legal, but Morgoth passes no knowledge to an
    elvish power at all. Every elvish receiver, Feanorian or not, is therefore
    refused with the elvish-power ban rather than with the Feanorians' own
    source rule.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    hooks.girdle.add("MORGOTH")
    game.grant("MORGOTH", "MEN", GIRDLE_FACT)
    assert "MEN" in hooks.girdle
    for receiver in ("FINGOLFIN", "FEANORIANS"):
        with pytest.raises(GameError, match="Morgoth passes no knowledge to elvish powers"):
            game.grant("MORGOTH", receiver, GIRDLE_FACT)


def test_morgoth_enumeration_never_offers_masks() -> None:
    """Offer Morgoth no mask directive of any kind.

    Enumeration agrees with the resolution-side guards: a Morgoth unit standing
    in a dwarf city is never offered ``MakeMasks``, and Morgoth never sees
    either transfer directive, as the ordering power or as a ``GiveMask``
    receiver. Only the transfer half is Morgoth's own rule; making is the
    Naugrim's alone, so Morgoth is refused there as one non-dwarf among many
    (:func:`test_only_naugrim_enumeration_offers_masks`).
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    units = [
        Unit("MORGOTH", UnitKind.ARMY, Location("BEL"), 1),
        Unit("NAUGRIM", UnitKind.ARMY, Location("DOL"), 1),
    ]
    state = State.of("beleriand", "S25M", units)
    hooks.register_start(state, {})
    naugrim_entry = next(e for e in hooks.ledger.alive() if e.power == "NAUGRIM")
    naugrim_entry.masks = 1  # adjacent to Morgoth's BEL (BEL-DOL is a real edge)
    morgoth_orders = hooks.legal_extra_movement_orders("MORGOTH", state)
    assert not any(isinstance(order, MakeMasks) for order in morgoth_orders)
    assert hooks._transfer_orders("MORGOTH", state) == []
    naugrim_orders = hooks._transfer_orders("NAUGRIM", state)
    assert not any(
        isinstance(order, GiveMask) and order.receiver.power == "MORGOTH"
        for order in naugrim_orders
    )


def test_only_naugrim_enumeration_offers_masks() -> None:
    """Offer ``MakeMasks`` to the Naugrim alone, and leave passing open to all.

    The rulebook's "Dwarven Masks" section says *"only the Naugrim may make
    masks"*, and enumeration agrees with the resolution-side guard: a foreign
    unit standing in a dwarf city is offered no ``MakeMasks``, while the
    Naugrim unit beside it is. Holding and passing carry no such restriction,
    so the foreign unit's own mask still enumerates a ``GiveMask``.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    units = [
        Unit("FEANORIANS", UnitKind.ARMY, Location("BEL"), 1),
        Unit("NAUGRIM", UnitKind.ARMY, Location("DOL"), 1),
    ]
    state = State.of("beleriand", "S25M", units)
    hooks.register_start(state, {})
    feanorian_entry = next(e for e in hooks.ledger.alive() if e.power == "FEANORIANS")
    feanorian_entry.masks = 1  # BEL-DOL is a real edge, so a pass enumerates
    feanorian_orders = hooks.legal_extra_movement_orders("FEANORIANS", state)
    assert not any(isinstance(order, MakeMasks) for order in feanorian_orders)
    assert any(isinstance(order, GiveMask) for order in feanorian_orders)
    naugrim_orders = hooks.legal_extra_movement_orders("NAUGRIM", state)
    assert any(isinstance(order, MakeMasks) for order in naugrim_orders)


def _beleriand_game(
    units: dict[str, list[str]], masks: dict[str, int], phase_tag: str = "S25M"
) -> Game:
    """Build a Beleriand movement position by hand, with masks on the ledger.

    Positions this file needs are assembled directly rather than played into
    existence, so a scenario can start from any board the rules allow.
    """
    from entente.engine.orders import parse_unit_spec

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    parsed = Phase.parse(phase_tag)
    phase = Phase(parsed.year, parsed.season, parsed.kind, year_step=-1)
    placed = [
        Unit(power, *parse_unit_spec(spec)) for power, specs in units.items() for spec in specs
    ]
    state = State.of("beleriand", phase.tag, placed)
    present = {u.power for u in placed}
    ownership = {
        center: power
        for power, centers in graph.home_centers.items()
        if power in present
        for center in centers
    }
    hooks.register_start(state, ownership)
    for key, count in masks.items():
        power, _, spec = key.partition(" ")
        kind, location, strength = parse_unit_spec(spec)
        entry = next(
            e
            for e in hooks.ledger.alive()
            if e.power == power
            and e.kind is kind
            and e.strength == strength
            and e.location == location
        )
        entry.masks = count
    return Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)


def test_gondolin_shield_survives_a_support_order() -> None:
    """Keep the Gondolin shield when the double is ordered to support.

    The shield falls only when the starting double is written a Move or a
    Dislodge naming its own unit spec: the rulebook treats the sortie order
    itself as the sally, whatever it achieves. A Support is not a sortie, so
    the shield stands. This is the non-relocating shape that
    ``_lift_gondolin_shield_on_order``'s ``isinstance(order, (Move, Dislodge))``
    gate excludes; the Hold case is exercised end to end through the case files.
    """
    game = _beleriand_game({"GONDOLIN": ["2A GON"], "FINROD": ["A TOL"]}, {})
    game.submit("GONDOLIN", [parse_order("2A GON S A TOL")])
    game.submit("FINROD", [parse_order("A TOL H")])
    game.adjudicate()
    assert game.hooks.gondolin_shield is True


def test_gondolin_shield_falls_on_a_written_dislodge_even_though_it_is_invalid() -> None:
    """Lift the Gondolin shield on a written Dislodge that cannot be valid.

    The shield falls by the written order, not by its outcome. This is the one
    order shape no played position can reach: GON has capacity 1, so the shield
    double never has a live Dislodge target on a real board, and the case files
    can only exercise the Move side. A Dislodge naming the double's own spec
    still lifts the shield the instant it is written, even though Phase 0
    necessarily invalidates the order itself -- no victim stands at DIM to
    dislodge.
    """
    game = _beleriand_game({"GONDOLIN": ["2A GON"], "FINROD": ["A TOL"]}, {})
    game.submit("GONDOLIN", [parse_order("2A GON DIS FINROD A DIM")])
    game.submit("FINROD", [parse_order("A TOL H")])
    game.adjudicate()
    assert game.hooks.gondolin_shield is False


def test_move_then_give_mask_resolves_end_of_phase() -> None:
    """Pass a mask from a unit that moved this phase, at its end position.

    A mask passes at the end of the phase, at the units' end-of-phase
    positions, so a holder may move and still give. The orders are hand-built
    to bypass enumeration: the giver moves DOL->THA, its ``GIVES`` names the
    receiver near the destination, and the counterpart's ``ACCEPTS`` names the
    giver at its end spot THA. The pass carries, which is what lets enumeration
    anticipate movement at all.
    """
    game = _beleriand_game({"NAUGRIM": ["A DOL"], "DORIATH": ["A NEB"]}, {"NAUGRIM A DOL": 1})
    game.submit(
        "NAUGRIM",
        [parse_order("A DOL - THA"), parse_order("A DOL GIVES MASK TO DORIATH A NEB")],
    )
    game.submit("DORIATH", [parse_order("A NEB ACCEPTS MASK FROM NAUGRIM A THA")])
    record = game.adjudicate()
    assert "NAUGRIM A THA passes a mask to DORIATH A NEB" in record.events
    holders = {f"{e.power} {e.spec()}": e.masks for e in game.hooks.ledger.alive() if e.masks}
    assert holders == {"DORIATH A NEB": 1}


def test_both_sides_moving_mask_pass_resolves_end_of_phase() -> None:
    """Pass a mask when both counterparts move in the same phase.

    Neither reference names a province either unit currently occupies: the
    Naugrim holder at DOL moves DOL->THA and gives to Doriath at its move-end
    NEB, while Doriath moves FOR->NEB and accepts from the giver at its
    move-end THA (THA-NEB is a real edge). The pass carries, because the
    resolution reads post-move positions on both sides -- the premise the
    enumeration's cross product rests on. The orders go in through
    :meth:`Game.submit`, which does not check enumeration membership, so this
    exercises the resolution alone.

    FOR lies inside the Girdle, but the Girdle check reads end positions (THA
    and NEB, both outside), so it plays no part: the pass succeeds for one
    reason only.
    """
    game = _beleriand_game({"NAUGRIM": ["A DOL"], "DORIATH": ["A FOR"]}, {"NAUGRIM A DOL": 1})
    game.submit(
        "NAUGRIM",
        [parse_order("A DOL - THA"), parse_order("A DOL GIVES MASK TO DORIATH A NEB")],
    )
    game.submit(
        "DORIATH",
        [parse_order("A FOR - NEB"), parse_order("A FOR ACCEPTS MASK FROM NAUGRIM A THA")],
    )
    record = game.adjudicate()
    rows = {row.text: row.resolution.value for row in record.resolutions}
    assert rows["A DOL - THA"] == "succeeds"
    assert rows["A FOR - NEB"] == "succeeds"
    assert rows["A DOL GIVES MASK TO DORIATH A NEB"] == "succeeds"
    assert rows["A FOR ACCEPTS MASK FROM NAUGRIM A THA"] == "succeeds"
    assert "NAUGRIM A THA passes a mask to DORIATH A NEB" in record.events
    holders = {f"{e.power} {e.spec()}": e.masks for e in game.hooks.ledger.alive() if e.masks}
    assert holders == {"DORIATH A NEB": 1}  # the giver's mask moved to the receiver
    ended = sorted(f"{u.power} {u}" for u in game.state.units)
    assert ended == ["DORIATH A NEB", "NAUGRIM A THA"]  # both moved


def test_transfer_enumeration_anticipates_counterpart_ends() -> None:
    """Enumerate transfer directives at the counterpart's anticipated ends.

    For a holder near a movable counterpart, ``_transfer_orders`` names that
    counterpart at each end it can reach, while the ordering unit's own spec
    stays at its current sheet location, because the main binding binds by
    slot. DOL is not adjacent to NEB, so the pass needs a move (DOL->THA, and
    THA-NEB is a real edge); the directives are offered on both sides anyway.
    """
    game = _beleriand_game({"NAUGRIM": ["A DOL"], "DORIATH": ["A NEB"]}, {"NAUGRIM A DOL": 1})
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    gives = [o for o in hooks._transfer_orders("NAUGRIM", game.state) if isinstance(o, GiveMask)]
    accepts = [
        o for o in hooks._transfer_orders("DORIATH", game.state) if isinstance(o, AcceptMask)
    ]
    give_texts = {format_order(o) for o in gives}
    accept_texts = {format_order(o) for o in accepts}
    # the receiver anticipated at its current NEB (giver could move to reach it)
    assert "A DOL GIVES MASK TO DORIATH A NEB" in give_texts
    # the move-coupled ACCEPT names the giver at its move-end THA -- the text the
    # stationary enumeration never offered (its docstring's "gap")
    assert "A NEB ACCEPTS MASK FROM NAUGRIM A THA" in accept_texts
    # the giver-at-current accept is offered too (here via the acceptor's OWN
    # move into reach -- NEB->NOS/THA are adjacent to a stationary DOL giver)
    assert "A NEB ACCEPTS MASK FROM NAUGRIM A DOL" in accept_texts
    # every OWN spec stays at the unit's current province (never an end)
    assert gives and all(o.location.province == "DOL" for o in gives)
    assert accepts and all(o.location.province == "NEB" for o in accepts)


def test_transfer_enumeration_offers_both_move_cross_product() -> None:
    """Enumerate the both-move pass as a cross product, for both powers.

    The Naugrim hold a mask at DOL; Doriath stands at FOR. DOL is not adjacent
    to NEB and FOR is not adjacent to THA, so no single move makes the pass
    reachable: it needs the giver's DOL->THA and the receiver's FOR->NEB at
    once (THA-NEB is a real edge). Enumeration therefore offers the
    geometry-filtered cross product of both sides' anticipated ends -- the give
    names Doriath at NEB, the accept names the Naugrim at THA -- rather than
    only the ends one moving side can reach.
    """
    game = _beleriand_game({"NAUGRIM": ["A DOL"], "DORIATH": ["A FOR"]}, {"NAUGRIM A DOL": 1})
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    gives = [o for o in hooks._transfer_orders("NAUGRIM", game.state) if isinstance(o, GiveMask)]
    accepts = [
        o for o in hooks._transfer_orders("DORIATH", game.state) if isinstance(o, AcceptMask)
    ]
    give_texts = {format_order(o) for o in gives}
    accept_texts = {format_order(o) for o in accepts}
    # the both-move give: receiver named at its move-end NEB (needs giver at THA)
    assert "A DOL GIVES MASK TO DORIATH A NEB" in give_texts
    # the both-move accept: giver named at its move-end THA (needs acceptor at NEB)
    assert "A FOR ACCEPTS MASK FROM NAUGRIM A THA" in accept_texts
    # own specs stay at the current sheet location on both sides (never an end)
    assert gives and all(o.location.province == "DOL" for o in gives)
    assert accepts and all(o.location.province == "FOR" for o in accepts)


def test_transfer_enumeration_geometry_matches_resolution_direction() -> None:
    """Keep enumeration's mask geometry pointed the way the resolution reads it.

    Every enumerated directive names its counterpart at an end for which some
    own end clears the engine's mask geometry in the resolution's own
    direction: ``_mask_geometry_ok`` asks whether the acceptor lies in the
    giver's neighbours, giver first. Enumeration standardises on that
    direction, because the map has one-way edges and the reversed
    acceptor-to-giver test disagrees on them; standardising is what keeps it
    from offering a pair the resolution would reject. The
    own-spec-stays-current invariant is re-checked sheet-wide over a
    holder-rich position, so dozens of pairs are actually exercised.
    """
    game = _beleriand_game(
        {
            "NAUGRIM": ["A DOL", "A BEL"],
            "DORIATH": ["A NEB", "A FOR"],
            "MEN": ["A EST"],
        },
        {"NAUGRIM A DOL": 1, "NAUGRIM A BEL": 1},
    )
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    checked = 0
    for power in game.powers:
        own_provinces = {u.location.province for u in game.state.units if u.power == power}
        for order in hooks._transfer_orders(power, game.state):
            # own spec stays at a CURRENT own-unit province, never an anticipated end
            assert order.location.province in own_provinces, format_order(order)
            own_ends = hooks._anticipated_ends(order.unit_kind, order.location.province)
            if isinstance(order, GiveMask):  # giver own; receiver ref is the acceptor end
                ref = order.receiver.location.province
                assert any(hooks._mask_geometry_provinces_ok(g, ref) for g in own_ends), (
                    format_order(order)
                )
                checked += 1
            elif isinstance(order, AcceptMask):  # acceptor own; giver ref is the giver end
                ref = order.giver.location.province
                assert any(hooks._mask_geometry_provinces_ok(ref, a) for a in own_ends), (
                    format_order(order)
                )
                checked += 1
    assert checked > 30  # the position is genuinely holder-rich (measured 45)


def test_serialization_rejects_bad_versions_and_grants() -> None:
    game = Game.new("beleriand", "beleriand", seed=1)
    data = game_to_json(game)
    assert data["version"] == 2 and data["seed"] == 1
    bad = dict(data)
    bad["version"] = 3
    with pytest.raises(SerializationError, match="unsupported version"):
        game_from_json(bad)
    game.grant("TELERI", "MEN", FLEET_FACT)
    game.adjudicate()
    data = game_to_json(game)
    assert data["phases"][0]["grants"] == [["TELERI", "MEN", FLEET_FACT]]
    data["phases"][0]["grants"] = ["not-a-triple"]
    with pytest.raises(SerializationError, match="malformed grant"):
        game_from_json(data)


def test_l2_snapshot_mismatch_fails_loudly() -> None:
    game = Game.new("beleriand", "beleriand", seed=1)
    game.adjudicate()
    data = game_to_json(game)
    data["snapshot"]["l2"]["eriador_centers"] = 99
    with pytest.raises(SerializationError, match="snapshot mismatch on 'l2'"):
        game_from_json(data)


def test_standard_serialization_still_emits_v1() -> None:
    game = Game.new("standard", "standard")
    data = game_to_json(game)
    assert data["version"] == 1
    assert "seed" not in data and "l2" not in data["snapshot"]
    assert game_from_text(game_to_text(game)).phase == game.phase


def test_ledger_spec_prints_the_special_designator() -> None:
    """Render the ``(M)``/``(G)`` designator for the two special identities.

    Equal units are fungible values, so an event line or a mask key naming a
    bare ``3A ANF`` could mean either special or an ordinary twin.
    :meth:`Ledger.spec` prints the designator that narrows it to the named
    special, and prints nothing for an untagged unit.
    """
    ledger = Ledger()
    unit = Unit("MORGOTH", UnitKind.ARMY, Location("ANF"), 3)
    morgoth = ledger.create(unit, special=MORGOTH_UNIT)
    glaurung = ledger.create(unit, special=GLAURUNG)
    plain = ledger.create(unit)
    assert morgoth.spec() == "3A(M) ANF"
    assert glaurung.spec() == "3A(G) ANF"
    assert plain.spec() == "3A ANF"  # an untagged 3A carries no designator


def test_ledger_assign_prefers_credentialed_identities() -> None:
    ledger = Ledger()
    plain = ledger.create(Unit("NAUGRIM", UnitKind.ARMY, Location("BEL")))
    masked = ledger.create(Unit("NAUGRIM", UnitKind.ARMY, Location("BEL")))
    masked.masks = 1
    needs_mask = ledger.assign([plain.uid, masked.uid], [lambda e: e.masks > 0, None])
    assert needs_mask == [masked.uid, plain.uid]
    unsatisfiable = ledger.assign([plain.uid], [lambda e: e.masks > 0])
    assert unsatisfiable == [None]
    assert ledger.special("nothing") is None


def test_movement_plan_merge_keeps_sheet_order() -> None:
    game = Game.new("beleriand", "beleriand", seed=0)
    game.submit(
        "NAUGRIM",
        [
            parse_order("A BEL MSK"),
            parse_order("A DOL - THA"),  # a real edge (DOL-ELA is gone)
            parse_order("A NOG MSK"),
        ],
    )
    record = game.adjudicate()
    naugrim_rows = [row.text for row in record.resolutions if row.power == "NAUGRIM"]
    assert naugrim_rows == ["A BEL MSK", "A DOL - THA", "A NOG MSK"]


def test_beleriand_defaults_start_at_year_25_countdown() -> None:
    game = Game.new("beleriand", "beleriand")
    assert game.phase == "S25M"
    assert game.seed == 0  # the variant layer defaults an unset seed to 0
    game.adjudicate()
    game.adjudicate()
    assert game.year == 24  # descending calendar


def test_movement_plan_defaults() -> None:
    plan = MovementPlan(adjudicated={})
    assert plan.deferred == {} and plan.invalid == {}


def _limbo_double(hooks: BeleriandHooks, province: str, *, bound_to: str | None = None):
    """Seed one dislodged FEANORIANS double in limbo at ``province``."""
    entry = hooks.ledger.create(
        Unit("FEANORIANS", UnitKind.ARMY, Location(province), 2), bound_to=bound_to
    )
    entry.limbo = True
    return entry


def _retreat_movement(*dislodged: Dislodgement) -> PhaseResult:
    """Fabricate the movement result a retreat phase reads: the dislodgements, an empty board."""
    return PhaseResult(
        resolutions=(),
        new_state=State.of("beleriand", "S23M", []),
        dislodgements=dislodged,
        embattled=frozenset(),
    )


def test_preprocess_retreat_spends_one_mask_per_line_the_adjudicator_binds() -> None:
    """Spend one mask per retreat into devastated ground, from the class of the unit it binds.

    Two dislodged FEANORIANS doubles at THA carry one mask between them, and
    both are ordered into devastated NEB. The value-level gate lets both lines
    through, since a mask is carried in the class; the first line binds one
    double and spends the mask, the second binds the other and finds none
    left, so it is withheld as invalid. A line written without the doubles'
    base strength binds neither of them and spends nothing.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    hooks.wrap_ruleset(RULESETS["beleriand"])
    masked = _limbo_double(hooks, "THA")
    masked.masks = 1
    _limbo_double(hooks, "THA")
    hooks.devastated.add("NEB")
    phase = Phase(23, Season.SPRING, PhaseKind.RETREAT, year_step=-1)
    double = Dislodgement(Unit("FEANORIANS", UnitKind.ARMY, Location("THA"), 2), "HEL", False)
    movement = _retreat_movement(double, double)
    texts = ("A THA - NEB", "2A THA - NEB", "2A THA - NEB")
    sheet = [parse_retreat_order(text) for text in texts]

    gated, invalid = hooks.preprocess_retreat(phase, {"FEANORIANS": sheet}, movement)
    assert gated["FEANORIANS"] == sheet[:2]
    assert invalid == [("FEANORIANS", sheet[2])]


def test_post_retreat_rebinds_dislodged_double_instead_of_disbanding_it() -> None:
    """Rebind a dislodged double to its landing instead of disbanding it.

    A double that retreats successfully, through a plain destination or a gated
    one, keeps its ledger identity and its double-centre binding:
    :meth:`post_retreat` moves an identity of the dislodgement the adjudicator
    bound -- the 2A's own class, base strength included -- to the landing
    rather than sweeping the 2A into the disband set.
    """
    graph = load_map("beleriand")
    phase = Phase(23, Season.SPRING, PhaseKind.RETREAT, year_step=-1)
    empty = State.of("beleriand", phase.tag, [])
    succeed = (OrderResolution("FEANORIANS", "2A THA - NEB", Resolution.SUCCEEDS),)
    dislodged = Dislodgement(Unit("FEANORIANS", UnitKind.ARMY, Location("THA"), 2), "HEL", False)
    retreated = ((dislodged, Location("NEB")),)

    hooks = BeleriandHooks(graph, 0)
    double = _limbo_double(hooks, "THA", bound_to="HID")
    hooks.post_retreat(phase, RetreatResult(succeed, empty, (), retreated))
    assert double.uid in hooks.ledger.units  # not swept into the disband set
    assert not double.limbo and double.location == Location("NEB")
    assert double.bound_to == "HID"

    hooks = BeleriandHooks(graph, 0)
    hooks.devastated.add("NEB")
    masked = _limbo_double(hooks, "THA")
    masked.masks = 1
    hooks.post_retreat(phase, RetreatResult(succeed, empty, (), retreated))
    assert not masked.limbo and masked.location == Location("NEB") and masked.masks == 1


def test_devastate_dislodgement_threads_through_the_game_facade() -> None:
    """Thread a devastation dislodgement through the game facade intact.

    Devastation dislodges the units standing in the province, so
    :meth:`postprocess_movement` returns extra ``Dislodgement`` records beyond
    the ones the movement calculus produced itself. Those records must reach
    ``Game._execute`` unchanged: ``pending_dislodgements``, the stored
    ``PhaseRecord`` and the ledger's own limbo flag all have to agree, and the
    records carry no dislodger origin, so nothing bans a retreat destination.
    The case files cover the rulebook behaviour; this covers the plumbing.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    units = [
        Unit("MORGOTH", UnitKind.ARMY, Location("HIT"), 3),
        Unit("FEANORIANS", UnitKind.ARMY, Location("HIT"), 1),
    ]
    state = State.of("beleriand", "S25M", units)
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    hooks.register_start(state, {})
    for entry in hooks.ledger.alive():
        if entry.power == "MORGOTH" and entry.strength == 3:
            entry.special = GLAURUNG

    game = Game(RULESETS["beleriand"], graph, state, {}, phase, hooks=hooks)
    game.submit("MORGOTH", [parse_order("3A(G) HIT DEV")])
    record = game.adjudicate()

    assert record.dislodgements == game.pending_dislodgements
    (dislodgement,) = game.pending_dislodgements
    assert dislodgement.unit == Unit("FEANORIANS", UnitKind.ARMY, Location("HIT"), 1)
    assert dislodgement.dislodger_origin is None
    assert dislodgement.by_convoy is False
    assert not any(u.power == "FEANORIANS" for u in game.state.units)

    limbo = hooks.ledger.limbo()
    assert len(limbo) == 1
    assert limbo[0].power == "FEANORIANS"
    assert limbo[0].location == Location("HIT")

    # a None dislodger_origin never bans a retreat destination (the origin
    # test is a plain equality against a province name) -- the unit retreats
    # out normally, same as a combat-dislodged one
    game.submit("FEANORIANS", [parse_retreat_order("A HIT - DLN")])
    retreat = game.adjudicate()
    assert retreat.resolutions[0].resolution.value == "succeeds"
    assert any(u == Unit("FEANORIANS", UnitKind.ARMY, Location("DLN")) for u in game.state.units)


def test_a_retreat_into_angband_or_eriador_claims_no_credential() -> None:
    """Leave Angband and Eriador to the value-level gate: neither needs a credential.

    Entry into Angband and Eriador from the board is forbidden outright, with
    Morgoth's strength-3 specials excepted for Angband. A retreat binds the
    dislodgement its spec names, whose class carries the base strength, so no
    credential has to tell a 3A from an ordinary army: the game layer
    withholds nothing, and the adjudicator's entry permission lets the
    Morgoth unit into Angband and fails the ordinary army's retreat there, and
    any retreat into Eriador.
    """
    graph = load_map("beleriand")
    phase = Phase(23, Season.SPRING, PhaseKind.RETREAT, year_step=-1)

    def retreat(power: str, origin: str, strength: int, dest: str) -> Resolution:
        hooks = BeleriandHooks(graph, 0)
        ruleset = hooks.wrap_ruleset(RULESETS["beleriand"])
        unit = Unit(power, UnitKind.ARMY, Location(origin), strength)
        special = MORGOTH_UNIT if strength == 3 else None
        hooks.ledger.create(unit, special=special).limbo = True
        designator = "M" if special is not None else None
        movement = _retreat_movement(Dislodgement(unit, None, False, designator))
        spec = f"3A(M) {origin}" if special is not None else f"A {origin}"
        sheet = [parse_retreat_order(f"{spec} - {dest}")]
        gated, invalid = hooks.preprocess_retreat(phase, {power: sheet}, movement)
        assert gated[power] == sheet and invalid == []
        return adjudicate_retreats(ruleset, movement, gated, graph=graph).resolutions[0].resolution

    assert retreat("MORGOTH", "ANF", 3, "ANG") is Resolution.SUCCEEDS  # the Morgoth unit
    assert retreat("MORGOTH", "ANF", 1, "ANG") is Resolution.FAILS  # an ordinary army
    assert retreat("MEN", "ELA", 1, "ERI") is Resolution.FAILS  # nobody enters Eriador


def test_post_retreat_binds_angband_to_the_strength3_identity() -> None:
    """Rebind an Angband retreat to Glaurung, not to the lower-uid army beside him.

    A mixed Morgoth group is dislodged from Anfauglith: an early-built ordinary
    army with a low uid, and Glaurung, a 3A with a higher one, both in limbo.
    The retreat into Angband bound Glaurung's dislodgement -- base strength 3,
    designator (G) -- and only an identity of that class may move: a uid-only
    bind would disband Glaurung and drift a strength-1 army into Angband, and a
    Morgoth unit drifting that way would be a false defeat.
    """
    graph = load_map("beleriand")
    phase = Phase(20, Season.SPRING, PhaseKind.RETREAT, year_step=-1)
    empty = State.of("beleriand", phase.tag, [])
    hooks = BeleriandHooks(graph, 0)
    at_anf = Unit("MORGOTH", UnitKind.ARMY, Location("ANF"), 3)
    ordinary = hooks.ledger.create(Unit("MORGOTH", UnitKind.ARMY, Location("ANF"), 1))
    glaurung = hooks.ledger.create(at_anf, special=GLAURUNG)
    ordinary.limbo = True
    glaurung.limbo = True
    assert ordinary.uid < glaurung.uid  # a uid-only bind would pick the ordinary twin
    row = (OrderResolution("MORGOTH", "3A(G) ANF - ANG", Resolution.SUCCEEDS),)
    dislodged = Dislodgement(at_anf, "LAD", False, "G")
    hooks.post_retreat(phase, RetreatResult(row, empty, (), ((dislodged, Location("ANG")),)))
    assert glaurung.uid in hooks.ledger.units and not glaurung.limbo
    assert glaurung.location == Location("ANG")
    assert ordinary.uid not in hooks.ledger.units  # the ordinary twin disbanded


def test_winter_rejects_special_unit_designators() -> None:
    """Reject the special designators in winter, in either quota direction.

    Neither special identity is buildable or removable: they exist from the
    start and leave only by dying. A winter sheet that writes a ``(M)`` or
    ``(G)`` designator therefore fails, whether it asks for a Build or a
    Disband.
    """
    graph = load_map("beleriand")

    # Build direction: MORGOTH owns ANG+ANF, its one 3A exempt -> a build allowance.
    hooks = BeleriandHooks(graph, 0)
    ownership = {"ANG": "MORGOTH", "ANF": "MORGOTH"}
    state = State.of("beleriand", "W25A", [Unit("MORGOTH", UnitKind.ARMY, Location("ANG"), 3)])
    hooks.register_start(state, ownership)
    built = adjudicate_winter(
        hooks, state, ownership, {"MORGOTH": [parse_adjustment_order("Build 3A(M) ANG")]}
    )
    assert built.resolutions[0].resolution is Resolution.FAILS
    assert built.built == ()

    # Disband direction: MORGOTH owns nothing, two ordinary armies over quota.
    hooks = BeleriandHooks(graph, 0)
    units = [
        Unit("MORGOTH", UnitKind.ARMY, Location("ANF"), 3),  # the Morgoth unit: exempt
        Unit("MORGOTH", UnitKind.ARMY, Location("NEB")),
        Unit("MORGOTH", UnitKind.ARMY, Location("LAD")),
    ]
    state = State.of("beleriand", "W25A", units)
    hooks.register_start(state, {})
    removed = adjudicate_winter(
        hooks, state, {}, {"MORGOTH": [parse_adjustment_order("Disband 3A(M) ANF")]}
    )
    assert removed.resolutions[0].resolution is Resolution.FAILS
    # the special is exempt and survives; only the ordinary armies can be trimmed
    assert any(u.power == "MORGOTH" and u.strength == 3 for u in removed.new_state.units)


# -- Morgoth win-placement and the defeat flag ------------------------------------

_MORGOTH_FORTY_OWNED = [
    "BEL",
    "BRI",
    "DIM",
    "DLN",
    "DNN",
    "DOL",
    "EGL",
    "EIT",
    "ELA",
    "FON",
    "FOR",
    "GEL",
    "GON",
    "HEL",
    "HID",
    "HIG",
    "IOB",
    "LAD",
    "MEN",
    "MIT",
    "MMA",
    "NAR",
    "NEV",
    "NFA",
    "NOG",
    "NWB",
    "RIV",
    "SFA",
    "TGA",
    "THA",
    "TOL",
    "VIN",
]
"""The 32 centres that, with Morgoth's home ANG (7) and ANF (1), weigh exactly 40.

Forty is the unraised victory bar, so this list is the smallest board that
carries a Morgoth win. The same position is played through the case files as
BATC-L2.V.2.
"""


def _morgoth_forty_win(
    *,
    oath: str | None = None,
    broken: int = 0,
    year: int = 25,
    expect_win: bool = True,
) -> Game:
    """Build a finished Beleriand game whose result is a solo Morgoth win at forty.

    The board is the forty-centre position driven through an Autumn boundary,
    assembled directly the way :func:`_beleriand_game` assembles one. Such a
    game does not replay, so the second-place round trip is checked save-side
    here and load-side on a real ``Game.new`` document below. The same position
    is played through the case files as BATC-L2.V.2.

    ``oath`` swears the Oath of the Second before the first phase and
    ``broken`` charges that many prices onto the bar, through the verbs, so the
    price is paid the way play pays it. With ``expect_win`` false the caller is
    asking for a board the raised bar denies.
    """
    from entente.engine.orders import parse_unit_spec

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    phase = Phase(year, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    state = State.of("beleriand", phase.tag, [Unit("MORGOTH", *parse_unit_spec("3A ANG"))])
    ownership = {c: p for p, cs in graph.home_centers.items() if p == "MORGOTH" for c in cs}
    ownership.update(dict.fromkeys(_MORGOTH_FORTY_OWNED, "MORGOTH"))
    hooks.register_start(state, ownership)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
    for _ in range(broken):
        game.swear_oath("OSSIRIAND")
        game.break_oath()
    if oath is not None:
        game.swear_oath(oath)
    game.adjudicate()  # Spring -> Autumn
    game.adjudicate()  # Autumn -> the victory check fires
    assert game.is_over
    assert (game.winners == ("MORGOTH",)) is expect_win
    return game


def test_second_place_designation_snapshot_and_serialization() -> None:
    """Let a winning Morgoth name one second place, once, after the game ends.

    The rulebook says *"he may choose one Player to get second place. The rest
    of the Players are unplaced."* The act is post-game and happens once; the
    snapshot and the save file both carry the name, and both leave the key out
    while no name has been given.
    """
    game = _morgoth_forty_win()
    assert game.second is None
    assert game.hooks.snapshot()["second"] is None  # type: ignore[index]
    assert "second" not in game_to_json(game)  # additive key absent until named

    # refused for Morgoth himself and for a power that is not in the game
    with pytest.raises(GameError, match="Morgoth cannot be designated"):
        game.designate_second("MORGOTH")
    with pytest.raises(GameError, match="unknown power"):
        game.designate_second("SAURON")

    game.designate_second("DORIATH")
    assert game.second == "DORIATH"
    assert game.hooks.snapshot()["second"] == "DORIATH"  # type: ignore[index]
    data = game_to_json(game)
    assert data["second"] == "DORIATH"  # top-level: replayed on load
    assert data["snapshot"]["l2"]["second"] == "DORIATH"  # verified on load

    with pytest.raises(GameError, match="already DORIATH"):  # once only
        game.designate_second("MEN")


def test_second_place_refused_off_a_morgoth_win() -> None:
    """Refuse second place except on Morgoth's own victory.

    The designation is refused while the game is still running and again after
    a Valar intervention, which is an ending Morgoth did not win.
    """
    running = Game.new("beleriand", "beleriand", seed=0)
    with pytest.raises(GameError, match="after the game ends"):
        running.designate_second("DORIATH")

    intervention = Game.new("beleriand", "beleriand", start_year=0, seed=0)
    intervention.adjudicate()  # S0M -> F0M
    intervention.adjudicate()  # F0M -> the Valar intervene at Autumn of year 0
    assert intervention.is_over and "MORGOTH" not in intervention.winners
    with pytest.raises(GameError, match="only designated after a Morgoth win"):
        intervention.designate_second("DORIATH")


def test_countdown_intervention_marks_and_round_trips_defeat() -> None:
    """Report Morgoth defeated after the plain year-0 countdown intervention.

    Every Valar intervention defeats Morgoth, the countdown included, and the
    flag survives a serialization round trip. This is a real ``Game.new``
    document, so it replays, and ``second`` stays absent on an ending Morgoth
    did not win.
    """
    game = Game.new("beleriand", "beleriand", start_year=0, seed=0)
    game.adjudicate()
    game.adjudicate()
    assert game.is_over
    assert game.hooks.snapshot()["morgoth_defeated"] is True  # type: ignore[index]
    restored = game_from_text(game_to_text(game))
    assert restored.is_over
    assert restored.hooks.snapshot()["morgoth_defeated"] is True  # type: ignore[index]
    assert restored.hooks.snapshot()["second"] is None  # type: ignore[index]


def test_serialization_rejects_an_impossible_second() -> None:
    """Fail the load when a document carries a second place it cannot have.

    The load path re-applies a top-level ``"second"`` through
    :meth:`designate_second` and lets the refusal through: a Valar-intervention
    document has no second place, so an injected one is a hard load failure
    rather than a snapshot that quietly disagrees with the game.
    """
    game = Game.new("beleriand", "beleriand", start_year=0, seed=0)
    game.adjudicate()
    game.adjudicate()
    data = game_to_json(game)  # winners exclude Morgoth
    data["second"] = "DORIATH"
    with pytest.raises(SerializationError, match="second-place designation rejected"):
        game_from_json(data)


# -- the Oath of the Second ----------------------------------------------------


def test_an_oath_binds_second_place_at_the_win_and_closes_the_free_pick() -> None:
    """Place second by a standing Oath at the win, and close the free pick.

    The rulebook says *"If an Oath stands when the game ends, the Power named
    by it gets second place"*: the placement happens at the win itself, not by
    a later act, and the game being over closes the only channel that could
    break the Oath, so the post-game designation refuses by name. The snapshot
    carries the whole instrument, while the save carries no top-level
    ``"second"``, because the Oath reproduces it on load.
    """
    game = _morgoth_forty_win(oath="DORIATH")
    assert game.second == "DORIATH"
    assert (game.oath, game.oath_year, game.oaths_broken) == ("DORIATH", 25, 0)
    fragment = game.hooks.snapshot()
    assert fragment is not None
    assert fragment["second"] == "DORIATH"
    assert fragment["oath"] == "DORIATH" and fragment["oath_year"] == 25
    assert fragment["oaths_broken"] == 0 and fragment["oath_stabbed"] is False

    with pytest.raises(GameError, match="bound by the Oath sworn to DORIATH"):
        game.designate_second("MEN")

    data = game_to_json(game)
    assert data["oath"] == "DORIATH" and data["oath_year"] == 25
    assert "second" not in data  # the Oath places it; re-designating is refused
    assert "oaths_broken" not in data and "oath_stabbed" not in data  # defaults stay absent


def test_an_untouched_oath_channel_writes_no_key_to_the_save() -> None:
    """Write no Oath key at all when the game never touched the Oath channel.

    Every default is written only when it is not the default, so a game that
    never swore or broke an Oath produces a document, and an ``"l2"`` snapshot
    fragment, with no Oath key anywhere. That is what lets a save written by a
    game with no Oath survive its own snapshot check on load.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.adjudicate()
    data = game_to_json(game)
    assert not [key for key in data if "oath" in key]
    assert not [key for key in data["snapshot"]["l2"] if "oath" in key]
    assert game_from_text(game_to_text(game)).oath is None


def test_the_bar_rises_by_one_for_every_broken_oath() -> None:
    """Raise Morgoth's victory bar by exactly one centre per broken Oath.

    The rulebook says: *"Each Oath Morgoth has broken raises the number of Supply Centres he
    must Own for victory by one, for the rest of the game: forty-one after the
    first, forty-two after the second, and so on." The rise is flat and
    cumulative rather than escalating, and the reason string keeps its shape,
    so an unbroken game still prints "victory at 40".
    """
    hooks = BeleriandHooks(load_map("beleriand"), 0)
    assert hooks.victory_bar == 40
    for expected in (41, 42, 43):
        hooks.oath = "DORIATH"
        hooks.apply_break(25)
        assert hooks.victory_bar == expected
    assert _morgoth_forty_win().hooks.game_result().reason.endswith("(victory at 40)")  # type: ignore[union-attr]


def test_forty_with_one_broken_oath_falls_through_to_the_valar_in_year_zero() -> None:
    """Drop Morgoth to the Valar when a broken Oath lifts the bar above his count.

    The win check runs ahead of the year-0 countdown, so a raised bar does not
    postpone the win: at forty owned centres and a bar of forty-one, Morgoth
    falls through to the Valar, who defeat him. Breaking an Oath can therefore
    lose him the game outright, and *"An Oath has no effect if Morgoth does not
    win"*, so the power he courted is left unplaced with everyone else.
    """
    game = _morgoth_forty_win(oath="DORIATH", broken=1, year=0, expect_win=False)
    assert game.is_over and "MORGOTH" not in game.winners
    assert game.second is None and game.oath == "DORIATH"
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    assert hooks.morgoth_defeated and hooks.victory_bar == 41
    assert "Valar intervene" in hooks.game_result().reason  # type: ignore[union-attr]


def test_the_pending_sheet_gate_holds_both_verbs() -> None:
    """Refuse both Oath verbs once Morgoth has placed his orders for the season.

    An Oath is sworn or broken before Morgoth writes his sheet. Announced
    alongside it, the sworn power is already committed and cannot react, which
    would leave the raised bar as its only protection. The gate is the sheet
    itself -- an empty one still counts -- so it lifts the moment a fresh phase
    opens.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.submit("MORGOTH", [parse_order("3A(M) ANG H")])
    with pytest.raises(GameError, match="before Morgoth places his orders"):
        game.swear_oath("DORIATH")
    game.submit("MORGOTH", [])  # an empty sheet is still a sheet
    with pytest.raises(GameError, match="before Morgoth places his orders"):
        game.swear_oath("DORIATH")
    game.adjudicate()

    game.swear_oath("DORIATH")  # fresh phase, no sheet
    game.submit("MORGOTH", [parse_order("3A(M) ANG H")])
    with pytest.raises(GameError, match="before Morgoth places his orders"):
        game.break_oath()


def test_oath_refusal_matrix() -> None:
    """Refuse exactly the illegal Oath verbs, and nothing else.

    At most one Oath stands at a time, Morgoth cannot swear to himself or to a
    power that does not exist, a break needs a standing Oath, and both verbs
    are refused once the game is over. Breaking and swearing again in the same
    phase costs one raised centre, not two. Naming a power with no units is
    legal: the Oath binds an eliminated power all the same.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    with pytest.raises(GameError, match="stands under no Oath"):
        game.break_oath()
    with pytest.raises(GameError, match="cannot swear the Oath of the Second to himself"):
        game.swear_oath("MORGOTH")
    with pytest.raises(GameError, match="unknown power"):
        game.swear_oath("SAURON")

    game.swear_oath("DORIATH")
    assert game.oath == "DORIATH" and game.oath_year == 25
    with pytest.raises(GameError, match="already stands under an Oath to DORIATH"):
        game.swear_oath("MEN")

    game.break_oath()  # break-then-swear is one price, not two
    game.swear_oath("MEN")
    assert (game.oath, game.oaths_broken) == ("MEN", 1)

    finished = _morgoth_forty_win()
    with pytest.raises(GameError, match="the game is over"):
        finished.swear_oath("DORIATH")
    with pytest.raises(GameError, match="the game is over"):
        finished.break_oath()


def test_the_stab_exemption_reads_an_implicit_dislodge_not_coexistence() -> None:
    """Read the stab exemption as a dislodge, implicit or written, not coexistence.

    Only a dislodge, or the implicit dislodge a move into a capacity-1 province
    performs, frees Morgoth from the Oath's price. Moving into Anfauglith --
    capacity 6 -- puts an elf beside a Morgoth army without displacing it, so
    it frees nobody, while the Dislodge the same unit could have written
    instead does. The evidence is the written order and not its outcome: the
    Dislodge here fails against the garrison and the exemption fires anyway.
    The flag is per-Oath, so a break leaves it standing, with nothing left to
    free, and the next swearing clears it, which prices the second break.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.swear_oath("GONDOLIN")
    assert not game.oath_stabbed
    game.submit("FINROD", [parse_order("A DNN - ANF")])  # a different power's attack
    game.submit("GONDOLIN", [parse_order("A RIV H")])  # and none of the sworn power's own
    game.adjudicate()
    assert not game.oath_stabbed

    game.submit("GONDOLIN", [parse_order("A RIV - ANF")])  # an Area: coexistence, not an attack
    game.adjudicate()
    assert not game.oath_stabbed  # coexistence in an Area frees nobody

    game.submit("GONDOLIN", [parse_order("A RIV DIS MORGOTH A ANF")])
    record = game.adjudicate()
    rows = {row.text: row.resolution.value for row in record.resolutions}
    assert rows["A RIV DIS MORGOTH A ANF"] == "fails"  # outcome is still irrelevant
    assert game.oath_stabbed

    game.break_oath()
    assert game.oaths_broken == 0 and game.oath is None  # free
    assert game.oath_stabbed  # per-Oath, and there is no Oath left to free
    game.swear_oath("GONDOLIN")
    assert not game.oath_stabbed  # cleared by the new Oath
    game.break_oath()
    assert game.oaths_broken == 1  # so this one is priced


def test_a_support_of_an_attack_on_morgoth_is_a_stab() -> None:
    """Free Morgoth when the sworn power only supported an attack on him.

    The stab exemption reaches an order that *"has Supported such an Order"*,
    so the sworn power writing no attack of its own is enough. The support's
    subject is unwrapped and then tested exactly like a written order, which
    means the same capacity gate governs it -- and that is why one test covers
    both paths: supporting a move into an Area supports coexistence and frees
    nobody, while supporting the same move into a capacity-1 province supports
    an implicit dislodge and frees him.
    """
    coexistence = Game.new("beleriand", "beleriand", seed=0)
    coexistence.swear_oath("GONDOLIN")
    coexistence.submit("FINROD", [parse_order("A DNN - ANF")])
    coexistence.submit("GONDOLIN", [parse_order("A RIV S A DNN - ANF")])
    coexistence.adjudicate()
    assert not coexistence.oath_stabbed

    # Morgoth's own board holds him only in Anfauglith (an Area) and Angband
    # (unbounded, and foreign units are refused entry), so the capacity-1 half
    # needs a position where he stands in an ordinary province.
    game = _beleriand_game(
        {"MORGOTH": ["A DLN"], "FINGOLFIN": ["A MIT"], "GONDOLIN": ["A NEV"]}, {}
    )
    game.swear_oath("GONDOLIN")
    game.submit("MORGOTH", [parse_order("A DLN H")])
    game.submit("FINGOLFIN", [parse_order("A MIT - DLN")])
    game.submit("GONDOLIN", [parse_order("A NEV S A MIT - DLN")])
    game.adjudicate()
    assert game.oath_stabbed


def test_a_written_dislodge_of_a_morgoth_unit_is_a_stab() -> None:
    """Free Morgoth on a written Dislodge, whatever the order achieves.

    A Dislodge reads its target province off the victim the order itself names,
    so the exemption needs no capacity or occupancy inference for this shape.
    The attempt is enough: this one fails, a single army against the Anfauglith
    garrison, and the exemption fires anyway.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.swear_oath("GONDOLIN")
    game.submit("GONDOLIN", [parse_order("A RIV DIS MORGOTH A ANF")])
    record = game.adjudicate()
    rows = {row.text: row.resolution.value for row in record.resolutions}
    assert rows["A RIV DIS MORGOTH A ANF"] == "fails"
    assert game.oath_stabbed


def test_oath_state_round_trips_with_a_broken_count_and_a_live_pending_sheet() -> None:
    """Round-trip every unflushed thing a save can carry at once.

    A save may hold a standing Oath, a price already paid, a free-break
    exemption already earned, a fact granted but not yet flushed, and a sheet
    drafted but not adjudicated -- all at the same time. The Oath state is
    re-applied before the pending sheets, so the document cannot refuse itself
    under the placed-orders gate on its own way back in.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.swear_oath("DORIATH")
    game.break_oath()
    game.swear_oath("GONDOLIN")
    game.submit("GONDOLIN", [parse_order("A RIV DIS MORGOTH A ANF")])  # earns the exemption
    game.adjudicate()
    game.grant("TELERI", "MEN", FLEET_FACT)  # an unflushed grant rides alongside
    game.submit("MORGOTH", [parse_order("3A(M) ANG H")])
    assert (game.oath, game.oaths_broken, game.oath_stabbed) == ("GONDOLIN", 1, True)

    data = game_to_json(game)
    assert data["oath"] == "GONDOLIN" and data["oaths_broken"] == 1
    assert data["oath_stabbed"] is True  # a set flag needs its own top-level key

    text = game_to_text(game)
    restored = game_from_text(text)
    assert (restored.oath, restored.oath_year, restored.oaths_broken) == ("GONDOLIN", 25, 1)
    assert restored.oath_stabbed
    assert restored.pending_sheets == game.pending_sheets
    assert game_to_text(restored) == text  # byte-for-byte, including the fragment
    # The restored sheet is live, so the placed-orders gate holds on the way
    # out too: the save came back into exactly the window it was taken in.
    with pytest.raises(GameError, match="before Morgoth places his orders"):
        restored.break_oath()


def test_the_recorded_stab_flag_outranks_what_the_replay_recomputes() -> None:
    """Let the recorded stab flag outrank whatever the replay recomputes.

    The Oath is restored before the phase replay, so that the bar and any
    Oath-bound win come out right, and that widens the window the free-break
    flag is derived over: the replay sees the Oath standing during phases that
    ran before it was sworn. The recorded value is therefore re-applied
    afterwards and wins. Without that second pass this document would not load.

    The pre-Oath phase must therefore hold an order the predicate would fire
    on, or the seam is tested against a replay that recomputes False anyway --
    a plain move into Anfauglith would not do, because coexistence in an Area
    frees nobody. The control below shows that this Dislodge is such an order.
    """
    control = Game.new("beleriand", "beleriand", seed=0)
    control.swear_oath("GONDOLIN")
    control.submit("GONDOLIN", [parse_order("A RIV DIS MORGOTH A ANF")])
    control.adjudicate()
    assert control.oath_stabbed  # the same phase, with the Oath standing, DOES stab

    game = Game.new("beleriand", "beleriand", seed=0)
    game.submit("GONDOLIN", [parse_order("A RIV DIS MORGOTH A ANF")])  # BEFORE any Oath
    game.adjudicate()
    game.swear_oath("GONDOLIN")
    assert not game.oath_stabbed  # nothing has happened since it was sworn

    data = game_to_json(game)
    assert "oath_stabbed" not in data  # a False flag needs no top-level key ...
    assert data["snapshot"]["l2"]["oath_stabbed"] is False  # ... but the fragment pins it

    text = game_to_text(game)
    restored = game_from_text(text)
    assert restored.oath == "GONDOLIN" and not restored.oath_stabbed
    assert game_to_text(restored) == text


def test_a_restored_oath_binds_a_win_that_comes_after_it() -> None:
    """Bind a win that lands after the Oath was sworn, on the restored state.

    The Oath is not a post-game fact the way ``second`` is, which is why the
    loader restores it before the phase replay rather than only after: it sets
    the bar every Autumn boundary tests against and places second at the moment
    the win lands, so a game replayed without it would run at the wrong bar and
    finish with nobody placed.

    A ``Game.new`` Beleriand game cannot be driven to forty centres inside the
    suite, so no round-trip document exercises this; what the mechanism needs
    is checked here, on the same call the loader makes.
    """
    from entente.engine.orders import parse_unit_spec

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    phase = Phase(25, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    state = State.of("beleriand", phase.tag, [Unit("MORGOTH", *parse_unit_spec("3A ANG"))])
    ownership = {c: p for p, cs in graph.home_centers.items() if p == "MORGOTH" for c in cs}
    ownership.update(dict.fromkeys(_MORGOTH_FORTY_OWNED, "MORGOTH"))
    hooks.register_start(state, ownership)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)

    game.restore_oath_state("DORIATH", 25, 1, stabbed=True)  # the loader's own call
    assert game.oaths_broken == 1 and game.oath_stabbed
    game.adjudicate()
    game.adjudicate()
    assert not game.is_over  # forty against a bar of forty-one

    game.break_oath()  # free: the flag came back with the state
    assert game.oaths_broken == 1 and game.oath is None


def test_serialization_rejects_impossible_oath_states() -> None:
    """Fail the load on every malformed Oath field rather than dropping it.

    A corrupt broken-count, an unknown or forbidden power, or an Oath with no
    year is corruption in the document, so the loader's door refuses it
    outright instead of ignoring the field and returning a game that disagrees
    with its own save.
    """
    game = Game.new("beleriand", "beleriand", seed=0)
    game.swear_oath("DORIATH")
    base = game_to_json(game)
    for patch, why in [
        ({"oath": "SAURON"}, "unknown power"),
        ({"oath": "MORGOTH"}, "cannot swear the Oath of the Second to himself"),
        ({"oaths_broken": -1}, "non-negative integer"),
        ({"oaths_broken": "two"}, "non-negative integer"),
        ({"oath_year": None}, "must record the year"),
        ({"oath": None, "oath_year": 25}, "no year can be recorded"),
    ]:
        data = {**base, **patch}
        with pytest.raises(SerializationError, match=f"oath state rejected.*{why}"):
            game_from_json(data)


def test_a_classic_save_carrying_an_oath_fails_to_load() -> None:
    """Refuse a document that records an Oath for a game which has no Oath.

    The base hooks answer that a standard game has no Oath of the Second, and
    the loader lets that refusal through: an Oath in such a document is
    corruption, not a field to ignore.
    """
    game = Game.new("standard", "standard")
    game.adjudicate()
    data = {**game_to_json(game), "oath": "FRANCE", "oath_year": 1901}
    with pytest.raises(SerializationError, match="no Oath of the Second"):
        game_from_json(data)


def _countdown_scoreboard(owned: dict[str, list[str]], units: dict[str, str]) -> Game:
    """Drive a minimal year-0 game to the Valar countdown with a set ownership.

    The scoreboard the intervention ranks is then exactly the supplied
    positions. The game is assembled directly, like
    :func:`_morgoth_forty_win`, so it does not replay; an owned centre with no
    unit on it survives the Autumn boundary, which is what lets a scoreboard be
    written out this way.
    """
    from entente.engine.orders import parse_unit_spec

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    phase = Phase(0, Season.SPRING, PhaseKind.MOVEMENT, year_step=-1)
    state = State.of(
        "beleriand", phase.tag, [Unit(p, *parse_unit_spec(s)) for p, s in units.items()]
    )
    ownership = {center: power for power, centers in owned.items() for center in centers}
    hooks.register_start(state, ownership)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)
    game.adjudicate()  # S0M -> F0M
    game.adjudicate()  # F0M -> the Valar countdown intervenes
    assert game.is_over and game.phase == "F0R"
    return game


def test_intervention_breaks_a_net_gain_tie_by_absolute_weight() -> None:
    """Break a net-gain tie at the intervention by absolute owned weight.

    When the Valar intervene the victory goes to the greatest net gain, and a
    tie there breaks to the greatest absolute owned weight. The position sets
    the Feanorians, Doriath and the Teleri all at net +2, but the Feanorians
    own 8 centres to the others' 5, so the Feanorians win alone rather than
    the three sharing.
    """
    feanor_home = ["GEL", "HEL", "HID", "HIG", "MMA", "THA"]
    game = _countdown_scoreboard(
        owned={
            "FEANORIANS": [*feanor_home, "ARV", "CBB"],  # home 6 + 2 neutral = abs 8
            "DORIATH": ["FON", "FOR", "MEN", "EIT", "GVA"],  # home 3 + 2 = abs 5
            "TELERI": ["BRI", "EGL", "SFA", "IOB", "LAM"],  # home 3 + 2 = abs 5
        },
        units={"FEANORIANS": "A THA", "DORIATH": "A FOR", "TELERI": "A BRI"},
    )
    result = game.hooks.game_result()
    assert result is not None
    assert result.scores["FEANORIANS"] == 2  # net gain: the three-way tie
    assert result.scores["DORIATH"] == 2
    assert result.scores["TELERI"] == 2
    assert game.supply_count("FEANORIANS") == 8  # absolute weight: the tiebreak
    assert game.supply_count("DORIATH") == 5
    assert game.supply_count("TELERI") == 5
    assert game.winners == ("FEANORIANS",)
    assert "ties broken by total centers owned" in result.reason


def test_intervention_keeps_a_full_tie_joint() -> None:
    """Keep a win joint when the tiebreak itself ties.

    Powers equal on both net gain and absolute owned weight share the win: the
    tiebreak resolves nothing and the rulebook names no further rank, so a
    joint victory stands. Doriath and the Teleri each net +2 at exactly 5
    centres.
    """
    game = _countdown_scoreboard(
        owned={
            "DORIATH": ["FON", "FOR", "MEN", "EIT", "GVA"],  # net +2, abs 5
            "TELERI": ["BRI", "EGL", "SFA", "IOB", "LAM"],  # net +2, abs 5
        },
        units={"DORIATH": "A FOR", "TELERI": "A BRI"},
    )
    assert game.winners == ("DORIATH", "TELERI")


# -- supply attribution: which centre funds which unit ---------------------------


def _elvish_start() -> Game:
    return Game.new("beleriand", "beleriand", seed=0)


def test_starting_armies_spend_their_home_centers() -> None:
    """Charge every starting army to a home centre, leaving no unspent supply.

    The armies a power starts with are already supplied by its own centres, so
    the opening position holds nothing spare and no home centre can fund a
    fresh double. Every power opens with its unit count equal to its owned
    weight, which is what makes the accounting close exactly.
    """
    game = _elvish_start()
    hooks = game.hooks
    for province, owner in game.ownership.items():
        weight = game.graph.province(province).supply_center_count
        assert hooks.spent_supply.get(province, 0) == weight, (
            f"{province} ({owner}) should open fully spent on the starting armies"
        )


def test_a_captured_center_arrives_unspent_for_its_new_owner() -> None:
    """Free a captured centre's supply point for its new owner.

    Each unit remembers the centre that supplies it, so a centre taken from
    another power stops supplying that power's army and arrives unspent; the
    capturing army keeps the centre it started on.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    units = [
        Unit("FINROD", UnitKind.ARMY, Location("RON")),
        Unit("TELERI", UnitKind.ARMY, Location("NAR")),
    ]
    state = State.of("beleriand", "S25M", units)
    hooks.register_start(state, {"NAR": "TELERI", "RON": "FINROD"})
    assert hooks.spent_supply == {"NAR": 1, "RON": 1}
    hooks._reconcile_spent(state, {"NAR": "FINROD", "RON": "FINROD"})
    assert hooks.spent_supply == {"RON": 1}


def test_a_starting_unit_is_supplied_by_the_centre_it_starts_on() -> None:
    """Charge a starting unit to the centre it stands on.

    The rulebook: "the Units a Player starts the game with are supplied by
    their Home Supply Centres". Ossiriand's army starts on TGA, so TGA is spent
    and both points of the two-Centre SOS stay free.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    state = State.of("beleriand", "S25M", [Unit("OSSIRIAND", UnitKind.ARMY, Location("TGA"))])
    ownership = {"SOS": "OSSIRIAND", "TGA": "OSSIRIAND"}
    hooks.register_start(state, ownership)
    assert hooks.spent_supply == {"TGA": 1}
    assert hooks._free_supply("SOS") == 2
    assert hooks._free_supply("TGA") == 0


def test_a_unit_without_a_centre_takes_the_newest_free_one() -> None:
    """Move a unit whose centre was lost to the newest centre its power owns.

    The centres waiting out their year for a double keep their wait: TELERI's
    army loses BRI, and of EGL (owned from the start) and SFA (taken later)
    it takes SFA, so EGL stays free for a double.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    state = State.of("beleriand", "S25M", [Unit("TELERI", UnitKind.ARMY, Location("BRI"))])
    hooks.register_start(state, {"BRI": "TELERI", "EGL": "TELERI"})
    hooks.owner_since["SFA"] = 1
    hooks._reconcile_spent(state, {"BRI": "FINROD", "EGL": "TELERI", "SFA": "TELERI"})
    assert hooks.spent_supply == {"SFA": 1}


def test_a_dead_unit_releases_the_supply_point_it_held() -> None:
    """Return a dead unit's supply point to the centre that funded it.

    Each unit remembers its centre, so a death releases that very point, and a
    centre that was fully spent becomes able to fund a double again.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    two = [
        Unit("FINROD", UnitKind.ARMY, Location("NAR")),
        Unit("FINROD", UnitKind.ARMY, Location("RON")),
    ]
    ownership = {"NAR": "FINROD", "RON": "FINROD"}
    hooks.register_start(State.of("beleriand", "S25M", two), ownership)
    assert hooks.spent_supply == {"NAR": 1, "RON": 1}
    dead = next(e for e in hooks.ledger.alive() if e.location.province == "RON")
    hooks._kill_entry(dead, "dies in a test")
    hooks._reconcile_spent(State.of("beleriand", "F25M", two[:1]), ownership)
    assert hooks.spent_supply == {"NAR": 1}


def test_a_dying_twin_frees_the_older_centre() -> None:
    """Free the centre held longest when one of two identical twins dies.

    Twins cannot be told apart, so which one dies is the ledger's choice, and
    the centre set free is the older one, which may be waiting for a double.
    Two FINROD armies stand in Talath Dimen, supplied by NAR and by the newer
    RON; whichever identity dies, NAR is freed.
    """
    graph = load_map("beleriand")
    for victim_centre in ("NAR", "RON"):
        hooks = BeleriandHooks(graph, 0)
        hooks.owner_since["RON"] = 1
        twins = [Unit("FINROD", UnitKind.ARMY, Location("TAL"))] * 2
        state = State.of("beleriand", "S25M", twins)
        ownership = {"NAR": "FINROD", "RON": "FINROD"}
        hooks.register_start(state, ownership)
        assert hooks.spent_supply == {"NAR": 1, "RON": 1}
        dead = next(e for e in hooks.ledger.alive() if e.supplied_by == victim_centre)
        hooks._kill_entry(dead, "dies in a test")
        hooks._reconcile_spent(State.of("beleriand", "F25M", twins[:1]), ownership)
        assert hooks.spent_supply == {"RON": 1}


def test_a_men_build_takes_the_centre_it_is_built_in() -> None:
    """Supply a Men build from the centre it stands on when that centre is free.

    Men build at any owned centre, and a unit built there takes that centre's
    point before the newest one.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    state = State.of("beleriand", "S25M", [])
    ownership = {"BRI": "MEN", "EGL": "MEN"}
    hooks.register_start(state, ownership)
    hooks.owner_since["EGL"] = 1
    built = hooks.ledger.create(Unit("MEN", UnitKind.ARMY, Location("BRI")))
    hooks._reconcile_spent(state, ownership, {built.uid: "BRI"})
    assert hooks.spent_supply == {"BRI": 1}


def test_a_bare_winter_disband_does_not_remove_a_double() -> None:
    """Fail a bare ``Disband`` that names a base strength the board does not hold.

    Beleriand runs its own winter adjudication, so the base-strength law holds
    there too: a written base strength binds, and a bare spec means base
    strength 1. A player owing one removal, with a doubled ``2A`` standing alone
    at NOG, writes ``Disband A NOG``; that names a unit of base strength 1 which
    is not there, so the order fails.
    The quota is still owed, so the civil-disorder sweep removes the double
    anyway -- the point is that the order does not do it silently, and the
    enumeration offers only the ``Disband 2A NOG`` that says what it means.
    """
    from entente.engine.game import Game
    from entente.engine.map import load_map
    from entente.engine.orders import format_order, parse_adjustment_order, parse_unit_spec
    from entente.engine.phase import Phase, PhaseKind, Season
    from entente.engine.rules import RULESETS
    from entente.engine.state import State, Unit
    from entente.variants.beleriand.layer2 import BeleriandHooks

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    phase = Phase(25, Season.WINTER, PhaseKind.ADJUSTMENT, year_step=-1)
    state = State.of(
        "beleriand",
        phase.tag,
        [Unit("NAUGRIM", *parse_unit_spec("2A NOG")), Unit("MORGOTH", *parse_unit_spec("A ANF"))],
    )
    ownership = {"ANF": "MORGOTH", "NOG": "MORGOTH"}
    hooks.register_start(state, ownership)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)

    assert game.build_quota("NAUGRIM") == -1
    assert {format_order(o) for o in game.legal_orders("NAUGRIM")} == {"Disband 2A NOG"}

    game.submit("NAUGRIM", [parse_adjustment_order("Disband A NOG")])
    record = game.adjudicate()
    outcome = {r.text: r.resolution for r in record.resolutions if r.power == "NAUGRIM"}
    assert outcome["Disband A NOG"] is Resolution.FAILS


def test_twins_written_alike_are_disbanded_one_per_line() -> None:
    """Remove one twin per identically worded Disband line.

    The rulebook: "identically worded Disband Orders remove different twins". A
    power owing two removals writes ``Disband A NEB`` twice for its two NEB
    twins; both go, and civil disorder takes nothing it did not name. A line
    repeated with no unit left for it is still the same order again (DATC 6.J.2).
    """
    from entente.engine.game import Game
    from entente.engine.map import load_map
    from entente.engine.orders import parse_adjustment_order, parse_unit_spec
    from entente.engine.phase import Phase, PhaseKind, Season
    from entente.engine.rules import RULESETS
    from entente.engine.state import State, Unit
    from entente.variants.beleriand.layer2 import BeleriandHooks

    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, seed=0)
    phase = Phase(25, Season.WINTER, PhaseKind.ADJUSTMENT, year_step=-1)
    units = [Unit("FEANORIANS", *parse_unit_spec(spec)) for spec in ("A NEB", "A NEB", "A SEB")]
    state = State.of("beleriand", phase.tag, units)
    ownership = {"HEL": "FEANORIANS"}
    hooks.register_start(state, ownership)
    game = Game(RULESETS["beleriand"], graph, state, ownership, phase, hooks=hooks)

    assert game.build_quota("FEANORIANS") == -2
    game.submit(
        "FEANORIANS",
        [
            parse_adjustment_order(text)
            for text in ("Disband A NEB", "Disband A NEB", "Disband A NEB")
        ],
    )
    record = game.adjudicate()
    rows = [r.resolution for r in record.resolutions if r.power == "FEANORIANS"]
    assert rows == [Resolution.SUCCEEDS, Resolution.SUCCEEDS]
    assert sorted(str(u) for u in game.state.units if u.power == "FEANORIANS") == ["A SEB"]


# -- Beleriand's own civil-disorder law ------------------------------------------


def test_the_removal_metric_is_measured_per_unit_kind() -> None:
    """Measure the civil-disorder removal distance on each kind's own graph.

    The rulebook allows armies *"to Move over any territory"* but fleets *"to only Move
    over places where fleets may Move", so the two kinds read different graphs:
    the army metric spans the whole board including the seas, the fleet metric
    only the fleet lanes, on which no inland province ever sits. Both are seeded
    from the power's owned Centres.
    """
    graph = load_map("beleriand")
    army = _army_steps(graph, {"CBB"})
    fleet = _fleet_steps(graph, {"CBB"})

    assert set(army) == set(graph.provinces), "every province is on the army metric"
    assert army["BOB"] == 1, "an army's metric crosses the Bay of Balar -- 'any territory'"

    fleet_provinces = {location.province for location in fleet}
    inland = {
        key for key, province in graph.provinces.items() if province.terrain is Terrain.INLAND
    }
    assert not (fleet_provinces & inland), "no inland province is on the fleet metric"
    # An inland Centre seeds the army metric and nothing at all for fleets.
    assert _army_steps(graph, {"GVA"})["GVA"] == 0
    assert _fleet_steps(graph, {"GVA"}) == {}


def test_a_fleet_with_no_owned_coastal_centre_is_removed_before_any_army() -> None:
    """Remove a fleet first when the power owns no coastal Centre to measure to.

    The rulebook's clause *"if no coastal supply centre exists ... then the
    fleets get prio over armies in disbands"* needs no branch of its own. A
    fleet counts only sea and coastal lanes, so a power holding nothing but
    inland Centres gives its fleets no destination at all: they are infinitely
    far and outrank every army. Owning one coastal Centre restores an ordinary
    distance, and the army goes instead.
    """
    graph = load_map("beleriand")
    army = Unit("TELERI", UnitKind.ARMY, Location("CNI"))
    fleet = Unit("TELERI", UnitKind.FLEET, Location("NBE"))

    inland_only = {"GVA": "TELERI"}  # Gelion Vale: a Centre no fleet can reach
    assert rulebook_civil_disorder_removals(graph, inland_only, [army, fleet], 1) == (fleet,)

    coastal = {"CBB": "TELERI"}  # Coasts of the Bay of Balar: one lane from NBE
    assert rulebook_civil_disorder_removals(graph, coastal, [army, fleet], 1) == (army,)

    # The gate is power-level: one owned coastal Centre keeps every fleet on
    # the ordinary sea-and-coast metric, however many inland Centres sit beside
    # it -- "when there is a coastal-owned-SC, the standard only-coasts-and-seas
    # rule applies".
    both = {"GVA": "TELERI", "CBB": "TELERI"}
    assert rulebook_civil_disorder_removals(graph, both, [army, fleet], 1) == (army,)


def test_a_defeated_morgoths_winter_sheet_is_discarded_outright() -> None:
    """Discard a defeated Morgoth's winter sheet before reading a line of it.

    Any death of the Morgoth unit defeats him permanently and leaves his units
    in civil disorder, so every later sheet is thrown away: each line resolves
    as a failure and nothing is built, even at Angband, where his eight owned
    supply points would otherwise entitle him. The branch is exercised here
    rather than left to whichever seeded game in the suite happens to reach it.
    """
    graph = load_map("beleriand")
    hooks = BeleriandHooks(graph, 0)
    ownership = {"ANG": "MORGOTH", "ANF": "MORGOTH"}
    state = State.of("beleriand", "W25A", [Unit("MORGOTH", UnitKind.ARMY, Location("ANF"))])
    hooks.register_start(state, ownership)
    hooks.morgoth_defeated = True

    result = adjudicate_winter(
        hooks, state, ownership, {"MORGOTH": [parse_adjustment_order("Build A ANG")]}
    )
    assert [(r.text, r.resolution) for r in result.resolutions] == [
        ("Build A ANG", Resolution.FAILS)
    ]
    assert result.built == ()
    assert result.new_state.units == state.units


def test_stranded_fleets_are_ranked_against_each_other_on_the_full_graph() -> None:
    """Rank stranded fleets against each other on the full graph, not the alphabet.

    The rulebook says *"when all fleets are unreachable, then the one is
    disbanded that is the farthest from owned SCs when considering a full
    graph"*. Stranded fleets still outrank every army, but among themselves the
    ordering stays a distance -- the army metric, "over any territory" --
    rather than falling straight to the alphabet.

    Play cannot currently reach a position where this changes the outcome: over
    the 1224 pairs of non-Centre fleet provinces reachable from an owned inland
    Centre, the full-graph ordering never contradicts the alphabet, and a fleet
    parked on a coastal Centre would claim it at the Fall->Winter update and
    un-strand the whole fleet. This is therefore a direct test of the
    function's contract, which is where the rule has to live until the board or
    a variant produces a case.
    """
    graph = load_map("beleriand")
    ownership = {"DLN": "TELERI"}  # Dor-Lomin: inland, so no fleet can measure

    near = Unit("TELERI", UnitKind.FLEET, Location("NEV"))  # full graph: 1 step
    far = Unit("TELERI", UnitKind.FLEET, Location("NBE"))  # full graph: 2 steps
    inland_army = Unit("TELERI", UnitKind.ARMY, Location("CNI"))

    # Distance decides, though "Nevrast" sorts before "North Belegaer".
    assert rulebook_civil_disorder_removals(graph, ownership, [near, far], 1) == (far,)
    # Both still outrank an army the metric can measure, however far inland.
    assert rulebook_civil_disorder_removals(graph, ownership, [inland_army, near, far], 2) == (
        far,
        near,
    )


def test_removals_are_a_no_op_when_nothing_is_owed() -> None:
    graph = load_map("beleriand")
    unit = Unit("TELERI", UnitKind.ARMY, Location("GOS"))
    assert rulebook_civil_disorder_removals(graph, {"CBB": "TELERI"}, [unit], 0) == ()
    assert rulebook_civil_disorder_removals(graph, {"CBB": "TELERI"}, [], 1) == ()
