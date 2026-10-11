"""Assert the Beleriand variant's whole-game invariants over random play.

Random seeded games over the variant map, driven through the public ``Game``
facade (movement sheets from the uniform RandomAgent -- which draws
Layer-2 orders from ``legal_layer2_orders`` too -- plus seeded knowledge
grants).
Per executed phase:

- **ledger-board bijection** -- the identity ledger's on-board entries are
  exactly the state's units as a multiset (limbo entries exactly the pending
  dislodgements), so no identity is ever invented, duplicated, or leaked;
- **mask conservation** -- the total mask count changes only by the phase's
  make events and the masks lost with dead identities (transfers conserve);
- **credential soundness** -- no identity stands in devastated ground without
  a mask (Morgoth's excepted) or inside the Girdle without its power holding
  the permission, so the credential-claiming discipline holds under fire;
- **one-way provinces** -- Angband ever holds only MORGOTH units and Eriador
  only MEN, and no successful move or retreat lands a unit in Angband unless
  it is a MORGOTH strength-3 special, nor any unit in Eriador at all;
- **Morgoth's knowledge and mask exclusion** -- Morgoth never holds the
  fleet-building knowledge or a mask, at every phase boundary; Girdle
  membership is not asserted against, because he may legitimately acquire it,
  by being attacked from inside or by a written grant from Men or the
  Naugrim;
- **determinism given the seed** -- two identically-driven games serialize
  identically (the Ulmo dice and Teleri picks come from the seed alone);
- **serialization** -- the v2 document round-trips at every phase.
"""

from __future__ import annotations

import contextlib
import random
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from entente.agents import RandomAgent
from entente.engine.game import Game
from entente.engine.serialization import game_from_text, game_to_text
from entente.variants.beleriand.layer2 import (
    FLEET_FACT,
    GIRDLE_FACT,
    MEN,
    MORGOTH,
    ULMO_PROVINCES,
    BeleriandHooks,
)

FACTS = (FLEET_FACT, GIRDLE_FACT)


def _maybe_grant(game: Game, seed: int) -> None:
    """Offer one knowledge grant per phase, on a seeded and replay-stable roll."""
    rng = random.Random(f"grant:{seed}:{game.phase}")
    if rng.random() > 0.25:
        return
    powers = list(game.powers)
    giver, receiver = rng.choice(powers), rng.choice(powers)
    fact = rng.choice(FACTS)
    with contextlib.suppress(Exception):  # illegal grants are the common case
        game.grant(giver, receiver, fact)


def play_beleriand_game(seed: int, phases: int, *, roundtrip: bool = False) -> Game:
    """Play a seeded game for ``phases`` phases, asserting the invariants each time.

    With ``roundtrip`` the serialized document is checked at every phase as
    well, which is a separate property and costs enough to be opt-in.
    """
    game = Game.new("beleriand", "beleriand", seed=seed)
    agent = RandomAgent(seed)
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    for _ in range(phases):
        if game.is_over:
            break
        _maybe_grant(game, seed)
        for power in game.powers:
            sheet = list(agent.orders(game, power))
            if sheet:
                game.submit(power, sheet)
        before = {u.uid: u.masks for u in hooks.ledger.alive()}
        before_loc = {u.uid: u.location.province for u in hooks.ledger.alive()}
        record = game.adjudicate()
        assert_ledger_board_bijection(game)
        assert_mask_conservation(hooks, before, record.events)
        assert_credential_soundness(game)
        assert_offmap_ownership(game)
        assert_no_illegal_offmap_landing(hooks, before_loc)
        assert_morgoth_knowledge_and_masks(hooks)
        if roundtrip:
            assert_roundtrip(game)
    return game


def assert_ledger_board_bijection(game: Game) -> None:
    """Assert the ledger's entries and the board's units are the same multiset.

    On-board entries match the state's units and limbo entries match the
    pending dislodgements, so no identity is invented, duplicated or leaked.
    """
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    board = Counter(
        (u.power, u.kind.value, u.strength, u.location.province, u.location.coast)
        for u in hooks.ledger.board()
    )
    state = Counter(
        (u.power, u.kind.value, u.strength, u.location.province, u.location.coast)
        for u in game.state.units
    )
    assert board == state, f"ledger/board drift at {game.phase}"
    limbo = Counter(
        (u.power, u.kind.value, u.strength, u.location.province) for u in hooks.ledger.limbo()
    )
    pending = Counter(
        (d.unit.power, d.unit.kind.value, d.unit.strength, d.unit.location.province)
        for d in game.pending_dislodgements
    )
    assert limbo == pending, f"limbo/dislodgement drift at {game.phase}"


def assert_mask_conservation(
    hooks: BeleriandHooks, before: dict[int, int], events: tuple[str, ...]
) -> None:
    """Assert the phase changed the mask total only by making and by deaths.

    A transfer moves a mask without creating or destroying one, so the total
    can only rise by the phase's own make events and fall with the masks that
    died with their carriers.
    """
    after = {u.uid: u.masks for u in hooks.ledger.alive()}
    # Only the Naugrim make masks, and a Naugrim order makes two, so counting
    # the events is enough. " makes " is unambiguous: no other emitted event
    # uses that verb.
    made = 2 * sum(1 for e in events if " makes " in e)
    died = sum(masks for uid, masks in before.items() if uid not in after)
    total_before = sum(before.values())
    total_after = sum(after.values())
    assert total_after == total_before + made - died, (
        f"mask conservation broke: {total_before} + {made} - {died} != {total_after}"
    )
    assert all(m >= 0 for m in after.values())


def assert_credential_soundness(game: Game) -> None:
    """Assert no identity stands anywhere its power holds no credential for.

    Devastated ground needs a mask, Morgoth's own units excepted, and the
    Girdle needs the passage.
    """
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    for entry in hooks.ledger.board():
        province = entry.location.province
        if province in hooks.devastated and entry.power != MORGOTH:
            assert entry.masks > 0, f"unmasked {entry.power} inside devastated {province}"
        if province in ("FOR", "FON"):
            assert entry.power in hooks.girdle, f"{entry.power} inside the Girdle"


def assert_morgoth_knowledge_and_masks(hooks: BeleriandHooks) -> None:
    """Assert Morgoth holds neither the fleet-building knowledge nor a mask.

    The rulebook's "Doriath" and "Dwarven Masks" sections shut both channels
    to him outright, so the assertion holds at every phase boundary over the
    random-game corpus. The Girdle is not asserted against: Morgoth may
    legitimately come to hold it, so a game where he does is not a fault, and
    the case files exercise that channel directly.
    """
    assert MORGOTH not in hooks.fleet_knowledge
    for entry in hooks.ledger.alive():
        if entry.power == MORGOTH:
            assert entry.masks == 0, f"MORGOTH {entry.spec()} holds {entry.masks} masks"


def assert_offmap_ownership(game: Game) -> None:
    """Assert Angband holds only Morgoth's units and Eriador only Men's.

    Each is restricted to one power, and entry from the board is banned
    outright, so no other power can ever appear in either.
    """
    for unit in game.state.units:
        if unit.location.province == "ANG":
            assert unit.power == MORGOTH, f"non-MORGOTH {unit.power} in Angband"
        elif unit.location.province == "ERI":
            assert unit.power == MEN, f"non-MEN {unit.power} in Eriador"


def assert_no_illegal_offmap_landing(hooks: BeleriandHooks, before: dict[int, str]) -> None:
    """Assert nothing entered Angband or Eriador from the board this phase.

    Only a Morgoth strength-3 special may land in Angband, and nothing at all
    may land in Eriador. An identity that already existed before the phase and
    now stands in one of them arrived by move or retreat; a winter build, a
    migration and Glaurung's arrival all create fresh uids, which are absent
    from ``before`` and so correctly ignored.
    """
    for entry in hooks.ledger.alive():
        prev = before.get(entry.uid)
        if prev is None:
            continue  # created this phase (build / migration / arrival)
        province = entry.location.province
        if province == "ANG" and prev != "ANG":
            assert entry.power == MORGOTH and entry.strength >= 3, (
                f"{entry.power} {entry.strength}A landed in Angband from {prev}"
            )
        if province == "ERI" and prev != "ERI":
            raise AssertionError(f"{entry.power} landed in Eriador from {prev}")


def assert_roundtrip(game: Game) -> None:
    """Assert the game survives a save and a load with everything intact."""
    text = game_to_text(game)
    restored = game_from_text(text)
    assert restored.state == game.state
    assert restored.ownership == game.ownership
    assert restored.phase == game.phase
    assert restored.is_over == game.is_over
    assert game_to_text(restored) == text


@settings(max_examples=15, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 12))
def test_random_l2_games_maintain_invariants(seed: int, phases: int) -> None:
    play_beleriand_game(seed, phases)


def test_double_unit_retreat_keeps_ledger_bijection_regression() -> None:
    """Regression pin for the ledger/board drift the property found.

    A dislodged double that retreats keeps its identity. The ledger once
    keyed a retreat's limbo identity on the base strength its text implied,
    and a text written without one never matched a 2F identity: the entry
    was swept into the disband set while the adjudicator kept the double on
    the board. The ledger now moves an identity of the very dislodgement the
    adjudicator bound, base strength included. :func:`play_beleriand_game`
    asserts the bijection at every phase, so naming a seed with a successful
    retreat of a double pins it deterministically, without depending on the
    hypothesis example database; under seed 11 the double ``2F SFA``
    retreats to SBE at S24R.

    The seed is chosen, not arbitrary: it is the earliest-firing hit in
    seeds 0..399 for the trigger this needs, a dislodged unit of strength 2
    or more whose retreat resolves as a success. Such a seed is only as
    stable as the draws it rides on -- any change that widens an order menu
    shifts every later draw -- so re-scanning that range for the trigger is
    how the pin is re-established when it goes stale.
    """
    play_beleriand_game(11, 8)


@settings(max_examples=6, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 6))
def test_l2_serialization_round_trips_at_every_phase(seed: int, phases: int) -> None:
    play_beleriand_game(seed, phases, roundtrip=True)


@settings(max_examples=8, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 8))
def test_same_seed_same_beleriand_transcript(seed: int, phases: int) -> None:
    first = play_beleriand_game(seed, phases)
    second = play_beleriand_game(seed, phases)
    assert game_to_text(first) == game_to_text(second)


def test_ulmo_dice_are_seeded_and_reproducible() -> None:
    first = Game.new("beleriand", "beleriand", seed=99).hooks
    second = Game.new("beleriand", "beleriand", seed=99).hooks
    third = Game.new("beleriand", "beleriand", seed=100).hooks
    assert isinstance(first, BeleriandHooks)
    assert isinstance(second, BeleriandHooks)
    assert isinstance(third, BeleriandHooks)
    assert first.ulmo_delays == second.ulmo_delays
    assert set(first.ulmo_delays) == set(ULMO_PROVINCES)
    assert all(delay in (1, 2, 3) for delay in first.ulmo_delays.values())
    assert any(
        Game.new("beleriand", "beleriand", seed=s).hooks.ulmo_delays != first.ulmo_delays  # type: ignore[attr-defined]
        for s in (100, 101, 102)
    ), third  # different seeds roll different dice (with overwhelming probability)


def test_full_countdown_reaches_the_valar() -> None:
    """The demo path: a hold-heavy full game runs 25->0 and the Valar end it."""
    game = Game.new("beleriand", "beleriand", seed=5)
    steps = 0
    while not game.is_over and steps < 400:
        game.adjudicate()
        steps += 1
    assert game.is_over
    hooks = game.hooks
    assert isinstance(hooks, BeleriandHooks)
    result = hooks.game_result()
    assert result is not None and "Valar" in result.reason
    assert hooks.glaurung_arrived
    assert hooks.eriador_centers == 0  # the migration schedule ran its course
