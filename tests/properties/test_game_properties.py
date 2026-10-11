"""Assert the whole-game invariants over randomly played games.

Random games are driven through the public Game facade by the seeded
RandomAgent for a drawn number of engine phases. At *every* executed phase
the invariants below must hold, and the serialized form must round-trip:

- **placement legality / occupancy <= capacity** -- ``check_state`` over every
  recorded state (capacity is map data; classic maps say 1 everywhere);
- **conservation** -- movement removes exactly the dislodged units (as a
  (power, kind) multiset), the retreat stage re-places exactly the
  dislodged-minus-disbanded, winter adds builds and removes disbands and
  civil-disorder removals, and no other phase invents or loses units;
- **SC-count sanity** -- ownership maps supply centers to known powers,
  a power's unit count never exceeds its center count after a winter, and
  the winner (if any) holds a strict majority;
- **phase-machine sanity** -- history tags advance strictly along the
  S->S->F->F->W cycle (skips allowed, reordering not);
- **serialization** -- ``game_from_text(game_to_text(g))`` reproduces state,
  ownership, phase and history at every step, and the text form is a
  fixpoint.

Determinism of the whole loop (same seed => same transcript) gets its own
property.
"""

from __future__ import annotations

import random
from collections import Counter

from hypothesis import given, settings
from hypothesis import strategies as st

from entente.agents import RandomAgent
from entente.engine.game import Game, PhaseRecord
from entente.engine.map import Location, load_map
from entente.engine.orders import AnyOrder, Hold, Move, Support
from entente.engine.phase import Phase, PhaseKind
from entente.engine.serialization import game_from_text, game_to_text
from entente.engine.state import State, check_state

GRAPH = load_map("standard")

SUPPLY_CENTERS = frozenset(key for key, p in GRAPH.provinces.items() if p.is_supply_center)


def belligerent_sheet(game: Game, power: str, rng: random.Random) -> list[AnyOrder]:
    """Draw a movement sheet biased toward supported attacks on foreign units.

    The uniform random agent almost never dislodges anything, and the
    retreat-stage invariants say nothing at all until a dislodgement happens.
    """
    legal = game.legal_orders(power)
    moves_by_unit: dict[Location, list[Move]] = {}
    supports: list[Support] = []
    for order in legal:
        if isinstance(order, Move):
            moves_by_unit.setdefault(order.location, []).append(order)
        elif isinstance(order, Support) and isinstance(order.subject, Move):
            supports.append(order)
    foreign = {u.location.province for u in game.state.units if u.power != power}
    attacks = [
        support
        for support in supports
        if isinstance(support.subject, Move)
        and support.subject.location in moves_by_unit
        and support.subject.destination.province in foreign
    ]
    rng.shuffle(attacks)
    orders: dict[Location, AnyOrder] = {}
    for support in attacks:
        subject = support.subject
        assert isinstance(subject, Move)
        if support.location in orders or subject.location in orders:
            continue
        concrete = [
            move
            for move in moves_by_unit[subject.location]
            if move.destination.province == subject.destination.province and not move.via_convoy
        ]
        if not concrete:
            continue
        orders[subject.location] = concrete[0]
        orders[support.location] = support
    for location, options in moves_by_unit.items():
        if location not in orders:
            hold = Hold(options[0].unit_kind, location)
            orders[location] = rng.choice([*options, hold])
    return [orders[location] for location in sorted(orders, key=str)]


def play_random_game(seed: int, phases: int, *, roundtrip_every_phase: bool = False) -> Game:
    """Drive a game for ``phases`` engine phases, belligerently most of the time.

    Two movement sheets in three come from the belligerent sampler, which is
    what produces dislodgements, retreats and eliminations; the uniform random
    agent supplies the rest, and every retreat and adjustment decision.
    """
    game = Game.new("standard", "standard")
    agent = RandomAgent(seed)
    for _ in range(phases):
        if game.is_over:
            break
        for power in game.powers:
            rng = random.Random(f"drive:{seed}:{power}:{game.phase}")
            if rng.random() < 0.15:
                continue  # absent power: phase defaults, incl. civil disorder
            if game.phase_kind is PhaseKind.MOVEMENT and rng.random() < 2 / 3:
                sheet: list[AnyOrder] = belligerent_sheet(game, power, rng)
            else:
                sheet = list(agent.orders(game, power))
            if sheet:
                game.submit(power, sheet)
        game.adjudicate()
        if roundtrip_every_phase:
            assert_roundtrip(game)
    return game


def assert_roundtrip(game: Game) -> None:
    """Assert the game survives a save and a load with everything intact."""
    text = game_to_text(game)
    restored = game_from_text(text)
    assert restored.state == game.state
    assert restored.ownership == game.ownership
    assert restored.phase == game.phase
    assert restored.is_over == game.is_over
    assert [(r.phase, r.auto) for r in restored.history] == [
        (r.phase, r.auto) for r in game.history
    ]
    assert game_to_text(restored) == text  # fixpoint


def unit_multiset(state: State) -> Counter[tuple[str, str]]:
    """Count a state's units by ``(power, kind)``."""
    return Counter((unit.power, unit.kind.value) for unit in state.units)


def units_of(units: tuple) -> Counter[tuple[str, str]]:
    """Count a loose collection of units by ``(power, kind)``."""
    return Counter((unit.power, unit.kind.value) for unit in units)


def assert_conservation(before: State, record: PhaseRecord) -> None:
    """Assert one phase's unit bookkeeping, as ``(power, kind)`` multisets.

    The retreat stage's full equation also needs the preceding movement's
    dislodgements, so the caller asserts that half: it is the one that holds
    both records.
    """
    kind = Phase.parse(record.phase).kind
    pre = unit_multiset(before)
    post = unit_multiset(record.state_after)
    if kind is PhaseKind.MOVEMENT:
        dislodged = units_of(tuple(d.unit for d in record.dislodgements))
        assert post + dislodged == pre
    elif kind is PhaseKind.RETREAT:
        assert record.built == () and record.civil_disorder == ()
        assert not (pre - post), "a retreat stage never removes on-board units"
    else:
        built = units_of(record.built)
        removed = units_of(record.disbanded) + units_of(record.civil_disorder)
        assert post == pre + built - removed


def assert_sc_sanity(game: Game, record: PhaseRecord) -> None:
    """Assert supply-centre ownership stays coherent with the units on the board.

    Ownership names only real centres and real powers, no power leaves a
    winter with more units than centres, and a winner holds a strict majority.
    """
    ownership = record.ownership_after
    assert set(ownership) <= SUPPLY_CENTERS
    assert set(ownership.values()) <= set(game.powers)
    if Phase.parse(record.phase).kind is PhaseKind.ADJUSTMENT:
        counts = Counter(unit.power for unit in record.state_after.units)
        for power, count in counts.items():
            owned = sum(1 for owner in ownership.values() if owner == power)
            assert count <= owned, (power, count, owned)


def assert_cycle_order(history: tuple[PhaseRecord, ...]) -> None:
    """Assert the recorded phases advance strictly along the season cycle."""
    previous: Phase | None = None
    for record in history:
        phase = Phase.parse(record.phase)
        if previous is not None:
            cursor = previous
            for _ in range(5):
                cursor = cursor.successor()
                if cursor == phase:
                    break
            else:
                raise AssertionError(f"{record.phase} does not follow {previous.tag} in the cycle")
        previous = phase


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 10))
def test_random_games_maintain_invariants(seed: int, phases: int) -> None:
    game = play_random_game(seed, phases)
    graph = game.graph

    check_state(graph, game.state)
    before = Game.new("standard", "standard").state
    movement_dislodged: Counter[tuple[str, str]] = Counter()
    for record in game.history:
        check_state(graph, record.state_after)
        assert_conservation(before, record)
        assert_sc_sanity(game, record)
        kind = Phase.parse(record.phase).kind
        if kind is PhaseKind.MOVEMENT:
            movement_dislodged = units_of(tuple(d.unit for d in record.dislodgements))
        elif kind is PhaseKind.RETREAT:
            placed = unit_multiset(record.state_after) - unit_multiset(before)
            assert placed + units_of(record.disbanded) == movement_dislodged
        before = record.state_after
    assert_cycle_order(game.history)
    if game.winner is not None:
        majority = len(SUPPLY_CENTERS) // 2 + 1
        assert game.supply_count(game.winner) >= majority


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 6))
def test_serialization_round_trips_at_every_phase(seed: int, phases: int) -> None:
    play_random_game(seed, phases, roundtrip_every_phase=True)


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), phases=st.integers(1, 6))
def test_same_seed_same_transcript(seed: int, phases: int) -> None:
    first = play_random_game(seed, phases)
    second = play_random_game(seed, phases)
    assert game_to_text(first) == game_to_text(second)
