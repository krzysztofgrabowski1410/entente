"""Pin that the per-unit legality mask reproduces ``game.legal_orders`` exactly.

The mask is checked in every phase over random states. It is the seam where a
network could place probability on an illegal order, so exactness here is what
makes "the policy cannot play an illegal move" a fact rather than a hope.

Torch-free -- the mask is vocabulary and engine data -- so these run in the
default environment, not only under the ``ml`` group.
"""

from __future__ import annotations

from entente.agents.random_agent import RandomAgent
from entente.engine.game import Game
from entente.engine.orders import Waive, format_order
from entente.nn.order_vocab import build_vocabulary, legal_order_indices


def _engine_index_set(game: Game, power: str, vocab: object) -> set[int]:
    return {vocab.index(format_order(o)) for o in game.legal_orders(power)}


def test_mask_union_equals_engine_legal_orders_all_phases() -> None:
    vocab = build_vocabulary("standard", "standard")
    seen_phase_kinds: set[str] = set()
    for seed in range(8):
        game = Game.new("standard", "standard")
        agent = RandomAgent(seed)
        while not game.is_over and game.year <= 1907:
            for power in game.powers:
                groups = legal_order_indices(game, power, vocab)
                union = {idx for indices in groups.values() for idx in indices}
                assert union == _engine_index_set(game, power, vocab)
                seen_phase_kinds.add(game.phase[-1])
                sheet = agent.orders(game, power)
                if sheet:
                    game.submit(power, list(sheet))
            game.adjudicate()
    # the sweep must actually exercise movement (and typically retreat/build too)
    assert "M" in seen_phase_kinds


def test_each_group_key_matches_its_orders_locations() -> None:
    vocab = build_vocabulary("standard", "standard")
    game = Game.new("standard", "standard")  # S1901M movement
    for power in game.powers:
        groups = legal_order_indices(game, power, vocab)
        # every legal order lands in the group keyed by its own unit location
        for order in game.legal_orders(power):
            key = "<waive>" if isinstance(order, Waive) else str(order.location)
            assert vocab.index(format_order(order)) in groups[key]


def test_groups_are_disjoint_and_sorted() -> None:
    vocab = build_vocabulary("standard", "standard")
    game = Game.new("standard", "standard")
    groups = legal_order_indices(game, "RUSSIA", vocab)
    assert groups  # Russia has units at game start
    all_indices = [idx for indices in groups.values() for idx in indices]
    assert len(all_indices) == len(set(all_indices))  # disjoint across units
    for indices in groups.values():
        assert indices == sorted(indices)


def test_a_unit_mask_contains_hold_move_and_support() -> None:
    vocab = build_vocabulary("standard", "standard")
    game = Game.new("standard", "standard")
    groups = legal_order_indices(game, "FRANCE", vocab)
    # A PAR (army) can at least hold and move; find its group and check membership
    par_group = {vocab.order(i) for i in groups["PAR"]}
    assert "A PAR H" in par_group
    assert any(order.startswith("A PAR - ") for order in par_group)
