"""Pin the global order vocabulary's stability, its content and its coverage.

The superset guarantee is what makes the policy mask engine-exact, so it is
checked against real play rather than assumed.

Torch-free -- the vocabulary drives the engine, not a network -- so this runs in
the default environment too. Index stability is what keeps a checkpoint
meaningful, so its check must not hide behind the ``ml`` group.
"""

from __future__ import annotations

import hashlib

import pytest

from entente.agents.random_agent import RandomAgent
from entente.engine.game import Game
from entente.engine.map import load_map
from entente.engine.orders import format_order
from entente.engine.rules import RULESETS
from entente.nn.order_vocab import _build_entries, build_vocabulary

# Pinned snapshot of the standard-map / standard-ruleset vocabulary. A change to
# the enumeration (or the engine's order set) shifts these, which is exactly when
# a checkpoint's head would silently mis-align -- so they are asserted, not
# informational. 36343 entries; sha256 over the newline-joined sorted strings.
_EXPECTED_SIZE = 36343
_EXPECTED_SHA256 = "35e29ed5e2d8549f21b835151022ce456b494f85f5b18525ab9549bd460fc329"


def _hash(entries: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def test_vocab_matches_pinned_snapshot() -> None:
    vocab = build_vocabulary("standard", "standard")
    assert len(vocab) == _EXPECTED_SIZE
    assert _hash(vocab.entries) == _EXPECTED_SHA256


def test_entries_are_sorted_and_unique() -> None:
    vocab = build_vocabulary("standard", "standard")
    assert list(vocab.entries) == sorted(vocab.entries)
    assert len(set(vocab.entries)) == len(vocab.entries)


def test_index_round_trips() -> None:
    vocab = build_vocabulary("standard", "standard")
    for i in range(0, len(vocab), 500):
        assert vocab.index(vocab.order(i)) == i
    assert vocab.order(vocab.index("Waive")) == "Waive"


def test_build_is_deterministic() -> None:
    graph = load_map("standard")
    ruleset = RULESETS["standard"]
    # A fresh build (bypassing the lru_cache) reproduces the cached entries.
    assert _build_entries(graph, ruleset) == build_vocabulary("standard", "standard").entries


def test_index_accepts_order_objects_and_strings() -> None:
    vocab = build_vocabulary("standard", "standard")
    game = Game.new("standard", "standard")
    order = game.legal_orders("FRANCE")[0]
    assert vocab.index(order) == vocab.index(format_order(order))


def test_unknown_order_is_not_a_member() -> None:
    vocab = build_vocabulary("standard", "standard")
    assert vocab.get("A PAR - MOON") is None
    assert "A PAR - MOON" not in vocab


def test_representative_orders_are_present() -> None:
    vocab = build_vocabulary("standard", "standard")
    # one of each grammar the head must be able to emit
    for order in ("A PAR H", "A PAR - BUR", "A MAR S A PAR - BUR", "F NTH C A LON - HOL", "Waive"):
        assert order in vocab, order
    # a multi-sea convoy route the two-unit enumeration cannot reach on its own
    assert "A LON - TUN via convoy" in vocab


def test_superset_of_engine_legal_orders_on_random_games() -> None:
    """Every order the engine enumerates in real play is in the vocabulary.

    This is the invariant the per-unit legality mask relies on. A quick sweep
    here; the deep sweep is slow-marked below.
    """
    vocab = build_vocabulary("standard", "standard")
    for seed in range(6):
        game = Game.new("standard", "standard")
        agent = RandomAgent(seed)
        while not game.is_over and game.year <= 1905:
            for power in game.powers:
                for order in game.legal_orders(power):
                    assert order in vocab, format_order(order)
                sheet = agent.orders(game, power)
                if sheet:
                    game.submit(power, list(sheet))
            game.adjudicate()


@pytest.mark.slow
def test_superset_of_engine_legal_orders_deep() -> None:
    """A wider sweep the vocabulary must still cover exactly.

    More games and later years bring more convoys, retreats and builds into
    reach than the quick sweep sees.
    """
    vocab = build_vocabulary("standard", "standard")
    for seed in range(60):
        game = Game.new("standard", "standard")
        agent = RandomAgent(seed)
        while not game.is_over and game.year <= 1910:
            for power in game.powers:
                for order in game.legal_orders(power):
                    assert order in vocab, format_order(order)
                sheet = agent.orders(game, power)
                if sheet:
                    game.submit(power, list(sheet))
            game.adjudicate()


# -- the vocabulary covers its board, or refuses it -------------------------
#
# The builder rests on three classic-map facts and would silently return an
# INCOMPLETE vocabulary where any fails -- exactly the "masking bug" that
# OrderVocabulary.index exists to fail loudly about. The learned seats are
# standard-only; these pin the refusal at the source, so no caller can obtain a
# vocabulary that cannot express legal orders.


def test_the_standard_vocabulary_names_no_base_strength_and_needs_none() -> None:
    """The standard vocabulary writes no strength prefix, and needs none.

    Every standard unit has base strength 1, so the canonical grammar writes
    every standard order bare and the vocabulary is complete without a prefix.
    That is what makes the whole learned stack safe from stronger units by
    construction rather than by care.
    """
    import re

    from entente.engine.game import Game
    from entente.engine.orders import format_order

    vocab = build_vocabulary("standard", "standard")
    prefixed = re.compile(r"(?:^|\s)\d[AF]")
    assert not [entry for entry in vocab.entries if prefixed.search(entry)]

    game = Game.new("standard", "standard")
    missing = [
        format_order(order)
        for power in game.powers
        for order in game.legal_orders(power)
        if order not in vocab
    ]
    assert not missing, missing


def test_a_board_of_stronger_units_is_refused_rather_than_under_generated() -> None:
    """A board of stronger units is refused outright rather than under-generated.

    Beleriand breaks all three assumptions at once -- Areas, base strengths above 1 and variant
    order types -- and the error names each, so whoever hits it learns what a
    Beleriand vocabulary would actually take.
    """
    import pytest

    with pytest.raises(ValueError) as excinfo:
        build_vocabulary("beleriand", "beleriand")
    message = str(excinfo.value)
    assert "capacity != 1" in message
    assert "above base strength 1" in message
    assert "variant order types" in message


def test_the_refusal_reads_the_board_not_a_map_name() -> None:
    """The refusal reads the board, not a map name.

    The guard asks the graph and the ruleset, so a later variant that
    introduces Areas or stronger units is refused without anyone remembering to add its
    name to a list.
    """
    from entente.engine.map import load_map
    from entente.engine.rules import RULESETS
    from entente.nn.order_vocab import _reject_uncoverable

    _reject_uncoverable(load_map("standard"), RULESETS["standard"])  # no raise
    with pytest.raises(ValueError):
        _reject_uncoverable(load_map("beleriand"), RULESETS["beleriand"])
