"""kuhn.py: the game-tree shape and the terminal payoffs, both derived by hand.

The shape means the deals, the terminal histories and the information sets. The
payoffs follow Kuhn's rules: a fold forfeits the ante only, and a showdown after
a bet and a call moves twice that.
"""

from __future__ import annotations

import pytest

from entente.gametheory.kuhn import ACTIONS, BET, PASS, KuhnState, all_deals, initial_states


def test_all_deals_are_the_six_ordered_two_card_draws() -> None:
    deals = all_deals()
    assert len(deals) == 6
    assert len(set(deals)) == 6
    for deal in deals:
        assert deal.player_card[0] != deal.player_card[1]


def test_initial_states_are_one_per_deal_at_the_empty_history() -> None:
    states = initial_states()
    assert len(states) == 6
    assert all(state.history == "" for state in states)
    assert {state.deal for state in states} == set(all_deals())


@pytest.mark.parametrize(
    "history", ["", "p", "b", "pb"], ids=["root", "after-pass", "after-bet", "after-pass-bet"]
)
def test_nonterminal_histories_have_both_actions_legal(history: str) -> None:
    state = KuhnState(all_deals()[0], history)
    assert not state.is_terminal
    assert state.legal_actions == ACTIONS


@pytest.mark.parametrize("history", ["pp", "pbp", "pbb", "bp", "bb"])
def test_terminal_histories_have_no_legal_actions(history: str) -> None:
    state = KuhnState(all_deals()[0], history)
    assert state.is_terminal
    assert state.legal_actions == ()


def test_player_alternates_by_history_length_starting_with_zero() -> None:
    deal = all_deals()[0]
    assert KuhnState(deal, "").player == 0
    assert KuhnState(deal, "p").player == 1
    assert KuhnState(deal, "pb").player == 0
    assert KuhnState(deal, "b").player == 1


def test_information_set_is_own_card_plus_history_not_opponents_card() -> None:
    deal = next(d for d in all_deals() if d.player_card == ("J", "K"))
    root = KuhnState(deal, "")
    assert root.information_set() == "J"  # player 0's own card, never K
    after_pass = KuhnState(deal, "p")
    assert after_pass.information_set() == "Kp"  # player 1's own card


def test_after_advances_history_and_rejects_terminal_or_unknown_action() -> None:
    deal = all_deals()[0]
    state = KuhnState(deal, "p")
    assert state.after(BET) == KuhnState(deal, "pb")
    with pytest.raises(ValueError, match="unknown action"):
        state.after("raise")
    with pytest.raises(ValueError, match="terminal"):
        KuhnState(deal, "pp").after(PASS)


@pytest.mark.parametrize(
    ("history", "magnitude"),
    [("pp", 1.0), ("bp", 1.0), ("pbp", 1.0), ("pbb", 2.0), ("bb", 2.0)],
)
def test_terminal_payoff_magnitude_matches_the_stake(history: str, magnitude: float) -> None:
    for deal in all_deals():
        state = KuhnState(deal, history)
        assert abs(state.payoff(0)) == pytest.approx(magnitude)


def test_terminal_payoff_is_zero_sum() -> None:
    for deal in all_deals():
        for history in ("pp", "bp", "pbp", "pbb", "bb"):
            state = KuhnState(deal, history)
            assert state.payoff(0) == pytest.approx(-state.payoff(1))


def test_fold_always_pays_the_bettor_regardless_of_cards() -> None:
    """After ``"bp"`` player 0 always wins, even holding the worst card.

    Player 0 bet and player 1 folded, and a fold never reaches a showdown, so
    the cards are never compared.
    """
    j_holder_deal = next(d for d in all_deals() if d.player_card[0] == "J")
    assert KuhnState(j_holder_deal, "bp").payoff(0) == pytest.approx(1.0)


def test_showdown_winner_is_the_higher_card() -> None:
    deal = next(d for d in all_deals() if d.player_card == ("K", "J"))
    assert KuhnState(deal, "bb").payoff(0) == pytest.approx(2.0)
    assert KuhnState(deal, "bb").payoff(1) == pytest.approx(-2.0)


def test_payoff_raises_on_a_nonterminal_history() -> None:
    with pytest.raises(ValueError, match="not terminal"):
        KuhnState(all_deals()[0], "p").payoff(0)
