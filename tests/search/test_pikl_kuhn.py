"""piKL on Kuhn poker (docs/pikl.tex Sec. 8 item 4).

Kuhn poker is reduced to its normal form -- each player's actions are their 64
pure strategies (one action per information set), with exact expected payoffs
over the six deals -- and piKL runs on that matrix exactly as it runs on any
candidate subgame. The resulting mean mixed strategies convert to behavioral
strategies (Kuhn's theorem, reachability-aware) and are scored with the
*extensive-form* best-response machinery
(:mod:`entente.gametheory.exploitability`) -- an independent implementation that
knows nothing about piKL. The normal-form and extensive-form
exploitability computations must agree to ~1e-9, which validates the
reduction itself; the known equilibrium facts (game value -1/18,
exploitability 0) then anchor the lam -> 0 limit, and Theorem 1's KL bound
anchors the large-lam limit.

Scale notes for the tolerances: payoffs are chips in [-2, 2] (D = 2 bounds
rewards); the pass-biased behavioral anchor (P(pass) = 0.6 at every infoset)
gives beta = max_s log(1/tau(s)) = 6*log(1/0.4) ~ 5.50 over pure strategies.
"""

from __future__ import annotations

import math
from itertools import product

import pytest

from entente.gametheory.exploitability import (
    best_response_value,
    expected_value,
    exploitability,
)
from entente.gametheory.kuhn import ACTIONS, KuhnState, initial_states
from entente.gametheory.normal_form import MatrixGame, self_play
from entente.search.pikl import AnchoredHedge

PASS_PROB = 0.6
KUHN_VALUE = -1.0 / 18.0

PureStrategy = dict[str, str]


def _player_infosets(player: int) -> list[str]:
    found: set[str] = set()

    def visit(state: KuhnState) -> None:
        if state.is_terminal:
            return
        if state.player == player:
            found.add(state.information_set())
        for action in state.legal_actions:
            visit(state.after(action))

    for root in initial_states():
        visit(root)
    return sorted(found)


def _pure_strategies(player: int) -> list[PureStrategy]:
    keys = _player_infosets(player)
    return [dict(zip(keys, combo, strict=True)) for combo in product(ACTIONS, repeat=len(keys))]


def _payoff(root: KuhnState, strategies: tuple[PureStrategy, PureStrategy]) -> float:
    state = root
    while not state.is_terminal:
        state = state.after(strategies[state.player][state.information_set()])
    return state.payoff(0)


def _kuhn_matrix(rows: list[PureStrategy], cols: list[PureStrategy]) -> MatrixGame:
    roots = initial_states()
    payoffs = [
        [sum(_payoff(root, (r, c)) for root in roots) / len(roots) for c in cols] for r in rows
    ]
    names_r = tuple(str(i) for i in range(len(rows)))
    names_c = tuple(str(j) for j in range(len(cols)))
    return MatrixGame.zero_sum(names_r, names_c, payoffs)


def _anchor(pures: list[PureStrategy]) -> tuple[float, ...]:
    """Product anchor: behavioral P(pass) = PASS_PROB at every infoset."""
    probs = []
    for strategy in pures:
        p = 1.0
        for action in strategy.values():
            p *= PASS_PROB if action == "p" else 1.0 - PASS_PROB
        probs.append(p)
    total = sum(probs)
    return tuple(p / total for p in probs)


def _reachable_infosets(strategy: PureStrategy, player: int) -> frozenset[str]:
    """Collect the own infosets a pure strategy can actually reach.

    Its own earlier moves may close some off, which is the consistency
    condition of Kuhn's theorem.
    """
    found: set[str] = set()

    def visit(state: KuhnState) -> None:
        if state.is_terminal:
            return
        if state.player == player:
            infoset = state.information_set()
            found.add(infoset)
            visit(state.after(strategy[infoset]))
        else:
            for action in state.legal_actions:
                visit(state.after(action))

    for root in initial_states():
        visit(root)
    return frozenset(found)


def _to_behavioral(
    mixed: list[float], pures: list[PureStrategy], player: int
) -> dict[str, list[float]]:
    reach = [_reachable_infosets(s, player) for s in pures]
    behavioral: dict[str, list[float]] = {}
    for infoset in _player_infosets(player):
        mass = [0.0] * len(ACTIONS)
        for weight, strategy, reached in zip(mixed, pures, reach, strict=True):
            if infoset in reached:
                mass[ACTIONS.index(strategy[infoset])] += weight
        total = sum(mass)
        if total <= 1e-12:
            behavioral[infoset] = [1.0 / len(ACTIONS)] * len(ACTIONS)
        else:
            behavioral[infoset] = [m / total for m in mass]
    return behavioral


def _kl(p: list[float] | tuple[float, ...], q: tuple[float, ...]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0.0)


def _run(
    lam: float, iterations: int
) -> tuple[
    MatrixGame,
    list[PureStrategy],
    list[PureStrategy],
    list[float],
    list[float],
    dict[str, list[float]],
]:
    rows = _pure_strategies(0)
    cols = _pure_strategies(1)
    game = _kuhn_matrix(rows, cols)
    row = AnchoredHedge(_anchor(rows), lam=lam)
    col = AnchoredHedge(_anchor(cols), lam=lam)
    self_play(game, row, col, iterations)
    mu_row = row.average_strategy
    mu_col = col.average_strategy
    behavioral = _to_behavioral(mu_row, rows, 0) | _to_behavioral(mu_col, cols, 1)
    return game, rows, cols, mu_row, mu_col, behavioral


class TestReductionAgreesWithExtensiveForm:
    """The 64-strategy reduction, checked against the tree it stands in for."""

    def test_normal_form_and_tree_scores_match(self) -> None:
        """The two exploitability computations must agree to numerical precision.

        One is a matrix best response over the normal form, the other a
        brute-force tree best response on the behavioral conversion, and the
        game values must match as well. Agreement is the evidence that the
        64-strategy reduction and the Kuhn's-theorem conversion are both right.
        """
        game, _, _, mu_row, mu_col, behavioral = _run(lam=0.05, iterations=400)
        roots = initial_states()

        tree_value = expected_value(roots, behavioral, player=0)
        matrix_value = game.expected_payoffs(mu_row, mu_col)[0]
        assert tree_value == pytest.approx(matrix_value, abs=1e-9)

        tree_expl = exploitability(roots, behavioral)
        matrix_expl = (
            max(game.row_utility_vector(mu_col)) + max(game.col_utility_vector(mu_row))
        ) / 2.0
        assert tree_expl == pytest.approx(matrix_expl, abs=1e-9)

        # Best responses are genuine upper envelopes of the value.
        assert best_response_value(roots, 0, behavioral) >= tree_value - 1e-12


class TestLambdaSmallFindsTheEquilibrium:
    """The weak-anchor limit: search should find Kuhn's equilibrium anyway."""

    def test_exploitability_and_value(self) -> None:
        """At ``lam = 0.002`` the mean policy must land near the equilibrium.

        The theory gap is ``lam*beta`` at about 0.011 chips, so with T = 2500
        exploitability must fall below 0.05 and the game value must sit within
        0.03 of the known -1/18 -- despite the pass-biased anchor pulling
        elsewhere.
        """
        _, _, _, _, _, behavioral = _run(lam=0.002, iterations=2500)
        roots = initial_states()
        value = expected_value(roots, behavioral, player=0)
        assert value == pytest.approx(KUHN_VALUE, abs=0.03)
        assert exploitability(roots, behavioral) < 0.05


class TestLambdaLargeStaysWithTheAnchor:
    """The strong-anchor limit: search should stay on the anchor, and be beatable."""

    def test_kl_bound_and_anchor_closeness(self) -> None:
        """At ``lam = 30`` the mean policy must stay within the anchor's KL ball.

        With D = 2, Theorem 1's asymptotic KL bound is ``D/lam``, about 0.067
        over the 64-strategy mixed policies. The behavioral conversion must then
        sit on the anchor's (0.6, 0.4) at every infoset -- and such play is
        *not* an equilibrium, so it must stay exploitable.
        """
        _, rows, cols, mu_row, mu_col, behavioral = _run(lam=30.0, iterations=300)
        assert _kl(mu_row, _anchor(rows)) < 2.0 / 30.0
        assert _kl(mu_col, _anchor(cols)) < 2.0 / 30.0
        for distribution in behavioral.values():
            assert distribution[0] == pytest.approx(PASS_PROB, abs=0.03)
        assert exploitability(initial_states(), behavioral) > 0.1
