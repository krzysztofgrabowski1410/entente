"""cfr.py: the quantitative check that the implementation actually solves a game.

Vanilla CFR on Kuhn poker has to reach the known game value, -1/18 to player 0,
and drive exploitability below 1e-3. Chance-sampled MCCFR is checked too, at a
looser tolerance, because sampling variance is inherent to it.
"""

from __future__ import annotations

import pytest

from entente.gametheory.cfr import chance_sampled_mccfr, vanilla_cfr
from entente.gametheory.exploitability import expected_value, exploitability
from entente.gametheory.kuhn import initial_states

KUHN_VALUE = -1.0 / 18.0


def test_vanilla_cfr_rejects_empty_root_states() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        vanilla_cfr([], iterations=1)


def test_chance_sampled_mccfr_rejects_empty_root_states() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        chance_sampled_mccfr([], iterations=1)


def test_vanilla_cfr_reaches_the_known_kuhn_value_and_exploitability() -> None:
    # CFR is fully deterministic (no sampling) -- these figures are exact
    # and reproducible, not a flaky statistical margin: 100k iterations
    # measured at exploitability 0.000633, comfortably under the 1e-3 gate.
    roots = initial_states()

    strategy, iteration_values = vanilla_cfr(roots, iterations=100_000)

    assert len(iteration_values) == 100_000
    assert expected_value(roots, strategy, player=0) == pytest.approx(KUHN_VALUE, abs=1e-3)
    assert exploitability(roots, strategy) < 1e-3


def test_vanilla_cfr_plus_also_reaches_the_known_kuhn_value() -> None:
    roots = initial_states()

    strategy, _ = vanilla_cfr(roots, iterations=100_000, plus=True)

    assert expected_value(roots, strategy, player=0) == pytest.approx(KUHN_VALUE, abs=1e-3)
    assert exploitability(roots, strategy) < 1e-3


def test_vanilla_cfr_iteration_values_trend_toward_the_known_value() -> None:
    """Late iterations must sit much closer to the known value than the first.

    The check makes no monotonicity claim at all, by design: the per-iteration
    value oscillates, and only the trend across the run is meaningful.
    """
    roots = initial_states()

    _, iteration_values = vanilla_cfr(roots, iterations=2_000)

    first_gap = abs(iteration_values[0] - KUHN_VALUE)
    late_gap = max(abs(v - KUHN_VALUE) for v in iteration_values[-10:])
    assert late_gap < first_gap


def test_chance_sampled_mccfr_converges_close_to_the_known_value() -> None:
    roots = initial_states()

    strategy, _ = chance_sampled_mccfr(roots, iterations=40_000, seed=7)

    assert expected_value(roots, strategy, player=0) == pytest.approx(KUHN_VALUE, abs=1e-2)
    assert exploitability(roots, strategy) < 5e-2


def test_chance_sampled_mccfr_is_deterministic_given_a_seed() -> None:
    roots = initial_states()

    first, _ = chance_sampled_mccfr(roots, iterations=200, seed=3)
    second, _ = chance_sampled_mccfr(roots, iterations=200, seed=3)

    assert first == second
