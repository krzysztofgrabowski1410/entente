"""scoring.py: Sum-of-Squares, survival and solo, per game and per agent name.

The aggregates run over a small set of hand-built ``SeatedOutcome`` values, so
every expected number is worked out by hand.
"""

from __future__ import annotations

import pytest

from entente.eval.scoring import (
    per_agent_solo,
    per_agent_sos,
    per_agent_survival,
    soloed,
    sos_scores,
    survived,
)
from entente.eval.tournament import SeatedOutcome
from entente.sim.parallel import GameOutcome


def _outcome(supply_counts: dict[str, int], winner: str | None = None) -> GameOutcome:
    return GameOutcome(
        seed=0,
        ruleset="standard",
        map_name="standard",
        max_year=1910,
        num_phases=1,
        final_phase="S1901M",
        winner=winner,
        supply_counts=supply_counts,
    )


def test_sos_scores_solo_winner_takes_everything() -> None:
    counts = {"FRANCE": 18, "GERMANY": 5, "AUSTRIA": 0}
    scores = sos_scores(counts, winner="FRANCE")
    assert scores == {"FRANCE": 1.0, "GERMANY": 0.0, "AUSTRIA": 0.0}


def test_sos_scores_no_solo_is_proportional_to_squares() -> None:
    counts = {"FRANCE": 4, "GERMANY": 3}
    scores = sos_scores(counts, winner=None)
    total = 4**2 + 3**2
    assert scores == pytest.approx({"FRANCE": 16 / total, "GERMANY": 9 / total})
    assert sum(scores.values()) == pytest.approx(1.0)


def test_sos_scores_eliminated_power_contributes_zero() -> None:
    counts = {"FRANCE": 10, "GERMANY": 0}
    scores = sos_scores(counts, winner=None)
    assert scores["GERMANY"] == 0.0
    assert scores["FRANCE"] == pytest.approx(1.0)


def test_sos_scores_falls_back_to_even_split_if_everyone_is_at_zero() -> None:
    # Should not arise from a real game (see the module docstring), but a
    # score function should stay total-preserving rather than divide by zero.
    scores = sos_scores({"FRANCE": 0, "GERMANY": 0}, winner=None)
    assert scores == {"FRANCE": 0.5, "GERMANY": 0.5}


def test_survived_reports_nonzero_centers() -> None:
    assert survived({"FRANCE": 3, "GERMANY": 0}) == {"FRANCE": True, "GERMANY": False}


def test_soloed_matches_the_winner() -> None:
    assert soloed("FRANCE", winner="FRANCE") is True
    assert soloed("GERMANY", winner="FRANCE") is False
    assert soloed("FRANCE", winner=None) is False


def test_per_agent_sos_groups_by_agent_name_not_power() -> None:
    # "greedy" plays FRANCE in game 1 and GERMANY in game 2; its two scores
    # must land in the same list despite the seat change.
    results = [
        SeatedOutcome(
            outcome=_outcome({"FRANCE": 10, "GERMANY": 0}, winner=None),
            seats={"FRANCE": "greedy", "GERMANY": "random"},
        ),
        SeatedOutcome(
            outcome=_outcome({"FRANCE": 0, "GERMANY": 20}, winner="GERMANY"),
            seats={"FRANCE": "random", "GERMANY": "greedy"},
        ),
    ]

    scores = per_agent_sos(results)

    assert scores["greedy"] == pytest.approx([1.0, 1.0])
    assert scores["random"] == pytest.approx([0.0, 0.0])


def test_per_agent_survival_and_solo_group_by_agent_name() -> None:
    results = [
        SeatedOutcome(
            outcome=_outcome({"FRANCE": 0, "GERMANY": 34}, winner="GERMANY"),
            seats={"FRANCE": "random", "GERMANY": "greedy"},
        ),
    ]

    assert per_agent_survival(results) == {"random": [False], "greedy": [True]}
    assert per_agent_solo(results) == {"random": [False], "greedy": [True]}
