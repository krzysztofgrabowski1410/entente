"""Diplomacy scoring: Sum-of-Squares, survival and solo-win indicators.

Sum-of-Squares (SoS) is the standard no-press tournament scoring system, and
the one the evaluation protocol reports. Every measure is computed per game and
then aggregated **per named agent** across a tournament's seated outcomes
(:class:`~entente.eval.tournament.SeatedOutcome`): an agent name, not a power,
is what a config compares, because an agent may play different powers across a
1-vs-6 rotation or a round-robin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from entente.eval.tournament import SeatedOutcome


def sos_scores(supply_counts: Mapping[str, int], winner: str | None) -> dict[str, float]:
    """Compute the Sum-of-Squares shares for one finished game.

    A solo winner takes the whole board: 1.0 for them, 0.0 for everyone else.
    Otherwise each power's share is its supply-centre count squared, normalized
    to sum to one across every power in ``supply_counts``, so an eliminated
    power contributes exactly 0.
    """
    if winner is not None:
        return {power: (1.0 if power == winner else 0.0) for power in supply_counts}
    total_sq = sum(count * count for count in supply_counts.values())
    if total_sq == 0:  # every power eliminated (should not happen; stays total-preserving)
        share = 1.0 / len(supply_counts) if supply_counts else 0.0
        return dict.fromkeys(supply_counts, share)
    return {power: (count * count) / total_sq for power, count in supply_counts.items()}


def survived(supply_counts: Mapping[str, int]) -> dict[str, bool]:
    """Report, per power, whether it still owned a supply centre at game end.

    Owning none is elimination.
    """
    return {power: count > 0 for power, count in supply_counts.items()}


def soloed(power: str, winner: str | None) -> bool:
    """Report whether ``power`` took a solo, that is a majority, victory."""
    return power == winner


def per_agent_sos(results: Sequence[SeatedOutcome]) -> dict[str, list[float]]:
    """Collect every game's SoS share, one list per agent *name*.

    Keying by name rather than by power is what makes the lists comparable: an
    agent may sit in a different seat from one game to the next.
    """
    scores: dict[str, list[float]] = {}
    for seated in results:
        shares = sos_scores(seated.outcome.supply_counts, seated.outcome.winner)
        for power, agent_name in seated.seats.items():
            scores.setdefault(agent_name, []).append(shares[power])
    return scores


def per_agent_survival(results: Sequence[SeatedOutcome]) -> dict[str, list[bool]]:
    """Every game's survival indicator, one list per agent name."""
    survival: dict[str, list[bool]] = {}
    for seated in results:
        alive = survived(seated.outcome.supply_counts)
        for power, agent_name in seated.seats.items():
            survival.setdefault(agent_name, []).append(alive[power])
    return survival


def per_agent_solo(results: Sequence[SeatedOutcome]) -> dict[str, list[bool]]:
    """Every game's solo-win indicator, one list per agent name."""
    solo: dict[str, list[bool]] = {}
    for seated in results:
        winner = seated.outcome.winner
        for power, agent_name in seated.seats.items():
            solo.setdefault(agent_name, []).append(soloed(power, winner))
    return solo
