"""Agent name -> factory registry: one table both callers build agents from.

The CLI and the tournament/eval harness each turn a config string (``"greedy"``)
into an agent through :func:`build_agent`, so neither hardcodes its own
name-to-class mapping and a new agent becomes available to both at once.
"""

from __future__ import annotations

from collections.abc import Callable

from entente.agents.base import Agent
from entente.agents.greedy import GreedyAgent
from entente.agents.random_agent import RandomAgent

AgentFactory = Callable[[int], Agent]

REGISTRY: dict[str, AgentFactory] = {
    "random": RandomAgent,
    "greedy": GreedyAgent,
}


def build_agent(name: str, seed: int) -> Agent:
    """Build a seeded agent by registry name.

    Raises :class:`ValueError` (not :class:`KeyError`) on an unknown name,
    listing what is available -- the same shape :class:`entente.engine.game.GameError`
    and :mod:`entente.cli` already use for a bad ``--agents``/ruleset name.
    """
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r} (available: {sorted(REGISTRY)})") from None
    return factory(seed)
