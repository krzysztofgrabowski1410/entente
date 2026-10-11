"""Agents: policies mapping a game position to an order sheet.

The dependency runs one way: the engine never imports this package, which is
what lets :mod:`entente.engine` stay stdlib-only and free of any random draw
of its own. Agents consume the public :class:`entente.engine.game.Game`
surface only, and every draw they make is seeded.
"""

from entente.agents.base import Agent
from entente.agents.greedy import GreedyAgent
from entente.agents.random_agent import RandomAgent
from entente.agents.registry import REGISTRY, AgentFactory, build_agent

__all__ = [
    "REGISTRY",
    "Agent",
    "AgentFactory",
    "GreedyAgent",
    "RandomAgent",
    "build_agent",
]
