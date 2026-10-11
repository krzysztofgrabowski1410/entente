"""registry.py: name -> factory lookup."""

from __future__ import annotations

import pytest

from entente.agents.greedy import GreedyAgent
from entente.agents.random_agent import RandomAgent
from entente.agents.registry import REGISTRY, build_agent


def test_registry_lists_random_and_greedy() -> None:
    assert set(REGISTRY) == {"random", "greedy"}


def test_build_agent_random_returns_a_random_agent() -> None:
    agent = build_agent("random", 7)
    assert isinstance(agent, RandomAgent)
    assert agent.seed == 7


def test_build_agent_greedy_returns_a_greedy_agent() -> None:
    assert isinstance(build_agent("greedy", 7), GreedyAgent)


def test_build_agent_unknown_name_raises_with_available_names() -> None:
    with pytest.raises(ValueError, match=r"random.*greedy|greedy.*random"):
        build_agent("nonexistent", 0)
