# Agents API

An agent maps a game position to an ordered sheet for one power. The engine
never imports this package (a one-way dependency); agents consume the
public `Game` surface only, and all randomness lives here, seeded.

## The `Agent` protocol and registry

```python
class Agent(Protocol):
    def orders(self, game: Game, power: str) -> Sequence[GameOrder]: ...

REGISTRY: dict[str, AgentFactory]           # "random", "greedy"
AgentFactory = Callable[[int], Agent]       # seed -> agent

def build_agent(name: str, seed: int) -> Agent: ...
```

`build_agent` raises `ValueError` (listing the available names) on an unknown
name. The two baseline agents are `RandomAgent(seed)` and `GreedyAgent(seed)`
(an original one-ply heuristic).

```python
from entente.agents import build_agent
from entente.engine.game import Game

game = Game.new("standard", "standard")
agent = build_agent("greedy", seed=11)
sheet = agent.orders(game, "FRANCE")
game.submit("FRANCE", list(sheet))
```

## The learned agents (`ml` group)

`entente.agents.bc_agent.BCAgent` plays a trained behavioral-cloning checkpoint;
`entente.agents.pikl_agent.PiklAgent` wraps it in piKL anchored-hedge search.
Both satisfy the `Agent` protocol and need the `ml` dependency group (torch).

```python
class BCAgent:
    def __init__(self, checkpoint: str | Path, *, seed: int = 0,
                 device: str = "cpu", sample: bool = False) -> None: ...
    def orders(self, game: Game, power: str) -> Sequence[GameOrder]: ...

class PiklAgent:
    def __init__(self, config: PiklAgentConfig) -> None: ...
    def orders(self, game: Game, power: str) -> Sequence[GameOrder]: ...
```

`bc_agent.bc_lockstep_policy` / `pikl_agent.pikl_lockstep_policy` are the
vectorized `PolicyCallback`s the tournament/sweep harnesses batch through the
lockstep simulator; `bc_agent.bc_vs_greedy` is the BC-vs-greedy evaluation
entry point. See [search.md](search.md) for `PiklAgentConfig`.

<!-- names
entente.agents:Agent
entente.agents:REGISTRY
entente.agents:AgentFactory
entente.agents:build_agent
entente.agents:GreedyAgent
entente.agents:RandomAgent
entente.agents.bc_agent:BCAgent
entente.agents.bc_agent:bc_lockstep_policy
entente.agents.bc_agent:bc_vs_greedy
entente.agents.pikl_agent:PiklAgent
entente.agents.pikl_agent:pikl_lockstep_policy
entente.agents.pikl_agent:BCCandidatePolicy
entente.agents.pikl_agent:value_scores
-->
