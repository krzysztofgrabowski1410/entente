# Search API (piKL)

The KL-anchored hedge search: `AnchoredHedge` (deliberately game-agnostic --
players are indices, candidates are opaque) plus the config surface and the
Diplomacy-specific candidate/rollout plumbing. Spec: [docs/pikl.pdf](../pikl.pdf);
implemented from Jacob et al. 2022 (arXiv:2112.07544). Stdlib + pydantic (the
torch dependency lives one layer up in `entente.agents.pikl_agent`).

## The learner

```python
class AnchoredHedge:                            # entente.search.pikl
    def __init__(self, anchor: Sequence[float], *, lam: float,
                 eta: float | None = None,
                 adaptive_scale: float = ADAPTIVE_ETA_SCALE) -> None: ...
    strategy: list[float]                        # this iteration's policy
    def observe_utility(self, utility, *, weight=1.0) -> list[float]: ...
    average_strategy: list[float]                # the mean iterate (paper eq. 7)

def run_pikl(...) -> list[PiklResult]: ...       # n-player orchestrator
def reweight_anchor(anchor, bias) -> list[float]: ...          # the negotiation hook
def anchored_quantal_response(utilities, anchor, lam) -> list[float]: ...
```

`lam = 0` recovers plain Hedge; `lam = inf` plays the anchor exactly.
`reweight_anchor` is the single seam `negotiation/` uses, kept
game-agnostic. `AnchoredHedge` satisfies the `Learner` protocol so
`entente.gametheory.normal_form.self_play` drives it unchanged -- the
validation suite checks it against hand-computed iterates, matrix-game
equilibria, and Kuhn poker before any Diplomacy use.

### Minimal working example

```python
from entente.search.pikl import AnchoredHedge

hedge = AnchoredHedge([0.5, 0.5], lam=1.0, eta=1.0)
print(hedge.strategy)                 # first-iteration policy
hedge.observe_utility([0.9, 0.1])     # fold in a utility vector
print(hedge.average_strategy)         # the object the guarantees attach to
```

## The Diplomacy config and plumbing

`entente.search.config` holds the typed surface: `PiklSearchConfig` (k,
attempts, iterations, lam, eta, sampling), `PiklAgentConfig` (checkpoint +
search + device), `LambdaSweepConfig`, `EtaConfig`.
`entente.search.candidates.collect_candidates` samples candidate order-sets
into a `CandidateSet`; `entente.search.rollout` steps the engine and evaluates
successors (`profile_utility_fn`, `supply_share_value`, `branch_game`).

<!-- names
entente.search.pikl:AnchoredHedge
entente.search.pikl:run_pikl
entente.search.pikl:PiklResult
entente.search.pikl:reweight_anchor
entente.search.pikl:anchored_quantal_response
entente.search.config:PiklSearchConfig
entente.search.config:PiklAgentConfig
entente.search.config:LambdaSweepConfig
entente.search.config:EtaConfig
entente.search.candidates:CandidateSet
entente.search.candidates:ScoredOrderSet
entente.search.candidates:CandidatePolicy
entente.search.candidates:collect_candidates
entente.search.rollout:branch_game
entente.search.rollout:profile_utility_fn
entente.search.rollout:supply_share_value
-->
