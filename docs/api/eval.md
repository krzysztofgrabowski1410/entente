# Evaluation API

Tournaments over the spawn-safe parallel simulator, Sum-of-Squares scoring, and
the two significance tools (bootstrap CIs, permutation test). Stdlib-only.
Standard ruleset only (the `POWERS` tuple
is pinned; a non-standard map is rejected at entry).

## Tournaments

```python
def one_vs_six(agent_a: str, agent_b: str, seeds: Sequence[int], *,
               ruleset: str = "standard", map_name: str = "standard",
               max_year: int = 1910, max_workers: int | None = None
               ) -> list[SeatedOutcome]: ...
def round_robin(agent_names, seeds, *, ...) -> list[SeatedOutcome]: ...
def write_seated_jsonl(results, path) -> int: ...
def read_seated_jsonl(path) -> Iterator[SeatedOutcome]: ...
```

`one_vs_six` runs `agent_a` in each of the 7 seats against six of `agent_b`
(`7 * len(seeds)` games). A `SeatedOutcome` pairs a `GameOutcome` with the
seat -> agent-name map every scoring function needs.

## Scoring and statistics

```python
# entente.eval.scoring
def sos_scores(supply_counts, winner) -> dict[str, float]: ...
def per_agent_sos(results) -> dict[str, list[float]]: ...
def per_agent_survival(results) -> dict[str, list[bool]]: ...
def per_agent_solo(results) -> dict[str, list[bool]]: ...

# entente.eval.stats
def bootstrap_ci(samples, *, confidence=0.95, resamples=10_000, seed=0) -> tuple[float, float]: ...
def permutation_test(a, b, *, iterations=10_000, seed=0) -> float: ...
```

### Minimal working example

```python
from entente.eval.tournament import one_vs_six
from entente.eval.scoring import per_agent_sos
from entente.eval.stats import bootstrap_ci

results = one_vs_six("greedy", "random", seeds=[0, 1], max_year=1902)
sos = per_agent_sos(results)
print("greedy 95% CI:", bootstrap_ci(sos["greedy"]))
```

`entente.eval.config.load_tournament_config` parses the tournament YAML (see
`configs/tournament/main.yaml`); `entente.eval.lambda_sweep` and
`entente.eval.press_results` own the piKL-sweep and press-eval record schemas
and their regenerable results tables.

<!-- names
entente.eval.tournament:one_vs_six
entente.eval.tournament:round_robin
entente.eval.tournament:SeatedOutcome
entente.eval.tournament:write_seated_jsonl
entente.eval.tournament:read_seated_jsonl
entente.eval.tournament:POWERS
entente.eval.scoring:sos_scores
entente.eval.scoring:per_agent_sos
entente.eval.scoring:per_agent_survival
entente.eval.scoring:per_agent_solo
entente.eval.stats:bootstrap_ci
entente.eval.stats:permutation_test
entente.eval.config:load_tournament_config
entente.eval.config:TournamentConfig
entente.eval.config:ConfigError
entente.eval.lambda_sweep:SweepRecord
entente.eval.lambda_sweep:aggregate
entente.eval.press_results:PressRecord
entente.eval.press_results:summarize
entente.eval.press_results:sos_delta
-->
