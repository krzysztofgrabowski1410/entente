# Negotiation API

The isolated, deletable LLM negotiation prototype: a `Provider` turns a
piKL-grounded `Intent` into a press message, and a parsed message reweights
piKL's candidate beliefs. Nothing outside `negotiation/`
imports it (the CLI reaches it via `importlib`), so it is cut by one
`git rm -r src/entente/negotiation` plus dropping the `llm`/`finetune` groups.

## The `Provider` seam

```python
class Provider(Protocol):                       # entente.negotiation.generate
    def generate(self, intent: Intent) -> str: ...

class StubProvider:      # deterministic, template-based; the only one parse.py reads
    def generate(self, intent: Intent) -> str: ...
class AnthropicProvider: # defers the `anthropic` import to __init__ (llm group)
    def __init__(self, *, model: str | None = None) -> None: ...
class LocalLoraProvider: # a fine-tuned adapter (finetune group)
    def __init__(self, adapter_dir: str | None = None) -> None: ...
```

`Intent` (`entente.negotiation.intents`) is `(proposer, ally, phase,
proposer_action, ally_action, rationale)`; the actions are order-sets from the
proposer's own piKL `CandidateSet`, which is what lets `parse.reweight_from_intent`
match an intent straight against candidates. `parse.parse_message` reads
`StubProvider`'s fixed grammar back into an `Intent` (`MessageParseError` on a
non-match).

### Minimal working example (offline stub)

```python
from entente.negotiation.scenario import load_scenario, run_predictions

scenario = load_scenario("src/entente/negotiation/scenarios/s1.json")
run = run_predictions(scenario)     # stub generate -> parse -> reweight, fully offline
print(run)
```

The CLI wrapper is `entente negotiate-demo <scenario.json>` (writes a stub
transcript; a real-model `*.anthropic.json` transcript when `ANTHROPIC_API_KEY`
is set -- see [usage.md](../usage.md) and docs/setup.md). Limitations are in
`src/entente/negotiation/README.md`.

<!-- names
entente.negotiation.generate:Provider
entente.negotiation.generate:StubProvider
entente.negotiation.generate:AnthropicProvider
entente.negotiation.generate:LocalLoraProvider
entente.negotiation.intents:Intent
entente.negotiation.parse:parse_message
entente.negotiation.parse:reweight_from_intent
entente.negotiation.parse:MessageParseError
entente.negotiation.scenario:load_scenario
entente.negotiation.scenario:run_predictions
entente.negotiation.scenario:Scenario
-->
