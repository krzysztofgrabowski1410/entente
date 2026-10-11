# Variants API

Variant rulesets are two things: a `Ruleset` instance (the frozen adjudication
surface) and an optional `GameHooks` (the game-loop seam for engine-level
state and events). Beleriand is the reference variant.

## The `GameHooks` seam

`entente.engine.game_hooks.GameHooks` is the extension surface for everything
the variant keeps *out* of the adjudicator: knowledge facts,
masks, seeded timers, timed events, per-power victory. The base class *is* the
standard game (every hook a documented no-op); variants register a factory in
`GAME_HOOKS` and the `Game` facade consults the registry -- never a ruleset
name.

```python
GAME_HOOKS: dict[str, Callable[[MapGraph, int | None], GameHooks]]

def hooks_for(name: str, graph: MapGraph, seed: int | None) -> GameHooks: ...

class GameHooks:
    def __init__(self, graph: MapGraph, seed: int | None = None) -> None: ...
    def wrap_ruleset(self, base: Ruleset) -> Ruleset: ...
    def game_result(self) -> GameResult | None: ...
    # ... the full hook set is documented in the module
```

`GameResult` (winners, reason, scores), `GrantRecord` (a knowledge transfer),
and `MovementPlan` (what `preprocess_movement` hands back) are the public data
types.

## The Beleriand variant

Importing `entente.variants.beleriand` registers the ruleset (in `RULESETS`),
its hooks (in `GAME_HOOKS`) and its spec (in `VARIANT_SPECS`) as an import
side effect -- `entente/__init__` does this so
`Game.new("beleriand", "beleriand")` works out of the box.

```python
from entente.engine.game import Game

game = Game.new("beleriand", "beleriand", seed=0)   # years count 25 down to 0
print(game.phase)                                    # "S25M"
print(sorted(game.legal_layer2_orders("MORGOTH"))[:1])  # Layer-2 orders
```

The ruleset object is `entente.variants.beleriand.ruleset.BELERIAND`; the
Layer-2 hooks are `entente.variants.beleriand.layer2.BeleriandHooks`. Layer 1
(the movement and retreat axes, the directional mountain barriers included) is
specified in [docs/adjudication.pdf](../adjudication.pdf); Layer 2 (state,
timers, victory) follows the rulebook and is documented in its module.

<!-- names
entente.engine.game_hooks:GameHooks
entente.engine.game_hooks:GAME_HOOKS
entente.engine.game_hooks:GameResult
entente.engine.game_hooks:GrantRecord
entente.engine.game_hooks:MovementPlan
entente.engine.game_hooks:hooks_for
entente.engine.variant_meta:VARIANT_SPECS
entente.engine.variant_meta:VariantSpec
entente.variants.beleriand.ruleset:BELERIAND
entente.variants.beleriand.ruleset:BeleriandRuleset
entente.variants.beleriand.layer2:BeleriandHooks
-->
