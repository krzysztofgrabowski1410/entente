# API reference

Curated reference for every public surface of `entente`. One page per surface;
each states the surface's purpose, its real signatures, a minimal working
example, and links to the spec that owns it, where one does.

| Page | Surface |
|:--|:--|
| [engine.md](engine.md) | The `Game` facade, `adjudicate(..., engine=...)`, orders, state, map |
| [variants.md](variants.md) | The `GameHooks` game-loop seam and the Beleriand variant |
| [agents.md](agents.md) | The `Agent` protocol, the registry, the BC and piKL agents |
| [data.md](data.md) | The behavioral-cloning data pipeline: source -> preprocess -> `ShardDataset` |
| [training.md](training.md) | The BC training entry points and the typed config loader |
| [eval.md](eval.md) | Tournaments, Sum-of-Squares scoring, bootstrap CIs, permutation tests |
| [search.md](search.md) | piKL: `AnchoredHedge`, the search config surface, candidates, rollouts |
| [negotiation.md](negotiation.md) | The negotiation `Provider` seam and intent schema |
| [viz.md](viz.md) | The stdlib SVG board renderer and the Pillow GIF rasterizer |

## What "public" means here

The engine (`entente.engine.*`) is stdlib-only and its surface is stable:
`Game`, the adjudicator entry, orders/state/map. The ML, search, eval,
negotiation and viz layers target the standard ruleset and depend on
optional dependency groups (`ml`, `viz`, `llm`, `finetune`); their
signatures are documented as they stand today and may still move (the project
is pre-1.0). Every name on a page is imported by `tests/test_api_docs.py`, so
a rename that these pages miss fails a test rather than rotting silently.

## The import-pinning contract

Each page ends with a machine-readable block::

    <!-- names
    entente.engine.game:Game
    ... -->

listing the fully-qualified `module:attribute` of every name the page
documents. `tests/test_api_docs.py` parses every block and imports every name
(skipping a page's optional-dependency modules when that group is absent, the
same way the suites do). Adding a name to a page without exporting it, or
renaming an exported name without updating its page, fails that test.
