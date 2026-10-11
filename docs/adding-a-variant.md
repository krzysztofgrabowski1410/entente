# Adding a variant

How to put a new Diplomacy variant on this engine, ordered from "an
afternoon" to "needs an engine change". Calibration: the families on
vdiplomacy.com (surveyed 2026-07-30, 150+ variants -- new maps, historic
starts, two-player duels, 30+ player monsters, build-anywhere, fog,
economic layers) all land in tiers 1-3 below except fog of war and
economic layers, which are tier 4. Nothing was copied from that site;
it is used only to calibrate what "a variant" means in practice.

The engine's variant machinery is three registries, each populated by the
variant package importing itself (`entente/__init__.py` holds one import
line per variant):

| Registry | Owns | Where |
|---|---|---|
| `RULESETS` | adjudication law (the 16-member `Ruleset` protocol) | `engine/rules/__init__.py` |
| `GAME_HOOKS` | stateful game-loop law (victory, timers, whole-phase overrides) | `engine/game_hooks.py` |
| `VARIANT_SPECS` | packaging + presentation (map file, calendar, caption, off-board tiles) | `engine/variant_meta.py` |

A map JSON is both the board *and* the scenario: provinces, adjacency,
supply centers, starting units, home centers. **Powers are derived** --
whoever appears in `starting_units`/`home_centers` is a power; there is
no power list anywhere else. Victory is likewise derived: the standard
rule is a strict majority of the map's supply centers (18 of 34 on the
standard board), computed from the map, so a new board inherits a correct
threshold for free.

## Tier 1 -- a new map, unchanged rules

Covers most variants in the wild: new boards (an Ancient Mediterranean,
an Africa, a South America), historic starts on the classic board (a
"1898"), two-player duels, many-power monsters. No Python law at all.

1. **Package.** `src/entente/variants/<name>/` with `__init__.py` and
   `data/<name>.json`. Hand-transcribe the map from primary sources and
   record provenance in the JSON's `provenance` field; if you download
   anything, record where it came from and under which licence.
2. **The map file.** Top-level keys (see
   [`engine/map/data/standard.json`](../src/entente/engine/map/data/standard.json)
   and Beleriand's for a maximal example):
   - `name` -- must equal the filename stem; the loader checks.
   - `provenance` -- where every fact was transcribed from.
   - `provinces` -- key -> `{name, terrain (land|coastal|sea),
     supply_center (bool or int count), capacity (default 1), fortress,
     restricted_to, one_way, coasts ["NC", ...]}`. The last four are
     optional rule-flavored *data*: entry bans, one-way edges, fortress
     bonuses and multi-unit Areas are map facts here, not code.
   - `army_adjacency` / `fleet_adjacency` -- the second in `PROV/COAST`
     location form for split coasts.
   - `starting_units` -- power -> `["A PAR", "F STP/SC", ...]`.
   - `home_centers` -- power -> province keys (also the initial
     ownership; centers outside every home list start neutral).
   - Optional: `default_capacity`, `mountain_barriers` (directed
     `[from, to]` pairs), `centroids` (only needed by the
     schematic SVG renderer).
3. **Validation is free.** The loader validates aggressively at load --
   symmetric adjacency per unit kind, terrain sanity (no army edges into
   sea, no fleet edges into inland), declared coasts only, barriers on
   real edges. A transcription typo fails loudly at first load; that is
   the design (transcription errors are a testing problem).
4. **Register.** In the package `__init__.py`:

   ```python
   from entente.engine.variant_meta import VARIANT_SPECS, VariantSpec

   VARIANT_SPECS.setdefault(
       "mediterranean",
       VariantSpec(
           name="mediterranean",
           ruleset="standard",          # unchanged law: reuse it
           map_name="mediterranean",
           start_year=1901,
           year_direction=1,
           caption="counts up from 1901",
           data_package="entente.variants.mediterranean",
       ),
   )
   ```

   plus one line in `entente/__init__.py`:
   `import entente.variants.mediterranean`.
5. **Check.** `uv run pytest tests/engine -q` (the loader validation and
   registry tests ride along), then play it:
   `Game.new("standard", "mediterranean")` from Python. The API and CLI
   work immediately.

A historic start ("same board, 1898 positions") is simply a second map
file -- geometry copied, `starting_units`/`home_centers` re-done -- plus
its own spec with the right `start_year`. A two-player duel is a map
whose `starting_units` name two powers. A 30-power monster is just a
long `starting_units`; nothing else changes.

**Off-map provinces.** Beleriand-style provinces that live on their own
off-map tile go in the spec's `off_board`, so that anything drawing the
board, a GUI for instance, could leave them off the main picture.

## Tier 2 -- rule changes inside the existing axes

The `Ruleset` protocol's 16 members are the menu of what may lawfully
vary without an engine change (the specification,
[docs/adjudication.pdf](adjudication.pdf), has the full standard-vs-Beleriand
table). Implement a frozen dataclass like
[`rules/standard.py`](../src/entente/engine/rules/standard.py), change
only the members you mean to change, `RULESETS.setdefault(...)`, and
point your spec's `ruleset` at it.

| You want | Change |
|---|---|
| unit strengths, units above base strength 1 | `base_strength`, `modified_base` |
| terrain penalties, fortress bonuses | map data (`mountain_barriers`, `fortress`) + `modified_base` |
| a toll on where a convoy lands | `modified_base`, through its `disembark` keyword: the landing seas the move's own convoy orders reach |
| who may convoy whom | `convoy_capability` |
| entry bans, one-way edges, multi-unit provinces | map data only (`restricted_to`, `one_way`, `capacity`) |
| targeted Dislodge / Bar available or not | `order_registry` |
| support-cut protection scope | `cut_exception_scope` |
| supply-center capture rule | `update_supply_ownership` |
| strict unit designations | `strict_designation` |

Write conformance cases for what you changed, in the BATC style
(`tests/batc/` -- hand-encoded YAML, one loader, exact-count guards).
The differential twin adjudicators are the only oracle an original
variant has, so cases are not optional.

## Tier 3 -- stateful, timed, or whole-phase law

Anything the adjudicator must not know about -- victory conditions,
timers and scheduled events, knowledge transfers, replacing a whole
phase's adjudication -- is a `GameHooks` subclass registered in
`GAME_HOOKS` (`entente/variants/beleriand/layer2.py` is the
worked example, `winter.py` shows a full winter-adjudication
replacement). Two vdiplomacy-family examples:

- **Build anywhere**: override the adjustment step in hooks (the
  Beleriand winter replacement is the precedent for owning that phase).
- **Custom victory threshold or scoring**: `decides_victory` +
  `game_result` -- return `False`/`None` to keep the standard
  strict-majority rule, or own it entirely.

One current limit worth knowing: initial ownership is derived from
`home_centers`, so an "every center owned from the start" setup
(NoNeutrals style) needs a small schema addition rather than
a hook -- the first variant to need it should propose it.

## Tier 4 -- beyond the current engine

- **New order words** (a variant verb like Beleriand's `MSK`): the
  grammar is not yet a registry, and the third variant with its own hook
  orders is the one that should make it one. New *adjudicator-grammar*
  order types change the `Ruleset` protocol itself: a new axis needs a
  design of its own and a concrete ruleset that proves it.
- **Fog of war / hidden information**: the engine is full-information by
  design; out of scope short of an engine-scale redesign.
- **Economic layers, new unit kinds (wings)**: same -- engine-scale.

## Checklist

- [ ] `src/entente/variants/<name>/` with `data/<name>.json` (provenance
      recorded, downloads with their source and licence)
- [ ] `VariantSpec` registered in the package `__init__.py`
      (+ ruleset / hooks registrations if tier 2/3)
- [ ] one import line in `entente/__init__.py`
- [ ] `uv run pytest tests/engine -q` green (loader + registry)
- [ ] conformance YAML for any changed law (tier 2/3)
- [ ] out of scope by decree: tournament harness and NN checkpoints stay
      standard-only -- their guards will refuse your
      variant, and that is correct
