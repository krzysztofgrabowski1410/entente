# Engine API

The stdlib-only, deterministic core: a ruleset-parameterized adjudicator and
the `Game` facade that drives the phase machine over it. Spec:
[docs/adjudication.pdf](../adjudication.pdf).

## The `Game` facade

`entente.engine.game` is the one object most callers need. It owns the phase
pointer, supply-center ownership, pending submissions, victory, and history.

```python
class Game:
    @classmethod
    def new(cls, ruleset: Ruleset | str, map_name: str, *,
            start_year: int | None = None, seed: int | None = None) -> Game: ...

    # read surface (properties)
    state: State
    ownership: Mapping[str, str]
    phase: str                 # tag, e.g. "S1901M"
    phase_kind: PhaseKind
    year: int
    powers: tuple[str, ...]
    history: tuple[PhaseRecord, ...]
    is_over: bool
    winner: str | None

    def legal_orders(self, power: str) -> tuple[AnyOrder, ...]: ...
    def legal_layer2_orders(self, power: str) -> tuple[GameOrder, ...]: ...
    def submit(self, power: str, orders: Sequence[GameOrder]) -> None: ...
    def adjudicate(self) -> PhaseRecord: ...
    def supply_count(self, power: str) -> int: ...
    def build_quota(self, power: str) -> int: ...
    def grant(self, giver: str, receiver: str, fact: str) -> None: ...
```

`submit` takes an ordered list (sequence carries meaning) and replaces any
earlier sheet by that power (DATC 4.D.1 choice C). Absent powers get the phase
defaults. `adjudicate()` resolves the current phase and advances the machine,
auto-executing or skipping decision-free phases. `GameError` is raised for
facade contract violations (unknown power, wrong-phase order kinds, acting on
a finished game).

### Minimal working example

```python
from entente.engine.game import Game
from entente.engine.orders import parse_order

game = Game.new("standard", "standard")
game.submit("FRANCE", [parse_order("A PAR - BUR")])
record = game.adjudicate()
print(record.phase, [(r.text, r.resolution.value) for r in record.resolutions])
# S1901M [('A PAR - BUR', 'succeeds'), ...]
```

## Adjudication (the pure function)

`entente.engine.adjudicator.adjudicate` is the pure movement-phase resolver
underneath the facade. `engine=` selects the implementation -- the staged
`"resolver"` (default) or the stable-labeling `"enumerator"` (the executable
spec and permanent differential oracle). Both are deterministic and
observationally identical.

```python
def adjudicate(ruleset: Ruleset, state: State,
               sheets: Mapping[str, Sequence[Order]], *,
               graph: MapGraph | None = None,
               engine: EngineName = "resolver") -> PhaseResult: ...
```

`PhaseResult` carries `resolutions`, `dislodgements`, and `new_state`;
`Resolution` is the `SUCCEEDS`/`FAILS` enum; `OrderResolution` and
`Dislodgement` are the per-order and per-dislodgement records.

## Orders, state, map

- `entente.engine.orders` -- the order dataclasses (`Hold`, `Move`, `Support`,
  `Convoy`, `Dislodge`, `Bar`, plus retreat/adjustment and Layer-2 variants),
  the `format_order`/`parse_order` notation, and the `AnyOrder`/`GameOrder`
  unions. `OrderSyntaxError` is raised for malformed text (distinct from a
  well-formed-but-illegal order, which becomes a Hold in Phase 0). Every
  unit-bearing order and `UnitRef` carries an optional `special` field -- the
  Beleriand `(M)`/`(G)` designator naming one of MORGOTH's two `3A` specials
  (`3A(M) ANF - LOT`); it round-trips through the notation but is
  adjudicator-invisible, consumed only by the Beleriand game layer.
- `entente.engine.state` -- `Unit` and the immutable `State` (a canonically
  sorted multiset of units); `check_state` validates placement + capacity.
- `entente.engine.map` -- `load_map(name)` returns a `MapGraph`; `Location`
  (province + optional coast), `Province`, `Terrain`, `UnitKind` are the value
  types. `Location != Province` from day one (DATC 6.B).
- `entente.engine.rules` -- `RULESETS` (name -> ruleset) and the `Ruleset`
  protocol (the frozen parameterization surface).

```python
from entente.engine.map import load_map
from entente.engine.serialization import game_to_text, game_from_text

graph = load_map("standard")
text = game_to_text(game)          # versioned JSON of the full decision record
restored = game_from_text(text)    # replays + verifies the snapshot
```

<!-- names
entente.engine.game:Game
entente.engine.game:GameError
entente.engine.game:PhaseRecord
entente.engine.adjudicator:adjudicate
entente.engine.adjudicator:ENGINES
entente.engine.adjudicator:EngineName
entente.engine.adjudicator:PhaseResult
entente.engine.adjudicator:Resolution
entente.engine.adjudicator:OrderResolution
entente.engine.adjudicator:Dislodgement
entente.engine.adjudicator:AdjudicationError
entente.engine.orders:Hold
entente.engine.orders:Move
entente.engine.orders:Support
entente.engine.orders:Convoy
entente.engine.orders:Dislodge
entente.engine.orders:Bar
entente.engine.orders:UnitRef
entente.engine.orders:format_order
entente.engine.orders:parse_order
entente.engine.orders:OrderSyntaxError
entente.engine.state:State
entente.engine.state:Unit
entente.engine.state:check_state
entente.engine.map:load_map
entente.engine.map:MapGraph
entente.engine.map:Location
entente.engine.map:Province
entente.engine.map:Terrain
entente.engine.map:UnitKind
entente.engine.rules:RULESETS
entente.engine.rules:Ruleset
entente.engine.serialization:game_to_text
entente.engine.serialization:game_from_text
entente.engine.serialization:SerializationError
-->
