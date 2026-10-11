# Data pipeline API

The behavioral-cloning data path (standard ruleset only, `ml` group): a game
source is preprocessed to Parquet shards, which a `ShardDataset` serves as
per-(phase, power) training examples.

## Source -> preprocess -> ShardDataset

```python
# 1. a source of GameRecords
class PaquetteJSONLSource:                     # entente.data.paquette
    def __init__(self, path: str | Path) -> None: ...
def game_from_json(raw: dict[str, Any]) -> GameRecord: ...

# 2. preprocess to sharded Parquet
def preprocess(source, out_dir: str | Path, *,        # entente.data.preprocess
               shard_size: int = ..., max_games: int | None = None,
               skip_games: int = 0) -> PreprocessStats: ...

# 3. serve examples (torch tensors)
class ShardDataset:                            # entente.data.dataset
    def __init__(self, shards_dir, *, game_ids=None, cache_shards=...) -> None: ...
def game_split(shards_dir, val_fraction: float, *, seed: int = 0): ...
def load_bc_datasets(cfg: DataConfig) -> tuple[ShardDataset, ShardDataset]: ...
def collate(examples) -> Batch: ...
```

`preprocess` accepts any object satisfying the `GameSource` protocol; the
committed `PaquetteJSONLSource` reads the Paquette no-press dump, and
`entente.data.synthetic.synthesize_games` produces the greedy-teacher fallback
when the dataset is not mirrored (docs/setup.md). `game_split` partitions by
game (no phase leaks across the train/val boundary).

### Minimal working example (synthetic, no download)

```python
from entente.data.synthetic import synthesize_games
from entente.data.paquette import game_from_json
from entente.data.preprocess import preprocess
from entente.data.dataset import ShardDataset

games = [game_from_json(g) for g in synthesize_games(range(4), teacher="greedy")]
preprocess(games, "out_shards", shard_size=100_000)
ds = ShardDataset("out_shards")
print(len(ds), ds[0].features.shape)   # (81, 32) location-token tensor
```

The dataset row schema and the order vocabulary are the seam the BC model's
legality masking pins exactly (`entente.nn.order_vocab`, tested in
`tests/nn/test_masking.py`).

<!-- names
entente.data.paquette:PaquetteJSONLSource
entente.data.paquette:GameRecord
entente.data.paquette:GameSource
entente.data.paquette:game_from_json
entente.data.preprocess:preprocess
entente.data.preprocess:PreprocessStats
entente.data.synthetic:synthesize_games
entente.data.dataset:ShardDataset
entente.data.dataset:game_split
entente.data.dataset:load_bc_datasets
entente.data.dataset:collate
-->
