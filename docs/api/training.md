# Training API

The behavioral-cloning training loop and the project's single typed-config
mechanism (`ml` group for `train`; the config loader is stdlib + pydantic).

## Config loading (shared across the project)

`entente.training.config` is the one YAML + dotted-override loader the whole
project uses (the tournament, sweep, and press configs are all built on it).

```python
def load_config(model: type[M], path: str | Path | None = None,
                overrides: Iterable[str] = ()) -> M: ...
def load_yaml(path: str | Path) -> dict[str, Any]: ...
def apply_overrides(data: dict, overrides: Iterable[str]) -> dict: ...
class ConfigError(ValueError): ...
```

`load_config` reads a YAML file, applies `key=value` (dotted) overrides, and
validates against a frozen pydantic model that forbids unknown keys.
`BCConfig` (with nested `ModelConfig`, `OptimConfig`, `DataConfig`) is the BC
run specification.

```python
from entente.training.config import BCConfig, load_config

cfg = load_config(BCConfig, "src/entente/training/configs/bc/smoke.yaml",
                  overrides=["optim.max_steps=10", "device=cpu"])
```

## Training (`ml` group)

```python
def train(cfg: BCConfig, *, train_ds: ShardDataset | None = None,
          val_ds: ShardDataset | None = None) -> TrainResult: ...
def evaluate_top1(model, dataset, device, *, max_examples=...) -> tuple[float, int]: ...
def resolve_device(spec: str) -> torch.device: ...
```

`train` builds datasets from `cfg.data` unless supplied, writes a JSONL metrics
stream and a checkpoint under `cfg.out_dir`, and returns a `TrainResult`
(steps, checkpoint path, metrics path). `MetricTracker`
(`entente.training.tracking`) is the always-JSONL, optionally-wandb sink.

The command-line entry point is `scripts/train_bc.py`; GPU-scale runs are
WSL2/CUDA (docs/setup.md, "Behavioral cloning"), the CPU smoke config proves
the loop learns.

<!-- names
entente.training.config:load_config
entente.training.config:load_yaml
entente.training.config:apply_overrides
entente.training.config:ConfigError
entente.training.config:BCConfig
entente.training.config:ModelConfig
entente.training.config:OptimConfig
entente.training.config:DataConfig
entente.training.bc:train
entente.training.bc:evaluate_top1
entente.training.bc:resolve_device
entente.training.bc:TrainResult
entente.training.tracking:MetricTracker
-->
