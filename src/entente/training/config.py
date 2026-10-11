"""Typed configuration: pydantic v2 models + YAML files + dotted CLI overrides.

A run is reproducible only if its whole configuration is in one file, so every
knob is a validated field with a default and an experiment is a YAML file plus
dotted ``--set a.b=c`` overrides on the command line. Three small, reusable
pieces:

- :func:`load_yaml` -- read a YAML file into a mapping, turning every failure
  (missing file, malformed YAML, non-mapping top level) into a
  :class:`ConfigError` with a stable message;
- :func:`apply_overrides` -- fold a list of ``a.b.c=value`` strings into that
  mapping, each ``value`` parsed as a YAML scalar so ``--set optim.lr=3e-4`` and
  ``--set data.shards=[a,b]`` are typed, not stringly;
- :func:`load_config` -- validate the merged mapping against a pydantic model.

The concrete behavioral-cloning schemas (:class:`BCConfig` and its parts) live
here too, and ``entente.eval.config``'s tournament loader builds its own pydantic
``TournamentConfig`` on :func:`load_yaml` and :func:`apply_overrides`, so the
project has exactly one YAML-plus-override mechanism to learn.
This module imports no torch -- it loads in the default (non-``ml``) environment.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

if TYPE_CHECKING:
    from collections.abc import Iterable

M = TypeVar("M", bound=BaseModel)


class ConfigError(ValueError):
    """Report a malformed or invalid config.

    That covers an unreadable file, bad YAML and a value the target model
    rejects. Pydantic's ``ValidationError`` is re-raised as this, so a caller
    catches one project-owned type rather than two library ones.
    """


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a mapping.

    Every failure mode -- unreadable file, malformed YAML, a top level that is
    not a mapping -- becomes a :class:`ConfigError`. An empty file reads as an
    empty mapping, so a config may hold nothing but comments.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {path}: {error}") from error
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return data


def apply_overrides(data: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Fold ``a.b.c=value`` override strings into a copy of ``data``.

    Each ``value`` is parsed with ``yaml.safe_load`` so scalars keep their type
    (``lr=3e-4`` is a float, ``fresh=true`` a bool, ``tags=[x,y]`` a list). A
    dotted key walks/creates nested mappings; a non-mapping intermediate is
    overwritten with a fresh mapping, so the override wins. The input is not
    mutated.
    """
    result = copy.deepcopy(data)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ConfigError(f"override must be key=value, got {item!r}")
        value = yaml.safe_load(raw)
        parts = key.split(".")
        node = result
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return result


def load_config(model: type[M], path: str | Path | None = None, overrides: Iterable[str] = ()) -> M:
    """Load, override, and validate a config against a pydantic ``model``.

    ``path`` may be ``None`` (start from an empty mapping and rely on model
    defaults and overrides). A pydantic ``ValidationError`` is re-raised as a
    :class:`ConfigError` so the caller sees one error type.
    """
    data = load_yaml(path) if path is not None else {}
    overrides = list(overrides)
    if overrides:
        data = apply_overrides(data, overrides)
    try:
        return model.model_validate(data)
    except ValidationError as error:
        raise ConfigError(f"invalid config: {error}") from error


# -- behavioral-cloning config schema -----------------------------------------
#
# Frozen models (a config is read once and never mutated mid-run). Bounds are
# validation, not decoration: a d_model that is not a multiple of the head count
# would crash attention far from here, so it is rejected at load time.


class _Frozen(BaseModel):
    """Base for every config section: immutable, and it refuses unknown keys.

    A misspelled key is the failure mode a config system has to catch, because
    the run would otherwise validate, ignore the setting and quietly train
    something other than what the file says.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelConfig(_Frozen):
    """Fix the encoder and head geometry.

    The bounds keep a model inside the design's under-10M-parameter budget at
    d_model 192-256 and 6-8 layers. ``independent_heads`` selects the simpler
    policy: per-unit heads with no autoregressive conditioning between units.
    """

    d_model: int = Field(default=192, ge=32, le=512)
    n_layers: int = Field(default=6, ge=1, le=16)
    n_heads: int = Field(default=8, ge=1, le=16)
    ff_mult: int = Field(default=4, ge=1, le=8)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    independent_heads: bool = False

    def model_post_init(self, _context: Any) -> None:
        """Reject a geometry multi-head attention could not build."""
        # A d_model not divisible by n_heads would crash multi-head attention
        # far from here; reject it at load time so it surfaces as a ConfigError.
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model {self.d_model} must be divisible by n_heads {self.n_heads}")


class OptimConfig(_Frozen):
    """Set the AdamW optimizer and its warmup-plus-cosine schedule."""

    lr: float = Field(default=3e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_steps: int = Field(default=500, ge=0)
    max_steps: int = Field(default=20000, ge=1)
    batch_size: int = Field(default=512, ge=1)
    grad_clip: float = Field(default=1.0, ge=0.0)
    value_loss_weight: float = Field(default=0.5, ge=0.0)


class DataConfig(_Frozen):
    """Say where the preprocessed shards live and how to split them.

    ``val_fraction`` is held out by *game*, through a seeded hash of the game id
    that cuts across shards: positions from one game are near-duplicates of each
    other, so a by-row split would report a top-1 the model has effectively seen.
    """

    shards_dir: str = "data/processed"
    val_fraction: float = Field(default=0.05, ge=0.0, lt=1.0)
    num_workers: int = Field(default=0, ge=0)
    shuffle: bool = True


class BCConfig(_Frozen):
    """Describe a full behavioral-cloning run: model, optimizer, data and bookkeeping.

    ``seed`` fixes every generator the run draws from, ``out_dir`` receives the
    checkpoint and the JSONL metrics stream, and ``run_name`` labels the metrics
    and the hosted run when one exists.
    """

    model: ModelConfig = ModelConfig()
    optim: OptimConfig = OptimConfig()
    data: DataConfig = DataConfig()
    seed: int = 0
    out_dir: str = "checkpoints/bc"
    run_name: str = "bc"
    device: str = "auto"
    log_every: int = Field(default=50, ge=1)
    eval_every: int = Field(default=500, ge=1)
