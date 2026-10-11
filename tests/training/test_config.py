"""Pin the shared config system and the behavioral-cloning schema over it.

That covers the YAML load, the dotted overrides and the validation. Torch-free,
so these run in the default environment with no ``ml`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from entente.training.config import (
    BCConfig,
    ConfigError,
    ModelConfig,
    apply_overrides,
    load_config,
    load_yaml,
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class _Nested(BaseModel):
    lr: float = 1.0
    name: str = "x"


class _Root(BaseModel):
    a: int = 0
    child: _Nested = _Nested()


def test_load_yaml_empty_file_is_empty_mapping(tmp_path: Path) -> None:
    assert load_yaml(_write(tmp_path, "")) == {}


def test_load_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(_write(tmp_path, "- a\n- b\n"))


def test_load_yaml_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_malformed(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(_write(tmp_path, "a: [unterminated\n"))


def test_apply_overrides_types_scalars_via_yaml() -> None:
    out = apply_overrides({}, ["a=3", "child.lr=2.5e-4", "child.name=hi", "flag=true"])
    assert out == {"a": 3, "child": {"lr": 2.5e-4, "name": "hi"}, "flag": True}


def test_apply_overrides_does_not_mutate_input() -> None:
    original = {"child": {"lr": 1.0}}
    apply_overrides(original, ["child.lr=9"])
    assert original == {"child": {"lr": 1.0}}


def test_apply_overrides_overwrites_non_mapping_intermediate() -> None:
    out = apply_overrides({"child": 5}, ["child.lr=2"])
    assert out == {"child": {"lr": 2}}


def test_apply_overrides_requires_equals() -> None:
    with pytest.raises(ConfigError, match="key=value"):
        apply_overrides({}, ["just_a_key"])


def test_load_config_merges_file_and_overrides(tmp_path: Path) -> None:
    path = _write(tmp_path, "a: 1\nchild:\n  name: fromfile\n")
    cfg = load_config(_Root, path, overrides=["child.lr=0.5"])
    assert cfg.a == 1
    assert cfg.child.name == "fromfile"
    assert cfg.child.lr == 0.5


def test_load_config_no_path_uses_defaults_and_overrides() -> None:
    cfg = load_config(_Root, None, overrides=["a=7"])
    assert cfg.a == 7
    assert cfg.child.name == "x"


def test_load_config_invalid_value_raises_config_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "a: not-an-int\n")
    with pytest.raises(ConfigError, match="invalid config"):
        load_config(_Root, path)


def test_bcconfig_defaults_are_in_the_documented_ranges() -> None:
    cfg = BCConfig()
    assert 192 <= cfg.model.d_model <= 256
    assert 6 <= cfg.model.n_layers <= 8
    assert cfg.model.d_model % cfg.model.n_heads == 0
    assert cfg.optim.batch_size == 512


def test_bcconfig_rejects_extra_keys(tmp_path: Path) -> None:
    path = _write(tmp_path, "typo_field: 3\n")
    with pytest.raises(ConfigError):
        load_config(BCConfig, path)


def test_modelconfig_rejects_indivisible_dmodel() -> None:
    with pytest.raises(ConfigError, match="divisible"):
        load_config(ModelConfig, None, overrides=["d_model=100", "n_heads=8"])
