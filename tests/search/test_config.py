"""piKL config models, and the committed ``configs/pikl/*.yaml`` files themselves.

Both the models and every shipped file must parse and validate
(docs/pikl.tex Sec. 7). Torch-free.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from entente.search.config import (
    EtaConfig,
    LambdaSweepConfig,
    PiklAgentConfig,
    PiklSearchConfig,
)
from entente.training.config import ConfigError, load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "pikl"


def test_committed_base_yaml_loads() -> None:
    cfg = load_config(PiklAgentConfig, CONFIGS / "base.yaml")
    assert cfg.search.k == 12
    assert cfg.search.iterations == 128
    assert cfg.search.eta.mode == "adaptive"
    assert cfg.search.eta.constant is None
    assert cfg.fallback == "greedy"


def test_committed_sweep_yaml_loads_with_plan_grid() -> None:
    cfg = load_config(LambdaSweepConfig, CONFIGS / "lambda_sweep.yaml")
    assert cfg.lambdas[:4] == (0.0, 0.3, 1.0, 3.0)
    assert math.isinf(cfg.lambdas[4])  # YAML .inf -> the anchor end
    assert cfg.rotate_seats and cfg.games * 7 >= 100  # plan's search-agent floor
    assert cfg.agreement_shards is None  # honest default: no human data yet


def test_dotted_overrides_reach_nested_search_fields() -> None:
    cfg = load_config(
        LambdaSweepConfig,
        CONFIGS / "lambda_sweep.yaml",
        overrides=[
            "agent.search.k=2",
            "agent.search.iterations=4",
            "games=1",
            "lambdas=[0.0, .inf]",
        ],
    )
    assert cfg.agent.search.k == 2
    assert cfg.agent.search.iterations == 4
    assert cfg.games == 1
    assert len(cfg.lambdas) == 2 and math.isinf(cfg.lambdas[1])


def test_domain_validation() -> None:
    with pytest.raises(ValueError, match="lam must be"):
        PiklSearchConfig(lam=-1.0)
    with pytest.raises(ValueError, match="lam must be"):
        PiklSearchConfig(lam=math.nan)
    with pytest.raises(ValueError, match=r"eta\.value is required"):
        EtaConfig(mode="constant")
    assert EtaConfig(mode="constant", value=0.5).constant == 0.5
    with pytest.raises(ValueError, match="every lambda"):
        LambdaSweepConfig(agent=PiklAgentConfig(checkpoint="x.pt"), lambdas=(0.5, -0.1))
    with pytest.raises(ConfigError, match="k"):
        load_config(PiklAgentConfig, CONFIGS / "base.yaml", overrides=["search.k=0"])


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ConfigError):
        load_config(PiklAgentConfig, CONFIGS / "base.yaml", overrides=["search.typo=1"])
