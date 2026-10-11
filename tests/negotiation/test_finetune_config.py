"""Pin the dialogue-LoRA config's YAML load, its overrides and its validation.

Torch-free, so these run in the default environment, as
tests/training/test_config.py does for the training config.
"""

from __future__ import annotations

import pytest

from entente.negotiation.finetune.config import DialogueLoraConfig, LoraConfig
from entente.training.config import ConfigError, load_config

SMOKE_CONFIG = "src/entente/negotiation/finetune/configs/smoke.yaml"
BASE_CONFIG = "src/entente/negotiation/finetune/configs/base.yaml"


class TestDialogueLoraConfig:
    """The fine-tune config's defaults and its shipped config files."""

    def test_defaults(self) -> None:
        cfg = DialogueLoraConfig()
        assert cfg.base_model == "HuggingFaceTB/SmolLM3-3B"
        assert cfg.load_in_4bit is False
        assert cfg.lora.target_modules == ("q_proj", "v_proj")

    def test_smoke_config_loads(self) -> None:
        cfg = load_config(DialogueLoraConfig, SMOKE_CONFIG)
        assert cfg.base_model == "hf-internal-testing/tiny-random-gpt2"
        assert cfg.load_in_4bit is False
        assert cfg.lora.target_modules == ("c_attn",)
        assert cfg.device == "cpu"

    def test_base_config_loads(self) -> None:
        cfg = load_config(DialogueLoraConfig, BASE_CONFIG)
        assert cfg.base_model == "HuggingFaceTB/SmolLM3-3B"
        assert cfg.load_in_4bit is True
        assert cfg.lora.target_modules == ("q_proj", "v_proj")

    def test_overrides_apply(self) -> None:
        cfg = load_config(DialogueLoraConfig, SMOKE_CONFIG, overrides=["optim.max_steps=5"])
        assert cfg.optim.max_steps == 5

    def test_rejects_unknown_field(self) -> None:
        with pytest.raises(ConfigError):
            load_config(DialogueLoraConfig, overrides=["not_a_field=1"])

    def test_rejects_non_positive_lr(self) -> None:
        with pytest.raises(ConfigError):
            load_config(DialogueLoraConfig, overrides=["optim.lr=0"])

    def test_frozen(self) -> None:
        cfg = DialogueLoraConfig()
        with pytest.raises(ValueError, match="frozen"):
            cfg.seed = 1  # type: ignore[misc]

    def test_lora_config_defaults(self) -> None:
        lora = LoraConfig()
        assert lora.r == 8
        assert lora.target_modules == ("q_proj", "v_proj")
