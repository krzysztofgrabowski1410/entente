"""Pin ``LocalLoraProvider.generate()`` against a freshly trained smoke adapter.

``finetune``-marked: needs the `finetune` group and network access to fetch the
tiny test model, cached after the first run. The offline construction-time
validation is in
tests/negotiation/test_generate.py::TestLocalLoraProviderConstruction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from entente.engine.orders import parse_order
from entente.negotiation.finetune.config import DialogueLoraConfig
from entente.negotiation.finetune.train import train
from entente.negotiation.generate import LocalLoraProvider
from entente.negotiation.intents import Intent
from entente.training.config import load_config

pytestmark = pytest.mark.finetune

SMOKE_CONFIG = "src/entente/negotiation/finetune/configs/smoke.yaml"


@pytest.fixture(scope="module")
def smoke_adapter_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a real adapter, tiny and fast.

    A handful of steps is enough to exercise save, load and generate; whether
    the loop learns is tests/negotiation/test_finetune.py's job.
    """
    out = tmp_path_factory.mktemp("lora_smoke")
    cfg = load_config(
        DialogueLoraConfig, SMOKE_CONFIG, overrides=[f"out_dir={out}", "optim.max_steps=4"]
    )
    result = train(cfg)
    return result.adapter_dir


def _intent() -> Intent:
    return Intent(
        proposer="GERMANY",
        ally="RUSSIA",
        phase="S1901M",
        proposer_action=(parse_order("A BER - SIL"),),
        ally_action=(parse_order("A WAR - GAL"),),
        rationale="coordinate against Austria",
    )


class TestLocalLoraProviderGenerate:
    """Generating from a real, if tiny, trained adapter."""

    def test_loads_and_generates_a_string(self, smoke_adapter_dir: Path) -> None:
        provider = LocalLoraProvider(str(smoke_adapter_dir), max_new_tokens=8)
        text = provider.generate(_intent())
        assert isinstance(text, str)
        assert provider.base_model == "hf-internal-testing/tiny-random-gpt2"
        assert provider.adapter_dir == str(smoke_adapter_dir)

    def test_satisfies_the_provider_protocol_structurally(self, smoke_adapter_dir: Path) -> None:
        # Provider (generate.py) is a plain (non-runtime-checkable) Protocol,
        # like the rest of this module's design -- satisfied structurally,
        # the same way StubProvider/AnthropicProvider are (no isinstance check).
        from entente.negotiation.generate import Provider

        def accepts_a_provider(provider: Provider) -> str:
            return provider.generate(_intent())

        text = accepts_a_provider(LocalLoraProvider(str(smoke_adapter_dir), max_new_tokens=4))
        assert isinstance(text, str)

    def test_env_var_fallback_loads_the_same_adapter(
        self, smoke_adapter_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ENTENTE_DIALOGUE_LORA_DIR", str(smoke_adapter_dir))
        provider = LocalLoraProvider(max_new_tokens=4)
        assert provider.adapter_dir == str(smoke_adapter_dir)

    def test_greedy_generation_is_repeatable(self, smoke_adapter_dir: Path) -> None:
        provider = LocalLoraProvider(str(smoke_adapter_dir), max_new_tokens=8, temperature=0.0)
        intent = _intent()
        assert provider.generate(intent) == provider.generate(intent)
