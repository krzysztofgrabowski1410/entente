"""Pin the provider protocol and everything about it that is checkable offline.

That is StubProvider whole, the pure prompt and response helpers behind
AnthropicProvider, and the errors it raises at construction. A live call needs
ANTHROPIC_API_KEY and the `llm` group; nothing here exercises one, and no
result below should be read as covering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from entente.engine.orders import parse_order
from entente.negotiation.generate import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicProvider,
    StubProvider,
    _build_prompt,
    _extract_text,
)
from entente.negotiation.intents import Intent


def _intent(rationale: str = "") -> Intent:
    return Intent(
        proposer="GERMANY",
        ally="RUSSIA",
        phase="S1901M",
        proposer_action=(parse_order("A BER - SIL"), parse_order("A MUN H")),
        ally_action=(parse_order("A WAR - GAL"),),
        rationale=rationale,
    )


class TestStubProvider:
    """The offline template provider the demo and the tests generate with."""

    def test_generate_is_deterministic_and_matches_the_shared_grammar(self) -> None:
        text = StubProvider().generate(_intent("take Galicia instead"))
        assert text == (
            "PROPOSAL from GERMANY to RUSSIA for S1901M: "
            "GERMANY orders [A BER - SIL, A MUN H]; RUSSIA orders [A WAR - GAL]. "
            "take Galicia instead"
        )

    def test_generate_omits_trailing_space_without_a_rationale(self) -> None:
        text = StubProvider().generate(_intent())
        assert text.endswith("RUSSIA orders [A WAR - GAL].")

    def test_generate_is_repeatable(self) -> None:
        intent = _intent("consistent")
        provider = StubProvider()
        assert provider.generate(intent) == provider.generate(intent)


class TestBuildPrompt:
    """The prompt both real-model providers are given."""

    def test_prompt_names_every_party_and_action(self) -> None:
        prompt = _build_prompt(_intent("avoid a bounce"))
        assert "GERMANY" in prompt
        assert "RUSSIA" in prompt
        assert "A BER - SIL" in prompt
        assert "A WAR - GAL" in prompt
        assert "avoid a bounce" in prompt

    def test_prompt_omits_rationale_sentence_when_absent(self) -> None:
        prompt = _build_prompt(_intent())
        assert "Reasoning to weave in" not in prompt


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeMessage:
    content: list[object] = field(default_factory=list)
    stop_reason: str | None = "end_turn"


class TestExtractText:
    """Pulling the message text out of a provider response."""

    def test_returns_the_first_text_block(self) -> None:
        message = _FakeMessage(content=[_FakeTextBlock("hello there")])
        assert _extract_text(message) == "hello there"

    def test_skips_non_text_blocks(self) -> None:
        message = _FakeMessage(
            content=[object(), _FakeTextBlock("the real message")],
        )
        assert _extract_text(message) == "the real message"

    def test_raises_with_stop_reason_when_no_text_block(self) -> None:
        message = _FakeMessage(content=[], stop_reason="refusal")
        with pytest.raises(RuntimeError, match="refusal"):
            _extract_text(message)


class TestAnthropicProviderConstruction:
    """What the hosted provider checks before it will construct."""

    def test_requires_the_package_or_a_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        try:
            import anthropic  # noqa: F401

            installed = True
        except ImportError:
            installed = False
        expected = "ANTHROPIC_API_KEY" if installed else "anthropic"
        with pytest.raises(RuntimeError, match=expected):
            AnthropicProvider()

    def test_default_model_constant_is_opus_4_8(self) -> None:
        # Pinned so changing the default model id has to be done on purpose.
        assert DEFAULT_ANTHROPIC_MODEL == "claude-opus-4-8"


class TestLocalLoraProviderConstruction:
    """What the adapter provider checks before it will construct.

    These are the errors checkable without the model, as
    TestAnthropicProviderConstruction is for the hosted provider. The real
    generate() path needs the `finetune` group and a trained adapter, and lives
    in tests/negotiation/test_generate_lora.py.
    """

    def test_requires_the_group_or_an_adapter_dir(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        from entente.negotiation.generate import LocalLoraProvider

        monkeypatch.delenv("ENTENTE_DIALOGUE_LORA_DIR", raising=False)
        try:
            import peft  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401

            installed = True
        except ImportError:
            installed = False
        expected = "ENTENTE_DIALOGUE_LORA_DIR" if installed else "finetune"
        with pytest.raises(RuntimeError, match=expected):
            LocalLoraProvider()

    def test_missing_adapter_directory_raises(self, tmp_path: object) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from entente.negotiation.generate import LocalLoraProvider

        with pytest.raises(RuntimeError, match="no adapter found"):
            LocalLoraProvider(str(tmp_path))

    def test_env_var_fallback_also_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        pytest.importorskip("transformers")
        pytest.importorskip("peft")
        from entente.negotiation.generate import LocalLoraProvider

        monkeypatch.setenv("ENTENTE_DIALOGUE_LORA_DIR", "no/such/directory")
        with pytest.raises(RuntimeError, match="no adapter found"):
            LocalLoraProvider()
