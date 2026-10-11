"""Turn a structured intent into an outgoing press message, behind a provider protocol.

The protocol is what lets the offline stub, a hosted model and a locally
fine-tuned adapter be substituted for one another without the rest of the
prototype knowing which is in use.

:class:`StubProvider` is deterministic and template-based, fully offline: it
is what tests and the committed demo transcript use, and its template is the
exact grammar :func:`entente.negotiation.parse.parse_message` parses back (see
that module's docstring -- the round trip is what keeps the offline demo
reproducible without an LLM). :class:`AnthropicProvider` is real-model
generation, gated on ``ANTHROPIC_API_KEY``; its one dependency (the
``anthropic`` package) lives in the ``llm`` dependency group (pyproject.toml)
so importing this module -- and using :class:`StubProvider` -- never requires
it. :class:`LocalLoraProvider` is real-model generation from a locally
fine-tuned LoRA adapter (:mod:`entente.negotiation.finetune`), gated on the
adapter directory existing; its dependencies (``torch``, ``transformers``,
``peft``) live in the ``finetune`` group, deferred to construction the same
way. All three satisfy :class:`Provider` structurally (no inheritance
needed).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from entente.engine.orders import format_order

if TYPE_CHECKING:
    from collections.abc import Sequence

    from entente.engine.orders import GameOrder
    from entente.negotiation.intents import Intent

DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
"""Default model id for :class:`AnthropicProvider`; any call may name another.

A caller overrides it with the ``ENTENTE_ANTHROPIC_MODEL`` environment variable
or the constructor's ``model`` argument. Generating one short press message is
cheap under any model, so the default favours quality; a caller on a tight
budget may prefer a smaller one.
"""


class Provider(Protocol):
    """Turn a structured proposal into an outgoing natural-language message."""

    def generate(self, intent: Intent) -> str:
        """Write the message this intent proposes."""
        ...


def _format_action(action: Sequence[GameOrder]) -> str:
    return ", ".join(format_order(order) for order in action)


class StubProvider:
    """Generate a message from a fixed template, deterministically and offline.

    The committed demo transcript and the whole test suite use only this
    provider, so neither needs a network or a key to reproduce. The template is
    intentionally rigid: it is the grammar :mod:`entente.negotiation.parse`
    reads back, and the round trip is what the demo rests on.
    """

    def generate(self, intent: Intent) -> str:
        """Fill the template from ``intent``, appending its rationale when it has one."""
        text = (
            f"PROPOSAL from {intent.proposer} to {intent.ally} for {intent.phase}: "
            f"{intent.proposer} orders [{_format_action(intent.proposer_action)}]; "
            f"{intent.ally} orders [{_format_action(intent.ally_action)}]."
        )
        if intent.rationale:
            text = f"{text} {intent.rationale}"
        return text


def _build_prompt(intent: Intent) -> str:
    """Pure prompt construction -- testable without the ``anthropic`` package."""
    parts = [
        "You are drafting a short in-character Diplomacy press message.",
        f"{intent.proposer} is proposing an arrangement to {intent.ally} for the "
        f"{intent.phase} phase: {intent.proposer} will play "
        f"{_format_action(intent.proposer_action)}, and is asking {intent.ally} to play "
        f"{_format_action(intent.ally_action)}.",
    ]
    if intent.rationale:
        parts.append(f"Reasoning to weave in: {intent.rationale}")
    parts.append(
        f"Write ONLY the message text (1-3 sentences, plausible in-game diplomatic "
        f"tone, signed as {intent.proposer}) -- no preamble, no commentary."
    )
    return " ".join(parts)


class _TextBlockLike(Protocol):
    type: str
    text: str


class _MessageLike(Protocol):
    content: Sequence[object]
    stop_reason: str | None


def _extract_text(response: _MessageLike) -> str:
    """Read the first text block off a Messages-API response.

    Structural (not ``anthropic.types.Message``) so this is unit-testable
    with a plain fake regardless of whether the ``anthropic`` package is
    installed (see ``tests/negotiation/test_generate.py``).
    """
    for block in response.content:
        if getattr(block, "type", None) == "text":
            return str(getattr(block, "text", ""))
    raise RuntimeError(
        f"AnthropicProvider: no text block in response (stop_reason={response.stop_reason!r})"
    )


class AnthropicProvider:
    """Real-model generation, gated on ``ANTHROPIC_API_KEY``.

    The ``anthropic`` import is deferred to construction so importing this
    module -- to use :class:`StubProvider`, say -- never requires the package.
    It lives in the ``llm`` dependency group, which the default install omits,
    keeping a heavy stack out of the core install.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 300,
    ) -> None:
        try:
            import anthropic
        except ImportError as error:
            raise RuntimeError(
                "AnthropicProvider needs the `anthropic` package: "
                "uv sync --group llm --system-certs"
            ) from error
        key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "AnthropicProvider requires ANTHROPIC_API_KEY (or an explicit api_key=)"
            )
        self._model = model or os.environ.get("ENTENTE_ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=key)

    @property
    def model(self) -> str:
        """Give the resolved model id.

        That is the constructor argument if one was given, else
        ``ENTENTE_ANTHROPIC_MODEL``, else :data:`DEFAULT_ANTHROPIC_MODEL`.
        """
        return self._model

    def generate(self, intent: Intent) -> str:
        """Ask the hosted model for the message ``intent`` proposes."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": _build_prompt(intent)}],
        )
        return _extract_text(response)


class LocalLoraProvider:
    """Generate from a locally fine-tuned LoRA adapter, if its directory exists.

    The adapter is the one :mod:`entente.negotiation.finetune` produces.

    ``torch``/``transformers``/``peft`` are deferred to construction, like
    ``AnthropicProvider`` defers ``anthropic`` -- the dependencies live in the
    ``finetune`` group, so importing this module for :class:`StubProvider`
    alone never requires them. The base model id is read from the adapter's
    own ``adapter_config.json`` (``peft``'s own
    ``base_model_name_or_path`` record) rather than asked of the caller -- an
    adapter directory produced by :func:`entente.negotiation.finetune.train.train`
    is a self-describing artifact. The prompt comes from :func:`_build_prompt`,
    so the fine-tuned adapter is prompted identically to
    :class:`AnthropicProvider` and only the model differs.

    Generation is greedy by default (``temperature=0.0``): reproducible demo
    transcripts matter more here than sampling diversity for a short press
    message; pass a positive ``temperature`` to sample instead.
    """

    def __init__(
        self,
        adapter_dir: str | None = None,
        *,
        device: str = "cpu",
        max_new_tokens: int = 120,
        temperature: float = 0.0,
    ) -> None:
        try:
            import torch
            import truststore
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "LocalLoraProvider needs the `finetune` group: "
                "uv sync --group finetune --system-certs"
            ) from error

        resolved_dir = adapter_dir or os.environ.get("ENTENTE_DIALOGUE_LORA_DIR")
        if not resolved_dir:
            raise RuntimeError(
                "LocalLoraProvider requires adapter_dir= (or ENTENTE_DIALOGUE_LORA_DIR)"
            )
        adapter_path = Path(resolved_dir)
        if not (adapter_path / "adapter_config.json").is_file():
            raise RuntimeError(f"LocalLoraProvider: no adapter found at {adapter_path}")

        truststore.inject_into_ssl()  # this machine's TLS interception; see finetune/train.py
        peft_config = PeftConfig.from_pretrained(adapter_path)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
        self._device = torch.device(device)
        self._model = PeftModel.from_pretrained(base_model, adapter_path).to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        self._adapter_dir = str(adapter_path)
        self._base_model = str(peft_config.base_model_name_or_path)
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature

    @property
    def adapter_dir(self) -> str:
        """Give the adapter directory this provider loaded."""
        return self._adapter_dir

    @property
    def base_model(self) -> str:
        """Give the base model id, read from the adapter's own config."""
        return self._base_model

    def generate(self, intent: Intent) -> str:
        """Ask the fine-tuned adapter for the message ``intent`` proposes."""
        import torch

        prompt = _build_prompt(intent)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        pad_id = self._tokenizer.pad_token_id or self._tokenizer.eos_token_id
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=self._temperature > 0.0,
                temperature=self._temperature if self._temperature > 0.0 else None,
                pad_token_id=pad_id,
            )
        completion = output_ids[0, inputs["input_ids"].shape[1] :]
        return str(self._tokenizer.decode(completion, skip_special_tokens=True)).strip()


__all__ = [
    "DEFAULT_ANTHROPIC_MODEL",
    "AnthropicProvider",
    "LocalLoraProvider",
    "Provider",
    "StubProvider",
]
