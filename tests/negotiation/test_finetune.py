"""Pin the dialogue fine-tune's tokenization and its training loop.

Both run against the genuinely tiny ``hf-internal-testing/tiny-random-gpt2``
test model. ``finetune``-marked: needs the `finetune` group and network access
to fetch it once, cached afterwards. The torch-free config tests are in
tests/negotiation/test_finetune_config.py.

The committed smoke curve's improvement is small but real: an untrained
32-dim model has essentially no language prior to leverage, unlike the
purpose-built architecture the BC smoke test overfits.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from entente.negotiation.finetune.config import DialogueLoraConfig
from entente.training.config import load_config

pytestmark = pytest.mark.finetune

SMOKE_CONFIG = "src/entente/negotiation/finetune/configs/smoke.yaml"


@pytest.fixture(scope="module")
def tokenizer():  # type: ignore[no-untyped-def]
    import truststore

    truststore.inject_into_ssl()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class TestEncodeExample:
    """Tokenization of one instruction pair, including the loss mask."""

    def test_prompt_span_is_masked(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from entente.negotiation.data.instructions import InstructionExample
        from entente.negotiation.finetune.data import IGNORE_INDEX, encode_example

        example = InstructionExample(
            prompt="p p p", completion="c c", truthful=True, game_id=1, message_index=0
        )
        encoded = encode_example(example, tokenizer, max_seq_length=64)
        prompt_len = len(tokenizer("p p p", add_special_tokens=False)["input_ids"])
        assert list(encoded["labels"][:prompt_len]) == [IGNORE_INDEX] * prompt_len
        assert encoded["labels"][prompt_len] != IGNORE_INDEX
        assert len(encoded["input_ids"]) == len(encoded["labels"])

    def test_completion_ends_with_eos(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from entente.negotiation.data.instructions import InstructionExample
        from entente.negotiation.finetune.data import encode_example

        example = InstructionExample(
            prompt="hi", completion="bye", truthful=True, game_id=1, message_index=0
        )
        encoded = encode_example(example, tokenizer, max_seq_length=64)
        assert encoded["input_ids"][-1].item() == tokenizer.eos_token_id

    def test_truncates_to_max_seq_length(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from entente.negotiation.data.instructions import InstructionExample
        from entente.negotiation.finetune.data import encode_example

        example = InstructionExample(
            prompt="word " * 200,
            completion="reply " * 200,
            truthful=True,
            game_id=1,
            message_index=0,
        )
        encoded = encode_example(example, tokenizer, max_seq_length=32)
        assert len(encoded["input_ids"]) <= 32
        assert len(encoded["labels"]) <= 32


class TestCollate:
    """Padding a batch to its longest sequence."""

    def test_pads_to_the_longest_sequence(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        import torch

        from entente.negotiation.finetune.data import collate

        batch = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([-100, -100, 3])},
            {"input_ids": torch.tensor([1]), "labels": torch.tensor([-100])},
        ]
        out = collate(batch, pad_token_id=tokenizer.pad_token_id)
        assert out["input_ids"].shape == (2, 3)
        assert out["labels"][1].tolist() == [-100, -100, -100]
        assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
        assert out["input_ids"][1, 0].item() == 1
        assert out["input_ids"][1, 1].item() == tokenizer.pad_token_id


class TestPromptCompletionDataset:
    """The eagerly tokenized dataset the fine-tune iterates."""

    def test_len_and_getitem(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from entente.negotiation.data.instructions import InstructionExample
        from entente.negotiation.finetune.data import PromptCompletionDataset

        examples = [
            InstructionExample(
                prompt="a", completion="b", truthful=True, game_id=1, message_index=i
            )
            for i in range(3)
        ]
        dataset = PromptCompletionDataset(examples, tokenizer, max_seq_length=32)
        assert len(dataset) == 3
        assert set(dataset[0]) == {"input_ids", "labels"}

    def test_empty_examples_raises(self, tokenizer) -> None:  # type: ignore[no-untyped-def]
        from entente.negotiation.finetune.data import PromptCompletionDataset

        with pytest.raises(ValueError, match="no instruction examples"):
            PromptCompletionDataset([], tokenizer, max_seq_length=32)


class TestLoadBaseModel:
    """Loading a base model and its tokenizer, quantized or plain."""

    def test_load_in_4bit_requires_cuda(self) -> None:
        import torch

        from entente.negotiation.finetune.train import load_base_model

        cfg = load_config(DialogueLoraConfig, SMOKE_CONFIG, overrides=["load_in_4bit=true"])
        with pytest.raises(RuntimeError, match="CUDA"):
            load_base_model(cfg, torch.device("cpu"))

    def test_loads_the_tiny_model_and_a_pad_token(self) -> None:
        import torch

        from entente.negotiation.finetune.train import load_base_model

        cfg = load_config(DialogueLoraConfig, SMOKE_CONFIG)
        model, tok = load_base_model(cfg, torch.device("cpu"))
        assert tok.pad_token is not None
        assert sum(p.numel() for p in model.parameters()) < 1_000_000


class TestTrainSmoke:
    """A whole fine-tune run over a tiny model, adapter included."""

    def test_train_runs_and_produces_an_adapter(self, tmp_path: Path) -> None:
        from entente.negotiation.finetune.train import train

        cfg = load_config(
            DialogueLoraConfig,
            SMOKE_CONFIG,
            overrides=[f"out_dir={tmp_path / 'out'}", "optim.max_steps=6", "log_every=2"],
        )
        result = train(cfg)
        assert result.steps == 6
        assert (result.adapter_dir / "adapter_config.json").exists()
        assert result.metrics_path.exists()
        metrics = [json.loads(line) for line in result.metrics_path.read_text().splitlines()]
        assert [m["step"] for m in metrics] == [2, 4, 6]

    @pytest.mark.slow
    def test_full_smoke_config_loss_decreases(self, tmp_path: Path) -> None:
        """A fresh run reproduces the committed smoke curve's trend.

        The curve is in docs/results/dialogue_lora_smoke_curve.jsonl: a small
        but real decrease from a near-uniform start, not the dramatic drop the
        behavioral-cloning smoke test shows. An untrained 32-dim model has no
        language prior to leverage, so a small decrease is the honest bar.
        """
        from entente.negotiation.finetune.train import train

        cfg = load_config(
            DialogueLoraConfig, SMOKE_CONFIG, overrides=[f"out_dir={tmp_path / 'out'}"]
        )
        result = train(cfg)
        metrics = [
            json.loads(line) for line in result.metrics_path.read_text().splitlines() if line
        ]
        losses = [m["loss"] for m in metrics]
        assert len(losses) == 20
        assert losses[-1] < losses[0], f"loss did not fall: {losses[0]:.4f} -> {losses[-1]:.4f}"
