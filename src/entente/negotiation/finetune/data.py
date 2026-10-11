"""Tokenize instruction examples into the tensors the fine-tune trains on.

The input is :mod:`entente.negotiation.data.instructions` and the output is a
torch ``Dataset``.

Standard instruction-tuning masking: each example is ``prompt`` tokens
followed by ``completion`` tokens (plus an EOS), with the prompt span masked
to ``-100`` in ``labels`` so the loss only ever scores completion tokens --
the model is trained to *write* the message, not to reproduce the instruction
that elicited it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from transformers import PreTrainedTokenizerBase

    from entente.negotiation.data.instructions import InstructionExample

IGNORE_INDEX = -100
"""``torch.nn.CrossEntropyLoss``'s default ``ignore_index`` -- prompt tokens
carry this label so they never contribute to the loss."""


def encode_example(
    example: InstructionExample,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_seq_length: int,
) -> dict[str, torch.Tensor]:
    """Tokenize one pair, masking the prompt span out of the loss.

    When the pair exceeds ``max_seq_length`` the truncation takes from the left
    of the *prompt*, so the completion -- the only span the loss scores -- is
    never cut to make room for its own instruction.
    """
    prompt_ids: list[int] = tokenizer(example.prompt, add_special_tokens=False)["input_ids"]
    eos = tokenizer.eos_token or ""
    completion_ids: list[int] = tokenizer(example.completion + eos, add_special_tokens=False)[
        "input_ids"
    ]
    completion_ids = completion_ids[-max_seq_length:]
    budget = max(max_seq_length - len(completion_ids), 0)
    prompt_ids = prompt_ids[-budget:] if budget else []
    input_ids = prompt_ids + completion_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + completion_ids
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class PromptCompletionDataset(Dataset[dict[str, torch.Tensor]]):
    """Serve pre-tokenized instruction examples.

    Tokenization is eager: at roughly 17k short messages the whole corpus fits
    in memory pre-tokenized, so no step of training waits on a tokenizer.
    """

    def __init__(
        self,
        examples: Iterable[InstructionExample],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_seq_length: int,
    ) -> None:
        self._encoded = [
            encode_example(example, tokenizer, max_seq_length=max_seq_length)
            for example in examples
        ]
        if not self._encoded:
            raise ValueError("no instruction examples to train on")

    def __len__(self) -> int:
        return len(self._encoded)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._encoded[index]


def collate(
    batch: Sequence[dict[str, torch.Tensor]], *, pad_token_id: int
) -> dict[str, torch.Tensor]:
    """Right-pad a batch to its longest sequence.

    ``labels`` pad with :data:`IGNORE_INDEX` so padding scores nothing, and
    ``attention_mask`` marks the positions that are real.
    """
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for row, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[row, :n] = item["input_ids"]
        labels[row, :n] = item["labels"]
        attention_mask[row, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


__all__ = ["IGNORE_INDEX", "PromptCompletionDataset", "collate", "encode_example"]
