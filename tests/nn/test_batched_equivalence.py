"""Prove the batched policy-head paths equal the per-board loops they restructure.

The batched paths are what makes a full-size run finish, and nothing else
checks that they compute the same thing, so this file is the whole guarantee
(``ml`` group).

`teacher_forced_batched` must reproduce the summed `teacher_forced` loss, its
gradients, and the top-1/slot counts; `decode_batched` must pick identical
orders to per-board `decode`. Randomized ragged batches (variable slot counts,
variable candidate-set sizes, empty boards, forced chunking) over both head
modes. Everything runs in eval mode on CPU fp32 -- dropout off, one dtype -- so
"equal" means allclose up to batched-reduction rounding.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from entente.nn.heads import PolicyHead

pytestmark = pytest.mark.ml

D_MODEL = 24
VOCAB = 60


def _head(independent: bool, seed: int = 0) -> PolicyHead:
    torch.manual_seed(seed)
    head = PolicyHead(
        d_model=D_MODEL, vocab_size=VOCAB, independent_heads=independent, dropout=0.0
    ).eval()
    return head


def _ragged_batch(
    seed: int, boards: int
) -> tuple[list[torch.Tensor], list[list[list[int]]], list[list[int]]]:
    """Build a random ragged batch of boards.

    Each board has 0-6 slots and 2-9 distinct candidates per slot, with the
    label inside the candidate set. Board 1, when present, is empty on purpose,
    since a board contributing no slots is the case batching gets wrong.
    """
    generator = torch.Generator().manual_seed(seed)
    encs: list[torch.Tensor] = []
    legal: list[list[list[int]]] = []
    labels: list[list[int]] = []
    for b in range(boards):
        slots = 0 if b == 1 else int(torch.randint(1, 7, (), generator=generator))
        encs.append(torch.randn(slots, D_MODEL, generator=generator))
        board_legal: list[list[int]] = []
        board_labels: list[int] = []
        for _ in range(slots):
            count = int(torch.randint(2, 10, (), generator=generator))
            cands = torch.randperm(VOCAB, generator=generator)[:count].tolist()
            board_legal.append(cands)
            pick = int(torch.randint(0, count, (), generator=generator))
            board_labels.append(cands[pick])
        legal.append(board_legal)
        labels.append(board_labels)
    return encs, legal, labels


def _loop_reference(
    head: PolicyHead,
    encs: list[torch.Tensor],
    legal: list[list[list[int]]],
    labels: list[list[int]],
) -> tuple[torch.Tensor, int, int]:
    total = encs[0].new_zeros(())
    correct = slots = 0
    for enc, board_legal, board_labels in zip(encs, legal, labels, strict=True):
        if not board_labels:
            continue
        loss, right = head.teacher_forced(enc, board_legal, board_labels)
        total = total + loss
        correct += right
        slots += len(board_labels)
    return total, correct, slots


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_teacher_forced_batched_matches_loop_loss_and_counts(independent: bool, seed: int) -> None:
    head = _head(independent, seed=1)
    encs, legal, labels = _ragged_batch(seed, boards=6)
    loop_total, loop_correct, loop_slots = _loop_reference(head, encs, legal, labels)
    batched_total, batched_correct, batched_slots = head.teacher_forced_batched(encs, legal, labels)
    assert torch.allclose(batched_total, loop_total, atol=1e-5)
    assert batched_correct == loop_correct
    assert batched_slots == loop_slots


def test_teacher_forced_batched_matches_loop_gradients() -> None:
    head = _head(independent=False, seed=2)
    encs, legal, labels = _ragged_batch(11, boards=5)

    loop_total, _, _ = _loop_reference(head, encs, legal, labels)
    loop_total.backward()
    loop_grads = {
        name: param.grad.clone()
        for name, param in head.named_parameters()
        if param.grad is not None
    }
    head.zero_grad()

    batched_total, _, _ = head.teacher_forced_batched(encs, legal, labels)
    batched_total.backward()
    for name, param in head.named_parameters():
        if name not in loop_grads:
            continue
        assert param.grad is not None
        assert torch.allclose(param.grad, loop_grads[name], atol=1e-5), name


def test_teacher_forced_batched_chunking_changes_nothing() -> None:
    head = _head(independent=False, seed=3)
    encs, legal, labels = _ragged_batch(13, boards=6)
    whole = head.teacher_forced_batched(encs, legal, labels)
    chunked = head.teacher_forced_batched(encs, legal, labels, logit_chunk=3)
    assert torch.allclose(whole[0], chunked[0], atol=1e-6)
    assert whole[1:] == chunked[1:]


def test_teacher_forced_batched_all_empty_boards() -> None:
    head = _head(independent=False)
    encs = [torch.randn(0, D_MODEL), torch.randn(0, D_MODEL)]
    total, correct, slots = head.teacher_forced_batched(encs, [[], []], [[], []])
    assert float(total) == 0.0 and correct == 0 and slots == 0


@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize("seed", [1, 23])
def test_decode_batched_matches_per_board_decode(independent: bool, seed: int) -> None:
    head = _head(independent, seed=4)
    encs, legal, _ = _ragged_batch(seed, boards=6)
    batched = head.decode_batched(encs, legal)
    for enc, board_legal, picks in zip(encs, legal, batched, strict=True):
        expected = head.decode(enc, board_legal) if board_legal else []
        assert picks == expected
        assert all(
            pick in cands for pick, cands in zip(picks, board_legal, strict=True)
        )  # legality by construction survives batching
