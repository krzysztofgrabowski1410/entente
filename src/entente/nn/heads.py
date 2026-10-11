"""Decode orders per unit and predict the final score share.

**Policy** -- an autoregressive per-unit decoder over the global order vocabulary
(:mod:`entente.nn.order_vocab`). Units are decoded in a canonical order; each
unit's order distribution is conditioned on its board-location encoding and, via
a GRU, on the orders already chosen this phase -- the coordination the
``independent_heads`` flag drops (the documented simplification: per-unit heads
that see only their own encoding). Output logits are the dot product of a unit's
decoder state with the shared order embedding (weight tying keeps the ~36k-way
head within the <10M-parameter budget), then **masked to exactly that unit's
legal orders** so the network can never place probability on an illegal move.

**Value** -- mean-pooled board encoding -> MLP -> a 7-power distribution over the
final Sum-of-Squares share, trained by cross-entropy against the realized shares.
That distribution is exactly what a piKL rollout consumes as its leaf value, so
the head predicts the scoring quantity itself rather than a scalar proxy.

:func:`legal_order_indices` is the masking primitive: it groups
``game.legal_orders`` by the ordering unit and maps each to its vocabulary index.
Because the vocabulary is a superset of every legal order, this reproduces the
engine's legal set exactly -- the property ``tests/nn/test_masking.py`` pins
against ``game.legal_orders`` on random states.
"""

from __future__ import annotations

import math

import torch
from torch import nn

# re-exported: the masking primitive lives with the vocabulary (torch-free) but
# is part of the policy head's public surface.
from entente.nn.order_vocab import legal_order_indices

__all__ = ["PolicyHead", "ValueHead", "legal_order_indices"]


def _nucleus(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Zero out everything past the smallest probability mass reaching ``top_p``.

    The highest-probability order always survives and the remainder is
    renormalized. A non-finite or all-zero input -- NaN probabilities from
    non-finite logits, say -- would divide by a zero mass and return NaN, a
    distribution ``torch.multinomial`` reads out of bounds on CUDA, so that case
    collapses to the top-1 one-hot instead. :func:`PolicyHead.decode_scored`
    guards the same numeric seam with its own argmax fallback.
    """
    sorted_probs, order = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative - sorted_probs < top_p
    filtered = torch.zeros_like(probs)
    kept_indices = order[keep]
    filtered[kept_indices] = probs[kept_indices]
    total = filtered.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        onehot = torch.zeros_like(probs)
        onehot[order[0]] = 1.0
        return onehot
    return filtered / total


class PolicyHead(nn.Module):
    """Autoregressive per-unit order decoder with engine-exact legality masking.

    Owns the order embedding (``vocab_size x d_model``); output logits over a set
    of candidate orders are ``decoder_state @ order_embedding[candidates].T`` --
    the embedding doubles as the tied output projection and as the autoregressive
    input embedding of the order decoded at the preceding slot.
    """

    def __init__(
        self, *, d_model: int, vocab_size: int, independent_heads: bool, dropout: float
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.independent_heads = independent_heads
        self.order_embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.order_embedding.weight, std=0.02)
        self.query = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout))
        if not independent_heads:
            self.gru = nn.GRUCell(d_model, d_model)
            self.start = nn.Parameter(torch.zeros(d_model))

    def _decoder_states(
        self, loc_encodings: torch.Tensor, prev_order_ids: list[int] | None
    ) -> torch.Tensor:
        """Per-slot decoder states ``(n_slots, d_model)`` for one board.

        ``loc_encodings`` is the encoder output gathered at each slot's location,
        in decode order. ``prev_order_ids`` (teacher forcing) are the true order
        chosen at each slot; state *i* is conditioned on slots ``< i`` through the
        GRU. ``None`` / ``independent_heads`` gives per-slot states with no
        cross-unit conditioning.
        """
        queries = self.query(loc_encodings)  # (n_slots, d)
        if self.independent_heads:
            return queries
        states = []
        hidden = torch.zeros(self.d_model, device=loc_encodings.device)
        prev_emb = self.start
        for i in range(queries.shape[0]):
            hidden = self.gru(queries[i] + prev_emb, hidden)
            states.append(hidden)
            if prev_order_ids is not None:
                prev_emb = self.order_embedding.weight[prev_order_ids[i]]
        return torch.stack(states)

    def candidate_logits(self, state: torch.Tensor, candidate_ids: list[int]) -> torch.Tensor:
        """Score one slot over its legal candidate orders.

        A masked softmax over these logits is the unit's order distribution.
        """
        return state @ self.order_embedding.weight[candidate_ids].T

    def teacher_forced(
        self,
        loc_encodings: torch.Tensor,
        legal_ids: list[list[int]],
        label_ids: list[int],
    ) -> tuple[torch.Tensor, int]:
        """Score one board's unit slots under teacher forcing.

        Returns the cross-entropy summed over the slots and the top-1-correct
        count. ``legal_ids[i]`` are slot *i*'s legal vocabulary indices, and
        ``label_ids[i]`` is the true order, which must be in ``legal_ids[i]``.
        """
        states = self._decoder_states(loc_encodings, label_ids)
        total = loc_encodings.new_zeros(())
        correct = 0
        for i, (candidates, label) in enumerate(zip(legal_ids, label_ids, strict=True)):
            logits = self.candidate_logits(states[i], candidates)
            target = candidates.index(label)
            total = total + nn.functional.cross_entropy(
                logits.unsqueeze(0), logits.new_tensor([target], dtype=torch.long)
            )
            correct += int(torch.argmax(logits).item() == target)
        return total, correct

    def teacher_forced_batched(
        self,
        loc_encodings: list[torch.Tensor],
        legal_ids: list[list[list[int]]],
        label_ids: list[list[int]],
        *,
        logit_chunk: int = 1024,
    ) -> tuple[torch.Tensor, int, int]:
        """Run :meth:`teacher_forced` over many boards at once.

        The math is the same -- loss summed over every slot, top-1-correct count,
        slot count -- restructured so one GRUCell call advances every board's slot
        *t* together. Under teacher forcing the whole autoregressive input
        sequence is known up front, so nothing is sequential across boards, and
        ``tests/nn/test_batched_equivalence.py`` pins per-board equivalence with
        the loop on loss, gradients and counts. Candidate logits run in padded
        ``logit_chunk``-slot chunks to bound the peak
        ``chunk x max_candidates x d_model`` gather. Boards with no slots
        contribute nothing.
        """
        boards = [i for i, labels in enumerate(label_ids) if labels]
        if not boards:
            anchor = loc_encodings[0] if loc_encodings else self.order_embedding.weight
            return anchor.new_zeros(()), 0, 0
        encs = [loc_encodings[i] for i in boards]
        labels = [label_ids[i] for i in boards]
        legals = [legal_ids[i] for i in boards]
        lengths = [len(lab) for lab in labels]
        batch, horizon = len(boards), max(lengths)
        device = encs[0].device

        padded = encs[0].new_zeros((batch, horizon, self.d_model))
        for b, enc in enumerate(encs):
            padded[b, : lengths[b]] = enc
        queries = self.query(padded)
        if self.independent_heads:
            states = queries
        else:
            prev = encs[0].new_zeros((batch, horizon, self.d_model))
            prev[:, 0] = self.start
            for b, lab in enumerate(labels):
                if len(lab) > 1:
                    ids = torch.tensor(lab[:-1], dtype=torch.long, device=device)
                    prev[b, 1 : lengths[b]] = self.order_embedding(ids)
            hidden = encs[0].new_zeros((batch, self.d_model))
            per_step = []
            for t in range(horizon):
                hidden = self.gru(queries[:, t] + prev[:, t], hidden)
                per_step.append(hidden)
            states = torch.stack(per_step, dim=1)

        flat_states = torch.cat([states[b, : lengths[b]] for b in range(batch)])
        flat_legals = [cands for board in legals for cands in board]
        flat_targets = [
            cands.index(label)
            for board_legals, board_labels in zip(legals, labels, strict=True)
            for cands, label in zip(board_legals, board_labels, strict=True)
        ]
        total = flat_states.new_zeros(())
        correct = 0
        for start in range(0, len(flat_legals), logit_chunk):
            chunk = flat_legals[start : start + logit_chunk]
            width = max(len(cands) for cands in chunk)
            cand = torch.tensor(
                [cands + [0] * (width - len(cands)) for cands in chunk],
                dtype=torch.long,
                device=device,
            )
            lens = torch.tensor([len(cands) for cands in chunk], device=device)
            mask = torch.arange(width, device=device).unsqueeze(0) < lens.unsqueeze(1)
            logits = torch.einsum(
                "nd,nwd->nw", flat_states[start : start + logit_chunk], self.order_embedding(cand)
            ).masked_fill(~mask, float("-inf"))
            targets = torch.tensor(
                flat_targets[start : start + logit_chunk], dtype=torch.long, device=device
            )
            total = total + nn.functional.cross_entropy(logits, targets, reduction="sum")
            correct += int((logits.argmax(dim=1) == targets).sum().item())
        return total, correct, len(flat_legals)

    @torch.no_grad()
    def decode_batched(
        self,
        loc_encodings: list[torch.Tensor],
        legal_ids: list[list[list[int]]],
    ) -> list[list[int]]:
        """Run argmax :meth:`decode` across many boards, lockstep over slot positions.

        Each board's GRU chain feeds its own preceding pick and the rows are
        independent, so batching changes no result;
        ``tests/nn/test_batched_equivalence.py`` pins that equivalence. Sampling
        stays on the per-board :meth:`decode` path, because interleaving the
        draws would reorder a shared generator's stream and a committed run must
        replay exactly from its seed.
        """
        lengths = [len(board) for board in legal_ids]
        out: list[list[int]] = [[] for _ in legal_ids]
        boards = [i for i, n in enumerate(lengths) if n]
        if not boards:
            return out
        batch, horizon = len(boards), max(lengths[i] for i in boards)
        device = loc_encodings[boards[0]].device
        padded = loc_encodings[boards[0]].new_zeros((batch, horizon, self.d_model))
        for row, i in enumerate(boards):
            padded[row, : lengths[i]] = loc_encodings[i]
        queries = self.query(padded)
        hidden = padded.new_zeros((batch, self.d_model))
        prev = (
            None
            if self.independent_heads
            else self.start.unsqueeze(0).expand(batch, self.d_model).contiguous()
        )
        for t in range(horizon):
            if self.independent_heads:
                state = queries[:, t]
            else:
                hidden = self.gru(queries[:, t] + prev, hidden)
                state = hidden
            active = [row for row, i in enumerate(boards) if t < lengths[i]]
            cands_t = [legal_ids[boards[row]][t] for row in active]
            width = max(len(cands) for cands in cands_t)
            cand = torch.tensor(
                [cands + [0] * (width - len(cands)) for cands in cands_t],
                dtype=torch.long,
                device=device,
            )
            lens = torch.tensor([len(cands) for cands in cands_t], device=device)
            mask = torch.arange(width, device=device).unsqueeze(0) < lens.unsqueeze(1)
            active_rows = torch.tensor(active, dtype=torch.long, device=device)
            logits = torch.einsum(
                "nd,nwd->nw", state[active_rows], self.order_embedding(cand)
            ).masked_fill(~mask, float("-inf"))
            picks = cand.gather(1, logits.argmax(dim=1, keepdim=True)).squeeze(1)
            for row, pick in zip(active, picks.tolist(), strict=True):
                out[boards[row]].append(int(pick))
            if prev is not None:
                prev = prev.clone()
                prev[active_rows] = self.order_embedding(picks)
        return out

    @torch.no_grad()
    def decode(
        self,
        loc_encodings: torch.Tensor,
        legal_ids: list[list[int]],
        *,
        sample: bool = False,
        generator: torch.Generator | None = None,
    ) -> list[int]:
        """Pick each slot's order from its legal set, one unit at a time.

        The choice feeds autoregressively into the next slot. ``sample`` draws
        from the masked softmax, which is what stochastic play needs for a
        tournament interval to have any spread; the default takes the
        highest-logit order. The returned vocabulary index per slot is legal by
        construction, because the candidate set it chooses from is the legal one.
        """
        return self.decode_scored(loc_encodings, legal_ids, sample=sample, generator=generator)[0]

    @torch.no_grad()
    def decode_scored(
        self,
        loc_encodings: torch.Tensor,
        legal_ids: list[list[int]],
        *,
        sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[list[int], float]:
        """Decode a sequence and score it under the **temperature-1** policy.

        piKL candidate sampling needs the untempered log-probability as its
        anchor score, so ``temperature`` and ``top_p`` shape only the *sampling*
        distribution and never the returned score (docs/pikl.tex Sec. 5).
        ``top_p`` keeps the smallest nucleus of orders whose probability mass
        reaches it (Holtzman et al. 2020), renormalized before sampling.
        """
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        queries = self.query(loc_encodings)
        chosen: list[int] = []
        log_prob = 0.0
        hidden = torch.zeros(self.d_model, device=loc_encodings.device)
        prev_emb = None if self.independent_heads else self.start
        for i, candidates in enumerate(legal_ids):
            if self.independent_heads:
                state = queries[i]
            else:
                hidden = self.gru(queries[i] + prev_emb, hidden)
                state = hidden
            logits = self.candidate_logits(state, candidates)
            if sample and bool(torch.isfinite(logits).all()):
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    probs = _nucleus(probs, top_p)
                # draw on the generator's device: callers seed plain CPU
                # generators (bc_vs_greedy, the sweep's candidate sampling) and
                # multinomial requires generator and probs to share a device --
                # the first CUDA tournament crashed here. Drawing the tiny
                # per-slot vector on CPU also makes same-seed picks
                # device-independent, and this loop syncs per slot via .item()
                # regardless.
                draw = probs if generator is None else probs.to(generator.device)
                local = int(torch.multinomial(draw, 1, generator=generator).item())
            else:
                # argmax is the safe fallback for non-sampling AND for the rare
                # non-finite-logit board state that would otherwise poison the
                # sampling distribution -- feeding NaN to multinomial is a CUDA
                # illegal memory access, which kills the process rather than the
                # draw. argmax over non-finite logits still returns a valid
                # in-range index; sampling degrades to the mode for that one slot.
                local = int(torch.argmax(logits).item())
            slot_lp = float(torch.log_softmax(logits, dim=-1)[local].item())
            log_prob += slot_lp if math.isfinite(slot_lp) else 0.0  # keep the anchor score finite
            pick = candidates[local]
            chosen.append(pick)
            if not self.independent_heads:
                prev_emb = self.order_embedding.weight[pick]
        return chosen, log_prob


class ValueHead(nn.Module):
    """Predict each power's share of the final Sum-of-Squares score.

    The board encoding is mean-pooled, passed through an MLP to one logit per
    power, and softmaxed into a distribution over the shares.
    """

    def __init__(self, *, d_model: int, n_powers: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_powers),
        )

    def forward(self, encoding: torch.Tensor) -> torch.Tensor:
        """Reduce a ``(B, L, d_model)`` encoding to ``(B, n_powers)`` share logits."""
        return self.mlp(encoding.mean(dim=1))
