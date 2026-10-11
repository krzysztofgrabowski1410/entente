"""Pin decode_scored: its log-prob accounting, nucleus filtering and delegation.

The score must always be the temperature-1 policy's, and decoding must agree
exactly with the plain decode path (``ml`` group).
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("torch")
import torch

from entente.engine.game import Game
from entente.nn.features import _unit_token, encode_game
from entente.nn.heads import _nucleus
from entente.nn.model import BCModel
from entente.nn.order_vocab import legal_order_indices

pytestmark = pytest.mark.ml


@pytest.fixture(scope="module")
def setup() -> tuple[BCModel, torch.Tensor, list[list[int]]]:
    torch.manual_seed(0)
    model = BCModel(d_model=64, n_layers=2, n_heads=4).eval()
    game = Game.new("standard", "standard")
    enc = model.encode(encode_game(game).unsqueeze(0))[0]
    units = sorted(
        (u for u in game.state.units if u.power == "FRANCE"), key=lambda u: str(u.location)
    )
    groups = legal_order_indices(game, "FRANCE", model.vocab)
    legal_ids = [groups[str(u.location)] for u in units]
    slots = enc[[model.board.location_index(_unit_token(u)) for u in units]]
    return model, slots, legal_ids


def _replay_log_prob(
    model: BCModel, slots: torch.Tensor, legal_ids: list[list[int]], chosen: list[int]
) -> float:
    """Recompute a sequence's log-probability independently of the decoder.

    The chosen ids are teacher-forced through ``_decoder_states`` and
    ``candidate_logits``, and the temperature-1 log-softmax terms summed.
    """
    states = model.policy._decoder_states(slots, chosen)
    total = 0.0
    for i, candidates in enumerate(legal_ids):
        logits = model.policy.candidate_logits(states[i], candidates)
        total += float(torch.log_softmax(logits, dim=-1)[candidates.index(chosen[i])].item())
    return total


class TestDecodeScored:
    """Decoding with a score, and what that score is measured under."""

    def test_greedy_log_prob_matches_replay(self, setup) -> None:
        model, slots, legal_ids = setup
        chosen, log_prob = model.policy.decode_scored(slots, legal_ids)
        assert len(chosen) == len(legal_ids)
        expected = _replay_log_prob(model, slots, legal_ids, chosen)
        assert log_prob == pytest.approx(expected, abs=1e-4)
        assert log_prob < 0.0

    def test_sampled_score_is_temperature_one_even_when_tempered(self, setup) -> None:
        """``temperature`` and ``top_p`` shape the draw, never the returned score.

        The score must stay the untempered policy's log-probability of what was
        drawn, because piKL consumes it as an anchor score (docs/pikl.tex Sec. 5).
        """
        model, slots, legal_ids = setup
        generator = torch.Generator().manual_seed(7)
        chosen, log_prob = model.policy.decode_scored(
            slots, legal_ids, sample=True, temperature=0.31, top_p=0.9, generator=generator
        )
        expected = _replay_log_prob(model, slots, legal_ids, chosen)
        assert log_prob == pytest.approx(expected, abs=1e-4)
        for pick, candidates in zip(chosen, legal_ids, strict=True):
            assert pick in candidates

    def test_decode_delegates_identically(self, setup) -> None:
        model, slots, legal_ids = setup
        scored, _ = model.policy.decode_scored(slots, legal_ids)
        assert model.policy.decode(slots, legal_ids) == scored
        a = model.policy.decode(
            slots, legal_ids, sample=True, generator=torch.Generator().manual_seed(3)
        )
        b, _ = model.policy.decode_scored(
            slots, legal_ids, sample=True, generator=torch.Generator().manual_seed(3)
        )
        assert a == b  # same generator stream, same draws -- no behavior change

    def test_rejects_bad_sampling_params(self, setup) -> None:
        model, slots, legal_ids = setup
        with pytest.raises(ValueError, match="temperature"):
            model.policy.decode_scored(slots, legal_ids, temperature=0.0)
        with pytest.raises(ValueError, match="top_p"):
            model.policy.decode_scored(slots, legal_ids, top_p=0.0)

    def test_non_finite_logits_fall_back_to_argmax_not_a_crash(self, setup) -> None:
        """Non-finite logits must not reach torch.multinomial as a NaN distribution.

        On CUDA that is an illegal memory access, which kills the process rather
        than the draw. Two candidates' embeddings are poisoned to +inf so their
        logits collide to NaN, and sampled decode must still return legal picks
        and a finite score.
        """
        model, slots, legal_ids = setup
        poisoned = [c for cands in legal_ids for c in cands][:2]
        with torch.no_grad():
            for idx in poisoned:
                model.policy.order_embedding.weight[idx] = float("inf")
        try:
            chosen, log_prob = model.policy.decode_scored(
                slots, legal_ids, sample=True, top_p=0.9, generator=torch.Generator().manual_seed(1)
            )
            assert all(p in c for p, c in zip(chosen, legal_ids, strict=True))  # still legal
            assert math.isfinite(log_prob)  # ranking score stays usable
        finally:  # module-scoped fixture: undo the poison for other tests
            with torch.no_grad():
                for idx in poisoned:
                    model.policy.order_embedding.weight[idx] = 0.0


class TestNucleus:
    """Nucleus filtering of a candidate distribution."""

    def test_keeps_smallest_mass_reaching_top_p(self) -> None:
        probs = torch.tensor([0.5, 0.3, 0.15, 0.05])
        filtered = _nucleus(probs, 0.8)
        # smallest prefix reaching 0.8 is {0.5, 0.3} (cumsum hits 0.8 exactly);
        # 0.15 and 0.05 are cut, survivors renormalize over 0.8.
        assert filtered[2] == 0.0 and filtered[3] == 0.0
        assert filtered[:2] == pytest.approx(torch.tensor([0.625, 0.375]))

    def test_top_choice_always_survives(self) -> None:
        probs = torch.tensor([0.9, 0.1])
        filtered = _nucleus(probs, 0.01)
        assert filtered == pytest.approx(torch.tensor([1.0, 0.0]))

    def test_degenerate_input_collapses_to_top1_not_nan(self) -> None:
        """A non-finite or all-zero probability vector collapses to the top-1 one-hot.

        Dividing by a zero mass would return NaN, and a NaN distribution reaching
        multinomial is a CUDA illegal memory access; the one-hot is a valid
        distribution to fall back on.
        """
        nan = _nucleus(torch.tensor([float("nan"), float("nan"), float("nan")]), 0.9)
        assert torch.isfinite(nan).all() and float(nan.sum()) == pytest.approx(1.0)
        zeros = _nucleus(torch.zeros(4), 0.9)
        assert torch.isfinite(zeros).all() and float(zeros.sum()) == pytest.approx(1.0)
