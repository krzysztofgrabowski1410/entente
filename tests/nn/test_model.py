"""Encoder + policy/value heads + assembly/checkpoint (``ml`` group)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import torch

from entente.engine.game import Game
from entente.engine.orders import Hold, format_order
from entente.nn.features import _unit_token, encode_game
from entente.nn.model import BCModel
from entente.nn.order_vocab import legal_order_indices

pytestmark = pytest.mark.ml


def _movement_example(model: BCModel, power: str = "FRANCE"):
    """Build a minimal teacher-forcing batch of one board.

    That is the slots, legal index lists and hold-defaulted labels for one
    power at the S1901M start position.
    """
    game = Game.new("standard", "standard")
    enc = model.encode(encode_game(game).unsqueeze(0))
    units = sorted((u for u in game.state.units if u.power == power), key=lambda u: str(u.location))
    groups = legal_order_indices(game, power, model.vocab)
    legal_ids = [groups[str(u.location)] for u in units]
    labels = [model.vocab.index(format_order(Hold(u.kind, u.location))) for u in units]
    loc_enc = enc[0, [model.board.location_index(_unit_token(u)) for u in units]]
    return loc_enc, legal_ids, labels


def test_base_geometry_is_under_ten_million_params() -> None:
    model = BCModel(d_model=192, n_layers=6, n_heads=8)
    params = model.num_parameters()
    assert 5_000_000 < params < 10_000_000, params


def test_load_puts_the_module_on_map_location(tmp_path) -> None:
    """Every call site treats map_location as "the device".

    The returned MODULE -- parameters and the rebuilt rel buffer, not just the
    checkpoint tensors -- must live there, or the first CUDA forward pass after
    a load crashes. Only the cpu case is assertable without a GPU, but it pins
    the contract.
    """
    model = BCModel(d_model=32, n_layers=1, n_heads=2)
    model.save(tmp_path / "m.pt")
    loaded = BCModel.load(tmp_path / "m.pt", map_location=torch.device("cpu"))
    devices = {p.device.type for p in loaded.parameters()}
    devices.add(loaded.rel.device.type)
    assert devices == {"cpu"}


def test_encode_and_value_shapes() -> None:
    model = BCModel(d_model=64, n_layers=2, n_heads=4).eval()
    game = Game.new("standard", "standard")
    enc = model.encode(encode_game(game).unsqueeze(0))
    assert enc.shape == (1, 81, 64)
    value = torch.softmax(model.value(enc), dim=-1)
    assert value.shape == (1, 7)
    assert torch.allclose(value.sum(), torch.tensor(1.0), atol=1e-5)


def test_policy_teacher_forcing_produces_gradient() -> None:
    model = BCModel(d_model=64, n_layers=2, n_heads=4)
    loc_enc, legal_ids, labels = _movement_example(model)
    loss, correct = model.policy.teacher_forced(loc_enc, legal_ids, labels)
    assert torch.isfinite(loss)
    assert 0 <= correct <= len(labels)
    loss.backward()
    assert model.policy.order_embedding.weight.grad is not None


def test_decode_only_returns_legal_orders() -> None:
    model = BCModel(d_model=64, n_layers=2, n_heads=4).eval()
    loc_enc, legal_ids, _ = _movement_example(model)
    chosen = model.policy.decode(loc_enc, legal_ids)
    assert len(chosen) == len(legal_ids)
    assert all(pick in candidates for pick, candidates in zip(chosen, legal_ids, strict=True))


def test_independent_heads_flag_has_no_recurrence() -> None:
    model = BCModel(d_model=64, n_layers=2, n_heads=4, independent_heads=True)
    assert not hasattr(model.policy, "gru")
    loc_enc, legal_ids, labels = _movement_example(model)
    loss, _ = model.policy.teacher_forced(loc_enc, legal_ids, labels)
    loss.backward()  # independent path is differentiable too
    assert torch.isfinite(loss)


def test_checkpoint_round_trip(tmp_path) -> None:
    model = BCModel(d_model=64, n_layers=2, n_heads=4).eval()
    game = Game.new("standard", "standard")
    feats = encode_game(game).unsqueeze(0)
    path = tmp_path / "bc.pt"
    model.save(path)
    restored = BCModel.load(path).eval()
    assert restored.num_parameters() == model.num_parameters()
    assert restored.geometry == model.geometry
    assert torch.allclose(model.encode(feats), restored.encode(feats), atol=1e-5)


def test_load_rejects_mismatched_vocabulary(tmp_path) -> None:
    model = BCModel(d_model=32, n_layers=1, n_heads=2)
    path = tmp_path / "bc.pt"
    model.save(path)
    blob = torch.load(path, weights_only=False)
    blob["vocab_sha256"] = "deadbeef"
    torch.save(blob, path)
    with pytest.raises(ValueError, match="vocabulary"):
        BCModel.load(path)


def test_load_rejects_wrong_format(tmp_path) -> None:
    path = tmp_path / "bad.pt"
    torch.save({"format": "something-else"}, path)
    with pytest.raises(ValueError, match="checkpoint"):
        BCModel.load(path)
