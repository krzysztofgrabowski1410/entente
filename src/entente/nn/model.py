"""Assemble the encoder and heads into one model and checkpoint it safely.

:class:`BCModel` wires the board encoder to the policy and value heads over a
fixed (map, ruleset) coordinate system -- the location tokens, adjacency relations
and order vocabulary are all derived from the map, so a checkpoint is only
meaningful against the same map/ruleset. :meth:`BCModel.save` records the
geometry and a vocabulary fingerprint; :meth:`BCModel.load` rebuilds the network
and refuses any checkpoint whose fingerprint differs from the vocabulary it is
being loaded against, because its head indices would then point at different
orders and the model would play plausible nonsense instead of failing.

Geometry is passed as plain kwargs, not a config object, so this package stays
independent of :mod:`entente.training` (training translates its ``ModelConfig``
into these kwargs).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from entente.nn.encoder import BoardEncoder
from entente.nn.features import N_RELATIONS, adjacency_relations, board_vocab, feature_dim
from entente.nn.heads import PolicyHead, ValueHead
from entente.nn.order_vocab import build_vocabulary

_CHECKPOINT_FORMAT = "entente-bc-1"


def _vocab_fingerprint(entries: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


class BCModel(nn.Module):
    """Board encoder + autoregressive policy head + 7-power value head."""

    def __init__(
        self,
        *,
        map_name: str = "standard",
        ruleset_name: str = "standard",
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
        independent_heads: bool = False,
    ) -> None:
        super().__init__()
        self.map_name = map_name
        self.ruleset_name = ruleset_name
        self.geometry: dict[str, Any] = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "ff_mult": ff_mult,
            "dropout": dropout,
            "independent_heads": independent_heads,
        }
        self.vocab = build_vocabulary(map_name, ruleset_name)
        self.board = board_vocab(map_name)
        # relation matrix is pure map data -- rebuilt on load, not persisted.
        self.register_buffer("rel", adjacency_relations(map_name), persistent=False)
        self.encoder = BoardEncoder(
            feature_dim=feature_dim(map_name),
            n_locations=self.board.n_locations,
            n_relations=N_RELATIONS,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            dropout=dropout,
        )
        self.policy = PolicyHead(
            d_model=d_model,
            vocab_size=len(self.vocab),
            independent_heads=independent_heads,
            dropout=dropout,
        )
        self.value = ValueHead(d_model=d_model, n_powers=self.board.n_powers, dropout=dropout)

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        """Encode a board-feature batch ``(B, L, F)`` into ``(B, L, d_model)``."""
        return self.encoder(feats, self.rel)

    def num_parameters(self) -> int:
        """Count every parameter in the network, which the budget is stated against."""
        return sum(p.numel() for p in self.parameters())

    def save(self, path: str | Path) -> None:
        """Write a checkpoint holding everything :meth:`load` needs to rebuild the model.

        That is the format tag, the map and ruleset names, the geometry, a
        fingerprint of the order vocabulary, and the weights.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": _CHECKPOINT_FORMAT,
                "map_name": self.map_name,
                "ruleset_name": self.ruleset_name,
                "geometry": self.geometry,
                "vocab_size": len(self.vocab),
                "vocab_sha256": _vocab_fingerprint(self.vocab.entries),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, map_location: str | torch.device = "cpu") -> BCModel:
        """Rebuild a model from a checkpoint, refusing one whose vocabulary differs.

        A vocabulary that does not fingerprint the same would misalign the policy
        head's indices. The returned module lives on ``map_location``.
        """
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        if ckpt.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"not an {_CHECKPOINT_FORMAT} checkpoint: {path}")
        model = cls(
            map_name=ckpt["map_name"], ruleset_name=ckpt["ruleset_name"], **ckpt["geometry"]
        )
        if _vocab_fingerprint(model.vocab.entries) != ckpt["vocab_sha256"]:
            raise ValueError(
                "checkpoint vocabulary fingerprint does not match the current "
                f"{model.map_name}/{model.ruleset_name} vocabulary -- head indices would misalign"
            )
        model.load_state_dict(ckpt["state_dict"])
        # the MODULE must land where the caller asked, not just the checkpoint
        # tensors: load_state_dict copies values into the freshly built CPU
        # module regardless of map_location, and every call site treats
        # map_location as "the device", so a module left on the CPU crashes the
        # first CUDA forward pass after the load.
        return model.to(map_location)
