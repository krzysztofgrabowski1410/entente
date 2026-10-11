"""Encode the board as location tokens with a learned adjacency bias on attention.

The encoder is a transformer of ~6-8 layers at d_model 192-256, under 10M
parameters. A graph convolution over the board (DipNet's approach) is the
obvious alternative; this design keeps the graph as a prior instead.

The adjacency prior is injected as a *learned* additive term on the pre-softmax
attention logits: a parameter of shape ``(n_relations, n_heads)`` indexed by the
relation category between every token pair (:func:`entente.nn.features.adjacency_relations`
-- self / same-province / movement-adjacent / none) gives a per-head scalar bias.
So the model starts with no structural assumption (biases init to zero) and
*learns* how much each head should attend along the board graph -- cheaper and
more flexible than hard-wiring a GCN, and it keeps attention full-rank (every
token can still attend anywhere; the bias only tilts it).
"""

from __future__ import annotations

import math

import torch
from torch import nn


class RelationalSelfAttention(nn.Module):
    """Attend over the tokens with a learned per-head bias for each pair's relation.

    The bias is added to the pre-softmax logits, so a head can learn how much
    to follow the board graph without any pair being cut off entirely.
    """

    def __init__(self, d_model: int, n_heads: int, n_relations: int, dropout: float) -> None:
        super().__init__()
        if d_model % n_heads != 0:  # pragma: no cover - config validation guards this
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.rel_bias = nn.Parameter(torch.zeros(n_relations, n_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """Attend ``x`` against itself, tilted by the pairwise relations in ``rel``."""
        # x: (B, L, D); rel: (L, L) long relation categories
        batch, length, _ = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, L, d_head)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)  # (B, H, L, L)
        bias = self.rel_bias[rel].permute(2, 0, 1)  # (H, L, L)
        attn = torch.softmax(scores + bias.unsqueeze(0), dim=-1)
        attn = self.dropout(attn)
        context = (attn @ v).transpose(1, 2).reshape(batch, length, -1)  # (B, L, D)
        return self.out(context)


class EncoderLayer(nn.Module):
    """Pre-norm transformer block: relational attention + position-wise FFN."""

    def __init__(
        self, d_model: int, n_heads: int, ff_mult: int, n_relations: int, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelationalSelfAttention(d_model, n_heads, n_relations, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """Run the attention and feed-forward sublayers, each around a residual."""
        x = x + self.dropout(self.attn(self.norm1(x), rel))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class BoardEncoder(nn.Module):
    """Encode a ``(B, L, F)`` board-feature tensor into ``(B, L, d_model)`` per location.

    A learned location embedding is added to the projected features, and every
    layer carries the relational attention bias.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        n_locations: int,
        n_relations: int,
        d_model: int = 192,
        n_layers: int = 6,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.loc_embedding = nn.Parameter(torch.zeros(n_locations, d_model))
        nn.init.normal_(self.loc_embedding, std=0.02)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, ff_mult, n_relations, dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, feats: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        """Encode one batch of board features into per-location representations."""
        x = self.input_proj(feats) + self.loc_embedding.unsqueeze(0)
        for layer in self.layers:
            x = layer(x, rel)
        return self.norm(x)
