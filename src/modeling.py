"""Set encoder, strategy/retrieval probe, and memory-selection gate."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Trunk(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 256, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1, max_pos: int = 128):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, d_model)
        self.role_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.d_model = d_model

    def forward(self, emb: torch.Tensor, role: torch.Tensor, pos: torch.Tensor,
                keep: torch.Tensor, bias: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """emb [B,L,D], role/pos long [B,L], keep float [B,L] in {0,1}.

        `bias` is an additive log-gate on attention keys; log(0) entries must
        already be large negative so a relaxed gate of 0 behaves as deletion.
        """
        bsz, length, _ = emb.shape
        x = self.in_proj(emb)
        x = x + self.role_emb(role) + self.pos_emb(pos.clamp(max=self.pos_emb.num_embeddings - 1))
        cls = self.cls.expand(bsz, 1, -1)
        x = torch.cat([cls, x], dim=1)
        pad = torch.cat(
            [
                torch.zeros(bsz, 1, dtype=torch.bool, device=emb.device),
                keep < 0.5,
            ],
            dim=1,
        )
        if bias is None:
            h = self.encoder(x, src_key_padding_mask=pad)
        else:
            key_bias = torch.cat(
                [torch.zeros(bsz, 1, device=emb.device, dtype=x.dtype), bias.to(x.dtype)], dim=1
            ).masked_fill(pad, -1e4)
            h = self.encoder(x, src_key_padding_mask=key_bias)
        return h, pad


class Probe(nn.Module):
    def __init__(self, n_strat: int = 8, in_dim: int = 768, d_model: int = 256):
        super().__init__()
        self.trunk = Trunk(in_dim=in_dim, d_model=d_model)
        self.pool = nn.Linear(d_model, 1)
        self.strat = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, n_strat),
        )
        self.query = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, in_dim),
        )

    def forward(self, emb, role, pos, keep, return_weights: bool = False, bias=None):
        h, pad = self.trunk(emb, role, pos, keep, bias=bias)
        weight_logits = self.pool(h).squeeze(-1)
        if bias is not None:
            weight_logits = weight_logits + torch.cat([torch.zeros_like(bias[:, :1]), bias], dim=1)
        weight_logits = weight_logits.masked_fill(pad, -1e4)
        weights = torch.softmax(weight_logits, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * h, dim=1)
        logits = self.strat(pooled)
        query = F.normalize(self.query(pooled), dim=-1)
        if return_weights:
            return logits, query, weights[:, 1:]
        return logits, query


class Selector(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 256):
        super().__init__()
        self.trunk = Trunk(in_dim=in_dim, d_model=d_model)
        self.gate = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.1),
            nn.Linear(d_model, 1),
        )

    def forward(self, emb, role, pos, keep):
        h, _pad = self.trunk(emb, role, pos, keep)
        return self.gate(h[:, 1:]).squeeze(-1)


def class_weights(labels: torch.Tensor, n_strat: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=n_strat).float().clamp(min=1.0)
    w = counts.sum() / (n_strat * counts)
    return w
