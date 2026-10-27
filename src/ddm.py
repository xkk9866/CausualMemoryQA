"""Decision-dominant memory.

A memory is not the smallest subset that copies the full-history decision.
Full history is a contaminated observation: recency and emotion wording move
the prediction without being the evidence the next support move depends on.

One network therefore minimises a single constrained risk. The gate is a
function of the whole dialogue, but the strategy and the retrieval query are
functions of the gated turns only:

    R(M) + λ |density(M) - ρ|
        + γ Σ_s max(0, R(M) - R(M_s))
        + β KL(p(y|M) || p(y|M, T(H)))

M_s are non-causal shortcuts (the most recent turns, and the turns most
similar to the latest one). T is a nuisance edit (emotion-lexeme substitution)
that should not change a decision made from the right evidence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionDominantMemory(nn.Module):
    def __init__(self, n_strat: int = 8, in_dim: int = 768, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 3, dropout: float = 0.1, max_pos: int = 128):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.in_proj = nn.Linear(in_dim, d_model)
        self.role_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))
        self.pool = nn.Linear(d_model, 1)
        self.strat = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, n_strat))
        self.query = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, in_dim))

    def _tokens(self, emb: torch.Tensor, role: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(emb)
        x = x + self.role_emb(role) + self.pos_emb(pos.clamp(max=self.pos_emb.num_embeddings - 1))
        cls = self.cls.expand(emb.shape[0], 1, -1)
        return torch.cat([cls, x], dim=1)

    def _attend(self, tokens: torch.Tensor, valid: torch.Tensor, key_bias: torch.Tensor | None) -> torch.Tensor:
        """valid [B, L] over turns. key_bias [B, L] is log(gate), added to keys.

        Padding is written into the same float mask. A boolean padding mask is
        not combined with it: that mix stalls on this PyTorch build.
        """
        bsz, seq, _ = tokens.shape
        if key_bias is None:
            bias = torch.zeros(bsz, seq, device=tokens.device, dtype=tokens.dtype)
        else:
            bias = torch.cat(
                [torch.zeros(bsz, 1, device=tokens.device, dtype=tokens.dtype), key_bias.to(tokens.dtype)],
                dim=1,
            )
        pad = torch.cat(
            [torch.zeros(bsz, 1, dtype=torch.bool, device=valid.device), ~valid],
            dim=1,
        )
        bias = bias.masked_fill(pad, -1e4)
        attn = bias[:, None, :].expand(bsz, seq, seq).contiguous()
        attn = attn.repeat_interleave(self.n_heads, dim=0)
        return self.encoder(tokens, mask=attn)

    def selection_logits(self, emb, role, pos, valid) -> torch.Tensor:
        tokens = self._tokens(emb, role, pos)
        # The selector may read every turn. Stop-grad so the decision cannot
        # hide the label inside this pass; only the second pass represents M.
        states = self._attend(tokens, valid, None).detach()
        logits = self.gate(states[:, 1:]).squeeze(-1)
        return logits.masked_fill(~valid, -20.0)

    def decide(self, emb, role, pos, valid, gate: torch.Tensor, return_weights: bool = False):
        """gate [B, L] in [0, 1]. Decision attends only in proportion to the gate."""
        tokens = self._tokens(emb, role, pos)
        log_g = torch.log(gate.clamp(min=1e-4))
        states = self._attend(tokens, valid, log_g)
        score = self.pool(states).squeeze(-1)
        score = score + torch.cat([torch.zeros(gate.shape[0], 1, device=gate.device, dtype=score.dtype), log_g], dim=1)
        pad = torch.cat(
            [torch.zeros(valid.shape[0], 1, dtype=torch.bool, device=valid.device), ~valid],
            dim=1,
        )
        score = score.masked_fill(pad, -1e4)
        weights = torch.softmax(score, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * states, dim=1)
        logits = self.strat(pooled)
        query = F.normalize(self.query(pooled), dim=-1)
        if return_weights:
            return logits, query, weights[:, 1:]
        return logits, query


def concrete_gate(logits: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
    logistic = torch.log(u) - torch.log(1.0 - u)
    return torch.sigmoid((logits + logistic) / tau)


def shortcut_masks(emb: torch.Tensor, valid: torch.Tensor, hlen: torch.Tensor, rho: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Recency mask and similarity-to-last mask, each with budget round(ρ n)."""
    bsz, length, _ = emb.shape
    device = emb.device
    last_index = (hlen - 1).clamp(min=0)
    last = emb[torch.arange(bsz, device=device), last_index]
    sim = torch.einsum("bld,bd->bl", emb, last).masked_fill(~valid, -1e4)
    recency = torch.arange(length, device=device)[None, :].expand(bsz, length)
    recency = recency.masked_fill(~valid, -1).float()
    budget = (hlen.float() * rho).round().clamp(min=1).long()
    rec_mask = torch.zeros(bsz, length, dtype=torch.bool, device=device)
    sim_mask = torch.zeros_like(rec_mask)
    # Budgets differ per row, so top-k uses the max and then drops extras.
    kmax = int(budget.max().item())
    rec_idx = recency.topk(kmax, dim=1).indices
    sim_idx = sim.topk(kmax, dim=1).indices
    rec_mask.scatter_(1, rec_idx, True)
    sim_mask.scatter_(1, sim_idx, True)
    rank = torch.arange(kmax, device=device)[None, :]
    rec_mask.scatter_(1, rec_idx, rank < budget[:, None])
    sim_mask.scatter_(1, sim_idx, rank < budget[:, None])
    return rec_mask & valid, sim_mask & valid
