"""Point at the one earlier turn the decision model is missing.

On the training dialogues only, each earlier turn is swapped into the oldest
slot of the recent window and scored by the frozen decision model. That
comparison uses the training label. A pointer is then trained to reproduce
the resulting distribution from the dialogue alone. At test time it either
keeps the window or performs the single exchange it ranks first.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.causal import forward_metrics
from src.joint_exchange import _window
from src.modeling import Trunk
from src.train_eval import set_seed


def utility_table(probe, store, idx, k=4, batch_size=64):
    """Utility of the window and of every single exchange. Train labels only."""
    length = store.max_l
    table = np.full((len(idx), length + 1), -1e4, dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        emb, role, pos, valid = view["emb"], view["role"], view["pos"], view["valid"]
        bsz = emb.shape[0]
        window = _window(valid, k)
        rows = torch.arange(bsz, device=store.device)
        oldest = window.argmax(1)
        hlen = valid.sum(1).long()
        start_w = (hlen - k).clamp(min=0)
        with torch.inference_mode():
            base = forward_metrics(probe, emb, role, pos, window, view["y"], view["resp"], view["cand"])
            table[start:start + bsz, length] = base["u"].float().cpu().numpy()
            max_j = int(start_w.max().item()) if bsz else 0
            for j in range(max_j):
                pos_j = start_w - 1 - j
                ok = pos_j >= 0
                if not bool(ok.any()):
                    continue
                trial = window.clone()
                trial[rows[ok], oldest[ok]] = 0
                trial[rows[ok], pos_j[ok]] = 1
                alt = forward_metrics(probe, emb, role, pos, trial, view["y"], view["resp"], view["cand"])
                wrote = np.zeros(bsz, dtype=bool)
                wrote[ok.cpu().numpy()] = True
                # Write only the legal early positions.
                vals = alt["u"].float().cpu().numpy()
                for row in np.where(wrote)[0]:
                    table[start + row, int(pos_j[row].item())] = vals[row]
    return table


def load_table(path, probe, store, idx, k=4):
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return np.load(path)["table"]
    table = utility_table(probe, store, idx, k=k)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, table=table, idx=np.asarray(idx))
    return table


class Pointer(nn.Module):
    def __init__(self, in_dim: int = 768, d_model: int = 256):
        super().__init__()
        self.trunk = Trunk(in_dim=in_dim, d_model=d_model)
        self.token = nn.Linear(d_model, 1)
        self.stay = nn.Linear(d_model, 1)
        nn.init.zeros_(self.token.weight)
        nn.init.zeros_(self.token.bias)
        nn.init.zeros_(self.stay.weight)
        nn.init.zeros_(self.stay.bias)

    def logits(self, emb, role, pos, valid):
        h, _pad = self.trunk(emb, role, pos, valid)
        token = self.token(h[:, 1:]).squeeze(-1)
        stay = self.stay(h[:, 0]).squeeze(-1)
        return token.masked_fill(valid < 0.5, -1e4), stay


def _window_np(hlen, k, length):
    pos = np.arange(length)[None, :]
    return (pos >= (hlen[:, None] - k)) & (pos < hlen[:, None])


def masks_from_scores(token, stay, hlen, k, tau):
    """token [N,L], stay [N]. Swap only when the best early turn clears `tau`."""
    length = token.shape[1]
    mask = _window_np(hlen, k, length)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        if n <= k:
            continue
        early = token[i, : n - k]
        j = int(np.argmax(early))
        if early[j] > stay[i] + tau:
            mask[i, n - k] = False
            mask[i, j] = True
    return mask


def train_pointer(store, train_idx, dev_idx, table_tr, table_dv, epochs, batch_size, lr, seed, device, k=4):
    set_seed(seed)
    model = Pointer(in_dim=store.emb.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    length = store.max_l
    hlen_tr = store.cache["hlen"][train_idx]
    best_state, best_f1, best_tau = None, -1.0, 0.0

    def epoch(ep):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(len(train_idx))
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            batch = train_idx[rows]
            view = store.view(torch.tensor(batch, device=device))
            token, stay = model.logits(view["emb"], view["role"], view["pos"], view["valid"])
            values = torch.tensor(table_tr[rows], device=device)
            # Last column is the window. Early columns are exchanges.
            # Positions inside the window are already -1e4 in the table.
            target = F.softmax(values / 0.1, dim=-1)
            logits = torch.cat([token, stay[:, None]], dim=-1)
            loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(rows)
            seen += len(rows)
        return total / max(seen, 1)

    @torch.no_grad()
    def score_split(idx):
        model.eval()
        tokens, stays = [], []
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            token, stay = model.logits(view["emb"], view["role"], view["pos"], view["valid"])
            tokens.append(token.float().cpu().numpy())
            stays.append(stay.float().cpu().numpy())
        return np.concatenate(tokens), np.concatenate(stays)

    return model, epoch, score_split, (best_state, best_f1, best_tau, hlen_tr, length)
