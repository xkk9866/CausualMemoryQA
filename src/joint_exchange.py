"""Co-adaptive sufficiency exchange.

The recent window is the memory a strategy model can justify without looking
at the label. Whether an earlier turn is worth exchanging into that window
depends on the strategy model itself, so the two are trained from the same
comparison: an exchange is a positive example only when it lowers the current
model's own next-strategy loss. The scorer reads that model's contextual
states and, at test time, performs at most one exchange. The label is used
only as training supervision.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modeling import Probe, class_weights
from src.train_eval import N_STRAT, pack_metrics, set_seed


class ExchangeNet(nn.Module):
    def __init__(self, in_dim: int = 768, n_strat: int = 8, d_model: int = 256, beta: float = 0.35):
        super().__init__()
        self.dec = Probe(n_strat=n_strat, in_dim=in_dim, d_model=d_model)
        self.score = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)
        self.beta = beta

    def turn_scores(self, emb, role, pos, valid):
        """Score every turn from the decision model's own contextual states.

        Hidden states are detached: the comparison loss trains the scorer,
        while the decision loss trains the reader. The two still co-adapt
        because the scorer's input is the reader's current representation.
        """
        h, _pad = self.dec.trunk(emb, role, pos, valid)
        turn_h = h[:, 1:].detach()
        window = _window(valid, 4)
        ctx = (turn_h * window.unsqueeze(-1)).sum(1) / window.sum(1).clamp(min=1).unsqueeze(-1)
        feat = torch.cat([turn_h, ctx[:, None, :].expand_as(turn_h), turn_h - ctx[:, None, :]], -1)
        raw = self.score(feat).squeeze(-1)
        length = emb.shape[1]
        dist = (valid.sum(1).long()[:, None] - 1 - torch.arange(length, device=emb.device)).clamp(min=0).float()
        return (raw - self.beta * torch.log1p(dist)).masked_fill(valid < 0.5, -1e4)


def _window(valid: torch.Tensor, k: int) -> torch.Tensor:
    length = valid.shape[1]
    hlen = valid.sum(1).long()
    pos = torch.arange(length, device=valid.device)[None, :]
    return ((pos >= (hlen[:, None] - k).clamp(min=0)) & (valid > 0.5)).float()


def exchange_keep(model, emb, role, pos, valid, k: int, tau: float = 0.0, explore: float = 0.0):
    """Last-k, with the oldest slot replaced when an earlier turn scores higher."""
    window = _window(valid, k)
    scores = model.turn_scores(emb, role, pos, valid)
    outside = (valid > 0.5) & (window < 0.5)
    pick = scores
    if explore > 0:
        u = torch.rand_like(scores).clamp(1e-6, 1 - 1e-6)
        pick = scores + explore * (-torch.log(-torch.log(u)))
    choice = pick.masked_fill(~outside, -1e4).argmax(1)
    rows = torch.arange(emb.shape[0], device=emb.device)
    oldest = window.argmax(1)
    can = outside.any(1) & (choice < oldest)
    take = can & (scores[rows, choice] > scores[rows, oldest] + tau)
    out = window.clone()
    out[rows[take], oldest[take]] = 0
    out[rows[take], choice[take]] = 1
    return out, scores, choice, oldest, can


def _loss_parts(model, emb, role, pos, mask, y, resp, weight):
    logits, query = model.dec(emb, role, pos, mask)
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    cos = (query * resp).sum(-1)
    return ce - 0.15 * cos, logits, query


def confidence_masks(model, store, idx, k, batch_size, margin, device):
    """One exchange, chosen without the label.

    Each earlier turn is tried in the oldest window slot. The exchange that
    most raises the decision model's confidence is kept when the gain exceeds
    `margin`.
    """
    masks = np.zeros((len(idx), store.max_l), dtype=bool)
    model.eval()
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=device))
        emb, role, pos, valid = view["emb"], view["role"], view["pos"], view["valid"]
        bsz, length, _ = emb.shape
        window = _window(valid, k)
        rows = torch.arange(bsz, device=device)
        oldest = window.argmax(1)
        hlen = valid.sum(1).long()
        start_w = (hlen - k).clamp(min=0)
        with torch.inference_mode():
            logits_w, _q = model.dec(emb, role, pos, window)
            conf_w = logits_w.softmax(-1).amax(-1)
            best_conf = conf_w.clone()
            best = window.clone()
            max_j = int(start_w.max().item()) if bsz else 0
            for j in range(max_j):
                pos_j = start_w - 1 - j
                ok = pos_j >= 0
                if not bool(ok.any()):
                    continue
                trial = window.clone()
                trial[rows[ok], oldest[ok]] = 0
                trial[rows[ok], pos_j[ok]] = 1
                logits, _q = model.dec(emb, role, pos, trial)
                conf = logits.softmax(-1).amax(-1)
                take = ok & (conf > best_conf)
                best_conf = torch.where(take, conf, best_conf)
                best = torch.where(take[:, None], trial, best)
            keep = best_conf > conf_w + margin
            chosen = torch.where(keep[:, None], best, window)
        masks[start:start + len(batch)] = chosen.cpu().numpy() > 0.5
    return masks


def collect_masks(model, store, idx, k, batch_size, tau, device):
    masks = np.zeros((len(idx), store.max_l), dtype=bool)
    model.eval()
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=device))
        with torch.inference_mode():
            keep, _s, _c, _o, _can = exchange_keep(
                model, view["emb"], view["role"], view["pos"], view["valid"], k, tau=tau
            )
        masks[start:start + len(batch)] = keep.cpu().numpy() > 0.5
    return masks


def train_joint(store, train_idx, dev_idx, epochs_warm: int, epochs_joint: int, batch_size: int,
                lr: float, seed: int, device, k: int = 4):
    set_seed(seed)
    model = ExchangeNet(in_dim=store.emb.shape[1]).to(device)
    weight = class_weights(store.y[train_idx].detach().cpu(), N_STRAT).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    best, best_state = -1.0, None

    def run_epoch(epoch, update_scorer: bool):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        total, wins, seen, deltas = 0.0, 0, 0, []
        n_cand = 4
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            emb, role, pos = view["emb"], view["role"], view["pos"]
            valid, y, resp = view["valid"], view["y"], view["resp"]
            bsz, length, dim = emb.shape
            window = _window(valid, k)
            scores = model.turn_scores(emb, role, pos, valid)
            outside = (valid > 0.5) & (window < 0.5)
            noise = torch.rand(bsz, length, device=device).clamp(1e-6, 1 - 1e-6)
            gumbel = -torch.log(-torch.log(noise))
            cands = gumbel.masked_fill(~outside, -1e4).topk(n_cand, dim=1).indices
            cand_ok = torch.gather(outside, 1, cands)
            oldest = window.argmax(1)
            rows = torch.arange(bsz, device=device)
            masks = window[:, None, :].expand(bsz, n_cand, length).clone()
            masks[rows[:, None], torch.arange(n_cand, device=device)[None, :], oldest[:, None]] = 0
            masks[rows[:, None], torch.arange(n_cand, device=device)[None, :], cands] = cand_ok.float()
            # Invalid candidates keep the window with the oldest turn removed and
            # nothing put back; their loss is ignored via cand_ok.
            flat = masks.reshape(bsz * n_cand, length)
            loss_w, _, query_w = _loss_parts(model, emb, role, pos, window, y, resp, weight)
            with torch.no_grad():
                loss_c, _, _q = _loss_parts(
                    model,
                    emb[:, None].expand(bsz, n_cand, length, dim).reshape(bsz * n_cand, length, dim),
                    role[:, None].expand(bsz, n_cand, length).reshape(bsz * n_cand, length),
                    pos[:, None].expand(bsz, n_cand, length).reshape(bsz * n_cand, length),
                    flat,
                    y[:, None].expand(bsz, n_cand).reshape(bsz * n_cand),
                    resp[:, None].expand(bsz, n_cand, resp.shape[-1]).reshape(bsz * n_cand, resp.shape[-1]),
                    weight,
                )
                loss_c = loss_c.view(bsz, n_cand).masked_fill(~cand_ok, 1e4)
            best_j = loss_c.argmin(1)
            best_loss = loss_c[rows, best_j]
            has_out = outside.any(1)
            winner = has_out & cand_ok.any(1) & (best_loss + 0.05 < loss_w.detach())
            nce = F.cross_entropy(query_w @ resp.T / 0.07, torch.arange(bsz, device=device))
            loss = loss_w.mean() + 0.5 * nce
            if winner.any():
                chosen = masks[rows, best_j]
                loss_star, _, _ = _loss_parts(model, emb, role, pos, chosen, y, resp, weight)
                loss = loss + loss_star[winner].mean()
            if update_scorer:
                s_cand = torch.gather(scores, 1, cands)
                s_old = scores[rows, oldest]
                # The surviving candidate must outrank the turn it replaces.
                # When no candidate survives, every candidate stays below it.
                logit = s_cand - s_old[:, None]
                target = torch.zeros_like(logit)
                target[rows[winner], best_j[winner]] = 1.0
                pref = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
                pref = pref.masked_fill(~cand_ok, 0.0)
                denom = cand_ok.float().sum().clamp(min=1)
                loss = loss + pref.sum() / denom
                wins += int(winner.sum().item())
                deltas.append(float((loss_w.detach() - best_loss.detach())[has_out].mean().item()) if has_out.any() else 0.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        delta = float(np.mean(deltas)) if deltas else 0.0
        return total / max(seen, 1), wins / max(seen, 1), delta

    def evaluate(idx):
        from src.train_eval import run_fixed
        model.eval()
        masks = collect_masks(model, store, idx, k, batch_size, tau=0.0, device=device)
        last = np.zeros_like(masks)
        hl = store.cache["hlen"][idx]
        for i, n in enumerate(hl.tolist()):
            n = int(n)
            kk = min(k, n)
            last[i, max(0, n - kk):n] = True
        y = store.cache["y"][idx]
        pred_m, rank_m = run_fixed(model.dec, store, idx, masks, batch_size)
        pred_l, rank_l = run_fixed(model.dec, store, idx, last, batch_size)
        swapped = float((masks != last).any(1).mean())
        return (
            pack_metrics(y, pred_m, rank_m, masks.sum(1), hl),
            pack_metrics(y, pred_l, rank_l, last.sum(1), hl),
            swapped,
        )

    for epoch in range(epochs_warm):
        loss, _win, _delta = run_epoch(epoch, update_scorer=False)
        ours, base, rate = evaluate(dev_idx)
        print(f"warm {epoch+1}/{epochs_warm} loss {loss:.3f} "
              f"dev ours {ours['macro_f1']*100:.2f} last {base['macro_f1']*100:.2f} "
              f"MRR {ours['mrr']*100:.2f}/{base['mrr']*100:.2f} swap {rate*100:.1f}", flush=True)
    ours, _base, _rate = evaluate(dev_idx)
    best = ours["macro_f1"]
    best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    for epoch in range(epochs_joint):
        loss, win, delta = run_epoch(100 + epoch, update_scorer=True)
        ours, base, rate = evaluate(dev_idx)
        print(f"joint {epoch+1}/{epochs_joint} loss {loss:.3f} swap-win {win*100:.1f} "
              f"delta {delta:.4f} dev ours {ours['macro_f1']*100:.2f} last {base['macro_f1']*100:.2f} "
              f"MRR {ours['mrr']*100:.2f}/{base['mrr']*100:.2f} swap {rate*100:.1f}", flush=True)
        if ours["macro_f1"] > best:
            best = ours["macro_f1"]
            best_state = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model
