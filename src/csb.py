"""Counterfactual Sufficiency Bottleneck (CSB).

Memory selection is posed as learning a budgeted sufficient statistic of the
dialogue history for the supporter's next decision. A gate g scores turns, a
differentiable k-subset bottleneck keeps k of them, and a shared frozen decision
model f must recover, from the kept turns only, the decision it makes from the
full history. The same requirement is imposed in counterfactual environments
(emotion-lexeme substitution, turn order permutation, distractor insertion).
A turn set that stays sufficient in every environment cannot rely on what the
environments change, which is the invariance route to causal parents.

    L = sum_e [ KL(f(H) || f(z_e * H_e)) + a * CE(y, f(z_e * H_e))
                + b * (1 - cos(q(z_e * H_e), r)) ]
        + c * Var_e[ sufficiency loss ]        (risk extrapolation)
        + d * gate disagreement across environments
        + w * score of the inserted distractor
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.causal import Store, forward_metrics
from src.modeling import Probe, Selector, class_weights
from src.train_eval import N_STRAT, pack_metrics, run_fixed, set_seed

RW = float(__import__("os").environ.get("CSB_RW", "1.0"))


def subset_keep(valid: torch.Tensor, gen: torch.Generator | None = None) -> torch.Tensor:
    """Mixture of full, random, recency, and random-plus-recency subsets."""
    bsz, length = valid.shape
    dev = valid.device
    u = torch.rand(bsz, 1, device=dev)
    rate = torch.rand(bsz, 1, device=dev) * 0.85 + 0.1
    rand_keep = (torch.rand(bsz, length, device=dev) < rate).float() * valid
    n = valid.sum(1, keepdim=True)
    m = torch.randint(1, 9, (bsz, 1), device=dev).float()
    pos = torch.arange(length, device=dev)[None, :].float()
    recent = ((pos >= n - m) & (pos < n)).float() * valid
    mixed = torch.clamp(recent + (torch.rand(bsz, length, device=dev) < 0.15).float() * valid, max=1.0)
    keep = torch.where(u < 0.3, valid, torch.where(u < 0.6, rand_keep, torch.where(u < 0.8, recent, mixed)))
    empty = (keep.sum(1) == 0) & (valid.sum(1) > 0)
    if empty.any():
        last = (valid.sum(1) - 1).clamp(min=0).long()
        keep = keep.clone()
        keep[empty, last[empty]] = 1.0
    return keep


def train_robust_probe(store: Store, train_idx, dev_idx, epochs: int, batch_size: int, lr: float,
                       seed: int, device) -> Probe:
    """Shared decision model that is well defined on history subsets."""
    set_seed(seed)
    probe = Probe(n_strat=N_STRAT, in_dim=store.emb.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    weights = class_weights(store.y[train_idx].detach().cpu(), N_STRAT).to(device)
    hl = store.cache["hlen"][dev_idx]
    y = store.cache["y"][dev_idx]
    full = np.arange(store.max_l)[None, :] < hl[:, None]
    rng = np.random.default_rng(7)
    half = full & (rng.random(full.shape) < 0.5)
    for i, n in enumerate(hl.tolist()):
        half[i, int(n) - 1] = True
    last4 = full & (np.arange(store.max_l)[None, :] >= (hl[:, None] - 4))
    best, best_state = -1e9, None
    for epoch in range(epochs):
        probe.train()
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            keep = subset_keep(view["valid"])
            logits, query = probe(view["emb"], view["role"], view["pos"], keep)
            loss = F.cross_entropy(logits, view["y"], weight=weights, label_smoothing=0.05)
            sim = query @ view["resp"].T / 0.07
            loss = loss + 0.5 * F.cross_entropy(sim, torch.arange(len(batch), device=device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
        sched.step()
        scores = {}
        for name, mask in (("full", full), ("half", half), ("last4", last4)):
            pred, rank = run_fixed(probe, store, dev_idx, mask, batch_size)
            scores[name] = pack_metrics(y, pred, rank, mask.sum(1), hl)
        key = np.mean([scores[n]["macro_f1"] + 0.15 * scores[n]["mrr"] for n in scores])
        print(
            f"robust probe epoch {epoch+1}/{epochs} "
            + " ".join(f"{n} F1 {s['macro_f1']*100:.2f} MRR {s['mrr']*100:.2f}" for n, s in scores.items())
        )
        if key > best:
            best = key
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    probe.to(device).eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    return probe


def relaxed_topk(scores: torch.Tensor, valid: torch.Tensor, k: torch.Tensor, temp: float,
                 noise: bool) -> torch.Tensor:
    """Relaxed k-subset sample (successive softmax, Xie & Ermon 2019)."""
    s = scores.masked_fill(valid < 0.5, -1e4)
    if noise:
        g = -torch.log(-torch.log(torch.rand_like(s).clamp(1e-6, 1 - 1e-6)))
        s = s + 0.25 * g * (valid > 0.5).float()
    kmax = int(k.max().item())
    soft = torch.zeros_like(s)
    a = s / temp
    for j in range(kmax):
        p = torch.softmax(a, dim=-1)
        active = (j < k).float()[:, None]
        soft = soft + p * active
        a = a + torch.log((1.0 - p).clamp(min=1e-6))
    soft = soft.clamp(max=1.0) * valid
    order = s.argsort(dim=-1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(s.shape[1], device=s.device)[None, :].expand_as(order))
    hard = ((ranks < k[:, None]) & (valid > 0.5)).float()
    return hard + soft - soft.detach()


class PriorGate(torch.nn.Module):
    """Gate score = recency base measure + learned counterfactual residual.

    With a zero residual the ranking is exactly LastK, so learning can only move
    away from recency where sufficiency in the environments demands it.
    """

    def __init__(self, in_dim: int = 768, beta: float = 1.0):
        super().__init__()
        self.sel = Selector(in_dim=in_dim)
        self.beta = beta
        torch.nn.init.zeros_(self.sel.gate[-1].weight)
        torch.nn.init.zeros_(self.sel.gate[-1].bias)

    @property
    def trunk(self):
        return self.sel.trunk

    def forward(self, emb, role, pos, keep):
        hlen = keep.sum(1, keepdim=True)
        prior = -torch.log1p((hlen - 1 - pos.float()).clamp(min=0))
        return self.sel(emb, role, pos, keep) + self.beta * prior


def _decide(probe, emb, role, pos, z, vmask):
    hard = (z.detach() > 0.5).float()
    return probe(emb * z.unsqueeze(-1), role, pos, hard)


def _shuffle_env(view, rng: np.random.Generator):
    emb, role = view["emb"].clone(), view["role"].clone()
    bsz, length = view["valid"].shape
    perm_all = torch.arange(length, device=emb.device)[None, :].repeat(bsz, 1)
    for b in range(bsz):
        n = int(view["hlen"][b])
        if n > 1:
            perm = torch.tensor(rng.permutation(n), device=emb.device)
            perm_all[b, :n] = perm
            emb[b, :n] = view["emb"][b, perm]
            role[b, :n] = view["role"][b, perm]
    return emb, role, perm_all


def _insert_env(store: Store, view, batch):
    bsz, length, dim = view["emb"].shape
    dev = view["emb"].device
    emb = torch.zeros(bsz, length + 1, dim, device=dev)
    role = torch.zeros(bsz, length + 1, dtype=torch.long, device=dev)
    valid = torch.zeros(bsz, length + 1, device=dev)
    src = torch.full((bsz, length + 1), -1, dtype=torch.long, device=dev)
    did = store.ins_donor[torch.tensor(batch, device=dev)].clamp(min=0)
    for b in range(bsz):
        n = int(view["hlen"][b])
        at = max(0, n - 1)
        idx = torch.arange(n, device=dev)
        before, after = idx[:at], idx[at:]
        emb[b, :at] = view["emb"][b, before]
        role[b, :at] = view["role"][b, before]
        src[b, :at] = before
        emb[b, at] = store.emb[did[b]]
        role[b, at] = store.role_all[did[b]]
        src[b, at] = -2
        emb[b, at + 1:n + 1] = view["emb"][b, after]
        role[b, at + 1:n + 1] = view["role"][b, after]
        src[b, at + 1:n + 1] = after
        valid[b, :n + 1] = 1.0
    pos = torch.arange(length + 1, device=dev)[None, :].expand(bsz, -1)
    return emb, role, pos, valid, src


def exchange_masks(probe, store: Store, indices: np.ndarray, k: int, n_try: int = 12,
                  batch_size: int = 128) -> np.ndarray:
    """Budgeted exchange: start from the last k turns and replace the oldest
    window turn by an earlier turn when that raises the decision model's utility.
    Utility uses the training label, so the mask is supervision, not a test rule.
    """
    hl = store.cache["hlen"][indices]
    length = store.max_l
    window = (np.arange(length)[None, :] >= (hl[:, None] - k)) & (np.arange(length)[None, :] < hl[:, None])
    chosen = window.copy()

    def _utility(mask: np.ndarray) -> np.ndarray:
        parts = []
        for start in range(0, len(indices), batch_size):
            batch = indices[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=store.device))
            z = torch.tensor(mask[start:start + batch_size], device=store.device, dtype=torch.float32)
            with torch.inference_mode():
                met = forward_metrics(probe, view["emb"], view["role"], view["pos"], z,
                                      view["y"], view["resp"], view["cand"])
            parts.append(met["u"].float().cpu().numpy())
        return np.concatenate(parts)

    current = _utility(chosen)
    for t in range(n_try):
        trial = chosen.copy()
        active = np.zeros(len(indices), dtype=bool)
        for i, n in enumerate(hl.tolist()):
            n = int(n)
            if n <= k:
                continue
            oldest = n - k
            j = oldest - 1 - t
            if j < 0 or not chosen[i, oldest]:
                continue
            trial[i, oldest] = False
            trial[i, j] = True
            active[i] = True
        if not active.any():
            break
        alt = _utility(trial)
        take = active & (alt > current + 1e-4)
        chosen[take] = trial[take]
        current[take] = alt[take]
    return chosen


def train_exchange(probe, store: Store, train_idx, dev_idx, target: np.ndarray, epochs: int,
                   batch_size: int, lr: float, seed: int, device, use_env: bool = True,
                   use_prior: bool = True, name: str = "csb", eval_k: int = 4):
    """Learn the budgeted exchange. The recency window is the base measure; the
    residual scorer only has to promote the earlier turn the exchange inserts
    and demote the window turn it evicts. The same ranking is required after
    emotion substitution and after a position permutation.
    """
    set_seed(seed)
    gate = (PriorGate(store.emb.shape[1], beta=0.15) if use_prior else Selector(in_dim=store.emb.shape[1])).to(device)
    gate.trunk.load_state_dict(probe.trunk.state_dict())
    opt = torch.optim.AdamW(gate.parameters(), lr=lr, weight_decay=0.01)
    rng = np.random.default_rng(seed + 5)
    hl_dev = store.cache["hlen"][dev_idx]
    y_dev = store.cache["y"][dev_idx]
    best, best_state = -1e9, {k2: v.detach().cpu().clone() for k2, v in gate.state_dict().items()}

    def rank_loss(scores, tgt, valid):
        # Every kept turn must outrank every rejected turn, so top-k recovers the set.
        diff = scores[:, None, :] - scores[:, :, None]
        pair = (tgt[:, :, None] > 0.5) & (tgt[:, None, :] < 0.5) & (valid[:, None, :] > 0.5)
        return (F.softplus(diff) * pair).sum() / pair.sum().clamp(min=1)

    for epoch in range(epochs):
        gate.train()
        order = rng.permutation(train_idx)
        tot = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            tgt = torch.tensor(target[batch], device=device, dtype=torch.float32)
            valid = view["valid"]
            scores = gate(view["emb"], view["role"], view["pos"], valid)
            loss = rank_loss(scores, tgt, valid)
            pos_ix = torch.arange(valid.shape[1], device=device)[None, :]
            in_window = (pos_ix >= (view["hlen"] - eval_k).clamp(min=0)[:, None]) & (valid > 0.5)
            inserted = (tgt > 0.5) & ~in_window
            evicted = (tgt < 0.5) & in_window
            has = inserted.any(1) & evicted.any(1)
            if has.any():
                s_in = scores.masked_fill(~inserted, -1e4).max(1).values
                s_out = scores.masked_fill(~evicted, 1e4).min(1).values
                loss = loss + 8.0 * (F.softplus(s_out - s_in)[has]).mean()
            if use_env:
                loss = loss + rank_loss(gate(view["emb_emo"], view["role"], view["pos"], valid), tgt, valid)
                emb_s, role_s, perm = _shuffle_env(view, rng)
                tgt_s = torch.zeros_like(tgt)
                tgt_s.scatter_(1, perm, tgt)
                loss = loss + rank_loss(gate(emb_s, role_s, view["pos"], valid), tgt_s, valid)
                loss = loss / 3.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(batch)
        masks = gate_topk_masks(gate, store, dev_idx, eval_k, batch_size)
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        met = pack_metrics(y_dev, pred, rank, masks.sum(1), hl_dev)
        overlap = (masks & target[dev_idx]).sum(1) / np.maximum(target[dev_idx].sum(1), 1)
        key = met["macro_f1"] + 0.15 * met["mrr"]
        print(f"{name} epoch {epoch+1}/{epochs} loss {tot/len(order):.4f} "
              f"dev@{eval_k} F1 {met['macro_f1']*100:.2f} MRR {met['mrr']*100:.2f} "
              f"overlap {100*float(overlap.mean()):.1f}")
        if key > best:
            best = key
            best_state = {k2: v.detach().cpu().clone() for k2, v in gate.state_dict().items()}
    gate.load_state_dict(best_state)
    gate.to(device).eval()
    return gate


def train_csb(probe, store: Store, train_idx, dev_idx, epochs: int, batch_size: int, lr: float, seed: int,
              device, use_env: bool = True, use_rex: bool = True, use_kl: bool = True, k_max: int = 8,
              name: str = "csb", eval_k: int = 4, use_prior: bool = True) -> torch.nn.Module:
    set_seed(seed)
    gate = (PriorGate(store.emb.shape[1]) if use_prior else Selector(in_dim=store.emb.shape[1])).to(device)
    gate.trunk.load_state_dict(probe.trunk.state_dict())
    cw = class_weights(store.y[train_idx].detach().cpu(), N_STRAT).to(device)
    head = [p for n, p in gate.named_parameters() if n.endswith("gate.2.weight") or n.endswith("gate.2.bias")]
    head_ids = {id(p) for p in head}
    body = [p for p in gate.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": lr * 0.3}, {"params": head, "lr": lr}], weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed + 123)
    best, best_state = -1e9, None
    hl_dev = store.cache["hlen"][dev_idx]
    y_dev = store.cache["y"][dev_idx]

    def _dev_key(tag: str) -> float:
        nonlocal best, best_state
        masks = gate_topk_masks(gate, store, dev_idx, eval_k, batch_size)
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        met = pack_metrics(y_dev, pred, rank, masks.sum(1), hl_dev)
        key = met["macro_f1"] + 0.15 * met["mrr"]
        print(f"{name} {tag} dev@{eval_k} F1 {met['macro_f1']*100:.2f} MRR {met['mrr']*100:.2f}")
        if key > best:
            best = key
            best_state = {k2: v.detach().cpu().clone() for k2, v in gate.state_dict().items()}
        return key

    gate.eval()
    _dev_key("init")
    for epoch in range(epochs):
        gate.train()
        temp = max(0.2, 1.0 * (0.75 ** epoch))
        order = rng.permutation(train_idx)
        tot = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            valid = view["valid"]
            hlen = view["hlen"]
            kk = int(torch.randint(2, k_max + 1, (1,), device=device).item())
            k = torch.minimum(torch.full_like(hlen, kk), hlen)
            with torch.no_grad():
                t_logits, t_q = probe(view["emb"], view["role"], view["pos"], valid)
                t_logp = F.log_softmax(t_logits, -1)
                pos_ix = torch.arange(valid.shape[1], device=device)[None, :]
                z_last = ((pos_ix >= (hlen - k).clamp(min=0)[:, None]) & (valid > 0.5)).float()
                lo_l, q_l = probe(view["emb"] * z_last.unsqueeze(-1), view["role"], view["pos"], z_last)

            def per_of(logits, q):
                logp = F.log_softmax(logits, -1)
                per = 0.5 * F.nll_loss(logp, view["y"], weight=cw, reduction="none")
                if use_kl:
                    per = per + (t_logp.exp() * (t_logp - logp)).sum(-1)
                per = per + RW * (0.5 * (1 - (q * view["resp"]).sum(-1)) + 0.25 * (1 - (q * t_q).sum(-1)))
                return per

            floor = per_of(lo_l, q_l)

            def suff(emb, role, pos, vmask, scores, use_floor: bool):
                z = relaxed_topk(scores, vmask, k, temp, noise=True)
                logits, q = _decide(probe, emb, role, pos, z, vmask)
                per = per_of(logits, q)
                if use_floor:
                    per = per + F.relu(per - floor)
                return per.mean()

            s_clean = gate(view["emb"], view["role"], view["pos"], valid)
            losses = [suff(view["emb"], view["role"], view["pos"], valid, s_clean, True)]
            extra = torch.zeros((), device=device)
            if use_env:
                s_emo = gate(view["emb_emo"], view["role"], view["pos"], valid)
                losses.append(suff(view["emb_emo"], view["role"], view["pos"], valid, s_emo, False))
                emb_s, role_s, perm = _shuffle_env(view, rng)
                s_shuf = gate(emb_s, role_s, view["pos"], valid)
                losses.append(suff(emb_s, role_s, view["pos"], valid, s_shuf, False))
                emb_i, role_i, pos_i, valid_i, src = _insert_env(store, view, batch)
                s_ins = gate(emb_i, role_i, pos_i, valid_i)
                z = relaxed_topk(s_ins, valid_i, k, temp, noise=True)
                logits, q = _decide(probe, emb_i, role_i, pos_i, z, valid_i)
                logp = F.log_softmax(logits, -1)
                per = 0.5 * F.nll_loss(logp, view["y"], weight=cw, reduction="none")
                if use_kl:
                    per = per + (t_logp.exp() * (t_logp - logp)).sum(-1)
                per = per + RW * (0.5 * (1 - (q * view["resp"]).sum(-1)) + 0.25 * (1 - (q * t_q).sum(-1)))
                losses.append((per + F.relu(per - floor)).mean())
                donor = (src == -2).float()
                extra = extra + 0.2 * F.binary_cross_entropy_with_logits(
                    s_ins, torch.zeros_like(s_ins), weight=donor, reduction="sum"
                ) / donor.sum().clamp(min=1)

                def norm(s):
                    m = valid > 0.5
                    mu = (s * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
                    sd = (((s - mu) ** 2) * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
                    return (s - mu) / (sd + 1e-4).sqrt()

                ref = norm(s_clean)
                back = torch.zeros_like(s_shuf)
                back.scatter_(1, perm, s_shuf)
                cons = ((norm(s_emo) - ref) ** 2 + (norm(back) - ref) ** 2) * valid
                extra = extra + 0.05 * cons.sum() / valid.sum().clamp(min=1)
            stack = torch.stack(losses)
            loss = stack.mean() + extra
            if use_prior:
                last = gate.sel.gate[-1]
                loss = loss + 1e-3 * last.weight.pow(2).mean()
            if use_rex and len(losses) > 1:
                loss = loss + 1.0 * stack.var()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(batch)
        sched.step()
        print(f"{name} epoch {epoch+1}/{epochs} loss {tot/max(len(order),1):.4f} temp {temp:.2f}")
        _dev_key(f"epoch {epoch+1}/{epochs}")
    gate.load_state_dict(best_state)
    gate.to(device).eval()
    return gate


def gate_scores(gate, store: Store, idx, batch_size: int) -> np.ndarray:
    out = []
    for start in range(0, len(idx), batch_size):
        b = idx[start:start + batch_size]
        v = store.view(torch.tensor(b, device=store.device))
        with torch.inference_mode():
            s = gate(v["emb"], v["role"], v["pos"], v["valid"])
        out.append(s.masked_fill(v["valid"] < 0.5, -1e4).float().cpu().numpy())
    return np.concatenate(out)


def gate_topk_masks(gate, store: Store, idx, k: int, batch_size: int) -> np.ndarray:
    s = gate_scores(gate, store, idx, batch_size)
    hl = store.cache["hlen"][idx]
    masks = np.zeros_like(s, dtype=bool)
    for i, n in enumerate(hl.tolist()):
        kk = min(k, int(n))
        masks[i, np.argsort(-s[i, :int(n)])[:kk]] = True
    return masks


def calibrate_adaptive(gate, probe, store: Store, dev_idx, alpha: float, batch_size: int):
    """Smallest per-instance prefix of the gate ranking whose decision matches the
    full-history decision. A conformal quantile over dev gives a size rule
    k(H) = ceil(q * |H|) clipped to [1, |H|] with coverage >= 1 - alpha."""
    s = gate_scores(gate, store, dev_idx, batch_size)
    hl = store.cache["hlen"][dev_idx]
    full = np.arange(store.max_l)[None, :] < hl[:, None]
    ref_pred, _ = run_fixed(probe, store, dev_idx, full, batch_size)
    need = np.full(len(dev_idx), np.nan)
    order = np.argsort(-s, axis=1)
    for k in range(1, store.max_l + 1):
        open_ = np.isnan(need)
        if not open_.any():
            break
        masks = np.zeros_like(full)
        for i in np.where(open_)[0]:
            kk = min(k, int(hl[i]))
            masks[i, order[i, :kk]] = True
        sel = np.where(open_)[0]
        pred, _ = run_fixed(probe, store, dev_idx[sel], masks[sel], batch_size)
        hit = pred == ref_pred[sel]
        done = sel[hit | (k >= hl[sel])]
        need[done] = k
    frac = need / np.maximum(hl, 1)
    n = len(frac)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    q = float(np.quantile(frac, level, method="higher"))
    return q


def adaptive_masks(gate, store: Store, idx, q: float, batch_size: int, k_min: int = 1) -> np.ndarray:
    s = gate_scores(gate, store, idx, batch_size)
    hl = store.cache["hlen"][idx]
    masks = np.zeros_like(s, dtype=bool)
    for i, n in enumerate(hl.tolist()):
        n = int(n)
        kk = int(min(n, max(k_min, np.ceil(q * n))))
        masks[i, np.argsort(-s[i, :n])[:kk]] = True
    return masks
