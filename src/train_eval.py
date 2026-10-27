"""Train the probe and CauMem selector, then evaluate memory-selection methods."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.causal import Store, forward_metrics
from src.lexicon import emotion_tokens, jaccard, substitute_emotion
from src.modeling import Probe, Selector, class_weights

N_STRAT = 8
LABELS = list(range(N_STRAT))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(cache: dict, split: int) -> np.ndarray:
    return np.where(cache["split"] == split)[0]


def weighted_bce(logits, target, valid, pos_weight: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(target > 0.5, torch.full_like(loss, pos_weight), torch.ones_like(loss))
    denom = valid.sum().clamp(min=1.0)
    return (loss * weights * valid).sum() / denom


def _turn_dropout(valid: torch.Tensor, drop_p: float = 0.15, apply_p: float = 0.45) -> torch.Tensor:
    bsz, length = valid.shape
    apply = torch.rand(bsz, 1, device=valid.device) < apply_p
    keep = valid * (torch.rand(bsz, length, device=valid.device) >= drop_p).float()
    keep = torch.where(apply, keep, valid)
    empty = (keep.sum(dim=1) == 0) & (valid.sum(dim=1) > 0)
    if empty.any():
        score = torch.rand(bsz, length, device=valid.device).masked_fill(valid < 0.5, -1.0)
        choice = score.argmax(dim=1)
        keep = keep.clone()
        keep[empty, choice[empty]] = 1.0
    return keep


def run_fixed(probe, store: Store, indices: np.ndarray, masks: np.ndarray, batch_size: int = 128):
    """masks aligns with `indices` (not the global instance axis)."""
    probe.eval()
    preds, ranks = [], []
    device = store.device
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=device))
        keep = torch.tensor(masks[start:start + batch_size], device=device, dtype=torch.float32)
        with torch.inference_mode():
            met = forward_metrics(probe, view["emb"], view["role"], view["pos"], keep, view["y"], view["resp"], view["cand"])
        preds.append(met["pred"].cpu().numpy())
        ranks.append(met["rank"].cpu().numpy())
    return np.concatenate(preds), np.concatenate(ranks)


def pack_metrics(y, pred, rank, counts, hlens) -> dict:
    y = np.asarray(y)
    pred = np.asarray(pred)
    rank = np.asarray(rank).astype(np.float64).clip(min=1.0)
    hlens = np.maximum(np.asarray(hlens).astype(np.float64), 1.0)
    counts = np.asarray(counts).astype(np.float64)
    mmr = float(np.mean(counts / hlens))
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0)),
        "accuracy": float(np.mean(y == pred)) if len(y) else 0.0,
        "r1": float(np.mean(rank <= 1)) if len(y) else 0.0,
        "r5": float(np.mean(rank <= 5)) if len(y) else 0.0,
        "mrr": float(np.mean(1.0 / rank)) if len(y) else 0.0,
        "mmr": mmr,
        "hcr": float(1.0 - mmr),
        "n": int(len(y)),
    }


def per_class_f1(y, pred) -> dict:
    scores = f1_score(y, pred, average=None, labels=LABELS, zero_division=0)
    return {str(i): float(s) for i, s in enumerate(scores)}


def train_probe(store: Store, train_idx: np.ndarray, dev_idx: np.ndarray, epochs: int, batch_size: int,
                lr: float, seed: int, device: torch.device) -> Probe:
    set_seed(seed)
    probe = Probe(n_strat=N_STRAT, in_dim=store.emb.shape[1]).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)
    weights = class_weights(store.y[train_idx].detach().cpu(), N_STRAT).to(device)
    best_state, best_score, wait = None, -1e9, 0
    for epoch in range(epochs):
        probe.train()
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            keep = _turn_dropout(view["valid"])
            logits, query = probe(view["emb"], view["role"], view["pos"], keep)
            loss_ce = F.cross_entropy(logits, view["y"], weight=weights)
            sim = query @ view["resp"].T / 0.07
            loss_nce = F.cross_entropy(sim, torch.arange(len(batch), device=device))
            loss = loss_ce + 0.5 * loss_nce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        masks = np.ones((len(dev_idx), store.max_l), dtype=np.bool_)
        hl = store.cache["hlen"][dev_idx]
        for i, n in enumerate(hl):
            masks[i, int(n):] = False
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        y = store.cache["y"][dev_idx]
        metrics = pack_metrics(y, pred, rank, hl, hl)
        score = metrics["macro_f1"] + 0.15 * metrics["mrr"]
        print(
            f"probe epoch {epoch+1}/{epochs} loss {total/max(seen,1):.4f} "
            f"dev_macF1 {metrics['macro_f1']*100:.2f} dev_MRR {metrics['mrr']*100:.2f}"
        )
        if score > best_score + 1e-4:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 4:
                print("probe early stop")
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.to(device).eval()
    return probe


def _selector_dev_score(selector, probe, store, dev_idx, batch_size, pos_weight_unused=None):
    """Tune the gate threshold on dev. Returns best tau, score, metrics, mean selected count."""
    selector.eval()
    probs = []
    device = store.device
    for start in range(0, len(dev_idx), batch_size):
        batch = dev_idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=device))
        with torch.inference_mode():
            logit = selector(view["emb"], view["role"], view["pos"], view["valid"])
            prob = torch.sigmoid(logit) * view["valid"]
        probs.append(prob.cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    y = store.cache["y"][dev_idx]
    hl = store.cache["hlen"][dev_idx]
    best = None
    records = []
    for tau in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        masks = probs >= tau
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        counts = masks.sum(axis=1)
        metrics = pack_metrics(y, pred, rank, counts, hl)
        records.append((tau, metrics))
    best_f1 = max(m["macro_f1"] for _, m in records)
    eligible = [(tau, m) for tau, m in records if m["macro_f1"] >= best_f1 - 0.01]
    tau, metrics = min(eligible, key=lambda item: (item[1]["mmr"], -item[1]["macro_f1"]))
    mean_count = float(np.mean((probs >= tau).sum(axis=1)))
    return tau, metrics, mean_count, probs


def _choose_checkpoint(records: list[dict]) -> dict:
    """Among epochs within 1 Macro-F1 point of the best, keep the smallest memory."""
    best_f1 = max(item["f1"] for item in records)
    eligible = [item for item in records if item["f1"] >= best_f1 - 0.01]
    return min(eligible, key=lambda item: (item["mmr"], -item["f1"]))


def train_selector(probe, store, train_idx, dev_idx, target_masks, epochs, batch_size, lr, seed, device,
                   use_cf: bool, name: str):
    set_seed(seed)
    selector = Selector(in_dim=store.emb.shape[1]).to(device)
    selector.trunk.load_state_dict(probe.trunk.state_dict())
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=0.01)
    valid_count = float(store.cache["hlen"][train_idx].sum())
    pos = float(target_masks[train_idx].sum())
    pos_weight = float(min(5.0, max(1.0, (valid_count - pos) / max(pos, 1.0))))
    target_density = float(pos / max(valid_count, 1.0))
    print(f"{name} pos_weight {pos_weight:.2f} target_density {target_density:.3f}")
    records = []
    for epoch in range(epochs):
        selector.train()
        order = np.random.default_rng(seed + 17 + epoch).permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            target = torch.tensor(target_masks[batch], device=device, dtype=torch.float32)
            valid = view["valid"]
            clean_logits = selector(view["emb"], view["role"], view["pos"], valid)
            loss = weighted_bce(clean_logits, target, valid, pos_weight)
            if use_cf:
                loss = loss + weighted_bce(
                    selector(view["emb_emo"], view["role"], view["pos"], valid), target, valid, pos_weight
                )
                emb_s, role_s, tgt_s, valid_s = _shuffle_like(view, target, batch, epoch)
                pos_s = torch.arange(emb_s.shape[1], device=device)[None, :].expand(len(batch), -1)
                loss = loss + weighted_bce(selector(emb_s, role_s, pos_s, valid_s), tgt_s, valid_s, pos_weight)
                emb_i, role_i, tgt_i, valid_i = _insert_like(store, view, target, batch)
                pos_i = torch.arange(emb_i.shape[1], device=device)[None, :].expand(len(batch), -1)
                loss = loss + 0.5 * weighted_bce(selector(emb_i, role_i, pos_i, valid_i), tgt_i, valid_i, pos_weight)
                loss = loss / 3.5
            dens = (torch.sigmoid(clean_logits) * valid).sum() / valid.sum().clamp(min=1.0)
            loss = loss + 0.35 * (dens - target_density).abs()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        tau, metrics, mean_count, _ = _selector_dev_score(selector, probe, store, dev_idx, batch_size)
        records.append({
            "epoch": epoch + 1,
            "f1": metrics["macro_f1"],
            "mmr": metrics["mmr"],
            "tau": tau,
            "state": {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()},
        })
        print(
            f"{name} epoch {epoch+1}/{epochs} loss {total/max(seen,1):.4f} "
            f"tau {tau:.1f} dev_macF1 {metrics['macro_f1']*100:.2f} "
            f"MMR {metrics['mmr']*100:.1f} k~ {mean_count:.2f}"
        )
    chosen = _choose_checkpoint(records)
    print(
        f"{name} chosen epoch {chosen['epoch']} tau {chosen['tau']:.1f} "
        f"dev_macF1 {chosen['f1']*100:.2f} MMR {chosen['mmr']*100:.1f}"
    )
    selector.load_state_dict(chosen["state"])
    selector.to(device).eval()
    return selector, chosen["tau"]


def train_rl_selector(probe, store, train_idx, dev_idx, epochs, batch_size, lr, seed, device):
    set_seed(seed)
    selector = Selector(in_dim=store.emb.shape[1]).to(device)
    selector.trunk.load_state_dict(probe.trunk.state_dict())
    opt = torch.optim.AdamW(selector.parameters(), lr=lr, weight_decay=0.01)
    records = []
    for epoch in range(epochs):
        selector.train()
        order = np.random.default_rng(seed + 99 + epoch).permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            view = store.view(torch.tensor(batch, device=device))
            valid = view["valid"]
            logits = selector(view["emb"], view["role"], view["pos"], valid)
            probs = torch.sigmoid(logits) * valid
            with torch.no_grad():
                sample = torch.bernoulli(probs.clamp(1e-4, 1 - 1e-4)) * valid
                empty = (sample.sum(dim=1) == 0) & (valid.sum(dim=1) > 0)
                if empty.any():
                    choice = probs.masked_fill(valid < 0.5, -1).argmax(dim=1)
                    sample = sample.clone()
                    sample[empty, choice[empty]] = 1.0
                met = forward_metrics(
                    probe, view["emb"], view["role"], view["pos"], sample, view["y"], view["resp"], view["cand"]
                )
                density = sample.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
                reward = met["p"] + 0.30 * met["cos01"] - 0.12 * density
                adv = ((reward - reward.mean()) / reward.std().clamp(min=1e-3)).clone()
            logp = sample * torch.log(probs.clamp(min=1e-6)) + (1 - sample) * torch.log((1 - probs).clamp(min=1e-6))
            loss = -(adv[:, None] * logp * valid).sum() / valid.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch)
            seen += len(batch)
        tau, metrics, mean_count, _ = _selector_dev_score(selector, probe, store, dev_idx, batch_size)
        records.append({
            "epoch": epoch + 1,
            "f1": metrics["macro_f1"],
            "mmr": metrics["mmr"],
            "tau": tau,
            "state": {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()},
        })
        print(
            f"rl epoch {epoch+1}/{epochs} loss {total/max(seen,1):.4f} "
            f"tau {tau:.1f} dev_macF1 {metrics['macro_f1']*100:.2f} MMR {metrics['mmr']*100:.1f}"
        )
    chosen = _choose_checkpoint(records)
    print(
        f"rl chosen epoch {chosen['epoch']} tau {chosen['tau']:.1f} "
        f"dev_macF1 {chosen['f1']*100:.2f} MMR {chosen['mmr']*100:.1f}"
    )
    selector.load_state_dict(chosen["state"])
    selector.to(device).eval()
    return selector, chosen["tau"]


def _shuffle_like(view, target, batch_idx, epoch: int):
    emb, role, tgt = view["emb"].clone(), view["role"].clone(), target.clone()
    valid = view["valid"].clone()
    for local, inst in enumerate(batch_idx):
        n = int(view["hlen"][local].item())
        if n <= 1:
            continue
        perm = np.random.default_rng(80_000 + epoch * 997 + int(inst)).permutation(n)
        perm_t = torch.tensor(perm, device=view["emb"].device)
        emb[local, :n] = view["emb"][local, perm_t]
        role[local, :n] = view["role"][local, perm_t]
        tgt[local, :n] = target[local, perm_t]
    return emb, role, tgt, valid


def _insert_like(store: Store, view, target, batch_idx):
    bsz, length, dim = view["emb"].shape
    emb_i = torch.zeros(bsz, length + 1, dim, device=view["emb"].device)
    role_i = torch.zeros(bsz, length + 1, dtype=torch.long, device=view["emb"].device)
    tgt_i = torch.zeros(bsz, length + 1, device=view["emb"].device)
    valid_i = torch.zeros(bsz, length + 1, device=view["emb"].device)
    donors = store.emb[store.ins_donor[torch.tensor(batch_idx, device=store.device)]]
    donor_role = store.role_all[store.ins_donor[torch.tensor(batch_idx, device=store.device)].clamp(min=0)]
    for local in range(bsz):
        n = int(view["hlen"][local].item())
        ins_at = max(0, n - 1) if n > 1 else 0
        if ins_at:
            emb_i[local, :ins_at] = view["emb"][local, :ins_at]
            role_i[local, :ins_at] = view["role"][local, :ins_at]
            tgt_i[local, :ins_at] = target[local, :ins_at]
        emb_i[local, ins_at] = donors[local]
        role_i[local, ins_at] = donor_role[local]
        rest = n - ins_at
        if rest:
            emb_i[local, ins_at + 1:ins_at + 1 + rest] = view["emb"][local, ins_at:n]
            role_i[local, ins_at + 1:ins_at + 1 + rest] = view["role"][local, ins_at:n]
            tgt_i[local, ins_at + 1:ins_at + 1 + rest] = target[local, ins_at:n]
        valid_i[local, :n + 1] = 1.0
    return emb_i, role_i, tgt_i, valid_i


def _lastk_mask(hlen: torch.Tensor, length: int, k: int) -> torch.Tensor:
    bsz = hlen.shape[0]
    mask = torch.zeros(bsz, length, dtype=torch.bool, device=hlen.device)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        if n <= 0:
            continue
        mask[i, max(0, n - k):n] = True
    return mask


def _topk_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    filled = scores.masked_fill(valid < 0.5, -1e9)
    kk = min(k, filled.shape[1])
    idx = filled.topk(kk, dim=1).indices
    mask = torch.zeros_like(valid, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask & (valid > 0.5)


def _sim_scores(view) -> torch.Tensor:
    emb = view["emb"]
    bsz = emb.shape[0]
    last_index = (view["hlen"] - 1).clamp(min=0)
    last = emb[torch.arange(bsz, device=emb.device), last_index]
    return torch.einsum("bld,bd->bl", emb, last)


def _lex_scores(cache, ids: np.ndarray, hlen: np.ndarray, substituted: bool) -> np.ndarray:
    scores = np.full(ids.shape, -1e4, dtype=np.float32)
    for b, n in enumerate(hlen.tolist()):
        n = int(n)
        if n <= 0:
            continue
        last_id = int(ids[b, n - 1])
        last_text = cache["texts"][last_id]
        if substituted:
            last_text = substitute_emotion(last_text)[0]
        last_set = emotion_tokens(last_text)
        for t in range(n):
            tid = int(ids[b, t])
            if tid < 0:
                continue
            if t == n - 1:
                scores[b, t] = 1.0
                continue
            text = cache["texts"][tid]
            if substituted:
                text = substitute_emotion(text)[0]
            scores[b, t] = jaccard(emotion_tokens(text), last_set)
    return scores


def _tfidf_scores(cache, ids: np.ndarray, hlen: np.ndarray, substituted: bool) -> np.ndarray:
    scores = np.full(ids.shape, -1e4, dtype=np.float32)
    vec = cache["vectorizer"]
    for b, n in enumerate(hlen.tolist()):
        n = int(n)
        if n <= 0:
            continue
        texts = []
        for t in range(n):
            raw = cache["texts"][int(ids[b, t])]
            if substituted:
                raw = substitute_emotion(raw)[0]
            texts.append(raw)
        mat = vec.transform(texts)
        sims = (mat @ mat[-1].T).toarray().ravel()
        scores[b, :n] = sims.astype(np.float32)
    return scores


def _placed_scores(clean_scores: np.ndarray, origin: np.ndarray) -> np.ndarray:
    out = np.full(origin.shape, -1e9, dtype=np.float32)
    for b in range(origin.shape[0]):
        for t, src in enumerate(origin[b]):
            if src >= 0:
                out[b, t] = clean_scores[b, int(src)]
    return out


def _make_views(store: Store, batch_idx: np.ndarray):
    device = store.device
    clean = store.view(torch.tensor(batch_idx, device=device), kind="clean")
    emo = store.view(torch.tensor(batch_idx, device=device), kind="emo")
    # Shuffle keeps turn identity in `origin` so precomputed scores can follow content.
    shuf = store.view(torch.tensor(batch_idx, device=device), kind="clean")
    bsz, length = shuf["ids"].shape
    origin_s = np.full((bsz, length), -1, dtype=np.int16)
    emb_s = shuf["emb"].clone()
    role_s = shuf["role"].clone()
    ids_s = shuf["ids"].clone()
    emo_s = shuf["emb_emo"].clone()
    for local, inst in enumerate(batch_idx):
        n = int(shuf["hlen"][local].item())
        origin_s[local, :n] = np.arange(n)
        if n <= 1:
            continue
        perm = np.random.default_rng(50_000 + int(inst)).permutation(n)
        emb_s[local, :n] = shuf["emb"][local, torch.tensor(perm, device=device)]
        role_s[local, :n] = shuf["role"][local, torch.tensor(perm, device=device)]
        ids_s[local, :n] = shuf["ids"][local, torch.tensor(perm, device=device)]
        emo_s[local, :n] = shuf["emb_emo"][local, torch.tensor(perm, device=device)]
        origin_s[local, :n] = perm
    shuf["emb"], shuf["role"], shuf["ids"], shuf["emb_emo"] = emb_s, role_s, ids_s, emo_s

    ins = _build_insert_view(store, clean, batch_idx)
    return {"clean": clean, "emo": emo, "shuffle": shuf, "insert": ins}, {"shuffle": origin_s, "insert": ins["origin"]}


def _build_insert_view(store: Store, clean, batch_idx: np.ndarray):
    bsz, length, dim = clean["emb"].shape
    new_l = length + 1
    device = clean["emb"].device
    emb = torch.zeros(bsz, new_l, dim, device=device)
    emb_emo = torch.zeros(bsz, new_l, dim, device=device)
    role = torch.zeros(bsz, new_l, dtype=torch.long, device=device)
    ids = torch.full((bsz, new_l), -1, dtype=torch.long, device=device)
    valid = torch.zeros(bsz, new_l, device=device)
    hlen = clean["hlen"] + 1
    origin = np.full((bsz, new_l), -1, dtype=np.int16)
    donor_ids = store.ins_donor[torch.tensor(batch_idx, device=device)].clamp(min=0)
    donors = store.emb[donor_ids]
    donor_role = store.role_all[donor_ids]
    donor_emo = store.emb_emo[donor_ids]
    for local in range(bsz):
        n = int(clean["hlen"][local].item())
        ins_at = max(0, n - 1) if n > 1 else 0
        if ins_at:
            emb[local, :ins_at] = clean["emb"][local, :ins_at]
            emb_emo[local, :ins_at] = clean["emb_emo"][local, :ins_at]
            role[local, :ins_at] = clean["role"][local, :ins_at]
            ids[local, :ins_at] = clean["ids"][local, :ins_at]
            origin[local, :ins_at] = np.arange(ins_at)
        emb[local, ins_at] = donors[local]
        emb_emo[local, ins_at] = donor_emo[local]
        role[local, ins_at] = donor_role[local]
        ids[local, ins_at] = donor_ids[local]
        origin[local, ins_at] = -2
        rest = n - ins_at
        if rest:
            emb[local, ins_at + 1:ins_at + 1 + rest] = clean["emb"][local, ins_at:n]
            emb_emo[local, ins_at + 1:ins_at + 1 + rest] = clean["emb_emo"][local, ins_at:n]
            role[local, ins_at + 1:ins_at + 1 + rest] = clean["role"][local, ins_at:n]
            ids[local, ins_at + 1:ins_at + 1 + rest] = clean["ids"][local, ins_at:n]
            origin[local, ins_at + 1:ins_at + 1 + rest] = np.arange(ins_at, n)
        valid[local, :n + 1] = 1.0
    pos = torch.arange(new_l, device=device)[None, :].expand(bsz, new_l)
    return {
        "emb": emb,
        "emb_emo": emb_emo,
        "role": role,
        "pos": pos,
        "valid": valid,
        "y": clean["y"],
        "resp": clean["resp"],
        "cand": clean["cand"],
        "hlen": hlen,
        "ids": ids,
        "origin": origin,
    }


def _selector_mask(selector, view, tau: float | None, k: int | None) -> torch.Tensor:
    with torch.inference_mode():
        prob = torch.sigmoid(selector(view["emb"], view["role"], view["pos"], view["valid"]))
    if k is not None:
        return _topk_mask(prob, view["valid"], k)
    return (prob >= tau) & (view["valid"] > 0.5)


def _attn_mask(probe, view, k: int) -> torch.Tensor:
    with torch.inference_mode():
        _logits, _q, weights = probe(view["emb"], view["role"], view["pos"], view["valid"], return_weights=True)
    return _topk_mask(weights, view["valid"], k)


def evaluate_methods(probe, selectors: dict, taus: dict, store: Store, effects: dict, indices: np.ndarray,
                     k: int, batch_size: int = 64) -> dict:
    """Evaluate deployable methods under clean, shuffle, insertion, and emotion substitution."""
    probe.eval()
    names = [
        "FullHist", "NoHist", "Last1", "Last2", "LastK", "RandomK", "SimK", "EmoK",
        "TfidfK", "AttnK", "CMIConfK", "CauMemK", "CauMem",
    ]
    for extra in selectors:
        if extra not in ("caumem",):
            names.append(extra)
    buckets = {
        name: {cond: {"pred": [], "rank": [], "count": [], "ids": []} for cond in ("clean", "shuffle", "insert", "emo")}
        for name in names
    }
    y_all, hlen_clean = [], []
    cache = store.cache
    conf_score = effects["conf_util"] + 0.25 * effects["conf_stab"]
    for start in tqdm(range(0, len(indices), batch_size), desc="eval"):
        batch = indices[start:start + batch_size]
        views, origins = _make_views(store, batch)
        y_all.append(views["clean"]["y"].cpu().numpy())
        hlen_clean.append(cache["hlen"][batch])
        clean_scores = {
            "CMIConfK": conf_score[batch],
        }
        for cond, view in views.items():
            valid = view["valid"]
            hlen = view["hlen"]
            ids_np = view["ids"].detach().cpu().numpy()
            hlen_np = hlen.detach().cpu().numpy()
            substituted = cond == "emo"
            sim = _sim_scores(view)
            lex = torch.tensor(_lex_scores(cache, ids_np, hlen_np, substituted), device=store.device)
            tfidf = torch.tensor(_tfidf_scores(cache, ids_np, hlen_np, substituted), device=store.device)
            if cond == "clean" or cond == "emo":
                cmi_np = clean_scores["CMIConfK"]
            else:
                cmi_np = _placed_scores(clean_scores["CMIConfK"], origins[cond])
            if cond == "insert":
                with torch.inference_mode():
                    only = torch.zeros_like(valid)
                    origin_np = origins["insert"]
                    for i in range(len(batch)):
                        donors = np.where(origin_np[i] == -2)[0]
                        if len(donors):
                            only[i, int(donors[0])] = 1.0
                    one = forward_metrics(probe, view["emb"], view["role"], view["pos"], only, view["y"], view["resp"], view["cand"])
                    empty = forward_metrics(
                        probe, view["emb"], view["role"], view["pos"], torch.zeros_like(valid),
                        view["y"], view["resp"], view["cand"],
                    )
                    dscore = (one["conf"] - empty["conf"]).detach().cpu().numpy()
                for i in range(len(batch)):
                    donors = np.where(origins["insert"][i] == -2)[0]
                    if len(donors):
                        cmi_np[i, int(donors[0])] = dscore[i]
            cmi = torch.tensor(cmi_np, device=store.device)
            cond_id = {"clean": 1, "shuffle": 2, "insert": 3, "emo": 4}[cond]
            rng = np.random.default_rng(4_000 + start + cond_id * 97)
            rand = np.zeros((len(batch), view["emb"].shape[1]), dtype=np.bool_)
            for i, n in enumerate(hlen_np.tolist()):
                kk = min(k, int(n))
                if kk > 0:
                    rand[i, rng.choice(int(n), size=kk, replace=False)] = True
            masks = {
                "FullHist": valid > 0.5,
                "NoHist": torch.zeros_like(valid, dtype=torch.bool),
                "Last1": _lastk_mask(hlen, view["emb"].shape[1], 1),
                "Last2": _lastk_mask(hlen, view["emb"].shape[1], 2),
                "LastK": _lastk_mask(hlen, view["emb"].shape[1], k),
                "RandomK": torch.tensor(rand, device=store.device),
                "SimK": _topk_mask(sim, valid, k),
                "EmoK": _topk_mask(lex, valid, k),
                "TfidfK": _topk_mask(tfidf, valid, k),
                "AttnK": _attn_mask(probe, view, k),
                "CMIConfK": _topk_mask(cmi, valid, k),
                "CauMemK": _selector_mask(selectors["caumem"], view, None, k),
                "CauMem": _selector_mask(selectors["caumem"], view, taus["caumem"], None),
            }
            for extra, model in selectors.items():
                if extra == "caumem":
                    continue
                masks[extra] = _selector_mask(model, view, taus[extra], None)
            for name, mask in masks.items():
                with torch.inference_mode():
                    met = forward_metrics(
                        probe, view["emb"], view["role"], view["pos"], mask.float(), view["y"], view["resp"], view["cand"]
                    )
                chosen = []
                mask_np = mask.detach().cpu().numpy()
                for i in range(len(batch)):
                    picked = [int(ids_np[i, t]) for t in range(mask_np.shape[1]) if mask_np[i, t] and ids_np[i, t] >= 0]
                    chosen.append(picked)
                buckets[name][cond]["pred"].append(met["pred"].cpu().numpy())
                buckets[name][cond]["rank"].append(met["rank"].cpu().numpy())
                buckets[name][cond]["count"].append(mask_np.sum(axis=1))
                buckets[name][cond]["ids"].extend(chosen)
    y = np.concatenate(y_all)
    hlen = np.concatenate(hlen_clean)
    summary = {}
    detail_preds = {}
    for name in names:
        summary[name] = {}
        detail_preds[name] = {}
        clean_ids = buckets[name]["clean"]["ids"]
        shuf_ids = buckets[name]["shuffle"]["ids"]
        stabs = []
        for a, b in zip(clean_ids, shuf_ids):
            sa, sb = set(a), set(b)
            stabs.append(1.0 if not sa and not sb else len(sa & sb) / max(1, len(sa | sb)))
        for cond in ("clean", "shuffle", "insert", "emo"):
            pred = np.concatenate(buckets[name][cond]["pred"])
            rank = np.concatenate(buckets[name][cond]["rank"])
            count = np.concatenate(buckets[name][cond]["count"])
            # Compression is defined on the clean history length, not the inserted length.
            denom = hlen if cond != "insert" else np.maximum(hlen, 1)
            if cond == "insert":
                # count includes the possible distractor; still divide by original history length
                metrics = pack_metrics(y, pred, rank, count, denom)
            else:
                metrics = pack_metrics(y, pred, rank, count, hlen)
            summary[name][cond] = metrics
            detail_preds[name][cond] = pred
            detail_preds[name][cond + "_rank"] = rank
            detail_preds[name][cond + "_count"] = count
        pc = detail_preds[name]["clean"]
        summary[name]["cc_emo"] = float(np.mean(pc == detail_preds[name]["emo"]))
        summary[name]["cc_ins"] = float(np.mean(pc == detail_preds[name]["insert"]))
        summary[name]["cc"] = 0.5 * (summary[name]["cc_emo"] + summary[name]["cc_ins"])
        summary[name]["sel_stability"] = float(np.mean(stabs))
        summary[name]["clean_turn_ids"] = buckets[name]["clean"]["ids"]
    summary["_y"] = y
    summary["_hlen"] = hlen
    summary["_generic"] = cache["generic"][indices]
    summary["_preds"] = detail_preds
    return summary


def oracle_rows(probe, store, effects, masks_by_name: dict, indices: np.ndarray, k: int) -> dict:
    rows = {}
    y = store.cache["y"][indices]
    hl = store.cache["hlen"][indices]
    for name, global_masks in masks_by_name.items():
        pred, rank = run_fixed(probe, store, indices, global_masks[indices])
        counts = global_masks[indices].sum(axis=1)
        rows[name] = pack_metrics(y, pred, rank, counts, hl)
        rows[name]["pred"] = pred
    # LOO top-k and CMI gold rule are score masks, built here.
    loo = np.zeros((len(indices), store.max_l), dtype=np.bool_)
    cmi = np.zeros_like(loo)
    scores = effects["delta_del"][indices]
    gold_u = effects["gold_util"][indices]
    gold_s = effects["gold_stab"][indices]
    for i, n in enumerate(hl.tolist()):
        n = int(n)
        kk = min(k, n)
        if kk > 0:
            choice = np.argpartition(-scores[i, :n], kk - 1)[:kk]
            loo[i, choice] = True
        cmi[i, :n] = (gold_u[i, :n] > 0) & (gold_s[i, :n] >= 0)
    for name, mask in (("LOO-K", loo), ("CMI-gold", cmi)):
        pred, rank = run_fixed(probe, store, indices, mask)
        rows[name] = pack_metrics(y, pred, rank, mask.sum(axis=1), hl)
    return rows


def bootstrap_macro_diff(y, pred_a, pred_b, n_boot: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        fa = f1_score(y[ix], pred_a[ix], average="macro", labels=LABELS, zero_division=0)
        fb = f1_score(y[ix], pred_b[ix], average="macro", labels=LABELS, zero_division=0)
        diffs.append(fa - fb)
    arr = np.asarray(diffs)
    return {
        "mean": float(arr.mean()),
        "low": float(np.quantile(arr, 0.025)),
        "high": float(np.quantile(arr, 0.975)),
    }


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.2f}"


def render_main_table(summary: dict, strategies: list[str]) -> str:
    header = (
        "| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR | HCR | CC | "
        "MacF1-shuf | MacF1-ins | MRR-shuf | MRR-ins | SelStab |"
    )
    sep = "|" + "---|" * 13
    lines = [
        header, sep,
    ]
    order = [
        "FullHist", "NoHist", "Last1", "Last2", "LastK", "RandomK", "SimK", "EmoK",
        "TfidfK", "AttnK", "CMIConfK", "CauMemK", "CauMem",
    ]
    for name in order:
        if name not in summary:
            continue
        row = summary[name]
        c, s, ins = row["clean"], row["shuffle"], row["insert"]
        lines.append(
            f"| {name} | {fmt_pct(c['macro_f1'])} | {fmt_pct(c['weighted_f1'])} | {fmt_pct(c['r1'])} | "
            f"{fmt_pct(c['r5'])} | {fmt_pct(c['mrr'])} | {fmt_pct(c['mmr'])} | {fmt_pct(c['hcr'])} | "
            f"{fmt_pct(row['cc'])} | {fmt_pct(s['macro_f1'])} | {fmt_pct(ins['macro_f1'])} | "
            f"{fmt_pct(s['mrr'])} | {fmt_pct(ins['mrr'])} | {fmt_pct(row['sel_stability'])} |"
        )
    return "\n".join(lines)


def render_oracle_table(rows: dict) -> str:
    lines = [
        "| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR | HCR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {fmt_pct(row['macro_f1'])} | {fmt_pct(row['weighted_f1'])} | {fmt_pct(row['r1'])} | "
            f"{fmt_pct(row['r5'])} | {fmt_pct(row['mrr'])} | {fmt_pct(row['mmr'])} | {fmt_pct(row['hcr'])} |"
        )
    return "\n".join(lines)


def subset_report(summary: dict, strategies: list[str]) -> str:
    y = summary["_y"]
    hlen = summary["_hlen"]
    generic = summary["_generic"]
    preds = summary["_preds"]
    lines = ["| Slice | Method | n | MacF1 | R@1 | MRR | MMR |", "|---|---|---|---|---|---|---|"]
    slices = {
        "short(1-8)": hlen <= 8,
        "mid(9-20)": (hlen >= 9) & (hlen <= 20),
        "long(>=21)": hlen >= 21,
        "generic-last": generic.astype(bool),
    }
    for slice_name, mask in slices.items():
        if mask.sum() < 5:
            continue
        for method in ("FullHist", "LastK", "SimK", "AttnK", "CauMem"):
            pred = preds[method]["clean"][mask]
            rank = preds[method]["clean_rank"][mask]
            count = preds[method]["clean_count"][mask]
            met = pack_metrics(y[mask], pred, rank, count, hlen[mask])
            lines.append(
                f"| {slice_name} | {method} | {met['n']} | {fmt_pct(met['macro_f1'])} | "
                f"{fmt_pct(met['r1'])} | {fmt_pct(met['mrr'])} | {fmt_pct(met['mmr'])} |"
            )
    # per-class for full and caumem
    lines.append("")
    lines.append("| Strategy | FullHist F1 | CauMem F1 |")
    lines.append("|---|---|---|")
    full_c = per_class_f1(y, preds["FullHist"]["clean"])
    cau_c = per_class_f1(y, preds["CauMem"]["clean"])
    for i, name in enumerate(strategies):
        lines.append(f"| {name} | {fmt_pct(full_c[str(i)])} | {fmt_pct(cau_c[str(i)])} |")
    return "\n".join(lines)


def export_cases(cache, indices, summary, k_show: int = 6) -> list:
    y = summary["_y"]
    hlen = summary["_hlen"]
    preds = summary["_preds"]
    cau_ok = preds["CauMem"]["clean"] == y
    sim_bad = preds["SimK"]["clean"] != y
    long = hlen >= 12
    cand = np.where(cau_ok & sim_bad & long)[0]
    if len(cand) == 0:
        cand = np.where(cau_ok & long)[0]
    counts = preds["CauMem"]["clean_count"]
    cand = sorted(cand.tolist(), key=lambda i: (counts[i] / max(hlen[i], 1), -hlen[i]))[:k_show]
    strategies = cache["strategies"]
    picked_lists = summary["CauMem"]["clean_turn_ids"]
    cases = []
    for local in cand:
        inst = int(indices[local])
        n = int(cache["hlen"][inst])
        ids = [int(t) for t in cache["hist"][inst, :n]]
        picked = set(picked_lists[local])
        history = []
        for t in ids:
            mark = "KEEP" if t in picked else "DROP"
            history.append({"keep": mark, "text": cache["texts"][t][:240]})
        cases.append({
            "instance": inst,
            "history_len": n,
            "gold_strategy": strategies[int(y[local])],
            "caumem_pred": strategies[int(preds["CauMem"]["clean"][local])],
            "sim_pred": strategies[int(preds["SimK"]["clean"][local])],
            "full_pred": strategies[int(preds["FullHist"]["clean"][local])],
            "caumem_count": int(counts[local]),
            "gold_response": cache["texts"][int(cache["resp"][inst])][:320],
            "history": history,
        })
    return cases
