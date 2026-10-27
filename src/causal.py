"""Interventional effects and the minimal sufficient memory search.

Utility of a history subset S is
    u(S) = log p(y*|S) + 0.35 * cos(q(S), r*).

For each turn i we estimate three interventions against the full history H:
    deletion        Δdel = u(H) - u(H without i)
    emotion swap    Δemo = u(H) - u(H with emotion lexemes in i replaced)
    background swap Δbg  = u(H) - u(H with i replaced by an emotion-matched,
                                   content-different turn)

Surface dependence is the part of the emotion effect that is not shared with
the background effect. The content-causal score is
    surface = max(0, Δemo - max(Δbg, 0))
    φ       = Δdel - 0.5 * surface + 0.25 * max(Δbg, 0)

Turns with near-zero individual deletion effects but a large joint deletion
effect are substitutes. MISSET keeps one representative of each substitute
group, grows a set by φ until it is ε-sufficient for the full history, then
deletes any turn that is no longer necessary.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def forward_metrics(probe, emb, role, pos, keep, y, resp, cand):
    logits, query = probe(emb, role, pos, keep)
    prob = torch.softmax(logits, dim=-1)
    pred = logits.argmax(dim=-1)
    p_gold = prob.gather(1, y[:, None]).squeeze(1)
    cos = (query * resp).sum(dim=-1)
    cos01 = (cos + 1.0) * 0.5
    # Log-probability makes single-turn interventions visible. cos stays in [-1, 1].
    utility = torch.log(p_gold.clamp(min=1e-6)) + 0.35 * cos
    sims = torch.einsum("bd,bkd->bk", query, cand)
    rank = 1 + (sims > sims[:, :1]).sum(dim=1)
    conf = prob.max(dim=-1).values
    return {
        "u": utility,
        "pred": pred,
        "p": p_gold,
        "cos": cos,
        "cos01": cos01,
        "rank": rank,
        "conf": conf,
        "logits": logits,
        "q": query,
    }


def suffices(current, reference, y, eps_p: float, rank_slack: int):
    """A subset is sufficient when it preserves the full-history decision.

    Gold labels score the decision; they are not used to search for a different
    label than the full history produced.
    """
    del y
    pred_ok = current["pred"].eq(reference["pred"])
    prob_ok = current["p"] >= (reference["p"] - eps_p)
    rank_ok = current["rank"] <= (reference["rank"] + rank_slack)
    return pred_ok & prob_ok & rank_ok


class Store:
    def __init__(self, cache: dict, device: torch.device):
        self.cache = cache
        self.device = device
        self.emb = torch.tensor(cache["emb"], device=device)
        self.emb_emo = torch.tensor(cache["emb_emo"], device=device)
        self.role_all = torch.tensor(cache["role"], device=device, dtype=torch.long)
        self.donor = torch.tensor(cache["donor"], device=device, dtype=torch.long)
        self.has_emo = torch.tensor(cache["has_emo"], device=device)
        self.hist = torch.tensor(cache["hist"], device=device, dtype=torch.long)
        self.hlen = torch.tensor(cache["hlen"], device=device, dtype=torch.long)
        self.y = torch.tensor(cache["y"], device=device, dtype=torch.long)
        self.resp_ids = torch.tensor(cache["resp"], device=device, dtype=torch.long)
        self.cand_ids = torch.tensor(cache["cand"], device=device, dtype=torch.long)
        self.ins_donor = torch.tensor(cache["ins_donor"], device=device, dtype=torch.long)
        self.max_l = int(cache["max_l"])

    def _gather(self, ids: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
        flat = ids.clamp(min=0)
        return table[flat]

    def view(self, idxs: torch.Tensor, kind: str = "clean"):
        hist = self.hist[idxs]
        hlen = self.hlen[idxs]
        bsz, length = hist.shape
        valid = torch.arange(length, device=self.device)[None, :] < hlen[:, None]
        ids = hist.clamp(min=0)
        table = self.emb_emo if kind == "emo" else self.emb
        emb = table[ids]
        emb = emb * valid.unsqueeze(-1).float()
        emb_emo = self.emb_emo[ids] * valid.unsqueeze(-1).float()
        role = self.role_all[ids]
        pos = torch.arange(length, device=self.device)[None, :].expand(bsz, length)
        y = self.y[idxs]
        resp = self.emb[self.resp_ids[idxs]]
        cand = self.emb[self.cand_ids[idxs]]
        has = self.has_emo[ids] & valid
        donor_ids = self.donor[ids].clamp(min=0)
        emb_bg = self.emb[donor_ids] * valid.unsqueeze(-1).float()
        return {
            "emb": emb,
            "emb_emo": emb_emo,
            "emb_bg": emb_bg,
            "role": role,
            "pos": pos,
            "valid": valid.float(),
            "has_emo": has,
            "y": y,
            "resp": resp,
            "cand": cand,
            "hlen": hlen,
            "ids": hist,
        }


def _empty_effects(n: int, length: int) -> dict:
    z = lambda: np.zeros((n, length), dtype=np.float32)
    return {
        "delta_del": z(),
        "delta_emo": z(),
        "delta_bg": z(),
        "conf_util": z(),
        "conf_stab": z(),
        "gold_util": z(),
        "gold_stab": z(),
        "pair_i": np.full((n, 15), -1, dtype=np.int16),
        "pair_j": np.full((n, 15), -1, dtype=np.int16),
        "pair_dij": np.zeros((n, 15), dtype=np.float32),
        "u_full": np.zeros(n, dtype=np.float32),
        "pred_full": np.zeros(n, dtype=np.int16),
        "p_full": np.zeros(n, dtype=np.float32),
        "rank_full": np.ones(n, dtype=np.int16),
    }


def compute_effects(probe, store: Store, batch_size: int = 96) -> dict:
    probe.eval()
    n, length = store.hist.shape
    out = _empty_effects(n, length)
    device = store.device
    for start in tqdm(range(0, n, batch_size), desc="effects"):
        idx_np = np.arange(start, min(n, start + batch_size))
        idxs = torch.tensor(idx_np, device=device)
        view = store.view(idxs, kind="clean")
        emb, role, pos = view["emb"], view["role"], view["pos"]
        valid = view["valid"]
        y, resp, cand = view["y"], view["resp"], view["cand"]
        with torch.inference_mode():
            ref = forward_metrics(probe, emb, role, pos, valid, y, resp, cand)
            empty = forward_metrics(probe, emb, role, pos, torch.zeros_like(valid), y, resp, cand)
        out["u_full"][idx_np] = ref["u"].float().cpu().numpy()
        out["pred_full"][idx_np] = ref["pred"].cpu().numpy().astype(np.int16)
        out["p_full"][idx_np] = ref["p"].float().cpu().numpy()
        out["rank_full"][idx_np] = ref["rank"].cpu().numpy().astype(np.int16)
        bsz = len(idx_np)
        delta_del = np.zeros((bsz, length), dtype=np.float32)
        delta_emo = np.zeros((bsz, length), dtype=np.float32)
        delta_bg = np.zeros((bsz, length), dtype=np.float32)
        conf_util = np.zeros((bsz, length), dtype=np.float32)
        conf_stab = np.zeros((bsz, length), dtype=np.float32)
        gold_util = np.zeros((bsz, length), dtype=np.float32)
        gold_stab = np.zeros((bsz, length), dtype=np.float32)
        valid_np = valid.cpu().numpy().astype(bool)
        has_np = view["has_emo"].cpu().numpy().astype(bool)
        with torch.inference_mode():
            for p in range(length):
                if not valid_np[:, p].any():
                    continue
                keep = valid.clone()
                keep[:, p] = 0
                deleted = forward_metrics(probe, emb, role, pos, keep, y, resp, cand)
                delta_del[:, p] = (ref["u"] - deleted["u"]).float().cpu().numpy()

                if has_np[:, p].any():
                    emb_e = emb.clone()
                    emb_e[:, p] = view["emb_emo"][:, p]
                    swapped = forward_metrics(probe, emb_e, role, pos, valid, y, resp, cand)
                    delta_emo[:, p] = (ref["u"] - swapped["u"]).float().cpu().numpy()

                emb_b = emb.clone()
                emb_b[:, p] = view["emb_bg"][:, p]
                backed = forward_metrics(probe, emb_b, role, pos, valid, y, resp, cand)
                delta_bg[:, p] = (ref["u"] - backed["u"]).float().cpu().numpy()

                only = torch.zeros_like(valid)
                only[:, p] = valid[:, p]
                one = forward_metrics(probe, emb, role, pos, only, y, resp, cand)
                gold_util[:, p] = (one["u"] - empty["u"]).float().cpu().numpy()
                conf_util[:, p] = (one["conf"] - empty["conf"]).float().cpu().numpy()
                if has_np[:, p].any():
                    one_e = forward_metrics(probe, emb_e, role, pos, only, y, resp, cand)
                    gold_stab[:, p] = (one["u"] - one_e["u"]).float().cpu().numpy()
                    conf_stab[:, p] = (one["conf"] - one_e["conf"]).float().cpu().numpy()

            # Joint deletion for the turns leave-one-out cannot see.
            pair_i = np.full((bsz, 15), -1, dtype=np.int16)
            pair_j = np.full((bsz, 15), -1, dtype=np.int16)
            pair_dij = np.zeros((bsz, 15), dtype=np.float32)
            pair_lists: list[list[tuple[int, int]]] = []
            max_pairs = 0
            for b in range(bsz):
                positions = np.where(valid_np[b])[0]
                if len(positions) < 2:
                    pair_lists.append([])
                    continue
                order = positions[np.argsort(delta_del[b, positions])[:6]]
                pairs = [(int(order[a]), int(order[c])) for a in range(len(order)) for c in range(a + 1, len(order))]
                pair_lists.append(pairs)
                max_pairs = max(max_pairs, len(pairs))
            for t in range(max_pairs):
                keep = valid.clone()
                active = []
                for b, pairs in enumerate(pair_lists):
                    if t >= len(pairs):
                        continue
                    i, j = pairs[t]
                    keep[b, i] = 0
                    keep[b, j] = 0
                    active.append(b)
                    pair_i[b, t] = i
                    pair_j[b, t] = j
                if not active:
                    continue
                joint = forward_metrics(probe, emb, role, pos, keep, y, resp, cand)
                dij = (ref["u"] - joint["u"]).float().cpu().numpy()
                for b in active:
                    pair_dij[b, t] = dij[b]
        # Zero-out padded positions.
        delta_del = np.where(valid_np, delta_del, 0.0)
        delta_emo = np.where(valid_np, delta_emo, 0.0)
        delta_bg = np.where(valid_np, delta_bg, 0.0)
        out["delta_del"][idx_np] = delta_del
        out["delta_emo"][idx_np] = delta_emo
        out["delta_bg"][idx_np] = delta_bg
        out["conf_util"][idx_np] = np.where(valid_np, conf_util, 0.0)
        out["conf_stab"][idx_np] = np.where(valid_np, conf_stab, 0.0)
        out["gold_util"][idx_np] = np.where(valid_np, gold_util, 0.0)
        out["gold_stab"][idx_np] = np.where(valid_np, gold_stab, 0.0)
        out["pair_i"][idx_np] = pair_i
        out["pair_j"][idx_np] = pair_j
        out["pair_dij"][idx_np] = pair_dij
    return out


def factor_scores(effects: dict, mode: str, hlen: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_del = effects["delta_del"]
    delta_emo = effects["delta_emo"]
    delta_bg = effects["delta_bg"]
    emo_pos = np.maximum(delta_emo, 0.0)
    bg_pos = np.maximum(delta_bg, 0.0)
    if mode == "full":
        surface = np.maximum(0.0, delta_emo - bg_pos)
        phi = delta_del - 0.5 * surface + 0.25 * bg_pos
        spec = delta_bg - 0.5 * emo_pos
    elif mode == "del":
        phi = delta_del.copy()
        spec = delta_del.copy()
    elif mode == "no_emo":
        phi = delta_del + 0.25 * bg_pos
        spec = delta_bg.copy()
    elif mode == "no_bg":
        phi = delta_del - 0.5 * emo_pos
        spec = -emo_pos
    else:
        raise ValueError(mode)
    valid = np.arange(phi.shape[1])[None, :] < hlen[:, None]
    phi = np.where(valid, phi, -1e9).astype(np.float32)
    spec = np.where(valid, spec, -1e9).astype(np.float32)
    return phi, spec


def redundancy_diagnostics(effects: dict) -> dict:
    """How often a pair is individually weak but jointly necessary."""
    valid = effects["pair_i"] >= 0
    if not np.any(valid):
        return {"n_pairs": 0}
    di = np.zeros(valid.sum(), dtype=np.float32)
    dj = np.zeros_like(di)
    dij = effects["pair_dij"][valid]
    # Gather the deletion effects of the paired positions.
    cursor = 0
    delta = effects["delta_del"]
    for i in range(delta.shape[0]):
        for t in range(effects["pair_i"].shape[1]):
            a = int(effects["pair_i"][i, t])
            b = int(effects["pair_j"][i, t])
            if a < 0:
                continue
            di[cursor] = delta[i, a]
            dj[cursor] = delta[i, b]
            cursor += 1
    synergy = dij - di - dj
    weak = (di < 0.08) & (dj < 0.08)
    return {
        "n_pairs": int(valid.sum()),
        "p50_dij": float(np.percentile(dij, 50)),
        "p95_dij": float(np.percentile(dij, 95)),
        "p95_synergy": float(np.percentile(synergy, 95)),
        "frac_weak_synergy_0.10": float(np.mean(weak & (synergy > 0.10) & (dij > 0.12))),
        "frac_rule_0.08_0.20": float(np.mean((di < 0.08) & (dj < 0.08) & (dij > 0.20) & (synergy > 0.10))),
    }


def substitute_groups(
    effects: dict,
    spec: np.ndarray,
    hlen: np.ndarray,
    di_max: float = 0.08,
    dij_min: float = 0.20,
    synergy: float = 0.10,
) -> np.ndarray:
    """Suppress all but one representative in each redundant substitute group."""
    n, length = spec.shape
    suppressed = np.zeros((n, length), dtype=np.bool_)
    delta = effects["delta_del"]
    for i in range(n):
        parent = list(range(length))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for t in range(15):
            a = int(effects["pair_i"][i, t])
            b = int(effects["pair_j"][i, t])
            if a < 0 or b < 0:
                continue
            di = float(delta[i, a])
            dj = float(delta[i, b])
            dij = float(effects["pair_dij"][i, t])
            if di < di_max and dj < di_max and dij > dij_min and dij > di + dj + synergy:
                union(a, b)
        groups: dict[int, list[int]] = {}
        for p in range(int(hlen[i])):
            groups.setdefault(find(p), []).append(p)
        for members in groups.values():
            if len(members) <= 1:
                continue
            rep = max(members, key=lambda p: float(spec[i, p]))
            for p in members:
                if p != rep:
                    suppressed[i, p] = True
    return suppressed


def build_masks(
    probe,
    store: Store,
    indices: np.ndarray,
    phi: np.ndarray,
    spec: np.ndarray,
    suppressed: np.ndarray,
    eps_p: float,
    rank_slack: int,
    batch_size: int = 128,
) -> np.ndarray:
    probe.eval()
    length = store.max_l
    masks = np.zeros((len(indices), length), dtype=np.bool_)
    device = store.device
    for start in tqdm(range(0, len(indices), batch_size), desc="misset", leave=False):
        loc = indices[start:start + batch_size]
        bsz = len(loc)
        idxs = torch.tensor(loc, device=device)
        view = store.view(idxs)
        emb, role, pos = view["emb"], view["role"], view["pos"]
        valid = view["valid"]
        y, resp, cand = view["y"], view["resp"], view["cand"]
        phi_b = torch.tensor(phi[loc], device=device)
        spec_b = torch.tensor(spec[loc], device=device)
        sup_b = torch.tensor(suppressed[loc], device=device)
        valid_b = valid > 0.5
        phi_b = phi_b.masked_fill(~valid_b, -1e9)
        spec_b = spec_b.masked_fill(~valid_b, -1e9)
        with torch.inference_mode():
            ref = forward_metrics(probe, emb, role, pos, valid, y, resp, cand)
        selected = torch.zeros(bsz, length, dtype=torch.bool, device=device)
        active = torch.ones(bsz, dtype=torch.bool, device=device)
        score = phi_b.masked_fill(sup_b, -1e9)
        ar = torch.arange(bsz, device=device)

        def _ok(keep_bool):
            met = forward_metrics(probe, emb, role, pos, keep_bool.float(), y, resp, cand)
            return suffices(met, ref, y, eps_p, rank_slack)

        with torch.inference_mode():
            for _ in range(length):
                if not active.any():
                    break
                nxt = score.argmax(dim=1)
                can = score.max(dim=1).values > -1e8
                take = active & can
                if not take.any():
                    break
                selected[ar[take], nxt[take]] = True
                score[ar[take], nxt[take]] = -1e9
                ok = _ok(selected)
                active = active & ~ok & can

            need = ~_ok(selected)
            score2 = spec_b.masked_fill(~sup_b | ~valid_b | selected, -1e9)
            active = need
            for _ in range(length):
                if not active.any():
                    break
                nxt = score2.argmax(dim=1)
                can = score2.max(dim=1).values > -1e8
                take = active & can
                if not take.any():
                    break
                selected[ar[take], nxt[take]] = True
                score2[ar[take], nxt[take]] = -1e9
                ok = _ok(selected)
                active = active & ~ok & can

            still = ~_ok(selected)
            if still.any():
                selected[still] = valid_b[still]

            locked = torch.zeros_like(selected)
            for _ in range(length):
                removable = selected & ~locked
                has = removable.any(dim=1)
                if not has.any():
                    break
                tmp = phi_b.masked_fill(~removable, 1e9)
                weak = tmp.argmin(dim=1)
                trial = selected.clone()
                trial[ar[has], weak[has]] = False
                ok = _ok(trial)
                commit = has & ok
                fail = has & ~ok
                if not commit.any() and not fail.any():
                    break
                if commit.any():
                    selected = torch.where(commit[:, None], trial, selected)
                if fail.any():
                    locked[ar[fail], weak[fail]] = True
        masks[start:start + bsz] = selected.cpu().numpy()
    return masks


def utility_gap_stats(effects: dict, hlen: np.ndarray) -> dict:
    valid = np.arange(effects["delta_del"].shape[1])[None, :] < hlen[:, None]
    del_v = effects["delta_del"][valid]
    emo_v = effects["delta_emo"][valid]
    bg_v = effects["delta_bg"][valid]
    return {
        "mean_abs_del": float(np.mean(np.abs(del_v))),
        "mean_pos_del": float(np.mean(np.maximum(del_v, 0))),
        "frac_emo_pos": float(np.mean(emo_v > 0.02)),
        "frac_bg_pos": float(np.mean(bg_v > 0.02)),
        "frac_del_pos": float(np.mean(del_v > 0.05)),
        "p50_abs_del": float(np.percentile(np.abs(del_v), 50)),
        "p95_abs_del": float(np.percentile(np.abs(del_v), 95)),
        "p95_pos_emo": float(np.percentile(emo_v, 95)),
        "p95_pos_bg": float(np.percentile(bg_v, 95)),
    }
