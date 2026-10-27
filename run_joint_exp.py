"""Train the joint exchange model and compare it with budget-matched selectors."""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store
from src.joint_exchange import ExchangeNet, collect_masks, exchange_keep, train_joint
from src.prepare import build_cache
from src.train_eval import LABELS, pack_metrics, run_fixed, split_indices

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "joint")


def topk_from_scores(scores, hlen, k):
    mask = np.zeros_like(scores, dtype=bool)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        kk = min(k, n)
        if kk:
            mask[i, np.argsort(-scores[i, :n])[:kk]] = True
    return mask


def last_mask(hlen, k, length):
    pos = np.arange(length)[None, :]
    return (pos >= (hlen[:, None] - k)) & (pos < hlen[:, None])


def model_scores(model, store, idx, batch_size, shuffled=False, inserted=False):
    chunks = []
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        emb, role, valid = view["emb"], view["role"], view["valid"]
        if shuffled or inserted:
            emb, role, valid = _perturb(store, view, batch, shuffled, inserted)
        with torch.inference_mode():
            s = model.turn_scores(emb, role, valid)
        chunks.append(s.float().cpu().numpy())
    return np.concatenate(chunks)


def _perturb(store, view, batch, shuffled, inserted):
    emb, role, valid = view["emb"].clone(), view["role"].clone(), view["valid"].clone()
    if shuffled:
        for i, inst in enumerate(batch):
            n = int(view["hlen"][i])
            if n > 1:
                perm = torch.tensor(np.random.default_rng(50_000 + int(inst)).permutation(n), device=emb.device)
                emb[i, :n] = view["emb"][i, perm]
                role[i, :n] = view["role"][i, perm]
    if inserted:
        donor = store.emb[store.ins_donor[torch.tensor(batch, device=store.device)].clamp(min=0)]
        at = (view["hlen"] - 1).clamp(min=0)
        rows = torch.arange(len(batch), device=emb.device)
        emb = emb.clone()
        emb[rows, at] = donor
    return emb, role, valid


def _predict_perturbed(model, store, idx, k, cond, kind, batch_size):
    preds = []
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        emb, role, valid = _perturb(store, view, batch, cond == "shuffle", cond == "insert")
        hlen = valid.sum(1).cpu().numpy().astype(int)
        if kind == "joint":
            with torch.inference_mode():
                keep, _s, _c, _o, _can = exchange_keep(
                    model, emb, role, view["pos"], valid, k, tau=0.0
                )
            mask = keep.cpu().numpy() > 0.5
        elif kind == "last":
            mask = last_mask(hlen, k, emb.shape[1])
        else:
            last = emb[torch.arange(len(batch), device=emb.device), (valid.sum(1).long() - 1).clamp(min=0)]
            sc = torch.einsum("bld,bd->bl", emb, last).float().cpu().numpy()
            sc = np.where(valid.cpu().numpy() > 0.5, sc, -1e4)
            mask = topk_from_scores(sc, hlen, k)
        keep = torch.tensor(mask, device=store.device, dtype=torch.float32)
        with torch.inference_mode():
            logits, _q = model.dec(emb, role, view["pos"], keep)
        preds.append(logits.argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def predict(model, store, idx, mask, batch_size):
    return run_fixed(model.dec, store, idx, mask, batch_size)


def main():
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda")
    cache = build_cache(ROOT, 0, 42)
    store = Store(cache, device)
    tr, dv, te = (split_indices(cache, s) for s in (0, 1, 2))
    seeds = [0, 1, 2]
    k = 4
    models = []
    for seed in seeds:
        path = os.path.join(RES, f"model_s{seed}.pt")
        model = ExchangeNet(store.emb.shape[1]).to(device)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            print("loaded", path)
        else:
            model = train_joint(store, tr, dv, 8, 6, 128, 3e-4, seed, device, k=k)
            torch.save(model.state_dict(), path)
        models.append(model)

    y = cache["y"][te]
    hl = cache["hlen"][te]
    length = store.max_l
    rows = {name: [] for name in ("Joint", "LastK", "RandomK", "SimK", "AttnK", "TfidfK")}
    cond_f1 = {c: {n: [] for n in ("Joint", "LastK", "SimK")} for c in ("shuffle", "insert")}
    for seed, model in enumerate(models):
        model.eval()
        y_dev = cache["y"][dv]
        hl_dev = cache["hlen"][dv]
        chosen_tau, chosen_f1 = 0.0, -1.0
        for tau in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 10.0):
            dev_mask = collect_masks(model, store, dv, k, 128, tau, device)
            pred_d, _rank_d = predict(model, store, dv, dev_mask, 128)
            f1_d = float(f1_score(y_dev, pred_d, average="macro", labels=LABELS, zero_division=0))
            if f1_d > chosen_f1:
                chosen_tau, chosen_f1 = tau, f1_d
        print(f"seed {seed} dev tau {chosen_tau} F1 {chosen_f1*100:.2f}")
        view_scores = {
            "SimK": cache["sim_scores"][te],
            "TfidfK": cache["tfidf_scores"][te],
        }
        attn = []
        for start in range(0, len(te), 128):
            batch = te[start:start + 128]
            view = store.view(torch.tensor(batch, device=device))
            with torch.inference_mode():
                _lg, _q, w = model.dec(view["emb"], view["role"], view["pos"], view["valid"], return_weights=True)
            attn.append(w.float().cpu().numpy())
        view_scores["AttnK"] = np.concatenate(attn)
        rng = np.random.default_rng(1000 + seed)
        rand = np.full((len(te), length), -1e4, dtype=np.float32)
        for i, n in enumerate(hl.tolist()):
            rand[i, :int(n)] = rng.random(int(n))
        view_scores["RandomK"] = rand
        masks = {
            "Joint": collect_masks(model, store, te, k, 128, chosen_tau, device),
            "LastK": last_mask(hl, k, length),
        }
        for name, sc in view_scores.items():
            masks[name] = topk_from_scores(sc, hl, k)
        for name, mask in masks.items():
            pred, rank = predict(model, store, te, mask, 128)
            met = pack_metrics(y, pred, rank, mask.sum(1), hl)
            met["pred"] = pred
            rows[name].append(met)
            print(f"seed {seed} {name} F1 {met['macro_f1']*100:.2f} MRR {met['mrr']*100:.2f}")
        for cond in ("shuffle", "insert"):
            pred_j = _predict_perturbed(model, store, te, k, cond, "joint", 128)
            pred_l = _predict_perturbed(model, store, te, k, cond, "last", 128)
            pred_s = _predict_perturbed(model, store, te, k, cond, "sim", 128)
            for name, pred in (("Joint", pred_j), ("LastK", pred_l), ("SimK", pred_s)):
                cond_f1[cond][name].append(float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)))

    def agg(vals):
        return float(np.mean(vals)), float(np.std(vals))

    lines = ["# Joint exchange, test, k=4, three seeds", ""]
    lines.append("| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR |")
    lines.append("|---|---|---|---|---|---|---|")
    summary = {}
    for name, mets in rows.items():
        summary[name] = {
            key: agg([m[key] for m in mets]) for key in ("macro_f1", "weighted_f1", "r1", "r5", "mrr", "mmr")
        }
        s = summary[name]
        lines.append(
            f"| {name} | {100*s['macro_f1'][0]:.2f} ± {100*s['macro_f1'][1]:.2f} | "
            f"{100*s['weighted_f1'][0]:.2f} ± {100*s['weighted_f1'][1]:.2f} | "
            f"{100*s['r1'][0]:.2f} ± {100*s['r1'][1]:.2f} | "
            f"{100*s['r5'][0]:.2f} ± {100*s['r5'][1]:.2f} | "
            f"{100*s['mrr'][0]:.2f} ± {100*s['mrr'][1]:.2f} | "
            f"{100*s['mmr'][0]:.2f} ± {100*s['mmr'][1]:.2f} |"
        )
    lines.append("")
    lines.append("| Condition | Joint | LastK | SimK |")
    lines.append("|---|---|---|---|")
    for cond, table in cond_f1.items():
        cell = []
        for name in ("Joint", "LastK", "SimK"):
            m, s = agg(table[name])
            cell.append(f"{100*m:.2f} ± {100*s:.2f}")
        lines.append(f"| {cond} | " + " | ".join(cell) + " |")
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        ix = rng.integers(0, len(y), len(y))
        seed_i = int(rng.integers(0, len(seeds)))
        a = rows["Joint"][seed_i]["pred"]
        b = rows["LastK"][seed_i]["pred"]
        diffs.append(
            f1_score(y[ix], a[ix], average="macro", labels=LABELS, zero_division=0)
            - f1_score(y[ix], b[ix], average="macro", labels=LABELS, zero_division=0)
        )
    d = np.asarray(diffs)
    lines.append("")
    lines.append(
        f"Bootstrap Joint minus LastK: mean {100*d.mean():.2f}, "
        f"95% CI [{100*np.quantile(d, 0.025):.2f}, {100*np.quantile(d, 0.975):.2f}], "
        f"P(diff<=0)={float(np.mean(d <= 0)):.3f}"
    )
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "bootstrap_mean": float(d.mean()),
                   "bootstrap_low": float(np.quantile(d, 0.025)),
                   "bootstrap_high": float(np.quantile(d, 0.975))}, f, indent=2)


if __name__ == "__main__":
    main()
