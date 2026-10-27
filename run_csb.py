"""CSB experiments: shared robust decision model, baselines, CSB, ablations, Pareto."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.causal import Store, build_masks, compute_effects, factor_scores, forward_metrics, substitute_groups
from src.csb import PriorGate, calibrate_adaptive, exchange_masks, train_exchange, train_robust_probe
from src.modeling import Probe, Selector
from src.prepare import build_cache
from src.train_eval import (
    LABELS,
    _attn_mask,
    _lastk_mask,
    _lex_scores,
    _make_views,
    _sim_scores,
    _tfidf_scores,
    _topk_mask,
    split_indices,
    train_selector,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "csb")


def _cmi_live(probe, view):
    """CMI rule scored by the decision model's confidence, recomputed on each view."""
    valid = view["valid"]
    bsz, length = valid.shape
    with torch.inference_mode():
        empty = forward_metrics(probe, view["emb"], view["role"], view["pos"], torch.zeros_like(valid),
                                view["y"], view["resp"], view["cand"])["conf"]
        out = torch.full((bsz, length), -1e4, device=valid.device)
        for p in range(length):
            if not (valid[:, p] > 0.5).any():
                continue
            only = torch.zeros_like(valid)
            only[:, p] = valid[:, p]
            one = forward_metrics(probe, view["emb"], view["role"], view["pos"], only,
                                  view["y"], view["resp"], view["cand"])["conf"]
            emb_e = view["emb"].clone()
            emb_e[:, p] = view["emb_emo"][:, p]
            one_e = forward_metrics(probe, emb_e, view["role"], view["pos"], only,
                                    view["y"], view["resp"], view["cand"])["conf"]
            out[:, p] = (one - empty) + 0.25 * (one - one_e)
    return out.masked_fill(valid < 0.5, -1e4)


def _gate(model, view):
    with torch.inference_mode():
        return model(view["emb"], view["role"], view["pos"], view["valid"])


def _adaptive(scores, valid, hlen, q):
    s = scores.masked_fill(valid < 0.5, -1e4)
    order = s.argsort(dim=1, descending=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(s.shape[1], device=s.device)[None].expand_as(order))
    kk = torch.clamp(torch.ceil(q * hlen.float()), min=1).long()
    kk = torch.minimum(kk, hlen)
    return (ranks < kk[:, None]) & (valid > 0.5)


def evaluate(probe, store, idx, k, gates, conds=("clean", "emo", "shuffle", "insert"), only=None,
             batch_size=64):
    """gates: name -> dict(kind='topk'|'tau'|'adaptive', model, tau/q)."""
    cache = store.cache
    rec = {}
    y_all = []
    for start in tqdm(range(0, len(idx), batch_size), desc=f"eval k={k}", leave=False):
        batch = idx[start:start + batch_size]
        views, origins = _make_views(store, batch)
        y_all.append(views["clean"]["y"].cpu().numpy())
        for cond in conds:
            view = views[cond]
            valid, hlen = view["valid"], view["hlen"]
            L = view["emb"].shape[1]
            ids_np = view["ids"].cpu().numpy()
            hl_np = hlen.cpu().numpy()
            masks = {}

            def want(n):
                return only is None or n in only

            if want("LastK"):
                masks["LastK"] = _lastk_mask(hlen, L, k)
            if want("RandomK"):
                for sd in range(3):
                    cid = {"clean": 1, "emo": 2, "shuffle": 3, "insert": 4}[cond]
                    r = np.random.default_rng(1000 * sd + start + 97 * cid)
                    m = np.zeros((len(batch), L), dtype=bool)
                    for i, n in enumerate(hl_np.tolist()):
                        m[i, r.choice(int(n), size=min(k, int(n)), replace=False)] = True
                    masks[f"RandomK#{sd}"] = torch.tensor(m, device=store.device)
            if want("SimK"):
                masks["SimK"] = _topk_mask(_sim_scores(view), valid, k)
            if want("EmoK"):
                masks["EmoK"] = _topk_mask(torch.tensor(
                    _lex_scores(cache, ids_np, hl_np, cond == "emo"), device=store.device), valid, k)
            if want("TfidfK"):
                masks["TfidfK"] = _topk_mask(torch.tensor(
                    _tfidf_scores(cache, ids_np, hl_np, cond == "emo"), device=store.device), valid, k)
            if want("AttnK"):
                masks["AttnK"] = _attn_mask(probe, view, k)
            if want("CMIConfK"):
                masks["CMIConfK"] = _topk_mask(_cmi_live(probe, view), valid, k)
            for name, g in gates.items():
                base = name.split("#")[0]
                if not want(base):
                    continue
                s = _gate(g["model"], view)
                if g["kind"] == "topk":
                    masks[name] = _topk_mask(s, valid, k)
                elif g["kind"] == "tau":
                    masks[name] = (torch.sigmoid(s) >= g["tau"]) & (valid > 0.5)
                else:
                    masks[name] = _adaptive(s, valid, hlen, g["q"])
            if want("FullHist"):
                masks["FullHist"] = valid > 0.5
            donor = None
            if cond == "insert":
                donor = torch.tensor(origins["insert"] == -2, device=store.device)
            for name, mask in masks.items():
                with torch.inference_mode():
                    met = forward_metrics(probe, view["emb"], view["role"], view["pos"], mask.float(),
                                          view["y"], view["resp"], view["cand"])
                r = rec.setdefault(name, {}).setdefault(cond, {"pred": [], "rank": [], "count": [],
                                                               "donor": [], "ids": []})
                m = mask if donor is None else (mask & ~donor)
                r["pred"].append(met["pred"].cpu().numpy())
                r["rank"].append(met["rank"].cpu().numpy())
                r["count"].append(m.sum(1).cpu().numpy())
                r["donor"].append((mask & donor).any(1).cpu().numpy() if donor is not None
                                  else np.zeros(len(batch), dtype=bool))
                mn = mask.cpu().numpy()
                r["ids"].extend([set(ids_np[i][mn[i]].tolist()) for i in range(len(batch))])
    y = np.concatenate(y_all)
    hl = store.cache["hlen"][idx]
    out = {}
    for name, conds_rec in rec.items():
        row = {}
        cat = {c: {kk: (np.concatenate(v) if kk != "ids" else v) for kk, v in d.items()}
               for c, d in conds_rec.items()}
        for c, d in cat.items():
            rank = d["rank"].astype(float)
            row[c] = {
                "macro_f1": float(f1_score(y, d["pred"], average="macro", labels=LABELS, zero_division=0)),
                "weighted_f1": float(f1_score(y, d["pred"], average="weighted", labels=LABELS, zero_division=0)),
                "r1": float(np.mean(rank <= 1)),
                "r5": float(np.mean(rank <= 5)),
                "mrr": float(np.mean(1.0 / rank)),
                "mmr": float(np.mean(d["count"] / np.maximum(hl, 1))),
                "donor_rate": float(np.mean(d["donor"])),
            }
        row["_pred"] = cat["clean"]["pred"]
        if "emo" in cat and "insert" in cat:
            ok = cat["clean"]["pred"] == y
            stab = [np.mean(cat[c]["pred"][ok] == y[ok]) for c in ("emo", "insert")]
            row["cc"] = float(np.mean(stab))
        if "shuffle" in cat:
            js = [len(a & b) / max(1, len(a | b)) for a, b in zip(cat["clean"]["ids"], cat["shuffle"]["ids"])]
            row["selstab"] = float(np.mean(js))
        out[name] = row
    out["_y"] = y
    return out


def merge_seeds(res):
    """Average rows named X#s into X with std."""
    groups = {}
    for name in list(res):
        if name.startswith("_"):
            continue
        groups.setdefault(name.split("#")[0], []).append(name)
    merged = {}
    for base, names in groups.items():
        rows = [res[n] for n in names]

        def agg(get):
            vals = np.array([get(r) for r in rows])
            return float(vals.mean()), float(vals.std())

        m = {"n_seeds": len(rows), "_preds": [r["_pred"] for r in rows]}
        for c in ("clean", "emo", "shuffle", "insert"):
            if c in rows[0]:
                m[c] = {key: agg(lambda r, c=c, key=key: r[c][key]) for key in rows[0][c]}
        for key in ("cc", "selstab"):
            if key in rows[0]:
                m[key] = agg(lambda r, key=key: r[key])
        merged[base] = m
    return merged


def bootstrap(y, a_list, b_list, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    for t in range(n):
        ix = rng.integers(0, len(y), len(y))
        a = a_list[t % len(a_list)]
        b = b_list[t % len(b_list)]
        diffs.append(f1_score(y[ix], a[ix], average="macro", labels=LABELS, zero_division=0)
                     - f1_score(y[ix], b[ix], average="macro", labels=LABELS, zero_division=0))
    d = np.array(diffs)
    return {"mean": float(d.mean()), "low": float(np.quantile(d, 0.025)), "high": float(np.quantile(d, 0.975)),
            "p_le_0": float(np.mean(d <= 0))}


def fmt(v, std=True):
    m, s = v
    return f"{100*m:.2f}±{100*s:.2f}" if std and s > 0 else f"{100*m:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-epochs", type=int, default=16)
    ap.add_argument("--csb-epochs", type=int, default=10)
    ap.add_argument("--sel-epochs", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    cache = build_cache(ROOT, limit_dialogs=0, seed=42)
    store = Store(cache, device)
    tr, dv, te = (split_indices(cache, s) for s in (0, 1, 2))
    log = {}

    ppath = os.path.join(RES, "probe_robust.pt")
    if os.path.exists(ppath):
        probe = Probe(n_strat=8, in_dim=768).to(device)
        probe.load_state_dict(torch.load(ppath, map_location=device, weights_only=True))
        probe.eval()
        for p in probe.parameters():
            p.requires_grad_(False)
        print("loaded robust probe")
    else:
        probe = train_robust_probe(store, tr, dv, args.probe_epochs, args.batch_size, 5e-4, 42, device)
        torch.save(probe.state_dict(), ppath)

    # Distilled MISSET selector from the previous pipeline, rebuilt on this probe.
    mpath = os.path.join(RES, "misset_masks.npz")
    need_misset = any(not os.path.exists(os.path.join(RES, f"caumem_s{sd}.pt")) for sd in range(args.seeds))
    misset = None
    if need_misset:
        if os.path.exists(mpath) and os.path.getsize(mpath) > 1000:
            misset = np.load(mpath)["MISSET"]
        else:
            eff = compute_effects(probe, store, batch_size=128)
            phi, spec = factor_scores(eff, "full", cache["hlen"])
            sup = substitute_groups(eff, spec, cache["hlen"])
            misset = build_masks(probe, store, np.arange(len(cache["y"])), phi, spec, sup, 0.15, 5, 128)
            np.savez_compressed(mpath, MISSET=misset)
        print("MISSET train density", float(misset[tr].sum() / cache["hlen"][tr].sum()))

    gates = {}
    for sd in range(args.seeds):
        path = os.path.join(RES, f"caumem_s{sd}.pt")
        sel = Selector(in_dim=768).to(device)
        if os.path.exists(path):
            ck = torch.load(path, map_location=device, weights_only=False)
            sel.load_state_dict(ck["state"])
            tau = ck["tau"]
        else:
            sel, tau = train_selector(probe, store, tr, dv, misset, args.sel_epochs, args.batch_size, args.lr,
                                      42 + sd, device, use_cf=True, name=f"caumem_s{sd}")
            torch.save({"state": sel.state_dict(), "tau": tau}, path)
        sel.eval()
        gates[f"CauMemK#{sd}"] = {"kind": "topk", "model": sel}
        gates[f"CauMem#{sd}"] = {"kind": "tau", "model": sel, "tau": tau}

    xmask_path = os.path.join(RES, f"exchange_k{args.k}.npz")
    if os.path.exists(xmask_path) and os.path.getsize(xmask_path) > 1000:
        exchange = np.load(xmask_path)["mask"]
    else:
        exchange = exchange_masks(probe, store, np.arange(len(cache["y"])), args.k)
        np.savez_compressed(xmask_path, mask=exchange)
    changed = exchange[tr] != ((np.arange(cache["max_l"])[None, :] >= cache["hlen"][tr, None] - args.k)
                               & (np.arange(cache["max_l"])[None, :] < cache["hlen"][tr, None]))
    print("train instances with an exchange", int(changed.any(1).mean() * 1000) / 10)

    variants = {
        "CSB": {},
        "CSB-noEnv": {"use_env": False},
        "CSB-noPrior": {"use_prior": False},
    }
    csb_models = {}
    for vname, kw in variants.items():
        n_seed = args.seeds if vname == "CSB" else 1
        for sd in range(n_seed):
            path = os.path.join(RES, f"{vname}_s{sd}.pt")
            g = (Selector(in_dim=768) if not kw.get("use_prior", True) else PriorGate(768, beta=0.15)).to(device)
            if os.path.exists(path):
                g.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            else:
                g = train_exchange(
                    probe, store, tr, dv, exchange, args.csb_epochs, args.batch_size, args.lr,
                    42 + sd, device, name=f"{vname}_s{sd}", eval_k=args.k, **kw,
                )
                torch.save(g.state_dict(), path)
            g.eval()
            csb_models[(vname, sd)] = g
            key = "CSB-K" if vname == "CSB" else vname
            gates[f"{key}#{sd}"] = {"kind": "topk", "model": g}

    for alpha in (0.10, 0.20):
        qs = []
        for sd in range(args.seeds):
            q = calibrate_adaptive(csb_models[("CSB", sd)], probe, store, dv, alpha, args.batch_size)
            qs.append(q)
            gates[f"CSB-a{int(alpha*100)}#{sd}"] = {"kind": "adaptive", "model": csb_models[("CSB", sd)], "q": q}
        log[f"conformal_q_alpha{alpha}"] = qs
        print("alpha", alpha, "q", qs)

    k = args.k
    print("evaluating test")
    res = evaluate(probe, store, te, k, gates)
    y = res["_y"]
    merged = merge_seeds(res)

    # Pareto over budgets, clean condition.
    pareto = {}
    pareto_gates = {n: g for n, g in gates.items() if g["kind"] == "topk"}
    for kk in (1, 2, 3, 6, 8):
        r = evaluate(probe, store, te, kk, pareto_gates, conds=("clean",),
                     only={"LastK", "SimK", "AttnK", "CMIConfK", "CauMemK", "CSB-K", "TfidfK"})
        pareto[kk] = {n: v["clean"] for n, v in merge_seeds(r).items()}
    pareto[k] = {n: merged[n]["clean"] for n in ("LastK", "SimK", "AttnK", "CMIConfK", "CauMemK", "CSB-K", "TfidfK")}

    # Long-history slice.
    hl = cache["hlen"][te]
    slices = {"short(1-8)": hl <= 8, "mid(9-20)": (hl >= 9) & (hl <= 20), "long(>=21)": hl >= 21}
    slice_rows = {}
    for sname, msk in slices.items():
        for name in ("FullHist", "LastK", "SimK", "AttnK", "CMIConfK", "CauMemK", "CSB-K"):
            preds = merged[name]["_preds"]
            f = [f1_score(y[msk], p[msk], average="macro", labels=LABELS, zero_division=0) for p in preds]
            slice_rows.setdefault(sname, {})[name] = (float(np.mean(f)), float(np.std(f)), int(msk.sum()))

    boots = {}
    for other in ("LastK", "SimK", "AttnK", "CMIConfK", "TfidfK", "CauMemK", "RandomK"):
        boots[other] = bootstrap(y, merged["CSB-K"]["_preds"], merged[other]["_preds"])
    boots["FullHist"] = bootstrap(y, merged["CSB-K"]["_preds"], merged["FullHist"]["_preds"])
    for sname, msk in slices.items():
        boots[f"LastK@{sname}"] = bootstrap(y[msk], [p[msk] for p in merged["CSB-K"]["_preds"]],
                                            [p[msk] for p in merged["LastK"]["_preds"]])

    # Tables.
    L = []
    L.append(f"# Main table (test, k={k}, mean±std over seeds where applicable)\n")
    L.append("| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR | HCR | CC | MacF1-shuf | MacF1-ins | MRR-shuf | MRR-ins | Donor | SelStab |")
    L.append("|" + "---|" * 15)
    order = ["LastK", "RandomK", "SimK", "EmoK", "TfidfK", "AttnK", "CMIConfK", "CauMemK", "CauMem",
             "CSB-K", "CSB-a20", "CSB-a10"]
    for n in order:
        r = merged[n]
        c = r["clean"]
        hcr = (1 - c["mmr"][0], c["mmr"][1])
        L.append(
            f"| {n} | {fmt(c['macro_f1'])} | {fmt(c['weighted_f1'])} | {fmt(c['r1'])} | {fmt(c['r5'])} | "
            f"{fmt(c['mrr'])} | {fmt(c['mmr'])} | {fmt(hcr)} | {fmt(r['cc'])} | {fmt(r['shuffle']['macro_f1'])} | "
            f"{fmt(r['insert']['macro_f1'])} | {fmt(r['shuffle']['mrr'])} | {fmt(r['insert']['mrr'])} | "
            f"{fmt(r['insert']['donor_rate'])} | {fmt(r['selstab'])} |"
        )
    fh = merged["FullHist"]
    L.append("\nReference, full history (no selection): "
             f"MacF1 {fmt(fh['clean']['macro_f1'])}, W-F1 {fmt(fh['clean']['weighted_f1'])}, "
             f"R@1 {fmt(fh['clean']['r1'])}, R@5 {fmt(fh['clean']['r5'])}, MRR {fmt(fh['clean']['mrr'])}, "
             f"CC {fmt(fh['cc'])}, MacF1-shuf {fmt(fh['shuffle']['macro_f1'])}, "
             f"MacF1-ins {fmt(fh['insert']['macro_f1'])}, MRR-shuf {fmt(fh['shuffle']['mrr'])}, "
             f"MRR-ins {fmt(fh['insert']['mrr'])}\n")
    L.append("# Ablation (test, k=%d)\n" % k)
    L.append("| Variant | MacF1 | MRR | CC | MacF1-shuf | MacF1-ins | Donor | SelStab |")
    L.append("|" + "---|" * 8)
    for n in ("CSB-K", "CSB-noEnv", "CSB-noPrior", "CauMemK"):
        r = merged[n]
        L.append(f"| {n} | {fmt(r['clean']['macro_f1'])} | {fmt(r['clean']['mrr'])} | {fmt(r['cc'])} | "
                 f"{fmt(r['shuffle']['macro_f1'])} | {fmt(r['insert']['macro_f1'])} | "
                 f"{fmt(r['insert']['donor_rate'])} | {fmt(r['selstab'])} |")
    L.append("\n# Budget sweep (test, clean MacF1 / MRR)\n")
    names = ["LastK", "SimK", "TfidfK", "AttnK", "CMIConfK", "CauMemK", "CSB-K"]
    L.append("| k | " + " | ".join(names) + " |")
    L.append("|" + "---|" * (len(names) + 1))
    for kk in sorted(pareto):
        L.append(f"| {kk} | " + " | ".join(
            f"{fmt(pareto[kk][n]['macro_f1'], False)} / {fmt(pareto[kk][n]['mrr'], False)}" for n in names) + " |")
    L.append("\n# History-length slices (test MacF1, k=%d)\n" % k)
    snames = ["FullHist", "LastK", "SimK", "AttnK", "CMIConfK", "CauMemK", "CSB-K"]
    L.append("| Slice | n | " + " | ".join(snames) + " |")
    L.append("|" + "---|" * (len(snames) + 2))
    for sname, rows in slice_rows.items():
        n = next(iter(rows.values()))[2]
        L.append(f"| {sname} | {n} | " + " | ".join(fmt(rows[s][:2]) for s in snames) + " |")
    L.append("\n# Paired bootstrap, MacF1(CSB-K) minus other (1000 resamples)\n")
    L.append("| Other | mean | 95% CI | P(diff<=0) |")
    L.append("|---|---|---|---|")
    for n, b in boots.items():
        L.append(f"| {n} | {100*b['mean']:.2f} | [{100*b['low']:.2f}, {100*b['high']:.2f}] | {b['p_le_0']:.3f} |")
    L.append(f"\nconformal q: {json.dumps(log)}")
    txt = "\n".join(L)
    print(txt)
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write(txt)

    def clean_json(o):
        if isinstance(o, dict):
            return {str(k2): clean_json(v) for k2, v in o.items() if not str(k2).startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean_json(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(clean_json({"main": merged, "pareto": pareto, "slices": slice_rows,
                              "bootstrap": boots, "log": log}), f, indent=1)
    print("wrote", RES)


if __name__ == "__main__":
    main()
