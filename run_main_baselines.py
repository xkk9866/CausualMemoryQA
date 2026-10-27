"""Literature memory baselines on the same robust decision model.

Budgets match the main table. Decay and MMR each have one coefficient,
chosen on the development split at k=4 and then frozen.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import LABELS, pack_metrics, run_fixed, split_indices
from run_sx import exchange

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "baselines")
TOKEN = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def last_mask(hlen, k, length):
    pos = np.arange(length)[None, :]
    return (pos >= (hlen[:, None] - k)) & (pos < hlen[:, None])


def full_mask(hlen, length):
    return np.arange(length)[None, :] < hlen[:, None]


def head_tail_mask(hlen, k, length):
    mask = np.zeros((len(hlen), length), dtype=bool)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        kk = min(k, n)
        head = kk // 2
        tail = kk - head
        if head:
            mask[i, :head] = True
        if tail:
            mask[i, n - tail:n] = True
    return mask


def topk_mask(scores, hlen, k):
    mask = np.zeros_like(scores, dtype=bool)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        kk = min(k, n)
        if kk:
            mask[i, np.argsort(-scores[i, :n])[:kk]] = True
    return mask


def summary_mask(scores, hlen, k):
    """Keep the recent k-1 turns and one earlier turn, the extractive memory slot."""
    mask = last_mask(hlen, max(k - 1, 1), scores.shape[1])
    if k <= 1:
        return last_mask(hlen, 1, scores.shape[1])
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        if n <= k:
            mask[i, :n] = True
            continue
        early = scores[i, : n - (k - 1)]
        mask[i, int(np.argmax(early))] = True
    return mask


def mmr_mask(emb_hist, hlen, sim, k, lam):
    """Greedy MMR. Relevance is cosine to the latest turn."""
    mask = np.zeros_like(sim, dtype=bool)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        kk = min(k, n)
        if kk == 0:
            continue
        chosen = [n - 1]
        mask[i, n - 1] = True
        vecs = emb_hist[i, :n]
        while len(chosen) < kk:
            best_j, best_s = 0, -1e9
            for j in range(n):
                if j in chosen:
                    continue
                red = max(float(vecs[j] @ vecs[c]) for c in chosen)
                value = lam * float(sim[i, j]) - (1.0 - lam) * red
                if value > best_s:
                    best_j, best_s = j, value
            chosen.append(best_j)
            mask[i, best_j] = True
    return mask


def ppr_mask(emb_hist, hlen, k, alpha=0.85):
    """Personalized PageRank from the latest turn on the nonnegative similarity graph."""
    scores = np.full_like(emb_hist[:, :, 0], -1e4, dtype=np.float32)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        if n == 0:
            continue
        vec = emb_hist[i, :n]
        sim = vec @ vec.T
        np.fill_diagonal(sim, 0.0)
        sim = np.clip(sim, 0.0, None)
        deg = sim.sum(1, keepdims=True)
        empty = deg[:, 0] <= 1e-8
        deg[empty] = 1.0
        trans = sim / deg
        rank = np.zeros(n, dtype=np.float64)
        rank[-1] = 1.0
        tele = rank.copy()
        for _ in range(12):
            rank = alpha * (trans.T @ rank) + (1.0 - alpha) * tele
        rank[empty] = 0.0
        scores[i, :n] = rank.astype(np.float32)
    return topk_mask(scores, hlen, k)


def bm25_matrix(cache, idx):
    texts = cache["texts"]
    hist = cache["hist"]
    hlen = cache["hlen"][idx]
    train_turns = np.where(cache["turn_split"] == 0)[0]
    df = Counter()
    n_docs = 0
    lengths = []
    for tid in train_turns.tolist():
        toks = tokenize(texts[int(tid)])
        if not toks:
            continue
        n_docs += 1
        lengths.append(len(toks))
        df.update(set(toks))
    avgdl = float(np.mean(lengths)) if lengths else 1.0
    n_docs = max(n_docs, 1)
    idf = {w: math.log((n_docs - c + 0.5) / (c + 0.5) + 1.0) for w, c in df.items()}
    k1, b = 1.5, 0.75
    scores = np.full((len(idx), cache["max_l"]), -1e4, dtype=np.float32)
    for i, inst in enumerate(idx.tolist()):
        n = int(hlen[i])
        ids = [int(x) for x in hist[inst, :n]]
        docs = [tokenize(texts[t]) for t in ids]
        query = Counter(docs[-1]) if docs else Counter()
        if not query:
            scores[i, :n] = np.linspace(0.0, 1.0, n, dtype=np.float32)
            continue
        raw = np.zeros(n, dtype=np.float32)
        for j, doc in enumerate(docs):
            if not doc:
                continue
            tf = Counter(doc)
            dl = len(doc)
            acc = 0.0
            for w, qf in query.items():
                if w not in tf:
                    continue
                freq = tf[w]
                denom = freq + k1 * (1.0 - b + b * dl / avgdl)
                acc += idf.get(w, 0.0) * freq * (k1 + 1.0) / denom
            raw[j] = acc
        scores[i, :n] = raw
    return scores


def decay_scores(sim, hlen, tau):
    length = sim.shape[1]
    pos = np.arange(length)[None, :]
    dist = (hlen[:, None] - 1 - pos).clip(min=0).astype(np.float32)
    out = sim.astype(np.float32) * np.exp(-dist / tau)
    out[pos >= hlen[:, None]] = -1e4
    return out


def rrf_scores(a, b, hlen, constant=60.0):
    """Reciprocal rank fusion of two per-turn score matrices."""
    out = np.full_like(a, -1e4, dtype=np.float32)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        if n == 0:
            continue
        ra = np.argsort(np.argsort(-a[i, :n]))
        rb = np.argsort(np.argsort(-b[i, :n]))
        out[i, :n] = 1.0 / (constant + ra + 1.0) + 1.0 / (constant + rb + 1.0)
    return out


def attention_scores(probe, store, idx, batch_size=128):
    chunks = []
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        with torch.inference_mode():
            _lg, _q, w = probe(view["emb"], view["role"], view["pos"], view["valid"], return_weights=True)
        chunks.append(w.float().cpu().numpy())
    scores = np.concatenate(chunks).astype(np.float32)
    hlen = store.cache["hlen"][idx]
    pos = np.arange(scores.shape[1])[None, :]
    scores[pos >= hlen[:, None]] = -1e4
    return scores


def hist_emb(store, idx, batch_size=256):
    chunks = []
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        chunks.append(view["emb"].float().cpu().numpy())
    return np.concatenate(chunks)


def evaluate(probe, store, idx, mask):
    y = store.cache["y"][idx]
    hl = store.cache["hlen"][idx]
    pred, rank = run_fixed(probe, store, idx, mask, 128)
    met = pack_metrics(y, pred, rank, mask.sum(1), hl)
    met["pred"] = pred
    return met


def fmt(met):
    return (
        f"{100*met['macro_f1']:.2f}",
        f"{100*met['weighted_f1']:.2f}",
        f"{100*met['r1']:.2f}",
        f"{100*met['r5']:.2f}",
        f"{100*met['mrr']:.2f}",
        f"{100*met['mmr']:.2f}",
    )


def main():
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda")
    cache = build_cache(ROOT, 0, 42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load(
        os.path.join(ROOT, "results", "csb", "probe_robust.pt"),
        map_location=device, weights_only=True,
    ))
    probe.eval()
    splits = {"dev": split_indices(cache, 1), "test": split_indices(cache, 2)}
    prepared = {}
    for name, idx in splits.items():
        print("preparing", name, flush=True)
        hlen = cache["hlen"][idx]
        sim = cache["sim_scores"][idx].astype(np.float32)
        tfidf = cache["tfidf_scores"][idx].astype(np.float32)
        emo = cache["emo_scores"][idx].astype(np.float32)
        bm25 = bm25_matrix(cache, idx)
        attn = attention_scores(probe, store, idx)
        emb = hist_emb(store, idx)
        prepared[name] = {
            "idx": idx, "hlen": hlen, "sim": sim, "tfidf": tfidf, "emo": emo,
            "bm25": bm25, "attn": attn, "emb": emb,
            "rrf": rrf_scores(bm25, sim, hlen),
        }

    dev = prepared["dev"]
    # One development choice each, at the main budget k=4.
    tau_choice, tau_f1 = 4.0, -1.0
    for tau in (2.0, 4.0, 8.0):
        mask = topk_mask(decay_scores(dev["sim"], dev["hlen"], tau), dev["hlen"], 4)
        f1 = evaluate(probe, store, dev["idx"], mask)["macro_f1"]
        print(f"dev decay tau {tau} F1 {100*f1:.2f}", flush=True)
        if f1 > tau_f1:
            tau_choice, tau_f1 = tau, f1
    lam_choice, lam_f1 = 0.5, -1.0
    for lam in (0.3, 0.5, 0.7):
        mask = mmr_mask(dev["emb"], dev["hlen"], dev["sim"], 4, lam)
        f1 = evaluate(probe, store, dev["idx"], mask)["macro_f1"]
        print(f"dev MMR lambda {lam} F1 {100*f1:.2f}", flush=True)
        if f1 > lam_f1:
            lam_choice, lam_f1 = lam, f1
    print(f"chosen tau {tau_choice} lambda {lam_choice}", flush=True)

    rows = {sp: {} for sp in splits}
    preds = {sp: {} for sp in splits}
    for sp, pack in prepared.items():
        idx, hlen = pack["idx"], pack["hlen"]
        length = cache["max_l"]
        for k in (2, 4, 6):
            methods = {
                "Full": full_mask(hlen, length),
                "Last": last_mask(hlen, k, length),
                "HeadTail": head_tail_mask(hlen, k, length),
                "Dense": topk_mask(pack["sim"], hlen, k),
                "BM25": topk_mask(pack["bm25"], hlen, k),
                "Tfidf": topk_mask(pack["tfidf"], hlen, k),
                "RRF": topk_mask(pack["rrf"], hlen, k),
                "Decay": topk_mask(decay_scores(pack["sim"], hlen, tau_choice), hlen, k),
                "MMR": mmr_mask(pack["emb"], hlen, pack["sim"], k, lam_choice),
                "PageRank": ppr_mask(pack["emb"], hlen, k),
                "Emotion": topk_mask(pack["emo"], hlen, k),
                "Attention": topk_mask(pack["attn"], hlen, k),
                "Summary": summary_mask(pack["bm25"], hlen, k),
                "SX": exchange(probe, store, idx, k, 8, "deploy"),
            }
            rng_mets = []
            for seed in (0, 1, 2):
                rng = np.random.default_rng(1000 + seed + k)
                rand = np.full((len(idx), length), -1e4, dtype=np.float32)
                for i, n in enumerate(hlen.tolist()):
                    rand[i, :int(n)] = rng.random(int(n))
                methods_seed = topk_mask(rand, hlen, k)
                met = evaluate(probe, store, idx, methods_seed)
                rng_mets.append(met)
                if seed == 0 and k == 4:
                    preds[sp]["Random"] = met["pred"]
            mean_rand = {
                key: float(np.mean([m[key] for m in rng_mets]))
                for key in ("macro_f1", "weighted_f1", "r1", "r5", "mrr", "mmr")
            }
            std_rand = float(np.std([m["macro_f1"] for m in rng_mets]))
            mean_rand["macro_std"] = std_rand
            rows[sp][f"Random@{k}"] = mean_rand
            for name, mask in methods.items():
                met = evaluate(probe, store, idx, mask)
                rows[sp][f"{name}@{k}"] = {key: met[key] for key in ("macro_f1", "weighted_f1", "r1", "r5", "mrr", "mmr")}
                if k == 4:
                    preds[sp][name] = met["pred"]
                print(f"{sp} {name}@{k} F1 {100*met['macro_f1']:.2f} MRR {100*met['mrr']:.2f}", flush=True)
            print(f"{sp} Random@{k} F1 {100*mean_rand['macro_f1']:.2f} ± {100*std_rand:.2f}", flush=True)

    # Long histories, test, k=4. Reuse predictions only for strategy F1.
    te = splits["test"]
    y = cache["y"][te]
    long = cache["hlen"][te] >= 21
    long_f1 = {}
    for name, pred in preds["test"].items():
        if name == "Random":
            continue
        long_f1[name] = float(f1_score(y[long], pred[long], average="macro", labels=LABELS, zero_division=0))
        print(f"long {name} n {int(long.sum())} F1 {100*long_f1[name]:.2f}", flush=True)

    rng = np.random.default_rng(0)
    boots = {}
    base = preds["test"]["Last"]
    for name, pred in preds["test"].items():
        if name == "Last":
            continue
        diffs = []
        for _ in range(1000):
            ix = rng.integers(0, len(y), len(y))
            diffs.append(
                f1_score(y[ix], pred[ix], average="macro", labels=LABELS, zero_division=0)
                - f1_score(y[ix], base[ix], average="macro", labels=LABELS, zero_division=0)
            )
        d = np.asarray(diffs)
        boots[name] = {
            "mean": float(d.mean()),
            "low": float(np.quantile(d, 0.025)),
            "high": float(np.quantile(d, 0.975)),
            "p_le_0": float(np.mean(d <= 0)),
        }
        print(f"boot {name} {100*d.mean():.2f} [{100*np.quantile(d, 0.025):.2f}, {100*np.quantile(d, 0.975):.2f}]", flush=True)

    order = ["Full", "Last", "HeadTail", "Random", "Dense", "BM25", "Tfidf", "RRF",
             "Decay", "MMR", "PageRank", "Emotion", "Attention", "Summary", "SX"]
    lines = [
        "# Main-protocol baselines, robust probe",
        "",
        f"Decay tau chosen on dev at k=4: {tau_choice}. MMR lambda: {lam_choice}.",
        "Random is the mean of three seeds. Full history does not depend on k;",
        "its three rows are the same evaluation.",
        "",
    ]
    for sp in ("dev", "test"):
        lines.append(f"## {sp}")
        lines.append("| Method | k | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for k in (2, 4, 6):
            for name in order:
                met = rows[sp][f"{name}@{k}"]
                mac = f"{100*met['macro_f1']:.2f}"
                if name == "Random":
                    mac += f" ± {100*met['macro_std']:.2f}"
                mmr_cell = "全历史" if name == "Full" else f"{100*met['mmr']:.2f}"
                lines.append(
                    f"| {name} | {k} | {mac} | {100*met['weighted_f1']:.2f} | "
                    f"{100*met['r1']:.2f} | {100*met['r5']:.2f} | {100*met['mrr']:.2f} | "
                    f"{mmr_cell} |"
                )
        lines.append("")
    lines.append(f"## test, history length at least 21, n={int(long.sum())}, k=4")
    lines.append("| Method | MacF1 |")
    lines.append("|---|---|")
    for name in order:
        if name == "Random":
            continue
        lines.append(f"| {name} | {100*long_f1[name]:.2f} |")
    lines.append("")
    lines.append("## Bootstrap Macro-F1 minus Last@4 on test")
    lines.append("| Method | mean | low | high | P(diff<=0) |")
    lines.append("|---|---|---|---|---|")
    for name, b in boots.items():
        lines.append(
            f"| {name} | {100*b['mean']:.2f} | {100*b['low']:.2f} | {100*b['high']:.2f} | {b['p_le_0']:.3f} |"
        )
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    serial = {sp: {k: v for k, v in table.items()} for sp, table in rows.items()}
    with open(os.path.join(RES, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": serial, "bootstrap": boots, "tau": tau_choice, "lambda": lam_choice,
                   "long_f1": long_f1, "long_n": int(long.sum())}, f, indent=2)


if __name__ == "__main__":
    main()
