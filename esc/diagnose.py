"""Diagnosis of self-annotated strategy labels (Section 3).

    python -m esc.diagnose esconv

1. recoverability: accuracy / Macro-F1 of a recogniser that reads the gold message;
2. clustering: turn-weighted between-dialogue variance of label-reading disagreement, against
   a permutation null;
3. recurrence: share of disagreements whose (reading, label) confusion recurs in the same
   dialogue, against the rate expected from the global confusion distribution;
4. in-context channel: label NLL / Macro-F1 of the recogniser's reading pushed through the
   global channel vs. a per-dialogue channel estimated from earlier messages only
   (prior strength chosen on dev).
"""

import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from esc.data import load, samples


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def between_var(x, d):
    ids = np.unique(d)
    m = np.array([x[d == i].mean() for i in ids])
    w = np.array([(d == i).sum() for i in ids])
    return np.average((m - x.mean()) ** 2, weights=w)


def channel_eval(pr, y, d, C, kappa):
    K = C.shape[0]
    nll_g, nll_u, pg, pu = [], [], [], []
    acc, last = None, None
    for i in range(len(y)):
        if d[i] != last:
            acc, last = np.zeros((K, K)), d[i]
        Cu = (acc + kappa * C) / (acc.sum(1, keepdims=True) + kappa)
        qg, qu = pr[i] @ C, pr[i] @ Cu
        nll_g.append(-np.log(qg[y[i]]))
        nll_u.append(-np.log(qu[y[i]]))
        pg.append(qg.argmax())
        pu.append(qu.argmax())
        acc = acc + np.outer(pr[i], np.eye(K)[y[i]])
    f = lambda p: 100 * f1_score(y, p, average="macro", labels=list(range(K)))
    return np.mean(nll_g), np.mean(nll_u), f(pg), f(pu)


def main():
    ds = sys.argv[1]
    splits, names = load(ds)
    K = len(names)
    z = np.load(os.path.join("runs", ds, "recog_roberta-large" if ds == "esconv" else "recog_full", "logits.npz"))
    S = {s: samples(splits[s]) for s in ["valid", "test"]}
    y = {s: np.array([r["label"] for r in S[s]]) for s in S}
    d = {s: np.array([r["dialog"] for r in S[s]]) for s in S}
    pr = {"valid": softmax(z["valid"]), "test": softmax(z["test"])}
    yt, dt, rt = y["test"], d["test"], pr["test"].argmax(1)

    res = {"recog_acc": 100 * accuracy_score(yt, rt), "recog_macro_f1": 100 * f1_score(yt, rt, average="macro", labels=list(range(K)))}
    print(f"[{ds}] recoverability: acc {res['recog_acc']:.1f}  MacF1 {res['recog_macro_f1']:.1f}")

    dis = (yt != rt).astype(float)
    obs = between_var(dis, dt)
    rng = np.random.default_rng(0)
    null = np.array([between_var(rng.permutation(dis), dt) for _ in range(2000)])
    res.update(disagree=float(dis.mean()), between_var=float(obs), between_var_null=float(null.mean()), between_p=float((null >= obs).mean()))
    print(f"[{ds}] disagreement {dis.mean():.3f}; between-dialogue variance {obs:.4f} vs permuted "
          f"{null.mean():.4f} (ratio {obs / null.mean():.2f}, p={(null >= obs).mean():.4f})")

    pairs = list(zip(yt[yt != rt], rt[yt != rt]))
    glob = Counter(pairs)
    pglob = np.array(list(glob.values())) / len(pairs)
    rep = tot = 0
    exp = 0.0
    for i in np.unique(dt):
        m = (dt == i) & (yt != rt)
        P = list(zip(yt[m], rt[m]))
        k = len(P)
        if k > 1:
            tot += k
            rep += sum(P.count(p) > 1 for p in P)
            exp += k * float((1 - (1 - pglob) ** (k - 1)) @ pglob)
    res.update(recur=rep / tot, recur_chance=exp / tot)
    print(f"[{ds}] confusion recurrence within dialogue {rep / tot:.3f} vs chance {exp / tot:.3f} "
          f"(ratio {rep / exp:.2f})")

    C = pr["valid"].T @ np.eye(K)[y["valid"]] + 1e-3
    C /= C.sum(1, keepdims=True)
    best = max([1, 2, 4, 8, 16], key=lambda kp: -channel_eval(pr["valid"], y["valid"], d["valid"], C, kp)[1])
    g, u, fg, fu = channel_eval(pr["test"], yt, dt, C, best)
    res.update(kappa=best, nll_global=float(g), nll_incontext=float(u), f1_global=float(fg), f1_incontext=float(fu))
    print(f"[{ds}] in-context channel (kappa={best} on dev): NLL {g:.3f} -> {u:.3f}  MacF1 {fg:.1f} -> {fu:.1f}")
    json.dump(res, open(os.path.join("runs", ds, "diagnosis.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
