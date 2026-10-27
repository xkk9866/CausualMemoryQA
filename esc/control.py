"""Control analysis for the label-noise diagnosis.

    python -m esc.control <recog_dir> [<recog_dir> ...]

For each recognizer (message only, or context + message) on the test split:
  agree      accuracy of the reading against the label
  top2       share of disagreements whose label is the recognizer's second choice (ambiguity)
  ratio      between-dialogue variance of disagreement / permutation mean (F2)
  ratio|topic the same after permuting only within dialogues of the same problem type, so that
             topic-level differences are controlled
"""

import json
import os
import sys

import numpy as np

from esc.data import EXT, load, samples


def cluster_ratio(delta, groups, strata, rng, n=2000):
    def V(d):
        tot = 0.0
        for g in np.unique(groups):
            m = groups == g
            tot += m.sum() * (d[m].mean() - d.mean()) ** 2
        return tot / len(d)
    obs = V(delta)
    perm = []
    for _ in range(n):
        d = delta.copy()
        for s in np.unique(strata):
            m = np.where(strata == s)[0]
            d[m] = d[rng.permutation(m)]
        perm.append(V(d))
    perm = np.array(perm)
    return obs / perm.mean(), float(np.mean(perm >= obs))


def main():
    splits, names = load("esconv")
    raw = json.load(open(os.path.join(EXT, "data/esconv/ESConv.json"), encoding="utf-8"))
    test = samples(splits["test"])
    groups = np.array([r["dialog"] for r in test])
    # problem type of each test dialogue (same shuffle as esc.data.load)
    import random
    idx = list(range(len(raw)))
    random.seed(13)
    random.shuffle(idx)
    n = int(0.15 * len(raw))
    topic_of = {i: raw[j]["problem_type"] for i, j in enumerate(idx[n:2 * n])}
    strata = np.array([topic_of[g] for g in groups])
    out = {}
    for d in sys.argv[1:]:
        z = np.load(os.path.join(d, "logits.npz"))
        lt, y = z["test"], z["yt"]
        order = np.argsort(-lt, 1)
        delta = (order[:, 0] != y).astype(float)
        rng = np.random.default_rng(0)
        r_all, p_all = cluster_ratio(delta, groups, np.zeros(len(y)), rng)
        r_top, p_top = cluster_ratio(delta, groups, strata, rng)
        out[os.path.basename(d)] = {"agree": 100 * (1 - delta.mean()),
                                    "top2_of_disagree": 100 * float(np.mean(order[delta == 1, 1] == y[delta == 1])),
                                    "ratio": r_all, "p": p_all, "ratio_topic": r_top, "p_topic": p_top}
    print(json.dumps(out, indent=1))
    json.dump(out, open("runs/esconv/control.json", "w"), indent=1)


if __name__ == "__main__":
    main()
