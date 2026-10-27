"""Aggregate runs into paper tables: mean +- std over seeds, per-class F1, and a paired
bootstrap of the seed-averaged prediction against a reference method.

    python -m esc.report runs/esconv --ref roberta-large_w5
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from sklearn.metrics import f1_score

from esc.metrics import report


def runs(root):
    """group run directories by name without the _s<seed> suffix"""
    groups = defaultdict(list)
    for d in sorted(glob.glob(os.path.join(root, "*", "best.json"))):
        name = os.path.basename(os.path.dirname(d))
        m = re.match(r"(.*)_s(\d+)$", name)
        if m and os.path.exists(os.path.join(os.path.dirname(d), "logits.npz")):
            groups[m.group(1)].append(os.path.dirname(d))
    return groups


def bootstrap(y, pa, pb, k, n=2000, seed=0):
    """P(macro-F1 of a <= macro-F1 of b) over test resamples, and the mean difference."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1_score(y[i], pa[i], average="macro", labels=list(range(k)))
                     - f1_score(y[i], pb[i], average="macro", labels=list(range(k))))
    diffs = np.array(diffs) * 100
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), (diffs <= 0).mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--ref", default="roberta-large_w5")
    p.add_argument("--names", default="")
    a = p.parse_args()
    g = runs(a.root)
    rows = {}
    for name, dirs in g.items():
        ms, preds = [], []
        for d in dirs:
            z = np.load(os.path.join(d, "logits.npz"))
            k = z["test"].shape[1]
            y = z["yt"]
            preds.append(z["test"].argmax(1))
            ms.append(report(y, preds[-1], k) | {"per_class": (100 * f1_score(y, preds[-1], average=None, labels=list(range(k)))).tolist()})
        rows[name] = (ms, preds, y, k)
    print(f"{'method':28s} n  Acc            MacF1          W-F1           Bias")
    for name, (ms, _, _, _) in sorted(rows.items(), key=lambda r: np.mean([m["macro_f1"] for m in r[1][0]])):
        cell = lambda key: f"{np.mean([m[key] for m in ms]):6.2f}±{np.std([m[key] for m in ms]):4.2f}"
        print(f"{name:28s} {len(ms)}  {cell('acc')}  {cell('macro_f1')}  {cell('weighted_f1')}  {cell('bias')}")
    if a.ref in rows:
        ms0, p0, y, k = rows[a.ref]
        print(f"\npaired bootstrap of each seed against the same seed of {a.ref} (MacF1 difference)")
        for name, (ms, preds, _, _) in rows.items():
            if name == a.ref:
                continue
            out = []
            for s in range(min(len(preds), len(p0))):
                mean, ci, pval = bootstrap(y, preds[s], p0[s], k)
                out.append(f"s{s}: {mean:+.2f} [{ci[0]:+.2f},{ci[1]:+.2f}] p={pval:.3f}")
            print(f"{name:28s} " + " | ".join(out))
        print("\nper-class F1 (mean over seeds)")
        for name, (ms, _, _, _) in rows.items():
            pc = np.mean([m["per_class"] for m in ms], 0)
            print(f"{name:28s} " + " ".join(f"{v:5.1f}" for v in pc))
    json.dump({n: r[0] for n, r in rows.items()}, open(os.path.join(a.root, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
