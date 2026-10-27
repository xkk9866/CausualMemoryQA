"""Pooled paired bootstrap over seeds.

    python -m esc.pooled runs/esconv roberta-large_plan kd_insample_plan lcn2_biaspairskd

Each bootstrap draw resamples test examples once; on that resample every method's Macro-F1 is
averaged over its seeds, and the difference to the reference (first name) is recorded.
"""

import glob
import os
import sys

import numpy as np
from sklearn.metrics import f1_score


def preds(root, name):
    out = []
    for d in sorted(glob.glob(os.path.join(root, f"{name}_s[0-9]"))):
        z = np.load(os.path.join(d, "logits.npz"))
        out.append(z["test"].argmax(1))
        y = z["yt"]
    return np.array(out), y


def macro(y, p, k, idx):
    return np.mean([f1_score(y[idx], q[idx], average="macro", labels=list(range(k))) for q in p]) * 100


def main():
    root, names = sys.argv[1], sys.argv[2:]
    P = {}
    for n in names:
        P[n], y = preds(root, n)
    k = int(y.max()) + 1
    rng = np.random.default_rng(0)
    full = np.arange(len(y))
    print(f"{'method':24s} seeds  MacF1(mean over seeds)")
    for n in names:
        print(f"{n:24s} {len(P[n]):5d}  {macro(y, P[n], k, full):.2f}")
    ref = names[0]
    draws = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    for n in names[1:]:
        d = np.array([macro(y, P[n], k, i) - macro(y, P[ref], k, i) for i in draws])
        print(f"{n} - {ref}: {d.mean():+.2f}  95% CI [{np.percentile(d, 2.5):+.2f}, {np.percentile(d, 97.5):+.2f}]  "
              f"p(diff<=0)={np.mean(d <= 0):.3f}")


if __name__ == "__main__":
    main()
