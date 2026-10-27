"""Dialogue-cluster paired bootstrap.

Messages of one dialogue are correlated, so every draw resamples whole test dialogues with
replacement. On each resample a system's score is its Macro-F1 averaged over its seeds, and both
systems are scored on the same resample (paired). Returns the difference and each system's
absolute score with 95% percentile intervals, plus the one-sided p(diff <= 0).

    from esc.cluster_boot import compare
    compare(preds_a, preds_b, y, groups)       # preds_*: list over seeds of label arrays
"""

import numpy as np
from sklearn.metrics import f1_score

from esc.data import load, samples


def test_groups(dataset="esconv"):
    splits, _ = load(dataset)
    return np.array([r["dialog"] for r in samples(splits["test"])])


def _macro(y, P, idx, k):
    return 100 * np.mean([f1_score(y[idx], p[idx], average="macro", labels=list(range(k))) for p in P])


def compare(A, B, y, groups, n=2000, seed=0):
    k = int(y.max()) + 1
    dial = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in dial}
    rng = np.random.default_rng(seed)
    da, db, dd = [], [], []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(dial, len(dial), replace=True)])
        a, b = _macro(y, A, idx, k), _macro(y, B, idx, k)
        da.append(a)
        db.append(b)
        dd.append(a - b)
    full = np.arange(len(y))
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"a": _macro(y, A, full, k), "a_ci": ci(da), "b": _macro(y, B, full, k), "b_ci": ci(db),
            "diff": _macro(y, A, full, k) - _macro(y, B, full, k), "diff_ci": ci(dd),
            "p": float(np.mean(np.array(dd) <= 0)), "n_dialogues": int(len(dial)), "draws": n}
