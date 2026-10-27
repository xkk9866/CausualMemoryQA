"""Labeler-channel statistics.

Every supporter message j has a canonical reading r_j = p(s | R_j) from a response recogniser
(out-of-fold on train, the full-train recogniser on dev/test) and the label y_j its writer chose.
For a decision i, the past pairs of the same dialogue give the count matrix

    N_i[s, y] = sum_{j < i, same dialogue} r_j[s] * 1[y_j = y],   n_i = number of past messages,

and the global channel C[s, y] is estimated from all training pairs. Only messages before i
are used, so everything here is available at test time.
"""

import os

import numpy as np

from esc.data import load, samples


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def readings(dataset):
    """canonical readings of every gold response, per split"""
    root = os.path.join("runs", dataset)
    full = os.path.join(root, "recog_roberta-large" if dataset == "esconv" else "recog_full", "logits.npz")
    z = np.load(full)
    return {"train": _softmax(np.load(os.path.join(root, "recog_foof.npz"))["train"]),
            "valid": _softmax(z["valid"]), "test": _softmax(z["test"])}


def build(dataset):
    splits, names = load(dataset)
    k = len(names)
    r = readings(dataset)
    out = {}
    for s in ["train", "valid", "test"]:
        rows = samples(splits[s])
        y = np.array([x["label"] for x in rows])
        d = np.array([x["dialog"] for x in rows])
        N = np.zeros((len(rows), k, k), dtype=np.float32)
        n = np.zeros(len(rows), dtype=np.float32)
        acc, cnt, last = None, 0, None
        for i in range(len(rows)):  # samples are in dialogue order, turns in time order
            if d[i] != last:
                acc, cnt, last = np.zeros((k, k), dtype=np.float32), 0, d[i]
            N[i], n[i] = acc, cnt
            acc = acc + np.outer(r[s][i], np.eye(k)[y[i]])
            cnt += 1
        out[s] = {"N": N, "n": n, "canon": r[s].astype(np.float32)}
    C = (r["train"].T @ np.eye(k)[np.array([x["label"] for x in samples(splits["train"])])]) + 1e-3
    out["C"] = (C / C.sum(1, keepdims=True)).astype(np.float32)
    return out
