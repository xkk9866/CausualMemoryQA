"""Frozen embeddings of every gold supporter response, the memory the mediator attends over."""

import argparse
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from esc.data import load, render, samples


def path(dataset, encoder):
    return os.path.join("runs", dataset, "bank_" + encoder.replace("/", "_") + ".npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    a = p.parse_args()
    splits, names = load(a.dataset)
    enc = SentenceTransformer(a.encoder, device="cuda")
    out = {}
    for s in ["train", "valid", "test"]:
        rows = samples(splits[s])
        e = enc.encode([r["response"] for r in rows], batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
        out[s] = e.astype(np.float16)
        c = enc.encode([render(r["turns"], names, 5) for r in rows], batch_size=128, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
        out[s + "_ctx"] = c.astype(np.float16)
        out[s + "_label"] = np.array([r["label"] for r in rows])
        out[s + "_dialog"] = np.array([r["dialog"] for r in rows])
    os.makedirs(os.path.dirname(path(a.dataset, a.encoder)), exist_ok=True)
    np.savez(path(a.dataset, a.encoder), **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
