"""Reaction types: k-means over embeddings of the seeker's reaction to each supporter message.

Centroids are fit on the train split only; every split gets its nearest-centroid type, but
only train types are ever used as supervision.

    python -m esc.reactions --dataset esconv --k 32
"""

import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from esc.data import load, samples


def path(dataset, k):
    return os.path.join("runs", dataset, f"reactions_k{k}.npz")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--encoder", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    splits, _ = load(a.dataset)
    enc = SentenceTransformer(a.encoder, device=a.device)
    emb = {}
    for s in ["train", "valid", "test"]:
        texts = [r["reaction"] or "(no reply)" for r in samples(splits[s])]
        emb[s] = enc.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    km = KMeans(a.k, n_init=10, random_state=0).fit(emb["train"])
    out = {s: km.predict(emb[s]) for s in emb}
    os.makedirs(os.path.dirname(path(a.dataset, a.k)), exist_ok=True)
    np.savez(path(a.dataset, a.k), centroids=km.cluster_centers_, **out)
    print(a.dataset, "cluster sizes", np.bincount(out["train"], minlength=a.k))


if __name__ == "__main__":
    main()
