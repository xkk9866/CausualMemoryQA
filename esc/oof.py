"""Assemble out-of-fold teacher logits for the whole train split: python -m esc.oof runs/esconv/hind_f 5"""
import sys

import numpy as np

prefix, n = sys.argv[1], int(sys.argv[2])
parts = [np.load(f"{prefix}{f}/logits.npz") for f in range(n)]
size = sum(len(p["held_idx"]) for p in parts)
train = np.zeros((size, parts[0]["held"].shape[1]), dtype=np.float32)
seen = np.zeros(size, dtype=bool)
for p in parts:
    train[p["held_idx"]] = p["held"]
    seen[p["held_idx"]] = True
assert seen.all()
np.savez(prefix + "oof.npz", train=train)
print("oof", train.shape)
