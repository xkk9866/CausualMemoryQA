"""Dev diagnostic: scorer exchange versus label-free confidence exchange."""

import torch
from sklearn.metrics import f1_score

from src.causal import Store
from src.joint_exchange import collect_masks, confidence_masks, train_joint
from src.prepare import build_cache
from src.train_eval import LABELS, run_fixed, split_indices


def f1_of(model, store, idx, mask):
    pred, _rank = run_fixed(model.dec, store, idx, mask, 128)
    y = store.cache["y"][idx]
    return float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0))


def main():
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    store = Store(cache, device)
    tr, dv, te = (split_indices(cache, s) for s in (0, 1, 2))
    model = train_joint(store, tr, dv, 4, 4, 128, 3e-4, 0, device, k=4)
    hl = cache["hlen"][dv]
    length = store.max_l
    last = (torch.arange(length)[None, :] >= (torch.tensor(hl)[:, None] - 4)) & (
        torch.arange(length)[None, :] < torch.tensor(hl)[:, None]
    )
    last = last.numpy()
    print("last", round(100 * f1_of(model, store, dv, last), 2), flush=True)
    for tau in (0.0, 0.25, 0.5, 1.0):
        mask = collect_masks(model, store, dv, 4, 128, tau, device)
        rate = float((mask != last).any(1).mean())
        print(f"scorer tau {tau} F1 {100 * f1_of(model, store, dv, mask):.2f} swap {100 * rate:.1f}", flush=True)
    for margin in (0.0, 0.02, 0.05, 0.1):
        mask = confidence_masks(model, store, dv, 4, 64, margin, device)
        rate = float((mask != last).any(1).mean())
        print(f"conf margin {margin} F1 {100 * f1_of(model, store, dv, mask):.2f} swap {100 * rate:.1f}", flush=True)


if __name__ == "__main__":
    main()
