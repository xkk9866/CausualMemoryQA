"""Train the missing-turn pointer and compare it with the recent window."""

import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store
from src.modeling import Probe
from src.pointer import load_table, masks_from_scores, train_pointer
from src.prepare import build_cache
from src.train_eval import LABELS, run_fixed, split_indices

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "pointer")


def f1_mrr(probe, store, idx, mask):
    y = store.cache["y"][idx]
    pred, rank = run_fixed(probe, store, idx, mask, 128)
    f1 = float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0))
    mrr = float(np.mean(1.0 / rank))
    return f1, mrr, pred


def main():
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda")
    cache = build_cache(ROOT, 0, 42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load(os.path.join(ROOT, "results", "csb", "probe_robust.pt"),
                                     map_location=device, weights_only=True))
    probe.eval()
    for p in probe.parameters():
        p.requires_grad_(False)
    tr, dv, te = (split_indices(cache, s) for s in (0, 1, 2))
    print("building utility tables", flush=True)
    table_tr = load_table(os.path.join(RES, "util_train.npz"), probe, store, tr)
    table_dv = load_table(os.path.join(RES, "util_dev.npz"), probe, store, dv)
    # Oracle ceiling on dev, using the dev label. Reported as a ceiling only.
    oracle = masks_from_scores(
        table_dv[:, :store.max_l], table_dv[:, store.max_l], cache["hlen"][dv], 4, tau=0.0
    )
    # The table's "stay" column is a utility, not a pointer score, so tau=0
    # keeps an exchange only when its utility exceeds the window. That is the ceiling.
    f1_o, mrr_o, _ = f1_mrr(probe, store, dv, oracle)
    last = masks_from_scores(
        np.full((len(dv), store.max_l), -1e4, dtype=np.float32),
        np.zeros(len(dv), dtype=np.float32),
        cache["hlen"][dv], 4, tau=0.0,
    )
    f1_l, mrr_l, _ = f1_mrr(probe, store, dv, last)
    print(f"dev ceiling F1 {100*f1_o:.2f} MRR {100*mrr_o:.2f} | last F1 {100*f1_l:.2f} MRR {100*mrr_l:.2f}", flush=True)

    model, epoch_fn, score_fn, _rest = train_pointer(
        store, tr, dv, table_tr, table_dv, 8, 64, 3e-4, 0, device
    )
    taus = (-0.5, -0.25, 0.0, 0.25, 0.5, 1.0)
    best = (-1.0, 0.0, None)
    for ep in range(8):
        loss = epoch_fn(ep)
        token, stay = score_fn(dv)
        hl = cache["hlen"][dv]
        line = f"epoch {ep+1} loss {loss:.3f}"
        for tau in taus:
            mask = masks_from_scores(token, stay, hl, 4, tau)
            f1, mrr, _pred = f1_mrr(probe, store, dv, mask)
            rate = float((mask != last).any(1).mean())
            line += f" | t{tau} {100*f1:.2f}/{100*rate:.0f}"
            if f1 > best[0]:
                best = (f1, tau, {n: p.detach().cpu().clone() for n, p in model.state_dict().items()})
        print(line, flush=True)
    print(f"best dev F1 {100*best[0]:.2f} tau {best[1]} last {100*f1_l:.2f}", flush=True)
    if best[2] is not None:
        torch.save(best[2], os.path.join(RES, "pointer_s0.pt"))


if __name__ == "__main__":
    main()
