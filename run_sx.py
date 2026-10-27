"""Budgeted sufficiency exchange.

Last-k is the recency window. For a fixed budget k the oldest window turn is
replaced by an earlier turn when that replacement makes the frozen decision
model agree with the decision it makes from the full history, and raises its
confidence. The rule uses the model's own full-history prediction, never the
gold label. A label-using utility search is reported only as a ceiling.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store, forward_metrics
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import LABELS, pack_metrics, run_fixed, split_indices

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results", "sx")


def _batches(probe, store, indices, mask, batch_size=128):
    pred, conf, pgold, cos = [], [], [], []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        z = torch.tensor(mask[start:start + batch_size], device=store.device, dtype=torch.float32)
        with torch.inference_mode():
            met = forward_metrics(
                probe, view["emb"], view["role"], view["pos"], z, view["y"], view["resp"], view["cand"]
            )
        pred.append(met["pred"].cpu().numpy())
        conf.append(met["conf"].cpu().numpy())
        pgold.append(met["p"].cpu().numpy())
        cos.append(met["cos"].cpu().numpy())
    return np.concatenate(pred), np.concatenate(conf), np.concatenate(pgold), np.concatenate(cos)


def window_mask(hlen, k, length):
    pos = np.arange(length)[None, :]
    return (pos >= (hlen[:, None] - k)) & (pos < hlen[:, None])


def exchange(probe, store, indices, k, n_try, mode, batch_size=128):
    """mode='deploy' follows the full-history decision. mode='gold' maximizes label utility."""
    hlen = store.cache["hlen"][indices]
    length = store.max_l
    base = window_mask(hlen, k, length)
    chosen = base.copy()
    b_pred, b_conf, b_p, b_cos = _batches(probe, store, indices, base, batch_size)
    if mode == "deploy":
        full = np.arange(length)[None, :] < hlen[:, None]
        f_pred, f_conf, _, _ = _batches(probe, store, indices, full, batch_size)
        key = (b_pred == f_pred).astype(np.float64) * 2.0 + b_conf
    else:
        f_pred = None
        key = np.log(np.clip(b_p, 1e-6, 1)) + 0.35 * b_cos
    for t in range(n_try):
        trial = chosen.copy()
        active = np.zeros(len(indices), dtype=bool)
        for i, n in enumerate(hlen.tolist()):
            n = int(n)
            if n <= k:
                continue
            oldest = int(np.where(chosen[i, :n])[0][0])
            j = oldest - 1 - t
            if j < 0:
                continue
            trial[i, oldest] = False
            trial[i, j] = True
            active[i] = True
        if not active.any():
            break
        pred, conf, p, cos = _batches(probe, store, indices, trial, batch_size)
        if mode == "deploy":
            alt = (pred == f_pred).astype(np.float64) * 2.0 + conf
        else:
            alt = np.log(np.clip(p, 1e-6, 1)) + 0.35 * cos
        take = active & (alt > key + 1e-6)
        chosen[take] = trial[take]
        key[take] = alt[take]
    return chosen


def metric_row(probe, store, indices, mask):
    y = store.cache["y"][indices]
    hl = store.cache["hlen"][indices]
    pred, rank = run_fixed(probe, store, indices, mask, 128)
    return pack_metrics(y, pred, rank, mask.sum(1), hl), pred


def main():
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda")
    cache = build_cache(ROOT, limit_dialogs=0, seed=42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load(os.path.join(ROOT, "results", "csb", "probe_robust.pt"),
                                     map_location=device, weights_only=True))
    probe.eval()
    te = split_indices(cache, 2)
    dv = split_indices(cache, 1)
    lines = ["# Budgeted sufficiency exchange", ""]
    preds = {}
    for split_name, idx in (("dev", dv), ("test", te)):
        lines.append(f"## {split_name}")
        lines.append("| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | MMR |")
        lines.append("|---|---|---|---|---|---|---|")
        hl = cache["hlen"][idx]
        for k in (2, 4, 6):
            last = window_mask(hl, k, store.max_l)
            for name, mask in (
                (f"Last{k}", last),
                (f"SX-deploy-k{k}", exchange(probe, store, idx, k, 8, "deploy")),
                (f"SX-gold-k{k}", exchange(probe, store, idx, k, 8, "gold")),
            ):
                met, pred = metric_row(probe, store, idx, mask)
                preds[(split_name, name)] = pred
                lines.append(
                    f"| {name} | {100*met['macro_f1']:.2f} | {100*met['weighted_f1']:.2f} | "
                    f"{100*met['r1']:.2f} | {100*met['r5']:.2f} | {100*met['mrr']:.2f} | {100*met['mmr']:.2f} |"
                )
                print(split_name, name, f"{100*met['macro_f1']:.2f}", flush=True)
        full = np.arange(store.max_l)[None, :] < hl[:, None]
        met, pred = metric_row(probe, store, idx, full)
        preds[(split_name, "Full")] = pred
        lines.append(
            f"| Full history | {100*met['macro_f1']:.2f} | {100*met['weighted_f1']:.2f} | "
            f"{100*met['r1']:.2f} | {100*met['r5']:.2f} | {100*met['mrr']:.2f} | {100*met['mmr']:.2f} |"
        )
        print(split_name, "full", f"{100*met['macro_f1']:.2f}", flush=True)
    y = cache["y"][te]
    rng = np.random.default_rng(0)
    diffs = []
    a, b = preds[("test", "SX-deploy-k4")], preds[("test", "Last4")]
    for _ in range(1000):
        ix = rng.integers(0, len(y), len(y))
        diffs.append(
            f1_score(y[ix], a[ix], average="macro", labels=LABELS, zero_division=0)
            - f1_score(y[ix], b[ix], average="macro", labels=LABELS, zero_division=0)
        )
    d = np.array(diffs)
    lines.append("")
    lines.append(
        f"Bootstrap MacF1(SX-deploy-k4) - MacF1(Last4) on test: "
        f"mean {100*d.mean():.2f}, 95% CI [{100*np.quantile(d,0.025):.2f}, {100*np.quantile(d,0.975):.2f}], "
        f"P(diff<=0)={np.mean(d<=0):.3f}"
    )
    # Gold ceiling is not deployable.
    lines.append("SX-gold uses the training label inside the exchange and is a ceiling, not a method.")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
