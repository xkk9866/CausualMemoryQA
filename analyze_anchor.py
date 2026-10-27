"""Label-free exchange: reproduce the full-history strategy and response query."""

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store, forward_metrics
from src.joint_exchange import _window
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import LABELS, run_fixed, split_indices


def choose(probe, store, idx, k=4, batch_size=64):
    """Return masks and the fraction of rows that leave the recency window."""
    masks = np.zeros((len(idx), store.max_l), dtype=bool)
    moved = np.zeros(len(idx), dtype=bool)
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        emb, role, pos, valid = view["emb"], view["role"], view["pos"], view["valid"]
        bsz = emb.shape[0]
        window = _window(valid, k)
        rows = torch.arange(bsz, device=store.device)
        oldest = window.argmax(1)
        hlen = valid.sum(1).long()
        start_w = (hlen - k).clamp(min=0)
        with torch.inference_mode():
            full = forward_metrics(probe, emb, role, pos, valid, view["y"], view["resp"], view["cand"])
            base = forward_metrics(probe, emb, role, pos, window, view["y"], view["resp"], view["cand"])
            # Agreement with the full dialogue, then fidelity of the response query.
            # Neither term uses the gold label.
            q_full = full["q"]
            pred_full = full["pred"]
            score = (base["pred"] == pred_full).float() * 2.0 + (base["q"] * q_full).sum(-1)
            best = window.clone()
            best_score = score.clone()
            max_j = int(start_w.max().item()) if bsz else 0
            for j in range(max_j):
                pos_j = start_w - 1 - j
                ok = pos_j >= 0
                if not bool(ok.any()):
                    continue
                trial = window.clone()
                trial[rows[ok], oldest[ok]] = 0
                trial[rows[ok], pos_j[ok]] = 1
                alt = forward_metrics(probe, emb, role, pos, trial, view["y"], view["resp"], view["cand"])
                alt_score = (alt["pred"] == pred_full).float() * 2.0 + (alt["q"] * q_full).sum(-1)
                take = ok & (alt_score > best_score + 0.01)
                best_score = torch.where(take, alt_score, best_score)
                best = torch.where(take[:, None], trial, best)
        masks[start:start + bsz] = best.cpu().numpy() > 0.5
        moved[start:start + bsz] = (best_score > score + 0.01).cpu().numpy()
    return masks, moved


def report(name, probe, store, idx, mask):
    y = store.cache["y"][idx]
    pred, rank = run_fixed(probe, store, idx, mask, 128)
    f1 = 100 * f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
    mrr = 100 * float(np.mean(1.0 / rank))
    print(f"{name} F1 {f1:.2f} MRR {mrr:.2f}", flush=True)
    return pred


def main():
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load("results/csb/probe_robust.pt", map_location=device, weights_only=True))
    probe.eval()
    dv = split_indices(cache, 1)
    hl = cache["hlen"][dv]
    length = store.max_l
    last = (np.arange(length)[None, :] >= (hl[:, None] - 4)) & (np.arange(length)[None, :] < hl[:, None])
    report("last4", probe, store, dv, last)
    masks, moved = choose(probe, store, dv)
    print(f"moved {moved.mean()*100:.1f}", flush=True)
    report("anchor", probe, store, dv, masks)


if __name__ == "__main__":
    main()
