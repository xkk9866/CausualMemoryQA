"""Is the loss-reducing early turn shared by dialogues with a similar recent window?"""

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store, forward_metrics
from src.joint_exchange import _window
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import LABELS, run_fixed, split_indices


def best_exchange(probe, store, idx, k=4, margin=0.02, batch_size=128):
    """Training-label utility of one exchange. Returns the chosen position or -1."""
    chosen = np.full(len(idx), -1, dtype=np.int64)
    helped = np.zeros(len(idx), dtype=bool)
    base_u = np.zeros(len(idx), dtype=np.float32)
    best_u = np.zeros(len(idx), dtype=np.float32)
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
            base = forward_metrics(probe, emb, role, pos, window, view["y"], view["resp"], view["cand"])
            u0 = base["u"].clone()
            ub = u0.clone()
            pick = torch.full((bsz,), -1, device=store.device, dtype=torch.long)
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
                take = ok & (alt["u"] > ub + margin)
                ub = torch.where(take, alt["u"], ub)
                pick = torch.where(take, pos_j, pick)
        sl = slice(start, start + bsz)
        chosen[sl] = pick.cpu().numpy()
        helped[sl] = (pick.cpu().numpy() >= 0)
        base_u[sl] = u0.cpu().numpy()
        best_u[sl] = ub.cpu().numpy()
    return chosen, helped, best_u - base_u


def window_repr(store, idx, k=4):
    reps, picks = [], []
    hl = store.cache["hlen"][idx]
    emb_all = []
    for start in range(0, len(idx), 256):
        batch = idx[start:start + 256]
        view = store.view(torch.tensor(batch, device=store.device))
        emb = view["emb"]
        valid = view["valid"]
        window = _window(valid, k)
        rep = (emb * window.unsqueeze(-1)).sum(1) / window.sum(1).clamp(min=1).unsqueeze(-1)
        reps.append(rep.cpu().numpy())
        emb_all.append(emb.cpu().numpy())
    return np.concatenate(reps), np.concatenate(emb_all), hl


def main():
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load("results/csb/probe_robust.pt", map_location=device, weights_only=True))
    probe.eval()
    tr, dv = split_indices(cache, 0), split_indices(cache, 1)
    print("scanning train", len(tr), flush=True)
    tr_pick, tr_help, tr_gain = best_exchange(probe, store, tr)
    print(f"train help {tr_help.mean()*100:.1f} mean gain {tr_gain.mean():.3f}", flush=True)
    print("scanning dev", len(dv), flush=True)
    dv_pick, dv_help, dv_gain = best_exchange(probe, store, dv)
    print(f"dev help {dv_help.mean()*100:.1f} mean gain {dv_gain.mean():.3f}", flush=True)

    tr_rep, tr_emb, tr_hl = window_repr(store, tr)
    dv_rep, dv_emb, dv_hl = window_repr(store, dv)
    tr_rep = tr_rep / np.linalg.norm(tr_rep, axis=1, keepdims=True).clip(1e-6)
    dv_rep = dv_rep / np.linalg.norm(dv_rep, axis=1, keepdims=True).clip(1e-6)
    # Prototype of the early turn that actually reduced training loss.
    proto_src = np.zeros_like(tr_rep)
    for i, j in enumerate(tr_pick.tolist()):
        if j >= 0:
            proto_src[i] = tr_emb[i, j]
    proto_src /= np.linalg.norm(proto_src, axis=1, keepdims=True).clip(1e-6)

    y = cache["y"][dv]
    length = store.max_l
    last = np.zeros((len(dv), length), dtype=bool)
    knn = np.zeros_like(last)
    hit = []
    for i, n in enumerate(dv_hl.tolist()):
        n = int(n)
        kk = min(4, n)
        last[i, max(0, n - kk):n] = True
        sim = tr_rep @ dv_rep[i]
        nb = np.argpartition(-sim, 32)[:32]
        nb = nb[tr_help[nb]]
        knn[i] = last[i]
        if n <= 4 or len(nb) == 0:
            continue
        proto = proto_src[nb].mean(0)
        proto /= np.linalg.norm(proto).clip(1e-6)
        early = dv_emb[i, : n - kk]
        early = early / np.linalg.norm(early, axis=1, keepdims=True).clip(1e-6)
        j = int(np.argmax(early @ proto))
        oldest = n - kk
        knn[i, oldest] = False
        knn[i, j] = True
        if dv_pick[i] >= 0:
            hit.append(int(j == dv_pick[i]))
    pred_l, _ = run_fixed(probe, store, dv, last, 128)
    pred_k, _ = run_fixed(probe, store, dv, knn, 128)
    # Oracle ceiling on the same split, label-using, not a method.
    oracle = last.copy()
    for i, j in enumerate(dv_pick.tolist()):
        n = int(dv_hl[i])
        if j >= 0 and n > 4:
            oracle[i, n - 4] = False
            oracle[i, j] = True
    pred_o, _ = run_fixed(probe, store, dv, oracle, 128)
    def f1(pred):
        return 100 * f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
    print(f"dev last {f1(pred_l):.2f} knn {f1(pred_k):.2f} oracle {f1(pred_o):.2f} "
          f"neighbor-hit {100*np.mean(hit):.1f} n {len(hit)}", flush=True)


if __name__ == "__main__":
    main()
