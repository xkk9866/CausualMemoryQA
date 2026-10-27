"""Which label-free property identifies the early turn that raises gold utility?"""

import numpy as np
import torch
from sklearn.metrics import f1_score

from src.causal import Store, forward_metrics
from src.joint_exchange import _window
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import LABELS, run_fixed, split_indices


def scan(probe, store, idx, k=4, batch_size=128):
    pick = np.full(len(idx), -1, dtype=np.int64)
    gain = np.zeros(len(idx), dtype=np.float32)
    attn = np.zeros((len(idx), store.max_l), dtype=np.float32)
    for start in range(0, len(idx), batch_size):
        batch = idx[start:start + batch_size]
        view = store.view(torch.tensor(batch, device=store.device))
        emb, role, pos, valid = view["emb"], view["role"], view["pos"], view["valid"]
        bsz, length, _ = emb.shape
        window = _window(valid, k)
        rows = torch.arange(bsz, device=store.device)
        oldest = window.argmax(1)
        hlen = valid.sum(1).long()
        start_w = (hlen - k).clamp(min=0)
        with torch.inference_mode():
            _lg, _q, w = probe(emb, role, pos, valid, return_weights=True)
            attn[start:start + bsz] = w.float().cpu().numpy()
            base = forward_metrics(probe, emb, role, pos, window, view["y"], view["resp"], view["cand"])
            ub = base["u"].clone()
            chosen = torch.full((bsz,), -1, device=store.device, dtype=torch.long)
            u0 = ub.clone()
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
                take = ok & (alt["u"] > ub)
                ub = torch.where(take, alt["u"], ub)
                chosen = torch.where(take, pos_j, chosen)
        pick[start:start + bsz] = chosen.cpu().numpy()
        gain[start:start + bsz] = (ub - u0).cpu().numpy()
    return pick, gain, attn


def apply_choice(choice, hlen, k, length):
    mask = np.zeros((len(hlen), length), dtype=bool)
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        kk = min(k, n)
        mask[i, max(0, n - kk):n] = True
        j = int(choice[i])
        if n > k and 0 <= j < n - k:
            mask[i, n - k] = False
            mask[i, j] = True
    return mask


def main():
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    store = Store(cache, device)
    probe = Probe(n_strat=8, in_dim=768).to(device)
    probe.load_state_dict(torch.load("results/csb/probe_robust.pt", map_location=device, weights_only=True))
    probe.eval()
    dv = split_indices(cache, 1)
    pick, gain, attn = scan(probe, store, dv)
    hl = cache["hlen"][dv]
    length = store.max_l
    y = cache["y"][dv]
    # Turn embeddings and the gold response live in the cache view.
    emb = np.zeros((len(dv), length, store.emb.shape[1]), dtype=np.float32)
    resp = np.zeros((len(dv), store.emb.shape[1]), dtype=np.float32)
    for start in range(0, len(dv), 256):
        batch = dv[start:start + 256]
        view = store.view(torch.tensor(batch, device=device))
        emb[start:start + len(batch)] = view["emb"].float().cpu().numpy()
        resp[start:start + len(batch)] = view["resp"].float().cpu().numpy()

    choice = {
        "oracle": pick.copy(),
        "nearest": np.array([int(n) - 5 if int(n) > 4 else -1 for n in hl], dtype=np.int64),
    }
    sim_last = np.full(len(dv), -1, dtype=np.int64)
    sim_resp = np.full(len(dv), -1, dtype=np.int64)
    sim_win = np.full(len(dv), -1, dtype=np.int64)
    attn_pick = np.full(len(dv), -1, dtype=np.int64)
    for i, n in enumerate(hl.tolist()):
        n = int(n)
        if n <= 4:
            continue
        early = emb[i, : n - 4]
        last = emb[i, n - 1]
        window = emb[i, n - 4:n].mean(0)
        sim_last[i] = int(np.argmax(early @ last))
        sim_resp[i] = int(np.argmax(early @ resp[i]))
        sim_win[i] = int(np.argmax(early @ window))
        score = attn[i, : n - 4].copy()
        attn_pick[i] = int(np.argmax(score))
    choice["sim-last"] = sim_last
    choice["sim-resp"] = sim_resp
    choice["sim-window"] = sim_win
    choice["attn"] = attn_pick

    helped = pick >= 0
    print(f"dev helped {helped.mean()*100:.1f} mean gain {gain.mean():.3f} gain|help {gain[helped].mean():.3f}")
    for name, ch in choice.items():
        if name == "oracle":
            hit = 100.0
        else:
            hit = 100 * np.mean(ch[helped] == pick[helped])
        mask = apply_choice(ch, hl, 4, length)
        pred, rank = run_fixed(probe, store, dv, mask, 128)
        f1 = 100 * f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)
        mrr = 100 * float(np.mean(1.0 / rank))
        print(f"{name:12s} hit {hit:5.1f}  F1 {f1:.2f}  MRR {mrr:.2f}", flush=True)


if __name__ == "__main__":
    main()
