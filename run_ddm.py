"""Train and evaluate decision-dominant memory on the cached ESConv split."""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.utils import clip_grad_norm_

from src.ddm import DecisionDominantMemory, concrete_gate, shortcut_masks
from src.modeling import class_weights

LABELS = list(range(8))
RHO = 0.32
LAM = 0.45
GAMMA = 0.35
BETA = 0.15
AUX = 0.25


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class View:
    def __init__(self, cache: dict, device: torch.device):
        self.cache = cache
        self.device = device
        self.emb = torch.tensor(cache["emb"], device=device)
        self.emb_emo = torch.tensor(cache["emb_emo"], device=device)
        self.role = torch.tensor(cache["role"], device=device, dtype=torch.long)
        self.hist = torch.tensor(cache["hist"], device=device, dtype=torch.long)
        self.hlen = torch.tensor(cache["hlen"], device=device, dtype=torch.long)
        self.y = torch.tensor(cache["y"], device=device, dtype=torch.long)
        self.resp = torch.tensor(cache["resp"], device=device, dtype=torch.long)
        self.cand = torch.tensor(cache["cand"], device=device, dtype=torch.long)
        self.donor = torch.tensor(cache["ins_donor"], device=device, dtype=torch.long)
        self.emo_score = torch.tensor(cache["emo_scores"], device=device)
        self.tfidf_score = torch.tensor(cache["tfidf_scores"], device=device)
        self.max_l = int(cache["max_l"])

    def gather(self, idx: torch.Tensor, kind: str = "clean"):
        hist = self.hist[idx]
        hlen = self.hlen[idx]
        bsz, length = hist.shape
        valid = torch.arange(length, device=self.device)[None, :] < hlen[:, None]
        ids = hist.clamp(min=0)
        table = self.emb_emo if kind == "emo" else self.emb
        emb = table[ids]
        role = self.role[ids]
        pos = torch.arange(length, device=self.device)[None, :].expand(bsz, length)
        y = self.y[idx]
        resp = F.normalize(self.emb[self.resp[idx]], dim=-1)
        cand = F.normalize(self.emb[self.cand[idx]], dim=-1)
        return {
            "emb": emb, "role": role, "pos": pos, "valid": valid, "y": y,
            "resp": resp, "cand": cand, "hlen": hlen, "ids": hist,
        }


def _per_example_ce(logits, y, weight):
    return F.cross_entropy(logits, y, weight=weight, reduction="none", label_smoothing=0.05)


def _risk(logits, query, y, resp, weight):
    ce = _per_example_ce(logits, y, weight)
    sim = query @ resp.T / 0.07
    nce = F.cross_entropy(sim, torch.arange(logits.shape[0], device=logits.device))
    return ce.mean() + 0.5 * nce, ce


def _batch_loss(model, batch, emo_emb, weight, use_hinge, use_inv, use_sparse):
    valid = batch["valid"]
    sel = model.selection_logits(batch["emb"], batch["role"], batch["pos"], valid)
    prob = torch.sigmoid(sel)
    gate = concrete_gate(sel) * valid.float()
    logits, query = model.decide(batch["emb"], batch["role"], batch["pos"], valid, gate)
    risk, ce = _risk(logits, query, batch["y"], batch["resp"], weight)
    loss = risk
    full_logits, full_query = model.decide(batch["emb"], batch["role"], batch["pos"], valid, valid.float())
    loss = loss + AUX * _risk(full_logits, full_query, batch["y"], batch["resp"], weight)[0]
    if use_hinge:
        rec, sim = shortcut_masks(batch["emb"], valid, batch["hlen"], RHO)
        with torch.no_grad():
            rec_logits, _ = model.decide(batch["emb"], batch["role"], batch["pos"], valid, rec.float())
            sim_logits, _ = model.decide(batch["emb"], batch["role"], batch["pos"], valid, sim.float())
            ce_rec = _per_example_ce(rec_logits, batch["y"], weight)
            ce_sim = _per_example_ce(sim_logits, batch["y"], weight)
        loss = loss + GAMMA * (F.relu(ce - ce_rec).mean() + F.relu(ce - ce_sim).mean())
    if use_inv:
        edited, _ = model.decide(emo_emb, batch["role"], batch["pos"], valid, gate)
        kl = F.kl_div(
            F.log_softmax(edited, dim=-1),
            F.softmax(logits.detach(), dim=-1),
            reduction="batchmean",
        )
        loss = loss + BETA * kl
    if use_sparse:
        density = (prob * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
        loss = loss + LAM * (density - RHO).abs()
    return loss


def _predict_from_gate(model, batch, gate):
    logits, query = model.decide(batch["emb"], batch["role"], batch["pos"], batch["valid"], gate)
    pred = logits.argmax(dim=-1)
    sims = torch.einsum("bd,bkd->bk", query, batch["cand"])
    rank = 1 + (sims > sims[:, :1]).sum(dim=1)
    return pred, rank


def _hard_gate(prob, valid, tau: float):
    chosen = (prob >= tau) & valid
    empty = valid.any(dim=1) & ~chosen.any(dim=1)
    if empty.any():
        fallback = prob.masked_fill(~valid, -1e4).argmax(dim=1)
        chosen = chosen.clone()
        chosen[empty, fallback[empty]] = True
    return chosen


@torch.inference_mode()
def _collect(model, view: View, indices: np.ndarray, tau: float, batch_size: int):
    """Clean predictions for DDM, full history, and matched-budget shortcuts."""
    model.eval()
    device = view.device
    store = {name: {"pred": [], "rank": [], "count": []} for name in ("DDM", "Full", "Last", "Sim", "Emo", "Tfidf", "Attn", "Rand")}
    ys, hlens = [], []
    probs_all = []
    for start in range(0, len(indices), batch_size):
        idx = torch.tensor(indices[start:start + batch_size], device=device)
        batch = view.gather(idx)
        valid = batch["valid"]
        prob = torch.sigmoid(model.selection_logits(batch["emb"], batch["role"], batch["pos"], valid))
        ddm = _hard_gate(prob, valid, tau)
        k = max(1, int(round(float(ddm.float().sum(dim=1).mean().item()))))
        rec, simm = shortcut_masks(batch["emb"], valid, batch["hlen"], max(k / max(float(batch["hlen"].float().mean()), 1.0), 1e-6))
        # Rebuild shortcuts at the exact global budget k, not the training density.
        rec, simm = _budget_masks(batch, k)
        emo = _score_mask(view.emo_score[idx], valid, k)
        tfidf = _score_mask(view.tfidf_score[idx], valid, k)
        _logits, _query, attn = model.decide(batch["emb"], batch["role"], batch["pos"], valid, valid.float(), return_weights=True)
        attn_mask = _score_mask(attn, valid, k)
        rng = np.random.default_rng(2_000 + start)
        rand = _random_mask(valid, k, rng)
        masks = {
            "DDM": ddm.float(),
            "Full": valid.float(),
            "Last": rec.float(),
            "Sim": simm.float(),
            "Emo": emo.float(),
            "Tfidf": tfidf.float(),
            "Attn": attn_mask.float(),
            "Rand": rand.float(),
        }
        ys.append(batch["y"].cpu().numpy())
        hlens.append(batch["hlen"].cpu().numpy())
        probs_all.append(prob.cpu().numpy())
        for name, gate in masks.items():
            pred, rank = _predict_from_gate(model, batch, gate)
            store[name]["pred"].append(pred.cpu().numpy())
            store[name]["rank"].append(rank.cpu().numpy())
            store[name]["count"].append(gate.sum(dim=1).cpu().numpy())
    y = np.concatenate(ys)
    hlen = np.concatenate(hlens)
    packed = {}
    for name, buckets in store.items():
        packed[name] = _pack(y, np.concatenate(buckets["pred"]), np.concatenate(buckets["rank"]), np.concatenate(buckets["count"]), hlen)
        packed[name]["pred"] = np.concatenate(buckets["pred"])
    packed["_y"] = y
    packed["_hlen"] = hlen
    return packed


def _budget_masks(batch, k: int):
    valid = batch["valid"]
    hlen = batch["hlen"]
    emb = batch["emb"]
    bsz, length, _ = emb.shape
    device = emb.device
    rec = torch.zeros(bsz, length, dtype=torch.bool, device=device)
    positions = torch.arange(length, device=device)[None, :]
    rec = (positions >= (hlen[:, None] - k)) & valid
    last = emb[torch.arange(bsz, device=device), (hlen - 1).clamp(min=0)]
    sim = torch.einsum("bld,bd->bl", emb, last).masked_fill(~valid, -1e4)
    kk = min(k, length)
    idx = sim.topk(kk, dim=1).indices
    chosen = torch.zeros_like(rec)
    chosen.scatter_(1, idx, True)
    chosen = chosen & valid
    # If top-k collided with padding, fill from the remaining valid turns.
    short = chosen.sum(dim=1) < torch.minimum(hlen, torch.full_like(hlen, k))
    if short.any():
        order = sim.argsort(dim=1, descending=True)
        for row in torch.where(short)[0].tolist():
            need = int(min(k, int(hlen[row])))
            take = order[row, :need]
            chosen[row, take] = True
            chosen[row, ~valid[row]] = False
    return rec, chosen


def _score_mask(scores: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    filled = scores.masked_fill(~valid, -1e4)
    kk = min(k, filled.shape[1])
    idx = filled.topk(kk, dim=1).indices
    mask = torch.zeros_like(valid)
    mask.scatter_(1, idx, True)
    return mask & valid


def _random_mask(valid: torch.Tensor, k: int, rng: np.random.Generator) -> torch.Tensor:
    mask = torch.zeros_like(valid)
    lengths = valid.sum(dim=1).tolist()
    for i, n in enumerate(lengths):
        n = int(n)
        kk = min(k, n)
        if kk <= 0:
            continue
        pick = rng.choice(n, size=kk, replace=False)
        mask[i, pick] = True
    return mask


def _pack(y, pred, rank, count, hlen) -> dict:
    rank = np.asarray(rank).astype(np.float64).clip(min=1.0)
    hlen = np.maximum(np.asarray(hlen).astype(np.float64), 1.0)
    count = np.asarray(count).astype(np.float64)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", labels=LABELS, zero_division=0)),
        "r1": float(np.mean(rank <= 1)),
        "r5": float(np.mean(rank <= 5)),
        "mrr": float(np.mean(1.0 / rank)),
        "kept": float(np.mean(count)),
        "keep_ratio": float(np.mean(count / hlen)),
        "n": int(len(y)),
    }


@torch.inference_mode()
def _condition_accuracy(model, view, indices, tau, batch_size, kind: str):
    """Accuracy when the gate is recomputed after a nuisance or order edit."""
    model.eval()
    device = view.device
    correct_both, correct_edit, kept_donor, n_donor = [], [], [], []
    for start in range(0, len(indices), batch_size):
        idx_np = indices[start:start + batch_size]
        idx = torch.tensor(idx_np, device=device)
        clean = view.gather(idx)
        prob = torch.sigmoid(model.selection_logits(clean["emb"], clean["role"], clean["pos"], clean["valid"]))
        gate = _hard_gate(prob, clean["valid"], tau).float()
        pred, _rank = _predict_from_gate(model, clean, gate)
        edited = _edit_batch(view, clean, idx, kind)
        eprob = torch.sigmoid(model.selection_logits(edited["emb"], edited["role"], edited["pos"], edited["valid"]))
        egate = _hard_gate(eprob, edited["valid"], tau).float()
        epred, _ = _predict_from_gate(model, edited, egate)
        y = clean["y"]
        correct_both.append(((pred == y) & (epred == y)).cpu().numpy())
        correct_edit.append((epred == y).cpu().numpy())
        if kind == "insert":
            donor_pos = edited["donor_pos"]
            took = egate[torch.arange(len(idx), device=device), donor_pos]
            kept_donor.append(took.cpu().numpy())
            n_donor.append(np.ones(len(idx)))
    out = {
        "edit_acc": float(np.mean(np.concatenate(correct_edit))),
        "both_acc": float(np.mean(np.concatenate(correct_both))),
    }
    if kept_donor:
        out["distractor_rate"] = float(np.mean(np.concatenate(kept_donor)))
    return out


def _edit_batch(view: View, clean, idx, kind: str):
    if kind == "emo":
        return view.gather(idx, kind="emo")
    bsz, length = clean["valid"].shape
    device = clean["emb"].device
    if kind == "shuffle":
        emb = clean["emb"].clone()
        role = clean["role"].clone()
        for local, inst in enumerate(idx.tolist()):
            n = int(clean["hlen"][local])
            if n <= 1:
                continue
            perm = torch.tensor(np.random.default_rng(90_000 + int(inst)).permutation(n), device=device)
            emb[local, :n] = clean["emb"][local, perm]
            role[local, :n] = clean["role"][local, perm]
        out = dict(clean)
        out["emb"], out["role"] = emb, role
        return out
    # Insert an emotion-matched turn from another dialogue just before the last turn.
    new_l = length + 1
    emb = torch.zeros(bsz, new_l, clean["emb"].shape[-1], device=device)
    role = torch.zeros(bsz, new_l, dtype=torch.long, device=device)
    valid = torch.zeros(bsz, new_l, dtype=torch.bool, device=device)
    donor_ids = view.donor[idx].clamp(min=0)
    donors = view.emb[donor_ids]
    donor_role = view.role[donor_ids]
    donor_pos = torch.zeros(bsz, dtype=torch.long, device=device)
    for local in range(bsz):
        n = int(clean["hlen"][local])
        at = max(n - 1, 0)
        if at:
            emb[local, :at] = clean["emb"][local, :at]
            role[local, :at] = clean["role"][local, :at]
        emb[local, at] = donors[local]
        role[local, at] = donor_role[local]
        rest = n - at
        if rest:
            emb[local, at + 1:at + 1 + rest] = clean["emb"][local, at:n]
            role[local, at + 1:at + 1 + rest] = clean["role"][local, at:n]
        valid[local, :n + 1] = True
        donor_pos[local] = at
    pos = torch.arange(new_l, device=device)[None, :].expand(bsz, new_l)
    hlen = clean["hlen"] + 1
    return {
        "emb": emb, "role": role, "pos": pos, "valid": valid, "y": clean["y"],
        "resp": clean["resp"], "cand": clean["cand"], "hlen": hlen, "donor_pos": donor_pos,
    }


def _dev_tau(model, view, dev_idx, batch_size):
    # One forward of selection probabilities, then a decision per threshold.
    model.eval()
    device = view.device
    best = None
    records = []
    for tau in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        preds, ranks, counts, ys, hlens = [], [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(dev_idx), batch_size):
                idx = torch.tensor(dev_idx[start:start + batch_size], device=device)
                batch = view.gather(idx)
                prob = torch.sigmoid(model.selection_logits(batch["emb"], batch["role"], batch["pos"], batch["valid"]))
                gate = _hard_gate(prob, batch["valid"], tau).float()
                pred, rank = _predict_from_gate(model, batch, gate)
                preds.append(pred.cpu().numpy())
                ranks.append(rank.cpu().numpy())
                counts.append(gate.sum(dim=1).cpu().numpy())
                ys.append(batch["y"].cpu().numpy())
                hlens.append(batch["hlen"].cpu().numpy())
        met = _pack(np.concatenate(ys), np.concatenate(preds), np.concatenate(ranks), np.concatenate(counts), np.concatenate(hlens))
        records.append((tau, met))
        print(f"  tau {tau:.2f} dev_macF1 {met['macro_f1']*100:.2f} kept {met['kept']:.2f} MRR {met['mrr']*100:.2f}")
    tau, met = max(records, key=lambda item: (item[1]["macro_f1"], item[1]["mrr"], -item[1]["kept"]))
    print(f"  chosen tau {tau:.2f} dev_macF1 {met['macro_f1']*100:.2f} kept {met['kept']:.2f}")
    return tau, met


@torch.inference_mode()
def _dev_reference(model, view, dev_idx, batch_size):
    """Full history and a fixed last-4 memory, on the same network."""
    model.eval()
    device = view.device
    buckets = {name: {"pred": [], "y": []} for name in ("Full", "Last4")}
    for start in range(0, len(dev_idx), batch_size):
        idx = torch.tensor(dev_idx[start:start + batch_size], device=device)
        batch = view.gather(idx)
        valid = batch["valid"]
        rec, _sim = _budget_masks(batch, 4)
        for name, gate in (("Full", valid.float()), ("Last4", rec.float())):
            pred, _rank = _predict_from_gate(model, batch, gate)
            buckets[name]["pred"].append(pred.cpu().numpy())
            buckets[name]["y"].append(batch["y"].cpu().numpy())
    out = {}
    for name, bucket in buckets.items():
        y = np.concatenate(bucket["y"])
        pred = np.concatenate(bucket["pred"])
        out[name] = float(f1_score(y, pred, average="macro", labels=LABELS, zero_division=0))
    return out


def train_one(view, train_idx, dev_idx, epochs, batch_size, lr, seed, device, use_hinge, use_inv, use_sparse, tag, ckpt_path):
    set_seed(seed)
    model = DecisionDominantMemory(in_dim=view.emb.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    weight = class_weights(view.y[train_idx].detach().cpu(), 8).to(device)
    best_state, best_f1, best_tau, wait, start_epoch = None, -1.0, 0.45, 0, 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        best_state = ckpt["best_state"]
        best_f1 = ckpt["best_f1"]
        best_tau = ckpt["best_tau"]
        wait = ckpt["wait"]
        start_epoch = ckpt["next_epoch"]
        if ckpt.get("finished"):
            print(f"{tag} training already finished, best_dev {best_f1*100:.2f}")
            model.load_state_dict(best_state)
            model.to(device).eval()
            return model, best_tau, best_f1
        print(f"{tag} resume at epoch {start_epoch+1} best_dev {best_f1*100:.2f}")
    for epoch in range(start_epoch, epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_idx)
        total, seen = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch_np = order[start:start + batch_size]
            batch = view.gather(torch.tensor(batch_np, device=device))
            emo = view.gather(torch.tensor(batch_np, device=device), kind="emo")["emb"]
            loss = _batch_loss(model, batch, emo, weight, use_hinge, use_inv, use_sparse)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(batch_np)
            seen += len(batch_np)
        print(f"{tag} epoch {epoch+1}/{epochs} loss {total / max(seen, 1):.4f}")
        tau, met = _dev_tau(model, view, dev_idx, batch_size)
        ref = _dev_reference(model, view, dev_idx, batch_size)
        print(f"  ref full {ref['Full']*100:.2f} last4 {ref['Last4']*100:.2f}")
        if met["macro_f1"] > best_f1 + 1e-4:
            best_f1 = met["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_tau = tau
            wait = 0
        else:
            wait += 1
        torch.save({
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "opt": opt.state_dict(),
            "best_state": best_state,
            "best_f1": best_f1,
            "best_tau": best_tau,
            "wait": wait,
            "next_epoch": epoch + 1,
            "finished": wait >= 4 or epoch + 1 >= epochs,
        }, ckpt_path)
        if wait >= 4:
            print(f"{tag} early stop")
            break
    model.load_state_dict(best_state)
    model.to(device).eval()
    return model, best_tau, best_f1


def _slice_f1(y, pred, hlen):
    rows = {}
    bands = {
        "short": hlen <= 8,
        "mid": (hlen >= 9) & (hlen <= 20),
        "long": hlen >= 21,
    }
    for name, mask in bands.items():
        if mask.sum() < 5:
            continue
        rows[name] = {
            "n": int(mask.sum()),
            "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", labels=LABELS, zero_division=0)),
        }
    return rows


def _bootstrap(y, pred_a, pred_b, n_boot=400, seed=0):
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        diffs.append(
            f1_score(y[ix], pred_a[ix], average="macro", labels=LABELS, zero_division=0)
            - f1_score(y[ix], pred_b[ix], average="macro", labels=LABELS, zero_division=0)
        )
    arr = np.asarray(diffs)
    return {"mean": float(arr.mean()), "low": float(np.quantile(arr, 0.025)), "high": float(np.quantile(arr, 0.975))}


def _fmt_pm(values):
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean() * 100:.2f} ± {arr.std(ddof=1) * 100:.2f}" if len(arr) > 1 else f"{arr.mean() * 100:.2f}"


def _fmt_kept(values):
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}" if len(arr) > 1 else f"{arr.mean():.2f}"


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    print("device", device)
    with open(os.path.join(root, "data", "cache", "esconv_v1.pkl"), "rb") as f:
        cache = pickle.load(f)
    view = View(cache, device)
    train_idx = np.where(cache["split"] == 0)[0]
    dev_idx = np.where(cache["split"] == 1)[0]
    test_idx = np.where(cache["split"] == 2)[0]
    print("instances", len(train_idx), len(dev_idx), len(test_idx), "mean history", float(cache["hlen"].mean()))

    seeds = (0, 1, 2)
    variants = {
        "DDM": dict(use_hinge=True, use_inv=True, use_sparse=True),
        "w/o dominance": dict(use_hinge=False, use_inv=True, use_sparse=True),
        "w/o invariance": dict(use_hinge=True, use_inv=False, use_sparse=True),
        "w/o budget": dict(use_hinge=True, use_inv=True, use_sparse=False),
    }
    # Main method: three seeds. Ablations: seed 0 only, same budget of epochs.
    jobs = [("DDM", seed, variants["DDM"]) for seed in seeds]
    jobs += [(name, 0, cfg) for name, cfg in variants.items() if name != "DDM"]

    done_path = os.path.join(root, "results", "ddm_done.json")
    summaries = []
    if os.path.exists(done_path):
        with open(done_path, encoding="utf-8") as f:
            summaries = json.load(f)
    done_keys = {(row["name"], int(row["seed"])) for row in summaries}
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    for name, seed, cfg in jobs:
        if (name, seed) in done_keys:
            print("skip finished", name, seed)
            continue
        tag = f"{name}/s{seed}"
        ckpt_path = os.path.join(root, "results", f"partial_{name.replace('/', '_').replace(' ', '_')}_s{seed}.pt")
        model, tau, dev_f1 = train_one(
            view, train_idx, dev_idx, epochs=12, batch_size=32, lr=3e-4, seed=seed,
            device=device, tag=tag, ckpt_path=ckpt_path, **cfg,
        )
        path = os.path.join(root, "results", f"ddm_{name.replace('/', '_').replace(' ', '_')}_s{seed}.pt")
        torch.save({"state": model.state_dict(), "tau": tau, "dev_f1": dev_f1, "cfg": cfg}, path)
        dev = _collect_at_k(model, view, dev_idx, tau, k=4, batch_size=64)
        k = max(1, int(round(dev["DDM"]["kept"])))
        test = _collect_at_k(model, view, test_idx, tau, k, batch_size=64)
        emo = _condition_accuracy(model, view, test_idx, tau, 64, "emo")
        shuf = _condition_accuracy(model, view, test_idx, tau, 64, "shuffle")
        ins = _condition_accuracy(model, view, test_idx, tau, 64, "insert")
        slices = {m: _slice_f1(test["_y"], test[m]["pred"], test["_hlen"]) for m in ("DDM", "Last", "Full", "Sim")}
        boot = _bootstrap(test["_y"], test["DDM"]["pred"], test["Last"]["pred"])
        row = {
            "name": name, "seed": seed, "tau": tau, "k": k, "dev_f1": dev_f1,
            "test": {m: {kk: vv for kk, vv in test[m].items() if kk != "pred"} for m in ("DDM", "Full", "Last", "Sim", "Emo", "Tfidf", "Attn", "Rand")},
            "emo": emo, "shuffle": shuf, "insert": ins, "slices": slices, "bootstrap_vs_last": boot,
        }
        summaries.append(row)
        with open(done_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2, default=float)
        print(tag, "TEST", json.dumps(row["test"]["DDM"]), "vs last", json.dumps(row["test"]["Last"]), "boot", boot)
        del model
        torch.cuda.empty_cache()

    _write_report(root, summaries)
    print("wrote results/ddm_tables.md results/ddm_metrics.json")


def _collect_at_k(model, view, indices, tau, k, batch_size):
    model.eval()
    device = view.device
    names = ("DDM", "Full", "Last", "Sim", "Emo", "Tfidf", "Attn", "Rand")
    store = {name: {"pred": [], "rank": [], "count": []} for name in names}
    ys, hlens = [], []
    for start in range(0, len(indices), batch_size):
        idx_np = indices[start:start + batch_size]
        idx = torch.tensor(idx_np, device=device)
        batch = view.gather(idx)
        valid = batch["valid"]
        prob = torch.sigmoid(model.selection_logits(batch["emb"], batch["role"], batch["pos"], valid))
        ddm = _hard_gate(prob, valid, tau)
        rec, simm = _budget_masks(batch, k)
        emo = _score_mask(view.emo_score[idx], valid, k)
        tfidf = _score_mask(view.tfidf_score[idx], valid, k)
        _logits, _query, attn = model.decide(batch["emb"], batch["role"], batch["pos"], valid, valid.float(), return_weights=True)
        attn_mask = _score_mask(attn, valid, k)
        rand = _random_mask(valid, k, np.random.default_rng(3_000 + start))
        masks = {
            "DDM": ddm.float(), "Full": valid.float(), "Last": rec.float(), "Sim": simm.float(),
            "Emo": emo.float(), "Tfidf": tfidf.float(), "Attn": attn_mask.float(), "Rand": rand.float(),
        }
        ys.append(batch["y"].cpu().numpy())
        hlens.append(batch["hlen"].cpu().numpy())
        for name, gate in masks.items():
            pred, rank = _predict_from_gate(model, batch, gate)
            store[name]["pred"].append(pred.cpu().numpy())
            store[name]["rank"].append(rank.cpu().numpy())
            store[name]["count"].append(gate.sum(dim=1).cpu().numpy())
    y = np.concatenate(ys)
    hlen = np.concatenate(hlens)
    packed = {}
    for name in names:
        packed[name] = _pack(y, np.concatenate(store[name]["pred"]), np.concatenate(store[name]["rank"]), np.concatenate(store[name]["count"]), hlen)
        packed[name]["pred"] = np.concatenate(store[name]["pred"])
    packed["_y"] = y
    packed["_hlen"] = hlen
    return packed


def _write_report(root, summaries):
    main_rows = [row for row in summaries if row["name"] == "DDM"]
    methods = ("DDM", "Full", "Last", "Sim", "Emo", "Tfidf", "Attn", "Rand")
    lines = [
        "# Decision-dominant memory",
        "",
        "Three training seeds. Cells are mean ± sample standard deviation.",
        "",
        "| Method | MacF1 | W-F1 | R@1 | R@5 | MRR | Kept turns | Keep ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method in methods:
        cells = []
        for key, scaler in (
            ("macro_f1", _fmt_pm), ("weighted_f1", _fmt_pm), ("r1", _fmt_pm),
            ("r5", _fmt_pm), ("mrr", _fmt_pm), ("kept", _fmt_kept), ("keep_ratio", _fmt_pm),
        ):
            cells.append(scaler([row["test"][method][key] for row in main_rows]))
        lines.append("| " + " | ".join([method, *cells]) + " |")
    lines += ["", "## Robustness of DDM", "",
              "| Edit | Accuracy on the edited history | Accuracy on both versions |",
              "|---|---|---|"]
    for label, key in (("emotion substitution", "emo"), ("turn shuffle", "shuffle"), ("irrelevant insertion", "insert")):
        acc = _fmt_pm([row[key]["edit_acc"] for row in main_rows])
        both = _fmt_pm([row[key]["both_acc"] for row in main_rows])
        lines.append(f"| {label} | {acc} | {both} |")
    dist = _fmt_pm([row["insert"]["distractor_rate"] for row in main_rows])
    lines.append("")
    lines.append(f"Share of inserted distractor turns that DDM keeps: {dist}.")
    lines += ["", "## Long histories (at least 21 turns), Macro-F1", "",
              "| Method | MacF1 |", "|---|---|"]
    for method in ("DDM", "Full", "Last", "Sim"):
        vals = [row["slices"][method]["long"]["macro_f1"] for row in main_rows if "long" in row["slices"][method]]
        lines.append(f"| {method} | {_fmt_pm(vals)} |")
    lines += ["", "## Ablations on seed 0 (the learned gate, same test protocol)", "",
              "| Variant | MacF1 | MRR | Kept turns |", "|---|---|---|---|"]
    for row in summaries:
        if row["name"] != "DDM" or row["seed"] == 0:
            met = row["test"]["DDM"]
            lines.append(
                f"| {row['name']} (seed {row['seed']}) | {met['macro_f1']*100:.2f} | {met['mrr']*100:.2f} | {met['kept']:.2f} |"
            )
    lines += ["", "## Paired bootstrap, DDM minus Last, Macro-F1", ""]
    for row in main_rows:
        b = row["bootstrap_vs_last"]
        lines.append(f"- seed {row['seed']}: mean {b['mean']*100:.2f}, 95% interval [{b['low']*100:.2f}, {b['high']*100:.2f}]")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(root, "results", "ddm_tables.md"), "w", encoding="utf-8") as f:
        f.write(text)
    payload = []
    for row in summaries:
        item = json.loads(json.dumps(row, default=float))
        payload.append(item)
    with open(os.path.join(root, "results", "ddm_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(text)


if __name__ == "__main__":
    main()
