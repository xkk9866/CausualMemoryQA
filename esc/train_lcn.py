"""Labeler-Channel Network (LCN).

Supporters label their own messages, and each one does it through a personal channel from the
strategy a message canonically realizes (s) to the label they pick (y). LCN predicts the
canonical strategy of the next message and explains the gold label through that supporter's
channel, which is inferred in context from the dialogue's past (reading, label) pairs:

    p(y | H) = sum_s p_theta(s | H) T_u(y | s),    T_u[s] = (N[s] + kappa C[s]) / (sum_y N[s, y] + kappa)

    L = -log p(y | H)  -  sum_s r(s) log p_theta(s | H)

r is the cross-fitted recogniser's reading of the gold response (the canonical target), C the
global channel, kappa > 0 a learned prior strength. Ablations:
    --channel global   T_u = C for every dialogue (no in-context inference)
    --channel none     p(y | H) = p_theta(s | H) (no channel)
    --no_canon         drop the canonical-target term
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from esc.channel import build
from esc.data import load, render, samples
from esc.metrics import report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--with_plan", action="store_true")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--channel", default="incontext", choices=["incontext", "global", "none", "bias"])
    p.add_argument("--evidence", default="pairs", choices=["pairs", "labels"],
                   help="bias mode: full (reading, label) pairs, or only the label histogram (ablation)")
    p.add_argument("--no_canon", action="store_true")
    p.add_argument("--kappa", type=float, default=4.0, help="initial prior strength (learned)")
    p.add_argument("--distill", default="", help="npz with teacher logits for the train split (key 'train')")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="smoke test on the first N samples per split")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    data = {s: samples(splits[s]) for s in ["train", "valid", "test"]}
    ch = build(a.dataset)
    if a.limit:
        data = {s: v[:a.limit] for s, v in data.items()}
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "left"

    def enc(s):
        rows = data[s]
        ids = tok([render(r["turns"], names, a.window, with_plan=a.with_plan) for r in rows],
                  truncation=True, max_length=a.max_len)["input_ids"]
        return [{"input_ids": x, "label": r["label"], "idx": i} for i, (x, r) in enumerate(zip(ids, rows))]

    def collate(s):
        N, n, canon = (torch.tensor(ch[s][key]) for key in ["N", "n", "canon"])

        def f(batch):
            out = tok.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
            idx = torch.tensor([b["idx"] for b in batch])
            out["labels"] = torch.tensor([b["label"] for b in batch])
            out["N"], out["n"], out["canon"] = N[idx], n[idx], canon[idx]
            out["idx"] = idx
            return out
        return f

    E = {s: enc(s) for s in data}
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=k)
    model.gradient_checkpointing_enable()
    model.cuda()
    C = torch.tensor(ch["C"]).cuda()
    teacher = torch.tensor(np.load(a.distill)["train"]).float().cuda() if a.distill else None
    rho = nn.Parameter(torch.tensor(math.log(math.expm1(a.kappa)), device="cuda"))  # kappa = softplus(rho)
    # bias mode: labeler evidence enters as an additive logit term g(T_u, n); zero-init = baseline at start
    ev_dim = k * k if a.evidence == "pairs" else k
    g_net = nn.Sequential(nn.Linear(ev_dim + 1, 256), nn.GELU(), nn.Linear(256, k)).cuda()
    nn.init.zeros_(g_net[2].weight)
    nn.init.zeros_(g_net[2].bias)

    def label_logp(logits, N, n):
        if a.channel == "bias":
            kappa = F.softplus(rho)
            T = (N + kappa * C) / (N.sum(-1, keepdim=True) + kappa)    # this supporter's channel, rows sum to 1
            ev = T.flatten(1) if a.evidence == "pairs" else (N.sum(1) + kappa * C.mean(0)) / (n[:, None] + kappa)
            z = logits.float() + g_net(torch.cat([ev, torch.log1p(n)[:, None]], -1))
            return z.log_softmax(-1), z.log_softmax(-1)
        log_ps = logits.float().log_softmax(-1)
        if a.channel == "none":
            return log_ps, log_ps
        if a.channel == "global":
            T = C.expand(len(N), -1, -1)
        else:
            kappa = F.softplus(rho)
            T = (N + kappa * C) / (N.sum(-1, keepdim=True) + kappa)    # row s: posterior of y given reading s
        return torch.log((log_ps.exp()[:, None, :] @ T).squeeze(1).clamp(min=1e-8)), log_ps

    g = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(E["train"], batch_size=a.bs, shuffle=True, collate_fn=collate("train"), generator=g)
    ev = {s: DataLoader(E[s], batch_size=a.bs * 2, collate_fn=collate(s)) for s in ["valid", "test"]}
    params = list(model.parameters()) + list(g_net.parameters())
    opt = torch.optim.AdamW([{"params": params, "lr": a.lr}, {"params": [rho], "lr": 1e-2, "weight_decay": 0.0}],
                            weight_decay=0.01)
    steps = a.epochs * len(tr)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, len(tr) // a.evals_per_epoch)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        out = []
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=b["input_ids"].cuda(), attention_mask=b["attention_mask"].cuda()).logits
            out.append(label_logp(logits, b["N"].cuda(), b["n"].cuda())[0].cpu())
        model.train()
        return torch.cat(out).numpy()

    yv = np.array([r["label"] for r in data["valid"]])
    yt = np.array([r["label"] for r in data["test"]])
    best, log, t0, it = -1, [], time.time(), 0
    model.train()
    for ep in range(a.epochs):
        for b in tr:
            b = {key: v.cuda() for key, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            log_py, log_ps = label_logp(logits, b["N"], b["n"])
            loss = F.nll_loss(log_py, b["labels"])
            if teacher is not None:  # posterior targets from a same-input teacher
                kd = F.kl_div(F.log_softmax(log_py / a.T, -1), F.softmax(teacher[b["idx"]] / a.T, -1), reduction="batchmean")
                loss = (1 - a.alpha) * loss + a.alpha * a.T ** 2 * kd
            if not a.no_canon:
                loss = loss - (b["canon"] * log_ps).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % eval_every == 0:
                lv = predict(ev["valid"])
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "kappa": float(F.softplus(rho)), "time": time.time() - t0}
                if mv["macro_f1"] > best:
                    best = mv["macro_f1"]
                    lt = predict(ev["test"])
                    row["test"] = report(yt, lt.argmax(1), k)
                    np.savez(os.path.join(a.out, "logits.npz"), valid=lv, test=lt, yv=yv, yt=yt)
                    json.dump({"args": vars(a), "valid": mv, "test": row["test"], "it": it, "kappa": row["kappa"]},
                              open(os.path.join(a.out, "best.json"), "w"), indent=1)
                log.append(row)
                print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
