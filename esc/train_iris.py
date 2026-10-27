"""IRIS: Imagined-Response Inference for Strategy planning.

A strategy label is a property of the message that realises it, and the message is observed
exactly while the self-annotated label is not (39.8% of ESConv labels disagree with what their
message reads as). IRIS therefore plans in the space of responses instead of the space of labels:

    p(y | H) = sum_m pi_m(H) * softmax(g(z_m(H)))_y

z_1..z_M are M imagined responses: latent vectors that the context encoder places in the same
space as the encoding r(R) of real supporter messages, pi their mixture weights, and g a single
strategy reader shared by imagined and real responses. Three terms train the whole thing:

    L = -log p(y | H)                              (plan: label through imagined responses)
        + lambda_r * CE(g(r(R)), y)                (read: the reader learns on real messages)
        - lambda_c * log [ sum_m pi_m e^{s(z_m,r)/tau} / sum_b' sum_m pi_m e^{s(z_m,r_b')/tau} ]
                                                   (imagine: a mixture InfoNCE on the real reply)

The response R is needed only for training; at test time IRIS reads the context alone.
Ablations: --M 1 (single imagined response), --lam_c 0 (no imagination target),
--lam_r 0 (reader only learns through the plan), --direct (no response space: g on the context).
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
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from esc.data import load, render, samples
from esc.metrics import report


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--with_plan", action="store_true")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--resp_len", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--M", type=int, default=4, help="number of imagined responses")
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_c", type=float, default=1.0)
    p.add_argument("--queue", type=int, default=1024, help="detached past replies used as extra negatives")
    p.add_argument("--direct", action="store_true", help="ablation: plain classifier head on the context")
    p.add_argument("--tied", type=int, default=0, help="1: one encoder for context and reply")
    p.add_argument("--sg", type=int, default=1, help="1: the reply is a fixed target of the imagination loss (JEPA-style)")
    p.add_argument("--train_frac", type=float, default=1.0, help="pilot runs on a dialogue subset")
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--out", required=True)
    return p.parse_args()


class IRIS(nn.Module):
    def __init__(self, name, k, a):
        super().__init__()
        self.a = a
        self.enc = AutoModel.from_pretrained(name)
        self.enc.gradient_checkpointing_enable()
        self.renc = self.enc if a.tied else AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.imagine = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, a.M * a.dim))
        self.mix = nn.Linear(h, a.M)
        self.rproj = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, a.dim))
        self.norm = nn.LayerNorm(a.dim)
        self.reader = nn.Linear(a.dim, k)       # g: strategy of a (real or imagined) response
        self.direct = nn.Linear(h, k)
        self.k = k

    def pool(self, enc, ids, mask):
        return enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]

    def plan(self, ids, mask):
        hc = self.pool(self.enc, ids, mask)
        if self.a.direct:
            return self.direct(hc).log_softmax(-1), None, None
        z = self.norm(self.imagine(hc).view(len(hc), self.a.M, self.a.dim))
        log_pi = self.mix(hc).float().log_softmax(-1)
        log_py_m = self.reader(z).float().log_softmax(-1)               # B x M x K
        log_py = torch.logsumexp(log_pi[..., None] + log_py_m, 1)       # B x K
        return log_py, z, log_pi

    def read(self, ids, mask):
        return self.norm(self.rproj(self.pool(self.renc, ids, mask)))


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    if a.train_frac < 1:
        rng = np.random.default_rng(0)
        keep = rng.permutation(len(splits["train"]))[: int(a.train_frac * len(splits["train"]))]
        splits["train"] = [splits["train"][i] for i in sorted(keep)]
    data = {s: samples(splits[s]) for s in ["train", "valid", "test"]}
    if a.label_noise > 0:  # same corruption as esc.train_cls
        rng = np.random.default_rng(12345)
        flip = rng.random(len(data["train"])) < a.label_noise
        shift = rng.integers(1, k, len(data["train"]))
        data["train"] = [dict(r, label=int((r["label"] + sh) % k)) if f else r
                         for r, f, sh in zip(data["train"], flip, shift)]
    tok = AutoTokenizer.from_pretrained(a.model)

    def enc(rows):
        tok.truncation_side = "left"   # keep the most recent turns
        ctx = tok([render(r["turns"], names, a.window, with_plan=a.with_plan) for r in rows],
                  truncation=True, max_length=a.max_len)["input_ids"]
        tok.truncation_side = "right"
        rsp = tok([r["response"] or "." for r in rows], truncation=True, max_length=a.resp_len)["input_ids"]
        return [{"ctx": c, "rsp": q, "label": r["label"], "idx": i} for i, (c, q, r) in enumerate(zip(ctx, rsp, rows))]

    def collate(batch):
        c = tok.pad([{"input_ids": b["ctx"]} for b in batch], return_tensors="pt")
        q = tok.pad([{"input_ids": b["rsp"]} for b in batch], return_tensors="pt")
        return {"ids": c["input_ids"], "mask": c["attention_mask"], "rids": q["input_ids"],
                "rmask": q["attention_mask"], "labels": torch.tensor([b["label"] for b in batch])}

    E = {s: enc(data[s]) for s in data}
    model = IRIS(a.model, k, a).cuda()
    g = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(E["train"], batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
    ev = {s: DataLoader(E[s], batch_size=a.bs * 2, collate_fn=collate) for s in ["valid", "test"]}
    enc_params = list(model.enc.parameters()) + ([] if a.tied else list(model.renc.parameters()))
    enc_ids = {id(p_) for p_ in enc_params}
    head = [p_ for p_ in model.parameters() if id(p_) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * len(tr)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, len(tr) // a.evals_per_epoch)
    queue = torch.zeros(0, a.dim, device="cuda")

    @torch.no_grad()
    def predict(loader):
        model.eval()
        out = []
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model.plan(b["ids"].cuda(), b["mask"].cuda())[0].float().cpu())
        model.train()
        return torch.cat(out).numpy()

    yv = np.array([r["label"] for r in data["valid"]])
    yt = np.array([r["label"] for r in data["test"]])
    best, log, t0, it = -1, [], time.time(), 0
    model.train()
    for ep in range(a.epochs):
        for b in tr:
            b = {key: v.cuda() for key, v in b.items()}
            y = b["labels"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                log_py, z, log_pi = model.plan(b["ids"], b["mask"])
                loss = F.nll_loss(log_py, y)
                parts = {"plan": loss.item()}
                if not a.direct and (a.lam_r > 0 or a.lam_c > 0):
                    r = model.read(b["rids"], b["rmask"])
                    if a.lam_r > 0:
                        lr_ = F.cross_entropy(model.reader(r).float(), y)
                        loss = loss + a.lam_r * lr_
                        parts["read"] = lr_.item()
                    if a.lam_c > 0:
                        zn = F.normalize(z.float(), dim=-1)
                        rn = F.normalize((r.detach() if a.sg else r).float(), dim=-1)
                        bank = torch.cat([rn, queue])                               # (B+Q) x d
                        s = torch.einsum("bmd,nd->bmn", zn, bank) / a.tau            # B x M x (B+Q)
                        s = torch.logsumexp(log_pi[..., None] + s, 1)                # mixture score
                        lc = F.cross_entropy(s, torch.arange(len(y), device="cuda"))
                        loss = loss + a.lam_c * lc
                        parts["nce"] = lc.item()
                        queue = torch.cat([rn.detach(), queue])[: a.queue]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % eval_every == 0:
                lv = predict(ev["valid"])
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "loss": parts, "time": time.time() - t0}
                if mv["macro_f1"] > best:
                    best = mv["macro_f1"]
                    lt = predict(ev["test"])
                    row["test"] = report(yt, lt.argmax(1), k)
                    np.savez(os.path.join(a.out, "logits.npz"), valid=lv, test=lt, yv=yv, yt=yt)
                    json.dump({"args": vars(a), "valid": mv, "test": row["test"], "it": it},
                              open(os.path.join(a.out, "best.json"), "w"), indent=1)
                log.append(row)
                print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
