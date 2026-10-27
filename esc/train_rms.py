"""Reaction-Marginalized Strategy Network (RMSN).

The seeker's reaction c is an effect of the supporter's strategy, observed only in hindsight.
RMSN builds the law of total probability into the network:

    p(y | H) = sum_c pi(c | H) q(y | H, c)

pi anticipates the reaction type from the context; q decides the strategy given a reaction
type. Training supervises both factors directly with the observed reaction type c* and the
marginal with the gold strategy:

    L = -log p(y | H)  -  log q(y | H, c*)  -  log pi(c* | H)

At test time only the marginal is used, so no future information is needed.
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
from esc.reactions import path as reaction_path


class RMSN(nn.Module):
    def __init__(self, backbone, k, n_types, d=256, use_hind=True, use_pi=True):
        super().__init__()
        self.enc = AutoModel.from_pretrained(backbone)
        self.enc.gradient_checkpointing_enable()
        hd = self.enc.config.hidden_size
        self.pi = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, n_types))
        self.h = nn.Sequential(nn.LayerNorm(hd), nn.Linear(hd, d))
        self.rtype = nn.Embedding(n_types, d)
        self.q = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, k))
        self.use_hind, self.use_pi = use_hind, use_pi

    def forward(self, input_ids, attention_mask):
        h = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        log_pi = self.pi(h).log_softmax(-1)                                    # (B, C)
        hp = self.h(h)
        z = torch.cat([hp[:, None].expand(-1, self.rtype.num_embeddings, -1),
                       self.rtype.weight[None].expand(len(h), -1, -1)], -1)
        log_q = self.q(z).log_softmax(-1)                                      # (B, C, K)
        log_p = torch.logsumexp(log_pi[:, :, None] + log_q, 1)                 # (B, K)
        return log_p, log_q, log_pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--types", type=int, default=32)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--with_plan", action="store_true")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--no_hind", action="store_true", help="ablation: drop the -log q(y|H,c*) term")
    p.add_argument("--no_pi", action="store_true", help="ablation: drop the -log pi(c*|H) term")
    p.add_argument("--shuffle_types", action="store_true",
                   help="control: permute reaction types across training examples, destroying the reaction signal")
    p.add_argument("--distill", default="", help="optional out-of-fold teacher logits on train")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    data = {s: samples(splits[s]) for s in ["train", "valid", "test"]}
    types = np.load(reaction_path(a.dataset, a.types))["train"]
    if a.shuffle_types:
        types = np.random.default_rng(a.seed).permutation(types)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "left"

    def enc(rows, with_type):
        ids = tok([render(r["turns"], names, a.window, with_plan=a.with_plan) for r in rows],
                  truncation=True, max_length=a.max_len)["input_ids"]
        return [{"input_ids": x, "label": r["label"], "idx": i, "type": int(types[i]) if with_type else -1}
                for i, (x, r) in enumerate(zip(ids, rows))]

    def collate(batch):
        out = tok.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
        for key in ["label", "idx", "type"]:
            out[key] = torch.tensor([b[key] for b in batch])
        return out

    E = {"train": enc(data["train"], True), "valid": enc(data["valid"], False), "test": enc(data["test"], False)}
    model = RMSN(a.model, k, a.types).cuda()
    teacher = torch.tensor(np.load(a.distill)["train"]).float().cuda() if a.distill else None

    g = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(E["train"], batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
    ev = {s: DataLoader(E[s], batch_size=a.bs * 2, collate_fn=collate) for s in ["valid", "test"]}
    enc_p = [q for n, q in model.named_parameters() if n.startswith("enc.")]
    head_p = [q for n, q in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_p, "lr": a.lr}, {"params": head_p, "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * len(tr)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, len(tr) // a.evals_per_epoch)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        out = []
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(model(b["input_ids"].cuda(), b["attention_mask"].cuda())[0].float().cpu())
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
                log_p, log_q, log_pi = model(b["input_ids"], b["attention_mask"])
            log_p, log_q, log_pi = log_p.float(), log_q.float(), log_pi.float()
            loss = F.nll_loss(log_p, b["label"])
            if teacher is not None:
                kd = F.kl_div(F.log_softmax(log_p / a.T, -1), F.softmax(teacher[b["idx"]] / a.T, -1), reduction="batchmean")
                loss = (1 - a.alpha) * loss + a.alpha * a.T ** 2 * kd
            if not a.no_hind:
                loss = loss + F.nll_loss(log_q[torch.arange(len(log_q)), b["type"]], b["label"])
            if not a.no_pi:
                loss = loss + F.nll_loss(log_pi, b["type"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_p + head_p, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % eval_every == 0:
                lv = predict(ev["valid"])
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "time": time.time() - t0}
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
