"""Train RMN; saves dev/test log-probabilities of the best dev checkpoint."""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from esc.bank import path as bank_path
from esc.data import load, render, samples
from esc.metrics import report
from esc.rmn import RMN


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--bank", default="BAAI/bge-large-en-v1.5")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the response InfoNCE")
    p.add_argument("--no_med", action="store_true")
    p.add_argument("--no_value", action="store_true")
    p.add_argument("--memory", default="response", choices=["response", "context"],
                   help="ablation: key the memory by the neighbours' contexts instead of their responses")
    p.add_argument("--shuffle_bank", action="store_true", help="ablation: break response-strategy pairing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    data = {s: samples(splits[s]) for s in ["train", "valid", "test"]}
    bank = np.load(bank_path(a.dataset, a.bank))
    assert (bank["train_label"] == np.array([r["label"] for r in data["train"]])).all()
    blab = bank["train_label"].copy()
    if a.shuffle_bank:
        np.random.default_rng(a.seed).shuffle(blab)

    decoder = bool(a.lora)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" if decoder else "right"
    tok.truncation_side = "left"

    def enc(rows):
        ids = tok([render(r["turns"], names, a.window) for r in rows], truncation=True, max_length=a.max_len)["input_ids"]
        return [{"input_ids": x, "label": r["label"], "dialog": r["dialog"], "index": i} for i, (x, r) in enumerate(zip(ids, rows))]

    def collate(batch):
        out = tok.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
        for key in ["label", "dialog", "index"]:
            out[key] = torch.tensor([b[key] for b in batch])
        return out

    E = {s: enc(data[s]) for s in data}
    lora = None
    if decoder:
        from peft import LoraConfig
        lora = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    mem = bank["train"] if a.memory == "response" else bank["train_ctx"]
    model = RMN(a.model, torch.tensor(mem), torch.tensor(blab), torch.tensor(bank["train_dialog"]), k,
                decoder=decoder, lora=lora, use_med=not a.no_med, use_value=not a.no_value,
                dtype=torch.bfloat16 if decoder else torch.float32).cuda()

    g = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(E["train"], batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
    ev = {s: DataLoader(E[s], batch_size=a.bs * 2, collate_fn=collate) for s in ["valid", "test"]}
    enc_params = [p for n, p in model.named_parameters() if n.startswith("enc.") and p.requires_grad]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head_params, "lr": a.head_lr}], weight_decay=0.01)
    steps = a.epochs * len(tr)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, len(tr) // a.evals_per_epoch)

    @torch.no_grad()
    def predict(loader):
        model.eval()
        lp, gate, hit = [], [], []
        for b in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b["input_ids"].cuda(), b["attention_mask"].cuda())
            lp.append(o["logp"].float().cpu())
            gate.append(o["gate"].float().cpu() if "gate" in o else torch.ones(len(b["label"]), 1))
        model.train()
        return torch.cat(lp).numpy(), torch.cat(gate).numpy()

    yv = np.array([r["label"] for r in data["valid"]])
    yt = np.array([r["label"] for r in data["test"]])
    best, log, t0, it = -1, [], time.time(), 0
    model.train()
    for ep in range(a.epochs):
        for b in tr:
            b = {key: v.cuda() for key, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(b["input_ids"], b["attention_mask"], dialog=b["dialog"])
            ls = F.nll_loss(o["logp"].float(), b["label"])
            lr_ = F.cross_entropy(o["score"].float(), b["index"]) if a.lam > 0 else torch.zeros((), device="cuda")
            loss = ls + a.lam * lr_
            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc_params + head_params, 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % 100 == 0:
                print(json.dumps({"it": it, "ls": round(ls.item(), 4), "lr": round(lr_.item(), 4),
                                  "tau": round(model.log_tau.exp().item(), 4)}), flush=True)
            if it % eval_every == 0:
                lv, gv = predict(ev["valid"])
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "gate": float(gv.mean()), "time": time.time() - t0}
                if mv["macro_f1"] > best:
                    best = mv["macro_f1"]
                    lt, gt = predict(ev["test"])
                    row["test"] = report(yt, lt.argmax(1), k)
                    np.savez(os.path.join(a.out, "logits.npz"), valid=lv, test=lt, yv=yv, yt=yt, gate_v=gv, gate_t=gt)
                    json.dump({"args": vars(a), "valid": mv, "test": row["test"], "it": it},
                              open(os.path.join(a.out, "best.json"), "w"), indent=1)
                log.append(row)
                print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
