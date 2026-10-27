"""Fine-tune a next-strategy classifier and save dev/test logits of the best dev epoch.

Encoders (roberta-large, deberta-v3-large) are fully fine-tuned; decoder LLMs (Qwen)
get LoRA on every linear layer plus a trainable score head.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from esc.data import load, render, samples
from esc.metrics import report


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--window", type=int, default=5, help="0 = full history")
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--lora", type=int, default=0)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--load_4bit", action="store_true", help="QLoRA: NF4 base weights for 24GB shared cards")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--input", default="context", choices=["context", "response", "context+reaction", "context+response"],
                   help="response: recognise the strategy of a written supporter message")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--with_plan", action="store_true", help="prepend the full supporter strategy trajectory")
    p.add_argument("--fold", type=int, default=-1, help="cross-fitting: hold out train dialogues with id %% nfold == fold")
    p.add_argument("--nfold", type=int, default=5)
    p.add_argument("--distill", default="", help="npz with out-of-fold teacher logits for the train split")
    p.add_argument("--save_train_logits", action="store_true",
                   help="ablation teacher without cross-fitting: also score the training split it was fit on")
    p.add_argument("--label_noise", type=float, default=0.0,
                   help="flip this fraction of TRAIN labels to a random other class (fixed noise seed)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--hist_teacher", default="", help="npz with teacher train logits: history labels of training "
                   "inputs are sampled from the teacher posterior, as a deployed planner sees its own decisions")
    p.add_argument("--elr", type=float, default=0.0, help="early-learning regularization weight (Liu et al. 2020)")
    p.add_argument("--elr_beta", type=float, default=0.7)
    p.add_argument("--smooth", type=float, default=0.0, help="label smoothing baseline")
    p.add_argument("--conf_pen", type=float, default=0.0, help="confidence penalty baseline (Pereyra et al. 2017)")
    p.add_argument("--boot", type=float, default=0.0, help="soft bootstrapping baseline (Reed et al. 2015): target beta*y+(1-beta)*p")
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def encode(tok, rows, names, window, max_len, decoder, field="context", with_plan=False):
    if field == "response":
        texts = [r["response"] for r in rows]
    else:
        texts = [render(r["turns"], names, window, with_plan=with_plan) for r in rows]
        if field == "context+reaction":  # hindsight teacher: sees how the seeker reacted
            texts = [t + " [reaction] " + r["reaction"] for t, r in zip(texts, rows)]
        elif field == "context+response":  # hindsight teacher: sees the message itself
            texts = [t + " [response] " + r["response"] for t, r in zip(texts, rows)]
    # keep the end of the context: the most recent turns decide the next move
    tok.truncation_side = "left"
    enc = tok(texts, truncation=True, max_length=max_len)
    return [{"input_ids": i, "label": r["label"], "idx": j} for j, (i, r) in enumerate(zip(enc["input_ids"], rows))]


def collate(tok):
    def f(batch):
        out = tok.pad([{"input_ids": b["input_ids"]} for b in batch], return_tensors="pt")
        out["labels"] = torch.tensor([b["label"] for b in batch])
        out["idx"] = torch.tensor([b["idx"] for b in batch])
        return out
    return f


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    logits = []
    for b in loader:
        b = {k: v.to(dev) for k, v in b.items() if k not in ("labels", "idx")}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(**b).logits.float().cpu())
    model.train()
    return torch.cat(logits).numpy()


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    data = {s: samples(splits[s]) for s in ["train", "valid", "test"]}
    if a.label_noise > 0:  # same corruption for every run at this level; dev/test stay clean
        rng = np.random.default_rng(12345)
        flip = rng.random(len(data["train"])) < a.label_noise
        shift = rng.integers(1, k, len(data["train"]))
        data["train"] = [dict(r, label=int((r["label"] + sh) % k)) if f else r
                         for r, f, sh in zip(data["train"], flip, shift)]
        print(f"label noise {a.label_noise}: flipped {int(flip.sum())} of {len(flip)}", flush=True)
    if a.hist_teacher:
        tz = np.load(a.hist_teacher)["train"]
        tp = np.exp(tz - tz.max(1, keepdims=True))
        tp /= tp.sum(1, keepdims=True)
        rng = np.random.default_rng(1000 + a.seed)
        drawn = np.array([rng.choice(k, p=q) for q in tp])
        start, new = {}, []
        for i, r in enumerate(data["train"]):
            start.setdefault(r["dialog"], i)
            j, turns = start[r["dialog"]], []
            for sp, text, lab in r["turns"]:
                if lab >= 0:
                    turns.append((sp, text, int(drawn[j])))
                    j += 1
                else:
                    turns.append((sp, text, lab))
            new.append(dict(r, turns=turns))
        data["train"] = new
    decoder = bool(a.lora)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left" if decoder else "right"
    enc = {s: encode(tok, data[s], names, a.window, a.max_len, decoder, a.input, a.with_plan) for s in data}

    dtype = torch.bfloat16 if decoder else torch.float32
    kw = {}
    if a.load_4bit:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=k, torch_dtype=dtype, **kw)
    model.config.pad_token_id = tok.pad_token_id
    if decoder:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="SEQ_CLS",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)  # no kbit prep: it upcasts the 152k-row embedding to fp32
        model.gradient_checkpointing_enable() if hasattr(model, "gradient_checkpointing_enable") else None
        model.enable_input_require_grads()
        for n, p_ in model.named_parameters():
            if "score" in n:
                p_.data = p_.data.float()
                p_.requires_grad = True
    else:
        model.gradient_checkpointing_enable()  # same math, fits next to other jobs on a 24GB card
    dev = "cuda"
    if not a.load_4bit:
        model.to(dev)

    held = []
    if a.fold >= 0:
        held = [e for e, r in zip(enc["train"], data["train"]) if r["dialog"] % a.nfold == a.fold]
        enc["train"] = [e for e, r in zip(enc["train"], data["train"]) if r["dialog"] % a.nfold != a.fold]
    teacher = torch.tensor(np.load(a.distill)["train"]).float().to("cuda") if a.distill else None
    elr_t = torch.zeros(len(enc["train"]), k, device="cuda") if a.elr > 0 else None
    g = torch.Generator().manual_seed(a.seed)
    tr = DataLoader(enc["train"], batch_size=a.bs, shuffle=True, collate_fn=collate(tok), generator=g)
    ev = {s: DataLoader(enc[s], batch_size=a.bs * 2, collate_fn=collate(tok)) for s in ["valid", "test"]}
    if held:
        ev["held"] = DataLoader(held, batch_size=a.bs * 2, collate_fn=collate(tok))
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.01)
    steps = a.epochs * math.ceil(len(tr) / a.grad_acc)
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, len(tr) // a.evals_per_epoch)

    best, log, t0 = -1, [], time.time()
    yv = np.array([r["label"] for r in data["valid"]])
    yt = np.array([r["label"] for r in data["test"]])
    model.train()
    it = 0
    for ep in range(a.epochs):
        for b in tr:
            b = {k_: v.to(dev) for k_, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(**{k_: v for k_, v in b.items() if k_ not in ("labels", "idx")}).logits.float()
            loss = F.cross_entropy(logits, b["labels"], label_smoothing=a.smooth)
            if a.conf_pen > 0:
                lp = logits.log_softmax(-1)
                loss = loss - a.conf_pen * (-(lp.exp() * lp).sum(-1)).mean()
            if a.elr > 0:  # temporal-ensemble target of each example; penalize moving away from it
                pr = logits.softmax(-1).clamp(1e-4, 1 - 1e-4)
                with torch.no_grad():
                    q = (pr / pr.sum(-1, keepdim=True)).detach()
                    elr_t[b["idx"]] = a.elr_beta * elr_t[b["idx"]] + (1 - a.elr_beta) * q
                loss = loss + a.elr * torch.log(1 - (elr_t[b["idx"]] * pr).sum(-1)).mean()
            if a.boot > 0:
                lp = logits.log_softmax(-1)
                tgt = a.boot * F.one_hot(b["labels"], k).float() + (1 - a.boot) * lp.exp().detach()
                loss = -(tgt * lp).sum(-1).mean()
            if teacher is not None:
                kd = F.kl_div(F.log_softmax(logits / a.T, -1), F.softmax(teacher[b["idx"]] / a.T, -1), reduction="batchmean")
                loss = (1 - a.alpha) * loss + a.alpha * a.T ** 2 * kd
            loss = loss / a.grad_acc
            loss.backward()
            it += 1
            if it % a.grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
            if it % eval_every == 0:
                lv = predict(model, ev["valid"], dev)
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "time": time.time() - t0}
                if mv["macro_f1"] > best:
                    best = mv["macro_f1"]
                    lt = predict(model, ev["test"], dev)
                    row["test"] = report(yt, lt.argmax(1), k)
                    extra = {}
                    if held:
                        extra = {"held": predict(model, ev["held"], dev), "held_idx": np.array([e["idx"] for e in held])}
                    np.savez(os.path.join(a.out, "logits.npz"), valid=lv, test=lt, yv=yv, yt=yt, **extra)
                    if a.save_train_logits:
                        tr_eval = DataLoader(enc["train"], batch_size=a.bs * 2, collate_fn=collate(tok))
                        np.savez(os.path.join(a.out, "train_logits.npz"), train=predict(model, tr_eval, dev))
                    if a.save_model:
                        model.save_pretrained(os.path.join(a.out, "model"))
                        tok.save_pretrained(os.path.join(a.out, "model"))
                    json.dump({"args": vars(a), "valid": mv, "test": row["test"], "it": it},
                              open(os.path.join(a.out, "best.json"), "w"), indent=1)
                log.append(row)
                print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
