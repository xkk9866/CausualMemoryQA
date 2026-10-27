"""IRIS: Imagined-Response Inference for Strategy planning (dialogue-level model).

Self-annotated strategy labels are noisy descriptions of messages that are themselves observed
exactly. IRIS plans in the space of messages and reads the strategy off the imagined message:

    u_j      = E(text_j)                                    utterance encoder (shared)
    s_t      = D(u_<t, speaker_<t, label_<t)                causal dialogue transformer over the
                                                            whole history incl. the supporter's own
                                                            past (message, label) pairs
    z_tm     = f_m(s_t),  pi_t = softmax(w s_t)             M imagined next messages
    p(y_t|H) = sum_m pi_tm  softmax(g(z_tm))                read the imagined messages
    r_t      = h(u_t)                                       the real message, same space

    L = NLL(y_t | H)                                        plan
      + lam_r CE(g(r_t), y_t)                               read: g learns on real messages
      + lam_c InfoNCE_mix(z_t, sg(r_t))                     imagine: the real reply among all
                                                            replies of the batch

Each dialogue is encoded once and yields a prediction at every labelled supporter turn, in the
same order as esc.data.samples, so logits are directly comparable with the window classifiers.
Ablations: --M 1, --lam_c 0, --lam_r 0, --direct (classifier on s_t), --no_labels (labels of past
turns hidden), --mem W (dialogue transformer sees only the last W utterances).
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
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from esc.data import load
from esc.metrics import report


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--utt_len", type=int, default=64)
    p.add_argument("--lora", type=int, default=0, help="decoder LLM utterance encoder: LoRA rank (QLoRA with --load_4bit)")
    p.add_argument("--load_4bit", action="store_true")
    p.add_argument("--pair", type=int, default=1, help="encode each utterance together with the previous one")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--head_lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--bs", type=int, default=4, help="dialogues per batch")
    p.add_argument("--chunk", type=int, default=64, help="utterances per encoder forward")
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d", type=int, default=512)
    p.add_argument("--M", type=int, default=4)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--lam_r", type=float, default=1.0)
    p.add_argument("--lam_c", type=float, default=0.5)
    p.add_argument("--queue", type=int, default=2048)
    p.add_argument("--direct", action="store_true")
    p.add_argument("--hybrid", type=int, default=0, help="1: fuse a token-level window+trajectory encoding into s_t")
    p.add_argument("--win_len", type=int, default=320)
    p.add_argument("--adapt", type=int, default=0, help="1: the reader is FiLM-conditioned on the labeler state s_t")
    p.add_argument("--no_labels", action="store_true")
    p.add_argument("--mem", type=int, default=0, help=">0: attend to the last W utterances only")
    p.add_argument("--distill", default="", help="npz with per-sample teacher logits for train (key 'train')")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--label_noise", type=float, default=0.0)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--evals_per_epoch", type=int, default=2)
    p.add_argument("--save_train", action="store_true", help="also save train-split logits (teacher)")
    p.add_argument("--out", required=True)
    return p.parse_args()


SEEKER, SUPPORTER = 0, 1


class IRIS(nn.Module):
    def __init__(self, a, k):
        super().__init__()
        self.a, self.k = a, k
        self.decoder = bool(a.lora)
        if self.decoder:  # LLM utterance encoder: NF4 weights + LoRA, last-token pooling
            from peft import LoraConfig, get_peft_model
            kw = {"torch_dtype": torch.bfloat16}
            if a.load_4bit:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                               bnb_4bit_compute_dtype=torch.bfloat16,
                                                               bnb_4bit_use_double_quant=True)
                kw["device_map"] = {"": 0}
            self.enc = AutoModel.from_pretrained(a.model, **kw)
            self.enc.gradient_checkpointing_enable()
            self.enc.enable_input_require_grads()
            self.enc = get_peft_model(self.enc, LoraConfig(
                r=a.lora, lora_alpha=2 * a.lora, lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        else:
            self.enc = AutoModel.from_pretrained(a.model)
            self.enc.gradient_checkpointing_enable()
        h = self.enc.config.hidden_size
        self.inp = nn.Linear(h, a.d)
        self.spk = nn.Embedding(2, a.d)
        self.lab = nn.Embedding(k + 1, a.d)          # k = no label (seeker, or hidden)
        self.pos = nn.Embedding(1024, a.d)
        self.start = nn.Parameter(torch.zeros(a.d))
        layer = nn.TransformerEncoderLayer(a.d, 8, 4 * a.d, dropout=0.1, batch_first=True, norm_first=True)
        self.dial = nn.TransformerEncoder(layer, a.layers)
        self.out_norm = nn.LayerNorm(a.d)
        self.imagine = nn.Sequential(nn.Linear(a.d, a.d), nn.GELU(), nn.Linear(a.d, a.M * a.dim))
        self.mix = nn.Linear(a.d, a.M)
        self.rproj = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, a.dim))
        self.norm = nn.LayerNorm(a.dim)
        self.reader = nn.Linear(a.dim, k)
        self.film = nn.Linear(a.d, 2 * a.dim)          # labeler-adaptive reading (zero-init: plain g)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.direct = nn.Linear(a.d, k)
        self.win = nn.Sequential(nn.Linear(h, a.d), nn.GELU(), nn.Linear(a.d, a.d))
        self.fuse = nn.LayerNorm(a.d)

    def encode(self, ids, mask):
        out = []
        for i in range(0, len(ids), self.a.chunk):
            m = mask[i:i + self.a.chunk]
            hs = self.enc(input_ids=ids[i:i + self.a.chunk], attention_mask=m).last_hidden_state
            if self.decoder:  # right padding: the last real token summarises the utterance
                out.append(hs[torch.arange(len(hs), device=hs.device), m.sum(1) - 1].float())
            else:
                out.append(hs[:, 0])
        return torch.cat(out)

    def forward(self, b):
        """b: ids/mask (U x L) for all utterances of the batch; dlg (B x N) utterance index or -1;
        spk, lab (B x N); returns states at every position (B x N x d) and utterance vectors."""
        u = self.encode(b["ids"], b["mask"])                                      # U x h
        B, N = b["dlg"].shape
        x = self.inp(u.float())[b["dlg"].clamp(min=0)]                            # B x N x d
        x = x + self.spk(b["spk"]) + self.lab(b["lab_in"]) + self.pos(torch.arange(N, device=x.device) + 1)
        x = torch.cat([(self.start + self.pos.weight[0]).expand(B, 1, -1), x], 1)  # START first
        L = N + 1
        idx = torch.arange(L, device=x.device)
        causal = idx[None, :] > idx[:, None]
        if self.a.mem > 0:  # only the last W utterances (plus START) are visible
            causal = causal | ((idx[:, None] - idx[None, :] > self.a.mem) & (idx[None, :] > 0))
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), b["dlg"] < 0], 1)
        s = self.out_norm(self.dial(x, mask=causal, src_key_padding_mask=pad))    # s[:, t] sees START, utt_<t
        return s[:, :N], u

    def g(self, x, s):
        """strategy reader; with --adapt it reads a message the way this labeler names it"""
        if self.a.adapt:
            gam, bet = self.film(s).chunk(2, -1)
            if x.dim() == s.dim() + 1:  # x carries the M hypotheses
                gam, bet = gam.unsqueeze(-2), bet.unsqueeze(-2)
            x = x * (1 + gam) + bet
        return self.reader(x)

    def local(self, s_m, b):
        """hybrid: add the token-level reading of the window (+ trajectory) at every labelled turn"""
        if not self.a.hybrid:
            return s_m
        out = []
        for i in range(0, len(b["wids"]), self.a.chunk // 2):
            out.append(self.enc(input_ids=b["wids"][i:i + self.a.chunk // 2],
                                attention_mask=b["wmask"][i:i + self.a.chunk // 2]).last_hidden_state[:, 0])
        return self.fuse(s_m + self.win(torch.cat(out).float()))

    def plan(self, s):
        if self.a.direct:
            return self.direct(s).float().log_softmax(-1), None, None
        z = self.norm(self.imagine(s).view(*s.shape[:-1], self.a.M, self.a.dim))
        log_pi = self.mix(s).float().log_softmax(-1)
        log_py = torch.logsumexp(log_pi[..., None] + self.g(z, s).float().log_softmax(-1), -2)
        return log_py, z, log_pi

    def read(self, u):
        return self.norm(self.rproj(u.float()))


def build(dialogues, names, tok, a, noise=None):
    from esc.data import render
    """Per dialogue: tokenised utterances, speakers, labels and the sample index of every
    labelled supporter turn (matching esc.data.samples order)."""
    out, n = [], 0
    for d in dialogues:
        texts = [t for _, t, _ in d]
        if a.pair:
            enc_in = [(texts[j - 1] if j else "<START>", texts[j]) for j in range(len(texts))]
            ids = tok([p for p, _ in enc_in], [c for _, c in enc_in], truncation="longest_first",
                      max_length=a.utt_len * 2)["input_ids"]
        else:
            ids = tok(texts, truncation=True, max_length=a.utt_len)["input_ids"]
        spk = [SUPPORTER if s == "supporter" else SEEKER for s, _, _ in d]
        lab = [y for _, _, y in d]
        sidx, wins, hist = [], [], [("none", "<START>", -1)]
        for j, y in enumerate(lab):
            if y >= 0 and a.hybrid:
                wins.append(render(hist, names, 5, with_plan=True))
            hist.append(d[j])
            if y >= 0:
                sidx.append(n)
                n += 1
            else:
                sidx.append(-1)
        if a.hybrid:
            tok.truncation_side = "left"
            wins = tok(wins, truncation=True, max_length=a.win_len)["input_ids"] if wins else []
            tok.truncation_side = "right"
        out.append({"ids": ids, "spk": spk, "lab": lab, "sidx": sidx, "wins": wins})
    if noise is not None:
        flip, shift, k = noise
        for d in out:
            d["lab"] = [int((y + shift[i]) % k) if (i >= 0 and flip[i]) else y for y, i in zip(d["lab"], d["sidx"])]
    return out, n


def collate(batch, tok, k, a):
    utt, dlg = [], []
    N = max(len(d["ids"]) for d in batch)
    D = torch.full((len(batch), N), -1, dtype=torch.long)
    spk = torch.zeros(len(batch), N, dtype=torch.long)
    lab = torch.full((len(batch), N), -1, dtype=torch.long)
    sidx = torch.full((len(batch), N), -1, dtype=torch.long)
    for i, d in enumerate(batch):
        for j, ids in enumerate(d["ids"]):
            D[i, j] = len(utt)
            utt.append({"input_ids": ids})
        n = len(d["ids"])
        spk[i, :n] = torch.tensor(d["spk"])
        lab[i, :n] = torch.tensor(d["lab"])
        sidx[i, :n] = torch.tensor(d["sidx"])
    p = tok.pad(utt, return_tensors="pt")
    if a.hybrid:
        w = tok.pad([{"input_ids": x} for d in batch for x in d["wins"]], return_tensors="pt")
    lab_in = lab.clone()
    lab_in[lab_in < 0] = k
    if a.no_labels:
        lab_in[:] = k
    return {"ids": p["input_ids"], "mask": p["attention_mask"], "dlg": D, "spk": spk,
            "lab": lab, "lab_in": lab_in, "sidx": sidx,
            **({"wids": w["input_ids"], "wmask": w["attention_mask"]} if a.hybrid else {})}


def main():
    a = parse()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    splits, names = load(a.dataset)
    k = len(names)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.truncation_side = "right"
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    noise = None
    if a.label_noise > 0:  # identical corruption to esc.train_cls (same rng, same sample order)
        ntr = sum(1 for d in splits["train"] for _, _, y in d if y >= 0)
        rng = np.random.default_rng(12345)
        noise = (rng.random(ntr) < a.label_noise, rng.integers(1, k, ntr), k)
    data, count = {}, {}
    for s in ["train", "valid", "test"]:
        data[s], count[s] = build(splits[s], names, tok, a, noise if s == "train" else None)
    train_dialogs = list(range(len(data["train"])))
    if a.train_frac < 1:
        rng = np.random.default_rng(0)
        train_dialogs = sorted(rng.permutation(len(train_dialogs))[: int(a.train_frac * len(train_dialogs))])
    teacher = torch.tensor(np.load(a.distill)["train"]).float().cuda() if a.distill else None

    model = IRIS(a, k).cuda()
    enc_params = list(model.enc.parameters())
    enc_ids = {id(p_) for p_ in enc_params}
    head = [p_ for p_ in model.parameters() if id(p_) not in enc_ids]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": a.lr}, {"params": head, "lr": a.head_lr}], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_dialogs) / a.bs)
    steps = a.epochs * steps_per_epoch
    sch = get_cosine_schedule_with_warmup(opt, int(0.06 * steps), steps)
    eval_every = max(1, steps_per_epoch // a.evals_per_epoch)
    queue = torch.zeros(0, a.dim, device="cuda")
    rng = np.random.default_rng(a.seed)

    def batches(split, order):
        for i in range(0, len(order), a.bs):
            yield collate([data[split][j] for j in order[i:i + a.bs]], tok, k, a)

    @torch.no_grad()
    def predict(split):
        model.eval()
        out = np.zeros((count[split], k), dtype=np.float32)
        order = list(range(len(data[split])))
        for b in batches(split, order):
            b = {key: v.cuda() for key, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, _ = model(b)
                m = b["sidx"] >= 0
                log_py = model.plan(model.local(s[m], b))[0]
            out[b["sidx"][m].cpu().numpy()] = log_py.float().cpu().numpy()
        model.train()
        return out

    @torch.no_grad()
    def imagination(split):
        """how well the imagined messages pick out the real reply among all replies of the split"""
        if a.direct:
            return {}
        model.eval()
        Z, P, Rr, Hm = [], [], [], []
        for b in batches(split, list(range(len(data[split])))):
            b = {key: v.cuda() for key, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, u = model(b)
                m = b["sidx"] >= 0
                _, z, log_pi = model.plan(model.local(s[m], b))
                r = model.read(u[b["dlg"][m]])
            Z.append(F.normalize(z.float(), dim=-1))
            P.append(log_pi.float())
            Rr.append(F.normalize(r.float(), dim=-1))
            Hm.append(log_pi.float().argmax(-1))
        Z, P, Rr, Hm = torch.cat(Z), torch.cat(P), torch.cat(Rr), torch.cat(Hm)
        ranks = []
        for i in range(0, len(Z), 256):
            sc = torch.logsumexp(P[i:i + 256, :, None] + torch.einsum("bmd,nd->bmn", Z[i:i + 256], Rr) / a.tau, 1)
            gold = sc[torch.arange(len(sc)), torch.arange(i, i + len(sc))]
            ranks.append((sc > gold[:, None]).sum(1) + 1)
        ranks = torch.cat(ranks).float()
        use = torch.bincount(Hm, minlength=a.M).float() / len(Hm)
        model.train()
        return {"r1": (ranks <= 1).float().mean().item() * 100, "r10": (ranks <= 10).float().mean().item() * 100,
                "mrr": (1 / ranks).mean().item() * 100, "n": len(ranks), "pi_entropy": -(P.exp() * P).sum(-1).mean().item(),
                "hyp_usage": use.tolist()}

    def labels(split):
        y = np.zeros(count[split], dtype=np.int64)
        for d in data[split]:
            for j, i in enumerate(d["sidx"]):
                if i >= 0:
                    y[i] = d["lab"][j]
        return y

    yv, yt = labels("valid"), labels("test")
    best, log, t0, it = -1, [], time.time(), 0
    model.train()
    for ep in range(a.epochs):
        order = [train_dialogs[i] for i in rng.permutation(len(train_dialogs))]
        for b in batches("train", order):
            b = {key: v.cuda() for key, v in b.items()}
            m = b["sidx"] >= 0                                   # labelled supporter turns
            y = b["lab"][m]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                s, u = model(b)
                sm = model.local(s[m], b)
                log_py, z, log_pi = model.plan(sm)
                loss = F.nll_loss(log_py, y)
                if teacher is not None:
                    kd = F.kl_div((log_py / a.T).log_softmax(-1), F.softmax(teacher[b["sidx"][m]] / a.T, -1), reduction="batchmean")
                    loss = (1 - a.alpha) * loss + a.alpha * a.T ** 2 * kd
                parts = {"plan": loss.item()}
                if not a.direct and (a.lam_r > 0 or a.lam_c > 0):
                    r = model.read(u[b["dlg"][m]])
                    if a.lam_r > 0:
                        lr_ = F.cross_entropy(model.g(r, sm).float(), y)
                        loss = loss + a.lam_r * lr_
                        parts["read"] = lr_.item()
                    if a.lam_c > 0:
                        zn = F.normalize(z.float(), dim=-1)
                        rn = F.normalize(r.detach().float(), dim=-1)
                        bank = torch.cat([rn, queue])
                        sc = torch.logsumexp(log_pi[..., None] + torch.einsum("bmd,nd->bmn", zn, bank) / a.tau, 1)
                        lc = F.cross_entropy(sc, torch.arange(len(y), device="cuda"))
                        loss = loss + a.lam_c * lc
                        parts["nce"] = lc.item()
                        queue = torch.cat([rn, queue])[: a.queue]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
            it += 1
            if it % eval_every == 0:
                lv = predict("valid")
                mv = report(yv, lv.argmax(1), k)
                row = {"epoch": ep, "it": it, "valid": mv, "loss": parts, "time": time.time() - t0}
                if mv["macro_f1"] > best:
                    best = mv["macro_f1"]
                    lt = predict("test")
                    row["test"] = report(yt, lt.argmax(1), k)
                    extra = {"train": predict("train")} if a.save_train else {}
                    np.savez(os.path.join(a.out, "logits.npz"), valid=lv, test=lt, yv=yv, yt=yt, **extra)
                    row["imag"] = imagination("test")
                    json.dump({"args": vars(a), "valid": mv, "test": row["test"], "imag": row["imag"], "it": it},
                              open(os.path.join(a.out, "best.json"), "w"), indent=1)
                log.append(row)
                print(json.dumps(row), flush=True)
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
