"""Downstream effect of strategy planning on generated support responses.

    python -m esc.generate --plans base=<run>/logits.npz post=<run>/logits.npz --n 400 --out runs/esconv/downstream

A fixed LLM (Qwen2.5-3B-Instruct by default, 4-bit NF4) writes the next supporter message for a random sample
of test turns under four conditions: no strategy, the strategy of each planner, and the gold
strategy. Each response is scored by
  BLEU-2 / ROUGE-L against the human reply,
  adherence: whether a message-only strategy recognizer reads the intended strategy,
  judge: a 1-5 rating of supportiveness and appropriateness by the same LLM, blind to condition.
"""

import argparse
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from esc.data import load, render, samples

LLM = os.environ.get("GEN_LLM", "Qwen/Qwen2.5-3B-Instruct")
DEF = {
    "Question": "ask a question to help the seeker explore their situation",
    "Restatement or Paraphrasing": "restate what the seeker said in your own words",
    "Reflection of feelings": "reflect the feelings the seeker is expressing",
    "Self-disclosure": "share a brief, relevant experience of your own",
    "Affirmation and Reassurance": "affirm the seeker's strengths and reassure them",
    "Providing Suggestions": "offer a concrete suggestion",
    "Information": "provide useful information",
    "Others": "respond naturally to keep the conversation going",
}


def bleu2(h, r):
    from collections import Counter
    h, r = h.lower().split(), r.lower().split()
    if not h:
        return 0.0
    ps = []
    for n in (1, 2):
        hc = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
        rc = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        ps.append((sum(min(c, rc[g]) for g, c in hc.items()) + 1e-9) / max(1, sum(hc.values())))
    bp = min(1.0, np.exp(1 - len(r) / len(h)))
    return float(bp * np.sqrt(ps[0] * ps[1]))


def rougel(h, r):
    h, r = h.lower().split(), r.lower().split()
    if not h or not r:
        return 0.0
    dp = [[0] * (len(r) + 1) for _ in range(len(h) + 1)]
    for i in range(len(h)):
        for j in range(len(r)):
            dp[i + 1][j + 1] = dp[i][j] + 1 if h[i] == r[j] else max(dp[i][j + 1], dp[i + 1][j])
    l = dp[-1][-1]
    p, q = l / len(h), l / len(r)
    return 0.0 if l == 0 else 2 * p * q / (p + q)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--plans", nargs="+", required=True)
    a.add_argument("--recog", default="")
    a.add_argument("--n", type=int, default=400)
    a.add_argument("--bs", type=int, default=16)
    a.add_argument("--out", required=True)
    a = a.parse_args()
    os.makedirs(a.out, exist_ok=True)
    splits, names = load("esconv")
    test = samples(splits["test"])
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(test), a.n, replace=False))
    plans = {k: np.load(v)["test"].argmax(1) for k, v in (p.split("=") for p in a.plans)}
    conds = {"none": None, **{k: [names[plans[k][i]] for i in idx] for k in plans},
             "gold": [names[test[i]["label"]] for i in idx]}

    tok = AutoTokenizer.from_pretrained(LLM)
    tok.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(LLM, device_map={"": 0}, torch_dtype=torch.bfloat16,
                                               quantization_config=BitsAndBytesConfig(
                                                   load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16))

    def chat(prompts, max_new=60):
        out = []
        for i in range(0, len(prompts), a.bs):
            msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
                    for p in prompts[i:i + a.bs]]
            enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
            g = llm.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
            out += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for x in g]
        return out

    ctx = [render(test[i]["turns"], names, window=5, tags=False).replace("<START>", "") for i in idx]
    cache = os.path.join(a.out, "gens.json")
    gens = json.load(open(cache)) if os.path.exists(cache) else {}
    for c, strat in conds.items():
        if c in gens:
            continue
        prompts = []
        for j, h in enumerate(ctx):
            p = ("You are a supporter in an emotional support conversation. Conversation so far:\n" + h +
                 "\nWrite the supporter's next message in one or two sentences.")
            if strat is not None:
                p += f" In this message, {DEF[strat[j]]} (strategy: {strat[j]})."
            prompts.append(p + " Output only the message.")
        gens[c] = chat(prompts)
        print("generated", c, flush=True)
        json.dump(gens, open(cache, "w"))

    # blind judge: conditions shuffled per item
    judge = {c: [None] * len(idx) for c in conds}
    jp, keys = [], []
    for j, h in enumerate(ctx):
        for c in conds:
            jp.append("Rate the supporter's reply in this emotional support conversation from 1 (poor) to 5 "
                      "(excellent) for how supportive, empathetic and appropriate it is.\nConversation:\n" + h +
                      "\nReply: " + gens[c][j] + "\nAnswer with a single digit.")
            keys.append((c, j))
    perm = rng.permutation(len(jp))
    outs = chat([jp[i] for i in perm], max_new=3)
    for i, o in zip(perm, outs):
        m = re.search(r"[1-5]", o)
        c, j = keys[i]
        judge[c][j] = int(m.group()) if m else 3

    adh = {}
    if a.recog:
        del llm
        torch.cuda.empty_cache()
        rt = AutoTokenizer.from_pretrained(os.path.join(a.recog, "model"))
        rm = AutoModelForSequenceClassification.from_pretrained(os.path.join(a.recog, "model")).cuda().eval()
        for c, strat in conds.items():
            if strat is None:
                continue
            with torch.no_grad():
                pr = []
                for i in range(0, len(gens[c]), 32):
                    e = rt(gens[c][i:i + 32], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
                    pr += rm(**e).logits.argmax(1).tolist()
            adh[c] = 100 * float(np.mean([names[p] == s for p, s in zip(pr, strat)]))
    ref = [test[i]["response"] for i in idx]
    res = {c: {"bleu2": 100 * np.mean([bleu2(g, r) for g, r in zip(gens[c], ref)]),
               "rougeL": 100 * np.mean([rougel(g, r) for g, r in zip(gens[c], ref)]),
               "judge": float(np.mean(judge[c])), "adherence": adh.get(c)} for c in conds}
    for k in plans:
        res[k]["plan_acc"] = 100 * float(np.mean([plans[k][i] == test[i]["label"] for i in idx]))
    json.dump({"results": res, "idx": [int(i) for i in idx], "gens": gens}, open(os.path.join(a.out, "downstream.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
