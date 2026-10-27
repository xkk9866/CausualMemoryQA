"""Prompted LLM strategy prediction (zero-shot / k-shot) with a quantized model.

    python -m esc.prompt_llm --dataset esconv --model <path to Qwen2.5-14B-Instruct-AWQ> --shots 2 --out runs/esconv/qwen14b_awq_2shot_s0

The model reads the same input as our classifiers (strategy trajectory + five-utterance window)
and every strategy name is scored by its log-likelihood as the answer, so each test example gets
a valid prediction without answer parsing. Shots are drawn from the training split with --seed.
Writes logits.npz / best.json in the same format as train_cls, so the reporting scripts apply.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from esc.data import load, render, samples
from esc.metrics import report

DEFS = {
    "esconv": "Strategies: Question; Restatement or Paraphrasing; Reflection of feelings; Self-disclosure; "
              "Affirmation and Reassurance; Providing Suggestions; Information; Others.",
    "annomi": "Therapist behaviours: Reflection simple; Reflection complex; Question closed; Question open; "
              "Provide information; Provide suggestion; Other.",
}


def prompt(names, ds, ctx, shots):
    sys_msg = ("You are an expert in counseling. Given a conversation, choose the strategy the supporter "
               "uses in the next message. " + DEFS[ds] + " Answer with the strategy name only.")
    msgs = [{"role": "system", "content": sys_msg}]
    for c, y in shots:
        msgs += [{"role": "user", "content": "Conversation:\n" + c + "\nNext strategy:"},
                 {"role": "assistant", "content": names[y]}]
    msgs.append({"role": "user", "content": "Conversation:\n" + ctx + "\nNext strategy:"})
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", required=True)
    p.add_argument("--shots", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_ctx", type=int, default=1536)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    splits, names = load(a.dataset)
    k = len(names)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16, device_map={"": 0}).eval()
    rng = np.random.default_rng(a.seed)
    train = samples(splits["train"])
    shots = [(render(train[i]["turns"], names, 5, with_plan=True), train[i]["label"])
             for i in rng.choice(len(train), a.shots, replace=False)] if a.shots else []
    name_ids = [tok(n, add_special_tokens=False)["input_ids"] for n in names]

    def score(rows):
        out = []
        for r in rows:
            msgs = prompt(names, a.dataset, render(r["turns"], names, 5, with_plan=True), shots)
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            pre_ids = tok(pre, add_special_tokens=False)["input_ids"][-a.max_ctx:]
            # one forward pass per candidate name; mean log-prob of the name tokens
            batch = [pre_ids + ids for ids in name_ids]
            L = max(map(len, batch))
            inp = torch.full((len(batch), L), tok.pad_token_id or 0)
            att = torch.zeros((len(batch), L), dtype=torch.long)
            for i, b in enumerate(batch):
                inp[i, :len(b)] = torch.tensor(b)
                att[i, :len(b)] = 1
            with torch.no_grad():
                # hidden states only; the LM head is applied just where the strategy-name tokens are predicted
                h = model.model(input_ids=inp.cuda(), attention_mask=att.cuda()).last_hidden_state
                s = []
                for i, ids in enumerate(name_ids):
                    pos = torch.arange(len(pre_ids) - 1, len(pre_ids) - 1 + len(ids), device=h.device)
                    lp = model.lm_head(h[i, pos]).float().log_softmax(-1)
                    s.append(lp[torch.arange(len(ids)), torch.tensor(ids, device=h.device)].mean().item())
            out.append(s)
        return np.array(out)

    res = {}
    for split in ["valid", "test"]:
        rows = samples(splits[split])
        res[split] = score(rows)
        res["y" + split[0]] = np.array([r["label"] for r in rows])
        print(split, report(res["y" + split[0]], res[split].argmax(1), k), flush=True)
    np.savez(os.path.join(a.out, "logits.npz"), valid=res["valid"], test=res["test"], yv=res["yv"], yt=res["yt"])
    json.dump({"args": vars(a), "valid": report(res["yv"], res["valid"].argmax(1), k),
               "test": report(res["yt"], res["test"].argmax(1), k), "it": 0},
              open(os.path.join(a.out, "best.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
