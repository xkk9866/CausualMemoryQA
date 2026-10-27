"""Simulate the supporter's next message with a zero-shot instruction LLM.

The simulator never sees the gold response or strategy, and it reads the same window as
the decision model, so train and test drafts come from the same distribution.
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from esc.data import load, samples

SYSTEM = ("You are an experienced emotional supporter trained in counseling helping skills. "
          "You talk with a help-seeker who is going through a hard time.")
ASK = ("Here is the latest part of the conversation.\n\n{dialog}\n\n"
       "Write the supporter's next message. Reply with that one message only, one or two sentences.")


def dialog_text(turns, window):
    lines = []
    for sp, text, _ in turns[-window:]:
        if sp == "none":
            continue
        lines.append(f"{'Help-seeker' if sp == 'seeker' else 'Supporter'}: {text}")
    return "\n".join(lines) if lines else "(The conversation has not started yet.)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--splits", nargs="+", default=["valid", "test", "train"])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--n", type=int, default=1, help="drafts per context (n>1 samples at T=0.8)")
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--max_new", type=int, default=64)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    splits, _ = load(a.dataset)
    for s in a.splits:
        path = os.path.join(a.out, f"{s}.jsonl")
        done = sum(1 for _ in open(path, encoding="utf-8")) if os.path.exists(path) else 0
        rows = samples(splits[s])
        prompts = [tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                            {"role": "user", "content": ASK.format(dialog=dialog_text(r["turns"], a.window))}],
                                           tokenize=False, add_generation_prompt=True) for r in rows]
        with open(path, "a", encoding="utf-8") as f:
            for i in range(done, len(prompts), a.bs):
                enc = tok(prompts[i:i + a.bs], return_tensors="pt", padding=True).to("cuda")
                kw = dict(do_sample=True, temperature=0.8, top_p=0.95, num_return_sequences=a.n) if a.n > 1 else dict(do_sample=False)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id, **kw)
                text = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for j in range(len(text) // a.n):
                    f.write(json.dumps({"i": i + j, "drafts": [t.strip() for t in text[j * a.n:(j + 1) * a.n]]}) + "\n")
                f.flush()
                print(s, i + a.bs, "/", len(prompts), flush=True)


if __name__ == "__main__":
    main()
