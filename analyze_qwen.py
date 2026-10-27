"""Does a language model use the early turn that the probe only finds with the label?"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pointer import masks_from_scores
from src.prepare import build_cache
from src.train_eval import LABELS, split_indices

MODEL = r"C:\Users\Admin\.cache\huggingface\hub\models--Qwen--Qwen2.5-3B-Instruct\snapshots\aa8e72537993ba99e69dfaafa59ed015b17504d1"
NAMES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]


def render(texts, roles, hist_row, keep):
    lines = []
    for j, tid in enumerate(hist_row):
        if tid < 0 or not keep[j]:
            continue
        who = "seeker" if int(roles[tid]) == 0 else "supporter"
        lines.append(f"{who}: {texts[tid].strip()}")
    body = "\n".join(lines)
    options = "\n".join(f"- {n}" for n in NAMES)
    return (
        "Predict the counseling strategy the supporter will use next. "
        "Reply with exactly one strategy name from the list and nothing else.\n"
        f"{options}\n\nDialogue:\n{body}\n\nNext supporter strategy:"
    )


def parse(text):
    found = [i for i, name in enumerate(NAMES) if name.lower() in text.lower()]
    if not found:
        return -1
    return max(found, key=lambda i: len(NAMES[i]))


def main():
    cache = build_cache(".", 0, 42)
    dv = split_indices(cache, 1)
    table = np.load("results/pointer/util_dev.npz")["table"]
    hl = cache["hlen"][dv]
    long = np.where(hl >= 8)[0]
    rng = np.random.default_rng(0)
    take = np.sort(rng.choice(long, size=200, replace=False))
    oracle = masks_from_scores(table[:, :-1], table[:, -1], hl, 4, tau=0.0)
    length = cache["max_l"]
    last = (np.arange(length)[None, :] >= (hl[:, None] - 4)) & (np.arange(length)[None, :] < hl[:, None])
    full = np.arange(length)[None, :] < hl[:, None]
    texts, roles, hist = cache["texts"], cache["role"], cache["hist"]
    y = cache["y"][dv][take]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True, device_map={"": 0}
    )
    model.eval()
    preds = {name: [] for name in ("last4", "full", "oracle")}
    masks = {"last4": last, "full": full, "oracle": oracle}
    for name in preds:
        for n_done, row in enumerate(take):
            prompt = render(texts, roles, hist[dv[row]], masks[name][row])
            messages = [{"role": "user", "content": prompt}]
            ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                out = model.generate(ids, max_new_tokens=16, do_sample=False)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
            preds[name].append(parse(text))
            if n_done % 25 == 24:
                print(f"  {name} {n_done+1}/200", flush=True)
        pred = np.asarray(preds[name])
        valid = pred >= 0
        f1 = f1_score(y[valid], pred[valid], average="macro", labels=list(range(8)), zero_division=0)
        acc = float((pred[valid] == y[valid]).mean()) if valid.any() else 0.0
        print(f"{name} parsed {valid.mean()*100:.0f} acc {acc*100:.1f} macroF1 {f1*100:.1f}", flush=True)


if __name__ == "__main__":
    main()
