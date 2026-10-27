"""Deployment-time evaluation of a saved trajectory classifier.

    python -m esc.rollout <run_dir> [--dataset esconv] [--window 5]

The offline protocol feeds the gold labels of earlier supporter messages. A deployed planner only
has its own past decisions. Two settings replace them:
  rollout  every earlier label in the history (trajectory and window tags) is the model's own
           prediction for that turn, produced turn by turn in dialogue order;
  noise e  each earlier label is independently replaced by a random other strategy with rate e.
Writes <run_dir>/rollout.json.
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from esc.data import load, render
from esc.metrics import report


@torch.no_grad()
def score(model, tok, texts, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.float().cpu())
    return torch.cat(out).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--window", type=int, default=5)
    a = p.parse_args()
    splits, names = load(a.dataset)
    k = len(names)
    tok = AutoTokenizer.from_pretrained(os.path.join(a.run, "model"))
    tok.truncation_side = "left"
    model = AutoModelForSequenceClassification.from_pretrained(os.path.join(a.run, "model")).cuda().eval()
    dialogs = splits["test"]
    res, preds = {}, {}

    def run_with(relabel, tag):
        """relabel(dialog index, turn index, gold) -> label shown in the history"""
        texts, ys = [], []
        for di, d in enumerate(dialogs):
            hist = [("none", "<START>", -1)]
            for t, (sp, text, s) in enumerate(d):
                if s >= 0:
                    texts.append(render(hist, names, a.window, with_plan=True))
                    ys.append(s)
                hist.append((sp, text, relabel(di, t, s) if s >= 0 else s))
        z = score(model, tok, texts)
        preds[tag] = z.argmax(1)
        return report(np.array(ys), z.argmax(1), k)

    res["gold"] = run_with(lambda di, t, s: s, "gold")
    rng = np.random.default_rng(0)
    for e in [0.1, 0.2, 0.4]:
        flips = {}

        def noisy(di, t, s, e=e):
            key = (di, t)
            if key not in flips:
                flips[key] = int((s + rng.integers(1, k)) % k) if rng.random() < e else s
            return flips[key]
        res[f"noise{e}"] = run_with(noisy, f"noise{e}")

    # rollout: the model's own predictions, turn by turn
    ys, rpred = [], []
    for d in dialogs:
        hist = [("none", "<START>", -1)]
        for sp, text, s in d:
            if s >= 0:
                y_hat = int(score(model, tok, [render(hist, names, a.window, with_plan=True)])[0].argmax())
                ys.append(s)
                rpred.append(y_hat)
                hist.append((sp, text, y_hat))
            else:
                hist.append((sp, text, s))
    res["rollout"] = report(np.array(ys), np.array(rpred), k)
    preds["rollout"] = np.array(rpred)
    np.savez(os.path.join(a.run, "rollout_preds.npz"), y=np.array(ys), **preds)
    json.dump(res, open(os.path.join(a.run, "rollout.json"), "w"), indent=1)
    print(json.dumps({key: round(v["macro_f1"], 2) for key, v in res.items()}))


if __name__ == "__main__":
    main()
