"""Strategy distribution of simulated replies, read by the response recogniser."""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--drafts", required=True, help="directory with <split>.jsonl")
    p.add_argument("--splits", nargs="+", default=["valid", "test"])
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model).cuda().eval()
    out = {}
    for s in a.splits:
        rows = [json.loads(l) for l in open(os.path.join(a.drafts, f"{s}.jsonl"), encoding="utf-8")]
        rows.sort(key=lambda r: r["i"])
        texts = [d for r in rows for d in r["drafts"]]
        n = len(rows[0]["drafts"])
        logp = []
        for i in range(0, len(texts), 128):
            enc = tok(texts[i:i + 128], truncation=True, max_length=128, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                logp.append(model(**enc).logits.float().log_softmax(-1).cpu())
        out[s] = torch.cat(logp).numpy().reshape(len(rows), n, -1)
        print(s, out[s].shape)
    np.savez(os.path.join(a.drafts, "recog.npz"), **out)


if __name__ == "__main__":
    main()
