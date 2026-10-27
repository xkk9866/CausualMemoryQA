"""Evaluate the full-history RoBERTa on the recent window, and the reverse."""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from run_roberta import MODEL, build_text, masks_for, run_split
from src.prepare import build_cache
from src.train_eval import split_indices

def main():
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    dv, te = split_indices(cache, 1), split_indices(cache, 2)
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.truncation_side = "left"
    texts = {
        "dev-last": build_text(cache, dv, masks_for(cache, dv, "last")),
        "dev-full": build_text(cache, dv, masks_for(cache, dv, "full")),
        "test-last": build_text(cache, te, masks_for(cache, te, "last")),
        "test-full": build_text(cache, te, masks_for(cache, te, "full")),
    }
    ys = {"dev": cache["y"][dv], "test": cache["y"][te]}
    for name in ("last", "full"):
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=8).to(device)
        model.load_state_dict(torch.load(f"results/roberta/{name}.pt", map_location=device, weights_only=True))
        for split in ("dev", "test"):
            for view in ("last", "full"):
                f1, _ = run_split(model, tok, texts[f"{split}-{view}"], ys[split], device)
                print(f"reader {name} input {view} {split} F1 {100*f1:.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
