"""Does a finetuned contextual encoder use turns outside the recent window?"""

import os

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.pointer import masks_from_scores
from src.prepare import build_cache
from src.train_eval import split_indices

MODEL = r"C:\Users\Admin\.cache\huggingface\hub\models--roberta-base\snapshots\e2da8e2f811d1448a5b465c236feacd80ffbac7b"
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "roberta")


def render(texts, roles, hist_row, keep):
    parts = []
    for j, tid in enumerate(hist_row):
        if tid < 0 or not keep[j]:
            continue
        who = "seeker" if int(roles[int(tid)]) == 0 else "supporter"
        parts.append(f"{who}: {texts[int(tid)].strip()}")
    return " </s> ".join(parts)


class Rows(Dataset):
    def __init__(self, texts, ys):
        self.texts = texts
        self.ys = ys

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.texts[i], int(self.ys[i])


def collate(tok, batch):
    text, y = zip(*batch)
    enc = tok(list(text), padding=True, truncation=True, max_length=384, return_tensors="pt")
    enc["labels"] = torch.tensor(y, dtype=torch.long)
    return enc


def masks_for(cache, idx, kind, table=None):
    hl = cache["hlen"][idx]
    length = cache["max_l"]
    if kind == "last":
        return (np.arange(length)[None, :] >= (hl[:, None] - 4)) & (np.arange(length)[None, :] < hl[:, None])
    if kind == "full":
        return np.arange(length)[None, :] < hl[:, None]
    return masks_from_scores(table[:, :-1], table[:, -1], hl, 4, tau=0.0)


def build_text(cache, idx, mask):
    texts, roles, hist = cache["texts"], cache["role"], cache["hist"]
    out = []
    for row, inst in enumerate(idx):
        out.append(render(texts, roles, hist[inst], mask[row]))
    return out


def run_split(model, tok, texts, y, device, batch_size=32):
    model.eval()
    preds = []
    loader = DataLoader(Rows(texts, y), batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate(tok, b))
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.inference_mode():
            logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu().numpy())
    pred = np.concatenate(preds)
    return float(f1_score(y, pred, average="macro", labels=list(range(8)), zero_division=0)), pred


def train_one(tok, texts, y, dev_texts, dev_y, device, epochs=4, batch_size=16, lr=2e-5):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=8).to(device)
    counts = np.bincount(np.asarray(y), minlength=8).astype(np.float32)
    weights = torch.tensor(counts.sum() / (8 * np.clip(counts, 1, None)), device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loader = DataLoader(
        Rows(texts, y), batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate(tok, b)
    )
    best, state = -1.0, None
    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, labels, weight=weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
        f1, _ = run_split(model, tok, dev_texts, dev_y, device)
        print(f"  epoch {ep+1} loss {total/max(len(loader),1):.3f} dev F1 {100*f1:.2f}", flush=True)
        if f1 > best:
            best = f1
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    model.to(device)
    return model


def main():
    os.makedirs(RES, exist_ok=True)
    device = torch.device("cuda")
    cache = build_cache(".", 0, 42)
    tr, dv, te = (split_indices(cache, s) for s in (0, 1, 2))
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.truncation_side = "left"
    dev_table = np.load(os.path.join(RES, "..", "pointer", "util_dev.npz"))["table"]
    # Train two readers on the same labels: one sees only the recent window,
    # the other sees the whole history. The oracle mask is a ceiling on dev/test.
    specs = {
        "last": (masks_for(cache, tr, "last"), masks_for(cache, dv, "last"), masks_for(cache, te, "last")),
        "full": (masks_for(cache, tr, "full"), masks_for(cache, dv, "full"), masks_for(cache, te, "full")),
    }
    for name, (mtr, mdv, mte) in specs.items():
        print(name, flush=True)
        model = train_one(
            tok,
            build_text(cache, tr, mtr), cache["y"][tr],
            build_text(cache, dv, mdv), cache["y"][dv],
            device,
        )
        f1_te, _ = run_split(model, tok, build_text(cache, te, mte), cache["y"][te], device)
        print(f"{name} test F1 {100*f1_te:.2f}", flush=True)
        torch.save(model.state_dict(), os.path.join(RES, f"{name}.pt"))
        del model
        torch.cuda.empty_cache()
    last_model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=8).to(device)
    last_model.load_state_dict(torch.load(os.path.join(RES, "last.pt"), map_location=device, weights_only=True))
    oracle_dv = masks_from_scores(dev_table[:, :-1], dev_table[:, -1], cache["hlen"][dv], 4, tau=0.0)
    f1_or, _ = run_split(last_model, tok, build_text(cache, dv, oracle_dv), cache["y"][dv], device)
    f1_last, _ = run_split(
        last_model, tok, build_text(cache, dv, masks_for(cache, dv, "last")), cache["y"][dv], device
    )
    print(f"dev last-reader on last {100*f1_last:.2f} on oracle-memory {100*f1_or:.2f}", flush=True)


if __name__ == "__main__":
    main()
