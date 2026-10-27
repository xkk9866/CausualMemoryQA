"""ESConv / AnnoMI next-strategy samples under the EmoDynamiX protocol.

The splits are rebuilt from the raw json with the same shuffling seed as
EmoDynamiX (`data/<dataset>/make.py`), so every sample matches their csv row
for row. Unlike the csv, each sample also keeps its dialogue id and the full
history, which the window-5 protocol throws away.
"""

import json
import os
import random

EXT = os.environ.get("EMODX", os.path.join(os.path.dirname(__file__), "..", "..", "ext", "EmoDynamiX-v2"))

ESCONV_NAMES = ["Reflection of feelings", "Self-disclosure", "Question", "Affirmation and Reassurance",
                "Providing Suggestions", "Restatement or Paraphrasing", "Information", "Others"]
ANNOMI_NAMES = ["Reflection simple", "Reflection complex", "Question closed", "Question open",
                "Provide information", "Other", "Provide suggestion"]


def _norm(x):
    return " ".join(x.strip().split())


def _turns_esconv(d, sid):
    out = []
    for t in d["dialog"]:
        s = t["annotation"].get("strategy")
        out.append(("seeker" if t["speaker"] == "seeker" else "supporter", _norm(t["content"]), sid[s] if s else -1))
    return out


def _turns_annomi(d, sid):
    out = []
    for t in d["dialog"]:
        a = t["annotation"]
        out.append(("seeker" if t["interlocutor"] == "client" else "supporter", _norm(t["content"]), sid[a] if a else -1))
    return out


def load(dataset):
    """Returns (splits, names). splits[name] is a list of dialogues; a dialogue is a list of
    (speaker, text, strategy) with strategy = -1 for unlabelled turns."""
    if dataset == "esconv":
        raw = json.load(open(os.path.join(EXT, "data/esconv/ESConv.json"), encoding="utf-8"))
        sid = json.load(open(os.path.join(EXT, "data/esconv/strategies.json"), encoding="utf-8"))
        dial = [_turns_esconv(d, sid) for d in raw]
        random.seed(13)
        random.shuffle(dial)
        n = int(0.15 * len(dial))
        splits = {"valid": dial[:n], "test": dial[n:2 * n], "train": dial[2 * n:]}
        names = ESCONV_NAMES
    elif dataset == "annomi":
        raw = json.load(open(os.path.join(EXT, "data/annomi/AnnoMI-clean.json"), encoding="utf-8"))
        raw = [d for d in raw if d["quality"] == "high"]
        sid = json.load(open(os.path.join(EXT, "data/annomi/strategies.json"), encoding="utf-8"))
        dial = [_turns_annomi(d, sid) for d in raw]
        random.seed(77)
        random.shuffle(dial)
        n = int(0.1 * len(dial))
        splits = {"test": dial[:n], "valid": dial[n:2 * n], "train": dial[2 * n:]}
        names = ANNOMI_NAMES
    else:
        raise ValueError(dataset)
    assert [names[v] for v in range(len(names))] == sorted(sid, key=sid.get)
    return splits, names


def samples(dialogues):
    """One sample per labelled supporter turn. `turns` is the full history before the target,
    with the EmoDynamiX <START> pseudo-turn first; `window` keeps the last 5 entries, which is
    exactly the csv context."""
    out = []
    for did, d in enumerate(dialogues):
        hist = [("none", "<START>", -1)]
        for t, (sp, text, s) in enumerate(d):
            if s >= 0:
                # training-only hindsight: the seeker's first reaction after this message
                nxt = next((x[1] for x in d[t + 1:] if x[0] == "seeker"), "")
                out.append({"dialog": did, "turns": list(hist), "label": s, "response": text, "reaction": nxt})
            hist.append((sp, text, s))
    return out


def plan(turns, names):
    """The supporter's whole strategy trajectory so far, oldest first."""
    return " > ".join(names[s] for sp, _, s in turns if sp == "supporter" and s >= 0) or "none"


def render(turns, names, window=5, tags=True, with_plan=False):
    """Flattened context in the EmoDynamiX format: [seeker] text / [supporter] [Strategy] text.
    with_plan prepends the full strategy trajectory, which the window otherwise cuts off."""
    head = f"[plan] {plan(turns, names)} [context] " if with_plan else ""
    if window:
        turns = turns[-window:]
    parts = []
    for sp, text, s in turns:
        if sp == "seeker":
            parts.append("[seeker]" + text)
        elif sp == "supporter":
            parts.append((f"[supporter] [{names[s]}] " if tags and s >= 0 else "[supporter] ") + text)
        else:
            parts.append(text)
    return head + "".join(parts)
