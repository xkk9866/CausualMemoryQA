"""Independent re-annotation of ESConv strategy labels by an LLM annotator panel.

    python -m esc.panel annotate <model> --n 400
    python -m esc.panel analyze

Every annotator reads the eight strategy definitions of Liu et al. (2021), the two preceding
utterances and the supporter's message, and names one strategy (greedy, 4-bit). The same random
sample of test messages is used for every annotator. `analyze` reports inter-annotator agreement
(Fleiss' kappa), how often the original self-label matches the panel majority, and whether
self-label/panel disagreement clusters by dialogue beyond topic (same permutation test as
esc.control). The panel is a proxy for independent expert annotation, not a replacement.
"""

import json
import os
import re
import sys

import numpy as np

from esc.data import load, samples

OUT = "runs/esconv/panel"
DEFS = {
    "Question": "asking for information related to the problem to help the seeker articulate their issues",
    "Restatement or Paraphrasing": "a simple, more concise rephrasing of the seeker's statements",
    "Reflection of feelings": "articulating and describing the seeker's feelings",
    "Self-disclosure": "divulging similar experiences the supporter has had, to express empathy",
    "Affirmation and Reassurance": "affirming the seeker's strengths, motivation and capabilities, and providing reassurance",
    "Providing Suggestions": "providing suggestions about how to change, without telling the seeker what to do",
    "Information": "providing useful information, e.g. data, facts, opinions or resources",
    "Others": "exchanging pleasantries or using strategies beyond the categories above",
}


def sample(n):
    splits, names = load("esconv")
    test = samples(splits["test"])
    idx = sorted(np.random.default_rng(1).choice(len(test), n, replace=False).tolist())
    return test, names, idx


def annotate(model, n):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    test, names, idx = sample(n)
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    llm = AutoModelForCausalLM.from_pretrained(model, device_map={"": 0}, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                               quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                                      bnb_4bit_compute_dtype=torch.bfloat16))
    menu = "\n".join(f"{i + 1}. {k}: {v}" for i, k in enumerate(names) for v in [DEFS[k]])
    prompts = []
    for i in idx:
        ctx = [f"{sp}: {t}" for sp, t, _ in test[i]["turns"][-2:] if sp != "none"]
        prompts.append("You are an expert annotator of emotional support conversations. Support strategies:\n" + menu +
                       "\n\nContext:\n" + "\n".join(ctx) + f"\n\nSupporter's message: {test[i]['response']}\n\n"
                       "Which ONE strategy does the supporter's message use? Answer with the number only.")
    labels = []
    for b in range(0, len(prompts), 16):
        kw = {"enable_thinking": False} if "Qwen3" in model else {}
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, **kw)
                for p in prompts[b:b + 16]]
        enc = tok(msgs, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            g = llm.generate(**enc, max_new_tokens=4, do_sample=False, pad_token_id=tok.eos_token_id)
        for x in g:
            m = re.search(r"[1-8]", tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True))
            labels.append(int(m.group()) - 1 if m else len(names) - 1)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"model": model, "idx": idx, "labels": labels}, open(os.path.join(OUT, model.split("/")[-1] + ".json"), "w"))
    print(model, "done", flush=True)


def fleiss(R, k):
    n, m = R.shape
    counts = np.stack([(R == c).sum(1) for c in range(k)], 1)
    p_j = counts.sum(0) / (n * m)
    P_i = ((counts ** 2).sum(1) - m) / (m * (m - 1))
    Pe = (p_j ** 2).sum()
    return (P_i.mean() - Pe) / (1 - Pe)


def analyze():
    from esc.control import cluster_ratio
    files = sorted(f for f in os.listdir(OUT) if f.endswith(".json") and f != "analysis.json")
    A = [json.load(open(os.path.join(OUT, f))) for f in files]
    test, names, idx = sample(len(A[0]["idx"]))
    k = len(names)
    R = np.array([a["labels"] for a in A]).T                         # n x annotators
    y = np.array([test[i]["label"] for i in idx])
    maj = np.array([np.bincount(r, minlength=k).argmax() for r in R])
    unanimous = (R == R[:, :1]).all(1)
    out = {"annotators": [a["model"] for a in A], "n": len(idx),
           "fleiss_kappa_panel": float(fleiss(R, k)),
           "fleiss_kappa_panel_plus_self": float(fleiss(np.concatenate([R, y[:, None]], 1), k)),
           "self_vs_majority_agree": 100 * float((maj == y).mean()),
           "annotator_vs_majority_agree": [100 * float((R[:, j] == maj).mean()) for j in range(R.shape[1])],
           "unanimous_share": 100 * float(unanimous.mean()),
           "self_vs_unanimous_agree": 100 * float((maj[unanimous] == y[unanimous]).mean())}
    # does self-label / panel disagreement cluster by dialogue beyond topic?
    import random
    from esc.data import EXT
    raw = json.load(open(os.path.join(EXT, "data/esconv/ESConv.json"), encoding="utf-8"))
    order = list(range(len(raw)))
    random.seed(13)
    random.shuffle(order)
    n15 = int(0.15 * len(raw))
    topic = {i: raw[j]["problem_type"] for i, j in enumerate(order[n15:2 * n15])}
    groups = np.array([test[i]["dialog"] for i in idx])
    delta = (maj != y).astype(float)
    rng = np.random.default_rng(0)
    out["cluster_ratio"], out["cluster_p"] = cluster_ratio(delta, groups, np.zeros(len(y)), rng)
    out["cluster_ratio_topic"], out["cluster_p_topic"] = cluster_ratio(delta, groups, np.array([topic[g] for g in groups]), rng)
    json.dump(out, open(os.path.join(OUT, "analysis.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "annotate":
        annotate(sys.argv[2], int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 400)
    else:
        analyze()
