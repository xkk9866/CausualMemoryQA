"""Build the ESConv instance cache: splits, embeddings, donors, candidates."""

from __future__ import annotations

import gc
import json
import os
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.lexicon import content_tokens, emotion_tokens, jaccard, normalize_text, substitute_emotion

MAX_L = 64
HEAD = 8
CACHE_TAG = "v1"
STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
GENERIC = {
    "yeah", "yes", "ok", "okay", "i see", "thanks", "thank you", "i guess",
    "hmm", "right", "sure", "no", "yep", "nah", "oh", "hi", "hello", "fine",
    "alright", "i know", "me too", "wow", "cool", "nice", "true", "exactly",
    "k", "kk", "ya", "yup", "nope", "ugh", "hmm ok", "ok thanks",
}


def is_generic(text: str) -> bool:
    raw = normalize_text(text)
    if raw in GENERIC:
        return True
    return len(raw.split()) <= 3


def _split_counts(n: int) -> tuple[int, int, int]:
    raw = [0.6 * n, 0.2 * n, 0.2 * n]
    floors = [int(np.floor(x)) for x in raw]
    rem = n - sum(floors)
    frac_order = sorted(range(3), key=lambda i: (raw[i] - floors[i], -i), reverse=True)
    for i in range(rem):
        floors[frac_order[i]] += 1
    return floors[0], floors[1], floors[2]


def split_dialogs(problem_types: list[str], seed: int = 42) -> np.ndarray:
    """Conversation-level 6:2:2 split, stratified by problem bucket."""
    counts = Counter(problem_types)
    buckets = [p if counts[p] >= 20 else "rare" for p in problem_types]
    rng = np.random.RandomState(seed)
    split = np.full(len(problem_types), -1, dtype=np.int8)
    grouped: dict[str, list[int]] = defaultdict(list)
    for i, b in enumerate(buckets):
        grouped[b].append(i)
    for ids in grouped.values():
        ids = list(ids)
        rng.shuffle(ids)
        n_tr, n_dv, n_te = _split_counts(len(ids))
        for i in ids[:n_tr]:
            split[i] = 0
        for i in ids[n_tr:n_tr + n_dv]:
            split[i] = 1
        for i in ids[n_tr + n_dv:]:
            split[i] = 2
    assert (split >= 0).all()
    return split


def _best_donor(i, cand_ids, emo, content, dialog, problem) -> int:
    best_j, best_s = -1, -1e9
    ei, ci, di, pi = emo[i], content[i], dialog[i], problem[i]
    for j in cand_ids:
        if j == i or dialog[j] == di:
            continue
        ej = jaccard(ei, emo[j])
        cj = jaccard(ci, content[j])
        if cj > 0.45:
            continue
        diff = 0.3 if problem[j] != pi else 0.0
        score = ej - 0.75 * cj + diff
        if score > best_s:
            best_s, best_j = score, j
    return best_j


def assign_donors(emo, content, dialog, problem, role, split, seed: int = 42) -> np.ndarray:
    rng = random.Random(seed)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    emo_index: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for i, (sp, ro) in enumerate(zip(split, role)):
        buckets[(int(sp), int(ro))].append(i)
        for w in emo[i]:
            emo_index[(int(sp), int(ro), w)].append(i)
    donor = np.full(len(role), -1, dtype=np.int32)
    for i in tqdm(range(len(role)), desc="donors"):
        key_bucket = buckets[(int(split[i]), int(role[i]))]
        cands: list[int] = []
        seen = set()
        for w in emo[i]:
            for j in emo_index[(int(split[i]), int(role[i]), w)]:
                if j not in seen:
                    seen.add(j)
                    cands.append(j)
                if len(cands) >= 300:
                    break
            if len(cands) >= 300:
                break
        if len(cands) < 40 and key_bucket:
            extra = key_bucket if len(key_bucket) <= 80 else rng.sample(key_bucket, 80)
            for j in extra:
                if j not in seen:
                    cands.append(j)
        j = _best_donor(i, cands, emo, content, dialog, problem)
        if j < 0 and len(key_bucket) > 1:
            for _ in range(20):
                k = key_bucket[rng.randrange(len(key_bucket))]
                if dialog[k] != dialog[i]:
                    j = k
                    break
        donor[i] = i if j < 0 else j
    return donor


def _truncate(ids: list[int]) -> list[int]:
    if len(ids) <= MAX_L:
        return ids
    return ids[:HEAD] + ids[-(MAX_L - HEAD):]


def build_cache(root: str, limit_dialogs: int = 0, seed: int = 42, encoder_name: str = "sentence-transformers/all-mpnet-base-v2") -> dict:
    os.makedirs(os.path.join(root, "data", "cache"), exist_ok=True)
    tag = CACHE_TAG if limit_dialogs <= 0 else f"{CACHE_TAG}-lim{limit_dialogs}"
    path = os.path.join(root, "data", "cache", f"esconv_{tag}.pkl")
    if os.path.exists(path):
        print(f"loading cache {path}")
        with open(path, "rb") as f:
            return pickle.load(f)

    with open(os.path.join(root, "data", "ESConv.json"), "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(data))
    if limit_dialogs > 0:
        order = order[:limit_dialogs]
    data = [data[i] for i in order]
    problems = [d.get("problem_type") or "unknown" for d in data]
    dialog_split = split_dialogs(problems, seed=seed)

    turns_text: list[str] = []
    turns_emo: list[str] = []
    role = []
    dialog_id = []
    turn_split = []
    problem = []
    strategy = []
    emo_sets = []
    content_sets = []
    changed_flags = []

    # map (dialog, local turn) -> global turn id
    index_of: list[list[int]] = []
    strat2id = {s: i for i, s in enumerate(STRATEGIES)}

    for d_i, dial in enumerate(data):
        local = []
        for utt in dial["dialog"]:
            text = normalize_text(utt.get("content") or "")
            emo_text, changed = substitute_emotion(text)
            spk = utt.get("speaker")
            ann = utt.get("annotation") or {}
            strat_name = ann.get("strategy") if isinstance(ann, dict) else None
            sid = strat2id.get(strat_name, -1) if spk == "supporter" else -1
            local.append(len(turns_text))
            turns_text.append(text)
            turns_emo.append(emo_text if changed else text)
            role.append(0 if spk == "seeker" else 1)
            dialog_id.append(d_i)
            turn_split.append(int(dialog_split[d_i]))
            problem.append(problems[d_i])
            strategy.append(sid)
            emo_sets.append(emotion_tokens(text))
            content_sets.append(content_tokens(text))
            changed_flags.append(bool(changed))
        index_of.append(local)

    print(f"turns {len(turns_text)} dialogs {len(data)} emotion-edited {sum(changed_flags)}")
    print("encoding with", encoder_name)
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(encoder_name, device=device)
    emb = encoder.encode(
        turns_text, batch_size=128, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    # Encode only texts that actually changed; copy the rest.
    emo_idx = [i for i, ch in enumerate(changed_flags) if ch]
    emb_emo = emb.copy()
    if emo_idx:
        emo_emb = encoder.encode(
            [turns_emo[i] for i in emo_idx], batch_size=128, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        emb_emo[np.array(emo_idx)] = emo_emb
    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    role_a = np.asarray(role, dtype=np.int64)
    dialog_a = np.asarray(dialog_id, dtype=np.int32)
    split_a = np.asarray(turn_split, dtype=np.int8)
    strat_a = np.asarray(strategy, dtype=np.int16)
    donor = assign_donors(emo_sets, content_sets, dialog_a, problem, role_a, split_a, seed=seed)
    same = float((donor == np.arange(len(donor))).mean())
    emo_j, con_j, diff_p = [], [], []
    for i, j in enumerate(donor[: min(5000, len(donor))]):
        if j == i:
            continue
        emo_j.append(jaccard(emo_sets[i], emo_sets[j]))
        con_j.append(jaccard(content_sets[i], content_sets[j]))
        diff_p.append(float(problem[i] != problem[j]))
    print(
        f"donor self-rate {same:.3f} emo_j {np.mean(emo_j):.3f} "
        f"content_j {np.mean(con_j):.3f} diff_problem {np.mean(diff_p):.3f}"
    )

    # Instances: one per supporter decision with nonempty history.
    hist_rows = []
    hlen = []
    y = []
    resp = []
    inst_split = []
    inst_dialog = []
    generic = []
    last_ids = []
    for d_i, dial in enumerate(data):
        ids = index_of[d_i]
        for local_i, gidx in enumerate(ids):
            if strat_a[gidx] < 0:
                continue
            prev = ids[:local_i]
            if not prev:
                continue
            kept = _truncate(prev)
            hist_rows.append(kept)
            hlen.append(len(kept))
            y.append(int(strat_a[gidx]))
            resp.append(int(gidx))
            inst_split.append(int(dialog_split[d_i]))
            inst_dialog.append(d_i)
            generic.append(is_generic(turns_text[kept[-1]]))
            last_ids.append(kept[-1])

    n = len(y)
    hist = np.full((n, MAX_L), -1, dtype=np.int32)
    for i, row in enumerate(hist_rows):
        hist[i, : len(row)] = row
    hlen_a = np.asarray(hlen, dtype=np.int16)
    y_a = np.asarray(y, dtype=np.int16)
    resp_a = np.asarray(resp, dtype=np.int32)
    split_i = np.asarray(inst_split, dtype=np.int8)
    dialog_i = np.asarray(inst_dialog, dtype=np.int32)
    generic_a = np.asarray(generic, dtype=np.bool_)

    cand = np.full((n, 32), -1, dtype=np.int32)
    # Candidate pool per split: gold + last-turn neighbors + same-strategy + random.
    emb_t = torch.tensor(emb)
    for sp in (0, 1, 2):
        inst_ids = np.where(split_i == sp)[0]
        resp_pool = np.where((strat_a >= 0) & (split_a == sp))[0]
        if len(inst_ids) == 0 or len(resp_pool) == 0:
            continue
        pool_emb = emb_t[resp_pool]  # [P,D]
        by_strat: dict[int, list[int]] = defaultdict(list)
        for ridx in resp_pool.tolist():
            by_strat[int(strat_a[ridx])].append(int(ridx))
        pool_dialog = dialog_a[resp_pool]
        print(f"candidates split {sp}: instances {len(inst_ids)} pool {len(resp_pool)}")
        rng_py = random.Random(seed + sp)
        bs = 256
        for s in tqdm(range(0, len(inst_ids), bs), desc=f"cand-{sp}"):
            batch_ids = inst_ids[s:s + bs]
            last = torch.tensor([last_ids[i] for i in batch_ids], dtype=torch.long)
            q = emb_t[last]
            sims = q @ pool_emb.T  # [B,P]
            sims_np = sims.numpy()
            for bi, inst in enumerate(batch_ids):
                gold = int(resp_a[inst])
                banned_dialog = int(dialog_i[inst])
                order = np.argsort(-sims_np[bi])
                negs = []
                for oi in order:
                    ridx = int(resp_pool[oi])
                    if ridx == gold or int(pool_dialog[oi]) == banned_dialog:
                        continue
                    negs.append(ridx)
                    if len(negs) >= 12:
                        break
                same = [r for r in by_strat[int(y_a[inst])] if r != gold and int(dialog_a[r]) != banned_dialog]
                rng_py.shuffle(same)
                for ridx in same:
                    if ridx not in negs:
                        negs.append(ridx)
                    if len(negs) >= 22:
                        break
                guard = 0
                while len(negs) < 31 and guard < 400:
                    guard += 1
                    ridx = int(resp_pool[rng_py.randrange(len(resp_pool))])
                    if ridx == gold or ridx in negs:
                        continue
                    negs.append(ridx)
                if not negs:
                    negs = [gold]
                while len(negs) < 31:
                    negs.append(negs[guard % len(negs)])
                chosen = [gold] + negs[:31]
                cand[inst] = chosen
        del pool_emb, sims
    del emb_t
    gc.collect()

    # Insertion donor: emotionally close to the last turn, different dialogue.
    ins_donor = np.full(n, -1, dtype=np.int32)
    for i in tqdm(range(n), desc="insert-donors"):
        last = int(hist[i, hlen_a[i] - 1])
        j = int(donor[last])
        if dialog_a[j] == dialog_i[i]:
            j = last
        ins_donor[i] = j

    # Baseline scores.
    sim_scores = np.full((n, MAX_L), -1e4, dtype=np.float32)
    emo_scores = np.full((n, MAX_L), -1e4, dtype=np.float32)
    for i in range(n):
        length = int(hlen_a[i])
        ids = hist[i, :length]
        last = ids[-1]
        sim_scores[i, :length] = emb[ids] @ emb[last]
        last_emo = emo_sets[last]
        for t, tid in enumerate(ids):
            emo_scores[i, t] = 1.0 if t == length - 1 else jaccard(emo_sets[tid], last_emo)

    print("tf-idf salience")
    gc.collect()
    min_df = 2 if len(turns_text) > 5000 else 1
    # Unigrams keep the vocabulary inside RAM. Bigram expansion on the full
    # training split previously exhausted the allocator during feature sorting.
    vectorizer = TfidfVectorizer(
        max_features=12000, ngram_range=(1, 1), min_df=min_df, dtype=np.float32,
    )
    train_mask = split_a == 0
    vectorizer.fit([turns_text[i] for i in np.where(train_mask)[0]])
    x_all = vectorizer.transform(turns_text)
    tfidf_scores = np.full((n, MAX_L), -1e4, dtype=np.float32)
    for i in range(n):
        length = int(hlen_a[i])
        ids = hist[i, :length]
        last_vec = x_all[ids[-1]]
        sims = (x_all[ids] @ last_vec.T).toarray().ravel()
        tfidf_scores[i, :length] = sims

    cache = {
        "emb": emb,
        "emb_emo": emb_emo,
        "donor": donor,
        "role": role_a,
        "dialog_of_turn": dialog_a,
        "turn_split": split_a,
        "strategy_of_turn": strat_a,
        "texts": turns_text,
        "has_emo": np.asarray(changed_flags, dtype=np.bool_),
        "hist": hist,
        "hlen": hlen_a,
        "y": y_a,
        "resp": resp_a,
        "cand": cand,
        "split": split_i,
        "dialog": dialog_i,
        "generic": generic_a,
        "ins_donor": ins_donor,
        "sim_scores": sim_scores,
        "emo_scores": emo_scores,
        "tfidf_scores": tfidf_scores,
        "vectorizer": vectorizer,
        "strategies": STRATEGIES,
        "max_l": MAX_L,
        "dialog_split_counts": {
            "train": int((dialog_split == 0).sum()),
            "dev": int((dialog_split == 1).sum()),
            "test": int((dialog_split == 2).sum()),
        },
        "instance_counts": {
            "train": int((split_i == 0).sum()),
            "dev": int((split_i == 1).sum()),
            "test": int((split_i == 2).sum()),
        },
        "generic_rate": float(generic_a.mean()),
        "encoder": encoder_name,
    }
    with open(path, "wb") as f:
        pickle.dump(cache, f, protocol=4)
    meta_path = os.path.join(root, "data", "cache", f"esconv_{tag}_meta.json")
    meta = {k: cache[k] for k in ("dialog_split_counts", "instance_counts", "generic_rate", "encoder", "strategies")}
    meta["n_turns"] = len(turns_text)
    meta["n_instances"] = n
    meta["emotion_edit_rate"] = float(np.mean(changed_flags))
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    return cache
