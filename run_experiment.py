"""Run CauMem: data cache, probe, interventional masks, selector, evaluation."""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.causal import (
    Store,
    build_masks,
    compute_effects,
    factor_scores,
    redundancy_diagnostics,
    substitute_groups,
    utility_gap_stats,
)
from src.modeling import Probe
from src.prepare import build_cache
from src.train_eval import (
    bootstrap_macro_diff,
    evaluate_methods,
    export_cases,
    oracle_rows,
    pack_metrics,
    render_main_table,
    render_oracle_table,
    run_fixed,
    split_indices,
    subset_report,
    train_probe,
    train_rl_selector,
    train_selector,
)


def _diagnostic(probe, store, indices, batch_size: int) -> None:
    cache = store.cache
    hl = cache["hlen"][indices]
    length = store.max_l
    full = np.zeros((len(indices), length), dtype=bool)
    last1 = np.zeros_like(full)
    last4 = np.zeros_like(full)
    for i, n in enumerate(hl.tolist()):
        n = int(n)
        full[i, :n] = True
        if n:
            last1[i, n - 1] = True
            last4[i, max(0, n - 4):n] = True
    y = cache["y"][indices]
    for name, mask in (("full", full), ("last1", last1), ("last4", last4)):
        pred, rank = run_fixed(probe, store, indices, mask, batch_size)
        met = pack_metrics(y, pred, rank, mask.sum(1), hl)
        print(
            f"diag {name} macF1 {met['macro_f1']*100:.2f} "
            f"R@1 {met['r1']*100:.2f} MRR {met['mrr']*100:.2f}"
        )


def _choose_eps(probe, store, effects, dev_idx, phi, spec, suppressed, batch_size: int):
    hl = store.cache["hlen"][dev_idx]
    y = store.cache["y"][dev_idx]
    records = []
    for eps, slack in ((0.03, 2), (0.03, 5), (0.08, 2), (0.08, 5), (0.15, 5)):
        masks = build_masks(probe, store, dev_idx, phi, spec, suppressed, eps, slack, batch_size)
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        met = pack_metrics(y, pred, rank, masks.sum(1), hl)
        print(
            f"eps {eps:.2f} slack {slack} dev_macF1 {met['macro_f1']*100:.2f} "
            f"MRR {met['mrr']*100:.2f} MMR {met['mmr']*100:.1f}"
        )
        records.append((eps, slack, met))
    best_f1 = max(item[2]["macro_f1"] for item in records)
    eligible = [item for item in records if item[2]["macro_f1"] >= best_f1 - 0.01]
    eps, slack, met = min(eligible, key=lambda item: (item[2]["mmr"], -item[2]["macro_f1"]))
    print(f"chosen eps {eps} slack {slack} dev_macF1 {met['macro_f1']*100:.2f} MMR {met['mmr']*100:.1f}")
    return eps, slack


def _choose_redundancy(probe, store, effects, dev_idx, phi, spec, eps, slack, batch_size: int):
    """Pick substitute-group thresholds on dev. Strict rule is the default."""
    hl = store.cache["hlen"][dev_idx]
    y = store.cache["y"][dev_idx]
    hlen = store.cache["hlen"]
    grid = ((0.08, 0.20, 0.10), (0.05, 0.12, 0.06), (0.035, 0.08, 0.04))
    records = []
    valid = np.arange(store.max_l)[None, :] < hlen[:, None]
    for di_max, dij_min, synergy in grid:
        sup = substitute_groups(effects, spec, hlen, di_max, dij_min, synergy)
        frac = float(sup[valid].mean())
        masks = build_masks(probe, store, dev_idx, phi, spec, sup, eps, slack, batch_size)
        pred, rank = run_fixed(probe, store, dev_idx, masks, batch_size)
        met = pack_metrics(y, pred, rank, masks.sum(1), hl)
        print(
            f"redun di<{di_max:.3f} dij>{dij_min:.2f} syn>{synergy:.2f} "
            f"frac {frac:.4f} dev_macF1 {met['macro_f1']*100:.2f} MMR {met['mmr']*100:.1f}"
        )
        records.append((di_max, dij_min, synergy, frac, met))
    best_f1 = max(item[4]["macro_f1"] for item in records)
    eligible = [item for item in records if item[4]["macro_f1"] >= best_f1 - 0.01]
    di_max, dij_min, synergy, frac, met = min(
        eligible, key=lambda item: (item[4]["mmr"], -item[3], -item[4]["macro_f1"])
    )
    print(
        f"chosen redun di<{di_max:.3f} dij>{dij_min:.2f} syn>{synergy:.2f} "
        f"frac {frac:.4f} MMR {met['mmr']*100:.1f}"
    )
    return di_max, dij_min, synergy


def _mean_jaccard(a: np.ndarray, b: np.ndarray, hlen: np.ndarray) -> float:
    scores = []
    for i, n in enumerate(hlen.tolist()):
        n = int(n)
        left = set(np.flatnonzero(a[i, :n]).tolist())
        right = set(np.flatnonzero(b[i, :n]).tolist())
        if not left and not right:
            scores.append(1.0)
        else:
            scores.append(len(left & right) / len(left | right))
    return float(np.mean(scores)) if scores else 0.0


def _slim_method(summary: dict) -> dict:
    out = {}
    for name, row in summary.items():
        if str(name).startswith("_"):
            continue
        out[name] = {
            "clean": row["clean"],
            "shuffle": row["shuffle"],
            "insert": row["insert"],
            "emo": row["emo"],
            "cc": row["cc"],
            "cc_emo": row["cc_emo"],
            "cc_ins": row["cc_ins"],
            "sel_stability": row["sel_stability"],
        }
    return out


def _slim_oracle(rows: dict) -> dict:
    return {name: {k: v for k, v in row.items() if k != "pred"} for name, row in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-dialogs", type=int, default=0)
    parser.add_argument("--probe-epochs", type=int, default=16)
    parser.add_argument("--sel-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--effects-batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--retrain-probe", action="store_true")
    parser.add_argument("--recompute-effects", action="store_true")
    parser.add_argument("--rebuild-masks", action="store_true")
    parser.add_argument("--retrain-selectors", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(root, "results"), exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device)

    cache = build_cache(root, limit_dialogs=args.limit_dialogs, seed=args.seed)
    assert int((cache["cand"] < 0).sum()) == 0, "candidate pool contains empty ids"
    store = Store(cache, device)
    train_idx = split_indices(cache, 0)
    dev_idx = split_indices(cache, 1)
    test_idx = split_indices(cache, 2)
    print("instances", len(train_idx), len(dev_idx), len(test_idx))

    probe_path = os.path.join(root, "results", "probe.pt")
    if os.path.exists(probe_path) and not args.retrain_probe:
        probe = Probe(n_strat=8, in_dim=int(store.emb.shape[1])).to(device)
        probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
        probe.eval()
        print("loaded probe", probe_path)
    else:
        probe = train_probe(
            store, train_idx, dev_idx, args.probe_epochs, args.batch_size, args.lr, args.seed, device
        )
        torch.save(probe.state_dict(), probe_path)
    _diagnostic(probe, store, dev_idx, args.batch_size)

    effects_path = os.path.join(root, "results", "effects_v1.npz")
    if os.path.exists(effects_path) and not args.recompute_effects:
        loaded = np.load(effects_path)
        effects = {key: loaded[key] for key in loaded.files}
        print("loaded effects", effects_path)
    else:
        effects = compute_effects(probe, store, batch_size=args.effects_batch)
        np.savez_compressed(effects_path, **effects)
        print("saved effects", effects_path)
    stats = utility_gap_stats(effects, cache["hlen"])
    print("effect stats", json.dumps(stats))
    pair_stats = redundancy_diagnostics(effects)
    print("pair stats", json.dumps(pair_stats))
    from src.modeling import Selector

    masks_path = os.path.join(root, "results", "masks_v1.npz")
    masks_meta_path = os.path.join(root, "results", "masks_v1_meta.json")
    valid = np.arange(cache["max_l"])[None, :] < cache["hlen"][:, None]
    if os.path.exists(masks_path) and os.path.exists(masks_meta_path) and not args.rebuild_masks:
        loaded = np.load(masks_path)
        masks = {key: loaded[key] for key in loaded.files}
        with open(masks_meta_path, encoding="utf-8") as f:
            masks_meta = json.load(f)
        eps = float(masks_meta["eps"])
        slack = int(masks_meta["rank_slack"])
        redun = (
            float(masks_meta["di_max"]),
            float(masks_meta["dij_min"]),
            float(masks_meta["synergy"]),
        )
        suppressed = np.asarray(masks_meta.get("suppressed_placeholder", 0.0))
        print("loaded masks", masks_path, "eps", eps, "slack", slack, "redun", redun)
        for name, arr in masks.items():
            density = arr[train_idx].sum() / max(1, cache["hlen"][train_idx].sum())
            print(f"{name} train density {float(density):.3f}")
        suppressed_frac = float(masks_meta.get("suppressed_fraction", 0.0))
    else:
        phi, spec = factor_scores(effects, "full", cache["hlen"])
        suppressed = substitute_groups(effects, spec, cache["hlen"])
        print("suppressed fraction", float(suppressed[valid].mean()))
        eps, slack = _choose_eps(probe, store, effects, dev_idx, phi, spec, suppressed, args.batch_size)
        redun = _choose_redundancy(
            probe, store, effects, dev_idx, phi, spec, eps, slack, args.batch_size
        )
        suppressed = substitute_groups(effects, spec, cache["hlen"], *redun)
        suppressed_frac = float(suppressed[valid].mean())
        print("suppressed fraction", suppressed_frac, "thresholds", redun)

        all_idx = np.arange(len(cache["y"]))
        variant_spec = {
            "MISSET": ("full", True),
            "MISSET-no-redun": ("full", False),
            "MISSET-del": ("del", False),
            "MISSET-no-emo": ("no_emo", True),
            "MISSET-no-bg": ("no_bg", True),
        }
        masks = {}
        for name, (mode, use_sup) in variant_spec.items():
            phi_m, spec_m = factor_scores(effects, mode, cache["hlen"])
            sup_m = suppressed if (use_sup and mode == "full") else (
                substitute_groups(effects, spec_m, cache["hlen"], *redun) if use_sup else np.zeros_like(suppressed)
            )
            print("building", name)
            masks[name] = build_masks(
                probe, store, all_idx, phi_m, spec_m, sup_m, eps, slack, batch_size=args.batch_size
            )
            density = masks[name][train_idx].sum() / max(1, cache["hlen"][train_idx].sum())
            print(f"{name} train density {float(density):.3f}")
        np.savez_compressed(masks_path, **masks)
        with open(masks_meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "eps": eps,
                    "rank_slack": slack,
                    "di_max": redun[0],
                    "dij_min": redun[1],
                    "synergy": redun[2],
                    "suppressed_fraction": suppressed_frac,
                },
                f,
                indent=2,
            )
        print("saved masks", masks_path)

    def _load_or_train_selector(name, trainer):
        path = os.path.join(root, "results", f"{name}.pt")
        if os.path.exists(path) and not args.retrain_selectors:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = Selector(in_dim=int(store.emb.shape[1])).to(device)
            model.load_state_dict(ckpt["state"])
            model.eval()
            print("loaded selector", name, "tau", ckpt["tau"])
            return model, float(ckpt["tau"])
        model, tau = trainer()
        torch.save({"state": model.state_dict(), "tau": tau}, path)
        print("saved selector", path)
        return model, tau

    selectors = {}
    taus = {}
    selectors["caumem"], taus["caumem"] = _load_or_train_selector(
        "caumem",
        lambda: train_selector(
            probe, store, train_idx, dev_idx, masks["MISSET"], args.sel_epochs, args.batch_size,
            args.lr, args.seed, device, use_cf=True, name="caumem",
        ),
    )
    selectors["no_cf"], taus["no_cf"] = _load_or_train_selector(
        "no_cf",
        lambda: train_selector(
            probe, store, train_idx, dev_idx, masks["MISSET"], args.sel_epochs, args.batch_size,
            args.lr, args.seed + 1, device, use_cf=False, name="no_cf",
        ),
    )
    selectors["del_only"], taus["del_only"] = _load_or_train_selector(
        "del_only",
        lambda: train_selector(
            probe, store, train_idx, dev_idx, masks["MISSET-del"], args.sel_epochs, args.batch_size,
            args.lr, args.seed + 2, device, use_cf=True, name="del_only",
        ),
    )
    if not args.skip_rl:
        selectors["rl"], taus["rl"] = _load_or_train_selector(
            "rl",
            lambda: train_rl_selector(
                probe, store, train_idx, dev_idx, max(4, args.sel_epochs // 2), args.batch_size,
                args.lr * 0.5, args.seed + 3, device,
            ),
        )

    # Budget matched to the deployable CauMem operating point on dev.
    from src.train_eval import _selector_dev_score
    _tau, dev_metrics, mean_count, _ = _selector_dev_score(
        selectors["caumem"], probe, store, dev_idx, args.batch_size
    )
    k = max(1, int(round(mean_count)))
    print(f"matched k {k} dev CauMem macF1 {dev_metrics['macro_f1']*100:.2f} MMR {dev_metrics['mmr']*100:.1f}")

    print("evaluating dev")
    dev_summary = evaluate_methods(
        probe, selectors, taus, store, effects, dev_idx, k, batch_size=max(32, args.batch_size // 2)
    )
    print(render_main_table(dev_summary, cache["strategies"]))
    print("evaluating test")
    test_summary = evaluate_methods(
        probe, selectors, taus, store, effects, test_idx, k, batch_size=max(32, args.batch_size // 2)
    )
    oracle = oracle_rows(probe, store, effects, masks, test_idx, k)
    oracle_jaccard = {
        name: _mean_jaccard(masks["MISSET"][test_idx], masks[name][test_idx], cache["hlen"][test_idx])
        for name in masks if name != "MISSET"
    }
    print("oracle jaccard vs MISSET", json.dumps(oracle_jaccard))
    main_txt = render_main_table(test_summary, cache["strategies"])
    oracle_txt = render_oracle_table(oracle)
    subset_txt = subset_report(test_summary, cache["strategies"])
    print(main_txt)
    print(oracle_txt)
    print(subset_txt)

    y = test_summary["_y"]
    preds = test_summary["_preds"]
    boots = {}
    for other in ("SimK", "LastK", "AttnK", "FullHist"):
        boots[f"CauMem-{other}"] = bootstrap_macro_diff(y, preds["CauMem"]["clean"], preds[other]["clean"])
        print(f"bootstrap CauMem - {other}", boots[f"CauMem-{other}"])

    cases = export_cases(cache, test_idx, test_summary)
    ablation_names = [n for n in ("CauMem", "no_cf", "del_only", "rl") if n in test_summary]
    ablation_txt_lines = [
        "| Method | MacF1 | R@1 | MRR | MMR | CC | MacF1-shuf | MacF1-ins | SelStab |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ablation_names:
        row = test_summary[name]
        ablation_txt_lines.append(
            f"| {name} | {100*row['clean']['macro_f1']:.2f} | {100*row['clean']['r1']:.2f} | "
            f"{100*row['clean']['mrr']:.2f} | {100*row['clean']['mmr']:.2f} | {100*row['cc']:.2f} | "
            f"{100*row['shuffle']['macro_f1']:.2f} | {100*row['insert']['macro_f1']:.2f} | "
            f"{100*row['sel_stability']:.2f} |"
        )
    ablation_txt = "\n".join(ablation_txt_lines)

    report = "\n\n".join([
        "# Test deployable methods",
        main_txt,
        "# Oracle interventional selectors (use gold utility; not deployable)",
        oracle_txt,
        "# Ablations",
        ablation_txt,
        "# Slices and per-strategy F1",
        subset_txt,
        "# Setup",
        (
            f"split counts {cache['dialog_split_counts']} instances {cache['instance_counts']}\n"
            f"encoder {cache['encoder']}\n"
            f"eps {eps} rank_slack {slack} matched_k {k} taus {taus}\n"
            f"redundancy_thresholds {redun}\n"
            f"effect_stats {json.dumps(stats)}\n"
            f"pair_stats {json.dumps(pair_stats)}\n"
            f"suppressed_fraction {suppressed_frac:.4f}\n"
            f"oracle_jaccard {json.dumps(oracle_jaccard)}\n"
            f"bootstrap {json.dumps(boots)}"
        ),
    ])
    with open(os.path.join(root, "results", "tables.md"), "w", encoding="utf-8") as f:
        f.write(report)
    payload = {
        "dev": _slim_method(dev_summary),
        "test": _slim_method(test_summary),
        "oracle_test": _slim_oracle(oracle),
        "bootstrap_macro_f1_diff": boots,
        "eps": eps,
        "rank_slack": slack,
        "matched_k": k,
        "taus": taus,
        "effect_stats": stats,
        "pair_stats": pair_stats,
        "redundancy_thresholds": {"di_max": redun[0], "dij_min": redun[1], "synergy": redun[2]},
        "oracle_jaccard_vs_misset": oracle_jaccard,
        "suppressed_fraction": suppressed_frac,
        "dialog_split_counts": cache["dialog_split_counts"],
        "instance_counts": cache["instance_counts"],
        "encoder": cache["encoder"],
        "generic_rate": cache["generic_rate"],
        "strategies": cache["strategies"],
    }
    with open(os.path.join(root, "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(root, "results", "cases.json"), "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=2)
    print("wrote results/tables.md results/metrics.json results/cases.json")


if __name__ == "__main__":
    main()
