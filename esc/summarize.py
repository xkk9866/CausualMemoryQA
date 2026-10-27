"""Print the best-dev row of every run: python -m esc.summarize runs/esconv"""
import glob, json, os, sys

for d in sorted(glob.glob(os.path.join(sys.argv[1], "*", "best.json"))):
    b = json.load(open(d))
    v, t = b["valid"], b["test"]
    print(f"{os.path.basename(os.path.dirname(d)):40s} it{b['it']:5d} | dev MacF1 {v['macro_f1']:.2f} | "
          f"test Acc {t['acc']:.2f} MacF1 {t['macro_f1']:.2f} W-F1 {t['weighted_f1']:.2f} B {t['bias']:.2f}")
