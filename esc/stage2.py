"""When all folds of a teacher family are done, build its out-of-fold targets and queue students.

    python -m esc.stage2 --dataset esconv --jobs jobs.txt
"""
import argparse
import os
import subprocess
import sys
import time

# teacher family -> student variants that distill from it
STUDENTS = {
    "hind": [("w5", ""), ("plan", "--with_plan")],
    "self": [("w5", "")],
    "resp": [("w5", "")],
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="esconv")
    p.add_argument("--jobs", default="jobs.txt")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--extra", default="", help="extra args for every student, e.g. --epochs 8")
    a = p.parse_args()
    tag = "" if a.dataset == "esconv" else a.dataset[:2]
    built = set()
    while len(built) < len(STUDENTS):
        for fam in STUDENTS:
            prefix = f"runs/{a.dataset}/{fam}_f"
            if fam in built:
                continue
            if all(os.path.exists(f"logs/{tag}{fam}f{f}.done") for f in range(5)):
                subprocess.run([sys.executable, "-m", "esc.oof", prefix, "5"], check=True)
                queue(a, fam, prefix, tag)
                built.add(fam)
        time.sleep(60)


def queue(a, fam, prefix, tag):
    with open(a.jobs, "a", encoding="utf-8") as f:
        for name, flag in STUDENTS[fam]:
            for s in a.seeds:
                job = f"{tag}kd_{fam}_{name}_s{s}"
                f.write(f"{job}\t-m esc.train_cls --dataset {a.dataset} --model roberta-large {flag} --seed {s} "
                        f"--distill {prefix}oof.npz {a.extra} --out runs/{a.dataset}/kd_{fam}_{name}_s{s}\n")
    print(time.strftime("%H:%M:%S"), "queued students of", fam, flush=True)


if __name__ == "__main__":
    main()
