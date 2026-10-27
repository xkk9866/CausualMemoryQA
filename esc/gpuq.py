"""Opportunistic GPU queue for a shared server.

    python -m esc.gpuq jobs.txt --gpus 0 1 2 3 4 5 6 7 --need 10000

jobs.txt: one job per line, `name<TAB>python args`. A job starts on the first allowed GPU
whose free memory exceeds --need MiB (and that this queue is not already using), and is
skipped if runs/.../best.json of a previous successful run exists in logs/<name>.done.
New lines appended to jobs.txt are picked up while the queue runs.
"""

import argparse
import os
import subprocess
import sys
import time


def free_mem():
    """free MiB per GPU; {} when nvidia-smi fails or hangs (treated as: nothing free)"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.total,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60).stdout
        res = {}
        for line in out.strip().splitlines():
            i, tot, used = [int(x) for x in line.split(",")]
            res[i] = tot - used
        return res
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--gpus", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--need", type=int, default=10000)
    p.add_argument("--per_gpu", type=int, default=1)
    p.add_argument("--forever", action="store_true", help="keep polling the job file after it drains")
    a = p.parse_args()
    os.makedirs("logs", exist_ok=True)
    running = {}  # name -> (proc, gpu)
    started = set()
    retry_at = {}  # name -> time before which a failed job is not retried
    while True:
        for name, (proc, gpu) in list(running.items()):
            if proc.poll() is not None:
                ok = proc.returncode == 0
                open(f"logs/{name}.{'done' if ok else 'failed'}", "w").write(str(proc.returncode))
                print(time.strftime("%H:%M:%S"), "finished", name, proc.returncode, flush=True)
                del running[name]
                if not ok:  # most failures here are OOM races with other users: retry later
                    started.discard(name)
                    retry_at[name] = time.time() + 600
        jobs = []
        for line in open(a.jobs, encoding="utf-8"):
            if line.strip() and not line.startswith("#"):
                name, args = line.rstrip("\n").split("\t", 1)
                jobs.append((name, args))
        todo = [(n, x) for n, x in jobs if n not in started and not os.path.exists(f"logs/{n}.done")
                and retry_at.get(n, 0) <= time.time()]
        if not todo and not running and not a.forever:
            print("queue empty", flush=True)
            break
        if todo:
            free = free_mem()
            busy = {}
            for _, g in running.values():
                busy[g] = busy.get(g, 0) + 1
            for g in a.gpus:
                if not todo:
                    break
                if free.get(g, 0) >= a.need and busy.get(g, 0) < a.per_gpu:
                    name, args = todo.pop(0)
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                    log = open(f"logs/{name}.log", "w")
                    proc = subprocess.Popen([sys.executable] + args.split(), stdout=log, stderr=subprocess.STDOUT, env=env)
                    running[name] = (proc, g)
                    started.add(name)
                    busy[g] = busy.get(g, 0) + 1
                    print(time.strftime("%H:%M:%S"), "start", name, "gpu", g, "free", free[g], flush=True)
                    time.sleep(90)  # let it allocate before measuring free memory again
                    free = free_mem()
        time.sleep(30)


if __name__ == "__main__":
    main()
