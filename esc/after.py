"""Queue jobs once their prerequisites are done.

    python -m esc.after --wait recogf0,recogf1 --run "-m esc.oof runs/esconv/recog_f 5" --jobs jobs.txt --add new_jobs.txt

Waits until logs/<name>.done exists for every --wait job, runs each --run command with this
interpreter, then appends the lines of --add to the queue's job file.
"""

import argparse
import os
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wait", required=True)
    p.add_argument("--run", action="append", default=[])
    p.add_argument("--jobs", default="jobs.txt")
    p.add_argument("--add", required=True)
    a = p.parse_args()
    names = a.wait.split(",")
    while not all(os.path.exists(f"logs/{n}.done") for n in names):
        time.sleep(60)
    for cmd in a.run:
        subprocess.run([sys.executable] + cmd.split(), check=True)
    with open(a.jobs, "a", encoding="utf-8") as f:
        f.write(open(a.add, encoding="utf-8").read())
    print(time.strftime("%H:%M:%S"), "queued", a.add, flush=True)


if __name__ == "__main__":
    main()
