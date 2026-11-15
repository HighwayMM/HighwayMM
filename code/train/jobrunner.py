"""Multi-GPU job runner (STEP 1.3): one worker per GPU pulls commands from a queue.
Best for embarrassingly parallel work: feature extraction, pseudo-labelling, inference, per-seed / per-ablation runs.

  python code\\train\\jobrunner.py --gpus 0,1,2 --jobs jobs.txt
  python code\\train\\jobrunner.py --gpus 0,1,2 --cmd "code\\train\\main.py --config X --set seed={seed}" --grid seed=0,1,2

jobs.txt: one command per line (python script + args; '#' comments). Each job runs with CUDA_VISIBLE_DEVICES=<gpu>
and WORLD_SIZE=1, so training jobs still create their own run folder via runlog. The runner's own log and one
stdout/stderr file per job go to experiments\\_jobrunner\\<timestamp>\\ (summary.csv lists gpu, exit code, duration).
"""
import argparse
import csv
import datetime
import itertools
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(r"D:\highwaymm")
EXP = Path(os.environ.get("EXP_ROOT", ROOT / "experiments"))


def expand_grid(cmd, grid):
    if not grid:
        return [cmd]
    keys, vals = zip(*[(k, v.split(",")) for k, v in (g.split("=", 1) for g in grid)])
    return [cmd.format(**dict(zip(keys, combo))) for combo in itertools.product(*vals)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2")
    ap.add_argument("--jobs")
    ap.add_argument("--cmd")
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per GPU (small inference jobs)")
    a = ap.parse_args()
    if a.jobs:
        # utf-8-sig: PowerShell's Set-Content -Encoding utf8 writes a BOM that would otherwise corrupt the first command
        cmds = [l.strip() for l in open(a.jobs, encoding="utf-8-sig") if l.strip() and not l.strip().startswith("#")]
    elif a.cmd:
        cmds = expand_grid(a.cmd, a.grid)
    else:
        sys.exit("give --jobs or --cmd")

    out = EXP / "_jobrunner" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    (out / "jobs.txt").write_text("\n".join(cmds), encoding="utf-8")
    q = queue.Queue()
    for i, c in enumerate(cmds):
        q.put((i, c))
    results, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                i, c = q.get_nowait()
            except queue.Empty:
                return
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WORLD_SIZE="1", USE_LIBUV="0", PYTHONUNBUFFERED="1",
                       LAUNCH_CMD=f"jobrunner gpu={gpu}: {c}")
            t0 = time.time()
            with open(out / f"job{i:04d}_gpu{gpu}.log", "w", encoding="utf-8") as f:
                f.write(f"# {c}\n")
                f.flush()
                rc = subprocess.run([PY, "-u"] + shlex.split(c, posix=False), cwd=ROOT, env=env, stdout=f,
                                    stderr=subprocess.STDOUT).returncode
            dt = time.time() - t0
            with lock:
                results.append({"job": i, "gpu": gpu, "exit": rc, "seconds": round(dt, 1), "cmd": c})
                print(f"[jobrunner] job {i} gpu {gpu} exit {rc} {dt:.0f}s  ({len(results)}/{len(cmds)})", flush=True)

    gpus = [g for g in a.gpus.split(",") if g != ""]
    ts = [threading.Thread(target=worker, args=(g,)) for g in gpus for _ in range(a.per_gpu)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["job", "gpu", "exit", "seconds", "cmd"])
        w.writeheader()
        w.writerows(sorted(results, key=lambda r: r["job"]))
    n_bad = sum(r["exit"] != 0 for r in results)
    print(f"[jobrunner] done: {len(results) - n_bad} ok, {n_bad} failed -> {out}")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
