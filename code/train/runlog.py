"""Mandatory experiment logging (Requirements.txt STEP 5.5).

Every run gets  D:\\highwaymm\\experiments\\<YYYYMMDD_HHMM>_<task>_<model>_<tag>\\  with
config.yaml, cmd.txt, env.txt, seed.txt, code_snapshot\\, train_log.txt (+ per-rank logs),
metrics.jsonl (global) + metrics_rank<r>.jsonl, final_metrics.json, and a row in
experiments\\runs_index.csv that is created at start (status=running) and updated at the end
(completed / failed / aborted). Run folders are never deleted.
"""
import csv
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
import yaml

from . import ddp

EXP_ROOT = Path(os.environ.get("EXP_ROOT", r"D:\highwaymm\experiments"))
CODE_ROOT = Path(__file__).resolve().parents[1]  # D:\highwaymm\code
INDEX_FIELDS = ["run_id", "date", "task", "model", "data_version", "split_version", "seed", "gpus",
                "world_size", "wall_time_s", "key_metrics", "status", "notes", "run_dir"]


# ---------------------------------------------------------------- index (file-locked; job runner writes concurrently)
class _Lock:
    def __init__(self, path, timeout=60):
        self.path, self.timeout = str(path) + ".lock", timeout

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                if time.time() - t0 > self.timeout:  # stale lock
                    try:
                        os.remove(self.path)
                    except OSError:
                        pass
                time.sleep(0.05)

    def __exit__(self, *a):
        os.close(self.fd)
        os.remove(self.path)


def _upsert_index(row: dict):
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    idx = EXP_ROOT / "runs_index.csv"
    with _Lock(idx):
        rows = []
        if idx.exists():
            with open(idx, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        for r in rows:
            if r["run_id"] == row["run_id"]:
                r.update({k: v for k, v in row.items() if k in INDEX_FIELDS})
                break
        else:
            rows.append({k: row.get(k, "") for k in INDEX_FIELDS})
        tmp = idx.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        for k in range(50):  # Windows: a reader holding the CSV open makes os.replace fail with PermissionError -> retry
            try:
                os.replace(tmp, idx)
                break
            except PermissionError:
                time.sleep(0.2 * (k + 1))
        else:
            raise PermissionError(f"could not update {idx}")


def mark_run(run_id, status, notes=""):
    """Mark an existing run (e.g. status='invalid') without deleting anything."""
    _upsert_index({"run_id": run_id, "status": status, "notes": notes})


# ---------------------------------------------------------------- env capture
def _sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except Exception as e:  # noqa
        return f"<failed: {e}>"


def code_hash(root=CODE_ROOT):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        if ".venv" in p.parts or "__pycache__" in p.parts:
            continue
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _env_text(ctx):
    lines = [f"date: {datetime.datetime.now().isoformat()}", f"host: {socket.gethostname()}",
             f"platform: {platform.platform()}", f"python: {sys.version}", f"executable: {sys.executable}",
             f"torch: {torch.__version__}  cuda: {torch.version.cuda}  cudnn: {torch.backends.cudnn.version()}",
             f"dist backend: gloo  world_size: {ctx.world_size}", f"code_hash(sha256[:16] of code/**/*.py): {code_hash()}"]
    for k in ["CUDA_VISIBLE_DEVICES", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "USE_LIBUV", "OMP_NUM_THREADS"]:
        lines.append(f"env {k}={os.environ.get(k, '')}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        lines.append(f"gpu{i}: {p.name} {p.total_memory / 2**30:.1f} GiB")
    lines += ["", "== nvidia-smi ==", _sh(["nvidia-smi"]), "== pip freeze ==", _sh([sys.executable, "-m", "pip", "freeze"])]
    return "\n".join(lines)


class _Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", encoding="utf-8", buffering=1)

    def write(self, s):
        self.stream.write(s)
        self.f.write(s)

    def flush(self):
        self.stream.flush()
        self.f.flush()

    def __getattr__(self, k):
        return getattr(self.stream, k)


# ---------------------------------------------------------------- run logger
class RunLogger:
    def __init__(self, cfg: dict, ctx, task: str, model: str, tag: str = "", notes: str = ""):
        self.cfg, self.ctx, self.t0 = cfg, ctx, time.time()
        self.status = "running"
        run_dir = None
        if ctx.is_main:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
            base = "_".join(x for x in [stamp, task, model, tag] if x).replace("/", "-").replace("\\", "-")
            EXP_ROOT.mkdir(parents=True, exist_ok=True)
            run_dir, k = EXP_ROOT / base, 1
            while True:  # atomic: concurrent job-runner processes may pick the same name
                try:
                    run_dir.mkdir()
                    break
                except FileExistsError:
                    k += 1
                    run_dir = EXP_ROOT / f"{base}_r{k}"
        self.dir = Path(ddp.broadcast_object(str(run_dir) if run_dir else None, ctx))
        self.run_id = self.dir.name

        # every rank: tee stdout/stderr
        log_name = "train_log.txt" if ctx.is_main else f"train_log_rank{ctx.rank}.txt"
        sys.stdout = _Tee(sys.stdout, self.dir / log_name)
        sys.stderr = _Tee(sys.stderr, self.dir / log_name)
        self._rank_metrics = open(self.dir / f"metrics_rank{ctx.rank}.jsonl", "a", encoding="utf-8", buffering=1)
        self._metrics = open(self.dir / "metrics.jsonl", "a", encoding="utf-8", buffering=1) if ctx.is_main else None

        if ctx.is_main:
            cfg = dict(cfg, run_id=self.run_id, run_dir=str(self.dir))
            (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
            launch = os.environ.get("LAUNCH_CMD", "")
            (self.dir / "cmd.txt").write_text(
                (f"launch: {launch}\n" if launch else "") + "argv: " + " ".join([sys.executable] + sys.argv) + "\n"
                f"cwd: {os.getcwd()}\n", encoding="utf-8")
            (self.dir / "seed.txt").write_text(str(cfg.get("seed", "")), encoding="utf-8")
            (self.dir / "env.txt").write_text(_env_text(ctx), encoding="utf-8")
            snap = self.dir / "code_snapshot"
            for p in CODE_ROOT.rglob("*"):
                if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".ps1", ".md"} and "__pycache__" not in p.parts:
                    dst = snap / p.relative_to(CODE_ROOT)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
            self._index_row = {
                "run_id": self.run_id, "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "task": task,
                "model": model, "data_version": cfg.get("data", {}).get("version", "") if isinstance(cfg.get("data"), dict) else "",
                "split_version": cfg.get("data", {}).get("split", "") if isinstance(cfg.get("data"), dict) else "",
                "seed": cfg.get("seed", ""), "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(ctx.world_size)))),
                "world_size": ctx.world_size, "status": "running", "notes": notes, "run_dir": str(self.dir)}
            _upsert_index(self._index_row)
            print(f"[runlog] run {self.run_id} -> {self.dir}")
        ddp.barrier(ctx)

    # per-rank raw metrics (every rank) ------------------------------------------
    def log_rank(self, **m):
        m = {"time": round(time.time() - self.t0, 3), "rank": self.ctx.rank, **m}
        self._rank_metrics.write(json.dumps(m, default=float) + "\n")

    # global (already reduced) metrics, rank 0 only ------------------------------
    def log(self, **m):
        if self._metrics:
            m = {"time": round(time.time() - self.t0, 3), **m}
            self._metrics.write(json.dumps(m, default=float) + "\n")

    def save_json(self, name, obj):
        if self.ctx.is_main:
            (self.dir / name).write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")

    def finish(self, status="completed", final_metrics=None, key_metrics=None, notes=None):
        if self.status != "running":
            return
        self.status = status
        wall = time.time() - self.t0
        if self.ctx.is_main:
            fm = dict(final_metrics or {}, status=status, wall_time_s=round(wall, 1))
            self.save_json("final_metrics.json", fm)
            km = key_metrics if key_metrics is not None else {k: v for k, v in (final_metrics or {}).items()
                                                               if isinstance(v, (int, float))}
            row = dict(self._index_row, status=status, wall_time_s=round(wall, 1),
                       key_metrics=json.dumps(km, default=float)[:500])
            if notes is not None:
                row["notes"] = notes
            _upsert_index(row)
            print(f"[runlog] run {self.run_id} finished: {status} ({wall:.0f}s)")
        for f in (self._rank_metrics, self._metrics):
            if f:
                f.flush()

    # context manager: failures/aborts are recorded, never lost -------------------
    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            self.finish("completed")
        elif issubclass(et, KeyboardInterrupt):
            self.finish("aborted", notes="KeyboardInterrupt")
        else:
            print("".join(traceback.format_exception(et, ev, tb)), file=sys.stderr)
            self.finish("failed", notes=f"{et.__name__}: {str(ev)[:200]}")
        return False
