"""Shared training entry point.

  powershell -File code\\train\\launch.ps1 -Gpus 3 -Config code\\train\\configs\\synthetic.yaml [-Set "train.epochs=2","seed=1"] [-Tag x]

Config must define: task (module in train/tasks), model.name, seed, train.{epochs,batch_size,...}.
Every run (incl. failures/aborts) is logged under D:\\highwaymm\\experiments (see runlog.py).
"""
import argparse
import importlib
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # D:\highwaymm\code

from train import ddp  # noqa: E402
from train.runlog import RunLogger  # noqa: E402
from train.trainer import Trainer  # noqa: E402


def apply_overrides(cfg, sets):
    for s in sets or []:
        k, v = s.split("=", 1)
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = yaml.safe_load(v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    cfg = apply_overrides(yaml.safe_load(open(a.config, encoding="utf-8")), a.set)
    cfg["config_file"] = str(Path(a.config).resolve())

    ctx = ddp.setup()
    ddp.seed_all(cfg.get("seed", 0), ctx.rank)
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        tag = "_".join(x for x in [a.tag or cfg.get("tag", ""), f"s{cfg.get('seed', 0)}"] if x)
        with RunLogger(cfg, ctx, task=cfg["task"], model=cfg["model"]["name"], tag=tag, notes=a.notes) as log:
            parts = importlib.import_module(f"train.tasks.{cfg['task']}").build(cfg, ctx)
            tr = Trainer(cfg, ctx, log, **parts)
            final = tr.fit()
            log.finish("completed", final_metrics=final,
                       key_metrics={k: v for k, v in final.items() if k.startswith(("val/", "best/"))})
    finally:
        ddp.cleanup()


if __name__ == "__main__":
    main()
