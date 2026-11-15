"""Generic DDP(gloo) trainer: AMP, gradient accumulation with no_sync(), partial freezing,
warmup+cosine LR, best/last checkpoints with exact resume, per-rank + global metric logging,
prediction dumps and loss/metric plots. Task-specific parts come from a task module (tasks/*.py).
"""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from . import ddp


class ShardSampler(Sampler):
    """Deterministic eval sharding without the padding/duplication of DistributedSampler."""

    def __init__(self, n, rank, world):
        self.idx = list(range(rank, n, world))

    def __iter__(self):
        return iter(self.idx)

    def __len__(self):
        return len(self.idx)


# ------------------------------------------------------------------ checkpoints
def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state()}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"].cpu())  # ckpt may have been mapped to cuda; RNG states must be CPU ByteTensors
    torch.cuda.set_rng_state(s["cuda"].cpu())


def save_ckpt(path, model, opt, sched=None, scaler=None, **extra):
    tmp = Path(str(path) + ".tmp")
    torch.save({"model": ddp.unwrap(model).state_dict(), "opt": opt.state_dict(),
                "sched": sched.state_dict() if sched else None, "scaler": scaler.state_dict() if scaler else None,
                "rng": rng_state(), **extra}, tmp)
    tmp.replace(path)


def load_ckpt(path, model, opt=None, sched=None, scaler=None, device="cpu", restore_rng=True):
    ck = torch.load(path, map_location=device, weights_only=False)
    ddp.unwrap(model).load_state_dict(ck["model"])
    if opt is not None and ck.get("opt"):
        opt.load_state_dict(ck["opt"])
    if sched is not None and ck.get("sched"):
        sched.load_state_dict(ck["sched"])
    if scaler is not None and ck.get("scaler"):
        scaler.load_state_dict(ck["scaler"])
    if restore_rng and ck.get("rng"):
        set_rng_state(ck["rng"])
    return ck


# ------------------------------------------------------------------ default eval (classification)
@torch.no_grad()
def default_eval(model, loader, ctx, amp_dtype):
    model.eval()
    stats = torch.zeros(3, dtype=torch.float64)  # loss_sum, correct, n
    preds = []
    for batch in loader:
        x, y = batch[0].to(ctx.device, non_blocking=True), batch[1].to(ctx.device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.float(), y, reduction="sum")
        p = logits.argmax(1)
        stats += torch.tensor([loss.item(), (p == y).sum().item(), y.numel()], dtype=torch.float64)
        ids = batch[2].tolist() if len(batch) > 2 else [None] * len(y)
        preds += [{"id": i, "y": int(t), "pred": int(q)} for i, t, q in zip(ids, y.tolist(), p.tolist())]
    stats = ddp.all_reduce_sum(stats, ctx)
    return {"val/loss": stats[0].item() / max(stats[2].item(), 1), "val/acc": stats[1].item() / max(stats[2].item(), 1)}, preds


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, cfg, ctx, logger, model, train_ds, val_ds=None, loss_fn=None, eval_fn=None,
                 monitor="val/acc", mode="max", collate_fn=None):
        self.cfg, self.ctx, self.log = cfg, ctx, logger
        t = cfg["train"]
        self.t = t
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(t.get("amp", "bf16"))
        self.accum = int(t.get("accum_steps", 1))
        self.monitor, self.mode = monitor, mode
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.eval_fn = eval_fn or default_eval

        # freezing (prefix list), channels_last
        for name, p in model.named_parameters():
            if any(name.startswith(f) for f in t.get("freeze", []) or []):
                p.requires_grad_(False)
        model = model.to(ctx.device)
        if t.get("channels_last", True):
            model = model.to(memory_format=torch.channels_last)
        if ctx.distributed:
            model = DDP(model, device_ids=[ctx.local_rank], bucket_cap_mb=t.get("bucket_cap_mb", 50),
                        gradient_as_bucket_view=True, find_unused_parameters=t.get("find_unused", False))
        self.model = model

        nw = int(t.get("num_workers", 4))
        dl_kw = dict(num_workers=nw, pin_memory=True, persistent_workers=nw > 0, collate_fn=collate_fn)
        if nw > 0:
            dl_kw["prefetch_factor"] = t.get("prefetch_factor", 4)
        self.train_sampler = DistributedSampler(train_ds, ctx.world_size, ctx.rank, shuffle=True,
                                                seed=cfg.get("seed", 0), drop_last=True) if ctx.distributed else None
        self.train_dl = DataLoader(train_ds, batch_size=t["batch_size"], sampler=self.train_sampler,
                                   shuffle=self.train_sampler is None, drop_last=True, **dl_kw)
        self.val_dl = DataLoader(val_ds, batch_size=t.get("eval_batch_size", t["batch_size"]),
                                 sampler=ShardSampler(len(val_ds), ctx.rank, ctx.world_size), **dl_kw) if val_ds else None

        params = [p for p in self.model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(params, lr=t.get("lr", 1e-3), weight_decay=t.get("weight_decay", 0.05),
                                     fused=True)
        self.steps_per_epoch = max(len(self.train_dl) // self.accum, 1)
        total = self.steps_per_epoch * t["epochs"]
        warm = int(t.get("warmup_frac", 0.05) * total)
        self.sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_dtype == torch.float16)
        self.epoch, self.step = 0, 0
        self.best = -float("inf") if mode == "max" else float("inf")
        self.history = []

        resume = t.get("resume")
        if resume:
            path = Path(resume)
            if path.is_dir():
                path = path / "checkpoints" / "last.pt"
            ck = load_ckpt(path, self.model, self.opt, self.sched, self.scaler, device=ctx.device)
            self.epoch, self.step, self.best = ck["epoch"], ck["step"], ck["best"]
            if ctx.is_main:
                print(f"[trainer] resumed from {path} at epoch {self.epoch} step {self.step}")

    # ------------------------------------------------------------------
    def _better(self, v):
        return v > self.best if self.mode == "max" else v < self.best

    def train_epoch(self):
        m, ctx, t = self.model, self.ctx, self.t
        m.train()
        if self.train_sampler:
            self.train_sampler.set_epoch(self.epoch)
        t0, seen, loss_acc, n_acc = time.time(), 0, 0.0, 0
        log_every = t.get("log_every", 20)
        self.opt.zero_grad(set_to_none=True)
        for i, batch in enumerate(self.train_dl):
            x, y = batch[0].to(ctx.device, non_blocking=True), batch[1].to(ctx.device, non_blocking=True)
            if x.ndim == 4 and t.get("channels_last", True):
                x = x.contiguous(memory_format=torch.channels_last)
            sync = (i + 1) % self.accum == 0
            with ddp.maybe_no_sync(m, sync):
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
                    out = m(x)
                    loss = self.loss_fn(out.float() if torch.is_tensor(out) else out, y)
                self.scaler.scale(loss / self.accum).backward()
            loss_acc += loss.item()
            n_acc += 1
            seen += x.shape[0]
            if not sync:
                continue
            if t.get("grad_clip"):
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], t["grad_clip"])
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad(set_to_none=True)
            self.sched.step()
            self.step += 1
            if self.step % log_every == 0:
                el = max(time.time() - t0, 1e-6)
                self.log.log_rank(step=self.step, epoch=self.epoch, loss=loss_acc / n_acc, samples_per_s=seen / el,
                                  gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30)
                g = ddp.all_reduce_mean(loss_acc / n_acc, ctx)
                if ctx.is_main:
                    lr = self.sched.get_last_lr()[0]
                    self.log.log(step=self.step, epoch=self.epoch, **{"train/loss": g, "lr": lr,
                                                                       "train/samples_per_s_global": seen / el * ctx.world_size})
                    print(f"ep {self.epoch} step {self.step} loss {g:.4f} lr {lr:.2e} "
                          f"{seen / el * ctx.world_size:.0f} samp/s")
                loss_acc, n_acc = 0.0, 0
        dt = max(time.time() - t0, 1e-6)
        return {"train/epoch_time_s": dt, "train/samples_per_s_global": seen / dt * ctx.world_size}

    def evaluate(self, split="val"):
        if self.val_dl is None:
            return {}
        metrics, preds = self.eval_fn(self.model, self.val_dl, self.ctx, self.amp_dtype)
        allp = ddp.all_gather_objects(preds, self.ctx)
        if self.ctx.is_main:
            with open(self.log.dir / f"predictions_{split}_ep{self.epoch}.jsonl", "w", encoding="utf-8") as f:
                for part in allp:
                    for p in part:
                        f.write(json.dumps(p) + "\n")
        return metrics

    def fit(self):
        ctx, t = self.ctx, self.t
        ck_dir = self.log.dir / "checkpoints"
        if ctx.is_main:
            ck_dir.mkdir(exist_ok=True)
        final = {}
        while self.epoch < t["epochs"]:
            tm = self.train_epoch()
            self.epoch += 1
            vm = self.evaluate() if (self.epoch % t.get("eval_every", 1) == 0 or self.epoch == t["epochs"]) else {}
            rec = {"epoch": self.epoch, "step": self.step, **tm, **vm}
            self.history.append(rec)
            if ctx.is_main:
                self.log.log(**rec)
                print(f"[epoch {self.epoch}] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items() if isinstance(v, float)))
                if self.monitor in vm and self._better(vm[self.monitor]):
                    self.best = vm[self.monitor]
                    save_ckpt(ck_dir / "best.pt", self.model, self.opt, self.sched, self.scaler,
                              epoch=self.epoch, step=self.step, best=self.best, cfg=self.cfg)
                save_ckpt(ck_dir / "last.pt", self.model, self.opt, self.sched, self.scaler,
                          epoch=self.epoch, step=self.step, best=self.best, cfg=self.cfg)
            self.best = ddp.broadcast_object(self.best, ctx)
            final = rec
        final = dict(final, **{f"best/{self.monitor}": self.best})
        if ctx.is_main:
            self.plot()
        return final

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[trainer] matplotlib not installed; skipping plots")
            return
        rows = [json.loads(l) for l in open(self.log.dir / "metrics.jsonl", encoding="utf-8")]
        keys = sorted({k for r in rows for k in r if k.startswith(("train/loss", "val/"))})
        if not keys:
            return
        fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 3), squeeze=False)
        for ax, k in zip(axes[0], keys):
            pts = [(r["step"], r[k]) for r in rows if k in r]
            ax.plot(*zip(*pts), marker="." if len(pts) < 50 else None)
            ax.set_title(k)
            ax.set_xlabel("step")
        fig.tight_layout()
        fig.savefig(self.log.dir / "curves.png", dpi=120)
        plt.close(fig)

