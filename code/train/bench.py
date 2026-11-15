"""STEP 1.6 sanity benchmark for gloo DDP on 3x RTX 3090.

  launch.ps1 -Gpus 3 code\\train\\bench.py --mode all
  launch.ps1 -Gpus 1 code\\train\\bench.py --mode all

Checks: (a) DDP gradients == single-process full-batch gradients (also with no_sync accumulation),
(b) checkpoint/resume reproduces the same loss trajectory, (c) training throughput (synthetic GPU tensors,
so this measures compute + gloo sync, not data loading) for ResNet-50 and ViT-B/16, with/without accumulation.
Logged as a normal run under experiments/.
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import ddp  # noqa: E402
from train.runlog import RunLogger  # noqa: E402
from train.trainer import load_ckpt, save_ckpt  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402


def small_net(c=10):
    # GroupNorm (no BatchNorm) so per-shard statistics don't break exact gradient equality
    return nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.GroupNorm(8, 32), nn.GELU(),
                         nn.Conv2d(32, 64, 3, 2, 1), nn.GroupNorm(8, 64), nn.GELU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, c))


def batch_for(step, n, device, size=64, c=10):
    g = torch.Generator().manual_seed(10_000 + step)
    return torch.randn(n, 3, size, size, generator=g).to(device), torch.randint(0, c, (n,), generator=g).to(device)


def test_grad_equality(ctx, B=16):
    torch.manual_seed(0)
    base = small_net().to(ctx.device)
    ref = copy.deepcopy(base)
    model = DDP(base, device_ids=[ctx.local_rank]) if ctx.distributed else base
    W, r = ctx.world_size, ctx.rank
    lf = nn.CrossEntropyLoss()
    res = {}
    # (1) single step
    x, y = batch_for(0, B * W, ctx.device)
    lf(model(x[r * B:(r + 1) * B]), y[r * B:(r + 1) * B]).backward()
    lf(ref(x), y).backward()
    d = max((p.grad - q.grad).abs().max().item() for p, q in zip(ddp.unwrap(model).parameters(), ref.parameters()))
    s = max(q.grad.abs().max().item() for q in ref.parameters())
    res["grad_maxabs_diff"], res["grad_maxabs_ref"] = ddp.all_reduce_mean(d, ctx), s
    # (2) accumulation over 2 micro-batches with no_sync on the first
    for m in (model, ref):
        m.zero_grad(set_to_none=True)
    xs = [batch_for(k + 1, B * W, ctx.device) for k in range(2)]
    for k, (x, y) in enumerate(xs):
        with ddp.maybe_no_sync(model, sync=k == 1):
            (lf(model(x[r * B:(r + 1) * B]), y[r * B:(r + 1) * B]) / 2).backward()
        (lf(ref(x), y) / 2).backward()
    d2 = max((p.grad - q.grad).abs().max().item() for p, q in zip(ddp.unwrap(model).parameters(), ref.parameters()))
    res["grad_accum_maxabs_diff"] = ddp.all_reduce_mean(d2, ctx)
    res["grad_equal_pass"] = bool(res["grad_maxabs_diff"] < 1e-5 * max(s, 1) and res["grad_accum_maxabs_diff"] < 1e-5 * max(s, 1))
    return res


def test_resume(ctx, run_dir, B=16, steps=20, at=10):
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    lf = nn.CrossEntropyLoss()

    def make(seed):
        torch.manual_seed(seed)
        net = small_net().to(ctx.device)
        m = DDP(net, device_ids=[ctx.local_rank]) if ctx.distributed else net
        return m, torch.optim.AdamW(m.parameters(), lr=1e-3)

    def run(m, opt, s0, s1):
        out = []
        for s in range(s0, s1):
            x, y = batch_for(100 + s * ctx.world_size + ctx.rank, B, ctx.device)
            loss = lf(m(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            out.append(ddp.all_reduce_mean(loss.item(), ctx))
        return out

    ck = Path(run_dir) / "checkpoints" / "resume_test.pt"
    m, opt = make(0)
    run(m, opt, 0, at)
    if ctx.is_main:
        ck.parent.mkdir(exist_ok=True)
        save_ckpt(ck, m, opt, step=at)
    ddp.barrier(ctx)
    a = run(m, opt, at, steps)
    m2, opt2 = make(999)  # different init: must be fully overwritten by the checkpoint
    load_ckpt(ck, m2, opt2, device=ctx.device, restore_rng=False)
    b = run(m2, opt2, at, steps)
    diff = max(abs(u - v) for u, v in zip(a, b))
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = False, True
    return {"resume_max_loss_diff": diff, "resume_pass": bool(diff < 1e-5)}


def bench_throughput(ctx, name, B, accum, size=224, warm=10, iters=40):
    import timm
    torch.manual_seed(0)
    net = timm.create_model(name, pretrained=False, num_classes=10).to(ctx.device).to(memory_format=torch.channels_last)
    m = DDP(net, device_ids=[ctx.local_rank], gradient_as_bucket_view=True, bucket_cap_mb=50) if ctx.distributed else net
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, fused=True)
    lf = nn.CrossEntropyLoss()
    x = torch.randn(B, 3, size, size, device=ctx.device).contiguous(memory_format=torch.channels_last)
    y = torch.randint(0, 10, (B,), device=ctx.device)
    torch.cuda.reset_peak_memory_stats()

    def micro(k):
        with ddp.maybe_no_sync(m, sync=(k + 1) % accum == 0):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = lf(m(x).float(), y) / accum
            loss.backward()
        if (k + 1) % accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    for k in range(warm * accum):
        micro(k)
    torch.cuda.synchronize()
    ddp.barrier(ctx)
    t0 = time.time()
    for k in range(iters * accum):
        micro(k)
    torch.cuda.synchronize()
    el = time.time() - t0
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    r = {"model": name, "per_gpu_batch": B, "accum": accum, "params_M": round(n_params, 1),
         "img_per_s_global": ddp.all_reduce_mean(B * iters * accum / el, ctx) * ctx.world_size,
         "opt_step_s": el / iters, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    del m, net, opt
    torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all", choices=["all", "correctness", "throughput"])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    ctx = ddp.setup()
    cfg = {"task": "bench_ddp", "seed": 0, "mode": a.mode, "world_size": ctx.world_size,
           "data": {"version": "synthetic", "split": "none"}}
    try:
        with RunLogger(cfg, ctx, task="bench", model="ddp", tag=a.tag or f"{ctx.world_size}gpu") as log:
            out = {"world_size": ctx.world_size}
            if a.mode in ("all", "correctness"):
                out.update(test_grad_equality(ctx))
                out.update(test_resume(ctx, log.dir))
                if ctx.is_main:
                    print("[correctness]", {k: v for k, v in out.items()})
            tp = []
            if a.mode in ("all", "throughput"):
                torch.backends.cudnn.benchmark = True
                for name, B in [("resnet50", 96), ("vit_base_patch16_224", 64)]:
                    for acc in (1, 4):
                        r = bench_throughput(ctx, name, B, acc)
                        tp.append(r)
                        log.log(**r)
                        if ctx.is_main:
                            print("[throughput]", r)
                out["throughput"] = tp
            km = {k: v for k, v in out.items() if k.endswith("pass")}
            km.update({f"{r['model']}_acc{r['accum']}_img_s": round(r["img_per_s_global"], 1) for r in tp})
            log.finish("completed", final_metrics=out, key_metrics=km)
    finally:
        ddp.cleanup()


if __name__ == "__main__":
    main()
