"""Distributed helpers for native-Windows torch (gloo only, no NCCL).

Launch with torchrun (see launch.ps1, which sets USE_LIBUV=0 for the Windows rendezvous).
Works unchanged as a plain single-process script (world_size=1).
"""
import datetime
import os
import random
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist


class Ctx:
    def __init__(self, rank, local_rank, world_size, device):
        self.rank, self.local_rank, self.world_size, self.device = rank, local_rank, world_size, device

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def distributed(self):
        return self.world_size > 1


def setup(timeout_min=30) -> Ctx:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        rank = int(os.environ["RANK"])
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        dist.init_process_group(backend="gloo", timeout=datetime.timedelta(minutes=timeout_min))
    else:
        rank = local = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    dev = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    return Ctx(rank, local, world, dev)


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(ctx):
    if ctx.distributed:
        dist.barrier()


def seed_all(seed, rank=0):
    s = seed + rank
    random.seed(s)
    np.random.seed(s % (2**32))
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def broadcast_object(obj, ctx, src=0):
    if not ctx.distributed:
        return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=src)
    return lst[0]


def all_reduce_mean(value, ctx):
    """Average a python float / tensor across ranks (returns float)."""
    t = torch.as_tensor(value, dtype=torch.float64, device="cpu").clone()
    if ctx.distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= ctx.world_size
    return t.item() if t.numel() == 1 else t


def all_reduce_sum(t: torch.Tensor, ctx):
    t = t.detach().to("cpu", torch.float64).clone()
    if ctx.distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_gather_objects(obj, ctx):
    if not ctx.distributed:
        return [obj]
    out = [None] * ctx.world_size
    dist.all_gather_object(out, obj)
    return out


def maybe_no_sync(model, sync: bool):
    """Skip DDP gradient all-reduce on accumulation micro-steps (big win with slow gloo)."""
    if sync or not hasattr(model, "no_sync"):
        return nullcontext()
    return model.no_sync()


def unwrap(model):
    return model.module if hasattr(model, "module") else model
