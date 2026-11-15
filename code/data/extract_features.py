"""Frozen SigLIP2-SO400M/16-384 frame embeddings for every prepared item (masked + unmasked frames).

Output: D:\\highwaymm\\HighwayMM_release\\features\\siglip2\\shard{i}.npz with
  ids [N], nfr [N], masked [N,8,1152] fp16, raw [N,8,1152] fp16   (frames beyond nfr are zero)
Sharded via jobrunner (one shard per GPU). Items = every folder under frames\\ (ours) and frames_ext\\ (external).
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REL = Path(r"D:\highwaymm\HighwayMM_release")
MODEL = "local-dir:D:/highwaymm/models/SigLIP2-SO400M-384"
K = 8


class DS(Dataset):
    def __init__(self, dirs, pre):
        self.dirs, self.pre = dirs, pre

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        n = len(list(d.glob("m*.jpg")))
        xs = []
        for kind in ("m", "r"):
            for k in range(K):
                p = d / f"{kind}{min(k, n - 1)}.jpg"
                xs.append(self.pre(Image.open(p).convert("RGB")))
        return torch.stack(xs), n, d.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="", help="only item dirs starting with this prefix (incremental add)")
    ap.add_argument("--name", default="", help="output shard name suffix, e.g. iowa -> shard_iowa0.npz")
    a = ap.parse_args()
    dirs = sorted([p for root in ("frames", "frames_ext") if (REL / root).exists() for p in (REL / root).iterdir()
                   if (p / "m0.jpg").exists() and p.name.startswith(a.only)])[a.shard::a.nshards]
    sfx = f"_{a.name}" if a.name else ""
    out = REL / "features" / "siglip2"
    out.mkdir(parents=True, exist_ok=True)
    m, _, pre = open_clip.create_model_and_transforms(MODEL)
    m = m.cuda().eval().half()
    dl = DataLoader(DS(dirs, pre), batch_size=4, num_workers=6, pin_memory=True)
    ids, nfr, em, er = [], [], [], []
    with torch.no_grad():
        for b, (x, n, names) in enumerate(dl):
            B = x.shape[0]
            e = m.encode_image(x.view(-1, *x.shape[2:]).cuda().half())
            e = torch.nn.functional.normalize(e.float(), dim=-1).view(B, 2, K, -1).cpu().numpy().astype(np.float16)
            for j in range(B):
                nn_ = int(n[j])
                e[j, :, nn_:] = 0
            em.append(e[:, 0]); er.append(e[:, 1]); ids += list(names); nfr += [int(v) for v in n]
            if b % 200 == 0:
                print(f"[feat shard {a.shard}] {len(ids)}/{len(dirs)}", flush=True)
    np.savez(out / f"shard{sfx}{a.shard}.npz", ids=np.array(ids), nfr=np.array(nfr), masked=np.concatenate(em), raw=np.concatenate(er))
    print(f"[feat shard {a.shard}] done {len(ids)}")


if __name__ == "__main__":
    main()

