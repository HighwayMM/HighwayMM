"""Spatial SigLIP2 features for TaxoWarn-S: patch tokens of 3 frames (clips: m1, m4, m6; images: m0) average-pooled from the
24x24 patch grid to 6x6 = 36 tokens per frame (masked frames only).
Output: HighwayMM_release\\features\\siglip2_spatial\\shard{i}.npz : ids [N], tok [N,3,36,1152] fp16
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REL = Path(r"D:\highwaymm\HighwayMM_release")
FR = (1, 4, 6)
G = 6


class DS(Dataset):
    def __init__(self, dirs, pre):
        self.dirs, self.pre = dirs, pre

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, i):
        d = self.dirs[i]
        files = [d / f"m{k}.jpg" for k in FR] if (d / "m7.jpg").exists() else [d / "m0.jpg"] * 3
        return torch.stack([self.pre(Image.open(f).convert("RGB")) for f in files]), d.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--only", default="", help="only item dirs starting with this prefix (incremental add)")
    ap.add_argument("--name", default="", help="output shard name suffix, e.g. iowa -> shard_iowa0.npz")
    a = ap.parse_args()
    dirs = sorted(p for p in (REL / "frames").iterdir() if (p / "m0.jpg").exists() and p.name.startswith(a.only))[a.shard::a.nshards]
    sfx = f"_{a.name}" if a.name else ""
    m, _, pre = open_clip.create_model_and_transforms("local-dir:D:/highwaymm/models/SigLIP2-SO400M-384")
    trunk = m.visual.trunk.cuda().eval().half()
    out = REL / "features" / "siglip2_spatial"
    out.mkdir(parents=True, exist_ok=True)
    dl = DataLoader(DS(dirs, pre), batch_size=8, num_workers=6, pin_memory=True)
    ids, toks = [], []
    with torch.no_grad():
        for b, (x, names) in enumerate(dl):
            B = x.shape[0]
            t = trunk.forward_features(x.view(-1, *x.shape[2:]).cuda().half())  # [B*3, 576, 1152] (no cls token)
            n = int(t.shape[1] ** 0.5)
            t = t[:, -n * n:].transpose(1, 2).reshape(t.shape[0], -1, n, n).float()
            t = F.adaptive_avg_pool2d(t, G).flatten(2).transpose(1, 2)  # [B*3, 36, 1152]
            toks.append(t.view(B, 3, G * G, -1).cpu().numpy().astype(np.float16))
            ids += list(names)
            if b % 300 == 0:
                print(f"[spatial shard {a.shard}] {len(ids)}/{len(dirs)}", flush=True)
    np.savez(out / f"shard{sfx}{a.shard}.npz", ids=np.array(ids), tok=np.concatenate(toks))
    print(f"[spatial shard {a.shard}] done {len(ids)}")


if __name__ == "__main__":
    main()

