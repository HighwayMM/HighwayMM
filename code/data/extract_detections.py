"""Object-detection features per frame (COCO Faster R-CNN R50-FPN v2, torchvision) for every item with frames.

Reads HighwayMM_release/frames/<item>/m{k}.jpg (the same overlay-masked frames the models see; read-only) and writes
HighwayMM_release/features/det/shard{i}.npz with ids [N], nfr [N], det [N, 8, D] float16 (zero rows = no frame).
Per-frame descriptor (D = 20), "veh" = car+truck+bus+motorcycle at score >= 0.4:
  0-4  log1p counts: car, truck, bus, motorcycle, person
  5    log1p vehicles
  6    mean vehicle score             7  fraction of vehicle candidates (score>=0.2) that are weak (<0.5)
  8    mean vehicle box area (frac)   9  max vehicle box area (frac)     10 vehicle occupancy (sum area, clipped 1)
  11   mean vehicle y-centre          12 std vehicle x-centre            13 log1p candidates 0.2-0.4 (uncertain)
  14   mean score of ALL detections >= 0.2 (scene "detectability"; drops in fog/snow/night)
  15   matched-to-previous-frame fraction    16 mean displacement of matched vehicles (frac of width)
  17   fraction of matched vehicles that are (nearly) static     18 count change vs previous frame
  19   frame present flag
  python extract_detections.py --shard i --nshards n
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torchvision.transforms.functional import to_tensor

REL = Path(r"D:\highwaymm\HighwayMM_release")
K, D = 8, 20
COCO = {"car": 3, "motorcycle": 4, "bus": 6, "truck": 8, "person": 1}
VEH = [3, 4, 6, 8]


def frame_paths(item):
    d = REL / "frames" / item
    ps = [d / f"m{k}.jpg" for k in range(K)]
    return [p for p in ps if p.exists()]


def describe(det, W, H, prev):
    s, l, b = det["scores"].cpu().numpy(), det["labels"].cpu().numpy(), det["boxes"].cpu().numpy()
    f = np.zeros(D, np.float32)
    keep = s >= 0.4
    for j, c in enumerate(["car", "truck", "bus", "motorcycle", "person"]):
        f[j] = math.log1p(int(((l == COCO[c]) & keep).sum()))
    vm = np.isin(l, VEH)
    v = vm & keep
    f[5] = math.log1p(int(v.sum()))
    cand = vm & (s >= 0.2)
    if v.any():
        bb = b[v]
        area = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1]) / (W * H)
        cx, cy = (bb[:, 0] + bb[:, 2]) / 2 / W, (bb[:, 1] + bb[:, 3]) / 2 / H
        f[6] = s[v].mean(); f[8] = area.mean(); f[9] = area.max(); f[10] = min(area.sum(), 1.0)
        f[11] = cy.mean(); f[12] = cx.std()
    else:
        cx = cy = np.zeros(0)
    f[7] = ((s[cand] < 0.5).mean()) if cand.any() else 0.0
    f[13] = math.log1p(int((vm & (s >= 0.2) & (s < 0.4)).sum()))
    f[14] = s[s >= 0.2].mean() if (s >= 0.2).any() else 0.0
    cur = np.stack([cx, cy], 1) if len(cx) else np.zeros((0, 2))
    if prev is not None and len(prev) and len(cur):
        dist = np.linalg.norm(cur[:, None] - prev[None], axis=-1)
        mn = dist.min(1)
        m = mn < 0.08
        f[15] = m.mean()
        if m.any():
            f[16] = mn[m].mean(); f[17] = (mn[m] < 0.01).mean()
        f[18] = (len(cur) - len(prev)) / max(len(prev), 1)
    f[19] = 1.0
    return f, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--labels", default=str(REL / "annotations" / "labels_v2t.csv"))
    a = ap.parse_args()
    out = REL / "features" / "det"
    out.mkdir(parents=True, exist_ok=True)
    dst = out / f"shard{a.shard}.npz"
    if dst.exists():
        print("exists", dst); return
    ids = sorted(pd.read_csv(a.labels, usecols=["item_id"]).item_id.unique())[a.shard::a.nshards]
    dev = "cuda"
    w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=0.2, min_size=512, max_size=912)
    model.eval().to(dev)
    jobs = [(i, k, p) for i, it in enumerate(ids) for k, p in enumerate(frame_paths(it))]
    feats = np.zeros((len(ids), K, D), np.float16)
    nfr = np.zeros(len(ids), np.int16)
    for it_i, k, _ in jobs:
        nfr[it_i] = max(nfr[it_i], k + 1)
    prev = {}
    print(f"shard {a.shard}: {len(ids)} items, {len(jobs)} frames", flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(jobs), a.bs):
            batch = jobs[s:s + a.bs]
            ims = [Image.open(p).convert("RGB") for _, _, p in batch]
            dets = model([to_tensor(im).to(dev) for im in ims])
            for (i, k, _), im, d in zip(batch, ims, dets):
                f, cur = describe(d, im.width, im.height, prev.get(i) if k > 0 else None)
                feats[i, k] = f
                prev[i] = cur
            if (s // a.bs) % 200 == 0:
                print(f"  {s}/{len(jobs)}", flush=True)
    np.savez(dst, ids=np.array(ids), nfr=nfr, det=feats)
    print("wrote", dst, feats.shape)


if __name__ == "__main__":
    main()
