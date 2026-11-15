"""Build the public HighwayMM release package -> D:\\highwaymm\\HighwayMM_public  (never modifies HighwayMM_release).

  python code\\data\\build_public_release.py --blur --shard 0 --nshard 3 --gpu 0     # one per GPU, in parallel
  python code\\data\\build_public_release.py --finalize                              # metadata, features, checks

What is released
  frames   : overlay-MASKED frames (the model input) of redistributable sources only: our ITS/KEC snapshots and
             sequences, our winter pilot clips (minus excluded files), NYSDOT 511NY (CC BY 4.0), FloodCam-NC.
             Persons are blurred in full; the lower part of every vehicle large enough to carry a readable plate is
             blurred (COCO Faster R-CNN R50-FPN v2, torchvision). Per-frame counts are logged in blur_log.csv.
  labels   : taxonomy labels with per-attribute provenance for ALL items (ID-only sources included), splits,
             gold labels (annotators anonymised), hazard-mining audit, VLM pseudo-labels, OCR overlay boxes.
  features : SigLIP 2 global + spatial tokens and detection descriptors of released-frame items only.
  ID-only  : FVEI, V-FloodNet sequences, Pixabay flood photos, Norway road cameras, Iowa DOT RWIS, TSBOW
             -> item IDs, source references and labels; frames/features are rebuilt with code/data/build_external.py.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REL = Path(r"D:\highwaymm\HighwayMM_release")
OUT = Path(r"D:\highwaymm\HighwayMM_public")
FRAME_SOURCES = {"itssnap", "itslive", "hmm_pilot", "nysdot_511ny", "floodcam_nc"}
PERSON, VEHICLES = 1, {3: "car", 4: "motorcycle", 6: "bus", 8: "truck", 2: "bicycle"}
MIN_PLATE_W = 40          # px at 640-px frame width: below this no plate is readable
SCORE = 0.3


def released_items():
    lab = pd.read_csv(REL / "annotations" / "labels_v2t.csv", keep_default_na=False)
    man = pd.read_csv(REL / "manifest_ours.csv")
    excluded = set(man.loc[man.exclude, "clip_id"])
    lab = lab[~lab.item_id.isin(excluded)]
    return lab, excluded


def blur_region(img, x0, y0, x1, y1):
    import cv2
    h, w = img.shape[:2]
    x0, y0, x1, y1 = max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    roi = img[y0:y1, x0:x1]
    k = max(9, (min(x1 - x0, y1 - y0) // 3) | 1)
    img[y0:y1, x0:x1] = cv2.GaussianBlur(roi, (k, k), 0)


def run_blur(a):
    import cv2
    import torch
    import torchvision
    from torchvision.transforms.functional import to_tensor
    lab, _ = released_items()
    items = sorted(lab.loc[lab.source.isin(FRAME_SOURCES), "item_id"])
    items = items[a.shard::a.nshard]
    dev = torch.device(f"cuda:{a.gpu}")
    w = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=w, box_score_thresh=SCORE,
                                                                     min_size=640, max_size=1100).eval().to(dev)
    log = []
    print(f"shard {a.shard}: {len(items)} items", flush=True)
    for n, item in enumerate(items):
        src = REL / "frames" / item
        dst = OUT / "frames" / item
        frames = sorted(src.glob("m*.jpg"))
        if not frames:          # corrupt recording without frames: not released
            continue
        if dst.exists() and len(list(dst.glob("m*.jpg"))) == len(frames):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        imgs = [cv2.imread(str(f)) for f in frames]
        with torch.no_grad():
            dets = model([to_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).to(dev) for im in imgs])
        for f, im, d in zip(frames, imgs, dets):
            npers = nveh = 0
            for (x0, y0, x1, y1), c in zip(d["boxes"].cpu().numpy(), d["labels"].cpu().numpy()):
                if c == PERSON:
                    blur_region(im, x0, y0, x1, y1); npers += 1
                elif int(c) in VEHICLES and (x1 - x0) >= MIN_PLATE_W:
                    blur_region(im, x0, y0 + 0.55 * (y1 - y0), x1, y1); nveh += 1
            cv2.imwrite(str(dst / f.name), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            log.append((item, f.name, npers, nveh))
        if n % 500 == 0:
            print(f"shard {a.shard}: {n}/{len(items)}", flush=True)
    pd.DataFrame(log, columns=["item_id", "frame", "persons_blurred", "vehicles_blurred"]).to_csv(
        OUT / f"blur_log_shard{a.shard}.csv", index=False)
    print(f"shard {a.shard}: done", flush=True)


def filter_npz(src, dst, keep):
    z = np.load(src, allow_pickle=True)
    key = next(k for k in ("ids", "item_ids", "item_id") if k in z.files)
    ids = np.asarray(z[key]).astype(str)
    m = np.isin(ids, list(keep))
    if not m.any():
        return 0
    out = {}
    for k in z.files:
        v = z[k]
        out[k] = v[m] if (hasattr(v, "shape") and v.ndim >= 1 and len(v) == len(ids)) else v
    np.savez_compressed(dst, **out)
    return int(m.sum())


def finalize():
    lab, excluded = released_items()
    rel_frames = {i for i in lab.loc[lab.source.isin(FRAME_SOURCES), "item_id"]
                  if any((REL / "frames" / i).glob("m*.jpg"))}          # 10 corrupt recordings have no frames
    for p in (OUT / "frames").iterdir():
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    have = {p.name for p in (OUT / "frames").iterdir()} if (OUT / "frames").exists() else set()
    missing = rel_frames - have
    assert not missing, f"{len(missing)} items not blurred yet, e.g. {sorted(missing)[:3]}"
    (OUT / "annotations").mkdir(parents=True, exist_ok=True)
    (OUT / "splits").mkdir(exist_ok=True)
    (OUT / "gold").mkdir(exist_ok=True)
    (OUT / "features").mkdir(exist_ok=True)

    lab = lab.assign(frames_released=lab.item_id.isin(rel_frames))
    lab.to_csv(OUT / "annotations" / "labels.csv", index=False)
    sp = pd.read_csv(REL / "splits" / "split_v2t.csv")
    sp[~sp.item_id.isin(excluded)].to_csv(OUT / "splits" / "splits.csv", index=False)

    # manifests without local paths
    man = pd.read_csv(REL / "manifest_ours.csv")
    man[~man.exclude].drop(columns=["path", "exclude", "exclude_reason"]).to_csv(OUT / "manifest_ours.csv", index=False)
    ext = pd.read_csv(REL / "manifest_external.csv", low_memory=False)
    ext = ext.assign(frames_released=ext.item_id.isin(rel_frames),
                     original_file=ext.path.astype(str).str.replace(r"^.*?external[\\/]", "", regex=True)
                     .str.replace("\\", "/", regex=False)).drop(columns=["path"])
    ext.to_csv(OUT / "manifest_external.csv", index=False)

    # gold: anonymise annotators, keep only what the evaluation uses
    g = pd.read_csv(REL / "gold" / "gold_labels_merged.csv", keep_default_na=False)
    names = {n: f"annotator_{i + 1}" for i, n in enumerate(sorted(g.labeller.unique()))}
    g = g.assign(labeller=g.labeller.map(names))
    v = pd.read_csv(REL / "gold" / "gold_v1.csv", keep_default_na=False)
    g.merge(v[["gid", "clip_id"]], on="gid", how="left").to_csv(OUT / "gold" / "gold_labels.csv", index=False)
    v.to_csv(OUT / "gold" / "gold_selection.csv", index=False)
    for f in ("hunt_v1.csv", "hunt_labels_audit.csv"):
        shutil.copy2(REL / "gold" / f, OUT / "gold" / f)

    # OCR overlay boxes and VLM pseudo-labels (labels -> released for every item)
    o = OUT / "annotations" / "ocr"
    o.mkdir(exist_ok=True)
    for f in (REL / "annotations" / "ocr").glob("*.json"):
        if f.stem not in excluded:
            shutil.copy2(f, o / f.name)
    shutil.copytree(REL / "annotations" / "vlm_qwen25vl7b", OUT / "annotations" / "vlm_qwen25vl7b", dirs_exist_ok=True)

    # features of released-frame items only
    nfeat = {}
    for kind in ("siglip2", "siglip2_spatial", "det"):
        (OUT / "features" / kind).mkdir(parents=True, exist_ok=True)
        for f in sorted((REL / "features" / kind).glob("shard*.npz")):
            if "iowa" in f.name or "tsbow" in f.name:
                continue
            nfeat[f"{kind}/{f.name}"] = filter_npz(f, OUT / "features" / kind / f.name, rel_frames)
    shutil.copy2(REL / "features" / "text_protos_siglip2.npz", OUT / "features" / "text_protos_siglip2.npz")

    for f in (Path(__file__).resolve().parent / "release_static").glob("*.md"):
        shutil.copy2(f, OUT / f.name)

    logs = sorted(OUT.glob("blur_log_shard*.csv"))
    if logs:
        pd.concat(pd.read_csv(f) for f in logs).to_csv(OUT / "blur_log.csv", index=False)
        for f in logs:
            f.unlink()

    # checks: nothing excluded, no annotator names, no local paths
    txt = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in OUT.rglob("*")
                    if p.is_file() and p.suffix in {".csv", ".json", ".jsonl", ".md"})
    bad = [s for s in ["owner", "D:\\highwaymm", "D:/highwaymm", "image902"] if s in txt]
    assert not bad, f"forbidden strings in release: {bad}"
    leaked = [p.name for p in (OUT / "frames").iterdir() if p.name in excluded]
    assert not leaked, leaked

    # checksums
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS.txt" and "frames" not in p.parts:
            rows.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(OUT).as_posix()}")
    (OUT / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    stats = {"items_with_labels": int(len(lab)), "items_with_frames": len(rel_frames),
             "frames": sum(1 for _ in (OUT / "frames").rglob("*.jpg")),
             "by_source": lab.groupby("source").frames_released.agg(["size", "sum"]).astype(int).to_dict("index"),
             "features_rows": nfeat}
    (OUT / "release_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur", action="store_true")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    if a.blur:
        run_blur(a)
    if a.finalize:
        finalize()
