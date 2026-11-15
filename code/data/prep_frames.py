"""Decode clips -> K uniformly spaced frames, OCR overlay text, build per-clip overlay mask, save masked + raw frames.

Reads originals only. Output (per clip) under D:\\highwaymm\\HighwayMM_release\\:
  frames\\<clip_id>\\m{k}.jpg   overlay-masked frames (model input)
  frames\\<clip_id>\\r{k}.jpg   unmasked frames (for the masked-vs-unmasked shortcut ablation only)
  annotations\\ocr\\<clip_id>.json   OCR boxes/text (alert banner -> weak hazard labels), mask boxes
Run sharded via the job runner:  prep_frames.py --shard i --nshards n
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

ROOT = Path(r"D:\highwaymm")
REL = ROOT / "HighwayMM_release"
K = 8
OUT_W = 640


def read_frames(path, k=K):
    if str(path).lower().endswith((".jpg", ".jpeg", ".png")):
        im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        return [im] if im is not None else []
    c = av.open(path)
    s = c.streams.video[0]
    n = s.frames or int(float(c.duration or 0) / 1e6 * float(s.average_rate or 30))
    frames = []
    for fr in c.decode(s):
        frames.append(fr)  # 10-60 s clips; long ones are subsampled below
        if len(frames) > 2400:
            break
    c.close()
    if not frames:
        return []
    idx = np.linspace(0, len(frames) - 1, k + 2)[1:-1].round().astype(int)

    def to_bgr(fr):
        try:
            return fr.to_ndarray(format="bgr24")
        except Exception:  # e.g. yuvj420p with ycgco colorspace unsupported by swscale -> convert planes ourselves
            return cv2.cvtColor(fr.to_ndarray(), cv2.COLOR_YUV2BGR_I420)  # yuv(j)420p planes stacked (H*3/2, W)
    return [to_bgr(frames[i]) for i in idx]


def crop_content(img, box320):
    if not isinstance(box320, str) or not box320.startswith("["):
        return img
    x0, y0, x1, y1 = json.loads(box320)
    h, w = img.shape[:2]
    sx, sy = w / 320, h / 180
    return img[int(y0 * sy):int(y1 * sy), int(x0 * sx):int(x1 * sx)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--manifest", default="manifest_ours.csv")
    a = ap.parse_args()
    import easyocr
    reader = easyocr.Reader(["ko", "en"], gpu=True, model_storage_directory=str(ROOT / "analysis" / ".easyocr"), verbose=False)
    man = pd.read_csv(REL / a.manifest)
    if "exclude" in man:
        man = man[~man.exclude].reset_index(drop=True)
    if "clip_id" not in man:
        man = man.rename(columns={"item_id": "clip_id"})
    for c in ("content_box",):
        if c not in man:
            man[c] = ""
    todo = man.iloc[a.shard::a.nshards]
    (REL / "annotations" / "ocr").mkdir(parents=True, exist_ok=True)
    done = fail = 0
    for _, r in todo.iterrows():
        cid = r.clip_id
        od = REL / "frames" / cid
        oj = REL / "annotations" / "ocr" / f"{cid}.json"
        if oj.exists() and ((od / f"m{K - 1}.jpg").exists() or (od / "m0.jpg").exists() and not str(r.path).endswith(".mp4")):
            continue
        try:
            fr = read_frames(r.path)
            if not fr:
                raise RuntimeError("no frames")
            fr = [crop_content(f, r.content_box) for f in fr]
            h, w = fr[0].shape[:2]
            scale = OUT_W / w
            fr = [cv2.resize(f, (OUT_W, int(round(h * scale))), interpolation=cv2.INTER_AREA) for f in fr]
            H, W = fr[0].shape[:2]
            # OCR on 2 frames (overlays are static; union catches flicker / low contrast), at 2x for small text
            boxes, texts = [], []
            for f in ([fr[0]] if len(fr) == 1 else [fr[0], fr[len(fr) // 2]]):
                big = cv2.resize(f, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
                for bb, txt, conf in reader.readtext(big, text_threshold=0.5, low_text=0.3):
                    pts = np.array(bb) / 2
                    x0, y0 = pts.min(0)
                    x1, y1 = pts.max(0)
                    boxes.append([float(x0), float(y0), float(x1), float(y1)])
                    texts.append({"text": txt, "conf": float(conf), "box": [float(x0), float(y0), float(x1), float(y1)],
                                  "y_rel": float((y0 + y1) / 2 / H)})
            mask = np.zeros((H, W), np.uint8)
            for x0, y0, x1, y1 in boxes:
                pad = 4 + 0.25 * (y1 - y0)
                cv2.rectangle(mask, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), 255, -1)
            if r.source in ("itssnap", "itslive") or (r.source == "hmm_pilot" and texts):
                mask[: int(0.09 * H)] = 255  # top band: branch office + alert banner (always masked on ITS/KEC layouts)
            od.mkdir(parents=True, exist_ok=True)
            for k, f in enumerate(fr):
                cv2.imwrite(str(od / f"r{k}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
                m = f.copy()
                m[mask > 0] = (127, 127, 127)
                cv2.imwrite(str(od / f"m{k}.jpg"), m, [cv2.IMWRITE_JPEG_QUALITY, 90])
            top = " ".join(t["text"] for t in texts if t["y_rel"] < 0.12)
            oj.write_text(json.dumps({"clip_id": cid, "W": W, "H": H, "mask_frac": float((mask > 0).mean()),
                                      "top_text": top, "ocr": texts}, ensure_ascii=False), encoding="utf-8")
            done += 1
            if done % 200 == 0:
                print(f"[shard {a.shard}] {done}/{len(todo)}", flush=True)
        except Exception as e:  # noqa
            fail += 1
            print(f"FAIL {cid}: {e}", file=sys.stderr)
            traceback.print_exc()
    print(f"[shard {a.shard}] done {done} fail {fail}")


if __name__ == "__main__":
    main()
