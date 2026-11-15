"""Sample N evenly spaced frames per TSBOW video -> external/tsbow/frames/<video>_<k>.jpg (single-image items).
TSBOW (SKKU AutoLab, AAAI-26, CC BY-NC-ND 4.0): Suwon city CCTV, daytime, one human weather tag per video
(n normal / h haze / r rain / s snow). Used as a TEST-ONLY Korean-CCTV source (no training), never redistributed."""
from pathlib import Path

import cv2

BASE = Path(r"D:\highwaymm\external\tsbow")
N = 10
out = BASE / "frames"
out.mkdir(exist_ok=True)
vids = {}
for p in BASE.rglob("*.mp4"):
    vids.setdefault(p.stem, p)
n = 0
for vid, p in sorted(vids.items()):
    cap = cv2.VideoCapture(str(p))
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k in range(N):
        f = out / f"{vid.replace('#', '_')}_{k}.jpg"
        if f.exists():
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, int((k + 0.5) * tot / N))
        ok, fr = cap.read()
        if ok:
            cv2.imwrite(str(f), fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
    cap.release()
print(len(vids), "videos,", n, "new frames")
