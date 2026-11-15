"""TSBOW coverage check: first frame of every video -> OCR the burned-in timestamp -> month / season / hour table.
Reads external/tsbow (downloaded via download_TSBOW-style hf_hub_download). Writes external/tsbow/inspect.csv and
a contact sheet of first frames per weather tag for a visual check.
"""
import re
import sys
from pathlib import Path

import cv2
import easyocr
import pandas as pd

BASE = Path(r"D:\highwaymm\external\tsbow")
meta = pd.read_csv(BASE / "metadata" / "TSBOW_info.csv")
vids = {p.stem: p for p in BASE.rglob("*.mp4")}
print("videos found", len(vids), "metadata rows", len(meta))
rd = easyocr.Reader(["en", "ko"], gpu=True)
rows = []
(BASE / "first_frames").mkdir(exist_ok=True)
for r in meta.itertuples():
    p = vids.get(r.VIDEO_ID)
    rec = dict(video_id=r.VIDEO_ID, weather=r.WEATHER, daytime=r.DAYTIME, scenario=r.SCENARIO, found=p is not None)
    if p is not None:
        cap = cv2.VideoCapture(str(p))
        ok, fr = cap.read()
        rec.update(w=int(cap.get(3)), h=int(cap.get(4)), fps=cap.get(5), nframes=int(cap.get(7)))
        cap.release()
        if ok:
            cv2.imwrite(str(BASE / "first_frames" / f"{r.VIDEO_ID.replace('#', '_')}.jpg"), fr)
            txt = " ".join(t for _, t, c in rd.readtext(fr) if c > 0.2)
            rec["ocr"] = txt
            m = re.search(r"(20\d\d)[-./ ]?(\d{1,2})[-./ ]?(\d{1,2})", txt)
            if m:
                rec.update(year=int(m.group(1)), month=int(m.group(2)), day=int(m.group(3)))
            h = re.search(r"\b(\d{1,2}):(\d\d)(?::\d\d)?\b", txt)
            if h:
                rec["hour"] = int(h.group(1))
    rows.append(rec)
    print(rec.get("video_id"), rec.get("weather"), rec.get("year"), rec.get("month"), rec.get("hour"), flush=True)
df = pd.DataFrame(rows)
df.to_csv(BASE / "inspect.csv", index=False, encoding="utf-8-sig")
season = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring", 6: "summer", 7: "summer", 8: "summer",
          9: "autumn", 10: "autumn", 11: "autumn"}
df["season"] = df.month.map(season)
print(df.groupby(["weather", "season"], dropna=False).size().unstack(fill_value=0))
print(df.groupby(["year", "month"], dropna=False).size())

