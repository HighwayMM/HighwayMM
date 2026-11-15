"""Build the clip manifest for our own footage (originals are only READ, never modified).

Output: D:\\highwaymm\\HighwayMM_release\\manifest_ours.csv
  clip_id, source, path, camera_id, ts_kst, date, hour, month, season, lon, lat, road_class, sequence_id, exclude, exclude_reason, content_box
"""
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\highwaymm")
OUT = ROOT / "HighwayMM_release"
PAT = re.compile(r"^(ex|its)_(\d+)p(\d+)_(\d+)p(\d+)_(\d{8}T\d{6})KST$")


def season(m):
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer"}.get(m, "autumn")


def its_rows(folder, source):
    rows = []
    for p in sorted(Path(folder).rglob("*.mp4")):
        m = PAT.match(p.stem)
        if not m:
            print("skip", p, file=sys.stderr)
            continue
        kind, lo1, lo2, la1, la2, ts = m.groups()
        t = datetime.strptime(ts, "%Y%m%dT%H%M%S")
        cam = f"{kind}_{lo1}p{lo2}_{la1}p{la2}"
        rows.append(dict(clip_id=f"{source}_{p.stem}", source=source, path=str(p), camera_id=cam,
                         ts_kst=t.isoformat(), date=t.date().isoformat(), hour=t.hour, month=t.month, season=season(t.month),
                         lon=float(f"{lo1}.{lo2}"), lat=float(f"{la1}.{la2}"),
                         road_class="expressway" if kind == "ex" else "national",
                         sequence_id=cam if source == "itslive" else "", exclude=False, exclude_reason="", content_box=""))
    return rows


def hmm_rows():
    m = pd.read_csv(ROOT / "analysis" / "tables" / "master.csv")
    rows = []
    for _, r in m.iterrows():
        flags = str(r.get("quality_flags", "") or "")
        excl = [f for f in ("non_traffic_content", "frozen", "no_overlay_unknown_source", "black_frames") if f in flags]
        ts = pd.to_datetime(r["creation_time_kst"])
        cam = r["camera_key_ocr"] if isinstance(r["camera_key_ocr"], str) and r["camera_key_ocr"] else f"hmmview{int(r['view_cluster'])}"
        rows.append(dict(clip_id="hmm_" + Path(r["file"]).stem.replace(" ", "_"), source="hmm_pilot",
                         path=str(ROOT / "HighwayMM" / r["file"]), camera_id=cam, ts_kst=ts.isoformat(),
                         date=ts.date().isoformat(), hour=ts.hour, month=ts.month, season=season(ts.month), lon="", lat="",
                         road_class={"expressway": "expressway", "national": "national", "urban": "urban"}.get(r["source_type"], "unknown"),
                         sequence_id="", exclude=bool(excl), exclude_reason=";".join(excl), content_box=r["content_box_320"]))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(hmm_rows() + its_rows(ROOT / "raw_sample", "itssnap") + its_rows(ROOT / "raw_live", "itslive"))
    df.to_csv(OUT / "manifest_ours.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["source", "exclude"]).size())
    print("cameras:", df.groupby("source").camera_id.nunique().to_dict())
    print(df.groupby(["source", "season"]).size())


if __name__ == "__main__":
    main()
