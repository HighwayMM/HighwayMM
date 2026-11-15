"""Assemble the unified label table -> HighwayMM_release\\annotations\\labels_v1.csv and splits\\split_v1.csv

Label priority per attribute: human GT (external, weight 1.0) > physics (illumination from solar elevation, 1.0)
> VLM pseudo-label (Qwen2.5-VL-7B; weight high .8 / medium .5 / low .2). Every label keeps its source (lab_src_<attr>).
Weak official warnings: KEC/ITS alert-banner OCR (overlay is masked from model input) -> alert_tags.
Splits (v1): ours = camera-disjoint (hash(camera_id) % 10: 0-6 train, 7 val, 8-9 test);
external = source-provided test sets, FloodCam camera-held-out (Buxton+Canal test), NYSDOT + storm sequences test-only.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, parse_alert, warning_level  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
VER = sys.argv[1] if len(sys.argv) > 1 else "v1"  # v2 (2026-11-06) adds Iowa RWIS summer cameras
CONF_W = {"high": 0.8, "medium": 0.5, "low": 0.2}


def solar_elev(lat, lon, ts_kst):
    t = pd.Timestamp(ts_kst) - pd.Timedelta(hours=9)
    doy = t.dayofyear
    hr = t.hour + t.minute / 60
    g = 2 * math.pi / 365 * (doy - 1 + (hr - 12) / 24)
    decl = 0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g) + 0.000907 * math.sin(2 * g)
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g) - 0.040849 * math.sin(2 * g))
    tst = hr * 60 + eqt + 4 * lon
    ha = math.radians(tst / 4 - 180)
    la = math.radians(lat)
    cz = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return math.degrees(math.asin(max(-1, min(1, cz))))


def illum_from_elev(e):
    return "day" if e > 6 else ("twilight" if e > -6 else "night")


def hsplit(key, train=7, val=1):
    h = int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 10
    return "train" if h < train else ("val" if h < train + val else "test")


def load_vlm():
    d = {}
    for f in (REL / "annotations" / "vlm_qwen25vl7b").glob("shard*.jsonl"):
        for l in open(f, encoding="utf-8"):
            j = json.loads(l)
            if j.get("ok"):
                d[j["item_id"]] = j
    return d


def main():
    ours = pd.read_csv(REL / "manifest_ours.csv")
    ours = ours[~ours.exclude].copy()
    ext = pd.read_csv(REL / "manifest_external.csv", keep_default_na=False)
    vlm = load_vlm()
    print("vlm labels:", len(vlm))
    rows = []
    for r in ours.itertuples():
        lab, src = {}, {}
        ocr = REL / "annotations" / "ocr" / f"{r.clip_id}.json"
        top = json.loads(ocr.read_text(encoding="utf-8")).get("top_text", "") if ocr.exists() else ""
        if r.source != "hmm_pilot" and not pd.isna(r.lat):
            lab["illumination"], src["illumination"] = illum_from_elev(solar_elev(r.lat, r.lon, r.ts_kst)), "physics"
        rows.append(dict(item_id=r.clip_id, domain="ours", source=r.source, camera_id=r.camera_id,
                         sequence_id=r.sequence_id if isinstance(r.sequence_id, str) else "", ts=r.ts_kst, season=r.season,
                         split=hsplit(r.camera_id), alert_text=top, alert_tags=";".join(parse_alert(top)), _lab=lab, _src=src, note=""))
    hmm = pd.read_csv(Path(r"D:\highwaymm\analysis\tables\master.csv"))
    hmm_ill = {"hmm_" + Path(f).stem.replace(" ", "_"): v for f, v in zip(hmm.file, hmm.illum_time)}
    for r in rows:
        if r["source"] == "hmm_pilot" and isinstance(hmm_ill.get(r["item_id"]), str):
            r["_lab"]["illumination"], r["_src"]["illumination"] = {"day": "day", "night": "night"}.get(hmm_ill[r["item_id"]], "twilight"), "physics"
    floodcam_test = {"ncdot_Buxton", "ncdot_Canal"}
    for r in ext.itertuples():
        lab = {a: getattr(r, f"gt_{a}") for a in ATTRS if getattr(r, f"gt_{a}")}
        src = {a: ("sensor" if r.source == "iowa_rwis" else "gt") for a in lab}
        role = r.split_role
        if role in ("test", "test_sequence"):
            sp = "test"
        elif role == "trainval_camdisjoint":  # Iowa RWIS: whole cameras held out (60/10/30)
            sp = hsplit(r.camera_id, 6, 1)
        elif r.source == "floodcam_nc":
            sp = "test" if r.camera_id in floodcam_test else hsplit(r.item_id, 8, 2)
        else:
            sp = hsplit(r.item_id, 9, 1) if r.source != "vfloodnet_pixabay" else hsplit(r.item_id, 8, 1)
        rows.append(dict(item_id=r.item_id, domain="external", source=r.source, camera_id=r.camera_id, sequence_id=r.sequence_id,
                         ts=r.ts, season="", split=sp, alert_text="", alert_tags="", _lab=lab, _src=src, note=r.gt_note))
    out = []
    for r in rows:
        lab, src = r.pop("_lab"), r.pop("_src")
        v = vlm.get(r["item_id"])
        w = {a: 1.0 for a in lab}
        trans = False
        if v:
            cw = CONF_W.get(v.get("confidence"), 0.4)
            trans = bool(v.get("transitional"))
            for a, val in v["labels"].items():
                if val is None or a in lab:
                    continue
                if r["source"] == "floodcam_nc" and a == "surface" and ("no_water" in r["note"] or "not_sure" in r["note"]):
                    if val == "water" or "not_sure" in r["note"]:
                        continue  # contradicts / uncertain w.r.t. human label
                lab[a], src[a], w[a] = val, "vlm", cw
        if "not_sure" in r["note"]:
            trans = True
        for a in ATTRS:
            r[f"lab_{a}"] = lab.get(a, "")
            r[f"src_{a}"] = src.get(a, "")
            r[f"w_{a}"] = round(w.get(a, 0.0), 2)
        # partial (negative) human labels: FloodCam "no_water" = surface is NOT water, even when its exact state is unknown
        r["neg_surface"] = "water" if (r["source"] == "floodcam_nc" and "no_water" in r["note"]) else ""
        r["transitional"] = int(trans)
        r["has_vlm"] = int(v is not None)
        r["warn_level"] = warning_level(lab, trans) if lab else -1
        if r["source"] == "tsbow":
            r["warn_level"] = -1  # per-video weather tag only (no intensity / surface): not a warning-level label
        out.append(r)
    df = pd.DataFrame(out)
    (REL / "annotations").mkdir(exist_ok=True)
    (REL / "splits").mkdir(exist_ok=True)
    df.to_csv(REL / "annotations" / f"labels_{VER}.csv", index=False, encoding="utf-8-sig")
    df[["item_id", "domain", "source", "camera_id", "split"]].to_csv(REL / "splits" / f"split_{VER}.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["source", "split"]).size().unstack(fill_value=0))
    for a in ATTRS:
        print(a, df[f"lab_{a}"].replace("", np.nan).dropna().value_counts().to_dict())
    print("warn", df.warn_level.value_counts().to_dict())
    print("alerts", df.alert_tags.replace("", np.nan).dropna().value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()

