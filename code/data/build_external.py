"""Map downloaded external datasets onto the HighwayMM taxonomy -> HighwayMM_release\\manifest_external.csv

Only READS D:\\highwaymm\\external\\* (Norway parquet images are exported once to external\\norway_trafficcam\\images\\).
GT columns gt_<attr> hold taxonomy values where the source annotates them; blank = not annotated (filled by VLM pseudo-labels later,
flagged as pseudo). Columns: item_id, source, path, camera_id, sequence_id, ts, split_role, licence, redistribute, gt_*, gt_note
"""
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

EXT = Path(r"D:\highwaymm\external")
REL = Path(r"D:\highwaymm\HighwayMM_release")
ATTR = ["precip", "intensity", "surface", "visibility", "illumination", "traffic"]


def row(item_id, source, path, licence, redistribute, camera_id="", sequence_id="", ts="", split_role="trainval", note="", **gt):
    r = dict(item_id=item_id, source=source, path=str(path), camera_id=camera_id, sequence_id=sequence_id, ts=ts,
             split_role=split_role, licence=licence, redistribute=redistribute, gt_note=note)
    for a in ATTR:
        r[f"gt_{a}"] = gt.get(a, "")
    return r


def dhash(im, flip=False):
    g = im.convert("L").resize((9, 8))
    a = np.asarray(g, dtype=np.int16)
    if flip:
        a = a[:, ::-1]
    return (a[:, 1:] > a[:, :-1]).tobytes()


def fvei():
    vis = {0: "fog_dense", 1: "fog_dense", 2: "fog_moderate", 3: "haze", 4: "clear"}
    out = []
    for split in ("train", "test"):
        files = sorted((EXT / "fvei_fog" / ("extracted" if split == "train" else "extracted_test") / split).rglob("*.jpg"))
        seen, kept = {}, 0
        for f in files:
            im = Image.open(f)
            h, hf = dhash(im), dhash(im, True)
            if hf in seen and hf != h:  # horizontal-flip augmentation copy of an image we already kept
                continue
            seen[h] = f
            g, _id, v = f.stem.split("-")[:3]
            out.append(row(f"fvei_{split}_{f.stem}", "fvei_fog", f, "unstated (research use; scripts+IDs only)", False,
                           camera_id="fvei", split_role="test" if split == "test" else "trainval",
                           note=f"visibility_m={v};grade={g}", visibility=vis[int(g)]))
            kept += 1
        print("fvei", split, len(files), "->", kept)
    return out


def floodcam():
    out = []
    for f in sorted((EXT / "floodcam_nc" / "extracted" / "TrainPhotosRecoded").glob("*.jpg")):
        m = re.match(r"^(water|no_water|not_sure)_X_(\d{4}-\d\d-\d\d)_(\d\d)_(\d\d)_(\d\d)\.\d+-(\w+)$", f.stem)
        if not m:
            print("floodcam skip", f.name)
            continue
        lab, d, hh, mm, ss, cam = m.groups()
        gt = {"surface": "water"} if lab == "water" else {}
        out.append(row(f"floodcam_{f.stem}", "floodcam_nc", f, "MIT repo; NCDOT public camera imagery", True,
                       camera_id=f"ncdot_{cam}", sequence_id=f"ncdot_{cam}", ts=f"{d}T{hh}:{mm}:{ss}",
                       note=f"floodcam_label={lab}", **gt))
    return out


def nysdot():
    xl = EXT / "nysdot_511ny" / "extracted" / "NYSDOT_quantitative_content_analysis" / "NYSDOT_ICR_Example.xlsx"
    imgs = {p.name: p for p in (EXT / "nysdot_511ny" / "extracted").rglob("*.jpg") if "__MACOSX" not in str(p)}
    lab = {}
    for s in ("trial1_labels", "trial2_labels", "trial3_labels", "trial4_labels"):
        d = pd.read_excel(xl, sheet_name=s)
        cols = [c for c in d.columns if str(c).startswith("Label")]
        for _, r in d.iterrows():
            if not isinstance(r.get("Img Name"), str):
                continue
            v = pd.to_numeric(r[cols], errors="coerce").dropna().astype(int)
            if len(v):
                lab[r["Img Name"].strip()] = (int(v.mode().iloc[0]), float((v == v.mode().iloc[0]).mean()))
    m = {1: dict(precip="snow", intensity="heavy", surface="snow"), 2: dict(surface="snow"), 3: dict(surface="wet"),
         4: dict(surface="dry"), 5: dict(visibility="fog_dense")}
    out = []
    for name, (c, agree) in lab.items():
        if c == 6 or name not in imgs:
            continue
        mt = re.search(r"_(\d+)_(\d{4}-\d\d-\d\d)-(\d\d)_(\d\d)_(\d\d)$", Path(name).stem)
        cam = mt.group(1) if mt else "unk"
        ts = f"{mt.group(2)}T{mt.group(3)}:{mt.group(4)}:{mt.group(5)}" if mt else ""
        out.append(row(f"nysdot_{Path(name).stem}", "nysdot_511ny", imgs[name], "CC-BY-4.0", True,
                       camera_id=f"nysdot_{cam}", sequence_id=f"nysdot_{cam}", ts=ts, split_role="test",
                       note=f"nysdot_class={c};agree={agree:.2f}", **m[c]))
    print("nysdot", len(out))
    return out


def norway():
    tmap = {0: "congested", 1: "free", 2: "dense", 3: "free"}  # high / low / medium / no traffic
    od = EXT / "norway_trafficcam" / "images"
    od.mkdir(exist_ok=True)
    out = []
    for pq in sorted((EXT / "norway_trafficcam").glob("*.parquet")):
        d = pd.read_parquet(pq)
        for i, r in d.iterrows():
            name = f"{pq.stem}_{i:05d}.jpg"
            p = od / name
            if not p.exists():
                Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p, quality=92)
            out.append(row(f"norway_{pq.stem}_{i:05d}", "norway_trafficcam", p, "unstated (HF; Statens vegvesen webcams?)", False,
                           camera_id="norway", split_role="test" if "validation" in pq.stem else "trainval",
                           note=f"traffic_label={r['label']}", traffic=tmap[int(r["label"])]))
    print("norway", len(out))
    return out


def vfloodnet():
    out = []
    base = EXT / "vfloodnet" / "extracted"
    # fixed-camera flood / storm timelapses: subsample (Harvey 1-min frames -> every 20 min; Boston every 30 min)
    for seq, step in [("houston_buffalo_20170825_20170901", 20), ("boston_harbor_20190119_20190123", 4),
                      ("boston_harbor_20220129_20220131", 4), ("boston_harbor_20200617_20200619", 4), ("LSU_20200624", 1),
                      ("LSU_20200526", 1)]:
        fs = sorted((base / "test_videos" / seq).glob("*.jpg"))[::step]
        for f in fs:
            out.append(row(f"vflood_{seq}_{f.stem}", "vfloodnet_seq", f, "unstated (research; scripts+IDs only)", False,
                           camera_id=seq.rsplit("_", 2)[0], sequence_id=seq, ts=f.stem, split_role="test_sequence",
                           note="fixed-camera storm/flood timelapse; labels = VLM pseudo + water seg"))
    for f in sorted((base / "train_images" / "JPEGImages" / "flooding_pixabay").glob("*")):
        out.append(row(f"vflood_pixabay_{f.stem}", "vfloodnet_pixabay", f, "Pixabay licence", True, camera_id="pixabay",
                       note="flooded street photo", surface="water"))
    print("vfloodnet", len(out))
    return out


def iowa():
    """Iowa DOT RWIS roadside cameras (IEM archive), summers 2023-2025; labels from co-located pavement sensors
    (surface), nearest ASOS METAR (precip/intensity/visibility) and solar elevation (illumination). See iowa_select.py."""
    base = EXT / "iowa_rwis"
    if not (base / "fetched.csv").exists():
        return []
    f = pd.read_csv(base / "fetched.csv", dtype=str).drop_duplicates("file")
    s = pd.read_csv(base / "selection.csv", dtype=str)
    m = f.merge(s, left_on=["query_utc", "cid"], right_on=["utc", "cid"], how="left")
    out = []
    for r in m.itertuples():
        p = base / r.file
        if not p.exists():
            continue
        gt = {a: getattr(r, a) for a in ("precip", "intensity", "surface", "visibility", "illumination")
              if isinstance(getattr(r, a, None), str) and getattr(r, a)}
        out.append(row(f"iowa_{Path(r.file).stem}", "iowa_rwis", p, "Iowa DOT imagery via IEM archive (scripts+IDs only)", False,
                       camera_id=f"iowa_{r.cid}", sequence_id=f"iowa_{r.view}", ts=r.img_utc,
                       split_role="trainval_camdisjoint",
                       note=f"sensor;cat={r.cat};wx={r.wx};vsby_mi={r.vsby_mi};rwis={r.rwis};asos={r.asos}", **gt))
    print("iowa", len(out))
    return out


def tsbow():
    """TSBOW (Suwon city CCTV, daytime): per-video human weather tag -> frame labels. TEST ONLY (split_role test)."""
    base = EXT / "tsbow"
    if not (base / "frames").exists():
        return []
    meta = pd.read_csv(base / "metadata" / "TSBOW_info.csv")
    tag = {r.VIDEO_ID.replace("#", "_"): r.WEATHER for r in meta.itertuples()}
    lab = {"n": dict(precip="none", intensity="none"), "r": dict(precip="rain"), "s": dict(precip="snow"),
           "h": dict(precip="none", intensity="none", visibility="haze")}
    out = []
    for f in sorted((base / "frames").glob("*.jpg")):
        vid = f.stem.rsplit("_", 1)[0]
        if vid not in tag:
            continue
        cam = "tsbow_" + vid.split("_")[1]  # suwon_<camera>_<seq>_<part>
        out.append(row(f"tsbow_{f.stem}", "tsbow", f, "CC BY-NC-ND 4.0 (research only; IDs+script only)", False,
                       camera_id=cam, sequence_id=f"tsbow_{vid}", split_role="test",
                       note=f"tsbow_weather={tag[vid]};video={vid}", illumination="day", **lab[tag[vid]]))
    print("tsbow", len(out))
    return out


def main():
    rows = fvei() + floodcam() + nysdot() + norway() + vfloodnet() + iowa() + tsbow()
    df = pd.DataFrame(rows)
    df.to_csv(REL / "manifest_external.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["source", "split_role"]).size())
    for a in ATTR:
        print(a, df[f"gt_{a}"].replace("", np.nan).dropna().value_counts().to_dict())


if __name__ == "__main__":
    main()
