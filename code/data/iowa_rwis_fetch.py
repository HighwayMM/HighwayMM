"""Fetch archived Iowa DOT RWIS camera images (Iowa Environmental Mesonet archive).

  python iowa_rwis_fetch.py cameras                      -> <out>/cameras.json (cid, lat, lon, n views)
  python iowa_rwis_fetch.py fetch --selection sel.csv    -> images for rows (utc, cid[, view]); one view per row,
                                                            chosen deterministically if 'view' is empty
Writes <out>/images/<YYYYMMDD>/<view>_<YYYYmmddHHMM>.jpg and appends to <out>/fetched.csv. Throttled; resumable.
"""
import argparse, csv, hashlib, json, time, urllib.request
from collections import defaultdict
from pathlib import Path

API = "https://mesonet.agron.iastate.edu/api/1/idot_rwiscam.geojson?valid={}:00Z"
UA = {"User-Agent": "academic-dataset-fetch (road-weather research)"}
OUT = Path("D:/highwaymm/external/iowa_rwis")


def get(url, tries=3):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * (k + 1))


def views_at(utc):
    gj = json.loads(get(API.format(utc)))
    out = {}
    for f in gj.get("features", []):
        p = f["properties"]
        if not f.get("geometry"):
            continue
        urls = [p[f"imgurl{k}"] for k in range(10) if p.get(f"imgurl{k}")]
        out[p["cid"]] = dict(lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1], urls=urls)
    return out


def cameras():
    cams = {}
    for t in ["2023-07-15T18:00", "2024-07-15T18:00", "2025-07-15T18:00"]:
        for cid, v in views_at(t).items():
            cams.setdefault(cid, dict(cid=cid, lat=v["lat"], lon=v["lon"], n_views=0))
            cams[cid]["n_views"] = max(cams[cid]["n_views"], len(v["urls"]))
        time.sleep(1)
    cams = [c for c in cams.values() if c["n_views"] > 0 and not c["cid"].endswith("-000")]
    (OUT / "cameras.json").write_text(json.dumps(cams, indent=1))
    print(len(cams), "cameras")


def fetch(selection, sleep):
    rows = list(csv.DictReader(open(selection, encoding="utf-8")))
    by_t = defaultdict(list)
    for r in rows:
        by_t[r["utc"]].append(r)
    log = OUT / "fetched.csv"
    done = set()
    if log.exists():
        done = {(r["query_utc"], r["cid"]) for r in csv.DictReader(open(log, encoding="utf-8"))}
    new = not log.exists()
    fl = open(log, "a", newline="", encoding="utf-8")
    w = csv.writer(fl)
    if new:
        w.writerow(["file", "view", "cid", "query_utc", "img_utc", "tag", "url"])
    n = 0
    for i, (t, rs) in enumerate(sorted(by_t.items())):
        rs = [r for r in rs if (t, r["cid"]) not in done]
        if not rs:
            continue
        try:
            va = views_at(t)
        except Exception as e:
            print("api fail", t, e, flush=True)
            continue
        for r in rs:
            urls = va.get(r["cid"], {}).get("urls", [])
            if r.get("view"):
                urls = [u for u in urls if f"/{r['view']}_" in u]
            if not urls:
                continue
            url = urls[int(hashlib.md5((r["cid"] + t).encode()).hexdigest(), 16) % len(urls)] if not r.get("view") else urls[0]
            stem = url.rsplit("/", 1)[1][:-4]
            view, ts = stem.rsplit("_", 1)
            rel = f"images/{ts[:8]}/{stem}.jpg"
            dst = OUT / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                if not dst.exists():
                    dst.write_bytes(get(url))
                    time.sleep(sleep)
            except Exception as e:
                print("img fail", url, e, flush=True)
                continue
            w.writerow([rel, view, r["cid"], t, ts, r.get("cat", r.get("tag", "")), url])
            n += 1
        fl.flush()
        time.sleep(sleep)
        if i % 50 == 0:
            print(f"[{i + 1}/{len(by_t)}] {t} images {n}", flush=True)
    fl.close()
    print("done", n)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cameras", "fetch"])
    ap.add_argument("--selection")
    ap.add_argument("--sleep", type=float, default=0.3)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cameras() if a.cmd == "cameras" else fetch(a.selection, a.sleep)
