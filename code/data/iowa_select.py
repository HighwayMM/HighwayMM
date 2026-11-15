"""Select labelled timestamps for Iowa DOT RWIS cameras from co-located sensors.

Labels (provenance "sensor"):
  surface      <- RWIS pavement sensors at the camera site (Dry -> dry, Wet -> wet; moisture/other ignored)
  precip/int.  <- nearest ASOS METAR (<=20 km, +-20 min): RA/TSRA/DZ with -/none/+ -> light/moderate/heavy
  visibility   <- same ASOS vsby (miles): <=0.062 fog_dense, <=0.124 fog_moderate, <=0.31 haze, else clear
                  (same metre thresholds as the FVEI mapping: 100 / 200 / 500 m)
  illumination <- solar elevation (physics)
Output: external/iowa_rwis/selection.csv (one row per camera-time) used by iowa_rwis_fetch.py --selection.
"""
import glob, json, math, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_labels import solar_elev, illum_from_elev  # noqa: E402

BASE = Path("D:/highwaymm/external/iowa_rwis")
RNG = np.random.default_rng(0)
QUOTA = {"rain_heavy": 900, "rain_moderate": 700, "rain_light": 600, "fog": 700, "wet_norain": 500,
         "dry_day": 600, "dry_night": 500, "dry_twilight": 300,
         "wet_surface_only": 500, "dry_surface_only": 300}


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def load_rwis():
    fs = sorted(glob.glob(str(BASE / "sensors" / "atmos_*.csv")))
    d = pd.concat([pd.read_csv(f, na_values=["M", "NA", ""], low_memory=False) for f in fs])
    d["t"] = pd.to_datetime(d.obtime)
    txt = d[[c for c in d.columns if c.endswith("_text")]]
    nd, nw = (txt == "Dry").sum(axis=1), (txt == "Wet").sum(axis=1)
    d["surface"] = np.where((nw > 0) & (nw >= nd), "wet", np.where((nd > 0) & (nw == 0) & ~txt.isin(["Trace Moisture", "Moist"]).any(axis=1), "dry", ""))
    return d[["station", "longitude", "latitude", "t", "surface"]]


def load_asos():
    fs = sorted(glob.glob(str(BASE / "asos" / "asos_*.csv")))
    d = pd.concat([pd.read_csv(f, low_memory=False) for f in fs])
    d["t"] = pd.to_datetime(d.valid)
    d["wxcodes"] = d.wxcodes.fillna("").astype(str)
    return d


def precip_from_wx(wx, p01i):
    toks = wx.split()
    rain = [x for x in toks if re.search(r"(RA|DZ)", x) and "SN" not in x]
    if any("SN" in x for x in toks):
        return None, None
    if rain:
        x = rain[0]
        inten = "heavy" if x.startswith("+") else ("light" if x.startswith("-") or "DZ" in x else "moderate")
        return "rain", inten
    if any("TS" in x for x in toks) or any(x.startswith("UP") for x in toks):
        return None, None  # thunder / unknown precip: leave unlabelled
    if pd.notna(p01i) and p01i > 0.001:
        return None, None
    return "none", "none"


def vis_from_mi(v):
    if pd.isna(v):
        return None
    return "fog_dense" if v <= 0.062 else "fog_moderate" if v <= 0.124 else "haze" if v <= 0.31 else "clear"


def main():
    cams = pd.read_json(BASE / "cameras.json")
    rw, asos = load_rwis(), load_asos()
    st_rw = rw.groupby("station")[["latitude", "longitude"]].first()
    st_as = asos.groupby("station")[["lat", "lon"]].first()
    rows = []
    for c in cams.itertuples():
        drw = hav(c.lat, c.lon, st_rw.latitude.values, st_rw.longitude.values)
        das = hav(c.lat, c.lon, st_as.lat.values, st_as.lon.values)
        if drw.min() > 5:
            continue
        near = das.min() <= 20  # precip / visibility only from an ASOS within 20 km
        srw, sas = st_rw.index[drw.argmin()], st_as.index[das.argmin()]
        r = rw[(rw.station == srw) & (rw.surface != "")][["t", "surface"]].sort_values("t")
        a = asos[asos.station == sas][["t", "wxcodes", "vsby", "p01i"]].sort_values("t")
        if not near:
            a = a.iloc[0:0]
        if r.empty:
            continue
        grid = pd.DataFrame({"t": pd.date_range(r.t.min().floor("h"), r.t.max(), freq="10min")})
        grid = grid[grid.t.dt.normalize().isin(r.t.dt.normalize().unique())]
        m = pd.merge_asof(grid, r, on="t", direction="nearest", tolerance=pd.Timedelta("10min"))
        m = pd.merge_asof(m, a, on="t", direction="nearest", tolerance=pd.Timedelta("20min"))
        m = m.dropna(subset=["surface"])
        m["wxcodes"] = m.wxcodes.astype(object).where(m.wxcodes.notna(), None)
        for x in m.itertuples():
            pr, it = precip_from_wx(x.wxcodes, x.p01i) if x.wxcodes is not None else (None, None)
            rows.append(dict(cid=c.cid, lat=c.lat, lon=c.lon, rwis=srw, asos=sas, d_rwis_km=round(drw.min(), 2),
                             d_asos_km=round(das.min(), 1), utc=x.t, surface=x.surface, precip=pr, intensity=it,
                             visibility=vis_from_mi(x.vsby), vsby_mi=x.vsby, wx=x.wxcodes))
        print(c.cid, srw, sas, len(m), flush=True)
    df = pd.DataFrame(rows)
    df["illumination"] = [illum_from_elev(solar_elev(la, lo, t + pd.Timedelta(hours=9))) for la, lo, t in zip(df.lat, df.lon, df.utc)]
    vis_fog = df.visibility.isin(["fog_dense", "fog_moderate", "haze"])
    cat = np.full(len(df), "", dtype=object)
    cat[(df.precip == "none") & (df.surface == "dry") & (df.visibility == "clear") & (df.illumination == "twilight")] = "dry_twilight"
    cat[(df.precip == "none") & (df.surface == "dry") & (df.visibility == "clear") & (df.illumination == "night")] = "dry_night"
    cat[(df.precip == "none") & (df.surface == "dry") & (df.visibility == "clear") & (df.illumination == "day")] = "dry_day"
    cat[(df.precip == "none") & (df.surface == "wet") & ~vis_fog] = "wet_norain"
    cat[vis_fog & (df.precip != "rain")] = "fog"
    for k in ("light", "moderate", "heavy"):
        cat[(df.precip == "rain") & (df.intensity == k)] = f"rain_{k}"
    cat[(cat == "") & df.precip.isna() & (df.surface == "wet")] = "wet_surface_only"
    cat[(cat == "") & df.precip.isna() & (df.surface == "dry")] = "dry_surface_only"
    df["cat"] = cat
    print(df.cat.value_counts().to_dict())
    sel = []
    for k, q in QUOTA.items():
        g = df[df.cat == k].sample(frac=1, random_state=0)
        g = g.assign(hour=g.utc.dt.floor("h")).drop_duplicates(["cid", "hour"])  # at most 1 per camera-hour
        cap = max(4, int(math.ceil(q / max(1, g.cid.nunique()) * 1.5)))
        g = g.groupby("cid").head(cap).head(q)
        sel.append(g)
        print(k, len(g), "cams", g.cid.nunique())
    s = pd.concat(sel).drop(columns=["hour"])
    s["utc"] = s.utc.dt.strftime("%Y-%m-%dT%H:%M")
    s.to_csv(BASE / "selection.csv", index=False)
    df.to_parquet(BASE / "sensor_grid.parquet")
    print("selected", len(s))


if __name__ == "__main__":
    main()

