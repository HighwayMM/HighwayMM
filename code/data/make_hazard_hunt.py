"""Hazard hunt: candidate clips of OUR data for snow / blizzard / flooded road, for human verification.

Ranks every our-domain clip (not already in the gold set) by (a) Qwen2.5-VL pseudo-labels and (b) zero-shot SigLIP2
similarity to the taxonomy text prototypes, takes the top candidates per category, renders unmasked review strips
(same layout as the gold page) and writes HighwayMM_release/gold/hunt_v1.csv + hunt_clips.json + strips_raw/<hid>.jpg.
Read-only on frames/features; all outputs are new files.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REL = Path(r"D:\highwaymm\HighwayMM_release")
ATTRS = {"precip": ["none", "rain", "snow"], "intensity": ["none", "light", "moderate", "heavy"],
         "surface": ["dry", "wet", "water", "snow", "ice"], "visibility": ["clear", "haze", "fog_moderate", "fog_dense"]}
N_PER = {"snow": 45, "blizzard": 30, "flood": 15}  # flood: our Oct footage has none; top-15 only to confirm


def zeroshot(ids):
    tp = np.load(REL / "features" / "text_protos_siglip2.npz")
    idx, X = {}, []
    for f in sorted((REL / "features" / "siglip2").glob("shard[0-9]*.npz")):  # our + original external shards
        z = np.load(f)
        for i, k in enumerate(z["ids"]):
            idx[k] = (len(X), i)
        x = z["masked"].astype(np.float32)
        v = (np.abs(x).sum(-1) > 0)[..., None]
        m = (x * v).sum(1) / np.maximum(v.sum(1), 1)
        X.append(m / np.linalg.norm(m, axis=1, keepdims=True).clip(1e-6))
    out = {}
    for a in ATTRS:
        P = tp[a].astype(np.float32)
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        sc = {}
        for k in ids:
            if k in idx:
                s, i = idx[k]
                lg = 100 * X[s][i] @ P.T
                e = np.exp(lg - lg.max())
                sc[k] = e / e.sum()
        out[a] = sc
    return out


def strip(clip_id, dst):
    d = REL / "frames" / clip_id
    pre = "r" if (d / "r4.jpg").exists() else "m"
    f = [Image.open(d / f"{pre}{i}.jpg").convert("RGB") for i in (1, 4, 6)] if (d / f"{pre}6.jpg").exists() else \
        [Image.open(sorted(d.glob(f"{pre}*.jpg"))[0]).convert("RGB")] * 3
    W, H = f[1].size
    sw, gap = 512, 4
    sh = round(sw * H / W)
    bh = 2 * sh + gap
    big = f[1].resize((round(bh * W / H), bh), Image.LANCZOS)
    c = Image.new("RGB", (big.width + gap + sw, bh), (20, 20, 20))
    c.paste(big, (0, 0))
    c.paste(f[0].resize((sw, sh), Image.LANCZOS), (big.width + gap, 0))
    c.paste(f[2].resize((sw, sh), Image.LANCZOS), (big.width + gap, sh + gap))
    c.save(dst, quality=88)


def main():
    l = pd.read_csv(REL / "annotations" / "labels_v2t.csv", keep_default_na=False)
    o = l[l.domain == "ours"].copy()
    gold = set(pd.read_csv(REL / "gold" / "gold_v1.csv").clip_id)
    o = o[~o.item_id.isin(gold)]
    zs = zeroshot(o.item_id.tolist())
    g = lambda a, v: np.array([zs[a].get(k, np.zeros(len(ATTRS[a])))[ATTRS[a].index(v)] for k in o.item_id])  # noqa: E731
    o["zs_snow"] = np.maximum(g("precip", "snow"), g("surface", "snow"))
    o["zs_heavy"] = g("intensity", "heavy")
    o["zs_lowvis"] = g("visibility", "fog_moderate") + g("visibility", "fog_dense")
    o["zs_water"] = g("surface", "water")
    vlm_snow = (o.lab_precip == "snow") | (o.lab_surface == "snow")
    vlm_bliz = (o.lab_precip == "snow") & ((o.lab_intensity == "heavy") | o.lab_visibility.isin(["fog_moderate", "fog_dense"]))
    o["s_snow"] = vlm_snow * 1.0 + o.zs_snow
    o["s_blizzard"] = vlm_bliz * 1.0 + o.zs_snow * (o.zs_heavy + o.zs_lowvis)
    o["s_flood"] = (o.lab_surface == "water") * 1.0 + o.zs_water
    picked, rows = set(), []
    for cat in ("blizzard", "snow", "flood"):
        cand = o[~o.item_id.isin(picked)].sort_values(f"s_{cat}", ascending=False)
        # at most 3 clips per camera so one snowy camera does not fill the batch
        cand = cand.groupby("camera_id").head(3).head(N_PER[cat])
        for r in cand.itertuples():
            picked.add(r.item_id)
            rows.append({"hid": "h" + hashlib.md5(r.item_id.encode()).hexdigest()[:10], "clip_id": r.item_id,
                         "category": cat, "score": round(getattr(r, f"s_{cat}"), 3), "source": r.source,
                         "camera_id": r.camera_id, "split": r.split, "vlm": f"{r.lab_precip}/{r.lab_intensity}/{r.lab_surface}/{r.lab_visibility}"})
    h = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)  # shuffled: no category hint
    (REL / "gold" / "strips_raw").mkdir(exist_ok=True)
    for r in h.itertuples():
        strip(r.clip_id, REL / "gold" / "strips_raw" / f"{r.hid}.jpg")
    h.to_csv(REL / "gold" / "hunt_v1.csv", index=False)
    (REL / "gold" / "hunt_clips.json").write_text(json.dumps([{"id": x} for x in h.hid]), encoding="utf-8")
    print(len(h), h.category.value_counts().to_dict(), h.split.value_counts().to_dict(), h.source.value_counts().to_dict())
    print(h.groupby("category").score.describe()[["min", "50%", "max"]])


if __name__ == "__main__":
    main()
