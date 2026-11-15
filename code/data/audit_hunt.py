"""Hazard-hunt audit: 90 VLM/zero-shot-mined candidates (make_hazard_hunt.py) checked visually by the assistant
(Claude, from the unmasked 3-frame strips) instead of a human pass. Writes
  gold/hunt_labels_audit.csv     audited labels (precip/intensity/surface/visibility/illumination/traffic, tunnel, unusable)
  gold/hunt_audit_report.txt     VLM-vs-audit agreement + verified hazard counts per category
  annotations/labels_v3.csv      labels_v2 with the audited TRAIN/VAL hunt clips overriding the VLM pseudo-labels
                                 (src 'audit', w 1.0); test hunt clips untouched so the test protocol is unchanged.
Audit labels are NOT human gold and are reported as such in the paper.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, warning_level  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
A = list(ATTRS)
# index in hunt_v1.csv order: precip/intensity/surface/visibility/illumination/traffic ; flags t=tunnel u=unusable ?=unsure
RAW = """
0 snow/light/wet/haze/night/free
1 rain/moderate/wet/haze/night/free
2 none/none/dry/clear/unknown/free t
3 none/none/wet/clear/night/free
4 none/none/dry/clear/unknown/free t
5 none/none/dry/clear/day/free
6 none/none/dry/clear/night/free
7 rain/light/wet/haze/night/free ?
8 none/none/dry/clear/day/free
9 none/none/dry/haze/day/free
10 rain/heavy/wet/haze/night/free
11 rain/moderate/wet/haze/night/free
12 none/none/wet/clear/night/free
13 none/none/dry/clear/unknown/free t
14 rain/moderate/wet/haze/night/free
15 rain/light/wet/haze/night/free ?
16 rain/light/wet/haze/night/free ?
17 rain/heavy/wet/haze/night/free
18 none/none/dry/clear/day/free
19 none/none/dry/clear/unknown/dense t
20 rain/heavy/wet/haze/night/free
21 rain/heavy/wet/haze/night/free
22 rain/light/wet/haze/night/free
23 rain/light/wet/haze/night/free ?
24 none/none/dry/clear/day/free
25 rain/moderate/wet/haze/night/free
26 none/none/dry/clear/night/free u
27 none/none/wet/clear/day/free
28 none/none/dry/clear/day/free
29 none/none/wet/haze/day/free
30 none/none/dry/clear/night/free u
31 none/none/wet/clear/day/free
32 rain/light/wet/haze/night/free
33 rain/moderate/wet/haze/night/free
34 rain/light/wet/clear/day/dense
35 none/none/wet/haze/night/free ?
36 none/none/dry/clear/unknown/free t
37 none/none/dry/clear/night/dense
38 rain/heavy/wet/haze/night/free
39 none/none/dry/clear/night/free u
40 rain/heavy/wet/haze/night/free
41 rain/heavy/wet/haze/night/free
42 rain/moderate/wet/haze/night/free
43 none/none/dry/clear/unknown/free t
44 rain/heavy/wet/fog_moderate/night/free
45 rain/light/wet/clear/night/free
46 none/none/dry/clear/unknown/free t
47 snow/moderate/snow/fog_moderate/day/free
48 snow/light/wet/haze/day/free
49 none/none/wet/fog_moderate/night/free
50 none/none/wet/fog_dense/night/free
51 snow/light/snow/haze/night/free
52 none/none/dry/clear/unknown/dense t
53 rain/light/wet/haze/night/free ?
54 none/none/dry/haze/unknown/dense t
55 none/none/wet/haze/day/free
56 rain/light/wet/haze/day/dense
57 none/none/dry/clear/night/free
58 none/none/wet/haze/day/free ?
59 none/none/wet/fog_moderate/night/free
60 none/none/dry/clear/day/free
61 none/none/dry/clear/night/free
62 none/none/dry/haze/night/free
63 none/none/dry/clear/day/free
64 rain/heavy/wet/haze/night/free
65 none/none/dry/haze/night/free ?
66 none/none/wet/clear/night/free ?
67 none/none/dry/clear/day/free
68 none/none/dry/clear/day/free
69 rain/moderate/wet/haze/night/free
70 rain/light/wet/haze/night/free ?
71 none/none/wet/haze/night/free
72 none/none/dry/clear/night/free u
73 rain/heavy/wet/haze/night/free
74 none/none/wet/haze/day/free
75 none/none/dry/clear/night/free
76 none/none/wet/clear/night/free
77 none/none/dry/clear/unknown/free t
78 snow/light/snow/haze/night/free
79 none/none/dry/fog_moderate/unknown/free t
80 none/none/dry/clear/unknown/free t
81 snow/light/wet/haze/day/free
82 none/none/wet/clear/night/free
83 rain/moderate/wet/haze/night/free
84 none/none/wet/fog_moderate/night/free
85 rain/light/wet/haze/day/free
86 none/none/dry/clear/unknown/free t
87 rain/light/wet/haze/night/free
88 none/none/dry/clear/night/free
89 none/none/dry/haze/night/free
"""


def main():
    h = pd.read_csv(REL / "gold" / "hunt_v1.csv", keep_default_na=False)
    rows = []
    for line in RAW.strip().splitlines():
        p = line.split()
        i, vals, fl = int(p[0]), p[1].split("/"), "".join(p[2:])
        r = h.iloc[i]
        rows.append({"hid": r.hid, "clip_id": r.clip_id, "category": r.category, "split": r.split, "vlm": r.vlm,
                     **dict(zip(A, vals)), "tunnel": int("t" in fl), "unusable": int("u" in fl), "unsure": int("?" in fl)})
    au = pd.DataFrame(rows)
    assert len(au) == len(h) == au.hid.nunique()
    for a in A:
        ok = set(ATTRS[a]) | ({"unknown"} if a == "illumination" else set())
        bad = au[~au[a].isin(ok)]
        assert bad.empty, (a, bad[a].tolist())
    au.to_csv(REL / "gold" / "hunt_labels_audit.csv", index=False, encoding="utf-8")

    use = au[au.unusable == 0].copy()
    rep = [f"hunt clips {len(au)} | unusable {int(au.unusable.sum())} | tunnel {int(au.tunnel.sum())} | unsure {int(au.unsure.sum())}"]
    v = use.vlm.str.split("/", expand=True)
    for j, a in enumerate(A[:4]):
        rep.append(f"VLM agrees on {a:10s}: {100 * (v[j] == use[a]).mean():5.1f}%")
    use["is_snow"] = use.precip.eq("snow") | use.surface.isin(["snow", "ice"])
    use["is_blizzard"] = use.precip.eq("snow") & (use.intensity.eq("heavy") | use.visibility.isin(["fog_moderate", "fog_dense"]))
    use["is_flood"] = use.surface.eq("water")
    use["is_rain"] = use.precip.eq("rain")
    use["is_fog"] = use.visibility.isin(["fog_moderate", "fog_dense"])
    rep.append("\nverified per mined category (rows) vs audited hazard (cols):")
    rep.append(use.groupby("category")[["is_snow", "is_blizzard", "is_flood", "is_rain", "is_fog"]].sum().to_string())
    rep.append("\ntotals: " + ", ".join(f"{c[3:]} {int(use[c].sum())}" for c in ["is_snow", "is_blizzard", "is_flood", "is_rain", "is_fog"]))
    rep.append("precip: " + str(use.precip.value_counts().to_dict()))
    rep.append("surface: " + str(use.surface.value_counts().to_dict()))
    rep.append("by split: " + str(use.split.value_counts().to_dict()))
    txt = "\n".join(rep)
    (REL / "gold" / "hunt_audit_report.txt").write_text(txt, encoding="utf-8")
    print(txt)

    # labels_v3: audited train/val hunt clips override VLM pseudo-labels
    lab = pd.read_csv(REL / "annotations" / "labels_v2.csv", keep_default_na=False, dtype=str)
    idx = {c: i for i, c in enumerate(lab.item_id)}
    n = 0
    for r in au[au.split.isin(["train", "val"])].itertuples():
        i = idx.get(r.clip_id)
        if i is None:
            print("missing", r.clip_id); continue
        if r.unusable:
            for a in A:
                lab.at[i, f"w_{a}"] = "0.0"
            lab.at[i, "note"] = "audit:unusable"
            continue
        for a in A:
            val = getattr(r, a)
            if a == "illumination" and val == "unknown":
                continue  # keep the physics (sun-angle) label
            lab.at[i, f"lab_{a}"] = val; lab.at[i, f"src_{a}"] = "audit"; lab.at[i, f"w_{a}"] = "1.0"
        att = {a: lab.at[i, f"lab_{a}"] for a in A if lab.at[i, f"lab_{a}"]}
        lab.at[i, "warn_level"] = str(warning_level(att, lab.at[i, "transitional"] == "1"))
        lab.at[i, "note"] = "audit"
        n += 1
    lab.to_csv(REL / "annotations" / "labels_v3.csv", index=False, encoding="utf-8")
    print("labels_v3:", n, "train/val clips overridden ->", REL / "annotations" / "labels_v3.csv")


if __name__ == "__main__":
    main()
