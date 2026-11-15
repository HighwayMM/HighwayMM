"""Dataset figures + statistics for the paper (from labels_v1.csv only).
Outputs paper\\figs\\{camera_map,coverage_matrix,example_grid}.pdf/png and experiments\\report\\dataset_stats.json"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, HAZARDS, hazards_from_attrs  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
FIG = Path(r"D:\highwaymm\paper\figs")
OUT = Path(r"D:\highwaymm\experiments\report")
NICE = {"hmm_pilot": "Ours-pilot", "itssnap": "Ours-ITS", "itslive": "Ours-ITS-seq", "fvei_fog": "FVEI (fog)",
        "floodcam_nc": "FloodCam-NC", "nysdot_511ny": "NYSDOT blizzard", "norway_trafficcam": "Norway cams",
        "vfloodnet_seq": "Storm timelapses", "vfloodnet_pixabay": "Flood photos"}


# Paper Fig. 2: curated examples shown UNMASKED (original frames r*.jpg). Masking is applied to model inputs only; for
# readers the overlay text is harmless and the masks hid scene content. Row 1: our footage with human GOLD labels; row 2:
# external sources with their human labels (Harvey: VLM label). Labels in the titles are the dataset labels of the item.
EXAMPLES = [
    ("itssnap_its_128p52448_37p32559_20261026T165334KST", "Normal (dry, clear, day)", "Ours, gold label"),
    ("itssnap_its_129p11610_35p54906_20261025T214737KST", "Rain, wet road, night", "Ours, gold label"),
    ("itslive_ex_129p07509_37p60178_20261025T164843KST", "Rain, wet road, day", "Ours, gold label"),
    ("hmm_Rec_0147", "Snowfall, fog (moderate)", "Ours, gold label"),
    ("hmm_Rec_0103", "Heavy snow, icy road, night", "Ours, gold label"),
    ("floodcam_water_X_2020-09-20_13_36_32.537296-Buxton", "Flooding (water on road)", "FloodCam-NC, human label"),
    ("vflood_houston_buffalo_20170825_20170901_2017-08-29-17-30-00", "Rising water, Hurricane Harvey", "Storm timelapse, VLM label"),
    ("nysdot_I_495_at_Terry_Road_(Exits_59_58)__Westbound__Skyline_1878_2022-01-29-16_10_04", "Blizzard (heavy snow, snow cover)", "NYSDOT, human label"),
    ("fvei_train_1-01814-59", "Dense fog (visibility <100 m)", "FVEI, measured"),
    ("norway_train-00001-of-00002_00440", "Congestion", "Norway road cameras, human label"),
]


def _autocrop(im):
    """Cut white margins and caption text rendered around some source images (e.g. NYSDOT header lines)."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    nonwhite = a < 235
    rows = nonwhite.mean(1) > 0.6
    if rows.sum() < 0.3 * len(rows):
        return im
    idx = np.flatnonzero(rows)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    r = max(runs, key=len)
    cols = nonwhite[r[0]:r[-1] + 1].mean(0) > 0.6
    c = np.flatnonzero(cols)
    return im.crop((int(c[0]), int(r[0]), int(c[-1]) + 1, int(r[-1]) + 1))


def _fit(im, ar=16 / 10):
    w, h = im.size
    if w / h > ar:
        nw = int(h * ar); x = (w - nw) // 2
        return im.crop((x, 0, x + nw, h))
    nh = int(w / ar); y = (h - nh) // 2
    return im.crop((0, y, w, y + nh))


def example_grid_unmasked():
    fig, axes = plt.subplots(2, 5, figsize=(10.0, 3.55))
    for ax, (item, title, src) in zip(axes.ravel(), EXAMPLES):
        d = REL / "frames" / item
        f = next(d / n for n in ("r4.jpg", "r3.jpg", "r0.jpg") if (d / n).exists())
        im = _fit(_autocrop(Image.open(f).convert("RGB")))
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4)
        ax.set_title(title, fontsize=7.2, pad=2)
        ax.set_xlabel(src, fontsize=6.2, labelpad=1.5, color="#333333")
    fig.text(0.004, 0.75, "Our footage", rotation=90, va="center", fontsize=7.5, weight="bold")
    fig.text(0.004, 0.27, "External sources", rotation=90, va="center", fontsize=7.5, weight="bold")
    fig.tight_layout(pad=0.3, w_pad=0.25, h_pad=0.5, rect=(0.018, 0, 1, 1))
    fig.savefig(FIG / "example_grid.pdf", dpi=300)
    fig.savefig(FIG / "example_grid.png", dpi=200)
    plt.close(fig)
    print("example_grid (unmasked) ->", FIG / "example_grid.pdf")


MASK_DEMO = ["itssnap_its_129p11610_35p54906_20261025T214737KST", "hmm_Rec_0147"]


def mask_demo():
    """Supplement figure: original frame vs. the overlay-masked model input."""
    fig, axes = plt.subplots(len(MASK_DEMO), 2, figsize=(5.0, 1.6 * len(MASK_DEMO)))
    for row, item in zip(axes, MASK_DEMO):
        d = REL / "frames" / item
        n = "4" if (d / "r4.jpg").exists() else "0"
        for ax, (f, t) in zip(row, ((d / f"r{n}.jpg", "original (shown to annotators)"), (d / f"m{n}.jpg", "masked (model input)"))):
            ax.imshow(Image.open(f)); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=7, pad=2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG / "mask_demo.pdf", dpi=300); fig.savefig(FIG / "mask_demo.png", dpi=200)
    plt.close(fig)
    print("mask_demo ->", FIG / "mask_demo.pdf")


def main():
    if "--examples" in sys.argv:
        example_grid_unmasked()
        return mask_demo()
    FIG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REL / "annotations" / "labels_v1.csv", keep_default_na=False)
    man = pd.read_csv(REL / "manifest_ours.csv")
    # 1. camera map
    m = man[~man.exclude & man.lat.notna()].drop_duplicates("camera_id")
    fig, ax = plt.subplots(figsize=(3.3, 4.0))
    for rc, col in (("expressway", "#1f77b4"), ("national", "#ff7f0e")):
        s = m[m.road_class == rc]
        ax.scatter(s.lon, s.lat, s=0.6, c=col, label=f"{rc} ({len(s):,})", alpha=0.7, linewidths=0)
    ax.set_aspect(1.2); ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(markerscale=8, fontsize=7, loc="lower right")
    ax.set_title(f"{len(m):,} CCTV cameras", fontsize=9); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"camera_map.{ext}", dpi=200)
    plt.close(fig)
    # 2. coverage matrix (source x hazard), GT-supported vs pseudo
    rows = []
    for r in df.itertuples():
        lab = {a: getattr(r, f"lab_{a}") or None for a in ATTRS}
        hz = hazards_from_attrs(lab)
        gt = all(getattr(r, f"src_{a}") in ("gt", "physics") for a in ATTRS if getattr(r, f"lab_{a}"))
        for h in hz or ["none"]:
            rows.append({"source": NICE.get(r.source, r.source), "hazard": h, "gt": gt})
    cv = pd.DataFrame(rows)
    mat = cv.pivot_table(index="source", columns="hazard", aggfunc="size", fill_value=0)
    mat = mat[[h for h in ["none"] + HAZARDS if h in mat.columns]]
    mat.to_csv(OUT / "coverage_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 3))
    im = ax.imshow(np.log10(mat.values + 1), cmap="Blues", aspect="auto")
    ax.set_xticks(range(mat.shape[1])); ax.set_xticklabels(mat.columns, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(mat.shape[0])); ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if v:
                ax.text(j, i, f"{v}", ha="center", va="center", fontsize=6, color="k" if v < 1000 else "w")
    fig.colorbar(im, ax=ax, label="log10(count)"); fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"coverage_matrix.{ext}", dpi=200)
    plt.close(fig)
    # 3. example grid: one item per hazard (prefer GT-labelled, high-confidence)
    picks = []
    rng = np.random.default_rng(3)
    # prefer sources whose labels for that hazard are human GT (or high-confidence VLM on plausible seasons)
    pref = {"none": ["itssnap"], "heavy_rain": ["hmm_pilot", "itssnap"], "flooding": ["floodcam_nc"],
            "snowfall": ["nysdot_511ny", "hmm_pilot"], "blizzard": ["nysdot_511ny"], "snow_cover": ["nysdot_511ny", "hmm_pilot"],
            "dense_fog": ["fvei_fog"], "low_visibility": ["fvei_fog"], "congestion": ["norway_trafficcam"]}
    for h in ["none", "heavy_rain", "flooding", "snowfall", "blizzard", "snow_cover", "dense_fog", "low_visibility", "congestion"]:
        cand = [r for r in df.itertuples() if h in (hazards_from_attrs({a: getattr(r, f"lab_{a}") or None for a in ATTRS}) or ["none"])]
        good = [r for r in cand if r.source in pref[h] and all(getattr(r, f"src_{a}") in ("gt", "physics") or getattr(r, f"w_{a}") >= 0.8
                                                               for a in ATTRS if getattr(r, f"lab_{a}"))]
        good = sorted(good, key=lambda r: pref[h].index(r.source))
        pool = [r for r in good if r.source == good[0].source] if good else cand
        used = {p[1].item_id for p in picks}
        pool = [r for r in pool if r.item_id not in used] or pool
        if not pool:
            continue
        r = pool[rng.integers(len(pool))]
        picks.append((h, r))
    n = len(picks)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(2.0 * ((n + 1) // 2), 3.2))
    for ax, (h, r) in zip(axes.ravel(), picks):
        d = REL / "frames" / r.item_id
        f = d / ("m4.jpg" if (d / "m4.jpg").exists() else "m0.jpg")
        ax.imshow(Image.open(f)); ax.set_title(f"{h}\n{NICE.get(r.source, r.source)}", fontsize=6); ax.axis("off")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"example_grid.{ext}", dpi=200)
    plt.close(fig)
    stats = {"items": len(df), "by_source": df.source.value_counts().to_dict(), "by_split": {f"{k[0]}/{k[1]}": int(v) for k, v in df.groupby(["domain", "split"]).size().items()},
             "cameras_ours": int(man[~man.exclude].camera_id.nunique()),
             "label_src": {a: df[f"src_{a}"].replace("", "none").value_counts().to_dict() for a in ATTRS},
             "warn_levels": df.warn_level.value_counts().to_dict(), "alert_tags": df.alert_tags.replace("", np.nan).dropna().value_counts().to_dict()}
    (OUT / "dataset_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False, default=str)[:3000])


if __name__ == "__main__":
    main()
