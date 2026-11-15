"""Main-paper dataset figures in the shared style (figstyle.py).
  python code\\report\\paper_figs.py            -> paper\\figs\\teaser.pdf/png, coverage.pdf/png
Teaser: curated examples, shown UNMASKED (masking is for model inputs only), each with its label-derived warning level.
Coverage: source x hazard counts from labels_v2t.csv (the label set of the paper, incl. Iowa and test-only TSBOW)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from data.taxonomy import ATTRS, hazards_from_attrs, warning_level  # noqa: E402
import figstyle as S  # noqa: E402
from dataset_figs import EXAMPLES, _autocrop, _fit  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
FIG = Path(r"D:\highwaymm\paper\figs")
LABELS = REL / "annotations" / "labels_v2t.csv"


def _save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print(name, "->", FIG / f"{name}.pdf")


SHORT = [("Dry, clear, day", "ours · human gold label"), ("Rain, wet road, night", "ours · human gold label"),
         ("Rain, wet road, day", "ours · human gold label"), ("Snowfall, moderate fog", "ours · human gold label"),
         ("Heavy snow, icy road, night", "ours · human gold label"), ("Flooded road", "FloodCam-NC · human label"),
         ("Storm surge, Hurricane Harvey", "storm timelapse · VLM label"), ("Blizzard", "NYSDOT · human label"),
         ("Dense fog, visibility <100 m", "FVEI · measured"), ("Congestion", "Norway road cams · human label")]


def gold_levels():
    """Warning level of each gold clip from its HUMAN labels (the teaser titles describe the gold labels)."""
    g = pd.read_csv(REL / "gold" / "gold_labels_merged.csv", keep_default_na=False)
    v = pd.read_csv(REL / "gold" / "gold_v1.csv", keep_default_na=False)[["gid", "clip_id"]]
    g = g.merge(v, on="gid").drop_duplicates("gid")
    out = {}
    for r in g.itertuples():
        a = {k: (getattr(r, k) or None) for k in ("precip", "intensity", "surface", "visibility", "illumination", "traffic")}
        if a["illumination"] not in ("day", "twilight", "night"):
            a["illumination"] = None
        out[r.clip_id] = warning_level(a, transitional=str(r.transition) == "1")
    return out


def teaser(df):
    lvl = dict(zip(df.item_id, df.warn_level))
    lvl.update(gold_levels())
    W, H = S.TEXTW, 2.62
    fig = plt.figure(figsize=(W, H))
    left, gap, top_pad = 0.30, 0.045, 0.16          # inches
    cw = (W - left - 4 * gap - 0.02) / 5
    ch = cw / 1.6
    row_h = ch + 0.42
    for k, ((item, _, _), (title, src)) in enumerate(zip(EXAMPLES, SHORT)):
        r, c = divmod(k, 5)
        x0 = left + c * (cw + gap)
        y0 = H - top_pad - r * row_h - ch - 0.12 * r
        ax = fig.add_axes([x0 / W, y0 / H, cw / W, ch / H])
        d = REL / "frames" / item
        f = next(d / n for n in ("r4.jpg", "r3.jpg", "r0.jpg") if (d / n).exists())
        ax.imshow(_fit(_autocrop(Image.open(f).convert("RGB"))), interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        L = int(lvl.get(item, 0))
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(S.LCOL[L]); s.set_linewidth(1.6)
        # level badge, top-left inside the frame
        ax.text(0.035, 0.94, S.LNAME[L].upper(), transform=ax.transAxes, ha="left", va="top", fontsize=5.6,
                weight="bold", color="white" if L != 1 else S.INK,
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.3", fc=S.LCOL[L], ec="none"))
        fig.text((x0 + cw / 2) / W, (y0 - 0.05) / H, title, ha="center", va="top", fontsize=6.4, weight="bold")
        fig.text((x0 + cw / 2) / W, (y0 - 0.19) / H, src, ha="center", va="top", fontsize=5.8, color=S.MUTED)
    for r, name in enumerate(["Our Korean CCTV", "External sources"]):
        yc = H - top_pad - r * row_h - 0.12 * r - ch / 2
        fig.text(0.13 / W, yc / H, name, rotation=90, ha="center", va="center", fontsize=6.8, weight="bold",
                 color=S.ACCENT if r == 0 else S.INK)
    _save(fig, "teaser")


COLS = [("none", "no hazard", 0), ("wet", "wet road", 1), ("snowfall", "snowfall", 1),
        ("low_visibility", "low visibility", 1), ("congestion", "congestion", 1), ("heavy_rain", "heavy rain", 2),
        ("snow_cover", "snow cover", 2), ("flooding", "flooding", 3), ("blizzard", "blizzard", 3),
        ("icy_road", "icy road", 3), ("dense_fog", "dense fog", 3)]


def coverage(df):
    keys = [k for k, _, _ in COLS]
    cnt = {s: dict.fromkeys(keys, 0) for s in S.OURS + S.EXTERNAL}
    for r in df.itertuples():
        if r.source not in cnt:
            continue
        lab = {a: getattr(r, f"lab_{a}") or None for a in ATTRS}
        hz = hazards_from_attrs(lab) or []
        for h in hz:
            cnt[r.source][h] += 1
        if lab.get("surface") == "wet":
            cnt[r.source]["wet"] += 1
        if not hz:
            cnt[r.source]["none"] += 1
    rows = S.OURS + S.EXTERNAL
    M = np.array([[cnt[s][k] for k in keys] for s in rows])
    pd.DataFrame(M, index=[S.SOURCE_NAME[s] for s in rows], columns=keys).to_csv(
        Path(r"D:\highwaymm\experiments\report") / "coverage_matrix_v2t.csv")

    W = S.COLW
    lab_w, cell_w, cell_h, head_h, top_h = 1.02, (S.COLW - 1.02 - 0.30) / len(COLS), 0.158, 0.50, 0.0
    H = top_h + head_h + cell_h * len(rows) + 0.10 + 0.05
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([lab_w / W, 0.05 / H, cell_w * len(COLS) / W, (cell_h * len(rows) + 0.10) / H])
    ax.set_xlim(0, len(COLS)); ax.set_ylim(len(rows) + 0.63, 0); ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("c", ["#eef3f9", "#9fbfe0", "#2f6fb3", "#123e72"])
    lo, hi = 0, np.log10(M.max())
    for i, s in enumerate(rows):
        y = i + (0.63 if i >= len(S.OURS) else 0)
        for j in range(len(COLS)):
            v = M[i, j]
            if v:
                t = (np.log10(v) - lo) / (hi - lo)
                ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, fc=cmap(t), ec="none"))
                txt = f"{v / 1000:.1f}k" if v >= 1000 else f"{v}"
                ax.text(j + 0.5, y + 0.52, txt, ha="center", va="center", fontsize=4.9,
                        color="white" if t > 0.55 else S.INK)
            else:
                ax.add_patch(Rectangle((j + 0.04, y + 0.06), 0.92, 0.88, fc="none", ec="#e3e6ea", lw=0.4))
        fig.text((lab_w - 0.05) / W, (0.05 + (cell_h * len(rows) + 0.10) * (1 - (y + 0.5) / (len(rows) + 0.63))) / H,
                 S.SOURCE_NAME[s], ha="right", va="center", fontsize=5.9, color=S.INK)
    # group labels
    ax.text(-0.15, -0.0, "", fontsize=1)
    ax.plot([0, len(COLS)], [len(S.OURS) + 0.31] * 2, color=S.RULE, lw=0.6, clip_on=False)
    # column headers: rotated names + level colour bar
    bar_y = (0.05 + cell_h * len(rows) + 0.10 + 0.02) / H
    for j, (_, name, L) in enumerate(COLS):
        xc = (lab_w + (j + 0.5) * cell_w) / W
        fig.patches.append(Rectangle(((lab_w + j * cell_w + 0.006) / W, bar_y), (cell_w - 0.012) / W, 0.045 / H,
                                     transform=fig.transFigure, fc=S.LCOL[L], ec="none"))
        fig.text(xc, bar_y + 0.075 / H, name, rotation=50, ha="left", va="bottom", fontsize=5.7, color=S.INK,
                 rotation_mode="anchor")
    # group labels on the left edge
    def ytxt(yrow):
        return (0.05 + (cell_h * len(rows) + 0.10) * (1 - yrow / (len(rows) + 0.63))) / H
    gx = -(lab_w - 0.03) / cell_w
    ax.text(gx, len(S.OURS) / 2, "ours", rotation=90, ha="center", va="top", rotation_mode="anchor", fontsize=6, weight="bold",
            color=S.ACCENT, clip_on=False)
    ax.text(gx, len(S.OURS) + 0.63 + len(S.EXTERNAL) / 2, "external", rotation=90, ha="center", va="top", rotation_mode="anchor",
            fontsize=6, weight="bold", color=S.INK, clip_on=False)
    # level legend (top-left)
    fig.text(0.06 / W, (H - 0.07) / H, "hazard level:", fontsize=5.6, va="center", color=S.MUTED)
    for L in range(4):
        xx, yy = 0.06 + (L % 2) * 0.47, H - 0.20 - (L // 2) * 0.12
        fig.patches.append(Rectangle((xx / W, (yy - 0.037) / H), 0.075 / W, 0.075 / H,
                                     transform=fig.transFigure, fc=S.LCOL[L], ec="none"))
        fig.text((xx + 0.1) / W, yy / H, S.LNAME[L], fontsize=5.6, va="center", color=S.INK)
    _save(fig, "coverage")


def main():
    S.apply()
    df = pd.read_csv(LABELS, keep_default_na=False)
    if "--teaser" not in sys.argv[1:] or True:
        teaser(df)
    coverage(df)


if __name__ == "__main__":
    main()
