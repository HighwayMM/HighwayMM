"""Supplement figure: where HighwayMM comes from.
(a) our 8.6k Korean ITS/KEC cameras drawn as CCTV icons on a Korea outline, with a zoom on the Seoul area;
(b) world map of all sources (ours + 7 external), so readers see the benchmark is not Korea-only.
Outlines: Natural Earth admin-0 countries (public domain), D:\\highwaymm\\tools\\naturalearth.
Outputs paper\\figs\\source_map.{pdf,png} and code\\report\\github_static\\docs\\assets\\source_map.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
NE = Path(r"D:\highwaymm\tools\naturalearth")
FIG = Path(r"D:\highwaymm\paper\figs")
GH = Path(r"D:\highwaymm\code\report\github_static\docs\assets")
IOWA = Path(r"D:\highwaymm\external\iowa_rwis\selection.csv")

LAND, EDGE, SEA = "#eceae4", "#9a968c", "#f6f9fc"
C_EX, C_NAT = "#1f5fa8", "#e07b16"


def cctv_marker():
    """A small CCTV camera glyph (housing + lens hood + wall bracket) as one compound path."""
    def poly(pts):
        return [(MPath.MOVETO, pts[0])] + [(MPath.LINETO, p) for p in pts[1:]] + [(MPath.CLOSEPOLY, pts[0])]
    body = [(-0.6, 0.1), (0.25, 0.1), (0.25, 0.7), (-0.6, 0.7)]
    hood = [(0.25, 0.02), (0.65, -0.1), (0.65, 0.82), (0.25, 0.78)]
    arm = [(-0.3, 0.1), (-0.12, 0.1), (-0.12, -0.6), (-0.3, -0.6)]
    wall = [(-0.55, -0.82), (0.13, -0.82), (0.13, -0.6), (-0.55, -0.6)]
    th = np.deg2rad(-15)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    codes, verts = [], []
    for part in (body, hood, arm, wall):
        for c, v in poly([tuple(R @ np.array(p)) for p in part]):
            codes.append(c); verts.append(v)
    return MPath(verts, codes)


CAM = cctv_marker()
# compact legend labels for the main-paper version (counts are in the supplement version and Table 1)
SHORT = {"Iowa DOT RWIS": "Iowa DOT RWIS (49 cams)", "NYSDOT blizzard, Long Island": "NYSDOT blizzard",
         "FloodCam-NC, Outer Banks": "FloodCam-NC", "Storm timelapses (Houston, Baton Rouge, Boston)": "storm timelapses",
         "Ours (ITS/KEC)": "ours (ITS/KEC)", "TSBOW, Suwon (test only)": "TSBOW (test only)",
         "FVEI fog, Yunnan expressways": "FVEI fog (Yunnan)", "Norway road cameras": "Norway road cams"}


def load_countries(res):
    gj = json.loads((NE / f"ne_{res}_admin_0_countries.geojson").read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        out.append((f["properties"].get("ADM0_A3", ""), polys))
    return out


def draw_land(ax, countries, highlight=(), hl_color="#dfe7f1"):
    for code, polys in countries:
        for poly in polys:
            ring = np.asarray(poly[0])
            ax.fill(ring[:, 0], ring[:, 1], color=hl_color if code in highlight else LAND, lw=0, zorder=1, rasterized=True)
            ax.plot(ring[:, 0], ring[:, 1], color=EDGE, lw=0.3, zorder=2, rasterized=True)


def main():
    S.apply()
    man = pd.read_csv(REL / "manifest_ours.csv")
    m = man[~man.exclude & man.lat.notna()].drop_duplicates("camera_id")
    n_ex, n_nat = (m.road_class == "expressway").sum(), (m.road_class == "national").sum()
    ext = pd.read_csv(REL / "manifest_external.csv", low_memory=False)
    cnt = ext.source.value_counts()
    iowa = pd.read_csv(IOWA).drop_duplicates("cid")
    print(f"ours: {len(m)} cameras ({n_ex} expressway, {n_nat} national); iowa cams with coords: {len(iowa)}")

    c10, c50 = load_countries("10m"), load_countries("50m")
    for layout in ("supp", "main"):
        render(layout, m, n_ex, n_nat, cnt, iowa, c10, c50)


def render(layout, m, n_ex, n_nat, cnt, iowa, c10, c50):
    """supp: Korea left, USA / Europe+Asia stacked right. main: compact single row for the main paper."""
    if layout == "supp":
        fig = plt.figure(figsize=(7.0, 4.05))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32], wspace=0.03, hspace=0.12)
        k_spec, u_spec, e_spec, name, fs_leg, fs_t = gs[:, 0], gs[0, 1], gs[1, 1], "source_map", 5.4, 7.5
    else:
        fig = plt.figure(figsize=(7.0, 1.86))
        gs = fig.add_gridspec(1, 3, width_ratios=[0.906, 1.436, 1.959], wspace=0.025)
        k_spec, u_spec, e_spec, name, fs_leg, fs_t = gs[0], gs[1], gs[2], "source_map_main", 5.6, 7.0

    # ---- (a) Korea --------------------------------------------------------------------------------
    ax = fig.add_subplot(k_spec)
    ax.set_facecolor(SEA)
    draw_land(ax, c10)
    for rc, col, z in (("national", C_NAT, 3), ("expressway", C_EX, 4)):
        s = m[m.road_class == rc]
        ax.scatter(s.lon, s.lat, marker=CAM, s=4 if layout == "supp" else 2.2, c=col, linewidths=0, alpha=0.85, zorder=z, rasterized=True)
    X0, X1, Y0, Y1 = 125.75, 131.2, 33.9, 38.75
    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_aspect(1 / np.cos(np.deg2rad(36.3)))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color("#8a8f96"); sp.set_linewidth(0.5)
    ax.set_title(f"(a) Ours: {len(m):,} ITS/KEC cameras, Korea" if layout == "supp" else f"(a) Ours: {len(m):,} cameras, Korea", fontsize=fs_t, pad=3, loc="left")
    # zoom on the Seoul metropolitan area, placed over the East Sea
    zx0, zx1, zy0, zy1 = 126.80, 127.20, 37.38, 37.66
    ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec="k", lw=0.6, zorder=6))
    ins = ax.inset_axes([0.64, 0.6, 0.355, 0.33] if layout == "supp" else [0.6, 0.62, 0.395, 0.33])
    ins.set_facecolor(SEA)
    draw_land(ins, c10)
    for rc, col, z in (("national", C_NAT, 3), ("expressway", C_EX, 4)):
        s = m[(m.road_class == rc) & m.lon.between(zx0, zx1) & m.lat.between(zy0, zy1)]
        ins.scatter(s.lon, s.lat, marker=CAM, s=80 if layout == "supp" else 40, c=col, ec="white", linewidths=0.25, zorder=z, rasterized=True)
    ins.set_xlim(zx0, zx1); ins.set_ylim(zy0, zy1); ins.set_aspect(1 / np.cos(np.deg2rad(37.5)))
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.6)
    ins.set_title("Seoul area", fontsize=6, pad=1.5)
    ax.indicate_inset_zoom(ins, edgecolor="k", lw=0.5, alpha=0.6)
    h = [Line2D([], [], ls="", marker=CAM, ms=8, color=C_EX, label=f"expressway ({n_ex:,})"),
         Line2D([], [], ls="", marker=CAM, ms=8, color=C_NAT, label=f"national road ({n_nat:,})")]
    ax.legend(handles=h, loc="lower right", fontsize=6 if layout == "supp" else 5.6, frameon=True, framealpha=0.9, borderpad=0.4, handletextpad=0.3)

    # ---- (b) USA, (c) Europe + Asia: external sources ----------------------------------------------
    # each: (label, [(lon, lat), ...], colour, marker, located). located=False: only the region is known (hollow)
    usa = [
        (f"Iowa DOT RWIS: {iowa.cid.nunique()} cams, {cnt.get('iowa_rwis', 0):,} img (sensor labels)",
         list(zip(iowa.lon, iowa.lat)), "#d4ac0d", "^", True),
        (f"NYSDOT blizzard, Long Island: 3 cams, {cnt.get('nysdot_511ny', 0)} img",
         [(-73.12, 40.80), (-73.24, 40.79), (-72.99, 40.80)], "#2e86c1", "s", True),
        (f"FloodCam-NC, Outer Banks: 4 cams, {cnt.get('floodcam_nc', 0)} img",
         [(-75.95, 35.12), (-75.54, 35.26), (-75.52, 35.25), (-75.47, 35.61)], "#17a589", "v", True),
        (f"Storm timelapses (Houston, Baton Rouge, Boston): {cnt.get('vfloodnet_seq', 0)} fr.",
         [(-95.37, 29.76), (-91.18, 30.41), (-71.05, 42.35)], "#e67e22", "P", True),
    ]
    eura = [
        (f"Ours (ITS/KEC): {len(m):,} cams, Korea", [(127.8, 36.4)], "#c0392b", "*", True),
        (f"TSBOW, Suwon (test only): {cnt.get('tsbow', 0):,} fr.", [(127.03, 37.26)], "#7d3c98", "D", True),
        (f"FVEI fog, Yunnan expressways: {cnt.get('fvei_fog', 0):,} img", [(100.2, 26.9)], "#566573", "o", False),
        (f"Norway road cameras: {cnt.get('norway_trafficcam', 0):,} img", [(9.5, 61.0)], "#16a085", "o", False),
    ]
    panels = (
        (u_spec, usa, (-104, -64.5, 24.5, 46.5) if layout == "main" else (-117, -64, 22.5, 49.5), 37, ("USA",), "(b) External: USA", "lower left" if layout == "supp" else "lower right"),
        (e_spec, eura, (-2, 142, 18, 70) if layout == "main" else (-8, 146, 16, 72), 45, ("KOR", "NOR", "CHN"), "(c) External: Europe and Asia", "lower left"),
    )
    for spec, items, (x0, x1, y0, y1), lat0, hl, title, loc in panels:
        aw = fig.add_subplot(spec)
        aw.set_facecolor(SEA)
        draw_land(aw, c50, highlight=hl)
        hs = []
        for lab, pts, col, mk, located in items:
            if layout == "main":
                lab = SHORT.get(lab.split(":")[0], lab)
            xs, ys = zip(*pts)
            big = 70 if mk == "*" else (14 if len(pts) > 10 else 30)
            kw = dict(c=col, ec="k", lw=0.3) if located else dict(c="white", ec=col, lw=1.2)
            aw.scatter(xs, ys, marker=mk, s=big, zorder=5, **kw)
            hs.append(Line2D([], [], ls="", marker=mk, ms=6 if mk == "*" else 4.5,
                             mfc=col if located else "white", mec="k" if located else col,
                             mew=0.3 if located else 1.0, label=lab))
        aw.set_xlim(x0, x1); aw.set_ylim(y0, y1); aw.set_aspect(1 / np.cos(np.deg2rad(lat0)))
        aw.set_xticks([]); aw.set_yticks([])
        aw.set_title(title, fontsize=fs_t, pad=3, loc="left" if layout == "main" else "center")
        for sp in aw.spines.values():
            sp.set_visible(True); sp.set_color("#8a8f96"); sp.set_linewidth(0.5)
        aw.legend(handles=hs, loc=loc, fontsize=fs_leg, frameon=True, framealpha=0.92, borderpad=0.35,
                  handletextpad=0.25, labelspacing=0.25, borderaxespad=0.3)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.95 if layout == "supp" else 0.885, bottom=0.01)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", dpi=300)
    fig.savefig(FIG / f"{name}.png", dpi=220)
    plt.close(fig)
    print("->", FIG / f"{name}.pdf")


if __name__ == "__main__":
    main()
