"""Figures / GIFs from the replay-demo outputs (demo\\out, see run_demo.ps1):
  paper\\figs\\demo_replay.{pdf,png}                           supplement figure (HUD frames + timelines)
  code\\report\\github_static\\docs\\assets\\demo_*.gif         README animations (small)"""
from pathlib import Path

import av
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r"D:\highwaymm")
OUT = ROOT / "demo" / "out"
FIG = ROOT / "paper" / "figs"
GH = ROOT / "code" / "report" / "github_static" / "docs" / "assets"
HOLD = 4  # video frames per observation (live_demo default)


def frames(mp4):
    c = av.open(str(mp4))
    fr = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    c.close()
    return fr[HOLD - 1::HOLD]  # one per observation


def gif(mp4, path, width=720, ms=700, every=1, max_frames=60):
    fr = frames(mp4)[::every][:max_frames]
    ims = [Image.fromarray(f).resize((width, int(width * f.shape[0] / f.shape[1])), Image.LANCZOS)
           .quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in fr]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=ms, loop=0, optimize=True)
    print(path, f"{path.stat().st_size / 2**20:.2f} MB, {len(ims)} frames")


def obs_index(events, cam, cond):
    e = events[events.camera == cam].reset_index(drop=True)
    return int(e.index[cond(e)][0])


def main():
    GH.mkdir(parents=True, exist_ok=True)
    gif(OUT / "nysdot" / "demo_nysdot_1878.mp4", GH / "demo_nysdot_1878.gif")
    # our footage: the snow clips back to back
    fr = []
    for c in ("Rec_0147", "Rec_0151", "Rec_0103"):
        fr += frames(OUT / "ours_winter" / f"demo_{c}.mp4")
    ims = [Image.fromarray(f).resize((720, 405), Image.LANCZOS).quantize(colors=96, method=Image.Quantize.MEDIANCUT) for f in fr]
    p = GH / "demo_ours_winter.gif"
    ims[0].save(p, save_all=True, append_images=ims[1:], duration=900, loop=0, optimize=True)
    print(p, f"{p.stat().st_size / 2**20:.2f} MB")

    # supplement figure
    ny = pd.read_csv(OUT / "nysdot" / "events.csv")
    i_ny = obs_index(ny, "nysdot_1878", lambda e: (e.level == 3) & (e.gt_level == 3))
    hud_ny = frames(OUT / "nysdot" / "demo_nysdot_1878.mp4")[i_ny]
    hud_ours = frames(OUT / "ours_winter" / "demo_Rec_0147.mp4")[3]
    tl = plt.imread(OUT / "nysdot" / "timeline_nysdot_1878.png")
    r_hud, r_tl = hud_ny.shape[0] / hud_ny.shape[1], tl.shape[0] / tl.shape[1]
    fig = plt.figure(figsize=(7.0, 3.5 * (r_hud + r_tl) + 0.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[r_hud, r_tl], hspace=0.12, wspace=0.03,
                          left=0, right=1, top=0.95, bottom=0)
    for k, (im, t) in enumerate([(hud_ny, "(a) NYSDOT blizzard, camera 1878"), (hud_ours, "(b) our winter footage (Rec 0147)")]):
        ax = fig.add_subplot(gs[0, k]); ax.imshow(im); ax.set_axis_off(); ax.set_title(t, fontsize=7, pad=2)
    for k, (png, t) in enumerate([(OUT / "nysdot" / "timeline_nysdot_1878.png", "(c) camera 1878, all levels"),
                                  (OUT / "floodcam_focus" / "timeline_ncdot_Buxton.png", "(d) FloodCam Buxton, flooding only")]):
        ax = fig.add_subplot(gs[1, k]); ax.imshow(plt.imread(png)); ax.set_axis_off(); ax.set_title(t, fontsize=7, pad=2)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"demo_replay.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("figure", FIG / "demo_replay.pdf")


if __name__ == "__main__":
    main()
