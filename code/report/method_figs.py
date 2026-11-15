"""Method figures from the released TaxoWarn-S checkpoint (real model outputs, no mock-ups).
  python code\\report\\method_figs.py            # compute (GPU) + plot
  python code\\report\\method_figs.py --plot     # re-plot from the cache
Main paper: paper\\figs\\method_pipeline.pdf -- one NYSDOT blizzard observation through every stage (input + overlay mask,
  frozen SigLIP 2 tokens, per-attribute taxonomy-query attention, calibrated attribute posteriors, hazard rule layer,
  taxonomy pushforward, warning system over the storm from the replay demo).
Supplement: paper\\figs\\query_attention.pdf -- where each taxonomy query looks, on examples from several sources.
Attention = last cross-attention layer of the query decoder, head-averaged, summed over the 3 frames' 6x6 tokens.
Frames are shown unmasked (masked regions outlined); the model input is the masked frame. In the main figure the NYSDOT
frame is cropped to the camera view for display (the white sensor-text header above it is masked in the input); image
and attention map are cropped identically. The supplement shows full frames."""
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, HAZARDS, HAZARD_LEVEL  # noqa: E402

ROOT = Path(r"D:\highwaymm")
REL = ROOT / "HighwayMM_release" / "frames"
FIG = ROOT / "paper" / "figs"
CACHE = ROOT / "experiments" / "report" / "method_figs_cache.pkl"
CKPT = ROOT / "release_models" / "taxowarn_s_highwaymm.pt"
EVENTS = ROOT / "demo" / "out" / "nysdot" / "events.csv"

MAIN = "nysdot_I_495_at_Terry_Road_(Exits_59_58)__Westbound__Skyline_1878_2022-01-29-16_10_04"
SUPP = [
    (MAIN, "NYSDOT blizzard"),
    ("hmm_Rec_0147", "Ours: snowfall, fog"),
    ("itssnap_its_129p11610_35p54906_20261025T214737KST", "Ours: rain, night"),
    ("floodcam_water_X_2020-09-20_08_45_44.376626-Mirlo", "FloodCam: flooding"),
    ("fvei_train_1-01814-59", "FVEI: dense fog"),
    ("norway_train-00001-of-00002_00440", "Norway: congestion"),
]
A = list(ATTRS)
ANAME = {"precip": "precipitation", "intensity": "intensity", "surface": "surface", "visibility": "visibility",
         "illumination": "illumination", "traffic": "traffic"}
import figstyle as _S  # noqa: E402
_S.apply()
LCOL = _S.LCOL
LNAME = ["normal", "advisory", "warning", "danger"]


def load(item):
    d = REL / item
    ms = sorted(d.glob("m*.jpg")); rs = sorted(d.glob("r*.jpg"))
    return [cv2.imread(str(p)) for p in ms], [cv2.imread(str(p)) for p in rs]


def compute():
    import torch
    from warning.live_demo import Pipeline, warn_probs_from_attrs
    pipe = Pipeline(str(CKPT))
    model = pipe.model
    att = {}
    mha = model.dec.layers[-1].multihead_attn
    orig = mha.forward

    def fwd(*a, **k):
        k["need_weights"] = True; k["average_attn_weights"] = True
        out = orig(*a, **k)
        att["w"] = out[1].detach().float().cpu().numpy()
        return out
    mha.forward = fwd
    gate = torch.sigmoid(model.gate).detach().cpu().numpy()
    res = {"gate": dict(zip(A, gate.tolist()))}
    for item, _ in SUPP:
        masked, raw = load(item)
        with torch.no_grad():
            out = model(pipe.features(masked))
        P = {}
        for a in A:
            lp = out["logits"][a].float().log_softmax(-1)
            P[a] = (lp / pipe.temps.get(f"p_{a}", 1.0)).softmax(-1)[0].cpu().numpy()
        pw = warn_probs_from_attrs({a: P[a][None] for a in A})[0]
        w = att["w"][0]                                   # [n_attr, 2 context + 3*36 patch tokens]
        sp = w[:, 2:].reshape(len(A), 3, 6, 6).sum(1)     # sum over the 3 frames
        mid = len(raw) // 2
        mask = (np.abs(raw[mid].astype(int) - masked[mid].astype(int)).sum(-1) > 0) & \
               (np.abs(masked[mid].astype(int) - 127).sum(-1) < 12)
        res[item] = dict(P=P, pw=pw, hz=out["hazard"][0].float().cpu().numpy(), att=sp, ctx=w[:, :2].sum(1),
                         raw=raw[mid], mask=mask)
        top = {a: ATTRS[a][int(P[a].argmax())] for a in A}
        print(item[:40], top, "P(level)", np.round(pw, 3), "ctx-attn", np.round(w[:, :2].sum(1), 2))
    print("gates", np.round(gate, 2))
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(CACHE, "wb"))
    return res


# ---------------------------------------------------------------------------------------------------------- plotting
def _rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def view_box(img):
    """camera view inside frames that carry a white text header (NYSDOT): largest block of non-white rows"""
    a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = (a < 235).mean(1) > 0.6
    if rows.sum() < 0.3 * len(rows):
        return 0, 0, img.shape[1], img.shape[0]
    idx = np.flatnonzero(rows)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    r = max(runs, key=len)
    cols = np.flatnonzero((a[r[0]:r[-1] + 1] < 235).mean(0) > 0.6)
    return int(cols[0]), int(r[0]), int(cols[-1]) + 1, int(r[-1]) + 1


def heat_up(img, a):
    H, W = img.shape[:2]
    return cv2.resize((a / a.max()).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1)


def _clean(ax, lw=0.4):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(lw)


def _show(ax, img, heat=None, mask=None, box=None, lw=0.7):
    x0, y0, x1, y1 = box or (0, 0, img.shape[1], img.shape[0])
    ax.imshow(_rgb(img[y0:y1, x0:x1]))
    if heat is not None:
        ax.imshow(heat[y0:y1, x0:x1], cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    if mask is not None and mask[y0:y1, x0:x1].any():
        ax.contour(mask[y0:y1, x0:x1].astype(float), levels=[0.5], colors="#ffd400", linewidths=lw, linestyles="--")
    _clean(ax)


def _stage(fig, x, y, txt, sub=None):
    fig.text(x, y, txt, fontsize=7, weight="bold", ha="left", va="bottom", color="#111111")
    if sub:
        fig.text(x, y - 0.006, sub, fontsize=6.2, ha="left", va="top", color="#333333", linespacing=1.15)


def _arrow(fig, p0, p1, conn="arc3"):
    from matplotlib.patches import FancyArrowPatch
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=8,
                                       lw=0.9, color="#555555", connectionstyle=conn))


def plot_main(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    r = res[MAIN]
    img, mask = r["raw"], r["mask"]
    box = view_box(img)
    bw, bh = box[2] - box[0], box[3] - box[1]
    FW, FH = 7.0, 3.6
    fig = plt.figure(figsize=(FW, FH))
    iw = 0.155                                   # image width (figure fraction); height keeps the aspect
    ih = iw * FW * bh / bw / FH
    y1 = 0.57                                    # bottom of row-1 images

    # 1 observation (shown with header for the mask outline: full frame, small)
    ax = fig.add_axes([0.0, y1, iw, ih])
    _show(ax, img, mask=mask, box=box)
    _stage(fig, 0.0, 0.935, "1  Observation", "10 s clip, 8 frames;\noverlay text masked (dashed)")
    _arrow(fig, (iw + 0.004, y1 + ih / 2), (iw + 0.03, y1 + ih / 2))

    # 2 frozen encoder: 3 stacked frames + token grid
    x2 = iw + 0.035
    for k, off in enumerate((0.016, 0.008, 0.0)):
        ax = fig.add_axes([x2 + off, y1 + off * 1.8, iw * 0.92, ih * 0.92])
        _show(ax, img, box=box)
        if k < 2:
            ax.images[0].set_alpha(0.5)
        else:
            for i in range(1, 6):
                ax.axvline(i * bw / 6, color="white", lw=0.5); ax.axhline(i * bh / 6, color="white", lw=0.5)
    _stage(fig, x2, 0.935, "2  Frozen SigLIP 2", "one pass, 3 frames:\n3\u00d76\u00d76 tokens + global $g$")
    _arrow(fig, (x2 + iw + 0.022, y1 + ih / 2), (x2 + iw + 0.045, y1 + ih / 2))

    # 3 taxonomy-query attention maps
    x3 = x2 + iw + 0.05
    tw = 0.098
    th = tw * FW * bh / bw / FH
    for j, a in enumerate(A):
        cx = x3 + (j % 3) * (tw + 0.005)
        cy = y1 + ih - th - (j // 3) * (th + 0.035) + 0.0
        ax = fig.add_axes([cx, cy, tw, th])
        _show(ax, img, heat=heat_up(img, r["att"][j]), box=box)
        ax.set_title(ANAME[a], fontsize=6.3, pad=1.2)
    _stage(fig, x3, 0.935, "3  Taxonomy queries", "one learnable query per attribute (2-layer decoder);\nits cross-attention over the tokens is shown")
    x4l = x3 + 3 * (tw + 0.005) + 0.004
    _arrow(fig, (x4l, y1 + ih / 2), (x4l + 0.03, y1 + ih / 2))

    # 4 attribute posteriors: top class dark, others light
    x4 = x4l + 0.105
    ax = fig.add_axes([x4, y1 - 0.02, 0.995 - x4, ih + 0.02])
    for i, a in enumerate(A):
        p = r["P"][a]; c = int(p.argmax())
        left = 0.0
        for k in np.argsort(-p):
            ax.barh(i, p[k], left=left, color="#1f5fa8" if k == c else "#c9d6e6", height=0.7, ec="white", lw=0.4)
            left += p[k]
        ax.text(0.03, i, f"{ATTRS[a][c].replace('_', '-')} {p[c]:.2f}", va="center", ha="left", fontsize=6.1, color="white",
                weight="bold")
    ax.set_yticks(range(len(A))); ax.set_yticklabels([ANAME[a] for a in A], fontsize=6.2)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"]); ax.tick_params(labelsize=5.9, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _stage(fig, x4l + 0.035, 0.935, "4  Attribute posteriors $p_a$",
           "gate $\\gamma_a$ mixes query and global\nheads; temperature-calibrated")

    # wrap arrow to row 2
    _arrow(fig, (0.80, 0.462), (0.012, 0.462))

    # 5 hazard rule layer
    yb, hb = 0.075, 0.245
    ax = fig.add_axes([0.08, yb, 0.19, hb])
    hz = r["hz"]; order = np.argsort(-hz)
    ax.barh(range(len(HAZARDS)), hz[order], color=[LCOL[HAZARD_LEVEL[HAZARDS[k]]] for k in order], height=0.72)
    ax.set_yticks(range(len(HAZARDS))); ax.set_yticklabels([HAZARDS[k].replace("_", " ") for k in order], fontsize=6)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xticks([0, 0.5, 1]); ax.set_xticklabels(["0", ".5", "1"]); ax.tick_params(labelsize=5.9, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _stage(fig, 0.0, 0.40, "5  Hazard rule layer $p_h$",
           "rules, e.g. blizzard = snow AND heavy AND fog;\nbar colour = base warning level")
    _arrow(fig, (0.285, yb + hb / 2), (0.32, yb + hb / 2))

    # 6 pushforward
    ax = fig.add_axes([0.345, yb, 0.13, hb])
    ax.bar(range(4), r["pw"], color=LCOL, width=0.72)
    for i, v in enumerate(r["pw"]):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=5.9)
    ax.set_xticks(range(4)); ax.set_xticklabels(["norm.", "adv.", "warn.", "dang."], fontsize=6)
    ax.set_ylim(0, 1.15); ax.set_yticks([0, 0.5, 1]); ax.tick_params(labelsize=5.9, length=2, pad=1)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _stage(fig, 0.33, 0.40, "6  Pushforward $P$(level)", "exact sum over all 2,160\nattribute combinations")
    _arrow(fig, (0.49, yb + hb / 2), (0.525, yb + hb / 2))

    # 7 warning system over the storm (replay demo events of the same camera)
    ev = pd.read_csv(EVENTS)
    ev = ev[ev.camera == "nysdot_1878"].reset_index(drop=True)
    n = len(ev)
    x7, w7 = 0.585, 0.41
    a_is = fig.add_axes([x7, yb + hb - 0.035, w7, 0.035])
    a_gt = fig.add_axes([x7, yb + hb - 0.075, w7, 0.035])
    a_p = fig.add_axes([x7, yb, w7, hb - 0.085])
    for axx, col, lab in ((a_is, ev.level, "issued"), (a_gt, ev.gt_level, "human")):
        for i, lv in enumerate(col):
            if lv == lv:
                axx.axvspan(i - 0.5, i + 0.5, color=LCOL[int(lv)], lw=0)
        axx.set_xlim(-0.5, n - 0.5); _clean(axx, 0.3)
        axx.set_ylabel(lab, fontsize=5.9, rotation=0, ha="right", va="center", labelpad=2)
    e = ev.index[ev.early.astype(str) == "True"]
    if len(e):
        a_is.scatter(e, np.full(len(e), 1.35), marker="v", s=7, c="#1f5fa8", transform=a_is.get_xaxis_transform(),
                     clip_on=False)
    p3 = ev.p_level.apply(lambda s: float(s.strip("[]").split(",")[3]))
    a_p.plot(np.arange(n), p3, color="black", lw=0.7)
    a_p.axhline(0.5, color="black", lw=0.4, ls=":")
    a_p.text(n - 0.5, 0.52, "$\\tau$", fontsize=6.1, ha="right", va="bottom")
    resets = ev.index[ev.reset.astype(str) == "True"]
    for rs in resets:
        for axx in (a_is, a_gt, a_p):
            axx.axvline(rs - 0.5, color="white" if axx is not a_p else "#999999", lw=0.8, ls="-" if axx is not a_p else "--")
    a_p.set_xlim(-0.5, n - 0.5); a_p.set_ylim(0, 1.02)
    a_p.set_yticks([0, 0.5, 1]); a_p.tick_params(labelsize=5.9, length=2, pad=1)
    a_p.set_ylabel("$P$(danger)", fontsize=6, labelpad=1)
    t0, t1 = pd.to_datetime(ev.ts.iloc[0]), pd.to_datetime(ev.ts.iloc[-1])
    a_p.set_xlabel(f"{n} observations of camera 1878, {t0:%d %b} \u2013 {t1:%d %b} 2022 (dashed: stream gap)",
                   fontsize=6, labelpad=1)
    _stage(fig, 0.545, 0.40, "7  Warning system over the storm",
           "issue level $L$ if $P(y{\\geq}L){>}\\tau$; CUSUM early advisory (\u25bc);\nhysteresis: de-escalate after 2 lower observations")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(LCOL, LNAME)], loc="lower right", ncol=2, fontsize=5.9,
               frameon=False, handlelength=0.9, columnspacing=0.6, handletextpad=0.3, labelspacing=0.2,
               bbox_to_anchor=(0.998, 0.395))

    for ext, dpi in (("pdf", 300), ("png", 220)):
        fig.savefig(FIG / f"method_pipeline.{ext}", dpi=dpi)
    plt.close(fig)
    print("->", FIG / "method_pipeline.pdf")


def plot_supp(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(SUPP)
    fig, axes = plt.subplots(n, 7, figsize=(7.0, 0.95 * n + 0.2))
    for i, (item, name) in enumerate(SUPP):
        r = res[item]
        img = r["raw"]
        _show(axes[i, 0], img, mask=r["mask"], lw=0.5)
        axes[i, 0].set_ylabel(name, fontsize=6.5)
        for j, a in enumerate(A):
            p = r["P"][a]; c = int(p.argmax())
            _show(axes[i, j + 1], img, heat=heat_up(img, r["att"][j]))
            axes[i, j + 1].set_xlabel(f"{ATTRS[a][c].replace('_', '-')} {p[c]:.2f}", fontsize=6.1, labelpad=1)
        if i == 0:
            axes[i, 0].set_title("frame (input masked)", fontsize=6.9, pad=2)
            for j, a in enumerate(A):
                axes[i, j + 1].set_title(f"{ANAME[a]} ($\\gamma$={res['gate'][a]:.2f})", fontsize=6.9, pad=2)
    fig.tight_layout(pad=0.2, w_pad=0.15, h_pad=0.25)
    for ext, dpi in (("pdf", 300), ("png", 200)):
        fig.savefig(FIG / f"query_attention.{ext}", dpi=dpi)
    plt.close(fig)
    print("->", FIG / "query_attention.pdf")


if __name__ == "__main__":
    res = pickle.load(open(CACHE, "rb")) if "--plot" in sys.argv else compute()
    plot_main(res)
    plot_supp(res)
