"""HighwayMM replay demo: run the full road-hazard warning pipeline on recorded footage as if it were arriving live.

    video / image files  ->  8 frames per observation  ->  overlay-text masking (OCR, as in training)
    ->  SigLIP 2 (frozen)  ->  TaxoWarn-S (6 attributes, 9 hazards, calibrated)  ->  warning system
    (thresholds, early advisory, hysteresis)  ->  console feed, events.csv, timelines, annotated video, HTML report

Inputs (any mix of files and folders):
  * a folder of per-camera sub-folders      demo_in/<camera>/<files>      camera = sub-folder name
  * a flat folder or single files           camera = --camera, or one camera per file with --camera per_file
  * long videos are cut into --window-second observations (default 10 s, the clip length of HighwayMM);
    a video no longer than 1.5 windows is one observation (identical frame sampling to training)
Timestamps are parsed from file names (20261025T164923, 2022-01-29-16_10_04, 2020-09-20_08_45_44, ISO, ...); files
without one are ordered by name and spaced --interval seconds apart. Window k of a video is at file time + k*window.

Modes:
  replay   (default)  process everything in time order; --speed N paces it at N x real time (0 = as fast as possible)
  watch    --watch    poll a drop folder and process each new file once it has finished being written. This works with
                      any external recorder; the demo itself never connects to a camera or stream.

Examples:
  python code\\warning\\live_demo.py --input demo_in\\nysdot --gt demo_in\\nysdot_gt.csv --out demo_out\\nysdot
  python code\\warning\\live_demo.py --input my_dashcam.mp4 --camera phone --window 5 --out demo_out\\phone --speed 10
  python code\\warning\\live_demo.py --watch D:\\drop --out demo_out\\watch

Outputs in --out: events.csv / events.jsonl (one row per observation), timeline_<camera>.png, demo_<camera>.mp4 (and .gif
with --gif), report.html (self-contained), run.json (settings, model, timing).
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prep_frames import read_frames  # noqa: E402
from data.taxonomy import ATTRS, HAZARDS, LEVEL_NAMES, warn_probs_from_attrs  # noqa: E402
from models.taxowarn_s import TaxoWarnS  # noqa: E402
from warning.system import WarnConfig, WarningSystem  # noqa: E402

VID = (".mp4", ".avi", ".mkv", ".mov", ".ts", ".m4v", ".webm", ".mpg", ".mpeg", ".flv")
IMG = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
K, OUT_W = 8, 640
LOCAL_ENCODER = Path(r"D:\highwaymm\models\SigLIP2-SO400M-384")
HUB_ENCODER = "hf-hub:timm/ViT-SO400M-16-SigLIP2-384"
LEVEL_RGB = [(46, 160, 67), (230, 184, 0), (240, 120, 20), (214, 39, 40)]   # normal / advisory / warning / danger
ATTR_SHORT = {"precip": "precipitation", "intensity": "intensity", "surface": "road surface", "visibility": "visibility",
              "illumination": "light", "traffic": "traffic"}


# ----------------------------------------------------------------------------------------------- inputs and timestamps
_TS_PATTERNS = [
    (re.compile(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})"), None),
    (re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})[-_ T](\d{1,2})[-_:.](\d{2})[-_:.](\d{2})"), None),
    (re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})(?!\d)"), None),
]


def ts_from_name(name):
    for rx, _ in _TS_PATTERNS:
        m = rx.search(name)
        if m:
            try:
                return datetime(*map(int, m.groups()))
            except ValueError:
                pass
    return None


def discover(inputs, camera=None):
    """-> list of dict(path, camera, ts|None)"""
    out = []
    for inp in inputs:
        p = Path(inp)
        if p.is_file():
            files, base = [p], p.parent
        elif p.is_dir():
            files, base = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in VID + IMG), p
        else:
            print(f"[warn] not found: {p}")
            continue
        for f in files:
            rel = f.relative_to(base)
            if camera == "per_file":
                cam = f.stem
            elif camera:
                cam = camera
            elif len(rel.parts) > 1:
                cam = rel.parts[0]
            else:
                cam = p.stem if p.is_dir() else "camera"
            out.append({"path": f, "camera": cam, "ts": ts_from_name(f.name)})
    return out


def assign_times(items, interval):
    """files without a parsable timestamp: name order within the camera, spaced `interval` seconds"""
    by = {}
    for it in items:
        by.setdefault(it["camera"], []).append(it)
    t0 = datetime(2000, 1, 1)
    for cam, lst in by.items():
        lst.sort(key=lambda d: (d["ts"] or t0, d["path"].name))
        for j, it in enumerate(lst):
            it["ts_source"] = "filename" if it["ts"] else "order"
            if it["ts"] is None:
                it["ts"] = t0 + timedelta(seconds=interval * j)
    return sorted(items, key=lambda d: (d["ts"], d["camera"], d["path"].name))


def _to_bgr(fr):
    try:
        return fr.to_ndarray(format="bgr24")
    except Exception:  # same fallback as prep_frames.read_frames
        return cv2.cvtColor(fr.to_ndarray(), cv2.COLOR_YUV2BGR_I420)


def video_duration(path):
    try:
        c = av.open(str(path))
        s = c.streams.video[0]
        d = float(c.duration) / 1e6 if c.duration else (float(s.duration * s.time_base) if s.duration else 0.0)
        c.close()
        return d
    except Exception:
        return 0.0


def observations(item, window):
    """yield (offset_seconds, list of BGR frames) for one input file"""
    p = item["path"]
    if p.suffix.lower() in IMG:
        fr = read_frames(str(p), K)
        if fr:
            yield 0.0, fr
        return
    dur = video_duration(p)
    if window <= 0 or dur <= 1.5 * window:
        fr = read_frames(str(p), K)  # whole clip, exactly as in training
        if fr:
            yield 0.0, fr
        return
    # long video: stream-decode and take K evenly spaced frames inside each window (no full decode into memory)
    c = av.open(str(p))
    s = c.streams.video[0]
    s.thread_type = "AUTO"
    w, got = 0, []
    tgt = [window * (j + 1) / (K + 1) for j in range(K)]
    for fr in c.decode(s):
        if fr.time is None:
            continue
        t = fr.time - w * window
        while len(got) < K and t >= tgt[len(got)]:
            got.append(_to_bgr(fr))
        if len(got) == K:
            yield w * window, got
            w, got = w + 1, []
    c.close()
    if len(got) >= 3:  # last partial window
        yield w * window, got + [got[-1]] * (K - len(got))


# --------------------------------------------------------------------------------------------- masking (as in training)
class Masker:
    """Same procedure as data/prep_frames.py: frames resized to 640 px width, OCR (EasyOCR ko+en) on the first and
    middle frame at 2x, boxes dilated, plus the top band of operator layouts; masked pixels set to grey 127.
    layout (the training rule of each source):
      generic = OCR boxes only (all external sources: NYSDOT, FloodCam, FVEI, ..., and any new footage)
      its     = OCR boxes + top 9% band always (Korean ITS / KEC snapshot and live layouts)
      anytext = OCR boxes + top band if OCR finds any text (our winter-pilot recordings)"""

    def __init__(self, mode="ocr", layout="generic", gpu=True):
        self.mode, self.layout = mode, layout
        self.reader = None
        if mode == "ocr":
            import easyocr
            store = Path(r"D:\highwaymm\analysis\.easyocr")
            kw = {"model_storage_directory": str(store)} if store.exists() else {}
            self.reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False, **kw)

    def __call__(self, fr):
        H, W = fr[0].shape[:2]
        mask = np.zeros((H, W), np.uint8)
        any_text = False
        if self.reader is not None:
            for f in ([fr[0]] if len(fr) == 1 else [fr[0], fr[len(fr) // 2]]):
                big = cv2.resize(f, (W * 2, H * 2), interpolation=cv2.INTER_CUBIC)
                for bb, _txt, _conf in self.reader.readtext(big, text_threshold=0.5, low_text=0.3):
                    pts = np.array(bb) / 2
                    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
                    pad = 4 + 0.25 * (y1 - y0)
                    cv2.rectangle(mask, (int(x0 - pad), int(y0 - pad)), (int(x1 + pad), int(y1 + pad)), 255, -1)
                    any_text = True
        if self.layout == "its" or (self.layout == "anytext" and any_text):
            mask[: int(0.09 * H)] = 255
        return mask


def _jpeg(img, q=90):
    """JPEG round trip: training frames were stored as JPEG (quality 90) before encoding"""
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def prepare(frames, masker):
    h, w = frames[0].shape[:2]
    s = OUT_W / w
    fr = [cv2.resize(f, (OUT_W, int(round(h * s))), interpolation=cv2.INTER_AREA) for f in frames]
    mask = masker(fr)
    raw, masked = [], []
    for f in fr:
        m = f.copy()
        m[mask > 0] = (127, 127, 127)
        raw.append(_jpeg(f)); masked.append(_jpeg(m))
    return raw, masked, mask


# ----------------------------------------------------------------------------------------------------------- the model
class Pipeline:
    def __init__(self, checkpoint, encoder=None, device=None, cfg=WarnConfig()):
        import open_clip
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dt = torch.float16 if self.dev.type == "cuda" else torch.float32
        enc_id = encoder or (f"local-dir:{LOCAL_ENCODER.as_posix()}" if LOCAL_ENCODER.exists() else HUB_ENCODER)
        enc, _, self.pre = open_clip.create_model_and_transforms(enc_id)
        self.visual = enc.visual.to(self.dev).eval().to(self.dt)
        del enc
        ck = self._load(checkpoint)
        sd = {k: v.float() for k, v in ck["state_dict"].items()}
        mc = ck["model_cfg"]
        keys = ("d", "enc_layers", "dec_layers", "dropout", "use_queries", "use_spatial", "use_text", "use_hazard_layer",
                "hierarchical", "use_global_residual", "use_dual")
        protos = {a: np.zeros(tuple(sd[f"txt_{a}"].shape), np.float32) for a in ATTRS} if f"txt_{A0}" in sd else None
        self.model = TaxoWarnS(text_protos=protos, **{k: mc[k] for k in keys if k in mc})
        self.model.load_state_dict(sd)
        self.model.to(self.dev).eval()
        self.temps = ck["temperatures"]
        self.meta = {"run_id": ck.get("run_id"), "encoder": enc_id, "device": str(self.dev), "k": ck.get("k", 1)}
        self.ws = WarningSystem(cfg)

    @staticmethod
    def _load(path):
        p = Path(path)
        if p.is_dir():  # a training run directory
            import yaml
            ck = torch.load(p / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
            cfg = yaml.safe_load(open(p / "config.yaml", encoding="utf-8"))
            return {"state_dict": ck["model"], "model_cfg": cfg["model"], "k": cfg["data"].get("k", 8), "run_id": p.name,
                    "temperatures": json.loads((p / "temperatures.json").read_text(encoding="utf-8"))}
        return torch.load(p, map_location="cpu", weights_only=False)

    @torch.no_grad()
    def features(self, masked):
        """single-frame TaxoWarn-S input (k=1): global embedding of the middle frame + 6x6-pooled patch tokens of frames
        1, 4, 6 (one trunk pass; identical to features/siglip2 + features/siglip2_spatial used in training)"""
        fr = masked + [masked[-1]] * (K - len(masked))
        x = torch.stack([self.pre(Image.fromarray(fr[i][:, :, ::-1])) for i in (1, 4, 6)]).to(self.dev, self.dt)
        tk = self.visual.trunk.forward_features(x)
        g = F.normalize(self.visual.head(self.visual.trunk.forward_head(tk[1:2])).float(), dim=-1)
        gs = int(tk.shape[1] ** 0.5)
        tp = F.adaptive_avg_pool2d(tk[:, -gs * gs:].transpose(1, 2).reshape(3, -1, gs, gs).float(), 6)
        tp = tp.flatten(2).transpose(1, 2).reshape(1, 108, -1)
        # cached training features were stored in fp16: round the same way
        return torch.cat([g.half().float()[:, None], torch.zeros(1, 7, g.shape[-1], device=g.device), tp.half().float()], 1)

    @torch.no_grad()
    def predict(self, masked):
        out = self.model(self.features(masked))
        P = {}
        for a in ATTRS:
            lp = out["logits"][a].float().log_softmax(-1)
            P[a] = (lp / self.temps.get(f"p_{a}", 1.0)).softmax(-1)[0].cpu().numpy()  # post-hoc temperature (val-fitted)
        pw = warn_probs_from_attrs({a: P[a][None] for a in ATTRS})[0]
        hz = out["hazard"][0].float().cpu().numpy()
        onset = float(torch.sigmoid(out["onset_logit"][0]))
        return P, pw, hz, onset

    def step(self, camera, ts, masked, reset_gap_min=0.0, focus=None):
        P, pw, hz, onset = self.predict(masked)
        if focus:  # hazard-specific warning, as in the paper's warning evaluation (eval_warning.py)
            s = float(P["surface"][ATTRS["surface"].index(FOCUS_SURFACE[focus])]) if focus in FOCUS_SURFACE \
                else float(hz[HAZARDS.index(focus)])
            pw_d, hz_d, onset = np.array([1 - s, 0.0, s, 0.0]), np.array([s]), 0.0
        else:
            pw_d, hz_d = pw, hz
        st = self.ws.cams.get(camera)
        reset = bool(st and st.history and reset_gap_min > 0
                     and (ts - st.history[-1]["ts"]).total_seconds() > 60 * reset_gap_min)
        if reset:  # stream gap: start the camera's state afresh (hysteresis / CUSUM do not carry over a long outage)
            del self.ws.cams[camera]
        ev = self.ws.step(camera, ts, pw_d, hz_d, onset)
        ev["reset"] = reset
        if focus:
            ev["hazard"] = focus
        return P, pw_d, hz, ev


A0 = list(ATTRS)[0]
FOCUS_SURFACE = {"flooding": "water", "snow_cover": "snow"}  # scored by the surface posterior, as in eval_warning.py


# ------------------------------------------------------------------------------------------------------------ rendering
def _font(size, bold=False):
    import matplotlib
    from matplotlib import font_manager
    try:
        return ImageFont.truetype(font_manager.findfont(f"DejaVu Sans{':bold' if bold else ''}"), size)
    except Exception:
        try:
            return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"), size)
        except Exception:
            return ImageFont.load_default()


class HUD:
    """1280x720 frame: camera image with level border (left), attribute / hazard / level panel (right),
    level timeline of this camera so far (bottom)."""
    W, H = 1280, 720

    def __init__(self, gt_cap=3):
        self.gt_cap = gt_cap
        self.f = {s: _font(s) for s in (13, 15, 18, 22)}
        self.fb = {s: _font(s, True) for s in (15, 18, 26)}

    def render(self, img_bgr, ev, P, pw, hz, hist, gt_hist=None, mask=None):
        W, H = self.W, self.H
        c = Image.new("RGB", (W, H), (22, 24, 28))
        d = ImageDraw.Draw(c)
        L = int(ev["level"])
        col = LEVEL_RGB[L]
        # camera image (unmasked by default; optional outline of masked overlay text)
        im = img_bgr[:, :, ::-1]
        if mask is not None:
            cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            im = np.ascontiguousarray(im)
            cv2.drawContours(im, cnt, -1, (0, 200, 255), 1)
        pil = Image.fromarray(im)
        bw, bh = 850, 478
        pil.thumbnail((bw, bh))
        x0, y0 = 16, 60
        d.rectangle([x0 - 6, y0 - 6, x0 + pil.width + 5, y0 + pil.height + 5], fill=col)
        c.paste(pil, (x0, y0))
        # header
        d.text((16, 14), str(ev["camera"])[:60], font=self.fb[18], fill=(235, 235, 235))
        d.text((16, 38), ev["ts_str"], font=self.f[13], fill=(170, 170, 170))
        tag = f"LEVEL {L}  {LEVEL_NAMES[L].upper()}"
        tw = d.textlength(tag, font=self.fb[26])
        d.rounded_rectangle([x0 + bw - tw - 30, 10, x0 + bw, 52], 8, fill=col)
        d.text((x0 + bw - tw - 15, 15), tag, font=self.fb[26], fill=(255, 255, 255) if L != 1 else (20, 20, 20))
        if ev.get("early"):
            d.rounded_rectangle([x0 + bw - tw - 210, 16, x0 + bw - tw - 40, 46], 6, outline=(230, 184, 0), width=2)
            d.text((x0 + bw - tw - 200, 21), "early advisory", font=self.f[15], fill=(230, 184, 0))
        # right panel: attributes
        px, py = 900, 16
        d.text((px, py), "Attributes (calibrated)", font=self.fb[15], fill=(235, 235, 235))
        py += 28
        for a, vs in ATTRS.items():
            p = P[a]
            j = int(np.argmax(p))
            d.text((px, py), ATTR_SHORT[a], font=self.f[13], fill=(160, 160, 160))
            d.text((px + 118, py - 2), f"{vs[j].replace('_', ' ')}", font=self.fb[15], fill=(240, 240, 240))
            d.text((px + 318, py), f"{p[j]:.2f}", font=self.f[13], fill=(200, 200, 200))
            # distribution bar
            bx, by, bwid = px, py + 19, 360
            acc = 0.0
            for k, v in enumerate(p):
                x1 = bx + acc * bwid
                acc += float(v)
                shade = 235 if k == j else 90 + 25 * (k % 3)
                d.rectangle([x1, by, bx + acc * bwid, by + 7], fill=(shade, shade, shade) if k != j else (120, 180, 255))
            py += 44
        # hazards
        py += 4
        d.text((px, py), "Hazards", font=self.fb[15], fill=(235, 235, 235))
        py += 24
        order = np.argsort(-hz)[:4]
        for k in order:
            v = float(hz[k])
            d.text((px, py), HAZARDS[k].replace("_", " "), font=self.f[13], fill=(200, 200, 200))
            d.rectangle([px + 130, py + 3, px + 130 + 200, py + 13], fill=(55, 58, 64))
            d.rectangle([px + 130, py + 3, px + 130 + 200 * v, py + 13], fill=(240, 120, 20) if v > 0.5 else (150, 150, 150))
            d.text((px + 338, py), f"{v:.2f}", font=self.f[13], fill=(200, 200, 200))
            py += 20
        # level distribution
        py += 8
        d.text((px, py), "P(warning level)", font=self.fb[15], fill=(235, 235, 235))
        py += 24
        for k in range(4):
            d.text((px, py), LEVEL_NAMES[k], font=self.f[13], fill=(200, 200, 200))
            d.rectangle([px + 130, py + 3, px + 330, py + 13], fill=(55, 58, 64))
            d.rectangle([px + 130, py + 3, px + 130 + 200 * float(pw[k]), py + 13], fill=LEVEL_RGB[k])
            d.text((px + 338, py), f"{pw[k]:.2f}", font=self.f[13], fill=(200, 200, 200))
            py += 20
        # timeline (this camera so far)
        ty0, ty1, tx0, tx1 = 570, 700, 90, 1260
        d.text((16, 552), "Warning level over time (system" + (", dots = human label)" if gt_hist else ")"),
               font=self.f[13], fill=(170, 170, 170))
        for k in range(4):
            yy = ty1 - (k + 0.5) * (ty1 - ty0) / 4
            d.text((16, yy - 8), LEVEL_NAMES[k], font=self.f[13], fill=LEVEL_RGB[k])
            d.line([tx0, yy, tx1, yy], fill=(45, 48, 54))
        n = max(len(hist), 2)
        span = max(n, 30)
        for i, lv in enumerate(hist):
            xa = tx0 + (tx1 - tx0) * i / span
            xb = tx0 + (tx1 - tx0) * (i + 1) / span
            yy = ty1 - (lv + 0.5) * (ty1 - ty0) / 4
            d.rectangle([xa, yy - 5, xb - 1, yy + 5], fill=LEVEL_RGB[lv])
            if gt_hist and gt_hist[i] is not None and gt_hist[i] >= 0:
                yg = ty1 - (min(gt_hist[i], self.gt_cap) + 0.5) * (ty1 - ty0) / 4
                cx = (xa + xb) / 2
                d.ellipse([cx - 3, yg - 3, cx + 3, yg + 3], outline=(255, 255, 255), width=1)
        return np.asarray(c)[:, :, ::-1]


class VideoOut:
    """H.264 mp4 via PyAV (plays in browsers); one per camera"""

    def __init__(self, path, fps):
        self.c = av.open(str(path), "w")
        self.s = self.c.add_stream("libx264", rate=fps)
        self.s.width, self.s.height, self.s.pix_fmt = HUD.W, HUD.H, "yuv420p"
        self.s.options = {"crf": "23", "preset": "veryfast"}

    def write(self, bgr, n=1):
        fr = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        for _ in range(n):
            for pkt in self.s.encode(fr):
                self.c.mux(pkt)

    def close(self):
        for pkt in self.s.encode():
            self.c.mux(pkt)
        self.c.close()


def timeline_png(cam, rows, path, gt_cap=3):
    """gt_cap: in --focus mode the system issues at most 'warning', so human hazard labels are drawn at that level"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # x = observation index (observations can be minutes or days apart); dashed lines = state reset after a gap
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.step(x, [r["level"] for r in rows], where="post", lw=2, color="#d62728", label="system warning level")
    g = [(i, min(r["gt_level"], gt_cap)) for i, r in enumerate(rows) if r.get("gt_level", -1) >= 0]
    if g:
        ax.scatter([i for i, _ in g], [y for _, y in g], s=14, c="k", zorder=3, label="human label")
    ax.plot(x, [3 * r["hazard_score"] for r in rows], alpha=0.5, color="#1f77b4", label="max hazard score (x3)")
    for i, r in enumerate(rows):
        if r.get("reset"):
            ax.axvline(i - 0.5, ls="--", lw=0.7, color="0.6")
    tick = np.unique(np.linspace(0, len(rows) - 1, min(6, len(rows))).round().astype(int))
    ax.set_xticks(tick)
    ax.set_xticklabels([f"{rows[i]['ts']:%m-%d}\n{rows[i]['ts']:%H:%M}" for i in tick], fontsize=7)
    ax.set_xlabel("observation (dashed: gap, camera state reset)" if any(r.get("reset") for r in rows) else "observation",
                  fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels(LEVEL_NAMES); ax.set_ylim(-0.3, 3.3)
    ax.set_title(cam, fontsize=9); ax.legend(fontsize=6, loc="upper left"); fig.tight_layout()
    fig.savefig(path, dpi=130); plt.close(fig)


def _b64(img_bgr, w=320, q=72):
    h = int(img_bgr.shape[0] * w / img_bgr.shape[1])
    ok, buf = cv2.imencode(".jpg", cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def write_report(out, rows, thumbs, meta):
    css = """body{font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:0;background:#f6f7f9;color:#1d2125}
main{max-width:1200px;margin:auto;padding:20px 16px}h1{font-size:22px}h2{font-size:17px;margin-top:28px}
.meta{color:#5b6470}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.card{background:#fff;border-radius:8px;overflow:hidden;border-top:6px solid var(--c);box-shadow:0 1px 2px #0002}
.card img{width:100%;display:block}.card div{padding:6px 8px;font-size:12px}.lv{font-weight:600;color:var(--c)}
.tl{width:100%;max-width:1000px;background:#fff;border-radius:8px}
@media (prefers-color-scheme:dark){body{background:#15171a;color:#e6e6e6}.card{background:#1f2226}.meta{color:#9aa3ad}}"""
    hexc = ["#2ea043", "#e6b800", "#f07814", "#d62728"]
    h = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>HighwayMM replay</title><style>{css}</style></head><body><main><h1>HighwayMM replay demo</h1>"
         f"<p class='meta'>model {meta.get('run_id')} &middot; {len(rows)} observations &middot; "
         f"{len({r['camera'] for r in rows})} cameras &middot; generated {datetime.now():%Y-%m-%d %H:%M}</p>"]
    for cam in dict.fromkeys(r["camera"] for r in rows):
        rr = [r for r in rows if r["camera"] == cam]
        h.append(f"<h2>{cam}</h2>")
        tl = out / f"timeline_{_safe(cam)}.png"
        if tl.exists():
            h.append(f"<img class='tl' alt='timeline' src='data:image/png;base64,{base64.b64encode(tl.read_bytes()).decode()}'>")
        h.append("<div class='grid'>")
        for r in rr:
            L = r["level"]
            top = ", ".join(f"{a}: {r['top_' + a].replace('_', ' ')}" for a in ("precip", "surface", "visibility"))
            gt = f" &middot; human {r['gt_level']}" if r.get("gt_level", -1) >= 0 else ""
            h.append(f"<div class='card' style='--c:{hexc[L]}'><img alt='' src='data:image/jpeg;base64,{thumbs[r['obs_id']]}'>"
                     f"<div><span class='lv'>{L} {LEVEL_NAMES[L]}</span>{' &middot; early' if r['early'] else ''}{gt}<br>"
                     f"{r['ts_str']}<br>{top}<br>top hazard {r['hazard'].replace('_', ' ')} {r['hazard_score']:.2f}</div></div>")
        h.append("</div>")
    h.append("</main></body></html>")
    (out / "report.html").write_text("".join(h), encoding="utf-8")


def _safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))[:80]


# ----------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="*", default=[], help="files and/or folders")
    ap.add_argument("--watch", default="", help="drop folder to poll for new files (live-style)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=r"D:\highwaymm\release_models\taxowarn_s_highwaymm.pt",
                    help="exported checkpoint (export_model.py) or a training run directory")
    ap.add_argument("--encoder", default=None, help="open_clip id; default local SigLIP2 dir or " + HUB_ENCODER)
    ap.add_argument("--camera", default=None, help="camera name for flat inputs, or 'per_file'")
    ap.add_argument("--window", type=float, default=10.0, help="seconds per observation for long videos (0 = whole file)")
    ap.add_argument("--interval", type=float, default=60.0, help="spacing (s) for files without a timestamp in the name")
    ap.add_argument("--mask", default="ocr", choices=["ocr", "none"])
    ap.add_argument("--layout", default="generic", choices=["generic", "its", "anytext"],
                    help="overlay rule: generic = OCR boxes (external / new footage), its = + top band (Korean ITS/KEC), "
                         "anytext = + top band when any text is found (HighwayMM winter pilot)")
    ap.add_argument("--gt", default="", help="optional CSV with columns file,gt_level (human warning level) for overlays")
    ap.add_argument("--speed", type=float, default=0.0, help="replay at N x real time (0 = as fast as possible)")
    ap.add_argument("--max_sleep", type=float, default=3.0)
    ap.add_argument("--poll", type=float, default=2.0)
    ap.add_argument("--show", default="raw", choices=["raw", "masked", "raw+outline"],
                    help="what the video/report show: unmasked frame (default), the masked model input, or outline of masks")
    ap.add_argument("--fps", type=int, default=4)
    ap.add_argument("--hold", type=int, default=4, help="video frames per observation")
    ap.add_argument("--no_video", action="store_true")
    ap.add_argument("--gif", action="store_true", help="also write a small GIF per camera (for READMEs)")
    ap.add_argument("--tau", type=float, default=0.5, help="level threshold tau (same for all levels)")
    ap.add_argument("--focus", default=None, choices=HAZARDS,
                    help="hazard-specific warning on one hazard (flooding / snow_cover use the surface posterior), as in "
                         "the paper's warning table; use the paper's operating points --tau 0.9 (flooding), 0.7 (snow_cover)")
    ap.add_argument("--reset_gap", type=float, default=180.0,
                    help="minutes without an observation after which a camera's state is reset (0 = never, as in the "
                         "paper's sequence evaluation)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not a.input and not a.watch:
        ap.error("give --input and/or --watch")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    if a.gt:
        import pandas as pd
        g = pd.read_csv(a.gt)
        gt = {str(r.file): int(r.gt_level) for r in g.itertuples()}

    t_load = time.time()
    tau = (a.tau, a.tau, 0.999) if a.focus else (a.tau, a.tau, a.tau)  # focus mode issues at most "warning" (as in the paper)
    pipe = Pipeline(a.checkpoint, a.encoder, a.device, WarnConfig(tau=tau))
    masker = Masker(a.mask, a.layout, gpu=pipe.dev.type == "cuda")
    hud = HUD(gt_cap=2 if a.focus else 3)
    print(f"[demo] model {pipe.meta['run_id']} on {pipe.meta['device']} loaded in {time.time() - t_load:.1f}s")

    rows, thumbs, hist, gth, vids, gifs = [], {}, {}, {}, {}, {}
    fj = open(out / "events.jsonl", "w", encoding="utf-8")
    t_proc, n_obs = 0.0, 0
    last_wall, last_ts = None, None

    def process(item):
        nonlocal t_proc, n_obs, last_wall, last_ts
        for off, frames in observations(item, a.window):
            ts = item["ts"] + timedelta(seconds=off)
            if a.speed > 0 and last_ts is not None:  # pace like a live feed
                wait = min(max((ts - last_ts).total_seconds() / a.speed - (time.time() - last_wall), 0), a.max_sleep)
                time.sleep(wait)
            s = time.time()
            raw, masked, mask = prepare(frames, masker)
            t_m = time.time() - s
            P, pw, hz, ev = pipe.step(item["camera"], ts, masked, a.reset_gap, a.focus)
            if pipe.dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - s
            t_proc += dt; n_obs += 1
            last_wall, last_ts = time.time(), ts
            cam = item["camera"]
            g = gt.get(item["path"].name, -1)
            no_clock = item.get("ts_source") == "order"
            ts_str = f"t = +{(ts - datetime(2000, 1, 1)).total_seconds():.0f} s (no timestamp in file name)" if no_clock \
                else f"{ts:%Y-%m-%d %H:%M:%S}"
            r = {"obs_id": n_obs, "camera": cam, "ts": ts, "ts_str": ts_str,
                 "file": item["path"].name, "offset_s": round(off, 2), "level": ev["level"], "level_name": LEVEL_NAMES[ev["level"]],
                 "raw_level": ev["raw_level"], "early": ev["early"], "hazard": ev["hazard"], "hazard_score": ev["hazard_score"],
                 "conf": ev["conf"], "gt_level": g, "mask_frac": round(float((mask > 0).mean()), 4), "latency_ms": round(1000 * dt, 1),
                 "mask_ms": round(1000 * t_m, 1), "reset": ev["reset"]}
            for a_, vs in ATTRS.items():
                r[f"top_{a_}"] = vs[int(np.argmax(P[a_]))]
                r[f"p_{a_}"] = [round(float(v), 4) for v in P[a_]]
            r["p_level"] = [round(float(v), 4) for v in pw]
            r["hazards"] = {h_: round(float(v), 4) for h_, v in zip(HAZARDS, hz)}
            rows.append(r)
            fj.write(json.dumps({**r, "ts": r["ts"].isoformat()}) + "\n"); fj.flush()
            mid = len(raw) // 2
            show = masked[mid] if a.show == "masked" else raw[mid]
            thumbs[n_obs] = _b64(show)
            hist.setdefault(cam, []).append(ev["level"]); gth.setdefault(cam, []).append(g)
            if not a.no_video or a.gif:
                frame = hud.render(show, {**ev, "ts_str": r["ts_str"]}, P, pw, hz, hist[cam], gth[cam] if gt else None,
                                   mask if a.show == "raw+outline" else None)
                if not a.no_video:
                    if cam not in vids:
                        vids[cam] = VideoOut(out / f"demo_{_safe(cam)}.mp4", a.fps)
                    vids[cam].write(frame, a.hold)
                if a.gif:
                    gifs.setdefault(cam, []).append(Image.fromarray(frame[:, :, ::-1]).resize((800, 450), Image.LANCZOS))
            hz_txt = f"{ev['hazard'].replace('_', ' ')} {ev['hazard_score']:.2f}"
            print(f"{r['ts_str']}  {cam[:28]:<28}  L{ev['level']} {LEVEL_NAMES[ev['level']]:<8}{' early' if ev['early'] else '      '}"
                  f"  {r['top_precip']}/{r['top_intensity']}  surface {r['top_surface']:<5} vis {r['top_visibility']:<12}"
                  f" {r['top_illumination']:<8} traffic {r['top_traffic']:<9} | {hz_txt}"
                  + (f" | human {g}" if g >= 0 else "") + f"  [{1000 * dt:.0f} ms]", flush=True)

    try:
        if a.input:
            items = assign_times(discover(a.input, a.camera), a.interval)
            print(f"[demo] {len(items)} files, {len({i['camera'] for i in items})} cameras")
            for it in items:
                process(it)
        if a.watch:
            seen, sizes = set(), {}
            print(f"[demo] watching {a.watch} (Ctrl+C to stop)")
            while True:
                found = discover([a.watch], a.camera)
                ready = []
                for it in found:
                    k = str(it["path"])
                    if k in seen:
                        continue
                    sz = it["path"].stat().st_size
                    if sizes.get(k) == sz and sz > 0:  # unchanged since last poll -> fully written
                        ready.append(it)
                    sizes[k] = sz
                for it in assign_times(ready, a.interval):
                    seen.add(str(it["path"]))
                    if it.get("ts_source") == "order":
                        it["ts"] = datetime.now()
                    process(it)
                time.sleep(a.poll)
    except KeyboardInterrupt:
        print("[demo] stopped")
    finally:
        fj.close()
        for v in vids.values():
            v.close()
        for cam, fr in gifs.items():
            fr[0].save(out / f"demo_{_safe(cam)}.gif", save_all=True, append_images=fr[1:], duration=1000 * a.hold // a.fps,
                       loop=0, optimize=True)
        if rows:
            import pandas as pd
            flat = [{k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()} for r in rows]
            pd.DataFrame(flat).to_csv(out / "events.csv", index=False)
            for cam in dict.fromkeys(r["camera"] for r in rows):
                timeline_png(cam, [r for r in rows if r["camera"] == cam], out / f"timeline_{_safe(cam)}.png",
                             gt_cap=2 if a.focus else 3)
            write_report(out, rows, thumbs, pipe.meta)
            run = {**pipe.meta, "args": vars(a), "n_observations": n_obs,
                   "mean_latency_ms": round(1000 * t_proc / max(n_obs, 1), 1),
                   "mean_mask_ms": round(float(np.mean([r["mask_ms"] for r in rows])), 1),
                   "levels": {LEVEL_NAMES[k]: sum(r["level"] == k for r in rows) for k in range(4)}}
            (out / "run.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
            print(f"[demo] {n_obs} observations, mean latency {run['mean_latency_ms']} ms (mask + encode + model + decision)"
                  f" -> {out}")


if __name__ == "__main__":
    main()
