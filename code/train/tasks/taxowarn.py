"""Task: taxonomy-structured hazard recognition + warning on cached SigLIP2 features (TaxoWarn and feature baselines).

cfg.data: features (dir with shard*.npz), labels (labels_v1.csv), use_raw (unmasked frames -> shortcut ablation),
          k (frames used: 8 video / 1 single middle frame), sources_train (list or 'all'), min_w (drop pseudo labels below)
cfg.model: name taxowarn | flat | linear | mlp, + TaxoWarn ablation switches
Target vector per item (float): [attr idx x6 | attr weight x6 | warn level | warn weight | transitional | source idx]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.taxonomy import ATTRS, warn_probs_from_attrs, warning_level  # noqa: E402
from models.taxowarn import TaxoWarn  # noqa: E402
from train import ddp  # noqa: E402

A = list(ATTRS)
NA = len(A)
_FEAT_CACHE = {}


def load_features(fdir, use_raw):
    key = (fdir, use_raw)
    if key not in _FEAT_CACHE:
        ids, nfr, X = [], [], []
        for f in sorted(Path(fdir).glob("shard*.npz")):
            z = np.load(f)
            ids += list(z["ids"]); nfr.append(z["nfr"]); X.append(z["raw" if use_raw else "masked"])
        _FEAT_CACHE[key] = ({k: i for i, k in enumerate(ids)}, np.concatenate(nfr), np.concatenate(X))
    return _FEAT_CACHE[key]


_DET_CACHE = {}
DET_D = 20


def load_det(fdir):
    """object-detection descriptors (extract_detections.py): {item_id: row}, det [N, 8, DET_D] (fp16)"""
    if fdir not in _DET_CACHE:
        ids, X = [], []
        for f in sorted(Path(fdir).glob("shard*.npz")):
            z = np.load(f)
            ids += list(z["ids"]); X.append(z["det"])
        _DET_CACHE[fdir] = ({k: i for i, k in enumerate(ids)}, np.concatenate(X))
    return _DET_CACHE[fdir]


class FeatDS(Dataset):
    def __init__(self, df, feats, k, sources, det=None):
        idx, nfr, X = feats
        df = df[df.item_id.isin(idx)].reset_index(drop=True)
        self.df, self.rows = df, np.array([idx[i] for i in df.item_id])
        self.nfr, self.X, self.k = nfr, X, k
        self.sources = sources
        self.t = make_targets(df, sources)
        self.det = None
        if det is not None:  # fusion: per-frame detection descriptor appended after the 1152 SigLIP dims
            didx, D = det
            self.det = np.zeros((len(df), 8, DET_D), np.float16)
            hit = 0
            for j, it in enumerate(df.item_id):
                if it in didx:
                    self.det[j] = D[didx[it]]; hit += 1
            print(f"[det] {hit}/{len(df)} items have detection features")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.rows[i]
        n = int(self.nfr[r])
        x = torch.from_numpy(self.X[r])  # fp16; models cast to fp32 on the GPU (CPU-side conversion was the bottleneck)
        if self.k == 1:
            x = torch.cat([x[n // 2: n // 2 + 1], torch.zeros(7, x.shape[1], dtype=x.dtype)])
        if self.det is not None:
            dt = torch.from_numpy(self.det[i])
            if self.k == 1:
                dt = torch.cat([dt[n // 2: n // 2 + 1], torch.zeros(7, DET_D, dtype=dt.dtype)])
            x = torch.cat([x, dt], -1)
        return x, self.t[i], i


def make_targets(df, sources):
    if True:
        t = np.full((len(df), 2 * NA + 5), -100.0, dtype=np.float32)
        if "neg_surface" in df:
            sm = {v: i for i, v in enumerate(ATTRS["surface"])}
            t[:, 2 * NA + 4] = [sm.get(v, -100) for v in df.neg_surface.fillna("").astype(str)]
        for j, a in enumerate(A):
            vals = df[f"lab_{a}"].fillna("").astype(str)
            m = {v: i for i, v in enumerate(ATTRS[a])}
            t[:, j] = [m.get(v, -100) for v in vals]
            t[:, NA + j] = np.where(t[:, j] >= 0, df[f"w_{a}"].astype(float), 0.0)
        t[:, 2 * NA] = df.warn_level.astype(float)
        t[:, 2 * NA + 1] = np.where(df.warn_level >= 0, df[[f"w_{a}" for a in A]].replace(0, np.nan).mean(1).fillna(0), 0)
        t[:, 2 * NA + 2] = df.transitional.astype(float)
        t[:, 2 * NA + 3] = [sources.index(s) if s in sources else -1 for s in df.source]
    return torch.from_numpy(t)


# ------------------------------------------------------------------ baselines on the same features
class FeatBaseline(nn.Module):
    """linear / mlp: independent attribute heads on mean-pooled frames. flat: one softmax over observed attribute
    combinations (flat 'weather class' classifier) - decoded back to attributes for evaluation."""

    def __init__(self, kind, d_in=1152, flat_classes=None, d_det=0):
        super().__init__()
        self.kind = kind
        self.d_in, self.d_det = d_in, d_det
        if d_det:  # detection branch: [mean, std, max] over valid frames -> MLP -> concatenated to the image embedding
            self.det_body = nn.Sequential(nn.LayerNorm(3 * d_det), nn.Linear(3 * d_det, 64), nn.GELU(), nn.Dropout(0.1))
        # input LayerNorm (as in TaxoWarn): L2-normalised SigLIP features otherwise give near-zero logits and the probes
        # collapse to the majority class (observed in runs 20261104_0600_*base_linear*, 0603_*base_mlp*; marked invalid)
        body = nn.LayerNorm(d_in) if kind in ("linear", "flat") else nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(0.1))
        d = (d_in if kind in ("linear", "flat") else 512) + (64 if d_det else 0)
        self.body = body
        if kind == "flat":
            self.flat_classes = flat_classes  # list of tuples
            self.head = nn.Linear(d, len(flat_classes))
            M = torch.zeros(NA, len(flat_classes), dtype=torch.long)
            for c, combo in enumerate(flat_classes):
                for j, v in enumerate(combo):
                    M[j, c] = v
            self.register_buffer("M", M)
        else:
            self.heads = nn.ModuleDict({a: nn.Linear(d, len(v)) for a, v in ATTRS.items()})

    def forward(self, x, valid=None):
        x = x.float()
        dt = None
        if self.d_det:
            x, dt = x[..., :self.d_in], x[..., self.d_in:]
        if valid is None:
            valid = x.abs().sum(-1) > 0
        w = valid.float().unsqueeze(-1)
        z = self.body((x * w).sum(1) / w.sum(1).clamp(min=1))
        if dt is not None:
            n = w.sum(1).clamp(min=1)
            mu = (dt * w).sum(1) / n
            sd = (((dt - mu.unsqueeze(1)) ** 2) * w).sum(1).div(n).clamp(min=1e-6).sqrt()
            mx = (dt * w - (1 - w) * 1e4).max(1).values.clamp(min=-10)
            z = torch.cat([z, self.det_body(torch.cat([mu, sd, mx], -1))], -1)
        if self.kind == "flat":
            lf = self.head(z)
            pf = lf.softmax(-1)
            logits = {}
            for j, a in enumerate(A):  # marginalise the flat distribution onto each attribute
                pa = torch.zeros(z.shape[0], len(ATTRS[a]), device=z.device).index_add_(1, self.M[j], pf)
                logits[a] = torch.log(pa.clamp(min=1e-6))
            return {"logits": logits, "flat_logits": lf}
        return {"logits": {a: h(z) for a, h in self.heads.items()}}


# ------------------------------------------------------------------ loss
class Loss(nn.Module):
    def __init__(self, cfg, src_weight):
        super().__init__()
        L = cfg.get("loss", {})
        self.lw, self.lo, self.lr = L.get("warn", 0.5), L.get("onset", 0.2), L.get("hazard_reg", 0.01)
        self.ls = L.get("label_smoothing", 0.05)
        self.register_buffer("src_w", torch.as_tensor(src_weight, dtype=torch.float32))
        self.flat_index = None

    def forward(self, out, t):
        sw = self.src_w[t[:, 2 * NA + 3].long().clamp(min=0)]
        tot = 0.0
        if "flat_logits" in out and self.flat_index is not None:
            yi = self.flat_index(t[:, :NA].long())
            m = yi >= 0
            if m.any():
                tot = tot + (F.cross_entropy(out["flat_logits"][m], yi[m], reduction="none") * sw[m]).mean()
            return tot
        for j, a in enumerate(A):
            y = t[:, j].long()
            w = t[:, NA + j] * sw
            m = y >= 0
            if m.any():
                ce = F.cross_entropy(out["logits"][a][m], y[m], reduction="none", label_smoothing=self.ls)
                tot = tot + (ce * w[m]).sum() / w[m].sum().clamp(min=1e-3)
        # deep supervision of each path of a dual-path model (weight 0.5 each), same masked/weighted CE
        for aux in out.get("aux_logits", []) or []:
            for j, a in enumerate(A):
                y = t[:, j].long()
                w = t[:, NA + j] * sw
                m = y >= 0
                if m.any():
                    ce = F.cross_entropy(aux[a][m].float(), y[m], reduction="none", label_smoothing=self.ls)
                    tot = tot + 0.5 * (ce * w[m]).sum() / w[m].sum().clamp(min=1e-3)
        # negative partial labels ("surface is not water"): -log(1 - p_neg)
        neg = t[:, 2 * NA + 4].long()
        mn = neg >= 0
        if mn.any():
            p = out["logits"]["surface"][mn].float().softmax(-1).gather(1, neg[mn].unsqueeze(1)).squeeze(1)
            tot = tot + (-torch.log((1 - p).clamp(min=1e-4)) * sw[mn]).mean()
        if "warn_cum_logit" in out:
            yl = t[:, 2 * NA].long()
            m = yl >= 0
            if m.any():
                tgt = torch.stack([(yl > j).float() for j in range(3)], 1)
                bce = F.binary_cross_entropy_with_logits(out["warn_cum_logit"][m], tgt[m], reduction="none").sum(1)
                ww = t[m, 2 * NA + 1] * sw[m]
                tot = tot + self.lw * (bce * ww).sum() / ww.sum().clamp(min=1e-3)
        if "onset_logit" in out:
            yo = t[:, 2 * NA + 2]
            tot = tot + self.lo * (F.binary_cross_entropy_with_logits(out["onset_logit"].float(), yo, reduction="none") * sw).mean()
        if out.get("rule") is not None:
            tot = tot + self.lr * (out["hazard_logit"].float() - torch.logit(out["rule"].float())).pow(2).mean()
        return tot


# ------------------------------------------------------------------ evaluation
def macro_f1(y, p, C):
    f = []
    for c in range(C):
        tp = ((p == c) & (y == c)).sum(); fp = ((p == c) & (y != c)).sum(); fn = ((p != c) & (y == c)).sum()
        if (y == c).sum() == 0:
            continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else float("nan")


def ece(conf, correct, bins=15):
    e, n = 0.0, len(conf)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (conf > lo) & (conf <= lo + 1 / bins)
        if m.sum():
            e += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def make_eval(sources):
    @torch.no_grad()
    def evaluate(model, loader, ctx, amp_dtype):
        model.eval()
        preds = []
        for x, t, i in loader:
            x = x.to(ctx.device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                out = model(x)
            P = {a: out["logits"][a].float().softmax(-1).cpu().numpy() for a in A}
            W = out["warn_prob"].float().cpu().numpy() if "warn_prob" in out else None
            for b in range(x.shape[0]):
                r = {"idx": int(i[b]), "t": t[b].tolist(), **{f"p_{a}": P[a][b].round(4).tolist() for a in A}}
                if W is not None:
                    r["p_warn"] = W[b].round(4).tolist()
                    r["hazard"] = out["hazard"][b].float().cpu().numpy().round(4).tolist()
                    r["onset"] = float(torch.sigmoid(out["onset_logit"][b].float()))
                preds.append(r)
        allp = [p for part in ddp.all_gather_objects(preds, ctx) for p in part]
        met = summarize(allp)
        return {f"val/{k}": v for k, v in met.items()}, preds
    return evaluate


def summarize(preds, gt_only=False):
    met, f1s = {}, []
    T = np.array([p["t"] for p in preds])
    for j, a in enumerate(A):
        y = T[:, j].astype(int)
        m = y >= 0
        if gt_only:
            m &= T[:, NA + j] >= 0.99
        if m.sum() < 5:
            continue
        P = np.array([p[f"p_{a}"] for p in preds])[m]
        pr = P.argmax(1)
        met[f"{a}_f1"] = macro_f1(y[m], pr, len(ATTRS[a]))
        met[f"{a}_acc"] = float((pr == y[m]).mean())
        met[f"{a}_ece"] = ece(P.max(1), (pr == y[m]).astype(float))
        f1s.append(met[f"{a}_f1"])
    met["mean_f1"] = float(np.nanmean(f1s)) if f1s else float("nan")
    # model-selection score on HUMAN/physics labels only (fog visibility, traffic, illumination): avoids selecting
    # checkpoints for agreement with the VLM pseudo-labeller
    if not gt_only:
        g = summarize(preds, gt_only=True)
        met["gt_score"] = float(np.nanmean([g.get(k, np.nan) for k in ("visibility_f1", "traffic_f1", "illumination_f1")]))
    # warning level from attribute predictions via taxonomy rules (for every model) and from the warn head (TaxoWarn)
    yl = T[:, 2 * NA].astype(int)
    m = yl >= 0
    if m.sum() >= 5:
        rule_lv = np.array([warning_level({a: ATTRS[a][int(np.argmax(p[f"p_{a}"]))] for a in A}) for p in preds])
        met["warn_rule_f1"] = macro_f1(yl[m], rule_lv[m], 4)
        met["warn_rule_mae"] = float(np.abs(rule_lv[m] - yl[m]).mean())
        # taxonomy pushforward of the attribute posterior (exact): distribution over levels, argmax + calibration
        Pw = warn_probs_from_attrs({a: np.array([p[f"p_{a}"] for p in preds]) for a in A})
        tl = Pw.argmax(1)
        met["warn_tax_f1"] = macro_f1(yl[m], tl[m], 4)
        met["warn_tax_mae"] = float(np.abs(tl[m] - yl[m]).mean())
        met["warn_tax_ece"] = ece(Pw.max(1)[m], (tl[m] == yl[m]).astype(float))
        if "p_warn" in preds[0]:
            W = np.array([p["p_warn"] for p in preds])
            wl = W.argmax(1)
            met["warn_f1"] = macro_f1(yl[m], wl[m], 4)
            met["warn_mae"] = float(np.abs(wl[m] - yl[m]).mean())
            met["warn_ece"] = ece(W.max(1)[m], (wl[m] == yl[m]).astype(float))
    return met


# ------------------------------------------------------------------ build
def test_dataset(cfg, te, sources):
    det = load_det(cfg["data"]["det_features"]) if cfg["data"].get("det_features") else None
    return FeatDS(te, load_features(cfg["data"]["features"], cfg["data"].get("use_raw", False)), cfg["data"].get("k", 8), sources, det)


def build(cfg, ctx):
    d = cfg["data"]
    df = pd.read_csv(d["labels"], keep_default_na=False)
    for c in ["warn_level", "transitional"] + [f"w_{a}" for a in A]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1 if c == "warn_level" else 0)
    min_w = d.get("min_w", 0.0)
    for a in A:
        drop = (df[f"w_{a}"] < min_w) & (df[f"src_{a}"] == "vlm")
        df.loc[drop, f"lab_{a}"] = ""
        df.loc[drop, f"w_{a}"] = 0.0
    if d.get("no_pseudo"):
        for a in A:
            m = df[f"src_{a}"] == "vlm"
            df.loc[m, f"lab_{a}"] = ""; df.loc[m, f"w_{a}"] = 0.0
    sources = sorted(df.source.unique())
    st = d.get("sources_train", "all")
    tr = df[df.split == "train"]
    if st != "all":
        tr = tr[tr.source.isin(st)]
    va = df[df.split == "val"]
    if d.get("sources_val", "all") != "all":
        va = va[va.source.isin(d["sources_val"])]
    feats = load_features(d["features"], d.get("use_raw", False))
    det = load_det(d["det_features"]) if d.get("det_features") else None
    train_ds = FeatDS(tr, feats, d.get("k", 8), sources, det)
    val_ds = FeatDS(va, feats, d.get("k", 8), sources, det)
    d_det = DET_D if det is not None else 0
    # source re-weighting: inverse sqrt frequency (keeps small hazard sources from being swamped by 8.7k ITS clips)
    cnt = tr.source.value_counts()
    sw = np.array([1.0 / np.sqrt(max(cnt.get(s, 1), 1)) for s in sources])
    sw = sw / (sw * np.array([cnt.get(s, 0) for s in sources])).sum() * len(tr)
    if d.get("no_source_balance"):
        sw = np.ones(len(sources))
    m = cfg["model"]
    if m["name"] == "taxowarn":
        tp = np.load(d["text_protos"]) if m.get("use_text", True) else None
        model = TaxoWarn(text_protos={a: tp[a] for a in A} if tp is not None else None,
                         **{k: m[k] for k in ("use_temporal", "use_delta", "use_text", "use_hazard_layer", "hierarchical", "d", "layers", "dropout") if k in m})
    elif m["name"] == "flat":
        T = train_ds.t[:, :NA].long()
        full = (T >= 0).all(1)
        combos = sorted({tuple(r.tolist()) for r in T[full]})
        model = FeatBaseline("flat", flat_classes=combos, d_det=d_det)
    else:
        model = FeatBaseline(m["name"], d_det=d_det)
    if d_det and m["name"] == "taxowarn":
        raise ValueError("det_features fusion is implemented for the linear / mlp / flat feature models")
    loss = Loss(cfg, sw)
    if m["name"] == "flat":
        lut = {c: i for i, c in enumerate(model.flat_classes)}

        def flat_index(y):
            return torch.tensor([lut.get(tuple(r.tolist()), -1) for r in y.cpu()], device=y.device)
        loss.flat_index = flat_index
    loss = loss.to(ctx.device)
    if ctx.is_main:
        print(f"[taxowarn] train {len(train_ds)}  val {len(val_ds)}  sources {sources}")
    return dict(model=model, train_ds=train_ds, val_ds=val_ds, loss_fn=loss, eval_fn=make_eval(sources),
                monitor=cfg["train"].get("monitor", "val/mean_f1"), mode="max")


