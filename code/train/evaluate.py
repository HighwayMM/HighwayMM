"""Test-set evaluation of a trained run (or of zero-shot / VLM baselines) -> run_dir\\test_metrics.json, test_per_source.csv,
test_predictions.jsonl. Metrics are computed on HUMAN/physics GT labels only (pseudo labels excluded) per source, plus
a 'silver' table on our data vs VLM labels.

  python code\\train\\evaluate.py --run <run_dir>              (TaxoWarn / feature baselines)
  python code\\train\\evaluate.py --zeroshot siglip2 --tag zs  (zero-shot SigLIP2 with taxonomy text prototypes)
  python code\\train\\evaluate.py --vlm                        (Qwen2.5-VL pseudo-labeller as a baseline)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, warn_probs_from_attrs  # noqa: E402
from train import ddp  # noqa: E402
from train.runlog import RunLogger  # noqa: E402
from train.tasks import taxowarn as T  # noqa: E402

A = list(ATTRS)
NA = len(A)
REL = Path(r"D:\highwaymm\HighwayMM_release")


def binary_metrics(preds, df):
    """Hazard-detection metrics from human labels that are binary by nature:
    flooding (FloodCam water vs no_water) and snow-covered road (NYSDOT severe snow/snow vs wet/dry)."""
    from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
    if "note" not in df and "gt_note" not in df:
        return {}
    notes = df.note.values if "note" in df else df.gt_note.values
    src = df.source.values
    out = {}
    tasks = {"flood_floodcam": ("floodcam_nc", "surface", "water", lambda n: 1 if "label=water" in n else (0 if "no_water" in n else -1)),
             "snowcover_nysdot": ("nysdot_511ny", "surface", "snow",
                                  lambda n: {1: 1, 2: 1, 3: 0, 4: 0}.get(int(n.split("nysdot_class=")[1].split(";")[0]), -1) if "nysdot_class=" in n else -1)}
    # Iowa RWIS (summer, held-out cameras): sensor labels. rain = METAR rain vs no precipitation; wet = pavement wet vs dry
    lab = {a: (df[f"lab_{a}"].fillna("").values if f"lab_{a}" in df else df[f"gt_{a}"].fillna("").values) for a in ("precip", "surface")}
    tasks["rain_iowa"] = ("iowa_rwis", "precip", "rain", lambda i: {"rain": 1, "none": 0}.get(lab["precip"][i], -1))
    tasks["wet_iowa"] = ("iowa_rwis", "surface", ("wet", "water"), lambda i: {"wet": 1, "dry": 0}.get(lab["surface"][i], -1))
    for name, (s, a, c, fy) in tasks.items():
        y, sc = [], []
        for p in preds:
            i = p["idx"]
            if src[i] != s:
                continue
            t = fy(i) if s == "iowa_rwis" else fy(notes[i])
            if t < 0:
                continue
            cs = c if isinstance(c, tuple) else (c,)
            y.append(t); sc.append(sum(p[f"p_{a}"][ATTRS[a].index(k)] for k in cs))
        if len(set(y)) == 2:
            y, sc = np.array(y), np.array(sc)
            out[f"{name}_auroc"] = float(roc_auc_score(y, sc))
            out[f"{name}_ap"] = float(average_precision_score(y, sc))
            out[f"{name}_f1"] = float(f1_score(y, sc >= 0.5))
            out[f"{name}_n"] = int(len(y))
    return out


def per_source(preds, df):
    rows = []
    src = df.source.values
    for s in sorted(set(src[[p["idx"] for p in preds]])):
        ps = [p for p in preds if src[p["idx"]] == s]
        for kind, gt_only in (("gt", True), ("all", False)):
            m = T.summarize(ps, gt_only=gt_only)
            rows.append({"source": s, "labels": kind, "n": len(ps), **m})
    return pd.DataFrame(rows)


def predict(model, ds, device, bs=256):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False)
    preds = []
    model.eval()
    with torch.no_grad():
        for x, t, i in dl:
            out = model(x.to(device))
            P = {a: out["logits"][a].float().softmax(-1).cpu().numpy() for a in A}
            Wt = warn_probs_from_attrs(P)  # taxonomy pushforward (available for every model with attribute heads)
            for b in range(x.shape[0]):
                r = {"idx": int(i[b]), "item_id": ds.df.item_id.iloc[int(i[b])], "t": t[b].tolist(),
                     **{f"p_{a}": P[a][b].round(4).tolist() for a in A}, "p_warn_tax": Wt[b].round(5).tolist()}
                if "warn_prob" in out:
                    r["p_warn"] = out["warn_prob"][b].float().cpu().numpy().round(4).tolist()
                if "hazard" in out:
                    r["hazard"] = out["hazard"][b].float().cpu().numpy().round(4).tolist()
                if "onset_logit" in out:
                    r["onset"] = float(torch.sigmoid(out["onset_logit"][b].float()))
                preds.append(r)
    return preds


def _targets(preds, key):
    """indices into the target vector for each head key"""
    if key == "p_warn":
        return np.array([p["t"][2 * NA] for p in preds]).astype(int)
    j = A.index(key[2:])
    return np.array([p["t"][j] for p in preds]).astype(int)


def fit_temperatures(vpreds):
    temps = {}
    keys = [f"p_{a}" for a in A] + (["p_warn"] if vpreds and "p_warn" in vpreds[0] else [])
    for k in keys:
        y = _targets(vpreds, k)
        m = y >= 0
        if m.sum() < 20:
            temps[k] = 1.0
            continue
        lp = torch.log(torch.tensor([vpreds[i][k] for i in np.where(m)[0]]).clamp(min=1e-6))
        yt = torch.tensor(y[m])
        logt = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([logt], lr=0.1, max_iter=100)

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(lp / logt.exp(), yt)
            loss.backward()
            return loss
        opt.step(closure)
        temps[k] = float(logt.exp().clamp(0.05, 20))
    return temps


def apply_temperatures(preds, temps):
    out = []
    for p in preds:
        q = dict(p)
        for k, t in temps.items():
            if k in q:
                lp = np.log(np.clip(np.array(q[k]), 1e-6, None)) / t
                e = np.exp(lp - lp.max())
                q[k] = (e / e.sum()).round(5).tolist()
        out.append(q)
    return out


class ZeroShot(torch.nn.Module):
    def __init__(self, protos, scale):
        super().__init__()
        self.p = {a: torch.as_tensor(protos[a]).float() for a in A}
        self.scale = float(scale)

    def forward(self, x):
        x = x.float()
        valid = (x.abs().sum(-1) > 0).float().unsqueeze(-1)
        z = torch.nn.functional.normalize((x * valid).sum(1) / valid.sum(1).clamp(min=1), dim=-1)
        return {"logits": {a: self.scale * z @ self.p[a].to(z.device).T for a in A}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--zeroshot")
    ap.add_argument("--vlm", action="store_true")
    ap.add_argument("--labels", default=None, help="default: the run's cfg data.labels, else labels_v1.csv")
    ap.add_argument("--features", default=str(REL / "features" / "siglip2"))
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    ctx = ddp.setup()
    if a.labels is None:
        a.labels = str(REL / "annotations" / "labels_v1.csv")
        if a.run:
            a.labels = yaml.safe_load(open(Path(a.run) / "config.yaml", encoding="utf-8"))["data"]["labels"]
    ver = Path(a.labels).stem.replace("labels_", "")
    df = pd.read_csv(a.labels, keep_default_na=False)
    for c in ["warn_level", "transitional"] + [f"w_{x}" for x in A]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1 if c == "warn_level" else 0)
    te = df[df.split == a.split].reset_index(drop=True)
    sources = sorted(df.source.unique())

    if a.run:
        run = Path(a.run)
        cfg = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
        cfg["train"]["resume"] = None
        import importlib
        mod = importlib.import_module(f"train.tasks.{cfg['task']}")
        parts = mod.build(cfg, ctx)
        model = parts["model"].to(ctx.device)
        ck = torch.load(run / "checkpoints" / "best.pt", map_location=ctx.device, weights_only=False)
        model.load_state_dict(ck["model"])
        ds = mod.test_dataset(cfg, te, sources)
        preds = predict(model, ds, ctx.device, bs=64 if cfg["task"] != "taxowarn" else 256)
        # post-hoc temperature scaling per head, fitted on the validation split (softmax(log p / T) == softmax(logit / T))
        va = df[df.split == "val"].reset_index(drop=True)
        vpreds = predict(model, mod.test_dataset(cfg, va, sources), ctx.device, bs=64 if cfg["task"] != "taxowarn" else 256)
        temps = fit_temperatures(vpreds)
        (run / "temperatures.json").write_text(json.dumps(temps, indent=2), encoding="utf-8")
        uncal = {"gt": T.summarize(preds, gt_only=True), "all": T.summarize(preds)}
        (run / f"{a.split}_metrics_uncalibrated.json").write_text(json.dumps(uncal, indent=2), encoding="utf-8")
        preds = apply_temperatures(preds, temps)
        # recompute the taxonomy pushforward from the CALIBRATED attribute posteriors
        Wt = warn_probs_from_attrs({a: np.array([p[f"p_{a}"] for p in preds]) for a in A})
        for p, w in zip(preds, Wt):
            p["p_warn_tax"] = w.round(5).tolist()
        out_dir, name = run, a.split  # val evaluation (for model selection) never overwrites test files
        # no new run folder: evaluation outputs live next to the trained run
        tab = per_source(preds, ds.df)
        tab.to_csv(out_dir / f"{name}_per_source.csv", index=False)
        with open(out_dir / f"{name}_predictions.jsonl", "w", encoding="utf-8") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        overall = {"gt": T.summarize(preds, gt_only=True), "all": T.summarize(preds), "binary": binary_metrics(preds, ds.df)}
        (out_dir / f"{name}_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
        print(tab.to_string())
        return

    cfg = {"task": "evaluate", "seed": 0, "mode": "zeroshot" if a.zeroshot else "vlm", "data": {"version": ver, "labels": a.labels, "split": a.split}}
    with RunLogger(cfg, ctx, task="eval", model=("zs_" + a.zeroshot) if a.zeroshot else "qwen25vl7b", tag=a.split if ver == "v1" else f"{a.split}_{ver}") as log:
        if a.zeroshot:
            tp = np.load(REL / "features" / "text_protos_siglip2.npz")
            model = ZeroShot({x: tp[x] for x in A}, 100.0)
            ds = T.FeatDS(te, T.load_features(a.features, False), 8, sources)
            preds = predict(model, ds, "cpu")
        else:
            vl = {}
            for f in (REL / "annotations" / "vlm_qwen25vl7b").glob("shard*.jsonl"):
                for l in open(f, encoding="utf-8"):
                    j = json.loads(l)
                    if j.get("ok"):
                        vl[j["item_id"]] = j["labels"]
            ds = T.FeatDS(te, T.load_features(a.features, False), 8, sources)
            preds = []
            for i in range(len(ds)):
                iid = ds.df.item_id.iloc[i]
                if iid not in vl:
                    continue
                r = {"idx": i, "item_id": iid, "t": ds.t[i].tolist()}
                for x in A:
                    oh = np.full(len(ATTRS[x]), 1e-3)
                    v = vl[iid].get(x)
                    if v in ATTRS[x]:
                        oh[ATTRS[x].index(v)] = 1.0
                    r[f"p_{x}"] = (oh / oh.sum()).tolist()
                preds.append(r)
        tab = per_source(preds, ds.df)
        tab.to_csv(log.dir / "test_per_source.csv", index=False)
        overall = {"gt": T.summarize(preds, gt_only=True), "all": T.summarize(preds), "binary": binary_metrics(preds, ds.df)}
        log.save_json("test_metrics.json", overall)
        print(tab.to_string())
        log.finish("completed", final_metrics=overall, key_metrics={k: v for k, v in overall["gt"].items() if k.endswith("f1")})
    ddp.cleanup()


if __name__ == "__main__":
    main()



