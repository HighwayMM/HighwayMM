"""National-scale warning snapshot + agreement with OFFICIAL operator alert banners.

The KEC/ITS overlay alert banners ([ê°•ìš°] rain / [ì•ˆê°œ] fog / ...) are masked out of every model input and never used as
labels, so they are an independent weak reference: does TaxoWarn's P(rain or wet road) rank the cameras with an official rain
alert above the others? Also renders the national warning map (camera coords coloured by system warning level, after
multi-camera aggregation). Outputs in the run dir: national_alert_agreement.json, national_map.png/pdf, national_events.csv
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
from train.tasks import taxowarn as T  # noqa: E402
from warning.system import WarningSystem  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", default="test", help="test | all")
    a = ap.parse_args()
    from sklearn.metrics import average_precision_score, roc_auc_score
    ctx = ddp.setup()
    run = Path(a.run)
    cfg = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
    cfg["train"]["resume"] = None
    import importlib
    mod = importlib.import_module(f"train.tasks.{cfg['task']}")
    model = mod.build(cfg, ctx)["model"].to(ctx.device).eval()
    model.load_state_dict(torch.load(run / "checkpoints" / "best.pt", map_location=ctx.device, weights_only=False)["model"])
    df = pd.read_csv(REL / "annotations" / "labels_v1.csv", keep_default_na=False)
    for c in ["warn_level", "transitional"] + [f"w_{x}" for x in ATTRS]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1 if c == "warn_level" else 0)
    sel = df[(df.source == "itssnap") & ((df.split == "test") if a.splits == "test" else True)].reset_index(drop=True)
    sources = sorted(df.source.unique())
    ds = mod.test_dataset(cfg, sel, sources)
    dl = torch.utils.data.DataLoader(ds, batch_size=512)
    P, H, W = [], [], []
    with torch.no_grad():
        for x, t, i in dl:
            o = model(x.to(ctx.device))
            P.append({k: v.softmax(-1).cpu().numpy() for k, v in o["logits"].items()})
            pp = {k: v.float().softmax(-1).cpu().numpy() for k, v in o["logits"].items()}
            H.append(o["hazard"].float().cpu().numpy()); W.append(warn_probs_from_attrs(pp))  # taxonomy pushforward
    pr = {k: np.concatenate([p[k] for p in P]) for k in P[0]}
    haz, pw = np.concatenate(H), np.concatenate(W)
    d = ds.df.copy()
    d["p_rain"] = pr["precip"][:, ATTRS["precip"].index("rain")]
    d["p_wet"] = pr["surface"][:, ATTRS["surface"].index("wet")] + pr["surface"][:, ATTRS["surface"].index("water")]
    d["p_rainwet"] = 1 - (1 - d.p_rain) * (1 - d.p_wet)
    d["p_fog"] = pr["visibility"][:, 2] + pr["visibility"][:, 3]
    out = {"n_cameras": int(len(d))}
    for tag, col in (("rain", "p_rainwet"), ("fog", "p_fog")):
        y = d.alert_tags.str.contains(tag).astype(int).to_numpy()
        out[f"official_{tag}_alerts"] = int(y.sum())
        if 0 < y.sum() < len(y):
            out[f"{tag}_auroc_vs_official"] = float(roc_auc_score(y, d[col]))
            out[f"{tag}_ap_vs_official"] = float(average_precision_score(y, d[col]))
            out[f"{tag}_ap_chance"] = float(y.mean())
            k = int(y.sum())
            top = np.argsort(-d[col].to_numpy())[:k]
            out[f"{tag}_precision_at_k"] = float(y[top].mean())
    # warning decisions + multi-camera aggregation on this national snapshot
    man = pd.read_csv(REL / "manifest_ours.csv").set_index("clip_id")
    coords = {r.item_id: (man.loc[r.item_id, "lat"], man.loc[r.item_id, "lon"]) for r in d.itertuples()}
    ws = WarningSystem()
    ev = [ws.step(r.item_id, r.ts, pw[k], haz[k]) for k, r in enumerate(d.itertuples())]
    # aggregate within hour slices
    d["hour"] = pd.to_datetime(d.ts).dt.floor("h")
    agg = []
    for _, g in d.groupby("hour"):
        idx = g.index.to_list()
        agg += WarningSystem.aggregate([ev[k] for k in idx], {c: coords[c] for c in g.item_id})
    e = pd.DataFrame(agg)
    e.to_csv(run / "national_events.csv", index=False)
    out["level_counts_raw"] = pd.Series([x["level"] for x in ev]).value_counts().sort_index().to_dict()
    out["level_counts_aggregated"] = e.level.value_counts().sort_index().to_dict()
    out["downgraded_isolated_danger"] = int(e.get("downgraded", pd.Series(dtype=bool)).fillna(False).sum()) if "downgraded" in e else 0
    out["regional_raises"] = int(e.get("regional", pd.Series(dtype=bool)).fillna(False).sum()) if "regional" in e else 0
    (run / "national_alert_agreement.json").write_text(json.dumps(out, indent=2, default=int), encoding="utf-8")
    print(json.dumps(out, indent=2, default=int))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lv = e.set_index("camera").level
    fig, ax = plt.subplots(figsize=(3.4, 4.2))
    cols = {0: "#bdbdbd", 1: "#fdd835", 2: "#fb8c00", 3: "#d32f2f"}
    for L in (0, 1, 2, 3):
        ids = lv[lv == L].index
        if len(ids):
            ax.scatter([coords[i][1] for i in ids], [coords[i][0] for i in ids], s=3 if L else 1, c=cols[L],
                       label=["normal", "advisory", "warning", "danger"][L] + f" ({len(ids)})", linewidths=0, zorder=L)
    off = d[d.alert_tags.str.contains("rain")]
    ax.scatter(off.lon if "lon" in off else [coords[i][1] for i in off.item_id], [coords[i][0] for i in off.item_id],
               s=14, facecolors="none", edgecolors="k", linewidths=0.6, label=f"official rain alert ({len(off)})", zorder=5)
    ax.set_aspect(1.2); ax.legend(fontsize=6, markerscale=2, loc="lower right"); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(run / f"national_map.{ext}", dpi=200)
    ddp.cleanup()


if __name__ == "__main__":
    main()

