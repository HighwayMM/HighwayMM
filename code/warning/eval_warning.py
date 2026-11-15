"""Warning-system evaluation (C2) on camera sequences of the test split, using a trained TaxoWarn run's test predictions.

GT warning levels on sequences come from HUMAN labels only:
  FloodCam NC (test cams Buxton, Canal): water -> 3 (flooding), no_water -> 0, not_sure -> -1 (ignored)
  NYSDOT blizzard: severe snow -> 3, snow -> 2, wet -> 1, dry -> 0, poor visibility -> 3
Outputs (in the run dir): warning_events.csv, warning_metrics.json, warning_tau_sweep.csv, warning_timeline_<seq>.png
Also reports storm timelapses (Harvey, Boston) qualitatively (no human GT).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warning.system import WarnConfig, run_sequences, sequence_metrics  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")


def parse_ts(x):
    """ISO (naive or tz-aware) or timelapse stems like 2017-08-25-16-10-00 / 2020-5-26-10-45-29 -> naive Timestamp."""
    import re
    s = str(x)
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})", s)
    if m:
        y, mo, d, h, mi, se = map(int, m.groups())
        return pd.Timestamp(y, mo, d, h, mi, se)
    t = pd.to_datetime(s, errors="coerce", utc=False)
    if t is pd.NaT or pd.isna(t):
        return pd.NaT
    return t.tz_localize(None) if t.tzinfo is not None else t


def gt_level(row):
    n = row["note"]
    if row["source"] == "floodcam_nc":
        return 3 if "floodcam_label=water" in n else (0 if "no_water" in n else -1)
    if row["source"] == "nysdot_511ny":
        c = int(n.split("nysdot_class=")[1].split(";")[0])
        return {1: 3, 2: 2, 3: 1, 4: 0, 5: 3}.get(c, -1)
    return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--warn_source", default="tax", choices=["tax", "head", "fused"],
                    help="tax = taxonomy pushforward of attribute posteriors; head = ordinal warning head; fused = mean")
    a = ap.parse_args()
    run = Path(a.run)
    preds = [json.loads(l) for l in open(run / "test_predictions.jsonl", encoding="utf-8")]
    lab = pd.read_csv(REL / "annotations" / "labels_v1.csv", keep_default_na=False).set_index("item_id")
    rows = []
    for p in preds:
        if ("p_warn" not in p and "p_warn_tax" not in p) or p["item_id"] not in lab.index:
            continue
        if "hazard" not in p:  # baselines: hazard scores from the attribute posteriors through the same taxonomy rules
            import torch
            from models.taxowarn import rule_hazard_probs
            p["hazard"] = rule_hazard_probs({k[2:]: torch.tensor([v]) for k, v in p.items()
                                             if k.startswith("p_") and k not in ("p_warn", "p_warn_tax")})[0].tolist()
        pw_tax = np.array(p.get("p_warn_tax", p.get("p_warn")))
        pw_head = np.array(p.get("p_warn", pw_tax))  # models without a warning head: head == taxonomy pushforward
        pw = {"head": pw_head, "tax": pw_tax, "fused": (pw_head + pw_tax) / 2}[a.warn_source]
        r = lab.loc[p["item_id"]]
        if not r["sequence_id"]:
            continue
        rows.append({"camera": r["sequence_id"], "ts": r["ts"], "source": r["source"], "p_warn": pw, "hazard": p["hazard"],
                     "onset": p.get("onset", 0.0), "gt_level": gt_level({"note": r["note"], "source": r["source"]}), "item_id": p["item_id"]})
    df = pd.DataFrame(rows)
    df["t"] = [parse_ts(x) for x in df.ts]
    df["ts"] = df["t"]
    df = df.sort_values(["camera", "t"])
    out = {}
    sweep = []
    for src in ["floodcam_nc", "nysdot_511ny"]:
        s = df[df.source == src].to_dict("records")
        if not s:
            continue
        ev = run_sequences(s, WarnConfig())
        summ, per = sequence_metrics(ev, onset_level=2)
        out[src] = summ
        pd.DataFrame(ev).assign(source=src).to_csv(run / f"warning_events_{a.warn_source}_{src}.csv", index=False)
        per.to_csv(run / f"warning_events_per_cam_{a.warn_source}_{src}.csv", index=False)
        for tau in np.linspace(0.2, 0.9, 15):
            for early in (True, False):
                cfg = WarnConfig(tau=(tau, tau, tau), cusum_h=0.25 if early else 99.0, onset_tau=0.6 if early else 9.0)
                sm, _ = sequence_metrics(run_sequences(s, cfg), onset_level=2)
                sweep.append({"source": src, "tau": round(tau, 3), "early_module": early, **sm})
        # per-observation level metrics vs human GT
        e = pd.DataFrame(ev)
        e = e[e.gt_level >= 0]
        if len(e):
            out[src]["obs_level_acc"] = float((e.level == e.gt_level).mean())
            out[src]["obs_hazard_recall(gt>=2 -> level>=2)"] = float((e[e.gt_level >= 2].level >= 2).mean()) if (e.gt_level >= 2).any() else np.nan
            out[src]["obs_false_alarm_rate(gt=0 -> level>=2)"] = float((e[e.gt_level == 0].level >= 2).mean()) if (e.gt_level == 0).any() else np.nan
    pd.DataFrame(sweep).to_csv(run / f"warning_tau_sweep_{a.warn_source}.csv", index=False)
    # ---- hazard-specific early warning: each source evaluated on the hazard its human labels define -------------------
    # FloodCam: flooding (water vs no_water); NYSDOT: snow-covered road (severe snow/snow vs wet/dry). The score is the
    # attribute posterior (p_surface[water] / p_surface[snow]) so every model with attribute heads is comparable.
    pmap = {p["item_id"]: p for p in preds}
    hz_out, hz_sweep = {}, []
    spec = {"floodcam_nc": ("water", lambda n: 1 if "label=water" in n else (0 if "no_water" in n else -1)),
            "nysdot_511ny": ("snow", lambda n: {1: 1, 2: 1, 3: 0, 4: 0}.get(int(n.split("nysdot_class=")[1].split(";")[0]), -1))}
    from data.taxonomy import ATTRS
    for src, (cls, fy) in spec.items():
        g = df[df.source == src].copy()
        if g.empty:
            continue
        ci = ATTRS["surface"].index(cls)
        g["s"] = [pmap[i]["p_surface"][ci] for i in g.item_id]
        g["gt_bin"] = [fy(lab.loc[i, "note"]) for i in g.item_id]
        for tau in np.round(np.linspace(0.1, 0.9, 17), 3):
            for early in (True, False):
                rows_h = [{"camera": r.camera, "ts": r.ts, "p_warn": [1 - r.s, 0.0, r.s, 0.0], "hazard": [r.s], "onset": 0.0,
                           "gt_level": (2 if r.gt_bin == 1 else (0 if r.gt_bin == 0 else -1))} for r in g.itertuples()]
                cfg = WarnConfig(tau=(tau, tau, 0.999), cusum_h=0.25 if early else 99.0, onset_tau=9.0)
                sm, per = sequence_metrics(run_sequences(rows_h, cfg), onset_level=2)
                e = pd.DataFrame(run_sequences(rows_h, cfg))
                e = e[e.gt_level >= 0]
                sm["obs_recall"] = float((e[e.gt_level == 2].level >= 2).mean()) if (e.gt_level == 2).any() else np.nan
                sm["obs_false_alarm_rate"] = float((e[e.gt_level == 0].level >= 2).mean()) if (e.gt_level == 0).any() else np.nan
                hz_sweep.append({"source": src, "tau": tau, "early_module": early, **sm})
        sw = pd.DataFrame([r for r in hz_sweep if r["source"] == src and r["early_module"]])
        # operating point: highest recall with observation false-alarm rate <= 10%
        ok = sw[sw.obs_false_alarm_rate <= 0.10]
        best = (ok.sort_values(["obs_recall", "median_lead_min_warning"], ascending=False).iloc[0] if len(ok)
                else sw.sort_values("obs_false_alarm_rate").iloc[0])
        hz_out[src] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in best.items()}
    pd.DataFrame(hz_sweep).to_csv(run / f"warning_hazard_sweep_{a.warn_source}.csv", index=False)
    (run / "warning_hazard_metrics.json").write_text(json.dumps(hz_out, indent=2, default=str), encoding="utf-8")
    out["hazard_specific"] = hz_out
    (run / f"warning_metrics_{a.warn_source}.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    # timeline figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for cam, g in df.groupby("camera"):
            if len(g) < 8:
                continue
            ev = pd.DataFrame(run_sequences(g.to_dict("records"), WarnConfig()))
            fig, ax = plt.subplots(figsize=(8, 2.4))
            ax.step(g.t, ev.level, where="post", label="system warning level", lw=2)
            if (g.gt_level >= 0).any():
                gg = g[g.gt_level >= 0]
                ax.scatter(gg.t, gg.gt_level, s=10, c="k", label="human GT level", zorder=3)
            ax.plot(g.t, [max(h) * 3 for h in g.hazard], alpha=0.5, label="max hazard score (x3)")
            ax.set_yticks([0, 1, 2, 3]); ax.set_yticklabels(["normal", "advisory", "warning", "danger"])
            ax.set_title(cam); ax.legend(fontsize=6, loc="upper left"); fig.autofmt_xdate(); fig.tight_layout()
            fig.savefig(run / f"warning_timeline_{cam}.png", dpi=150); plt.close(fig)
    except Exception as ex:  # noqa
        print("plot failed", ex)


if __name__ == "__main__":
    main()

