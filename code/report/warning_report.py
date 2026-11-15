"""Warning-system comparison across methods (from each run's warning_hazard_sweep_tax.csv, i.e. logs only).
Operating point per (run, source): the threshold with the highest observation recall whose observation false-alarm rate
<= 10%, early module ON; compared with early module OFF at the same tau (ablation). Mean +- std over seeds per tag.
Outputs experiments\\report\\warning_by_tag.csv, paper\\tables\\warning.tex, paper\\figs\\warning_tradeoff.{pdf,png}"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EXP = Path(r"D:\highwaymm\experiments")
TAGS = {"p4_base_linear": "Linear probe", "p4_base_mlp": "MLP", "p4_base_flat": "Flat classes", "p4_base_ft_convnext": "ConvNeXt-B FT",
        "p4_main_taxowarn": "TaxoWarn (v1)", "p4_s_k8": "TaxoWarn-S (K=8)", "p4_final": "TaxoWarn-S (ours)"}  # labels_v2 runs (p2 version in git-less history: experiments\report)
SRC = {"floodcam_nc": "Flooding (FloodCam)", "nysdot_511ny": "Snow cover (NYSDOT blizzard)"}


def tag_of(run_id, task, model):
    model = str(model).replace("/", "-")
    t = run_id.split(f"_{task}_{model}_", 1)[-1]
    return t.rsplit("_s", 1)[0]


def main():
    idx = pd.read_csv(EXP / "runs_index.csv")
    idx = idx[idx.status == "completed"]
    rows, curves = [], []
    for r in idx.itertuples():
        f = Path(r.run_dir) / "warning_hazard_sweep_tax.csv"
        if not f.exists():
            continue
        tag = tag_of(r.run_id, r.task, r.model)
        if tag not in TAGS:
            continue
        sw = pd.read_csv(f)
        for src, g in sw.groupby("source"):
            on, off = g[g.early_module == True], g[g.early_module == False]  # noqa: E712
            ok = on[on.obs_false_alarm_rate <= 0.10]
            best = ok.sort_values(["obs_recall", "median_lead_min_warning"], ascending=False).iloc[0] if len(ok) else on.sort_values("obs_false_alarm_rate").iloc[0]
            b_off = off[off.tau == best.tau].iloc[0]
            rows.append({"tag": tag, "run_id": r.run_id, "source": src, "tau": best.tau, "recall": best.obs_recall,
                         "far": best.obs_false_alarm_rate, "fa_per_cam_day": best.false_alarms_per_cam_day,
                         "lead_warn_min": best.median_lead_min_warning, "lead_any_min": best.median_lead_min_any,
                         "lead_any_min_no_early": b_off.median_lead_min_any, "detection_rate": best.detection_rate})
            for _, c in on.iterrows():
                curves.append({"tag": tag, "source": src, "seed_run": r.run_id, "far": c.obs_false_alarm_rate, "recall": c.obs_recall})
    df = pd.DataFrame(rows)
    out = EXP / "report"
    out.mkdir(exist_ok=True)
    df.to_csv(out / "warning_runs.csv", index=False)
    agg = df.drop(columns=["run_id"]).groupby(["source", "tag"]).agg(["mean", "std"])
    agg.to_csv(out / "warning_by_tag.csv")
    print(df.groupby(["source", "tag"])[["recall", "far", "fa_per_cam_day", "lead_warn_min", "lead_any_min", "lead_any_min_no_early"]].mean().round(3).to_string())
    # LaTeX table
    L = ["\\begin{tabular}{ll" + "c" * 4 + "}", "\\toprule", "Hazard & Method & Recall@FAR$\\le$10\\% & FA/cam-day & Lead (min) & Lead w/o early \\\\", "\\midrule"]
    for src in SRC:
        for t in TAGS:
            s = df[(df.source == src) & (df.tag == t)]
            if s.empty:
                continue
            f = lambda c, k=100, d=1: f"{k * s[c].mean():.{d}f}" + (f"$\\pm${k * s[c].std():.{d}f}" if len(s) > 1 else "")  # noqa: E731
            L.append(f"{SRC[src] if t == list(TAGS)[0] else ''} & {TAGS[t]} & {f('recall')} & {f('fa_per_cam_day', 1, 2)} & "
                     f"{f('lead_any_min', 1)} & {f('lead_any_min_no_early', 1)} \\\\")
        L.append("\\midrule")
    L[-1] = "\\bottomrule"
    L.append("\\end{tabular}")
    Path(r"D:\highwaymm\paper\tables").mkdir(parents=True, exist_ok=True)
    Path(r"D:\highwaymm\paper\tables\warning.tex").write_text("\n".join(L), encoding="utf-8")
    # trade-off figure: recall vs observation false-alarm rate (mean over seeds per tau)
    cv = pd.DataFrame(curves)
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8))
    for ax, src in zip(axes, SRC):
        for t, lab in TAGS.items():
            c = cv[(cv.source == src) & (cv.tag == t)]
            if c.empty:
                continue
            c = c.groupby(c.index % 17).mean(numeric_only=True).sort_values("far")
            ax.plot(c.far, c.recall, marker=".", label=lab, lw=2 if "ours" in lab else 1)
        ax.set_title(SRC[src], fontsize=8); ax.set_xlabel("false-alarm rate (GT normal obs.)", fontsize=7)
        ax.set_ylabel("hazard recall", fontsize=7); ax.grid(alpha=.3); ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(Path(r"D:\highwaymm\paper\figs") / f"warning_tradeoff.{ext}", dpi=200)


if __name__ == "__main__":
    main()




