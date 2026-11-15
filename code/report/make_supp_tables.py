"""Supplementary tables, generated from logged results only.
  python code\\report\\make_supp_tables.py   (after make_tables.py, eval_gold.py --out experiments\\gold_eval_p6, audit_hunt.py)
Writes paper/tables/supp_{gold_full,gold_stats,audit_retrain,hunt,run_ids}.tex
"""
import json
from pathlib import Path

import pandas as pd

EXP = Path(r"D:\highwaymm\experiments")
REL = Path(r"D:\highwaymm\HighwayMM_release")
TEX = Path(r"D:\highwaymm\paper\tables")
A = ["precip", "intensity", "surface", "visibility", "illumination", "traffic"]
GOLD_ROWS = [
    ("Qwen2.5-VL-7B (pseudo-labeller)", "vlm_qwen25vl7b"), ("Flat weather classes", "p4_base_flat"),
    ("ConvNeXt-B fine-tuned", "p4_base_ft_convnext"), ("Linear probe (K=8)", "p4_base_linear"),
    ("Linear probe (K=1)", "p4_base_linear_k1"), ("MLP probe (K=8)", "p4_base_mlp"), ("MLP probe (K=1)", "p4_base_mlp_k1"),
    ("MLP + detection descriptors", "p5_det_mlp"), ("Linear + detection descriptors", "p5_det_linear"),
    ("MLP, unmasked frames", "p4_abl_unmasked_mlp"), ("MLP, audited hunt labels", "p6_audit_mlp"),
    ("TaxoWarn-v1", "p4_main_taxowarn"), ("TaxoWarn-S (K=8)", "p4_s_k8"), ("TaxoWarn-S (final)", "p4_final"),
    ("TaxoWarn-S, human/sensor labels only", "p4_abls_gt_only"), ("TaxoWarn-S, ours only", "p4_datas_ours_only"),
    ("TaxoWarn-S, external only", "p4_datas_ext_only"), ("TaxoWarn-S, without Iowa", "p4_datas_no_iowa"),
]


def w(name, lines):
    (TEX / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", name)


def gold_full():
    t = pd.read_csv(EXP / "gold_eval_p6" / "per_tag.csv").set_index("tag")
    cols = A + ["attr_mean", "warn"]
    L = ["\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
         "Method & Prec. & Int. & Surf. & Vis. & Illum. & Traf. & Attr & Warn \\\\", "\\midrule"]
    for name, tag in GOLD_ROWS:
        if tag not in t.index:
            continue
        r = t.loc[tag]
        cells = []
        for c in cols:
            m, s = r[f"{c}_f1_mean"], r.get(f"{c}_f1_std")
            cells.append(f"{m:.1f}" + (f"$\\pm${s:.1f}" if pd.notna(s) and tag != "vlm_qwen25vl7b" else ""))
        L.append(f"{name} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    w("supp_gold_full.tex", L)


def gold_stats():
    import sys
    sys.path.insert(0, r"D:\highwaymm\code\train"); sys.path.insert(0, r"D:\highwaymm\code")
    from eval_gold import load_gold
    g = load_gold(REL / "gold" / "gold_labels_merged.csv")
    ag = pd.read_csv(REL / "gold" / "gold_agreement.csv").set_index("attr")
    vn = pd.read_csv(EXP / "gold_eval_p6" / "vlm_noise.csv").set_index("attr")
    L = ["\\begin{tabular}{@{}llccc@{}}", "\\toprule",
         "Attribute & Gold label distribution & Human--human (\\%, $\\kappa$) & VLM--human (\\%) \\\\", "\\midrule"]
    for a in A:
        dist = ", ".join(f"{k.replace('_', '-')} {v}" for k, v in g[a].value_counts().items())
        L.append(f"{a} & {dist} & {ag.at[a, 'agree_pct']:.0f} ({ag.at[a, 'kappa']:.2f}) & {vn.at[a, 'vlm_agree_pct']:.0f} \\\\")
    L += ["\\midrule", f"warning level & " + ", ".join(f"L{k} {v}" for k, v in sorted(g.warn.value_counts().items()))
          + f" & n={int(ag.n.iloc[0])} & all clips \\\\", "\\bottomrule", "\\end{tabular}"]
    L[2] = L[2].replace("{@{}llccc@{}}", "{@{}lp{6.2cm}cc@{}}")
    L[0] = "\\begin{tabular}{@{}lp{6.2cm}cc@{}}"
    w("supp_gold_stats.tex", L)


def audit_retrain():
    r = pd.read_csv(EXP / "report" / "all_runs_test.csv")
    keys = [("fog", "Fog"), ("flood", "Flood"), ("snow", "Snow"), ("traffic", "Traffic"), ("illum", "Illum."),
            ("rain", "Rain"), ("wet", "Wet"), ("kor", "Kor.\\ prec."), ("gt_mean", "GT-mean"), ("warn", "Warn")]
    rows = [("MLP probe, VLM labels", "p4_base_mlp"), ("MLP probe, audited hunt labels", "p6_audit_mlp"),
            ("MLP + det., VLM labels", "p5_det_mlp"), ("MLP + det., audited hunt labels", "p6_audit_det_mlp"),
            ("Linear probe, VLM labels", "p4_base_linear"), ("Linear probe, audited hunt labels", "p6_audit_linear")]
    L = ["\\begin{tabular}{@{}l" + "c" * len(keys) + "@{}}", "\\toprule", "Training labels & " + " & ".join(h for _, h in keys) + " \\\\", "\\midrule"]
    for name, tag in rows:
        s = r[r.tag == tag]
        if s.empty:
            continue
        L.append(name + " & " + " & ".join(f"{100 * s[k].mean():.1f}" for k, _ in keys) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    w("supp_audit_retrain.tex", L)


def hunt():
    a = pd.read_csv(REL / "gold" / "hunt_labels_audit.csv", keep_default_na=False)
    u = a[a.unusable == 0].copy()
    u["snow"] = u.precip.eq("snow") | u.surface.isin(["snow", "ice"])
    u["blizzard"] = u.precip.eq("snow") & (u.intensity.eq("heavy") | u.visibility.isin(["fog_moderate", "fog_dense"]))
    u["flood"] = u.surface.eq("water"); u["rain"] = u.precip.eq("rain"); u["fog"] = u.visibility.isin(["fog_moderate", "fog_dense"])
    u["none"] = ~(u.snow | u.flood | u.rain | u.fog)
    L = ["\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
         "Mined as & Clips & Snow & Blizzard & Flooded & Rain & Fog & None of these \\\\", "\\midrule"]
    for c in ["snow", "blizzard", "flood"]:
        s = u[u.category == c]
        L.append(f"{c} & {len(a[a.category == c])} & " + " & ".join(str(int(s[k].sum())) for k in ["snow", "blizzard", "flood", "rain", "fog", "none"]) + " \\\\")
    L += ["\\midrule", f"total & {len(a)} & " + " & ".join(str(int(u[k].sum())) for k in ["snow", "blizzard", "flood", "rain", "fog", "none"]) + " \\\\",
          "\\bottomrule", "\\end{tabular}"]
    w("supp_hunt.tex", L)


def run_ids():
    d = json.loads((EXP / "report" / "run_ids_per_tag.json").read_text())
    idx = pd.read_csv(EXP / "runs_index.csv")
    idx = idx[idx.status == "completed"]
    for t in ["p5_det_linear", "p5_det_flat", "p5_det_unmasked_mlp", "p6_audit_mlp", "p6_audit_det_mlp", "p6_audit_linear"]:
        ids = idx[idx.run_id.str.contains(f"_{t}_s")].run_id.tolist()
        if ids:
            d[t] = sorted(set(ids))
    # run ID = <yyyymmdd_hhmm>_<task>_<model>_<tag>_s<seed>; list tag, task_model and the timestamp of each seed
    L = ["\\begin{longtable}{@{}lll@{}}", "\\toprule", "Tag & Task\\_model & Timestamps (seeds 0, 1, 2) \\\\", "\\midrule", "\\endhead"]
    for t in sorted(d):
        ids = sorted(d[t], key=lambda x: x.rsplit("_s", 1)[-1])
        mid = ids[0][14:].split("_" + t + "_s")[0]
        stamps = ", ".join(x[:13] for x in ids)
        e = lambda s: s.replace("_", "\\_")  # noqa: E731
        L.append(f"\\texttt{{{e(t)}}} & \\texttt{{{e(mid)}}} & \\texttt{{{e(stamps)}}} \\\\")
    L += ["\\bottomrule", "\\end{longtable}"]
    w("supp_run_ids.tex", L)


if __name__ == "__main__":
    gold_full(); gold_stats(); audit_retrain(); hunt(); run_ids()
