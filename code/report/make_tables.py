"""Paper tables from experiment logs ONLY (every number traces to run ids in experiments\\runs_index.csv).

Protocol v2 (final): tags prefixed p2_ (checkpoint selection on human/physics val score), plus the zero-shot and VLM
baselines (task=eval). Per run, the HUMAN/physics-label test metrics are:
  fog = FVEI visibility F1, flood = FloodCam flood AUROC (held-out cameras), snow = NYSDOT snow-cover AUROC (unseen blizzard),
  traffic = Norway traffic F1, illum = illumination F1 on our ITS test cameras (solar-elevation labels);
  GT-mean = mean of the five. Also reported: warning-level F1 via taxonomy pushforward (vs. mixed labels) and agreement with
  the VLM pseudo-labels on our test clips ("VLM-agree", not a correctness measure).
Outputs experiments\\report\\{all_runs_test,by_tag_p2}.csv and paper\\tables\\{main_results,ablations}.tex
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(r"D:\highwaymm\experiments")
OUT = EXP / "report"
TEX = Path(r"D:\highwaymm\paper\tables")
# Final TaxoWarn-S configuration = K=1 global frame (+ spatial tokens of 3 frames), selected on VALIDATION (val score 89.2 vs
# 88.5 for K=8, 88.9 for the MLP probe); its runs carry the tag p2_abls_k1_image. Ablations of the final config: p3_*.
MAIN_V1 = [("zs_siglip2", "SigLIP\\,2 zero-shot"), ("vlm_qwen25vl7b", "Qwen2.5-VL-7B (prompted)"),
        ("p2_base_linear", "Linear probe (K=8)"), ("p3_base_linear_k1", "Linear probe (K=1)"),
        ("p2_base_mlp", "MLP probe (K=8)"), ("p3_base_mlp_k1", "MLP probe (K=1)"),
        ("p2_base_flat", "Flat weather classes (K=8)"), ("p3_base_flat_k1", "Flat weather classes (K=1)"),
        ("p2_base_ft_convnext", "ConvNeXt-B fine-tuned"), ("p2_main_taxowarn", "TaxoWarn-v1 (temporal, K=8)"),
        ("p2_main_taxowarn_s", "TaxoWarn-S (K=8)"), ("p2_abls_k1_image", "\\textbf{TaxoWarn-S (ours, final)}")]
ABL_V1 = [("p2_abls_k1_image", "TaxoWarn-S (final)"), ("p3_abls_no_dual", "-- global path (spatial only)"),
       ("p3_abls_no_queries", "-- taxonomy queries"), ("p3_abls_no_spatial", "-- spatial tokens"), ("p3_abls_no_text", "-- text anchors"),
       ("p3_abls_no_hazard_layer", "-- hazard rule layer"), ("p3_abls_flat_heads", "-- parent conditioning"),
       ("p2_main_taxowarn_s", "8 global frames (K=8)"), ("p3_abls_gt_only", "human labels only (no VLM pseudo)"),
       ("p3_datas_ours_only", "train on ours only"), ("p3_datas_ext_only", "train on external only"),
       ("p2_base_mlp", "MLP probe, masked frames"), ("p2_abl_unmasked_mlp", "MLP probe, UNMASKED frames")]
# labels_v2 (2026-11-06): + Iowa DOT RWIS summer cameras (sensor labels, held-out cameras); tags p4_*. GT-mean over 7 metrics.
MAIN = [("zs_siglip2_v2t", "SigLIP\\,2 zero-shot"), ("vlm_qwen25vl7b_v2t", "Qwen2.5-VL-7B (prompted)"),
        ("p4_base_linear", "Linear probe (K=8)"), ("p4_base_linear_k1", "Linear probe (K=1)"),
        ("p4_base_mlp", "MLP probe (K=8)"), ("p4_base_mlp_k1", "MLP probe (K=1)"),
        ("p4_base_flat", "Flat weather classes (K=8)"), ("p4_base_flat_k1", "Flat weather classes (K=1)"),
        ("p4_base_ft_convnext", "ConvNeXt-B fine-tuned"), ("p4_main_taxowarn", "TaxoWarn-v1 (temporal, K=8)"),
        ("p4_s_k8", "TaxoWarn-S (K=8)"), ("p4_final", "\\textbf{TaxoWarn-S (ours, final)}")]
ABL = [("p4_final", "TaxoWarn-S (final)"), ("p4_abls_no_dual", "-- global path (spatial only)"),
       ("p4_abls_no_queries", "-- taxonomy queries"), ("p4_abls_no_spatial", "-- spatial tokens"), ("p4_abls_no_text", "-- text anchors"),
       ("p4_abls_no_hazard_layer", "-- hazard rule layer"), ("p4_abls_flat_heads", "-- parent conditioning"),
       ("p4_s_k8", "8 global frames (K=8)"), ("p4_abls_gt_only", "human/sensor labels only (no VLM pseudo)"),
       ("p4_datas_ours_only", "train on ours only"), ("p4_datas_ext_only", "train on external only"),
       ("p4_datas_no_iowa", "without the summer source (Iowa)"),
       ("p4_base_mlp", "MLP probe, masked frames"), ("p4_abl_unmasked_mlp", "MLP probe, UNMASKED frames"),
       ("p5_det_mlp", "MLP probe + detection descriptors")]
COLS = [("fog", "Fog F1"), ("flood", "Flood AUROC"), ("snow", "Snow AUROC"), ("traffic", "Traffic F1"), ("illum", "Illum.\\ F1"),
        ("rain", "Rain AUROC"), ("wet", "Wet AUROC"), ("kor", "Korean CCTV precip.\\ F1"),
        ("gt_mean", "GT-mean"), ("warn", "Warn F1"), ("vlm_agree", "VLM-agree")]


def run_metrics(d):
    m = json.loads((d / "test_metrics.json").read_text())
    ps = pd.read_csv(d / "test_per_source.csv")
    g = ps[ps.labels == "gt"].set_index("source")
    get = lambda s, c: float(g.loc[s, c]) if s in g.index and c in g.columns and not pd.isna(g.loc[s, c]) else np.nan  # noqa: E731
    b = m.get("binary", {})
    r = {"fog": get("fvei_fog", "visibility_f1"), "flood": b.get("flood_floodcam_auroc", np.nan),
         "snow": b.get("snowcover_nysdot_auroc", np.nan), "traffic": get("norway_trafficcam", "traffic_f1"),
         "illum": get("itssnap", "illumination_f1"), "rain": b.get("rain_iowa_auroc", np.nan), "wet": b.get("wet_iowa_auroc", np.nan),
         "kor": get("tsbow", "precip_f1")}
    r["gt_mean"] = float(np.nanmean(list(r.values())))
    a = m["all"]
    r["warn"] = a.get("warn_tax_f1", a.get("warn_rule_f1", np.nan))
    ours = ps[(ps.labels == "all") & ps.source.isin(["itssnap", "hmm_pilot", "itslive"])]
    r["vlm_agree"] = float(np.average(ours.mean_f1, weights=ours.n)) if len(ours) else np.nan
    return r


def tag_of(r):
    if r.task == "eval":
        t = "zs_siglip2" if "zs_" in str(r.model) else "vlm_qwen25vl7b"
        rid = str(r.run_id)
        return t + ("_v2t" if rid.endswith("_v2t") else "_v2" if rid.endswith("_v2") else "")
    model = str(r.model).replace("/", "-").replace("\\", "-")
    t = r.run_id.split(f"_{r.task}_{model}_", 1)[-1]
    return t.rsplit("_s", 1)[0]


def fmt(s, k):
    v = s[k].dropna()
    if v.empty:
        return "--"
    return f"{100 * v.mean():.1f}" + (f"\\,$\\pm$\\,{100 * v.std():.1f}" if len(v) > 1 else "")


def table(runs, spec, path, bold_best=True):
    lines = ["\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule", "Method & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
    for t, name in spec:
        s = runs[runs.tag == t]
        if s.empty:
            continue
        lines.append(name + " & " + " & ".join(fmt(s, k) for k, _ in COLS) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines), encoding="utf-8")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    TEX.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(EXP / "runs_index.csv")
    idx = idx[(idx.status == "completed") & idx.task.isin(["taxowarn", "taxowarn_s", "finetune_img", "eval"])]
    rows = []
    for r in idx.itertuples():
        d = Path(r.run_dir)
        if not (d / "test_metrics.json").exists():
            continue
        try:
            rows.append({"run_id": r.run_id, "tag": tag_of(r), "seed": r.seed, **run_metrics(d)})
        except Exception as e:  # noqa
            print("skip", r.run_id, e)
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "all_runs_test.csv", index=False)
    num = runs.drop(columns=["run_id"]).groupby("tag").agg(["mean", "std", "count"])
    num.to_csv(OUT / "by_tag.csv")
    print(table(runs, MAIN, TEX / "main_results.tex"))
    print(table(runs, ABL, TEX / "ablations.tex"))
    used = runs[runs.tag.isin([t for t, _ in MAIN + ABL])]
    (OUT / "run_ids_per_tag.json").write_text(json.dumps(used.groupby("tag").run_id.apply(list).to_dict(), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()


