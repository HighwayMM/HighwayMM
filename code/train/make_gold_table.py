"""Paper table: models vs. the human gold set of our Korean CCTV (experiments/gold_eval_p6/per_tag.csv, from eval_gold.py).
  python code\\train\\make_gold_table.py  -> paper/tables/gold.tex (+ prints max seed std)
"""
from pathlib import Path

import pandas as pd

T = pd.read_csv(r"D:\highwaymm\experiments\gold_eval_p6\per_tag.csv").set_index("tag")
ROWS = [
    ("Qwen2.5-VL-7B (pseudo-labeller)", "vlm_qwen25vl7b"),
    ("Flat weather classes", "p4_base_flat"),
    ("ConvNeXt-B fine-tuned", "p4_base_ft_convnext"),
    ("Linear probe", "p4_base_linear"),
    ("MLP probe", "p4_base_mlp"),
    ("\\quad + detection descriptors", "p5_det_mlp"),
    ("\\quad unmasked frames", "p4_abl_unmasked_mlp"),
    ("\\quad human/sensor labels only", "p4_abls_gt_only"),
    ("\\quad train on external only", "p4_datas_ext_only"),
    ("TaxoWarn-S (final)", "p4_final"),
]
COLS = [("precip", "Prec."), ("surface", "Surf."), ("visibility", "Vis."), ("illumination", "Illum."),
        ("traffic", "Traf."), ("attr_mean", "Attr"), ("warn", "Warn")]
# hazard-F1 (level>=1) is omitted: 42% of gold clips are level>=1, so "always hazard" already scores 59.2


def main():
    best = {c: max(T.loc[t, f"{c}_f1_mean"] for _, t in ROWS) for c, _ in COLS}
    lines = ["\\begin{tabular}{@{}l" + "c" * len(COLS) + "@{}}", "\\toprule",
             "Method & " + " & ".join(h for _, h in COLS) + " \\\\", "\\midrule"]
    mx = 0.0
    for name, tag in ROWS:
        r = T.loc[tag]
        cells = []
        for c, _ in COLS:
            v = r[f"{c}_f1_mean"]
            s = f"{v:.1f}"
            cells.append(f"\\textbf{{{s}}}" if abs(v - best[c]) < 0.05 else s)
            sd = r.get(f"{c}_f1_std")
            if tag != "vlm_qwen25vl7b" and pd.notna(sd):
                mx = max(mx, sd)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
        if tag == "vlm_qwen25vl7b":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    out = Path(r"D:\highwaymm\paper\tables\gold.tex")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("n gold:", int(T.loc["p4_base_mlp", "n_mean"]), "| max seed std:", round(mx, 1))


if __name__ == "__main__":
    main()
