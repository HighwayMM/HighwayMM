"""Main-paper (8-page) variants of the generated tables; the full tables go to the supplement.
Run after make_tables.py and warning_report.py:
  python code\\report\\compact_tables.py
  main_results.tex -> main_results_main.tex  (drops K=1 probe/flat rows; reported in the supplement)
  ablations.tex    -> ablations_main.tex     (drops the K=8 row, identical to TaxoWarn-S (K=8) in the main table)
  warning.tex      -> warning_main.tex       (drops TaxoWarn-v1 and K=8 rows and the "lead w/o early" column)
"""
from pathlib import Path

T = Path(r"D:\highwaymm\paper\tables")


def filt(src, dst, drop_rows, drop_last_col=False):
    out = []
    for line in (T / src).read_text(encoding="utf-8").splitlines():
        if any(line.lstrip().startswith(d) or (" & " + d + " & ") in (" " + line) for d in drop_rows):
            continue
        if drop_last_col and "&" in line and line.rstrip().endswith("\\\\"):
            line = " & ".join(line.rstrip()[:-2].split(" & ")[:-1]) + " \\\\"
        if drop_last_col and line.startswith("\\begin{tabular}"):
            line = line.replace("cccc}", "ccc}")
        out.append(line)
    (T / dst).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(dst, len(out), "lines")


filt("main_results.tex", "main_results_main.tex", ["Linear probe (K=1)", "MLP probe (K=1)", "Flat weather classes (K=1)"])
filt("ablations.tex", "ablations_main.tex", ["8 global frames (K=8)"])
filt("warning.tex", "warning_main.tex", ["TaxoWarn (v1)", "TaxoWarn-S (K=8)"], drop_last_col=True)
