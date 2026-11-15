"""Mean±std over seeds of GT-test metrics (test_metrics.json 'gt' + 'binary') for given tags.
  python code\\train\\compare_tags.py p4_base_mlp p5_det_mlp ...
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

runs = pd.read_csv(r"D:\highwaymm\experiments\runs_index.csv")
runs = runs[runs.status == "completed"]
keys = ["mean_f1", "precip_f1", "surface_f1", "visibility_f1", "traffic_f1", "warn_tax_f1"]
rows = []
for tag in sys.argv[1:]:
    for r in runs[runs.run_id.str.contains(rf"_{re.escape(tag)}_s\d+$")].itertuples():
        f = Path(r.run_dir) / "test_metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        row = {"tag": tag, **{k: 100 * m["gt"][k] for k in keys}}
        for k, v in m.get("binary", {}).items():
            if k.endswith("_auroc"):
                row[k.replace("_auroc", "")] = 100 * v
        rows.append(row)
d = pd.DataFrame(rows)
g = d.groupby("tag", sort=False)
out = g.mean().round(1).astype(str) + "±" + g.std().round(1).astype(str)
out.insert(0, "n", g.size())
pd.set_option("display.width", 250)
print(out.T.to_string())
