"""Model selection on the VALIDATION split only (human/physics labels), never on test.
Usage: python code\\report\\val_select.py <tag-substring> [...]   (evaluates runs on val if needed, prints a table)
Score = mean of fog F1 (FVEI val), traffic F1 (Norway val), illumination F1 (ours val, physics), flood AUROC (FloodCam val)."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(r"D:\highwaymm\experiments")
PY = sys.executable


def row(run):
    if not (run / "val_metrics.json").exists():
        subprocess.run([PY, r"D:\highwaymm\code\train\evaluate.py", "--run", str(run), "--split", "val"], capture_output=True)
    m = json.loads((run / "val_metrics.json").read_text())
    d = pd.read_csv(run / "val_per_source.csv")
    g = d[d.labels == "gt"].set_index("source")
    get = lambda s, c: float(g.loc[s, c]) if s in g.index and c in g.columns else np.nan  # noqa: E731
    r = {"run": run.name, "fog": get("fvei_fog", "visibility_f1"), "traffic": get("norway_trafficcam", "traffic_f1"),
         "illum": get("itssnap", "illumination_f1"), "flood_auc": m.get("binary", {}).get("flood_floodcam_auroc", np.nan),
         "silver": m["all"].get("mean_f1"), "warn_tax": m["all"].get("warn_tax_f1")}
    r["score"] = np.nanmean([r["fog"], r["traffic"], r["illum"], r["flood_auc"]])
    return r


def main():
    keys = sys.argv[1:]
    idx = pd.read_csv(EXP / "runs_index.csv")
    idx = idx[(idx.status == "completed") & idx.run_id.apply(lambda x: any(k in x for k in keys))]
    rows = [row(Path(r)) for r in idx.run_dir]
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    pd.set_option("display.width", 200)
    print((df.set_index("run") * 100).round(1).to_string())


if __name__ == "__main__":
    main()
