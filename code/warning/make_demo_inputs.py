"""Build the replay-demo input folders (hard links into D:\\highwaymm\\demo\\inputs; originals are never modified).

  nysdot/<camera>/*.jpg     + nysdot_gt.csv     NYSDOT 511NY blizzard, Jan 2022 (CC BY 4.0, test split, human labels)
  floodcam/<camera>/*.jpg   + floodcam_gt.csv   FloodCam NC, Hurricane Teddy 2020 (held-out cameras Buxton, Canal)
  ours_winter.txt           list of our winter-pilot test clips (passed directly, one camera per clip)
GT levels follow warning/eval_warning.py (human labels only)."""
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from warning.eval_warning import gt_level  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
OUT = Path(r"D:\highwaymm\demo\inputs")
OURS = ["hmm_Rec_0084", "hmm_Rec_0103", "hmm_Rec_0147", "hmm_Rec_0151", "hmm_Rec_0183", "hmm_Rec_0021"]


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    me = pd.read_csv(REL / "manifest_external.csv", keep_default_na=False)
    for name, src, cams in (("nysdot", "nysdot_511ny", None), ("floodcam", "floodcam_nc", ["ncdot_Buxton", "ncdot_Canal"])):
        g = me[me.source == src]
        if cams:
            g = g[g.camera_id.isin(cams)]
        rows = []
        for r in g.itertuples():
            p = Path(r.path)
            link(p, OUT / name / r.camera_id / p.name)
            rows.append({"file": p.name, "gt_level": gt_level({"note": r.gt_note, "source": r.source})})
        pd.DataFrame(rows).to_csv(OUT / f"{name}_gt.csv", index=False)
        print(name, len(rows), g.camera_id.value_counts().to_dict())
    mo = pd.read_csv(REL / "manifest_ours.csv").set_index("clip_id")
    (OUT / "ours_winter.txt").write_text("\n".join(mo.loc[c, "path"] for c in OURS), encoding="utf-8")
    print("ours", len(OURS))


if __name__ == "__main__":
    main()
