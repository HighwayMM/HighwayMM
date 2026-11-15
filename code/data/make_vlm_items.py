"""Item list for VLM pseudo-labelling -> HighwayMM_release\\annotations\\vlm_items.csv
ours: 3 masked frames (m1, m4, m6) per clip; external: the single masked image (FVEI skipped: GT visibility is what it is for;
Norway subsampled to 2,500 for weather pseudo-labels; traffic GT exists for all)."""
from pathlib import Path

import pandas as pd

REL = Path(r"D:\highwaymm\HighwayMM_release")
rows = []
for d in sorted((REL / "frames").iterdir()):
    if not (d / "m0.jpg").exists():
        continue
    name = d.name
    if name.startswith("fvei_train"):
        continue  # FVEI test (900) is labelled so the VLM baseline can be scored against fog GT
    if (d / "m7.jpg").exists():
        imgs = [d / f"m{k}.jpg" for k in (1, 4, 6)]
    else:
        imgs = [d / "m0.jpg"]
    rows.append({"item_id": name, "images": ";".join(map(str, imgs))})
df = pd.DataFrame(rows)
nor = df[df.item_id.str.startswith("norway_")]
keep = pd.concat([df[~df.item_id.str.startswith("norway_")], nor.sample(n=min(2500, len(nor)), random_state=0),
                  nor[nor.item_id.str.contains("validation")]]).drop_duplicates("item_id")
keep.to_csv(REL / "annotations" / "vlm_items.csv", index=False)
print(len(df), "->", len(keep))
