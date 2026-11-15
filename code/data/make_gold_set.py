"""D3: select the human-verified gold test set (400 clips of OUR data, camera-disjoint TEST split only) and render
3-frame strips for the labelling page. Oversamples clips with official alert banners (rain/snow/fog/ice), the winter pilot
and live sequences; fills the rest at random, balanced day/night. Output: HighwayMM_release\\gold\\{gold_v1.csv, strips\\*.jpg}"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.build_labels import hsplit  # noqa: E402
from data.taxonomy import parse_alert  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
N = 400


def main():
    man = pd.read_csv(REL / "manifest_ours.csv")
    man = man[~man.exclude].copy()
    man["split"] = [hsplit(c) for c in man.camera_id]
    has = np.array([(REL / "frames" / c / "m7.jpg").exists() for c in man.clip_id])
    te = man[(man.split == "test").to_numpy() & has].copy()
    tags = []
    for c in te.clip_id:
        j = REL / "annotations" / "ocr" / f"{c}.json"
        tags.append(";".join(parse_alert(json.loads(j.read_text(encoding="utf-8")).get("top_text", ""))) if j.exists() else "")
    te["alert_tags"] = tags
    weather = te.alert_tags.str.contains("rain|snow|fog|icy|heavy|flood|wind|typhoon")
    rng = np.random.default_rng(0)
    pick = pd.concat([te[weather], te[te.source == "hmm_pilot"], te[te.source == "itslive"].groupby("camera_id").head(2)]).drop_duplicates("clip_id")
    rest = te[~te.clip_id.isin(pick.clip_id)]
    need = max(N - len(pick), 0)
    pick = pd.concat([pick, rest.sample(n=min(need, len(rest)), random_state=0)]).head(N)
    pick = pick.sample(frac=1, random_state=1)  # shuffle presentation order
    out = REL / "gold"
    (out / "strips").mkdir(parents=True, exist_ok=True)
    rows = []
    for k, r in enumerate(pick.itertuples()):
        gid = "g" + hashlib.md5(r.clip_id.encode()).hexdigest()[:10]  # opaque id: no source/camera hint for labellers
        ims = [Image.open(REL / "frames" / r.clip_id / f"m{i}.jpg").convert("RGB") for i in (1, 4, 6)]
        w = 420
        ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
        strip = Image.new("RGB", (w * 3 + 8, ims[0].height), (20, 20, 20))
        for i, im in enumerate(ims):
            strip.paste(im, (i * (w + 4), 0))
        strip.save(out / "strips" / f"{gid}.jpg", quality=82)
        rows.append({"gid": gid, "clip_id": r.clip_id, "source": r.source, "alert_tags": r.alert_tags, "order": k})
    g = pd.DataFrame(rows)
    g.to_csv(out / "gold_v1.csv", index=False)
    (out / "clips.json").write_text(json.dumps([{"id": x} for x in g.gid]), encoding="utf-8")
    print(len(g), g.source.value_counts().to_dict(), "alerts:", (g.alert_tags != "").sum())


if __name__ == "__main__":
    main()
