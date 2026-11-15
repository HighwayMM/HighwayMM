"""Re-render gold labelling strips from the UNMASKED frames (r{k}.jpg) at a larger size, for the human labellers only
(models still see masked frames). Layout: middle frame large on the left, start and end frames stacked on the right.
Writes NEW files to HighwayMM_release\\gold\\strips_raw\\<gid>.jpg (old strips/ and all frames are untouched)."""
from pathlib import Path

import pandas as pd
from PIL import Image

REL = Path(r"D:\highwaymm\HighwayMM_release")
BIG_W, SMALL_W, GAP = 1024, 512, 4


def main():
    g = pd.read_csv(REL / "gold" / "gold_v1.csv", keep_default_na=False)
    out = REL / "gold" / "strips_raw"
    out.mkdir(exist_ok=True)
    miss = 0
    for r in g.itertuples():
        d = REL / "frames" / r.clip_id
        pre = "r" if (d / "r4.jpg").exists() else "m"
        miss += pre == "m"
        f = [Image.open(d / f"{pre}{i}.jpg").convert("RGB") for i in (1, 4, 6)]
        W, H = f[1].size
        sh = round(SMALL_W * H / W)  # small frames keep their aspect ratio
        bh = 2 * sh + GAP
        big = f[1].resize((round(bh * W / H), bh), Image.LANCZOS)
        sm = [im.resize((SMALL_W, sh), Image.LANCZOS) for im in (f[0], f[2])]
        canvas = Image.new("RGB", (big.width + GAP + SMALL_W, bh), (20, 20, 20))
        canvas.paste(big, (0, 0))
        canvas.paste(sm[0], (big.width + GAP, 0))
        canvas.paste(sm[1], (big.width + GAP, sh + GAP))
        canvas.save(out / f"{r.gid}.jpg", quality=88)
    print("strips:", len(g), "| fell back to masked:", miss)


if __name__ == "__main__":
    main()
