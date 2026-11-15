"""Offline labelling kit for a second labeller (File Explorer + Excel), clips FIRST..LAST in the web page's numbering
("Clip N of 400" = gold_v1 order + 1). Uses the unmasked, larger strips (gold/strips_raw). Copies only; never overwrites
an existing labels.csv. Output: D:/highwaymm/gold_labeling_friend/{images/, labels.csv, LABELING_GUIDE.md}

  python code/data/make_gold_friend_kit.py [FIRST LAST]     (default 181 400)
"""
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path("D:/highwaymm")
GOLD = ROOT / "HighwayMM_release" / "gold"
KIT = ROOT / "gold_labeling_friend"
COLS = ["clip", "file", "gid", "precip", "intensity", "surface", "visibility", "illumination", "traffic",
        "tunnel", "transition", "unusable", "unsure", "notes"]

GUIDE = """# Road-weather labelling guide (clips {first}-{last}, about 2 hours)

Thank you for helping! You will label what the weather and road look like in short roadside-camera clips.

## Easiest way: click, don't type
1. Unzip the whole folder (keep `images`, `clips.js` and `LABEL_HERE.html` together).
2. Double-click **`LABEL_HERE.html`**. It opens in Chrome or Edge, no internet needed.
3. Click the options (or use keys 1-5) and press Enter for the next clip. Your work is saved in the browser
   automatically, so you can close it and continue later (use the same browser and do not clear its data).
4. When finished, click **Download labels.csv** and send that file back.
   Tip: download it now and then as a backup; "resume from a CSV" loads it back.

The rest of this guide explains the labels. (You can also type codes into `labels.csv` in Excel instead.)

## What to open
- **Images:** the `images` folder. In File Explorer choose View > Extra large icons, or open the first image and use the
  arrow keys in the Photos app.
- Each image is ONE clip of about 10 seconds: the **large picture on the left is the middle** of the clip, the two small
  ones on the right are the **start (top)** and **end (bottom)**.
- **Sheet:** `labels.csv`. Open it in Excel. Column `clip` = the number at the start of the image name,
  e.g. image `183_g....jpg` goes in the row with clip = 183. Label in order.
- When Excel asks, save as **CSV (UTF-8)**. Do not rename, sort or delete columns.

## What to type: the number code (faster) or the word
| Column | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| **precip** (what is falling) | none | rain | snow | | |
| **intensity** (how hard it falls) | none | light | moderate | heavy | |
| **surface** (the road) | dry | wet | water | snow | ice |
| **visibility** | clear | haze | fog_moderate | fog_dense | |
| **illumination** (light) | day | twilight | night | unknown (can't tell) | |
| **traffic** | free | dense | congested | | |

Extra columns (type `1` for yes, leave blank for no):
- **tunnel**: the camera is inside a tunnel. Then type `3` (unknown) for illumination, and still label surface,
  visibility and traffic. Precip is usually 0 inside a tunnel.
- **transition**: the weather or light clearly changes between the start and the end frames.
- **unusable**: black, frozen or broken image, no road visible, or not a traffic scene. Leave the other columns blank.
- **unsure**: you guessed. Optionally list which columns, e.g. `surface,visibility`.
- **notes**: optional free text.

## Definitions (judge only what you see in the picture)
- **Judge from the scene, not from on-screen text.** Some cameras show text or alert banners (e.g. a rain warning).
  Ignore them and label only what the picture shows.
- **precip / intensity**
  - Rain: streaks, drops on the lens, splashes, spray behind vehicles. Snow: falling flakes.
  - Light = barely visible; moderate = clearly visible; heavy = it hides the scene or makes spray clouds.
  - If precip = none, intensity is none (you may leave intensity blank). If rain or snow, intensity must be 1, 2 or 3.
  - A wet road with nothing falling is precip none + surface wet.
- **surface**
  - Dry.
  - Wet: dark, glossy, reflections of lights.
  - Water: standing water or puddles in the lanes, or flooding.
  - Snow: snow or slush on the road itself, not only on the roadside.
  - Ice: only if clearly visible glazed ice. If in doubt choose wet and mark unsure.
- **visibility**
  - Clear: you can see far along the road.
  - Haze: distant things look washed out, but the road is visible far away.
  - fog_moderate: the distant road or objects disappear (roughly 100-500 m).
  - fog_dense: only the nearby road is visible (under about 100 m).
  - Dirt or raindrops on the lens are not fog (write them in notes).
- **illumination**
  - Day: daylight, including overcast. Twilight: dawn or dusk, sky dim but not dark.
  - Night: dark sky, scene lit by lamps or headlights. Unknown: you really cannot tell (e.g. tunnel).
- **traffic**
  - Free: vehicles with large gaps, or an empty road. Dense: many vehicles, short gaps, still moving.
  - Congested: queues, stopped or crawling. Compare the frames: if cars stay in place it is congested.

## Tips
- Fill illumination and traffic first (quick), then the weather columns.
- Most clips are ordinary clear weather: precip 0, intensity 0, surface 0, visibility 0.
  In Excel select cells and press Ctrl+D to copy a value down.
- When you finish, save the CSV and send back only `labels.csv`.
"""


def main():
    first, last = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (181, 400)
    rows = sorted(csv.DictReader(open(GOLD / "gold_v1.csv", encoding="utf-8")), key=lambda r: int(r["order"]))
    img = KIT / "images"
    img.mkdir(parents=True, exist_ok=True)
    out = KIT / "labels.csv"
    if out.exists() and "--force" not in sys.argv:
        print(f"{out} exists; not overwriting.")
        return
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            n = int(r["order"]) + 1  # page numbering
            if not first <= n <= last:
                continue
            name = f"{n:03d}_{r['gid']}.jpg"
            if not (img / name).exists():
                shutil.copy2(GOLD / "strips_raw" / f"{r['gid']}.jpg", img / name)
            w.writerow([n, name, r["gid"]] + [""] * (len(COLS) - 3))
    (KIT / "LABELING_GUIDE.md").write_text(GUIDE.format(first=first, last=last), encoding="utf-8")
    # clip list for the click-to-label page LABEL_HERE.html (works offline from file://)
    import json
    clips = [{"clip": int(r["order"]) + 1, "gid": r["gid"], "file": f"{int(r['order']) + 1:03d}_{r['gid']}.jpg"}
             for r in rows if first <= int(r["order"]) + 1 <= last]
    (KIT / "clips.js").write_text("const CLIPS = " + json.dumps(clips) + ";\n", encoding="utf-8")
    print(f"kit: clips {first}-{last} -> {KIT}")


if __name__ == "__main__":
    main()
