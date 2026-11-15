"""Build an offline (File Explorer + Excel) labelling kit for the 400-clip gold set.

Copies (never moves) HighwayMM_release/gold/strips/<gid>.jpg into
D:/highwaymm/gold_labeling/images/<NNN>_<gid>.jpg in labelling order and writes a
labels.csv template. Originals are untouched.
"""
import csv, shutil, sys
from pathlib import Path

ROOT = Path("D:/highwaymm")
GOLD = ROOT / "HighwayMM_release" / "gold"
KIT = ROOT / "gold_labeling"
COLS = ["no", "file", "gid", "precip", "intensity", "surface", "visibility", "illumination", "traffic",
        "unusable", "unsure", "notes"]


def main():
    rows = list(csv.DictReader(open(GOLD / "gold_v1.csv", encoding="utf-8")))
    rows.sort(key=lambda r: int(r["order"]))
    img = KIT / "images"
    img.mkdir(parents=True, exist_ok=True)
    out = KIT / "labels.csv"
    if out.exists() and "--force" not in sys.argv:
        print(f"{out} exists; not overwriting (pass --force).")
        return
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for i, r in enumerate(rows):
            name = f"{i:03d}_{r['gid']}.jpg"
            dst = img / name
            if not dst.exists():
                shutil.copy2(GOLD / "strips" / f"{r['gid']}.jpg", dst)
            w.writerow([i, name, r["gid"]] + [""] * (len(COLS) - 3))
    print(f"kit: {len(rows)} images -> {img}\ntemplate -> {out}")


if __name__ == "__main__":
    main()
