"""Validate a gold labelling CSV and convert it to HighwayMM_release/gold/gold_labels_<name>.csv.

Each attribute cell accepts the value name (e.g. fog_dense) or its number code (see LABELING_GUIDE.md).
illumination additionally accepts "unknown" (code 3; can't tell, e.g. inside a tunnel).
Works for the original kit (gold_labeling/labels.csv, column `no`) and the friend kit / LABEL_HERE.html export
(column `clip`, plus tunnel / transition).

  python import_gold_csv.py [--check]                                  (gold_labeling/labels.csv -> gold_labels_v1.csv)
  python import_gold_csv.py --src <labels.csv> --name friend [--check] (-> gold_labels_friend.csv)
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import ATTRS  # noqa: E402

ROOT = Path("D:/highwaymm")
YES = {"1", "y", "yes", "x", "true"}
EXTRA = {"illumination": ["unknown"]}


def parse(attr, v):
    v = v.strip().lower().replace(" ", "_")
    if v == "":
        return None
    vals = ATTRS[attr] + EXTRA.get(attr, [])
    if v in ("can't_tell", "cant_tell"):
        v = "unknown"
    if v.isdigit() and int(v) < len(vals):
        return vals[int(v)]
    if v in vals:
        return v
    raise ValueError(f"{attr}={v!r} not in {vals}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "gold_labeling" / "labels.csv"))
    ap.add_argument("--name", default="v1")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dst = ROOT / "HighwayMM_release" / "gold" / f"gold_labels_{a.name}.csv"
    rows = list(csv.DictReader(open(a.src, encoding="utf-8-sig")))
    out, errs, done, unusable = [], [], 0, 0
    for r in rows:
        rid = r.get("no") or r.get("clip") or "?"
        flag = lambda k: int(r.get(k, "").strip().lower() in YES)  # noqa: E731
        rec = {"gid": r["gid"], "unusable": flag("unusable"), "tunnel": flag("tunnel"), "transition": flag("transition"),
               "unsure": r.get("unsure", "").strip(), "notes": r.get("notes", "").strip()}
        if r.get("precip", "").strip().lower() in ("0", "none") and not r.get("intensity", "").strip():
            r["intensity"] = "none"  # no precipitation => intensity none (may be left blank)
        if rec["tunnel"] and not r.get("illumination", "").strip():
            r["illumination"] = "unknown"
        missing = []
        for att in ATTRS:
            try:
                rec[att] = parse(att, r.get(att, ""))
            except ValueError as e:
                errs.append(f"row {rid} ({r.get('file', '')}): {e}")
                rec[att] = None
            if rec[att] is None:
                missing.append(att)
        if rec["unusable"]:
            unusable += 1
        elif not missing:
            if (rec["precip"] == "none") != (rec["intensity"] == "none"):
                errs.append(f"row {rid} ({r.get('file', '')}): precip={rec['precip']} but intensity={rec['intensity']}")
            done += 1
        elif len(missing) < len(ATTRS):
            errs.append(f"row {rid} ({r.get('file', '')}): partially labelled, missing {missing}")
        out.append(rec)
    print(f"complete: {done}/{len(rows)}  unusable: {unusable}  problems: {len(errs)}")
    for e in errs[:50]:
        print("  " + e)
    if a.check:
        return
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gid", *ATTRS, "tunnel", "transition", "unusable", "unsure", "notes"])
        w.writeheader()
        for rec in out:
            w.writerow({k: ("" if v is None else v) for k, v in rec.items()})
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
