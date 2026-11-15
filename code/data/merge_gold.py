"""Merge gold labels from the web page (DB export JSON, labeller A) and offline kits (imported CSVs) into
HighwayMM_release/gold/gold_labels_merged.csv, and report inter-annotator agreement on clips labelled by both.

  python merge_gold.py <db_export_clips_dir> <gold_labels_X.csv> [...]
Conflict rule: first source listed wins (the page labels of the project owner); agreement table reports overlaps.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy import ATTRS  # noqa: E402

GOLD = Path(r"D:\highwaymm\HighwayMM_release\gold")
A = list(ATTRS)


def from_db(d):
    rows = []
    for f in sorted(Path(d).glob("*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        v = j.get("data", j)
        rows.append({"gid": j.get("id", f.stem), **{a: v.get(a, "") or "" for a in A},
                     "tunnel": int(bool(v.get("tunnel"))), "transition": int(bool(v.get("transitional"))),
                     "unusable": int(bool(v.get("unusable"))), "unsure": "1" if v.get("unsure") else "", "notes": ""})
    return pd.DataFrame(rows)


def main():
    srcs = []
    for s in sys.argv[1:]:
        p = Path(s)
        df = from_db(p) if p.is_dir() else pd.read_csv(p, keep_default_na=False, dtype=str)
        df = df[(df.unusable.astype(str) == "1") | (df[A] != "").all(axis=1)].copy()  # finished rows only
        df["labeller"] = "owner" if p.is_dir() else p.stem.replace("gold_labels_", "")
        srcs.append(df)
        print(f"{df.labeller.iloc[0] if len(df) else s}: {len(df)} clips")
    order = pd.read_csv(GOLD / "gold_v1.csv", keep_default_na=False)[["gid", "order"]]
    allr = pd.concat(srcs, ignore_index=True).merge(order, on="gid")
    allr["clip"] = allr["order"] + 1
    # agreement on overlaps
    labs = allr.labeller.unique()
    if len(labs) >= 2:
        a, b = (allr[allr.labeller == l].set_index("gid") for l in labs[:2])
        both = [g for g in a.index if g in b.index and str(a.at[g, "unusable"]) != "1" and str(b.at[g, "unusable"]) != "1"]
        print(f"overlap {labs[0]} vs {labs[1]}: {len(both)} clips")
        if both:
            rep = []
            for att in A:
                ids = [g for g in both if a.at[g, att] != "unknown" and b.at[g, att] != "unknown"]
                x, y = [a.at[g, att] for g in ids], [b.at[g, att] for g in ids]
                agree = 100 * sum(p == q for p, q in zip(x, y)) / max(len(ids), 1)
                k = cohen_kappa_score(x, y) if len(set(x) | set(y)) > 1 else float("nan")
                rep.append({"attr": att, "n": len(ids), "agree_pct": round(agree, 1), "kappa": round(k, 3)})
            rep = pd.DataFrame(rep)
            rep.to_csv(GOLD / "gold_agreement.csv", index=False)
            print(rep.to_string(index=False))
    merged = allr.sort_values("clip").drop_duplicates("gid", keep="first")  # first source listed wins
    merged.to_csv(GOLD / "gold_labels_merged.csv", index=False)
    print(f"merged: {len(merged)} clips ({(merged.unusable.astype(str) == '1').sum()} unusable) -> gold_labels_merged.csv")
    for att in A:
        print(att, merged[merged.unusable.astype(str) != "1"][att].value_counts().to_dict())


if __name__ == "__main__":
    main()
