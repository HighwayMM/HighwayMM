"""Score models on the human-labelled gold set (400 test-split clips of OUR Korean highway/national-road CCTV).

Reads the saved, calibrated test predictions of every completed p4 run (run_dir/test_predictions.jsonl, written by
evaluate.py on labels_v2t; all 400 gold clips are test-split) and the Qwen2.5-VL labels. No retraining, no GPU.
Also measures how often the VLM pseudo-labels (used as training labels on our data) agree with the humans.

  python code\\train\\eval_gold.py                      -> experiments\\gold_eval\\{per_run,per_tag,vlm_noise}.csv
  python code\\train\\eval_gold.py --gold <csv> --out <dir>   (dry run on another label file)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS, warning_level, warn_probs_from_attrs  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")
EXP = Path(r"D:\highwaymm\experiments")
A = list(ATTRS)


def load_gold(path):
    g = pd.read_csv(path, keep_default_na=False, dtype=str)
    g = g[g.unusable.astype(str) != "1"]
    g = g[(g[A] != "").all(axis=1)]  # completely labelled rows only
    # illumination "unknown" (can't tell, e.g. inside a tunnel): row kept, illumination not scored for it
    m = pd.read_csv(REL / "gold" / "gold_v1.csv", keep_default_na=False, dtype=str)[["gid", "clip_id"]]
    g = g.merge(m, on="gid")
    g["warn"] = [warning_level({a: r[a] for a in A if r[a] != "unknown"}, False) for _, r in g.iterrows()]
    return g.set_index("clip_id")


def score(pred_lab, g):
    """pred_lab: {clip_id: {attr: value, 'warn': int}} -> metrics dict"""
    ids = [i for i in g.index if i in pred_lab]
    out = {"n": len(ids)}
    if not ids:
        return out
    for a in A + ["warn"]:
        ia = [i for i in ids if str(g.at[i, a]) != "unknown"]
        y = [str(g.at[i, a]) for i in ia]
        p = [str(pred_lab[i][a]) for i in ia]
        out[f"{a}_f1"] = 100 * f1_score(y, p, labels=sorted(set(y)), average="macro", zero_division=0)
        out[f"{a}_acc"] = 100 * float(np.mean(np.array(y) == np.array(p)))
    # binary hazard view: any warning (level >= 1) vs normal
    y = np.array([int(g.at[i, "warn"]) >= 1 for i in ids])
    p = np.array([int(pred_lab[i]["warn"]) >= 1 for i in ids])
    out["hazard_f1"] = 100 * f1_score(y, p, zero_division=0) if y.any() else np.nan
    out["attr_mean_f1"] = float(np.mean([out[f"{a}_f1"] for a in A]))
    return out


def run_preds(run_dir):
    f = Path(run_dir) / "test_predictions.jsonl"
    if not f.exists():
        return None
    d = {}
    for l in open(f, encoding="utf-8"):
        p = json.loads(l)
        lab = {a: ATTRS[a][int(np.argmax(p[f"p_{a}"]))] for a in A}
        P = {a: np.array([p[f"p_{a}"]]) for a in A}
        lab["warn"] = int(np.argmax(warn_probs_from_attrs(P)[0]))  # taxonomy pushforward
        d[p["item_id"]] = lab
    return d


def vlm_preds():
    d = {}
    for f in (REL / "annotations" / "vlm_qwen25vl7b").glob("shard*.jsonl"):
        for l in open(f, encoding="utf-8"):
            j = json.loads(l)
            if j.get("ok"):
                lab = {a: (j["labels"].get(a) or "") for a in A}
                lab["warn"] = warning_level({a: v for a, v in lab.items() if v}, bool(j.get("transitional")))
                d[j["item_id"]] = lab
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(REL / "gold" / "gold_labels_v1.csv"))
    ap.add_argument("--out", default=str(EXP / "gold_eval"))
    ap.add_argument("--match", default="_p4_")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    g = load_gold(a.gold)
    print("gold clips usable:", len(g), "| warn levels:", g.warn.value_counts().to_dict())
    runs = pd.read_csv(EXP / "runs_index.csv")
    runs = runs[(runs.status == "completed") & runs.run_id.str.contains(a.match)]
    rows = []
    for r in runs.itertuples():
        pl = run_preds(r.run_dir)
        if pl is None:
            continue
        import re
        mt = re.search(r"_(p\d+_.+)_s\d+$", r.run_id)
        tag = mt.group(1) if mt else r.run_id
        rows.append({"run_id": r.run_id, "tag": tag, "seed": r.seed, **score(pl, g)})
    vl = vlm_preds()
    rows.append({"run_id": "vlm_qwen25vl7b", "tag": "vlm_qwen25vl7b", "seed": 0, **score(vl, g)})
    per_run = pd.DataFrame(rows)
    per_run.to_csv(out / "per_run.csv", index=False)
    num = per_run.select_dtypes("number").columns.drop("seed")
    per_tag = per_run.groupby("tag")[num].agg(["mean", "std"])
    per_tag.columns = [f"{c}_{s}" for c, s in per_tag.columns]
    per_tag = per_tag.sort_values("attr_mean_f1_mean", ascending=False)
    per_tag.to_csv(out / "per_tag.csv")
    # VLM pseudo-label noise on our domain (these labels train the models on our data)
    noise = []
    for att in A:
        ids = [i for i in g.index if i in vl and vl[i][att]]
        if ids:
            agree = np.mean([vl[i][att] == g.at[i, att] for i in ids])
            noise.append({"attr": att, "n": len(ids), "vlm_agree_pct": round(100 * agree, 1)})
    pd.DataFrame(noise).to_csv(out / "vlm_noise.csv", index=False)
    cols = [c for c in per_tag.columns if c.endswith("_mean") and ("f1" in c)]
    print(per_tag[cols].round(1).to_string())
    print(pd.DataFrame(noise).to_string(index=False))


if __name__ == "__main__":
    main()
