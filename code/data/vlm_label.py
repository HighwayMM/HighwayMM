"""Taxonomy pseudo-labels with Qwen2.5-VL-7B (one model per GPU, sharded via jobrunner).

Input CSV (--items): item_id, images  (images = ';'-separated paths, overlay-masked frames of one clip/sample)
Output JSONL (--out_dir/shard{i}.jsonl): {"item_id", "raw", "labels": {attr: value}, "conf": {attr: low|medium|high}, "transitional", "ok"}
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS  # noqa: E402

MODEL = r"D:\highwaymm\models\Qwen2.5-VL-7B-Instruct"

PROMPT = """You are a road-weather analyst looking at {n} frames from one fixed roadside traffic camera (gray boxes are masked text overlays; ignore them).
Classify the CURRENT conditions. Answer ONLY with a JSON object with exactly these keys and allowed values:
"precip": "none"|"rain"|"snow"            (falling precipitation visible: streaks, flakes, drops on lens, spray)
"intensity": "none"|"light"|"moderate"|"heavy"   (of the falling precipitation; "none" if precip is none)
"surface": "dry"|"wet"|"water"|"snow"|"ice"     ("water" = standing water or flooding on the road; "snow" = snow or slush covering the road; "ice" = glazed/icy road)
"visibility": "clear"|"haze"|"fog_moderate"|"fog_dense"   ("fog_dense" = far road barely visible, < ~100 m; heavy blowing snow also reduces visibility)
"illumination": "day"|"twilight"|"night"
"traffic": "free"|"dense"|"congested"   ("congested" = queues / stop-and-go)
"transitional": true|false   (conditions visibly changing between frames, e.g. rain starting, water rising, snow beginning to cover the road)
"confidence": "low"|"medium"|"high"
"""


def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    lab = {a: (d.get(a) if d.get(a) in v else None) for a, v in ATTRS.items()}
    if lab.get("precip") == "none":
        lab["intensity"] = "none"
    return {"labels": lab, "transitional": bool(d.get("transitional", False)), "confidence": d.get("confidence")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--bs", type=int, default=6)
    ap.add_argument("--max_side", type=int, default=560)
    a = ap.parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    of = out / f"shard{a.shard}.jsonl"
    done = set()
    if of.exists():
        done = {json.loads(l)["item_id"] for l in open(of, encoding="utf-8")}
    items = pd.read_csv(a.items).iloc[a.shard::a.nshards]
    items = items[~items.item_id.isin(done)]
    print(f"[vlm shard {a.shard}] {len(items)} items to label", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                               device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(MODEL, min_pixels=128 * 28 * 28, max_pixels=a.max_side * a.max_side)
    proc.tokenizer.padding_side = "left"
    f = open(of, "a", encoding="utf-8")
    rows = list(items.itertuples())
    for b in range(0, len(rows), a.bs):
        batch = rows[b:b + a.bs]
        texts, imgs = [], []
        for r in batch:
            paths = str(r.images).split(";")
            ims = []
            for p in paths:
                im = Image.open(p).convert("RGB")
                s = a.max_side / max(im.size)
                if s < 1:
                    im = im.resize((int(im.width * s), int(im.height * s)))
                ims.append(im)
            content = [{"type": "image"} for _ in ims] + [{"type": "text", "text": PROMPT.format(n=len(ims))}]
            msgs = [{"role": "user", "content": content}]
            texts.append(proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            imgs.extend(ims)
        inp = proc(text=texts, images=imgs, padding=True, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=160, do_sample=False)
        dec = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        for r, t in zip(batch, dec):
            p = parse(t)
            f.write(json.dumps({"item_id": r.item_id, "raw": t, "ok": p is not None, **(p or {})}, ensure_ascii=False) + "\n")
        f.flush()
        if (b // a.bs) % 50 == 0:
            print(f"[vlm shard {a.shard}] {b + len(batch)}/{len(rows)}", flush=True)
    print(f"[vlm shard {a.shard}] done")


if __name__ == "__main__":
    main()
