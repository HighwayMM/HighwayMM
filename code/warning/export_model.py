"""Export a trained TaxoWarn-S run as a small, self-contained inference checkpoint (weights + model config + calibration
temperatures), without optimizer / scheduler / scaler / RNG state. The original run directory is not modified.

  python code\\warning\\export_model.py --run experiments\\<run_id> --out release_models\\taxowarn_s.pt
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true", help="store weights in fp16 (half the size; tiny numerical change)")
    a = ap.parse_args()
    run = Path(a.run)
    ck = torch.load(run / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    cfg = yaml.safe_load(open(run / "config.yaml", encoding="utf-8"))
    temps = json.loads((run / "temperatures.json").read_text(encoding="utf-8"))
    sd = {k: (v.half() if a.fp16 and v.is_floating_point() else v).contiguous() for k, v in ck["model"].items()}
    out = {"format": "highwaymm-taxowarn-v1", "task": cfg["task"], "model_cfg": cfg["model"], "k": cfg["data"].get("k", 8),
           "temperatures": temps, "run_id": cfg.get("run_id", run.name), "epoch": ck.get("epoch"), "val_best": ck.get("best"),
           "encoder": "SigLIP2-SO400M/16-384 (open_clip hf-hub:timm/ViT-SO400M-16-SigLIP2-384)", "state_dict": sd}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, a.out)
    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(f"saved {a.out}: {Path(a.out).stat().st_size / 2**20:.1f} MB, {sum(v.numel() for v in sd.values()) / 1e6:.2f}M params, "
          f"sha256 {h[:16]}")


if __name__ == "__main__":
    main()
