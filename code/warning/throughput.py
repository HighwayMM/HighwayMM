"""End-to-end throughput of the warning pipeline on ONE GPU: decode 8 frames from raw ITS clips (CPU workers) -> SigLIP2 encode
(fp16) -> TaxoWarn -> warning decision. Reports clips/s and cameras servable per GPU at a 1-clip-per-minute refresh.
Logged as a run (task=bench, model=warning_pipeline)."""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.prep_frames import read_frames  # noqa: E402
from train import ddp  # noqa: E402
from train.runlog import RunLogger  # noqa: E402
from train.tasks import taxowarn as T  # noqa: E402
from warning.system import WarningSystem  # noqa: E402
from data.taxonomy import warn_probs_from_attrs  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")


class Clips(Dataset):
    def __init__(self, paths, pre):
        self.p, self.pre = paths, pre

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        fr = read_frames(self.p[i], 8)
        ims = [self.pre(Image.fromarray(f[:, :, ::-1])) for f in fr]
        while len(ims) < 8:
            ims.append(ims[-1])
        return torch.stack(ims), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="trained TaxoWarn run dir")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    ctx = ddp.setup()
    man = pd.read_csv(REL / "manifest_ours.csv")
    paths = man[(man.source == "itssnap") & ~man.exclude].path.sample(n=a.n, random_state=0).tolist()
    cfg = {"task": "bench_warning", "seed": 0, "n_clips": a.n, "workers": a.workers, "bs": a.bs, "model_run": a.run,
           "data": {"version": "v1", "split": "none"}}
    with RunLogger(cfg, ctx, task="bench", model="warning_pipeline", tag="1gpu") as log:
        enc, _, pre = open_clip.create_model_and_transforms("local-dir:D:/highwaymm/models/SigLIP2-SO400M-384")
        enc = enc.cuda().eval().half()
        rc = yaml.safe_load(open(Path(a.run) / "config.yaml", encoding="utf-8"))
        rc["train"]["resume"] = None
        import importlib
        mod = importlib.import_module(f"train.tasks.{rc['task']}")
        model = mod.build(rc, ctx)["model"].cuda().eval()
        spatial = rc["task"] == "taxowarn_s"
        k1 = rc["data"].get("k", 8) == 1
        model.load_state_dict(torch.load(Path(a.run) / "checkpoints" / "best.pt", map_location="cuda", weights_only=False)["model"])
        ws = WarningSystem()
        dl = DataLoader(Clips(paths, pre), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True)
        t_enc = t_head = 0.0
        n = 0
        t0 = time.time()
        with torch.no_grad():
            for b, (x, i) in enumerate(dl):
                if b == 2:
                    torch.cuda.synchronize(); t0 = time.time(); n = 0; t_enc = t_head = 0.0
                B = x.shape[0]
                s = time.time()
                if spatial and k1:
                    # ONE trunk pass over frames 1,4,6 gives both the patch tokens and (via the attention-pool head) the
                    # global embedding of the middle frame -- numerically identical to encode_image (verified, max diff 0)
                    xx = x[:, [1, 4, 6]].reshape(-1, *x.shape[2:]).cuda(non_blocking=True).half()
                    tk = enc.visual.trunk.forward_features(xx)
                    g = enc.visual.head(enc.visual.trunk.forward_head(tk.view(B, 3, *tk.shape[1:])[:, 1]))
                    g = torch.nn.functional.normalize(g.float(), dim=-1)
                    gs = int(tk.shape[1] ** 0.5)
                    tp = torch.nn.functional.adaptive_avg_pool2d(tk[:, -gs * gs:].transpose(1, 2).reshape(tk.shape[0], -1, gs, gs).float(), 6)
                    e = torch.cat([g[:, None], torch.zeros(B, 7, g.shape[-1], device=g.device),
                                   tp.flatten(2).transpose(1, 2).reshape(B, 108, -1)], 1)
                    torch.cuda.synchronize(); t_enc += time.time() - s; s = time.time()
                    out = model(e)
                    pw = warn_probs_from_attrs({k: v.float().softmax(-1).cpu().numpy() for k, v in out["logits"].items()})
                    hz = out["hazard"].float().cpu().numpy()
                    for j in range(B):
                        ws.step(f"cam{int(i[j])}", time.time(), pw[j], hz[j], float(torch.sigmoid(out["onset_logit"][j])))
                    torch.cuda.synchronize(); t_head += time.time() - s
                    n += B
                    continue
                e = enc.encode_image(x.view(-1, *x.shape[2:]).cuda(non_blocking=True).half())
                e = torch.nn.functional.normalize(e.float(), dim=-1).view(B, 8, -1)
                if spatial:  # patch tokens of frames 1,4,6 pooled to 6x6 (same as extract_spatial.py)
                    xx = x[:, [1, 4, 6]].reshape(-1, *x.shape[2:]).cuda(non_blocking=True).half()
                    tk = enc.visual.trunk.forward_features(xx)
                    gs = int(tk.shape[1] ** 0.5)
                    tk = torch.nn.functional.adaptive_avg_pool2d(tk[:, -gs * gs:].transpose(1, 2).reshape(tk.shape[0], -1, gs, gs).float(), 6)
                    e = torch.cat([e, tk.flatten(2).transpose(1, 2).reshape(B, 108, -1)], 1)
                torch.cuda.synchronize(); t_enc += time.time() - s; s = time.time()
                out = model(e)
                pw = warn_probs_from_attrs({k: v.float().softmax(-1).cpu().numpy() for k, v in out["logits"].items()})
                hz = out["hazard"].float().cpu().numpy()
                for j in range(B):
                    ws.step(f"cam{int(i[j])}", time.time(), pw[j], hz[j], float(torch.sigmoid(out["onset_logit"][j])))
                torch.cuda.synchronize(); t_head += time.time() - s
                n += B
        wall = time.time() - t0
        res = {"clips_per_s": n / wall, "encode_ms_per_clip": 1000 * t_enc / n, "head_and_decision_ms_per_clip": 1000 * t_head / n,
               "cameras_per_gpu_at_1_clip_per_min": int(60 * n / wall), "frames_encoded_per_clip": 3 if (spatial and k1) else (11 if spatial else 8), "gpu": torch.cuda.get_device_name(0),
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        print(res)
        log.finish("completed", final_metrics=res, key_metrics=res)
    ddp.cleanup()


if __name__ == "__main__":
    main()


