"""Baseline: end-to-end fine-tuned image classifier (ConvNeXt / ViT via timm) on the middle overlay-masked frame,
with independent attribute heads (same labels, weights, loss and evaluation as TaxoWarn).
cfg.model.name = timm model id (e.g. convnext_base.fb_in22k_ft_in1k), cfg.data.img_size."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as Tv

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data.taxonomy import ATTRS  # noqa: E402
from train.tasks.taxowarn import A, Loss, make_eval, make_targets  # noqa: E402

REL = Path(r"D:\highwaymm\HighwayMM_release")


class ImgDS(Dataset):
    def __init__(self, df, sources, size, train):
        keep = [(REL / "frames" / i / "m0.jpg").exists() for i in df.item_id]
        self.df = df[np.array(keep)].reset_index(drop=True)
        self.t = make_targets(self.df, sources)
        norm = Tv.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.tf = Tv.Compose(([Tv.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(1.2, 1.9)), Tv.RandomHorizontalFlip()] if train
                              else [Tv.Resize((size, size))]) + [Tv.ToTensor(), norm])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        d = REL / "frames" / self.df.item_id.iloc[i]
        f = d / "m4.jpg" if (d / "m4.jpg").exists() else d / "m0.jpg"
        return self.tf(Image.open(f).convert("RGB")), self.t[i], i


class Net(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.body = timm.create_model(name, pretrained=True, num_classes=0)
        d = self.body.num_features
        self.heads = nn.ModuleDict({a: nn.Linear(d, len(v)) for a, v in ATTRS.items()})

    def forward(self, x):
        z = self.body(x)
        return {"logits": {a: h(z) for a, h in self.heads.items()}}


def _prep(d):
    df = pd.read_csv(d["labels"], keep_default_na=False)
    for c in ["warn_level", "transitional"] + [f"w_{a}" for a in A]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1 if c == "warn_level" else 0)
    return df


def test_dataset(cfg, te, sources):
    return ImgDS(te, sources, cfg["data"].get("img_size", 224), train=False)


def build(cfg, ctx):
    d = cfg["data"]
    df = _prep(d)
    sources = sorted(df.source.unique())
    tr, va = df[df.split == "train"], df[df.split == "val"]
    size = d.get("img_size", 224)
    train_ds, val_ds = ImgDS(tr, sources, size, True), ImgDS(va, sources, size, False)
    cnt = tr.source.value_counts()
    sw = np.array([1.0 / np.sqrt(max(cnt.get(s, 1), 1)) for s in sources])
    sw = sw / (sw * np.array([cnt.get(s, 0) for s in sources])).sum() * len(tr)
    model = Net(cfg["model"]["name"])
    return dict(model=model, train_ds=train_ds, val_ds=val_ds, loss_fn=Loss(cfg, sw).to(ctx.device), eval_fn=make_eval(sources),
                monitor=cfg["train"].get("monitor", "val/mean_f1"), mode="max")

