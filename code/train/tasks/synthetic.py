"""Synthetic, learnable classification task for launcher sanity checks (no real data touched).

Each sample's image = class-specific pattern + noise, generated deterministically from its index.
"""
import timm
import torch
from torch.utils.data import Dataset


class SynthDS(Dataset):
    def __init__(self, n, num_classes, size, seed):
        self.n, self.c, self.size, self.seed = n, num_classes, size, seed
        g = torch.Generator().manual_seed(1234)
        self.protos = torch.randn(num_classes, 3, 8, 8, generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        y = i % self.c
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        base = torch.nn.functional.interpolate(self.protos[y:y + 1], size=self.size, mode="bilinear")[0]
        x = 0.5 * base + torch.randn(3, self.size, self.size, generator=g)
        return x, y, i


def build(cfg, ctx):
    d = cfg["data"]
    model = timm.create_model(cfg["model"]["name"], pretrained=False, num_classes=d["num_classes"])
    return dict(model=model,
                train_ds=SynthDS(d["n_train"], d["num_classes"], d["size"], seed=1),
                val_ds=SynthDS(d["n_val"], d["num_classes"], d["size"], seed=2),
                monitor="val/acc", mode="max")
