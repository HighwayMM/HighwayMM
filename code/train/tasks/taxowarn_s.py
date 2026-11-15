"""Task: TaxoWarn-S on global + spatial SigLIP2 features (same labels / loss / metrics as tasks.taxowarn).
cfg.data.spatial = dir with shard*.npz (ids, tok [N,3,36,1152]); cfg.model switches: use_queries, use_spatial, use_text,
use_hazard_layer, hierarchical, use_global_residual."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.taxowarn_s import TaxoWarnS  # noqa: E402
from train.tasks import taxowarn as T  # noqa: E402

_SP = {}


def load_spatial(d):
    if d not in _SP:
        ids, tok = [], []
        for f in sorted(Path(d).glob("shard*.npz")):
            z = np.load(f)
            ids += list(z["ids"]); tok.append(z["tok"])
        _SP[d] = ({k: i for i, k in enumerate(ids)}, np.concatenate(tok))
    return _SP[d]


class SpDS(T.FeatDS):
    def __init__(self, df, feats, k, sources, sp):
        sidx, self.S = sp
        df = df[df.item_id.isin(sidx)]
        super().__init__(df, feats, k, sources)
        self.srow = np.array([sidx[i] for i in self.df.item_id])

    def __getitem__(self, i):
        x, t, j = super().__getitem__(i)
        s = torch.from_numpy(self.S[self.srow[i]]).flatten(0, 1)  # [108, D] fp16 (cast on GPU)
        return torch.cat([x, s], 0), t, j


def _df(cfg):
    import pandas as pd
    df = pd.read_csv(cfg["data"]["labels"], keep_default_na=False)
    for c in ["warn_level", "transitional"] + [f"w_{a}" for a in T.A]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1 if c == "warn_level" else 0)
    return df


def test_dataset(cfg, te, sources):
    d = cfg["data"]
    return SpDS(te, T.load_features(d["features"], d.get("use_raw", False)), d.get("k", 8), sources, load_spatial(d["spatial"]))


def build(cfg, ctx):
    base = T.build(cfg | {"model": {"name": "mlp"}}, ctx)  # reuse label handling / splits / loss / eval
    d = cfg["data"]
    sp = load_spatial(d["spatial"])
    feats = T.load_features(d["features"], d.get("use_raw", False))
    tr, va = base["train_ds"].df, base["val_ds"].df
    sources = base["train_ds"].sources
    train_ds = SpDS(tr, feats, d.get("k", 8), sources, sp)
    val_ds = SpDS(va, feats, d.get("k", 8), sources, sp)
    m = cfg["model"]
    tp = np.load(d["text_protos"]) if m.get("use_text", True) else None
    model = TaxoWarnS(text_protos={a: tp[a] for a in T.A} if tp is not None else None,
                      **{k: m[k] for k in ("d", "enc_layers", "dec_layers", "dropout", "use_queries", "use_spatial", "use_text",
                                           "use_hazard_layer", "hierarchical", "use_global_residual", "use_dual") if k in m})
    if ctx.is_main:
        print(f"[taxowarn_s] train {len(train_ds)} val {len(val_ds)} params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return dict(model=model, train_ds=train_ds, val_ds=val_ds, loss_fn=base["loss_fn"], eval_fn=base["eval_fn"],
                monitor=cfg["train"].get("monitor", "val/mean_f1"), mode="max")



