"""TaxoWarn: taxonomy-structured early-warning model on frozen frame embeddings (C3, uses the C4 taxonomy).

x [B,K,D] frozen SigLIP2 embeddings of K overlay-masked frames (K=1 for still images), valid-frame mask [B,K].

1. Temporal Transition Encoder: frame tokens + frame-difference tokens (x_t - x_{t-1}) -> small Transformer with CLS.
2. Taxonomy heads: one head per attribute; classifier weights = (text prototype of the class, projected) + learned residual,
   so every taxonomy node has a language anchor (rare nodes like ice/water/fog_dense get a zero-shot prior).
   Child-of-parent conditioning: intensity head sees precip probabilities.
3. Differentiable hazard layer: product-t-norm soft logic of taxonomy.HAZARD rules on attribute probabilities + learned residual
   -> hazards consistent with the taxonomy by construction.
4. Warning head: ordinal (cumulative-link, levels 0..3) on [CLS, hazard probs]; onset head (transitional) from difference tokens.
Ablation switches: use_temporal, use_delta, use_text, use_hazard_layer, hierarchical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.taxonomy import ATTRS, HAZARDS

A = list(ATTRS)
IDX = {a: {v: i for i, v in enumerate(vs)} for a, vs in ATTRS.items()}


def rule_hazard_probs(p):
    """p: dict attr -> [B,C] probabilities. Returns [B,len(HAZARDS)] soft rule activations (taxonomy.hazards_from_attrs)."""
    g = lambda a, v: p[a][:, IDX[a][v]]  # noqa: E731
    heavy = g("intensity", "heavy")
    snowing = g("precip", "snow")
    lowvis = g("visibility", "fog_moderate") + g("visibility", "fog_dense")
    out = {
        "heavy_rain": g("precip", "rain") * heavy,
        "flooding": g("surface", "water"),
        "snowfall": snowing * (1 - g("intensity", "none")),
        "blizzard": snowing * heavy * lowvis,
        "snow_cover": g("surface", "snow"),
        "icy_road": g("surface", "ice"),
        "low_visibility": g("visibility", "fog_moderate"),
        "dense_fog": g("visibility", "fog_dense"),
        "congestion": g("traffic", "congested"),
    }
    return torch.stack([out[h] for h in HAZARDS], 1)


class TaxoWarn(nn.Module):
    def __init__(self, d_in=1152, d=512, layers=2, heads=8, k_max=8, dropout=0.1, text_protos=None,
                 use_temporal=True, use_delta=True, use_text=True, use_hazard_layer=True, hierarchical=True):
        super().__init__()
        self.cfg = dict(use_temporal=use_temporal, use_delta=use_delta, use_text=use_text and text_protos is not None,
                        use_hazard_layer=use_hazard_layer, hierarchical=hierarchical)
        self.inp = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU(), nn.Dropout(dropout))
        self.delta = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(1, 2 * k_max + 1, d))
        self.type_emb = nn.Parameter(torch.zeros(3, d))  # cls / frame / delta
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        # taxonomy heads
        self.heads = nn.ModuleDict()
        self.txt_proj = nn.ModuleDict()
        for a, vs in ATTRS.items():
            extra = len(ATTRS["precip"]) if (a == "intensity" and hierarchical) else 0
            self.heads[a] = nn.Linear(d + extra, len(vs))
            if self.cfg["use_text"]:
                self.register_buffer(f"txt_{a}", torch.as_tensor(text_protos[a], dtype=torch.float32))  # [C, d_in]
                self.txt_proj[a] = nn.Linear(d_in, d + extra, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.haz_res = nn.Linear(d, len(HAZARDS))
        nn.init.zeros_(self.haz_res.weight); nn.init.constant_(self.haz_res.bias, 0.0)
        self.warn = nn.Sequential(nn.Linear(d + len(HAZARDS), d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.warn_cut = nn.Parameter(torch.tensor([-1.0, 0.5, 2.0]))  # ordinal thresholds, 4 levels
        self.onset = nn.Linear(d, 1)

    def encode(self, x, valid):
        B, K, _ = x.shape
        f = self.inp(x)
        if not self.cfg["use_temporal"]:
            w = valid.float().unsqueeze(-1)
            z = self.norm((f * w).sum(1) / w.sum(1).clamp(min=1))
            return z, z  # no temporal encoder: onset head reads the pooled frame embedding
        toks = [self.cls.expand(B, -1, -1) + self.type_emb[0], f + self.type_emb[1]]
        masks = [torch.ones(B, 1, dtype=torch.bool, device=x.device), valid]
        dtok = None
        if self.cfg["use_delta"] and K > 1:
            dx = x[:, 1:] - x[:, :-1]
            dv = valid[:, 1:] & valid[:, :-1]
            dtok = self.delta(dx) + self.type_emb[2]
            toks.append(dtok)
            masks.append(dv)
        h = torch.cat(toks, 1)
        h = h + self.pos[:, : h.shape[1]]
        m = torch.cat(masks, 1)
        h = self.temporal(h, src_key_padding_mask=~m)
        cls = self.norm(h[:, 0])
        if dtok is not None:
            dh = h[:, 1 + K:]
            dvf = m[:, 1 + K:].float().unsqueeze(-1)
            dpool = (dh * dvf).sum(1) / dvf.sum(1).clamp(min=1)
            dpool = torch.where(dvf.sum(1) > 0, dpool, cls)
        else:
            dpool = cls
        return cls, dpool

    def head_logits(self, a, z):
        lin = self.heads[a](z)
        if self.cfg["use_text"]:
            proto = self.txt_proj[a](getattr(self, f"txt_{a}"))  # [C, d']
            lin = lin + self.logit_scale.exp() * F.normalize(z, dim=-1) @ F.normalize(proto, dim=-1).T
        return lin

    def forward(self, x, valid=None):
        x = x.float()
        if valid is None:
            valid = x.abs().sum(-1) > 0
        cls, dpool = self.encode(x, valid)
        logits, probs = {}, {}
        for a in A:
            z = cls
            if a == "intensity" and self.cfg["hierarchical"]:
                z = torch.cat([cls, probs["precip"].detach() if "precip" in probs else torch.zeros(cls.shape[0], 3, device=cls.device)], -1)
            logits[a] = self.head_logits(a, z)
            probs[a] = logits[a].softmax(-1)
        if self.cfg["use_hazard_layer"]:
            rule = rule_hazard_probs(probs).clamp(1e-4, 1 - 1e-4)
            haz_logit = torch.logit(rule) + self.haz_res(cls)
        else:
            rule = None
            haz_logit = self.haz_res(cls) + self.haz_res.bias.new_zeros(1)  # plain multi-label head
        haz = torch.sigmoid(haz_logit)
        s = self.warn(torch.cat([cls, haz], -1))  # [B,1]
        cuts = torch.cumsum(torch.cat([self.warn_cut[:1], F.softplus(self.warn_cut[1:])]), 0)  # monotone thresholds
        warn_cum = torch.sigmoid(s - cuts)  # P(level > j), j=0..2
        pw = torch.stack([1 - warn_cum[:, 0], warn_cum[:, 0] - warn_cum[:, 1], warn_cum[:, 1] - warn_cum[:, 2], warn_cum[:, 2]], 1).clamp(min=1e-6)
        return {"logits": logits, "hazard_logit": haz_logit, "hazard": haz, "rule": rule, "warn_cum_logit": s - cuts,
                "warn_prob": pw / pw.sum(1, keepdim=True), "onset_logit": self.onset(dpool).squeeze(-1), "emb": cls}

