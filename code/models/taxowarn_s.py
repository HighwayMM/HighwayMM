"""TaxoWarn-S: taxonomy-query model on spatial vision-language tokens (C3 v2).

Input x [B, 8 + F*36, D]: rows 0..7 = per-frame global SigLIP2 embeddings (zero rows = padding), rows 8.. = 6x6 pooled patch
tokens of F=3 frames. Design (motivated by v1 results: global pooled embeddings miss local road-surface cues, and a temporal
transformer over near-identical frames adds variance):
  1. Spatial encoder: patch tokens + 2-D position + frame embedding + two context tokens (frame-mean global embedding and
     frame-variation token |x_t - mean|, the onset cue) -> 2-layer Transformer.
  2. Taxonomy queries: one learnable query per taxonomy attribute cross-attends to the encoded tokens, so each attribute
     looks at its own evidence (surface -> road region, visibility -> far field, illumination -> whole frame).
     A global residual path (MLP on the frame-mean embedding) is added to every query output, so the model contains the
     strong pooled-feature baseline as a special case.
  3. Text-anchored heads (class-description prototypes), child conditioned on parent (intensity | precip).
  4. Differentiable hazard layer (taxonomy rules) and warning levels by exact taxonomy pushforward (no free-form warning head);
     an onset head reads the variation token.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.taxonomy import ATTRS, HAZARDS
from models.taxowarn import rule_hazard_probs

A = list(ATTRS)


class TaxoWarnS(nn.Module):
    def __init__(self, d_in=1152, d=384, enc_layers=2, dec_layers=2, heads=6, n_frames=3, grid=36, dropout=0.1,
                 text_protos=None, use_queries=True, use_spatial=True, use_text=True, use_hazard_layer=True, hierarchical=True,
                 use_global_residual=True, use_dual=True):
        super().__init__()
        self.nf, self.grid = n_frames, grid
        self.cfg = dict(use_queries=use_queries, use_spatial=use_spatial, use_text=use_text and text_protos is not None,
                        use_hazard_layer=use_hazard_layer, hierarchical=hierarchical, use_global_residual=use_global_residual)
        self.tok = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.ctx = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.var = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, d))
        self.pos = nn.Parameter(torch.zeros(1, 1, grid, d))
        self.fpos = nn.Parameter(torch.zeros(1, n_frames, 1, d))
        self.type_emb = nn.Parameter(torch.zeros(3, d))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.fpos, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, enc_layers)
        self.queries = nn.Parameter(torch.randn(len(A), d) * 0.02)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.dec = nn.TransformerDecoder(dec, dec_layers)
        self.glob = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512, d))
        self.norm = nn.LayerNorm(d)
        self.heads = nn.ModuleDict()
        self.txt_proj = nn.ModuleDict()
        for a, vs in ATTRS.items():
            extra = len(ATTRS["precip"]) if (a == "intensity" and hierarchical) else 0
            self.heads[a] = nn.Linear(d + extra, len(vs))
            if self.cfg["use_text"]:
                self.register_buffer(f"txt_{a}", torch.as_tensor(text_protos[a], dtype=torch.float32))
                self.txt_proj[a] = nn.Linear(d_in, d + extra, bias=False)
        # dual path: global pooled-embedding heads (strong on frame-level attributes such as illumination) and spatial
        # taxonomy-query heads (local evidence: surface, traffic), mixed per attribute by a learned gate in probability space
        self.use_dual = use_dual
        self.gpath = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 512), nn.GELU(), nn.Dropout(dropout))
        self.gheads = nn.ModuleDict({a: nn.Linear(512 + (len(ATTRS["precip"]) if (a == "intensity" and hierarchical) else 0), len(v))
                                     for a, v in ATTRS.items()})
        self.gate = nn.Parameter(torch.zeros(len(A)))
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.haz_res = nn.Linear(d, len(HAZARDS))
        nn.init.zeros_(self.haz_res.weight); nn.init.zeros_(self.haz_res.bias)
        self.onset = nn.Linear(d, 1)

    def forward(self, x, valid=None):
        x = x.float()
        g, s = x[:, :8], x[:, 8:]
        B = x.shape[0]
        gv = (g.abs().sum(-1) > 0).float().unsqueeze(-1)
        gm = (g * gv).sum(1) / gv.sum(1).clamp(min=1)                                  # frame-mean global embedding
        gvar = ((g - gm.unsqueeze(1)).abs() * gv).sum(1) / gv.sum(1).clamp(min=1)       # frame-variation (onset cue)
        ctx = [self.ctx(gm).unsqueeze(1) + self.type_emb[0], self.var(gvar).unsqueeze(1) + self.type_emb[1]]
        if self.cfg["use_spatial"]:
            t = self.tok(s).view(B, self.nf, self.grid, -1) + self.pos + self.fpos + self.type_emb[2]
            mem = torch.cat(ctx + [t.flatten(1, 2)], 1)
        else:
            mem = torch.cat(ctx, 1)
        mem = self.enc(mem)
        gres = self.glob(gm) if self.cfg["use_global_residual"] else 0.0
        if self.cfg["use_queries"]:
            q = self.dec(self.queries.unsqueeze(0).expand(B, -1, -1), mem)             # [B, n_attr, d]
            Z = self.norm(q + (gres.unsqueeze(1) if torch.is_tensor(gres) else 0.0))
        else:
            z = self.norm(mem[:, 0] + gres)
            Z = z.unsqueeze(1).expand(-1, len(A), -1)
        logits, probs = {}, {}
        aux_s, aux_g = {}, {}
        hg = self.gpath(gm) if self.use_dual else None
        for j, a in enumerate(A):
            z = Z[:, j]
            if a == "intensity" and self.cfg["hierarchical"]:
                z = torch.cat([z, probs["precip"].detach()], -1)
            lin = self.heads[a](z)
            if self.cfg["use_text"]:
                proto = self.txt_proj[a](getattr(self, f"txt_{a}"))
                lin = lin + self.logit_scale.exp() * F.normalize(z, dim=-1) @ F.normalize(proto, dim=-1).T
            if self.use_dual:
                zg = torch.cat([hg, probs["precip"].detach()], -1) if (a == "intensity" and self.cfg["hierarchical"]) else hg
                lg = self.gheads[a](zg)
                aux_s[a], aux_g[a] = lin, lg
                gt = torch.sigmoid(self.gate[j])
                p = gt * lin.float().softmax(-1) + (1 - gt) * lg.float().softmax(-1)
                lin = torch.log(p.clamp(min=1e-8))  # log of the mixture; softmax(lin) == p
            logits[a] = lin
            probs[a] = lin.softmax(-1)
        zbar = Z.mean(1)
        if self.cfg["use_hazard_layer"]:
            rule = rule_hazard_probs(probs).clamp(1e-4, 1 - 1e-4)
            haz_logit = torch.logit(rule) + self.haz_res(zbar)
        else:
            rule, haz_logit = None, self.haz_res(zbar)
        return {"logits": logits, "aux_logits": [aux_s, aux_g] if self.use_dual else [],
                "hazard_logit": haz_logit, "hazard": torch.sigmoid(haz_logit), "rule": rule,
                "onset_logit": self.onset(self.norm(mem[:, 1])).squeeze(-1), "emb": zbar}



