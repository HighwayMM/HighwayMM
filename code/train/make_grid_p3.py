"""Protocol v2, final configuration (selected on VALIDATION: TaxoWarn-S with K=1 global frame, val score 89.2 vs 88.5 for K=8
and 88.9 for the MLP probe). Ablations re-run on this configuration; baselines get the same K=1 option for fairness.
Tags p3_* (same protocol as p2: monitor val/gt_score, 3 seeds). -> experiments\\grids\\grid_p3.txt"""
from pathlib import Path

V1 = r"code\train\configs\taxowarn.yaml"
S = r"code\train\configs\taxowarn_s.yaml"
OURS = "[hmm_pilot,itssnap,itslive]"
EXT = "[floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay]"
BASE = ["train.monitor=val/gt_score", "data.k=1"]
G = [
    ("p3_abls_no_dual", S, ["model.use_dual=false"]),
    ("p3_abls_no_queries", S, ["model.use_queries=false"]),
    ("p3_abls_no_spatial", S, ["model.use_spatial=false"]),
    ("p3_abls_no_text", S, ["model.use_text=false"]),
    ("p3_abls_no_hazard_layer", S, ["model.use_hazard_layer=false"]),
    ("p3_abls_flat_heads", S, ["model.hierarchical=false"]),
    ("p3_abls_gt_only", S, ["data.no_pseudo=true"]),
    ("p3_datas_ours_only", S, [f"data.sources_train={OURS}"]),
    ("p3_datas_ext_only", S, [f"data.sources_train={EXT}"]),
    ("p3_base_mlp_k1", V1, ["model.name=mlp", "train.num_workers=0"]),
    ("p3_base_linear_k1", V1, ["model.name=linear", "train.num_workers=0"]),
    ("p3_base_flat_k1", V1, ["model.name=flat", "train.num_workers=0"]),
]
lines = [f"code\\train\\main.py --config {c} --tag {t} --set {' '.join(s + BASE + [f'seed={k}'])}" for t, c, s in G for k in (0, 1, 2)]
p = Path(r"D:\highwaymm\experiments\grids\grid_p3.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "->", p)
