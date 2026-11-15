"""Protocol v2 grid (FINAL): every method, 3 seeds, checkpoint selection on the human/physics validation score
(train.monitor=val/gt_score). Tags are prefixed p2_ so tables use only this protocol. -> experiments\\grids\\grid_p2.txt"""
from pathlib import Path

V1 = r"code\train\configs\taxowarn.yaml"
S = r"code\train\configs\taxowarn_s.yaml"
FT = r"code\train\configs\finetune_convnext.yaml"
OURS = "[hmm_pilot,itssnap,itslive]"
EXT = "[floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay]"
MON = "train.monitor=val/gt_score"
G = [  # (tag, config, sets) -- heavy jobs first so they overlap with the many light ones
    ("p2_base_ft_convnext", FT, []),
    ("p2_main_taxowarn_s", S, []),
    ("p2_abls_no_dual", S, ["model.use_dual=false"]),
    ("p2_abls_no_queries", S, ["model.use_queries=false"]),
    ("p2_abls_no_spatial", S, ["model.use_spatial=false"]),
    ("p2_abls_no_text", S, ["model.use_text=false"]),
    ("p2_abls_no_hazard_layer", S, ["model.use_hazard_layer=false"]),
    ("p2_abls_flat_heads", S, ["model.hierarchical=false"]),
    ("p2_abls_k1_image", S, ["data.k=1"]),
    ("p2_abls_gt_only", S, ["data.no_pseudo=true"]),
    ("p2_datas_ours_only", S, [f"data.sources_train={OURS}"]),
    ("p2_datas_ext_only", S, [f"data.sources_train={EXT}"]),
    ("p2_main_taxowarn", V1, ["train.num_workers=0"]),
    ("p2_base_linear", V1, ["model.name=linear", "train.num_workers=0"]),
    ("p2_base_mlp", V1, ["model.name=mlp", "train.num_workers=0"]),
    ("p2_base_flat", V1, ["model.name=flat", "train.num_workers=0"]),
    ("p2_abl_unmasked_mlp", V1, ["model.name=mlp", "data.use_raw=true", "train.num_workers=0"]),
]
lines = [f"code\\train\\main.py --config {c} --tag {t} --set {' '.join(s + [MON, f'seed={k}'])}" for t, c, s in G for k in (0, 1, 2)]
p = Path(r"D:\highwaymm\experiments\grids\grid_p2.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "->", p)
