"""Protocol v2 on labels_v2 (adds Iowa DOT RWIS summer cameras with sensor labels; TSBOW dropped). Same selection rule as
p2/p3 (monitor val/gt_score, 3 seeds); final config = TaxoWarn-S K=1 (chosen on validation in p2, NOT re-chosen here).
Tags p4_*. Adds the 'no_iowa' data ablation (does the summer gap-filling source help?). -> experiments\\grids\\grid_p4.txt"""
from pathlib import Path

V1 = r"code\train\configs\taxowarn.yaml"
S = r"code\train\configs\taxowarn_s.yaml"
FT = r"code\train\configs\finetune_convnext.yaml"
OURS = "[hmm_pilot,itssnap,itslive]"
EXT = "[floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay,iowa_rwis]"
NO_IOWA = "[hmm_pilot,itssnap,itslive,floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay]"
LAB = r"data.labels=D:\highwaymm\HighwayMM_release\annotations\labels_v2.csv"
MON = "train.monitor=val/gt_score"
K1 = "data.k=1"
G = [  # heavy first
    ("p4_base_ft_convnext", FT, []),
    ("p4_final", S, [K1]),
    ("p4_s_k8", S, []),
    ("p4_abls_no_dual", S, [K1, "model.use_dual=false"]),
    ("p4_abls_no_queries", S, [K1, "model.use_queries=false"]),
    ("p4_abls_no_spatial", S, [K1, "model.use_spatial=false"]),
    ("p4_abls_no_text", S, [K1, "model.use_text=false"]),
    ("p4_abls_no_hazard_layer", S, [K1, "model.use_hazard_layer=false"]),
    ("p4_abls_flat_heads", S, [K1, "model.hierarchical=false"]),
    ("p4_abls_gt_only", S, [K1, "data.no_pseudo=true"]),
    ("p4_datas_ours_only", S, [K1, f"data.sources_train={OURS}"]),
    ("p4_datas_ext_only", S, [K1, f"data.sources_train={EXT}"]),
    ("p4_datas_no_iowa", S, [K1, f"data.sources_train={NO_IOWA}"]),
    ("p4_main_taxowarn", V1, ["train.num_workers=0"]),
    ("p4_base_linear", V1, ["model.name=linear", "train.num_workers=0"]),
    ("p4_base_mlp", V1, ["model.name=mlp", "train.num_workers=0"]),
    ("p4_base_flat", V1, ["model.name=flat", "train.num_workers=0"]),
    ("p4_base_linear_k1", V1, ["model.name=linear", K1, "train.num_workers=0"]),
    ("p4_base_mlp_k1", V1, ["model.name=mlp", K1, "train.num_workers=0"]),
    ("p4_base_flat_k1", V1, ["model.name=flat", K1, "train.num_workers=0"]),
    ("p4_abl_unmasked_mlp", V1, ["model.name=mlp", "data.use_raw=true", "train.num_workers=0"]),
]
lines = [f"code\\train\\main.py --config {c} --tag {t} --set {' '.join(s + [LAB, MON, f'seed={k}'])}" for t, c, s in G for k in (0, 1, 2)]
p = Path(r"D:\highwaymm\experiments\grids\grid_p4.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "->", p)
