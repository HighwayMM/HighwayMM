"""Experiment grid v2 for TaxoWarn-S -> experiments\\grids\\grid_s_v1.txt"""
from pathlib import Path

CFG = r"code\train\configs\taxowarn_s.yaml"
OURS = "[hmm_pilot,itssnap,itslive]"
EXT = "[floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay]"
G = {
    "main_taxowarn_s": [],
    "abls_no_dual": ["model.use_dual=false"],
    "abls_no_queries": ["model.use_queries=false"],
    "abls_no_spatial": ["model.use_spatial=false"],
    "abls_no_text": ["model.use_text=false"],
    "abls_no_hazard_layer": ["model.use_hazard_layer=false"],
    "abls_flat_heads": ["model.hierarchical=false"],
    "abls_no_global_residual": ["model.use_global_residual=false"],
    "abls_k1_image": ["data.k=1"],
    "abls_gt_only": ["data.no_pseudo=true"],
    "datas_ours_only": [f"data.sources_train={OURS}"],
    "datas_ext_only": [f"data.sources_train={EXT}"],
}
lines = [f"code\\train\\main.py --config {CFG} --tag {t} --set {' '.join(s + [f'seed={k}'])}" for t, s in G.items() for k in (0, 1, 2)]
p = Path(r"D:\highwaymm\experiments\grids\grid_s_v1.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "->", p)

