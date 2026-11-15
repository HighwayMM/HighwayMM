"""P5: object-detection fusion. Same data/protocol as p4 (labels_v2 for training, monitor val/gt_score, 3 seeds); the
feature models get per-frame COCO Faster R-CNN descriptors (features/det, extract_detections.py) through a separate
detection branch fused with the SigLIP2 embedding (taxowarn.py FeatBaseline d_det). Compare against p4_base_* (no det).
Tags p5_det_*. -> experiments\\grids\\grid_p5.txt"""
from pathlib import Path

V1 = r"code\train\configs\taxowarn.yaml"
LAB = r"data.labels=D:\highwaymm\HighwayMM_release\annotations\labels_v2.csv"
DET = r"data.det_features=D:\highwaymm\HighwayMM_release\features\det"
MON = "train.monitor=val/gt_score"
G = [
    ("p5_det_mlp", ["model.name=mlp"]),
    ("p5_det_linear", ["model.name=linear"]),
    ("p5_det_flat", ["model.name=flat"]),
    ("p5_det_unmasked_mlp", ["model.name=mlp", "data.use_raw=true"]),
]
lines = [f"code\\train\\main.py --config {V1} --tag {t} --set {' '.join(s + [LAB, DET, MON, 'train.num_workers=0', f'seed={k}'])}"
         for t, s in G for k in (0, 1, 2)]
p = Path(r"D:\highwaymm\experiments\grids\grid_p5.txt")
p.write_text("\n".join(lines), encoding="utf-8")
print(len(lines), "->", p)
