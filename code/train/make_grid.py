"""Experiment grid v1 -> experiments\\grids\\grid_v1.txt (one command per line, run with jobrunner.py --per-gpu 2).
Every run logs itself (runlog). Tags name the experiment for the paper tables (code\\report)."""
from pathlib import Path

CFG = r"code\train\configs\taxowarn.yaml"
OURS = "[hmm_pilot,itssnap,itslive]"
EXT = "[floodcam_nc,fvei_fog,norway_trafficcam,vfloodnet_pixabay]"
SEEDS = [0, 1, 2]
G = {
    # main table
    "main_taxowarn": [],
    "base_linear": ["model.name=linear"],
    "base_mlp": ["model.name=mlp"],
    "base_flat": ["model.name=flat"],
    # ablations of TaxoWarn components
    "abl_unmasked": ["data.use_raw=true"],
    "abl_k1_image": ["data.k=1"],
    "abl_no_temporal": ["model.use_temporal=false"],
    "abl_no_delta": ["model.use_delta=false"],
    "abl_no_text": ["model.use_text=false"],
    "abl_no_hazard_layer": ["model.use_hazard_layer=false"],
    "abl_flat_heads": ["model.hierarchical=false"],
    "abl_no_source_balance": ["data.no_source_balance=true"],
    "abl_gt_only": ["data.no_pseudo=true"],
    "abl_conf_pseudo": ["data.min_w=0.5"],
    # data: combined vs single-source training (C1)
    "data_ours_only": [f"data.sources_train={OURS}"],
    "data_ext_only": [f"data.sources_train={EXT}"],
    "base_mlp_ours_only": ["model.name=mlp", f"data.sources_train={OURS}"],
}


def main():
    lines = []
    for tag, sets in G.items():
        for s in SEEDS:
            st = " ".join(sets + [f"seed={s}"])  # no spaces inside values -> no quoting needed (jobrunner uses shlex posix=False)
            lines.append(f"code\\train\\main.py --config {CFG} --tag {tag} --set {st} train.num_workers=0")
    p = Path(r"D:\highwaymm\experiments\grids")
    p.mkdir(parents=True, exist_ok=True)
    (p / "grid_v1.txt").write_text("\n".join(lines), encoding="utf-8")
    print(len(lines), "jobs ->", p / "grid_v1.txt")


if __name__ == "__main__":
    main()
