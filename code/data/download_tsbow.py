"""Download TSBOW (SKKUAutoLab, AAAI-26) from the gated HF repo, as directed by the authors (utils/README.md:
huggingface_hub, repo SKKUAutoLab/TSBOW, official_release). Needs an HF login whose account was granted access
(run once:  D:\\highwaymm\\analysis\\.venv\\Scripts\\hf.exe auth login).

We take what a weather/season analysis needs: metadata + yaml + split lists + the official-release VIDEOS (train, val;
~7.4 GB). Box annotations / semilabels are not needed (object labels, not road-weather labels). Resumable.
  python download_tsbow.py [--all_annotations]
"""
import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "SKKUAutoLab/TSBOW"
OUT = Path(r"D:\highwaymm\external\tsbow")
SMALL = ["LICENSE", "README.md", "classes.txt", "metadata/TSBOW_info.csv", "v1.0.0_official_release/TSBOW.yaml",
         "v1.0.0_official_release/train.txt", "v1.0.0_official_release/val.txt", "v1.0.0_official_release/test_public.txt"]
VIDEOS = ["v1.0.0_official_release/train/videos.zip", "v1.0.0_official_release/val/videos.zip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all_annotations", action="store_true", help="also the manually labelled frames + boxes (~7.7 GB)")
    a = ap.parse_args()
    files = SMALL + VIDEOS
    if a.all_annotations:
        files += [f"v1.0.0_official_release/{s}/annotations.zip" for s in ("train", "val", "test_public")]
    OUT.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=OUT)
        print("ok", p, flush=True)
        if p.endswith(".zip"):
            dst = Path(p).with_suffix("")
            if not (dst / ".extracted").exists():
                with zipfile.ZipFile(p) as z:
                    z.extractall(dst)
                (dst / ".extracted").write_text("ok")
                print("extracted", dst, flush=True)


if __name__ == "__main__":
    main()
