"""SigLIP2 text prototypes for every taxonomy node (used by TaxoWarn's text-anchored heads and the zero-shot baseline).
Output: HighwayMM_release\\features\\text_protos_siglip2.npz  (key per attribute: [C, 1152], prompt-ensembled, L2-normalised)
"""
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.taxonomy import ATTRS  # noqa: E402

DESC = {
    "precip": {"none": "no rain or snow falling", "rain": "rain falling", "snow": "snow falling"},
    "intensity": {"none": "no precipitation", "light": "light drizzle or light snow", "moderate": "moderate rain or snow",
                  "heavy": "heavy downpour or a heavy snowstorm"},
    "surface": {"dry": "a dry road surface", "wet": "a wet road surface with reflections",
                "water": "a flooded road with standing water", "snow": "a road covered with snow or slush",
                "ice": "an icy glazed road with black ice"},
    "visibility": {"clear": "clear visibility far into the distance", "haze": "light haze", "fog_moderate": "fog reducing visibility",
                   "fog_dense": "dense fog, almost nothing visible"},
    "illumination": {"day": "daytime", "twilight": "dusk or dawn twilight", "night": "night time with artificial lights"},
    "traffic": {"free": "light free-flowing traffic", "dense": "dense but moving traffic", "congested": "a traffic jam with queued cars"},
}
TEMPLATES = ["a traffic camera image of a highway with {}.", "a roadside CCTV view showing {}.", "a photo of a road: {}."]


def main():
    m, _, _ = open_clip.create_model_and_transforms("local-dir:D:/highwaymm/models/SigLIP2-SO400M-384")
    tok = open_clip.get_tokenizer("local-dir:D:/highwaymm/models/SigLIP2-SO400M-384")
    m = m.cuda().eval()
    out = {}
    with torch.no_grad():
        for a, vs in ATTRS.items():
            embs = []
            for v in vs:
                t = m.encode_text(tok([tp.format(DESC[a][v]) for tp in TEMPLATES]).cuda())
                t = torch.nn.functional.normalize(t.float(), dim=-1).mean(0)
                embs.append(torch.nn.functional.normalize(t, dim=0).cpu().numpy())
            out[a] = np.stack(embs)
        out["logit_scale"] = np.array(m.logit_scale.exp().item())
        out["logit_bias"] = np.array(getattr(m, "logit_bias", torch.tensor(0.0)).item())
    p = Path(r"D:\highwaymm\HighwayMM_release\features")
    p.mkdir(parents=True, exist_ok=True)
    np.savez(p / "text_protos_siglip2.npz", **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
