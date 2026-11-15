# HighwayMM dataset

A benchmark for road-weather hazard recognition and early warning from fixed roadside CCTV.
Code, models and the project page: https://highwaymm.github.io/HighwayMM/

## Contents

| Path | What |
|---|---|
| `frames/<item_id>/m{k}.jpg` | Overlay-masked frames (the model input) of the redistributable sources: up to 8 frames per clip, 640 px wide. Persons and licence-plate regions of vehicles are blurred. |
| `annotations/labels.csv` | Taxonomy labels for **every** item, including ID-only sources. Each of the 6 attributes has a value (`lab_<attr>`), a provenance (`src_<attr>`: gt, physics, sensor or vlm) and a training weight (`w_<attr>`). Also included: the derived warning level and the `frames_released` flag. |
| `annotations/vlm_qwen25vl7b/` | Raw Qwen2.5-VL-7B pseudo-label outputs. |
| `annotations/ocr/` | EasyOCR boxes of the burned-in overlay text, from which the masks are built. |
| `splits/splits.csv` | Train, val and test split per item. Ours: camera-disjoint. External: source test sets or held-out cameras. |
| `gold/` | Human gold labels of our own test clips (two anonymised annotators) and the hazard-mining audit. |
| `features/` | SigLIP 2 SO400M/16-384 global (masked and raw) and 6×6 spatial tokens, and COCO detection descriptors, for released-frame items. |
| `manifest_ours.csv`, `manifest_external.csv` | Per-item metadata: camera, time, location and road class for ours; source, licence and original file for external items. |
| `blur_log.csv` | Number of blurred persons and vehicles per frame. |
| `SHA256SUMS.txt` | Checksums of all non-frame files. |

## Sources

| Source | Items | Frames here | Licence |
|---|---|---|---|
| Ours: ITS/KEC snapshots, ITS/KEC sequences, winter pilot | 9,040 labelled (9,030 with frames) | yes | see below |
| NYSDOT 511NY blizzard | 121 | yes | CC BY 4.0 |
| FloodCam-NC | 554 | yes | MIT repository; NCDOT public camera imagery |
| FVEI highway fog | 4,502 | IDs only | unstated (research use) |
| V-FloodNet storm timelapses | 859 | IDs only | unstated (research use) |
| Web flood photos (Pixabay) | 149 | IDs only | Pixabay licence (no standalone redistribution) |
| Norway road cameras | 6,782 | IDs only | unstated |
| Iowa DOT RWIS | 4,932 | IDs only | Iowa DOT imagery via the IEM archive |
| TSBOW (test only) | 1,980 | IDs only | CC BY-NC-ND 4.0 |

For ID-only sources, download the originals from their providers and run `code/data/build_external.py` and the frame and feature scripts in the code repository. The labels in `labels.csv` then line up by `item_id`.

## Privacy

- Our footage comes from public traffic-camera streams of public roads.
- No personal data was sought.
- 23 pilot files are excluded: non-traffic content (one shows a person), frozen or black clips, and files of unknown source.
- In every released frame, persons are blurred in full and the lower part of every vehicle wide enough to carry a readable plate (40 px or more) is blurred. A COCO Faster R-CNN detector finds both, at score 0.3.
- If you find an identifiable person or plate, please report it and we will remove the frame.

## Licence

Labels, annotations, splits and features: CC BY 4.0. Frames of our footage: for research use, with attribution to the source road agencies. External frames keep their source licences (table above).

## Citation

See the project page.
