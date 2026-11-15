# Replay demo

`code/warning/live_demo.py` runs the full HighwayMM warning pipeline on recorded footage, one observation at a time and in
time order, the same way it would run on a live camera feed:

```
video / image files ─► 8 frames per observation ─► overlay-text masking (EasyOCR, training rules)
  ─► SigLIP 2 SO400M/16-384 (frozen) ─► TaxoWarn-S: 6 attributes + 9 hazards (temperature-calibrated)
  ─► taxonomy pushforward P(level) ─► warning system: thresholds, early advisory (CUSUM), hysteresis
  ─► console feed · events.csv/jsonl · timeline_<camera>.png · demo_<camera>.mp4 (+ .gif) · report.html · run.json
```

The demo never connects to a camera, stream or traffic API. It reads files that you supply. In `--watch` mode it reads
files as an external recorder writes them into a folder.

## Setup

1. Install the requirements: `pip install -r requirements.txt`. You need `torch`, `open_clip_torch`, `av`, `easyocr`, `opencv-python`, `matplotlib` and `pandas`.
2. Get the trained model, `taxowarn_s_highwaymm.pt` (52 MB). It is attached to the GitHub release (and to the dataset release).
   It contains the TaxoWarn-S weights, the model configuration and the calibration temperatures. It holds no optimizer state and no data.
   To build it yourself from a training run:
   `python code/warning/export_model.py --run experiments/<run_id> --out release_models/taxowarn_s_highwaymm.pt`
3. The SigLIP 2 encoder (`hf-hub:timm/ViT-SO400M-16-SigLIP2-384`, about 4.5 GB) downloads on first use.
   To use a local copy, pass `--encoder local-dir:<path>`.
   EasyOCR downloads its Korean and English models on first use.

A GPU is recommended. On one RTX 3090 an observation takes 0.2–0.6 s, most of it OCR; the encoder and model take about 0.1 s.
The demo also runs on a CPU, slowly.

## Usage

```bash
# a folder with one sub-folder per camera (camera = sub-folder name)
python code/warning/live_demo.py --input footage/ --out demo_out/ --checkpoint taxowarn_s_highwaymm.pt --gif

# single files; each file its own camera; long videos are cut into 10 s observations
python code/warning/live_demo.py --input a.mp4 b.mp4 --camera per_file --window 10 --out demo_out/

# paced like a live feed at 20x real time
python code/warning/live_demo.py --input footage/cam1 --camera cam1 --speed 20 --out demo_out/

# watch a drop folder (a file is processed once its size stops changing); Ctrl+C writes the outputs
python code/warning/live_demo.py --watch D:/drop --out demo_out/

# hazard-specific warning, as evaluated in the paper (flooding: tau 0.9; snow cover: tau 0.7)
python code/warning/live_demo.py --input floodcam/ --focus flooding --tau 0.9 --reset_gap 0 --out demo_out/
```

| Option | Meaning |
|---|---|
| `--window S` | Seconds per observation for long videos (default 10, the HighwayMM clip length). A video no longer than 1.5 windows is one observation, sampled exactly as in training. |
| `--layout` | Overlay masking rule. `generic` (default; OCR boxes, as for all external sources), `its` (adds the top band of Korean ITS/KEC layouts) or `anytext` (adds the top band when any text is found; our winter-pilot recordings). Use the rule that matches the footage: the model was trained on these masks. |
| `--show` | What the video and report show: `raw` (default, the recorded frame), `masked` (the model input) or `raw+outline` (the recorded frame with the masked regions outlined). |
| `--gt file.csv` | Optional human labels (`file,gt_level`), drawn as dots on the timelines. |
| `--reset_gap M` | Reset a camera's state after M minutes without observations (default 180). Use `0` to reproduce the paper's sequence evaluation. |
| `--tau`, `--focus` | Level threshold; `--focus` restricts the warning to one hazard. |
| `--interval S` | Spacing of files with no timestamp in their name (default 60 s). Timestamps such as `20261025T164923` or `2022-01-29-16_10_04` are parsed from file names. |

## Reproducing the demo figures

```powershell
python code\warning\make_demo_inputs.py      # NYSDOT, FloodCam (held-out cameras) and our winter clips -> demo\inputs (hard links)
.\code\warning\run_demo.ps1                  # four replays on 3 GPUs -> demo\out
python code\warning\make_demo_assets.py      # supplement figure + README GIFs
```

## Faithfulness check

On 24 test items from six sources, running from the raw files gives the same top class on every attribute as the logged
test predictions. The largest probability difference is 0.006 on the cached masked frames, from fp16 rounding.
With the paper's operating points and `--reset_gap 0`, the replay reproduces the paper's warning numbers on held-out cameras:

| Sequence | Mode | Recall | False-alarm rate (per observation) |
|---|---|---|---|
| FloodCam Buxton + Canal | `--focus flooding --tau 0.9` | 0.468 | 0.349 |
| NYSDOT blizzard (3 cameras) | `--focus snow_cover --tau 0.7` | 0.851 | 0.091 |

## What the replay shows (and its limits)

- **NYSDOT blizzard (all levels).** All 41 observations with human level 3 are issued as *danger*. On 25 observations with human level 0, 20 stay *normal*.
  Hysteresis makes the system de-escalate one or two observations late.
- **FloodCam.** On these beach-road cameras the general warning raises many false alarms: wind-blown sand on the road is
  read as water or snow cover. The flooding-only mode used in the paper is far quieter, but it still misses about half of
  the flooded observations. Flooding on unseen cameras is an open problem (see the paper).
- **Our winter clips.** Snow-covered roads with snowfall and fog reach *danger*. In the night-time clip the precipitation
  alternates between snow and rain from one 10 s window to the next. Snow and rain are hard to tell apart at night on
  infrared cameras (see the hazard-mining audit in the supplement).
- **Not included.** Multi-camera aggregation (`WarningSystem.aggregate`) needs camera coordinates and is not wired into
  the demo. A demo warning is not an operational warning.
