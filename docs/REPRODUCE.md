# Reproducing HighwayMM

Every script has a docstring at the top with its exact usage; this page gives the order. Steps 1–3 need the raw data (released after review); steps 4–7 run from the released labels and features.

## 1. Manifests and frames

| Script | Purpose |
|---|---|
| `code/data/build_manifest.py` | manifest of our ITS/KEC snapshots and pilot clips (camera, time, source) |
| `code/data/build_external.py` | external sources mapped to the taxonomy → `manifest_external.csv` |
| `code/data/iowa_rwis_fetch.py`, `iowa_rwis_sensors.py`, `iowa_asos.py`, `iowa_select.py` | Iowa DOT camera images, pavement sensors, METAR; camera selection |
| `code/data/download_tsbow.py`, `tsbow_frames.py` | TSBOW test-only frames (needs granted access on Hugging Face) |
| `code/data/prep_frames.py` | 8 frames per clip at 640 px, OCR (EasyOCR) overlay masking, alert-banner parsing |

## 2. Labels

| Script | Purpose |
|---|---|
| `code/data/make_vlm_items.py`, `vlm_label.py` | Qwen2.5-VL-7B pseudo-labels from three masked frames and the taxonomy definitions |
| `code/data/build_labels.py` | merges human, physics (solar elevation), sensor and VLM labels with provenance and weights → `labels_v2.csv` (+ TSBOW test rows → `labels_v2t.csv`) |
| `code/data/make_gold_set.py`, `make_gold_strips_raw.py`, `import_gold_csv.py`, `merge_gold.py` | gold-set selection, labelling strips, import and merging of annotators, agreement |
| `code/data/make_hazard_hunt.py`, `audit_hunt.py` | mining of rare-hazard candidates and the audit (`labels_v3.csv`) |

## 3. Features (frozen encoders)

| Script | Output |
|---|---|
| `code/data/extract_features.py` | SigLIP 2 SO400M/16-384 global embeddings of masked (and unmasked) frames |
| `code/data/extract_spatial.py` | 6×6-pooled patch tokens of 3 frames (TaxoWarn-S) |
| `code/data/extract_detections.py` | 20-d per-frame Faster R-CNN descriptors |
| `code/data/text_protos.py` | taxonomy text prototypes (zero-shot, text anchors) |

## 4. Training

```bash
python code/train/main.py --config code/train/configs/taxowarn.yaml --tag my_mlp \
       --set model.name=mlp data.labels=<labels_v2.csv> train.monitor=val/gt_score seed=0
```

- Model choices are `model.name` = `linear | mlp | flat`, plus `taxowarn` (v1). TaxoWarn-S uses `configs/taxowarn_s.yaml`, and ConvNeXt uses `configs/finetune_convnext.yaml`.
- Object-detection fusion: `data.det_features=<features/det>`. Unmasked inputs: `data.use_raw=true`.
- The paper grids are `code/train/make_grid_p4.py` (main and ablations), `make_grid_p5.py` (detection) and `pipeline_p6.ps1` (audit). Run them on several GPUs with:

```bash
python code/train/jobrunner.py --gpus 0,1,2 --per-gpu 2 --jobs experiments/grids/grid_p4.txt
```

Every run writes a directory `experiments/<timestamp>_<task>_<model>_<tag>_s<seed>/`. It contains the command, config, environment, code snapshot, logs, per-epoch validation predictions and checkpoints, and the run is indexed in `experiments/runs_index.csv`.

## 5. Evaluation

```bash
python code/train/evaluate.py --run <run_dir> --labels <labels_v2t.csv>      # calibrated test metrics + predictions
python code/train/eval_gold.py --gold <gold_labels_merged.csv> --out experiments/gold_eval --match "_p[456]_"
```

## 6. Early-warning system

| Script | Purpose |
|---|---|
| `code/warning/system.py` | level decision, CUSUM early advisory, hysteresis, multi-camera aggregation |
| `code/warning/eval_warning.py` | recall at FAR ≤ 10%, false alarms per camera-day, lead time on labelled sequences |
| `code/warning/national_eval.py` | 8.7k-camera national snapshot, agreement with operator banners |
| `code/warning/throughput.py` | end-to-end clips/s on one GPU |

## 7. Paper tables and figures

```bash
python code/report/make_tables.py          # main results + ablations
python code/report/warning_report.py       # warning-system table + trade-off figure
python code/train/make_gold_table.py       # gold-set table
python code/report/make_supp_tables.py     # supplementary tables + run IDs
python code/report/compact_tables.py       # 8-page variants of the tables
python code/report/dataset_figs.py         # camera map, examples, coverage
python code/report/make_overleaf.py        # self-contained Overleaf project
```
