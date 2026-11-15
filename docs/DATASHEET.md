# HighwayMM datasheet

This follows Gebru et al., "Datasheets for Datasets". Release v1.0, prepared 2026-11-14.

## Motivation
- **Purpose:** recognising road-weather hazards and issuing early warnings from fixed roadside CCTV under one taxonomy. The taxonomy has 6 observable attributes, which derive 9 hazards, which derive 4 warning levels.
- **Hazards covered:** rain, flooding, fog, snow, blizzard, ice and congestion.
- **Gap it fills:** public traffic-camera weather datasets are local, seasonally narrow, and labelled with incompatible flat classes.

## Composition
- **Ours, Korean ITS/KEC (all labelled; 10 corrupt recordings have no frames):**
  - 8,732 snapshot clips (10 s) from 8,611 expressway and national-road cameras, recorded 25–26 Oct 2026, day and night;
  - 82 live-sequence clips;
  - 226 screen-recorded winter pilot clips from about 225 cameras, Nov 2025–May 2026.
- **External, 7 sources:** see `README.md`. NYSDOT and FloodCam-NC frames are included; the other sources are released as item IDs with labels only.
- **Item:** one camera observation, with up to 8 frames at 640 px width.
- **Labels:** 6 attributes, each with its provenance and weight:
  - precipitation: none, rain, snow;
  - intensity: none, light, moderate, heavy;
  - surface: dry, wet, water, snow, ice;
  - visibility: clear, haze, fog-moderate, fog-dense;
  - illumination: day, twilight, night;
  - traffic: free, dense, congested.
- **Hazards and warning levels** are derived by the rules in `code/data/taxonomy.py`, never labelled directly.
- **Provenance per attribute:**
  - `gt`: human labels of the source dataset, mapped to the taxonomy;
  - `sensor`: Iowa RWIS pavement sensors and METAR within 20 km and 20 min;
  - `physics`: solar elevation at the camera position and time;
  - `vlm`: Qwen2.5-VL-7B pseudo-labels, weighted 0.8, 0.5 or 0.2 by stated confidence.
- **Gold set:** human labels on test-split clips of our own footage, from two annotators (anonymised). Unusable clips are flagged, and hard clips are double-labelled with the agreement reported in the paper.
- **Excluded:** 23 pilot files (non-traffic content, one showing a person; frozen or black clips; unknown source).
- **Splits** (`splits/splits.csv`):
  - our cameras are split camera-disjoint;
  - FloodCam-NC is held out by camera (Buxton, Canal);
  - NYSDOT, the storm timelapses and TSBOW are test-only;
  - the other external sources use their own test sets.

## Collection
- **ITS/KEC:** public CCTV streams of the Korean national ITS centre and Korea Expressway Corporation, sampled once per camera. Collection ended on 4 Nov 2026.
- **Pilot:** screen recordings of public CCTV web viewers.
- **External sources:** obtained from their public releases. See the paper's supplement for the selection criteria.

## Pre-processing
- **Frames:** content crop, then 8 uniformly spaced frames.
- **Overlay masking:**
  - EasyOCR boxes from 2 frames are dilated and masked;
  - on ITS layouts, the top 9% band is also masked.
  - Masking removes the route, kilometre and alert text, which would otherwise act as shortcuts.
  - Released frames are the masked model inputs.
- **Blurring:** persons and plate regions are blurred after masking (see `README.md`, Privacy). Counts are in `blur_log.csv`.

## Uses
- **Intended:** research on weather and hazard recognition, domain shift, calibration and early warning from roadside cameras.
- **Not intended:** identifying or tracking people or vehicles, or operational traffic control without local validation.

## Known limitations
- Our own footage has no summer monsoon clips. Summer rain comes from Iowa DOT RWIS.
- Black ice has no ground truth.
- Labels of our footage outside the gold set are VLM pseudo-labels. On our cameras, models trained on them reach only the VLM's own accuracy.
- Most cameras contribute a single 10 s clip, so long-horizon forecasting is out of scope.

## Distribution and maintenance
- **Licences:** labels, splits and features are CC BY 4.0. External frames keep their source licences.
- **Takedown:** frames reported as containing identifiable people or plates will be removed.
- **Versions:** the label version is v2t (the paper's labels, including Iowa and test-only TSBOW). Earlier label versions are not released.
