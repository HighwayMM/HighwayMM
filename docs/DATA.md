# Data sources

| Source | What | Size used | Labels mapped to the taxonomy | Role | Release |
|---|---|---|---|---|---|
| **HighwayMM-ITS (ours)** | one 10 s clip per camera from Korea's national ITS centre and Korea Expressway Corporation streams, 25–26 Oct 2026, day and night | 8,728 clips, 8,611 cameras (+77 live-sequence clips, 50 cameras) | illumination (physics), VLM pseudo-labels, operator alert banners (weak tags, never input) | train / val / test (camera-disjoint) | masked frames, labels, features |
| **HighwayMM-pilot (ours)** | screen-recorded clips, mostly winter | 226 clips, ~225 cameras | as above | train / val / test | as above |
| **Gold set (ours)** | 400 test clips labelled by hand on all six attributes | 291 labelled so far | human | test only | labels |
| FVEI | expressway fog images with measured visibility | 3,602 train + 900 test | visibility grade (human) | train / test | IDs + script |
| FloodCam-NC | 4 coastal-highway cameras, Sept 2020 storm | 554 | surface water / no water (human) | 2 cameras held out | redistributable |
| NYSDOT 511NY | highway cameras through the Jan 2022 blizzard, six annotators | 121 | snow, intensity, surface, visibility (human) | test only (unseen) | CC BY 4.0 |
| V-FloodNet | Hurricane Harvey and Boston Harbor timelapses; flooded-street photos | 859 + 149 | VLM / surface water | warning timelines; auxiliary | IDs + script |
| Norway road cameras | public road webcams | 6,103 + 679 | traffic level (human) | train / test | IDs + script |
| Iowa DOT RWIS | roadside cameras next to pavement sensors, summers 2023–2025 | held-out cameras | surface (sensor), precipitation, intensity, visibility (METAR) | train / val / test | IDs + script |
| TSBOW | Korean city CCTV (Suwon), 198 videos | 10 frames per video | per-video weather tag (human) | **test only** | IDs + script (CC BY-NC-ND) |

**Gaps (documented, not hidden):**
- no summer monsoon footage of our own;
- no ground truth for black ice;
- typhoons only through Hurricane Harvey;
- one clip per camera, so no long-horizon forecasting.

**Privacy:**
- public infrastructure views only, and no personal data was sought;
- licence plates and faces are blurred before release;
- non-traffic files are removed.
