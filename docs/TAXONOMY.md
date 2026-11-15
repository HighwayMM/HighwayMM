# The HighwayMM hazard taxonomy

Defined in [`code/data/taxonomy.py`](../code/data/taxonomy.py). All sources are mapped to **Level 1**. Hazards and warning levels are **derived** from Level 1 by rules and are never labelled directly.

## Level 1: observable attributes

| Attribute | Values |
|---|---|
| precipitation | none, rain, snow |
| intensity | none, light, moderate, heavy |
| surface | dry, wet, water, snow, ice |
| visibility | clear, haze, fog-moderate, fog-dense |
| illumination | day, twilight, night |
| traffic | free, dense, congested |

## Level 2: hazards (rules over Level 1)

| Base level | Hazards |
|---|---|
| Advisory | snowfall, low visibility, congestion |
| Warning | heavy rain, snow cover |
| Danger | flooding, blizzard (snow ∧ heavy ∧ low visibility), icy road, dense fog |

## Level 3: warning level

`normal / advisory / warning / danger` is the maximum base level over the active hazards, raised at night and for transitional (onset) states.

## Why factorise?

- Flat weather classes ("rainy", "foggy", "snowy") cannot express co-occurring conditions such as night, heavy snow and low visibility at once.
- Datasets labelled with different flat sets cannot be merged.
- With the taxonomy:
  - each source labels only the attributes it observes, and every label records its provenance (human, physics, sensor or VLM) and a weight;
  - a model predicts the six attribute posteriors;
  - the warning level comes from the **exact pushforward** of those posteriors through the rules (a sum over all 2,160 attribute combinations, under 0.1 ms per item).

## Examples of the mapping

| Source label | Level 1 |
|---|---|
| FVEI visibility ≤ 100 m | visibility = fog-dense |
| NYSDOT "severe snow" | precipitation = snow, intensity = heavy, surface = snow |
| FloodCam "water" | surface = water |
| FloodCam "no water" | partial label: surface ≠ water |
| Iowa RWIS pavement "wet" | surface = wet (sensor) |
| METAR `+RA` within 20 km / 20 min | precipitation = rain, intensity = heavy (sensor) |
| Solar elevation at camera and time | illumination (physics) |
