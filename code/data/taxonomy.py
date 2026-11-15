"""HighwayMM unified hazard taxonomy (C4) and taxonomy -> hazard -> warning-level rules (C2).

Level 1 attributes (observable, per clip):
  precip       : none | rain | snow
  intensity    : none | light | moderate | heavy                (heavy snow + low visibility => blizzard)
  surface      : dry | wet | water (standing water / flooded) | snow (snow/slush cover) | ice
  visibility   : clear | haze | fog_moderate | fog_dense        (fog_dense ~ < 100 m)
  illumination : day | twilight | night
  traffic      : free | dense | congested
Level 2 hazard types (derived, multi-label):  derived from attributes by HAZARD_RULES
Level 3 warning level (0 normal / 1 advisory / 2 warning / 3 danger): max over hazards, raised by night/transition.
"""

ATTRS = {
    "precip": ["none", "rain", "snow"],
    "intensity": ["none", "light", "moderate", "heavy"],
    "surface": ["dry", "wet", "water", "snow", "ice"],
    "visibility": ["clear", "haze", "fog_moderate", "fog_dense"],
    "illumination": ["day", "twilight", "night"],
    "traffic": ["free", "dense", "congested"],
}
IGNORE = -100  # missing attribute label (source doesn't annotate it)

HAZARDS = ["heavy_rain", "flooding", "snowfall", "blizzard", "snow_cover", "icy_road", "low_visibility", "dense_fog",
           "congestion"]

# Hazard -> base warning level
HAZARD_LEVEL = {"heavy_rain": 2, "flooding": 3, "snowfall": 1, "blizzard": 3, "snow_cover": 2, "icy_road": 3,
                "low_visibility": 1, "dense_fog": 3, "congestion": 1}
LEVEL_NAMES = ["normal", "advisory", "warning", "danger"]


def hazards_from_attrs(a: dict) -> list:
    """a: attribute dict with string values (missing -> None)."""
    g = a.get
    h = []
    if g("precip") == "rain" and g("intensity") == "heavy":
        h.append("heavy_rain")
    if g("surface") == "water":
        h.append("flooding")
    if g("precip") == "snow" and g("intensity") in ("light", "moderate", "heavy"):
        h.append("snowfall")
    if g("precip") == "snow" and g("intensity") == "heavy" and g("visibility") in ("fog_moderate", "fog_dense"):
        h.append("blizzard")
    if g("surface") == "snow":
        h.append("snow_cover")
    if g("surface") == "ice":
        h.append("icy_road")
    if g("visibility") == "fog_moderate":
        h.append("low_visibility")
    if g("visibility") == "fog_dense":
        h.append("dense_fog")
    if g("traffic") == "congested":
        h.append("congestion")
    return h


def warning_level(a: dict, transitional: bool = False) -> int:
    hz = hazards_from_attrs(a)
    lvl = max([HAZARD_LEVEL[x] for x in hz], default=0)
    # wet road + rain moderate is an advisory even without a named hazard
    if lvl == 0 and (a.get("surface") == "wet" or a.get("precip") in ("rain", "snow") or a.get("visibility") == "haze"):
        lvl = 1
    # night amplifies hazards (reduced perception); transitional states (onset) escalate advisory -> warning
    if lvl in (1, 2) and a.get("illumination") == "night" and hz:
        lvl += 1
    if transitional and lvl == 1:
        lvl = 2
    return min(lvl, 3)


# ---- taxonomy pushforward: P(warning level) from attribute posteriors (exact enumeration of all 2,160 combinations) ----
_LEVEL_TABLE = None


def level_table():
    """int array [3,4,5,4,3,3] -> warning level of every attribute combination (transitional=False)."""
    global _LEVEL_TABLE
    if _LEVEL_TABLE is None:
        import itertools
        import numpy as np
        keys = list(ATTRS)
        shape = [len(ATTRS[k]) for k in keys]
        t = np.zeros(shape, dtype=np.int64)
        for idx in itertools.product(*[range(s) for s in shape]):
            t[idx] = warning_level({k: ATTRS[k][i] for k, i in zip(keys, idx)})
        _LEVEL_TABLE = t
    return _LEVEL_TABLE


def warn_probs_from_attrs(P):
    """P: dict attr -> [B, C_a] probabilities (numpy). Returns [B, 4] = P(level) assuming attributes independent given
    the observation (intensity is already conditioned on precip by the model)."""
    import numpy as np
    keys = list(ATTRS)
    T = level_table()
    joint = np.einsum("bi,bj,bk,bl,bm,bn->bijklmn", *[np.asarray(P[k], dtype=np.float64) for k in keys], optimize=True)
    B = joint.shape[0]
    flat, lv = joint.reshape(B, -1), T.reshape(-1)
    out = np.stack([flat[:, lv == L].sum(1) for L in range(4)], 1)
    return out / out.sum(1, keepdims=True)


# ---- KEC/ITS overlay alert banners -> weak hazard labels (overlay is masked from the model input) ----
ALERT_KEYWORDS = {
    "강우": "rain", "빗길": "rain", "호우": "heavy_rain", "폭우": "heavy_rain", "침수": "flooding",
    "강설": "snow", "눈길": "snow", "폭설": "heavy_snow", "대설": "heavy_snow", "결빙": "icy_road", "블랙아이스": "icy_road",
    "안개": "fog", "짙은안개": "dense_fog", "강풍": "strong_wind", "태풍": "typhoon", "정체": "congestion", "지체": "congestion",
    "사고": "accident", "작업": "roadwork", "고장": "breakdown", "낙하물": "debris",
}


def parse_alert(text: str) -> list:
    t = (text or "").replace(" ", "")
    return sorted({v for k, v in ALERT_KEYWORDS.items() if k in t})
