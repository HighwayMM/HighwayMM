"""Smart early road-hazard warning system (C2).

Per camera, each observation (clip or image, timestamp) is turned into
  hazard scores (TaxoWarn hazard layer), calibrated warning-level probabilities (temperature-scaled ordinal head)
  and an onset score (transition head + hazard-trend detector).
Decision logic (per camera stream):
  * level L is issued if P(level >= L) > tau_L (tau chosen on val to bound the false-alarm rate);
  * EARLY advisory: an EWMA/CUSUM detector on the max hazard score (or the onset head) fires "developing <hazard>"
    before the hazard is full-blown (this is what produces lead time);
  * hysteresis: escalate immediately, de-escalate only after `cool` consecutive lower observations;
  * multi-camera aggregation: cameras within `radius_km` that are simultaneously >= 2 raise a regional warning, and an
    isolated single-camera danger is downgraded unless confidence is high (suppresses false alarms).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from data.taxonomy import HAZARDS, LEVEL_NAMES


@dataclass
class WarnConfig:
    tau: tuple = (0.5, 0.5, 0.5)       # thresholds on P(level>=1), P(>=2), P(>=3)
    ewma: float = 0.5                  # smoothing of hazard score
    cusum_k: float = 0.05              # CUSUM drift
    cusum_h: float = 0.25              # CUSUM alarm threshold -> early advisory
    onset_tau: float = 0.6
    cool: int = 2                      # observations below before de-escalation
    radius_km: float = 15.0
    single_cam_danger_conf: float = 0.8


@dataclass
class CameraState:
    level: int = 0
    below: int = 0
    s: float = 0.0          # EWMA hazard score
    cusum: float = 0.0
    history: list = field(default_factory=list)


def cum_from_probs(pw):
    pw = np.asarray(pw)
    return np.array([pw[1:].sum(), pw[2:].sum(), pw[3:].sum()])


class WarningSystem:
    def __init__(self, cfg: WarnConfig = WarnConfig()):
        self.cfg = cfg
        self.cams = {}

    def step(self, cam, ts, p_warn, hazard, onset=0.0):
        c = self.cfg
        st = self.cams.setdefault(cam, CameraState())
        cum = cum_from_probs(p_warn)
        raw = int(sum(cum[j] > c.tau[j] for j in range(3)))
        hz = float(np.max(hazard)) if len(hazard) else 0.0
        prev = st.s
        st.s = c.ewma * hz + (1 - c.ewma) * st.s if st.history else hz
        st.cusum = max(0.0, st.cusum + (st.s - prev) - c.cusum_k)
        early = (st.cusum > c.cusum_h or onset > c.onset_tau) and raw < 2 and hz > 0.15
        target = max(raw, 1 if early else 0)
        if target >= st.level:
            st.level, st.below = target, 0
        else:
            st.below += 1
            if st.below >= c.cool:
                st.level, st.below = target, 0
        top = HAZARDS[int(np.argmax(hazard))] if len(hazard) else ""
        ev = {"camera": cam, "ts": ts, "level": st.level, "raw_level": raw, "early": bool(early), "hazard": top,
              "hazard_score": round(hz, 3), "conf": round(float(cum[max(st.level - 1, 0)]) if st.level else float(1 - cum[0]), 3)}
        st.history.append(ev)
        return ev

    @staticmethod
    def aggregate(events, coords, radius_km=15.0, single_conf=0.8):
        """events: list of per-camera events at the same time slice; coords: cam -> (lat, lon). Returns adjusted events."""
        def dist(a, b):
            (la1, lo1), (la2, lo2) = a, b
            dx = (lo2 - lo1) * 111.32 * math.cos(math.radians((la1 + la2) / 2))
            return math.hypot(dx, (la2 - la1) * 110.57)
        hot = [e for e in events if e["level"] >= 2 and e["camera"] in coords]
        out = []
        for e in events:
            e = dict(e)
            nb = [h for h in hot if h["camera"] != e["camera"] and dist(coords.get(e["camera"], (0, 0)), coords[h["camera"]]) < radius_km] \
                if e["camera"] in coords else []
            e["neighbors_hot"] = len(nb)
            if e["level"] == 3 and not nb and e["conf"] < single_conf:
                e["level"], e["downgraded"] = 2, True
            if e["level"] < 2 and len(nb) >= 2:
                e["level"], e["regional"] = max(e["level"], 1), True
            out.append(e)
        return out


def run_sequences(pred_rows, cfg=WarnConfig()):
    """pred_rows: list of dicts (camera, ts, p_warn, hazard, onset, gt_level, gt_onset_flag) sorted per camera by ts."""
    ws = WarningSystem(cfg)
    out = []
    for r in pred_rows:
        ev = ws.step(r["camera"], r["ts"], r["p_warn"], r["hazard"], r.get("onset", 0.0))
        out.append({**ev, "gt_level": r.get("gt_level", -1)})
    return out


def _episodes(t, on):
    """contiguous runs of True in `on` -> list of (start_time, end_time, start_idx, end_idx)"""
    eps, s = [], None
    for k, v in enumerate(on):
        if v and s is None:
            s = k
        if (not v or k == len(on) - 1) and s is not None:
            e = k if v else k - 1
            eps.append((t[s], t[e], s, e))
            s = None
    return eps


def sequence_metrics(events, onset_level=2, window_min=360):
    """Episode-based early-warning metrics per camera sequence.
    Hazard onset = first observation with GT level >= onset_level (after at least one GT observation below it).
    Lead time (min) for alert threshold L in {1 (any alert), 2 (warning)}: if an alert episode (level >= L) is active at onset,
    lead = onset - episode start, capped at `window_min`; otherwise lead = -(delay until the first alert after onset)
    (negative = late), or NaN if never alerted. False alarm = a warning episode (level >= 2) that lies entirely on
    GT-level-0 observations and is not followed by an onset within `window_min`; reported per camera-day.
    (Implemented as: a warning episode covering any GT-normal observation that has no onset within the next `window_min`,
    so an always-on alarm cannot score as early warning.)"""
    import pandas as pd
    df = pd.DataFrame(events)
    df["t"] = pd.to_datetime(df["ts"].astype(object), errors="coerce")  # callers pass parsed (naive) timestamps
    res, fa, cam_days = [], 0, 0.0
    W = pd.Timedelta(minutes=window_min)
    for cam, g in df.dropna(subset=["t"]).sort_values("t").groupby("camera"):
        t, lv, gl = list(g.t), g.level.to_numpy(), g.gt_level.to_numpy()
        cam_days += max((t[-1] - t[0]).total_seconds() / 86400, 1 / 24)
        onsets = [k for k in range(len(t)) if gl[k] >= onset_level and any(gl[j] >= 0 and gl[j] < onset_level for j in range(k))]
        on_t = [t[k] for k in onsets]
        # an observation is "clear-far" if GT says normal and no onset follows within W: any warning there is a false alarm
        clear_far = [gl[k] == 0 and not any(t[k] < ot <= t[k] + W for ot in on_t) for k in range(len(t))]
        for (s, e, si, ei) in _episodes(t, lv >= 2):
            if any(clear_far[j] for j in range(si, ei + 1)):
                fa += 1
        if not onsets:
            continue
        k0 = onsets[0]
        rec = {"camera": cam, "gt_onset": t[k0]}
        for L, name in ((1, "any"), (2, "warning")):
            act = lv >= L
            if act[k0]:
                s = k0
                while s > 0 and act[s - 1]:
                    s -= 1
                rec[f"lead_min_{name}"] = min((t[k0] - t[s]).total_seconds() / 60, window_min)
            else:
                later = [k for k in range(k0, len(t)) if act[k]]
                rec[f"lead_min_{name}"] = -(t[later[0]] - t[k0]).total_seconds() / 60 if later else np.nan
        rec["detected"] = not np.isnan(rec["lead_min_warning"])
        res.append(rec)
    r = pd.DataFrame(res)
    summary = {"n_events": len(r), "detection_rate": float(r.detected.mean()) if len(r) else np.nan,
               "median_lead_min_any": float(r.lead_min_any.median()) if len(r) else np.nan,
               "median_lead_min_warning": float(r.lead_min_warning.median()) if len(r) else np.nan,
               "false_alarm_episodes": fa, "camera_days": round(cam_days, 2),
               "false_alarms_per_cam_day": fa / cam_days if cam_days else np.nan}
    return summary, r


def level_name(l):
    return LEVEL_NAMES[l]
