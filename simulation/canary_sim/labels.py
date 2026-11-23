"""Label policy (SYNTHETIC / SIMULATION). Labels are derived from the state timeline plus fault/event
metadata and live in their own table; they are never written into the state timeline.

Rule (per window of length L, per class c, f_c = overlap of the window with the union of c's active
intervals / L):
  * f_c >= min_overlap (0.5)  -> class c is ACTIVE in the window.
  * label = highest-priority active class (headline runs have exactly one condition, so no tie).
  * no class active, but some class has 0 < f_c < min_overlap -> "transition"
    (policy "touching"; excluded from training/scoring, kept with offsets for detection delay).
  * otherwise "normal".
Basin classes (gradual / rapid): the active interval runs from the event start to the END of the run,
because the ground stays tilted (aftermath keeps the event class). Faults and abnormal vibration are
active only from start to end; afterwards the label returns to normal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import classes as C

TRANSITION = "transition"


def intervals_from_state(state_df: pd.DataFrame) -> list[dict]:
    """Basin event interval from the state timeline's active_class (event start .. end of run)."""
    d = state_df[state_df.node_id == "NODE_01"].sort_values("time_s")
    act = d.active_class.to_numpy()
    nz = np.nonzero(act != "normal")[0]
    if not len(nz):
        return []
    cls = [a for a in act[nz] if a in ("gradual", "rapid")][0]
    dt = float(np.median(np.diff(d.time_s)))
    return [{"class": cls, "start_s": float(d.time_s.iloc[nz[0]]), "end_s": float(d.time_s.iloc[-1] + dt),
             "source": "state_timeline"}]


def intervals_from_meta(tmeta: dict) -> list[dict]:
    """Fault / abnormal-vibration intervals from telemetry metadata (confounders are NOT events: they stay normal)."""
    out = []
    for rec in list(tmeta.get("faults", [])) + list(tmeta.get("events", [])):
        cls = C.class_of_record(rec["type"])
        if cls is not None:
            out.append({"class": cls, "start_s": rec["start_s"], "end_s": rec["end_s"], "source": rec["type"]})
    return out


def _union_overlap(intervals, t0, t1) -> float:
    segs = sorted((max(a, t0), min(b, t1)) for a, b in intervals if min(b, t1) > max(a, t0))
    tot, cur_end = 0.0, -np.inf
    for a, b in segs:
        a = max(a, cur_end)
        if b > a:
            tot += b - a
            cur_end = b
    return tot


def label_windows(windows: np.ndarray, intervals: list[dict], min_overlap: float = 0.5,
                  transition_policy: str = "touching", priority=C.PRIORITY) -> pd.DataFrame:
    by_cls: dict[str, list] = {}
    for iv in intervals:
        by_cls.setdefault(iv["class"], []).append((iv["start_s"], iv["end_s"]))
    first_start = min((iv["start_s"] for iv in intervals), default=np.nan)
    rows = []
    for i, (t0, t1) in enumerate(windows):
        L = t1 - t0
        frac = {c: _union_overlap(v, t0, t1) / L for c, v in by_cls.items()}
        active = [c for c in priority if frac.get(c, 0.0) >= min_overlap - 1e-12]
        partial = [c for c in priority if 0.0 < frac.get(c, 0.0) < min_overlap - 1e-12]
        if active:
            label = active[0]
        elif partial and transition_policy == "touching":
            label = TRANSITION
        else:
            label = "normal"
        row = {"window_idx": i, "t_start": float(t0), "t_end": float(t1), "label": label,
               "label_set": "|".join(active), "partial_set": "|".join(partial),
               "t_rel_event_start_s": float((t0 + t1) / 2 - first_start) if not np.isnan(first_start) else np.nan}
        for c in priority:
            row[f"overlap_{c}"] = round(frac.get(c, 0.0), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def trainable(labels: pd.DataFrame) -> pd.Series:
    """Mask of windows usable for training and scoring (everything except transition)."""
    return labels.label != TRANSITION
