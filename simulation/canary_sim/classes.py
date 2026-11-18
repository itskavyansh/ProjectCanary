"""Six class definitions and run generators (SYNTHETIC / SIMULATION).

Fixed class names: normal, gradual, rapid, abnormal_vibration, sensor_fault, comm_fault.
Class labels live in a separate labels table derived from the state timeline plus fault/event
metadata, never written into the state timeline.

- normal: null settlement field + benign confounders (temperature drift, wind/vehicle bursts,
  harmless spikes) as hard negatives.
- gradual, rapid: Gaussian settlement basin runs.
- abnormal_vibration: null settlement field + sustained vibration and impulse bursts across >=3
  nodes with amplitude and duration above the documented confounder margin.
- sensor_fault: null settlement field + one IMU or hub fault (normal lead-in -> fault -> aftermath).
- comm_fault: null settlement field + one communication fault (normal lead-in -> fault -> aftermath).
- stress: combined conditions (e.g. gradual + sensor_fault, abnormal_vibration + comm_fault).
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from . import confounders as cf, engine, faults as fl, sensors as S

CLASSES = (
    "normal",
    "gradual",
    "rapid",
    "abnormal_vibration",
    "sensor_fault",
    "comm_fault",
)
ClassName = Literal[
    "normal", "gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault"
]

# Priority order when combined conditions occur in stress sets (single-condition headline runs have no ties)
PRIORITY = (
    "rapid",
    "gradual",
    "abnormal_vibration",
    "sensor_fault",
    "comm_fault",
    "normal",
)

SENSOR_FAULT_TYPES = (
    "bias_step",
    "slow_drift",
    "stuck_at",
    "isolated_spikes",
    "intermittent_disconnect",
    "implausible_reading",
    "motor_stall",
    "wrong_dwell_position",
    "stale_readings",
    "hub_self_motion",
)

COMM_FAULT_TYPES = (
    "packet_drops",
    "burst_loss",
    "added_latency",
    "node_unavailable",
    "network_outage",
)


def class_of_record(rec_type: str) -> str | None:
    """Map a fault or event record type to one of the 6 classes (or None for benign confounders)."""
    if rec_type in SENSOR_FAULT_TYPES:
        return "sensor_fault"
    if rec_type in COMM_FAULT_TYPES:
        return "comm_fault"
    if rec_type == "abnormal_vibration":
        return "abnormal_vibration"
    return None


def check_abnormal_vibration_margins(abn_cfg: dict, conf_cfg: dict) -> None:
    """Verify abnormal_vibration parameters strictly exceed confounder reference minimums."""
    ref = conf_cfg["abnormal_reference"]
    cf.check_margins(conf_cfg)
    checks = {
        "rms_add": abn_cfg["rms_add_range"][0] >= ref["rms_add_min"] - 1e-9,
        "duration_s": abn_cfg["duration_s_range"][0] >= ref["duration_s_min"] - 1e-9,
        "min_nodes": abn_cfg["min_nodes"] >= ref["min_nodes"],
        "tilt_jitter_sd_deg": abn_cfg["tilt_jitter_sd_deg_range"][0] >= ref["tilt_jitter_sd_deg_min"] - 1e-9,
        "impulse_peak": abn_cfg["impulse_peak_range"][0] >= ref["peak_min"] - 1e-9,
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"abnormal_vibration config violates confounder margin reference: {bad}")


def inject_abnormal_vibration(
    buf: S.Buffers,
    seed: int,
    abn_cfg: dict,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> dict:
    """Inject sustained multi-node vibration + impulse train above the confounder margin.
    Appends an 'abnormal_vibration' event record to buf.events. Never touches the state timeline.
    """
    r = S.rng_for(seed, 300)
    T = buf.state.duration_s
    if duration_s is None:
        duration_s = float(r.uniform(*abn_cfg["duration_s_range"]))
    if start_s is None:
        lo_f, hi_f = abn_cfg.get("start_frac_range", [0.20, 0.50])
        max_start = max(lo_f * T, T * 0.85 - duration_s)
        start_s = float(r.uniform(lo_f * T, max(lo_f * T + 1.0, max_start)))
    duration_s = float(min(duration_s, T * 0.90 - start_s))

    n_nodes = int(r.integers(abn_cfg["min_nodes"], 5))  # 3 or 4 nodes
    nodes = [int(k) for k in sorted(r.choice(4, size=n_nodes, replace=False))]

    ws = buf.window_s
    w0 = int(np.ceil(start_s / ws - 1e-9))
    w1 = int(np.ceil((start_s + duration_s) / ws - 1e-9))
    w = np.arange(max(w0, 0), min(w1, buf.W))
    if len(w) == 0:
        return {}

    u = (buf.t[w] - start_s) / max(duration_s, 1e-6)
    ramp = np.minimum(np.minimum(u / 0.1, (1.0 - u) / 0.1), 1.0)
    env = np.clip(ramp, 0.2, 1.0) * (0.85 + 0.30 * r.random(len(w)))

    rms_base = float(r.uniform(*abn_cfg["rms_add_range"]))
    jit_base = float(r.uniform(*abn_cfg["tilt_jitter_sd_deg_range"]))
    crest = float(abn_cfg.get("crest", 3.2))
    imp_rate = float(abn_cfg.get("impulse_rate_per_s", 0.35))
    imp_lo, imp_hi = abn_cfg["impulse_peak_range"]

    for j, k in enumerate(nodes):
        rk = S.rng_for(seed, 301, k)
        scale = 1.0 if j == 0 else float(rk.uniform(0.75, 1.0))
        rms_add = rms_base * scale * env
        peak_add = crest * rms_add
        imp_mask = rk.random(len(w)) < (imp_rate * ws)
        if not imp_mask.any():
            imp_mask[int(rk.integers(0, len(w)))] = True
        imp_peaks = rk.uniform(imp_lo, imp_hi, len(w)) * scale
        peak_total = np.where(imp_mask, np.maximum(peak_add, imp_peaks), peak_add)
        rms_total = np.hypot(rms_add, np.where(imp_mask, 0.35 * imp_peaks, 0.0))
        S.add_piezo_energy(buf, k, w, rms_total, peak_total)

        jit = jit_base * scale * env
        buf.tilt_x[w, k] += rk.normal(0.0, 1.0, len(w)) * jit
        buf.tilt_y[w, k] += rk.normal(0.0, 1.0, len(w)) * jit

    rec = {
        "type": "abnormal_vibration",
        "nodes": [S.NODES[k] for k in nodes],
        "start_s": float(start_s),
        "end_s": float(start_s + duration_s),
        "seed": int(seed),
        "params": {
            "rms_add": rms_base,
            "tilt_jitter_sd_deg": jit_base,
            "n_nodes": n_nodes,
        },
    }
    buf.events.append(rec)
    return rec


def sample_single_fault_spec(
    seed: int,
    duration_s: float,
    cls: Literal["sensor_fault", "comm_fault"],
    fault_kind: str | None = None,
    p_two_nodes: float = 0.2,
    start_frac: tuple[float, float] = (0.20, 0.50),
    duration_frac: tuple[float, float] = (0.20, 0.35),
) -> dict:
    """Sample one fault spec with a guaranteed lead-in and aftermath."""
    r = S.rng_for(seed, 310)
    pool = SENSOR_FAULT_TYPES if cls == "sensor_fault" else COMM_FAULT_TYPES
    kind = fault_kind or pool[int(r.integers(0, len(pool)))]
    scope = fl.FAULTS[kind][1]
    start_s = float(r.uniform(*start_frac) * duration_s)
    dur_s = float(min(r.uniform(*duration_frac) * duration_s, duration_s * 0.88 - start_s))
    if scope == "hub":
        target = "HUB"
    elif scope == "network":
        target = "ALL"
    else:
        n = 2 if r.random() < p_two_nodes else 1
        target = [S.NODES[i] for i in sorted(r.choice(4, size=n, replace=False))]
    fseed = int(r.integers(0, 2**31 - 1))
    params = {}
    if kind == "packet_drops":
        params["n"] = max(4, int(round(0.45 * dur_s)))
    elif kind == "isolated_spikes":
        params["n"] = max(4, int(round(0.35 * dur_s)))
    return fl.make_spec(kind, start_s, dur_s, target, fseed, **params)
