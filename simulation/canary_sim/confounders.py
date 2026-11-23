"""Benign hard negatives for class 0 (SYNTHETIC / SIMULATION).

Temperature-driven slow tilt drift, wind / passing-vehicle vibration bursts, isolated harmless spikes.
Limits live in configs/confounder_default.yaml and are kept at least `margin_factor` times below the
(placeholder) minimum of the future abnormal-vibration class in amplitude and duration, and strictly
below it in multi-node spread. check_margins() enforces this. Confounders edit telemetry buffers only.
"""
from __future__ import annotations

import numpy as np
import yaml

from . import sensors as S

KINDS = ("temperature_drift", "vibration_burst", "harmless_spikes")


def load_confounder_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    check_margins(cfg)
    return cfg


def check_margins(cfg: dict) -> None:
    m, b, a = cfg["margin_factor"], cfg["benign"], cfg["abnormal_reference"]
    vb = b["vibration_burst"]
    checks = {
        "rms amplitude": vb["rms_add_max"] * m <= a["rms_add_min"],
        "burst duration": vb["duration_s"][1] * m <= a["duration_s_min"],
        "tilt jitter": vb["tilt_jitter_sd_deg"] * m <= a["tilt_jitter_sd_deg_min"] + 1e-9,
        "spike peak": b["harmless_spike"]["peak_max"] * m <= a["peak_min"],
        "node spread (vibration)": vb["max_nodes"] < a["min_nodes"],
        "node spread (spike)": b["harmless_spike"]["max_nodes"] < a["min_nodes"],
    }
    bad = [k for k, ok in checks.items() if not ok]
    if bad:
        raise ValueError(f"benign confounder limits violate the margin to the abnormal class: {bad}")


def _rec(buf, name, node, start, dur, **p):
    buf.confounders.append({"type": f"confounder:{name}", "node": node, "start_s": float(start),
                            "end_s": float(start + dur), "params": p})


def temperature_drift(buf, seed, cfg):
    """Slow, smooth tilt drift on all nodes (thermal offset), amplitude <= max_tilt_deg."""
    amax = cfg["benign"]["temperature_drift"]["max_tilt_deg"]
    T = buf.state.duration_s
    shape = np.sin(np.pi * buf.t / T)                      # warms up and cools down over the run
    for k in range(4):
        r = S.rng_for(seed, 200, k)
        amp = amax * r.uniform(0.3, 1.0, 2) * np.where(r.random(2) < 0.5, -1, 1)
        buf.tilt_x[:, k] += amp[0] * shape
        buf.tilt_y[:, k] += amp[1] * shape
        _rec(buf, "temperature_drift", S.NODES[k], 0.0, T, peak_tilt_deg=amp.tolist())


def vibration_burst(buf, seed, cfg, kind=None, index=0):
    """Wind (1-2 nodes, irregular envelope) or passing vehicle (1 node, smooth pass-by)."""
    vb = cfg["benign"]["vibration_burst"]
    r = S.rng_for(seed, 201, index)
    kind = kind or ("wind", "vehicle")[int(r.integers(0, 2))]
    dur = float(r.uniform(*vb["duration_s"]))
    start = float(r.uniform(0.05, 0.95) * (buf.state.duration_s - dur))
    n_nodes = int(r.integers(1, vb["max_nodes"] + 1)) if kind == "wind" else 1
    nodes = [int(k) for k in r.choice(4, size=n_nodes, replace=False)]
    w = np.arange(int(np.ceil(start / buf.window_s)), min(int(np.ceil((start + dur) / buf.window_s)), buf.W))
    if len(w) == 0:
        return
    u = (buf.t[w] - start) / dur
    env = np.exp(-0.5 * ((u - 0.5) / 0.22) ** 2) if kind == "vehicle" else np.sin(np.pi * u) ** 2 * (0.7 + 0.3 * r.random(len(w)))
    env = env / env.max()
    for j, k in enumerate(nodes):
        scale = 1.0 if j == 0 else 0.6
        a = vb["rms_add_max"] * scale * env * r.uniform(0.5, 1.0)
        S.add_piezo_energy(buf, k, w, a, vb["crest"] * a)
        jit = vb["tilt_jitter_sd_deg"] * scale * env
        rr = S.rng_for(seed, 202, index * 4 + k)
        buf.tilt_x[w, k] += rr.normal(0, 1, len(w)) * jit
        buf.tilt_y[w, k] += rr.normal(0, 1, len(w)) * jit
        _rec(buf, kind, S.NODES[k], start, dur, rms_add_peak=float(a.max()), scale=scale)


def harmless_spikes(buf, seed, cfg):
    """A few isolated spikes on ONE node (piezo knock / tilt blip), below the benign caps."""
    sp = cfg["benign"]["harmless_spike"]
    r = S.rng_for(seed, 203)
    k = int(r.integers(0, 4))
    n = int(r.integers(1, sp["max_count"] + 1))
    w = r.choice(buf.W, size=min(n, buf.W), replace=False)
    peak = r.uniform(0.5, 1.0, len(w)) * sp["peak_max"]
    S.add_piezo_energy(buf, k, w, 0.3 * peak, peak)
    buf.tilt_x[w, k] += r.uniform(-1, 1, len(w)) * sp["tilt_max_deg"]
    for i, wi in enumerate(w):
        _rec(buf, "harmless_spike", S.NODES[k], buf.t[wi], buf.window_s, peak=float(peak[i]))


def scale_config(cfg: dict, scale: float) -> dict:
    """Scale benign amplitudes (never above the configured maxima, so margins still hold for scale <= 1)."""
    import copy
    c = copy.deepcopy(cfg)
    b = c["benign"]
    b["temperature_drift"]["max_tilt_deg"] *= scale
    b["vibration_burst"]["rms_add_max"] *= scale
    b["vibration_burst"]["tilt_jitter_sd_deg"] *= scale
    b["harmless_spike"]["peak_max"] *= scale
    b["harmless_spike"]["tilt_max_deg"] *= scale
    return c


def apply_confounders(buf, seed: int, cfg: dict, kinds: list[str] | None = None,
                      n_bursts: int | None = None) -> None:
    kinds = list(kinds) if kinds is not None else list(KINDS)
    if "temperature_drift" in kinds:
        temperature_drift(buf, seed, cfg)
    if "vibration_burst" in kinds:
        n = int(n_bursts) if n_bursts is not None else int(S.rng_for(seed, 204).integers(1, 3))
        for i in range(n):
            vibration_burst(buf, seed, cfg, index=i)
    if "harmless_spikes" in kinds:
        harmless_spikes(buf, seed, cfg)
