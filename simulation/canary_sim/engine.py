"""Scenario engine (SYNTHETIC / SIMULATION).

w(x,y,t) = -s(t) * exp(-[x'^2/(2 sx^2) + y'^2/(2 sy^2)])
(x',y') = (x-x0,y-y0) rotated by theta. s(t) = s_max * smoothstep((t-t_on)/tau).
Gradual and rapid share this code path; only the sampled tau differs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import node_geometry as ng
from .schema import COLUMNS, SCHEMA_VERSION, validate_frame

SIGMA_CAP_RADII = 1.5          # hard cap for the main sampling range (network radii)
SIGMA_BROAD_CAP_RADII = 2.5    # absolute cap even for the optional broad-basin subset
E_HALF = float(np.exp(-0.5))  # 0.6065: max of (r/s^2)exp(-r^2/2s^2) * s


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def s_of_t(t, s_max, t_on, tau):
    return s_max * smoothstep((np.asarray(t, float) - t_on) / tau)


def _shape(x, y, p):
    th = np.deg2rad(p["theta_deg"])
    c, s = np.cos(th), np.sin(th)
    dx, dy = np.asarray(x, float) - p["x0"], np.asarray(y, float) - p["y0"]
    xp = c * dx + s * dy
    yp = -s * dx + c * dy
    g = np.exp(-(xp**2 / (2 * p["sx"] ** 2) + yp**2 / (2 * p["sy"] ** 2)))
    return g, xp, yp, c, s


def settlement(x, y, s, p):
    """w at (x,y) for current settlement amplitude s (metres, >=0)."""
    g, *_ = _shape(x, y, p)
    return -s * g


def slopes(x, y, s, p):
    """World-frame (dw/dx, dw/dy), dimensionless (m/m). Analytic."""
    g, xp, yp, c, sn = _shape(x, y, p)
    dq_dx = xp / p["sx"] ** 2 * c - yp / p["sy"] ** 2 * sn
    dq_dy = xp / p["sx"] ** 2 * sn + yp / p["sy"] ** 2 * c
    w = -s * g
    return -w * dq_dx, -w * dq_dy


def sample_parameters(cfg: dict, seed: int | None = None, mode: str | None = None,
                      peak_tilt_deg: float | None = None) -> dict:
    """Sample every scenario parameter from config ranges. Returns a flat dict
    (this dict is saved verbatim in run metadata)."""
    seed = int(cfg["seed"] if seed is None else seed)
    mode = mode or cfg["mode"]
    R = cfg["ranges"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    rad = ng.NETWORK_RADIUS_M
    cx, cy = ng.NETWORK_CENTRE
    span = R["centre_radii"] * rad

    x0 = float(cx + rng.uniform(-span, span))
    y0 = float(cy + rng.uniform(-span, span))
    s_lo, s_hi = R["sigma_radii"]
    if s_hi > SIGMA_CAP_RADII + 1e-12:
        raise ValueError(f"sigma_radii upper bound {s_hi} exceeds the main-range cap {SIGMA_CAP_RADII}; "
                         "use ranges.broad_basin_subset for wider basins")
    broad = bool(R.get("broad_basin_subset", False))
    if broad:
        s_hi = min(float(R.get("sigma_radii_broad_max", SIGMA_BROAD_CAP_RADII)), SIGMA_BROAD_CAP_RADII)
    # cap applies to max(sx, sy): draw aspect first, then sx in [lo, hi / max(aspect, 1)]
    aspect = float(rng.uniform(*R["aspect_sy_over_sx"]))
    sx_hi = s_hi / max(aspect, 1.0)
    if sx_hi < s_lo:
        raise ValueError("sigma_radii lower bound incompatible with the max(sx, sy) cap at this aspect ratio")
    sx = float(rng.uniform(s_lo, sx_hi) * rad)
    sy = sx * aspect
    theta = float(rng.uniform(*R["theta_deg"]))
    lo, hi = R["peak_tilt_deg"]
    peak = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))   # draw always consumed (stream unchanged)
    if peak_tilt_deg is not None:                                 # optional stratification override
        peak = float(peak_tilt_deg)

    T = float(cfg["run"]["duration_s"])
    max_end = 1.0 - cfg["run"]["min_hold_fraction"]
    tau_rng = R["tau_frac"][mode]
    for attempt in range(1, R["max_resample"] + 1):
        t_on_f = float(rng.uniform(*R["t_on_frac"]))
        tau_f = float(rng.uniform(*tau_rng))
        if t_on_f + tau_f <= max_end:
            break
    else:
        raise RuntimeError("could not sample (t_on, tau) leaving a hold window")

    sigma_min = min(sx, sy)
    # max surface slope of Gaussian bowl = s_max * e^-0.5 / sigma_min (anywhere in plane)
    s_max = float(np.tan(np.deg2rad(peak)) * sigma_min / E_HALF)

    phys = float(cfg["time_compression"][mode]["physical_duration_s"])
    return {
        "seed": seed, "mode": mode, "x0": x0, "y0": y0, "sx": sx, "sy": sy,
        "aspect_sy_over_sx": aspect, "theta_deg": theta,
        "peak_tilt_deg": peak, "s_max": s_max,
        "t_on_frac": t_on_f, "tau_frac": tau_f, "t_on": t_on_f * T, "tau": tau_f * T,
        "t_on_tau_resample_attempts": attempt,
        "duration_s": T, "sample_rate_hz": float(cfg["run"]["sample_rate_hz"]),
        "physical_duration_s": phys, "compression_factor": phys / T,
        "placeholder_ranges": True, "broad_basin_subset": broad,
    }


def _classes(t, p):
    cls = np.full(t.shape, "normal", dtype=object)
    ev = (t >= p["t_on"]) & (t < p["t_on"] + p["tau"])
    cls[ev] = p["mode"]
    cls[t >= p["t_on"] + p["tau"]] = "aftermath"
    return cls


def node_tilt_deg(x, y, s, p, mounting: ng.Mounting):
    gx, gy = slopes(x, y, s, p)
    gxl, gyl = ng.world_to_sensor_slopes(gx, gy, mounting.effective_yaw_rad)
    tx, ty = ng.slopes_to_tilt_deg(gxl, gyl)
    return tx + mounting.tilt_x_offset_deg, ty + mounting.tilt_y_offset_deg


def run_scenario(cfg: dict, seed: int | None = None, mode: str | None = None,
                 peak_tilt_deg: float | None = None, null_field: bool = False):
    """Generate one run. Returns (timeline DataFrame, metadata dict)."""
    p = sample_parameters(cfg, seed, mode, peak_tilt_deg)
    mounts = ng.sample_mountings(p["seed"], **cfg["mounting"])
    mounts[ng.HUB_ID] = ng.hub_mounting()
    n = int(round(p["duration_s"] * p["sample_rate_hz"]))
    t = np.arange(n) / p["sample_rate_hz"]
    s = s_of_t(t, p["s_max"], p["t_on"], p["tau"])
    cls = _classes(t, p)
    sim_id = f"sim_{p['mode']}_{p['seed']}"
    if null_field:   # no settlement field at all (normal / vibration / fault runs): zero field, all "normal"
        s = np.zeros_like(s)
        cls = np.full(t.shape, "normal", dtype=object)
        sim_id = f"sim_null_{p['seed']}"
        p.update(null_field=True, s_max=0.0, peak_tilt_deg=0.0)

    frames = []
    xy = dict(ng.NODE_XY)
    xy[ng.HUB_ID] = ng.HUB_XY
    for nid, (x, y) in xy.items():
        w = settlement(x, y, s, p)
        tx, ty = node_tilt_deg(x, y, s, p, mounts[nid])
        frames.append(pd.DataFrame({
            "simulation_id": sim_id, "time_s": t, "node_id": nid,
            "true_x": x, "true_y": y, "true_z": w,  # nominal z0 = 0
            "true_settlement": w, "true_tilt_x_deg": tx, "true_tilt_y_deg": ty,
            "active_class": cls, "fault_flags": 0,
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["time_s", "node_id"], kind="stable").reset_index(drop=True)[list(COLUMNS)]
    validate_frame(df)

    meta = {
        "simulation_id": sim_id, "schema_version": SCHEMA_VERSION,
        "synthetic": True, "parameters": p,
        "mountings": {k: v.to_dict() for k, v in mounts.items()},
        "config": cfg,
    }
    return df, meta


def write_run(df: pd.DataFrame, meta: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir) / meta["simulation_id"]
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "state_timeline.csv", index=False)
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    return out
