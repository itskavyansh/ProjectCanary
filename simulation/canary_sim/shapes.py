"""Subsidence-shape mismatch stress test (SYNTHETIC / SIMULATION).

Alternative settlement fields w(x,y,t) = -s(t) * h(x,y) in the SAME frozen schema_v1 timeline format, with the same
S-curve time profile, run template, node geometry and mounting offsets as the frozen Gaussian engine
(`engine.py` is NOT modified). Shapes: gaussian_control, trough_panel, asymmetric, two_basin, planar_slope.

The existing dataset code (`dataset.generate_one_run`: sensors, faults, confounders, labels, windows, features) is run
UNCHANGED; `shape_engine()` only swaps the callable `engine.run_scenario` for the duration of a `with` block (monkey
patch in memory, no file touched). Normal runs use the untouched null-field path.

Peak matching: for every shape the amplitude s_max is set so that the maximum surface slope of the final field anywhere in
the plane, max|grad w| = s_max * max|grad h|, equals tan(peak_tilt_deg) - the same definition as the Gaussian engine -
and peak_tilt_deg is the same log-stratified draw for every shape. All ranges are PLACEHOLDERS.
"""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.special import erf

from . import engine, node_geometry as ng
from .schema import COLUMNS, SCHEMA_VERSION, validate_frame

SHAPES = ("gaussian_control", "trough_panel", "asymmetric", "two_basin", "planar_slope")
SHAPE_INDEX = {s: i for i, s in enumerate(SHAPES)}
SQ2 = float(np.sqrt(2.0))
SQ2PI = float(np.sqrt(2.0 / np.pi))
RAD = ng.NETWORK_RADIUS_M


def load_shape_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------------------------------ unit shapes h and grad h
def _rot(x, y, p):
    th = np.deg2rad(p["theta_deg"])
    c, s = np.cos(th), np.sin(th)
    dx, dy = np.asarray(x, float) - p["x0"], np.asarray(y, float) - p["y0"]
    return c * dx + s * dy, -s * dx + c * dy, c, s


def _E(u, a, sig):
    return erf((u + a) / (sig * SQ2)) - erf((u - a) / (sig * SQ2))


def _dE(u, a, sig):
    return SQ2PI / sig * (np.exp(-(u + a) ** 2 / (2 * sig**2)) - np.exp(-(u - a) ** 2 / (2 * sig**2)))


def h_and_grad(shape: str, x, y, p: dict):
    """Unit shape h(x,y) (so w = -s*h) and its analytic world-frame gradient (dh/dx, dh/dy)."""
    if shape == "gaussian_control":
        xp, yp, c, s = _rot(x, y, p)
        g = np.exp(-(xp**2 / (2 * p["sx"] ** 2) + yp**2 / (2 * p["sy"] ** 2)))
        dxp, dyp = -g * xp / p["sx"] ** 2, -g * yp / p["sy"] ** 2
        return g, c * dxp - s * dyp, s * dxp + c * dyp
    if shape == "trough_panel":
        xp, yp, c, s = _rot(x, y, p)
        a, b, sig = p["panel_half_width_m"], p["panel_half_length_m"], p["influence_sigma_m"]
        n = _E(0.0, a, sig) * _E(0.0, b, sig)
        ex, ey = _E(xp, a, sig), _E(yp, b, sig)
        h = ex * ey / n
        dxp, dyp = _dE(xp, a, sig) * ey / n, ex * _dE(yp, b, sig) / n
        return h, c * dxp - s * dyp, s * dxp + c * dyp
    if shape == "asymmetric":
        xp, yp, c, s = _rot(x, y, p)
        sxs = np.where(xp < 0, p["sx_neg"], p["sx_pos"])
        g = np.exp(-(xp**2 / (2 * sxs**2) + yp**2 / (2 * p["sy"] ** 2)))
        dxp, dyp = -g * xp / sxs**2, -g * yp / p["sy"] ** 2
        return g, c * dxp - s * dyp, s * dxp + c * dyp
    if shape == "two_basin":
        x, y = np.asarray(x, float), np.asarray(y, float)
        h = 0.0
        gx = 0.0
        gy = 0.0
        for (cx, cy, sg, m) in ((p["x0"], p["y0"], p["sigma1"], 1.0), (p["x2"], p["y2"], p["sigma2"], p["magnitude_ratio"])):
            dx, dy = x - cx, y - cy
            g = m * np.exp(-(dx**2 + dy**2) / (2 * sg**2))
            h = h + g
            gx = gx - g * dx / sg**2
            gy = gy - g * dy / sg**2
        return h, gx, gy
    if shape == "planar_slope":
        cphi, sphi = np.cos(np.deg2rad(p["direction_deg"])), np.sin(np.deg2rad(p["direction_deg"]))
        dx, dy = np.asarray(x, float) - p["x0"], np.asarray(y, float) - p["y0"]
        h = dx * cphi + dy * sphi
        one = np.ones_like(h)
        return h, cphi * one, sphi * one
    raise ValueError(f"unknown shape {shape}")


# ------------------------------------------------------------------------------------------ peak-slope search
def _search_box(shape: str, p: dict):
    """(centre_x, centre_y, half_extent_m, smallest_length_scale_m) bounding the region of non-negligible gradient."""
    if shape == "gaussian_control":
        return p["x0"], p["y0"], 5 * max(p["sx"], p["sy"]), min(p["sx"], p["sy"])
    if shape == "trough_panel":
        sig = p["influence_sigma_m"]
        return p["x0"], p["y0"], float(np.hypot(p["panel_half_width_m"], p["panel_half_length_m"])) + 5 * sig, sig
    if shape == "asymmetric":
        return p["x0"], p["y0"], 5 * max(p["sx_pos"], p["sx_neg"], p["sy"]), min(p["sx_neg"], p["sy"])
    if shape == "two_basin":
        cx, cy = 0.5 * (p["x0"] + p["x2"]), 0.5 * (p["y0"] + p["y2"])
        d = float(np.hypot(p["x2"] - p["x0"], p["y2"] - p["y0"]))
        return cx, cy, 0.5 * d + 5 * max(p["sigma1"], p["sigma2"]), min(p["sigma1"], p["sigma2"])
    raise ValueError(shape)


def max_gradient(shape: str, p: dict, step_fraction: float = 6.0, refine_halfwidth_steps: float = 2.0) -> float:
    """max over the plane of |grad h| (dimensionless slope per unit s). Analytic for planar / gaussian_control,
    grid + local refinement otherwise."""
    if shape == "planar_slope":
        return 1.0
    if shape == "gaussian_control":
        return float(engine.E_HALF / min(p["sx"], p["sy"]))
    cx, cy, half, ell = _search_box(shape, p)
    step = ell / step_fraction
    n = int(min(max(2 * half / step, 21), 1800)) | 1
    xs, ys = cx + np.linspace(-half, half, n), cy + np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, ys)
    _, gx, gy = h_and_grad(shape, X, Y, p)
    m = np.hypot(gx, gy)
    iy, ix = np.unravel_index(np.argmax(m), m.shape)
    st = 2 * half / (n - 1)
    hw = refine_halfwidth_steps * st
    xr, yr = np.meshgrid(xs[ix] + np.linspace(-hw, hw, 81), ys[iy] + np.linspace(-hw, hw, 81))
    _, gxr, gyr = h_and_grad(shape, xr, yr, p)
    return float(max(m.max(), np.hypot(gxr, gyr).max()))


# ------------------------------------------------------------------------------------------ parameters
def sample_shape_parameters(cfg: dict, shape_cfg: dict, shape: str, seed: int, mode: str, peak_tilt_deg: float) -> dict:
    """Base (Gaussian-engine) parameters for the seed + shape-specific parameters + amplitude s_max. Saved verbatim."""
    if shape not in SHAPES:
        raise ValueError(f"unknown shape {shape}")
    p = engine.sample_parameters(cfg, seed, mode, peak_tilt_deg)     # same centre / theta / timing stream as the control
    p["shape"] = shape
    p["shape_placeholder_ranges"] = True
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2, SHAPE_INDEX[shape]]))
    if shape == "trough_panel":
        c = shape_cfg["trough_panel"]
        p["panel_width_m"] = float(rng.uniform(*c["panel_width_radii"]) * RAD)
        p["panel_length_m"] = float(rng.uniform(*c["panel_length_radii"]) * RAD)
        p["influence_sigma_m"] = float(rng.uniform(*c["influence_sigma_radii"]) * RAD)
        p["panel_half_width_m"], p["panel_half_length_m"] = p["panel_width_m"] / 2, p["panel_length_m"] / 2
    elif shape == "asymmetric":
        c = shape_cfg["asymmetric"]
        p["skew_ratio"] = float(rng.uniform(*c["skew_ratio"]))
        p["sx_neg"] = p["sx"]
        p["sx_pos"] = float(min(p["sx"] * p["skew_ratio"], engine.SIGMA_CAP_RADII * RAD))
    elif shape == "two_basin":
        c = shape_cfg["two_basin"]
        p["sigma1"] = p["sx"]
        p["separation_sigma"] = float(rng.uniform(*c["separation_sigma"]))
        p["magnitude_ratio"] = float(rng.uniform(*c["magnitude_ratio"]))
        p["sigma2"] = float(p["sigma1"] * rng.uniform(*c["sigma2_over_sigma1"]))
        phi = float(rng.uniform(0.0, 360.0))
        p["separation_direction_deg"] = phi
        d = p["separation_sigma"] * p["sigma1"]
        p["x2"] = float(p["x0"] + d * np.cos(np.deg2rad(phi)))
        p["y2"] = float(p["y0"] + d * np.sin(np.deg2rad(phi)))
    elif shape == "planar_slope":
        c = shape_cfg["planar_slope"]
        p["direction_deg"] = float(rng.uniform(*c["direction_deg"]))
    ps = shape_cfg.get("peak_search", {})
    g = max_gradient(shape, p, ps.get("step_fraction", 6.0), ps.get("refine_halfwidth_steps", 2.0))
    p["max_unit_gradient"] = g
    if shape != "gaussian_control":                                   # control keeps the engine's own s_max exactly
        p["s_max"] = float(np.tan(np.deg2rad(p["peak_tilt_deg"])) / g)
    # tilt actually seen by the four nodes at full amplitude (world frame), diagnostic only
    gmag = [float(np.hypot(*[float(v) for v in h_and_grad(shape, x, y, p)[1:]])) for (x, y) in ng.NODE_XY.values()]
    p["max_node_tilt_deg"] = float(np.rad2deg(np.arctan(p["s_max"] * max(gmag))))
    p["mean_node_tilt_deg"] = float(np.rad2deg(np.arctan(p["s_max"] * float(np.mean(gmag)))))
    return p


def node_slopes(shape: str, x, y, s, p):
    """World-frame (dw/dx, dw/dy) for amplitude s: w = -s*h."""
    _, gx, gy = h_and_grad(shape, x, y, p)
    return -s * gx, -s * gy


def run_scenario_shaped(cfg: dict, shape_cfg: dict, shape: str, seed: int, mode: str, peak_tilt_deg: float):
    """Same output contract as engine.run_scenario for a non-null field: (schema_v1 DataFrame, metadata)."""
    p = sample_shape_parameters(cfg, shape_cfg, shape, seed, mode, peak_tilt_deg)
    mounts = ng.sample_mountings(p["seed"], **cfg["mounting"])
    mounts[ng.HUB_ID] = ng.hub_mounting()
    n = int(round(p["duration_s"] * p["sample_rate_hz"]))
    t = np.arange(n) / p["sample_rate_hz"]
    s = engine.s_of_t(t, p["s_max"], p["t_on"], p["tau"])
    cls = engine._classes(t, p)
    sim_id = f"sim_{p['mode']}_{p['seed']}"
    xy = dict(ng.NODE_XY)
    xy[ng.HUB_ID] = ng.HUB_XY
    frames = []
    for nid, (x, y) in xy.items():
        h, _, _ = h_and_grad(shape, x, y, p)
        w = -s * float(h)
        gx, gy = node_slopes(shape, x, y, s, p)
        gxl, gyl = ng.world_to_sensor_slopes(gx, gy, mounts[nid].effective_yaw_rad)
        tx, ty = ng.slopes_to_tilt_deg(gxl, gyl)
        frames.append(pd.DataFrame({
            "simulation_id": sim_id, "time_s": t, "node_id": nid, "true_x": x, "true_y": y, "true_z": w,
            "true_settlement": w, "true_tilt_x_deg": tx + mounts[nid].tilt_x_offset_deg,
            "true_tilt_y_deg": ty + mounts[nid].tilt_y_offset_deg, "active_class": cls, "fault_flags": 0}))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["time_s", "node_id"], kind="stable").reset_index(drop=True)[list(COLUMNS)]
    validate_frame(df)
    meta = {"simulation_id": sim_id, "schema_version": SCHEMA_VERSION, "synthetic": True, "parameters": p,
            "mountings": {k: v.to_dict() for k, v in mounts.items()}, "config": cfg}
    return df, meta


@contextlib.contextmanager
def shape_engine(shape: str, shape_cfg: dict):
    """Within the block, dataset.generate_one_run (which calls engine.run_scenario) produces `shape` basin runs.
    Null-field (normal) runs always take the untouched engine path; gaussian_control delegates to the frozen engine."""
    orig = engine.run_scenario

    def patched(cfg, seed=None, mode=None, peak_tilt_deg=None, null_field=False):
        if null_field:
            return orig(cfg, seed=seed, mode=mode, peak_tilt_deg=peak_tilt_deg, null_field=True)
        if shape == "gaussian_control":
            df, meta = orig(cfg, seed=seed, mode=mode, peak_tilt_deg=peak_tilt_deg, null_field=False)
            P = meta["parameters"]
            P["shape"] = "gaussian_control"
            gmag = [float(np.hypot(*[float(v) for v in h_and_grad(shape, x, y, P)[1:]])) for (x, y) in ng.NODE_XY.values()]
            P["max_unit_gradient"] = float(engine.E_HALF / min(P["sx"], P["sy"]))
            P["max_node_tilt_deg"] = float(np.rad2deg(np.arctan(P["s_max"] * max(gmag))))
            P["mean_node_tilt_deg"] = float(np.rad2deg(np.arctan(P["s_max"] * float(np.mean(gmag)))))
            return df, meta
        return run_scenario_shaped(cfg, shape_cfg, shape, int(seed), mode, float(peak_tilt_deg))

    engine.run_scenario = patched
    try:
        yield
    finally:
        engine.run_scenario = orig


# ------------------------------------------------------------------------------------------ run specs
def build_specs(shape_cfg: dict, scen_cfg: dict, stratified_peaks) -> dict[str, list[dict]]:
    """{shape: [spec,...]} plus key 'normal'. Same seeds and the same stratified peaks for every shape (paired)."""
    n_g, n_r = int(shape_cfg["runs"]["gradual_per_shape"]), int(shape_cfg["runs"]["rapid_per_shape"])
    n_n = int(shape_cfg["runs"]["normal_runs"])
    base, off = int(shape_cfg["seed_base"]), shape_cfg["seed_offsets"]
    lo, hi = scen_cfg["ranges"]["peak_tilt_deg"]
    pk = {"gradual": stratified_peaks(n_g, lo, hi, base + 11), "rapid": stratified_peaks(n_r, lo, hi, base + 12)}
    out: dict[str, list[dict]] = {}
    for shape in shape_cfg["shapes"]:
        specs = []
        for cls, n in (("gradual", n_g), ("rapid", n_r)):
            for i in range(n):
                specs.append({"seed": base + int(off[cls]) + i, "split": f"shape_{shape}", "run_class": cls,
                              "conditions": [cls], "broad_basin": False, "shifted_noise": False,
                              "peak_tilt_deg": pk[cls][i], "shape": shape})
        out[shape] = specs
    out["normal"] = [{"seed": base + int(off["normal"]) + i, "split": "shape_normal", "run_class": "normal",
                      "conditions": ["normal"], "broad_basin": False, "shifted_noise": False, "peak_tilt_deg": None,
                      "shape": "none"} for i in range(n_n)]
    return out


def assert_seeds_disjoint(specs: dict[str, list[dict]], existing_seeds: set[int], margin: int = 40) -> None:
    mine = {int(s["seed"]) for v in specs.values() for s in v}
    bad = [m for m in mine if any((m + d) in existing_seeds for d in range(-margin, margin + 1))]
    if bad:
        raise ValueError(f"{len(bad)} new seeds collide with (or lie within {margin} of) existing seeds, e.g. {bad[:3]}")


def generate_shape_set(shape: str, specs: list[dict], gen_one, shape_cfg: dict, args: tuple):
    """Run dataset.generate_one_run (passed as gen_one) for every spec inside shape_engine(shape)."""
    rows, labs, feats = [], [], []
    with shape_engine(shape, shape_cfg):
        for sp in specs:
            m, l, f = gen_one(sp, *args)
            P = __import__("json").loads(m["parameters_json"])
            m["shape"] = shape if sp["run_class"] != "normal" else "none"
            m["max_node_tilt_deg"] = float(P.get("max_node_tilt_deg", 0.0))
            m["mean_node_tilt_deg"] = float(P.get("mean_node_tilt_deg", 0.0))
            m["peak_tilt_deg_target"] = float(sp["peak_tilt_deg"]) if sp["peak_tilt_deg"] is not None else 0.0
            rows.append(m); labs.append(l); feats.append(f)
    return pd.DataFrame(rows), pd.concat(labs, ignore_index=True), pd.concat(feats, ignore_index=True)
