"""Literature-basis sensitivity checks & timescale audit (SYNTHETIC / SIMULATION).

Implements:
- Part B1: High-tilt Gaussian sensitivity arms (3-6 deg and 6-12 deg) via the unchanged dataset pipeline.
- Part B2: Realistic-depth Knothe influence spread sigma = H / (tan(beta) * sqrt(2*pi)) for cells above the
  frozen 1.5-radius main cap, using an in-memory context manager so `engine.py` is never modified.
- Part C: Timescale audit of all gradual runs in the frozen manifest (implied deg/day at 360x vs the
  0.1 deg/day DERIVED, ORDER OF MAGNITUDE reference, plus alternative 30-day compression factors).

Strictly evaluation-only: no model, threshold, feature, or operating point is ever fitted or tuned here.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import engine, node_geometry as ng, validation_checks as VC
from .schema import COLUMNS, SCHEMA_VERSION, validate_frame

RAD = ng.NETWORK_RADIUS_M
SQRT_2PI = math.sqrt(2.0 * math.pi)


def load_literature_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Part B2: Knothe influence spread formula
# -----------------------------------------------------------------------------
def knothe_sigma_m(H_m: float, tan_beta: float) -> float:
    """Knothe influence function exp(-pi * r^2 / R^2) with R = H / tan(beta) is a Gaussian
    exp(-r^2 / (2 * sigma^2)) with sigma = H / (tan(beta) * sqrt(2 * pi))."""
    if H_m <= 0 or tan_beta <= 0:
        raise ValueError(f"H_m and tan_beta must be positive, got H_m={H_m}, tan_beta={tan_beta}")
    return float(H_m / (tan_beta * SQRT_2PI))


def knothe_sigma_radii(H_m: float, tan_beta: float) -> float:
    """Convert Knothe Gaussian sigma (m) to network radii (1 radius = 10 * sqrt(2) = 14.1421356 m)."""
    return float(knothe_sigma_m(H_m, tan_beta) / RAD)


def build_depth_sigma_grid(lit_cfg: dict) -> pd.DataFrame:
    """Compute the full (H, tan_beta) grid and flag cells with sigma_radii > main_sigma_cap_radii (1.5)."""
    gcfg = lit_cfg["realistic_depth_grid"]
    cap = float(gcfg.get("main_sigma_cap_radii", engine.SIGMA_CAP_RADII))
    rows = []
    for H in gcfg["depths_m"]:
        for tb in gcfg["tan_beta_values"]:
            H_f, tb_f = float(H), float(tb)
            R_inf = H_f / tb_f
            sig_m = knothe_sigma_m(H_f, tb_f)
            sig_rad = sig_m / RAD
            tb_tag = str(tb_f).replace(".", "p")
            rows.append({
                "cell_id": f"H{int(round(H_f))}_tanB{tb_tag}",
                "H_m": H_f,
                "tan_beta": tb_f,
                "R_influence_m": round(R_inf, 4),
                "sigma_m": round(sig_m, 4),
                "sigma_radii": round(sig_rad, 4),
                "above_main_cap_1p5": bool(sig_rad > cap + 1e-12),
                "placeholder_grid": True,
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Part B1 & B2: Scenario generation above the frozen 1.5-radius cap without editing engine.py
# -----------------------------------------------------------------------------
def _node_tilt_summary(p: dict) -> tuple[float, float]:
    """Max and mean true surface tilt magnitude (deg, world frame, before mounting offset) across the 4 corners."""
    mags = []
    for (x, y) in ng.NODE_XY.values():
        gx, gy = engine.slopes(x, y, p["s_max"], p)
        mags.append(float(np.rad2deg(np.arctan(np.hypot(gx, gy)))))
    return float(max(mags)), float(np.mean(mags))


def sample_literature_parameters(
    scen_cfg: dict,
    seed: int,
    mode: str,
    peak_tilt_deg: float,
    fixed_sigma_m: float | None = None,
    extra_meta: dict | None = None,
) -> dict:
    """Sample scenario parameters using engine.sample_parameters stream, optionally overriding sigma=(sx, sy)
    for realistic-depth Knothe cells above the 1.5-radius cap."""
    p = engine.sample_parameters(scen_cfg, seed=seed, mode=mode, peak_tilt_deg=peak_tilt_deg)
    if fixed_sigma_m is not None:
        sig = float(fixed_sigma_m)
        p["sx"] = sig
        p["sy"] = sig
        p["aspect_sy_over_sx"] = 1.0
        p["sigma_radii"] = float(sig / RAD)
        p["s_max"] = float(np.tan(np.deg2rad(p["peak_tilt_deg"])) * sig / engine.E_HALF)
    else:
        p["sigma_radii"] = float(max(p["sx"], p["sy"]) / RAD)
    if extra_meta:
        p.update(extra_meta)
    mx_t, mn_t = _node_tilt_summary(p)
    p["max_node_tilt_deg"] = mx_t
    p["mean_node_tilt_deg"] = mn_t
    return p


def run_scenario_literature(
    scen_cfg: dict,
    seed: int,
    mode: str,
    peak_tilt_deg: float,
    fixed_sigma_m: float | None = None,
    extra_meta: dict | None = None,
):
    """Produce a schema_v1 state timeline DataFrame and metadata dict for B1 or B2."""
    p = sample_literature_parameters(
        scen_cfg, seed=seed, mode=mode, peak_tilt_deg=peak_tilt_deg, fixed_sigma_m=fixed_sigma_m, extra_meta=extra_meta
    )
    mounts = ng.sample_mountings(p["seed"], **scen_cfg["mounting"])
    mounts[ng.HUB_ID] = ng.hub_mounting()
    n = int(round(p["duration_s"] * p["sample_rate_hz"]))
    t = np.arange(n) / p["sample_rate_hz"]
    s = engine.s_of_t(t, p["s_max"], p["t_on"], p["tau"])
    cls = engine._classes(t, p)
    sim_id = f"sim_{p['mode']}_{p['seed']}"

    frames = []
    xy = dict(ng.NODE_XY)
    xy[ng.HUB_ID] = ng.HUB_XY
    for nid, (x, y) in xy.items():
        w = engine.settlement(x, y, s, p)
        tx, ty = engine.node_tilt_deg(x, y, s, p, mounts[nid])
        frames.append(pd.DataFrame({
            "simulation_id": sim_id,
            "time_s": t,
            "node_id": nid,
            "true_x": x,
            "true_y": y,
            "true_z": w,
            "true_settlement": w,
            "true_tilt_x_deg": tx,
            "true_tilt_y_deg": ty,
            "active_class": cls,
            "fault_flags": 0,
        }))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["time_s", "node_id"], kind="stable").reset_index(drop=True)[list(COLUMNS)]
    validate_frame(df)
    meta = {
        "simulation_id": sim_id,
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "parameters": p,
        "mountings": {k: v.to_dict() for k, v in mounts.items()},
        "config": scen_cfg,
    }
    return df, meta


@contextlib.contextmanager
def literature_engine(fixed_sigma_m: float | None = None, extra_meta: dict | None = None):
    """In-memory context manager so dataset.generate_one_run runs unchanged for high-tilt and realistic-depth arms."""
    orig = engine.run_scenario

    def patched(cfg, seed=None, mode=None, peak_tilt_deg=None, null_field=False):
        if null_field:
            return orig(cfg, seed=seed, mode=mode, peak_tilt_deg=peak_tilt_deg, null_field=True)
        return run_scenario_literature(
            cfg,
            seed=int(seed),
            mode=mode,
            peak_tilt_deg=float(peak_tilt_deg),
            fixed_sigma_m=fixed_sigma_m,
            extra_meta=extra_meta,
        )

    engine.run_scenario = patched
    try:
        yield
    finally:
        engine.run_scenario = orig


def build_b1_and_b2_specs(lit_cfg: dict, stratified_peaks_fn) -> dict[str, dict[str, Any]]:
    """Build deterministic run specs for B1 (high_tilt_3_6, high_tilt_6_12) and B2 (each cell with sigma > 1.5 radii)."""
    base = int(lit_cfg["seed_base"])
    arms: dict[str, dict[str, Any]] = {}

    # B1: High-tilt arms
    for arm_name, acfg in lit_cfg["high_tilt_arms"].items():
        lo, hi = float(acfg["peak_tilt_deg"][0]), float(acfg["peak_tilt_deg"][1])
        n_g, n_r = int(acfg["gradual_runs"]), int(acfg["rapid_runs"])
        off_g, off_r = int(acfg["seed_offset_gradual"]), int(acfg["seed_offset_rapid"])
        pk_g = stratified_peaks_fn(n_g, lo, hi, base + off_g + 11)
        pk_r = stratified_peaks_fn(n_r, lo, hi, base + off_r + 12)
        specs = []
        for cls, n, off, pk in (("gradual", n_g, off_g, pk_g), ("rapid", n_r, off_r, pk_r)):
            for i in range(n):
                specs.append({
                    "seed": base + off + i,
                    "split": f"lit_{arm_name}",
                    "run_class": cls,
                    "conditions": [cls],
                    "broad_basin": False,
                    "shifted_noise": False,
                    "peak_tilt_deg": pk[i],
                    "arm": arm_name,
                })
        arms[arm_name] = {
            "kind": "B1_high_tilt",
            "fixed_sigma_m": None,
            "extra_meta": {"arm": arm_name, "peak_tilt_range_deg": [lo, hi]},
            "specs": specs,
        }

    # B2: Realistic-depth cells above the 1.5-radius cap
    grid_df = build_depth_sigma_grid(lit_cfg)
    gcfg = lit_cfg["realistic_depth_grid"]
    lo_b2, hi_b2 = float(gcfg["peak_tilt_deg"][0]), float(gcfg["peak_tilt_deg"][1])
    n_g_b2, n_r_b2 = int(gcfg["gradual_runs_per_cell"]), int(gcfg["rapid_runs_per_cell"])
    off_base = int(gcfg["seed_offset_base"])

    above_cells = grid_df[grid_df["above_main_cap_1p5"]].reset_index(drop=True)
    for c_idx, crow in above_cells.iterrows():
        cid = str(crow["cell_id"])
        sig_m = knothe_sigma_m(float(crow["H_m"]), float(crow["tan_beta"]))
        cell_seed_start = base + off_base + int(c_idx) * 100
        pk_g = stratified_peaks_fn(n_g_b2, lo_b2, hi_b2, cell_seed_start + 11)
        pk_r = stratified_peaks_fn(n_r_b2, lo_b2, hi_b2, cell_seed_start + 12)
        specs = []
        for cls, n, sub_off, pk in (("gradual", n_g_b2, 0, pk_g), ("rapid", n_r_b2, 50, pk_r)):
            for i in range(n):
                specs.append({
                    "seed": cell_seed_start + sub_off + i,
                    "split": f"lit_{cid}",
                    "run_class": cls,
                    "conditions": [cls],
                    "broad_basin": False,
                    "shifted_noise": False,
                    "peak_tilt_deg": pk[i],
                    "arm": cid,
                })
        arms[cid] = {
            "kind": "B2_realistic_depth",
            "fixed_sigma_m": sig_m,
            "extra_meta": {
                "arm": cid,
                "H_m": float(crow["H_m"]),
                "tan_beta": float(crow["tan_beta"]),
                "knothe_sigma_m": sig_m,
                "knothe_sigma_radii": float(sig_m / RAD),
                "placeholder_grid": True,
            },
            "specs": specs,
        }

    return arms


def assert_all_seeds_disjoint(arms: dict[str, dict[str, Any]], existing_seeds: set[int], margin: int = 35) -> None:
    """Ensure every new seed across all B1 and B2 arms is disjoint from existing seeds and across arms."""
    seen: set[int] = set()
    for arm_name, ainfo in arms.items():
        for sp in ainfo["specs"]:
            s = int(sp["seed"])
            if s in seen:
                raise ValueError(f"Duplicate seed {s} inside literature_checks arms ({arm_name})")
            seen.add(s)
            if any((s + d) in existing_seeds for d in range(-margin, margin + 1)):
                raise ValueError(f"Seed {s} in {arm_name} collides with existing seed within margin {margin}")


def generate_arm_runs(arm_name: str, ainfo: dict[str, Any], gen_one, gen_args: tuple):
    """Run dataset.generate_one_run inside literature_engine for all specs of an arm."""
    rows, labs, feats = [], [], []
    with literature_engine(fixed_sigma_m=ainfo["fixed_sigma_m"], extra_meta=ainfo["extra_meta"]):
        for sp in ainfo["specs"]:
            m, l, f = gen_one(sp, *gen_args)
            P = json.loads(m["parameters_json"])
            m["arm"] = arm_name
            m["kind"] = ainfo["kind"]
            m["H_m"] = float(P.get("H_m", np.nan))
            m["tan_beta"] = float(P.get("tan_beta", np.nan))
            m["sigma_m"] = float(P["sx"])
            m["sigma_radii"] = float(P.get("sigma_radii", P["sx"] / RAD))
            m["s_max_m"] = float(P["s_max"])
            m["max_node_tilt_deg"] = float(P.get("max_node_tilt_deg", 0.0))
            m["mean_node_tilt_deg"] = float(P.get("mean_node_tilt_deg", 0.0))
            rows.append(m)
            labs.append(l)
            feats.append(f)
    return pd.DataFrame(rows), pd.concat(labs, ignore_index=True), pd.concat(feats, ignore_index=True)


def check_impossible_values_on_arm(arm_name: str, man_df: pd.DataFrame, feat_df: pd.DataFrame) -> dict[str, Any]:
    """Run validation_checks.check_impossible and inspect physical tilt vs 15 deg and 3.5 deg thresholds."""
    rep = VC.Report()
    VC.check_impossible(man_df, feat_df, rep)
    mean_cols = [f"{n}_mean" for n in ng.NODE_IDS]
    per_run_max_tilt = feat_df.groupby("simulation_id")[mean_cols].max().max(axis=1)
    n_runs_gt_15 = int((per_run_max_tilt > VC.TILT_LIMIT_NORMAL_DEG).sum())
    n_win_gt_15 = int((feat_df[mean_cols].max(axis=1) > VC.TILT_LIMIT_NORMAL_DEG).sum())
    n_runs_gt_3p5 = int((per_run_max_tilt >= 3.5).sum())
    n_win_gt_3p5 = int((feat_df[mean_cols].max(axis=1) >= 3.5).sum())
    failed_msgs = rep.failed.get("impossible_values", [])
    return {
        "arm": arm_name,
        "n_runs": len(man_df),
        "n_windows": len(feat_df),
        "max_window_node_mean_deg": round(float(per_run_max_tilt.max()), 4),
        "median_run_max_node_mean_deg": round(float(per_run_max_tilt.median()), 4),
        "runs_exceeding_15deg_normal_limit": n_runs_gt_15,
        "windows_exceeding_15deg_normal_limit": n_win_gt_15,
        "runs_exceeding_3p5deg_threshold_rule_implausible": n_runs_gt_3p5,
        "windows_exceeding_3p5deg_threshold_rule_implausible": n_win_gt_3p5,
        "validation_check_passed": rep.success,
        "validation_check_messages": "; ".join(failed_msgs) if failed_msgs else "NONE",
    }


# -----------------------------------------------------------------------------
# Part C: Timescale audit on all gradual runs in the frozen manifest
# -----------------------------------------------------------------------------
def audit_frozen_gradual_timescales(
    manifest_df: pd.DataFrame,
    ref_rate_deg_per_day: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute implied physical tilt rate (deg/day at 360x compression) for every gradual run in the
    frozen manifest and compare against the DERIVED, ORDER OF MAGNITUDE reference (~0.1 deg/day)."""
    grad = manifest_df[manifest_df["conditions"].astype(str).str.contains("gradual")].copy()
    records = []
    for _, r in grad.iterrows():
        P = json.loads(r["parameters_json"])
        peak_deg = float(r["peak_tilt_deg"])
        peak_mm_per_m = float(np.tan(np.deg2rad(peak_deg)) * 1000.0)
        tau_sim_s = float(P["tau"])
        dur_sim_s = float(r["duration_s"])
        comp = float(r["compression_factor"])
        # At 360x compression, 1 simulated s = 360 physical s, so 240 s = 86,400 s = 1.0 day (24 h)
        tau_phys_s_360x = tau_sim_s * comp
        tau_phys_hours_360x = tau_phys_s_360x / 3600.0
        tau_phys_days_360x = tau_phys_s_360x / 86400.0
        rate_deg_per_day_360x = peak_deg / tau_phys_days_360x
        rate_mm_m_per_day_360x = peak_mm_per_m / tau_phys_days_360x
        factor_vs_ref = rate_deg_per_day_360x / ref_rate_deg_per_day
        # Compression factor that would make this run's average tilt rate equal ref_rate_deg_per_day (0.1 deg/day)
        comp_for_ref_rate = comp * factor_vs_ref
        # At 10,800x (240 s run = 30 days) and 30,000x (240 s run = 83.3 days)
        rate_deg_per_day_10800x = peak_deg / (tau_sim_s * 10800.0 / 86400.0)
        rate_deg_per_day_30000x = peak_deg / (tau_sim_s * 30000.0 / 86400.0)
        records.append({
            "simulation_id": str(r["simulation_id"]),
            "split": str(r["split"]),
            "run_class": str(r["run_class"]),
            "conditions": str(r["conditions"]),
            "broad_basin_subset": bool(r["broad_basin_subset"]),
            "peak_tilt_deg": round(peak_deg, 5),
            "peak_tilt_mm_per_m": round(peak_mm_per_m, 3),
            "tau_sim_s": round(tau_sim_s, 3),
            "tau_frac_of_run": round(tau_sim_s / dur_sim_s, 4),
            "compression_factor_frozen": comp,
            "tau_phys_hours_at_360x": round(tau_phys_hours_360x, 3),
            "tau_phys_days_at_360x": round(tau_phys_days_360x, 4),
            "implied_tilt_rate_deg_per_day_at_360x": round(rate_deg_per_day_360x, 4),
            "implied_tilt_rate_mm_m_per_day_at_360x": round(rate_mm_m_per_day_360x, 3),
            "derived_ref_rate_deg_per_day": ref_rate_deg_per_day,
            "faster_than_0p1_deg_per_day": bool(rate_deg_per_day_360x > ref_rate_deg_per_day),
            "speedup_factor_vs_0p1_deg_per_day": round(factor_vs_ref, 3),
            "implied_tilt_rate_deg_per_day_at_10800x_30d_run": round(rate_deg_per_day_10800x, 5),
            "implied_tilt_rate_deg_per_day_at_30000x": round(rate_deg_per_day_30000x, 5),
            "compression_needed_for_0p1_deg_per_day": round(comp_for_ref_rate, 1),
        })
    per_run_df = pd.DataFrame(records)

    # Summary table overall and by split
    summary_rows = []
    groups = [("ALL_GRADUAL_RUNS", per_run_df)] + [
        (f"split:{sp}", g) for sp, g in per_run_df.groupby("split", sort=False)
    ]
    for label, g in groups:
        rates = g["implied_tilt_rate_deg_per_day_at_360x"].to_numpy(dtype=float)
        factors = g["speedup_factor_vs_0p1_deg_per_day"].to_numpy(dtype=float)
        fast_mask = rates > ref_rate_deg_per_day
        summary_rows.append({
            "subset": label,
            "n_gradual_runs": len(g),
            "n_faster_than_0p1_deg_per_day": int(fast_mask.sum()),
            "frac_faster_than_0p1_deg_per_day": round(float(fast_mask.mean()), 4),
            "rate_deg_per_day_min": round(float(np.min(rates)), 4),
            "rate_deg_per_day_p10": round(float(np.percentile(rates, 10)), 4),
            "rate_deg_per_day_p25": round(float(np.percentile(rates, 25)), 4),
            "rate_deg_per_day_p50_median": round(float(np.percentile(rates, 50)), 4),
            "rate_deg_per_day_mean": round(float(np.mean(rates)), 4),
            "rate_deg_per_day_p75": round(float(np.percentile(rates, 75)), 4),
            "rate_deg_per_day_p90": round(float(np.percentile(rates, 90)), 4),
            "rate_deg_per_day_max": round(float(np.max(rates)), 4),
            "factor_vs_0p1_min": round(float(np.min(factors)), 2),
            "factor_vs_0p1_p50_median": round(float(np.percentile(factors, 50)), 2),
            "factor_vs_0p1_mean": round(float(np.mean(factors)), 2),
            "factor_vs_0p1_p90": round(float(np.percentile(factors, 90)), 2),
            "factor_vs_0p1_max": round(float(np.max(factors)), 2),
        })
    return per_run_df, pd.DataFrame(summary_rows)
