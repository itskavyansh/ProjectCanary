"""Installation-yaw sensitivity analysis (SYNTHETIC / SIMULATION).

Evaluates how much the frozen v3 models and frozen v3 operating points degrade when the nominal
installation yaw (`yaw_deg`) used ONLY by the world-frame feature rotation has field installation
errors:
  - `sd_0deg`: zero error (reproduces v3 features and metrics 100% identically)
  - `sd_5deg`: per-node zero-mean Gaussian error with sd = 5 deg
  - `sd_10deg`: per-node zero-mean Gaussian error with sd = 10 deg
  - `sd_20deg`: per-node zero-mean Gaussian error with sd = 20 deg
  - `single_node_45deg`: one randomly chosen corner node per run has a fixed +/- 45 deg error,
    other 3 nodes have 0 deg error.

Strict invariants:
  - Sensor physics, true state timeline, telemetry, and window labels are 100% untouched.
  - Only the yaw-dependent observable features (`{NODE}_tilt_angle_rad`, `dir_pairwise_cos_mean`,
    `dir_pairwise_cos_min`, `dir_circular_spread`, `dir_win_convergence_*`, `dir_convergence_*`,
    `coherence_trailing_wx`, `coherence_trailing_wy`, `coherence_trailing_signed_mean`) are
    recomputed under the perturbed nominal yaw. All other 65 model features (`{NODE}_mean`,
    `piezo_*`, `hub_*`, `comm_*`, `long_slope_*`, `coherence_trailing_mag`, etc.) are yaw-invariant
    and remain bitwise identical.
  - No model or operating point is ever refit or retuned.
"""
from __future__ import annotations

import copy
import numpy as np
import pandas as pd

from . import (
    classes as C,
    confounders as cf,
    dataset as ds,
    engine,
    features as F,
    node_geometry as ng,
    sensors as S,
    windows as W,
)

NODES = ng.NODE_IDS

YAW_CONDITIONS: tuple[dict, ...] = (
    {"name": "sd_0deg", "kind": "gaussian", "sd_deg": 0.0, "label": "0 deg (nominal v3)"},
    {"name": "sd_5deg", "kind": "gaussian", "sd_deg": 5.0, "label": "Gaussian sd = 5 deg"},
    {"name": "sd_10deg", "kind": "gaussian", "sd_deg": 10.0, "label": "Gaussian sd = 10 deg"},
    {"name": "sd_20deg", "kind": "gaussian", "sd_deg": 20.0, "label": "Gaussian sd = 20 deg"},
    {"name": "single_node_45deg", "kind": "single_45", "sd_deg": 45.0, "label": "1 node fixed 45 deg"},
)

YAW_DEPENDENT_MODEL_COLS: tuple[str, ...] = (
    *[f"{n}_tilt_angle_rad" for n in NODES],
    "dir_pairwise_cos_mean",
    "dir_pairwise_cos_min",
    "dir_circular_spread",
    "dir_win_convergence_align_cos",
    "dir_win_convergence_weighted_cos",
    "dir_win_convergence_residual_m",
    "dir_convergence_align_cos",
    "dir_convergence_weighted_cos",
    "dir_convergence_residual_m",
    "coherence_trailing_wx",
    "coherence_trailing_wy",
    "coherence_trailing_signed_mean",
)


def sample_nominal_yaw_perturbation(
    run_seed: int,
    condition: str | dict,
    master_seed: int = 20261117,
) -> dict[str, float]:
    """Deterministically draw per-node nominal installation-yaw errors (deg) for a run and condition."""
    cond = (
        next(c for c in YAW_CONDITIONS if c["name"] == condition)
        if isinstance(condition, str)
        else condition
    )
    cname = cond["name"]
    if cname == "sd_0deg" or (cond["kind"] == "gaussian" and float(cond["sd_deg"]) == 0.0):
        return {n: 0.0 for n in NODES}

    # Condition-specific key so each condition has an independent, reproducible stream per run
    cond_idx = [c["name"] for c in YAW_CONDITIONS].index(cname)
    rng = S.rng_for(master_seed, 900 + cond_idx, int(run_seed))

    if cond["kind"] == "gaussian":
        sd = float(cond["sd_deg"])
        errs = rng.normal(0.0, sd, size=4)
        return {n: float(errs[k]) for k, n in enumerate(NODES)}
    elif cond["kind"] == "single_45":
        target_k = int(rng.integers(0, 4))
        sign = 1.0 if rng.random() < 0.5 else -1.0
        out = {n: 0.0 for n in NODES}
        out[NODES[target_k]] = float(sign * float(cond.get("sd_deg", 45.0)))
        return out
    else:
        raise ValueError(f"Unknown yaw perturbation kind: {cond}")


def recompute_yaw_dependent_features_for_run(
    base_run_features: pd.DataFrame,
    vx_win: np.ndarray,
    vy_win: np.ndarray,
    base_mountings: dict[str, dict],
    yaw_errors_deg: dict[str, float],
    long_horizon_windows: int = 12,
) -> pd.DataFrame:
    """Recompute ONLY the world-frame rotation and yaw-dependent features for one run (`W` windows)
    given its unperturbed sensor-frame window tilt changes `(vx_win, vy_win)` and the perturbed
    nominal installation yaws `yaw_deg + yaw_errors_deg`.

    All non-direction features (`{NODE}_mean`, `piezo_*`, `hub_*`, `comm_*`, `long_slope_*`,
    `coherence_trailing_mag`, `diag_*`, labels, etc.) are untouched.
    """
    out = base_run_features.copy()
    W_len = len(out)
    node_xy_arr = np.array([ng.NODE_XY[n] for n in NODES], dtype=float)

    # Perturbed nominal installation yaw ONLY (never touches true sensor physics or yaw_error_deg)
    nom_yaw_rad = np.array([
        float(np.deg2rad(base_mountings[n].get("yaw_deg", 0.0) + yaw_errors_deg.get(n, 0.0)))
        for n in NODES
    ], dtype=float)
    c_nom = np.cos(nom_yaw_rad)
    s_nom = np.sin(nom_yaw_rad)

    # Rotate local downslope vector (vy, -vx) into world frame for all W windows
    win_dip_wx = vx_win * s_nom[None, :] + vy_win * c_nom[None, :]
    win_dip_wy = -vx_win * c_nom[None, :] + vy_win * s_nom[None, :]

    for k, n in enumerate(NODES):
        out[f"{n}_tilt_angle_rad"] = np.arctan2(win_dip_wy[:, k], win_dip_wx[:, k])

    cos_mean = np.zeros(W_len, dtype=float)
    cos_min = np.zeros(W_len, dtype=float)
    circ_spread = np.zeros(W_len, dtype=float)
    w_align = np.zeros(W_len, dtype=float)
    w_w_align = np.zeros(W_len, dtype=float)
    w_res_m = np.zeros(W_len, dtype=float)

    for i in range(W_len):
        wx_i = win_dip_wx[i]
        wy_i = win_dip_wy[i]
        norms = np.hypot(wx_i, wy_i)
        valid = norms > 1e-9
        if valid.sum() >= 2:
            ux = wx_i[valid] / norms[valid]
            uy = wy_i[valid] / norms[valid]
            angs = np.arctan2(wy_i[valid], wx_i[valid])
            c_list = []
            m_v = len(ux)
            for a_i in range(m_v):
                for b_j in range(a_i + 1, m_v):
                    c_list.append(float(ux[a_i] * ux[b_j] + uy[a_i] * uy[b_j]))
            cos_mean[i] = float(np.mean(c_list))
            cos_min[i] = float(np.min(c_list))
            r_bar = float(np.hypot(np.mean(np.cos(angs)), np.mean(np.sin(angs))))
            circ_spread[i] = float(1.0 - r_bar)
        else:
            cos_mean[i] = 0.0
            cos_min[i] = 0.0
            circ_spread[i] = 1.0

        am, aw, fr = F._fit_radial_convergence(node_xy_arr, wx_i, wy_i)
        w_align[i] = am
        w_w_align[i] = aw
        w_res_m[i] = fr

    out["dir_pairwise_cos_mean"] = cos_mean
    out["dir_pairwise_cos_min"] = cos_min
    out["dir_circular_spread"] = circ_spread
    out["dir_win_convergence_align_cos"] = w_align
    out["dir_win_convergence_weighted_cos"] = w_w_align
    out["dir_win_convergence_residual_m"] = w_res_m

    N = max(2, int(long_horizon_windows))
    trail_align = np.zeros(W_len, dtype=float)
    trail_w_align = np.zeros(W_len, dtype=float)
    trail_res_m = np.zeros(W_len, dtype=float)
    coh_trail_wx = np.zeros(W_len, dtype=float)
    coh_trail_wy = np.zeros(W_len, dtype=float)

    for i in range(W_len):
        i0 = max(0, i - N + 1)
        wx_bar = np.mean(win_dip_wx[i0 : i + 1], axis=0)
        wy_bar = np.mean(win_dip_wy[i0 : i + 1], axis=0)
        am, aw, fr = F._fit_radial_convergence(node_xy_arr, wx_bar, wy_bar)
        trail_align[i] = am
        trail_w_align[i] = aw
        trail_res_m[i] = fr
        if i - i0 + 1 >= 3:
            coh_trail_wx[i] = F._pairwise_corr_mean(win_dip_wx[i0 : i + 1])
            coh_trail_wy[i] = F._pairwise_corr_mean(win_dip_wy[i0 : i + 1])

    out["dir_convergence_align_cos"] = trail_align
    out["dir_convergence_weighted_cos"] = trail_w_align
    out["dir_convergence_residual_m"] = trail_res_m
    out["coherence_trailing_wx"] = coh_trail_wx
    out["coherence_trailing_wy"] = coh_trail_wy
    out["coherence_trailing_signed_mean"] = 0.5 * (coh_trail_wx + coh_trail_wy)

    return out


def extract_sensor_frame_window_vectors_for_spec(
    spec: dict,
    scen_cfg: dict,
    sens_cfg: dict,
    conf_cfg: dict,
    ds_cfg: dict,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict], pd.DataFrame]:
    """Generate the run's true state timeline and telemetry (identically to `dataset.generate_one_run`)
    and return `(vx_win, vy_win, mountings, state_df)` where `(vx_win, vy_win)` are the `(W, 4)`
    sensor-frame window mean tilt changes before world-frame rotation.
    """
    seed = int(spec["seed"])
    split = str(spec["split"])
    run_class = str(spec["run_class"])
    conds = list(spec.get("conditions", [run_class]))
    broad = bool(spec.get("broad_basin", False))
    shifted = bool(spec.get("shifted_noise", False))
    peak_override = spec.get("peak_tilt_deg")

    sc = copy.deepcopy(scen_cfg)
    if broad:
        sc["ranges"]["broad_basin_subset"] = True
    has_basin = ("gradual" in conds) or ("rapid" in conds)
    basin_mode = "rapid" if "rapid" in conds else "gradual"

    state_df, state_meta = engine.run_scenario(
        sc,
        seed=seed,
        mode=basin_mode,
        peak_tilt_deg=peak_override if has_basin else None,
        null_field=not has_basin,
    )
    sim_id = f"sim_{split}_{run_class}_{seed}"
    state_df["simulation_id"] = sim_id

    s_cfg = copy.deepcopy(sens_cfg)
    c_cfg = copy.deepcopy(conf_cfg)
    n_bursts = None
    if shifted:
        sh = ds_cfg["shifted_test"]
        s_cfg["imu"]["noise_multiplier_range"] = list(sh["noise_multiplier_range"])
        s_cfg["imu"]["node_bias_sd_deg"] = float(sh["node_bias_sd_deg"])
        c_cfg = cf.scale_config(c_cfg, float(sh["confounder_scale"]))
        n_bursts = int(sh.get("confounder_n_bursts", 2))

    T = float(state_meta["parameters"]["duration_s"])
    fault_plan = []
    hooks = []
    use_conf = c_cfg if "normal" in conds else None
    if "abnormal_vibration" in conds:
        abn_cfg = ds_cfg["abnormal_vibration"]
        hooks.append(lambda buf, s=seed, ac=abn_cfg: C.inject_abnormal_vibration(buf, s, ac))
    if "sensor_fault" in conds:
        fk = spec.get("sensor_fault_kind")
        fault_plan.append(C.sample_single_fault_spec(seed + 17, T, "sensor_fault", fault_kind=fk))
    if "comm_fault" in conds:
        fk = spec.get("comm_fault_kind")
        fault_plan.append(C.sample_single_fault_spec(seed + 31, T, "comm_fault", fault_kind=fk))

    wcfg = ds_cfg["windowing"]
    lead_in_s = float(wcfg.get("lead_in_baseline_s", 30.0))

    tele_df, _ = S.generate_telemetry(
        state_df,
        seed=seed,
        sensor_cfg=s_cfg,
        profile=profile,
        fault_plan=fault_plan if fault_plan else None,
        confounder_cfg=use_conf,
        lead_in_end_s=lead_in_s if not has_basin else None,
        hooks=hooks,
        n_bursts=n_bursts,
    )

    d = tele_df[tele_df.node_id.isin(NODES)]
    T_len = len(d) // 4
    t = d["time_s"].to_numpy(dtype=float)[::4]
    tx = d["tilt_x_deg"].to_numpy(dtype=float).reshape(T_len, 4)
    ty = d["tilt_y_deg"].to_numpy(dtype=float).reshape(T_len, 4)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    duration_s = float(t[-1] + dt)

    lead_mask = t < min(lead_in_s, duration_s * 0.25)
    tx0 = np.where(np.isfinite(np.nanmedian(tx[lead_mask], axis=0)), np.nanmedian(tx[lead_mask], axis=0), 0.0)
    ty0 = np.where(np.isfinite(np.nanmedian(ty[lead_mask], axis=0)), np.nanmedian(ty[lead_mask], axis=0), 0.0)
    dtx = tx - tx0[None, :]
    dty = ty - ty0[None, :]

    wins = W.make_windows(duration_s, float(wcfg["window_s"]), float(wcfg["overlap"]))
    vx_win = np.zeros((len(wins), 4), dtype=float)
    vy_win = np.zeros((len(wins), 4), dtype=float)
    for w_i, (t0_w, t1_w) in enumerate(wins):
        sl = W.sample_slice(t, t0_w, t1_w)
        dtx_s = dtx[sl]
        dty_s = dty[sl]
        for k in range(4):
            vx_win[w_i, k] = float(np.nanmean(dtx_s[:, k])) if np.isfinite(dtx_s[:, k]).any() else 0.0
            vy_win[w_i, k] = float(np.nanmean(dty_s[:, k])) if np.isfinite(dty_s[:, k]).any() else 0.0

    return vx_win, vy_win, state_meta["mountings"], state_df
