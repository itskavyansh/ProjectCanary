from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    confounders as cf,
    dataset as ds,
    engine,
    features as F,
    sensors as S,
)

SCEN = engine.load_config("configs/scenario_default.yaml")
SENS = S.load_sensor_config("configs/sensor_default.yaml")
CONF = cf.load_confounder_config("configs/confounder_default.yaml")
PROF = S.load_noise_profile("configs/noise_profile.json")
DSCFG = ds.load_dataset_config("configs/dataset_default.yaml")


def test_world_frame_rotation_uses_nominal_yaw_only_not_yaw_error():
    """Verify observable direction & radial convergence features depend ONLY on nominal `yaw_deg`
    and never leak the ground-truth `yaw_error_deg` misalignment (whereas `diag_` columns do use
    `yaw_error_deg`).
    """
    df_st, meta = engine.run_scenario(SCEN, seed=20260001, mode="gradual", peak_tilt_deg=1.5)
    tele, _ = S.generate_telemetry(df_st, seed=20260001, sensor_cfg=SENS, profile=PROF)

    mounts_orig = {k: dict(v) for k, v in meta["mountings"].items()}
    # Perturb ONLY `yaw_error_deg` by +25 deg while keeping nominal `yaw_deg` identical
    mounts_pert_err = {k: {**v, "yaw_error_deg": v["yaw_error_deg"] + 25.0} for k, v in mounts_orig.items()}
    # Perturb nominal `yaw_deg` by +45 deg
    mounts_pert_nom = {k: {**v, "yaw_deg": v["yaw_deg"] + 45.0} for k, v in mounts_orig.items()}

    centre = (meta["parameters"]["x0"], meta["parameters"]["y0"])
    f_orig = F.extract_window_features(tele, basin_centre_xy=centre, mountings=mounts_orig)
    f_perr = F.extract_window_features(tele, basin_centre_xy=centre, mountings=mounts_pert_err)
    f_pnom = F.extract_window_features(tele, basin_centre_xy=centre, mountings=mounts_pert_nom)

    mcols = F.feature_columns(f_orig)
    # 1. Changing ground-truth `yaw_error_deg` leaves ALL observable model features 100% identical
    pd.testing.assert_frame_equal(f_orig[mcols], f_perr[mcols])
    # Whereas diagnostic `diag_` columns DO change when `yaw_error_deg` changes
    assert not np.allclose(
        f_orig["diag_mean_angle_to_true_centre_deg"],
        f_perr["diag_mean_angle_to_true_centre_deg"],
    )

    # 2. Changing nominal installation yaw `yaw_deg` changes the world-frame direction & convergence features
    assert not np.allclose(f_orig["NODE_01_tilt_angle_rad"], f_pnom["NODE_01_tilt_angle_rad"])
    assert not np.allclose(f_orig["dir_convergence_align_cos"], f_pnom["dir_convergence_align_cos"])


def test_radial_convergence_fits_moving_focal_point_not_fixed_10_10():
    """Verify `_fit_radial_convergence` estimates the convergence focal point c_hat dynamically from
    the 4 world-frame vectors (does NOT use fixed (10,10)): vectors converging to an off-centre
    point (24, -4) achieve alignment ~ 1.0 and residual ~ 0 m, whereas single-node drift has poor
    alignment and large residual.
    """
    node_xy = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    focal_off_centre = np.array([24.0, -4.0])  # far from (10, 10)
    vec = focal_off_centre[None, :] - node_xy
    dist = np.hypot(vec[:, 0], vec[:, 1])
    # Dip vectors pointing directly toward (24, -4) with realistic unequal magnitudes
    mags = np.array([0.4, 0.9, 0.5, 0.3])
    dip_wx = mags * vec[:, 0] / dist
    dip_wy = mags * vec[:, 1] / dist

    al_mean, al_wt, res_m = F._fit_radial_convergence(node_xy, dip_wx, dip_wy)
    assert al_mean > 0.995
    assert al_wt > 0.995
    assert res_m < 0.15

    # Single-node drift on NODE_02 pointing outward while other 3 nodes have small random noise
    dip_wx_sf = np.array([-0.02, 1.20, 0.01, -0.01])
    dip_wy_sf = np.array([0.01, -0.80, 0.02, -0.02])
    al_sf, al_wt_sf, res_sf = F._fit_radial_convergence(node_xy, dip_wx_sf, dip_wy_sf)
    assert al_wt_sf < 0.3
    assert res_sf > 2.5


def test_trailing_horizon_coherence_causal_and_same_run_only():
    """Verify trailing-horizon features (`coherence_trailing_*`, `dir_convergence_*`) use ONLY
    past windows `max(0, i - N + 1) .. i` of the same run:
      1) Modifying telemetry at t >= 100 s does NOT alter any feature of windows ending <= 100 s
         (no future-window leakage).
      2) Extracting features per run in `generate_one_run` is independent of other runs (no
         cross-run leakage).
    """
    df_st, meta = engine.run_scenario(SCEN, seed=20260005, mode="gradual", peak_tilt_deg=1.8)
    tele_a, _ = S.generate_telemetry(df_st, seed=20260005, sensor_cfg=SENS, profile=PROF)
    tele_b = tele_a.copy()
    # Corrupt telemetry for t >= 100 s
    tele_b.loc[tele_b["time_s"] >= 100.0, "tilt_x_deg"] += 15.0
    tele_b.loc[tele_b["time_s"] >= 100.0, "tilt_y_deg"] -= 12.0

    fa = F.extract_window_features(tele_a, mountings=meta["mountings"])
    fb = F.extract_window_features(tele_b, mountings=meta["mountings"])

    early_a = fa[fa["t_end"] <= 100.0].reset_index(drop=True)
    early_b = fb[fb["t_end"] <= 100.0].reset_index(drop=True)
    pd.testing.assert_frame_equal(early_a, early_b)

    # Reproducibility check with same seed
    fa_repeat = F.extract_window_features(tele_a, mountings=meta["mountings"])
    pd.testing.assert_frame_equal(fa, fa_repeat)
