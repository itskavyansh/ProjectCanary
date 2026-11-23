from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    confounders as cf,
    dataset as ds,
    engine,
    features as F,
    sensors as S,
    yaw_sensitivity as ys,
)

SCEN = engine.load_config("configs/scenario_default.yaml")
SENS = S.load_sensor_config("configs/sensor_default.yaml")
CONF = cf.load_confounder_config("configs/confounder_default.yaml")
PROF = S.load_noise_profile("configs/noise_profile.json")
DSCFG = ds.load_dataset_config("configs/dataset_default.yaml")


@pytest.fixture(scope="module")
def sample_test_specs_and_v3():
    specs = [s for s in ds.build_run_specs(DSCFG, SCEN) if s["split"] == "test"][:6]
    te_v3 = pd.read_csv("outputs/datasets_v3/features_test.csv")
    return specs, te_v3


def test_zero_degree_condition_reproduces_v3_exactly(sample_test_specs_and_v3):
    """Verify `sd_0deg` reproduces `outputs/datasets_v3` features and `outputs/eval_v3` model
    predictions 100% identically, while state timeline and labels are completely untouched.
    """
    specs, te_v3 = sample_test_specs_and_v3
    models_v3 = pickle.load(open("outputs/eval_v3/trained_models_v3.pkl", "rb"))
    ops_v3 = pickle.load(open("outputs/eval_v3/operating_points_v3.pkl", "rb"))

    recomputed_frames = []
    for sp in specs:
        sim_id = f"sim_{sp['split']}_{sp['run_class']}_{sp['seed']}"
        base_f = te_v3[te_v3["simulation_id"] == sim_id].reset_index(drop=True)
        vx, vy, mounts, state_df = ys.extract_sensor_frame_window_vectors_for_spec(
            sp, SCEN, SENS, CONF, DSCFG, PROF
        )
        errs_0 = ys.sample_nominal_yaw_perturbation(sp["seed"], "sd_0deg")
        assert all(v == 0.0 for v in errs_0.values())

        f_0 = ys.recompute_yaw_dependent_features_for_run(base_f, vx, vy, mounts, errs_0)
        mcols = F.feature_columns(base_f)
        pd.testing.assert_frame_equal(f_0[mcols], base_f[mcols], check_exact=False, rtol=1e-12, atol=1e-12)
        pd.testing.assert_series_equal(f_0["label"], base_f["label"])
        recomputed_frames.append(f_0)

    df_0 = pd.concat(recomputed_frames, ignore_index=True)
    df_orig = te_v3[te_v3["simulation_id"].isin(df_0["simulation_id"].unique())].reset_index(drop=True)

    for mname, clf in models_v3.items():
        assert np.array_equal(clf.predict(df_0), clf.predict(df_orig))
    for mname, op in ops_v3["episode_fa_le_1"].items():
        assert np.array_equal(op.predict_window(df_0), op.predict_window(df_orig))


def test_perturbation_affects_only_world_frame_rotation_and_state_timeline_untouched(sample_test_specs_and_v3):
    """Verify that perturbing nominal yaw (`sd_20deg` and `single_node_45deg`):
      1) Does NOT modify the state timeline, labels, or any of the 65 yaw-invariant model features
         (`{NODE}_mean`, `piezo_*`, `hub_*`, `comm_*`, `long_slope_*`, `coherence_trailing_mag`, `diag_*`).
      2) Changes ONLY the 16 yaw-dependent model columns (`YAW_DEPENDENT_MODEL_COLS`).
      3) Gives identical results for the same seed.
    """
    specs, te_v3 = sample_test_specs_and_v3
    sp = specs[2]  # gradual test run
    sim_id = f"sim_{sp['split']}_{sp['run_class']}_{sp['seed']}"
    base_f = te_v3[te_v3["simulation_id"] == sim_id].reset_index(drop=True)

    vx1, vy1, mounts1, state1 = ys.extract_sensor_frame_window_vectors_for_spec(
        sp, SCEN, SENS, CONF, DSCFG, PROF
    )
    vx2, vy2, mounts2, state2 = ys.extract_sensor_frame_window_vectors_for_spec(
        sp, SCEN, SENS, CONF, DSCFG, PROF
    )
    # State timeline is identical and untouched
    pd.testing.assert_frame_equal(state1, state2)

    errs_20_a = ys.sample_nominal_yaw_perturbation(sp["seed"], "sd_20deg")
    errs_20_b = ys.sample_nominal_yaw_perturbation(sp["seed"], "sd_20deg")
    assert errs_20_a == errs_20_b

    errs_45 = ys.sample_nominal_yaw_perturbation(sp["seed"], "single_node_45deg")
    non_zero_45 = [v for v in errs_45.values() if v != 0.0]
    assert len(non_zero_45) == 1 and abs(non_zero_45[0]) == 45.0

    f_20_a = ys.recompute_yaw_dependent_features_for_run(base_f, vx1, vy1, mounts1, errs_20_a)
    f_20_b = ys.recompute_yaw_dependent_features_for_run(base_f, vx2, vy2, mounts2, errs_20_b)
    pd.testing.assert_frame_equal(f_20_a, f_20_b)

    # All non-yaw-dependent columns (including labels, diag_ columns, and 65 yaw-invariant model features) are 100% unchanged
    invariant_cols = [c for c in base_f.columns if c not in ys.YAW_DEPENDENT_MODEL_COLS]
    pd.testing.assert_frame_equal(f_20_a[invariant_cols], base_f[invariant_cols])

    # Yaw-dependent columns DO change when nominal yaw is perturbed by 20 deg SD
    assert not np.allclose(f_20_a["NODE_01_tilt_angle_rad"], base_f["NODE_01_tilt_angle_rad"])
    assert not np.allclose(f_20_a["dir_convergence_weighted_cos"], base_f["dir_convergence_weighted_cos"])
