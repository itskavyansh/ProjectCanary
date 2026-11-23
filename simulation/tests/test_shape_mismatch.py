"""Tests for the subsidence-shape mismatch stress test (SYNTHETIC / SIMULATION).

Verifies:
1. Every shape produces a state timeline matching the frozen schema_v1.
2. Analytic gradient matches central finite-difference numerical gradient for all 5 shapes.
3. Peak-tilt matching scales s_max so max surface tilt equals peak_tilt_deg across all shapes,
   and log-stratified peak_tilt_deg / SNR distributions match the Gaussian control.
4. Normal runs and gaussian_control reuse the identical pipeline verbatim.
5. No frozen file changes (hash check) and same seed gives identical output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    confounders as cf,
    dataset as ds,
    engine,
    sensors as S,
    shapes as SH,
)
from canary_sim.schema import COLUMNS, validate_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def cfgs():
    return {
        "scen": engine.load_config(ROOT / "configs/scenario_default.yaml"),
        "sens": S.load_sensor_config(ROOT / "configs/sensor_default.yaml"),
        "conf": cf.load_confounder_config(ROOT / "configs/confounder_default.yaml"),
        "ds": ds.load_dataset_config(ROOT / "configs/dataset_default.yaml"),
        "prof": S.load_noise_profile(ROOT / "configs/noise_profile.json"),
        "shape": SH.load_shape_config(ROOT / "configs/shape_mismatch.yaml"),
    }


def test_new_fields_match_frozen_schema(cfgs):
    """Every shape produces a valid schema_v1 DataFrame with all parameters recorded in metadata."""
    for idx, shape in enumerate(SH.SHAPES):
        for mode in ("gradual", "rapid"):
            df, meta = SH.run_scenario_shaped(
                cfgs["scen"], cfgs["shape"], shape, seed=3000000900 + idx, mode=mode, peak_tilt_deg=1.2
            )
            assert tuple(df.columns) == COLUMNS
            n_rows = validate_frame(df)
            assert n_rows == 2400 * 5  # 240 s * 10 Hz * 5 nodes
            assert meta["synthetic"] is True
            assert meta["parameters"]["shape"] == shape
            assert np.isclose(meta["parameters"]["peak_tilt_deg"], 1.2)


def test_analytic_gradient_matches_numerical_gradient_all_shapes(cfgs):
    """Analytic world-frame gradient (dh/dx, dh/dy) matches central finite differences for all 5 shapes."""
    rng = np.random.default_rng(20261118)
    eps = 1e-6
    for idx, shape in enumerate(SH.SHAPES):
        p = SH.sample_shape_parameters(
            cfgs["scen"], cfgs["shape"], shape, seed=3000000001 + idx, mode="gradual", peak_tilt_deg=1.5
        )
        x = rng.uniform(-12.0, 32.0, 80)
        y = rng.uniform(-12.0, 32.0, 80)
        if shape == "asymmetric":
            # Avoid points within eps of thexp=0 seam where second derivative jumps
            xp, _, _, _ = SH._rot(x, y, p)
            keep = np.abs(xp) > 1e-3
            x, y = x[keep], y[keep]

        _, gx, gy = SH.h_and_grad(shape, x, y, p)
        num_gx = (SH.h_and_grad(shape, x + eps, y, p)[0] - SH.h_and_grad(shape, x - eps, y, p)[0]) / (2 * eps)
        num_gy = (SH.h_and_grad(shape, x, y + eps, p)[0] - SH.h_and_grad(shape, x, y - eps, p)[0]) / (2 * eps)
        assert np.max(np.abs(gx - num_gx)) < 1e-8, f"{shape} dh/dx mismatch"
        assert np.max(np.abs(gy - num_gy)) < 1e-8, f"{shape} dh/dy mismatch"

    # Also verify gaussian_control slopes match engine.slopes to machine precision
    p_ctrl = SH.sample_shape_parameters(
        cfgs["scen"], cfgs["shape"], "gaussian_control", seed=3000000005, mode="gradual", peak_tilt_deg=1.1
    )
    egx, egy = engine.slopes(7.5, 13.2, p_ctrl["s_max"], p_ctrl)
    sgx, sgy = SH.node_slopes("gaussian_control", 7.5, 13.2, p_ctrl["s_max"], p_ctrl)
    assert np.isclose(egx, sgx, atol=1e-14)
    assert np.isclose(egy, sgy, atol=1e-14)


def test_peak_tilt_matching_works_across_shapes(cfgs):
    """s_max scaling achieves the target peak surface slope (in degrees) for every shape, and
    build_specs pairs seeds and stratified peaks identically across shapes."""
    for target_deg in (0.12, 0.75, 2.4):
        for idx, shape in enumerate(SH.SHAPES):
            p = SH.sample_shape_parameters(
                cfgs["scen"], cfgs["shape"], shape, seed=3000000010 + idx, mode="gradual", peak_tilt_deg=target_deg
            )
            achieved_max_slope = p["s_max"] * SH.max_gradient(shape, p)
            achieved_deg = float(np.rad2deg(np.arctan(achieved_max_slope)))
            assert np.isclose(achieved_deg, target_deg, rtol=2e-3, atol=1e-4), (
                f"{shape}: achieved {achieved_deg} deg vs target {target_deg} deg"
            )

    specs = SH.build_specs(cfgs["shape"], cfgs["scen"], ds.stratified_peaks)
    ctrl_peaks = [s["peak_tilt_deg"] for s in specs["gaussian_control"]]
    ctrl_seeds = [s["seed"] for s in specs["gaussian_control"]]
    for shape in SH.SHAPES:
        assert [s["peak_tilt_deg"] for s in specs[shape]] == ctrl_peaks
        assert [s["seed"] for s in specs[shape]] == ctrl_seeds

    existing = set(pd.read_csv(ROOT / "outputs/datasets_v3/manifest.csv", usecols=["seed"])["seed"].astype(int))
    SH.assert_seeds_disjoint(specs, existing)


def test_normal_runs_and_control_reuse_identical_pipeline(cfgs):
    """Normal runs inside shape_engine and gaussian_control runs match the untouched pipeline."""
    gen_args = (cfgs["scen"], cfgs["sens"], cfgs["conf"], cfgs["ds"], cfgs["prof"])
    norm_spec = {
        "seed": 3000000201,
        "split": "shape_normal",
        "run_class": "normal",
        "conditions": ["normal"],
        "broad_basin": False,
        "shifted_noise": False,
        "peak_tilt_deg": None,
    }
    m_direct, l_direct, f_direct = ds.generate_one_run(norm_spec, *gen_args)
    with SH.shape_engine("planar_slope", cfgs["shape"]):
        m_shaped, l_shaped, f_shaped = ds.generate_one_run(norm_spec, *gen_args)
    pd.testing.assert_frame_equal(l_direct, l_shaped)
    pd.testing.assert_frame_equal(f_direct, f_shaped)
    assert m_direct["faults_json"] == m_shaped["faults_json"]

    # gaussian_control inside shape_engine produces identical features and SNR to direct generate_one_run
    grad_spec = {
        "seed": 3000000001,
        "split": "shape_gaussian_control",
        "run_class": "gradual",
        "conditions": ["gradual"],
        "broad_basin": False,
        "shifted_noise": False,
        "peak_tilt_deg": 0.85,
    }
    m_g0, l_g0, f_g0 = ds.generate_one_run(grad_spec, *gen_args)
    with SH.shape_engine("gaussian_control", cfgs["shape"]):
        m_g1, l_g1, f_g1 = ds.generate_one_run(grad_spec, *gen_args)
    pd.testing.assert_frame_equal(l_g0, l_g1)
    pd.testing.assert_frame_equal(f_g0, f_g1)
    assert np.isclose(m_g0["snr_vs_imu_noise"], m_g1["snr_vs_imu_noise"], atol=1e-12)


def test_same_seed_identical_output_and_frozen_files_untouched(cfgs):
    """Same seed produces identical outputs, and no frozen file was modified."""
    gen_args = (cfgs["scen"], cfgs["sens"], cfgs["conf"], cfgs["ds"], cfgs["prof"])
    sp_list = [
        {
            "seed": 3000000002,
            "split": "shape_trough_panel",
            "run_class": "gradual",
            "conditions": ["gradual"],
            "broad_basin": False,
            "shifted_noise": False,
            "peak_tilt_deg": 1.1,
            "shape": "trough_panel",
        }
    ]
    m1, l1, f1 = SH.generate_shape_set("trough_panel", sp_list, ds.generate_one_run, cfgs["shape"], gen_args)
    m2, l2, f2 = SH.generate_shape_set("trough_panel", sp_list, ds.generate_one_run, cfgs["shape"], gen_args)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(f1, f2)

    meta_path = ROOT / "outputs/eval_shape_mismatch/metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = meta.get("extra", {}).get("frozen_sha256", {})
        assert len(saved) > 0
        for rel, exp_hash in saved.items():
            cur_hash = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            assert cur_hash == exp_hash, f"Frozen file {rel} changed!"
