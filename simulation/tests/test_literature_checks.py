"""Tests for literature-basis sensitivity checks & timescale audit (SYNTHETIC / SIMULATION).

Verifies:
1. Knothe sigma formula matches hand calculation and Gaussian equivalence exp(-pi r^2/R^2) == exp(-r^2/(2 sigma^2)).
2. New B1 (high-tilt) and B2 (realistic-depth sigma > 1.5 radii) fields match the frozen schema_v1.
3. Peak-tilt matching works for both high-tilt and realistic-depth arms.
4. Pre-registration file (docs/literature_checks_expectations.md) was committed in git BEFORE any scoring
   output files in outputs/eval_literature/ existed.
5. No frozen file hash changes and no .fit( call in the new module or runner.
6. Same seed gives identical output, and Part C timescale audit math matches hand calculation.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    confounders as cf,
    dataset as ds,
    engine,
    literature_checks as LC,
    node_geometry as ng,
    sensors as S,
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
        "lit": LC.load_literature_config(ROOT / "configs/literature_checks.yaml"),
    }


def test_sigma_formula_matches_hand_calculation(cfgs):
    """Verify sigma = H / (tan(beta) * sqrt(2*pi)) and equivalence with exp(-pi * r^2 / R^2)."""
    H = 200.0
    tan_b = 2.0
    R_inf = H / tan_b  # 100.0 m
    expected_sigma_m = 100.0 / math.sqrt(2.0 * math.pi)  # 39.89422804014327 m
    expected_sigma_rad = expected_sigma_m / (10.0 * math.sqrt(2.0))  # 2.8209479177387813 radii

    got_m = LC.knothe_sigma_m(H, tan_b)
    got_rad = LC.knothe_sigma_radii(H, tan_b)
    assert np.isclose(got_m, expected_sigma_m, atol=1e-12)
    assert np.isclose(got_rad, expected_sigma_rad, atol=1e-12)

    # Verify Knothe influence exp(-pi * r^2 / R^2) == Gaussian exp(-r^2 / (2 * sigma^2))
    r = np.linspace(0.0, 250.0, 50)
    knothe_val = np.exp(-math.pi * (r**2) / (R_inf**2))
    gauss_val = np.exp(-(r**2) / (2.0 * (got_m**2)))
    assert np.allclose(knothe_val, gauss_val, atol=1e-14)

    # Verify the 12-cell grid has 7 cells above 1.5 radii and 5 cells <= 1.5 radii
    grid = LC.build_depth_sigma_grid(cfgs["lit"])
    assert len(grid) == 12
    assert int(grid["above_main_cap_1p5"].sum()) == 7
    assert int((~grid["above_main_cap_1p5"]).sum()) == 5

    with pytest.raises(ValueError):
        LC.knothe_sigma_m(-50.0, 2.0)
    with pytest.raises(ValueError):
        LC.knothe_sigma_m(100.0, 0.0)


def test_new_fields_match_frozen_schema_and_peak_tilt_matching(cfgs):
    """Both B1 high-tilt and B2 wide-sigma runs satisfy schema_v1 and match target peak_tilt_deg."""
    # B1 high-tilt (9.5 deg)
    df_b1, meta_b1 = LC.run_scenario_literature(
        cfgs["scen"], seed=4000000901, mode="gradual", peak_tilt_deg=9.5, fixed_sigma_m=None
    )
    assert tuple(df_b1.columns) == COLUMNS
    assert validate_frame(df_b1) == 2400 * 5
    p1 = meta_b1["parameters"]
    # Max surface slope occurs at distance sigma_min from (x0, y0) along the steeper principal axis
    sigma_min_1 = min(p1["sx"], p1["sy"])
    achieved_slope_1 = p1["s_max"] * engine.E_HALF / sigma_min_1
    assert np.isclose(np.rad2deg(np.arctan(achieved_slope_1)), 9.5, atol=1e-10)

    # B2 realistic-depth wide sigma (H=300 m, tanB=1.5 -> sigma = 79.788 m = 5.642 radii > 1.5 cap)
    sig_300_1p5 = LC.knothe_sigma_m(300.0, 1.5)
    assert sig_300_1p5 / ng.NETWORK_RADIUS_M > 5.0
    df_b2, meta_b2 = LC.run_scenario_literature(
        cfgs["scen"], seed=4000000902, mode="rapid", peak_tilt_deg=2.1, fixed_sigma_m=sig_300_1p5
    )
    assert tuple(df_b2.columns) == COLUMNS
    assert validate_frame(df_b2) == 2400 * 5
    p2 = meta_b2["parameters"]
    assert np.isclose(p2["sx"], sig_300_1p5, atol=1e-12)
    assert np.isclose(p2["sy"], sig_300_1p5, atol=1e-12)
    # Check analytic slope at (x0 + sigma, y0) equals tan(2.1 deg)
    gx, gy = engine.slopes(p2["x0"] + sig_300_1p5, p2["y0"], p2["s_max"], p2)
    achieved_deg_2 = float(np.rad2deg(np.arctan(np.hypot(gx, gy))))
    assert np.isclose(achieved_deg_2, 2.1, atol=1e-10)


def test_preregistration_committed_before_scoring_outputs():
    """Verify via git log and filesystem timestamps that docs/literature_checks_expectations.md
    was committed before any outputs/eval_literature/ scoring files existed."""
    prereg_rel = "docs/literature_checks_expectations.md"
    assert (ROOT / prereg_rel).is_file()

    # 1. Commit that added docs/literature_checks_expectations.md
    r_hash = subprocess.run(
        ["git", "log", "--diff-filter=A", "-n", "1", "--format=%H", "--", prereg_rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit_hash = r_hash.stdout.strip()
    assert len(commit_hash) >= 7, "docs/literature_checks_expectations.md must be committed in git"

    # 2. That pre-registration commit touched ONLY docs/literature_checks_expectations.md (and no outputs/eval_literature files)
    r_files = subprocess.run(
        ["git", "show", "--name-only", "--format=", commit_hash],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    files_in_prereg_commit = [line.strip() for line in r_files.stdout.splitlines() if line.strip()]
    assert files_in_prereg_commit == [prereg_rel], (
        f"Pre-registration commit {commit_hash} must contain only {prereg_rel}, got {files_in_prereg_commit}"
    )

    # 3. In git tree at commit_hash, outputs/eval_literature did NOT exist
    r_tree = subprocess.run(
        ["git", "ls-tree", commit_hash, "outputs/eval_literature"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert r_tree.stdout.strip() == "", "outputs/eval_literature must not exist in the pre-registration commit tree"

    # 4. If outputs/eval_literature is later committed, its first commit in git log must be a descendant after commit_hash
    r_out_hash = subprocess.run(
        ["git", "log", "--diff-filter=A", "-n", "1", "--format=%H", "--", "outputs/eval_literature"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    out_commit_hash = r_out_hash.stdout.strip()
    if out_commit_hash:
        assert out_commit_hash != commit_hash
        # commit_hash must be an ancestor of out_commit_hash
        anc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit_hash, out_commit_hash],
            cwd=ROOT,
        )
        assert anc.returncode == 0, "Pre-registration commit must be an ancestor of the scoring output commit"


def test_no_frozen_file_hash_changes_and_no_fit_calls():
    """Ensure zero .fit( calls in literature_checks.py / run_literature_checks.py and verify frozen SHA-256 hashes."""
    for src in (
        inspect.getsource(LC),
        (ROOT / "scripts/run_literature_checks.py").read_text(encoding="utf-8"),
    ):
        assert ".fit(" not in src
        assert "fit_transform(" not in src
        assert "fit_random_forest" not in src
        assert "fit_logistic_regression" not in src

    meta_path = ROOT / "outputs/eval_literature/metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        saved = meta.get("extra", {}).get("frozen_sha256", {})
        assert len(saved) > 0
        for rel, exp_hash in saved.items():
            cur_hash = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            assert cur_hash == exp_hash, f"Frozen file {rel} modified!"


def test_same_seed_identical_output_and_timescale_audit_hand_example(cfgs):
    """Same seed produces bitwise/numerically identical DataFrames, and timescale audit matches hand arithmetic."""
    gen_args = (cfgs["scen"], cfgs["sens"], cfgs["conf"], cfgs["ds"], cfgs["prof"])
    sig_m = LC.knothe_sigma_m(200.0, 2.0)
    ainfo = {
        "kind": "B2_realistic_depth",
        "fixed_sigma_m": sig_m,
        "extra_meta": {"arm": "H200_tanB2p0", "H_m": 200.0, "tan_beta": 2.0},
        "specs": [
            {
                "seed": 4000009901,
                "split": "lit_H200_tanB2p0",
                "run_class": "gradual",
                "conditions": ["gradual"],
                "broad_basin": False,
                "shifted_noise": False,
                "peak_tilt_deg": 1.25,
                "arm": "H200_tanB2p0",
            }
        ],
    }
    m1, l1, f1 = LC.generate_arm_runs("H200_tanB2p0", ainfo, ds.generate_one_run, gen_args)
    m2, l2, f2 = LC.generate_arm_runs("H200_tanB2p0", ainfo, ds.generate_one_run, gen_args)
    pd.testing.assert_frame_equal(m1, m2)
    pd.testing.assert_frame_equal(l1, l2)
    pd.testing.assert_frame_equal(f1, f2)

    # Hand-computed timescale audit check:
    # peak_tilt_deg = 1.2 deg, tau = 120 s, duration_s = 240 s, compression = 360 -> physical tau = 0.5 days (12 hours)
    # implied rate = 1.2 deg / 0.5 days = 2.4 deg/day -> 24.0x faster than 0.1 deg/day reference
    toy_man = pd.DataFrame([
        {
            "simulation_id": "toy_grad_1",
            "split": "test",
            "run_class": "gradual",
            "conditions": "gradual",
            "broad_basin_subset": False,
            "peak_tilt_deg": 1.2,
            "duration_s": 240.0,
            "compression_factor": 360.0,
            "parameters_json": json.dumps({"tau": 120.0}),
        }
    ])
    per_run, summary = LC.audit_frozen_gradual_timescales(toy_man, ref_rate_deg_per_day=0.1)
    assert np.isclose(per_run["tau_phys_days_at_360x"].iloc[0], 0.5, atol=1e-12)
    assert np.isclose(per_run["implied_tilt_rate_deg_per_day_at_360x"].iloc[0], 2.4, atol=1e-12)
    assert np.isclose(per_run["speedup_factor_vs_0p1_deg_per_day"].iloc[0], 24.0, atol=1e-12)
    assert int(summary["n_faster_than_0p1_deg_per_day"].iloc[0]) == 1
