import numpy as np
import pytest

from canary_sim import engine, node_geometry as ng
from canary_sim.schema import validate_frame

CFG = engine.load_config("configs/scenario_default.yaml")


def test_isotropic_max_slope():
    sigma, s_max = 3.0, 0.5
    p = dict(x0=0.0, y0=0.0, sx=sigma, sy=sigma, theta_deg=33.0)
    xs = np.linspace(-4 * sigma, 4 * sigma, 4001)
    gx, gy = engine.slopes(xs, np.zeros_like(xs), s_max, p)
    mag = np.hypot(gx, gy)
    assert mag.max() == pytest.approx(0.607 * s_max / sigma, rel=2e-3)
    assert abs(abs(xs[mag.argmax()]) - sigma) < 0.01 * sigma
    # also in a 2-D radial sense / rotated direction
    X, Y = np.meshgrid(xs[::20], xs[::20])
    gx, gy = engine.slopes(X, Y, s_max, p)
    assert np.hypot(gx, gy).max() == pytest.approx(0.6065 * s_max / sigma, rel=1e-2)


def test_analytic_slope_matches_numeric_anisotropic():
    p = dict(x0=3.0, y0=-2.0, sx=4.0, sy=7.0, theta_deg=40.0)
    x, y, h = 5.0, 1.0, 1e-5
    gx, gy = engine.slopes(x, y, 1.0, p)
    nx = (engine.settlement(x + h, y, 1.0, p) - engine.settlement(x - h, y, 1.0, p)) / (2 * h)
    ny = (engine.settlement(x, y + h, 1.0, p) - engine.settlement(x, y - h, 1.0, p)) / (2 * h)
    assert gx == pytest.approx(nx, rel=1e-6) and gy == pytest.approx(ny, rel=1e-6)


def test_tilt_sign_convention():
    # Bowl east of a yaw-0 node: ground falls toward +x => sensor +x side lower => tilt_y > 0
    p = dict(x0=10.0, y0=0.0, sx=5.0, sy=5.0, theta_deg=0.0)
    m = ng.Mounting("N", 0.0, 0.0, 0.0, 0.0)
    tx, ty = engine.node_tilt_deg(0.0, 0.0, 0.3, p, m)
    assert ty > 0 and abs(tx) < 1e-9
    # Bowl north: ground falls toward +y => sensor +y side lower => tilt_x < 0
    p2 = dict(x0=0.0, y0=10.0, sx=5.0, sy=5.0, theta_deg=0.0)
    tx, ty = engine.node_tilt_deg(0.0, 0.0, 0.3, p2, m)
    assert tx < 0 and abs(ty) < 1e-9


def test_same_seed_identical_different_seed_differs():
    a, ma = engine.run_scenario(CFG, seed=5)
    b, mb = engine.run_scenario(CFG, seed=5)
    c, _ = engine.run_scenario(CFG, seed=6)
    assert a.equals(b) and ma["parameters"] == mb["parameters"]
    assert not a.drop(columns="simulation_id").equals(c.drop(columns="simulation_id"))


def test_centre_outside_square_valid():
    found = 0
    for seed in range(200):
        p = engine.sample_parameters(CFG, seed)
        if not (0 <= p["x0"] <= 20 and 0 <= p["y0"] <= 20):
            df, _ = engine.run_scenario(CFG, seed=seed)
            assert np.isfinite(df.select_dtypes("number").to_numpy()).all()
            found += 1
            if found >= 5:
                break
    assert found >= 5


@pytest.mark.parametrize("mode", ["gradual", "rapid"])
def test_schema_validation_and_metadata(mode):
    for seed in range(5):
        df, meta = engine.run_scenario(CFG, seed=seed, mode=mode)
        assert validate_frame(df) == len(df)
        P = meta["parameters"]
        assert P["t_on"] + P["tau"] <= 0.9 * P["duration_s"] + 1e-9
        assert P["compression_factor"] > 0
        assert set(df.active_class) >= {"normal", mode, "aftermath"}
        # peak tilt parameter is the max slope of the final field
        assert np.rad2deg(np.arctan(P["s_max"] * engine.E_HALF / min(P["sx"], P["sy"]))) == \
            pytest.approx(P["peak_tilt_deg"])


def test_schema_rejects_bad_frame():
    df, _ = engine.run_scenario(CFG, seed=1)
    with pytest.raises(Exception):
        validate_frame(df.rename(columns={"true_z": "z"}))
    bad = df.copy()
    bad.loc[0, "active_class"] = "bogus"
    with pytest.raises(Exception):
        validate_frame(bad)


def test_sigma_cap_and_broad_basin_flag():
    rad = ng.NETWORK_RADIUS_M
    assert CFG["ranges"]["broad_basin_subset"] is False  # off by default
    def worst(cfg):
        ps = [engine.sample_parameters(cfg, s) for s in range(400)]
        return max(max(p["sx"], p["sy"]) for p in ps)
    assert worst(CFG) <= 1.5 * rad + 1e-9
    broad_cfg = {**CFG, "ranges": {**CFG["ranges"], "broad_basin_subset": True}}
    assert 1.5 * rad < worst(broad_cfg) <= 2.5 * rad + 1e-9
    bad = {**CFG, "ranges": {**CFG["ranges"], "sigma_radii": [0.4, 2.5]}}
    with pytest.raises(ValueError):
        engine.sample_parameters(bad, 1)
