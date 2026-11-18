import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from canary_sim import confounders as cf, engine, faults as fl, sensors as S

SCEN = engine.load_config("configs/scenario_default.yaml")
SCFG = S.load_sensor_config("configs/sensor_default.yaml")
PROF = S.load_noise_profile("configs/noise_profile.json")
CCFG = cf.load_confounder_config("configs/confounder_default.yaml")
VALUE_COLS = ["tilt_x_deg", "tilt_y_deg", "piezo_peak", "piezo_rms", "piezo_peak_to_peak"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    df, meta = engine.run_scenario(SCEN, seed=20260001, mode="gradual")
    out = engine.write_run(df, meta, tmp_path_factory.mktemp("runs"))
    return out / "state_timeline.csv"


def tele(path, seed=1, plan=None, conf=None, tl=None):
    return S.generate_telemetry(tl if tl is not None else path, seed, SCFG, PROF, plan, conf)


def node(df, n):
    return df[df.node_id == n].reset_index(drop=True)


def others_equal(a, b, n):
    for m in S.NODES:
        if m != n:
            pd.testing.assert_frame_equal(node(a, m), node(b, m))


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_same_seed_identical_different_seed_differs(run):
    a, ma = tele(run, 3, conf=CCFG)
    b, mb = tele(run, 3, conf=CCFG)
    c, _ = tele(run, 4, conf=CCFG)
    pd.testing.assert_frame_equal(a, b)
    assert ma == mb
    assert not a[VALUE_COLS].equals(c[VALUE_COLS])
    assert 0.5 <= ma["noise_multiplier"] <= 2.0


def test_faults_never_change_state_timeline(run):
    h = sha(run)
    before = pd.read_csv(run)
    plan = [fl.make_spec(k, 100 + i, 30, "ALL" if k == "network_outage" else ("HUB" if v[1] == "hub" else ["NODE_02"]), i)
            for i, (k, v) in enumerate(fl.FAULTS.items())]
    df, meta = tele(run, 2, plan, CCFG)
    S.write_telemetry(df, meta, run.parent, "faulty")
    assert sha(run) == h
    pd.testing.assert_frame_equal(pd.read_csv(run), before)
    assert "tilt_x_deg" not in before.columns
    assert len(meta["faults"]) == len(plan)


NODE_FAULTS = [k for k, v in fl.FAULTS.items() if v[1] == "node"]


@pytest.mark.parametrize("kind", NODE_FAULTS)
def test_single_node_fault_leaves_other_nodes_unchanged(run, kind):
    clean, _ = tele(run, 5)
    faulty, meta = tele(run, 5, [fl.make_spec(kind, 120, 60, "NODE_02", 7)])
    others_equal(clean, faulty, "NODE_02")
    assert any(f["node"] == "NODE_02" for f in meta["faults"])


def test_single_node_fault_changes_that_node(run):
    clean, _ = tele(run, 5)
    faulty, _ = tele(run, 5, [fl.make_spec("bias_step", 120, 60, "NODE_02", 7)])
    assert not node(clean, "NODE_02").equals(node(faulty, "NODE_02"))


def test_hub_self_motion_shifts_all_four_ranges_consistently(run):
    clean, _ = tele(run, 6)
    shift = 0.30
    faulty, meta = tele(run, 6, [fl.make_spec("hub_self_motion", 150, 60, "HUB", 9, shift_m=shift)])
    amp = meta["faults"][0]["params"]["shift_m"]
    assert abs(amp) == pytest.approx(shift)
    q = SCFG["lidar"]["quantisation_m"]
    d = (faulty.lidar_range_change_m - clean.lidar_range_change_m)
    plateau = (faulty.time_s >= 156) & (faulty.time_s <= 204) & (faulty.lidar_new == 1)
    assert plateau.sum() >= 4
    got = d[plateau].groupby(faulty.node_id[plateau]).mean()
    assert len(got) == 4                                    # all 4 nodes shifted
    assert np.all(np.sign(got) == np.sign(amp))             # same sign
    assert np.allclose(got, amp, atol=1.5 * q)              # same magnitude
    outside = (faulty.time_s < 148) & faulty.lidar_range_change_m.notna()
    assert np.allclose(d[outside], 0, atol=1e-12)


def test_single_node_displacement_shifts_only_its_own_range(run):
    tl = pd.read_csv(run)
    moved = tl.copy()
    sel = (moved.node_id == "NODE_03") & (moved.time_s >= 150)
    moved.loc[sel, "true_x"] += 3.0                         # in-memory only
    a, _ = tele(run, 8)
    b, _ = tele(run, 8, tl=moved)
    late = (a.time_s >= 150) & (a.lidar_new == 1)
    for n in S.NODES:
        m = late & (a.node_id == n)
        d = (b.lidar_range_change_m - a.lidar_range_change_m)[m]
        if n == "NODE_03":
            assert (d > 1.0).all() and len(d) > 0           # sqrt(13^2+10^2) - 14.14 ~ +2.2 m
        else:
            assert np.allclose(d, 0, atol=1e-12)
    pd.testing.assert_frame_equal(a[VALUE_COLS], b[VALUE_COLS])   # IMU/piezo untouched


def test_missing_packets_are_nan_not_zero(run):
    plan = [fl.make_spec("packet_drops", 100, 100, "NODE_01", 1, n=10),
            fl.make_spec("node_unavailable", 50, 30, "NODE_03", 2),
            fl.make_spec("network_outage", 200, 10, "ALL", 3)]
    df, _ = tele(run, 9, plan)
    lost = df.packet_received == 0
    assert lost.sum() >= 10 + 30 + 40
    assert df.loc[lost, VALUE_COLS].isna().all().all()
    assert df.loc[lost, "arrival_time_s"].isna().all()
    assert (df.loc[~lost, VALUE_COLS] == 0).sum().sum() == 0
    out = (df.time_s >= 200) & (df.time_s < 210)
    assert (df.loc[out, "hub_link_up"] == 0).all() and df.loc[out, "lidar_range_change_m"].isna().all()


def test_every_fault_in_metadata_with_correct_times(run):
    plan = [fl.make_spec("slow_drift", 60, 90, ["NODE_01", "NODE_04"], 1),
            fl.make_spec("hub_self_motion", 150, 40, "HUB", 2),
            fl.make_spec("burst_loss", 20, 15, "NODE_02", 3),
            fl.make_spec("network_outage", 210, 5, "ALL", 4)]
    df, meta = tele(run, 11, plan)
    recs = meta["faults"]
    assert len(recs) == 5                                   # two-node fault -> two records
    key = lambda r: (r["type"], r["node"], r["start_s"], r["end_s"])
    assert {key(r) for r in recs} == {
        ("slow_drift", "NODE_01", 60.0, 150.0), ("slow_drift", "NODE_04", 60.0, 150.0),
        ("hub_self_motion", "HUB", 150.0, 190.0), ("burst_loss", "NODE_02", 20.0, 35.0),
        ("network_outage", "ALL", 210.0, 215.0)}
    S.write_telemetry(df, meta, run.parent, "meta")
    assert json.load(open(run.parent / "telemetry_meta_metadata.json"))["faults"] == json.loads(json.dumps(recs))


def test_sampled_plan_defaults_one_node_sometimes_two():
    sizes = []
    for s in range(200):
        for sp in fl.sample_fault_plan(s, 240.0, kinds=["bias_step"]):
            sizes.append(len(sp["target"]))
    assert set(sizes) == {1, 2} and sizes.count(1) > sizes.count(2) > 0


def test_lidar_sparse_scanning_and_baseline(run):
    df, meta = tele(run, 12)
    P = SCFG["lidar"]["sweep_period_s"]
    for n in S.NODES:
        d = node(df, n)
        assert d.lidar_new.sum() == pytest.approx(240 / P, abs=1.5)    # one reading per pass
        assert d.lidar_new.mean() < 0.2                                # sparse, not every window
        lead = d[(d.time_s < meta["lead_in_end_s"]) & d.lidar_range_change_m.notna()]
        assert abs(lead.lidar_range_change_m.median()) < 0.05          # relative to own baseline
    assert "lidar_range_m" not in df.columns


def test_confounders_stay_small_and_margins_enforced(run):
    clean, _ = tele(run, 13)
    conf, meta = tele(run, 13, conf=CCFG)
    assert meta["confounders"]
    d = (conf.piezo_rms - clean.piezo_rms).abs()
    assert d.max() <= CCFG["benign"]["vibration_burst"]["rms_add_max"] + 0.03   # + impulse headroom
    assert (conf.tilt_x_deg - clean.tilt_x_deg).abs().max() < 0.3
    bad = json.loads(json.dumps(CCFG))
    bad["benign"]["vibration_burst"]["rms_add_max"] = 0.05
    with pytest.raises(ValueError):
        cf.check_margins(bad)
    nodes_hit = {r["node"] for r in meta["confounders"] if r["type"] in ("wind", "vehicle")}
    assert len(nodes_hit) <= 4
