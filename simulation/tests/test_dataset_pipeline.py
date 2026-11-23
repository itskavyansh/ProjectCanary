from __future__ import annotations

import copy
import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    classes as C,
    confounders as cf,
    dataset as ds,
    engine,
    features as F,
    labels as L,
    sensors as S,
    windows as W,
)

SCEN = engine.load_config("configs/scenario_default.yaml")
SENS = S.load_sensor_config("configs/sensor_default.yaml")
CONF = cf.load_confounder_config("configs/confounder_default.yaml")
PROF = S.load_noise_profile("configs/noise_profile.json")
DSCFG = ds.load_dataset_config("configs/dataset_default.yaml")


def _mini_ds_cfg(n_tr=2, n_va=1, n_te=1, n_sh=1, n_bb=1, n_st=4) -> dict:
    c = copy.deepcopy(DSCFG)
    c["splits"] = {"train": n_tr, "val": n_va, "test": n_te}
    c["shifted_test"]["runs_per_class"] = n_sh
    c["broad_basin_subset"]["runs_per_class"] = n_bb
    c["stress_set"]["n_runs"] = n_st
    return c


@pytest.fixture(scope="module")
def built_mini():
    return ds.build_datasets(_mini_ds_cfg(), SCEN, SENS, CONF, PROF)


def test_no_run_appears_in_more_than_one_split(built_mini):
    mf = built_mini["manifest"]
    assert mf["simulation_id"].is_unique
    assert mf["seed"].is_unique
    # Pairwise intersection of simulation_id across splits is empty
    by_split = {sp: set(g["simulation_id"]) for sp, g in mf.groupby("split")}
    splits = list(by_split.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            assert by_split[splits[i]].isdisjoint(by_split[splits[j]])


def test_windows_from_one_run_never_appear_in_two_splits(built_mini):
    feats = built_mini["features"]
    lbls = built_mini["labels"]
    for df in (feats, lbls):
        splits_per_run = df.groupby("simulation_id")["split"].nunique()
        assert (splits_per_run == 1).all()
        by_split = {sp: set(g["simulation_id"]) for sp, g in df.groupby("split")}
        splits = list(by_split.keys())
        for i in range(len(splits)):
            for j in range(i + 1, len(splits)):
                assert by_split[splits[i]].isdisjoint(by_split[splits[j]])


def test_labels_50_percent_rule_aftermath_and_transition_hand_built():
    # Windows of length 5.0 s, 50% overlap -> [0,5), [2.5,7.5), [5,10), [7.5,12.5), [10,15), [12.5,17.5), [15,20)
    wins = W.make_windows(20.0, window_s=5.0, overlap=0.5)

    # Case A: Basin event (gradual) starting at t = 6.0 s (runs to end of timeline = 20.0 s; aftermath keeps class)
    # [0, 5): overlap = 0.0 -> "normal"
    # [2.5, 7.5): overlap [6.0, 7.5] = 1.5 s / 5.0 s = 0.30 (<0.5) -> "transition"
    # [5.0, 10.0): overlap [6.0, 10.0] = 4.0 s / 5.0 s = 0.80 (>=0.5) -> "gradual"
    # [15.0, 20.0) (aftermath): overlap = 1.0 -> "gradual"
    iv_basin = [{"class": "gradual", "start_s": 6.0, "end_s": 20.0, "source": "state_timeline"}]
    lb_basin = L.label_windows(wins, iv_basin, min_overlap=0.5)
    assert list(lb_basin["label"]) == [
        "normal",
        "transition",
        "gradual",
        "gradual",
        "gradual",
        "gradual",
        "gradual",
    ]
    # Exact 50% boundary: event starts at 5.0 s -> [2.5, 7.5) has 2.5 / 5.0 = 0.50 overlap -> active ("gradual")
    iv_exact = [{"class": "gradual", "start_s": 5.0, "end_s": 20.0, "source": "state_timeline"}]
    lb_exact = L.label_windows(wins, iv_exact, min_overlap=0.5)
    assert lb_exact.loc[1, "label"] == "gradual"
    assert lb_exact.loc[0, "label"] == "normal"

    # Case B: Transient fault or abnormal_vibration from t = 6.0 s to t = 11.5 s (returns to normal in aftermath)
    # [0, 5): 0.0 -> "normal"
    # [2.5, 7.5): 1.5/5 = 0.30 -> "transition"
    # [5.0, 10.0): 4.0/5 = 0.80 -> "sensor_fault"
    # [7.5, 12.5): [7.5, 11.5] = 4.0/5 = 0.80 -> "sensor_fault"
    # [10.0, 15.0): [10.0, 11.5] = 1.5/5 = 0.30 -> "transition"
    # [12.5, 17.5): 0.0 -> "normal" (aftermath returns to normal!)
    # [15.0, 20.0): 0.0 -> "normal"
    iv_fault = [{"class": "sensor_fault", "start_s": 6.0, "end_s": 11.5, "source": "bias_step"}]
    lb_fault = L.label_windows(wins, iv_fault, min_overlap=0.5)
    assert list(lb_fault["label"]) == [
        "normal",
        "transition",
        "sensor_fault",
        "sensor_fault",
        "transition",
        "normal",
        "normal",
    ]
    # Transition windows are excluded by trainable() but retain time offsets for detection delay
    tr_mask = L.trainable(lb_fault)
    assert not tr_mask.iloc[1] and not tr_mask.iloc[4]
    assert np.isfinite(lb_fault["t_rel_event_start_s"]).all()


def test_fault_and_confounder_windows_labelled_correctly(built_mini):
    mf = built_mini["manifest"]
    lbls = built_mini["labels"]

    # 1. Normal runs (which contain benign confounders: temp drift, wind/vehicle bursts, spikes)
    norm_ids = mf[(mf.split == "train") & (mf.run_class == "normal")]["simulation_id"]
    assert (mf.loc[mf.simulation_id.isin(norm_ids), "n_confounders"] > 0).all()
    norm_lbls = lbls[lbls.simulation_id.isin(norm_ids)]
    assert set(norm_lbls["label"]) == {"normal"}

    # 2. Sensor fault, comm fault, and abnormal vibration runs: have lead-in normal, active class, and return to normal
    for cls in ("sensor_fault", "comm_fault", "abnormal_vibration"):
        c_ids = mf[(mf.split == "train") & (mf.run_class == cls)]["simulation_id"]
        sub = lbls[lbls.simulation_id.isin(c_ids)]
        assert cls in set(sub["label"])
        for sid, g in sub.groupby("simulation_id"):
            assert g.iloc[0]["label"] == "normal"       # lead-in is normal
            assert g.iloc[-1]["label"] == "normal"      # aftermath after fault/vibration returns to normal

    # 3. Gradual and rapid runs: lead-in is normal, aftermath keeps the basin class
    for cls in ("gradual", "rapid"):
        c_ids = mf[(mf.split == "train") & (mf.run_class == cls)]["simulation_id"]
        sub = lbls[lbls.simulation_id.isin(c_ids)]
        for sid, g in sub.groupby("simulation_id"):
            assert g.iloc[0]["label"] == "normal"
            assert g.iloc[-1]["label"] == cls           # ground stays tilted in aftermath


def test_diag_columns_excluded_from_model_features(built_mini):
    feats = built_mini["features"]
    mcols = F.feature_columns(feats)
    dcols = F.diagnostic_columns(feats)
    assert len(dcols) >= 5
    assert all(c.startswith("diag_") for c in dcols)
    assert not any(c.startswith("diag_") for c in mcols)
    assert set(mcols).isdisjoint(set(dcols))
    # Every model feature is finite (explicit NaN handling)
    assert np.isfinite(feats[mcols].to_numpy(dtype=float)).all()


def test_same_seed_gives_identical_datasets():
    cfg = _mini_ds_cfg(n_tr=1, n_va=1, n_te=1, n_sh=1, n_bb=1, n_st=2)
    a = ds.build_datasets(cfg, SCEN, SENS, CONF, PROF)
    b = ds.build_datasets(cfg, SCEN, SENS, CONF, PROF)
    pd.testing.assert_frame_equal(a["manifest"], b["manifest"])
    pd.testing.assert_frame_equal(a["labels"], b["labels"])
    pd.testing.assert_frame_equal(a["features"], b["features"])


def test_class_counts_per_split_match_config():
    # Verify both on full default spec list and on built mini dataset
    specs = ds.build_run_specs(DSCFG, SCEN)
    sp_df = pd.DataFrame(specs)
    for split, expected in DSCFG["splits"].items():
        counts = sp_df[sp_df.split == split]["run_class"].value_counts().to_dict()
        assert set(counts.keys()) == set(C.CLASSES)
        for cls in C.CLASSES:
            assert counts[cls] == expected

    sh_counts = sp_df[sp_df.split == "shifted_test"]["run_class"].value_counts().to_dict()
    for cls in C.CLASSES:
        assert sh_counts[cls] == DSCFG["shifted_test"]["runs_per_class"]

    bb_counts = sp_df[sp_df.split == "broad_basin"]["run_class"].value_counts().to_dict()
    assert set(bb_counts.keys()) == {"gradual", "rapid"}
    for cls in ("gradual", "rapid"):
        assert bb_counts[cls] == DSCFG["broad_basin_subset"]["runs_per_class"]

    assert (sp_df.split == "stress").sum() == DSCFG["stress_set"]["n_runs"]
