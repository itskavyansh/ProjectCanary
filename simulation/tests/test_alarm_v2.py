from __future__ import annotations

import pickle
import numpy as np
import pandas as pd
import pytest

from canary_sim import alarm as A, models as M


def test_persistence_logic_hand_built_sequences():
    # Sequence with isolated 1-window and 2-window blips, then a 3-window gradual event,
    # then a 1-window normal gap (should NOT clear when m=2), then 2 normal windows (clears).
    seq = [
        "normal",
        "gradual",                      # 1-window blip -> stays normal for k=3
        "normal",
        "sensor_fault", "sensor_fault", # 2-window blip -> stays normal for k=3
        "normal",
        "gradual", "gradual", "gradual", # 3 consecutive -> raises 'gradual' on 3rd window
        "normal",                       # 1 normal window -> stays 'gradual' because m=2
        "gradual",                      # resets normal counter
        "normal", "normal",             # 2 consecutive normal -> clears to 'normal' on 2nd
        "normal",
    ]

    # k = 1, m = 1: every window passes straight through
    out_k1, ep_k1 = A.apply_persistence_sequence(seq, k=1, m=1)
    assert list(out_k1) == seq
    assert ep_k1 == 4

    # k = 2, m = 2: 2-window sensor_fault raises, then 1 normal doesn't clear, then 2 gradual switches
    out_k2, ep_k2 = A.apply_persistence_sequence(seq, k=2, m=2)
    assert out_k2[3] == "normal" and out_k2[4] == "sensor_fault"
    assert ep_k2 >= 2

    # k = 3, m = 2: blips of length 1 and 2 are suppressed; raises at index 8; clears at index 12
    out_k3, ep_k3 = A.apply_persistence_sequence(seq, k=3, m=2)
    expected_k3 = [
        "normal",
        "normal",
        "normal",
        "normal", "normal",
        "normal",
        "normal", "normal", "gradual",
        "gradual",
        "gradual",
        "gradual", "normal",
        "normal",
    ]
    assert list(out_k3) == expected_k3
    assert ep_k3 == 1

    # k = 5, m = 2: nothing reaches 5 consecutive windows
    out_k5, ep_k5 = A.apply_persistence_sequence(seq, k=5, m=2)
    assert list(out_k5) == ["normal"] * len(seq)
    assert ep_k5 == 0


def test_no_test_data_in_persistence_and_operating_point_tuning():
    models = pickle.load(open("outputs/eval/trained_models.pkl", "rb"))
    va = pd.read_csv("outputs/datasets/features_val.csv").iloc[:40].copy()
    rf = models["random_forest"]
    raw_preds = rf.predict(va)

    for forbidden_split in ("test", "shifted_test", "stress", "broad_basin"):
        bad = va.copy()
        bad["split"] = forbidden_split

        with pytest.raises(ValueError, match="forbidden split"):
            A.evaluate_persistence_grid_on_val(bad, raw_preds)

        with pytest.raises(ValueError, match="forbidden split"):
            A.tune_min_prob_operating_point(rf, bad, max_fa_per_normal_run=1.0, k=3, m=2)


def test_false_alarm_target_met_on_val_and_same_seed_reproducible():
    models = pickle.load(open("outputs/eval/trained_models.pkl", "rb"))
    va = pd.read_csv("outputs/datasets/features_val.csv")

    # 1. Threshold Rule with persistence grid on val meets <= 1.0 false alarm per normal run
    th_raw = models["threshold_rule"].predict(va)
    th_best, _ = A.evaluate_persistence_grid_on_val(va, th_raw, max_fa_per_normal_run=1.0)
    assert th_best["meets_fa_target"] is True
    assert th_best["fa_episodes_per_normal_run"] <= 1.0 + 1e-9

    # 2. Random Forest and Logistic Regression operating points tuned on val meet <= 1.0 FA/run
    for mname in ("random_forest", "logistic_regression"):
        op1 = A.tune_min_prob_operating_point(
            models[mname], va, max_fa_per_normal_run=1.0, k=3, m=2, fa_metric="fa_windows_per_normal_run"
        )
        op2 = A.tune_min_prob_operating_point(
            models[mname], va, max_fa_per_normal_run=1.0, k=3, m=2, fa_metric="fa_windows_per_normal_run"
        )
        assert op1.group_thresholds == op2.group_thresholds
        assert op1.tuning_summary["best_val"]["val_fa_episodes_per_normal_run"] <= 1.0 + 1e-9
        assert op1.tuning_summary["best_val"]["val_fa_windows_per_normal_run"] <= 1.0 + 1e-9

        w_preds = op1.predict_window(va)
        res_a = A.evaluate_alarm_level(va, w_preds, k=op1.k, m=op1.m, n_boot=40, seed=77)
        res_b = A.evaluate_alarm_level(va, w_preds, k=op1.k, m=op1.m, n_boot=40, seed=77)
        assert res_a["fa_episodes_per_normal_run"] <= 1.0 + 1e-9
        assert res_a["fa_windows_per_normal_run"] <= 1.0 + 1e-9
        assert res_a["exact_detect_rate"] == res_b["exact_detect_rate"]
        assert res_a["exact_detect_ci95"] == res_b["exact_detect_ci95"]
