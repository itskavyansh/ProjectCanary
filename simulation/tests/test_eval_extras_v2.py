"""Tests for eval_extras_v2 (SYNTHETIC / SIMULATION)."""
from __future__ import annotations

import hashlib
import inspect
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canary_sim import eval_extras as EE, eval_extras_v2 as V2

ROOT = Path(__file__).resolve().parents[1]
MODELS_P = ROOT / "outputs/eval_v3/trained_models_v3.pkl"
OPS_P = ROOT / "outputs/eval_v3/operating_points_v3.pkl"
DATA = ROOT / "outputs/datasets_v3"


def test_per_hour_conversion_both_definitions_hand_computed():
    # 240 s sensor time, compression 360: A = 240*360/3600 = 24 h, B = 240/3600 = 1/15 h
    assert np.isclose(V2.hours_slow(240.0, 360.0), 24.0)
    assert np.isclose(V2.hours_sensor(240.0), 1.0 / 15.0)
    # 0.6 false alarms per run -> A: 0.6/24 = 0.025 per h (one every 40 h); B: 0.6*15 = 9 per h (one every 1/9 h)
    assert np.isclose(V2.rate_per_hour(0.6, 24.0), 0.025)
    assert np.isclose(V2.rate_per_hour(0.6, 1.0 / 15.0), 9.0)
    assert np.isclose(V2.every_n_hours(0.025), 40.0)
    assert np.isclose(V2.every_n_hours(9.0), 1.0 / 9.0)
    assert V2.every_n_hours(0.0) == float("inf") and V2.fmt_every(float("inf")) == "none observed"
    assert np.isclose(V2.rate_per_hour(1.0, 1.0 / 15.0) / V2.rate_per_hour(1.0, 24.0), 360.0)  # B/A = compression
    with pytest.raises(ValueError):
        V2.hours_sensor(0.0)


def test_normal_run_hours_rejects_mixed_compression():
    m = pd.DataFrame({"split": ["t", "t"], "run_class": ["normal", "normal"],
                      "duration_s": [240.0, 240.0], "compression_factor": [360.0, 1.0]})
    with pytest.raises(ValueError, match="per factor"):
        V2.normal_run_hours(m, "t")
    m["compression_factor"] = 360.0
    d, c, hA, hB = V2.normal_run_hours(m, "t")
    assert (d, c) == (240.0, 360.0) and np.isclose(hA, 24.0) and np.isclose(hB, 1 / 15)


def test_na_for_empty_classes():
    # a split containing only normal / gradual windows: vibration, sensor_fault, comm_fault, rapid have zero true windows
    n = 40
    df = pd.DataFrame({"simulation_id": ["a"] * 20 + ["b"] * 20, "label": ["normal"] * 20 + ["gradual"] * 20})
    preds = np.array(["normal"] * 20 + ["gradual"] * 10 + ["normal"] * 10)
    res = EE.evaluate_window_level_accuracy_and_f1(df, preds, n_boot=20, seed=1)
    row = V2.accuracy_row("toy", "m", "r", res)
    for c in ("abnormal_vibration", "sensor_fault", "comm_fault", "rapid"):
        assert row[f"supp_{c}"] == 0
        assert pd.isna(row[f"rec_{c}"]) and pd.isna(row[f"prec_{c}"])
        assert V2.fmt_pr(row[f"prec_{c}"], row[f"rec_{c}"]) == "N/A / N/A"
    assert row["rec_gradual"] == 0.5 and row["supp_gradual"] == 20
    assert V2.fmt_pr(0.5, 0.25) == "0.50 / 0.25"


def test_na_in_generated_csv_if_present():
    p = ROOT / "outputs/eval_extras_v2/window_accuracy_and_f1_v2.csv"
    if not p.is_file():
        pytest.skip("v2 outputs not generated")
    d = pd.read_csv(p)
    bb = d[d.split == "broad_basin"]
    for c in ("abnormal_vibration", "sensor_fault", "comm_fault"):
        assert bb[f"rec_{c}"].isna().all() and bb[f"prec_{c}"].isna().all()
    assert not d[d.split == "test"]["rec_abnormal_vibration"].isna().any()


def test_no_fit_calls_in_v2_sources():
    for src in (inspect.getsource(V2), (ROOT / "scripts/run_eval_extras_v2.py").read_text(encoding="utf-8")):
        assert ".fit(" not in src and "fit_transform(" not in src and "fit_logistic" not in src and "fit_random_forest" not in src


@pytest.mark.skipif(not MODELS_P.is_file(), reason="frozen models missing")
def test_models_untouched_and_same_seed_identical():
    h_before = hashlib.sha256(MODELS_P.read_bytes()).hexdigest()
    models, ops = pickle.load(open(MODELS_P, "rb")), pickle.load(open(OPS_P, "rb"))
    man = pd.read_csv(DATA / "manifest.csv", low_memory=False)
    df = pd.read_csv(DATA / "features_test.csv", low_memory=False)
    regs = {m: V2.regimes_for(m, models[m], ops, df) for m in ("threshold_rule", "random_forest")}
    r1 = pd.DataFrame(V2.fa_rows_for_split("test", df, regs, man, n_boot=20, seed=7))
    r2 = pd.DataFrame(V2.fa_rows_for_split("test", df, regs, man, n_boot=20, seed=7))
    pd.testing.assert_frame_equal(r1, r2)
    assert (r1["B_sensor_fa_episodes_per_hour"] / r1["A_slow_fa_episodes_per_hour"]).replace([np.inf, -np.inf], np.nan).dropna().between(359, 361).all()
    assert hashlib.sha256(MODELS_P.read_bytes()).hexdigest() == h_before


def test_per_run_class_summary_hand_example():
    det = pd.DataFrame({"simulation_id": ["a", "b", "c"], "run_class": ["gradual"] * 3, "detected": [1.0, 0.0, 1.0]})
    df = pd.DataFrame({"simulation_id": ["a"] * 4 + ["b"] * 4 + ["c"] * 2,
                       "label": ["gradual"] * 10,
                       "run_class": ["gradual"] * 10})
    preds = np.array(["gradual", "gradual", "normal", "normal"] + ["normal"] * 4 + ["gradual", "gradual"])
    pr = V2.per_run_class_table(df, preds, det, "gradual")
    s = V2.summarize_class(pr, n_boot=0, seed=1)
    assert s["window_recall"] == 4 / 10 and s["n_windows"] == 10
    assert np.isclose(s["run_detect_rate"], 2 / 3)       # detection (2/3) is far above window recall (0.4): shown together


def test_snr_bins():
    assert [V2.snr_bin_of(x) for x in (0.5, 1.0, 1.9, 2.0, 4.99, 5.0, 20)] == ["<1", "1-2", "1-2", "2-5", "2-5", ">5", ">5"]
    assert V2.snr_bin_of(float("nan")) is None


def test_old_eval_extras_byte_identical():
    snap = ROOT / "outputs/eval_extras_v2/eval_extras_original_sha256.json"
    if not snap.is_file():
        pytest.skip("hash snapshot missing")
    rec = json.loads(snap.read_text())
    cur = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT / "outputs/eval_extras").iterdir()) if p.is_file()}
    assert cur == rec, "outputs/eval_extras differs from the snapshot taken before v2"
