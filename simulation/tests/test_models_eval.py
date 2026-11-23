from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from canary_sim import (
    classes as C,
    evaluate as ev,
    features as F,
    labels as L,
    models as M,
)


@pytest.fixture(scope="module")
def ds_sub():
    """Load a small deterministic subset of runs from the frozen train/val/test/shifted_test splits."""
    tr_full = pd.read_csv("outputs/datasets/features_train.csv")
    va_full = pd.read_csv("outputs/datasets/features_val.csv")
    te_full = pd.read_csv("outputs/datasets/features_test.csv")
    sh_full = pd.read_csv("outputs/datasets/features_shifted_test.csv")

    def _take_runs(df: pd.DataFrame, k_per_cls: int) -> pd.DataFrame:
        ids = []
        for cls in C.CLASSES:
            rids = df[df["run_class"] == cls]["simulation_id"].unique()[:k_per_cls]
            ids.extend(rids)
        return df[df["simulation_id"].isin(ids)].copy().reset_index(drop=True)

    return {
        "train": _take_runs(tr_full, 4),
        "val": _take_runs(va_full, 2),
        "test": _take_runs(te_full, 2),
        "shifted_test": _take_runs(sh_full, 2),
    }


def test_no_test_data_in_fit_guard(ds_sub):
    tr = ds_sub["train"]
    va = ds_sub["val"]
    te = ds_sub["test"]
    sh = ds_sub["shifted_test"]

    for forbidden_split in ("test", "shifted_test", "stress", "broad_basin"):
        bad = tr.iloc[:10].copy()
        bad["split"] = forbidden_split
        mixed = pd.concat([tr, bad], ignore_index=True)

        with pytest.raises(ValueError, match="forbidden split"):
            M.guard_no_test_data(mixed)

        with pytest.raises(ValueError, match="forbidden split"):
            M.ThresholdRuleClassifier().fit(mixed, va)

        with pytest.raises(ValueError, match="forbidden split"):
            M.ThresholdRuleClassifier().fit(tr, bad)

        with pytest.raises(ValueError, match="forbidden split"):
            M.fit_logistic_regression(mixed, va)

        with pytest.raises(ValueError, match="forbidden split"):
            M.fit_random_forest(mixed, va, fixed_params={"n_estimators": 10, "max_depth": 5, "min_samples_leaf": 1})

    # Also verify diag_ columns are rejected by guard_model_features
    with pytest.raises(ValueError, match="Forbidden non-model columns"):
        M.guard_model_features(["NODE_01_mean", "diag_NODE_01_angle_to_true_centre_deg"])


def test_threshold_values_come_from_train_and_val_only(ds_sub):
    tr = ds_sub["train"]
    va = ds_sub["val"]
    clf1 = M.ThresholdRuleClassifier().fit(tr, va)
    clf2 = M.ThresholdRuleClassifier().fit(tr.copy(), va.copy())
    assert clf1.thresholds == clf2.thresholds
    assert clf1.fitted_on_splits == ("train", "val")

    # Perturbing train_df changes the calibrated quantiles, whereas test/shifted_test are never passed
    tr_mod = tr.copy()
    tr_mod.loc[tr_mod["label"] == "normal", "piezo_rms_network_mean"] *= 1.5
    clf_mod = M.ThresholdRuleClassifier().fit(tr_mod, va)
    assert clf_mod.thresholds["piezo_rms_network_mean_th"] != clf1.thresholds["piezo_rms_network_mean_th"]


def test_reproducibility_same_seed_gives_same_metrics(ds_sub):
    tr, va, te = ds_sub["train"], ds_sub["val"], ds_sub["test"]
    grid = {"n_estimators": [20], "max_depth": [8], "min_samples_leaf": [2]}

    rf1 = M.fit_random_forest(tr, va, seed=42, param_grid=grid)
    rf2 = M.fit_random_forest(tr, va, seed=42, param_grid=grid)
    lr1 = M.fit_logistic_regression(tr, va, seed=42, c_grid=(1.0,))
    lr2 = M.fit_logistic_regression(tr, va, seed=42, c_grid=(1.0,))

    p_rf1 = rf1.predict(te)
    p_rf2 = rf2.predict(te)
    assert np.array_equal(p_rf1, p_rf2)
    assert np.array_equal(lr1.predict(te), lr2.predict(te))

    ev1 = ev.evaluate_window_level_with_run_bootstrap(te, p_rf1, n_boot=50, seed=99)
    ev2 = ev.evaluate_window_level_with_run_bootstrap(te, p_rf2, n_boot=50, seed=99)
    assert ev1["macro_f1"] == ev2["macro_f1"]
    assert ev1["macro_f1_ci95"] == ev2["macro_f1_ci95"]
    assert np.array_equal(ev1["confusion_matrix"], ev2["confusion_matrix"])


def test_bootstrap_resamples_whole_runs_and_cm_sums_match_window_counts(ds_sub):
    tr, va, te = ds_sub["train"], ds_sub["val"], ds_sub["test"]
    th = M.ThresholdRuleClassifier().fit(tr, va)
    preds = th.predict(te)

    res = ev.evaluate_window_level_with_run_bootstrap(te, preds, n_boot=60, seed=123)
    trainable_count = int(L.trainable(te).sum())

    # Confusion matrix sum matches exact trainable window count
    assert int(res["confusion_matrix"].sum()) == trainable_count
    assert res["n_windows"] == trainable_count
    assert sum(v["support"] for v in res["per_class"].values()) == trainable_count

    # Verify first bootstrap replicate resampled whole runs (len == n_runs, all valid run IDs)
    unique_runs = set(te["simulation_id"].unique())
    assert res["n_runs"] == len(unique_runs)
    assert len(res["first_boot_run_ids"]) == len(unique_runs)
    assert set(res["first_boot_run_ids"]).issubset(unique_runs)

    # Reconstruct the exact confusion matrix of replicate 0 by pulling whole runs from `te`
    te_tr = te[L.trainable(te)].copy()
    te_tr["_pred"] = preds[L.trainable(te).to_numpy()]
    rep0_frames = [te_tr[te_tr["simulation_id"] == rid] for rid in res["first_boot_run_ids"]]
    rep0_df = pd.concat(rep0_frames, ignore_index=True)
    cm0 = ev.confusion_matrix_from_labels(rep0_df["label"].to_numpy(), rep0_df["_pred"].to_numpy())
    m0 = ev.metrics_from_cm(cm0)
    assert m0["macro_f1"] == pytest.approx(res["boot_macro_f1_samples"][0])
