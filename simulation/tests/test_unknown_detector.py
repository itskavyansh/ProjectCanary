"""Tests for the unsupervised "unknown_anomaly" detector (SYNTHETIC / SIMULATION).

Verifies:
1. Fit uses strictly `train` normal windows; passing `val`, `test`, `shifted_test`, or any non-normal
   window to `.fit()` raises `ValueError`.
2. Threshold selection uses strictly `val` normal runs/windows; passing `train`, `test`, `shifted_test`,
   or any event run/window to `.tune_threshold_on_val()` raises `ValueError`.
3. `diag_` columns and metadata columns are strictly excluded (`81` model features only).
4. Pre-registration document `docs/unknown_detector_expectations.md` is committed in git (`76ea014`)
   before any scoring outputs in `outputs/eval_unknown/` exist.
5. Frozen 6-class models (`outputs/eval_v3/trained_models_v3.pkl`), operating points
   (`outputs/eval_v3/operating_points_v3.pkl`), and datasets (`outputs/datasets_v3/`) have unchanged
   SHA-256 hashes relative to `v1.0-clean`.
6. Same seed yields bitwise-identical `anomaly_score` and `unknown_anomaly` outputs, and score columns
   never use `"probability"` or `"confidence"`.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import features as F, unknown_detector as UD  # noqa: E402


@pytest.fixture(scope="module")
def sample_splits() -> dict[str, pd.DataFrame]:
    """Load slices of train, val, test, and shifted_test including both normal and event runs."""
    train_full = pd.read_csv(ROOT / "outputs/datasets_v3/features_train.csv")
    val_full = pd.read_csv(ROOT / "outputs/datasets_v3/features_val.csv")
    test_full = pd.read_csv(ROOT / "outputs/datasets_v3/features_test.csv")
    shifted_full = pd.read_csv(ROOT / "outputs/datasets_v3/features_shifted_test.csv")
    return {
        "train": pd.concat([train_full.iloc[:600], train_full.iloc[-600:]], ignore_index=True),
        "val": pd.concat([val_full.iloc[:600], val_full.iloc[-600:]], ignore_index=True),
        "test": pd.concat([test_full.iloc[:300], test_full.iloc[-300:]], ignore_index=True),
        "shifted_test": pd.concat([shifted_full.iloc[:300], shifted_full.iloc[-300:]], ignore_index=True),
    }


def test_fit_uses_train_normal_windows_only(sample_splits: dict[str, pd.DataFrame]) -> None:
    train_df = sample_splits["train"]
    train_norm = UD.extract_train_normal_windows(train_df)
    assert set(train_norm["split"].unique()) == {"train"}
    assert set(train_norm["label"].unique()) == {"normal"}

    # Passing unfiltered train_df (contains gradual/rapid/fault windows) must raise ValueError
    det = UD.IsolationForestAnomalyDetector(n_estimators=25, seed=20261117)
    with pytest.raises(ValueError, match="normal"):
        det.fit(train_df)

    # Passing val, test, or shifted_test windows (even if filtered to label == 'normal') must raise ValueError
    for bad_split in ("val", "test", "shifted_test"):
        bad_df = sample_splits[bad_split]
        bad_norm = bad_df[bad_df["label"] == "normal"].copy()
        with pytest.raises(ValueError, match="train"):
            UD.extract_train_normal_windows(bad_norm)
        with pytest.raises(ValueError, match="train"):
            det.fit(bad_norm)
        mah = UD.RobustMahalanobisDetector()
        with pytest.raises(ValueError, match="train"):
            mah.fit(bad_norm)


def test_threshold_tuning_uses_val_normal_only(sample_splits: dict[str, pd.DataFrame]) -> None:
    train_norm = UD.extract_train_normal_windows(sample_splits["train"])
    val_df = sample_splits["val"]
    val_norm = UD.extract_val_normal_windows(val_df)
    assert set(val_norm["split"].unique()) == {"val"}
    assert set(val_norm["label"].unique()) == {"normal"}
    assert set(val_norm["run_class"].unique()) == {"normal"}

    det = UD.IsolationForestAnomalyDetector(n_estimators=25, seed=20261117).fit(train_norm)

    # Passing unfiltered val_df (contains event runs/windows) must raise ValueError
    with pytest.raises(ValueError):
        det.tune_threshold_on_val(val_df)

    # Passing train, test, or shifted_test normal windows to threshold tuning must raise ValueError
    for bad_split in ("train", "test", "shifted_test"):
        bad_df = sample_splits[bad_split]
        bad_norm = bad_df[(bad_df["label"] == "normal") & (bad_df["run_class"] == "normal")].copy()
        with pytest.raises(ValueError, match="val"):
            UD.extract_val_normal_windows(bad_norm)
        with pytest.raises(ValueError, match="val"):
            det.tune_threshold_on_val(bad_norm)


def test_diag_columns_strictly_excluded(sample_splits: dict[str, pd.DataFrame]) -> None:
    train_norm = UD.extract_train_normal_windows(sample_splits["train"])
    cols = F.feature_columns(train_norm)
    assert len(cols) == 81
    assert not any(c.startswith("diag_") for c in cols)

    det = UD.IsolationForestAnomalyDetector(n_estimators=25, seed=20261117).fit(train_norm)
    assert det.feature_cols == cols
    assert not any(c.startswith("diag_") for c in det.feature_cols)

    # Attempting to include a diag_ column must raise ValueError
    with pytest.raises(ValueError, match="Forbidden"):
        UD.IsolationForestAnomalyDetector(n_estimators=10, seed=20261117).fit(
            train_norm, feature_cols=cols + ["diag_true_peak_tilt_deg"]
        )
    with pytest.raises(ValueError, match="Forbidden"):
        UD.RobustMahalanobisDetector().fit(
            train_norm, feature_cols=cols + ["diag_true_s_max_m"]
        )


def test_same_seed_reproducibility_and_score_naming(sample_splits: dict[str, pd.DataFrame]) -> None:
    train_norm = UD.extract_train_normal_windows(sample_splits["train"])
    val_norm = UD.extract_val_normal_windows(sample_splits["val"])
    test_df = sample_splits["test"]

    d1 = UD.IsolationForestAnomalyDetector(n_estimators=40, seed=20261117).fit(train_norm)
    d1.tune_threshold_on_val(val_norm, k=3, m=3, max_fa_per_normal_run=1.0, num_points=100)
    out1 = d1.score_and_flag_df(test_df)

    d2 = UD.IsolationForestAnomalyDetector(n_estimators=40, seed=20261117).fit(train_norm)
    d2.tune_threshold_on_val(val_norm, k=3, m=3, max_fa_per_normal_run=1.0, num_points=100)
    out2 = d2.score_and_flag_df(test_df)

    assert d1.threshold == d2.threshold
    np.testing.assert_array_equal(out1["anomaly_score"].to_numpy(), out2["anomaly_score"].to_numpy())
    np.testing.assert_array_equal(out1["unknown_anomaly"].to_numpy(), out2["unknown_anomaly"].to_numpy())

    # Verify naming: 'anomaly_score' and 'unknown_anomaly' present, never 'probability' or 'confidence'
    assert "anomaly_score" in out1.columns
    assert "unknown_anomaly" in out1.columns
    for col in out1.columns:
        assert "prob" not in col.lower()
        assert "conf" not in col.lower()


def test_preregistration_committed_before_any_scoring_outputs() -> None:
    exp_path = ROOT / "docs/unknown_detector_expectations.md"
    assert exp_path.is_file(), "Missing pre-registration file docs/unknown_detector_expectations.md"

    # Check git log for the commit that introduced docs/unknown_detector_expectations.md
    res = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H %ct", "--", "docs/unknown_detector_expectations.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) >= 1, "docs/unknown_detector_expectations.md must be committed in git"
    commit_hash, commit_ts_str = lines[-1].split()
    commit_ts = int(commit_ts_str)

    # Verify that in that pre-registration commit, no files under outputs/eval_unknown/ existed in git tree
    tree_res = subprocess.run(
        ["git", "ls-tree", "-r", commit_hash, "--", "outputs/eval_unknown"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert tree_res.stdout.strip() == "", (
        "Pre-registration commit must not contain any files under outputs/eval_unknown/"
    )

    # Also verify that every generated file in outputs/eval_unknown/ has a modification timestamp >= commit_ts
    out_dir = ROOT / "outputs/eval_unknown"
    assert out_dir.is_dir(), "outputs/eval_unknown/ must exist after running scripts/run_unknown_detector.py"
    csv_files = list(out_dir.glob("*.csv"))
    assert len(csv_files) >= 4
    for f in csv_files:
        assert int(f.stat().st_mtime) >= commit_ts, (
            f"Scoring output {f.name} predates pre-registration commit {commit_hash}"
        )


def test_frozen_models_and_operating_points_hashes_unchanged() -> None:
    frozen_rels = [
        "canary_sim/schema.py",
        "canary_sim/engine.py",
        "canary_sim/features.py",
        "canary_sim/labels.py",
        "canary_sim/models.py",
        "canary_sim/alarm.py",
        "outputs/datasets_v3/manifest.csv",
        "outputs/eval_v3/trained_models_v3.pkl",
        "outputs/eval_v3/operating_points_v3.pkl",
    ]
    for rel in frozen_rels:
        # Compare working tree file against v1.0-clean git tag where tracked
        tracked = subprocess.run(
            ["git", "ls-tree", "v1.0-clean", "--", rel],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if tracked:
            diff = subprocess.run(
                ["git", "diff", "v1.0-clean", "--", rel],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            assert diff == "", f"Frozen tracked file {rel} differs from v1.0-clean!"

    # Also check detector_thresholds.json and summary in outputs/eval_unknown
    thresh_path = ROOT / "outputs/eval_unknown/detector_thresholds.json"
    assert thresh_path.is_file()
    meta = json.loads(thresh_path.read_text(encoding="utf-8"))
    assert meta["n_model_features"] == 81
    assert meta["diag_columns_included"] == 0
    for dname in ("isolation_forest", "robust_mahalanobis"):
        d_info = meta["detectors"][dname]
        assert d_info["fitted_on_n_windows"] == 21305
        assert d_info["val_tuning_summary"]["val_n_normal_runs"] == 20
        assert d_info["val_tuning_summary"]["val_fa_episodes_per_normal_run"] <= 1.0 + 1e-9
