"""Tests for pipeline v1, dataset versioning, and validation checks.

Verifies:
1. Pipeline runs end-to-end with --quick.
2. Same config and seed produce identical outputs.
3. metadata.json checksum verification passes on fresh generation and fails if tampered.
4. Integrity checks catch deliberately corrupted data.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from canary_sim.validation_checks import run_all
from canary_sim.versioning import verify_dataset_dir, write_dataset_metadata
from run_pipeline import run_pipeline


def test_versioning_tamper_detection(tmp_path: Path):
    """Test that metadata.json passes when valid and fails when tampered."""
    data_dir = tmp_path / "dataset"
    data_dir.mkdir()

    f1 = data_dir / "sample1.csv"
    f1.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    f2 = data_dir / "sample2.csv"
    f2.write_text("x,y,z\n4,5,6\n", encoding="utf-8")

    meta = write_dataset_metadata(data_dir, extra={"seed": 42, "notes": "test"})
    assert (data_dir / "metadata.json").is_file()

    # Clean check passes
    problems = verify_dataset_dir(data_dir)
    assert len(problems) == 0, f"Expected clean check, got errors: {problems}"

    # Tamper with sample1
    f1.write_text("a,b,c\n9,9,9\n", encoding="utf-8")
    problems_tampered = verify_dataset_dir(data_dir)
    assert len(problems_tampered) > 0
    assert any("sample1.csv" in err for err in problems_tampered)

    # Missing file
    f2.unlink()
    problems_missing = verify_dataset_dir(data_dir)
    assert len(problems_missing) > 0
    assert any("MISSING file: sample2.csv" in err for err in problems_missing)


@pytest.fixture(scope="module")
def quick_pipe_output(tmp_path_factory) -> Path:
    """Run quick pipeline once for the module tests."""
    out_dir = tmp_path_factory.mktemp("quick_pipe")
    cfg = Path("configs/pipeline_v1.yaml")
    res = run_pipeline(
        config_path=cfg,
        out_dir=out_dir,
        stages=["data", "features", "models", "eval"],
        quick=True,
    )
    assert res == 0
    return out_dir


def test_integrity_checks_on_tampered_dataset(quick_pipe_output: Path, tmp_path: Path):
    """Verify integrity checks catch deliberately corrupted dataset files."""
    data_dir = quick_pipe_output / "datasets"

    # 1. Clean dataset passes all integrity checks
    rep = run_all(data_dir, raise_on_fail=False)
    assert rep.success, f"Expected clean dataset to pass: {rep.render()}"

    # 2. Corrupted features with duplicate windows
    tamper_dir_dup = tmp_path / "tamper_dup"
    shutil.copytree(data_dir, tamper_dir_dup)
    fea_dup = pd.read_csv(tamper_dir_dup / "window_features.csv")
    fea_dup = pd.concat([fea_dup.iloc[0:1], fea_dup], ignore_index=True)
    fea_dup.to_csv(tamper_dir_dup / "window_features.csv", index=False)
    rep_dup = run_all(tamper_dir_dup, raise_on_fail=False)
    assert not rep_dup.success
    assert "duplicate_windows" in rep_dup.failed

    # 3. Corrupted features with impossible values (negative piezo RMS)
    tamper_dir_imp = tmp_path / "tamper_imp"
    shutil.copytree(data_dir, tamper_dir_imp)
    fea_imp = pd.read_csv(tamper_dir_imp / "window_features.csv")
    col = [c for c in fea_imp.columns if "piezo_rms_mean" in c][0]
    fea_imp.loc[0, col] = -99.0
    fea_imp.to_csv(tamper_dir_imp / "window_features.csv", index=False)
    rep_imp = run_all(tamper_dir_imp, raise_on_fail=False)
    assert not rep_imp.success
    assert "impossible_values" in rep_imp.failed

    # 4. Corrupted features with undocumented missing value (NaN in tilt)
    tamper_dir_nan = tmp_path / "tamper_nan"
    shutil.copytree(data_dir, tamper_dir_nan)
    fea_nan = pd.read_csv(tamper_dir_nan / "window_features.csv")
    col_tilt = [c for c in fea_nan.columns if "mean" in c and "NODE" in c][0]
    fea_nan.loc[0, col_tilt] = None
    fea_nan.to_csv(tamper_dir_nan / "window_features.csv", index=False)
    rep_nan = run_all(tamper_dir_nan, raise_on_fail=False)
    assert not rep_nan.success
    assert "missing_values" in rep_nan.failed


def test_pipeline_quick_execution(quick_pipe_output: Path, tmp_path: Path):
    """Test end-to-end pipeline execution with --quick and reproducibility."""
    out_dir_1 = quick_pipe_output
    out_dir_2 = tmp_path / "pipe_run_2"
    cfg = Path("configs/pipeline_v1.yaml")

    # Verify out_dir_1 dataset directory has valid metadata.json
    meta_problems = verify_dataset_dir(out_dir_1 / "datasets")
    assert len(meta_problems) == 0, f"Metadata verification failed: {meta_problems}"

    # Verify model and eval outputs exist in out_dir_1
    assert (out_dir_1 / "models" / "trained_models.pkl").is_file()
    assert (out_dir_1 / "models" / "operating_points.pkl").is_file()
    assert (out_dir_1 / "models" / "model_settings.json").is_file()
    assert (out_dir_1 / "eval" / "metrics_summary.json").is_file()
    assert (out_dir_1 / "eval" / "headline_metrics.csv").is_file()

    # Run 2 (identical config and seed)
    res2 = run_pipeline(
        config_path=cfg,
        out_dir=out_dir_2,
        stages=["data", "features", "models", "eval"],
        quick=True,
    )
    assert res2 == 0

    # Compare key artifacts between Run 1 and Run 2 for exact equality
    m1 = pd.read_csv(out_dir_1 / "datasets" / "manifest.csv")
    m2 = pd.read_csv(out_dir_2 / "datasets" / "manifest.csv")
    pd.testing.assert_frame_equal(m1, m2)

    l1 = pd.read_csv(out_dir_1 / "datasets" / "window_labels.csv")
    l2 = pd.read_csv(out_dir_2 / "datasets" / "window_labels.csv")
    pd.testing.assert_frame_equal(l1, l2)

    f1 = pd.read_csv(out_dir_1 / "datasets" / "window_features.csv")
    f2 = pd.read_csv(out_dir_2 / "datasets" / "window_features.csv")
    pd.testing.assert_frame_equal(f1, f2)

    e1 = pd.read_csv(out_dir_1 / "eval" / "headline_metrics.csv")
    e2 = pd.read_csv(out_dir_2 / "eval" / "headline_metrics.csv")
    pd.testing.assert_frame_equal(e1, e2)
