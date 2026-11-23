"""Tests for evaluation extras (SYNTHETIC / SIMULATION).

Verifies:
1. Per-hour conversion matches hand-computed examples.
2. Permutation importance accepts VAL only (rejects test/train/stress).
3. No test-split data enters any fit or tuning routine.
4. Same seed gives identical outputs.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from canary_sim import eval_extras as EE, features as F


def test_per_hour_conversion_matches_hand_computed_example():
    """Verify conversion from per-run false alarms to per-simulated-hour matches exact hand arithmetic."""
    # Hand-computed case 1:
    # 240 s duration at 360x compression
    dur1 = 240.0
    comp1 = 360.0
    # Expected hours: (240.0 * 360.0) / 3600.0 = 86400.0 / 3600.0 = 24.0 hours per run
    expected_hours_1 = 24.0
    got_hours_1 = EE.calculate_simulated_hours_per_run(dur1, comp1)
    assert np.isclose(got_hours_1, expected_hours_1, atol=1e-12)

    # Hand-computed case 2:
    # 3 runs with 6, 2, and 4 false alarm episodes:
    # total FA = 12, mean FA per run = 4.0
    # FA per simulated hour = 4.0 / 24.0 = 1/6 = 0.16666666666666666
    fa_episodes = [6.0, 2.0, 4.0]
    mean_fa_run = float(np.mean(fa_episodes))
    assert np.isclose(mean_fa_run, 4.0, atol=1e-12)
    fa_per_hour = mean_fa_run / got_hours_1
    assert np.isclose(fa_per_hour, 1.0 / 6.0, atol=1e-12)

    # Hand-computed case 3 (uncompressed / rapid run):
    # 240 s duration at 1.0x compression:
    # (240.0 * 1.0) / 3600.0 = 240 / 3600 = 1/15 hour = 0.06666666666666667 hours
    got_hours_uncompressed = EE.calculate_simulated_hours_per_run(240.0, 1.0)
    assert np.isclose(got_hours_uncompressed, 1.0 / 15.0, atol=1e-12)

    # Invalid non-positive parameters reject loudly
    with pytest.raises(ValueError):
        EE.calculate_simulated_hours_per_run(-10.0, 360.0)
    with pytest.raises(ValueError):
        EE.calculate_simulated_hours_per_run(240.0, 0.0)


def test_permutation_importance_uses_val_only(tmp_path: Path):
    """Verify that permutation importance strictly rejects any test, train, or stress splits."""
    models_path = Path("outputs/eval_v3/trained_models_v3.pkl")
    if not models_path.is_file():
        pytest.skip("Frozen v3 models not found")
    models = pickle.load(open(models_path, "rb"))
    rf = models["random_forest"]

    # Create dummy DataFrame with split == "test"
    dummy_test = pd.DataFrame({
        "simulation_id": ["run_test_01"] * 10,
        "split": ["test"] * 10,
        "window_idx": list(range(10)),
        "t_start": [float(i * 2.5) for i in range(10)],
        "t_end": [float(i * 2.5 + 5.0) for i in range(10)],
        "label": ["normal"] * 10,
        "NODE_01_mean": [0.1] * 10,
    })

    with pytest.raises(ValueError, match="VAL split only"):
        EE.compute_grouped_permutation_importance(
            rf, dummy_test, {"NODE_01_mean": ["NODE_01_mean"]}, n_repeats=2, seed=42,
        )

    # Dummy train split rejected
    dummy_train = dummy_test.copy()
    dummy_train["split"] = "train"
    with pytest.raises(ValueError, match="VAL split only"):
        EE.compute_grouped_permutation_importance(
            rf, dummy_train, {"NODE_01_mean": ["NODE_01_mean"]}, n_repeats=2, seed=42,
        )

    # Dummy stress split rejected
    dummy_stress = dummy_test.copy()
    dummy_stress["split"] = "stress"
    with pytest.raises(ValueError, match="VAL split only"):
        EE.compute_grouped_permutation_importance(
            rf, dummy_stress, {"NODE_01_mean": ["NODE_01_mean"]}, n_repeats=2, seed=42,
        )


def test_no_test_split_data_enters_any_fit():
    """Verify that evaluation extras module does not fit, retrain, or calibrate models."""
    import inspect

    funcs = [getattr(EE, name) for name in dir(EE) if inspect.isfunction(getattr(EE, name))]
    # Assert none of the extras functions contain .fit(
    for fn in funcs:
        src = inspect.getsource(fn)
        assert ".fit(" not in src, f"Function {fn.__name__} calls .fit()"
        assert "fit_transform(" not in src, f"Function {fn.__name__} calls fit_transform()"


def test_same_seed_gives_identical_outputs():
    """Verify that same seed produces bitwise identical permutation importance values."""
    models_path = Path("outputs/eval_v3/trained_models_v3.pkl")
    val_path = Path("outputs/datasets_v3/features_val.csv")
    if not models_path.is_file() or not val_path.is_file():
        pytest.skip("Frozen artifacts not found")

    models = pickle.load(open(models_path, "rb"))
    rf = models["random_forest"]
    val_df = pd.read_csv(val_path, nrows=190)  # 2 runs of 95 windows

    target = {"mag_01": ["NODE_01_mean", "NODE_01_std"]}
    res1 = EE.compute_grouped_permutation_importance(rf, val_df, target, n_repeats=3, seed=20261117)
    res2 = EE.compute_grouped_permutation_importance(rf, val_df, target, n_repeats=3, seed=20261117)

    for k in ("mean_importance", "std_importance", "min_importance", "max_importance"):
        assert res1["mag_01"][k] == res2["mag_01"][k], f"Mismatch in {k}: {res1} vs {res2}"
