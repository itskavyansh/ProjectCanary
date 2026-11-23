"""Pytest configuration for ProjectCanary/simulation tests.

When `simulation/` is copied into ProjectCanary without the large generated `outputs/`
directory (as required by repo size rules), tests that read generated CSVs or `.pkl`
models under `simulation/outputs/` are skipped with an explicit message pointing the
user to `python run_pipeline.py` instead of failing with a raw FileNotFoundError.
"""
from __future__ import annotations

from pathlib import Path
import pytest

SIM_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = SIM_ROOT / "outputs"

DATASET_DEPENDENT_MODULES = {
    "test_alarm_v2.py",
    "test_models_eval.py",
    "test_yaw_sensitivity.py",
    "test_unknown_detector.py",
}

DATASET_DEPENDENT_TESTS = {
    "test_peak_tilt_matching_works_across_shapes",
    "test_preregistration_committed_before_scoring_outputs",
    "test_preregistration_committed_before_any_scoring_outputs",
    "test_frozen_models_and_operating_points_hashes_unchanged",
}


def _datasets_generated() -> bool:
    return (
        (OUTPUTS_DIR / "datasets/manifest.csv").is_file()
        and (OUTPUTS_DIR / "datasets_v3/manifest.csv").is_file()
        and (OUTPUTS_DIR / "eval_v3/trained_models_v3.pkl").is_file()
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _datasets_generated():
        return
    skip_marker = pytest.mark.skip(
        reason=(
            "Generated simulation datasets/models are not present in simulation/outputs/ "
            "(excluded from git). Run `python run_pipeline.py` from `simulation/` first to generate them."
        )
    )
    for item in items:
        fspath_name = Path(str(item.fspath)).name
        if fspath_name in DATASET_DEPENDENT_MODULES or item.name in DATASET_DEPENDENT_TESTS:
            item.add_marker(skip_marker)
