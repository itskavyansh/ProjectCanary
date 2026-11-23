"""Tests for scripts/export_dashboard_data.py (SYNTHETIC / SIMULATION).

Verifies:
1. No `.fit(` or `fit_transform(` calls appear anywhere in `scripts/export_dashboard_data.py`.
2. Export is 100% deterministic: two independent runs produce byte-identical JSON files (SHA-256).
3. Frozen model file SHA-256 hashes (`trained_models_v3.pkl`, `operating_points_v3.pkl`, etc.)
   are unchanged before and after running the export.
4. Every exported JSON file is under 300 KB and total export is under 2 MB.
5. Every exported JSON file carries `"synthetic": true`, `"compression_factor"`,
   `"git_commit"`, and `"banner": "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA"`.
6. JSON schema for `summary.json`, `results_*.json`, `robustness.json`, `parameters.json`,
   and all 6 `replay_<class>.json` files matches expected keys, uses `model_score` (never
   `probability` or `confidence`), and contains no secret-looking strings.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_dashboard_data as E  # noqa: E402

EXPECTED_FILES = (
    "summary.json",
    "results_by_split.json",
    "results_false_alarms.json",
    "results_recall_vs_snr.json",
    "results_model_comparison.json",
    "robustness.json",
    "parameters.json",
    "replay_normal.json",
    "replay_gradual.json",
    "replay_rapid.json",
    "replay_abnormal_vibration.json",
    "replay_sensor_fault.json",
    "replay_comm_fault.json",
)

SECRET_PATTERNS = (
    re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}"),  # JWT token
    re.compile(r"sk_(live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"sbp_[A-Za-z0-9]{20,}"),
    re.compile(r"SUPABASE_SERVICE_ROLE_KEY", re.IGNORECASE),
    re.compile(r"FAST2SMS_API_KEY", re.IGNORECASE),
    re.compile(r"PRIVATE_KEY", re.IGNORECASE),
)


def _has_frozen_outputs() -> bool:
    return (
        (ROOT / "outputs/datasets_v3/features_test.csv").is_file()
        and (ROOT / "outputs/eval_v3/trained_models_v3.pkl").is_file()
        and (ROOT / "outputs/eval_v3/operating_points_v3.pkl").is_file()
    )


def _find_export_dir() -> Path:
    sim_out = ROOT / "outputs/dashboard_export"
    if sim_out.is_dir() and all((sim_out / f).is_file() for f in EXPECTED_FILES):
        return sim_out
    pub_sim = ROOT.parent / "public/sim"
    if pub_sim.is_dir() and all((pub_sim / f).is_file() for f in EXPECTED_FILES):
        return pub_sim
    if _has_frozen_outputs():
        E.export_all(out_dir=sim_out, root=ROOT)
        return sim_out
    pytest.skip(
        "Generated simulation outputs not found in outputs/ — run `python run_pipeline.py` "
        "and `python scripts/export_dashboard_data.py` to generate datasets and dashboard exports."
    )


def test_no_fit_calls_in_export_script() -> None:
    script_path = ROOT / "scripts/export_dashboard_data.py"
    assert script_path.is_file(), "Missing scripts/export_dashboard_data.py"
    src = script_path.read_text(encoding="utf-8")
    # Strip comments and docstrings before checking for any .fit( call, AND also check raw source
    assert ".fit(" not in src, "scripts/export_dashboard_data.py must not contain any .fit( call"
    assert "fit_transform(" not in src
    assert "fit_logistic_regression(" not in src
    assert "fit_random_forest(" not in src


def test_deterministic_export_and_frozen_model_hashes_unchanged(tmp_path: Path) -> None:
    if not _has_frozen_outputs():
        pytest.skip(
            "Frozen outputs/eval_v3 and outputs/datasets_v3 not present in this checkout — "
            "run `python run_pipeline.py` to generate datasets before running full export determinism test."
        )

    hashes_before = E.snapshot_model_hashes(ROOT)
    assert len(hashes_before) >= 2, "Expected frozen v3 model and operating-point pickle files"

    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    paths1 = E.export_all(out_dir=run1_dir, root=ROOT, seed=20261117)
    paths2 = E.export_all(out_dir=run2_dir, root=ROOT, seed=20261117)

    hashes_after = E.snapshot_model_hashes(ROOT)
    assert hashes_before == hashes_after, "Frozen model SHA-256 hashes changed during export!"

    assert set(paths1.keys()) == set(EXPECTED_FILES)
    assert set(paths2.keys()) == set(EXPECTED_FILES)

    for fname in EXPECTED_FILES:
        h1 = hashlib.sha256(paths1[fname].read_bytes()).hexdigest()
        h2 = hashlib.sha256(paths2[fname].read_bytes()).hexdigest()
        assert h1 == h2, f"Non-deterministic export detected for {fname}: {h1} != {h2}"


def test_export_file_sizes_and_common_envelope() -> None:
    exp_dir = _find_export_dir()
    total_bytes = 0

    for fname in EXPECTED_FILES:
        fpath = exp_dir / fname
        assert fpath.is_file(), f"Missing exported file {fname}"
        sz = fpath.stat().st_size
        assert 0 < sz <= E.MAX_FILE_BYTES, f"{fname} size {sz} exceeds 300 KB ({E.MAX_FILE_BYTES})"
        total_bytes += sz

        data = json.loads(fpath.read_text(encoding="utf-8"))
        assert data.get("synthetic") is True, f"{fname} missing synthetic: true"
        assert data.get("banner") == "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA"
        assert "compression_factor" in data and isinstance(data["compression_factor"], dict)
        assert data["compression_factor"].get("gradual") == 360
        assert data["compression_factor"].get("rapid") == 1
        assert isinstance(data.get("git_commit"), str) and len(data["git_commit"]) >= 7

    assert total_bytes <= E.MAX_TOTAL_BYTES, (
        f"Total export size {total_bytes} bytes exceeds 2 MB ({E.MAX_TOTAL_BYTES})"
    )


def test_json_schemas_and_model_score_naming() -> None:
    exp_dir = _find_export_dir()

    # 1. summary.json
    summary = json.loads((exp_dir / "summary.json").read_text(encoding="utf-8"))
    assert len(summary["headline_claims"]) >= 7
    for claim in summary["headline_claims"]:
        for req_k in ("id", "title", "number", "source_file", "caveat"):
            assert req_k in claim and claim[req_k], f"summary claim missing {req_k}"
    assert len(summary["limitations"]) >= 5

    # 2. results_*.json
    res_split = json.loads((exp_dir / "results_by_split.json").read_text(encoding="utf-8"))
    assert len(res_split["window_metrics_by_split"]) >= 36
    assert len(res_split["v2_vs_v3_all_snr"]) >= 30

    res_fa = json.loads((exp_dir / "results_false_alarms.json").read_text(encoding="utf-8"))
    assert len(res_fa["false_alarms_table"]) >= 18
    fa_row0 = res_fa["false_alarms_table"][0]
    assert "B_sensor_fa_episodes_per_hour" in fa_row0
    assert "A_slow_fa_episodes_per_hour" in fa_row0

    res_snr = json.loads((exp_dir / "results_recall_vs_snr.json").read_text(encoding="utf-8"))
    assert len(res_snr["gradual_rapid_recall_by_snr_v3"]) >= 40
    assert len(res_snr["gradual_rapid_recall_and_detection_v3"]) >= 24

    res_cmp = json.loads((exp_dir / "results_model_comparison.json").read_text(encoding="utf-8"))
    assert len(res_cmp["rf_vs_threshold_comparison"]) > 0
    assert len(res_cmp["permutation_importance_groups"]) >= 6
    assert len(res_cmp["rf_ablations_v3"]) >= 7

    # 3. robustness.json
    rob = json.loads((exp_dir / "robustness.json").read_text(encoding="utf-8"))
    for req_sec in (
        "yaw_sensitivity",
        "shape_mismatch",
        "literature_checks",
        "timescale_audit",
        "unknown_detector_complement",
    ):
        assert req_sec in rob, f"robustness.json missing {req_sec}"
    assert rob["unknown_detector_complement"]["advisory"] is True

    # 4. parameters.json
    params = json.loads((exp_dir / "parameters.json").read_text(encoding="utf-8"))
    assert len(params["literature_table"]) >= 6
    assert any(r["is_placeholder"] for r in params["literature_table"])
    assert len(params["unsourced_items"]) >= 4

    # 5. replay_<class>.json
    for cls in E.REPLAY_CLASSES:
        rep = json.loads((exp_dir / f"replay_{cls}.json").read_text(encoding="utf-8"))
        assert rep["run_class"] == cls
        assert rep["simulation_id"].startswith(f"sim_test_{cls}_")
        assert isinstance(rep["selection_rule"], str) and "median" in rep["selection_rule"].lower()
        assert len(rep["windows"]) == 95
        w0 = rep["windows"][0]
        for wk in (
            "window_idx",
            "t_start",
            "t_end",
            "true_label",
            "model_score",
            "model_scores_by_class",
            "predicted_class",
            "alarm_state",
            "node_mean_tilt_deg",
            "piezo_rms",
            "lidar_range_change_m",
            "comm_missing_fraction",
            "unknown_anomaly",
        ):
            assert wk in w0, f"replay_{cls}.json window missing {wk}"


def test_no_secrets_and_no_probability_or_confidence_fields() -> None:
    exp_dir = _find_export_dir()
    for fname in EXPECTED_FILES:
        raw_text = (exp_dir / fname).read_text(encoding="utf-8")
        lower_text = raw_text.lower()
        assert "probability" not in lower_text, f"{fname} must not use 'probability'"
        assert "confidence" not in lower_text, f"{fname} must not use 'confidence'"
        for pat in SECRET_PATTERNS:
            assert not pat.search(raw_text), f"Secret-like pattern {pat.pattern} found in {fname}"
