"""Tests for scripts/export_explainer_sample.py (SYNTHETIC / SIMULATION).

Verifies:
1. No `.fit(` or `fit_transform(` calls appear anywhere in `scripts/export_explainer_sample.py`.
2. Fixed seed produces byte-identical `packets.json` output across repeated runs.
3. Frozen model SHA-256 hashes are unchanged before and after running the export.
4. Exactly 150 stratified packets across all 7 required strata (`correct_alarm`,
   `missed_event`, `false_alarm`, `classifier_mistake`, `unknown_flag_normal`,
   `unknown_flag_event`, `fault_window`).
5. Each packet contains ONLY the permitted observable/classifier fields (never ground truth,
   never `"probability"` or `"confidence"`, and no real-world identity or true class leaked in `run_id`).
6. Ground truth is stored separately in `"truth"` keyed by `window_id`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import export_explainer_sample as ES  # noqa: E402

ALLOWED_PACKET_KEYS = {
    "window_id",
    "run_id",
    "time_s",
    "predicted_class",
    "model_score",
    "alarm_state",
    "threshold_rule_class",
    "unknown_anomaly",
    "anomaly_score",
    "node_mean_tilt_deg",
    "piezo_rms",
    "lidar_range_change_m",
    "comm",
    "radial_convergence_residual_m",
    "radial_convergence_weighted_cos",
    "snr_vs_imu_noise_bin",
}

FORBIDDEN_PACKET_KEYS = {
    "truth",
    "true_label",
    "label",
    "run_class",
    "stratum",
    "substratum",
    "simulation_id",
    "probability",
    "confidence",
}


def _has_frozen_data() -> bool:
    data_root = ES.resolve_data_root(ROOT) if hasattr(ES, "resolve_data_root") else ROOT
    return (
        (data_root / "outputs/datasets_v3/features_test.csv").is_file()
        and (data_root / "outputs/eval_v3/trained_models_v3.pkl").is_file()
        and (data_root / "outputs/eval_v3/operating_points_v3.pkl").is_file()
    )


def test_no_fit_calls_in_export_explainer_sample() -> None:
    script_path = ROOT / "scripts/export_explainer_sample.py"
    assert script_path.is_file(), "Missing scripts/export_explainer_sample.py"
    src = script_path.read_text(encoding="utf-8")
    assert ".fit(" not in src, "export_explainer_sample.py must not contain any .fit( call"
    assert "fit_transform(" not in src
    assert "fit_random_forest(" not in src
    assert "fit_logistic_regression(" not in src


def test_deterministic_export_and_frozen_hashes_unchanged(tmp_path: Path) -> None:
    if not _has_frozen_data():
        pytest.skip("Frozen v3 models/datasets not available in this checkout.")

    data_root = ES.resolve_data_root(ROOT) if hasattr(ES, "resolve_data_root") else ROOT
    hashes_before = ES.snapshot_model_hashes(data_root)
    assert len(hashes_before) >= 2

    out1 = tmp_path / "run1" / "packets.json"
    out2 = tmp_path / "run2" / "packets.json"
    ES.export_explainer_sample(out_path=out1, root=data_root, seed=20261123)
    ES.export_explainer_sample(out_path=out2, root=data_root, seed=20261123)

    hashes_after = ES.snapshot_model_hashes(data_root)
    assert hashes_before == hashes_after, "Frozen model SHA-256 hashes changed during export!"

    h1 = hashlib.sha256(out1.read_bytes()).hexdigest()
    h2 = hashlib.sha256(out2.read_bytes()).hexdigest()
    assert h1 == h2, f"Fixed seed must give identical output: {h1} != {h2}"

    data = json.loads(out1.read_text(encoding="utf-8"))
    assert data["n_packets"] == 150
    assert len(data["packets"]) == 150
    assert len(data["truth"]) == 150

    expected_strata = {
        "correct_alarm": 30,
        "missed_event": 24,
        "false_alarm": 20,
        "classifier_mistake": 26,
        "unknown_flag_normal": 15,
        "unknown_flag_event": 15,
        "fault_window": 20,
    }
    assert data["strata_counts"] == expected_strata

    for pkt in data["packets"]:
        assert set(pkt.keys()) == ALLOWED_PACKET_KEYS
        assert not (set(pkt.keys()) & FORBIDDEN_PACKET_KEYS)
        pkt_str = json.dumps(pkt).lower()
        assert "probability" not in pkt_str
        assert "confidence" not in pkt_str
        # Verify anonymised run_id does not leak class name
        for cls_name in ("normal", "gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault"):
            assert cls_name not in pkt["run_id"]
            assert cls_name not in pkt["window_id"]
        assert pkt["window_id"] in data["truth"]
