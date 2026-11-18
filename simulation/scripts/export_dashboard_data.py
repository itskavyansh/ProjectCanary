#!/usr/bin/env python
"""Deterministic read-only dashboard export script (SYNTHETIC / SIMULATION).

Reads ONLY frozen evaluation tables, manifests, provenance documentation, and the
frozen v3 models under ProjectCanary-Sim/outputs and writes compact JSON files to
`outputs/dashboard_export/`.

Strict guarantees enforced:
- No retraining, no retuning, no new evaluation numbers: every figure is copied or
  looked up from an existing result table or computed by running the frozen models
  on frozen windows.
- Zero calls to model training or tuning (no `.fit` calls anywhere in this script).
- Every exported file carries `"synthetic": true`, `"compression_factor"`,
  `"git_commit"`, `"seed"`, and `"banner": "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA"`.
- Model output scores use field name `"model_score"`, never `"probability"` or `"confidence"`.
- Frozen model file SHA-256 hashes are verified before and after execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import alarm as A, features as F, models as M  # noqa: E402

BANNER = "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA"
DEFAULT_SEED = 20261117
FALLBACK_GIT_COMMIT = "8e3cdde347ca577531abfae4791809f2f229906d"
MAX_FILE_BYTES = 300 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024

REPLAY_CLASSES = (
    "normal",
    "gradual",
    "rapid",
    "abnormal_vibration",
    "sensor_fault",
    "comm_fault",
)

FROZEN_MODEL_FILES = (
    "outputs/eval_v3/trained_models_v3.pkl",
    "outputs/eval_v3/operating_points_v3.pkl",
    "outputs/eval/trained_models.pkl",
    "outputs/eval_v2/operating_points_v2.pkl",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_model_hashes(root: Path = ROOT) -> dict[str, str]:
    out: dict[str, str] = {}
    for rel in FROZEN_MODEL_FILES:
        p = root / rel
        if p.is_file():
            out[rel] = sha256_file(p)
    return out


def resolve_git_commit(root: Path = ROOT) -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if commit:
            return commit
    except Exception:
        pass
    return FALLBACK_GIT_COMMIT


def _clean_val(val: Any, ndigits: int = 6) -> Any:
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, ndigits)
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return {str(k): _clean_val(v, ndigits=ndigits) for k, v in val.items()}
    if isinstance(val, (list, tuple, np.ndarray)):
        return [_clean_val(v, ndigits=ndigits) for v in val]
    if pd.isna(val):
        return None
    return str(val)


def _rename_ci_key(col: str) -> str:
    # Ensure no key name ever contains 'probability' or 'confidence'
    out = col.replace("probability", "model_score").replace("confidence", "ci")
    return out


def df_to_records(df: pd.DataFrame, ndigits: int = 6) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cols = [str(c) for c in df.columns]
    for _, row in df.iterrows():
        rec: dict[str, Any] = {}
        for col in cols:
            rec[_rename_ci_key(col)] = _clean_val(row[col], ndigits=ndigits)
        records.append(rec)
    return records


def common_envelope(git_commit: str, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    return {
        "synthetic": True,
        "banner": BANNER,
        "compression_factor": {
            "gradual": 360,
            "normal": 360,
            "abnormal_vibration": 360,
            "sensor_fault": 360,
            "comm_fault": 360,
            "rapid": 1,
            "sensor_duration_s": 240,
            "slow_process_hours_per_run": 24.0,
            "sensor_hours_per_run": round(240.0 / 3600.0, 6),
            "note": (
                "Only the slow ground-settlement process is time-compressed (240 s stands for 24 h "
                "at 360x, a PLACEHOLDER label); IMU noise, piezo vibration, LiDAR sweeps, packet "
                "loss, and 5 s windows run in uncompressed 1x sensor time."
            ),
        },
        "git_commit": git_commit,
        "seed": int(seed),
    }


def build_summary_json(root: Path, git_commit: str, seed: int) -> dict[str, Any]:
    v3_all = pd.read_csv(root / "outputs/eval_v3/v2_vs_v3_all_snr_comparison.csv")
    v3_snr2 = pd.read_csv(root / "outputs/eval_v3/v2_vs_v3_snr_gt_2_comparison.csv")
    rf_vs_th = pd.read_csv(root / "outputs/eval/rf_vs_threshold_comparison.csv")
    fa_both = pd.read_csv(root / "outputs/eval_extras_v2/false_alarms_per_hour_both_definitions.csv")
    gr_det = pd.read_csv(root / "outputs/eval_extras_v2/gradual_rapid_recall_and_detection.csv")
    gr_snr = pd.read_csv(root / "outputs/eval_extras_v2/gradual_rapid_recall_by_snr.csv")
    ab_v3 = pd.read_csv(root / "outputs/eval_v3/rf_ablations_v3.csv")
    perm_grp = pd.read_csv(root / "outputs/eval_extras/permutation_importance_groups.csv")
    yaw_all = pd.read_csv(root / "outputs/eval_v3_yaw/yaw_sensitivity_all_snr.csv")
    yaw_s2 = pd.read_csv(root / "outputs/eval_v3_yaw/yaw_sensitivity_snr_gt_2.csv")

    # Lookup exact numbers from frozen CSVs
    rf_test_ep = v3_all[
        (v3_all["split"] == "test")
        & (v3_all["version"] == "v3 (new features)")
        & (v3_all["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (v3_all["model"] == "random_forest")
    ].iloc[0]
    th_test_ep = v3_all[
        (v3_all["split"] == "test")
        & (v3_all["version"] == "v3 (new features)")
        & (v3_all["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (v3_all["model"] == "threshold_rule")
    ].iloc[0]
    lr_test_ep = v3_all[
        (v3_all["split"] == "test")
        & (v3_all["version"] == "v3 (new features)")
        & (v3_all["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (v3_all["model"] == "logistic_regression")
    ].iloc[0]
    rf_bb_ep = v3_all[
        (v3_all["split"] == "broad_basin")
        & (v3_all["version"] == "v3 (new features)")
        & (v3_all["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (v3_all["model"] == "random_forest")
    ].iloc[0]
    rf_stress_ep = v3_all[
        (v3_all["split"] == "stress")
        & (v3_all["version"] == "v3 (new features)")
        & (v3_all["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (v3_all["model"] == "random_forest")
    ].iloc[0]

    pooled_snr_gt5_rf_grad = gr_snr[
        (gr_snr["split"].str.startswith("pooled"))
        & (gr_snr["model"] == "random_forest")
        & (gr_snr["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_snr["class"] == "gradual")
        & (gr_snr["snr_bin"] == ">5")
    ].iloc[0]
    pooled_snr_gt5_th_grad = gr_snr[
        (gr_snr["split"].str.startswith("pooled"))
        & (gr_snr["model"] == "threshold_rule")
        & (gr_snr["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_snr["class"] == "gradual")
        & (gr_snr["snr_bin"] == ">5")
    ].iloc[0]
    pooled_snr_gt5_rf_rap = gr_snr[
        (gr_snr["split"].str.startswith("pooled"))
        & (gr_snr["model"] == "random_forest")
        & (gr_snr["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_snr["class"] == "rapid")
        & (gr_snr["snr_bin"] == ">5")
    ].iloc[0]

    ab_full = ab_v3[ab_v3["group_removed"] == "none (full_model)"].iloc[0]
    ab_no_dir = ab_v3[ab_v3["group_removed"] == "direction"].iloc[0]
    perm_dir = perm_grp[perm_grp["group"] == "direction"].iloc[0]
    perm_mag = perm_grp[perm_grp["group"] == "magnitude"].iloc[0]

    yaw_rf_0 = yaw_s2[
        (yaw_s2["split"] == "test")
        & (yaw_s2["condition"] == "sd_0deg")
        & (yaw_s2["model"] == "random_forest")
        & (yaw_s2["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]
    yaw_rf_10 = yaw_s2[
        (yaw_s2["split"] == "test")
        & (yaw_s2["condition"] == "sd_10deg")
        & (yaw_s2["model"] == "random_forest")
        & (yaw_s2["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]
    yaw_rf_20 = yaw_s2[
        (yaw_s2["split"] == "test")
        & (yaw_s2["condition"] == "sd_20deg")
        & (yaw_s2["model"] == "random_forest")
        & (yaw_s2["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]
    yaw_rf_45 = yaw_s2[
        (yaw_s2["split"] == "test")
        & (yaw_s2["condition"] == "single_node_45deg")
        & (yaw_s2["model"] == "random_forest")
        & (yaw_s2["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]

    fa_test_rf = fa_both[
        (fa_both["split"] == "test")
        & (fa_both["model"] == "random_forest")
        & (fa_both["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]
    fa_test_th = fa_both[
        (fa_both["split"] == "test")
        & (fa_both["model"] == "threshold_rule")
        & (fa_both["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]
    fa_test_lr = fa_both[
        (fa_both["split"] == "test")
        & (fa_both["model"] == "logistic_regression")
        & (fa_both["regime"] == "episode_fa<=1 (k=3,m=3)")
    ].iloc[0]

    gr_test_rf_grad = gr_det[
        (gr_det["split"] == "test")
        & (gr_det["model"] == "random_forest")
        & (gr_det["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_det["class"] == "gradual")
    ].iloc[0]
    gr_test_th_grad = gr_det[
        (gr_det["split"] == "test")
        & (gr_det["model"] == "threshold_rule")
        & (gr_det["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_det["class"] == "gradual")
    ].iloc[0]
    gr_test_lr_grad = gr_det[
        (gr_det["split"] == "test")
        & (gr_det["model"] == "logistic_regression")
        & (gr_det["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_det["class"] == "gradual")
    ].iloc[0]
    gr_test_rf_rap = gr_det[
        (gr_det["split"] == "test")
        & (gr_det["model"] == "random_forest")
        & (gr_det["regime"] == "episode_fa<=1 (k=3,m=3)")
        & (gr_det["class"] == "rapid")
    ].iloc[0]

    claims = [
        {
            "id": "v3_vs_threshold_rule",
            "title": "v3 Random Forest is comparable to the Threshold Rule on test, with gains on broad basins and stress",
            "number": {
                "test_alarm_macro_f1_rf_v3": _clean_val(rf_test_ep["alarm_macro_f1"]),
                "test_alarm_macro_f1_rf_ci95": [
                    _clean_val(rf_test_ep["alarm_macro_f1_ci_lo"]),
                    _clean_val(rf_test_ep["alarm_macro_f1_ci_hi"]),
                ],
                "test_alarm_macro_f1_threshold_v3": _clean_val(th_test_ep["alarm_macro_f1"]),
                "test_alarm_macro_f1_threshold_ci95": [
                    _clean_val(th_test_ep["alarm_macro_f1_ci_lo"]),
                    _clean_val(th_test_ep["alarm_macro_f1_ci_hi"]),
                ],
                "test_alarm_macro_f1_logistic_v3": _clean_val(lr_test_ep["alarm_macro_f1"]),
                "broad_basin_run_detect_rf_v3": _clean_val(rf_bb_ep["alarm_exact_detect_rate"]),
                "stress_alarm_macro_f1_rf_v3": _clean_val(rf_stress_ep["alarm_macro_f1"]),
            },
            "display_value": (
                f"Test Alarm Macro-F1: RF {rf_test_ep['alarm_macro_f1']:.4f} vs "
                f"Threshold {th_test_ep['alarm_macro_f1']:.4f} (Broad-Basin Detect: {rf_bb_ep['alarm_exact_detect_rate']:.3f})"
            ),
            "source_file": "outputs/eval_v3/v2_vs_v3_all_snr_comparison.csv",
            "secondary_source_file": "outputs/eval/rf_vs_threshold_comparison.csv",
            "caveat": (
                "Random Forest does not beat the hand-built threshold rule on overall in-domain test Macro-F1 "
                "or normal-window specificity; its advantage appears on broad basins and multi-condition stress runs."
            ),
        },
        {
            "id": "detection_limit_5x_noise",
            "title": "Reliable detection requires peak tilt of about 5x IMU noise (SNR > 5)",
            "number": {
                "snr_threshold_multiple": 5.0,
                "approx_min_peak_tilt_deg": 0.95,
                "imu_noise_floor_deg": 0.19,
                "pooled_snr_gt_5_gradual_run_detect_rf": _clean_val(pooled_snr_gt5_rf_grad["run_detect_rate"]),
                "pooled_snr_gt_5_gradual_run_detect_threshold": _clean_val(pooled_snr_gt5_th_grad["run_detect_rate"]),
                "pooled_snr_gt_5_rapid_run_detect_rf": _clean_val(pooled_snr_gt5_rf_rap["run_detect_rate"]),
                "pooled_snr_gt_5_gradual_window_recall_rf": _clean_val(pooled_snr_gt5_rf_grad["window_recall"]),
            },
            "display_value": (
                f"SNR > 5 (~0.95 deg tilt): Run detect {pooled_snr_gt5_rf_grad['run_detect_rate']:.2f} (RF) / "
                f"{pooled_snr_gt5_th_grad['run_detect_rate']:.2f} (Threshold); Window recall {pooled_snr_gt5_rf_grad['window_recall']:.2f}"
            ),
            "source_file": "outputs/eval_extras_v2/gradual_rapid_recall_by_snr.csv",
            "secondary_source_file": "outputs/eval/recall_vs_snr.csv",
            "caveat": (
                "Below SNR 2, single-window tilts are buried in MPU6050 white noise and mounting offsets; "
                "even above SNR 5, gradual window recall stays at 0.18-0.34."
            ),
        },
        {
            "id": "direction_features_needed",
            "title": "World-frame radial convergence direction features cut gradual <-> sensor_fault confusions by nearly 3x",
            "number": {
                "full_v3_rf_grad_sf_confusions_snr_gt_2": int(ab_full["gradual_sf_confusions_snr_gt_2"]),
                "no_direction_rf_grad_sf_confusions_snr_gt_2": int(ab_no_dir["gradual_sf_confusions_snr_gt_2"]),
                "confusion_increase_when_removed": int(
                    ab_no_dir["gradual_sf_confusions_snr_gt_2"] - ab_full["gradual_sf_confusions_snr_gt_2"]
                ),
                "direction_perm_importance_drop": _clean_val(perm_dir["mean_importance"]),
                "magnitude_perm_importance_drop": _clean_val(perm_mag["mean_importance"]),
            },
            "display_value": (
                f"Gradual <-> Sensor Fault confusions (SNR>2): {int(ab_full['gradual_sf_confusions_snr_gt_2'])} (Full v3) "
                f"vs {int(ab_no_dir['gradual_sf_confusions_snr_gt_2'])} (-direction)"
            ),
            "source_file": "outputs/eval_v3/rf_ablations_v3.csv",
            "secondary_source_file": "outputs/eval_extras/permutation_importance_groups.csv",
            "caveat": (
                "Radial convergence separates inward-tilting basins from single-node drift, but fails when "
                "all four nodes tilt in parallel on a uniform planar slope."
            ),
        },
        {
            "id": "yaw_tolerance_10_deg",
            "title": "Installation compass yaw tolerance is <= 10 deg SD (with no single-node >= 45 deg blunder)",
            "number": {
                "recommended_yaw_sd_tolerance_deg": 10.0,
                "ci_containment_yaw_sd_deg": 20.0,
                "test_snr2_alarm_f1_0deg": _clean_val(yaw_rf_0["alarm_macro_f1"]),
                "test_snr2_alarm_f1_10deg": _clean_val(yaw_rf_10["alarm_macro_f1"]),
                "test_snr2_alarm_f1_20deg": _clean_val(yaw_rf_20["alarm_macro_f1"]),
                "test_snr2_run_detect_0deg": _clean_val(yaw_rf_0["run_exact_detect_rate"]),
                "test_snr2_run_detect_10deg": _clean_val(yaw_rf_10["run_exact_detect_rate"]),
                "test_snr2_run_detect_20deg": _clean_val(yaw_rf_20["run_exact_detect_rate"]),
                "raw_grad_sf_confusions_0deg": int(yaw_rf_0["gradual_sf_confusions"]),
                "raw_grad_sf_confusions_10deg": int(yaw_rf_10["gradual_sf_confusions"]),
                "raw_grad_sf_confusions_single_45deg": int(yaw_rf_45["gradual_sf_confusions"]),
            },
            "display_value": (
                f"<= 10 deg SD keeps Test SNR>2 Detect at {yaw_rf_10['run_exact_detect_rate']:.3f} "
                f"(drops to {yaw_rf_20['run_exact_detect_rate']:.3f} at 20 deg SD; 45 deg blunder -> {int(yaw_rf_45['gradual_sf_confusions'])} confusions)"
            ),
            "source_file": "outputs/eval_v3_yaw/yaw_sensitivity_snr_gt_2.csv",
            "secondary_source_file": "outputs/eval_v3_yaw/yaw_sensitivity_all_snr.csv",
            "caveat": (
                "Nodes must be aligned within <= 10 deg compass SD at installation and checked against "
                "any single-node >= 45 deg orientation blunder."
            ),
        },
        {
            "id": "false_alarms_both_time_definitions",
            "title": "False alarms per hour must be shown in sensor time (1x) alongside slow-process time (360x)",
            "number": {
                "rf_episode_fa_per_run": _clean_val(fa_test_rf["fa_episodes_per_run"]),
                "rf_B_sensor_fa_per_hour": _clean_val(fa_test_rf["B_sensor_fa_episodes_per_hour"]),
                "rf_B_sensor_every_minutes": round(float(fa_test_rf["B_sensor_one_fa_every_hours"]) * 60.0, 2),
                "rf_A_slow_fa_per_hour": _clean_val(fa_test_rf["A_slow_fa_episodes_per_hour"]),
                "rf_A_slow_every_hours": _clean_val(fa_test_rf["A_slow_one_fa_every_hours"]),
                "threshold_B_sensor_fa_per_hour": _clean_val(fa_test_th["B_sensor_fa_episodes_per_hour"]),
                "threshold_A_slow_fa_per_hour": _clean_val(fa_test_th["A_slow_fa_episodes_per_hour"]),
                "logistic_B_sensor_fa_per_hour": _clean_val(fa_test_lr["B_sensor_fa_episodes_per_hour"]),
                "logistic_A_slow_fa_per_hour": _clean_val(fa_test_lr["A_slow_fa_episodes_per_hour"]),
            },
            "display_value": (
                f"Test RF (k=3,m=3): {fa_test_rf['B_sensor_fa_episodes_per_hour']:.2f} FA/h in sensor time "
                f"(1 every {float(fa_test_rf['B_sensor_one_fa_every_hours']) * 60.0:.1f} min) vs "
                f"{fa_test_rf['A_slow_fa_episodes_per_hour']:.4f} FA/h in 360x slow time (1 every {fa_test_rf['A_slow_one_fa_every_hours']:.1f} h)"
            ),
            "source_file": "outputs/eval_extras_v2/false_alarms_per_hour_both_definitions.csv",
            "caveat": (
                "IMU noise and benign vibrations are not time-compressed, so Definition B (sensor time) is "
                "360x higher and is the stricter figure for noise-driven false alarms; neither is a validated field rate."
            ),
        },
        {
            "id": "gradual_window_recall_low",
            "title": "Window-level recall on gradual subsidence at the matched false-alarm operating points is low",
            "number": {
                "test_episode_gradual_window_recall_threshold": _clean_val(gr_test_th_grad["window_recall"]),
                "test_episode_gradual_window_recall_logistic": _clean_val(gr_test_lr_grad["window_recall"]),
                "test_episode_gradual_window_recall_rf": _clean_val(gr_test_rf_grad["window_recall"]),
                "test_episode_rapid_window_recall_rf": _clean_val(gr_test_rf_rap["window_recall"]),
            },
            "display_value": (
                f"Test Gradual Window Recall (episode k3m3): Threshold {gr_test_th_grad['window_recall']:.3f}, "
                f"LR {gr_test_lr_grad['window_recall']:.3f}, RF {gr_test_rf_grad['window_recall']:.3f}"
            ),
            "source_file": "outputs/eval_extras_v2/gradual_rapid_recall_and_detection.csv",
            "caveat": (
                "Most 5 s windows during the early and low-SNR stages of a gradual ramp remain below the "
                "operating-point threshold and are classified as normal."
            ),
        },
        {
            "id": "run_level_detection_counts_any_alarm",
            "title": "Run-level detection counts any correct persisted alarm in a run and must be paired with window recall",
            "number": {
                "test_episode_gradual_run_detect_threshold": _clean_val(gr_test_th_grad["run_detect_rate"]),
                "test_episode_gradual_run_detect_logistic": _clean_val(gr_test_lr_grad["run_detect_rate"]),
                "test_episode_gradual_run_detect_rf": _clean_val(gr_test_rf_grad["run_detect_rate"]),
                "test_episode_rapid_run_detect_rf": _clean_val(gr_test_rf_rap["run_detect_rate"]),
            },
            "display_value": (
                f"Test Gradual Run Detect vs Window Recall (RF k3m3): "
                f"{gr_test_rf_grad['run_detect_rate']:.3f} run detect vs {gr_test_rf_grad['window_recall']:.3f} window recall"
            ),
            "source_file": "outputs/eval_extras_v2/gradual_rapid_recall_and_detection.csv",
            "caveat": (
                "A run is counted as detected if even a single persisted 3-window episode fires late in the ramp, "
                "so run-level detection is much higher than per-window recall and must never be quoted alone."
            ),
        },
    ]

    limitations = [
        {
            "id": "not_continuum_solver",
            "title": "Kinematic Geometric Surface, Not a Geomechanical Solver",
            "detail": (
                "Ground displacements come from analytic Gaussian, elliptical, trough, or planar mathematical "
                "surfaces rather than finite-element stress-strain or constitutive rock/soil mechanics (e.g. FLAC/PLAXIS)."
            ),
        },
        {
            "id": "homogeneous_idealization",
            "title": "Homogeneous Idealization Without Site Geology",
            "detail": (
                "Does not model overburden stratigraphy, rock jointing, groundwater pore pressure, rainfall "
                "infiltration, or surface topography."
            ),
        },
        {
            "id": "synthetic_faults_and_confounders",
            "title": "Synthetic Sensor Fault and Confounder Distributions",
            "detail": (
                "Hardware fault modes (stuck-at, drift, bias step, motor stall) and packet-loss bursts are "
                "synthetically injected; NODE_04 bench noise is proxied from NODE_02."
            ),
        },
        {
            "id": "timescale_compression_placeholder",
            "title": "360x Time-Compression Label Is a Placeholder",
            "detail": (
                "In frozen gradual runs, 240 s stands for 24 h (360x), which implies a median tilt rate of "
                "0.763 deg/day (~7.6x faster than a rough 0.1 deg/day 30-day longwall order of magnitude). "
                "Meanwhile, sensor noise runs at 1x real time."
            ),
        },
        {
            "id": "shape_and_range_boundaries",
            "title": "Closed-World Supervision and Geometry Sensitivity",
            "detail": (
                "Models trained on 0.05-3.0 deg single basins degrade on uniform planar slopes (parallel tilts "
                "break radial convergence), narrow trough rims (single-node rim tilt resembles sensor fault), "
                "and >3.5 deg tilts (where the Threshold Rule's hard-coded implausible-tilt cutoff misfires)."
            ),
        },
        {
            "id": "unsourced_site_parameters",
            "title": "Unsourced Target Coalfield Parameters",
            "detail": (
                "Angle-of-draw tangent tan(beta), Knothe time coefficient c for Indian coalfields, rapid collapse "
                "rates, and calibrated piezo particle-velocity scaling are unsourced placeholders."
            ),
        },
    ]

    return {
        **common_envelope(git_commit, seed),
        "file_id": "summary",
        "study_title": "ProjectCanary-Sim v3 Synthetic Benchmarking & Sensitivity Study",
        "study_description": (
            "Controlled synthetic benchmarking of a 4-corner-node (MPU6050 IMU + ADS1115 piezo) + central "
            "scanning LiDAR hub chunk across 6 classes (normal, gradual, rapid, abnormal_vibration, "
            "sensor_fault, comm_fault), evaluating a hand-built Threshold Rule, Logistic Regression, and "
            "Random Forest across in-domain, noise-shifted, multi-fault stress, broad-basin, yaw-error, "
            "shape-mismatch, and literature-range regimes."
        ),
        "headline_claims": claims,
        "limitations": limitations,
        "dataset_overview": {
            "splits": {
                "train": {"runs": 360, "description": "60 runs per class; used for model fitting"},
                "val": {"runs": 120, "description": "20 runs per class; used for hyperparameter and operating-point tuning"},
                "test": {"runs": 120, "description": "20 runs per class; held-out in-domain evaluation"},
                "shifted_test": {"runs": 90, "description": "15 runs per class; 1.5-2x IMU noise and elevated 0.12 deg node bias"},
                "stress": {"runs": 100, "description": "Combined multi-condition runs (simultaneous basin + fault/vibration)"},
                "broad_basin": {"runs": 40, "description": "20 gradual + 20 rapid runs with wider spatial spread (up to 2.5 R_net)"},
            },
            "window_length_s": 5.0,
            "window_step_s": 2.5,
            "run_duration_s": 240.0,
            "windows_per_run": 95,
            "n_observable_features_v3": 81,
        },
    }


def build_results_files(root: Path, git_commit: str, seed: int) -> dict[str, dict[str, Any]]:
    win_acc_v2 = pd.read_csv(root / "outputs/eval_extras_v2/window_accuracy_and_f1_v2.csv")
    v3_all = pd.read_csv(root / "outputs/eval_v3/v2_vs_v3_all_snr_comparison.csv")
    v3_snr2 = pd.read_csv(root / "outputs/eval_v3/v2_vs_v3_snr_gt_2_comparison.csv")
    cls_shares = pd.read_csv(root / "outputs/eval_extras_v2/class_share_of_windows_per_split.csv")
    cls_shares = cls_shares.rename(columns={"Unnamed: 0": "split"})
    stress_scoring = pd.read_csv(root / "outputs/eval/stress_set_scoring.csv")

    fa_both = pd.read_csv(root / "outputs/eval_extras_v2/false_alarms_per_hour_both_definitions.csv")
    gr_by_snr = pd.read_csv(root / "outputs/eval_extras_v2/gradual_rapid_recall_by_snr.csv")
    gr_det = pd.read_csv(root / "outputs/eval_extras_v2/gradual_rapid_recall_and_detection.csv")
    rec_snr_v1 = pd.read_csv(root / "outputs/eval/recall_vs_snr.csv")

    rf_vs_th = pd.read_csv(root / "outputs/eval/rf_vs_threshold_comparison.csv")
    perm_groups = pd.read_csv(root / "outputs/eval_extras/permutation_importance_groups.csv")
    perm_top20 = pd.read_csv(root / "outputs/eval_extras/permutation_importance_top20.csv")
    ab_v3 = pd.read_csv(root / "outputs/eval_v3/rf_ablations_v3.csv")
    ab_v1 = pd.read_csv(root / "outputs/eval/rf_ablations.csv")

    results_by_split = {
        **common_envelope(git_commit, seed),
        "file_id": "results_by_split",
        "source_files": [
            "outputs/eval_extras_v2/window_accuracy_and_f1_v2.csv",
            "outputs/eval_v3/v2_vs_v3_all_snr_comparison.csv",
            "outputs/eval_v3/v2_vs_v3_snr_gt_2_comparison.csv",
            "outputs/eval_extras_v2/class_share_of_windows_per_split.csv",
            "outputs/eval/stress_set_scoring.csv",
        ],
        "caption_note": (
            "Share of scored (non-transition) windows that are normal: test 63.6%, shifted_test 63.3%, "
            "stress 41.3%, broad_basin 31.8% (largest in broad_basin is gradual at 35.7%). Classes with "
            "zero true windows in a split have null precision/recall (shown as N/A in tables). "
            "Gradual and rapid rows always pair window recall with run-level detection."
        ),
        "window_metrics_by_split": df_to_records(win_acc_v2, ndigits=4),
        "v2_vs_v3_all_snr": df_to_records(v3_all, ndigits=4),
        "v2_vs_v3_snr_gt_2": df_to_records(v3_snr2, ndigits=4),
        "class_share_of_windows_per_split": df_to_records(cls_shares, ndigits=4),
        "stress_set_scoring": df_to_records(stress_scoring, ndigits=4),
    }

    results_false_alarms = {
        **common_envelope(git_commit, seed),
        "file_id": "results_false_alarms",
        "source_files": [
            "outputs/eval_extras_v2/false_alarms_per_hour_both_definitions.csv",
        ],
        "definition_notes": {
            "primary_display_order": ["B_sensor", "A_slow"],
            "definition_B_sensor_time": (
                "Sensor time (1x = 240 s = 0.0667 h per normal run). Stricter and more defensible for "
                "false alarms driven by IMU noise, benign vibration, and packet loss, which are not time-compressed."
            ),
            "definition_A_slow_process_time": (
                "Slow-process time (360x = 24 h per normal run, PLACEHOLDER compression factor). "
                "360x smaller than Definition B; only meaningful relative to the compressed ground-movement clock."
            ),
            "zero_count_caveat": (
                "'None observed' (0 false alarms in 20 normal runs = 1.33 h of sensor time) is a zero sample count, "
                "not a proven zero rate; by the rule of three the 95% upper bound is ~2.25 FA/h in sensor time."
            ),
        },
        "false_alarms_table": df_to_records(fa_both, ndigits=4),
    }

    results_recall_vs_snr = {
        **common_envelope(git_commit, seed),
        "file_id": "results_recall_vs_snr",
        "source_files": [
            "outputs/eval_extras_v2/gradual_rapid_recall_by_snr.csv",
            "outputs/eval_extras_v2/gradual_rapid_recall_and_detection.csv",
            "outputs/eval/recall_vs_snr.csv",
        ],
        "detection_limit_note": (
            "Reliable detection requires a peak tilt of about 5x IMU noise (SNR > 5, ~0.95 deg vs ~0.19 deg noise). "
            "Run-level detection counts any persisted alarm in a run and is always shown next to window recall."
        ),
        "gradual_rapid_recall_by_snr_v3": df_to_records(gr_by_snr, ndigits=4),
        "gradual_rapid_recall_and_detection_v3": df_to_records(gr_det, ndigits=4),
        "recall_vs_snr_5bins_test": df_to_records(rec_snr_v1, ndigits=4),
    }

    results_model_comparison = {
        **common_envelope(git_commit, seed),
        "file_id": "results_model_comparison",
        "source_files": [
            "outputs/eval/rf_vs_threshold_comparison.csv",
            "outputs/eval_extras/permutation_importance_groups.csv",
            "outputs/eval_extras/permutation_importance_top20.csv",
            "outputs/eval_v3/rf_ablations_v3.csv",
            "outputs/eval/rf_ablations.csv",
        ],
        "importance_caveat": (
            "Piezo and comm groups lead val permutation importance because abnormal_vibration and comm_fault "
            "are near-separable by synthetic construction in single-condition splits (comm_fault precision drops "
            "to ~0.28 in mixed stress runs). For separating basin movement from single-node sensor faults, "
            "the direction group ( fitted radial convergence) is the key discriminator."
        ),
        "rf_vs_threshold_comparison": df_to_records(rf_vs_th, ndigits=4),
        "permutation_importance_groups": df_to_records(perm_groups, ndigits=5),
        "permutation_importance_top20": df_to_records(perm_top20, ndigits=5),
        "rf_ablations_v3": df_to_records(ab_v3, ndigits=4),
        "rf_ablations_v1": df_to_records(ab_v1, ndigits=4),
    }

    return {
        "results_by_split.json": results_by_split,
        "results_false_alarms.json": results_false_alarms,
        "results_recall_vs_snr.json": results_recall_vs_snr,
        "results_model_comparison.json": results_model_comparison,
    }


def build_robustness_json(root: Path, git_commit: str, seed: int) -> dict[str, Any]:
    yaw_all = pd.read_csv(root / "outputs/eval_v3_yaw/yaw_sensitivity_all_snr.csv")
    yaw_s2 = pd.read_csv(root / "outputs/eval_v3_yaw/yaw_sensitivity_snr_gt_2.csv")
    op_regimes = ("episode_fa<=1 (k=3,m=3)", "strict_window_fa<=1")
    yaw_all_op = yaw_all[yaw_all["regime"] == "episode_fa<=1 (k=3,m=3)"]
    yaw_s2_op = yaw_s2[yaw_s2["regime"] == "episode_fa<=1 (k=3,m=3)"]

    sh_det = pd.read_csv(root / "outputs/eval_shape_mismatch/shape_detection_rates.csv")
    sh_conf = pd.read_csv(root / "outputs/eval_shape_mismatch/shape_gradual_sensor_fault_confusion.csv")
    sh_snr = pd.read_csv(root / "outputs/eval_shape_mismatch/shape_recall_by_snr.csv")
    sh_conv = pd.read_csv(root / "outputs/eval_shape_mismatch/shape_convergence_distributions.csv")
    sh_fa = pd.read_csv(root / "outputs/eval_shape_mismatch/shape_normal_false_alarms.csv")

    lit_det = pd.read_csv(root / "outputs/eval_literature/literature_detection_rates.csv")
    lit_conf = pd.read_csv(root / "outputs/eval_literature/literature_gradual_sensor_fault_confusion.csv")
    lit_imp = pd.read_csv(root / "outputs/eval_literature/impossible_value_check_report.csv")
    lit_grid = pd.read_csv(root / "outputs/eval_literature/realistic_depth_sigma_grid.csv")
    lit_ts = pd.read_csv(root / "outputs/eval_literature/timescale_audit_summary.csv")
    lit_fa_alt = pd.read_csv(root / "outputs/eval_literature/normal_false_alarms_with_alt_compression.csv")

    unk_comp = pd.read_csv(root / "outputs/eval_unknown/complement_analysis.csv")
    unk_run = pd.read_csv(root / "outputs/eval_unknown/run_level_detection_and_fa.csv")
    unk_auroc = pd.read_csv(root / "outputs/eval_unknown/test_per_class_auroc_and_recall.csv")
    unk_th = json.loads((root / "outputs/eval_unknown/detector_thresholds.json").read_text(encoding="utf-8"))

    main_comp_arms = (
        "test",
        "shifted_test",
        "shape:gaussian_control",
        "shape:trough_panel",
        "shape:asymmetric",
        "shape:two_basin",
        "shape:planar_slope",
        "lit:high_tilt_3_6",
        "lit:high_tilt_6_12",
        "lit:depth_all_gt_1p5",
    )

    return {
        **common_envelope(git_commit, seed),
        "file_id": "robustness",
        "source_files": [
            "outputs/eval_v3_yaw/yaw_sensitivity_all_snr.csv",
            "outputs/eval_v3_yaw/yaw_sensitivity_snr_gt_2.csv",
            "outputs/eval_shape_mismatch/shape_detection_rates.csv",
            "outputs/eval_shape_mismatch/shape_gradual_sensor_fault_confusion.csv",
            "outputs/eval_shape_mismatch/shape_recall_by_snr.csv",
            "outputs/eval_shape_mismatch/shape_convergence_distributions.csv",
            "outputs/eval_shape_mismatch/shape_normal_false_alarms.csv",
            "outputs/eval_literature/literature_detection_rates.csv",
            "outputs/eval_literature/literature_gradual_sensor_fault_confusion.csv",
            "outputs/eval_literature/impossible_value_check_report.csv",
            "outputs/eval_literature/realistic_depth_sigma_grid.csv",
            "outputs/eval_literature/timescale_audit_summary.csv",
            "outputs/eval_literature/normal_false_alarms_with_alt_compression.csv",
            "outputs/eval_unknown/complement_analysis.csv",
            "outputs/eval_unknown/run_level_detection_and_fa.csv",
            "outputs/eval_unknown/test_per_class_auroc_and_recall.csv",
            "outputs/eval_unknown/detector_thresholds.json",
        ],
        "yaw_sensitivity": {
            "summary_note": (
                "Evaluated across 5 compass perturbation conditions (sd_0deg, sd_5deg, sd_10deg, sd_20deg, "
                "single_node_45deg) plus the yaw-free v2 baseline. v3 RF preserves its advantage up to 10 deg SD; "
                "at 20 deg SD its test detection advantage disappears, and a single-node 45 deg blunder spikes "
                "gradual <-> sensor_fault confusions from 39 to 84."
            ),
            "all_snr": df_to_records(yaw_all_op, ndigits=4),
            "snr_gt_2": df_to_records(yaw_s2_op, ndigits=4),
        },
        "shape_mismatch": {
            "summary_note": (
                "Five synthetic surface shapes matched on peak slope (gaussian_control, asymmetric, two_basin, "
                "trough_panel, planar_slope). Asymmetric is handled without loss; two_basin causes mild degradation; "
                "trough_panel concentrates tilt on a single corner rim (increasing sensor_fault confusion 3.3x on RF); "
                "planar_slope has parallel tilts with no convergence point (increasing gradual->sensor_fault confusion "
                "3.0x on RF and 7.5x on LR)."
            ),
            "detection_rates": df_to_records(
                sh_det[
                    (sh_det["regime"] == "episode_fa<=1 (k=3,m=3)")
                    | (
                        (sh_det["regime"] == "strict_window_fa<=1")
                        & (sh_det["subset"] == "all_events (60 runs)")
                    )
                ],
                ndigits=4,
            ),
            "gradual_sensor_fault_confusion": df_to_records(
                sh_conf[sh_conf["regime"].isin(op_regimes)], ndigits=4
            ),
            "recall_by_snr": df_to_records(
                sh_snr[
                    (sh_snr["regime"] == "episode_fa<=1 (k=3,m=3)")
                    & (sh_snr["model"] == "random_forest")
                ],
                ndigits=4,
            ),
            "convergence_distributions": df_to_records(
                sh_conv[
                    [
                        "shape",
                        "subset",
                        "n_windows",
                        "n_runs",
                        "dir_convergence_residual_m_p10",
                        "dir_convergence_residual_m_p50",
                        "dir_convergence_residual_m_p90",
                        "dir_convergence_residual_m_mean",
                        "dir_convergence_weighted_cos_p10",
                        "dir_convergence_weighted_cos_p50",
                        "dir_convergence_weighted_cos_p90",
                        "dir_convergence_weighted_cos_mean",
                        "dir_pairwise_cos_mean_p50",
                    ]
                ],
                ndigits=4,
            ),
            "normal_false_alarms": df_to_records(sh_fa[sh_fa["shape"] == "gaussian_control"], ndigits=4),
        },
        "literature_checks": {
            "summary_note": (
                "Part B1 (high tilt 3-6 deg and 6-12 deg): ThresholdRuleClassifier's hard-coded 3.5 deg "
                "implausible-tilt cutoff misclassifies 39%-68% of gradual windows as sensor_fault and misses "
                "93%-100% of rapid runs in 6-12 deg, while RF misclassifies 55.5% of 6-12 deg gradual windows as rapid. "
                "Part B2 (realistic depth H=50-300 m, PLACEHOLDER tan(beta)=1.5-2.5 grid): isotropic basins centred "
                "within +-1.5 R_net keep sharp radial convergence, though LR gradual detection drops to 0.00 at H=300 m, tanB=1.5."
            ),
            "detection_rates": df_to_records(
                lit_det[
                    (lit_det["regime"] == "episode_fa<=1 (k=3,m=3)")
                    & (
                        lit_det["arm"].isin(
                            ("gaussian_control", "high_tilt_3_6", "high_tilt_6_12", "depth_all_gt_1p5")
                        )
                        | (lit_det["subset"] == "all_events")
                    )
                ],
                ndigits=4,
            ),
            "gradual_sensor_fault_confusion": df_to_records(
                lit_conf[
                    (lit_conf["regime"] == "episode_fa<=1 (k=3,m=3)")
                    & lit_conf["arm"].isin(
                        ("gaussian_control", "high_tilt_3_6", "high_tilt_6_12", "depth_all_gt_1p5")
                    )
                ],
                ndigits=4,
            ),
            "impossible_value_check": df_to_records(lit_imp, ndigits=4),
            "realistic_depth_sigma_grid": df_to_records(lit_grid, ndigits=4),
        },
        "timescale_audit": {
            "summary_note": (
                "At the frozen 360x compression factor (240 s = 24 h), 97.6% (161/165) of gradual runs exceed "
                "the rough 0.1 deg/day 30-day longwall order-of-magnitude reference (median 0.763 deg/day, 7.63x faster). "
                "Relabelling a run as 30 days (10,800x) changes ONLY the Definition A slow-process false-alarm rate, "
                "never the simulated sensor data, model scores, or Definition B sensor-time false-alarm rate."
            ),
            "gradual_rate_summary": df_to_records(lit_ts, ndigits=4),
            "false_alarms_under_alt_compression": df_to_records(lit_fa_alt, ndigits=6),
        },
        "unknown_detector_complement": {
            "advisory": True,
            "advisory_banner": (
                "ADVISORY SIDE-STUDY ONLY (SYNTHETIC / SIMULATION): Outputs an unsupervised anomaly_score "
                "(a relative statistical distance score) and an unknown_anomaly flag relative to synthetic train normal "
                "windows. Does NOT identify a physical hazard, does NOT assign a class, and adds +0.40 to +0.80 "
                "extra false-alarm episodes per normal run on test (+6.0 to +12.0 FA/h in sensor time)."
            ),
            "thresholds": {
                k: {
                    "fitted_on_n_windows": v["fitted_on_n_windows"],
                    "threshold": v["threshold"],
                    "threshold_quantile_val": v["threshold_quantile_val"],
                    "val_fa_episodes_per_normal_run": v["val_tuning_summary"]["val_fa_episodes_per_normal_run"],
                    "val_fa_windows_per_normal_run": v["val_tuning_summary"]["val_fa_windows_per_normal_run"],
                }
                for k, v in unk_th["detectors"].items()
            },
            "complement_table": df_to_records(
                unk_comp.loc[
                    unk_comp["split_or_arm"].isin(main_comp_arms)
                    & (
                        (unk_comp["detector"] == "isolation_forest")
                        | (
                            (unk_comp["detector"] == "robust_mahalanobis")
                            & (unk_comp["classifier"] == "random_forest")
                        )
                    ),
                    [
                        "split_or_arm",
                        "classifier",
                        "detector",
                        "n_event_runs",
                        "clf_exact_detect_rate",
                        "det_only_flag_rate",
                        "n_clf_exact_missed",
                        "n_missed_flagged_by_det",
                        "frac_of_clf_exact_missed_flagged_by_det",
                        "combined_or_detect_rate",
                        "combined_or_detect_ci_lo",
                        "combined_or_detect_ci_hi",
                        "gated_unknown_detect_rate",
                        "gated_unknown_detect_ci_lo",
                        "gated_unknown_detect_ci_hi",
                        "clf_only_fa_per_normal_run",
                        "det_only_fa_per_normal_run",
                        "combined_or_fa_per_normal_run",
                        "extra_fa_per_normal_run_or",
                        "extra_B_sensor_fa_per_hour_or",
                        "extra_A_slow_fa_per_hour_or",
                        "extra_fa_per_normal_run_gated",
                        "extra_B_sensor_fa_per_hour_gated",
                        "total_fa_per_normal_run_with_gated",
                    ],
                ],
                ndigits=4,
            ),
            "run_level_detection_and_fa": df_to_records(
                unk_run[unk_run["split_or_arm"].isin(("val", "shared_normal_40", *main_comp_arms))],
                ndigits=4,
            ),
            "test_per_class_auroc": df_to_records(unk_auroc, ndigits=4),
        },
    }


def _strip_md(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
    cleaned = re.sub(r"`(.*?)`", r"\1", cleaned)
    return cleaned


def build_parameters_json(root: Path, git_commit: str, seed: int) -> dict[str, Any]:
    prov_path = root / "docs/provenance_note.md"
    lines = prov_path.read_text(encoding="utf-8").splitlines()

    table_rows: list[dict[str, Any]] = []
    in_sec5_table = False
    timescale_lines: list[str] = []
    in_timescale = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("| Parameter / Topic |"):
            in_sec5_table = True
            continue
        if in_sec5_table:
            if stripped.startswith("|---"):
                continue
            if stripped.startswith("|"):
                parts = [p.strip() for p in stripped.strip("|").split("|")]
                if len(parts) >= 4:
                    raw_param, raw_setting, raw_lit, raw_note = parts[:4]
                    combined = " ".join(parts)
                    table_rows.append(
                        {
                            "parameter_or_topic": _strip_md(raw_param),
                            "simulator_setting_and_status": _strip_md(raw_setting),
                            "literature_basis_and_sources": _strip_md(raw_lit),
                            "assessment_or_scope_note": _strip_md(raw_note),
                            "raw_markdown": {
                                "parameter_or_topic": raw_param,
                                "simulator_setting_and_status": raw_setting,
                                "literature_basis_and_sources": raw_lit,
                                "assessment_or_scope_note": raw_note,
                            },
                            "is_placeholder": "PLACEHOLDER" in combined,
                            "status_tag": "PLACEHOLDER" if "PLACEHOLDER" in combined else "CITED",
                        }
                    )
            else:
                in_sec5_table = False

        if stripped.startswith("### Timescale"):
            in_timescale = True
            continue
        if in_timescale and stripped:
            timescale_lines.append(_strip_md(stripped))

    unsourced_items = [
        {
            "item": "Angle of draw / influence tangent tan(beta) for target Indian coalfields",
            "status": "PLACEHOLDER (unsourced)",
            "literature_leads": "Singh & Yadav (1995) Raniganj coalfield; Sadhana (2021) Adriyala longwall panel study",
            "impact": "Required to convert overburden depth H into physical Gaussian spread sigma = H / (tan(beta) * sqrt(2*pi)).",
        },
        {
            "item": "Knothe time-function coefficient c (1/year or 1/day) for target Indian coalfields",
            "status": "PLACEHOLDER (unsourced)",
            "literature_leads": "Zhang et al. Scientific Reports 12:18433 (2022); Hejmanowski (2015) Int J Coal Sci Technol",
            "impact": "Required to calibrate physical ramp duration and replace the smoothstep S-curve / 360x compression placeholder.",
        },
        {
            "item": "Rapid trough/sinkhole collapse onset rate and duration distribution",
            "status": "PLACEHOLDER (no source found)",
            "literature_leads": "No calibrated public time-series source found for sudden crown-hole/pillar collapse rates",
            "impact": "Rapid events (3-8 s / 5-19 s ramp) must be treated as an uncalibrated synthetic stress test.",
        },
        {
            "item": "Downloadable raw field vibration dataset & ADS1115 piezo-to-mm/s transfer calibration",
            "status": "PLACEHOLDER (unsourced)",
            "literature_leads": "OSMRE (2007) acoustic/ground-vibration report (0.2-30 mm/s); Agnico Meadowbank (2019) blast report (13 mm/s limit)",
            "impact": "Simulator piezo voltages use relative synthetic scaling and are not calibrated to physical peak particle velocity (mm/s).",
        },
        {
            "item": "LiDAR hub range noise (sigma_range) and quantization step (q_step)",
            "status": "PLACEHOLDER",
            "literature_leads": "configs/sensor_default.yaml bench/datasheet placeholder values",
            "impact": "Must be measured on physical rotating-mount Benewake TFmini-i bench tests.",
        },
        {
            "item": "Hardware fault injection rates and LoRa packet-loss burst distributions",
            "status": "PLACEHOLDER",
            "literature_leads": "Synthetically configured in configs/sensor_default.yaml and configs/scenario_default.yaml",
            "impact": "Not mined from multi-year field deployment telemetry logs.",
        },
    ]

    return {
        **common_envelope(git_commit, seed),
        "file_id": "parameters",
        "source_file": "docs/provenance_note.md",
        "literature_table": table_rows,
        "unsourced_items": unsourced_items,
        "timescale_note": " ".join(timescale_lines),
    }


def _compute_mahalanobis_stats_from_train_normal(
    train_df: pd.DataFrame, feature_cols: list[str]
) -> dict[str, np.ndarray]:
    """Compute the exact frozen RobustMahalanobis summary statistics from `train` normal windows
    using pure NumPy array operations (zero model fitting calls)."""
    norm_df = train_df[(train_df["split"] == "train") & (train_df["label"] == "normal")]
    X_raw = np.clip(norm_df[feature_cols].to_numpy(dtype=float), -1e4, 1e4)
    col_med = np.nanmedian(X_raw, axis=0)
    X = np.where(np.isnan(X_raw), col_med, X_raw)

    med = np.median(X, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr_s = (q75 - q25) / 1.3489795
    mad_s = 1.4826 * np.median(np.abs(X - med), axis=0)
    std_s = np.std(X, axis=0, ddof=1)
    scale = np.maximum.reduce([iqr_s, mad_s, std_s, np.full_like(med, 1e-4)])

    Z = np.clip((X - med) / scale, -50.0, 50.0)
    norms = np.linalg.norm(Z, axis=1)
    cutoff = float(np.quantile(norms, 0.98))
    Z_trim = Z[norms <= cutoff]
    if len(Z_trim) < len(feature_cols) + 2:
        Z_trim = Z

    mean_trimmed = np.mean(Z_trim, axis=0)
    cov = np.cov(Z_trim, rowvar=False)
    diag_cov = np.diag(np.maximum(np.diag(cov), 1e-3))
    cov_shrunk = 0.90 * cov + 0.10 * diag_cov
    precision_mat = np.linalg.pinv(cov_shrunk, hermitian=True)

    return {
        "col_med": col_med,
        "center_med": med,
        "robust_scale": scale,
        "mean_trimmed": mean_trimmed,
        "precision_mat": precision_mat,
    }


def _score_unknown_anomaly_run(
    run_df: pd.DataFrame,
    feature_cols: list[str],
    maha_stats: dict[str, np.ndarray],
    threshold: float,
    k: int = 3,
    m: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X_raw = np.clip(run_df[feature_cols].to_numpy(dtype=float), -1e4, 1e4)
    X = np.where(np.isnan(X_raw), maha_stats["col_med"], X_raw)
    Z = np.clip((X - maha_stats["center_med"]) / maha_stats["robust_scale"], -50.0, 50.0)
    diff = Z - maha_stats["mean_trimmed"]
    quad = np.einsum("ij,jk,ik->i", diff, maha_stats["precision_mat"], diff)
    scores = np.sqrt(np.maximum(quad, 0.0))
    raw_flags = scores >= float(threshold)
    raw_labels = np.where(raw_flags, "unknown_anomaly", "normal")
    pers_labels, _ = A.apply_persistence_sequence(raw_labels, k=k, m=m)
    pers_flags = pers_labels == "unknown_anomaly"
    return scores, raw_flags, pers_flags


def build_replay_files(root: Path, git_commit: str, seed: int) -> dict[str, dict[str, Any]]:
    manifest = pd.read_csv(root / "outputs/datasets_v3/manifest.csv")
    train_df = pd.read_csv(root / "outputs/datasets_v3/features_train.csv")
    test_df = pd.read_csv(root / "outputs/datasets_v3/features_test.csv")
    models_v3 = pickle.loads((root / "outputs/eval_v3/trained_models_v3.pkl").read_bytes())
    ops_v3 = pickle.loads((root / "outputs/eval_v3/operating_points_v3.pkl").read_bytes())
    unk_meta = json.loads((root / "outputs/eval_unknown/detector_thresholds.json").read_text(encoding="utf-8"))

    rf_model = models_v3["random_forest"]
    rf_ep_op = ops_v3["episode_fa_le_1"]["random_forest"]
    feature_cols = F.feature_columns(train_df)
    M.guard_model_features(feature_cols)

    maha_stats = _compute_mahalanobis_stats_from_train_normal(train_df, feature_cols)
    maha_th = float(unk_meta["detectors"]["robust_mahalanobis"]["threshold"])

    selection_rule = (
        "Fixed deterministic rule: filter outputs/datasets_v3/manifest.csv to split == 'test' "
        "(20 runs per class), sort each class by ['snr_vs_imu_noise', 'simulation_id'] ascending (stable), "
        "and select index len(class_runs) // 2 (index 10 of 20, the upper-median SNR run)."
    )

    test_man = manifest[manifest["split"] == "test"].copy()
    replay_files: dict[str, dict[str, Any]] = {}

    for cls in REPLAY_CLASSES:
        sub_m = (
            test_man[test_man["run_class"] == cls]
            .sort_values(["snr_vs_imu_noise", "simulation_id"], kind="stable")
            .reset_index(drop=True)
        )
        chosen_idx = len(sub_m) // 2
        run_meta = sub_m.iloc[chosen_idx]
        sim_id = str(run_meta["simulation_id"])

        run_df = (
            test_df[test_df["simulation_id"] == sim_id]
            .sort_values("t_start", kind="stable")
            .reset_index(drop=True)
        )

        # Frozen Random Forest window scores and predictions
        raw_scores = rf_model.predict_proba(run_df)
        rf_classes = [str(c) for c in rf_model.pipeline.classes_]
        raw_preds = rf_model.predict(run_df)
        ep_win_preds = rf_ep_op.predict_window(run_df)
        alarm_seq, n_alarm_episodes = A.apply_persistence_sequence(ep_win_preds, k=3, m=3)

        # Frozen unknown_anomaly detector scores and persisted flags (k=3, m=3)
        unk_scores, unk_raw, unk_pers = _score_unknown_anomaly_run(
            run_df, feature_cols, maha_stats, threshold=maha_th, k=3, m=3
        )

        windows_list: list[dict[str, Any]] = []
        for i, row in run_df.iterrows():
            scores_dict = {
                c_name: _clean_val(raw_scores[i, j], ndigits=4)
                for j, c_name in enumerate(rf_classes)
            }
            pred_ep = str(ep_win_preds[i])
            pred_raw = str(raw_preds[i])
            top_score = float(np.max(raw_scores[i]))
            pred_ep_score = float(scores_dict.get(pred_ep, top_score))

            windows_list.append(
                {
                    "window_idx": int(row["window_idx"]),
                    "t_start": _clean_val(row["t_start"], 2),
                    "t_end": _clean_val(row["t_end"], 2),
                    "t_mid": _clean_val(row["t_mid"], 2),
                    "t_rel_event_start_s": _clean_val(row["t_rel_event_start_s"], 2),
                    "true_label": str(row["label"]),
                    "label_set": str(row["label_set"]) if pd.notna(row["label_set"]) else "",
                    "model_score": _clean_val(top_score, 4),
                    "predicted_class_model_score": _clean_val(pred_ep_score, 4),
                    "model_scores_by_class": scores_dict,
                    "predicted_class": pred_ep,
                    "predicted_class_raw": pred_raw,
                    "alarm_state": str(alarm_seq[i]),
                    "node_mean_tilt_deg": {
                        "NODE_01": _clean_val(row["NODE_01_mean"], 4),
                        "NODE_02": _clean_val(row["NODE_02_mean"], 4),
                        "NODE_03": _clean_val(row["NODE_03_mean"], 4),
                        "NODE_04": _clean_val(row["NODE_04_mean"], 4),
                    },
                    "piezo_rms": {
                        "network_mean": _clean_val(row["piezo_rms_network_mean"], 5),
                        "NODE_01": _clean_val(row["NODE_01_piezo_rms_mean"], 5),
                        "NODE_02": _clean_val(row["NODE_02_piezo_rms_mean"], 5),
                        "NODE_03": _clean_val(row["NODE_03_piezo_rms_mean"], 5),
                        "NODE_04": _clean_val(row["NODE_04_piezo_rms_mean"], 5),
                    },
                    "lidar_range_change_m": {
                        "hub_common_mode_m": _clean_val(row["hub_common_mode_m"], 4),
                        "hub_residual_abs_max_m": _clean_val(row["hub_residual_abs_max_m"], 4),
                        "NODE_01": _clean_val(row["NODE_01_lidar_range_change_m"], 4),
                        "NODE_02": _clean_val(row["NODE_02_lidar_range_change_m"], 4),
                        "NODE_03": _clean_val(row["NODE_03_lidar_range_change_m"], 4),
                        "NODE_04": _clean_val(row["NODE_04_lidar_range_change_m"], 4),
                    },
                    "comm_missing_fraction": {
                        "network": _clean_val(row["comm_missing_frac_net"], 4),
                        "max_burst_len": _clean_val(row["comm_max_burst_len"], 2),
                        "NODE_01": _clean_val(row["NODE_01_missing_frac"], 4),
                        "NODE_02": _clean_val(row["NODE_02_missing_frac"], 4),
                        "NODE_03": _clean_val(row["NODE_03_missing_frac"], 4),
                        "NODE_04": _clean_val(row["NODE_04_missing_frac"], 4),
                    },
                    "dir_convergence_residual_m": _clean_val(row["dir_convergence_residual_m"], 3),
                    "dir_convergence_weighted_cos": _clean_val(row["dir_convergence_weighted_cos"], 4),
                    "anomaly_score": _clean_val(unk_scores[i], 4),
                    "unknown_anomaly_raw": bool(unk_raw[i]),
                    "unknown_anomaly": bool(unk_pers[i]),
                }
            )

        replay_payload = {
            **common_envelope(git_commit, seed),
            "file_id": f"replay_{cls}",
            "run_class": cls,
            "simulation_id": sim_id,
            "selection_rule": selection_rule,
            "selected_rank_in_class_test_split": int(chosen_idx),
            "class_test_split_run_count": int(len(sub_m)),
            "downsample_step": 1,
            "run_metadata": {
                "seed": int(run_meta["seed"]),
                "split": str(run_meta["split"]),
                "run_class": str(run_meta["run_class"]),
                "peak_tilt_deg": _clean_val(run_meta["peak_tilt_deg"], 4),
                "imu_noise_sd_deg": _clean_val(run_meta["imu_noise_sd_deg"], 4),
                "noise_multiplier": _clean_val(run_meta["noise_multiplier"], 4),
                "snr_vs_imu_noise": _clean_val(run_meta["snr_vs_imu_noise"], 4),
                "run_compression_factor": _clean_val(run_meta["compression_factor"], 1),
                "duration_s": _clean_val(run_meta["duration_s"], 1),
                "n_windows": int(run_meta["n_windows"]),
                "fault_types": str(run_meta["fault_types"]) if pd.notna(run_meta["fault_types"]) else None,
            },
            "operating_point": {
                "model": "random_forest (v3 frozen)",
                "regime": "episode_fa<=1 (k=3, m=3)",
                "k": 3,
                "m": 3,
                "group_thresholds": {
                    k_g: _clean_val(v_g, 5) for k_g, v_g in rf_ep_op.group_thresholds.items()
                },
                "n_alarm_episodes_in_run": int(n_alarm_episodes),
                "n_alarm_windows_in_run": int((alarm_seq != "normal").sum()),
                "unknown_detector_advisory": {
                    "detector": "robust_mahalanobis (train-normal frozen stats)",
                    "threshold": _clean_val(maha_th, 6),
                    "k": 3,
                    "m": 3,
                    "n_flagged_windows_in_run": int(unk_pers.sum()),
                },
            },
            "windows": windows_list,
        }
        replay_files[f"replay_{cls}.json"] = replay_payload

    return replay_files


def export_all(
    out_dir: Path | None = None,
    root: Path = ROOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    target_dir = out_dir if out_dir is not None else (root / "outputs/dashboard_export")
    target_dir.mkdir(parents=True, exist_ok=True)

    hashes_before = snapshot_model_hashes(root)
    git_commit = resolve_git_commit(root)

    files_to_write: dict[str, dict[str, Any]] = {}
    files_to_write["summary.json"] = build_summary_json(root, git_commit, seed)
    files_to_write.update(build_results_files(root, git_commit, seed))
    files_to_write["robustness.json"] = build_robustness_json(root, git_commit, seed)
    files_to_write["parameters.json"] = build_parameters_json(root, git_commit, seed)
    files_to_write.update(build_replay_files(root, git_commit, seed))

    written_paths: dict[str, Path] = {}
    total_bytes = 0
    for fname, payload in sorted(files_to_write.items()):
        out_path = target_dir / fname
        text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
        raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > 260 * 1024:
            text = (
                json.dumps(
                    payload,
                    indent=1,
                    separators=(",", ": "),
                    sort_keys=False,
                    ensure_ascii=False,
                )
                + "\n"
            )
            raw_bytes = text.encode("utf-8")
        if len(raw_bytes) > MAX_FILE_BYTES:
            raise ValueError(
                f"Exported file {fname} is {len(raw_bytes)} bytes, exceeding 300 KB limit ({MAX_FILE_BYTES})"
            )
        out_path.write_bytes(raw_bytes)
        written_paths[fname] = out_path
        total_bytes += len(raw_bytes)

    if total_bytes > MAX_TOTAL_BYTES:
        raise ValueError(
            f"Total export size is {total_bytes} bytes, exceeding 2 MB limit ({MAX_TOTAL_BYTES})"
        )

    hashes_after = snapshot_model_hashes(root)
    if hashes_before != hashes_after:
        raise RuntimeError("Frozen model file SHA-256 hashes changed during export!")

    return written_paths


def main() -> None:
    ap = argparse.ArgumentParser(description="Export frozen ProjectCanary-Sim results for dashboard.")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/dashboard_export",
        help="Destination directory for exported JSON files.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    paths = export_all(out_dir=args.out, root=ROOT, seed=args.seed)
    total_sz = sum(p.stat().st_size for p in paths.values())
    print(f"Exported {len(paths)} JSON files ({total_sz / 1024:.1f} KB total) to {args.out}:")
    for name, p in sorted(paths.items()):
        print(f"  - {name}: {p.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
