#!/usr/bin/env python
"""Deterministic evaluation sample export for the advisory AI explainer (SYNTHETIC / SIMULATION).

Reads ONLY frozen v3 models (`outputs/eval_v3/trained_models_v3.pkl`,
`outputs/eval_v3/operating_points_v3.pkl`, `outputs/eval_unknown/detector_thresholds.json`)
and the frozen `test` split (`outputs/datasets_v3/features_test.csv`, `manifest.csv`,
 plus `features_train.csv` for the frozen RobustMahalanobis normal reference statistics).

Strict guarantees enforced:
- Fixed seed, no retraining, zero model fitting calls anywhere in this script.
- Frozen model SHA-256 hashes verified before and after execution.
- Anonymised run_id (`run_test_000` .. `run_test_119`) and window_id (`<run_id>_w<idx>`)
  so the packet contains no real-world identity and leaks no true class name in the run ID.
- Stratified sample of 150 windows across 7 documented strata:
  1) `correct_alarm` (30 windows: 6 per non-normal class: gradual, rapid, abnormal_vibration,
     sensor_fault, comm_fault where predicted_class == alarm_state == true_label)
  2) `missed_event` (24 windows: 6 gradual, 6 rapid, 6 sensor_fault, 6 comm_fault where
     true_label != 'normal' and predicted_class == 'normal' and alarm_state == 'normal')
  3) `false_alarm` (20 windows: normal runs where predicted_class != 'normal' or alarm_state != 'normal')
  4) `classifier_mistake` (26 windows: event/fault windows where predicted_class != 'normal'
     and predicted_class != true_label, including gradual->sensor_fault, gradual->rapid,
     rapid->gradual, rapid->sensor_fault, sensor_fault->gradual, sensor_fault->rapid, etc.)
  5) `unknown_flag_normal` (15 windows: normal windows where unknown_anomaly == True)
  6) `unknown_flag_event` (15 windows: event/fault windows where unknown_anomaly == True)
  7) `fault_window` (20 windows: 10 comm_fault and 10 sensor_fault windows)
- Ground-truth fields (`true_label`, `run_class`, `stratum`, `simulation_id`, etc.) are stored
  ONLY inside the separate top-level `"truth"` dictionary (keyed by `window_id`), which is
  NEVER shown to the explainer and is used only by the evaluation script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import alarm as A, evaluate as EV, features as F, models as M  # noqa: E402

BANNER = "SYNTHETIC / SIMULATION - NOT VALIDATED ON REAL DATA"
DEFAULT_SEED = 20261123
TARGET_TOTAL_PACKETS = 150
FALLBACK_SIM_ROOT = Path(r"D:\Github Projects\ProjectCanary-Sim")

FROZEN_MODEL_FILES = (
    "outputs/eval_v3/trained_models_v3.pkl",
    "outputs/eval_v3/operating_points_v3.pkl",
    "outputs/eval/trained_models.pkl",
    "outputs/eval_v2/operating_points_v2.pkl",
    "outputs/eval_unknown/detector_thresholds.json",
)


def resolve_data_root(root: Path = ROOT) -> Path:
    if (root / "outputs/datasets_v3/features_test.csv").is_file():
        return root
    env_root = os.getenv("PROJECTCANARY_SIM_ROOT")
    if env_root and (Path(env_root) / "outputs/datasets_v3/features_test.csv").is_file():
        return Path(env_root)
    if (FALLBACK_SIM_ROOT / "outputs/datasets_v3/features_test.csv").is_file():
        return FALLBACK_SIM_ROOT
    return root


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_model_hashes(root: Path = ROOT) -> dict[str, str]:
    data_root = resolve_data_root(root)
    out: dict[str, str] = {}
    for rel in FROZEN_MODEL_FILES:
        p = data_root / rel
        if p.is_file():
            out[rel] = sha256_file(p)
    return out


def _clean_num(val: Any, ndigits: int = 4) -> float:
    f = float(val)
    if math.isnan(f) or math.isinf(f):
        return 0.0
    return round(f, ndigits)


def snr_to_bin(snr_val: float) -> str:
    f = float(snr_val)
    for b_lbl, lo, hi in EV.SNR_BINS:
        if lo <= f < hi:
            return b_lbl
    return ">5" if f >= 5.0 else "<0.5"


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


def _sample_indices(
    pool_indices: np.ndarray,
    n_pick: int,
    used: set[int],
    rng: np.random.Generator,
) -> list[int]:
    avail = [int(i) for i in pool_indices if int(i) not in used]
    if len(avail) <= n_pick:
        chosen = sorted(avail)
    else:
        picked = rng.choice(np.array(avail, dtype=int), size=n_pick, replace=False)
        chosen = sorted(int(x) for x in picked)
    used.update(chosen)
    return chosen


def build_explainer_sample(
    root: Path = ROOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    data_root = resolve_data_root(root)
    hashes_before = snapshot_model_hashes(data_root)

    manifest = pd.read_csv(data_root / "outputs/datasets_v3/manifest.csv")
    train_df = pd.read_csv(data_root / "outputs/datasets_v3/features_train.csv")
    test_df = pd.read_csv(data_root / "outputs/datasets_v3/features_test.csv")
    models_v3 = pickle.loads((data_root / "outputs/eval_v3/trained_models_v3.pkl").read_bytes())
    ops_v3 = pickle.loads((data_root / "outputs/eval_v3/operating_points_v3.pkl").read_bytes())
    unk_meta = json.loads((data_root / "outputs/eval_unknown/detector_thresholds.json").read_text(encoding="utf-8"))

    rf_model = models_v3["random_forest"]
    th_model = models_v3["threshold_rule"]
    rf_ep_op = ops_v3["episode_fa_le_1"]["random_forest"]
    feature_cols = F.feature_columns(train_df)
    M.guard_model_features(feature_cols)

    maha_stats = _compute_mahalanobis_stats_from_train_normal(train_df, feature_cols)
    maha_th = float(unk_meta["detectors"]["robust_mahalanobis"]["threshold"])

    test_df = test_df.sort_values(["simulation_id", "t_start"], kind="stable").reset_index(drop=True)

    # Map simulation_id to anonymised run_id (sorted deterministically by seed, simulation_id)
    test_man = (
        manifest[manifest["split"] == "test"]
        .sort_values(["seed", "simulation_id"], kind="stable")
        .reset_index(drop=True)
    )
    sim_to_anon: dict[str, str] = {
        str(row["simulation_id"]): f"run_test_{idx:03d}"
        for idx, row in test_man.iterrows()
    }
    sim_to_snr: dict[str, float] = {
        str(row["simulation_id"]): float(row["snr_vs_imu_noise"])
        for _, row in test_man.iterrows()
    }
    sim_to_fault: dict[str, str | None] = {
        str(row["simulation_id"]): (str(row["fault_types"]) if pd.notna(row["fault_types"]) else None)
        for _, row in test_man.iterrows()
    }

    # Frozen predictions and scores
    raw_scores = rf_model.predict_proba(test_df)
    rf_classes = [str(c) for c in rf_model.pipeline.classes_]
    cls_to_j = {c: j for j, c in enumerate(rf_classes)}
    raw_preds = rf_model.predict(test_df)
    ep_preds = rf_ep_op.predict_window(test_df)
    th_preds = th_model.predict(test_df)
    alarm_seq, _ = A.apply_persistence_df(test_df, ep_preds, k=3, m=3)

    unk_scores = np.zeros(len(test_df), dtype=float)
    unk_raw = np.zeros(len(test_df), dtype=bool)
    unk_pers = np.zeros(len(test_df), dtype=bool)
    for _, g in test_df.groupby("simulation_id", sort=False):
        idx = g.index.to_numpy()
        sc, rw, pr = _score_unknown_anomaly_run(g, feature_cols, maha_stats, threshold=maha_th, k=3, m=3)
        unk_scores[idx] = sc
        unk_raw[idx] = rw
        unk_pers[idx] = pr

    test_df["_ep_pred"] = ep_preds
    test_df["_raw_pred"] = raw_preds
    test_df["_th_pred"] = th_preds
    test_df["_alarm"] = alarm_seq
    test_df["_unk_pers"] = unk_pers
    test_df["_unk_score"] = unk_scores

    # Exclude transition windows from evaluation sampling
    valid = test_df[test_df["label"] != "transition"].copy()
    rng = np.random.default_rng(int(seed))
    used: set[int] = set()
    selected_rows: list[tuple[int, str, str]] = []  # (df_idx, stratum, substratum)

    # 1. correct_alarm (30 windows: 6 per non-normal class)
    event_classes = ("gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault")
    for cls in event_classes:
        pool = valid[
            (valid["label"] == cls)
            & (valid["_ep_pred"] == cls)
            & (valid["_alarm"] == cls)
        ].index.to_numpy()
        picks = _sample_indices(pool, 6, used, rng)
        for idx in picks:
            selected_rows.append((idx, "correct_alarm", f"correct_alarm_{cls}"))

    # 2. missed_event (24 windows: 6 gradual, 6 rapid, 6 sensor_fault, 6 comm_fault)
    for cls in ("gradual", "rapid", "sensor_fault", "comm_fault"):
        pool = valid[
            (valid["label"] == cls)
            & (valid["_ep_pred"] == "normal")
            & (valid["_alarm"] == "normal")
        ].index.to_numpy()
        picks = _sample_indices(pool, 6, used, rng)
        for idx in picks:
            selected_rows.append((idx, "missed_event", f"missed_{cls}"))

    # 3. false_alarm on normal runs (20 windows: 12 persisted alarm_state != normal, 8 window predicted_class != normal)
    pool_fa_pers = valid[
        (valid["run_class"] == "normal")
        & (valid["label"] == "normal")
        & (valid["_alarm"] != "normal")
    ].index.to_numpy()
    for idx in _sample_indices(pool_fa_pers, 12, used, rng):
        selected_rows.append((idx, "false_alarm", "false_alarm_persisted_on_normal_run"))

    pool_fa_win = valid[
        (valid["run_class"] == "normal")
        & (valid["label"] == "normal")
        & (valid["_ep_pred"] != "normal")
    ].index.to_numpy()
    for idx in _sample_indices(pool_fa_win, 8, used, rng):
        selected_rows.append((idx, "false_alarm", "false_alarm_window_on_normal_run"))

    # 4. classifier_mistake (26 windows: true_label != normal & predicted_class != normal & predicted_class != true_label)
    mistake_pairs = (
        ("gradual", "sensor_fault", 6),
        ("gradual", "rapid", 5),
        ("rapid", "gradual", 5),
        ("rapid", "sensor_fault", 4),
        ("sensor_fault", "gradual", 3),
        ("sensor_fault", "rapid", 3),
    )
    for t_cls, p_cls, k_pick in mistake_pairs:
        pool = valid[
            (valid["label"] == t_cls)
            & (valid["_ep_pred"] == p_cls)
        ].index.to_numpy()
        picks = _sample_indices(pool, k_pick, used, rng)
        for idx in picks:
            selected_rows.append((idx, "classifier_mistake", f"mistake_{t_cls}_as_{p_cls}"))

    # 5. unknown_flag_normal (15 windows: label == normal & unknown_anomaly == True)
    pool_unk_norm = valid[
        (valid["label"] == "normal")
        & (valid["_unk_pers"])
    ].index.to_numpy()
    for idx in _sample_indices(pool_unk_norm, 15, used, rng):
        selected_rows.append((idx, "unknown_flag_normal", "unknown_anomaly_on_normal"))

    # 6. unknown_flag_event (15 windows: label != normal & unknown_anomaly == True, 3 per event class)
    for cls in event_classes:
        pool_unk_ev = valid[
            (valid["label"] == cls)
            & (valid["_unk_pers"])
        ].index.to_numpy()
        for idx in _sample_indices(pool_unk_ev, 3, used, rng):
            selected_rows.append((idx, "unknown_flag_event", f"unknown_anomaly_on_{cls}"))

    # 7. fault_window (20 windows: 10 comm_fault and 10 sensor_fault windows)
    for cls in ("comm_fault", "sensor_fault"):
        pool_flt = valid[valid["label"] == cls].index.to_numpy()
        for idx in _sample_indices(pool_flt, 10, used, rng):
            selected_rows.append((idx, "fault_window", f"fault_window_{cls}"))

    assert len(selected_rows) == TARGET_TOTAL_PACKETS, (
        f"Expected {TARGET_TOTAL_PACKETS} packets, got {len(selected_rows)}"
    )

    packets: list[dict[str, Any]] = []
    truth_by_window: dict[str, dict[str, Any]] = {}
    strata_counts: dict[str, int] = {}
    substrata_counts: dict[str, int] = {}

    for df_idx, stratum, substratum in selected_rows:
        row = test_df.loc[df_idx]
        sim_id = str(row["simulation_id"])
        anon_run_id = sim_to_anon[sim_id]
        win_idx = int(row["window_idx"])
        window_id = f"{anon_run_id}_w{win_idx:02d}"

        pred_cls = str(row["_ep_pred"])
        pred_j = cls_to_j[pred_cls]
        pred_score = _clean_num(raw_scores[df_idx, pred_j], 4)
        snr_val = sim_to_snr[sim_id]

        # Packet contains ONLY the strictly permitted observable/model fields (NO ground-truth label)
        packet = {
            "window_id": window_id,
            "run_id": anon_run_id,
            "time_s": {
                "t_start": _clean_num(row["t_start"], 2),
                "t_end": _clean_num(row["t_end"], 2),
                "t_mid": _clean_num(row["t_mid"], 2),
            },
            "predicted_class": pred_cls,
            "model_score": pred_score,
            "alarm_state": str(row["_alarm"]),
            "threshold_rule_class": str(row["_th_pred"]),
            "unknown_anomaly": bool(row["_unk_pers"]),
            "anomaly_score": _clean_num(row["_unk_score"], 4),
            "node_mean_tilt_deg": {
                "NODE_01": _clean_num(row["NODE_01_mean"], 4),
                "NODE_02": _clean_num(row["NODE_02_mean"], 4),
                "NODE_03": _clean_num(row["NODE_03_mean"], 4),
                "NODE_04": _clean_num(row["NODE_04_mean"], 4),
            },
            "piezo_rms": {
                "network_mean": _clean_num(row["piezo_rms_network_mean"], 5),
                "NODE_01": _clean_num(row["NODE_01_piezo_rms_mean"], 5),
                "NODE_02": _clean_num(row["NODE_02_piezo_rms_mean"], 5),
                "NODE_03": _clean_num(row["NODE_03_piezo_rms_mean"], 5),
                "NODE_04": _clean_num(row["NODE_04_piezo_rms_mean"], 5),
            },
            "lidar_range_change_m": {
                "hub_common_mode_m": _clean_num(row["hub_common_mode_m"], 4),
                "NODE_01": _clean_num(row["NODE_01_lidar_range_change_m"], 4),
                "NODE_02": _clean_num(row["NODE_02_lidar_range_change_m"], 4),
                "NODE_03": _clean_num(row["NODE_03_lidar_range_change_m"], 4),
                "NODE_04": _clean_num(row["NODE_04_lidar_range_change_m"], 4),
            },
            "comm": {
                "missing_fraction_network": _clean_num(row["comm_missing_frac_net"], 4),
                "NODE_01_missing_fraction": _clean_num(row["NODE_01_missing_frac"], 4),
                "NODE_02_missing_fraction": _clean_num(row["NODE_02_missing_frac"], 4),
                "NODE_03_missing_fraction": _clean_num(row["NODE_03_missing_frac"], 4),
                "NODE_04_missing_fraction": _clean_num(row["NODE_04_missing_frac"], 4),
                "latency_mean_s": _clean_num(row["comm_latency_mean_s"], 4),
                "latency_max_s": _clean_num(row["comm_latency_max_s"], 4),
            },
            "radial_convergence_residual_m": _clean_num(row["dir_convergence_residual_m"], 3),
            "radial_convergence_weighted_cos": _clean_num(row["dir_convergence_weighted_cos"], 4),
            "snr_vs_imu_noise_bin": snr_to_bin(snr_val),
        }
        packets.append(packet)

        truth_by_window[window_id] = {
            "window_id": window_id,
            "run_id": anon_run_id,
            "simulation_id": sim_id,
            "window_idx": win_idx,
            "true_label": str(row["label"]),
            "run_class": str(row["run_class"]),
            "stratum": stratum,
            "substratum": substratum,
            "snr_vs_imu_noise": _clean_num(snr_val, 4),
            "fault_types": sim_to_fault[sim_id],
            "is_classifier_mistake": bool(
                str(row["label"]) != "normal"
                and pred_cls != "normal"
                and pred_cls != str(row["label"])
            ),
        }

        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
        substrata_counts[substratum] = substrata_counts.get(substratum, 0) + 1

    hashes_after = snapshot_model_hashes(data_root)
    if hashes_before != hashes_after:
        raise RuntimeError("Frozen model file SHA-256 hashes changed during explainer sample export!")

    return {
        "synthetic": True,
        "banner": BANNER,
        "seed": int(seed),
        "split": "test",
        "operating_point": "episode_fa<=1 (k=3, m=3)",
        "n_packets": len(packets),
        "strata_documentation": {
            "correct_alarm": "30 windows (6 per non-normal class) where frozen RF predicted_class == alarm_state == true_label.",
            "missed_event": "24 windows (6 gradual, 6 rapid, 6 sensor_fault, 6 comm_fault) where true_label != normal and RF predicted_class == alarm_state == normal.",
            "false_alarm": "20 windows on normal runs where RF predicted_class != normal (8) or persisted alarm_state != normal (12).",
            "classifier_mistake": "26 windows on event/fault runs where RF predicted_class != normal and predicted_class != true_label (gradual<->sensor_fault, gradual<->rapid, rapid->sensor_fault, sensor_fault->rapid).",
            "unknown_flag_normal": "15 normal windows where persisted RobustMahalanobis unknown_anomaly == True.",
            "unknown_flag_event": "15 event/fault windows (3 per event class) where persisted unknown_anomaly == True.",
            "fault_window": "20 hardware/network fault windows (10 comm_fault and 10 sensor_fault).",
        },
        "strata_counts": strata_counts,
        "substrata_counts": substrata_counts,
        "frozen_model_sha256": hashes_after,
        "packets": packets,
        "truth": truth_by_window,
    }


def export_explainer_sample(
    out_path: Path | None = None,
    root: Path = ROOT,
    seed: int = DEFAULT_SEED,
) -> Path:
    data_root = resolve_data_root(root)
    target = out_path if out_path is not None else (data_root / "outputs/explainer_sample/packets.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = build_explainer_sample(root=data_root, seed=seed)
    text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    target.write_bytes(text.encode("utf-8"))
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description="Export stratified evaluation packet sample for the advisory explainer.")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination path for packets.json.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    out_file = export_explainer_sample(out_path=args.out, root=ROOT, seed=args.seed)
    data = json.loads(out_file.read_text(encoding="utf-8"))
    print(f"Exported {data['n_packets']} packets to {out_file} ({out_file.stat().st_size / 1024:.1f} KB)")
    for k, v in data["strata_counts"].items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
