"""Evaluation Extras (SYNTHETIC / SIMULATION).

Contains:
1. Conversion of normal-run false alarm counts to false alarms per simulated hour using metadata compression factor.
2. Window accuracy alongside Macro-F1 and per-class precision and recall.
3. Grouped-by-run permutation feature importance for Random Forest strictly on VAL.
4. Uncalibrated model_score reliability tables and calibration diagrams on VAL.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import alarm as A, classes as C, evaluate as ev, features as F, labels as L

CLASS_LIST = list(C.CLASSES)

FEATURE_GROUPS: dict[str, list[str]] = {
    "direction": [
        "NODE_01_tilt_angle_rad", "NODE_02_tilt_angle_rad", "NODE_03_tilt_angle_rad", "NODE_04_tilt_angle_rad",
        "dir_circular_spread", "dir_pairwise_cos_mean", "dir_pairwise_cos_min",
        "dir_convergence_align_cos", "dir_convergence_residual_m", "dir_convergence_weighted_cos",
        "dir_win_convergence_align_cos", "dir_win_convergence_residual_m", "dir_win_convergence_weighted_cos",
    ],
    "coherence": [
        "coherence", "coherence_signed_mean", "coherence_signed_tx", "coherence_signed_ty",
        "coherence_trailing_mag", "coherence_trailing_signed_mean", "coherence_trailing_wx", "coherence_trailing_wy",
    ],
    "magnitude": [
        "NODE_01_mean", "NODE_01_std", "NODE_01_slope",
        "NODE_02_mean", "NODE_02_std", "NODE_02_slope",
        "NODE_03_mean", "NODE_03_std", "NODE_03_slope",
        "NODE_04_mean", "NODE_04_std", "NODE_04_slope",
    ],
    "piezo": [
        "NODE_01_piezo_impulse_count", "NODE_01_piezo_p2p_max", "NODE_01_piezo_peak_max", "NODE_01_piezo_rms_mean",
        "NODE_02_piezo_impulse_count", "NODE_02_piezo_p2p_max", "NODE_02_piezo_peak_max", "NODE_02_piezo_rms_mean",
        "NODE_03_piezo_impulse_count", "NODE_03_piezo_p2p_max", "NODE_03_piezo_peak_max", "NODE_03_piezo_rms_mean",
        "NODE_04_piezo_impulse_count", "NODE_04_piezo_p2p_max", "NODE_04_piezo_peak_max", "NODE_04_piezo_rms_mean",
        "piezo_impulse_total", "piezo_nodes_with_impulse", "piezo_rms_network_mean",
    ],
    "hub": [
        "NODE_01_lidar_range_change_m", "NODE_01_lidar_residual_m", "NODE_01_lidar_staleness_s",
        "NODE_02_lidar_range_change_m", "NODE_02_lidar_residual_m", "NODE_02_lidar_staleness_s",
        "NODE_03_lidar_range_change_m", "NODE_03_lidar_residual_m", "NODE_03_lidar_staleness_s",
        "NODE_04_lidar_range_change_m", "NODE_04_lidar_residual_m", "NODE_04_lidar_staleness_s",
        "hub_common_mode_m", "hub_residual_abs_max_m", "hub_staleness_max_s",
    ],
    "comm": [
        "NODE_01_missing_frac", "NODE_02_missing_frac", "NODE_03_missing_frac", "NODE_04_missing_frac",
        "comm_latency_max_s", "comm_latency_mean_s", "comm_max_burst_len",
        "comm_missing_frac_net", "comm_nodes_unavailable",
    ],
    "long_horizon": [
        "NODE_01_long_slope_deg_s", "NODE_02_long_slope_deg_s",
        "NODE_03_long_slope_deg_s", "NODE_04_long_slope_deg_s",
        "long_slope_max_deg_s",
    ],
}


# -----------------------------------------------------------------------------
# 1. False alarms per simulated hour
# -----------------------------------------------------------------------------
def calculate_simulated_hours_per_run(duration_s: float, compression_factor: float) -> float:
    """Calculate simulated physical hours represented by a run.

    simulated_hours = (duration_s * compression_factor) / 3600.0
    For normal runs: (240.0 * 360.0) / 3600.0 = 24.0 hours per run.
    """
    if duration_s <= 0 or compression_factor <= 0:
        raise ValueError(f"duration_s and compression_factor must be positive, got {duration_s}, {compression_factor}")
    return float((duration_s * compression_factor) / 3600.0)


def evaluate_alarm_level_with_hours(
    df_all_windows: pd.DataFrame,
    raw_preds: np.ndarray,
    manifest_df: pd.DataFrame,
    k: int = 3,
    m: int = 2,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict[str, Any]:
    """Evaluate alarm metrics including false alarms per simulated hour with 95% run-bootstrap CIs."""
    # Pre-merge compression factor and duration into run index
    man_map = manifest_df.set_index("simulation_id")[["compression_factor", "duration_s", "run_class"]].to_dict(orient="index")

    alarm_preds, run_stats = A.apply_persistence_df(df_all_windows, raw_preds, k=k, m=m)
    d = df_all_windows.copy()
    d["_alarm"] = alarm_preds
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])

    records = []
    for sim_id, g in d.groupby("simulation_id", sort=False):
        sid = str(sim_id)
        g = g.sort_values("t_start", kind="stable")
        r_cls = str(g["run_class"].iloc[0])
        st = run_stats[sid]
        meta_info = man_map.get(sid, {"compression_factor": 360.0, "duration_s": 240.0})
        comp = float(meta_info["compression_factor"])
        dur = float(meta_info["duration_s"])
        sim_hrs = calculate_simulated_hours_per_run(dur, comp)

        if r_cls == "normal":
            records.append({
                "simulation_id": sid,
                "run_class": "normal",
                "is_event_run": False,
                "compression_factor": comp,
                "simulated_hours": sim_hrs,
                "fa_episodes": float(st["n_alarm_episodes"]),
                "fa_windows": float(st["n_alarm_windows"]),
                "fa_episodes_per_hour": float(st["n_alarm_episodes"]) / sim_hrs,
                "fa_windows_per_hour": float(st["n_alarm_windows"]) / sim_hrs,
                "exact_detected": np.nan,
                "missed_event": np.nan,
            })
        else:
            t_rel = g["t_rel_event_start_s"].to_numpy(dtype=float)
            al_arr = g["_alarm"].to_numpy(dtype=object)
            post_mask = t_rel > -win_half + 1e-9
            al_post = al_arr[post_mask]

            if (al_post == r_cls).any():
                ex_det = 1.0
            else:
                ls_post = g["label_set"].fillna("").to_numpy(dtype=object)[post_mask]
                ex_matches = [
                    i for i, (pr, ls) in enumerate(zip(al_post, ls_post))
                    if pr != "normal" and (pr == r_cls or (ls and str(pr) in str(ls).split("|")))
                ]
                ex_det = 1.0 if len(ex_matches) > 0 else 0.0

            records.append({
                "simulation_id": sid,
                "run_class": r_cls,
                "is_event_run": True,
                "compression_factor": comp,
                "simulated_hours": sim_hrs,
                "fa_episodes": np.nan,
                "fa_windows": np.nan,
                "fa_episodes_per_hour": np.nan,
                "fa_windows_per_hour": np.nan,
                "exact_detected": ex_det,
                "missed_event": 1.0 - ex_det,
            })

    rdf = pd.DataFrame(records)
    ev_runs = rdf[rdf["is_event_run"]]
    norm_runs = rdf[~rdf["is_event_run"]]

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 802]))

    def _boot_stats(arr: np.ndarray) -> tuple[float, tuple[float, float]]:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, (0.0, 0.0)
        pt = float(np.mean(arr))
        if n_boot <= 0 or len(arr) == 1:
            return pt, (pt, pt)
        b_means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(n_boot)]
        return pt, (float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5)))

    # Compute stats on normal runs
    n_norm = len(norm_runs)
    if n_norm > 0:
        comp_factors = norm_runs["compression_factor"].unique()
        comp_str = "|".join(str(c) for c in sorted(comp_factors))
        hrs_per_run = float(norm_runs["simulated_hours"].iloc[0]) if len(comp_factors) == 1 else np.nan

        pt_eps_run, ci_eps_run = _boot_stats(norm_runs["fa_episodes"].to_numpy())
        pt_eps_hr, ci_eps_hr = _boot_stats(norm_runs["fa_episodes_per_hour"].to_numpy())
        pt_win_run, ci_win_run = _boot_stats(norm_runs["fa_windows"].to_numpy())
        pt_win_hr, ci_win_hr = _boot_stats(norm_runs["fa_windows_per_hour"].to_numpy())
    else:
        comp_str = "N/A"
        hrs_per_run = np.nan
        pt_eps_run, ci_eps_run = np.nan, (np.nan, np.nan)
        pt_eps_hr, ci_eps_hr = np.nan, (np.nan, np.nan)
        pt_win_run, ci_win_run = np.nan, (np.nan, np.nan)
        pt_win_hr, ci_win_hr = np.nan, (np.nan, np.nan)

    # Compute stats on event runs
    n_ev = len(ev_runs)
    if n_ev > 0:
        pt_ex, ci_ex = _boot_stats(ev_runs["exact_detected"].to_numpy())
        pt_miss, ci_miss = _boot_stats(ev_runs["missed_event"].to_numpy())
    else:
        pt_ex, ci_ex = np.nan, (np.nan, np.nan)
        pt_miss, ci_miss = np.nan, (np.nan, np.nan)

    return {
        "n_normal_runs": n_norm,
        "n_event_runs": n_ev,
        "compression_factor": comp_str,
        "simulated_hours_per_run": hrs_per_run,
        "fa_episodes_per_run": pt_eps_run,
        "fa_episodes_per_run_ci95": ci_eps_run,
        "fa_episodes_per_hour": pt_eps_hr,
        "fa_episodes_per_hour_ci95": ci_eps_hr,
        "fa_windows_per_run": pt_win_run,
        "fa_windows_per_run_ci95": ci_win_run,
        "fa_windows_per_hour": pt_win_hr,
        "fa_windows_per_hour_ci95": ci_win_hr,
        "exact_detect_rate": pt_ex,
        "exact_detect_ci95": ci_ex,
        "missed_event_rate": pt_miss,
        "missed_event_ci95": ci_miss,
    }


def generate_false_alarm_per_hour_table(
    datasets: dict[str, pd.DataFrame],
    models: dict[str, Any],
    ops: dict[str, Any],
    manifest_df: pd.DataFrame,
    splits: tuple[str, ...] = ("test", "shifted_test", "val", "train"),
    n_boot: int = 300,
    seed: int = 20261117,
) -> pd.DataFrame:
    """Generate table of false alarms per run and per simulated hour alongside missed-event rate."""
    rows = []
    pers_cfg = {
        "raw": {"k": 1, "m": 1},
        "episode_fa_le_1": {"k": 3, "m": 3},
        "strict_window_fa_le_1": {"threshold_rule": {"k": 5, "m": 2}, "default": {"k": 3, "m": 2}},
    }

    for sp in splits:
        df = datasets[sp]
        for mname in ("threshold_rule", "logistic_regression", "random_forest"):
            model = models[mname]
            raw_preds = model.predict(df)

            regimes = [
                ("raw (k=1)", raw_preds, pers_cfg["raw"]["k"], pers_cfg["raw"]["m"]),
                ("episode_fa<=1 (k=3,m=3)",
                 raw_preds if mname == "threshold_rule" else ops["episode_fa_le_1"][mname].predict_window(df),
                 pers_cfg["episode_fa_le_1"]["k"], pers_cfg["episode_fa_le_1"]["m"]),
                ("strict_window_fa<=1",
                 raw_preds if mname == "threshold_rule" else ops["strict_window_fa_le_1"][mname].predict_window(df),
                 pers_cfg["strict_window_fa_le_1"]["threshold_rule"]["k"] if mname == "threshold_rule" else pers_cfg["strict_window_fa_le_1"]["default"]["k"],
                 pers_cfg["strict_window_fa_le_1"]["threshold_rule"]["m"] if mname == "threshold_rule" else pers_cfg["strict_window_fa_le_1"]["default"]["m"]),
            ]

            for rname, w_preds, k, m in regimes:
                res = evaluate_alarm_level_with_hours(
                    df, w_preds, manifest_df=manifest_df, k=k, m=m, n_boot=n_boot, seed=seed,
                )
                rows.append({
                    "split": sp,
                    "model": mname,
                    "regime": rname,
                    "k": k,
                    "m": m,
                    "n_normal_runs": res["n_normal_runs"],
                    "n_event_runs": res["n_event_runs"],
                    "compression_factor": res["compression_factor"],
                    "simulated_hours_per_normal_run": res["simulated_hours_per_run"],
                    "fa_episodes_per_run": round(res["fa_episodes_per_run"], 4),
                    "fa_episodes_per_run_ci_lo": round(res["fa_episodes_per_run_ci95"][0], 4),
                    "fa_episodes_per_run_ci_hi": round(res["fa_episodes_per_run_ci95"][1], 4),
                    "fa_episodes_per_sim_hour": round(res["fa_episodes_per_hour"], 5),
                    "fa_episodes_per_sim_hour_ci_lo": round(res["fa_episodes_per_hour_ci95"][0], 5),
                    "fa_episodes_per_sim_hour_ci_hi": round(res["fa_episodes_per_hour_ci95"][1], 5),
                    "fa_windows_per_run": round(res["fa_windows_per_run"], 4),
                    "fa_windows_per_run_ci_lo": round(res["fa_windows_per_run_ci95"][0], 4),
                    "fa_windows_per_run_ci_hi": round(res["fa_windows_per_run_ci95"][1], 4),
                    "fa_windows_per_sim_hour": round(res["fa_windows_per_hour"], 5),
                    "fa_windows_per_sim_hour_ci_lo": round(res["fa_windows_per_hour_ci95"][0], 5),
                    "fa_windows_per_sim_hour_ci_hi": round(res["fa_windows_per_hour_ci95"][1], 5),
                    "exact_detect_rate": round(res["exact_detect_rate"], 4),
                    "exact_detect_ci_lo": round(res["exact_detect_ci95"][0], 4),
                    "exact_detect_ci_hi": round(res["exact_detect_ci95"][1], 4),
                    "missed_event_rate": round(res["missed_event_rate"], 4),
                    "missed_event_ci_lo": round(res["missed_event_ci95"][0], 4),
                    "missed_event_ci_hi": round(res["missed_event_ci95"][1], 4),
                })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 2. Accuracy alongside Macro-F1 and per-class precision and recall
# -----------------------------------------------------------------------------
def evaluate_window_level_accuracy_and_f1(
    df: pd.DataFrame,
    preds_col_or_arr: np.ndarray,
    n_boot: int = 300,
    seed: int = 20261117,
    classes: list[str] = CLASS_LIST,
) -> dict[str, Any]:
    """Evaluate window accuracy, Macro-F1, and per-class precision & recall with run-bootstrap CIs."""
    mask = L.trainable(df).to_numpy()
    sub = df.loc[mask].copy()
    preds = np.asarray(preds_col_or_arr)[mask] if not isinstance(preds_col_or_arr, str) else sub[preds_col_or_arr].to_numpy()
    y_true = sub["label"].to_numpy()

    cm = ev.confusion_matrix_from_labels(y_true, preds, classes)
    point = ev.metrics_from_cm(cm, classes)
    total_w = int(cm.sum())
    point_acc = float(np.trace(cm) / total_w) if total_w > 0 else 0.0

    run_ids = sub["simulation_id"].unique()
    n_runs = len(run_ids)
    k = len(classes)
    cm_by_run = np.zeros((n_runs, k, k), dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}
    ti_all = np.array([idx_map[v] for v in y_true], dtype=int)
    pi_all = np.array([idx_map[v] for v in preds], dtype=int)

    run_to_r = {rid: r for r, rid in enumerate(run_ids)}
    r_codes = np.array([run_to_r[rid] for rid in sub["simulation_id"]], dtype=int)
    np.add.at(cm_by_run, (r_codes, ti_all, pi_all), 1)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 702]))
    boot_acc = np.zeros(n_boot, dtype=float)
    boot_macro_f1 = np.zeros(n_boot, dtype=float)
    boot_prec = np.zeros((n_boot, k), dtype=float)
    boot_rec = np.zeros((n_boot, k), dtype=float)

    for b in range(n_boot):
        pick = rng.integers(0, n_runs, size=n_runs)
        cm_b = cm_by_run[pick].sum(axis=0)
        tot_b = cm_b.sum()
        boot_acc[b] = float(np.trace(cm_b) / tot_b) if tot_b > 0 else 0.0
        m_b = ev.metrics_from_cm(cm_b, classes)
        boot_macro_f1[b] = m_b["macro_f1"]
        boot_prec[b] = m_b["prec_arr"]
        boot_rec[b] = m_b["rec_arr"]

    acc_ci = (float(np.percentile(boot_acc, 2.5)), float(np.percentile(boot_acc, 97.5)))
    macro_ci = (float(np.percentile(boot_macro_f1, 2.5)), float(np.percentile(boot_macro_f1, 97.5)))

    per_class_res = {}
    for i, c in enumerate(classes):
        per_class_res[c] = {
            "precision": point["per_class"][c]["precision"],
            "precision_ci95": (float(np.percentile(boot_prec[:, i], 2.5)), float(np.percentile(boot_prec[:, i], 97.5))),
            "recall": point["per_class"][c]["recall"],
            "recall_ci95": (float(np.percentile(boot_rec[:, i], 2.5)), float(np.percentile(boot_rec[:, i], 97.5))),
            "support": point["per_class"][c]["support"],
        }

    return {
        "n_windows": total_w,
        "n_runs": n_runs,
        "accuracy": point_acc,
        "accuracy_ci95": acc_ci,
        "macro_f1": point["macro_f1"],
        "macro_f1_ci95": macro_ci,
        "per_class": per_class_res,
    }


def generate_accuracy_and_f1_table(
    datasets: dict[str, pd.DataFrame],
    models: dict[str, Any],
    ops: dict[str, Any],
    splits: tuple[str, ...] = ("test", "shifted_test", "stress", "broad_basin"),
    n_boot: int = 300,
    seed: int = 20261117,
) -> pd.DataFrame:
    """Generate comprehensive table of Window Accuracy, Macro-F1, and per-class precision/recall."""
    rows = []
    pers_cfg = {
        "raw": {"k": 1, "m": 1},
        "episode_fa_le_1": {"k": 3, "m": 3},
        "strict_window_fa_le_1": {"threshold_rule": {"k": 5, "m": 2}, "default": {"k": 3, "m": 2}},
    }

    for sp in splits:
        df = datasets[sp]
        for mname in ("threshold_rule", "logistic_regression", "random_forest"):
            model = models[mname]
            raw_preds = model.predict(df)

            regimes = [
                ("raw (k=1)", raw_preds),
                ("episode_fa<=1 (k=3,m=3)",
                 raw_preds if mname == "threshold_rule" else ops["episode_fa_le_1"][mname].predict_window(df)),
                ("strict_window_fa<=1",
                 raw_preds if mname == "threshold_rule" else ops["strict_window_fa_le_1"][mname].predict_window(df)),
            ]

            for rname, w_preds in regimes:
                res = evaluate_window_level_accuracy_and_f1(df, w_preds, n_boot=n_boot, seed=seed)
                row = {
                    "split": sp,
                    "model": mname,
                    "regime": rname,
                    "n_windows": res["n_windows"],
                    "n_runs": res["n_runs"],
                    "window_accuracy": round(res["accuracy"], 4),
                    "window_accuracy_ci_lo": round(res["accuracy_ci95"][0], 4),
                    "window_accuracy_ci_hi": round(res["accuracy_ci95"][1], 4),
                    "window_macro_f1": round(res["macro_f1"], 4),
                    "window_macro_f1_ci_lo": round(res["macro_f1_ci95"][0], 4),
                    "window_macro_f1_ci_hi": round(res["macro_f1_ci95"][1], 4),
                }
                for c in CLASS_LIST:
                    pc = res["per_class"][c]
                    row[f"prec_{c}"] = round(pc["precision"], 4)
                    row[f"prec_{c}_ci_lo"] = round(pc["precision_ci95"][0], 4)
                    row[f"prec_{c}_ci_hi"] = round(pc["precision_ci95"][1], 4)
                    row[f"rec_{c}"] = round(pc["recall"], 4)
                    row[f"rec_{c}_ci_lo"] = round(pc["recall_ci95"][0], 4)
                    row[f"rec_{c}_ci_hi"] = round(pc["recall_ci95"][1], 4)
                    row[f"supp_{c}"] = pc["support"]
                rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 3. Permutation feature importance on VAL only (grouped by run)
# -----------------------------------------------------------------------------
def compute_grouped_permutation_importance(
    model: Any,
    val_df: pd.DataFrame,
    features_to_permute: list[str] | dict[str, list[str]],
    n_repeats: int = 10,
    seed: int = 20261117,
) -> dict[str, dict[str, float]]:
    """Compute permutation importance grouped by whole runs on VAL only.

    Permutes feature values across runs (all windows of run A replaced with corresponding
    window values of run B), preserving within-run temporal autocorrelation.
    """
    # Guard: must be VAL split only
    split_vals = val_df["split"].unique()
    if len(split_vals) != 1 or split_vals[0] != "val":
        raise ValueError(f"Permutation importance must be computed on VAL split only, got: {split_vals}")

    trainable_mask = L.trainable(val_df).to_numpy()
    val_clean = val_df.copy()
    y_true_trainable = val_clean.loc[trainable_mask, "label"].to_numpy()

    # Baseline Macro-F1
    base_preds = model.predict(val_clean)
    base_cm = ev.confusion_matrix_from_labels(y_true_trainable, base_preds[trainable_mask], CLASS_LIST)
    base_macro_f1 = ev.metrics_from_cm(base_cm, CLASS_LIST)["macro_f1"]

    # Unique runs and run index mapping
    runs = val_clean["simulation_id"].unique()
    n_runs = len(runs)
    run_indices = [np.where(val_clean["simulation_id"] == rid)[0] for rid in runs]

    # Pre-extract model feature matrix X
    feature_cols = getattr(model, "feature_cols", F.feature_columns(val_clean))
    X_base = val_clean[feature_cols].to_numpy(dtype=float)
    col_to_idx = {c: i for i, c in enumerate(feature_cols)}

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 903]))

    # Target features/groups dict
    if isinstance(features_to_permute, list):
        targets = {f: [f] for f in features_to_permute}
    else:
        targets = features_to_permute

    results: dict[str, dict[str, float]] = {}

    for name, cols in targets.items():
        # Get integer column indices for these target features
        col_idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
        if not col_idxs:
            continue
        drops = []
        for _ in range(n_repeats):
            perm = rng.permutation(n_runs)
            # Create perturbed copy of the target columns in numpy
            X_perm = X_base.copy()
            for r_orig_idx, r_perm_idx in enumerate(perm):
                orig_rows = run_indices[r_orig_idx]
                perm_rows = run_indices[r_perm_idx]
                min_len = min(len(orig_rows), len(perm_rows))
                X_perm[np.ix_(orig_rows[:min_len], col_idxs)] = X_base[np.ix_(perm_rows[:min_len], col_idxs)]

            # Fast model predict
            if hasattr(model, "pipeline"):
                X_clipped = np.clip(X_perm, -1e4, 1e4)
                perm_preds = model.pipeline.predict(X_clipped)
            else:
                # Fallback for threshold rule or other models
                df_perm = val_clean.copy()
                for i, c in enumerate(feature_cols):
                    df_perm[c] = X_perm[:, i]
                perm_preds = model.predict(df_perm)

            perm_cm = ev.confusion_matrix_from_labels(y_true_trainable, perm_preds[trainable_mask], CLASS_LIST)
            perm_macro_f1 = ev.metrics_from_cm(perm_cm, CLASS_LIST)["macro_f1"]
            drops.append(float(base_macro_f1 - perm_macro_f1))

        results[name] = {
            "mean_importance": float(np.mean(drops)),
            "std_importance": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            "min_importance": float(np.min(drops)),
            "max_importance": float(np.max(drops)),
            "repeats": len(drops),
        }

    return results


# -----------------------------------------------------------------------------
# 4. Score labelling & Reliability Calibration on VAL only
# -----------------------------------------------------------------------------
def compute_reliability_calibration(
    models: dict[str, Any],
    val_df: pd.DataFrame,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute reliability diagram statistics for raw model_score on VAL only.

    Computes:
    - Top-1 predicted class calibration table (mean model_score vs empirical accuracy)
    - One-vs-rest per-class calibration table (mean model_score vs true positive rate)
    All column names use 'model_score' (never 'probability' or 'confidence').
    """
    split_vals = val_df["split"].unique()
    if len(split_vals) != 1 or split_vals[0] != "val":
        raise ValueError(f"Reliability analysis must be computed on VAL split only, got: {split_vals}")

    mask = L.trainable(val_df).to_numpy()
    val_t = val_df.loc[mask].copy()
    y_true = val_t["label"].to_numpy()

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    results: dict[str, Any] = {"top1_calibration": [], "per_class_calibration": []}

    for mname in ("random_forest", "logistic_regression"):
        model = models[mname]
        # Predict score array (N, K)
        scores = model.predict_proba(val_t)
        classes = model.pipeline.classes_
        top_indices = np.argmax(scores, axis=1)
        top_scores = np.max(scores, axis=1)
        pred_labels = classes[top_indices]
        is_correct = (pred_labels == y_true).astype(float)

        # Top-1 calibration
        for b in range(n_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            in_bin = (top_scores >= lo) & (top_scores < hi if b < n_bins - 1 else top_scores <= hi)
            cnt = int(in_bin.sum())
            mean_sc = float(np.mean(top_scores[in_bin])) if cnt > 0 else np.nan
            emp_acc = float(np.mean(is_correct[in_bin])) if cnt > 0 else np.nan
            results["top1_calibration"].append({
                "model": mname,
                "bin_idx": b,
                "bin_lower_model_score": round(lo, 2),
                "bin_upper_model_score": round(hi, 2),
                "mean_model_score": round(mean_sc, 4) if np.isfinite(mean_sc) else np.nan,
                "empirical_accuracy": round(emp_acc, 4) if np.isfinite(emp_acc) else np.nan,
                "window_count": cnt,
            })

        # One-vs-rest calibration per class
        for c_idx, c_name in enumerate(classes):
            c_scores = scores[:, c_idx]
            y_binary = (y_true == c_name).astype(float)
            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                in_bin = (c_scores >= lo) & (c_scores < hi if b < n_bins - 1 else c_scores <= hi)
                cnt = int(in_bin.sum())
                mean_sc = float(np.mean(c_scores[in_bin])) if cnt > 0 else np.nan
                emp_pos = float(np.mean(y_binary[in_bin])) if cnt > 0 else np.nan
                results["per_class_calibration"].append({
                    "model": mname,
                    "class_name": c_name,
                    "bin_idx": b,
                    "bin_lower_model_score": round(lo, 2),
                    "bin_upper_model_score": round(hi, 2),
                    "mean_model_score": round(mean_sc, 4) if np.isfinite(mean_sc) else np.nan,
                    "empirical_positive_rate": round(emp_pos, 4) if np.isfinite(emp_pos) else np.nan,
                    "window_count": cnt,
                })

    return {
        "top1_df": pd.DataFrame(results["top1_calibration"]),
        "per_class_df": pd.DataFrame(results["per_class_calibration"]),
    }
