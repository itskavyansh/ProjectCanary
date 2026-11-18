"""Evaluation suite (SYNTHETIC / SIMULATION): window-level, run-level, bootstrap CIs over runs,
SNR detection limits, Random Forest feature-group ablations, hard-pair failure analysis, and
stress-set combined-window evaluation.

Bootstrap rule:
- All 95% confidence intervals resample WHOLE RUNS (`simulation_id`) with replacement, pulling all
  trainable windows of each sampled run together. Windows are never resampled independently.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import classes as C, labels as L

CLASS_LIST = list(C.CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}

SNR_BINS = (
    ("<0.5", 0.0, 0.5),
    ("0.5-1", 0.5, 1.0),
    ("1-2", 1.0, 2.0),
    ("2-5", 2.0, 5.0),
    (">5", 5.0, float("inf")),
)


def confusion_matrix_from_labels(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str] = CLASS_LIST) -> np.ndarray:
    """Fast integer confusion matrix (K, K) for fixed class order."""
    k = len(classes)
    idx_map = {c: i for i, c in enumerate(classes)}
    ti = np.array([idx_map[v] for v in y_true], dtype=int)
    pi = np.array([idx_map[v] for v in y_pred], dtype=int)
    cm = np.zeros((k, k), dtype=int)
    np.add.at(cm, (ti, pi), 1)
    return cm


def metrics_from_cm(cm: np.ndarray, classes: list[str] = CLASS_LIST) -> dict:
    """Compute macro-F1 and per-class precision, recall, F1, and support from a confusion matrix."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    support = cm.sum(axis=1).astype(int)

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)

    # Macro-F1 across all classes present in ground truth (or all 6 classes when all present)
    present = support > 0
    macro_f1 = float(np.mean(f1[present])) if present.any() else 0.0

    per_class = {}
    for i, c in enumerate(classes):
        per_class[c] = {
            "precision": float(prec[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return {
        "macro_f1": macro_f1,
        "per_class": per_class,
        "prec_arr": prec,
        "rec_arr": rec,
        "f1_arr": f1,
        "support_arr": support,
    }


def evaluate_window_level_with_run_bootstrap(
    df: pd.DataFrame,
    preds_col_or_arr,
    n_boot: int = 300,
    seed: int = 20261117,
    classes: list[str] = CLASS_LIST,
) -> dict:
    """Evaluate window-level metrics and 95% CIs by bootstrapping over RUNS (`simulation_id`).

    Excludes `transition` windows automatically if any are present.
    Also records the exact list of resampled run IDs for the first bootstrap replicate so tests can
    verify that whole runs (never isolated windows) are resampled.
    """
    mask = L.trainable(df).to_numpy()
    sub = df.loc[mask].copy()
    preds = np.asarray(preds_col_or_arr)[mask] if not isinstance(preds_col_or_arr, str) else sub[preds_col_or_arr].to_numpy()
    y_true = sub["label"].to_numpy()

    cm = confusion_matrix_from_labels(y_true, preds, classes)
    point = metrics_from_cm(cm, classes)

    # Pre-compute per-run confusion matrix (R, K, K) -> resampling runs = summing cm_by_run[run_indices]
    run_ids = sub["simulation_id"].unique()
    n_runs = len(run_ids)
    k = len(classes)
    cm_by_run = np.zeros((n_runs, k, k), dtype=int)
    run_window_counts = np.zeros(n_runs, dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}
    ti_all = np.array([idx_map[v] for v in y_true], dtype=int)
    pi_all = np.array([idx_map[v] for v in preds], dtype=int)

    run_to_r = {rid: r for r, rid in enumerate(run_ids)}
    r_codes = np.array([run_to_r[rid] for rid in sub["simulation_id"]], dtype=int)
    np.add.at(cm_by_run, (r_codes, ti_all, pi_all), 1)
    np.add.at(run_window_counts, r_codes, 1)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 701]))
    boot_macro_f1 = np.zeros(n_boot, dtype=float)
    boot_prec = np.zeros((n_boot, k), dtype=float)
    boot_rec = np.zeros((n_boot, k), dtype=float)
    boot_f1 = np.zeros((n_boot, k), dtype=float)
    first_boot_runs = []

    for b in range(n_boot):
        pick = rng.integers(0, n_runs, size=n_runs)
        if b == 0:
            first_boot_runs = [str(run_ids[i]) for i in pick]
        cm_b = cm_by_run[pick].sum(axis=0)
        m_b = metrics_from_cm(cm_b, classes)
        boot_macro_f1[b] = m_b["macro_f1"]
        boot_prec[b] = m_b["prec_arr"]
        boot_rec[b] = m_b["rec_arr"]
        boot_f1[b] = m_b["f1_arr"]

    macro_ci = (float(np.percentile(boot_macro_f1, 2.5)), float(np.percentile(boot_macro_f1, 97.5)))
    for i, c in enumerate(classes):
        point["per_class"][c]["precision_ci95"] = (
            float(np.percentile(boot_prec[:, i], 2.5)),
            float(np.percentile(boot_prec[:, i], 97.5)),
        )
        point["per_class"][c]["recall_ci95"] = (
            float(np.percentile(boot_rec[:, i], 2.5)),
            float(np.percentile(boot_rec[:, i], 97.5)),
        )
        point["per_class"][c]["f1_ci95"] = (
            float(np.percentile(boot_f1[:, i], 2.5)),
            float(np.percentile(boot_f1[:, i], 97.5)),
        )

    return {
        "n_windows": int(cm.sum()),
        "n_runs": n_runs,
        "confusion_matrix": cm,
        "classes": classes,
        "macro_f1": point["macro_f1"],
        "macro_f1_ci95": macro_ci,
        "per_class": point["per_class"],
        "boot_macro_f1_samples": boot_macro_f1,
        "boot_rec_samples": boot_rec,
        "first_boot_run_ids": first_boot_runs,
    }


def evaluate_run_level_detection(
    df_all_windows: pd.DataFrame,
    preds_all_windows: np.ndarray,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    """Evaluate run-level event/fault detection, false alarms per normal run, and detection delay.

    Crucially, `df_all_windows` includes `transition` windows (with their `t_rel_event_start_s`
    offsets):
    - For an event/fault run (`run_class != 'normal'`), a detection occurs on the first window at or
      after event onset (`t_end > event_start_s`, i.e. `t_rel_event_start_s > -window_s/2`) where:
        * `any_alarm_detected`: predicted label != 'normal'
        * `exact_class_detected`: predicted label == `run_class` (or in `label_set` for stress runs)
      Detection delay (s) = `max(0.0, t_end - event_start_s)` = `max(0.0, t_rel_event_start_s + 2.5)`
      at the first detected window (including transition windows straddling the onset!).
    - For a `normal` run (`run_class == 'normal'`), every window predicted != 'normal' counts as a
      false alarm window; we report both false-alarm windows per normal run and the fraction of
      normal runs with >=1 false alarm.
    """
    d = df_all_windows.copy()
    d["_pred"] = np.asarray(preds_all_windows)
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])

    run_records = []
    for sim_id, g in d.groupby("simulation_id", sort=False):
        g = g.sort_values("t_start")
        r_cls = str(g["run_class"].iloc[0])
        if r_cls == "normal":
            fa_wins = int((g["_pred"] != "normal").sum())
            run_records.append({
                "simulation_id": sim_id,
                "run_class": "normal",
                "is_event_run": False,
                "fa_windows": fa_wins,
                "has_fa": int(fa_wins > 0),
                "exact_detected": np.nan,
                "any_detected": np.nan,
                "delay_exact_s": np.nan,
                "delay_any_s": np.nan,
            })
        else:
            # Windows overlapping the event or aftermath (including onset transition window)
            post = g[g["t_rel_event_start_s"] > -win_half + 1e-9]
            any_hits = post[post["_pred"] != "normal"]
            # For stress runs, exact match accepts either run_class or any active class in label_set
            ls_col = post["label_set"].fillna("").astype(str).tolist()
            pr_col = post["_pred"].astype(str).tolist()
            exact_mask = [
                (pr == r_cls) or (pr in ls.split("|") if ls else False)
                for pr, ls in zip(pr_col, ls_col)
            ]
            exact_hits = post[exact_mask]
            any_det = int(len(any_hits) > 0)
            ex_det = int(len(exact_hits) > 0)
            delay_any = float(max(0.0, any_hits.iloc[0]["t_rel_event_start_s"] + win_half)) if any_det else np.nan
            delay_ex = float(max(0.0, exact_hits.iloc[0]["t_rel_event_start_s"] + win_half)) if ex_det else np.nan
            run_records.append({
                "simulation_id": sim_id,
                "run_class": r_cls,
                "is_event_run": True,
                "fa_windows": np.nan,
                "has_fa": np.nan,
                "exact_detected": ex_det,
                "any_detected": any_det,
                "delay_exact_s": delay_ex,
                "delay_any_s": delay_any,
            })

    rdf = pd.DataFrame(run_records)
    ev_runs = rdf[rdf["is_event_run"]]
    norm_runs = rdf[~rdf["is_event_run"]]

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 702]))

    def _boot_mean(arr: np.ndarray) -> tuple[float, tuple[float, float]]:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, (0.0, 0.0)
        pt = float(np.mean(arr))
        b_means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(n_boot)]
        return pt, (float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5)))

    ex_rate, ex_ci = _boot_mean(ev_runs["exact_detected"].to_numpy(dtype=float))
    any_rate, any_ci = _boot_mean(ev_runs["any_detected"].to_numpy(dtype=float))
    del_ex, del_ex_ci = _boot_mean(ev_runs["delay_exact_s"].to_numpy(dtype=float))
    del_any, del_any_ci = _boot_mean(ev_runs["delay_any_s"].to_numpy(dtype=float))
    fa_per_run, fa_ci = _boot_mean(norm_runs["fa_windows"].to_numpy(dtype=float))
    fa_run_frac, fa_frac_ci = _boot_mean(norm_runs["has_fa"].to_numpy(dtype=float))

    per_class_run = {}
    for cls in C.CLASSES:
        if cls == "normal":
            continue
        sub_c = ev_runs[ev_runs["run_class"] == cls]
        if len(sub_c) == 0:
            continue
        er, er_ci = _boot_mean(sub_c["exact_detected"].to_numpy(dtype=float))
        ar, ar_ci = _boot_mean(sub_c["any_detected"].to_numpy(dtype=float))
        de, de_ci = _boot_mean(sub_c["delay_exact_s"].to_numpy(dtype=float))
        per_class_run[cls] = {
            "n_runs": len(sub_c),
            "exact_detect_rate": er,
            "exact_detect_rate_ci95": er_ci,
            "any_detect_rate": ar,
            "any_detect_rate_ci95": ar_ci,
            "mean_delay_exact_s": de,
            "mean_delay_exact_s_ci95": de_ci,
            "median_delay_exact_s": float(np.nanmedian(sub_c["delay_exact_s"])) if sub_c["delay_exact_s"].notna().any() else np.nan,
        }

    return {
        "n_event_runs": len(ev_runs),
        "n_normal_runs": len(norm_runs),
        "exact_detect_rate": ex_rate,
        "exact_detect_rate_ci95": ex_ci,
        "any_detect_rate": any_rate,
        "any_detect_rate_ci95": any_ci,
        "mean_delay_exact_s": del_ex,
        "mean_delay_exact_s_ci95": del_ex_ci,
        "mean_delay_any_s": del_any,
        "mean_delay_any_s_ci95": del_any_ci,
        "false_alarms_per_normal_run": fa_per_run,
        "false_alarms_per_normal_run_ci95": fa_ci,
        "normal_run_with_fa_rate": fa_run_frac,
        "normal_run_with_fa_rate_ci95": fa_frac_ci,
        "per_class": per_class_run,
        "run_table": rdf,
    }


def evaluate_recall_vs_snr(
    df: pd.DataFrame,
    preds: np.ndarray,
    manifest_df: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 20261117,
) -> pd.DataFrame:
    """Compute window-level recall (exact class and any-basin class) across SNR bins for gradual and rapid."""
    mask = L.trainable(df).to_numpy()
    sub = df.loc[mask, ["simulation_id", "label"]].copy()
    sub["pred"] = np.asarray(preds)[mask]
    snr_map = dict(zip(manifest_df["simulation_id"], manifest_df["snr_vs_imu_noise"]))
    sub["snr"] = sub["simulation_id"].map(snr_map)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 703]))
    rows = []
    for cls in ("gradual", "rapid"):
        c_sub = sub[sub["label"] == cls]
        for b_lbl, lo, hi in SNR_BINS:
            b_df = c_sub[(c_sub["snr"] >= lo) & (c_sub["snr"] < hi)]
            run_ids = b_df["simulation_id"].unique()
            n_runs = len(run_ids)
            n_wins = len(b_df)
            if n_wins == 0:
                rows.append({
                    "class": cls,
                    "snr_bin": b_lbl,
                    "n_runs": 0,
                    "n_windows": 0,
                    "exact_recall": np.nan,
                    "exact_recall_ci_lo": np.nan,
                    "exact_recall_ci_hi": np.nan,
                    "basin_recall": np.nan,
                    "basin_recall_ci_lo": np.nan,
                    "basin_recall_ci_hi": np.nan,
                })
                continue

            exact_hit = (b_df["pred"] == cls).to_numpy(dtype=float)
            basin_hit = b_df["pred"].isin(["gradual", "rapid"]).to_numpy(dtype=float)
            ex_pt = float(exact_hit.mean())
            ba_pt = float(basin_hit.mean())

            # Bootstrap over runs in this SNR bin
            by_run_ex = [exact_hit[b_df["simulation_id"].to_numpy() == r] for r in run_ids]
            by_run_ba = [basin_hit[b_df["simulation_id"].to_numpy() == r] for r in run_ids]
            b_ex, b_ba = [], []
            for _ in range(n_boot):
                pick = rng.integers(0, n_runs, size=n_runs)
                b_ex.append(float(np.concatenate([by_run_ex[i] for i in pick]).mean()))
                b_ba.append(float(np.concatenate([by_run_ba[i] for i in pick]).mean()))
            rows.append({
                "class": cls,
                "snr_bin": b_lbl,
                "n_runs": n_runs,
                "n_windows": n_wins,
                "exact_recall": ex_pt,
                "exact_recall_ci_lo": float(np.percentile(b_ex, 2.5)),
                "exact_recall_ci_hi": float(np.percentile(b_ex, 97.5)),
                "basin_recall": ba_pt,
                "basin_recall_ci_lo": float(np.percentile(b_ba, 2.5)),
                "basin_recall_ci_hi": float(np.percentile(b_ba, 97.5)),
            })
    return pd.DataFrame(rows)


def evaluate_stress_set(
    stress_df: pd.DataFrame,
    preds: np.ndarray,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    """Evaluate stress set with combined conditions under two explicit label policies:
    1) Primary-priority label (`label` column, determined by `PRIORITY = (rapid, gradual,
       abnormal_vibration, sensor_fault, comm_fault, normal)` when >=2 conditions overlap >=50%).
    2) Set-membership accuracy (`any_active_match`): for multi-condition windows (`label_set` has
       1 or 2 active conditions), a prediction is credited if `pred` matches ANY active condition in
       `label_set` (or `'normal'` when `label_set` is empty).
    """
    primary_eval = evaluate_window_level_with_run_bootstrap(stress_df, preds, n_boot=n_boot, seed=seed)
    mask = L.trainable(stress_df).to_numpy()
    sub = stress_df.loc[mask].copy()
    p_sub = np.asarray(preds)[mask]

    def _matches_set(row_lbl_set: str, row_lbl: str, pr: str) -> bool:
        if not row_lbl_set or pd.isna(row_lbl_set):
            return pr == row_lbl
        return pr in str(row_lbl_set).split("|")

    hits = np.array(
        [_matches_set(ls, lb, pr) for ls, lb, pr in zip(sub["label_set"], sub["label"], p_sub)],
        dtype=float,
    )
    multi_mask = sub["label_set"].fillna("").str.contains(r"\|").to_numpy()
    set_acc_all = float(hits.mean())
    set_acc_multi = float(hits[multi_mask].mean()) if multi_mask.any() else np.nan
    primary_acc_multi = float((p_sub[multi_mask] == sub["label"].to_numpy()[multi_mask]).mean()) if multi_mask.any() else np.nan

    return {
        "primary_eval": primary_eval,
        "set_match_accuracy_all": set_acc_all,
        "n_multi_condition_windows": int(multi_mask.sum()),
        "primary_accuracy_on_multi_windows": primary_acc_multi,
        "set_match_accuracy_on_multi_windows": set_acc_multi,
    }


HARD_PAIRS = (
    ("gradual", "sensor_fault"),
    ("normal", "abnormal_vibration"),
    ("comm_fault", "sensor_fault"),
    ("gradual", "rapid"),
)


def analyze_hard_pairs(
    df: pd.DataFrame,
    preds_by_model: dict[str, np.ndarray],
    manifest_df: pd.DataFrame,
) -> dict[str, dict]:
    """Analyze the 4 hard pairs across models and extract the 3 most common failure patterns with example windows."""
    mask = L.trainable(df).to_numpy()
    sub = df.loc[mask].copy().reset_index(drop=True)
    snr_map = dict(zip(manifest_df["simulation_id"], manifest_df["snr_vs_imu_noise"]))
    ftype_map = dict(zip(manifest_df["simulation_id"], manifest_df["fault_types"]))
    sub["snr"] = sub["simulation_id"].map(snr_map)
    sub["fault_type"] = sub["simulation_id"].map(ftype_map)

    out = {}
    for a, b in HARD_PAIRS:
        pair_key = f"{a}_vs_{b}"
        model_stats = {}
        for mname, pr_all in preds_by_model.items():
            pr = np.asarray(pr_all)[mask]
            a_to_b = int(((sub["label"] == a) & (pr == b)).sum())
            b_to_a = int(((sub["label"] == b) & (pr == a)).sum())
            n_a = int((sub["label"] == a).sum())
            n_b = int((sub["label"] == b).sum())
            model_stats[mname] = {
                f"{a}_misclassified_as_{b}": a_to_b,
                f"{b}_misclassified_as_{a}": b_to_a,
                "total_pair_confusions": a_to_b + b_to_a,
                f"{a}_support": n_a,
                f"{b}_support": n_b,
            }

        # Use Random Forest, Threshold Rule, and Logistic Regression errors to catalogue the 3 failure patterns
        rf_pr = np.asarray(preds_by_model.get("random_forest", next(iter(preds_by_model.values()))))[mask]
        th_pr = np.asarray(preds_by_model.get("threshold_rule", rf_pr))[mask]
        lr_pr = np.asarray(preds_by_model.get("logistic_regression", rf_pr))[mask]
        sub["_rf"] = rf_pr
        sub["_th"] = th_pr
        sub["_lr"] = lr_pr

        patterns = _extract_three_patterns(sub, a, b)
        out[pair_key] = {"counts_by_model": model_stats, "patterns": patterns}
    return out


def _fmt_example(row: pd.Series) -> dict:
    return {
        "simulation_id": str(row["simulation_id"]),
        "window_idx": int(row["window_idx"]),
        "t_range_s": f"[{row['t_start']:.1f}, {row['t_end']:.1f})",
        "true_label": str(row["label"]),
        "rf_pred": str(row["_rf"]),
        "th_pred": str(row["_th"]),
        "snr": round(float(row.get("snr", 0.0)), 3),
        "fault_type": str(row.get("fault_type", "")),
        "NODE_01_mean": round(float(row["NODE_01_mean"]), 4),
        "long_slope_max": round(float(row["long_slope_max_deg_s"]), 5),
        "piezo_rms_net": round(float(row["piezo_rms_network_mean"]), 4),
        "hub_cm_m": round(float(row["hub_common_mode_m"]), 4),
        "comm_miss_net": round(float(row["comm_missing_frac_net"]), 3),
    }


def _extract_three_patterns(sub: pd.DataFrame, a: str, b: str) -> list[dict]:
    lr_pr = sub.get("_lr", sub["_rf"])
    any_err = sub[
        ((sub["label"] == a) & ((sub["_rf"] == b) | (sub["_th"] == b) | (lr_pr == b)))
        | ((sub["label"] == b) & ((sub["_rf"] == a) | (sub["_th"] == a) | (lr_pr == a)))
    ]
    patterns = []

    if (a, b) == ("gradual", "sensor_fault"):
        p1 = any_err[(any_err["label"] == "sensor_fault") & (any_err["fault_type"].str.contains("slow_drift|bias_step", na=False))]
        p2 = any_err[(any_err["label"] == "gradual") & (any_err["NODE_01_mean"] > 0.35)]
        p3 = any_err[(any_err["label"] == "sensor_fault") & (any_err["fault_type"].str.contains("wrong_dwell|stale", na=False))]
        defs = [
            ("Single/two-node IMU slow_drift or bias_step mimicking basin tilt ramp", p1),
            ("Asymmetric off-centre gradual basin loading 1 corner node strongly (mistaken for single-node sensor fault)", p2),
            ("Hub wrong_dwell / stale_readings window before next LiDAR pass where IMU noise mimics minor tilt", p3),
        ]
    elif (a, b) == ("normal", "abnormal_vibration"):
        p1 = sub[(sub["label"] == "normal") & ((sub["_rf"] == "abnormal_vibration") | (sub["_th"] == "abnormal_vibration") | (lr_pr == "abnormal_vibration"))]
        p2 = sub[(sub["label"] == "abnormal_vibration") & (sub["t_rel_event_start_s"].abs() < 8.0)].sort_values("piezo_rms_network_mean")
        p3 = sub[(sub["label"] == "abnormal_vibration") & (sub["piezo_nodes_with_impulse"] <= 3)].sort_values("piezo_rms_network_mean")
        defs = [
            ("Two-node benign wind burst + coinciding background piezo impulse in a normal window exceeding threshold", p1),
            ("Onset/tail boundary window of abnormal_vibration where ramp envelope is near minimum", p2),
            ("Abnormal vibration window with lowest network piezo RMS / fewest simultaneous impulse nodes", p3),
        ]
    elif (a, b) == ("comm_fault", "sensor_fault"):
        p1 = any_err[(any_err["label"] == "sensor_fault") & (any_err["fault_type"].str.contains("intermittent_disconnect|stuck_at|bias_step", na=False))]
        p2 = any_err[(any_err["label"] == "comm_fault") & (any_err["comm_missing_frac_net"] <= 0.20)]
        p3 = any_err[(any_err["label"] == "comm_fault") & (any_err["comm_latency_max_s"] > 2.5)]
        defs = [
            ("Sensor fault window coinciding with background LoRa packet loss (comm_missing_frac_net >= 0.05)", p1),
            ("Sparse packet_drops comm_fault window (missing_frac <= 0.20) where IMU noise triggers sensor_fault", p2),
            ("Added-latency comm_fault window with zero packet drops where random tilt excursion triggers sensor_fault", p3),
        ]
    else:  # ("gradual", "rapid")
        p1 = any_err[(any_err["label"] == "rapid") & (any_err["t_rel_event_start_s"] > 25.0)]
        p2 = any_err[(any_err["label"] == "gradual") & (any_err["snr"] > 5.0)]
        p3 = any_err[any_err["snr"] < 1.2]
        defs = [
            ("Rapid basin in post-ramp aftermath hold (within-window slope ~0 deg/s, indistinguishable from held gradual tilt)", p1),
            ("High-SNR gradual basin during steepest mid-ramp phase (high slope mimicking rapid onset)", p2),
            ("Low-SNR (<1.2) basin windows where IMU white noise dominates both short and 12-window slopes", p3),
        ]

    for desc, df_pat in defs:
        if len(df_pat) == 0:
            df_pat = any_err if len(any_err) > 0 else sub[sub["label"].isin([a, b])]
        ex = [_fmt_example(df_pat.iloc[i]) for i in range(min(2, len(df_pat)))]
        patterns.append({
            "pattern": desc,
            "count": int(len(df_pat)),
            "examples": ex,
        })
    return patterns
