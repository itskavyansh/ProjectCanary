#!/usr/bin/env python
"""Runner for the unsupervised "unknown_anomaly" detector evaluation (SYNTHETIC / SIMULATION).

Executes:
1. Fits primary `IsolationForestAnomalyDetector` (fixed seed) and secondary `RobustMahalanobisDetector`
   strictly on `train` normal windows (21,305 windows, 81 model features from `feature_columns(df)`,
   never `diag_` columns).
2. Chooses flag thresholds strictly on `val` normal windows (20 normal runs, 1,900 windows) under
   `(k=3, m=3)` persistence to achieve at most `1.0` false-alarm episode per normal run.
3. Evaluates (without changing any 6-class classifier output):
   - Per-class vs `normal` on `test`: AUROC and window-level detection recall of each known class
     with run-bootstrap 95% CIs.
   - Run-level event flagged rate, detection delay, and normal-run false-alarm episodes per run
     (plus Definition A slow-process and Definition B sensor-time per-hour rates) across `val`,
     `test`, `shifted_test`, `shape_mismatch` arms, `B1` high-tilt arms, and `B2` realistic-depth arms.
   - `gradual` and `rapid` recall by `snr_vs_imu_noise` bin (`<1`, `1-2`, `2-5`, `>5`).
   - Complement analysis at the v3 episode operating point (`k=3, m=3`):
     * Fraction of classifier-missed event runs that the detector flags
     * Extra false-alarm episodes per normal run added by the detector
     * Combined rule `"classifier_alarm OR detector_flag"` and third rule
       `"unknown_only_when_detector_flags_and_classifier_says_normal"`, each with its false-alarm cost.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import (  # noqa: E402
    eval_extras_v2 as V2,
    features as F,
    models as M,
    shapes as SH,
    unknown_detector as UD,
    versioning as VR,
)

MODELS = ("random_forest", "threshold_rule", "logistic_regression")
FROZEN_CHECK_FILES = (
    "canary_sim/schema.py",
    "canary_sim/engine.py",
    "canary_sim/features.py",
    "canary_sim/labels.py",
    "canary_sim/models.py",
    "canary_sim/alarm.py",
    "configs/scenario_default.yaml",
    "configs/sensor_default.yaml",
    "configs/confounder_default.yaml",
    "configs/dataset_default.yaml",
    "outputs/datasets_v3/manifest.csv",
    "outputs/datasets_v3/features_train.csv",
    "outputs/datasets_v3/features_val.csv",
    "outputs/datasets_v3/features_test.csv",
    "outputs/datasets_v3/features_shifted_test.csv",
    "outputs/eval_v3/trained_models_v3.pkl",
    "outputs/eval_v3/operating_points_v3.pkl",
    "outputs/eval_extras/false_alarms_per_hour.csv",
    "outputs/eval_extras_v2/false_alarms_per_hour_both_definitions.csv",
)


def snapshot_frozen_hashes() -> dict[str, str]:
    return {
        rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        for rel in FROZEN_CHECK_FILES
        if (ROOT / rel).is_file()
    }


def _add_banner(fig: plt.Figure, subtitle: str = "") -> None:
    txt = "SYNTHETIC / SIMULATION — Compression Factor: 360x (Gradual/Normal: 24h/run) | 1x (Rapid: 240s/run)"
    if subtitle:
        txt += f" | {subtitle}"
    fig.text(
        0.5,
        0.97,
        txt,
        ha="center",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#d32f2f", edgecolor="none"),
    )


def ci_str(pt: float, lo: float, hi: float, d: int = 3) -> str:
    if pd.isna(pt):
        return "N/A"
    return f"{pt:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _ensure_stored_features_exist() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load stored feature and manifest tables from `outputs/eval_shape_mismatch/` and
    `outputs/eval_literature/`; regenerate via existing scripts if missing."""
    sm_feat_path = ROOT / "outputs/eval_shape_mismatch/window_features.csv"
    sm_man_path = ROOT / "outputs/eval_shape_mismatch/manifest.csv"
    if not (sm_feat_path.is_file() and sm_man_path.is_file()):
        print("[unknown_detector] regenerating outputs/eval_shape_mismatch/ via scripts/run_shape_mismatch.py ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/run_shape_mismatch.py")], check=True)

    lit_feat_path = ROOT / "outputs/eval_literature/window_features.csv"
    lit_man_path = ROOT / "outputs/eval_literature/manifest.csv"
    if not (lit_feat_path.is_file() and lit_man_path.is_file()):
        print("[unknown_detector] regenerating outputs/eval_literature/ via scripts/run_literature_checks.py ...", flush=True)
        subprocess.run([sys.executable, str(ROOT / "scripts/run_literature_checks.py")], check=True)

    sm_feat = pd.read_csv(sm_feat_path)
    sm_man = pd.read_csv(sm_man_path)
    lit_feat = pd.read_csv(lit_feat_path)
    lit_man = pd.read_csv(lit_man_path)
    return sm_feat, sm_man, lit_feat, lit_man


def _clf_episode_window_preds(mname: str, model_obj: object, ops: dict, df: pd.DataFrame) -> tuple[np.ndarray, int, int]:
    """Return frozen v3 `episode_fa<=1` window predictions (before persistence) and `(k, m)`."""
    regs = V2.regimes_for(mname, model_obj, ops, df)
    for rname, w_preds, k, m in regs:
        if rname == "episode_fa<=1 (k=3,m=3)":
            return np.asarray(w_preds, dtype=object), int(k), int(m)
    raise KeyError(f"episode_fa<=1 regime not found for {mname}")


# -----------------------------------------------------------------------------
# Plots (with red SYNTHETIC / SIMULATION banner)
# -----------------------------------------------------------------------------
def plot_test_auroc_and_snr_recall(
    auroc_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot per-class vs normal AUROC on `test` and SNR-binned recall on `test`."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.3))
    classes_order = ["gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault", "all_known_anomalies"]
    cls_labels = ["gradual", "rapid", "abnormal\nvibration", "sensor\nfault", "comm\nfault", "all known\nanomalies"]
    x = np.arange(len(classes_order))
    w = 0.35

    sub_if = auroc_df[auroc_df["detector"] == "isolation_forest"].set_index("class_vs_normal").reindex(classes_order)
    sub_rm = auroc_df[auroc_df["detector"] == "robust_mahalanobis"].set_index("class_vs_normal").reindex(classes_order)

    for offset, sdf, lbl, col in (
        (-w / 2, sub_if, "Isolation Forest (Primary)", "#1976d2"),
        (+w / 2, sub_rm, "Robust Mahalanobis (Secondary)", "#00897b"),
    ):
        y = sdf["auroc"].to_numpy(float)
        lo = np.maximum(0.0, y - sdf["auroc_ci_lo"].to_numpy(float))
        hi = np.maximum(0.0, sdf["auroc_ci_hi"].to_numpy(float) - y)
        axes[0].bar(x + offset, y, width=w, color=col, alpha=0.88, label=lbl, yerr=np.vstack([lo, hi]), capsize=3)

    axes[0].axhline(0.5, color="#757575", linestyle="--", linewidth=1.2, label="Random Guess (0.50)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cls_labels, fontsize=8.5)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Window AUROC vs Normal (Test Split)", fontsize=9.5)
    axes[0].set_title("Per-Class vs Normal Window AUROC on Test (95% Run-Bootstrap CI)", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6, axis="y")
    axes[0].legend(loc="lower right", fontsize=8.3)

    # Panel 2: SNR-binned window flag recall and run flag rate on test (Isolation Forest)
    bins = ["<1", "1-2", "2-5", ">5"]
    xb = np.arange(len(bins))
    s_test = snr_df[(snr_df["split_or_arm"] == "test") & (snr_df["detector"] == "isolation_forest")]
    for cls, col, mk in (("gradual", "#1976d2", "o"), ("rapid", "#e65100", "s")):
        sc = s_test[s_test["class"] == cls].set_index("snr_bin").reindex(bins)
        axes[1].plot(xb, sc["window_recall"].to_numpy(float), marker=mk, linewidth=2.2, color=col, label=f"{cls} — Window Flag Recall (raw)")
        axes[1].plot(xb, sc["run_detect_rate"].to_numpy(float), marker=mk, linewidth=1.8, linestyle="--", color=col, alpha=0.75, label=f"{cls} — Run Flag Rate (k=3,m=3)")

    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([f"SNR {b}" for b in bins], fontsize=9)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].set_ylabel("Recall / Run Flag Rate", fontsize=9.5)
    axes[1].set_title("Isolation Forest: Gradual & Rapid Recall by SNR Bin (Test)", fontsize=10, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=8.2)

    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.98, wspace=0.20)
    _add_banner(fig, "Unsupervised Unknown Anomaly Detector on Held-Out Test")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_run_detection_across_regimes(
    comp_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot run-level detection rates across regimes: Frozen RF vs Isolation Forest vs Combined Rules."""
    regimes = [
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
    ]
    r_labels = [
        "test",
        "shifted\ntest",
        "gaussian\ncontrol",
        "trough\npanel",
        "asymmetric",
        "two\nbasin",
        "planar\nslope",
        "high tilt\n3-6 deg",
        "high tilt\n6-12 deg",
        "depth\nsigma>1.5R",
    ]
    sub = comp_df[
        (comp_df["classifier"] == "random_forest") & (comp_df["detector"] == "isolation_forest")
    ].set_index("split_or_arm").reindex(regimes)

    fig, ax = plt.subplots(figsize=(15.5, 5.4))
    x = np.arange(len(regimes))
    w = 0.20

    bars = [
        (-1.5 * w, "clf_exact_detect_rate", "Frozen RF Alone (Exact Class Alarm)", "#1976d2"),
        (-0.5 * w, "det_only_flag_rate", "Isolation Forest Alone (Unknown Flag)", "#7b1fa2"),
        (+0.5 * w, "gated_unknown_detect_rate", "Third Rule: RF Exact OR Gated Unknown (Det & RF=Normal)", "#00897b"),
        (+1.5 * w, "combined_or_detect_rate", "Combined Rule: RF Exact Alarm OR Detector Flag", "#f57c00"),
    ]
    for offset, col_name, lbl, color in bars:
        y = sub[col_name].to_numpy(float)
        ax.bar(x + offset, y, width=w, color=color, alpha=0.88, label=lbl)

    ax.set_xticks(x)
    ax.set_xticklabels(r_labels, fontsize=8.6)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Event Run Detection / Flag Rate (k=3, m=3)", fontsize=9.5)
    ax.set_title(
        "Event Run Detection Across In-Domain, Shape-Mismatch, High-Tilt (B1), and Realistic-Depth (B2) Regimes",
        fontsize=10.2,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax.legend(loc="lower right", fontsize=8.5, ncol=2)

    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.05, right=0.98)
    _add_banner(fig, "Frozen Random Forest vs Unsupervised Unknown Anomaly Detector")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_complement_tradeoff(
    comp_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot fraction of RF-missed runs flagged by the detector vs false-alarm cost per normal run."""
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.2))

    # Left panel: Fraction of RF-missed runs recovered by Isolation Forest vs Robust Mahalanobis
    arms = [
        "test",
        "shifted_test",
        "shape:trough_panel",
        "shape:two_basin",
        "shape:planar_slope",
        "lit:high_tilt_3_6",
        "lit:high_tilt_6_12",
        "lit:depth_all_gt_1p5",
    ]
    a_labels = [
        "test",
        "shifted\ntest",
        "trough\npanel",
        "two\nbasin",
        "planar\nslope",
        "high tilt\n3-6 deg",
        "high tilt\n6-12 deg",
        "depth\nsigma>1.5R",
    ]
    xa = np.arange(len(arms))
    w = 0.34
    for offset, dname, lbl, col in (
        (-w / 2, "isolation_forest", "Isolation Forest (Primary)", "#1976d2"),
        (+w / 2, "robust_mahalanobis", "Robust Mahalanobis (Secondary)", "#00897b"),
    ):
        sub = comp_df[(comp_df["classifier"] == "random_forest") & (comp_df["detector"] == dname)].set_index("split_or_arm").reindex(arms)
        y = sub["frac_of_clf_exact_missed_flagged_by_det"].fillna(0.0).to_numpy(float)
        axes[0].bar(xa + offset, y, width=w, color=col, alpha=0.88, label=lbl)

    axes[0].set_xticks(xa)
    axes[0].set_xticklabels(a_labels, fontsize=8.5)
    axes[0].set_ylim(0.0, 1.08)
    axes[0].set_ylabel("Fraction of RF-Missed Event Runs Flagged", fontsize=9.5)
    axes[0].set_title("Fraction of Classifier-Missed Event Runs Flagged by Unknown Detector", fontsize=9.8, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6, axis="y")
    axes[0].legend(loc="upper left", fontsize=8.3)

    # Right panel: Normal-run false-alarm episodes per run across test, shifted_test, and shared_normal_40
    norm_splits = ["test", "shifted_test", "shape:gaussian_control"]
    n_labels = ["test\n(20 normal runs)", "shifted_test\n(15 normal runs)", "shared_normal_40\n(40 normal runs)"]
    xn = np.arange(len(norm_splits))
    wn = 0.20
    sub_n = comp_df[(comp_df["classifier"] == "random_forest") & (comp_df["detector"] == "isolation_forest")].set_index("split_or_arm").reindex(norm_splits)
    for offset, col_name, lbl, col in (
        (-1.5 * wn, "clf_only_fa_per_normal_run", "RF Alone (episode_fa<=1)", "#1976d2"),
        (-0.5 * wn, "det_only_fa_per_normal_run", "Isolation Forest Alone", "#7b1fa2"),
        (+0.5 * wn, "total_fa_per_normal_run_with_gated", "RF + Third Rule (Gated Unknown)", "#00897b"),
        (+1.5 * wn, "combined_or_fa_per_normal_run", "Combined OR Rule (RF OR Detector)", "#f57c00"),
    ):
        y = sub_n[col_name].to_numpy(float)
        axes[1].bar(xn + offset, y, width=wn, color=col, alpha=0.88, label=lbl)

    axes[1].axhline(1.0, color="#d32f2f", linestyle="--", linewidth=1.3, label="Val Budget (1.0 FA ep / normal run)")
    axes[1].set_xticks(xn)
    axes[1].set_xticklabels(n_labels, fontsize=8.8)
    axes[1].set_ylabel("False-Alarm Episodes per Normal Run (k=3, m=3)", fontsize=9.5)
    axes[1].set_title("Normal-Run False-Alarm Cost: Classifier vs Combined Rules", fontsize=9.8, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6, axis="y")
    axes[1].legend(loc="upper right", fontsize=8.0)

    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.98, wspace=0.22)
    _add_banner(fig, "Complement Recovery vs Normal-Run False-Alarm Cost")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Markdown Table Writers
# -----------------------------------------------------------------------------
def _write_markdown_tables(
    out_dir: Path,
    thresh_summary: dict,
    auroc_df: pd.DataFrame,
    run_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    comp_df: pd.DataFrame,
) -> None:
    # 1. Threshold summary markdown
    t_lines = [
        "# Unsupervised Unknown Anomaly Detector — Threshold Selection on `val` Normal Runs (`SYNTHETIC / SIMULATION`)\n",
        "- **Fit Data:** Strictly `train` windows with `label == 'normal'` (`21,305` windows across all `train` runs, including normal lead-in and benign confounder windows; `81` model features from `feature_columns(df)`; zero `diag_` columns).",
        "- **Threshold Tuning Data:** Strictly `val` normal-only runs (`20` runs, `1,900` windows), evaluated under `(k=3, m=3)` episode persistence with target `<= 1.0` false-alarm episode per normal run.",
        "- **Score Naming:** Output score is `anomaly_score` (never a probability or confidence); flag is `unknown_anomaly`.\n",
        "| Detector | Fitted `train` Normal Windows | `val` Normal Runs (`windows`) | Persistence `(k, m)` | Target FA / Normal Run | Chosen `anomaly_score` Threshold | `val` Normal Window Quantile | Achieved `val` FA Episodes / Run | Achieved `val` FA Windows / Run |",
        "| :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dname, info in thresh_summary["detectors"].items():
        vs = info["val_tuning_summary"]
        t_lines.append(
            f"| `{dname}` | {info['fitted_on_n_windows']} | {vs['val_n_normal_runs']} ({vs['val_n_normal_windows']}) | "
            f"`({vs['k']}, {vs['m']})` | <= {vs['fa_target_per_normal_run']:.2f} | "
            f"**{vs['chosen_threshold']:.6f}** | {vs['chosen_quantile_on_val_normal']:.4f} | "
            f"**{vs['val_fa_episodes_per_normal_run']:.3f}** | {vs['val_fa_windows_per_normal_run']:.2f} |"
        )
    (out_dir / "detector_thresholds.md").write_text("\n".join(t_lines) + "\n", encoding="utf-8")

    # 2. Per-class vs normal on test
    a_lines = [
        "# Per-Class vs Normal Window AUROC and Flag Recall on `test` (`SYNTHETIC / SIMULATION`)\n",
        "> **Note:** Evaluated on trainable windows of the held-out `test` split (`2,850` normal windows vs each known anomaly class). Unsupervised detectors never see class labels and output only `anomaly_score` and `unknown_anomaly`.\n",
        "| Detector | Class vs `normal` | Class Windows | Normal Windows | Window AUROC [95% CI] | Raw Window Flag Recall [95% CI] | Persisted `(k=3,m=3)` Window Flag Recall [95% CI] |",
        "| :--- | :--- | ---: | ---: | :---: | :---: | :---: |",
    ]
    for _, r in auroc_df.iterrows():
        a_lines.append(
            f"| `{r['detector']}` | `{r['class_vs_normal']}` | {int(r['n_class_windows'])} | {int(r['n_normal_windows'])} | "
            f"**{ci_str(r['auroc'], r['auroc_ci_lo'], r['auroc_ci_hi'], 3)}** | "
            f"{ci_str(r['window_flag_recall_raw'], r['window_flag_recall_raw_ci_lo'], r['window_flag_recall_raw_ci_hi'], 3)} | "
            f"{ci_str(r['window_flag_recall_persisted'], r['window_flag_recall_persisted_ci_lo'], r['window_flag_recall_persisted_ci_hi'], 3)} |"
        )
    (out_dir / "test_per_class_auroc_and_recall.md").write_text("\n".join(a_lines) + "\n", encoding="utf-8")

    # 3. Run-level detection and false alarms
    r_lines = [
        "# Run-Level Unknown Anomaly Flagging, Detection Delay, and False Alarms (`SYNTHETIC / SIMULATION`)\n",
        "| Split / Arm | Detector | Event Runs | Normal Runs | Event Flagged Rate [95% CI] | Missed Event Rate | Mean Delay (s) [95% CI] | Normal FA Episodes / Run [95% CI] | Def B Sensor FA / Hour (`1x`, `0.0667h/run`) | Def A Slow FA / Hour (`360x`, `24h/run`) |",
        "| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | ---: | ---: |",
    ]
    for _, r in run_df.iterrows():
        fl_s = ci_str(r["event_flagged_rate"], r["event_flagged_ci_lo"], r["event_flagged_ci_hi"], 3)
        dl_s = ci_str(r["mean_delay_s"], r["mean_delay_ci_lo"], r["mean_delay_ci_hi"], 2)
        fa_s = ci_str(r["fa_episodes_per_normal_run"], r["fa_episodes_ci_lo"], r["fa_episodes_ci_hi"], 3)
        ms_s = f"{r['missed_event_rate']:.3f}" if pd.notna(r["missed_event_rate"]) else "N/A"
        b_s = f"{r['B_sensor_fa_per_hour']:.3f}" if pd.notna(r["B_sensor_fa_per_hour"]) else "N/A"
        a_s = f"{r['A_slow_fa_per_hour']:.4f}" if pd.notna(r["A_slow_fa_per_hour"]) else "N/A"
        r_lines.append(
            f"| `{r['split_or_arm']}` | `{r['detector']}` | {int(r['n_event_runs'])} | {int(r['n_normal_runs'])} | "
            f"**{fl_s}** | {ms_s} | {dl_s} | {fa_s} | {b_s} | {a_s} |"
        )
    (out_dir / "run_level_detection_and_fa.md").write_text("\n".join(r_lines) + "\n", encoding="utf-8")

    # 4. SNR-binned recall
    s_lines = [
        "# Gradual & Rapid Recall by `snr_vs_imu_noise` Bin (`SYNTHETIC / SIMULATION`)\n",
        "| Split / Arm | Detector | Class | SNR Bin | Runs | Raw Window Flag Recall [95% CI] | Run-Level Flag Rate (`k=3,m=3`) [95% CI] |",
        "| :--- | :--- | :--- | :---: | ---: | :---: | :---: |",
    ]
    for _, r in snr_df.iterrows():
        w_s = ci_str(r["window_recall"], r["window_recall_ci_lo"], r["window_recall_ci_hi"], 3)
        d_s = ci_str(r["run_detect_rate"], r["run_detect_ci_lo"], r["run_detect_ci_hi"], 3)
        s_lines.append(
            f"| `{r['split_or_arm']}` | `{r['detector']}` | `{r['class']}` | `{r['snr_bin']}` | {int(r['n_runs'])} | {w_s} | **{d_s}** |"
        )
    (out_dir / "snr_bin_recall_gradual_rapid.md").write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    # 5. Complement analysis
    c_lines = [
        "# Complement Analysis at the Frozen v3 Episode Operating Point (`k=3, m=3`, `SYNTHETIC / SIMULATION`)\n",
        "- **Rule 1 (`classifier_alarm OR detector_flag`):** Run is detected if the classifier raises a persisted class alarm OR the unsupervised detector raises a persisted `unknown_anomaly` episode.",
        "- **Rule 2 (`unknown_only_when_detector_flags_and_classifier_says_normal`):** A window is flagged `unknown_anomaly` only when `anomaly_score >= threshold` AND the classifier's window prediction is `normal`, then persisted at `(k=3, m=3)`.\n",
        "| Split / Arm | Classifier | Detector | Event Runs | Clf Exact Detect | Det Flag Rate | Clf Missed Runs (`Flagged by Det`, `Frac`) | Rule 1 (`Clf OR Det`) Detect [95% CI] | Rule 2 (`Clf OR Gated Unknown`) Detect [95% CI] | Clf Alone FA / Run | Extra FA / Run (`OR` / `Gated`) | Total FA / Run (`OR` / `Gated`) | Def B Extra Sensor FA / h (`OR` / `Gated`) |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | :---: | ---: | :---: | :---: | :---: |",
    ]
    for _, r in comp_df.iterrows():
        n_m = int(r["n_clf_exact_missed"])
        n_rec = int(r["n_missed_flagged_by_det"])
        fr = f"{r['frac_of_clf_exact_missed_flagged_by_det']:.3f}" if pd.notna(r["frac_of_clf_exact_missed_flagged_by_det"]) else "N/A"
        or_d = ci_str(r["combined_or_detect_rate"], r["combined_or_detect_ci_lo"], r["combined_or_detect_ci_hi"], 3)
        gt_d = ci_str(r["gated_unknown_detect_rate"], r["gated_unknown_detect_ci_lo"], r["gated_unknown_detect_ci_hi"], 3)
        c_fa = f"{r['clf_only_fa_per_normal_run']:.3f}" if pd.notna(r["clf_only_fa_per_normal_run"]) else "N/A"
        ex_fa = (
            f"+{r['extra_fa_per_normal_run_or']:.3f} / +{r['extra_fa_per_normal_run_gated']:.3f}"
            if pd.notna(r["extra_fa_per_normal_run_or"])
            else "N/A"
        )
        tot_fa = (
            f"{r['combined_or_fa_per_normal_run']:.3f} / {r['total_fa_per_normal_run_with_gated']:.3f}"
            if pd.notna(r["combined_or_fa_per_normal_run"])
            else "N/A"
        )
        ex_b = (
            f"+{r['extra_B_sensor_fa_per_hour_or']:.2f} / +{r['extra_B_sensor_fa_per_hour_gated']:.2f}"
            if pd.notna(r["extra_B_sensor_fa_per_hour_or"])
            else "N/A"
        )
        c_lines.append(
            f"| `{r['split_or_arm']}` | `{r['classifier']}` | `{r['detector']}` | {int(r['n_event_runs'])} | "
            f"{r['clf_exact_detect_rate']:.3f} | {r['det_only_flag_rate']:.3f} | "
            f"{n_m} ({n_rec}, **{fr}**) | **{or_d}** | {gt_d} | {c_fa} | {ex_fa} | {tot_fa} | {ex_b} |"
        )
    (out_dir / "complement_analysis.md").write_text("\n".join(c_lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main Runner
# -----------------------------------------------------------------------------
def run_all_unknown_detector_evals(
    cfg_path: Path,
    out_dir: Path,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    t0 = time.time()
    hashes_before = snapshot_frozen_hashes()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = UD.load_unknown_detector_config(cfg_path)
    k = int(cfg["persistence"]["k"])
    m = int(cfg["persistence"]["m"])
    fa_target = float(cfg["persistence"]["fa_target_episodes_per_normal_run"])
    q_start = float(cfg["threshold_search"]["quantile_start"])
    q_stop = float(cfg["threshold_search"]["quantile_stop"])
    n_pts = int(cfg["threshold_search"]["num_points"])

    # 1. Load frozen datasets_v3 splits & manifest
    print("[unknown_detector] loading frozen v3 splits (train, val, test, shifted_test) ...", flush=True)
    train_df = pd.read_csv(ROOT / "outputs/datasets_v3/features_train.csv")
    val_df = pd.read_csv(ROOT / "outputs/datasets_v3/features_val.csv")
    test_df = pd.read_csv(ROOT / "outputs/datasets_v3/features_test.csv")
    shifted_df = pd.read_csv(ROOT / "outputs/datasets_v3/features_shifted_test.csv")
    man_v3 = pd.read_csv(ROOT / "outputs/datasets_v3/manifest.csv")

    feat_cols = F.feature_columns(train_df)
    M.guard_model_features(feat_cols)
    assert len(feat_cols) == 81, f"Expected 81 model features, got {len(feat_cols)}"

    # Extract strictly train normal windows and val normal runs
    train_norm_df = UD.extract_train_normal_windows(train_df)
    val_norm_df = UD.extract_val_normal_windows(val_df)
    print(
        f"[unknown_detector] fit set: {len(train_norm_df)} train normal windows | "
        f"threshold tuning set: {val_norm_df['simulation_id'].nunique()} val normal runs ({len(val_norm_df)} windows)",
        flush=True,
    )

    # 2. Fit primary (Isolation Forest) and secondary (Robust Mahalanobis) detectors & tune thresholds on val
    if_cfg = cfg["isolation_forest"]
    iso_det = UD.IsolationForestAnomalyDetector(
        n_estimators=int(if_cfg["n_estimators"]),
        max_samples=if_cfg["max_samples"],
        contamination=if_cfg["contamination"],
        clip_scaled=float(if_cfg["clip_scaled"]),
        seed=int(cfg["seed"]),
    ).fit(train_norm_df, feature_cols=feat_cols)
    iso_det.tune_threshold_on_val(
        val_norm_df, k=k, m=m, max_fa_per_normal_run=fa_target, quantile_start=q_start, quantile_stop=q_stop, num_points=n_pts
    )

    rm_cfg = cfg["robust_mahalanobis"]
    mah_det = UD.RobustMahalanobisDetector(
        trim_quantile=float(rm_cfg["trim_quantile"]),
        shrinkage=float(rm_cfg["shrinkage"]),
        min_scale=float(rm_cfg["min_scale"]),
        z_clip=float(rm_cfg["z_clip"]),
    ).fit(train_norm_df, feature_cols=feat_cols)
    mah_det.tune_threshold_on_val(
        val_norm_df, k=k, m=m, max_fa_per_normal_run=fa_target, quantile_start=q_start, quantile_stop=q_stop, num_points=n_pts
    )

    detectors = [iso_det, mah_det]
    thresh_summary = {
        "seed": int(cfg["seed"]),
        "n_model_features": len(feat_cols),
        "diag_columns_included": 0,
        "detectors": {
            d.name: {
                "fitted_on_n_windows": d.fitted_on_n_windows,
                "threshold": round(d.threshold, 6),
                "threshold_quantile_val": round(d.threshold_quantile_val, 5),
                "val_tuning_summary": d.val_tuning_summary,
            }
            for d in detectors
        },
    }
    (out_dir / "detector_thresholds.json").write_text(json.dumps(thresh_summary, indent=2), encoding="utf-8")

    # 3. Load stored feature files for shape_mismatch and literature_checks (B1 & B2)
    sm_feat, sm_man, lit_feat, lit_man = _ensure_stored_features_exist()

    # Load frozen 6-class models and operating points
    models = pickle.load(open(ROOT / "outputs/eval_v3/trained_models_v3.pkl", "rb"))
    ops = pickle.load(open(ROOT / "outputs/eval_v3/operating_points_v3.pkl", "rb"))

    # 4. Evaluate per-class vs normal AUROC and window flag recall on `test`
    print("[unknown_detector] evaluating per-class vs normal AUROC and recall on test ...", flush=True)
    auroc_frames = [
        UD.evaluate_per_class_vs_normal_on_test(d, test_df, n_boot=n_boot, seed=seed)
        for d in detectors
    ]
    auroc_df = pd.concat(auroc_frames, ignore_index=True)
    auroc_df.to_csv(out_dir / "test_per_class_auroc_and_recall.csv", index=False)

    # 5. Prepare evaluation datasets/arms across all studies
    sm_shape_map = sm_man.set_index("simulation_id")["shape"]
    sm_feat = sm_feat.copy()
    sm_feat["_shape_type"] = sm_feat["simulation_id"].map(sm_shape_map)
    sm_norm_df = sm_feat[sm_feat["run_class"] == "normal"].copy().reset_index(drop=True)

    lit_arm_map = lit_man.set_index("simulation_id")["arm"]
    lit_feat = lit_feat.copy()
    lit_feat["_arm"] = lit_feat["simulation_id"].map(lit_arm_map)
    lit_b2_cells = [
        "H100_tanB1p5",
        "H200_tanB2p5",
        "H200_tanB2p0",
        "H300_tanB2p5",
        "H200_tanB1p5",
        "H300_tanB2p0",
        "H300_tanB1p5",
    ]
    lit_depth_pooled_df = lit_feat[lit_feat["_arm"].isin(lit_b2_cells)].copy().reset_index(drop=True)

    eval_slices: list[tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]] = [
        ("val", val_df, man_v3, val_df[val_df["run_class"] == "normal"].copy().reset_index(drop=True)),
        ("test", test_df, man_v3, test_df[test_df["run_class"] == "normal"].copy().reset_index(drop=True)),
        ("shifted_test", shifted_df, man_v3, shifted_df[shifted_df["run_class"] == "normal"].copy().reset_index(drop=True)),
        ("shared_normal_40", sm_norm_df, sm_man, sm_norm_df),
    ]
    for sh in SH.SHAPES:
        sh_ev = sm_feat[(sm_feat["run_class"] != "normal") & (sm_feat["_shape_type"] == sh)].copy().reset_index(drop=True)
        sh_combined = pd.concat([sh_ev, sm_norm_df], ignore_index=True)
        eval_slices.append((f"shape:{sh}", sh_combined, sm_man, sm_norm_df))

    for l_arm in ["high_tilt_3_6", "high_tilt_6_12"]:
        l_ev = lit_feat[lit_feat["_arm"] == l_arm].copy().reset_index(drop=True)
        l_combined = pd.concat([l_ev, sm_norm_df], ignore_index=True)
        eval_slices.append((f"lit:{l_arm}", l_combined, lit_man, sm_norm_df))

    eval_slices.append(
        ("lit:depth_all_gt_1p5", pd.concat([lit_depth_pooled_df, sm_norm_df], ignore_index=True), lit_man, sm_norm_df)
    )
    for cell in lit_b2_cells:
        c_ev = lit_feat[lit_feat["_arm"] == cell].copy().reset_index(drop=True)
        eval_slices.append((f"lit:{cell}", pd.concat([c_ev, sm_norm_df], ignore_index=True), lit_man, sm_norm_df))

    # 6. Run-level detection, SNR-bin recall, and complement analysis across all slices
    print("[unknown_detector] running run-level detection, SNR-bin recall, and complement analysis ...", flush=True)
    run_rows = []
    snr_rows = []
    comp_rows = []

    for split_name, df_all, man_cur, df_norm in eval_slices:
        for d in detectors:
            res_r = UD.evaluate_run_level_unknown(d, df_all, split_name, n_boot=n_boot, seed=seed)
            res_r_clean = {k_: v_ for k_, v_ in res_r.items() if k_ != "per_run_df"}
            run_rows.append(res_r_clean)

            if split_name != "shared_normal_40":
                snr_rows.extend(
                    UD.evaluate_snr_bin_recall_unknown(d, df_all, man_cur, split_name, n_boot=n_boot, seed=seed)
                )

        if split_name not in ("val", "shared_normal_40"):
            df_ev_only = df_all[df_all["run_class"] != "normal"].copy().reset_index(drop=True)
            for mname in MODELS:
                w_ev, k_op, m_op = _clf_episode_window_preds(mname, models[mname], ops, df_ev_only)
                if df_norm is not None and not df_norm.empty:
                    w_norm, _, _ = _clf_episode_window_preds(mname, models[mname], ops, df_norm)
                    df_norm_with_p = df_norm.copy()
                    df_norm_with_p["_clf_wpred"] = w_norm
                else:
                    df_norm_with_p = None

                for d in detectors:
                    comp_rows.append(
                        UD.evaluate_complement_with_classifier(
                            detector=d,
                            clf_name=mname,
                            clf_window_preds=w_ev,
                            df_events=df_ev_only,
                            df_normals=df_norm_with_p,
                            split_name=split_name,
                            k=k_op,
                            m=m_op,
                            n_boot=n_boot,
                            seed=seed,
                        )
                    )

    run_df = pd.DataFrame(run_rows)
    snr_df = pd.DataFrame(snr_rows)
    comp_df = pd.DataFrame(comp_rows)

    run_df.to_csv(out_dir / "run_level_detection_and_fa.csv", index=False)
    snr_df.to_csv(out_dir / "snr_bin_recall_gradual_rapid.csv", index=False)
    comp_df.to_csv(out_dir / "complement_analysis.csv", index=False)

    # 7. Write Markdown tables and render plots
    _write_markdown_tables(out_dir, thresh_summary, auroc_df, run_df, snr_df, comp_df)
    plot_test_auroc_and_snr_recall(auroc_df, snr_df, out_dir / "unknown_test_auroc_and_snr_recall.png")
    plot_run_detection_across_regimes(comp_df, out_dir / "unknown_run_detection_across_regimes.png")
    plot_complement_tradeoff(comp_df, out_dir / "unknown_complement_tradeoff.png")

    # 8. Verify frozen hashes unchanged
    hashes_after = snapshot_frozen_hashes()
    assert hashes_before == hashes_after, "FROZEN FILE MODIFIED DURING UNKNOWN DETECTOR EVALUATION!"

    summary = {
        "elapsed_s": round(time.time() - t0, 2),
        "frozen_files_verified": len(hashes_after),
        "thresholds": thresh_summary["detectors"],
        "git": VR.git_state(ROOT),
        "libraries": VR.library_versions(),
    }
    (out_dir / "unknown_detector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    VR.write_dataset_metadata(out_dir, config_files={"unknown_detector": cfg_path}, repo_dir=ROOT)
    print(f"[unknown_detector] completed in {summary['elapsed_s']:.1f}s -> {out_dir}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Run unsupervised unknown anomaly detector evaluation.")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/unknown_detector.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/eval_unknown")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    run_all_unknown_detector_evals(args.config, args.out, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
