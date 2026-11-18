"""End-to-end evaluation script (SYNTHETIC / SIMULATION):
  1. Loads frozen datasets (`train`, `val`, `test`, `shifted_test`, `stress`, `broad_basin`) and `manifest.csv`.
  2. Fits ThresholdRule, LogisticRegression (balanced), and RandomForest (balanced, grid-tuned on `val` only).
  3. Evaluates window-level metrics (macro-F1, per-class P/R/F1, confusion matrices) with 95% CIs bootstrapped over RUNS.
  4. Evaluates run-level detection rate, false alarms per normal run, and detection delay (using transition offsets).
  5. Evaluates recall vs `snr_vs_imu_noise` bins (`<0.5`, `0.5-1`, `1-2`, `2-5`, `>5`) as the detection limit.
  6. Runs 6 feature-group ablations for Random Forest (`direction`, `coherence`, `piezo`, `hub`, `comm`, `long_horizon`).
  7. Runs hard-pair failure analysis across all 4 pairs with 3 failure patterns + example windows each.
  8. Explicitly reports every metric/class/split where Random Forest does NOT beat the Threshold Rule, with 95% CIs.
  9. Saves trained models, exact configs, CSV/Markdown tables, and plots (with red SYNTHETIC / SIMULATION banner & compression factor) to `outputs/eval/`.

Usage: python scripts/run_eval.py [--data DIR] [--out DIR] [--seed N] [--n-boot N]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canary_sim import (
    classes as C,
    evaluate as ev,
    features as F,
    labels as L,
    models as M,
)

BANNER = "SYNTHETIC / SIMULATION"
EVAL_SPLITS = ("test", "shifted_test", "stress", "broad_basin")
MODEL_NAMES = ("threshold_rule", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "threshold_rule": "Threshold Rule",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
}


def stamp_banner(fig, comp_grad: float = 360.0, comp_rap: float = 1.0, subtitle: str = "") -> None:
    fig.suptitle(
        f"{BANNER}  |  time compression: gradual x{comp_grad:g} (240 s = 24 h), rapid x{comp_rap:g} (real-time)\n{subtitle}",
        color="red",
        fontsize=11,
        fontweight="bold",
    )


def plot_confusion_matrices(eval_by_split_model: dict, out_dir: Path, comp_grad: float, comp_rap: float) -> Path:
    fig, axes = plt.subplots(len(EVAL_SPLITS), len(MODEL_NAMES), figsize=(16, 17))
    cls_short = ["norm", "grad", "rapid", "abn_vib", "sens_flt", "comm_flt"]

    for r_i, sp in enumerate(EVAL_SPLITS):
        for c_j, mname in enumerate(MODEL_NAMES):
            ax = axes[r_i, c_j]
            res = eval_by_split_model[sp][mname]["window"]
            cm = res["confusion_matrix"]
            row_sums = np.maximum(cm.sum(axis=1, keepdims=True), 1)
            cm_norm = cm / row_sums
            im = ax.imshow(cm_norm, vmin=0.0, vmax=1.0, cmap="Blues")
            for ii in range(len(cls_short)):
                for jj in range(len(cls_short)):
                    val = cm[ii, jj]
                    if val > 0 or ii == jj:
                        col = "white" if cm_norm[ii, jj] > 0.52 else "black"
                        ax.text(jj, ii, f"{val}", ha="center", va="center", color=col, fontsize=7.5)
            f1 = res["macro_f1"]
            lo, hi = res["macro_f1_ci95"]
            ax.set_title(
                f"{sp} | {MODEL_LABELS[mname]}\nmacro-F1 = {f1:.3f} [{lo:.3f}, {hi:.3f}]",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.text(0.5, 1.16, BANNER, transform=ax.transAxes, color="red", ha="center", fontsize=8, fontweight="bold")
            ax.set_xticks(range(6))
            ax.set_yticks(range(6))
            ax.set_xticklabels(cls_short, rotation=30, ha="right", fontsize=8)
            ax.set_yticklabels(cls_short, fontsize=8)
            if c_j == 0:
                ax.set_ylabel("True class", fontsize=9)
            if r_i == len(EVAL_SPLITS) - 1:
                ax.set_xlabel("Predicted class", fontsize=9)

    stamp_banner(fig, comp_grad, comp_rap, "Window-Level Confusion Matrices Across Evaluation Splits (Counts & Row-Normalised Shading)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p = out_dir / "plot_confusion_matrices.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_recall_vs_snr(snr_df: pd.DataFrame, out_dir: Path, comp_grad: float, comp_rap: float) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=True)
    colors = {"threshold_rule": "tab:orange", "logistic_regression": "tab:green", "random_forest": "tab:blue"}
    markers = {"threshold_rule": "s", "logistic_regression": "^", "random_forest": "o"}
    bins_order = [b[0] for b in ev.SNR_BINS]
    x = np.arange(len(bins_order))

    for r_i, sp in enumerate(["test", "broad_basin"]):
        for c_j, cls in enumerate(["gradual", "rapid"]):
            ax = axes[r_i, c_j]
            for off_i, mname in enumerate(MODEL_NAMES):
                sub = snr_df[(snr_df["split"] == sp) & (snr_df["class"] == cls) & (snr_df["model"] == mname)]
                sub = sub.set_index("snr_bin").reindex(bins_order).reset_index()
                y = sub["exact_recall"].to_numpy(dtype=float)
                lo = sub["exact_recall_ci_lo"].to_numpy(dtype=float)
                hi = sub["exact_recall_ci_hi"].to_numpy(dtype=float)
                dx = (off_i - 1) * 0.08
                yerr = np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)])
                ax.errorbar(
                    x + dx,
                    y,
                    yerr=yerr,
                    marker=markers[mname],
                    color=colors[mname],
                    lw=1.6,
                    capsize=3,
                    label=MODEL_LABELS[mname],
                )
            ax.axhline(0.5, color="grey", ls=":", lw=1.0, label="50% recall")
            ax.axvline(1.5, color="red", ls="--", lw=1.0, alpha=0.6, label="SNR = 1 boundary")
            ax.set_xticks(x)
            ax.set_xticklabels(bins_order)
            ax.set_ylim(-0.05, 1.08)
            ax.set_title(f"Split: {sp}  |  Class: {cls}", fontsize=10.5, fontweight="bold")
            ax.text(0.5, 1.09, BANNER, transform=ax.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
            if c_j == 0:
                ax.set_ylabel("Window recall (95% run-bootstrap CI)")
            if r_i == 1:
                ax.set_xlabel("snr_vs_imu_noise bin (peak_tilt_deg / imu_noise_sd_deg)")
            ax.grid(alpha=0.25)
            if r_i == 0 and c_j == 0:
                ax.legend(fontsize=8, loc="lower right")

    stamp_banner(fig, comp_grad, comp_rap, "Detection Limit: Window Recall vs snr_vs_imu_noise Bins (with 95% Run-Bootstrap CIs)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = out_dir / "plot_recall_vs_snr.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_ablation_bars(ablation_df: pd.DataFrame, out_dir: Path, comp_grad: float, comp_rap: float) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sub = ablation_df[ablation_df["group_removed"] != "none (full_model)"].copy()
    groups = sub["group_removed"].tolist()
    y_pos = np.arange(len(groups))

    # Left panel: delta Macro-F1 with 95% CI
    ax0 = axes[0]
    dm = sub["delta_macro_f1"].to_numpy(dtype=float)
    lo = sub["delta_macro_f1_ci_lo"].to_numpy(dtype=float)
    hi = sub["delta_macro_f1_ci_hi"].to_numpy(dtype=float)
    xerr = np.vstack([np.maximum(0, dm - lo), np.maximum(0, hi - dm)])
    ax0.barh(y_pos, dm, xerr=xerr, color="tab:red", alpha=0.8, capsize=4, edgecolor="k")
    ax0.axvline(0.0, color="black", lw=1.0)
    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([f"- {g}" for g in groups], fontsize=9.5)
    ax0.set_xlabel("Change in Test Macro-F1 (ablated - full RF, 95% run-bootstrap CI)")
    ax0.set_title("Macro-F1 Drop When Removing Each Feature Group", fontsize=10.5, fontweight="bold")
    ax0.text(0.5, 1.07, BANNER, transform=ax0.transAxes, color="red", ha="center", fontsize=9, fontweight="bold")
    ax0.grid(axis="x", alpha=0.3)

    # Right panel: heatmap of per-class recall change
    ax1 = axes[1]
    rec_cols = [f"delta_recall_{c}" for c in C.CLASSES]
    mat = sub[rec_cols].to_numpy(dtype=float)
    im = ax1.imshow(mat, cmap="RdBu", vmin=-0.45, vmax=0.45, aspect="auto")
    for i in range(len(groups)):
        for j in range(len(C.CLASSES)):
            v = mat[i, j]
            ax1.text(j, i, f"{v:+.2f}", ha="center", va="center", color="white" if abs(v) > 0.24 else "black", fontsize=8.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"- {g}" for g in groups], fontsize=9.5)
    ax1.set_xticks(range(len(C.CLASSES)))
    ax1.set_xticklabels(list(C.CLASSES), rotation=28, ha="right", fontsize=9)
    ax1.set_title("Per-Class Test Recall Change (Ablated - Full RF)", fontsize=10.5, fontweight="bold")
    ax1.text(0.5, 1.07, BANNER, transform=ax1.transAxes, color="red", ha="center", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax1, label="Delta recall")

    stamp_banner(fig, comp_grad, comp_rap, "Random Forest Feature-Group Ablations on Test Split")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    p = out_dir / "plot_rf_ablations.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def plot_detection_delay(run_eval_df: pd.DataFrame, out_dir: Path, comp_grad: float, comp_rap: float) -> Path:
    sub = run_eval_df[(run_eval_df["split"] == "test") & (run_eval_df["class"] != "ALL_EVENTS")].copy()
    ev_classes = [c for c in C.CLASSES if c != "normal"]
    x = np.arange(len(ev_classes))
    width = 0.25
    colors = {"threshold_rule": "tab:orange", "logistic_regression": "tab:green", "random_forest": "tab:blue"}

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    for j, mname in enumerate(MODEL_NAMES):
        m_sub = sub[sub["model"] == mname].set_index("class").reindex(ev_classes).reset_index()
        # Left: Run-level exact detection rate
        rate = m_sub["exact_detect_rate"].to_numpy(dtype=float)
        r_lo = m_sub["exact_detect_ci_lo"].to_numpy(dtype=float)
        r_hi = m_sub["exact_detect_ci_hi"].to_numpy(dtype=float)
        axes[0].bar(
            x + (j - 1) * width,
            rate,
            width=width,
            yerr=np.vstack([np.maximum(0, rate - r_lo), np.maximum(0, r_hi - rate)]),
            capsize=3,
            color=colors[mname],
            alpha=0.85,
            label=MODEL_LABELS[mname],
        )
        # Right: Mean detection delay (s) using transition offsets
        dly = m_sub["mean_delay_exact_s"].to_numpy(dtype=float)
        d_lo = m_sub["delay_exact_ci_lo"].to_numpy(dtype=float)
        d_hi = m_sub["delay_exact_ci_hi"].to_numpy(dtype=float)
        axes[1].bar(
            x + (j - 1) * width,
            dly,
            width=width,
            yerr=np.vstack([np.maximum(0, dly - d_lo), np.maximum(0, d_hi - dly)]),
            capsize=3,
            color=colors[mname],
            alpha=0.85,
            label=MODEL_LABELS[mname],
        )

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ev_classes, rotation=22, ha="right")
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("Run-level detection rate (95% CI)")
    axes[0].set_title("Test Run-Level Event/Fault Detection Rate", fontsize=10.5, fontweight="bold")
    axes[0].text(0.5, 1.08, BANNER, transform=axes[0].transAxes, color="red", ha="center", fontsize=9, fontweight="bold")
    axes[0].legend(fontsize=8.5, loc="lower right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ev_classes, rotation=22, ha="right")
    axes[1].set_ylabel("Mean detection delay from onset (sim s, 95% CI)")
    axes[1].set_title("Test Detection Delay (Using Transition Window Offsets)", fontsize=10.5, fontweight="bold")
    axes[1].text(0.5, 1.08, BANNER, transform=axes[1].transAxes, color="red", ha="center", fontsize=9, fontweight="bold")
    axes[1].legend(fontsize=8.5, loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)

    stamp_banner(fig, comp_grad, comp_rap, "Run-Level Detection Rate and Detection Delay on Test Split")
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    p = out_dir / "plot_detection_delay.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "outputs" / "datasets"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()

    data_dir = Path(a.data)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen datasets (do NOT regenerate or modify)
    manifest = pd.read_csv(data_dir / "manifest.csv")
    splits_df = {
        sp: pd.read_csv(data_dir / f"features_{sp}.csv")
        for sp in ("train", "val", "test", "shifted_test", "stress", "broad_basin")
    }
    comp_grad = float(manifest[manifest.run_class == "gradual"]["compression_factor"].iloc[0])
    comp_rap = float(manifest[manifest.run_class == "rapid"]["compression_factor"].iloc[0])

    train_df = splits_df["train"]
    val_df = splits_df["val"]
    model_cols = F.feature_columns(train_df)
    M.guard_model_features(model_cols)

    # 2. Fit models on train and tune on val ONLY (enforced by M.guard_no_test_data)
    th_clf = M.ThresholdRuleClassifier().fit(train_df, val_df)
    lr_clf = M.fit_logistic_regression(train_df, val_df, seed=a.seed, feature_cols=model_cols)
    rf_clf = M.fit_random_forest(train_df, val_df, seed=a.seed, feature_cols=model_cols)

    models = {
        "threshold_rule": th_clf,
        "logistic_regression": lr_clf,
        "random_forest": rf_clf,
    }

    # Save trained models and exact configs
    with open(out_dir / "trained_models.pkl", "wb") as f:
        pickle.dump(models, f)

    configs_saved = {
        "seed": a.seed,
        "n_boot": a.n_boot,
        "n_model_features": len(model_cols),
        "model_features": model_cols,
        "feature_groups": M.FEATURE_GROUPS,
        "threshold_rule": {
            "chosen_thresholds": th_clf.thresholds,
            "val_tuning_history": th_clf.tuning_history,
        },
        "logistic_regression": {
            "best_params": lr_clf.best_params,
            "val_grid": lr_clf.tuning_grid_results,
        },
        "random_forest": {
            "best_params": rf_clf.best_params,
            "val_grid": rf_clf.tuning_grid_results,
        },
    }
    with open(out_dir / "eval_configs_and_hyperparams.json", "w", encoding="utf-8") as f:
        json.dump(configs_saved, f, indent=2)

    # 3. Evaluate across val, test, shifted_test, stress, broad_basin
    eval_by_split_model: dict[str, dict] = {}
    win_summary_rows = []
    per_class_rows = []
    run_eval_rows = []
    snr_rows = []
    stress_rows = []
    rf_vs_th_comparisons = []

    for sp in ("val", *EVAL_SPLITS):
        df_sp = splits_df[sp]
        eval_by_split_model[sp] = {}
        preds_map = {}
        for mname, clf in models.items():
            pr = clf.predict(df_sp)
            preds_map[mname] = pr
            w_res = ev.evaluate_window_level_with_run_bootstrap(df_sp, pr, n_boot=a.n_boot, seed=a.seed)
            r_res = ev.evaluate_run_level_detection(df_sp, pr, n_boot=a.n_boot, seed=a.seed)
            eval_by_split_model[sp][mname] = {"window": w_res, "run": r_res, "preds": pr}

            win_summary_rows.append({
                "split": sp,
                "model": mname,
                "n_runs": w_res["n_runs"],
                "n_trainable_windows": w_res["n_windows"],
                "macro_f1": round(w_res["macro_f1"], 4),
                "macro_f1_ci_lo": round(w_res["macro_f1_ci95"][0], 4),
                "macro_f1_ci_hi": round(w_res["macro_f1_ci95"][1], 4),
                "run_exact_detect_rate": round(r_res["exact_detect_rate"], 4),
                "run_exact_detect_ci_lo": round(r_res["exact_detect_rate_ci95"][0], 4),
                "run_exact_detect_ci_hi": round(r_res["exact_detect_rate_ci95"][1], 4),
                "run_any_detect_rate": round(r_res["any_detect_rate"], 4),
                "mean_delay_exact_s": round(r_res["mean_delay_exact_s"], 3),
                "delay_exact_ci_lo": round(r_res["mean_delay_exact_s_ci95"][0], 3),
                "delay_exact_ci_hi": round(r_res["mean_delay_exact_s_ci95"][1], 3),
                "fa_windows_per_normal_run": round(r_res["false_alarms_per_normal_run"], 3),
                "fa_windows_ci_lo": round(r_res["false_alarms_per_normal_run_ci95"][0], 3),
                "fa_windows_ci_hi": round(r_res["false_alarms_per_normal_run_ci95"][1], 3),
            })

            for cls, cinfo in w_res["per_class"].items():
                if cinfo["support"] == 0:
                    continue
                per_class_rows.append({
                    "split": sp,
                    "model": mname,
                    "class": cls,
                    "support": cinfo["support"],
                    "precision": round(cinfo["precision"], 4),
                    "precision_ci_lo": round(cinfo["precision_ci95"][0], 4),
                    "precision_ci_hi": round(cinfo["precision_ci95"][1], 4),
                    "recall": round(cinfo["recall"], 4),
                    "recall_ci_lo": round(cinfo["recall_ci95"][0], 4),
                    "recall_ci_hi": round(cinfo["recall_ci95"][1], 4),
                    "f1": round(cinfo["f1"], 4),
                    "f1_ci_lo": round(cinfo["f1_ci95"][0], 4),
                    "f1_ci_hi": round(cinfo["f1_ci95"][1], 4),
                })

            run_eval_rows.append({
                "split": sp,
                "model": mname,
                "class": "ALL_EVENTS",
                "n_runs": r_res["n_event_runs"],
                "exact_detect_rate": round(r_res["exact_detect_rate"], 4),
                "exact_detect_ci_lo": round(r_res["exact_detect_rate_ci95"][0], 4),
                "exact_detect_ci_hi": round(r_res["exact_detect_rate_ci95"][1], 4),
                "any_detect_rate": round(r_res["any_detect_rate"], 4),
                "mean_delay_exact_s": round(r_res["mean_delay_exact_s"], 3),
                "delay_exact_ci_lo": round(r_res["mean_delay_exact_s_ci95"][0], 3),
                "delay_exact_ci_hi": round(r_res["mean_delay_exact_s_ci95"][1], 3),
            })
            for cls, rc in r_res["per_class"].items():
                run_eval_rows.append({
                    "split": sp,
                    "model": mname,
                    "class": cls,
                    "n_runs": rc["n_runs"],
                    "exact_detect_rate": round(rc["exact_detect_rate"], 4),
                    "exact_detect_ci_lo": round(rc["exact_detect_rate_ci95"][0], 4),
                    "exact_detect_ci_hi": round(rc["exact_detect_rate_ci95"][1], 4),
                    "any_detect_rate": round(rc["any_detect_rate"], 4),
                    "mean_delay_exact_s": round(rc["mean_delay_exact_s"], 3),
                    "delay_exact_ci_lo": round(rc["mean_delay_exact_s_ci95"][0], 3),
                    "delay_exact_ci_hi": round(rc["mean_delay_exact_s_ci95"][1], 3),
                })

            if sp in ("test", "shifted_test", "broad_basin", "stress"):
                s_df = ev.evaluate_recall_vs_snr(df_sp, pr, manifest, n_boot=a.n_boot, seed=a.seed)
                s_df.insert(0, "model", mname)
                s_df.insert(0, "split", sp)
                snr_rows.append(s_df)

            if sp == "stress":
                st_res = ev.evaluate_stress_set(df_sp, pr, n_boot=a.n_boot, seed=a.seed)
                stress_rows.append({
                    "model": mname,
                    "primary_macro_f1": round(st_res["primary_eval"]["macro_f1"], 4),
                    "primary_macro_f1_ci_lo": round(st_res["primary_eval"]["macro_f1_ci95"][0], 4),
                    "primary_macro_f1_ci_hi": round(st_res["primary_eval"]["macro_f1_ci95"][1], 4),
                    "set_match_accuracy_all": round(st_res["set_match_accuracy_all"], 4),
                    "n_multi_condition_windows": st_res["n_multi_condition_windows"],
                    "primary_accuracy_on_multi_windows": round(st_res["primary_accuracy_on_multi_windows"], 4),
                    "set_match_accuracy_on_multi_windows": round(st_res["set_match_accuracy_on_multi_windows"], 4),
                })

        # Compare Random Forest vs Threshold Rule paired across bootstrap replicates on this split
        rf_w = eval_by_split_model[sp]["random_forest"]["window"]
        th_w = eval_by_split_model[sp]["threshold_rule"]["window"]
        diff_macro = rf_w["boot_macro_f1_samples"] - th_w["boot_macro_f1_samples"]
        pt_diff = rf_w["macro_f1"] - th_w["macro_f1"]
        rf_vs_th_comparisons.append({
            "split": sp,
            "metric": "macro_f1",
            "rf_val": round(rf_w["macro_f1"], 4),
            "th_val": round(th_w["macro_f1"], 4),
            "rf_minus_th": round(pt_diff, 4),
            "diff_ci_lo": round(float(np.percentile(diff_macro, 2.5)), 4),
            "diff_ci_hi": round(float(np.percentile(diff_macro, 97.5)), 4),
            "rf_fails_to_beat_th": bool(pt_diff <= 0.0 or np.percentile(diff_macro, 2.5) <= 0.0),
        })
        for i, cls in enumerate(C.CLASSES):
            if rf_w["per_class"][cls]["support"] == 0:
                continue
            rf_r = rf_w["per_class"][cls]["recall"]
            th_r = th_w["per_class"][cls]["recall"]
            d_rec = rf_w["boot_rec_samples"][:, i] - th_w["boot_rec_samples"][:, i]
            pt_dr = rf_r - th_r
            rf_vs_th_comparisons.append({
                "split": sp,
                "metric": f"recall_{cls}",
                "rf_val": round(rf_r, 4),
                "th_val": round(th_r, 4),
                "rf_minus_th": round(pt_dr, 4),
                "diff_ci_lo": round(float(np.percentile(d_rec, 2.5)), 4),
                "diff_ci_hi": round(float(np.percentile(d_rec, 97.5)), 4),
                "rf_fails_to_beat_th": bool(pt_dr <= 0.0 or np.percentile(d_rec, 2.5) <= 0.0),
            })

    # 4. Random Forest Feature-Group Ablations (evaluated on test split with paired run-bootstrap CIs)
    test_df = splits_df["test"]
    full_rf_test = eval_by_split_model["test"]["random_forest"]["window"]
    ablation_rows = []
    base_row = {
        "group_removed": "none (full_model)",
        "n_features_kept": len(model_cols),
        "macro_f1": round(full_rf_test["macro_f1"], 4),
        "macro_f1_ci_lo": round(full_rf_test["macro_f1_ci95"][0], 4),
        "macro_f1_ci_hi": round(full_rf_test["macro_f1_ci95"][1], 4),
        "delta_macro_f1": 0.0,
        "delta_macro_f1_ci_lo": 0.0,
        "delta_macro_f1_ci_hi": 0.0,
    }
    for cls in C.CLASSES:
        base_row[f"recall_{cls}"] = round(full_rf_test["per_class"][cls]["recall"], 4)
        base_row[f"delta_recall_{cls}"] = 0.0
    ablation_rows.append(base_row)

    chosen_rf_params = {
        k: rf_clf.best_params[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")
    }
    for grp in M.FEATURE_GROUPS:
        kept_cols = M.features_without_group(model_cols, grp)
        ab_clf = M.fit_random_forest(
            train_df,
            val_df=None,
            seed=a.seed,
            feature_cols=kept_cols,
            fixed_params=chosen_rf_params,
        )
        ab_preds = ab_clf.predict(test_df)
        ab_res = ev.evaluate_window_level_with_run_bootstrap(test_df, ab_preds, n_boot=a.n_boot, seed=a.seed)
        d_boot = ab_res["boot_macro_f1_samples"] - full_rf_test["boot_macro_f1_samples"]
        row = {
            "group_removed": grp,
            "n_features_kept": len(kept_cols),
            "macro_f1": round(ab_res["macro_f1"], 4),
            "macro_f1_ci_lo": round(ab_res["macro_f1_ci95"][0], 4),
            "macro_f1_ci_hi": round(ab_res["macro_f1_ci95"][1], 4),
            "delta_macro_f1": round(ab_res["macro_f1"] - full_rf_test["macro_f1"], 4),
            "delta_macro_f1_ci_lo": round(float(np.percentile(d_boot, 2.5)), 4),
            "delta_macro_f1_ci_hi": round(float(np.percentile(d_boot, 97.5)), 4),
        }
        for i, cls in enumerate(C.CLASSES):
            r_ab = ab_res["per_class"][cls]["recall"]
            r_full = full_rf_test["per_class"][cls]["recall"]
            dr_boot = ab_res["boot_rec_samples"][:, i] - full_rf_test["boot_rec_samples"][:, i]
            row[f"recall_{cls}"] = round(r_ab, 4)
            row[f"delta_recall_{cls}"] = round(r_ab - r_full, 4)
            row[f"delta_recall_{cls}_ci_lo"] = round(float(np.percentile(dr_boot, 2.5)), 4)
            row[f"delta_recall_{cls}_ci_hi"] = round(float(np.percentile(dr_boot, 97.5)), 4)
        ablation_rows.append(row)

    # 5. Hard-pair analysis on test + shifted_test
    hard_pairs_test = ev.analyze_hard_pairs(
        test_df,
        {m: eval_by_split_model["test"][m]["preds"] for m in MODEL_NAMES},
        manifest,
    )
    with open(out_dir / "hard_pairs_analysis.json", "w", encoding="utf-8") as f:
        json.dump(hard_pairs_test, f, indent=2)

    # Convert tables to DataFrames and write CSV + Markdown
    win_summary_df = pd.DataFrame(win_summary_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    run_eval_df = pd.DataFrame(run_eval_rows)
    snr_all_df = pd.concat(snr_rows, ignore_index=True)
    stress_df_tab = pd.DataFrame(stress_rows)
    ablation_df = pd.DataFrame(ablation_rows)
    rf_vs_th_df = pd.DataFrame(rf_vs_th_comparisons)

    def _df_to_md(df_in: pd.DataFrame) -> str:
        cols_h = [str(c) for c in df_in.columns]
        lines = [
            "| " + " | ".join(cols_h) + " |",
            "| " + " | ".join(["---"] * len(cols_h)) + " |",
        ]
        for _, r in df_in.iterrows():
            vals = []
            for v in r:
                if isinstance(v, float):
                    vals.append("NaN" if np.isnan(v) else f"{v:g}")
                else:
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    for name, df_tab in [
        ("summary_by_split", win_summary_df),
        ("per_class_metrics", per_class_df),
        ("run_level_detection", run_eval_df),
        ("recall_vs_snr", snr_all_df),
        ("stress_set_scoring", stress_df_tab),
        ("rf_ablations", ablation_df),
        ("rf_vs_threshold_comparison", rf_vs_th_df),
    ]:
        df_tab.to_csv(out_dir / f"{name}.csv", index=False)
        with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
            f.write(f"# {name} (SYNTHETIC / SIMULATION)\n\n")
            f.write(_df_to_md(df_tab))
            f.write("\n")

    # 6. Plots
    p_cm = plot_confusion_matrices(eval_by_split_model, out_dir, comp_grad, comp_rap)
    p_snr = plot_recall_vs_snr(snr_all_df, out_dir, comp_grad, comp_rap)
    p_ab = plot_ablation_bars(ablation_df, out_dir, comp_grad, comp_rap)
    p_dl = plot_detection_delay(run_eval_df, out_dir, comp_grad, comp_rap)

    print("=== WINDOW & RUN SUMMARY ACROSS SPLITS ===")
    print(win_summary_df.to_string(index=False))
    print("\n=== RANDOM FOREST FEATURE-GROUP ABLATIONS (TEST) ===")
    print(ablation_df[["group_removed", "n_features_kept", "macro_f1", "delta_macro_f1", "delta_macro_f1_ci_lo", "delta_macro_f1_ci_hi"]].to_string(index=False))
    print("\n=== WHERE RANDOM FOREST DOES NOT BEAT THRESHOLD RULE (RF - TH <= 0 OR 95% CI INCLUDES <= 0) ===")
    non_beat = rf_vs_th_df[rf_vs_th_df["rf_fails_to_beat_th"]]
    print(non_beat.to_string(index=False))
    print("\nSaved plots:", p_cm, p_snr, p_ab, p_dl)


if __name__ == "__main__":
    main()
