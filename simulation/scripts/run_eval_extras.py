#!/usr/bin/env python
"""Runner for evaluation extras (SYNTHETIC / SIMULATION).

Usage:
    python scripts/run_eval_extras.py --data outputs/datasets_v3 --models outputs/eval_v3/trained_models_v3.pkl --ops outputs/eval_v3/operating_points_v3.pkl --out outputs/eval_extras/
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import eval_extras as EE, features as F, labels as L  # noqa: E402


def _add_synthetic_banner(fig: plt.Figure, title_suffix: str = ""):
    """Add standardized red SYNTHETIC / SIMULATION banner with compression factor."""
    banner_text = (
        "SYNTHETIC / SIMULATION — Time Compression: 360x (Normal Runs: 24h sim / run) | 1x (Rapid Runs)"
    )
    if title_suffix:
        banner_text += f" | {title_suffix}"
    fig.text(
        0.5,
        0.97,
        banner_text,
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#d32f2f", edgecolor="none"),
    )


def plot_reliability_diagram(top1_df: pd.DataFrame, out_path: Path):
    """Plot reliability diagram for Random Forest and Logistic Regression on VAL."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    models_info = [
        ("random_forest", "Random Forest (Frozen v3)", axes[0], "#1976d2"),
        ("logistic_regression", "Logistic Regression (Frozen v3)", axes[1], "#388e3c"),
    ]

    for mname, mtitle, ax, col in models_info:
        sub = top1_df[top1_df["model"] == mname].dropna(subset=["mean_model_score", "empirical_accuracy"])
        x = sub["mean_model_score"].to_numpy()
        y = sub["empirical_accuracy"].to_numpy()
        counts = sub["window_count"].to_numpy()

        # Diagonal perfect calibration
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect Calibration")
        ax.plot(x, y, marker="o", linewidth=2.2, markersize=7, color=col, label="Observed Accuracy")

        # Bar chart of sample counts on twin axis
        ax2 = ax.twinx()
        ax2.bar(x, counts, width=0.04, alpha=0.18, color=col, label="Window Count")
        ax2.set_ylabel("Window Count", fontsize=9, color="#555555")
        ax2.tick_params(axis="y", labelsize=8, labelcolor="#555555")
        ax2.grid(False)

        ax.set_title(mtitle, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Mean model_score (Top-1)", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("Empirical Accuracy", fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.28, left=0.08, right=0.92)
    _add_synthetic_banner(fig, "Score Reliability on VAL")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_false_alarm_vs_detection(fa_df: pd.DataFrame, out_path: Path):
    """Plot trade-off between False Alarms per simulated hour and Missed Event Rate."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    splits = [("test", "Test Split", axes[0]), ("shifted_test", "Shifted Test Split", axes[1])]

    markers = {"threshold_rule": "s", "logistic_regression": "^", "random_forest": "o"}
    colors = {"threshold_rule": "#f57c00", "logistic_regression": "#388e3c", "random_forest": "#1976d2"}

    for sp, title, ax in splits:
        sub = fa_df[fa_df["split"] == sp]
        for mname in ("threshold_rule", "logistic_regression", "random_forest"):
            msub = sub[sub["model"] == mname]
            x = msub["fa_episodes_per_sim_hour"].to_numpy()
            y = msub["missed_event_rate"].to_numpy()
            labels = msub["regime"].to_numpy()

            ax.plot(x, y, linestyle="-", alpha=0.4, color=colors[mname])
            for xi, yi, lbl in zip(x, y, labels):
                tag = "Raw" if "raw" in lbl else ("Persist" if "episode" in lbl else "Strict")
                ax.scatter(xi, yi, marker=markers[mname], s=90, color=colors[mname], label=mname if lbl == labels[0] else "")
                ax.annotate(f"{mname[:2].upper()}-{tag}", (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

        ax.set_title(title, fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("False Alarm Episodes per Simulated Hour", fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.6)

    axes[0].set_ylabel("Missed Event Rate (1 - Detection Rate)", fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.14, wspace=0.22, left=0.08, right=0.95)
    _add_synthetic_banner(fig, "FA / Simulated Hour vs Missed Events")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_permutation_importance(top20_df: pd.DataFrame, groups_df: pd.DataFrame, out_path: Path):
    """Plot top 20 individual features and group importance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))

    # Top 20 features
    t20 = top20_df.sort_values("mean_importance", ascending=True)
    y_pos = np.arange(len(t20))
    ax1.barh(y_pos, t20["mean_importance"], xerr=t20["std_importance"], color="#1976d2", alpha=0.85, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(t20["feature"], fontsize=8.5)
    ax1.set_xlabel("Mean Macro-F1 Drop on VAL (Whole-Run Permutation)", fontsize=9.5)
    ax1.set_title("Top 20 Features (Random Forest on VAL)", fontsize=11, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")

    # Feature groups
    grp = groups_df.sort_values("mean_importance", ascending=True)
    y_pos2 = np.arange(len(grp))
    ax2.barh(y_pos2, grp["mean_importance"], xerr=grp["std_importance"], color="#00897b", alpha=0.85, capsize=4)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([f"{g} ({n} feat)" for g, n in zip(grp["group"], grp["feature_count"])], fontsize=9.5)
    ax2.set_xlabel("Mean Macro-F1 Drop on VAL (Whole-Run Permutation)", fontsize=9.5)
    ax2.set_title("Importance by Feature Group", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.32, left=0.22, right=0.95)
    _add_synthetic_banner(fig, "Permutation Feature Importance on VAL Only")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_extras(
    data_dir: Path,
    models_path: Path,
    ops_path: Path,
    out_dir: Path,
    n_boot: int = 300,
    seed: int = 20261117,
    n_perm_repeats: int = 10,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[eval_extras] loading frozen models from {models_path} ...", flush=True)
    models = pickle.load(open(models_path, "rb"))
    ops = pickle.load(open(ops_path, "rb"))

    print(f"[eval_extras] loading datasets from {data_dir} ...", flush=True)
    manifest = pd.read_csv(data_dir / "manifest.csv", low_memory=False)
    splits = ("train", "val", "test", "shifted_test", "stress", "broad_basin")
    datasets = {s: pd.read_csv(data_dir / f"features_{s}.csv", low_memory=False) for s in splits}

    # 1. False alarms per simulated hour
    print("[eval_extras] 1. computing false alarms per simulated hour ...", flush=True)
    fa_df = EE.generate_false_alarm_per_hour_table(
        datasets, models, ops, manifest_df=manifest,
        splits=("test", "shifted_test", "val", "train"), n_boot=n_boot, seed=seed,
    )
    fa_df.to_csv(out_dir / "false_alarms_per_hour.csv", index=False)

    md_fa = ["# False Alarms per Simulated Hour & Missed-Event Rate (SYNTHETIC / SIMULATION)", "",
             "> **Note on Time Compression:** All normal runs across all splits use a uniform time-compression factor of **`360.0`** "
             "(1 simulated s = 360 physical s). A 240 s simulated normal run represents exactly **`24.0 simulated physical hours`** (1 full day). "
             "False alarms per simulated hour = (FA count / 24.0). For rapid events, compression factor is 1.0 (uncompressed).", "",
             "| split | model | regime | FA eps / run (95% CI) | FA eps / sim hour (95% CI) | FA win / run | FA win / sim hour | Missed-Event Rate (95% CI) | Exact Detect Rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in fa_df.iterrows():
        md_fa.append(
            f"| {r['split']} | {r['model']} | {r['regime']} | "
            f"{r['fa_episodes_per_run']:.3f} [{r['fa_episodes_per_run_ci_lo']:.3f}, {r['fa_episodes_per_run_ci_hi']:.3f}] | "
            f"**{r['fa_episodes_per_sim_hour']:.4f}** [{r['fa_episodes_per_sim_hour_ci_lo']:.4f}, {r['fa_episodes_per_sim_hour_ci_hi']:.4f}] | "
            f"{r['fa_windows_per_run']:.2f} | {r['fa_windows_per_sim_hour']:.3f} | "
            f"{r['missed_event_rate']:.3f} [{r['missed_event_ci_lo']:.3f}, {r['missed_event_ci_hi']:.3f}] | "
            f"{r['exact_detect_rate']:.3f} |"
        )
    (out_dir / "false_alarms_per_hour.md").write_text("\n".join(md_fa) + "\n", encoding="utf-8")

    # 2. Window accuracy alongside Macro-F1
    print("[eval_extras] 2. computing window accuracy alongside Macro-F1 ...", flush=True)
    acc_df = EE.generate_accuracy_and_f1_table(
        datasets, models, ops, splits=("test", "shifted_test", "stress", "broad_basin"),
        n_boot=n_boot, seed=seed,
    )
    acc_df.to_csv(out_dir / "window_accuracy_and_f1.csv", index=False)

    md_acc = [
        "# Window Accuracy and Macro-F1 (SYNTHETIC / SIMULATION)", "",
        "> **Table Caption / Notice:** *Class balance in these synthetic evaluation splits is balanced/controlled for benchmark purposes "
        "and does NOT represent the sparse event incidence rate of a real-world field deployment.*", "",
        "| split | model | regime | Window Accuracy (95% CI) | Window Macro-F1 (95% CI) | Prec/Rec Normal | Prec/Rec Gradual | Prec/Rec Rapid | Prec/Rec Vibration | Prec/Rec Sensor | Prec/Rec Comm |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in acc_df.iterrows():
        md_acc.append(
            f"| {r['split']} | {r['model']} | {r['regime']} | "
            f"**{r['window_accuracy']:.4f}** [{r['window_accuracy_ci_lo']:.4f}, {r['window_accuracy_ci_hi']:.4f}] | "
            f"**{r['window_macro_f1']:.4f}** [{r['window_macro_f1_ci_lo']:.4f}, {r['window_macro_f1_ci_hi']:.4f}] | "
            f"{r['prec_normal']:.2f} / {r['rec_normal']:.2f} | "
            f"{r['prec_gradual']:.2f} / {r['rec_gradual']:.2f} | "
            f"{r['prec_rapid']:.2f} / {r['rec_rapid']:.2f} | "
            f"{r['prec_abnormal_vibration']:.2f} / {r['rec_abnormal_vibration']:.2f} | "
            f"{r['prec_sensor_fault']:.2f} / {r['rec_sensor_fault']:.2f} | "
            f"{r['prec_comm_fault']:.2f} / {r['rec_comm_fault']:.2f} |"
        )
    (out_dir / "window_accuracy_and_f1.md").write_text("\n".join(md_acc) + "\n", encoding="utf-8")

    # 3. Permutation feature importance on VAL only
    print("[eval_extras] 3. computing permutation feature importance on VAL only (grouped by run) ...", flush=True)
    rf_model = models["random_forest"]
    val_df = datasets["val"]
    val_cols = F.feature_columns(val_df)

    # Individual features
    indiv_res = EE.compute_grouped_permutation_importance(
        rf_model, val_df, val_cols, n_repeats=n_perm_repeats, seed=seed,
    )
    indiv_rows = []
    # Reverse lookup for group
    col_to_group = {}
    for g, cols in EE.FEATURE_GROUPS.items():
        for c in cols:
            col_to_group[c] = g

    for c, stat in indiv_res.items():
        indiv_rows.append({
            "feature": c,
            "group": col_to_group.get(c, "unknown"),
            "mean_importance": round(stat["mean_importance"], 5),
            "std_importance": round(stat["std_importance"], 5),
            "min_importance": round(stat["min_importance"], 5),
            "max_importance": round(stat["max_importance"], 5),
            "repeats": stat["repeats"],
        })
    all_indiv_df = pd.DataFrame(indiv_rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    all_indiv_df["rank"] = all_indiv_df.index + 1
    top20_df = all_indiv_df.head(20).copy()
    top20_df.to_csv(out_dir / "permutation_importance_top20.csv", index=False)
    all_indiv_df.to_csv(out_dir / "permutation_importance_all_features.csv", index=False)

    md_t20 = [
        "# Top 20 Permutation Feature Importance (Random Forest on VAL Only; Grouped by Run)", "",
        "> **Methodology:** Permutations are executed across whole runs (preserving temporal autocorrelation within runs) on the **VAL split only**. "
        f"Repeats = {n_perm_repeats}. Importance metric = drop in Window Macro-F1.", "",
        "| Rank | Feature | Feature Group | Mean Macro-F1 Drop | Std (Spread) | Min | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in top20_df.iterrows():
        md_t20.append(
            f"| {int(r['rank'])} | `{r['feature']}` | {r['group']} | "
            f"**{r['mean_importance']:.5f}** | {r['std_importance']:.5f} | {r['min_importance']:.5f} | {r['max_importance']:.5f} |"
        )
    (out_dir / "permutation_importance_top20.md").write_text("\n".join(md_t20) + "\n", encoding="utf-8")

    # Feature groups
    group_res = EE.compute_grouped_permutation_importance(
        rf_model, val_df, EE.FEATURE_GROUPS, n_repeats=n_perm_repeats, seed=seed + 10,
    )
    group_rows = []
    for g, stat in group_res.items():
        group_rows.append({
            "group": g,
            "feature_count": len(EE.FEATURE_GROUPS[g]),
            "mean_importance": round(stat["mean_importance"], 5),
            "std_importance": round(stat["std_importance"], 5),
            "min_importance": round(stat["min_importance"], 5),
            "max_importance": round(stat["max_importance"], 5),
            "repeats": stat["repeats"],
        })
    groups_df = pd.DataFrame(group_rows).sort_values("mean_importance", ascending=False).reset_index(drop=True)
    groups_df["rank"] = groups_df.index + 1
    groups_df.to_csv(out_dir / "permutation_importance_groups.csv", index=False)

    md_grp = [
        "# Feature Group Permutation Importance (Random Forest on VAL Only; Grouped by Run)", "",
        "> **Methodology:** Permuting all features of a group simultaneously across whole runs on the **VAL split only**.", "",
        "| Rank | Feature Group | Feature Count | Mean Macro-F1 Drop | Std (Spread) | Min | Max |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in groups_df.iterrows():
        md_grp.append(
            f"| {int(r['rank'])} | **{r['group']}** | {int(r['feature_count'])} | "
            f"**{r['mean_importance']:.5f}** | {r['std_importance']:.5f} | {r['min_importance']:.5f} | {r['max_importance']:.5f} |"
        )
    (out_dir / "permutation_importance_groups.md").write_text("\n".join(md_grp) + "\n", encoding="utf-8")

    # 4. Reliability calibration on VAL only
    print("[eval_extras] 4. computing score calibration on VAL only ...", flush=True)
    calib = EE.compute_reliability_calibration(models, val_df, n_bins=10)
    calib["top1_df"].to_csv(out_dir / "score_calibration_top1_val.csv", index=False)
    calib["per_class_df"].to_csv(out_dir / "score_calibration_per_class_val.csv", index=False)

    md_cal = [
        "# Model Score Reliability & Calibration on VAL Only (SYNTHETIC / SIMULATION)", "",
        "> **Important Documentation on Model Scores:**",
        "> 1. Output scores are raw, uncalibrated classifier outputs designated strictly as **`model_score`** (never 'probability' or 'confidence').",
        "> 2. Random Forest and Logistic Regression probabilities are **NOT** calibrated Bayesian probabilities.",
        "> 3. Tree ensembles typically exhibit under-confidence in extreme regimes and push scores toward intermediate values due to finite-leaf smoothing.",
        "> 4. No 'uncertain' classification outcome is produced; each window receives raw model scores across the 6 condition classes.", "",
        "| Model | Bin Range (model_score) | Mean model_score | Empirical Accuracy | Window Count |",
        "|---|---|---|---|---|",
    ]
    for _, r in calib["top1_df"].iterrows():
        md_cal.append(
            f"| {r['model']} | [{r['bin_lower_model_score']:.1f}, {r['bin_upper_model_score']:.1f}) | "
            f"{r['mean_model_score'] if pd.notna(r['mean_model_score']) else 'N/A'} | "
            f"{r['empirical_accuracy'] if pd.notna(r['empirical_accuracy']) else 'N/A'} | "
            f"{int(r['window_count'])} |"
        )
    (out_dir / "score_calibration_top1_val.md").write_text("\n".join(md_cal) + "\n", encoding="utf-8")

    # 5. Publication Plots
    print("[eval_extras] 5. rendering publication plots with red banner ...", flush=True)
    plot_reliability_diagram(calib["top1_df"], out_dir / "reliability_diagram_val.png")
    plot_false_alarm_vs_detection(fa_df, out_dir / "false_alarm_per_hour_tradeoff.png")
    plot_permutation_importance(top20_df, groups_df, out_dir / "permutation_importance_val.png")

    print("[eval_extras] DONE. All outputs written to", out_dir, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs/datasets_v3")
    ap.add_argument("--models", default="outputs/eval_v3/trained_models_v3.pkl")
    ap.add_argument("--ops", default="outputs/eval_v3/operating_points_v3.pkl")
    ap.add_argument("--out", default="outputs/eval_extras")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--n-repeats", type=int, default=5)
    args = ap.parse_args()

    run_extras(
        data_dir=Path(args.data),
        models_path=Path(args.models),
        ops_path=Path(args.ops),
        out_dir=Path(args.out),
        n_boot=args.n_boot,
        seed=args.seed,
        n_perm_repeats=args.n_repeats,
    )
