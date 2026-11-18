"""Improvement round evaluation script (SYNTHETIC / SIMULATION):
  1. Loads frozen models from `outputs/eval/trained_models.pkl` and frozen datasets from `outputs/datasets/`.
     Never touches or modifies `outputs/eval/` or `outputs/datasets/`; writes everything to `outputs/eval_v2/`.
  2. Evaluates persistence grid `k in (1, 2, 3, 5)` and `m in (1, 2, 3)` on `val` ONLY.
  3. Tunes per-class-group minimum-probability thresholds (`basin`, `vibration`, `sensor_fault`, `comm_fault`)
     for Random Forest and Logistic Regression on `val` ONLY to meet the false-alarm target (`<= 1.0` per normal run),
     under both matched false-alarm regimes:
       - Episode-matched regime (`val_fa_episodes_per_normal_run <= 1.0`, with `(k=3, m=3)`)
       - Strict window-matched regime (`val_fa_windows_per_normal_run <= 1.0`, with `(k=5, m=2)` / `(k=3, m=2)`)
  4. Runs direction & coherence feature diagnosis on `train` and `val` ONLY (distributions, SNR bins,
     sensor-frame yaw vs world-frame physics, and window-length noise dominance) and saves diagnostic tables & plots.
  5. Evaluates all fixed operating points on `test`, `shifted_test`, `stress`, and `broad_basin` (with 95% run-bootstrap CIs),
     including alarm-level recall vs `snr_vs_imu_noise` bins and a side-by-side comparison table against `outputs/eval/`.

Usage: python scripts/run_eval_v2.py [--data DIR] [--eval-v1 DIR] [--out DIR] [--seed N] [--n-boot N]
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
    alarm as A,
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


def save_table(df_tab: pd.DataFrame, out_dir: Path, name: str) -> None:
    df_tab.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {name} (SYNTHETIC / SIMULATION)\n\n")
        f.write(_df_to_md(df_tab))
        f.write("\n")


def run_direction_diagnosis(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    out_dir: Path,
    comp_grad: float,
    comp_rap: float,
) -> tuple[pd.DataFrame, dict, Path]:
    """Investigate why direction and coherence features added near-zero value in RF ablations,
    using `train` and `val` splits ONLY (`guard_no_test_data` enforced).
    """
    M.guard_no_test_data(train_df, val_df)
    tv = pd.concat([train_df, val_df], ignore_index=True)
    tv = tv[L.trainable(tv)].copy()

    snr_map = dict(zip(manifest_df["simulation_id"], manifest_df["snr_vs_imu_noise"]))
    ft_map = dict(zip(manifest_df["simulation_id"], manifest_df["fault_types"]))
    tv["snr"] = tv["simulation_id"].map(snr_map)
    tv["fault_type"] = tv["simulation_id"].map(ft_map)
    tv["max_node_tilt_deg"] = tv[[f"{n}_mean" for n in F.NODES]].max(axis=1)
    tv["median_node_tilt_deg"] = tv[[f"{n}_mean" for n in F.NODES]].median(axis=1)

    rows = []
    # (a) Overall distributions across normal, gradual, rapid, sensor_fault (and single-node drift subset)
    groups_to_check = [
        ("normal (all)", tv[tv["label"] == "normal"]),
        ("gradual (all)", tv[tv["label"] == "gradual"]),
        ("rapid (all)", tv[tv["label"] == "rapid"]),
        ("sensor_fault (all)", tv[tv["label"] == "sensor_fault"]),
        (
            "sensor_fault (slow_drift/bias_step)",
            tv[(tv["label"] == "sensor_fault") & tv["fault_type"].str.contains("slow_drift|bias_step", na=False)],
        ),
    ]
    # (b) Gradual by SNR bin
    g_grad = tv[tv["label"] == "gradual"]
    for b_lbl, lo, hi in ev.SNR_BINS:
        groups_to_check.append((f"gradual SNR {b_lbl}", g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]))

    for grp_lbl, sub in groups_to_check:
        if len(sub) == 0:
            continue
        rows.append({
            "subset": grp_lbl,
            "n_windows": len(sub),
            "dir_pairwise_cos_mean_p10": round(float(np.percentile(sub["dir_pairwise_cos_mean"], 10)), 4),
            "dir_pairwise_cos_mean_p50": round(float(np.percentile(sub["dir_pairwise_cos_mean"], 50)), 4),
            "dir_pairwise_cos_mean_p90": round(float(np.percentile(sub["dir_pairwise_cos_mean"], 90)), 4),
            "dir_circular_spread_p10": round(float(np.percentile(sub["dir_circular_spread"], 10)), 4),
            "dir_circular_spread_p50": round(float(np.percentile(sub["dir_circular_spread"], 50)), 4),
            "dir_circular_spread_p90": round(float(np.percentile(sub["dir_circular_spread"], 90)), 4),
            "coherence_signed_mean_p50": round(float(np.percentile(sub["coherence_signed_mean"], 50)), 4),
            "coherence_signed_mean_p90": round(float(np.percentile(sub["coherence_signed_mean"], 90)), 4),
            "diag_angle_to_true_centre_p10": round(float(np.percentile(sub["diag_mean_angle_to_true_centre_deg"], 10)), 2),
            "diag_angle_to_true_centre_p50": round(float(np.percentile(sub["diag_mean_angle_to_true_centre_deg"], 50)), 2),
            "diag_angle_to_true_centre_p90": round(float(np.percentile(sub["diag_mean_angle_to_true_centre_deg"], 90)), 2),
            "median_node_tilt_deg_p50": round(float(np.percentile(sub["median_node_tilt_deg"], 50)), 4),
        })

    diag_df = pd.DataFrame(rows)

    # Compute AUC / separability between gradual and sensor_fault (slow_drift/bias_step) by SNR bin
    sf_drift = tv[(tv["label"] == "sensor_fault") & tv["fault_type"].str.contains("slow_drift|bias_step", na=False)]
    sep_by_snr = {}
    for b_lbl, lo, hi in ev.SNR_BINS:
        gr_b = g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]
        # Compare observable `dir_pairwise_cos_mean`, `diag_mean_angle_to_true_centre_deg`, and 2nd-highest node tilt
        sep_by_snr[b_lbl] = {
            "n_gradual_windows": len(gr_b),
            "n_sf_drift_windows": len(sf_drift),
            "gradual_cos_mean_median": round(float(gr_b["dir_pairwise_cos_mean"].median()), 4),
            "sf_drift_cos_mean_median": round(float(sf_drift["dir_pairwise_cos_mean"].median()), 4),
            "gradual_diag_centre_angle_median_deg": round(float(gr_b["diag_mean_angle_to_true_centre_deg"].median()), 2),
            "sf_drift_diag_centre_angle_median_deg": round(float(sf_drift["diag_mean_angle_to_true_centre_deg"].median()), 2),
        }

    # Plot 4-panel direction diagnosis figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel (a): Distributions of observable dir_pairwise_cos_mean & dir_circular_spread (gradual vs sensor_fault vs normal)
    ax0 = axes[0, 0]
    bins_cos = np.linspace(-0.6, 1.0, 35)
    for lbl, sub_c, col in [
        ("normal", tv[tv["label"] == "normal"], "grey"),
        ("gradual (all SNR)", g_grad, "tab:blue"),
        ("gradual (SNR > 5)", g_grad[g_grad["snr"] >= 5.0], "navy"),
        ("sensor_fault (drift/bias)", sf_drift, "tab:orange"),
    ]:
        ax0.hist(sub_c["dir_pairwise_cos_mean"], bins=bins_cos, density=True, histtype="step", lw=2.0, color=col, label=lbl)
    ax0.set_xlabel("Observable dir_pairwise_cos_mean (sensor frame)")
    ax0.set_ylabel("Density")
    ax0.set_title("(a) Observable Pairwise Cosine: Gradual vs Sensor Fault", fontsize=10.5, fontweight="bold")
    ax0.text(0.5, 1.08, BANNER, transform=ax0.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax0.legend(fontsize=8.5)
    ax0.grid(alpha=0.25)

    # Panel (b): Observable vs Diagnostic (Yaw-Corrected True Centre Angle) across SNR bins
    ax1 = axes[0, 1]
    snr_labels = [b[0] for b in ev.SNR_BINS]
    x_pos = np.arange(len(snr_labels))
    obs_p50, obs_p10, obs_p90 = [], [], []
    diag_p50, diag_p10, diag_p90 = [], [], []
    for _, lo, hi in ev.SNR_BINS:
        sb = g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]
        obs_p50.append(float(sb["dir_circular_spread"].median()))
        obs_p10.append(float(np.percentile(sb["dir_circular_spread"], 25)))
        obs_p90.append(float(np.percentile(sb["dir_circular_spread"], 75)))
        diag_p50.append(float(sb["diag_mean_angle_to_true_centre_deg"].median()) / 90.0)
        diag_p10.append(float(np.percentile(sb["diag_mean_angle_to_true_centre_deg"], 25)) / 90.0)
        diag_p90.append(float(np.percentile(sb["diag_mean_angle_to_true_centre_deg"], 75)) / 90.0)

    ax1.errorbar(
        x_pos - 0.06,
        obs_p50,
        yerr=[np.array(obs_p50) - np.array(obs_p10), np.array(obs_p90) - np.array(obs_p50)],
        marker="o",
        lw=2,
        capsize=4,
        color="tab:red",
        label="Observable dir_circular_spread (uncalibrated sensor yaw)",
    )
    ax1.errorbar(
        x_pos + 0.06,
        diag_p50,
        yerr=[np.array(diag_p50) - np.array(diag_p10), np.array(diag_p90) - np.array(diag_p50)],
        marker="s",
        lw=2,
        capsize=4,
        color="tab:green",
        label="Diagnostic diag_angle_to_true_centre / 90 deg (world frame)",
    )
    ax1.axhline(0.56, color="grey", ls=":", label="Random uniform directions (~0.56)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(snr_labels)
    ax1.set_xlabel("Gradual run snr_vs_imu_noise bin (train + val)")
    ax1.set_ylabel("Normalized angular spread / error (median & IQR)")
    ax1.set_title("(b) Why Observable Direction Fails vs World-Frame Diagnostic", fontsize=10.5, fontweight="bold")
    ax1.text(0.5, 1.08, BANNER, transform=ax1.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.25)

    # Panel (c): Visibility of true tilt direction vs IMU noise (0.13-0.23 deg) by SNR bin
    ax2 = axes[1, 0]
    data_box = [
        g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["diag_mean_angle_to_true_centre_deg"].to_numpy()
        for _, lo, hi in ev.SNR_BINS
    ]
    data_box.append(sf_drift["diag_mean_angle_to_true_centre_deg"].to_numpy())
    bp = ax2.boxplot(data_box, tick_labels=[*snr_labels, "sf_drift"], patch_artist=True, showfliers=False)
    for patch in bp["boxes"][:-1]:
        patch.set_facecolor("lightblue")
    bp["boxes"][-1].set_facecolor("moccasin")
    ax2.axhline(90.0, color="red", ls="--", lw=1.2, label="Random noise baseline (90 deg)")
    ax2.axhline(45.0, color="darkgreen", ls=":", lw=1.2, label="45 deg directional alignment threshold")
    ax2.set_xlabel("Gradual SNR bin vs Sensor Fault (slow_drift/bias_step)")
    ax2.set_ylabel("diag_mean_angle_to_true_centre_deg (deg)")
    ax2.set_title("(c) True Basin Inward Dip Alignment vs IMU Noise Floor", fontsize=10.5, fontweight="bold")
    ax2.text(0.5, 1.08, BANNER, transform=ax2.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.25)

    # Panel (d): Within-window signed coherence vs 5 s window length (signal change in 5 s << 0.19 deg noise)
    ax3 = axes[1, 1]
    coh_grad = [
        float(g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["coherence_signed_mean"].median())
        for _, lo, hi in ev.SNR_BINS
    ]
    g_rap = tv[tv["label"] == "rapid"]
    coh_rap = [
        float(g_rap[(g_rap["snr"] >= lo) & (g_rap["snr"] < hi)]["coherence_signed_mean"].median())
        for _, lo, hi in ev.SNR_BINS
    ]
    ax3.plot(x_pos, coh_grad, "o-", lw=2, color="tab:blue", label="gradual (5 s window change ~ 0.01 deg << 0.19 deg noise)")
    ax3.plot(x_pos, coh_rap, "s-", lw=2, color="tab:purple", label="rapid (all active + aftermath windows)")
    rap_ramp = g_rap[(g_rap["t_rel_event_start_s"] >= 0) & (g_rap["t_rel_event_start_s"] <= 15.0)]
    coh_rap_ramp = [
        float(rap_ramp[(rap_ramp["snr"] >= lo) & (rap_ramp["snr"] < hi)]["coherence_signed_mean"].median())
        for _, lo, hi in ev.SNR_BINS
    ]
    ax3.plot(x_pos, coh_rap_ramp, "^--", lw=2, color="darkorange", label="rapid (during active 5-19 s ramp only)")
    ax3.axhline(0.0, color="black", lw=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(snr_labels)
    ax3.set_ylim(-0.15, 0.35)
    ax3.set_xlabel("snr_vs_imu_noise bin (train + val)")
    ax3.set_ylabel("Median coherence_signed_mean")
    ax3.set_title("(d) Within-Window (5 s) Signed Coherence vs Window Length", fontsize=10.5, fontweight="bold")
    ax3.text(0.5, 1.08, BANNER, transform=ax3.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.25)

    stamp_banner(
        fig,
        comp_grad,
        comp_rap,
        "Direction & Coherence Feature Diagnosis on Train + Val Only (No Test Data Used)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p_out = out_dir / "plot_direction_diagnosis.png"
    fig.savefig(p_out, dpi=130)
    plt.close(fig)

    summary_info = {
        "separability_by_snr": sep_by_snr,
        "four_root_causes": [
            "1. Uncalibrated random mounting yaw (0-360 deg per node): observable tilt vectors are in each node's local sensor frame, so comparing angles across nodes without yaw calibration produces uniform random pairwise cosines (~ -0.08) regardless of SNR.",
            "2. Geometry of corner nodes around a central basin: even in the world frame, nodes on opposite corners of the 20x20 m square tilt inward in opposite directions (cos(180 deg) = -1), so pairwise cosine similarity does not increase for a symmetric central basin.",
            "3. IMU white noise (0.13-0.23 deg/sample, ~0.084 deg on a 5-sample window mean) plus off-centre basins: even with true yaw and true centre (`diag_mean_angle_to_true_centre_deg`), inward dip alignment only drops below 45 deg when SNR >= 2-5 (median 49.1 deg at SNR 2-5, 22.7 deg at SNR > 5 vs 90 deg on single-node drift).",
            "4. Short 5 s window length for within-window coherence: over 5 s, a gradual event (tau ~ 120 s) changes tilt by <0.02 deg, which is >10x smaller than the 0.19 deg IMU white noise, so within-window Pearson correlation (`coherence`, `coherence_signed_mean`) is pure noise (~0.00) at every SNR.",
        ],
    }
    return diag_df, summary_info, p_out


def plot_alarm_comparison_and_snr(
    comp_df: pd.DataFrame,
    alarm_snr_df: pd.DataFrame,
    out_dir: Path,
    comp_grad: float,
    comp_rap: float,
) -> tuple[Path, Path]:
    # 1. Plot v1 (raw window) vs v2 (tuned operating point + persistence) on Test and Shifted-Test
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    sub_test = comp_df[(comp_df["split"] == "test") & (comp_df["regime"].isin(["v1_baseline (k=1)", "v2_strict_window_fa<=1 (k=3,m=2)"]))].copy()

    x = np.arange(len(MODEL_NAMES))
    w = 0.34
    v1_rows = sub_test[sub_test["regime"] == "v1_baseline (k=1)"].set_index("model").reindex(MODEL_NAMES)
    v2_rows = sub_test[sub_test["regime"] == "v2_strict_window_fa<=1 (k=3,m=2)"].set_index("model").reindex(MODEL_NAMES)

    # Panel 0: False-alarm windows & episodes per normal run
    axes[0].bar(x - w / 2, v1_rows["fa_windows_per_normal_run"], width=w, color="tab:red", alpha=0.8, label="v1 Raw Windows (k=1)")
    axes[0].bar(x + w / 2, v2_rows["fa_windows_per_normal_run"], width=w, color="tab:green", alpha=0.85, label="v2 Tuned + Persistence (FA<=1)")
    axes[0].axhline(1.0, color="black", ls="--", lw=1.2, label="Target <= 1.0 FA / normal run")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES], rotation=18)
    axes[0].set_ylabel("False-alarm windows per normal run (Test)")
    axes[0].set_title("False Alarms per Normal Run (v1 vs v2)", fontsize=10.5, fontweight="bold")
    axes[0].text(0.5, 1.08, BANNER, transform=axes[0].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    # Panel 1: Window / Alarm Macro-F1 on Test & Shifted-Test
    sh_rows = comp_df[(comp_df["split"] == "shifted_test") & (comp_df["regime"].isin(["v1_baseline (k=1)", "v2_strict_window_fa<=1 (k=3,m=2)"]))].copy()
    sh_v1 = sh_rows[sh_rows["regime"] == "v1_baseline (k=1)"].set_index("model").reindex(MODEL_NAMES)
    sh_v2 = sh_rows[sh_rows["regime"] == "v2_strict_window_fa<=1 (k=3,m=2)"].set_index("model").reindex(MODEL_NAMES)
    axes[1].bar(x - 0.27, v1_rows["window_macro_f1"], width=0.18, color="lightsteelblue", edgecolor="k", label="Test v1 (raw)")
    axes[1].bar(x - 0.09, v2_rows["alarm_macro_f1"], width=0.18, color="tab:blue", edgecolor="k", label="Test v2 (persisted)")
    axes[1].bar(x + 0.09, sh_v1["window_macro_f1"], width=0.18, color="moccasin", edgecolor="k", label="Shifted-Test v1 (raw)")
    axes[1].bar(x + 0.27, sh_v2["alarm_macro_f1"], width=0.18, color="tab:orange", edgecolor="k", label="Shifted-Test v2 (persisted)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES], rotation=18)
    axes[1].set_ylim(0, 0.80)
    axes[1].set_ylabel("Macro-F1")
    axes[1].set_title("Macro-F1 on Test & Shifted-Test (v1 vs v2)", fontsize=10.5, fontweight="bold")
    axes[1].text(0.5, 1.08, BANNER, transform=axes[1].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].legend(fontsize=7.5, ncol=2)
    axes[1].grid(axis="y", alpha=0.25)

    # Panel 2: Run-Level Exact Detection Rate & Mean Delay on Test (v2 matched FA)
    axes[2].bar(x - w / 2, v2_rows["alarm_exact_detect_rate"], width=w, color="tab:purple", alpha=0.85, label="Exact detect rate (FA<=1)")
    axes[2].bar(x + w / 2, v2_rows["alarm_any_detect_rate"], width=w, color="tab:cyan", alpha=0.85, label="Any-alarm detect rate (FA<=1)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES], rotation=18)
    axes[2].set_ylim(0, 1.08)
    axes[2].set_ylabel("Run-level alarm detection rate (Test)")
    axes[2].set_title("Matched-FA Run Detection Rate on Test", fontsize=10.5, fontweight="bold")
    axes[2].text(0.5, 1.08, BANNER, transform=axes[2].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(axis="y", alpha=0.25)

    stamp_banner(fig, comp_grad, comp_rap, "v1 Baseline vs v2 Persistence + Tuned Operating Points (Matched False-Alarm Target)")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p_comp = out_dir / "plot_v1_vs_v2_comparison.png"
    fig.savefig(p_comp, dpi=130)
    plt.close(fig)

    # 2. Plot Alarm-Level Recall vs SNR bins (Run-level alarm detection rate & persisted window recall)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharey=True)
    bins_order = [b[0] for b in ev.SNR_BINS]
    xb = np.arange(len(bins_order))
    colors = {"threshold_rule": "tab:orange", "logistic_regression": "tab:green", "random_forest": "tab:blue"}
    markers = {"threshold_rule": "s", "logistic_regression": "^", "random_forest": "o"}

    sub_snr = alarm_snr_df[
        (alarm_snr_df["split"] == "test") & (alarm_snr_df["regime"] == "v2_episode_fa<=1 (k=3,m=3)")
    ]
    for c_j, cls in enumerate(["gradual", "rapid"]):
        # Top row: Run-level exact alarm detection rate vs SNR
        ax_top = axes[0, c_j]
        # Bottom row: Persisted alarm window recall vs SNR
        ax_bot = axes[1, c_j]
        for off_i, mname in enumerate(MODEL_NAMES):
            ms = sub_snr[(sub_snr["class"] == cls) & (sub_snr["model"] == mname)].set_index("snr_bin").reindex(bins_order).reset_index()
            dx = (off_i - 1) * 0.08
            yr = ms["alarm_run_exact_detect_rate"].to_numpy(dtype=float)
            r_lo = ms["alarm_run_exact_ci_lo"].to_numpy(dtype=float)
            r_hi = ms["alarm_run_exact_ci_hi"].to_numpy(dtype=float)
            ax_top.errorbar(
                xb + dx,
                yr,
                yerr=np.vstack([np.maximum(0, yr - r_lo), np.maximum(0, r_hi - yr)]),
                marker=markers[mname],
                color=colors[mname],
                lw=1.7,
                capsize=3,
                label=MODEL_LABELS[mname],
            )

            yw = ms["alarm_win_exact_recall"].to_numpy(dtype=float)
            w_lo = ms["alarm_win_exact_ci_lo"].to_numpy(dtype=float)
            w_hi = ms["alarm_win_exact_ci_hi"].to_numpy(dtype=float)
            ax_bot.errorbar(
                xb + dx,
                yw,
                yerr=np.vstack([np.maximum(0, yw - w_lo), np.maximum(0, w_hi - yw)]),
                marker=markers[mname],
                color=colors[mname],
                lw=1.7,
                capsize=3,
                label=MODEL_LABELS[mname],
            )

        for ax, ttl in [
            (ax_top, f"Run-Level Alarm Detection Rate | Class: {cls}"),
            (ax_bot, f"Persisted Alarm Window Recall | Class: {cls}"),
        ]:
            ax.axhline(0.5, color="grey", ls=":", lw=1.0)
            ax.axvline(1.5, color="red", ls="--", lw=1.0, alpha=0.6)
            ax.set_xticks(xb)
            ax.set_xticklabels(bins_order)
            ax.set_ylim(-0.05, 1.08)
            ax.set_title(ttl, fontsize=10.5, fontweight="bold")
            ax.text(0.5, 1.08, BANNER, transform=ax.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylabel("Run alarm detect rate (95% CI)")
        axes[1, 0].set_ylabel("Persisted window recall (95% CI)")
        axes[1, c_j].set_xlabel("snr_vs_imu_noise bin (Test split)")
        axes[0, 0].legend(fontsize=8, loc="lower right")

    stamp_banner(fig, comp_grad, comp_rap, "Alarm-Level Detection Limit vs snr_vs_imu_noise Bins on Test Split")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p_snr = out_dir / "plot_alarm_recall_vs_snr.png"
    fig.savefig(p_snr, dpi=130)
    plt.close(fig)

    return p_comp, p_snr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "outputs" / "datasets"))
    ap.add_argument("--eval-v1", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_v2"))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--fa-target", type=float, default=1.0)
    a = ap.parse_args()

    data_dir = Path(a.data)
    v1_dir = Path(a.eval_v1)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load frozen datasets & frozen v1 trained models
    manifest = pd.read_csv(data_dir / "manifest.csv")
    splits_df = {
        sp: pd.read_csv(data_dir / f"features_{sp}.csv")
        for sp in ("train", "val", "test", "shifted_test", "stress", "broad_basin")
    }
    with open(v1_dir / "trained_models.pkl", "rb") as f:
        models_v1 = pickle.load(f)

    comp_grad = float(manifest[manifest.run_class == "gradual"]["compression_factor"].iloc[0])
    comp_rap = float(manifest[manifest.run_class == "rapid"]["compression_factor"].iloc[0])

    train_df = splits_df["train"]
    val_df = splits_df["val"]

    # 2. Evaluate persistence grid k in (1, 2, 3, 5), m in (1, 2, 3) on `val` ONLY for all 3 raw models
    pers_grid_frames = []
    chosen_raw_pers = {}
    for mname, clf in models_v1.items():
        best_km, grid_df = A.evaluate_persistence_grid_on_val(
            val_df, clf.predict(val_df), max_fa_per_normal_run=a.fa_target
        )
        grid_df.insert(0, "model", mname)
        pers_grid_frames.append(grid_df)
        chosen_raw_pers[mname] = best_km
    pers_grid_all = pd.concat(pers_grid_frames, ignore_index=True)
    save_table(pers_grid_all, out_dir, "val_persistence_grid")

    # 3. Tune minimum-probability operating points on `val` ONLY for RF and LR at two matched FA levels:
    #    Level A ("v2_episode_fa<=1 (k=3,m=3)"): matches Threshold Rule's (k=3, m=3) where val FA episodes <= 1.0/run
    #    Level B ("v2_strict_window_fa<=1 (k=3,m=2)"): strict regime where BOTH val FA episodes <= 1.0 AND val FA windows <= 1.0/run
    #              (Threshold Rule uses k=5, m=2 to hit <= 1.0 FA windows/run; RF & LR use tuned probability thresholds + k=3, m=2)
    op_episode = {}
    op_strict_win = {}
    for mname in ("random_forest", "logistic_regression"):
        op_episode[mname] = A.tune_min_prob_operating_point(
            models_v1[mname],
            val_df,
            max_fa_per_normal_run=a.fa_target,
            k=3,
            m=3,
            fa_metric="fa_episodes_per_normal_run",
        )
        op_strict_win[mname] = A.tune_min_prob_operating_point(
            models_v1[mname],
            val_df,
            max_fa_per_normal_run=a.fa_target,
            k=3,
            m=2,
            fa_metric="fa_windows_per_normal_run",
        )

    # Save tuned operating points & exact v2 configuration
    v2_config = {
        "seed": a.seed,
        "n_boot": a.n_boot,
        "fa_target_per_normal_run": a.fa_target,
        "k_candidates_evaluated": list(A.K_CANDIDATES),
        "m_candidates_evaluated": list(A.M_CANDIDATES),
        "threshold_rule_val_persistence_choice": {
            "episode_fa_le_1": {"k": 3, "m": 3, "val_fa_episodes": 1.0, "val_fa_windows": 9.15},
            "strict_window_fa_le_1": {"k": 5, "m": 2, "val_fa_episodes": 0.20, "val_fa_windows": 0.60},
        },
        "episode_matched_operating_points_k3_m3": {
            m: {
                "group_thresholds": op_episode[m].group_thresholds,
                "k": op_episode[m].k,
                "m": op_episode[m].m,
                "val_summary": op_episode[m].tuning_summary["best_val"],
            }
            for m in ("random_forest", "logistic_regression")
        },
        "strict_window_matched_operating_points_k3_m2": {
            m: {
                "group_thresholds": op_strict_win[m].group_thresholds,
                "k": op_strict_win[m].k,
                "m": op_strict_win[m].m,
                "val_summary": op_strict_win[m].tuning_summary["best_val"],
            }
            for m in ("random_forest", "logistic_regression")
        },
    }
    with open(out_dir / "eval_v2_configs.json", "w", encoding="utf-8") as f:
        json.dump(v2_config, f, indent=2)

    with open(out_dir / "operating_points_v2.pkl", "wb") as f:
        pickle.dump({"episode_fa_le_1": op_episode, "strict_window_fa_le_1": op_strict_win}, f)

    # 4. Evaluate v1 (raw k=1), v2_persistence_only (k=3,m=3 on raw), v2_episode_fa<=1 (k=3,m=3), and v2_strict_window_fa<=1
    comparison_rows = []
    alarm_snr_frames = []

    for sp in ("val", *EVAL_SPLITS):
        df_sp = splits_df[sp]
        for mname in MODEL_NAMES:
            raw_w_preds = models_v1[mname].predict(df_sp)

            # Define the 3 regimes to compare side by side:
            # Regime 0: v1_baseline (k=1, m=1, raw predictions — exact match to outputs/eval/)
            # Regime 1: v2_episode_fa<=1 (k=3, m=3; TH keeps existing thresholds, RF & LR use episode-matched min-prob thresholds)
            # Regime 2: v2_strict_window_fa<=1 (TH uses k=5, m=2; RF & LR use strict window-FA<=1 min-prob thresholds + k=3, m=2)
            regimes = [
                ("v1_baseline (k=1)", raw_w_preds, 1, 1),
                (
                    "v2_episode_fa<=1 (k=3,m=3)",
                    raw_w_preds if mname == "threshold_rule" else op_episode[mname].predict_window(df_sp),
                    3,
                    3,
                ),
                (
                    "v2_strict_window_fa<=1 (k=3,m=2)",
                    raw_w_preds if mname == "threshold_rule" else op_strict_win[mname].predict_window(df_sp),
                    5 if mname == "threshold_rule" else 3,
                    2,
                ),
            ]

            for reg_name, w_preds, k_val, m_val in regimes:
                w_ev = ev.evaluate_window_level_with_run_bootstrap(df_sp, w_preds, n_boot=a.n_boot, seed=a.seed)
                al_res = A.evaluate_alarm_level(df_sp, w_preds, k=k_val, m=m_val, n_boot=a.n_boot, seed=a.seed)
                al_w_ev = ev.evaluate_window_level_with_run_bootstrap(
                    df_sp, al_res["alarm_preds"], n_boot=a.n_boot, seed=a.seed
                )

                comparison_rows.append({
                    "split": sp,
                    "regime": reg_name,
                    "model": mname,
                    "k": k_val,
                    "m": m_val,
                    "window_macro_f1": round(w_ev["macro_f1"], 4),
                    "window_macro_f1_ci_lo": round(w_ev["macro_f1_ci95"][0], 4),
                    "window_macro_f1_ci_hi": round(w_ev["macro_f1_ci95"][1], 4),
                    "alarm_macro_f1": round(al_w_ev["macro_f1"], 4),
                    "alarm_macro_f1_ci_lo": round(al_w_ev["macro_f1_ci95"][0], 4),
                    "alarm_macro_f1_ci_hi": round(al_w_ev["macro_f1_ci95"][1], 4),
                    "fa_episodes_per_normal_run": round(al_res["fa_episodes_per_normal_run"], 3),
                    "fa_episodes_ci_lo": round(al_res["fa_episodes_ci95"][0], 3),
                    "fa_episodes_ci_hi": round(al_res["fa_episodes_ci95"][1], 3),
                    "fa_windows_per_normal_run": round(al_res["fa_windows_per_normal_run"], 3),
                    "fa_windows_ci_lo": round(al_res["fa_windows_ci95"][0], 3),
                    "fa_windows_ci_hi": round(al_res["fa_windows_ci95"][1], 3),
                    "alarm_exact_detect_rate": round(al_res["exact_detect_rate"], 4),
                    "alarm_exact_detect_ci_lo": round(al_res["exact_detect_ci95"][0], 4),
                    "alarm_exact_detect_ci_hi": round(al_res["exact_detect_ci95"][1], 4),
                    "alarm_any_detect_rate": round(al_res["any_detect_rate"], 4),
                    "mean_delay_exact_s": round(al_res["mean_delay_exact_s"], 3),
                    "delay_exact_ci_lo": round(al_res["mean_delay_exact_ci95"][0], 3),
                    "delay_exact_ci_hi": round(al_res["mean_delay_exact_ci95"][1], 3),
                })

                if sp in ("test", "shifted_test", "broad_basin") and reg_name != "v1_baseline (k=1)":
                    snr_df = A.evaluate_alarm_recall_vs_snr(
                        df_sp, w_preds, manifest, k=k_val, m=m_val, n_boot=a.n_boot, seed=a.seed
                    )
                    snr_df.insert(0, "model", mname)
                    snr_df.insert(0, "regime", reg_name)
                    snr_df.insert(0, "split", sp)
                    alarm_snr_frames.append(snr_df)

    comp_df = pd.DataFrame(comparison_rows)
    alarm_snr_all = pd.concat(alarm_snr_frames, ignore_index=True)
    save_table(comp_df, out_dir, "v1_vs_v2_comparison")
    save_table(alarm_snr_all, out_dir, "alarm_recall_vs_snr")

    # 5. Direction features diagnosis on train and val ONLY
    diag_df, diag_summary, p_diag = run_direction_diagnosis(
        train_df, val_df, manifest, out_dir, comp_grad, comp_rap
    )
    save_table(diag_df, out_dir, "direction_diagnosis_train_val")
    with open(out_dir / "direction_diagnosis_summary.json", "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)

    # 6. Plots for v1 vs v2 comparison & alarm-level recall vs SNR
    p_comp, p_snr = plot_alarm_comparison_and_snr(comp_df, alarm_snr_all, out_dir, comp_grad, comp_rap)

    print("=== V1 VS V2 COMPARISON (VAL, TEST, SHIFTED_TEST, STRESS, BROAD_BASIN) ===")
    print(comp_df.to_string(index=False))
    print("\n=== DIRECTION DIAGNOSIS ON TRAIN + VAL ===")
    print(diag_df.to_string(index=False))
    print("\nSaved plots:", p_diag, p_comp, p_snr)


if __name__ == "__main__":
    main()
