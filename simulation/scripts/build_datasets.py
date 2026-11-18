"""Build full 6-class dataset suite, summary tables, and example window feature plots.

Splits are strictly by run (never by window):
  - train (60/class), val (20/class), test (20/class)
  - shifted_test (15/class, IMU noise multiplier 1.5-2x, shifted bias & confounders)
  - broad_basin (20 gradual + 20 rapid with broad_basin_subset=True)
  - stress (100 combined-condition runs)

Usage: python scripts/build_datasets.py [--config PATH] [--out DIR]
"""
from __future__ import annotations

import argparse
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
    confounders as cf,
    dataset as ds,
    engine,
    features as F,
    sensors as S,
)

BANNER = "SYNTHETIC / SIMULATION"


def plot_example_windows(features_df: pd.DataFrame, manifest_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot 3 example windows per class across the 6 classes showing key distinguishing features."""
    train_df = features_df[features_df["split"] == "train"].copy()
    comp_grad = float(manifest_df[manifest_df.run_class == "gradual"]["compression_factor"].iloc[0])
    comp_rap = float(manifest_df[manifest_df.run_class == "rapid"]["compression_factor"].iloc[0])

    key_feats = [
        ("NODE_01_mean", "N1 tilt mag (deg)"),
        ("long_slope_max_deg_s", "Long-horizon slope (deg/s)"),
        ("dir_pairwise_cos_mean", "Dir pairwise cos"),
        ("coherence_signed_mean", "Signed tilt coherence"),
        ("piezo_rms_network_mean", "Piezo RMS mean"),
        ("piezo_nodes_with_impulse", "Nodes w/ piezo impulse"),
        ("hub_common_mode_m", "Hub common-mode (m)"),
        ("comm_missing_frac_net", "Missing pkt frac"),
        ("comm_max_burst_len", "Max loss burst (pkts)"),
    ]
    cols = [k[0] for k in key_feats]
    labels = [k[1] for k in key_feats]

    # Compute z-scores across trainable train windows (skip implausible-reading extremes so SD isn't ~50 deg)
    tr_valid = train_df[(train_df["label"] != "transition") & (train_df["NODE_01_mean"] < 10.0)]
    mu = tr_valid[cols].mean()
    sd = tr_valid[cols].std().replace(0, 1.0)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), sharey=True)
    axes_flat = axes.ravel()
    bar_colors = ["tab:blue", "tab:orange", "tab:green"]

    for idx, cls in enumerate(C.CLASSES):
        ax = axes_flat[idx]
        # Pick 3 distinct runs of this class and select an active window of that class from each
        sub_runs = train_df[(train_df["run_class"] == cls) & (train_df["label"] == cls)]
        run_ids = sub_runs["simulation_id"].unique()[:3]
        picked_rows = []
        for rid in run_ids:
            r_wins = sub_runs[sub_runs["simulation_id"] == rid]
            # Pick middle active window
            picked_rows.append(r_wins.iloc[len(r_wins) // 2])

        x = np.arange(len(cols))
        width = 0.25
        for j, row in enumerate(picked_rows):
            z = ((row[cols] - mu) / sd).to_numpy(dtype=float)
            z_clip = np.clip(z, -3.5, 4.5)
            w_lbl = f"{row['simulation_id'].split('_')[-1]} [{row['t_start']:.0f}-{row['t_end']:.0f}s]"
            ax.bar(x + (j - 1) * width, z_clip, width=width, color=bar_colors[j], alpha=0.85, label=w_lbl)

        ax.axhline(0.0, color="black", lw=0.7, alpha=0.6)
        ax.set_title(f"Class: {cls}  (3 example windows)", fontsize=11, fontweight="bold")
        ax.text(
            0.5,
            1.10,
            BANNER,
            transform=ax.transAxes,
            color="red",
            ha="center",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right", fontsize=8.5)
        ax.set_ylim(-3.8, 5.0)
        if idx % 3 == 0:
            ax.set_ylabel("Standardised feature (z-score, clipped)")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{BANNER}  |  time compression: gradual x{comp_grad:g} (240 s = 24 h), rapid x{comp_rap:g} (real-time)  |  "
        "Key Window Features (3 Example Windows per Class)",
        color="red",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p_out = out_dir / "plot_class_example_windows.png"
    fig.savefig(p_out, dpi=130)
    plt.close(fig)
    return p_out


def plot_snr_distribution(manifest_df: pd.DataFrame, out_dir: Path) -> Path:
    """Plot SNR distribution (peak_tilt_deg / imu_noise_sd_deg) for gradual and rapid runs."""
    sub = manifest_df[
        manifest_df["split"].isin(["train", "val", "test", "shifted_test", "broad_basin"])
        & manifest_df["run_class"].isin(["gradual", "rapid"])
    ].copy()
    comp_grad = float(sub[sub.run_class == "gradual"]["compression_factor"].iloc[0])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for idx, cls in enumerate(["gradual", "rapid"]):
        ax = axes[idx]
        hl = sub[(sub.run_class == cls) & sub.split.isin(["train", "val", "test"])]["snr_vs_imu_noise"]
        sh = sub[(sub.run_class == cls) & (sub.split == "shifted_test")]["snr_vs_imu_noise"]
        bins = np.logspace(np.log10(max(0.1, sub["snr_vs_imu_noise"].min() * 0.9)),
                           np.log10(sub["snr_vs_imu_noise"].max() * 1.1), 20)
        ax.hist(hl, bins=bins, alpha=0.75, color="tab:blue", edgecolor="k",
                label=f"headline (n={len(hl)}, <1: {(hl < 1).mean()*100:.1f}%)")
        ax.hist(sh, bins=bins, alpha=0.65, color="tab:orange", edgecolor="k",
                label=f"shifted_test (n={len(sh)}, <1: {(sh < 1).mean()*100:.1f}%)")
        ax.axvline(1.0, color="red", ls="--", lw=1.5, label="SNR = 1 threshold")
        ax.set_xscale("log")
        ax.set_xlabel("snr_vs_imu_noise = peak_tilt_deg / imu_noise_sd_deg (log scale)")
        ax.set_title(f"{cls} runs", fontsize=11, fontweight="bold")
        ax.text(0.5, 1.08, BANNER, transform=ax.transAxes, color="red", ha="center",
                fontsize=9.5, fontweight="bold")
        ax.legend(fontsize=8.5)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Number of runs")
    fig.suptitle(
        f"{BANNER}  |  time compression: gradual x{comp_grad:g}, rapid x1  |  "
        "Distribution of snr_vs_imu_noise Across Stratified Basin Runs",
        color="red",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    p_out = out_dir / "plot_snr_distribution.png"
    fig.savefig(p_out, dpi=130)
    plt.close(fig)
    return p_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "dataset_default.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "datasets"))
    a = ap.parse_args()

    ds_cfg = ds.load_dataset_config(a.config)
    scen_cfg = engine.load_config(ROOT / "configs" / "scenario_default.yaml")
    sens_cfg = S.load_sensor_config(ROOT / "configs" / "sensor_default.yaml")
    conf_cfg = cf.load_confounder_config(ROOT / "configs" / "confounder_default.yaml")
    profile = S.load_noise_profile(ROOT / "configs" / "noise_profile.json")

    out_dir = Path(a.out)
    res = ds.build_datasets(ds_cfg, scen_cfg, sens_cfg, conf_cfg, profile, out_dir=out_dir)
    mf = res["manifest"]
    lbls = res["labels"]
    feats = res["features"]

    p1 = plot_example_windows(feats, mf, out_dir)
    p2 = plot_snr_distribution(mf, out_dir)

    print("=== RUN COUNTS BY SPLIT AND RUN_CLASS ===")
    run_tab = pd.crosstab(mf["split"], mf["run_class"], margins=True)
    print(run_tab.to_string())

    print("\n=== WINDOW COUNTS BY SPLIT AND WINDOW LABEL ===")
    win_tab = pd.crosstab(lbls["split"], lbls["label"], margins=True)
    print(win_tab.to_string())

    print("\n=== SNR_VS_IMU_NOISE DISTRIBUTION (GRADUAL & RAPID) ===")
    basin_mf = mf[mf["run_class"].isin(["gradual", "rapid"])].copy()
    for grp_name, sub in [
        ("Headline (train+val+test)", basin_mf[basin_mf.split.isin(["train", "val", "test"])]),
        ("Shifted-noise test", basin_mf[basin_mf.split == "shifted_test"]),
        ("Broad-basin subset", basin_mf[basin_mf.split == "broad_basin"]),
        ("All single-condition basin runs", basin_mf[basin_mf.split != "stress"]),
    ]:
        s = sub["snr_vs_imu_noise"]
        below1 = (s < 1.0).sum()
        frac = (s < 1.0).mean() * 100
        q = np.percentile(s, [5, 25, 50, 75, 95])
        print(
            f"{grp_name} (n={len(s)}): min={s.min():.3f}, p5={q[0]:.3f}, p25={q[1]:.3f}, "
            f"median={q[2]:.3f}, p75={q[3]:.3f}, p95={q[4]:.3f}, max={s.max():.3f} | "
            f"below 1.0: {below1}/{len(s)} ({frac:.1f}%)"
        )

    print(f"\nSaved plots: {p1} , {p2}")


if __name__ == "__main__":
    main()
