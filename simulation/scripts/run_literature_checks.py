#!/usr/bin/env python
"""Runner for literature-basis sensitivity checks & timescale audit (SYNTHETIC / SIMULATION).

Executes:
- Part B1: High-tilt sensitivity (3-6 deg and 6-12 deg, 30 gradual + 30 rapid runs per arm)
  + impossible-value check audit (tilt > 15 deg vs manifest [0.05, 3.0] deg vs 3.5 deg rule).
- Part B2: Realistic-depth Knothe sigma = H / (tanB * sqrt(2*pi)) for H in {50, 100, 200, 300} m and
  tanB in {1.5, 2.0, 2.5}, generating 10 gradual + 10 rapid runs for every cell with sigma > 1.5 radii.
- Part C: Timescale audit across all gradual runs in the frozen manifest (outputs/datasets_v3/manifest.csv),
  plus alternative compression-factor columns on the normal-run false-alarm table.

Strictly evaluation-only: no frozen file or model is modified or refit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
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
    alarm as A,
    confounders as cf,
    dataset as ds,
    engine,
    eval_extras as EE,
    eval_extras_v2 as V2,
    labels as L,
    literature_checks as LC,
    sensors as S,
    shapes as SH,
    versioning as VR,
)

MODELS = ("threshold_rule", "logistic_regression", "random_forest")
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
        fontsize=9.2,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="square,pad=0.35", facecolor="#d32f2f", edgecolor="none"),
    )


def ci_str(pt: float, lo: float, hi: float, d: int = 3) -> str:
    if pd.isna(pt):
        return "N/A"
    return f"{pt:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _boot_mean(arr: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    pt = float(np.mean(arr))
    if n_boot <= 0 or len(arr) == 1:
        return pt, pt, pt
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = arr[idx].mean(axis=1)
    return pt, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _boot_sum(arr: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[int, float, float]:
    if len(arr) == 0:
        return 0, 0.0, 0.0
    pt = int(np.sum(arr))
    if n_boot <= 0 or len(arr) == 1:
        return pt, float(pt), float(pt)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    sums = arr[idx].sum(axis=1)
    return pt, float(np.percentile(sums, 2.5)), float(np.percentile(sums, 97.5))


def score_arm(
    arm_name: str,
    df_ev: pd.DataFrame,
    man_ev: pd.DataFrame,
    models: dict,
    ops: dict,
    n_boot: int,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Score one sensitivity arm on all 3 frozen models and 3 regimes."""
    det_rows, snr_rows, conf_rows, conv_rows = [], [], [], []
    snr_map = man_ev.set_index("simulation_id")["snr_vs_imu_noise"]
    regs_ev = {m: V2.regimes_for(m, models[m], ops, df_ev) for m in MODELS}

    for mname in MODELS:
        for rname, w_preds_ev, k, m in regs_ev[mname]:
            det_rec = V2.run_detection_records(df_ev, w_preds_ev, k, m)
            rng_d = np.random.default_rng(np.random.SeedSequence([int(seed), 1301]))
            for sub_name, sub_df in (
                ("all_events", det_rec),
                ("gradual", det_rec[det_rec["run_class"] == "gradual"]),
                ("rapid", det_rec[det_rec["run_class"] == "rapid"]),
            ):
                arr_d = sub_df["detected"].to_numpy(dtype=float)
                pt_d, lo_d, hi_d = _boot_mean(arr_d, n_boot, rng_d)
                pt_m, lo_m, hi_m = _boot_mean(1.0 - arr_d, n_boot, rng_d)
                det_rows.append({
                    "arm": arm_name,
                    "subset": sub_name,
                    "model": mname,
                    "regime": rname,
                    "k": k,
                    "m": m,
                    "n_runs": len(sub_df),
                    "exact_detect_rate": round(pt_d, 4),
                    "exact_detect_ci_lo": round(lo_d, 4),
                    "exact_detect_ci_hi": round(hi_d, 4),
                    "missed_event_rate": round(pt_m, 4),
                    "missed_event_ci_lo": round(lo_m, 4),
                    "missed_event_ci_hi": round(hi_m, 4),
                })

            for cls in ("gradual", "rapid"):
                pr = V2.per_run_class_table(df_ev, w_preds_ev, det_rec, cls)
                pr["snr"] = pr["simulation_id"].map(snr_map)
                pr["snr_bin"] = pr["snr"].map(V2.snr_bin_of)
                s_all = V2.summarize_class(pr, n_boot, seed)
                snr_rows.append({
                    "arm": arm_name, "model": mname, "regime": rname, "class": cls, "snr_bin": "ALL",
                    **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_all.items()},
                })
                for b_name, _, _ in V2.SNR_BINS:
                    s_b = V2.summarize_class(pr[pr["snr_bin"] == b_name], n_boot, seed)
                    snr_rows.append({
                        "arm": arm_name, "model": mname, "regime": rname, "class": cls, "snr_bin": b_name,
                        **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_b.items()},
                    })

            # Gradual vs sensor_fault confusion counts
            alarm_preds_ev, _ = A.apply_persistence_df(df_ev, w_preds_ev, k=k, m=m)
            tmask = L.trainable(df_ev).to_numpy()
            sub_t = df_ev.loc[tmask].copy()
            sub_t["_wpred"] = np.asarray(w_preds_ev)[tmask]
            sub_t["_apred"] = np.asarray(alarm_preds_ev)[tmask]

            g_runs = sub_t[sub_t["run_class"] == "gradual"]
            per_run_w_sf, per_run_a_sf = [], []
            per_run_w_rap, per_run_w_norm, per_run_w_grad = [], [], []
            for _, rg in g_runs.groupby("simulation_id", sort=False):
                gm = rg["label"].to_numpy() == "gradual"
                wp = rg["_wpred"].to_numpy()[gm]
                ap = rg["_apred"].to_numpy()[gm]
                per_run_w_sf.append(int((wp == "sensor_fault").sum()))
                per_run_a_sf.append(int((ap == "sensor_fault").sum()))
                per_run_w_rap.append(int((wp == "rapid").sum()))
                per_run_w_norm.append(int((wp == "normal").sum()))
                per_run_w_grad.append(int((wp == "gradual").sum()))

            rng_c = np.random.default_rng(np.random.SeedSequence([int(seed), 1305]))
            w_sf_pt, w_sf_lo, w_sf_hi = _boot_sum(np.array(per_run_w_sf), n_boot, rng_c)
            a_sf_pt, a_sf_lo, a_sf_hi = _boot_sum(np.array(per_run_a_sf), n_boot, rng_c)
            n_grad_win = int((g_runs["label"] == "gradual").sum())
            conf_rows.append({
                "arm": arm_name,
                "model": mname,
                "regime": rname,
                "n_gradual_runs": int(g_runs["simulation_id"].nunique()),
                "n_gradual_windows": n_grad_win,
                "win_gradual_to_sensor_fault": w_sf_pt,
                "win_gradual_to_sensor_fault_ci_lo": round(w_sf_lo, 1),
                "win_gradual_to_sensor_fault_ci_hi": round(w_sf_hi, 1),
                "win_gradual_to_sensor_fault_rate": round(w_sf_pt / max(n_grad_win, 1), 4),
                "alarm_gradual_to_sensor_fault": a_sf_pt,
                "alarm_gradual_to_sensor_fault_ci_lo": round(a_sf_lo, 1),
                "alarm_gradual_to_sensor_fault_ci_hi": round(a_sf_hi, 1),
                "alarm_gradual_to_sensor_fault_rate": round(a_sf_pt / max(n_grad_win, 1), 4),
                "win_gradual_to_gradual": int(np.sum(per_run_w_grad)),
                "win_gradual_to_rapid": int(np.sum(per_run_w_rap)),
                "win_gradual_to_normal": int(np.sum(per_run_w_norm)),
            })

    # Convergence distributions on active windows
    df_ev_snr = df_ev.copy()
    df_ev_snr["snr"] = df_ev_snr["simulation_id"].map(snr_map)
    active_mask = L.trainable(df_ev_snr).to_numpy() & (df_ev_snr["label"].isin(["gradual", "rapid"]).to_numpy())
    for sub_label, mask in (
        ("active_windows_all_snr", active_mask),
        ("active_windows_snr_gt_2", active_mask & (df_ev_snr["snr"].to_numpy() > 2.0)),
    ):
        sub_c = df_ev_snr.loc[mask]
        if len(sub_c) == 0:
            continue
        row_c: dict[str, float | int | str] = {
            "arm": arm_name,
            "subset": sub_label,
            "n_windows": len(sub_c),
            "n_runs": int(sub_c["simulation_id"].nunique()),
        }
        for col in ("dir_convergence_residual_m", "dir_convergence_weighted_cos", "dir_pairwise_cos_mean"):
            vals = sub_c[col].to_numpy(dtype=float)
            row_c[f"{col}_mean"] = round(float(np.mean(vals)), 4)
            row_c[f"{col}_std"] = round(float(np.std(vals, ddof=1)), 4)
            for q in (10, 25, 50, 75, 90):
                row_c[f"{col}_p{q}"] = round(float(np.percentile(vals, q)), 4)
        conv_rows.append(row_c)

    return det_rows, snr_rows, conf_rows, conv_rows


def plot_b1_high_tilt(det_df: pd.DataFrame, conf_df: pd.DataFrame, out_path: Path) -> None:
    """Plot B1 high-tilt sensitivity: gradual vs rapid run detection and gradual->sensor_fault confusion rate."""
    arms = ["gaussian_control", "high_tilt_3_6", "high_tilt_6_12"]
    labels = ["Control\n(0.05 - 3 deg)", "High Tilt Arm 1\n(3 - 6 deg)", "High Tilt Arm 2\n(6 - 12 deg)"]
    colors = {"threshold_rule": "#f57c00", "logistic_regression": "#388e3c", "random_forest": "#1976d2"}
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.2))
    x = np.arange(len(arms))
    w = 0.25

    for ax, sub_name, title in (
        (axes[0], "gradual", "Gradual Run Detection Rate (episode_fa<=1)"),
        (axes[1], "rapid", "Rapid Run Detection Rate (episode_fa<=1)"),
    ):
        sub = det_df[(det_df["regime"] == "episode_fa<=1 (k=3,m=3)") & (det_df["subset"] == sub_name)]
        for i, mname in enumerate(MODELS):
            ms = sub[sub["model"] == mname].set_index("arm").reindex(arms)
            y = ms["exact_detect_rate"].to_numpy(dtype=float)
            lo = np.maximum(0, y - ms["exact_detect_ci_lo"].to_numpy(dtype=float))
            hi = np.maximum(0, ms["exact_detect_ci_hi"].to_numpy(dtype=float) - y)
            ax.bar(x + (i - 1) * w, y, width=w, color=colors[mname], alpha=0.88, label=mname, yerr=np.vstack([lo, hi]), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_ylim(0, 1.08)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("Exact Run Detection Rate", fontsize=9.5)

    # Panel 3: Gradual -> sensor_fault window confusion rate
    ax3 = axes[2]
    sub_c = conf_df[conf_df["regime"] == "episode_fa<=1 (k=3,m=3)"]
    for i, mname in enumerate(MODELS):
        ms = sub_c[sub_c["model"] == mname].set_index("arm").reindex(arms)
        y = ms["win_gradual_to_sensor_fault_rate"].to_numpy(dtype=float)
        ax3.bar(x + (i - 1) * w, y, width=w, color=colors[mname], alpha=0.88, label=mname)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=8.8)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Fraction of Gradual Windows -> sensor_fault", fontsize=9.5)
    ax3.set_title("Gradual -> sensor_fault Window Confusion Rate", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6, axis="y")
    ax3.legend(loc="best", fontsize=8)

    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.05, right=0.98, wspace=0.22)
    _add_banner(fig, "Part B1: High-Tilt Sensitivity (3-6 deg and 6-12 deg vs Control)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_b2_realistic_depth(
    grid_df: pd.DataFrame,
    det_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    conv_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Plot B2 realistic-depth cells ordered by sigma_radii vs control."""
    above = grid_df[grid_df["above_main_cap_1p5"]].sort_values("sigma_radii")
    cell_ids = ["gaussian_control"] + above["cell_id"].tolist()
    x_labels = ["Control\n(0.4-1.5R)"] + [
        f"{r['cell_id']}\n({r['sigma_radii']:.2f}R)" for _, r in above.iterrows()
    ]
    x = np.arange(len(cell_ids))
    colors = {"threshold_rule": "#f57c00", "logistic_regression": "#388e3c", "random_forest": "#1976d2"}

    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.3))
    # 1) All-event exact detection rate at episode_fa<=1
    sub_d = det_df[(det_df["regime"] == "episode_fa<=1 (k=3,m=3)") & (det_df["subset"] == "all_events")]
    for mname in MODELS:
        ms = sub_d[sub_d["model"] == mname].set_index("arm").reindex(cell_ids)
        axes[0].plot(x, ms["exact_detect_rate"].to_numpy(float), marker="o", linewidth=2.0, color=colors[mname], label=mname)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels, fontsize=7.5, rotation=25)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Exact Run Detection Rate (episode_fa<=1)", fontsize=9.2)
    axes[0].set_title("Run Detection Rate vs Knothe Spread (sigma/R)", fontsize=9.8, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="best", fontsize=8)

    # 2) Gradual -> sensor_fault window confusion rate
    sub_c = conf_df[conf_df["regime"] == "episode_fa<=1 (k=3,m=3)"]
    for mname in MODELS:
        ms = sub_c[sub_c["model"] == mname].set_index("arm").reindex(cell_ids)
        axes[1].plot(x, ms["win_gradual_to_sensor_fault_rate"].to_numpy(float), marker="s", linewidth=2.0, color=colors[mname], label=mname)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, fontsize=7.5, rotation=25)
    axes[1].set_ylabel("Gradual -> sensor_fault Window Rate", fontsize=9.2)
    axes[1].set_title("Gradual -> sensor_fault Confusion vs Spread", fontsize=9.8, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="best", fontsize=8)

    # 3) Convergence residual & pairwise cosine (SNR > 2)
    sub_v = conv_df[conv_df["subset"] == "active_windows_snr_gt_2"].set_index("arm").reindex(cell_ids)
    ax3 = axes[2]
    ax3.plot(x, sub_v["dir_convergence_residual_m_p50"].to_numpy(float), "o-", color="#d32f2f", linewidth=2.0, label="Residual p50 (m)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontsize=7.5, rotation=25)
    ax3.set_ylabel("Median Convergence Residual (m, SNR>2)", fontsize=9.2, color="#d32f2f")
    ax3.set_title("Convergence Residual & Pairwise Cosine (SNR>2)", fontsize=9.8, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3_r = ax3.twinx()
    ax3_r.plot(x, sub_v["dir_pairwise_cos_mean_p50"].to_numpy(float), "^--", color="#1976d2", linewidth=1.8, label="Pairwise Cos p50")
    ax3_r.set_ylabel("Median Pairwise Cos [-1, 1]", fontsize=9.0, color="#1976d2")
    ax3_r.set_ylim(-0.3, 1.05)

    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.05, right=0.94, wspace=0.26)
    _add_banner(fig, "Part B2: Realistic-Depth Knothe Spread Sensitivity (PLACEHOLDER Grid)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_c_timescale_audit(per_run_df: pd.DataFrame, out_path: Path) -> None:
    """Plot distribution of implied physical tilt rates (deg/day at 360x vs 10,800x and 30,000x)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.1))
    r_360 = per_run_df["implied_tilt_rate_deg_per_day_at_360x"].to_numpy(float)
    r_10800 = per_run_df["implied_tilt_rate_deg_per_day_at_10800x_30d_run"].to_numpy(float)
    r_30000 = per_run_df["implied_tilt_rate_deg_per_day_at_30000x"].to_numpy(float)

    bins = np.logspace(-3.2, 1.0, 28)
    axes[0].hist(r_360, bins=bins, color="#1976d2", alpha=0.82, edgecolor="k", label="Frozen 360x (240s = 24h)")
    axes[0].axvline(0.1, color="#d32f2f", linestyle="--", linewidth=2.2, label="0.1 deg/day (DERIVED, ORDER OF MAGNITUDE)")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Implied Physical Tilt Rate (deg / day, log scale)", fontsize=9.5)
    axes[0].set_ylabel("Gradual Run Count (Frozen Manifest, N=195)", fontsize=9.5)
    axes[0].set_title("Frozen 360x Compression vs 0.1 deg/day Reference", fontsize=10, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="upper left", fontsize=8.3)

    axes[1].hist(r_360, bins=bins, alpha=0.55, color="#1976d2", label="360x (24h / run)")
    axes[1].hist(r_10800, bins=bins, alpha=0.55, color="#00897b", label="10,800x (30d / run)")
    axes[1].hist(r_30000, bins=bins, alpha=0.55, color="#7b1fa2", label="30,000x (83.3d / run)")
    axes[1].axvline(0.1, color="#d32f2f", linestyle="--", linewidth=2.2, label="0.1 deg/day (DERIVED OOM)")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Implied Physical Tilt Rate (deg / day, log scale)", fontsize=9.5)
    axes[1].set_ylabel("Gradual Run Count", fontsize=9.5)
    axes[1].set_title("Effect of Relabelling Compression Factor (Scores Unchanged)", fontsize=10, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper left", fontsize=8.3)

    fig.subplots_adjust(top=0.83, bottom=0.14, left=0.06, right=0.97, wspace=0.20)
    _add_banner(fig, "Part C: Timescale Audit of Frozen Gradual Runs (N=195)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_all_literature_checks(
    lit_cfg_path: Path,
    out_dir: Path,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    t0 = time.time()
    hashes_before = snapshot_frozen_hashes()
    out_dir.mkdir(parents=True, exist_ok=True)

    scen_cfg = engine.load_config(ROOT / "configs/scenario_default.yaml")
    sens_cfg = S.load_sensor_config(ROOT / "configs/sensor_default.yaml")
    conf_cfg = cf.load_confounder_config(ROOT / "configs/confounder_default.yaml")
    ds_cfg = ds.load_dataset_config(ROOT / "configs/dataset_default.yaml")
    profile = S.load_noise_profile(ROOT / "configs/noise_profile.json")
    shape_cfg = SH.load_shape_config(ROOT / "configs/shape_mismatch.yaml")
    lit_cfg = LC.load_literature_config(lit_cfg_path)

    # 1. Verify seed disjointness against outputs/datasets_v3 and outputs/eval_shape_mismatch
    man_v3 = pd.read_csv(ROOT / "outputs/datasets_v3/manifest.csv")
    existing_seeds = set(man_v3["seed"].astype(int).tolist())
    sm_man_path = ROOT / "outputs/eval_shape_mismatch/manifest.csv"
    if sm_man_path.is_file():
        sm_man = pd.read_csv(sm_man_path)
        existing_seeds |= set(sm_man["seed"].astype(int).tolist())

    arms = LC.build_b1_and_b2_specs(lit_cfg, ds.stratified_peaks)
    LC.assert_all_seeds_disjoint(arms, existing_seeds)

    # Save full 12-cell realistic-depth Knothe sigma grid
    grid_df = LC.build_depth_sigma_grid(lit_cfg)
    grid_df.to_csv(out_dir / "realistic_depth_sigma_grid.csv", index=False)

    # 2. Load or generate the 40 shared shape-mismatch normal runs and the 60 gaussian_control runs
    gen_args = (scen_cfg, sens_cfg, conf_cfg, ds_cfg, profile)
    sm_specs = SH.build_specs(shape_cfg, scen_cfg, ds.stratified_peaks)
    print("[literature_checks] generating/loading 40 shared normal runs and 60 gaussian_control runs ...", flush=True)
    norm_man, norm_lab, norm_feat = SH.generate_shape_set(
        "gaussian_control", sm_specs["normal"], ds.generate_one_run, shape_cfg, gen_args
    )
    ctrl_man, ctrl_lab, ctrl_feat = SH.generate_shape_set(
        "gaussian_control", sm_specs["gaussian_control"], ds.generate_one_run, shape_cfg, gen_args
    )
    ctrl_man["arm"] = "gaussian_control"
    ctrl_man["kind"] = "control"
    ctrl_man["H_m"] = np.nan
    ctrl_man["tan_beta"] = np.nan
    ctrl_man["sigma_m"] = ctrl_man["parameters_json"].map(lambda s: float(json.loads(s)["sx"]))
    ctrl_man["sigma_radii"] = ctrl_man["parameters_json"].map(
        lambda s: float(max(json.loads(s)["sx"], json.loads(s)["sy"]) / LC.RAD)
    )
    ctrl_man["s_max_m"] = ctrl_man["parameters_json"].map(lambda s: float(json.loads(s)["s_max"]))

    # 3. Generate B1 (high_tilt_3_6, high_tilt_6_12) and B2 (7 realistic-depth cells)
    arm_mans = {"gaussian_control": ctrl_man}
    arm_labs = {"gaussian_control": ctrl_lab}
    arm_feats = {"gaussian_control": ctrl_feat}

    for arm_name, ainfo in arms.items():
        print(f"[literature_checks] generating {len(ainfo['specs'])} runs for arm={arm_name} ({ainfo['kind']}) ...", flush=True)
        m_df, l_df, f_df = LC.generate_arm_runs(arm_name, ainfo, ds.generate_one_run, gen_args)
        arm_mans[arm_name] = m_df
        arm_labs[arm_name] = l_df
        arm_feats[arm_name] = f_df

    # Pooled B2 arm (all 7 cells with sigma > 1.5 radii = 140 runs)
    b2_cell_names = [k for k, v in arms.items() if v["kind"] == "B2_realistic_depth"]
    arm_mans["depth_all_gt_1p5"] = pd.concat([arm_mans[c] for c in b2_cell_names], ignore_index=True)
    arm_labs["depth_all_gt_1p5"] = pd.concat([arm_labs[c] for c in b2_cell_names], ignore_index=True)
    arm_feats["depth_all_gt_1p5"] = pd.concat([arm_feats[c] for c in b2_cell_names], ignore_index=True)

    # Save combined manifest, labels, features
    norm_man_tagged = norm_man.copy()
    norm_man_tagged["arm"] = "shared_normal_40"
    norm_man_tagged["kind"] = "normal"
    norm_man_tagged["H_m"] = np.nan
    norm_man_tagged["tan_beta"] = np.nan
    norm_man_tagged["sigma_m"] = 0.0
    norm_man_tagged["sigma_radii"] = 0.0
    norm_man_tagged["s_max_m"] = 0.0

    all_non_pooled = ["gaussian_control"] + list(arms.keys())
    full_man = pd.concat([norm_man_tagged] + [arm_mans[a] for a in all_non_pooled], ignore_index=True)
    full_lab = pd.concat([norm_lab] + [arm_labs[a] for a in all_non_pooled], ignore_index=True)
    full_feat = pd.concat([norm_feat] + [arm_feats[a] for a in all_non_pooled], ignore_index=True)
    full_man.to_csv(out_dir / "manifest.csv", index=False)
    full_lab.to_csv(out_dir / "window_labels.csv", index=False)
    full_feat.to_csv(out_dir / "window_features.csv", index=False)

    # 4. Check impossible values on control, B1 arms, and B2 arms
    imp_rows = []
    for arm_name in ["gaussian_control", "high_tilt_3_6", "high_tilt_6_12", "depth_all_gt_1p5"] + b2_cell_names:
        imp_rows.append(LC.check_impossible_values_on_arm(arm_name, arm_mans[arm_name], arm_feats[arm_name]))
    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(out_dir / "impossible_value_check_report.csv", index=False)

    # 5. Score frozen v3 models across all arms
    print("[literature_checks] scoring frozen v3 models across all arms ...", flush=True)
    models = pickle.load(open(ROOT / "outputs/eval_v3/trained_models_v3.pkl", "rb"))
    ops = pickle.load(open(ROOT / "outputs/eval_v3/operating_points_v3.pkl", "rb"))

    det_all, snr_all, conf_all, conv_all = [], [], [], []
    eval_arm_order = ["gaussian_control", "high_tilt_3_6", "high_tilt_6_12", "depth_all_gt_1p5"] + b2_cell_names
    for arm_name in eval_arm_order:
        d_r, s_r, c_r, v_r = score_arm(
            arm_name, arm_feats[arm_name], arm_mans[arm_name], models, ops, n_boot=n_boot, seed=seed
        )
        det_all.extend(d_r)
        snr_all.extend(s_r)
        conf_all.extend(c_r)
        conv_all.extend(v_r)

    det_df = pd.DataFrame(det_all)
    snr_df = pd.DataFrame(snr_all)
    conf_df = pd.DataFrame(conf_all)
    conv_df = pd.DataFrame(conv_all)

    det_df.to_csv(out_dir / "literature_detection_rates.csv", index=False)
    snr_df.to_csv(out_dir / "literature_recall_by_snr.csv", index=False)
    conf_df.to_csv(out_dir / "literature_gradual_sensor_fault_confusion.csv", index=False)
    conv_df.to_csv(out_dir / "literature_convergence_distributions.csv", index=False)

    # 6. Normal-run false alarms with both frozen 360x and alternative 30-day compression factors (Part C)
    fa_rows = []
    alt_comps = [float(x) for x in lit_cfg["timescale_audit"]["alternative_compression_factors"]]
    regs_norm = {m: V2.regimes_for(m, models[m], ops, norm_feat) for m in MODELS}
    for mname in MODELS:
        for rname, w_preds, k, m in regs_norm[mname]:
            res_fa = EE.evaluate_alarm_level_with_hours(
                norm_feat, w_preds, manifest_df=norm_man, k=k, m=m, n_boot=n_boot, seed=seed
            )
            dur = 240.0
            e, (elo, ehi) = res_fa["fa_episodes_per_run"], res_fa["fa_episodes_per_run_ci95"]
            hB = V2.hours_sensor(dur)
            row_fa: dict[str, float | int | str] = {
                "model": mname,
                "regime": rname,
                "k": k,
                "m": m,
                "n_normal_runs": res_fa["n_normal_runs"],
                "fa_episodes_per_run": round(e, 4),
                "fa_episodes_per_run_ci_lo": round(elo, 4),
                "fa_episodes_per_run_ci_hi": round(ehi, 4),
                "B_sensor_fa_per_hour_1x": round(V2.rate_per_hour(e, hB), 4),
                "B_sensor_one_fa_every": V2.fmt_every(V2.every_n_hours(V2.rate_per_hour(e, hB))),
            }
            for comp in alt_comps:
                hA = V2.hours_slow(dur, comp)
                tag = f"{int(round(comp))}x"
                row_fa[f"A_slow_hours_per_run_{tag}"] = round(hA, 2)
                row_fa[f"A_slow_fa_per_hour_{tag}"] = round(V2.rate_per_hour(e, hA), 6)
                row_fa[f"A_slow_one_fa_every_{tag}"] = V2.fmt_every(V2.every_n_hours(V2.rate_per_hour(e, hA)))
            fa_rows.append(row_fa)
    fa_df = pd.DataFrame(fa_rows)
    fa_df.to_csv(out_dir / "normal_false_alarms_with_alt_compression.csv", index=False)

    # 7. Part C: Timescale audit on all gradual runs in the frozen manifest (outputs/datasets_v3/manifest.csv)
    print("[literature_checks] running Part C timescale audit on frozen manifest ...", flush=True)
    ref_rate = float(lit_cfg["timescale_audit"]["derived_reference_rate_deg_per_day"])
    ts_runs_df, ts_sum_df = LC.audit_frozen_gradual_timescales(man_v3, ref_rate_deg_per_day=ref_rate)
    ts_runs_df.to_csv(out_dir / "timescale_audit_gradual_runs.csv", index=False)
    ts_sum_df.to_csv(out_dir / "timescale_audit_summary.csv", index=False)

    # 8. Write Markdown tables and render plots
    _write_markdown_summaries(out_dir, grid_df, imp_df, det_df, snr_df, conf_df, conv_df, fa_df, ts_sum_df)
    print("[literature_checks] rendering plots with red banner ...", flush=True)
    plot_b1_high_tilt(det_df, conf_df, out_dir / "high_tilt_sensitivity.png")
    plot_b2_realistic_depth(grid_df, det_df, conf_df, conv_df, out_dir / "realistic_depth_sensitivity.png")
    plot_c_timescale_audit(ts_runs_df, out_dir / "timescale_audit_gradual_rates.png")

    hashes_after = snapshot_frozen_hashes()
    assert hashes_before == hashes_after, "Frozen file modified during run_literature_checks!"

    cfg_files = {
        "scenario": ROOT / "configs/scenario_default.yaml",
        "sensor": ROOT / "configs/sensor_default.yaml",
        "confounder": ROOT / "configs/confounder_default.yaml",
        "noise_profile": ROOT / "configs/noise_profile.json",
        "dataset": ROOT / "configs/dataset_default.yaml",
        "literature_checks": lit_cfg_path,
    }
    VR.write_dataset_metadata(
        out_dir,
        cfg_files,
        extra={
            "seed": seed,
            "n_boot": n_boot,
            "runtime_s": round(time.time() - t0, 2),
            "frozen_hashes_verified": True,
            "frozen_sha256": hashes_after,
        },
    )
    print(f"[literature_checks] DONE in {time.time() - t0:.1f} s; outputs in {out_dir}", flush=True)
    return {
        "grid": grid_df,
        "impossible": imp_df,
        "detection": det_df,
        "snr": snr_df,
        "confusion": conf_df,
        "convergence": conv_df,
        "false_alarms": fa_df,
        "timescale_summary": ts_sum_df,
    }


def _write_markdown_summaries(
    out_dir: Path,
    grid_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    det_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    conv_df: pd.DataFrame,
    fa_df: pd.DataFrame,
    ts_sum_df: pd.DataFrame,
) -> None:
    banner = "> **SYNTHETIC / SIMULATION** — Literature-basis sensitivity checks on frozen v3 models; nothing here is validated on real data."

    # 1. Knothe grid markdown
    md_g = [
        "# Realistic-Depth Knothe Spread Grid (PLACEHOLDER Grid; SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| cell_id | H (m) | tan(beta) | R = H/tan(beta) (m) | sigma (m) | sigma (network radii, R_net=14.14m) | Above 1.5-radius main cap? |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in grid_df.iterrows():
        md_g.append(
            f"| `{r['cell_id']}` | {r['H_m']:.0f} | {r['tan_beta']:.1f} | {r['R_influence_m']:.2f} | "
            f"**{r['sigma_m']:.2f}** | **{r['sigma_radii']:.3f}** | {'YES (simulated)' if r['above_main_cap_1p5'] else 'NO (<=1.5, in main range)'} |"
        )
    (out_dir / "realistic_depth_sigma_grid.md").write_text("\n".join(md_g) + "\n", encoding="utf-8")

    # 2. Impossible value check markdown
    md_i = [
        "# Impossible-Value Check Audit on B1 & B2 Arms (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| arm | runs | windows | max window node tilt (deg) | runs > 15 deg limit | runs >= 3.5 deg (ThresholdRule implausible) | validation_checks.check_impossible messages |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in imp_df.iterrows():
        md_i.append(
            f"| `{r['arm']}` | {r['n_runs']} | {r['n_windows']} | **{r['max_window_node_mean_deg']:.3f}** | "
            f"**{r['runs_exceeding_15deg_normal_limit']}** ({r['windows_exceeding_15deg_normal_limit']} win) | "
            f"**{r['runs_exceeding_3p5deg_threshold_rule_implausible']}** ({r['windows_exceeding_3p5deg_threshold_rule_implausible']} win) | "
            f"{r['validation_check_messages']} |"
        )
    (out_dir / "impossible_value_check_report.md").write_text("\n".join(md_i) + "\n", encoding="utf-8")

    # 3. Detection rates markdown
    md_d = [
        "# Run-Level Exact Detection & Missed-Event Rates Across B1 and B2 Arms (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| arm | subset | runs | model | regime | exact detect rate [95% CI] | missed-event rate [95% CI] |",
        "|---|---|---:|---|---|---|---|",
    ]
    for _, r in det_df[det_df["regime"] != "raw (k=1)"].iterrows():
        md_d.append(
            f"| `{r['arm']}` | {r['subset']} | {r['n_runs']} | {r['model']} | {r['regime']} | "
            f"**{ci_str(r['exact_detect_rate'], r['exact_detect_ci_lo'], r['exact_detect_ci_hi'])}** | "
            f"{ci_str(r['missed_event_rate'], r['missed_event_ci_lo'], r['missed_event_ci_hi'])} |"
        )
    (out_dir / "literature_detection_rates.md").write_text("\n".join(md_d) + "\n", encoding="utf-8")

    # 4. Confusion counts markdown
    md_c = [
        "# Gradual vs Sensor_Fault Confusion Counts Across B1 & B2 Arms (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| arm | model | regime | gradual win | win: gradual->sensor_fault [95% CI] (rate) | alarm: gradual->sensor_fault [95% CI] | win->gradual | win->rapid | win->normal |",
        "|---|---|---|---:|---|---|---:|---:|---:|",
    ]
    for _, r in conf_df[conf_df["regime"] != "raw (k=1)"].iterrows():
        md_c.append(
            f"| `{r['arm']}` | {r['model']} | {r['regime']} | {r['n_gradual_windows']} | "
            f"**{r['win_gradual_to_sensor_fault']}** [{r['win_gradual_to_sensor_fault_ci_lo']:.1f}, {r['win_gradual_to_sensor_fault_ci_hi']:.1f}] ({r['win_gradual_to_sensor_fault_rate']:.3f}) | "
            f"**{r['alarm_gradual_to_sensor_fault']}** [{r['alarm_gradual_to_sensor_fault_ci_lo']:.1f}, {r['alarm_gradual_to_sensor_fault_ci_hi']:.1f}] | "
            f"{r['win_gradual_to_gradual']} | {r['win_gradual_to_rapid']} | {r['win_gradual_to_normal']} |"
        )
    (out_dir / "literature_gradual_sensor_fault_confusion.md").write_text("\n".join(md_c) + "\n", encoding="utf-8")

    # 5. Convergence distributions markdown
    md_v = [
        "# Radial Convergence Feature Distributions Across B1 & B2 Arms (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| arm | subset | windows | residual_m p10 / **p50** / p90 (mean) | weighted_cos p10 / **p50** / p90 (mean) | pairwise_cos p50 |",
        "|---|---|---:|---|---|---:|",
    ]
    for _, r in conv_df.iterrows():
        md_v.append(
            f"| `{r['arm']}` | {r['subset']} | {r['n_windows']} | "
            f"{r['dir_convergence_residual_m_p10']:.2f} / **{r['dir_convergence_residual_m_p50']:.2f}** / {r['dir_convergence_residual_m_p90']:.2f} ({r['dir_convergence_residual_m_mean']:.2f}) | "
            f"{r['dir_convergence_weighted_cos_p10']:.3f} / **{r['dir_convergence_weighted_cos_p50']:.3f}** / {r['dir_convergence_weighted_cos_p90']:.3f} ({r['dir_convergence_weighted_cos_mean']:.3f}) | "
            f"{r['dir_pairwise_cos_mean_p50']:.3f} |"
        )
    (out_dir / "literature_convergence_distributions.md").write_text("\n".join(md_v) + "\n", encoding="utf-8")

    # 6. Timescale audit markdown
    md_t = [
        "# Part C: Timescale Audit of Frozen Gradual Runs (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "> **Reference Notice:** Real-world comparison value `0.1 deg/day` (from ~`60 mm/m` [`3.4 deg`] over ~`30 days`) is **DERIVED, ORDER OF MAGNITUDE**, not a cited value.",
        "",
        "| subset | gradual runs | faster than 0.1 deg/day (share) | implied deg/day at 360x: min / p10 / **p50** / mean / p90 / max | speedup factor vs 0.1 deg/day: min / **p50** / mean / p90 / max |",
        "|---|---:|---|---|---|",
    ]
    for _, r in ts_sum_df.iterrows():
        md_t.append(
            f"| `{r['subset']}` | {r['n_gradual_runs']} | **{r['n_faster_than_0p1_deg_per_day']} / {r['n_gradual_runs']}** ({r['frac_faster_than_0p1_deg_per_day']*100:.1f}%) | "
            f"{r['rate_deg_per_day_min']:.3f} / {r['rate_deg_per_day_p10']:.3f} / **{r['rate_deg_per_day_p50_median']:.3f}** / {r['rate_deg_per_day_mean']:.3f} / {r['rate_deg_per_day_p90']:.3f} / {r['rate_deg_per_day_max']:.3f} | "
            f"{r['factor_vs_0p1_min']:.2f}x / **{r['factor_vs_0p1_p50_median']:.2f}x** / {r['factor_vs_0p1_mean']:.2f}x / {r['factor_vs_0p1_p90']:.2f}x / {r['factor_vs_0p1_max']:.2f}x |"
        )
    (out_dir / "timescale_audit_summary.md").write_text("\n".join(md_t) + "\n", encoding="utf-8")

    # 7. Normal false alarms with alternative compression factors
    md_fa = [
        "# Normal-Run False Alarms Under Frozen 360x vs Alternative 30-Day Compression Factors (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| model | regime | FA eps / run [95% CI] | B: sensor time (1x, 0.0667h/run) | A (frozen 360x, 24h/run) | A_alt (10,000x, 666.7h/run) | A_alt (10,800x = 30d/run, 720h/run) | A_alt (30,000x, 2000h/run) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in fa_df.iterrows():
        md_fa.append(
            f"| {r['model']} | {r['regime']} | {ci_str(r['fa_episodes_per_run'], r['fa_episodes_per_run_ci_lo'], r['fa_episodes_per_run_ci_hi'])} | "
            f"{r['B_sensor_fa_per_hour_1x']:.2f}/h ({r['B_sensor_one_fa_every']}) | "
            f"{r['A_slow_fa_per_hour_360x']:.4f}/h ({r['A_slow_one_fa_every_360x']}) | "
            f"{r['A_slow_fa_per_hour_10000x']:.6f}/h ({r['A_slow_one_fa_every_10000x']}) | "
            f"{r['A_slow_fa_per_hour_10800x']:.6f}/h ({r['A_slow_one_fa_every_10800x']}) | "
            f"{r['A_slow_fa_per_hour_30000x']:.6f}/h ({r['A_slow_one_fa_every_30000x']}) |"
        )
    (out_dir / "normal_false_alarms_with_alt_compression.md").write_text("\n".join(md_fa) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/literature_checks.yaml")
    ap.add_argument("--out", default="outputs/eval_literature")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    run_all_literature_checks(Path(args.config), Path(args.out), n_boot=args.n_boot, seed=args.seed)
