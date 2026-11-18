#!/usr/bin/env python
"""Model-mismatch stress test runner (SYNTHETIC / SIMULATION).

Generates 30 gradual + 30 rapid runs for each of the 5 shapes (gaussian_control, trough_panel, asymmetric,
two_basin, planar_slope) plus 40 shared normal-only runs via the UNCHANGED dataset.generate_one_run pipeline,
then scores the FROZEN v3 models and FROZEN v3 operating points without any retraining, retuning, or selection.

Writes all tables, metadata, hashes, and plots to outputs/eval_shape_mismatch/.
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
    evaluate as ev,
    labels as L,
    node_geometry as ng,
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
    out = {}
    for rel in FROZEN_CHECK_FILES:
        p = ROOT / rel
        if p.is_file():
            out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


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


def plot_shape_examples(scen_cfg: dict, shape_cfg: dict, out_path: Path) -> None:
    """Plot 2-D settlement contours and corner downslope vectors for the 5 shapes on one representative seed."""
    fig, axes = plt.subplots(1, 5, figsize=(17.5, 3.9), sharex=True, sharey=True)
    xs = np.linspace(-5.0, 25.0, 121)
    ys = np.linspace(-5.0, 25.0, 121)
    X, Y = np.meshgrid(xs, ys)
    seed = int(shape_cfg["seed_base"]) + 4

    for ax, sh in zip(axes, SH.SHAPES):
        p = SH.sample_shape_parameters(scen_cfg, shape_cfg, sh, seed=seed, mode="gradual", peak_tilt_deg=1.5)
        h, _, _ = SH.h_and_grad(sh, X, Y, p)
        W_mm = -p["s_max"] * h * 1000.0
        cs = ax.contourf(X, Y, W_mm, levels=16, cmap="viridis")
        plt.colorbar(cs, ax=ax, fraction=0.046, pad=0.04, label="w (mm)")
        # Plot 20x20 square and 4 corner nodes + downslope arrows
        sq_x = [0, 20, 20, 0, 0]
        sq_y = [0, 0, 20, 20, 0]
        ax.plot(sq_x, sq_y, "w--", linewidth=1.1, alpha=0.85)
        for nid, (nx, ny) in ng.NODE_XY.items():
            _, ghx, ghy = SH.h_and_grad(sh, nx, ny, p)
            # Downslope direction in world frame is (-dw/dx, -dw/dy) = s * (dh/dx, dh/dy)
            ux, uy = float(ghx), float(ghy)
            nrm = np.hypot(ux, uy)
            if nrm > 1e-9:
                ux, uy = 4.0 * ux / nrm, 4.0 * uy / nrm
            ax.plot(nx, ny, "wo", markersize=5, markeredgecolor="k")
            ax.arrow(nx, ny, ux, uy, head_width=1.1, head_length=1.1, fc="#ff5252", ec="k", linewidth=0.8)
        ax.set_title(sh.replace("_", "\n"), fontsize=9.5, fontweight="bold")
        ax.set_xlabel("x (m)", fontsize=8.5)
        ax.set_xlim(-5, 25)
        ax.set_ylim(-5, 25)
        ax.set_aspect("equal")
    axes[0].set_ylabel("y (m)", fontsize=8.5)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.04, right=0.97, wspace=0.28)
    _add_banner(fig, "Example w(x,y) Fields & Corner Downslope Vectors (peak_tilt = 1.5 deg)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_detection_by_shape(det_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of run-level exact detection rate and missed-event rate across shapes."""
    regimes = ["episode_fa<=1 (k=3,m=3)", "strict_window_fa<=1"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)
    colors = {"threshold_rule": "#f57c00", "logistic_regression": "#388e3c", "random_forest": "#1976d2"}
    width = 0.25
    x = np.arange(len(SH.SHAPES))

    for ax, reg in zip(axes, regimes):
        sub = det_df[(det_df["regime"] == reg) & (det_df["subset"] == "all_events (60 runs)")]
        for m_idx, mname in enumerate(MODELS):
            msub = sub[sub["model"] == mname].set_index("shape").loc[list(SH.SHAPES)]
            y = msub["exact_detect_rate"].to_numpy(dtype=float)
            lo = y - msub["exact_detect_ci_lo"].to_numpy(dtype=float)
            hi = msub["exact_detect_ci_hi"].to_numpy(dtype=float) - y
            pos = x + (m_idx - 1) * width
            ax.bar(pos, y, width=width, color=colors[mname], alpha=0.88, label=mname,
                   yerr=np.vstack([np.maximum(0, lo), np.maximum(0, hi)]), capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in SH.SHAPES], fontsize=8.8)
        ax.set_ylim(0.0, 1.05)
        ax.set_title(f"Run Exact Detection Rate — {reg}", fontsize=10.5, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="upper right", fontsize=8.5)
    axes[0].set_ylabel("Exact Run Detection Rate (1 - Missed Rate)", fontsize=9.5)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.97, wspace=0.15)
    _add_banner(fig, "Frozen v3 Models Across 5 Subsidence Shapes (60 Event Runs / Shape)")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_recall_by_snr(snr_df: pd.DataFrame, out_path: Path) -> None:
    """Plot gradual and rapid window recall by SNR bin across shapes for Random Forest (episode_fa<=1)."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4), sharey=True)
    bins = [b[0] for b in V2.SNR_BINS]
    x = np.arange(len(bins))
    palette = {
        "gaussian_control": ("#1976d2", "o"),
        "trough_panel": ("#00897b", "s"),
        "asymmetric": ("#7b1fa2", "^"),
        "two_basin": ("#f57c00", "D"),
        "planar_slope": ("#d32f2f", "X"),
    }

    for ax, cls in zip(axes, ("gradual", "rapid")):
        sub = snr_df[
            (snr_df["model"] == "random_forest")
            & (snr_df["regime"] == "episode_fa<=1 (k=3,m=3)")
            & (snr_df["class"] == cls)
        ]
        for sh in SH.SHAPES:
            ssub = sub[sub["shape"] == sh].set_index("snr_bin").reindex(bins)
            col, mk = palette[sh]
            y_win = ssub["window_recall"].to_numpy(dtype=float)
            y_run = ssub["run_detect_rate"].to_numpy(dtype=float)
            ax.plot(x, y_win, marker=mk, linewidth=2.0, color=col, label=f"{sh} (win recall)")
            ax.plot(x, y_run, marker=mk, linewidth=1.2, linestyle="--", color=col, alpha=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels([f"SNR {b}" for b in bins], fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(
            f"Random Forest (episode_fa<=1): {cls} — Solid=Window Recall, Dashed=Run Detect",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Recall / Detection Rate", fontsize=9.5)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.06, right=0.97, wspace=0.15)
    _add_banner(fig, "Window Recall & Run Detection by SNR Bin")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_convergence_distributions(conv_df: pd.DataFrame, out_path: Path) -> None:
    """Plot median and IQR (p25-p75) & p10-p90 of dir_convergence_residual_m and dir_convergence_weighted_cos."""
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    sub_all = conv_df[conv_df["subset"] == "active_windows_all_snr"].set_index("shape").loc[list(SH.SHAPES)]
    sub_hi = conv_df[conv_df["subset"] == "active_windows_snr_gt_2"].set_index("shape").loc[list(SH.SHAPES)]
    x = np.arange(len(SH.SHAPES))
    w = 0.34

    for ax, feat, title, ylabel in (
        (axes[0], "dir_convergence_residual_m", "Fitted Convergence Residual (m) — Lower is More Convergent", "Residual (m)"),
        (axes[1], "dir_convergence_weighted_cos", "Magnitude-Weighted Convergence Cosine — Higher is More Convergent", "Weighted Cosine [-1, 1]"),
    ):
        for offset, sdf, lbl, col in (
            (-w / 2, sub_all, "All Active Windows", "#5c6bc0"),
            (+w / 2, sub_hi, "Active Windows (SNR > 2)", "#00897b"),
        ):
            med = sdf[f"{feat}_p50"].to_numpy(dtype=float)
            p25 = sdf[f"{feat}_p25"].to_numpy(dtype=float)
            p75 = sdf[f"{feat}_p75"].to_numpy(dtype=float)
            ax.bar(x + offset, med, width=w, color=col, alpha=0.85, label=f"{lbl} (median, IQR)",
                   yerr=np.vstack([med - p25, p75 - med]), capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace("_", "\n") for s in SH.SHAPES], fontsize=8.8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(True, linestyle=":", alpha=0.6, axis="y")
        ax.legend(loc="best", fontsize=8.5)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.06, right=0.97, wspace=0.22)
    _add_banner(fig, "Radial Convergence Feature Breakdown by Subsidence Shape")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_shape_mismatch(
    shape_cfg_path: Path,
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
    shape_cfg = SH.load_shape_config(shape_cfg_path)

    # Verify seed disjointness against frozen dataset manifest
    existing_man = pd.read_csv(ROOT / "outputs/datasets_v3/manifest.csv", usecols=["seed"])
    specs_by_key = SH.build_specs(shape_cfg, scen_cfg, ds.stratified_peaks)
    SH.assert_seeds_disjoint(specs_by_key, set(existing_man["seed"].astype(int).tolist()))

    # 1. Generate normal runs once (shared by all shapes) and 60 event runs per shape
    gen_args = (scen_cfg, sens_cfg, conf_cfg, ds_cfg, profile)
    print("[shape_mismatch] generating 40 shared normal runs ...", flush=True)
    norm_man, norm_lab, norm_feat = SH.generate_shape_set(
        "gaussian_control", specs_by_key["normal"], ds.generate_one_run, shape_cfg, gen_args
    )

    shape_mans, shape_labs, shape_feats = {}, {}, {}
    for sh in SH.SHAPES:
        print(f"[shape_mismatch] generating 60 runs (30 gradual, 30 rapid) for shape={sh} ...", flush=True)
        m_df, l_df, f_df = SH.generate_shape_set(sh, specs_by_key[sh], ds.generate_one_run, shape_cfg, gen_args)
        shape_mans[sh] = m_df
        shape_labs[sh] = l_df
        shape_feats[sh] = f_df

    # Save combined manifest, labels, features, and run metadata
    full_man = pd.concat([norm_man] + [shape_mans[s] for s in SH.SHAPES], ignore_index=True)
    full_lab = pd.concat([norm_lab] + [shape_labs[s] for s in SH.SHAPES], ignore_index=True)
    full_feat = pd.concat([norm_feat] + [shape_feats[s] for s in SH.SHAPES], ignore_index=True)
    full_man.to_csv(out_dir / "manifest.csv", index=False)
    full_lab.to_csv(out_dir / "window_labels.csv", index=False)
    full_feat.to_csv(out_dir / "window_features.csv", index=False)

    runs_meta = {
        r["simulation_id"]: {
            "shape": r["shape"],
            "split": r["split"],
            "run_class": r["run_class"],
            "seed": int(r["seed"]),
            "peak_tilt_deg": float(r["peak_tilt_deg"]),
            "snr_vs_imu_noise": float(r["snr_vs_imu_noise"]),
            "max_node_tilt_deg": float(r["max_node_tilt_deg"]),
            "mean_node_tilt_deg": float(r["mean_node_tilt_deg"]),
            "parameters": json.loads(r["parameters_json"]),
            "faults": json.loads(r["faults_json"]),
            "events": json.loads(r["events_json"]),
            "confounders": json.loads(r["confounders_json"]),
        }
        for _, r in full_man.iterrows()
    }
    (out_dir / "runs_metadata.json").write_text(json.dumps(runs_meta, indent=2), encoding="utf-8")

    # 2. Load FROZEN v3 models and FROZEN v3 operating points
    models = pickle.load(open(ROOT / "outputs/eval_v3/trained_models_v3.pkl", "rb"))
    ops = pickle.load(open(ROOT / "outputs/eval_v3/operating_points_v3.pkl", "rb"))

    # 3. Score each shape (60 event runs + 40 shared normal runs)
    print("[shape_mismatch] scoring frozen v3 models across all shapes ...", flush=True)
    det_rows = []
    snr_rows = []
    conf_rows = []
    fa_rows = []
    conv_rows = []

    for sh in SH.SHAPES:
        df_ev = shape_feats[sh]
        man_ev = shape_mans[sh]
        df_combined = pd.concat([df_ev, norm_feat], ignore_index=True)
        man_combined = pd.concat([man_ev, norm_man], ignore_index=True)
        snr_map = man_ev.set_index("simulation_id")["snr_vs_imu_noise"]

        # Regimes on combined (event + normal) and on event-only
        regs_comb = {m: V2.regimes_for(m, models[m], ops, df_combined) for m in MODELS}
        regs_ev = {m: V2.regimes_for(m, models[m], ops, df_ev) for m in MODELS}

        # 3a. False alarms on the 40 shared normal runs (both definitions A and B)
        for mname in MODELS:
            for rname, w_preds, k, m in regs_comb[mname]:
                res_fa = EE.evaluate_alarm_level_with_hours(
                    df_combined, w_preds, manifest_df=man_combined, k=k, m=m, n_boot=n_boot, seed=seed
                )
                dur, comp = 240.0, 360.0
                hA, hB = V2.hours_slow(dur, comp), V2.hours_sensor(dur)
                e, (elo, ehi) = res_fa["fa_episodes_per_run"], res_fa["fa_episodes_per_run_ci95"]
                fa_rows.append({
                    "shape": sh,
                    "model": mname,
                    "regime": rname,
                    "k": k,
                    "m": m,
                    "n_normal_runs": res_fa["n_normal_runs"],
                    "fa_episodes_per_run": round(e, 4),
                    "fa_episodes_per_run_ci_lo": round(elo, 4),
                    "fa_episodes_per_run_ci_hi": round(ehi, 4),
                    "A_slow_fa_per_hour": round(V2.rate_per_hour(e, hA), 5),
                    "A_slow_one_fa_every": V2.fmt_every(V2.every_n_hours(V2.rate_per_hour(e, hA))),
                    "B_sensor_fa_per_hour": round(V2.rate_per_hour(e, hB), 4),
                    "B_sensor_one_fa_every": V2.fmt_every(V2.every_n_hours(V2.rate_per_hour(e, hB))),
                    "fa_windows_per_run": round(res_fa["fa_windows_per_run"], 4),
                })

                # 3b. Run-level detection & missed-event rate (all 60 event runs, plus gradual and rapid separately)
                det_rec = V2.run_detection_records(df_ev, regs_ev[mname][[r[0] for r in regs_ev[mname]].index(rname)][1], k, m)
                rng_d = np.random.default_rng(np.random.SeedSequence([int(seed), 1101]))
                for sub_name, sub_df in (
                    ("all_events (60 runs)", det_rec),
                    ("gradual (30 runs)", det_rec[det_rec["run_class"] == "gradual"]),
                    ("rapid (30 runs)", det_rec[det_rec["run_class"] == "rapid"]),
                ):
                    arr_d = sub_df["detected"].to_numpy(dtype=float)
                    pt_d, lo_d, hi_d = _boot_mean(arr_d, n_boot, rng_d)
                    pt_m, lo_m, hi_m = _boot_mean(1.0 - arr_d, n_boot, rng_d)
                    det_rows.append({
                        "shape": sh,
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

        # 3c. Gradual and rapid window recall + run detection by SNR bin, and gradual vs sensor_fault confusion
        for mname in MODELS:
            for rname, w_preds_ev, k, m in regs_ev[mname]:
                det_rec = V2.run_detection_records(df_ev, w_preds_ev, k, m)
                for cls in ("gradual", "rapid"):
                    pr = V2.per_run_class_table(df_ev, w_preds_ev, det_rec, cls)
                    pr["snr"] = pr["simulation_id"].map(snr_map)
                    pr["snr_bin"] = pr["snr"].map(V2.snr_bin_of)
                    # All-SNR summary row
                    s_all = V2.summarize_class(pr, n_boot, seed)
                    snr_rows.append({
                        "shape": sh, "model": mname, "regime": rname, "class": cls, "snr_bin": "ALL",
                        **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_all.items()},
                    })
                    for b_name, _, _ in V2.SNR_BINS:
                        s_b = V2.summarize_class(pr[pr["snr_bin"] == b_name], n_boot, seed)
                        snr_rows.append({
                            "shape": sh, "model": mname, "regime": rname, "class": cls, "snr_bin": b_name,
                            **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_b.items()},
                        })

                # Confusion counts on trainable event windows (both window-level w_preds_ev and persisted alarm_preds)
                alarm_preds_ev, _ = A.apply_persistence_df(df_ev, w_preds_ev, k=k, m=m)
                tmask = L.trainable(df_ev).to_numpy()
                sub_t = df_ev.loc[tmask].copy()
                sub_t["_wpred"] = np.asarray(w_preds_ev)[tmask]
                sub_t["_apred"] = np.asarray(alarm_preds_ev)[tmask]

                # Per-run confusion counts for bootstrap CI
                g_runs = sub_t[sub_t["run_class"] == "gradual"]
                per_run_w_sf = []
                per_run_a_sf = []
                per_run_w_rap = []
                per_run_w_norm = []
                per_run_w_grad = []
                for _, rg in g_runs.groupby("simulation_id", sort=False):
                    gm = rg["label"].to_numpy() == "gradual"
                    wp = rg["_wpred"].to_numpy()[gm]
                    ap = rg["_apred"].to_numpy()[gm]
                    per_run_w_sf.append(int((wp == "sensor_fault").sum()))
                    per_run_a_sf.append(int((ap == "sensor_fault").sum()))
                    per_run_w_rap.append(int((wp == "rapid").sum()))
                    per_run_w_norm.append(int((wp == "normal").sum()))
                    per_run_w_grad.append(int((wp == "gradual").sum()))

                rng_c = np.random.default_rng(np.random.SeedSequence([int(seed), 1205]))
                w_sf_pt, w_sf_lo, w_sf_hi = _boot_sum(np.array(per_run_w_sf), n_boot, rng_c)
                a_sf_pt, a_sf_lo, a_sf_hi = _boot_sum(np.array(per_run_a_sf), n_boot, rng_c)
                n_grad_win = int((g_runs["label"] == "gradual").sum())
                conf_rows.append({
                    "shape": sh,
                    "model": mname,
                    "regime": rname,
                    "n_gradual_windows": n_grad_win,
                    "win_gradual_to_sensor_fault": w_sf_pt,
                    "win_gradual_to_sensor_fault_ci_lo": round(w_sf_lo, 1),
                    "win_gradual_to_sensor_fault_ci_hi": round(w_sf_hi, 1),
                    "win_gradual_to_sensor_fault_rate": round(w_sf_pt / max(n_grad_win, 1), 4),
                    "alarm_gradual_to_sensor_fault": a_sf_pt,
                    "alarm_gradual_to_sensor_fault_ci_lo": round(a_sf_lo, 1),
                    "alarm_gradual_to_sensor_fault_ci_hi": round(a_sf_hi, 1),
                    "win_gradual_to_gradual": int(np.sum(per_run_w_grad)),
                    "win_gradual_to_rapid": int(np.sum(per_run_w_rap)),
                    "win_gradual_to_normal": int(np.sum(per_run_w_norm)),
                })

        # 3d. Distribution of dir_convergence_residual_m and dir_convergence_weighted_cos
        df_ev_snr = df_ev.copy()
        df_ev_snr["snr"] = df_ev_snr["simulation_id"].map(snr_map)
        active_mask = L.trainable(df_ev_snr).to_numpy() & (df_ev_snr["label"].isin(["gradual", "rapid"]).to_numpy())
        for sub_label, mask in (
            ("active_windows_all_snr", active_mask),
            ("active_windows_snr_gt_2", active_mask & (df_ev_snr["snr"].to_numpy() > 2.0)),
            ("active_windows_snr_gt_5", active_mask & (df_ev_snr["snr"].to_numpy() > 5.0)),
        ):
            sub_c = df_ev_snr.loc[mask]
            row_c: dict[str, float | int | str] = {
                "shape": sh,
                "subset": sub_label,
                "n_windows": len(sub_c),
                "n_runs": int(sub_c["simulation_id"].nunique()),
            }
            for col in (
                "dir_convergence_residual_m",
                "dir_convergence_weighted_cos",
                "dir_convergence_align_cos",
                "dir_win_convergence_residual_m",
                "dir_win_convergence_weighted_cos",
                "dir_pairwise_cos_mean",
                "dir_circular_spread",
            ):
                vals = sub_c[col].to_numpy(dtype=float)
                row_c[f"{col}_mean"] = round(float(np.mean(vals)), 4)
                row_c[f"{col}_std"] = round(float(np.std(vals, ddof=1)), 4)
                for q in (10, 25, 50, 75, 90):
                    row_c[f"{col}_p{q}"] = round(float(np.percentile(vals, q)), 4)
            conv_rows.append(row_c)

    det_df = pd.DataFrame(det_rows)
    snr_df = pd.DataFrame(snr_rows)
    conf_df = pd.DataFrame(conf_rows)
    fa_df = pd.DataFrame(fa_rows)
    conv_df = pd.DataFrame(conv_rows)

    det_df.to_csv(out_dir / "shape_detection_rates.csv", index=False)
    snr_df.to_csv(out_dir / "shape_recall_by_snr.csv", index=False)
    conf_df.to_csv(out_dir / "shape_gradual_sensor_fault_confusion.csv", index=False)
    fa_df.to_csv(out_dir / "shape_normal_false_alarms.csv", index=False)
    conv_df.to_csv(out_dir / "shape_convergence_distributions.csv", index=False)

    # Write concise markdown tables
    _write_markdown_tables(out_dir, det_df, snr_df, conf_df, fa_df, conv_df, full_man)

    # Render plots with red banner
    print("[shape_mismatch] rendering plots with red banner ...", flush=True)
    plot_shape_examples(scen_cfg, shape_cfg, out_dir / "shape_field_profiles.png")
    plot_detection_by_shape(det_df, out_dir / "shape_detection_and_missed_rate.png")
    plot_recall_by_snr(snr_df, out_dir / "shape_recall_by_snr.png")
    plot_convergence_distributions(conv_df, out_dir / "shape_convergence_distributions.png")

    # Verify no frozen file changed
    hashes_after = snapshot_frozen_hashes()
    assert hashes_before == hashes_after, "Frozen file modified during run_shape_mismatch!"

    cfg_files = {
        "scenario": ROOT / "configs/scenario_default.yaml",
        "sensor": ROOT / "configs/sensor_default.yaml",
        "confounder": ROOT / "configs/confounder_default.yaml",
        "noise_profile": ROOT / "configs/noise_profile.json",
        "dataset": ROOT / "configs/dataset_default.yaml",
        "shape_mismatch": shape_cfg_path,
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
    print(f"[shape_mismatch] DONE in {time.time() - t0:.1f} s; all outputs in {out_dir}", flush=True)
    return {
        "manifest": full_man,
        "detection": det_df,
        "snr": snr_df,
        "confusion": conf_df,
        "false_alarms": fa_df,
        "convergence": conv_df,
    }


def _write_markdown_tables(
    out_dir: Path,
    det_df: pd.DataFrame,
    snr_df: pd.DataFrame,
    conf_df: pd.DataFrame,
    fa_df: pd.DataFrame,
    conv_df: pd.DataFrame,
    full_man: pd.DataFrame,
) -> None:
    banner = "> **SYNTHETIC / SIMULATION** — Model-mismatch stress test on synthetic mathematical surfaces; NOT field data and NOT validated on real data."

    # 1. Detection & missed-event rate markdown
    md1 = [
        "# Run-Level Exact Detection & Missed-Event Rate by Shape (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| shape | subset | model | regime | exact detect rate (95% CI) | missed-event rate (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in det_df[det_df["regime"] != "raw (k=1)"].iterrows():
        md1.append(
            f"| {r['shape']} | {r['subset']} | {r['model']} | {r['regime']} | "
            f"**{ci_str(r['exact_detect_rate'], r['exact_detect_ci_lo'], r['exact_detect_ci_hi'])}** | "
            f"{ci_str(r['missed_event_rate'], r['missed_event_ci_lo'], r['missed_event_ci_hi'])} |"
        )
    (out_dir / "shape_detection_rates.md").write_text("\n".join(md1) + "\n", encoding="utf-8")

    # 2. SNR bin recall markdown (v3 operating points)
    md2 = [
        "# Gradual & Rapid Window Recall and Run Detection by SNR Bin (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| shape | model | regime | class | SNR bin | runs | windows | window recall (95% CI) | run detect rate (95% CI) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in snr_df[snr_df["regime"] != "raw (k=1)"].iterrows():
        md2.append(
            f"| {r['shape']} | {r['model']} | {r['regime']} | {r['class']} | {r['snr_bin']} | {r['n_runs']} | {r['n_windows']} | "
            f"{ci_str(r['window_recall'], r['window_recall_ci_lo'], r['window_recall_ci_hi'])} | "
            f"{ci_str(r['run_detect_rate'], r['run_detect_ci_lo'], r['run_detect_ci_hi'])} |"
        )
    (out_dir / "shape_recall_by_snr.md").write_text("\n".join(md2) + "\n", encoding="utf-8")

    # 3. Confusion counts markdown
    md3 = [
        "# Gradual vs Sensor_Fault Confusion Counts by Shape (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| shape | model | regime | gradual windows | win: gradual->sensor_fault [95% CI] (rate) | alarm: gradual->sensor_fault [95% CI] | win: gradual->gradual | win: gradual->rapid | win: gradual->normal |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in conf_df.iterrows():
        md3.append(
            f"| {r['shape']} | {r['model']} | {r['regime']} | {r['n_gradual_windows']} | "
            f"**{r['win_gradual_to_sensor_fault']}** [{r['win_gradual_to_sensor_fault_ci_lo']:.1f}, {r['win_gradual_to_sensor_fault_ci_hi']:.1f}] ({r['win_gradual_to_sensor_fault_rate']:.3f}) | "
            f"**{r['alarm_gradual_to_sensor_fault']}** [{r['alarm_gradual_to_sensor_fault_ci_lo']:.1f}, {r['alarm_gradual_to_sensor_fault_ci_hi']:.1f}] | "
            f"{r['win_gradual_to_gradual']} | {r['win_gradual_to_rapid']} | {r['win_gradual_to_normal']} |"
        )
    (out_dir / "shape_gradual_sensor_fault_confusion.md").write_text("\n".join(md3) + "\n", encoding="utf-8")

    # 4. Convergence feature distribution markdown
    md4 = [
        "# Radial Convergence Feature Distributions by Shape (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| shape | subset | windows | residual_m p10 / **p50** / p90 (mean) | weighted_cos p10 / **p50** / p90 (mean) | pairwise_cos p50 |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in conv_df.iterrows():
        md4.append(
            f"| {r['shape']} | {r['subset']} | {r['n_windows']} | "
            f"{r['dir_convergence_residual_m_p10']:.2f} / **{r['dir_convergence_residual_m_p50']:.2f}** / {r['dir_convergence_residual_m_p90']:.2f} ({r['dir_convergence_residual_m_mean']:.2f}) | "
            f"{r['dir_convergence_weighted_cos_p10']:.3f} / **{r['dir_convergence_weighted_cos_p50']:.3f}** / {r['dir_convergence_weighted_cos_p90']:.3f} ({r['dir_convergence_weighted_cos_mean']:.3f}) | "
            f"{r['dir_pairwise_cos_mean_p50']:.3f} |"
        )
    (out_dir / "shape_convergence_distributions.md").write_text("\n".join(md4) + "\n", encoding="utf-8")

    # 5. Normal runs false-alarm markdown
    md5 = [
        "# Shared Normal Runs (40 Runs) False Alarms Across Shape Evaluations (SYNTHETIC / SIMULATION)",
        "",
        banner,
        "",
        "| shape | model | regime | FA eps / run (95% CI) | A: FA/h (slow, 24h/run) | A: every | B: FA/h (sensor, 0.0667h/run) | B: every |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in fa_df.iterrows():
        md5.append(
            f"| {r['shape']} | {r['model']} | {r['regime']} | "
            f"{ci_str(r['fa_episodes_per_run'], r['fa_episodes_per_run_ci_lo'], r['fa_episodes_per_run_ci_hi'])} | "
            f"{r['A_slow_fa_per_hour']:.4f} | {r['A_slow_one_fa_every']} | "
            f"{r['B_sensor_fa_per_hour']:.2f} | {r['B_sensor_one_fa_every']} |"
        )
    (out_dir / "shape_normal_false_alarms.md").write_text("\n".join(md5) + "\n", encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shape_mismatch.yaml")
    ap.add_argument("--out", default="outputs/eval_shape_mismatch")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args()
    run_shape_mismatch(Path(args.config), Path(args.out), n_boot=args.n_boot, seed=args.seed)
