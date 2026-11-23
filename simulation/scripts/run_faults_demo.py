"""Faults demo: one gradual run, three telemetry variants side by side.

(a) clean   (b) single-node IMU slow drift   (c) hub self-motion (all 4 ranges shift together)
Outputs go to outputs/faults_demo/<simulation_id>/ ; the state timeline is read-only input.

Usage: python scripts/run_faults_demo.py [--seed N] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from canary_sim import engine, faults as fl, sensors as S

BANNER = "SYNTHETIC / SIMULATION"
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260001)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "faults_demo"))
    a = ap.parse_args()

    cfg = engine.load_config(ROOT / "configs" / "scenario_default.yaml")
    df, meta = engine.run_scenario(cfg, seed=a.seed, mode="gradual")
    run_dir = engine.write_run(df, meta, a.out)
    state_path = run_dir / "state_timeline.csv"
    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    P = meta["parameters"]
    T = P["duration_s"]

    scfg = S.load_sensor_config(ROOT / "configs" / "sensor_default.yaml")
    prof = S.load_noise_profile(ROOT / "configs" / "noise_profile.json")
    tseed = a.seed
    plans = {
        "clean": None,
        "imu_drift": [fl.make_spec("slow_drift", 0.15 * T, 0.7 * T, "NODE_02", tseed, sensor="imu", rate_per_s=0.01)],
        "hub_self_motion": [fl.make_spec("hub_self_motion", 0.55 * T, 0.30 * T, "HUB", tseed, shift_m=0.30)],
    }
    titles = {"clean": "(a) clean", "imu_drift": "(b) NODE_02 IMU slow drift", "hub_self_motion": "(c) hub self-motion"}
    res = {}
    for tag, plan in plans.items():
        tdf, tmeta = S.generate_telemetry(state_path, tseed, scfg, prof, plan)
        S.write_telemetry(tdf, tmeta, run_dir, tag)
        res[tag] = (tdf, tmeta)
    assert hashlib.sha256(state_path.read_bytes()).hexdigest() == digest, "state timeline changed!"

    tl = engine.pd.read_csv(state_path)
    fig, axes = plt.subplots(3, 3, figsize=(17, 10), sharex=True)
    for j, (tag, (tdf, tmeta)) in enumerate(res.items()):
        for k, n in enumerate(S.NODES):
            d = tdf[tdf.node_id == n]
            tr = tl[tl.node_id == n]
            axes[0, j].plot(d.time_s, d.tilt_x_deg - d.tilt_x_deg.iloc[:int(tmeta["lead_in_end_s"])].mean(),
                            color=COLORS[k], lw=0.8, label=n)
            axes[0, j].plot(tr.time_s, tr.true_tilt_x_deg - tr.true_tilt_x_deg.iloc[0], color=COLORS[k], ls="--", lw=0.8, alpha=0.6)
            axes[1, j].plot(d.time_s, d.piezo_rms, color=COLORS[k], lw=0.7)
            r = d[d.lidar_new == 1]
            axes[2, j].plot(r.time_s, r.lidar_range_change_m, "o-", ms=3, lw=0.8, color=COLORS[k])
        for f in tmeta["faults"]:
            for ax in axes[:, j]:
                ax.axvspan(f["start_s"], f["end_s"], color="red", alpha=0.07)
        axes[0, j].set_title(titles[tag], fontsize=11)
        axes[0, j].text(0.5, 1.12, BANNER, transform=axes[0, j].transAxes, color="red", ha="center",
                        fontsize=12, fontweight="bold")
        axes[2, j].set_xlabel("simulated time (s)")
    axes[0, 0].set_ylabel("tilt_x change (deg)\nsolid = telemetry, dashed = truth")
    axes[1, 0].set_ylabel("piezo_rms")
    axes[2, 0].set_ylabel("LiDAR range change vs\nown lead-in baseline (m)")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{BANNER}  |  time compression x{P['compression_factor']:g} "
                 f"({T:g} s sim = {P['physical_duration_s'] / 3600:.2f} h physical)  |  {meta['simulation_id']}  "
                 f"(red shading = injected fault)",
                 color="red", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = run_dir / "plot_faults_demo.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # one plot per variant too, each with the banner + compression factor
    for tag, (tdf, tmeta) in res.items():
        fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        for k, n in enumerate(S.NODES):
            d = tdf[tdf.node_id == n]
            ax[0].plot(d.time_s, d.tilt_x_deg - d.tilt_x_deg.iloc[:int(tmeta["lead_in_end_s"])].mean(), color=COLORS[k], lw=0.8, label=n)
            r = d[d.lidar_new == 1]
            ax[1].plot(r.time_s, r.lidar_range_change_m, "o-", ms=3, lw=0.8, color=COLORS[k])
        ax[0].set_ylabel("tilt_x change (deg)"); ax[1].set_ylabel("LiDAR range change (m)")
        ax[1].set_xlabel("simulated time (s)"); ax[0].legend(fontsize=7, ncol=2)
        fig.suptitle(f"{BANNER}  |  time compression x{P['compression_factor']:g}\n{titles[tag]}",
                     color="red", fontsize=10, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(run_dir / f"plot_faults_{tag}.png", dpi=110)
        plt.close(fig)
    print("state timeline unchanged (sha256 verified)")
    print("wrote", out)


if __name__ == "__main__":
    main()
