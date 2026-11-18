"""One gradual run end to end: engine -> placeholder telemetry -> features -> LogReg -> plots.

Usage: python scripts/run_one.py [--seed N] [--config PATH] [--out DIR]
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from canary_sim import engine, node_geometry as ng
from canary_sim.features import feature_columns, window_features
from canary_sim.telemetry_placeholder import make_telemetry

BANNER = "SYNTHETIC / SIMULATION"


def stamp(fig, P, extra=""):
    fig.suptitle(f"{BANNER}  |  time compression x{P['compression_factor']:g} "
                 f"({P['duration_s']:g} s sim = {P['physical_duration_s']/3600:.2f} h physical)  {extra}",
                 color="red", fontsize=10, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "scenario_default.yaml"))
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    a = ap.parse_args()

    cfg = engine.load_config(a.config)
    df, meta = engine.run_scenario(cfg, seed=a.seed, mode="gradual")
    P = meta["parameters"]
    run_dir = engine.write_run(df, meta, a.out)
    print(f"run {meta['simulation_id']} -> {run_dir}")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in P.items()})

    # --- placeholder telemetry + features ---
    tcfg = cfg["telemetry_placeholder"]
    tele = make_telemetry(df, tcfg["tilt_noise_sd_deg"], P["seed"] + tcfg["seed_offset"])
    feats = window_features(tele, P["t_on"], P["tau"])
    cols = feature_columns(feats)
    feats.to_csv(run_dir / "window_features.csv", index=False)

    # --- quick logistic regression (plumbing proof only; train == eval on one run) ---
    tr = feats[feats.label >= 0]
    sc = StandardScaler().fit(tr[cols])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(tr[cols]), tr.label)
    feats["p_event"] = clf.predict_proba(sc.transform(feats[cols]))[:, 1]
    acc = clf.score(sc.transform(tr[cols]), tr.label)
    print(f"windows: total={len(feats)} train={len(tr)} (normal={int((tr.label==0).sum())}, "
          f"event={int((tr.label==1).sum())}); in-sample accuracy={acc:.3f}  [PLUMBING CHECK ONLY]")

    # --- plots ---
    out = run_dir
    # 1. settlement field (final hold state)
    L = ng.SIDE_M
    pad = max(L, 1.5 * ng.NETWORK_RADIUS_M)
    gx = np.linspace(L / 2 - pad - 5, L / 2 + pad + 5, 300)
    X, Y = np.meshgrid(gx, gx)
    W = engine.settlement(X, Y, P["s_max"], P)
    fig, ax = plt.subplots(figsize=(7, 6.3))
    im = ax.pcolormesh(X, Y, W * 1000, shading="auto", cmap="viridis")
    fig.colorbar(im, label="w (mm), final state, negative = settlement")
    for n, (x, y) in ng.NODE_XY.items():
        ax.plot(x, y, "ro"); ax.annotate(n[-2:], (x, y), color="w", xytext=(4, 4), textcoords="offset points")
    ax.plot(*ng.HUB_XY, "w^", label="HUB (not a reference)")
    ax.plot(P["x0"], P["y0"], "kx", ms=10, label="centre (x0,y0)")
    ax.plot([0, L, L, 0, 0], [0, 0, L, L, 0], "r-", lw=0.8)
    ax.set_aspect("equal"); ax.set_xlabel("x east (m)"); ax.set_ylabel("y north (m)"); ax.legend(loc="upper right")
    ax.set_title(f"Settlement field  s_max={P['s_max']*1000:.1f} mm, sx={P['sx']:.1f}, sy={P['sy']:.1f} m, "
                 f"theta={P['theta_deg']:.0f} deg", fontsize=9)
    stamp(fig, P); fig.tight_layout(); fig.savefig(out / "plot_settlement_field.png", dpi=130); plt.close(fig)

    # 2. per-node tilt vs time (truth + placeholder telemetry)
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    for n in ng.NODE_IDS:
        d = df[df.node_id == n]; tl = tele[tele.node_id == n]
        for ax, c in zip(axs, ("x", "y")):
            l, = ax.plot(d.time_s, d[f"true_tilt_{c}_deg"], lw=1.5, label=f"{n} truth")
            ax.plot(tl.time_s, tl[f"tilt_{c}_deg"], lw=0.4, alpha=0.5, color=l.get_color())
    for ax, c in zip(axs, ("x", "y")):
        ax.axvspan(P["t_on"], P["t_on"] + P["tau"], color="orange", alpha=0.12, label="event ramp")
        ax.set_ylabel(f"tilt_{c} (deg)")
    axs[0].legend(ncol=3, fontsize=7); axs[1].set_xlabel("simulated time (s)")
    axs[0].set_title("Thick = truth; thin = PLACEHOLDER telemetry (truth + Gaussian noise)", fontsize=9)
    stamp(fig, P); fig.tight_layout(); fig.savefig(out / "plot_tilt_vs_time.png", dpi=130); plt.close(fig)

    # 3. model output vs time
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(feats.t_mid, feats.p_event, "o-", ms=3, label="P(event) LogReg")
    ax.axvspan(P["t_on"], P["t_on"] + P["tau"], color="orange", alpha=0.15, label="true event ramp")
    ax.axvspan(P["t_on"] + P["tau"], P["duration_s"], color="grey", alpha=0.12, label="aftermath (not trained)")
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel("window centre (s)"); ax.set_ylabel("P(event)"); ax.legend(loc="center left")
    ax.set_title(f"Plumbing check ONLY: trained & evaluated on one run (in-sample acc {acc:.2f})", fontsize=9)
    stamp(fig, P); fig.tight_layout(); fig.savefig(out / "plot_model_output.png", dpi=130); plt.close(fig)
    print("plots saved in", out)


if __name__ == "__main__":
    main()
