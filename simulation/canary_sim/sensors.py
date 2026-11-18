"""Telemetry layer (SYNTHETIC / SIMULATION): IMU, piezo and LiDAR-hub sensor models.

Input : a state-timeline (frozen schema v1) as a CSV path or DataFrame. It is only READ.
Output: a separate telemetry table (and metadata) per run. Sensor values are never written into
        the state-timeline.

Pipeline: build_buffers (clean sensors) -> confounders (optional) -> faults (optional) -> finalize.
Every random stream is keyed by (seed, stream, node) and drawn at a fixed shape regardless of
faults, so a fault on one node cannot change the other nodes' values.

Telemetry columns (one row per window per corner node):
    simulation_id, time_s, node_id, packet_received, arrival_time_s,
    tilt_x_deg, tilt_y_deg, piezo_peak, piezo_rms, piezo_peak_to_peak,
    hub_link_up, lidar_new, lidar_range_change_m
Missing data are NaN (never 0). lidar_range_change_m is relative to that node's own lead-in
baseline (median of received readings before the event starts); absolute range is not reported.
LiDAR readings arrive once per directional pass (sparse); other windows are NaN with lidar_new = 0.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from . import node_geometry as ng

NODES = ng.NODE_IDS
HUB = ng.HUB_ID
CHANNELS = {"imu": ("tilt_x", "tilt_y"), "piezo": ("pz_peak", "pz_rms", "pz_p2p")}
TELEMETRY_COLUMNS = (
    "simulation_id", "time_s", "node_id", "packet_received", "arrival_time_s",
    "tilt_x_deg", "tilt_y_deg", "piezo_peak", "piezo_rms", "piezo_peak_to_peak",
    "hub_link_up", "lidar_new", "lidar_range_change_m",
)


def rng_for(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *[int(k) for k in key]]))


# --------------------------------------------------------------------------- loading
def load_sensor_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    lc = cfg["lidar"]
    if lc["dwell_s"] > lc["sweep_period_s"] / 4 + 1e-12:
        raise ValueError("lidar.dwell_s must be <= sweep_period_s / 4")
    return cfg


def load_noise_profile(path: str | Path) -> dict:
    """Derive model parameters from the real MPU6050/ADS1115 recording."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    step = d["continuous_step_noise"]

    def pooled(ch):
        return float(np.mean([v[ch] for v in step.values()]))

    ov = d["overall"]
    rms, pk, p2p = ov["piezo_rms"], ov["piezo_peak"], ov["piezo_peak_to_peak"]
    def white_sd(ch):
        """white sd = step / sqrt(2 (1 - rho1)); falls back to step / sqrt(2) only if rho1 is missing."""
        rho = ov.get(ch, {}).get("autocorr_lag1")
        den = math.sqrt(2.0 * (1.0 - rho)) if rho is not None and rho < 1.0 else math.sqrt(2.0)
        return pooled(ch) / den, rho

    wx, rx = white_sd("tilt_x")
    wy, ry = white_sd("tilt_y")
    return {
        "source": d.get("source"),
        "imu_white_sd_source": {
            "formula": "mean over NODE_01..NODE_03 of continuous_step_noise / sqrt(2*(1-autocorr_lag1)); /sqrt(2) if autocorr_lag1 missing",
            "autocorr_lag1": {"tilt_x": rx, "tilt_y": ry},
            "nodes_in_profile": sorted(step.keys()),
            "applied_to_nodes": "all four corner nodes (NODE_04 uses the same pooled value)",
        },
        "sampling_interval_s": float(d["sampling_interval_sec"]),
        # ASSUMPTION: continuous_step_noise is the sd of the first difference of an AR(1)-like series
        "imu_white_sd_deg": {"tilt_x": wx, "tilt_y": wy},
        "piezo_rms_median": float(rms["median"]),
        "piezo_rms_sigma_ln": float(math.sqrt(math.log(1 + (rms["std"] / rms["mean"]) ** 2))),
        "piezo_rms_phi": float(rms["autocorr_lag1"]),
        "piezo_crest_median": float(pk["median"] / rms["median"]),
        "piezo_p2p_over_peak_median": float(p2p["median"] / pk["median"]),
    }


@dataclass
class State:
    sim_id: str
    rate_hz: float
    duration_s: float
    hub_pos: np.ndarray        # (S, 3)
    node_pos: np.ndarray       # (S, 4, 3)
    tilt_x: np.ndarray         # (S, 4) true
    tilt_y: np.ndarray
    lead_in_end_s: float


def load_state(timeline) -> State:
    """Read a state timeline (path or DataFrame). The input is never modified."""
    df = pd.read_csv(timeline) if isinstance(timeline, (str, Path)) else timeline
    sim_id = str(df["simulation_id"].iloc[0])
    per = {n: df[df.node_id == n].sort_values("time_s") for n in (*NODES, HUB)}
    t = per[NODES[0]]["time_s"].to_numpy()
    rate = 1.0 / float(np.median(np.diff(t)))

    def pos(n):
        s = per[n]
        return np.column_stack([s.true_x, s.true_y, s.true_z])

    active = per[NODES[0]]["active_class"].to_numpy()
    nz = np.nonzero(active != "normal")[0]
    lead = float(t[nz[0]]) if len(nz) else float(t[-1] + 1 / rate)
    return State(
        sim_id=sim_id, rate_hz=rate, duration_s=len(t) / rate,
        hub_pos=pos(HUB), node_pos=np.stack([pos(n) for n in NODES], axis=1),
        tilt_x=np.column_stack([per[n].true_tilt_x_deg for n in NODES]),
        tilt_y=np.column_stack([per[n].true_tilt_y_deg for n in NODES]),
        lead_in_end_s=lead,
    )


def clockwise_order() -> list[int]:
    """Node indices in clockwise order (seen from above) as seen from the hub, starting at NODE_01."""
    ang = [math.atan2(ng.NODE_XY[n][1] - ng.HUB_XY[1], ng.NODE_XY[n][0] - ng.HUB_XY[0]) for n in NODES]
    return sorted(range(4), key=lambda k: (ang[0] - ang[k]) % (2 * math.pi))


# --------------------------------------------------------------------------- buffers
@dataclass
class Buffers:
    state: State
    seed: int
    cfg: dict
    profile: dict
    noise_multiplier: float
    t: np.ndarray              # (W,) window start times
    tilt_x: np.ndarray         # (W, 4)
    tilt_y: np.ndarray
    pz_peak: np.ndarray
    pz_rms: np.ndarray
    pz_p2p: np.ndarray
    # LiDAR dwell events
    ev_t: np.ndarray           # reading time
    ev_si: np.ndarray          # state index at dwell centre
    ev_aim: np.ndarray         # node the schedule says the hub is at (label of the reading)
    ev_meas: np.ndarray        # node actually ranged
    ev_true: np.ndarray        # true 3-D distance to the point actually ranged
    ev_noise: np.ndarray
    ev_add: np.ndarray
    ev_override: np.ndarray    # bool
    ev_override_val: np.ndarray
    ev_valid: np.ndarray
    hub_motion: list = field(default_factory=list)   # hook: callables f(t_array)->range shift (m), common-mode
    # links
    node_drop: np.ndarray = None   # (W, 4) bool
    hub_drop: np.ndarray = None    # (W,) bool
    latency: np.ndarray = None     # (W, 4) s
    faults: list = field(default_factory=list)
    confounders: list = field(default_factory=list)
    events: list = field(default_factory=list)       # non-fault, non-benign events (e.g. abnormal vibration)

    @property
    def W(self) -> int:
        return len(self.t)

    @property
    def window_s(self) -> float:
        return float(self.cfg["window_s"])

    node_ids = NODES

    def dist_to_points(self, ev_idx, points) -> np.ndarray:
        """3-D distance from the hub (at the event's state index) to given points (n,3)."""
        return np.linalg.norm(points - self.state.hub_pos[self.ev_si[ev_idx]], axis=1)

    def node_point(self, ev_idx, k) -> np.ndarray:
        return self.state.node_pos[self.ev_si[ev_idx], k]


def add_hub_motion(buf: Buffers, fn: Callable[[np.ndarray], np.ndarray]) -> None:
    """Hub-motion hook: fn(t) -> metres, added to ALL four ranges (common mode)."""
    buf.hub_motion.append(fn)


def add_piezo_energy(buf: Buffers, k: int, w, rms_add, peak_amp) -> None:
    """Add vibration energy to piezo features of node k at windows w (rms in quadrature)."""
    ratio = buf.profile["piezo_p2p_over_peak_median"]
    buf.pz_rms[w, k] = np.hypot(buf.pz_rms[w, k], rms_add)
    buf.pz_peak[w, k] = np.maximum(buf.pz_peak[w, k], peak_amp)
    buf.pz_p2p[w, k] = np.maximum(buf.pz_p2p[w, k], ratio * np.asarray(peak_amp))


def build_buffers(state: State, seed: int, cfg: dict, profile: dict) -> Buffers:
    ws = float(cfg["window_s"])
    W = int(state.duration_s / ws + 1e-9)
    t = np.arange(W) * ws
    S = len(state.hub_pos)
    si = np.clip(np.round(t * state.rate_hz).astype(int), 0, S - 1)

    # ---- IMU
    ic = cfg["imu"]
    lo, hi = ic["noise_multiplier_range"]
    mult = float(np.exp(rng_for(seed, 1).uniform(math.log(lo), math.log(hi))))
    tilt = {"tilt_x": np.zeros((W, 4)), "tilt_y": np.zeros((W, 4))}
    for k in range(4):
        r = rng_for(seed, 2, k)
        white = r.normal(0.0, 1.0, (W, 2))
        bias = r.normal(0.0, ic["node_bias_sd_deg"], 2)
        steps = r.normal(0.0, 1.0, (W, 2))
        drift = np.cumsum(steps * ic["drift_sd_deg_per_sqrt_s"] * math.sqrt(ws) * mult, axis=0)
        sd = profile["imu_white_sd_deg"]
        tilt["tilt_x"][:, k] = state.tilt_x[si, k] + bias[0] + drift[:, 0] + white[:, 0] * sd["tilt_x"] * mult
        tilt["tilt_y"][:, k] = state.tilt_y[si, k] + bias[1] + drift[:, 1] + white[:, 1] * sd["tilt_y"] * mult

    # ---- Piezo: AR(1) log-level baseline + Poisson impulse events
    pc = cfg["piezo"]
    peak, rms, p2p = (np.zeros((W, 4)) for _ in range(3))
    phi = profile["piezo_rms_phi"]
    imp_amp_all = []
    for k in range(4):
        r = rng_for(seed, 3, k)
        e = r.normal(0.0, 1.0, W)
        gain = math.exp(r.normal(0.0, pc["node_gain_sd_ln"]))
        zc, zp = r.normal(0.0, 1.0, W), r.normal(0.0, 1.0, W)
        imp_mask = r.random(W) < pc["impulse_rate_per_s"] * ws
        imp_amp = pc["impulse_amp_median"] * np.exp(r.normal(0.0, pc["impulse_amp_sigma_ln"], W))
        x = np.zeros(W)
        x[0] = e[0]
        for i in range(1, W):
            x[i] = phi * x[i - 1] + math.sqrt(1 - phi ** 2) * e[i]
        rms[:, k] = profile["piezo_rms_median"] * gain * np.exp(profile["piezo_rms_sigma_ln"] * x)
        peak[:, k] = np.maximum(rms[:, k] * profile["piezo_crest_median"] * np.exp(pc["crest_scatter_ln"] * zc), rms[:, k])
        p2p[:, k] = np.clip(peak[:, k] * profile["piezo_p2p_over_peak_median"] * np.exp(pc["p2p_scatter_ln"] * zp),
                            peak[:, k], 2 * peak[:, k])
        imp_amp_all.append((imp_mask, imp_amp))

    # ---- LiDAR dwell schedule
    lc = cfg["lidar"]
    P, dwell = lc["sweep_period_s"], lc["dwell_s"]
    cw = clockwise_order()
    n_pass = int(math.ceil(state.duration_s / P)) + 1
    noise = np.stack([rng_for(seed, 4, k).normal(0.0, lc["range_noise_sd_m"], n_pass) for k in range(4)], axis=1)
    ev = []
    for p in range(n_pass):
        order = cw if p % 2 == 0 else cw[::-1]       # clockwise pass, then anticlockwise pass
        for i, k in enumerate(order):
            centre = p * P + (i + 0.5) / 4 * P
            ts = centre + dwell / 2                  # reading is complete at end of dwell
            if ts < state.duration_s:
                ev.append((ts, min(int(round(centre * state.rate_hz)), S - 1), k, noise[p, k]))
    ev.sort()
    E = len(ev)
    ev_t = np.array([e[0] for e in ev]); ev_si = np.array([e[1] for e in ev], dtype=int)
    ev_aim = np.array([e[2] for e in ev], dtype=int); ev_noise = np.array([e[3] for e in ev])
    ev_true = np.linalg.norm(state.node_pos[ev_si, ev_aim] - state.hub_pos[ev_si], axis=1)

    # ---- Links (LoRa as packet loss + latency process only)
    cc = cfg["comms"]
    drop = np.zeros((W, 4), bool); lat = np.zeros((W, 4))
    for k in range(4):
        r = rng_for(seed, 5, k)
        drop[:, k] = r.random(W) < cc["node_loss_prob"]
        lat[:, k] = cc["latency_median_s"] * np.exp(r.normal(0.0, cc["latency_sigma_ln"], W))
    hub_drop = rng_for(seed, 5, 9).random(W) < cc["hub_loss_prob"]

    buf = Buffers(
        state=state, seed=seed, cfg=cfg, profile=profile, noise_multiplier=mult, t=t,
        tilt_x=tilt["tilt_x"], tilt_y=tilt["tilt_y"], pz_peak=peak, pz_rms=rms, pz_p2p=p2p,
        ev_t=ev_t, ev_si=ev_si, ev_aim=ev_aim, ev_meas=ev_aim.copy(), ev_true=ev_true, ev_noise=ev_noise,
        ev_add=np.zeros(E), ev_override=np.zeros(E, bool), ev_override_val=np.full(E, np.nan),
        ev_valid=np.ones(E, bool), node_drop=drop, hub_drop=hub_drop, latency=lat,
    )
    for k, (mask, amp) in enumerate(imp_amp_all):
        w = np.nonzero(mask)[0]
        add_piezo_energy(buf, k, w, pc["impulse_rms_fraction"] * amp[w], amp[w])
    return buf


def lidar_ranges(buf: Buffers) -> np.ndarray:
    """Current range reading per dwell event (NaN = no reading / no return)."""
    lc = buf.cfg["lidar"]
    r = buf.ev_true + buf.ev_noise + buf.ev_add
    for f in buf.hub_motion:
        r = r + f(buf.ev_t)
    q = lc["quantisation_m"]
    r = np.round(r / q) * q
    r = np.where(buf.ev_override, buf.ev_override_val, r)
    r = np.where((r > lc["max_range_m"]) | (r < 0), np.nan, r)
    r[~buf.ev_valid] = np.nan
    return r


# --------------------------------------------------------------------------- finalize
def finalize(buf: Buffers) -> tuple[pd.DataFrame, dict]:
    W = buf.W
    ok = ~buf.node_drop
    cols = {c: getattr(buf, c).copy() for c in ("tilt_x", "tilt_y", "pz_peak", "pz_rms", "pz_p2p")}
    for c in cols.values():
        c[~ok] = np.nan
    arrival = np.where(ok, buf.t[:, None] + buf.latency, np.nan)

    rng_ = lidar_ranges(buf)
    L = np.full((W, 4), np.nan)
    new = np.zeros((W, 4), int)
    win = (buf.ev_t / buf.window_s + 1e-9).astype(int)
    for e in range(len(buf.ev_t)):
        w, k = win[e], buf.ev_aim[e]
        if w < W and not buf.hub_drop[w] and buf.ev_valid[e]:
            L[w, k] = rng_[e]
            new[w, k] = 1
    baseline = np.full(4, np.nan)
    lead = buf.t < buf.state.lead_in_end_s
    for k in range(4):
        v = L[lead, k]; v = v[~np.isnan(v)]
        if len(v):
            baseline[k] = float(np.median(v))
    change = L - baseline[None, :]

    df = pd.DataFrame({
        "simulation_id": buf.state.sim_id,
        "time_s": np.repeat(buf.t, 4),
        "node_id": np.tile(NODES, W),
        "packet_received": ok.astype(int).ravel(),
        "arrival_time_s": arrival.ravel(),
        "tilt_x_deg": cols["tilt_x"].ravel(), "tilt_y_deg": cols["tilt_y"].ravel(),
        "piezo_peak": cols["pz_peak"].ravel(), "piezo_rms": cols["pz_rms"].ravel(),
        "piezo_peak_to_peak": cols["pz_p2p"].ravel(),
        "hub_link_up": np.repeat((~buf.hub_drop).astype(int), 4),
        "lidar_new": new.ravel(),
        "lidar_range_change_m": change.ravel(),
    })[list(TELEMETRY_COLUMNS)]
    meta = {
        "simulation_id": buf.state.sim_id, "synthetic": True, "telemetry_seed": int(buf.seed),
        "noise_multiplier": buf.noise_multiplier, "window_s": buf.window_s,
        "lead_in_end_s": buf.state.lead_in_end_s,
        "lidar_baseline_m": {n: (None if np.isnan(b) else float(b)) for n, b in zip(NODES, baseline)},
        "faults": buf.faults, "confounders": buf.confounders, "events": buf.events,
        "sensor_config": buf.cfg,
        "profile_derived": buf.profile,
        "note": "Telemetry is separate from the frozen state timeline; ground-truth faults are listed only here.",
    }
    return df, meta


def generate_telemetry(timeline, seed: int, sensor_cfg: dict, profile: dict,
                       fault_plan: list | None = None, confounder_cfg: dict | None = None,
                       confounder_kinds: list | None = None, lead_in_end_s: float | None = None,
                       hooks: list | None = None, n_bursts: int | None = None):
    """Full pipeline. confounder_cfg=None -> no benign confounders. fault_plan=None -> no faults."""
    state = load_state(timeline)
    if lead_in_end_s is not None:        # observable baseline window override (e.g. first 30 s of the run)
        state.lead_in_end_s = float(lead_in_end_s)
    buf = build_buffers(state, seed, sensor_cfg, profile)
    if confounder_cfg is not None:
        from .confounders import apply_confounders
        apply_confounders(buf, seed, confounder_cfg, confounder_kinds, n_bursts)
    for h in hooks or []:                # extra events (e.g. abnormal vibration), applied before faults
        h(buf)
    if fault_plan:
        from .faults import apply_faults
        apply_faults(buf, fault_plan)
    return finalize(buf)


def write_telemetry(df: pd.DataFrame, meta: dict, out_dir: str | Path, tag: str = "clean") -> Path:
    """Writes telemetry_<tag>.csv and telemetry_<tag>_metadata.json. Never touches state_timeline.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"telemetry_{tag}.csv", index=False)
    with open(out / f"telemetry_{tag}_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    return out / f"telemetry_{tag}.csv"
