"""Fault injection (SYNTHETIC / SIMULATION).

Every fault is a function  fn(buf, seed, start_s, duration_s, target, **params)  that edits the
telemetry BUFFERS only. The true state timeline is never touched (it is not even reachable for
writing: buffers hold copies). Every injected fault is appended to buf.faults as
{type, node, start_s, end_s, seed, params} (one record per affected node; hub faults use node "HUB";
network-wide outages use node "ALL").

Communication faults: LoRa is modelled ONLY as a packet / loss / latency process. There is no RF
propagation, path loss, fading, SNR, spreading-factor or collision model.

All default magnitudes in DEFAULT_PARAMS are PLACEHOLDERS.
"""
from __future__ import annotations

import math

import numpy as np

from . import sensors as S

DEFAULT_PARAMS = {  # PLACEHOLDERS (units: imu deg, piezo same unit as piezo_peak, lidar m)
    "bias_step": {"imu": 0.5, "piezo": 0.02},
    "slow_drift_rate": {"imu": 0.01, "piezo": 0.0005},    # per sim second
    "spike": {"imu": 3.0, "piezo": 0.2},
    "n_spikes": 3,
    "disconnect_p_off": 0.15, "disconnect_p_on": 0.3,
    "implausible_tilt_deg": (120.0, 175.0),
    "implausible_piezo": 5.0,                              # > ADS1115 full scale 4.096
    "wrong_dwell_offset_frac": 0.5,                        # fraction of the way to the next (clockwise) corner
    "hub_shift_m": 0.30, "hub_ramp_s": 3.0,
    "n_drops": 3, "burst_drop_prob": 0.8, "latency_added_s": 5.0,
}


# --------------------------------------------------------------------------- helpers
def _widx(buf, start_s, duration_s):
    ws = buf.window_s
    w0 = int(math.ceil(start_s / ws - 1e-9))
    w1 = int(math.ceil((start_s + duration_s) / ws - 1e-9))
    return np.arange(max(w0, 0), min(w1, buf.W))


def _eidx(buf, start_s, duration_s):
    return np.nonzero((buf.ev_t >= start_s) & (buf.ev_t < start_s + duration_s))[0]


def _nodes(buf, target) -> list[int]:
    names = [target] if isinstance(target, str) else list(target)
    return [S.NODES.index(n) for n in names]


def _log(buf, type_, node, start_s, duration_s, seed, **params):
    rec = {"type": type_, "node": node, "start_s": float(start_s), "end_s": float(start_s + duration_s),
           "seed": int(seed), "params": {k: (v if not isinstance(v, np.generic) else v.item()) for k, v in params.items()}}
    buf.faults.append(rec)
    return rec


def _arrays(buf, sensor):
    return [getattr(buf, c) for c in S.CHANNELS[sensor]]


def _sign(r):
    return 1.0 if r.random() < 0.5 else -1.0


# --------------------------------------------------------------------------- sensor faults
def bias_step(buf, seed, start_s, duration_s, target, sensor="imu", magnitude=None):
    w = _widx(buf, start_s, duration_s)
    mag = DEFAULT_PARAMS["bias_step"][sensor] if magnitude is None else magnitude
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 100, k)
        for a in _arrays(buf, sensor):
            a[w, k] += (_sign(r) if sensor == "imu" else 1.0) * mag
        _log(buf, "bias_step", S.NODES[k], start_s, duration_s, seed, sensor=sensor, magnitude=mag)


def slow_drift(buf, seed, start_s, duration_s, target, sensor="imu", rate_per_s=None):
    w = _widx(buf, start_s, duration_s)
    rate = DEFAULT_PARAMS["slow_drift_rate"][sensor] if rate_per_s is None else rate_per_s
    ramp = rate * (buf.t[w] - start_s)
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 101, k)
        for a in _arrays(buf, sensor):
            a[w, k] += (_sign(r) if sensor == "imu" else 1.0) * ramp
        _log(buf, "slow_drift", S.NODES[k], start_s, duration_s, seed, sensor=sensor, rate_per_s=rate)


def stuck_at(buf, seed, start_s, duration_s, target, sensor="imu", value=None):
    w = _widx(buf, start_s, duration_s)
    if len(w) == 0:
        return
    for k in _nodes(buf, target):
        for a in _arrays(buf, sensor):
            a[w, k] = a[w[0], k] if value is None else value
        _log(buf, "stuck_at", S.NODES[k], start_s, duration_s, seed, sensor=sensor, value=value)


def isolated_spikes(buf, seed, start_s, duration_s, target, sensor="imu", n=None, amplitude=None):
    w = _widx(buf, start_s, duration_s)
    n = DEFAULT_PARAMS["n_spikes"] if n is None else n
    amp = DEFAULT_PARAMS["spike"][sensor] if amplitude is None else amplitude
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 102, k)
        pick = r.choice(w, size=min(n, len(w)), replace=False) if len(w) else []
        for i in pick:
            sg = _sign(r)
            for a in _arrays(buf, sensor):
                a[i, k] += sg * amp if sensor == "imu" else amp
        _log(buf, "isolated_spikes", S.NODES[k], start_s, duration_s, seed, sensor=sensor, n=n, amplitude=amp)


def intermittent_disconnect(buf, seed, start_s, duration_s, target, sensor="imu"):
    """Sensor returns no value (NaN) in random off-segments; the packet itself still arrives."""
    w = _widx(buf, start_s, duration_s)
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 103, k)
        off = np.zeros(len(w), bool)
        state = True                               # starts disconnected so the fault is always visible
        u = r.random(len(w))
        for i in range(len(w)):
            off[i] = state
            state = (u[i] >= DEFAULT_PARAMS["disconnect_p_on"]) if state else (u[i] < DEFAULT_PARAMS["disconnect_p_off"])
        for a in _arrays(buf, sensor):
            a[w[off], k] = np.nan
        _log(buf, "intermittent_disconnect", S.NODES[k], start_s, duration_s, seed, sensor=sensor)


def implausible_reading(buf, seed, start_s, duration_s, target, sensor="imu"):
    w = _widx(buf, start_s, duration_s)
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 104, k)
        if sensor == "imu":
            lo, hi = DEFAULT_PARAMS["implausible_tilt_deg"]
            for a in _arrays(buf, "imu"):
                a[w, k] = r.uniform(lo, hi, len(w)) * np.where(r.random(len(w)) < 0.5, -1, 1)
        else:
            buf.pz_peak[w, k] = DEFAULT_PARAMS["implausible_piezo"]
            buf.pz_p2p[w, k] = 2 * DEFAULT_PARAMS["implausible_piezo"]
            buf.pz_rms[w, k] = -abs(r.normal(0.01, 0.002, len(w)))     # negative rms is impossible
        _log(buf, "implausible_reading", S.NODES[k], start_s, duration_s, seed, sensor=sensor)


# --------------------------------------------------------------------------- hub faults
def motor_stall(buf, seed, start_s, duration_s, target="HUB", stall_node=None):
    """Motor jams pointing at one node; the hub keeps ranging on schedule and labels readings as if
    sweeping, so every label receives the stalled node's distance."""
    e = _eidx(buf, start_s, duration_s)
    if stall_node is None:
        prior = np.nonzero(buf.ev_t < start_s)[0]
        k = int(buf.ev_aim[prior[-1]]) if len(prior) else int(S.rng_for(seed, 110).integers(0, 4))
    else:
        k = S.NODES.index(stall_node)
    buf.ev_meas[e] = k
    buf.ev_true[e] = buf.dist_to_points(e, buf.node_point(e, k))
    _log(buf, "motor_stall", S.HUB, start_s, duration_s, seed, stalled_on=S.NODES[k])


def wrong_dwell_position(buf, seed, start_s, duration_s, target, offset_frac=None):
    """Range for the target node is measured toward a wrong point (partway to the next clockwise corner)."""
    frac = DEFAULT_PARAMS["wrong_dwell_offset_frac"] if offset_frac is None else offset_frac
    cw = S.clockwise_order()
    for k in _nodes(buf, target):
        nxt = cw[(cw.index(k) + 1) % 4]
        e = np.nonzero((buf.ev_aim == k) & (buf.ev_t >= start_s) & (buf.ev_t < start_s + duration_s))[0]
        pt = buf.node_point(e, k) + frac * (buf.node_point(e, nxt) - buf.node_point(e, k))
        buf.ev_true[e] = buf.dist_to_points(e, pt)
        _log(buf, "wrong_dwell_position", S.NODES[k], start_s, duration_s, seed, offset_frac=frac, toward=S.NODES[nxt])


def stale_readings(buf, seed, start_s, duration_s, target="HUB"):
    """Sweep stops; hub keeps re-sending the last reading it had for each node."""
    cur = S.lidar_ranges(buf)
    e = _eidx(buf, start_s, duration_s)
    for k in range(4):
        prior = np.nonzero((buf.ev_aim == k) & (buf.ev_t < start_s) & ~np.isnan(cur))[0]
        last = cur[prior[-1]] if len(prior) else np.nan
        ek = e[buf.ev_aim[e] == k]
        buf.ev_override[ek] = True
        buf.ev_override_val[ek] = last
    _log(buf, "stale_readings", S.HUB, start_s, duration_s, seed)


def hub_self_motion(buf, seed, start_s, duration_s, target="HUB", shift_m=None, ramp_s=None):
    """Hub moves: all 4 ranges shift together by the same signed amount (common mode), via the hub-motion hook."""
    amp = (DEFAULT_PARAMS["hub_shift_m"] if shift_m is None else shift_m) * _sign(S.rng_for(seed, 113))
    ramp = min(DEFAULT_PARAMS["hub_ramp_s"] if ramp_s is None else ramp_s, duration_s / 2)
    end = start_s + duration_s

    def sstep(u):
        u = np.clip(u, 0, 1)
        return u * u * (3 - 2 * u)

    S.add_hub_motion(buf, lambda t: amp * sstep((t - start_s) / ramp) * sstep((end - t) / ramp))
    _log(buf, "hub_self_motion", S.HUB, start_s, duration_s, seed, shift_m=amp, ramp_s=ramp)


# --------------------------------------------------------------------------- communication faults
def packet_drops(buf, seed, start_s, duration_s, target, n=None):
    w = _widx(buf, start_s, duration_s)
    n = DEFAULT_PARAMS["n_drops"] if n is None else n
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 120, k)
        pick = r.choice(w, size=min(n, len(w)), replace=False) if len(w) else []
        buf.node_drop[pick, k] = True
        _log(buf, "packet_drops", S.NODES[k], start_s, duration_s, seed, n=n)


def burst_loss(buf, seed, start_s, duration_s, target, drop_prob=None):
    w = _widx(buf, start_s, duration_s)
    p = DEFAULT_PARAMS["burst_drop_prob"] if drop_prob is None else drop_prob
    for k in _nodes(buf, target):
        r = S.rng_for(seed, 121, k)
        buf.node_drop[w, k] |= r.random(len(w)) < p
        _log(buf, "burst_loss", S.NODES[k], start_s, duration_s, seed, drop_prob=p)


def added_latency(buf, seed, start_s, duration_s, target, added_s=None):
    w = _widx(buf, start_s, duration_s)
    a = DEFAULT_PARAMS["latency_added_s"] if added_s is None else added_s
    for k in _nodes(buf, target):
        buf.latency[w, k] += a
        _log(buf, "added_latency", S.NODES[k], start_s, duration_s, seed, added_s=a)


def node_unavailable(buf, seed, start_s, duration_s, target):
    w = _widx(buf, start_s, duration_s)
    for k in _nodes(buf, target):
        buf.node_drop[w, k] = True
        _log(buf, "node_unavailable", S.NODES[k], start_s, duration_s, seed)


def network_outage(buf, seed, start_s, duration_s, target="ALL"):
    w = _widx(buf, start_s, duration_s)
    buf.node_drop[w, :] = True
    buf.hub_drop[w] = True
    _log(buf, "network_outage", "ALL", start_s, duration_s, seed)


FAULTS = {  # name -> (function, scope)
    "bias_step": (bias_step, "node"), "slow_drift": (slow_drift, "node"), "stuck_at": (stuck_at, "node"),
    "isolated_spikes": (isolated_spikes, "node"), "intermittent_disconnect": (intermittent_disconnect, "node"),
    "implausible_reading": (implausible_reading, "node"),
    "motor_stall": (motor_stall, "hub"), "wrong_dwell_position": (wrong_dwell_position, "node"),
    "stale_readings": (stale_readings, "hub"), "hub_self_motion": (hub_self_motion, "hub"),
    "packet_drops": (packet_drops, "node"), "burst_loss": (burst_loss, "node"),
    "added_latency": (added_latency, "node"), "node_unavailable": (node_unavailable, "node"),
    "network_outage": (network_outage, "network"),
}


def make_spec(type_, start_s, duration_s, target, seed, **params) -> dict:
    return {"type": type_, "start_s": float(start_s), "duration_s": float(duration_s),
            "target": target, "seed": int(seed), "params": params}


def apply_faults(buf, plan: list[dict]) -> list[dict]:
    """Apply specs in start-time order. Returns the records appended to buf.faults."""
    n0 = len(buf.faults)
    for sp in sorted(plan, key=lambda s: s["start_s"]):
        fn, _ = FAULTS[sp["type"]]
        fn(buf, sp["seed"], sp["start_s"], sp["duration_s"], sp["target"], **sp.get("params", {}))
    return buf.faults[n0:]


def sample_fault_plan(seed: int, duration_s: float, n_faults: int = 1, kinds: list[str] | None = None,
                      p_two_nodes: float = 0.2, start_frac=(0.1, 0.8), duration_frac=(0.05, 0.3)) -> list[dict]:
    """Random plan. Default: one node per node-scoped fault, two nodes with probability p_two_nodes."""
    r = S.rng_for(seed, 130)
    kinds = list(kinds or FAULTS)
    plan = []
    for _ in range(n_faults):
        kind = kinds[int(r.integers(0, len(kinds)))]
        scope = FAULTS[kind][1]
        start = float(r.uniform(*start_frac) * duration_s)
        dur = float(min(r.uniform(*duration_frac) * duration_s, duration_s - start))
        if scope == "hub":
            target = "HUB"
        elif scope == "network":
            target = "ALL"
        else:
            n = 2 if r.random() < p_two_nodes else 1
            target = [S.NODES[i] for i in sorted(r.choice(4, size=n, replace=False))]
        plan.append(make_spec(kind, start, dur, target, int(r.integers(0, 2 ** 31 - 1))))
    return plan
