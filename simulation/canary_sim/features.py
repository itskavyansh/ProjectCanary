"""Windowed features (SYNTHETIC / SIMULATION): 5 s windows, 50% overlap (config-driven).

Groups:
1. Magnitude features (existing): per-node mean, std, least-squares slope of m = hypot(tx, ty),
   plus cross-node magnitude correlation `coherence`.
2. Direction features:
   - Observable (model): per-node tilt vector angle `atan2(dy, dx)` relative to lead-in baseline,
     pairwise cosine similarity of mean 2-D tilt vectors across nodes (`dir_pairwise_cos_mean`,
     `dir_pairwise_cos_min`), and circular spread `dir_circular_spread` = 1 - R_bar.
   - Diagnostic ONLY (`diag_` prefix, excluded from `feature_columns`): angle between each node's
     world-frame tilt vector and the true direction toward the basin centre `(x0, y0)`.
3. Signed cross-node coherence: mean pairwise correlation across the 4 nodes of signed `tilt_x`
   and `tilt_y` (`coherence_signed_tx`, `coherence_signed_ty`, `coherence_signed_mean`).
4. Piezo features: per-node peak max, rms mean, peak-to-peak max, impulse count (windows exceeding
   impulse threshold), plus `piezo_impulse_total` and `piezo_nodes_with_impulse`.
5. Hub features: per-node latest known `lidar_range_change_m`, common-mode mean across available
   nodes (`hub_common_mode_m`), per-node residual (`{node}_lidar_residual_m`), and sweep staleness
   (`{node}_lidar_staleness_s`, time since last received LiDAR reading).
6. Communication features: per-node and network missing-packet fraction, max consecutive missing
   burst length (`comm_max_burst_len`), latency mean/max, and `comm_nodes_unavailable` (nodes with
   100% packet loss in the window).
7. Long-horizon features: least-squares slope of window mean tilt magnitude over the last N windows
   (`{node}_long_slope_deg_s`, `long_slope_max_deg_s`, config-driven N, default 12).

NaN / missing-packet handling (documented per feature):
- Magnitude/direction/signed tilt: computed over non-NaN samples in the window; if <1 sample,
  mean/angle fall back to 0.0; if <2 samples, std/slope/correlations fall back to 0.0.
- Pairwise cosine & circular spread: computed over nodes that have >=1 valid sample and non-zero
  vector norm; if <2 valid nodes, cosine = 0.0 and circular spread = 1.0 (maximum dispersion).
- Piezo: NaNs ignored; if a node has all NaN in the window, peak/rms/p2p = 0.0 and impulse count = 0.
- Hub: `lidar_range_change_m` is forward-filled from the most recent received dwell at or before
  the window end; if no reading has arrived yet, range change and residual = 0.0 and staleness =
  `t_end` (time since run start).
- Comms: missing fraction = `1 - mean(packet_received)`. Latency stats over received packets; if
  none received, latency mean/max = `window_s` (timeout fallback).
- Long-horizon slope: fitted over available finite window means in the last N windows; if <2 finite
  windows, slope = 0.0.
Every model feature returned by `feature_columns(df)` is finite (no NaNs).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import node_geometry as ng
from .windows import make_windows, sample_slice

NODES = ng.NODE_IDS
META_COLS = {
    "simulation_id",
    "split",
    "run_class",
    "window_idx",
    "t_start",
    "t_end",
    "t_mid",
    "label",
    "label_set",
    "partial_set",
    "t_rel_event_start_s",
}


def _slope_nan(t: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return 0.0
    tc = t[ok] - t[ok].mean()
    denom = float((tc**2).sum())
    if denom < 1e-12:
        return 0.0
    return float((tc * (y[ok] - y[ok].mean())).sum() / denom)


def _pairwise_corr_mean(mat: np.ndarray) -> float:
    """Mean upper-triangle Pearson correlation across columns of mat (T, 4), NaN-safe."""
    vals = []
    n = mat.shape[1]
    for i in range(n):
        for j in range(i + 1, n):
            xi, xj = mat[:, i], mat[:, j]
            ok = np.isfinite(xi) & np.isfinite(xj)
            if ok.sum() < 2:
                vals.append(0.0)
                continue
            a = xi[ok] - xi[ok].mean()
            b = xj[ok] - xj[ok].mean()
            den = float(np.sqrt((a**2).sum() * (b**2).sum()))
            vals.append(float((a * b).sum() / den) if den > 1e-12 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _max_run_true(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    best = cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def _fit_radial_convergence(
    node_xy: np.ndarray,
    dip_wx: np.ndarray,
    dip_wy: np.ndarray,
    reg_lambda: float = 1e-3,
    reg_centre: tuple[float, float] = (10.0, 10.0),
) -> tuple[float, float, float]:
    """Fit 2-D convergence point c_hat = (cx, cy) of the 4 world-frame downslope ray lines and
    return `(mean_alignment_cos, directed_align_weighted, fit_residual_m)`.

    Crucially, `c_hat` is NOT fixed to (10, 10); it is estimated per window by magnitude-weighted
    orthogonal distance least squares across the 4 rays:
        min_c sum_k m_k^2 * ((c - p_k) . n_k)^2 + reg_lambda * ||c - reg_centre||^2
    where `u_k = (dip_wx[k], dip_wy[k]) / m_k` is the unit world-frame downslope vector from
    nominal installation yaw (`yaw_deg` only, never `yaw_error_deg`) and `n_k = (-u_y, u_x)`.
    """
    m = np.hypot(dip_wx, dip_wy)
    valid = np.isfinite(m) & (m > 1e-6)
    if valid.sum() < 2:
        return 0.0, 0.0, 20.0

    u_x = np.where(valid, dip_wx / np.maximum(m, 1e-12), 0.0)
    u_y = np.where(valid, dip_wy / np.maximum(m, 1e-12), 0.0)
    n_x = -u_y
    n_y = u_x
    w = np.where(valid, m**2, 0.0)
    w_sum = float(w.sum())
    if w_sum < 1e-12:
        return 0.0, 0.0, 20.0
    w_norm = w / w_sum

    # 2x2 normal equations for c_hat
    A00 = float(np.sum(w_norm * n_x * n_x)) + reg_lambda
    A01 = float(np.sum(w_norm * n_x * n_y))
    A11 = float(np.sum(w_norm * n_y * n_y)) + reg_lambda
    p_dot_n = node_xy[:, 0] * n_x + node_xy[:, 1] * n_y
    b0 = float(np.sum(w_norm * p_dot_n * n_x)) + reg_lambda * reg_centre[0]
    b1 = float(np.sum(w_norm * p_dot_n * n_y)) + reg_lambda * reg_centre[1]

    det = A00 * A11 - A01 * A01
    if abs(det) < 1e-12:
        return 0.0, 0.0, 20.0
    cx = (A11 * b0 - A01 * b1) / det
    cy = (A00 * b1 - A01 * b0) / det
    # Clip extreme parallel-ray intersections to +/- 100 m so features stay well-scaled
    cx = float(np.clip(cx, -100.0, 120.0))
    cy = float(np.clip(cy, -100.0, 120.0))

    dx = cx - node_xy[:, 0]
    dy = cy - node_xy[:, 1]
    dist = np.hypot(dx, dy)
    cos_k = np.where(
        valid & (dist > 1e-6),
        np.clip((dx * u_x + dy * u_y) / np.maximum(dist, 1e-6), -1.0, 1.0),
        0.0,
    )
    mean_align_cos = float(np.mean(cos_k[valid]))
    weighted_align_cos = float(np.sum(w_norm[valid] * cos_k[valid]))

    # Orthogonal distance residual (m), penalised when ray points away from c_hat (cos_k < 0)
    perp_d = np.abs(dx * n_x + dy * n_y)
    ray_d = np.where(cos_k >= 0.0, perp_d, dist)
    fit_res_m = float(np.sqrt(np.sum(w_norm[valid] * (ray_d[valid] ** 2))))
    return mean_align_cos, weighted_align_cos, min(fit_res_m, 50.0)


def extract_window_features(
    tele: pd.DataFrame,
    win_s: float = 5.0,
    overlap: float = 0.5,
    long_horizon_windows: int = 12,
    lead_in_s: float = 30.0,
    impulse_peak_thresh: float = 0.035,
    basin_centre_xy: tuple[float, float] | None = None,
    mountings: dict | None = None,
) -> pd.DataFrame:
    """Extract all windowed model and diagnostic (`diag_`) features from a telemetry DataFrame."""
    d = tele[tele.node_id.isin(NODES)]
    has_full = "piezo_rms" in d.columns

    # Fast reshape when telemetry is in standard (time_s, node_id) order
    n_rows = len(d)
    if (
        n_rows % 4 == 0
        and n_rows >= 4
        and tuple(d["node_id"].iloc[:4]) == tuple(NODES)
    ):
        T_len = n_rows // 4
        t = d["time_s"].to_numpy(dtype=float)[::4]
        tx = d["tilt_x_deg"].to_numpy(dtype=float).reshape(T_len, 4)
        ty = d["tilt_y_deg"].to_numpy(dtype=float).reshape(T_len, 4)
        if has_full:
            pk = d["piezo_peak"].to_numpy(dtype=float).reshape(T_len, 4)
            rms = d["piezo_rms"].to_numpy(dtype=float).reshape(T_len, 4)
            p2p = d["piezo_peak_to_peak"].to_numpy(dtype=float).reshape(T_len, 4)
            rcv = d["packet_received"].to_numpy(dtype=float).reshape(T_len, 4)
            arr = d["arrival_time_s"].to_numpy(dtype=float).reshape(T_len, 4)
            lnew = d["lidar_new"].to_numpy(dtype=float).reshape(T_len, 4)
            lrng = d["lidar_range_change_m"].to_numpy(dtype=float).reshape(T_len, 4)
        else:
            pk = rms = p2p = np.zeros_like(tx)
            rcv = np.isfinite(tx).astype(float)
            arr = t[:, None] + np.zeros_like(tx)
            lnew = np.zeros_like(tx)
            lrng = np.full_like(tx, np.nan)
    else:
        tx_df = d.pivot(index="time_s", columns="node_id", values="tilt_x_deg")[list(NODES)]
        ty_df = d.pivot(index="time_s", columns="node_id", values="tilt_y_deg")[list(NODES)]
        t = tx_df.index.to_numpy(dtype=float)
        tx = tx_df.to_numpy(dtype=float)
        ty = ty_df.to_numpy(dtype=float)
        if has_full:
            pk = d.pivot(index="time_s", columns="node_id", values="piezo_peak")[list(NODES)].to_numpy(dtype=float)
            rms = d.pivot(index="time_s", columns="node_id", values="piezo_rms")[list(NODES)].to_numpy(dtype=float)
            p2p = d.pivot(index="time_s", columns="node_id", values="piezo_peak_to_peak")[list(NODES)].to_numpy(dtype=float)
            rcv = d.pivot(index="time_s", columns="node_id", values="packet_received")[list(NODES)].to_numpy(dtype=float)
            arr = d.pivot(index="time_s", columns="node_id", values="arrival_time_s")[list(NODES)].to_numpy(dtype=float)
            lnew = d.pivot(index="time_s", columns="node_id", values="lidar_new")[list(NODES)].to_numpy(dtype=float)
            lrng = d.pivot(index="time_s", columns="node_id", values="lidar_range_change_m")[list(NODES)].to_numpy(dtype=float)
        else:
            pk = rms = p2p = np.zeros_like(tx)
            rcv = np.isfinite(tx).astype(float)
            arr = t[:, None] + np.zeros_like(tx)
            lnew = np.zeros_like(tx)
            lrng = np.full_like(tx, np.nan)

    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    duration_s = float(t[-1] + dt)

    # Baseline tilt per node from lead-in (observable: first lead_in_s of the run)
    lead_mask = t < min(lead_in_s, duration_s * 0.25)
    if not lead_mask.any():
        lead_mask = np.ones(len(t), dtype=bool)
    tx0 = np.nanmedian(tx[lead_mask], axis=0)
    ty0 = np.nanmedian(ty[lead_mask], axis=0)
    tx0 = np.where(np.isfinite(tx0), tx0, 0.0)
    ty0 = np.where(np.isfinite(ty0), ty0, 0.0)

    # Tilt change relative to lead-in and raw magnitude
    dtx = tx - tx0[None, :]
    dty = ty - ty0[None, :]
    mag = np.hypot(tx, ty)

    wins = make_windows(duration_s, win_s, overlap)
    rows = []
    last_lidar_val = np.full(4, np.nan)
    last_lidar_t = np.full(4, np.nan)
    scan_ptr = 0
    node_xy_arr = np.array([ng.NODE_XY[n] for n in NODES], dtype=float)
    # Nominal installation yaw ONLY (`yaw_deg`, never `yaw_error_deg`) for observable world-frame rotation
    nom_yaw_rad = np.array([
        float(np.deg2rad(mountings[n].get("yaw_deg", 0.0))) if (mountings and n in mountings) else 0.0
        for n in NODES
    ], dtype=float)
    c_nom = np.cos(nom_yaw_rad)
    s_nom = np.sin(nom_yaw_rad)

    win_dip_wx = np.zeros((len(wins), 4), dtype=float)
    win_dip_wy = np.zeros((len(wins), 4), dtype=float)

    for w_idx, (t0, t1) in enumerate(wins):
        # Advance LiDAR state up to t1
        end_idx = int(np.searchsorted(t, t1 - 1e-9))
        while scan_ptr < end_idx:
            for k in range(4):
                if lnew[scan_ptr, k] > 0.5 and np.isfinite(lrng[scan_ptr, k]):
                    last_lidar_val[k] = lrng[scan_ptr, k]
                    last_lidar_t[k] = t[scan_ptr]
            scan_ptr += 1

        sl = sample_slice(t, t0, t1)
        ts = t[sl]
        row: dict[str, float | int] = {
            "window_idx": w_idx,
            "t_start": float(t0),
            "t_end": float(t1),
            "t_mid": float(0.5 * (t0 + t1)),
        }

        # 1. Magnitude features (per node + cross-node magnitude coherence)
        m_seg = mag[sl]
        for k, n in enumerate(NODES):
            mk = m_seg[:, k]
            ok = np.isfinite(mk)
            row[f"{n}_mean"] = float(np.nanmean(mk)) if ok.any() else 0.0
            row[f"{n}_std"] = float(np.nanstd(mk, ddof=1)) if ok.sum() >= 2 else 0.0
            row[f"{n}_slope"] = _slope_nan(ts, mk)
        row["coherence"] = _pairwise_corr_mean(m_seg)

        # 2. Direction features: rotate into world frame with NOMINAL installation yaw (`yaw_deg`) ONLY
        dtx_seg = dtx[sl]
        dty_seg = dty[sl]
        unit_vecs = []
        angles = []
        diag_angles = []
        dip_wx_cur = np.zeros(4, dtype=float)
        dip_wy_cur = np.zeros(4, dtype=float)

        for k, n in enumerate(NODES):
            vx = float(np.nanmean(dtx_seg[:, k])) if np.isfinite(dtx_seg[:, k]).any() else 0.0
            vy = float(np.nanmean(dty_seg[:, k])) if np.isfinite(dty_seg[:, k]).any() else 0.0
            # Local downslope vector is (vy, -vx); rotate by NOMINAL yaw_deg into world frame
            w_dx = float(c_nom[k] * vy - s_nom[k] * (-vx))
            w_dy = float(s_nom[k] * vy + c_nom[k] * (-vx))
            dip_wx_cur[k] = w_dx
            dip_wy_cur[k] = w_dy
            win_dip_wx[w_idx, k] = w_dx
            win_dip_wy[w_idx, k] = w_dy

            ang = float(np.arctan2(w_dy, w_dx))
            row[f"{n}_tilt_angle_rad"] = ang
            norm = float(np.hypot(w_dx, w_dy))
            if norm > 1e-9:
                unit_vecs.append((w_dx / norm, w_dy / norm))
                angles.append(ang)

            # Diagnostic feature using TRUE basin centre + effective yaw (excluded from model features)
            if basin_centre_xy is not None:
                nx, ny = ng.NODE_XY[n]
                to_cx, to_cy = basin_centre_xy[0] - nx, basin_centre_xy[1] - ny
                psi = 0.0
                if mountings and n in mountings:
                    m_info = mountings[n]
                    psi = float(np.deg2rad(m_info.get("yaw_deg", 0.0) + m_info.get("yaw_error_deg", 0.0)))
                c_p, s_p = np.cos(psi), np.sin(psi)
                dip_wx_true = c_p * vy - s_p * (-vx)
                dip_wy_true = s_p * vy + c_p * (-vx)
                dn = float(np.hypot(dip_wx_true, dip_wy_true))
                cn = float(np.hypot(to_cx, to_cy))
                if dn > 1e-9 and cn > 1e-9:
                    cos_c = np.clip((dip_wx_true * to_cx + dip_wy_true * to_cy) / (dn * cn), -1.0, 1.0)
                    diag_ang = float(np.rad2deg(np.arccos(cos_c)))
                else:
                    diag_ang = 90.0
            else:
                diag_ang = 90.0
            row[f"diag_{n}_angle_to_true_centre_deg"] = diag_ang
            diag_angles.append(diag_ang)

        row["diag_mean_angle_to_true_centre_deg"] = float(np.mean(diag_angles))
        if len(unit_vecs) >= 2:
            cos_list = []
            for i in range(len(unit_vecs)):
                for j in range(i + 1, len(unit_vecs)):
                    cos_list.append(unit_vecs[i][0] * unit_vecs[j][0] + unit_vecs[i][1] * unit_vecs[j][1])
            row["dir_pairwise_cos_mean"] = float(np.mean(cos_list))
            row["dir_pairwise_cos_min"] = float(np.min(cos_list))
            r_bar = float(np.hypot(np.mean(np.cos(angles)), np.mean(np.sin(angles))))
            row["dir_circular_spread"] = float(1.0 - r_bar)
        else:
            row["dir_pairwise_cos_mean"] = 0.0
            row["dir_pairwise_cos_min"] = 0.0
            row["dir_circular_spread"] = 1.0

        # Fitted single-window radial convergence point c_hat of the 4 world-frame vectors
        al_mean, al_wt, fit_res = _fit_radial_convergence(node_xy_arr, dip_wx_cur, dip_wy_cur)
        row["dir_win_convergence_align_cos"] = al_mean
        row["dir_win_convergence_weighted_cos"] = al_wt
        row["dir_win_convergence_residual_m"] = fit_res

        # 3. Signed cross-node coherence
        c_tx = _pairwise_corr_mean(dtx_seg)
        c_ty = _pairwise_corr_mean(dty_seg)
        row["coherence_signed_tx"] = c_tx
        row["coherence_signed_ty"] = c_ty
        row["coherence_signed_mean"] = 0.5 * (c_tx + c_ty)

        # 4. Piezo features
        pk_seg, rms_seg, p2p_seg = pk[sl], rms[sl], p2p[sl]
        imp_tot = 0
        nodes_imp = 0
        for k, n in enumerate(NODES):
            pkk, rmsk, p2pk = pk_seg[:, k], rms_seg[:, k], p2p_seg[:, k]
            row[f"{n}_piezo_peak_max"] = float(np.nanmax(pkk)) if np.isfinite(pkk).any() else 0.0
            row[f"{n}_piezo_rms_mean"] = float(np.nanmean(rmsk)) if np.isfinite(rmsk).any() else 0.0
            row[f"{n}_piezo_p2p_max"] = float(np.nanmax(p2pk)) if np.isfinite(p2pk).any() else 0.0
            icnt = int(np.sum(np.isfinite(pkk) & (pkk > impulse_peak_thresh)))
            row[f"{n}_piezo_impulse_count"] = icnt
            imp_tot += icnt
            if icnt > 0:
                nodes_imp += 1
        row["piezo_impulse_total"] = imp_tot
        row["piezo_nodes_with_impulse"] = nodes_imp
        row["piezo_rms_network_mean"] = float(np.mean([row[f"{n}_piezo_rms_mean"] for n in NODES]))

        # 5. Hub features
        avail_lidar = np.isfinite(last_lidar_val)
        cm = float(np.nanmean(last_lidar_val)) if avail_lidar.any() else 0.0
        row["hub_common_mode_m"] = cm
        for k, n in enumerate(NODES):
            if avail_lidar[k]:
                val = float(last_lidar_val[k])
                res = val - cm
                stale = float(t1 - last_lidar_t[k])
            else:
                val = 0.0
                res = 0.0
                stale = float(t1)
            row[f"{n}_lidar_range_change_m"] = val
            row[f"{n}_lidar_residual_m"] = res
            row[f"{n}_lidar_staleness_s"] = stale
        row["hub_residual_abs_max_m"] = float(max(abs(row[f"{n}_lidar_residual_m"]) for n in NODES))
        row["hub_staleness_max_s"] = float(max(row[f"{n}_lidar_staleness_s"] for n in NODES))

        # 6. Communication features
        rcv_seg = rcv[sl]
        arr_seg = arr[sl]
        miss_fracs = []
        bursts = []
        unavail = 0
        for k, n in enumerate(NODES):
            rk = rcv_seg[:, k]
            miss = (rk < 0.5) | ~np.isfinite(rk)
            mf = float(np.mean(miss)) if len(miss) else 1.0
            row[f"{n}_missing_frac"] = mf
            miss_fracs.append(mf)
            bursts.append(_max_run_true(miss))
            if mf >= 1.0 - 1e-9:
                unavail += 1
        row["comm_missing_frac_net"] = float(np.mean(miss_fracs))
        row["comm_max_burst_len"] = int(max(bursts))
        row["comm_nodes_unavailable"] = int(unavail)

        lat = arr_seg - ts[:, None]
        lat_ok = np.isfinite(lat) & (rcv_seg >= 0.5)
        if lat_ok.any():
            row["comm_latency_mean_s"] = float(np.mean(lat[lat_ok]))
            row["comm_latency_max_s"] = float(np.max(lat[lat_ok]))
        else:
            row["comm_latency_mean_s"] = float(win_s)
            row["comm_latency_max_s"] = float(win_s)

        rows.append(row)

    df = pd.DataFrame(rows)

    # 7. Long-horizon features (slope of window mean tilt magnitude & trailing-horizon cross-node
    #    coherence over past windows of the SAME run only: indices max(0, i - N + 1) .. i)
    N = max(2, int(long_horizon_windows))
    t_mids = df["t_mid"].to_numpy(dtype=float)
    mag_win_mat = df[[f"{n}_mean" for n in NODES]].to_numpy(dtype=float)
    for k, n in enumerate(NODES):
        y_all = mag_win_mat[:, k]
        slopes_lh = np.zeros(len(df), dtype=float)
        for i in range(len(df)):
            i0 = max(0, i - N + 1)
            slopes_lh[i] = _slope_nan(t_mids[i0 : i + 1], y_all[i0 : i + 1])
        df[f"{n}_long_slope_deg_s"] = slopes_lh
    df["long_slope_max_deg_s"] = df[[f"{n}_long_slope_deg_s" for n in NODES]].abs().max(axis=1)

    # Trailing-horizon radial convergence: fit c_hat on trailing N-window mean world-frame tilt vectors
    # (strictly causal: uses windows max(0, i - N + 1) .. i of this run only)
    trail_align = np.zeros(len(df), dtype=float)
    trail_w_align = np.zeros(len(df), dtype=float)
    trail_res_m = np.zeros(len(df), dtype=float)
    coh_trail_wx = np.zeros(len(df), dtype=float)
    coh_trail_wy = np.zeros(len(df), dtype=float)
    coh_trail_mag = np.zeros(len(df), dtype=float)
    for i in range(len(df)):
        i0 = max(0, i - N + 1)
        wx_bar = np.mean(win_dip_wx[i0 : i + 1], axis=0)
        wy_bar = np.mean(win_dip_wy[i0 : i + 1], axis=0)
        al_m, al_w, f_res = _fit_radial_convergence(node_xy_arr, wx_bar, wy_bar)
        trail_align[i] = al_m
        trail_w_align[i] = al_w
        trail_res_m[i] = f_res
        if i - i0 + 1 >= 3:
            coh_trail_wx[i] = _pairwise_corr_mean(win_dip_wx[i0 : i + 1])
            coh_trail_wy[i] = _pairwise_corr_mean(win_dip_wy[i0 : i + 1])
            coh_trail_mag[i] = _pairwise_corr_mean(mag_win_mat[i0 : i + 1])
    df["dir_convergence_align_cos"] = trail_align
    df["dir_convergence_weighted_cos"] = trail_w_align
    df["dir_convergence_residual_m"] = trail_res_m
    df["coherence_trailing_wx"] = coh_trail_wx
    df["coherence_trailing_wy"] = coh_trail_wy
    df["coherence_trailing_signed_mean"] = 0.5 * (coh_trail_wx + coh_trail_wy)
    df["coherence_trailing_mag"] = coh_trail_mag
    return df


def window_features(
    tele: pd.DataFrame,
    t_on: float,
    tau: float,
    win_s: float = 5.0,
    overlap: float = 0.5,
    long_horizon_windows: int = 12,
    basin_centre_xy: tuple[float, float] | None = None,
    mountings: dict | None = None,
) -> pd.DataFrame:
    """Backward-compatible helper used by scripts/run_one.py, plus all new features."""
    df = extract_window_features(
        tele,
        win_s=win_s,
        overlap=overlap,
        long_horizon_windows=long_horizon_windows,
        basin_centre_xy=basin_centre_xy,
        mountings=mountings,
    )
    labels = []
    for _, r in df.iterrows():
        t0, t1 = r["t_start"], r["t_end"]
        if t1 <= t_on:
            labels.append(0)
        elif t0 >= t_on and t1 <= t_on + tau:
            labels.append(1)
        else:
            labels.append(-1)
    df["label"] = labels
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return model feature columns: excludes metadata, overlap columns, and all `diag_` columns."""
    out = []
    for c in df.columns:
        if c in META_COLS or c.startswith("diag_") or c.startswith("overlap_"):
            continue
        out.append(c)
    return out


def diagnostic_columns(df: pd.DataFrame) -> list[str]:
    """Return diagnostic-only (`diag_`) columns that use ground-truth centre coordinates."""
    return [c for c in df.columns if c.startswith("diag_")]
