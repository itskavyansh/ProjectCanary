"""Models and baselines (SYNTHETIC / SIMULATION): ThresholdRule, LogisticRegression, RandomForest.

Strict rules:
- Only model features returned by `feature_columns(df)` are ever used (`diag_` columns are forbidden).
- Only `train` and `val` splits are permitted in fitting or tuning functions. Passing any row with
  `split` in (`test`, `shifted_test`, `stress`, `broad_basin`) immediately raises `ValueError`.
- Scalers and imputers are fit on `train` only.
- All models are deterministic with fixed random seeds.

Feature group mapping (`FEATURE_GROUPS`) supports Random Forest single-group ablations:
- `direction`: per-node tilt vector angles, pairwise cosine similarities, circular spread
- `coherence`: magnitude coherence and signed `tilt_x` / `tilt_y` cross-node correlations
- `piezo`: per-node and network piezo peak, RMS, peak-to-peak, and impulse counts
- `hub`: per-node LiDAR range change, common-mode shift, node residuals, and sweep staleness
- `comm`: per-node and network missing-packet fractions, max burst length, unavailable count, latency
- `long_horizon`: trailing N-window least-squares slopes of tilt magnitude
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from . import classes as C, features as F, labels as L

FORBIDDEN_FIT_SPLITS = frozenset({"test", "shifted_test", "stress", "broad_basin"})
ALLOWED_FIT_SPLITS = frozenset({"train", "val"})

NODES = F.NODES

FEATURE_GROUPS: dict[str, list[str]] = {
    "direction": [
        *[f"{n}_tilt_angle_rad" for n in NODES],
        "dir_pairwise_cos_mean",
        "dir_pairwise_cos_min",
        "dir_circular_spread",
        "dir_win_convergence_align_cos",
        "dir_win_convergence_weighted_cos",
        "dir_win_convergence_residual_m",
        "dir_convergence_align_cos",
        "dir_convergence_weighted_cos",
        "dir_convergence_residual_m",
    ],
    "coherence": [
        "coherence",
        "coherence_signed_tx",
        "coherence_signed_ty",
        "coherence_signed_mean",
        "coherence_trailing_wx",
        "coherence_trailing_wy",
        "coherence_trailing_signed_mean",
        "coherence_trailing_mag",
    ],
    "piezo": [
        *[f"{n}_piezo_peak_max" for n in NODES],
        *[f"{n}_piezo_rms_mean" for n in NODES],
        *[f"{n}_piezo_p2p_max" for n in NODES],
        *[f"{n}_piezo_impulse_count" for n in NODES],
        "piezo_impulse_total",
        "piezo_nodes_with_impulse",
        "piezo_rms_network_mean",
    ],
    "hub": [
        "hub_common_mode_m",
        *[f"{n}_lidar_range_change_m" for n in NODES],
        *[f"{n}_lidar_residual_m" for n in NODES],
        *[f"{n}_lidar_staleness_s" for n in NODES],
        "hub_residual_abs_max_m",
        "hub_staleness_max_s",
    ],
    "comm": [
        *[f"{n}_missing_frac" for n in NODES],
        "comm_missing_frac_net",
        "comm_max_burst_len",
        "comm_nodes_unavailable",
        "comm_latency_mean_s",
        "comm_latency_max_s",
    ],
    "long_horizon": [
        *[f"{n}_long_slope_deg_s" for n in NODES],
        "long_slope_max_deg_s",
    ],
}


def guard_no_test_data(*dfs: pd.DataFrame) -> None:
    """Fail immediately if any forbidden evaluation split is passed to a fit or tuning function."""
    for df in dfs:
        if df is None:
            continue
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame with a 'split' column.")
        if "split" not in df.columns:
            raise ValueError("DataFrame passed to fit/tune is missing the required 'split' column.")
        splits = set(df["split"].dropna().unique())
        bad = splits & FORBIDDEN_FIT_SPLITS
        if bad:
            raise ValueError(
                f"Data leakage guard triggered: forbidden split(s) {sorted(bad)} passed to fit/tune!"
            )
        unknown = splits - ALLOWED_FIT_SPLITS
        if unknown:
            raise ValueError(f"Unrecognised split(s) {sorted(unknown)} passed to fit/tune.")


def guard_model_features(cols: list[str]) -> list[str]:
    """Ensure no diagnostic (`diag_`) or metadata columns are in the feature list."""
    bad = [c for c in cols if c.startswith("diag_") or c in F.META_COLS or c.startswith("overlap_")]
    if bad:
        raise ValueError(f"Forbidden non-model columns in feature list: {bad}")
    return list(cols)


def prepare_trainable(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out 'transition' windows for training or window-level scoring."""
    return df[L.trainable(df)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Hand-built Threshold Rule (tuned on train & val only at current noise)
# ---------------------------------------------------------------------------
@dataclass
class ThresholdRuleClassifier:
    """Hand-built hierarchical threshold classifier on tilt magnitude, piezo, hub range, and comms.

    All thresholds are calibrated on `train` quantiles (at the current MPU6050/ADS1115 noise profile)
    and fine-tuned on `val` macro-F1.
    """

    thresholds: dict[str, float] = field(default_factory=dict)
    tuning_history: list[dict] = field(default_factory=list)
    fitted_on_splits: tuple[str, ...] = ("train", "val")

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "ThresholdRuleClassifier":
        guard_no_test_data(train_df, val_df)
        if set(train_df["split"].unique()) != {"train"}:
            raise ValueError("train_df must contain only the 'train' split.")
        if set(val_df["split"].unique()) != {"val"}:
            raise ValueError("val_df must contain only the 'val' split.")

        tr = prepare_trainable(train_df)
        va = prepare_trainable(val_df)
        guard_model_features(F.feature_columns(tr))

        norm = tr[tr["label"] == "normal"]
        grad = tr[tr["label"] == "gradual"]
        rap = tr[tr["label"] == "rapid"]
        abn = tr[tr["label"] == "abnormal_vibration"]

        # Base quantiles derived strictly from `train` at current noise profile:
        # 1) Comm fault: background loss is 1%/pkt; >=2 missing packets in 5 s (20 pkts) is 0.10,
        #    and normal p99.9 latency max is ~2.8 s while added_latency fault adds +5 s.
        comm_miss_th = float(max(0.12, np.quantile(norm["comm_missing_frac_net"], 0.995)))
        comm_lat_th = float(max(3.0, np.quantile(norm["comm_latency_max_s"], 0.999)))

        # 2) Abnormal vibration: midpoint between normal p99.9 piezo_rms_network_mean and abnormal_vibration p1
        piezo_rms_th = float(
            0.5 * (np.quantile(norm["piezo_rms_network_mean"], 0.999) + np.quantile(abn["piezo_rms_network_mean"], 0.01))
        )
        piezo_nodes_th = 3.0

        # 3) Hub range faults: hub_self_motion shifts hub_common_mode_m (~0.30 m); motor_stall / wrong_dwell
        #    create large node-specific residuals (>0.12 m vs normal p99.9 ~0.10 m).
        hub_cm_th = float(max(0.09, np.quantile(norm["hub_common_mode_m"].abs(), 0.999)))
        hub_res_th = float(max(0.11, np.quantile(norm["hub_residual_abs_max_m"], 0.999)))

        # 4) Sensor fault (IMU): implausible (>3.5 deg), stuck_at (min std == 0 while packet_received),
        #    isolated spikes (max single-node std > 3x normal p99), or single-node tilt excursion
        #    where max_tilt / median_tilt ratio is high (or radial convergence residual is large) and max_tilt exceeds normal p99.
        norm_std_max = norm[[f"{n}_std" for n in NODES]].max(axis=1)
        imu_spike_std_th = float(np.quantile(norm_std_max, 0.998) * 1.5)
        imu_implausible_th = 3.5

        has_conv = "dir_convergence_residual_m" in tr.columns
        conv_res_th = (
            float(np.quantile(grad["dir_convergence_residual_m"], 0.85))
            if has_conv
            else 999.0
        )

        # Candidate grid on (basin_tilt_q, single_node_fault_ratio, rapid_vs_gradual_slope_q) tuned on `val`
        norm_tilt_max = norm[[f"{n}_mean" for n in NODES]].max(axis=1)
        norm_tilt_med = norm[[f"{n}_mean" for n in NODES]].median(axis=1)
        grad_slope = grad["long_slope_max_deg_s"]
        rap_slope = rap[[f"{n}_slope" for n in NODES]].abs().max(axis=1)

        best_f1 = -1.0
        best_th = {}
        history = []

        for q_tilt in (0.94, 0.96, 0.98, 0.99):
            tilt_max_th = float(np.quantile(norm_tilt_max, q_tilt))
            tilt_med_th = float(np.quantile(norm_tilt_med, q_tilt - 0.02))
            for sf_ratio in (2.0, 2.4, 2.8):
                for q_rap in (0.60, 0.75, 0.85):
                    rap_win_slope_th = float(np.quantile(rap_slope, q_rap))
                    grad_lh_slope_th = float(np.quantile(grad_slope, 0.55))
                    cand = {
                        "comm_missing_frac_net_th": round(comm_miss_th, 5),
                        "comm_max_burst_len_th": 2.0,
                        "comm_latency_max_s_th": round(comm_lat_th, 5),
                        "piezo_rms_network_mean_th": round(piezo_rms_th, 5),
                        "piezo_nodes_with_impulse_th": piezo_nodes_th,
                        "hub_common_mode_abs_m_th": round(hub_cm_th, 5),
                        "hub_residual_abs_max_m_th": round(hub_res_th, 5),
                        "imu_implausible_tilt_deg_th": imu_implausible_th,
                        "imu_stuck_std_deg_th": 1e-4,
                        "imu_spike_std_deg_th": round(imu_spike_std_th, 5),
                        "imu_single_node_ratio_th": sf_ratio,
                        "dir_convergence_residual_m_th": round(conv_res_th, 5),
                        "basin_tilt_max_deg_th": round(tilt_max_th, 5),
                        "basin_tilt_median_deg_th": round(tilt_med_th, 5),
                        "rapid_win_slope_deg_s_th": round(rap_win_slope_th, 5),
                        "gradual_long_slope_deg_s_th": round(grad_lh_slope_th, 5),
                    }
                    preds = self._apply_rule(va, cand)
                    macro_f1 = float(f1_score(va["label"], preds, labels=list(C.CLASSES), average="macro", zero_division=0))
                    history.append({
                        "q_tilt": q_tilt,
                        "sf_ratio": sf_ratio,
                        "q_rap": q_rap,
                        "val_macro_f1": round(macro_f1, 5),
                    })
                    if macro_f1 > best_f1:
                        best_f1 = macro_f1
                        best_th = cand

        self.thresholds = best_th
        self.tuning_history = history
        return self

    @staticmethod
    def _apply_rule(df: pd.DataFrame, th: dict[str, float]) -> np.ndarray:
        n = len(df)
        out = np.full(n, "normal", dtype=object)

        m_cols = [f"{nd}_mean" for nd in NODES]
        s_cols = [f"{nd}_std" for nd in NODES]
        sl_cols = [f"{nd}_slope" for nd in NODES]
        ms_cols = [f"{nd}_missing_frac" for nd in NODES]

        m_mat = df[m_cols].to_numpy(dtype=float)
        s_mat = df[s_cols].to_numpy(dtype=float)
        sl_mat = np.abs(df[sl_cols].to_numpy(dtype=float))
        ms_mat = df[ms_cols].to_numpy(dtype=float)

        tilt_max = np.max(m_mat, axis=1)
        tilt_med = np.median(m_mat, axis=1)
        tilt_second = np.sort(m_mat, axis=1)[:, 2]
        std_max = np.max(s_mat, axis=1)
        win_slope_max = np.max(sl_mat, axis=1)

        # Stuck-at or disconnect on a node that is still delivering packets (missing_frac < 0.8)
        node_stuck = ((s_mat <= th["imu_stuck_std_deg_th"]) & (ms_mat < 0.8)).any(axis=1)

        # 1. Basin tilt detection (gradual vs rapid)
        is_basin = (tilt_max >= th["basin_tilt_max_deg_th"]) | (tilt_med >= th["basin_tilt_median_deg_th"])
        is_rapid = is_basin & (
            (win_slope_max >= th["rapid_win_slope_deg_s_th"])
            | (
                (tilt_max >= 1.25 * th["basin_tilt_max_deg_th"])
                & (df["long_slope_max_deg_s"].to_numpy(dtype=float) < th["gradual_long_slope_deg_s_th"])
            )
        )
        out[is_basin & ~is_rapid] = "gradual"
        out[is_rapid] = "rapid"

        # 2. Sensor fault (hub range fault OR IMU fault)
        hub_fault = (
            (np.abs(df["hub_common_mode_m"].to_numpy(dtype=float)) >= th["hub_common_mode_abs_m_th"])
            | (df["hub_residual_abs_max_m"].to_numpy(dtype=float) >= th["hub_residual_abs_max_m_th"])
        )
        ratio_val = tilt_max / np.maximum(tilt_second, 0.15)
        if "dir_convergence_residual_m" in df.columns and "dir_convergence_residual_m_th" in th:
            conv_res = df["dir_convergence_residual_m"].to_numpy(dtype=float)
            conv_w_cos = df["dir_convergence_weighted_cos"].to_numpy(dtype=float)
            # When vectors strongly converge to a fitted basin point (weighted_cos > 0.75 & low residual),
            # require a higher single-node ratio so asymmetric gradual basins aren't flagged as faults;
            # when vectors fail to converge (conv_w_cos < 0), lower the ratio threshold to catch 1-2 node drift.
            eff_ratio_th = np.where(
                (conv_w_cos > 0.75) & (conv_res < th["dir_convergence_residual_m_th"]),
                th["imu_single_node_ratio_th"] * 1.25,
                np.where(conv_w_cos < -0.15, th["imu_single_node_ratio_th"] * 0.78, th["imu_single_node_ratio_th"]),
            )
            single_node_anomaly = (tilt_max >= th["basin_tilt_max_deg_th"] * 1.10) & (ratio_val >= eff_ratio_th)
        else:
            single_node_anomaly = (
                (tilt_max >= th["basin_tilt_max_deg_th"] * 1.15)
                & (ratio_val >= th["imu_single_node_ratio_th"])
            )
        imu_fault = (
            (tilt_max >= th["imu_implausible_tilt_deg_th"])
            | node_stuck
            | (std_max >= th["imu_spike_std_deg_th"])
            | single_node_anomaly
        )
        out[hub_fault | imu_fault] = "sensor_fault"

        # 3. Abnormal vibration (sustained multi-node piezo energy)
        is_abn_vib = (
            (df["piezo_rms_network_mean"].to_numpy(dtype=float) >= th["piezo_rms_network_mean_th"])
            | (
                (df["piezo_nodes_with_impulse"].to_numpy(dtype=float) >= th["piezo_nodes_with_impulse_th"])
                & (df["piezo_rms_network_mean"].to_numpy(dtype=float) >= 0.65 * th["piezo_rms_network_mean_th"])
            )
        )
        out[is_abn_vib] = "abnormal_vibration"

        # 4. Communication fault (packet loss, burst loss, node unavailability, or high latency)
        is_comm = (
            (df["comm_missing_frac_net"].to_numpy(dtype=float) >= th["comm_missing_frac_net_th"])
            | (df["comm_max_burst_len"].to_numpy(dtype=float) >= th["comm_max_burst_len_th"])
            | (df["comm_nodes_unavailable"].to_numpy(dtype=float) >= 1.0)
            | (df["comm_latency_max_s"].to_numpy(dtype=float) >= th["comm_latency_max_s_th"])
        )
        out[is_comm] = "comm_fault"

        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if not self.thresholds:
            raise RuntimeError("ThresholdRuleClassifier has not been fitted.")
        guard_model_features(F.feature_columns(df))
        return self._apply_rule(df, self.thresholds)


# ---------------------------------------------------------------------------
# 2. Logistic Regression (balanced class weights, train-only scaler/imputer)
# ---------------------------------------------------------------------------
@dataclass
class SklearnModelWrapper:
    name: str
    pipeline: Pipeline
    feature_cols: list[str]
    best_params: dict = field(default_factory=dict)
    tuning_grid_results: list[dict] = field(default_factory=list)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        guard_model_features(self.feature_cols)
        X = df[self.feature_cols].to_numpy(dtype=float)
        # Clip extreme implausible readings to +/- 1e4 so linear models never overflow
        X = np.clip(X, -1e4, 1e4)
        return self.pipeline.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        guard_model_features(self.feature_cols)
        X = df[self.feature_cols].to_numpy(dtype=float)
        X = np.clip(X, -1e4, 1e4)
        return self.pipeline.predict_proba(X)


def fit_logistic_regression(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    seed: int = 20261117,
    feature_cols: list[str] | None = None,
    c_grid: tuple[float, ...] = (0.1, 1.0, 5.0),
) -> SklearnModelWrapper:
    """Fit balanced Logistic Regression. Imputer and RobustScaler are fit on `train` ONLY."""
    guard_no_test_data(train_df, val_df)
    if set(train_df["split"].unique()) != {"train"}:
        raise ValueError("train_df must contain only the 'train' split.")
    if val_df is not None and set(val_df["split"].unique()) != {"val"}:
        raise ValueError("val_df must contain only the 'val' split.")

    tr = prepare_trainable(train_df)
    cols = guard_model_features(feature_cols if feature_cols is not None else F.feature_columns(tr))
    X_tr = np.clip(tr[cols].to_numpy(dtype=float), -1e4, 1e4)
    y_tr = tr["label"].to_numpy()

    if val_df is not None:
        va = prepare_trainable(val_df)
        X_va = np.clip(va[cols].to_numpy(dtype=float), -1e4, 1e4)
        y_va = va["label"].to_numpy()
    else:
        X_va, y_va = None, None

    best_pipe = None
    best_c = c_grid[0]
    best_f1 = -1.0
    grid_res = []

    for c_val in c_grid:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_val,
                    class_weight="balanced",
                    max_iter=300,
                    tol=1e-3,
                    random_state=int(seed),
                ),
            ),
        ])
        pipe.fit(X_tr, y_tr)
        if X_va is not None:
            preds = pipe.predict(X_va)
            f1 = float(f1_score(y_va, preds, labels=list(C.CLASSES), average="macro", zero_division=0))
        else:
            f1 = float(f1_score(y_tr, pipe.predict(X_tr), labels=list(C.CLASSES), average="macro", zero_division=0))
        grid_res.append({"C": c_val, "val_macro_f1": round(f1, 5)})
        if f1 > best_f1:
            best_f1 = f1
            best_c = c_val
            best_pipe = pipe

    return SklearnModelWrapper(
        name="logistic_regression",
        pipeline=best_pipe,
        feature_cols=cols,
        best_params={"C": best_c, "class_weight": "balanced", "max_iter": 1000, "random_state": int(seed)},
        tuning_grid_results=grid_res,
    )


# ---------------------------------------------------------------------------
# 3. Random Forest (balanced class weights, tuned on small val grid)
# ---------------------------------------------------------------------------
RF_DEFAULT_GRID = {
    "n_estimators": [120, 200],
    "max_depth": [12, 20],
    "min_samples_leaf": [1, 3],
}


def fit_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    seed: int = 20261117,
    feature_cols: list[str] | None = None,
    param_grid: dict | None = None,
    fixed_params: dict | None = None,
) -> SklearnModelWrapper:
    """Fit balanced Random Forest. Imputer is fit on `train` ONLY; hyperparameters tuned on `val` ONLY."""
    guard_no_test_data(train_df, val_df)
    if set(train_df["split"].unique()) != {"train"}:
        raise ValueError("train_df must contain only the 'train' split.")
    if val_df is not None and set(val_df["split"].unique()) != {"val"}:
        raise ValueError("val_df must contain only the 'val' split.")

    tr = prepare_trainable(train_df)
    cols = guard_model_features(feature_cols if feature_cols is not None else F.feature_columns(tr))
    X_tr = np.clip(tr[cols].to_numpy(dtype=float), -1e4, 1e4)
    y_tr = tr["label"].to_numpy()

    if fixed_params is not None:
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(fixed_params["n_estimators"]),
                    max_depth=fixed_params["max_depth"],
                    min_samples_leaf=int(fixed_params["min_samples_leaf"]),
                    class_weight="balanced",
                    random_state=int(seed),
                    n_jobs=1,
                ),
            ),
        ])
        pipe.fit(X_tr, y_tr)
        return SklearnModelWrapper(
            name="random_forest",
            pipeline=pipe,
            feature_cols=cols,
            best_params={**fixed_params, "class_weight": "balanced", "random_state": int(seed)},
            tuning_grid_results=[],
        )

    if val_df is None:
        raise ValueError("val_df is required when tuning Random Forest on the hyperparameter grid.")
    va = prepare_trainable(val_df)
    X_va = np.clip(va[cols].to_numpy(dtype=float), -1e4, 1e4)
    y_va = va["label"].to_numpy()

    grid = param_grid or RF_DEFAULT_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))

    best_pipe = None
    best_params = {}
    best_f1 = -1.0
    grid_res = []

    for vals in combos:
        p = dict(zip(keys, vals))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(p["n_estimators"]),
                    max_depth=p["max_depth"],
                    min_samples_leaf=int(p["min_samples_leaf"]),
                    class_weight="balanced",
                    random_state=int(seed),
                    n_jobs=1,
                ),
            ),
        ])
        pipe.fit(X_tr, y_tr)
        preds = pipe.predict(X_va)
        f1 = float(f1_score(y_va, preds, labels=list(C.CLASSES), average="macro", zero_division=0))
        rec = {**p, "val_macro_f1": round(f1, 5)}
        grid_res.append(rec)
        if f1 > best_f1:
            best_f1 = f1
            best_params = {**p, "class_weight": "balanced", "random_state": int(seed)}
            best_pipe = pipe

    return SklearnModelWrapper(
        name="random_forest",
        pipeline=best_pipe,
        feature_cols=cols,
        best_params=best_params,
        tuning_grid_results=grid_res,
    )


def features_without_group(all_cols: list[str], group_name: str) -> list[str]:
    """Return model feature columns with `FEATURE_GROUPS[group_name]` removed (for ablation)."""
    if group_name not in FEATURE_GROUPS:
        raise KeyError(f"Unknown feature group {group_name!r}; valid groups: {list(FEATURE_GROUPS)}")
    drop = set(FEATURE_GROUPS[group_name])
    return [c for c in guard_model_features(all_cols) if c not in drop]
