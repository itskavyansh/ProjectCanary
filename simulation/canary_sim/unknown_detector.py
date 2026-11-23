"""Unsupervised "unknown_anomaly" detector running beside the frozen 6-class classifier (SYNTHETIC / SIMULATION).

Rules enforced:
- Uses ONLY the 81 model features from `feature_columns(df)`; `diag_` and metadata columns are forbidden.
- Fits strictly on TRAIN windows labelled `normal` (including lead-in and benign confounder windows).
  Passing any non-train split or any non-normal window to `.fit()` immediately raises `ValueError`.
- Chooses the flag threshold strictly on VAL normal windows under `(k=3, m=3)` persistence to achieve
  at most `1.0` false-alarm episode per normal run. Passing any non-val split raises `ValueError`.
- Never assigns a known class and never modifies the frozen classifier's predictions; only outputs
  `anomaly_score` (never called "probability" or "confidence") and an `unknown_anomaly` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from . import alarm as A, classes as C, eval_extras_v2 as V2, features as F, labels as L, models as M

UNKNOWN_FLAG_LABEL = "unknown_anomaly"
KNOWN_EVENT_CLASSES = ("gradual", "rapid", "abnormal_vibration", "sensor_fault", "comm_fault")


def load_unknown_detector_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Strict data-hygiene guards
# -----------------------------------------------------------------------------
def extract_train_normal_windows(train_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a train DataFrame to windows with `label == 'normal'`, rejecting any non-train rows."""
    if not isinstance(train_df, pd.DataFrame) or "split" not in train_df.columns:
        raise ValueError("Expected a DataFrame with a 'split' column.")
    splits = set(train_df["split"].dropna().unique())
    if splits != {"train"}:
        raise ValueError(f"Detector fit requires 'train' split only; got split(s): {sorted(splits)}")
    if "label" not in train_df.columns:
        raise ValueError("DataFrame is missing required 'label' column.")
    out = train_df[train_df["label"] == "normal"].reset_index(drop=True)
    if out.empty:
        raise ValueError("No normal windows found in train_df.")
    return out


def extract_val_normal_windows(val_df: pd.DataFrame) -> pd.DataFrame:
    """Filter a val DataFrame to normal-only runs (`run_class == 'normal'` and `label == 'normal'`),
    rejecting any non-val rows."""
    if not isinstance(val_df, pd.DataFrame) or "split" not in val_df.columns:
        raise ValueError("Expected a DataFrame with a 'split' column.")
    splits = set(val_df["split"].dropna().unique())
    if splits != {"val"}:
        raise ValueError(f"Threshold tuning requires 'val' split only; got split(s): {sorted(splits)}")
    mask = val_df["label"] == "normal"
    if "run_class" in val_df.columns:
        mask = mask & (val_df["run_class"] == "normal")
    out = val_df[mask].reset_index(drop=True)
    if out.empty:
        raise ValueError("No normal windows/runs found in val_df.")
    return out


def guard_train_normal_only(df: pd.DataFrame) -> None:
    """Verify every row passed to .fit() has `split == 'train'` and `label == 'normal'`."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if "split" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame passed to fit() must have 'split' and 'label' columns.")
    splits = set(df["split"].dropna().unique())
    if splits != {"train"}:
        raise ValueError(f"Data leakage guard: detector fit() accepts 'train' split only, got {sorted(splits)}")
    labels = set(df["label"].dropna().unique())
    if labels != {"normal"}:
        raise ValueError(
            f"Unsupervised detector fit() accepts 'normal' windows only, got label(s): {sorted(labels)}"
        )


def guard_val_normal_only(df: pd.DataFrame) -> None:
    """Verify every row passed to .tune_threshold_on_val() has `split == 'val'` and `label == 'normal'`."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")
    if "split" not in df.columns or "label" not in df.columns:
        raise ValueError("DataFrame passed to tune_threshold_on_val() must have 'split' and 'label' columns.")
    splits = set(df["split"].dropna().unique())
    if splits != {"val"}:
        raise ValueError(
            f"Data leakage guard: threshold tuning accepts 'val' split only, got {sorted(splits)}"
        )
    labels = set(df["label"].dropna().unique())
    if labels != {"normal"}:
        raise ValueError(
            f"Threshold tuning accepts 'normal' windows only, got label(s): {sorted(labels)}"
        )
    if "run_class" in df.columns:
        rclasses = set(df["run_class"].dropna().unique())
        if rclasses != {"normal"}:
            raise ValueError(
                f"Threshold tuning accepts normal runs only, got run_class(es): {sorted(rclasses)}"
            )


# -----------------------------------------------------------------------------
# Base Unsupervised Detector Interface + Shared Threshold Tuning on VAL Normal
# -----------------------------------------------------------------------------
@dataclass
class BaseUnknownDetector:
    name: str
    feature_cols: list[str] = field(default_factory=list)
    threshold: float = float("inf")
    threshold_quantile_val: float = 1.0
    k: int = 3
    m: int = 3
    fa_target_per_normal_run: float = 1.0
    val_tuning_summary: dict[str, Any] = field(default_factory=dict)
    fitted_on_n_windows: int = 0

    def score_windows(self, df: pd.DataFrame) -> np.ndarray:
        """Return 1-D array of `anomaly_score` (higher = more anomalous). Never called probability."""
        raise NotImplementedError

    def tune_threshold_on_val(
        self,
        val_normal_df: pd.DataFrame,
        k: int = 3,
        m: int = 3,
        max_fa_per_normal_run: float = 1.0,
        quantile_start: float = 0.50,
        quantile_stop: float = 0.999,
        num_points: int = 500,
    ) -> "BaseUnknownDetector":
        """Choose the smallest `anomaly_score` threshold on VAL normal runs that achieves
        at most `max_fa_per_normal_run` persisted `(k, m)` false-alarm episodes per normal run."""
        guard_val_normal_only(val_normal_df)
        self.k = int(k)
        self.m = int(m)
        self.fa_target_per_normal_run = float(max_fa_per_normal_run)

        scores = self.score_windows(val_normal_df)
        # Pre-slice scores per run in temporal order for fast persistence evaluation
        run_slices: list[np.ndarray] = []
        for _, g in val_normal_df.groupby("simulation_id", sort=False):
            order = np.argsort(g["t_start"].to_numpy(dtype=float), kind="stable")
            idx = g.index.to_numpy()[order]
            run_slices.append(scores[idx])

        qs = np.linspace(float(quantile_start), float(quantile_stop), int(num_points))
        cand_ths = np.quantile(scores, qs)

        chosen_th = float(np.max(scores) + 1e-6)
        chosen_q = 1.0
        chosen_eps = 0.0
        chosen_wins = 0.0
        curve = []

        for q, th in zip(qs, cand_ths):
            th_f = float(th)
            eps_list = []
            win_list = []
            for r_scores in run_slices:
                raw_lbls = np.where(r_scores >= th_f, UNKNOWN_FLAG_LABEL, "normal")
                seq, n_eps = A.apply_persistence_sequence(raw_lbls, k=self.k, m=self.m)
                eps_list.append(n_eps)
                win_list.append(int((seq != "normal").sum()))
            mean_eps = float(np.mean(eps_list))
            mean_wins = float(np.mean(win_list))
            curve.append({
                "quantile": round(float(q), 5),
                "threshold": round(th_f, 6),
                "val_fa_episodes_per_normal_run": round(mean_eps, 4),
                "val_fa_windows_per_normal_run": round(mean_wins, 4),
            })
            if mean_eps <= self.fa_target_per_normal_run + 1e-12 and chosen_q == 1.0:
                chosen_th = th_f
                chosen_q = float(q)
                chosen_eps = mean_eps
                chosen_wins = mean_wins

        self.threshold = chosen_th
        self.threshold_quantile_val = chosen_q
        self.val_tuning_summary = {
            "chosen_threshold": round(chosen_th, 6),
            "chosen_quantile_on_val_normal": round(chosen_q, 5),
            "k": self.k,
            "m": self.m,
            "fa_target_per_normal_run": self.fa_target_per_normal_run,
            "val_n_normal_runs": len(run_slices),
            "val_n_normal_windows": len(val_normal_df),
            "val_fa_episodes_per_normal_run": round(chosen_eps, 4),
            "val_fa_windows_per_normal_run": round(chosen_wins, 4),
            "sample_curve": curve[:: max(1, len(curve) // 25)],
        }
        return self

    def flag_windows_raw(self, df: pd.DataFrame, scores: np.ndarray | None = None) -> np.ndarray:
        """Boolean array: True where window `anomaly_score >= self.threshold` (before persistence)."""
        sc = self.score_windows(df) if scores is None else scores
        return sc >= self.threshold

    def flag_windows_persisted(
        self, df: pd.DataFrame, scores: np.ndarray | None = None
    ) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
        """Apply `(k, m)` persistence per run to the raw anomaly flags.

        Returns `(persisted_bool_array, run_stats_dict)`.
        """
        raw_bool = self.flag_windows_raw(df, scores=scores)
        raw_str = np.where(raw_bool, UNKNOWN_FLAG_LABEL, "normal")
        pers_str, run_stats = A.apply_persistence_df(df, raw_str, k=self.k, m=self.m)
        return (pers_str == UNKNOWN_FLAG_LABEL), run_stats

    def score_and_flag_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a table with `anomaly_score`, `unknown_anomaly_raw`, and persisted `unknown_anomaly`."""
        sc = self.score_windows(df)
        raw_b = sc >= self.threshold
        pers_b, _ = self.flag_windows_persisted(df, scores=sc)
        out = pd.DataFrame({
            "simulation_id": df["simulation_id"].to_numpy(),
            "window_idx": df["window_idx"].to_numpy() if "window_idx" in df.columns else np.arange(len(df)),
            "t_start": df["t_start"].to_numpy(dtype=float),
            "t_end": df["t_end"].to_numpy(dtype=float),
            "anomaly_score": sc,
            "unknown_anomaly_raw": raw_b.astype(int),
            "unknown_anomaly": pers_b.astype(int),
        })
        return out


# -----------------------------------------------------------------------------
# 1. Primary Detector: Isolation Forest
# -----------------------------------------------------------------------------
@dataclass
class IsolationForestAnomalyDetector(BaseUnknownDetector):
    name: str = "isolation_forest"
    n_estimators: int = 200
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    clip_scaled: float = 10000.0
    seed: int = 20261117
    imputer: SimpleImputer | None = None
    scaler: RobustScaler | None = None
    iso: IsolationForest | None = None

    def fit(self, train_normal_df: pd.DataFrame, feature_cols: list[str] | None = None) -> "IsolationForestAnomalyDetector":
        guard_train_normal_only(train_normal_df)
        cols = list(feature_cols) if feature_cols is not None else F.feature_columns(train_normal_df)
        M.guard_model_features(cols)
        self.feature_cols = cols
        self.fitted_on_n_windows = len(train_normal_df)

        X_raw = train_normal_df[self.feature_cols].to_numpy(dtype=float)
        X_raw = np.clip(X_raw, -self.clip_scaled, self.clip_scaled)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = RobustScaler()
        X_imp = self.imputer.fit_transform(X_raw)
        X_sc = np.clip(self.scaler.fit_transform(X_imp), -self.clip_scaled, self.clip_scaled)

        self.iso = IsolationForest(
            n_estimators=int(self.n_estimators),
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=int(self.seed),
            n_jobs=1,
        )
        self.iso.fit(X_sc)
        return self

    def score_windows(self, df: pd.DataFrame) -> np.ndarray:
        if self.iso is None or self.imputer is None or self.scaler is None:
            raise RuntimeError("IsolationForestAnomalyDetector has not been fitted.")
        M.guard_model_features(self.feature_cols)
        X_raw = df[self.feature_cols].to_numpy(dtype=float)
        X_raw = np.clip(X_raw, -self.clip_scaled, self.clip_scaled)
        X_sc = np.clip(self.scaler.transform(self.imputer.transform(X_raw)), -self.clip_scaled, self.clip_scaled)
        # Negate score_samples so larger anomaly_score = more anomalous
        return np.asarray(-self.iso.score_samples(X_sc), dtype=float)


# -----------------------------------------------------------------------------
# 2. Secondary Detector: Robust Mahalanobis Distance
# -----------------------------------------------------------------------------
@dataclass
class RobustMahalanobisDetector(BaseUnknownDetector):
    name: str = "robust_mahalanobis"
    trim_quantile: float = 0.98
    shrinkage: float = 0.10
    min_scale: float = 1e-4
    z_clip: float = 50.0
    imputer: SimpleImputer | None = None
    center_med: np.ndarray | None = None
    robust_scale: np.ndarray | None = None
    mean_trimmed: np.ndarray | None = None
    precision_mat: np.ndarray | None = None

    def fit(self, train_normal_df: pd.DataFrame, feature_cols: list[str] | None = None) -> "RobustMahalanobisDetector":
        guard_train_normal_only(train_normal_df)
        cols = list(feature_cols) if feature_cols is not None else F.feature_columns(train_normal_df)
        M.guard_model_features(cols)
        self.feature_cols = cols
        self.fitted_on_n_windows = len(train_normal_df)

        X_raw = train_normal_df[self.feature_cols].to_numpy(dtype=float)
        X_raw = np.clip(X_raw, -1e4, 1e4)
        self.imputer = SimpleImputer(strategy="median")
        X = self.imputer.fit_transform(X_raw)

        med = np.median(X, axis=0)
        q25 = np.percentile(X, 25, axis=0)
        q75 = np.percentile(X, 75, axis=0)
        iqr_s = (q75 - q25) / 1.3489795
        mad_s = 1.4826 * np.median(np.abs(X - med), axis=0)
        std_s = np.std(X, axis=0, ddof=1)
        scale = np.maximum.reduce([iqr_s, mad_s, std_s, np.full_like(med, self.min_scale)])

        self.center_med = med
        self.robust_scale = scale

        Z = np.clip((X - med) / scale, -self.z_clip, self.z_clip)
        norms = np.linalg.norm(Z, axis=1)
        cutoff = float(np.quantile(norms, self.trim_quantile))
        Z_trim = Z[norms <= cutoff]
        if len(Z_trim) < len(cols) + 2:
            Z_trim = Z

        self.mean_trimmed = np.mean(Z_trim, axis=0)
        cov = np.cov(Z_trim, rowvar=False)
        diag_cov = np.diag(np.maximum(np.diag(cov), 1e-3))
        cov_shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * diag_cov
        self.precision_mat = np.linalg.pinv(cov_shrunk, hermitian=True)
        return self

    def score_windows(self, df: pd.DataFrame) -> np.ndarray:
        if self.imputer is None or self.center_med is None or self.robust_scale is None or self.precision_mat is None:
            raise RuntimeError("RobustMahalanobisDetector has not been fitted.")
        M.guard_model_features(self.feature_cols)
        X_raw = np.clip(df[self.feature_cols].to_numpy(dtype=float), -1e4, 1e4)
        X = self.imputer.transform(X_raw)
        Z = np.clip((X - self.center_med) / self.robust_scale, -self.z_clip, self.z_clip)
        diff = Z - self.mean_trimmed
        quad = np.einsum("ij,jk,ik->i", diff, self.precision_mat, diff)
        return np.sqrt(np.maximum(quad, 0.0))


# -----------------------------------------------------------------------------
# 3. Evaluation Suite: Per-Class AUROC, Run-Level Detection, SNR Bins, and Complement Analysis
# -----------------------------------------------------------------------------
def _boot_mean(arr: np.ndarray, n_boot: int, rng: np.random.Generator) -> tuple[float, tuple[float, float]]:
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, (np.nan, np.nan)
    pt = float(np.mean(arr))
    if n_boot <= 0 or len(arr) == 1:
        return pt, (pt, pt)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    b_means = arr[idx].mean(axis=1)
    return pt, (float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5)))


def evaluate_per_class_vs_normal_on_test(
    detector: BaseUnknownDetector,
    test_df: pd.DataFrame,
    n_boot: int = 300,
    seed: int = 20261117,
) -> pd.DataFrame:
    """Evaluate per-class vs normal AUROC and window-level flag recall on `test` with run-bootstrap 95% CIs."""
    tmask = L.trainable(test_df).to_numpy()
    sub = test_df.loc[tmask].copy().reset_index(drop=True)
    scores = detector.score_windows(sub)
    raw_flag = scores >= detector.threshold
    pers_flag, _ = detector.flag_windows_persisted(test_df)
    pers_sub = pers_flag[tmask]

    sub["_score"] = scores
    sub["_raw"] = raw_flag.astype(float)
    sub["_pers"] = pers_sub.astype(float)

    norm_df = sub[sub["label"] == "normal"]
    norm_runs = norm_df["simulation_id"].unique()
    norm_by_run = {rid: norm_df[norm_df["simulation_id"] == rid] for rid in norm_runs}

    rows = []
    targets = list(KNOWN_EVENT_CLASSES) + ["all_known_anomalies"]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1401]))

    for cls in targets:
        if cls == "all_known_anomalies":
            pos_df = sub[sub["label"] != "normal"]
        else:
            pos_df = sub[sub["label"] == cls]
        if pos_df.empty:
            continue

        pos_runs = pos_df["simulation_id"].unique()
        pos_by_run = {rid: pos_df[pos_df["simulation_id"] == rid] for rid in pos_runs}

        y_bin = np.concatenate([np.zeros(len(norm_df)), np.ones(len(pos_df))])
        s_bin = np.concatenate([norm_df["_score"].to_numpy(), pos_df["_score"].to_numpy()])
        pt_auc = float(roc_auc_score(y_bin, s_bin))
        pt_raw_rec = float(pos_df["_raw"].mean())
        pt_pers_rec = float(pos_df["_pers"].mean())

        # Run-bootstrap for AUROC and window flag recall
        b_aucs, b_raw, b_pers = [], [], []
        for _ in range(n_boot):
            p_pick = rng.choice(pos_runs, size=len(pos_runs), replace=True)
            n_pick = rng.choice(norm_runs, size=len(norm_runs), replace=True)
            p_scores = np.concatenate([pos_by_run[r]["_score"].to_numpy() for r in p_pick])
            n_scores = np.concatenate([norm_by_run[r]["_score"].to_numpy() for r in n_pick])
            p_raw = np.concatenate([pos_by_run[r]["_raw"].to_numpy() for r in p_pick])
            p_pers = np.concatenate([pos_by_run[r]["_pers"].to_numpy() for r in p_pick])
            yb = np.concatenate([np.zeros(len(n_scores)), np.ones(len(p_scores))])
            sb = np.concatenate([n_scores, p_scores])
            b_aucs.append(float(roc_auc_score(yb, sb)))
            b_raw.append(float(np.mean(p_raw)))
            b_pers.append(float(np.mean(p_pers)))

        rows.append({
            "detector": detector.name,
            "class_vs_normal": cls,
            "n_class_windows": len(pos_df),
            "n_normal_windows": len(norm_df),
            "auroc": round(pt_auc, 4),
            "auroc_ci_lo": round(float(np.percentile(b_aucs, 2.5)), 4),
            "auroc_ci_hi": round(float(np.percentile(b_aucs, 97.5)), 4),
            "window_flag_recall_raw": round(pt_raw_rec, 4),
            "window_flag_recall_raw_ci_lo": round(float(np.percentile(b_raw, 2.5)), 4),
            "window_flag_recall_raw_ci_hi": round(float(np.percentile(b_raw, 97.5)), 4),
            "window_flag_recall_persisted": round(pt_pers_rec, 4),
            "window_flag_recall_persisted_ci_lo": round(float(np.percentile(b_pers, 2.5)), 4),
            "window_flag_recall_persisted_ci_hi": round(float(np.percentile(b_pers, 97.5)), 4),
        })

    return pd.DataFrame(rows)


def evaluate_run_level_unknown(
    detector: BaseUnknownDetector,
    df: pd.DataFrame,
    split_name: str,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict[str, Any]:
    """Evaluate run-level event flagging, detection delay, and normal-run false alarms."""
    pers_bool, run_stats = detector.flag_windows_persisted(df)
    d = df[["simulation_id", "run_class", "t_start", "t_end", "t_rel_event_start_s"]].copy()
    d["_flag"] = pers_bool
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])

    records = []
    for sim_id, g in d.groupby("simulation_id", sort=False):
        sid = str(sim_id)
        g = g.sort_values("t_start", kind="stable")
        r_cls = str(g["run_class"].iloc[0])
        st = run_stats[sid]
        if r_cls == "normal":
            records.append({
                "simulation_id": sid,
                "run_class": "normal",
                "is_event_run": False,
                "fa_episodes": float(st["n_alarm_episodes"]),
                "fa_windows": float(st["n_alarm_windows"]),
                "event_flagged": np.nan,
                "delay_s": np.nan,
            })
        else:
            t_rel = g["t_rel_event_start_s"].to_numpy(dtype=float)
            fl_arr = g["_flag"].to_numpy(dtype=bool)
            post_mask = t_rel > -win_half + 1e-9
            t_post = t_rel[post_mask]
            fl_post = fl_arr[post_mask]
            idx = np.nonzero(fl_post)[0]
            flagged = 1.0 if len(idx) > 0 else 0.0
            delay = float(max(0.0, t_post[idx[0]] + win_half)) if len(idx) > 0 else np.nan
            records.append({
                "simulation_id": sid,
                "run_class": r_cls,
                "is_event_run": True,
                "fa_episodes": np.nan,
                "fa_windows": np.nan,
                "event_flagged": flagged,
                "delay_s": delay,
            })

    rdf = pd.DataFrame(records)
    ev_runs = rdf[rdf["is_event_run"]]
    norm_runs = rdf[~rdf["is_event_run"]]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1402]))

    pt_fl, (lo_fl, hi_fl) = _boot_mean(ev_runs["event_flagged"].to_numpy(float), n_boot, rng)
    pt_dl, (lo_dl, hi_dl) = _boot_mean(ev_runs["delay_s"].to_numpy(float), n_boot, rng)
    pt_fa, (lo_fa, hi_fa) = _boot_mean(norm_runs["fa_episodes"].to_numpy(float), n_boot, rng)
    pt_fw, (lo_fw, hi_fw) = _boot_mean(norm_runs["fa_windows"].to_numpy(float), n_boot, rng)

    hA, hB = V2.hours_slow(240.0, 360.0), V2.hours_sensor(240.0)
    return {
        "split_or_arm": split_name,
        "detector": detector.name,
        "n_event_runs": len(ev_runs),
        "n_normal_runs": len(norm_runs),
        "event_flagged_rate": round(pt_fl, 4) if pd.notna(pt_fl) else np.nan,
        "event_flagged_ci_lo": round(lo_fl, 4) if pd.notna(lo_fl) else np.nan,
        "event_flagged_ci_hi": round(hi_fl, 4) if pd.notna(hi_fl) else np.nan,
        "missed_event_rate": round(1.0 - pt_fl, 4) if pd.notna(pt_fl) else np.nan,
        "mean_delay_s": round(pt_dl, 3) if pd.notna(pt_dl) else np.nan,
        "mean_delay_ci_lo": round(lo_dl, 3) if pd.notna(lo_dl) else np.nan,
        "mean_delay_ci_hi": round(hi_dl, 3) if pd.notna(hi_dl) else np.nan,
        "fa_episodes_per_normal_run": round(pt_fa, 4) if pd.notna(pt_fa) else np.nan,
        "fa_episodes_ci_lo": round(lo_fa, 4) if pd.notna(lo_fa) else np.nan,
        "fa_episodes_ci_hi": round(hi_fa, 4) if pd.notna(hi_fa) else np.nan,
        "fa_windows_per_normal_run": round(pt_fw, 4) if pd.notna(pt_fw) else np.nan,
        "A_slow_fa_per_hour": round(pt_fa / hA, 5) if pd.notna(pt_fa) else np.nan,
        "B_sensor_fa_per_hour": round(pt_fa / hB, 4) if pd.notna(pt_fa) else np.nan,
        "per_run_df": rdf,
    }


def evaluate_snr_bin_recall_unknown(
    detector: BaseUnknownDetector,
    df: pd.DataFrame,
    manifest_df: pd.DataFrame,
    split_name: str,
    n_boot: int = 300,
    seed: int = 20261117,
) -> list[dict[str, Any]]:
    """Compute gradual and rapid window flag recall and run-level flag rate by SNR bin."""
    raw_bool = detector.flag_windows_raw(df)
    raw_preds_as_cls = np.where(raw_bool, df["label"].to_numpy(), "normal")
    pers_bool, _ = detector.flag_windows_persisted(df)
    d = df[["simulation_id", "run_class", "t_start", "t_end", "t_rel_event_start_s"]].copy()
    d["_flag"] = pers_bool
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])

    det_recs = []
    for sid, g in d.groupby("simulation_id", sort=False):
        r_cls = str(g["run_class"].iloc[0])
        if r_cls == "normal":
            continue
        g = g.sort_values("t_start", kind="stable")
        post = g["t_rel_event_start_s"].to_numpy(dtype=float) > -win_half + 1e-9
        fl = g["_flag"].to_numpy(dtype=bool)[post]
        det_recs.append({"simulation_id": str(sid), "run_class": r_cls, "detected": float(fl.any())})
    if not det_recs:
        return []
    det_df = pd.DataFrame(det_recs)

    snr_map = manifest_df.set_index("simulation_id")["snr_vs_imu_noise"]
    rows = []
    for cls in ("gradual", "rapid"):
        pr = V2.per_run_class_table(df, raw_preds_as_cls, det_df, cls)
        if pr.empty:
            continue
        pr["snr"] = pr["simulation_id"].map(snr_map)
        pr["snr_bin"] = pr["snr"].map(V2.snr_bin_of)
        s_all = V2.summarize_class(pr, n_boot, seed)
        rows.append({
            "split_or_arm": split_name,
            "detector": detector.name,
            "class": cls,
            "snr_bin": "ALL",
            **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_all.items()},
        })
        for b_name, _, _ in V2.SNR_BINS:
            s_b = V2.summarize_class(pr[pr["snr_bin"] == b_name], n_boot, seed)
            rows.append({
                "split_or_arm": split_name,
                "detector": detector.name,
                "class": cls,
                "snr_bin": b_name,
                **{k_: (round(v_, 4) if isinstance(v_, float) and np.isfinite(v_) else v_) for k_, v_ in s_b.items()},
            })
    return rows


def evaluate_complement_with_classifier(
    detector: BaseUnknownDetector,
    clf_name: str,
    clf_window_preds: np.ndarray,
    df_events: pd.DataFrame,
    df_normals: pd.DataFrame | None,
    split_name: str,
    k: int = 3,
    m: int = 3,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict[str, Any]:
    """Evaluate how the unsupervised detector complements a frozen classifier at the v3 episode operating point:
    1. Classifier alone (persisted k=3, m=3)
    2. Detector alone (persisted k=3, m=3)
    3. Combined Rule A: 'classifier_alarm OR detector_flag'
    4. Combined Rule B (Third Rule): 'unknown_only_when_detector_flags_and_classifier_says_normal'
    """
    # --- Event runs analysis ---
    clf_alarm_ev, _ = A.apply_persistence_df(df_events, clf_window_preds, k=k, m=m)
    det_raw_ev = detector.flag_windows_raw(df_events)
    det_pers_ev, _ = detector.flag_windows_persisted(df_events)

    # Third rule on event runs: window flagged as 'unknown_anomaly' only when detector flags AND clf_window_preds == 'normal'
    gated_raw_ev = np.where(det_raw_ev & (np.asarray(clf_window_preds) == "normal"), UNKNOWN_FLAG_LABEL, "normal")
    gated_pers_str_ev, _ = A.apply_persistence_df(df_events, gated_raw_ev, k=k, m=m)
    gated_pers_ev = gated_pers_str_ev == UNKNOWN_FLAG_LABEL

    d_ev = df_events[["simulation_id", "run_class", "label_set", "t_start", "t_end", "t_rel_event_start_s"]].copy()
    d_ev["_clf_al"] = clf_alarm_ev
    d_ev["_det_al"] = det_pers_ev
    d_ev["_gated_al"] = gated_pers_ev
    win_half = 0.5 * float((d_ev["t_end"] - d_ev["t_start"]).iloc[0])

    ev_recs = []
    for sid, g in d_ev.groupby("simulation_id", sort=False):
        r_cls = str(g["run_class"].iloc[0])
        if r_cls == "normal":
            continue
        g = g.sort_values("t_start", kind="stable")
        post = g["t_rel_event_start_s"].to_numpy(dtype=float) > -win_half + 1e-9
        al_c = g["_clf_al"].to_numpy(dtype=object)[post]
        al_d = g["_det_al"].to_numpy(dtype=bool)[post]
        al_g = g["_gated_al"].to_numpy(dtype=bool)[post]
        ls_post = g["label_set"].fillna("").to_numpy(dtype=object)[post]

        clf_any = float((al_c != "normal").any())
        if (al_c == r_cls).any():
            clf_exact = 1.0
        else:
            matches = [
                i for i, (pr, ls) in enumerate(zip(al_c, ls_post))
                if pr != "normal" and (pr == r_cls or (ls and str(pr) in str(ls).split("|")))
            ]
            clf_exact = 1.0 if len(matches) > 0 else 0.0

        det_fl = float(al_d.any())
        gated_fl = float(al_g.any())
        ev_recs.append({
            "simulation_id": str(sid),
            "run_class": r_cls,
            "clf_exact": clf_exact,
            "clf_any": clf_any,
            "det_flagged": det_fl,
            "gated_flagged": gated_fl,
            "rule_or_exact_or_det": float((clf_exact > 0.5) or (det_fl > 0.5)),
            "rule_or_any_or_det": float((clf_any > 0.5) or (det_fl > 0.5)),
            "rule_gated_exact_or_gated": float((clf_exact > 0.5) or (gated_fl > 0.5)),
        })
    ev_df = pd.DataFrame(ev_recs)

    exact_missed = ev_df[ev_df["clf_exact"] < 0.5]
    any_missed = ev_df[ev_df["clf_any"] < 0.5]
    n_ev = len(ev_df)
    n_exact_missed = len(exact_missed)
    n_any_missed = len(any_missed)
    n_recov_exact_or = int(exact_missed["det_flagged"].sum()) if n_exact_missed > 0 else 0
    n_recov_any_or = int(any_missed["det_flagged"].sum()) if n_any_missed > 0 else 0
    n_recov_exact_gated = int(exact_missed["gated_flagged"].sum()) if n_exact_missed > 0 else 0

    # --- Normal runs false-alarm cost analysis ---
    if df_normals is not None and not df_normals.empty:
        # Assume clf_window_preds_norm is passed or we compute on normal rows inside df_normals
        # Wait: if df_normals is separate, caller passes clf_window_preds_norm via df_normals["_clf_wpred"]
        clf_w_norm = df_normals["_clf_wpred"].to_numpy()
        _, clf_norm_stats = A.apply_persistence_df(df_normals, clf_w_norm, k=k, m=m)
        det_raw_norm = detector.flag_windows_raw(df_normals)
        _, det_norm_stats = detector.flag_windows_persisted(df_normals)

        # Rule 1 (OR): window is non-normal if clf_w_norm != 'normal' or det_raw_norm is True
        # Also evaluate episode union on persisted streams:
        clf_pers_norm, _ = A.apply_persistence_df(df_normals, clf_w_norm, k=k, m=m)
        det_pers_norm, _ = detector.flag_windows_persisted(df_normals)

        # Rule 2 (Gated): detector flags AND classifier window prediction == 'normal'
        gated_raw_norm = np.where(det_raw_norm & (clf_w_norm == "normal"), UNKNOWN_FLAG_LABEL, "normal")
        _, gated_norm_stats = A.apply_persistence_df(df_normals, gated_raw_norm, k=k, m=m)

        # Combined OR episode count per normal run:
        # A normal run has clf_episodes + gated_episodes (or episodes of the combined alarm stream)
        d_n = df_normals[["simulation_id", "t_start"]].copy()
        d_n["_or_active"] = (clf_pers_norm != "normal") | det_pers_norm
        or_eps_per_run = []
        clf_eps_per_run = []
        det_eps_per_run = []
        gated_eps_per_run = []
        for sid, g in d_n.groupby("simulation_id", sort=False):
            sid_s = str(sid)
            g = g.sort_values("t_start", kind="stable")
            act = g["_or_active"].to_numpy(dtype=bool)
            # Count distinct contiguous active episodes in the union stream
            n_union_eps = int(np.sum(act & ~np.concatenate([[False], act[:-1]])))
            c_e = clf_norm_stats[sid_s]["n_alarm_episodes"]
            d_e = det_norm_stats[sid_s]["n_alarm_episodes"]
            g_e = gated_norm_stats[sid_s]["n_alarm_episodes"]
            # Union episode count is at least c_e (never fewer episodes than classifier alone)
            or_eps_per_run.append(float(max(c_e, n_union_eps)))
            clf_eps_per_run.append(float(c_e))
            det_eps_per_run.append(float(d_e))
            gated_eps_per_run.append(float(g_e))

        clf_eps_arr = np.array(clf_eps_per_run)
        det_eps_arr = np.array(det_eps_per_run)
        or_eps_arr = np.array(or_eps_per_run)
        gated_eps_arr = np.array(gated_eps_per_run)
        extra_or_arr = or_eps_arr - clf_eps_arr
        total_with_gated_arr = clf_eps_arr + gated_eps_arr

        rng_n = np.random.default_rng(np.random.SeedSequence([int(seed), 1408]))
        clf_fa, _ = _boot_mean(clf_eps_arr, n_boot, rng_n)
        det_fa, _ = _boot_mean(det_eps_arr, n_boot, rng_n)
        or_fa, (or_lo, or_hi) = _boot_mean(or_eps_arr, n_boot, rng_n)
        ext_or_fa, _ = _boot_mean(extra_or_arr, n_boot, rng_n)
        gated_fa, _ = _boot_mean(gated_eps_arr, n_boot, rng_n)
        tot_gated_fa, (tg_lo, tg_hi) = _boot_mean(total_with_gated_arr, n_boot, rng_n)
    else:
        clf_fa = det_fa = or_fa = or_lo = or_hi = ext_or_fa = gated_fa = tot_gated_fa = tg_lo = tg_hi = np.nan

    rng_e = np.random.default_rng(np.random.SeedSequence([int(seed), 1409]))
    c_ex_pt, _ = _boot_mean(ev_df["clf_exact"].to_numpy(float), n_boot, rng_e)
    d_fl_pt, _ = _boot_mean(ev_df["det_flagged"].to_numpy(float), n_boot, rng_e)
    or_ex_pt, (or_ex_lo, or_ex_hi) = _boot_mean(ev_df["rule_or_exact_or_det"].to_numpy(float), n_boot, rng_e)
    g_ex_pt, (g_ex_lo, g_ex_hi) = _boot_mean(ev_df["rule_gated_exact_or_gated"].to_numpy(float), n_boot, rng_e)

    hA, hB = V2.hours_slow(240.0, 360.0), V2.hours_sensor(240.0)
    return {
        "split_or_arm": split_name,
        "classifier": clf_name,
        "detector": detector.name,
        "n_event_runs": n_ev,
        "clf_exact_detect_rate": round(c_ex_pt, 4),
        "det_only_flag_rate": round(d_fl_pt, 4),
        "n_clf_exact_missed": n_exact_missed,
        "n_missed_flagged_by_det": n_recov_exact_or,
        "frac_of_clf_exact_missed_flagged_by_det": round(n_recov_exact_or / max(n_exact_missed, 1), 4) if n_exact_missed > 0 else np.nan,
        "n_clf_any_missed": n_any_missed,
        "n_any_missed_flagged_by_det": n_recov_any_or,
        "frac_of_clf_any_missed_flagged_by_det": round(n_recov_any_or / max(n_any_missed, 1), 4) if n_any_missed > 0 else np.nan,
        "combined_or_detect_rate": round(or_ex_pt, 4),
        "combined_or_detect_ci_lo": round(or_ex_lo, 4),
        "combined_or_detect_ci_hi": round(or_ex_hi, 4),
        "gated_unknown_detect_rate": round(g_ex_pt, 4),
        "gated_unknown_detect_ci_lo": round(g_ex_lo, 4),
        "gated_unknown_detect_ci_hi": round(g_ex_hi, 4),
        "clf_only_fa_per_normal_run": round(clf_fa, 4) if pd.notna(clf_fa) else np.nan,
        "det_only_fa_per_normal_run": round(det_fa, 4) if pd.notna(det_fa) else np.nan,
        "combined_or_fa_per_normal_run": round(or_fa, 4) if pd.notna(or_fa) else np.nan,
        "combined_or_fa_ci_lo": round(or_lo, 4) if pd.notna(or_lo) else np.nan,
        "combined_or_fa_ci_hi": round(or_hi, 4) if pd.notna(or_hi) else np.nan,
        "extra_fa_per_normal_run_or": round(ext_or_fa, 4) if pd.notna(ext_or_fa) else np.nan,
        "extra_A_slow_fa_per_hour_or": round(ext_or_fa / hA, 5) if pd.notna(ext_or_fa) else np.nan,
        "extra_B_sensor_fa_per_hour_or": round(ext_or_fa / hB, 4) if pd.notna(ext_or_fa) else np.nan,
        "total_A_slow_fa_per_hour_or": round(or_fa / hA, 5) if pd.notna(or_fa) else np.nan,
        "total_B_sensor_fa_per_hour_or": round(or_fa / hB, 4) if pd.notna(or_fa) else np.nan,
        "extra_fa_per_normal_run_gated": round(gated_fa, 4) if pd.notna(gated_fa) else np.nan,
        "extra_A_slow_fa_per_hour_gated": round(gated_fa / hA, 5) if pd.notna(gated_fa) else np.nan,
        "extra_B_sensor_fa_per_hour_gated": round(gated_fa / hB, 4) if pd.notna(gated_fa) else np.nan,
        "total_fa_per_normal_run_with_gated": round(tot_gated_fa, 4) if pd.notna(tot_gated_fa) else np.nan,
        "total_A_slow_fa_per_hour_with_gated": round(tot_gated_fa / hA, 5) if pd.notna(tot_gated_fa) else np.nan,
        "total_B_sensor_fa_per_hour_with_gated": round(tot_gated_fa / hB, 4) if pd.notna(tot_gated_fa) else np.nan,
    }
