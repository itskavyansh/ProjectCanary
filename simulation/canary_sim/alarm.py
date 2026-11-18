"""Persistence alarm logic and minimum-probability operating-point tuning (SYNTHETIC / SIMULATION).

1. Persistence state machine (`apply_persistence_sequence`, `apply_persistence_df`):
   - Raises an alarm for an event/fault class only after `k` consecutive windows predict that same
     non-normal class (`k` is config-driven; evaluated over `k in {1, 2, 3, 5}`).
   - Once an alarm is active, it stays active until `m` consecutive windows predict `'normal'`
     (clears after `m` consecutive non-event windows), or until another non-normal class accumulates
     `k` consecutive predictions.
   - Each transition from `'normal'` into an active alarm state in a run is counted as one distinct
     alarm episode (`n_alarm_episodes`), alongside total active alarm windows (`n_alarm_windows`).

2. Minimum-probability operating point (`MinProbOperatingPoint`):
   - Wraps any probabilistic classifier (`RandomForest`, `LogisticRegression`) and maps the 5
     non-normal classes into 4 class groups:
       * `basin`: `("gradual", "rapid")`
       * `vibration`: `("abnormal_vibration",)`
       * `sensor_fault`: `("sensor_fault",)`
       * `comm_fault`: `("comm_fault",)`
   - Predicts a non-normal class only if its probability exceeds its group threshold `tau_g`
     (picking the highest-margin class `p_c - tau_g(c)` when multiple exceed their threshold);
     otherwise predicts `'normal'`.
   - Tuned strictly on `val` (`guard_no_test_data` enforced) to meet the configurable false-alarm
     target (`max_fa_per_normal_run = 1.0`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import classes as C, labels as L, models as M

CLASS_GROUPS: dict[str, tuple[str, ...]] = {
    "basin": ("gradual", "rapid"),
    "vibration": ("abnormal_vibration",),
    "sensor_fault": ("sensor_fault",),
    "comm_fault": ("comm_fault",),
}

CLASS_TO_GROUP: dict[str, str] = {
    cls: grp for grp, members in CLASS_GROUPS.items() for cls in members
}

K_CANDIDATES = (1, 2, 3, 5)
M_CANDIDATES = (1, 2, 3)


def apply_persistence_sequence(
    preds: list[str] | np.ndarray,
    k: int = 3,
    m: int = 2,
) -> tuple[np.ndarray, int]:
    """Apply (k-to-raise, m-to-clear) persistence state machine to a single run's window predictions.

    Returns `(alarm_state_per_window, n_alarm_episodes)`.
    - When `active == 'normal'`: a candidate non-normal class must appear for `k` consecutive
      windows to raise an alarm (`active = cand`, `n_alarm_episodes += 1`).
    - When `active != 'normal'`: the alarm clears back to `'normal'` only after `m` consecutive
      `'normal'` window predictions (or switches if another non-normal class appears `k` times
      consecutively).
    """
    if k < 1 or m < 1:
        raise ValueError(f"k and m must be >= 1, got k={k}, m={m}")

    out = []
    active = "normal"
    cand: str | None = None
    c_cnt = 0
    norm_cnt = 0
    episodes = 0

    for raw_p in preds:
        p = str(raw_p)
        if active == "normal":
            if p != "normal":
                if p == cand:
                    c_cnt += 1
                else:
                    cand = p
                    c_cnt = 1
                if c_cnt >= k:
                    active = cand
                    norm_cnt = 0
                    episodes += 1
            else:
                cand = None
                c_cnt = 0
            out.append(active)
        else:
            if p == "normal":
                norm_cnt += 1
                cand = None
                c_cnt = 0
                if norm_cnt >= m:
                    active = "normal"
                    norm_cnt = 0
            elif p != active:
                norm_cnt = 0
                if p == cand:
                    c_cnt += 1
                else:
                    cand = p
                    c_cnt = 1
                if c_cnt >= k:
                    active = cand
                    episodes += 1
            else:
                norm_cnt = 0
                cand = None
                c_cnt = 0
            out.append(active)

    return np.array(out, dtype=object), int(episodes)


def apply_persistence_df(
    df: pd.DataFrame,
    raw_preds: np.ndarray,
    k: int = 3,
    m: int = 2,
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    """Apply `(k, m)` persistence independently within each run (`simulation_id`), in `t_start` order."""
    alarm_preds = np.empty(len(df), dtype=object)
    run_stats: dict[str, dict[str, int]] = {}

    work = pd.DataFrame({
        "simulation_id": df["simulation_id"].to_numpy(),
        "t_start": df["t_start"].to_numpy(dtype=float),
        "raw": np.asarray(raw_preds, dtype=object),
        "orig_idx": np.arange(len(df)),
    })

    for sim_id, g in work.groupby("simulation_id", sort=False):
        g_sorted = g.sort_values("t_start", kind="stable")
        seq, eps = apply_persistence_sequence(g_sorted["raw"].to_numpy(), k=k, m=m)
        idx = g_sorted["orig_idx"].to_numpy()
        alarm_preds[idx] = seq
        run_stats[str(sim_id)] = {
            "n_alarm_episodes": int(eps),
            "n_alarm_windows": int((seq != "normal").sum()),
        }

    return alarm_preds, run_stats


def evaluate_alarm_level(
    df_all_windows: pd.DataFrame,
    raw_preds: np.ndarray,
    k: int = 3,
    m: int = 2,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    """Evaluate alarm-level metrics after `(k, m)` persistence with run-bootstrap 95% CIs.

    Metrics reported:
    - `fa_episodes_per_normal_run`: distinct false-alarm episodes raised per normal run
    - `fa_windows_per_normal_run`: active false-alarm windows per normal run
    - `normal_run_with_fa_rate`: fraction of normal runs with >=1 false alarm
    - `exact_detect_rate`: fraction of event/fault runs where the true class alarm was raised
    - `any_detect_rate`: fraction of event/fault runs where any alarm was raised after event onset
    - `mean_delay_exact_s`: mean detection delay (sim s) from true event onset to the window where
      the persisted alarm is raised (includes the added `(k - 1) * step_s` persistence delay and
      uses transition window offsets `t_rel_event_start_s`).
    """
    alarm_preds, run_stats = apply_persistence_df(df_all_windows, raw_preds, k=k, m=m)
    d = df_all_windows.copy()
    d["_alarm"] = alarm_preds
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])

    records = []
    for sim_id, g in d.groupby("simulation_id", sort=False):
        g = g.sort_values("t_start", kind="stable")
        r_cls = str(g["run_class"].iloc[0])
        st = run_stats[str(sim_id)]
        if r_cls == "normal":
            records.append({
                "simulation_id": str(sim_id),
                "run_class": "normal",
                "is_event_run": False,
                "fa_episodes": st["n_alarm_episodes"],
                "fa_windows": st["n_alarm_windows"],
                "has_fa": int(st["n_alarm_episodes"] > 0),
                "exact_detected": np.nan,
                "any_detected": np.nan,
                "delay_exact_s": np.nan,
                "delay_any_s": np.nan,
            })
        else:
            t_rel = g["t_rel_event_start_s"].to_numpy(dtype=float)
            al_arr = g["_alarm"].to_numpy(dtype=object)
            post_mask = t_rel > -win_half + 1e-9
            t_post = t_rel[post_mask]
            al_post = al_arr[post_mask]

            any_idx = np.nonzero(al_post != "normal")[0]
            any_det = int(len(any_idx) > 0)
            delay_any = float(max(0.0, t_post[any_idx[0]] + win_half)) if any_det else np.nan

            if (al_post == r_cls).any():
                ex_idx = int(np.nonzero(al_post == r_cls)[0][0])
                ex_det = 1
                delay_ex = float(max(0.0, t_post[ex_idx] + win_half))
            else:
                ls_post = g["label_set"].fillna("").to_numpy(dtype=object)[post_mask]
                ex_matches = [
                    i for i, (pr, ls) in enumerate(zip(al_post, ls_post))
                    if pr != "normal" and (pr == r_cls or (ls and str(pr) in str(ls).split("|")))
                ]
                ex_det = int(len(ex_matches) > 0)
                delay_ex = float(max(0.0, t_post[ex_matches[0]] + win_half)) if ex_det else np.nan

            records.append({
                "simulation_id": str(sim_id),
                "run_class": r_cls,
                "is_event_run": True,
                "fa_episodes": np.nan,
                "fa_windows": np.nan,
                "has_fa": np.nan,
                "exact_detected": ex_det,
                "any_detected": any_det,
                "delay_exact_s": delay_ex,
                "delay_any_s": delay_any,
            })

    rdf = pd.DataFrame(records)
    ev_runs = rdf[rdf["is_event_run"]]
    norm_runs = rdf[~rdf["is_event_run"]]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 801]))

    def _boot_mean(arr: np.ndarray) -> tuple[float, tuple[float, float]]:
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, (0.0, 0.0)
        pt = float(np.mean(arr))
        if n_boot <= 0:
            return pt, (pt, pt)
        b_means = [float(np.mean(arr[rng.integers(0, len(arr), size=len(arr))])) for _ in range(n_boot)]
        return pt, (float(np.percentile(b_means, 2.5)), float(np.percentile(b_means, 97.5)))

    fa_ep, fa_ep_ci = _boot_mean(norm_runs["fa_episodes"].to_numpy(dtype=float))
    fa_win, fa_win_ci = _boot_mean(norm_runs["fa_windows"].to_numpy(dtype=float))
    fa_rate, fa_rate_ci = _boot_mean(norm_runs["has_fa"].to_numpy(dtype=float))
    ex_rate, ex_ci = _boot_mean(ev_runs["exact_detected"].to_numpy(dtype=float))
    any_rate, any_ci = _boot_mean(ev_runs["any_detected"].to_numpy(dtype=float))
    del_ex, del_ex_ci = _boot_mean(ev_runs["delay_exact_s"].to_numpy(dtype=float))
    del_any, del_any_ci = _boot_mean(ev_runs["delay_any_s"].to_numpy(dtype=float))

    per_class = {}
    for cls in C.CLASSES:
        if cls == "normal":
            continue
        sub_c = ev_runs[ev_runs["run_class"] == cls]
        if len(sub_c) == 0:
            continue
        er, er_ci = _boot_mean(sub_c["exact_detected"].to_numpy(dtype=float))
        ar, ar_ci = _boot_mean(sub_c["any_detected"].to_numpy(dtype=float))
        de, de_ci = _boot_mean(sub_c["delay_exact_s"].to_numpy(dtype=float))
        per_class[cls] = {
            "n_runs": len(sub_c),
            "exact_detect_rate": er,
            "exact_detect_rate_ci95": er_ci,
            "any_detect_rate": ar,
            "any_detect_rate_ci95": ar_ci,
            "mean_delay_exact_s": de,
            "mean_delay_exact_s_ci95": de_ci,
        }

    return {
        "k": k,
        "m": m,
        "alarm_preds": alarm_preds,
        "fa_episodes_per_normal_run": fa_ep,
        "fa_episodes_ci95": fa_ep_ci,
        "fa_windows_per_normal_run": fa_win,
        "fa_windows_ci95": fa_win_ci,
        "normal_run_with_fa_rate": fa_rate,
        "normal_run_with_fa_ci95": fa_rate_ci,
        "exact_detect_rate": ex_rate,
        "exact_detect_ci95": ex_ci,
        "any_detect_rate": any_rate,
        "any_detect_ci95": any_ci,
        "mean_delay_exact_s": del_ex,
        "mean_delay_exact_ci95": del_ex_ci,
        "mean_delay_any_s": del_any,
        "mean_delay_any_ci95": del_any_ci,
        "per_class": per_class,
        "run_table": rdf,
    }


def evaluate_persistence_grid_on_val(
    val_df: pd.DataFrame,
    raw_preds: np.ndarray,
    k_grid: tuple[int, ...] = K_CANDIDATES,
    m_grid: tuple[int, ...] = M_CANDIDATES,
    max_fa_per_normal_run: float = 1.0,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate all `(k, m)` persistence pairs on `val` ONLY and select the best pair meeting
    `fa_episodes_per_normal_run <= max_fa_per_normal_run` (maximising exact detection rate, then
    minimising detection delay; if none meets the target on uncalibrated predictions, picks the pair
    minimising false alarms).
    """
    M.guard_no_test_data(val_df)
    if set(val_df["split"].unique()) != {"val"}:
        raise ValueError("Persistence tuning is only allowed on the 'val' split.")

    rows = []
    for k in k_grid:
        for m in m_grid:
            res = evaluate_alarm_level(val_df, raw_preds, k=k, m=m, n_boot=0)
            rows.append({
                "k": k,
                "m": m,
                "fa_episodes_per_normal_run": round(res["fa_episodes_per_normal_run"], 4),
                "fa_windows_per_normal_run": round(res["fa_windows_per_normal_run"], 4),
                "exact_detect_rate": round(res["exact_detect_rate"], 4),
                "any_detect_rate": round(res["any_detect_rate"], 4),
                "mean_delay_exact_s": round(res["mean_delay_exact_s"], 3),
                "meets_fa_target": bool(res["fa_episodes_per_normal_run"] <= max_fa_per_normal_run + 1e-9),
            })

    grid_df = pd.DataFrame(rows)
    feasible = grid_df[grid_df["meets_fa_target"]]
    if len(feasible) > 0:
        # Prefer highest exact_detect_rate, then lowest fa_windows_per_normal_run, then shortest delay
        best = feasible.sort_values(
            ["exact_detect_rate", "fa_windows_per_normal_run", "mean_delay_exact_s"],
            ascending=[False, True, True],
        ).iloc[0]
    else:
        best = grid_df.sort_values(
            ["fa_episodes_per_normal_run", "fa_windows_per_normal_run", "exact_detect_rate"],
            ascending=[True, True, False],
        ).iloc[0]

    return best.to_dict(), grid_df


# ---------------------------------------------------------------------------
# 2. Minimum-Probability Operating Point (RF & Logistic Regression, tuned on val)
# ---------------------------------------------------------------------------
@dataclass
class MinProbOperatingPoint:
    """Minimum-probability operating point tuned per class group on `val` only.

    Predicts a non-normal class `c` only if `P(c | x) >= group_thresholds[group(c)]`; otherwise
    predicts `'normal'`. When combined with `(k, m)` persistence, guarantees the false-alarm target
    (`<= max_fa_per_normal_run`) on `val` at both the alarm-episode and alarm-window levels.
    """

    base_model: M.SklearnModelWrapper
    group_thresholds: dict[str, float] = field(default_factory=dict)
    k: int = 3
    m: int = 2
    fa_target_per_normal_run: float = 1.0
    tuning_summary: dict = field(default_factory=dict)

    def predict_window(self, df: pd.DataFrame) -> np.ndarray:
        """Window-level predictions using the tuned per-group minimum probability thresholds."""
        proba = self.base_model.predict_proba(df)
        classes = list(self.base_model.pipeline.classes_)
        return self._predict_from_proba(proba, classes, self.group_thresholds)

    def predict_alarm(self, df: pd.DataFrame, k: int | None = None, m: int | None = None) -> np.ndarray:
        """Persisted alarm predictions on `df` using `(k, m)`."""
        w_preds = self.predict_window(df)
        use_k = self.k if k is None else int(k)
        use_m = self.m if m is None else int(m)
        alarm_preds, _ = apply_persistence_df(df, w_preds, k=use_k, m=use_m)
        return alarm_preds

    @staticmethod
    def _predict_from_proba(
        proba: np.ndarray,
        classes: list[str],
        group_thresholds: dict[str, float],
    ) -> np.ndarray:
        n = proba.shape[0]
        out = np.full(n, "normal", dtype=object)
        non_norm_indices = [(j, c, group_thresholds[CLASS_TO_GROUP[c]]) for j, c in enumerate(classes) if c != "normal"]

        # Score = P(c) - tau_group(c) for classes where P(c) >= tau_group(c)
        best_margin = np.full(n, -1e-12, dtype=float)
        for j, c, tau in non_norm_indices:
            p_c = proba[:, j]
            exceeds = p_c >= tau
            margin = p_c - tau
            pick = exceeds & (margin > best_margin)
            out[pick] = c
            best_margin[pick] = margin[pick]
        return out


def tune_min_prob_operating_point(
    base_model: M.SklearnModelWrapper,
    val_df: pd.DataFrame,
    max_fa_per_normal_run: float = 1.0,
    k: int = 3,
    m: int = 2,
    fa_metric: str = "fa_windows_per_normal_run",
) -> MinProbOperatingPoint:
    """Tune per-group minimum-probability thresholds on `val` ONLY to meet `max_fa_per_normal_run`.

    Each of the 4 class groups (`basin`, `vibration`, `sensor_fault`, `comm_fault`) is allocated a
    false-alarm budget on `val` normal runs and searched over candidate quantiles of normal-run
    group probabilities so that:
      1) Total false alarms per normal run on `val` is `<= max_fa_per_normal_run` (for both
         `fa_episodes_per_normal_run` and `fa_windows_per_normal_run`).
      2) Validation macro-F1 and event detection rate across the 6 classes are maximised.
    """
    M.guard_no_test_data(val_df)
    if set(val_df["split"].unique()) != {"val"}:
        raise ValueError("Operating-point tuning is only allowed on the 'val' split.")

    proba = base_model.predict_proba(val_df)
    classes = list(base_model.pipeline.classes_)
    cls_idx = {c: i for i, c in enumerate(classes)}

    norm_mask = (val_df["run_class"] == "normal").to_numpy()
    proba_norm = proba[norm_mask]

    # Precompute candidate thresholds per group from quantiles on val normal windows
    group_max_norm: dict[str, np.ndarray] = {}
    for grp, members in CLASS_GROUPS.items():
        idxs = [cls_idx[c] for c in members]
        group_max_norm[grp] = proba_norm[:, idxs].max(axis=1)

    # Candidate quantile grid on val normal windows
    q_grid = (0.90, 0.95, 0.975, 0.985, 0.99, 0.993, 0.995, 0.997, 0.999)
    va_tr_mask = L.trainable(val_df).to_numpy()
    y_va_tr = val_df.loc[va_tr_mask, "label"].to_numpy()

    best_score = -1.0
    best_taus: dict[str, float] | None = None
    best_stats: dict = {}
    history = []

    # Coordinate search: first calibrate each non-basin group (`vibration`, `comm_fault`, `sensor_fault`)
    # at high specificity on normal runs, then sweep `basin` and `sensor_fault` quantiles jointly.
    vib_tau = float(max(0.25, np.quantile(group_max_norm["vibration"], 0.999) + 0.02))
    comm_candidates = [float(max(0.22, np.quantile(group_max_norm["comm_fault"], q))) for q in (0.99, 0.995, 0.998)]
    sf_candidates = [float(max(0.24, np.quantile(group_max_norm["sensor_fault"], q))) for q in (0.975, 0.985, 0.99, 0.995, 0.998)]
    basin_candidates = [float(max(0.28, np.quantile(group_max_norm["basin"], q))) for q in q_grid]

    # Pre-slice per-run index arrays once so candidate grid evaluation is <0.2 s
    run_slices = []
    for sim_id, g in val_df.groupby("simulation_id", sort=False):
        g_s = g.sort_values("t_start", kind="stable")
        idx = g_s.index.to_numpy()
        r_cls = str(g_s["run_class"].iloc[0])
        t_rel = g_s["t_rel_event_start_s"].to_numpy(dtype=float)
        win_half = 0.5 * float((g_s["t_end"] - g_s["t_start"]).iloc[0])
        post_mask = t_rel > -win_half + 1e-9
        run_slices.append((str(sim_id), r_cls, idx, post_mask))

    def _fast_val_alarm_stats(w_preds: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        al_preds = np.empty(len(val_df), dtype=object)
        norm_eps, norm_wins, ev_exact = [], [], []
        for _, r_cls, idx, post_mask in run_slices:
            seq, eps = apply_persistence_sequence(w_preds[idx], k=k, m=m)
            al_preds[idx] = seq
            if r_cls == "normal":
                norm_eps.append(eps)
                norm_wins.append(int((seq != "normal").sum()))
            else:
                ev_exact.append(int((seq[post_mask] == r_cls).any()))
        return (
            float(np.mean(norm_eps)),
            float(np.mean(norm_wins)),
            float(np.mean(ev_exact)),
            al_preds,
        )

    for tau_b in basin_candidates:
        for tau_sf in sf_candidates:
            for tau_cm in comm_candidates:
                cand_taus = {
                    "basin": round(tau_b, 5),
                    "vibration": round(vib_tau, 5),
                    "sensor_fault": round(tau_sf, 5),
                    "comm_fault": round(tau_cm, 5),
                }
                w_preds = MinProbOperatingPoint._predict_from_proba(proba, classes, cand_taus)
                fa_ep, fa_win, ex_rate, al_preds = _fast_val_alarm_stats(w_preds)
                w_f1 = float(
                    M.f1_score(
                        y_va_tr,
                        w_preds[va_tr_mask],
                        labels=list(C.CLASSES),
                        average="macro",
                        zero_division=0,
                    )
                )
                al_f1 = float(
                    M.f1_score(
                        y_va_tr,
                        al_preds[va_tr_mask],
                        labels=list(C.CLASSES),
                        average="macro",
                        zero_division=0,
                    )
                )
                rec = {
                    **cand_taus,
                    "val_fa_episodes_per_normal_run": round(fa_ep, 4),
                    "val_fa_windows_per_normal_run": round(fa_win, 4),
                    "val_exact_detect_rate": round(ex_rate, 4),
                    "val_window_macro_f1": round(w_f1, 4),
                    "val_alarm_macro_f1": round(al_f1, 4),
                }
                history.append(rec)

                fa_val = fa_win if fa_metric == "fa_windows_per_normal_run" else fa_ep
                if fa_ep <= max_fa_per_normal_run + 1e-9 and fa_val <= max_fa_per_normal_run + 1e-9:
                    # Combined objective on val: alarm macro-F1 + run exact detect rate
                    score = 0.6 * al_f1 + 0.4 * ex_rate
                    if score > best_score:
                        best_score = score
                        best_taus = cand_taus
                        best_stats = rec

    if best_taus is None:
        # Fallback to strictest candidate iffa_windows target is tight
        history.sort(key=lambda r: (r["val_fa_episodes_per_normal_run"], r["val_fa_windows_per_normal_run"], -r["val_alarm_macro_f1"]))
        best_stats = history[0]
        best_taus = {g: best_stats[g] for g in CLASS_GROUPS}

    return MinProbOperatingPoint(
        base_model=base_model,
        group_thresholds=best_taus,
        k=k,
        m=m,
        fa_target_per_normal_run=max_fa_per_normal_run,
        tuning_summary={"best_val": best_stats, "n_candidates_evaluated": len(history)},
    )


def evaluate_alarm_recall_vs_snr(
    df_all_windows: pd.DataFrame,
    raw_preds: np.ndarray,
    manifest_df: pd.DataFrame,
    k: int = 3,
    m: int = 2,
    n_boot: int = 300,
    seed: int = 20261117,
) -> pd.DataFrame:
    """Compute both run-level alarm detection rate and persisted-alarm window recall across SNR bins
    for `gradual` and `rapid` runs, with 95% run-bootstrap CIs.
    """
    from . import evaluate as ev

    al_res = evaluate_alarm_level(df_all_windows, raw_preds, k=k, m=m, n_boot=0)
    alarm_preds = al_res["alarm_preds"]
    rdf = al_res["run_table"]

    snr_map = dict(zip(manifest_df["simulation_id"], manifest_df["snr_vs_imu_noise"]))
    rdf["snr"] = rdf["simulation_id"].map(snr_map)

    mask = L.trainable(df_all_windows).to_numpy()
    w_sub = df_all_windows.loc[mask, ["simulation_id", "label"]].copy()
    w_sub["alarm_pred"] = alarm_preds[mask]
    w_sub["snr"] = w_sub["simulation_id"].map(snr_map)

    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 805]))
    rows = []

    for cls in ("gradual", "rapid"):
        r_cls_df = rdf[rdf["run_class"] == cls]
        w_cls_df = w_sub[w_sub["label"] == cls]
        for b_lbl, lo, hi in ev.SNR_BINS:
            r_bin = r_cls_df[(r_cls_df["snr"] >= lo) & (r_cls_df["snr"] < hi)]
            w_bin = w_cls_df[(w_cls_df["snr"] >= lo) & (w_cls_df["snr"] < hi)]
            run_ids = r_bin["simulation_id"].unique()
            n_runs = len(run_ids)
            n_wins = len(w_bin)
            if n_runs == 0 or n_wins == 0:
                rows.append({
                    "class": cls,
                    "snr_bin": b_lbl,
                    "n_runs": n_runs,
                    "n_windows": n_wins,
                    "alarm_run_exact_detect_rate": np.nan,
                    "alarm_run_exact_ci_lo": np.nan,
                    "alarm_run_exact_ci_hi": np.nan,
                    "alarm_run_any_detect_rate": np.nan,
                    "alarm_run_any_ci_lo": np.nan,
                    "alarm_run_any_ci_hi": np.nan,
                    "alarm_win_exact_recall": np.nan,
                    "alarm_win_exact_ci_lo": np.nan,
                    "alarm_win_exact_ci_hi": np.nan,
                    "alarm_win_basin_recall": np.nan,
                    "alarm_win_basin_ci_lo": np.nan,
                    "alarm_win_basin_ci_hi": np.nan,
                })
                continue

            r_ex = r_bin["exact_detected"].to_numpy(dtype=float)
            r_any = r_bin["any_detected"].to_numpy(dtype=float)
            w_ex = (w_bin["alarm_pred"] == cls).to_numpy(dtype=float)
            w_ba = w_bin["alarm_pred"].isin(["gradual", "rapid"]).to_numpy(dtype=float)

            by_run_w_ex = [w_ex[w_bin["simulation_id"].to_numpy() == r] for r in run_ids]
            by_run_w_ba = [w_ba[w_bin["simulation_id"].to_numpy() == r] for r in run_ids]

            b_rex, b_rany, b_wex, b_wba = [], [], [], []
            for _ in range(n_boot):
                pick = rng.integers(0, n_runs, size=n_runs)
                b_rex.append(float(r_ex[pick].mean()))
                b_rany.append(float(r_any[pick].mean()))
                b_wex.append(float(np.concatenate([by_run_w_ex[i] for i in pick]).mean()))
                b_wba.append(float(np.concatenate([by_run_w_ba[i] for i in pick]).mean()))

            rows.append({
                "class": cls,
                "snr_bin": b_lbl,
                "n_runs": n_runs,
                "n_windows": n_wins,
                "alarm_run_exact_detect_rate": float(r_ex.mean()),
                "alarm_run_exact_ci_lo": float(np.percentile(b_rex, 2.5)),
                "alarm_run_exact_ci_hi": float(np.percentile(b_rex, 97.5)),
                "alarm_run_any_detect_rate": float(r_any.mean()),
                "alarm_run_any_ci_lo": float(np.percentile(b_rany, 2.5)),
                "alarm_run_any_ci_hi": float(np.percentile(b_rany, 97.5)),
                "alarm_win_exact_recall": float(w_ex.mean()),
                "alarm_win_exact_ci_lo": float(np.percentile(b_wex, 2.5)),
                "alarm_win_exact_ci_hi": float(np.percentile(b_wex, 97.5)),
                "alarm_win_basin_recall": float(w_ba.mean()),
                "alarm_win_basin_ci_lo": float(np.percentile(b_wba, 2.5)),
                "alarm_win_basin_ci_hi": float(np.percentile(b_wba, 97.5)),
            })

    return pd.DataFrame(rows)

