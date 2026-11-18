"""Evaluation extras v2 (SYNTHETIC / SIMULATION). Patches the presentation of eval_extras; NO model is fitted,
tuned or modified here. Frozen v3 models / operating points are only used to predict.

Adds:
  * false alarms per hour under two definitions (A: slow-process time, B: sensor time)
  * accuracy tables that show N/A (never 0.00) for classes with zero true windows
  * gradual / rapid window recall together with run-level detection, also split by snr_vs_imu_noise bin
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import alarm as A, eval_extras as EE, labels as L

CLASS_LIST = EE.CLASS_LIST
LABEL_A = "slow-process time (ground-movement timescale)"
LABEL_B = "sensor time (noise, vibration and window timescale)"
SNR_BINS = (("<1", 0.0, 1.0), ("1-2", 1.0, 2.0), ("2-5", 2.0, 5.0), (">5", 5.0, float("inf")))


# ------------------------------------------------------------------ regimes (same definitions as eval_extras)
def regimes_for(mname: str, model: Any, ops: dict, df: pd.DataFrame) -> list[tuple[str, np.ndarray, int, int]]:
    raw = model.predict(df)
    thr = mname == "threshold_rule"
    return [
        ("raw (k=1)", raw, 1, 1),
        ("episode_fa<=1 (k=3,m=3)", raw if thr else ops["episode_fa_le_1"][mname].predict_window(df), 3, 3),
        ("strict_window_fa<=1", raw if thr else ops["strict_window_fa_le_1"][mname].predict_window(df),
         5 if thr else 3, 2),
    ]


# ------------------------------------------------------------------ 1. two false-alarm time definitions
def hours_slow(duration_s: float, compression_factor: float) -> float:
    """Definition A: compressed physical duration = duration_s * compression_factor / 3600."""
    return EE.calculate_simulated_hours_per_run(duration_s, compression_factor)


def hours_sensor(duration_s: float) -> float:
    """Definition B: actual sensor time of the run = duration_s / 3600."""
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    return float(duration_s / 3600.0)


def rate_per_hour(count_per_run: float, hours: float) -> float:
    return float(count_per_run / hours)


def every_n_hours(rate_per_h: float) -> float:
    """'One false alarm every N hours'; inf when no false alarm was observed."""
    return float("inf") if rate_per_h <= 0 else float(1.0 / rate_per_h)


def fmt_every(n: float) -> str:
    if not np.isfinite(n):
        return "none observed"
    return f"{n * 60:.1f} min" if n < 1.0 else f"{n:.1f} h"


def normal_run_hours(manifest: pd.DataFrame, split: str) -> tuple[float, float, float, float]:
    """(duration_s, compression_factor, hours_A, hours_B) of the normal runs of a split. Raises when the
    normal runs use more than one duration/compression factor (they must then be reported per factor)."""
    m = manifest[(manifest["split"] == split) & (manifest["run_class"] == "normal")]
    if m.empty:
        raise ValueError(f"split {split} has no normal runs")
    durs, comps = m["duration_s"].unique(), m["compression_factor"].unique()
    if len(durs) != 1 or len(comps) != 1:
        raise ValueError(f"split {split}: normal runs use several (duration, compression) pairs "
                         f"{sorted(set(zip(m.duration_s, m.compression_factor)))}; report per factor")
    d, c = float(durs[0]), float(comps[0])
    return d, c, hours_slow(d, c), hours_sensor(d)


def fa_rows_for_split(sp: str, df: pd.DataFrame, regs: dict, manifest: pd.DataFrame, n_boot: int, seed: int) -> list[dict]:
    """regs[model_name] = regimes_for(...) list. Returns one row per model x regime with both definitions."""
    dur, comp, hA, hB = normal_run_hours(manifest, sp)
    rows = []
    for mname, regimes in regs.items():
        for rname, w_preds, k, m in regimes:
            r = EE.evaluate_alarm_level_with_hours(df, w_preds, manifest_df=manifest, k=k, m=m, n_boot=n_boot, seed=seed)
            e, elo, ehi = r["fa_episodes_per_run"], *r["fa_episodes_per_run_ci95"]
            w, wlo, whi = r["fa_windows_per_run"], *r["fa_windows_per_run_ci95"]
            rate_a, rate_b = rate_per_hour(e, hA), rate_per_hour(e, hB)
            rows.append({
                "split": sp, "model": mname, "regime": rname, "k": k, "m": m,
                "n_normal_runs": r["n_normal_runs"], "n_event_runs": r["n_event_runs"],
                "sensor_duration_s": dur, "compression_factor": comp,
                "hours_per_run_A_slow": round(hA, 5), "hours_per_run_B_sensor": round(hB, 5),
                "fa_episodes_per_run": round(e, 4), "fa_episodes_per_run_ci_lo": round(elo, 4),
                "fa_episodes_per_run_ci_hi": round(ehi, 4),
                "A_slow_fa_episodes_per_hour": round(rate_a, 5),
                "A_slow_ci_lo": round(rate_per_hour(elo, hA), 5), "A_slow_ci_hi": round(rate_per_hour(ehi, hA), 5),
                "A_slow_one_fa_every_hours": every_n_hours(rate_a),
                "B_sensor_fa_episodes_per_hour": round(rate_b, 4),
                "B_sensor_ci_lo": round(rate_per_hour(elo, hB), 4), "B_sensor_ci_hi": round(rate_per_hour(ehi, hB), 4),
                "B_sensor_one_fa_every_hours": every_n_hours(rate_b),
                "fa_windows_per_run": round(w, 4), "fa_windows_per_run_ci_lo": round(wlo, 4),
                "fa_windows_per_run_ci_hi": round(whi, 4),
                "A_slow_fa_windows_per_hour": round(rate_per_hour(w, hA), 5),
                "B_sensor_fa_windows_per_hour": round(rate_per_hour(w, hB), 4),
                "exact_detect_rate": round(r["exact_detect_rate"], 4),
                "exact_detect_ci_lo": round(r["exact_detect_ci95"][0], 4),
                "exact_detect_ci_hi": round(r["exact_detect_ci95"][1], 4),
                "missed_event_rate": round(r["missed_event_rate"], 4),
                "missed_event_ci_lo": round(r["missed_event_ci95"][0], 4),
                "missed_event_ci_hi": round(r["missed_event_ci95"][1], 4),
            })
    return rows


# ------------------------------------------------------------------ 2. accuracy rows with N/A for empty classes
def accuracy_row(split: str, model: str, regime: str, res: dict) -> dict:
    """Flatten evaluate_window_level_accuracy_and_f1 output. Classes with zero true windows get NaN (shown N/A)."""
    row = {"split": split, "model": model, "regime": regime, "n_windows": res["n_windows"], "n_runs": res["n_runs"],
           "window_accuracy": round(res["accuracy"], 4),
           "window_accuracy_ci_lo": round(res["accuracy_ci95"][0], 4),
           "window_accuracy_ci_hi": round(res["accuracy_ci95"][1], 4),
           "window_macro_f1": round(res["macro_f1"], 4),
           "window_macro_f1_ci_lo": round(res["macro_f1_ci95"][0], 4),
           "window_macro_f1_ci_hi": round(res["macro_f1_ci95"][1], 4)}
    for c in CLASS_LIST:
        pc = res["per_class"][c]
        empty = pc["support"] == 0
        for key, src in (("prec", "precision"), ("rec", "recall")):
            row[f"{key}_{c}"] = np.nan if empty else round(pc[src], 4)
            row[f"{key}_{c}_ci_lo"] = np.nan if empty else round(pc[f"{src}_ci95"][0], 4)
            row[f"{key}_{c}_ci_hi"] = np.nan if empty else round(pc[f"{src}_ci95"][1], 4)
        row[f"supp_{c}"] = pc["support"]
    return row


def fmt_pr(prec: float, rec: float) -> str:
    f = lambda v: "N/A" if pd.isna(v) else f"{v:.2f}"
    return f"{f(prec)} / {f(rec)}"


def split_class_shares(df: pd.DataFrame) -> dict[str, float]:
    """Share of each label among trainable (non-transition) windows."""
    sub = df[L.trainable(df)]
    return (sub["label"].value_counts(normalize=True)).to_dict()


# ------------------------------------------------------------------ 3. gradual / rapid recall + run-level detection
def run_detection_records(df: pd.DataFrame, w_preds: np.ndarray, k: int, m: int) -> pd.DataFrame:
    """Per event run: was an alarm of the run's OWN class raised after onset (after (k,m) persistence)?
    Strict own-class rule (stress runs: an alarm of a different co-occurring class does NOT count)."""
    alarm_preds, _ = A.apply_persistence_df(df, w_preds, k=k, m=m)
    d = df[["simulation_id", "run_class", "t_start", "t_end", "t_rel_event_start_s"]].copy()
    d["_alarm"] = alarm_preds
    win_half = 0.5 * float((d["t_end"] - d["t_start"]).iloc[0])
    recs = []
    for sid, g in d.groupby("simulation_id", sort=False):
        r_cls = str(g["run_class"].iloc[0])
        if r_cls == "normal":
            continue
        g = g.sort_values("t_start", kind="stable")
        post = g["t_rel_event_start_s"].to_numpy(dtype=float) > -win_half + 1e-9
        al = g["_alarm"].to_numpy(dtype=object)[post]
        recs.append({"simulation_id": str(sid), "run_class": r_cls, "detected": float((al == r_cls).any())})
    return pd.DataFrame(recs)


def per_run_class_table(df: pd.DataFrame, w_preds: np.ndarray, det: pd.DataFrame, cls: str) -> pd.DataFrame:
    """One row per run of class `cls`: window hits/total of `cls` (window level, w_preds) and run-level detection."""
    mask = L.trainable(df).to_numpy() & (df["label"].to_numpy() == cls)
    w = pd.DataFrame({"simulation_id": df["simulation_id"].to_numpy()[mask],
                      "hit": (np.asarray(w_preds)[mask] == cls).astype(int)})
    g = w.groupby("simulation_id").agg(win_tp=("hit", "sum"), win_n=("hit", "size")).reset_index()
    runs = det[det["run_class"] == cls][["simulation_id", "detected"]]
    out = runs.merge(g, on="simulation_id", how="left").fillna({"win_tp": 0, "win_n": 0})
    out["win_tp"], out["win_n"] = out["win_tp"].astype(int), out["win_n"].astype(int)
    return out


def _boot_ratio(tp: np.ndarray, n: np.ndarray, n_boot: int, rng: np.random.Generator):
    if n.sum() == 0:
        return np.nan, (np.nan, np.nan)
    pt = float(tp.sum() / n.sum())
    if n_boot <= 0 or len(n) == 1:
        return pt, (pt, pt)
    idx = rng.integers(0, len(n), size=(n_boot, len(n)))
    bn, btp = n[idx].sum(1), tp[idx].sum(1)
    ok = bn > 0
    v = btp[ok] / bn[ok]
    return pt, (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def summarize_class(pr: pd.DataFrame, n_boot: int, seed: int) -> dict:
    """Window recall (run-bootstrap) and run-level detection rate (run-bootstrap) for one set of runs."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 1002]))
    n_runs = len(pr)
    if n_runs == 0:
        return {"n_runs": 0, "n_windows": 0, "window_recall": np.nan, "window_recall_ci_lo": np.nan,
                "window_recall_ci_hi": np.nan, "run_detect_rate": np.nan, "run_detect_ci_lo": np.nan,
                "run_detect_ci_hi": np.nan}
    r, (rlo, rhi) = _boot_ratio(pr["win_tp"].to_numpy(float), pr["win_n"].to_numpy(float), n_boot, rng)
    d, (dlo, dhi) = _boot_ratio(pr["detected"].to_numpy(float), np.ones(n_runs), n_boot, rng)
    return {"n_runs": n_runs, "n_windows": int(pr["win_n"].sum()),
            "window_recall": r, "window_recall_ci_lo": rlo, "window_recall_ci_hi": rhi,
            "run_detect_rate": d, "run_detect_ci_lo": dlo, "run_detect_ci_hi": dhi}


def snr_bin_of(snr: float) -> str | None:
    if pd.isna(snr):
        return None
    for name, lo, hi in SNR_BINS:
        if lo <= snr < hi:
            return name
    return None
