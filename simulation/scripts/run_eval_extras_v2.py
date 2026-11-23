#!/usr/bin/env python
"""Eval extras v2 runner (SYNTHETIC / SIMULATION). Predict-only: frozen v3 models/operating points, no fitting.

    python scripts/run_eval_extras_v2.py --out outputs/eval_extras_v2
Never writes into outputs/eval_extras (hashes verified before and after).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from canary_sim import eval_extras as EE, eval_extras_v2 as V2  # noqa: E402

MODELS = ("threshold_rule", "logistic_regression", "random_forest")
EVAL_SPLITS = ("test", "shifted_test", "stress", "broad_basin")
FA_SPLITS = ("test", "shifted_test", "val", "train")
OLD_DIR = ROOT / "outputs" / "eval_extras"
BANNER = "> **SYNTHETIC / SIMULATION** - all data are simulated; nothing here is validated on real data or field-ready."


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def old_hashes() -> dict:
    return {p.name: sha(p) for p in sorted(OLD_DIR.iterdir()) if p.is_file()}


def ci(pt, lo, hi, d=3):
    return "N/A" if pd.isna(pt) else f"{pt:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def main(out: Path, data: Path, models_p: Path, ops_p: Path, n_boot: int, seed: int) -> None:
    before = old_hashes()
    out.mkdir(parents=True, exist_ok=True)
    models, ops = pickle.load(open(models_p, "rb")), pickle.load(open(ops_p, "rb"))
    manifest = pd.read_csv(data / "manifest.csv", low_memory=False)
    splits = sorted(set(FA_SPLITS) | set(EVAL_SPLITS))
    dfs = {s: pd.read_csv(data / f"features_{s}.csv", low_memory=False) for s in splits}
    regs = {(s, m): V2.regimes_for(m, models[m], ops, dfs[s]) for s in splits for m in MODELS}

    # ---- 1. false alarms per hour, definitions A and B
    print("[v2] 1. false alarms per hour (A and B)", flush=True)
    fa_rows = []
    for sp in FA_SPLITS:
        fa_rows += V2.fa_rows_for_split(sp, dfs[sp], {m: regs[(sp, m)] for m in MODELS}, manifest, n_boot, seed)
    fa = pd.DataFrame(fa_rows)
    fa.to_csv(out / "false_alarms_per_hour_both_definitions.csv", index=False)
    d0 = fa.iloc[0]
    md = ["# False alarms per hour - two time definitions (SYNTHETIC / SIMULATION)", "", BANNER, "",
          f"* **Definition A - {V2.LABEL_A}**: false alarms per normal run / {d0.hours_per_run_A_slow:g} h "
          f"({d0.sensor_duration_s:g} s x compression factor {d0.compression_factor:g} / 3600).",
          f"* **Definition B - {V2.LABEL_B}**: false alarms per normal run / {d0.hours_per_run_B_sensor:.4f} h "
          f"({d0.sensor_duration_s:g} s of actual sensor time / 3600). Definition B is the stricter figure "
          "(15x higher than A with these numbers) and the more defensible one for false alarms caused by sensor noise "
          "and benign vibration. Neither is a validated field rate.",
          "* All normal runs of every listed split use compression factor 360, so the factors are never mixed. "
          "Rows are false-alarm *episodes* (after persistence); CIs are run-bootstrap 95%; missed-event rate = event runs "
          "with no correct alarm.", "",
          "| split | model | regime | FA eps / run | A: FA / h (slow) | A: one FA every | B: FA / h (sensor) | B: one FA every | missed-event rate | exact detect |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in fa.iterrows():
        md.append(f"| {r.split} | {r.model} | {r.regime} | "
                  f"{ci(r.fa_episodes_per_run, r.fa_episodes_per_run_ci_lo, r.fa_episodes_per_run_ci_hi)} | "
                  f"{ci(r.A_slow_fa_episodes_per_hour, r.A_slow_ci_lo, r.A_slow_ci_hi, 4)} | {V2.fmt_every(r.A_slow_one_fa_every_hours)} | "
                  f"{ci(r.B_sensor_fa_episodes_per_hour, r.B_sensor_ci_lo, r.B_sensor_ci_hi, 2)} | {V2.fmt_every(r.B_sensor_one_fa_every_hours)} | "
                  f"{ci(r.missed_event_rate, r.missed_event_ci_lo, r.missed_event_ci_hi)} | {r.exact_detect_rate:.3f} |")
    (out / "false_alarms_per_hour_both_definitions.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- 3 (computed first so the accuracy table can show recall and run detection together)
    print("[v2] 3. gradual / rapid recall + run detection", flush=True)
    snr = manifest.set_index("simulation_id")["snr_vs_imu_noise"]
    prs = {}
    for sp in EVAL_SPLITS:
        for m in MODELS:
            for rname, w_preds, k, mm in regs[(sp, m)]:
                det = V2.run_detection_records(dfs[sp], w_preds, k, mm)
                for cls in ("gradual", "rapid"):
                    t = V2.per_run_class_table(dfs[sp], w_preds, det, cls)
                    t["snr"] = t["simulation_id"].map(snr)
                    t["snr_bin"] = t["snr"].map(V2.snr_bin_of)
                    prs[(sp, m, rname, cls)] = t
    cls_rows, snr_rows = [], []
    for (sp, m, rname, cls), t in prs.items():
        cls_rows.append({"split": sp, "model": m, "regime": rname, "class": cls, **V2.summarize_class(t, n_boot, seed)})
        for b in [x[0] for x in V2.SNR_BINS]:
            snr_rows.append({"split": sp, "model": m, "regime": rname, "class": cls, "snr_bin": b,
                             **V2.summarize_class(t[t.snr_bin == b], n_boot, seed)})
    for m in MODELS:                                # pooled across the four evaluation splits
        for rname in [r[0] for r in regs[("test", m)]]:
            for cls in ("gradual", "rapid"):
                pool = pd.concat([prs[(sp, m, rname, cls)] for sp in EVAL_SPLITS])
                for b in [x[0] for x in V2.SNR_BINS]:
                    snr_rows.append({"split": "pooled (test+shifted_test+stress+broad_basin)", "model": m,
                                     "regime": rname, "class": cls, "snr_bin": b,
                                     **V2.summarize_class(pool[pool.snr_bin == b], n_boot, seed)})
    cdf, sdf = pd.DataFrame(cls_rows), pd.DataFrame(snr_rows)
    cdf.to_csv(out / "gradual_rapid_recall_and_detection.csv", index=False)
    sdf.to_csv(out / "gradual_rapid_recall_by_snr.csv", index=False)
    note = ("Window recall = share of true gradual (rapid) windows predicted as that class by the model/operating point "
            "(before persistence). Run-level detection = share of gradual (rapid) runs with at least one alarm of that class "
            "after the regime's (k, m) persistence; it counts ANY correct alarm in the run, so it is always shown next to "
            "window recall and is never a substitute for it. Own-class rule: in the stress set an alarm of a co-occurring "
            "other class does not count. snr_vs_imu_noise = run's peak tilt / IMU noise (manifest).")
    md = ["# Gradual / rapid: window recall and run-level detection together (SYNTHETIC / SIMULATION)", "", BANNER, "", note, "",
          "| split | model | regime | class | runs | windows | window recall | run-level detection |", "|---|---|---|---|---|---|---|---|"]
    for _, r in cdf.iterrows():
        md.append(f"| {r.split} | {r.model} | {r.regime} | {r['class']} | {r.n_runs} | {r.n_windows} | "
                  f"{ci(r.window_recall, r.window_recall_ci_lo, r.window_recall_ci_hi)} | "
                  f"{ci(r.run_detect_rate, r.run_detect_ci_lo, r.run_detect_ci_hi)} |")
    (out / "gradual_rapid_recall_and_detection.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    md = ["# Gradual / rapid recall by snr_vs_imu_noise bin at the v3 operating points (SYNTHETIC / SIMULATION)", "", BANNER, "",
          note, "", "Small cells (few runs) have wide intervals; N/A = no run in that bin. Operating-point regimes only "
          "(episode_fa<=1 and strict_window_fa<=1) plus raw for reference.", "",
          "| split | model | regime | class | SNR bin | runs | windows | window recall | run-level detection |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sdf.iterrows():
        md.append(f"| {r.split} | {r.model} | {r.regime} | {r['class']} | {r.snr_bin} | {r.n_runs} | {r.n_windows} | "
                  f"{ci(r.window_recall, r.window_recall_ci_lo, r.window_recall_ci_hi)} | "
                  f"{ci(r.run_detect_rate, r.run_detect_ci_lo, r.run_detect_ci_hi)} |")
    (out / "gradual_rapid_recall_by_snr.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- 2. accuracy table with N/A and corrected caption
    print("[v2] 2. accuracy table", flush=True)
    shares = {sp: V2.split_class_shares(dfs[sp]) for sp in EVAL_SPLITS}
    pd.DataFrame(shares).T.fillna(0).round(4).to_csv(out / "class_share_of_windows_per_split.csv")
    rows = []
    for sp in EVAL_SPLITS:
        for m in MODELS:
            for rname, w_preds, k, mm in regs[(sp, m)]:
                res = EE.evaluate_window_level_accuracy_and_f1(dfs[sp], w_preds, n_boot=n_boot, seed=seed)
                row = V2.accuracy_row(sp, m, rname, res)
                for cls in ("gradual", "rapid"):
                    c = cdf[(cdf.split == sp) & (cdf.model == m) & (cdf.regime == rname) & (cdf["class"] == cls)].iloc[0]
                    row[f"{cls}_run_detect_rate"] = round(c.run_detect_rate, 4) if pd.notna(c.run_detect_rate) else np.nan
                    row[f"{cls}_run_detect_ci_lo"] = round(c.run_detect_ci_lo, 4) if pd.notna(c.run_detect_ci_lo) else np.nan
                    row[f"{cls}_run_detect_ci_hi"] = round(c.run_detect_ci_hi, 4) if pd.notna(c.run_detect_ci_hi) else np.nan
                rows.append(row)
    acc = pd.DataFrame(rows)
    acc.to_csv(out / "window_accuracy_and_f1_v2.csv", index=False)
    share_txt = "; ".join(f"{sp}: {shares[sp].get('normal', 0) * 100:.1f}%" for sp in EVAL_SPLITS)
    not_top = [sp for sp in EVAL_SPLITS if max(shares[sp], key=shares[sp].get) != "normal"]
    top_txt = ("`normal` is the largest class in every split" if not not_top else
               "`normal` is the largest class in " + ", ".join(sp for sp in EVAL_SPLITS if sp not in not_top) +
               " but NOT in " + ", ".join(f"{sp} (largest: {max(shares[sp], key=shares[sp].get)} "
                                          f"{max(shares[sp].values()) * 100:.1f}%)" for sp in not_top))
    md = ["# Window accuracy, Macro-F1 and per-class precision / recall (SYNTHETIC / SIMULATION)", "", BANNER, "",
          f"> **Caption.** Share of scored (non-transition) windows that are `normal` - {share_txt}; {top_txt}. The class mix "
          "in these synthetic splits is set by the dataset design and is NOT the real-world mix, so plain accuracy depends on "
          "that mix; read it next to Macro-F1 and per-class recall. A class with zero true windows in a split is shown "
          "as N/A (its recall and precision are undefined, not zero); Macro-F1 averages only the classes present.", "",
          "| split | model | regime | accuracy | Macro-F1 | P / R normal | P / R gradual | P / R rapid | P / R vibration | P / R sensor | P / R comm | gradual: window recall / run detect | rapid: window recall / run detect |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in acc.iterrows():
        gd = f"{V2.fmt_pr(r.rec_gradual, r.gradual_run_detect_rate).replace(' / ', ' / ')}"
        rd = f"{V2.fmt_pr(r.rec_rapid, r.rapid_run_detect_rate)}"
        md.append(f"| {r.split} | {r.model} | {r.regime} | {ci(r.window_accuracy, r.window_accuracy_ci_lo, r.window_accuracy_ci_hi, 3)} | "
                  f"{ci(r.window_macro_f1, r.window_macro_f1_ci_lo, r.window_macro_f1_ci_hi, 3)} | "
                  f"{V2.fmt_pr(r.prec_normal, r.rec_normal)} | {V2.fmt_pr(r.prec_gradual, r.rec_gradual)} | "
                  f"{V2.fmt_pr(r.prec_rapid, r.rec_rapid)} | {V2.fmt_pr(r.prec_abnormal_vibration, r.rec_abnormal_vibration)} | "
                  f"{V2.fmt_pr(r.prec_sensor_fault, r.rec_sensor_fault)} | {V2.fmt_pr(r.prec_comm_fault, r.rec_comm_fault)} | "
                  f"{gd} | {rd} |")
    (out / "window_accuracy_and_f1_v2.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    after = old_hashes()
    assert before == after, "outputs/eval_extras changed during the run!"
    (out / "run_manifest.json").write_text(json.dumps({
        "seed": seed, "n_boot": n_boot, "models": str(models_p), "operating_points": str(ops_p), "data": str(data),
        "fit_calls": 0, "old_eval_extras_unchanged": True, "old_eval_extras_sha256": after,
        "normal_run_share_of_windows": {sp: round(shares[sp].get("normal", 0), 4) for sp in EVAL_SPLITS}},
        indent=2), encoding="utf-8")
    print("[v2] DONE; outputs/eval_extras verified byte-identical", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/eval_extras_v2")
    ap.add_argument("--data", default="outputs/datasets_v3")
    ap.add_argument("--models", default="outputs/eval_v3/trained_models_v3.pkl")
    ap.add_argument("--ops", default="outputs/eval_v3/operating_points_v3.pkl")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    main(Path(a.out), Path(a.data), Path(a.models), Path(a.ops), a.n_boot, a.seed)
