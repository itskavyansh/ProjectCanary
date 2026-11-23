#!/usr/bin/env python
"""Single-command pipeline (SYNTHETIC / SIMULATION): data -> features -> models -> eval (-> compare).

    python run_pipeline.py --config configs/pipeline_v1.yaml --out outputs/pipeline_check/
    python run_pipeline.py --config configs/pipeline_v1.yaml --out outputs/pipeline_quick/ --quick
    python run_pipeline.py --config configs/pipeline_v1.yaml --out X --stages models,eval   (re-use X/datasets)

This file only WRAPS existing modules/scripts; it does not reimplement them:
  data      scripts/run_faults_demo.py + scripts/run_one.py (reference state timeline + telemetry, subprocess)
            canary_sim.dataset.build_datasets  (state timelines -> telemetry -> labels -> windows -> v3 features
            are chained inside dataset.generate_one_run, so they are produced in ONE pass in this stage and
            written as manifest/labels/features files + metadata.json with content hashes)
  features  checks the v3 feature set on the written files and runs canary_sim.validation_checks
  models    canary_sim.models (ThresholdRule, LR, RF with the FROZEN val-tuned settings) and
            canary_sim.alarm.tune_min_prob_operating_point (val only)
  eval      canary_sim.evaluate + canary_sim.alarm -> headline metrics table
  compare   scripts/compare_pipeline_to_frozen.py (default only for a full, non-quick run)
Nothing under the frozen directories is ever written.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # same single-thread BLAS as the frozen v3 training run
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import copy
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from canary_sim import (  # noqa: E402
    alarm as A, confounders as cf, dataset as ds, engine, evaluate as ev, features as F, models as M,
    sensors as S, validation_checks as V, versioning as VR,
)

STAGE_ORDER = ["data", "features", "models", "eval", "compare"]
MODEL_NAMES = ("threshold_rule", "logistic_regression", "random_forest")
FROZEN_DIRS = ("outputs/datasets", "outputs/datasets_v3", "outputs/eval", "outputs/eval_v2",
               "outputs/eval_v3", "outputs/eval_v3_yaw")


def load_pipeline_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _p(rel: str | Path) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def _assert_not_frozen(out: Path):
    out_r = out.resolve()
    for fd in FROZEN_DIRS:
        fr = _p(fd).resolve()
        if out_r == fr or fr in out_r.parents:
            raise SystemExit(f"REFUSING to write into frozen directory: {fr}")


def effective_dataset_config(pcfg: dict, quick: bool) -> dict:
    ds_cfg = ds.load_dataset_config(_p(pcfg["configs"]["dataset"]))
    ds_cfg = copy.deepcopy(ds_cfg)
    if quick:
        q = pcfg["quick"]
        ds_cfg["splits"] = dict(q["splits"])
        ds_cfg["shifted_test"]["runs_per_class"] = int(q["shifted_test_runs_per_class"])
        ds_cfg["broad_basin_subset"]["runs_per_class"] = int(q["broad_basin_runs_per_class"])
        ds_cfg["stress_set"]["n_runs"] = int(q["stress_n_runs"])
    return ds_cfg


# ----------------------------------------------------------------------------------- stages
def stage_data(pcfg: dict, out: Path, quick: bool, log: dict) -> None:
    ref = pcfg["reference_run"]
    refdir = out / "reference_runs"
    refdir.mkdir(parents=True, exist_ok=True)
    # reference state timeline + telemetry via the existing demo / run_one scripts
    for script, sub, extra in ((ref["demo_script"], "faults_demo", []),
                               (ref["run_one_script"], "run_one", ["--seed", str(ref["seed"])])):
        cmd = [sys.executable, str(_p(script)), "--out", str(refdir / sub), *extra]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        log.setdefault("subprocesses", []).append({"cmd": " ".join(cmd), "returncode": r.returncode})
        if r.returncode != 0:
            raise RuntimeError(f"reference script failed: {' '.join(cmd)}\n{r.stdout}\n{r.stderr}")

    cfgs = pcfg["configs"]
    ds_cfg = effective_dataset_config(pcfg, quick)
    scen = engine.load_config(_p(cfgs["scenario"]))
    sens = S.load_sensor_config(_p(cfgs["sensor"]))
    conf = cf.load_confounder_config(_p(cfgs["confounder"]))
    prof = S.load_noise_profile(_p(cfgs["noise_profile"]))
    cfg_files = {k: _p(v) for k, v in cfgs.items()}
    cfg_files["pipeline"] = _p(log["config_path"])
    dsdir = out / "datasets"
    ds.build_datasets(ds_cfg, scen, sens, conf, prof, out_dir=dsdir, config_files=cfg_files)
    # re-write metadata including the effective (possibly quick-overridden) dataset config
    VR.write_dataset_metadata(dsdir, cfg_files, extra={"quick": quick, "effective_dataset_config": ds_cfg})
    log["n_runs"] = int(len(pd.read_csv(dsdir / "manifest.csv")))


def stage_features(pcfg: dict, out: Path, quick: bool, log: dict) -> None:
    dsdir = out / "datasets"
    _require(dsdir / "manifest.csv", "data")
    train = pd.read_csv(dsdir / "features_train.csv", nrows=50)
    cols = F.feature_columns(train)
    M.guard_model_features(cols)
    assert not any(c.startswith("diag_") for c in cols)
    rep = V.run_all(dsdir, raise_on_fail=True)           # fails loudly with a readable report
    (out / "features_report.json").write_text(json.dumps({
        "n_model_features": len(cols), "model_features": cols,
        "yaw_dependent_note": "see docs/provenance_note.md (installation-yaw requirement)",
        "integrity_checks_passed": rep.passed,
    }, indent=2), encoding="utf-8")
    (out / "integrity_report.txt").write_text(rep.render(), encoding="utf-8")
    log["n_model_features"] = len(cols)


def _load_splits(dsdir: Path, names) -> dict:
    return {n: pd.read_csv(dsdir / f"features_{n}.csv", low_memory=False) for n in names}


def stage_models(pcfg: dict, out: Path, quick: bool, log: dict) -> None:
    dsdir = out / "datasets"
    _require(dsdir / "features_train.csv", "data")
    seed = int(pcfg["seed"])
    sp = _load_splits(dsdir, ("train", "val"))
    train, val = sp["train"], sp["val"]
    cols = F.feature_columns(train)
    mc = pcfg["models"]
    rf_p = dict(mc["random_forest"])
    if quick:
        rf_p["n_estimators"] = int(pcfg["quick"]["rf_n_estimators"])
    th = M.ThresholdRuleClassifier().fit(train, val)
    lr = M.fit_logistic_regression(train, val, seed=seed, feature_cols=cols,
                                   c_grid=(float(mc["logistic_regression"]["C"]),))
    rf = M.fit_random_forest(train, None, seed=seed, feature_cols=cols, fixed_params=rf_p)
    models = {"threshold_rule": th, "logistic_regression": lr, "random_forest": rf}

    op = pcfg["operating_points"]
    ops = {"episode_fa_le_1": {}, "strict_window_fa_le_1": {}}
    for name in ("random_forest", "logistic_regression"):
        for key in ("episode_fa_le_1", "strict_window_fa_le_1"):
            ops[key][name] = A.tune_min_prob_operating_point(
                models[name], val, max_fa_per_normal_run=float(op["fa_target_per_normal_run"]),
                k=int(op[key]["k"]), m=int(op[key]["m"]), fa_metric=op[key]["fa_metric"])
    mdir = out / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    with open(mdir / "trained_models.pkl", "wb") as f:
        pickle.dump(models, f)
    with open(mdir / "operating_points.pkl", "wb") as f:
        pickle.dump(ops, f)
    (mdir / "model_settings.json").write_text(json.dumps({
        "seed": seed, "quick": quick, "n_model_features": len(cols),
        "threshold_rule_thresholds": th.thresholds,
        "logistic_regression": lr.best_params, "random_forest": rf.best_params,
        "operating_points": {k: {m: {"group_thresholds": v.group_thresholds, "k": v.k, "m": v.m,
                                     "val_summary": v.tuning_summary["best_val"]}
                                 for m, v in d.items()} for k, d in ops.items()},
    }, indent=2, default=float), encoding="utf-8")


def stage_eval(pcfg: dict, out: Path, quick: bool, log: dict) -> None:
    dsdir = out / "datasets"
    mdir = out / "models"
    _require(mdir / "trained_models.pkl", "models")
    seed = int(pcfg["seed"])
    n_boot = int(pcfg["quick"]["n_boot"]) if quick else int(pcfg["evaluation"]["n_boot"])
    with open(mdir / "trained_models.pkl", "rb") as f:
        models = pickle.load(f)
    with open(mdir / "operating_points.pkl", "rb") as f:
        ops = pickle.load(f)
    pers = pcfg["operating_points"]["threshold_rule_persistence"]
    rows = []
    for sp in pcfg["evaluation"]["splits"]:
        df = pd.read_csv(dsdir / f"features_{sp}.csv", low_memory=False)
        for mname in MODEL_NAMES:
            raw = models[mname].predict(df)
            regimes = [
                ("raw (k=1)", raw, pers["raw"]["k"], pers["raw"]["m"]),
                ("episode_fa<=1 (k=3,m=3)",
                 raw if mname == "threshold_rule" else ops["episode_fa_le_1"][mname].predict_window(df),
                 pers["episode_fa_le_1"]["k"], pers["episode_fa_le_1"]["m"]),
                ("strict_window_fa<=1",
                 raw if mname == "threshold_rule" else ops["strict_window_fa_le_1"][mname].predict_window(df),
                 pers["strict_window_fa_le_1"]["k"] if mname == "threshold_rule" else 3,
                 pers["strict_window_fa_le_1"]["m"] if mname == "threshold_rule" else 2),
            ]
            for rname, w_p, k, m in regimes:
                w_ev = ev.evaluate_window_level_with_run_bootstrap(df, w_p, n_boot=n_boot, seed=seed)
                al = A.evaluate_alarm_level(df, w_p, k=k, m=m, n_boot=n_boot, seed=seed)
                al_w = ev.evaluate_window_level_with_run_bootstrap(df, al["alarm_preds"], n_boot=n_boot, seed=seed)
                rows.append({
                    "split": sp, "version": "v3 (new features)", "regime": rname, "model": mname, "k": k, "m": m,
                    "window_macro_f1": round(w_ev["macro_f1"], 4),
                    "window_macro_f1_ci_lo": round(w_ev["macro_f1_ci95"][0], 4),
                    "window_macro_f1_ci_hi": round(w_ev["macro_f1_ci95"][1], 4),
                    "alarm_macro_f1": round(al_w["macro_f1"], 4),
                    "alarm_macro_f1_ci_lo": round(al_w["macro_f1_ci95"][0], 4),
                    "alarm_macro_f1_ci_hi": round(al_w["macro_f1_ci95"][1], 4),
                    "fa_episodes_per_normal_run": round(al["fa_episodes_per_normal_run"], 3),
                    "fa_windows_per_normal_run": round(al["fa_windows_per_normal_run"], 3),
                    "alarm_exact_detect_rate": round(al["exact_detect_rate"], 4),
                    "alarm_exact_detect_ci_lo": round(al["exact_detect_ci95"][0], 4),
                    "alarm_exact_detect_ci_hi": round(al["exact_detect_ci95"][1], 4),
                    "mean_delay_exact_s": round(al["mean_delay_exact_s"], 3),
                })
    edir = out / "eval"
    edir.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame(rows)
    tab.to_csv(edir / "headline_metrics.csv", index=False)
    (edir / "metrics_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    head = tab[tab["regime"] == "episode_fa<=1 (k=3,m=3)"]
    md = ["# Headline metrics (SYNTHETIC / SIMULATION; episode-matched FA<=1 regime)", "",
          "| split | model | window macro-F1 | alarm macro-F1 | FA episodes / normal run | run exact detect rate |",
          "|---|---|---|---|---|---|"]
    for _, r in head.iterrows():
        md.append(f"| {r['split']} | {r['model']} | {r['window_macro_f1']} | {r['alarm_macro_f1']} | "
                  f"{r['fa_episodes_per_normal_run']} | {r['alarm_exact_detect_rate']} |")
    (edir / "headline_metrics.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def stage_compare(pcfg: dict, out: Path, quick: bool, log: dict) -> None:
    import compare_pipeline_to_frozen as C
    C.run_comparison(pcfg, out, root=ROOT)


def _require(path: Path, stage: str):
    if not path.exists():
        raise SystemExit(f"missing {path}: run the '{stage}' stage first (or include it in --stages)")


STAGE_FUNCS = {"data": stage_data, "features": stage_features, "models": stage_models,
               "eval": stage_eval, "compare": stage_compare}


def run_pipeline(config_path: str | Path, out_dir: str | Path,
                 stages: list[str] | None = None, quick: bool = False) -> int:
    argv = ["--config", str(config_path), "--out", str(out_dir)]
    if stages:
        argv += ["--stages", ",".join(stages)]
    if quick:
        argv.append("--quick")
    return run(argv)


def run(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_v1.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default=None, help="comma list of: data,features,models,eval,compare "
                    "(default: data,features,models,eval[,compare when not --quick])")
    ap.add_argument("--quick", action="store_true", help="small run counts, smoke test only")
    a = ap.parse_args(argv)

    out = _p(a.out)
    _assert_not_frozen(out)
    out.mkdir(parents=True, exist_ok=True)
    pcfg = load_pipeline_config(_p(a.config))
    default = ["data", "features", "models", "eval"] + ([] if a.quick else ["compare"])
    wanted = default if a.stages is None else [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in wanted if s not in STAGE_ORDER]
    if bad:
        raise SystemExit(f"unknown stage(s) {bad}; valid: {STAGE_ORDER}")
    stages = [s for s in STAGE_ORDER if s in wanted]          # always executed in canonical order

    log = {"config_path": str(_p(a.config)), "quick": a.quick, "stages_requested": wanted,
           "stages_run": [], "timings_s": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for s in stages:
        t0 = time.time()
        print(f"[pipeline] stage {s} ...", flush=True)
        STAGE_FUNCS[s](pcfg, out, a.quick, log)
        log["stages_run"].append(s)
        log["timings_s"][s] = round(time.time() - t0, 2)
        print(f"[pipeline] stage {s} done in {log['timings_s'][s]} s", flush=True)
        (out / "pipeline_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
