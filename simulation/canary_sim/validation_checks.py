"""Dataset integrity checks (SYNTHETIC / SIMULATION). Fail loudly with a readable report.

Run on a dataset directory containing `manifest.csv`, `window_labels.csv`, `window_features.csv`
and `runs_metadata.json` (as written by `dataset.build_datasets`).

IMPLEMENTED HERE (operate on the written files, not covered by an existing test):
  1. class_counts_after_labelling  - every run has the labels its run_class implies; per-run window
                                      counts match the manifest; label-count table is reported.
  2. missing_values                - NaN/inf only where documented as allowed (see ALLOWED_NAN).
  3. duplicate_windows             - (simulation_id, window_idx) unique; simulation_id/seed unique.
  4. impossible_values             - tilt beyond physical limits, negative counts, fractions outside
                                      [0, 1], negative staleness/latency, bad window geometry, ...
  5. label_consistency             - labels recomputed from the state-timeline parameters
                                      (basin onset) + fault/event metadata and compared window by window.

SKIPPED BECAUSE AN EXISTING TEST ALREADY COVERS THEM (see `COVERED_BY_EXISTING_TESTS`):
  * train/val/test leakage by run ID and by window
  * configured class counts per split (spec level)
  * diag_ columns excluded from model features / all model features finite
  * hand-built 50% overlap / aftermath / transition label cases
  * frozen schema validation of the state timeline
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import classes as C, features as F, labels as L, windows as W

COVERED_BY_EXISTING_TESTS: dict[str, list[str]] = {
    "train/val/test leakage by run ID": [
        "tests/test_dataset_pipeline.py::test_no_run_appears_in_more_than_one_split",
    ],
    "windows of one run never in two splits": [
        "tests/test_dataset_pipeline.py::test_windows_from_one_run_never_appear_in_two_splits",
    ],
    "configured class counts per split": [
        "tests/test_dataset_pipeline.py::test_class_counts_per_split_match_config",
    ],
    "diag_ columns excluded / model features finite": [
        "tests/test_dataset_pipeline.py::test_diag_columns_excluded_from_model_features",
    ],
    "label rule on hand-built cases (50%, aftermath, transition)": [
        "tests/test_dataset_pipeline.py::test_labels_50_percent_rule_aftermath_and_transition_hand_built",
        "tests/test_dataset_pipeline.py::test_fault_and_confounder_windows_labelled_correctly",
    ],
    "state-timeline schema": ["tests/test_engine.py::test_schema_validation_and_metadata",
                              "tests/test_engine.py::test_schema_rejects_bad_frame"],
}

# Documented NaN-allowed fields:
#   window_labels.csv: t_rel_event_start_s is NaN for runs with no event/fault interval (normal runs);
#                      label_set / partial_set are empty (read back as NaN) when no class is active/partial.
#   manifest.csv:      fault_types is empty (NaN) for runs without faults.
ALLOWED_NAN = {
    "labels": {"t_rel_event_start_s", "label_set", "partial_set"},
    "manifest": {"fault_types"},
    # window_features.csv carries copies of label_set and t_rel_event_start_s from the labels table
    "features": {"label_set", "t_rel_event_start_s"},
}

TILT_LIMIT_NORMAL_DEG = 15.0   # max basin 3 deg + offsets + noise; far below any gross sensor fault
TILT_LIMIT_ABSOLUTE_DEG = 360.0  # implausible_reading injects 120-175 deg per axis (hypot <= ~250)
WINDOW_SAMPLES = 5             # telemetry window_s = 1 s -> 5 samples per 5 s window


class IntegrityError(RuntimeError):
    pass


@dataclass
class Report:
    passed: list[str] = field(default_factory=list)
    failed: dict[str, list[str]] = field(default_factory=dict)
    info: dict[str, str] = field(default_factory=dict)

    def ok(self, name: str):
        self.passed.append(name)

    def fail(self, name: str, msgs: list[str]):
        self.failed[name] = msgs[:25] + ([f"... and {len(msgs) - 25} more"] if len(msgs) > 25 else [])

    @property
    def success(self) -> bool:
        return not self.failed

    def render(self) -> str:
        lines = ["DATASET INTEGRITY REPORT", "=" * 60]
        for n in self.passed:
            lines.append(f"[PASS] {n}")
        for n, msgs in self.failed.items():
            lines.append(f"[FAIL] {n}  ({len(msgs)} problem(s) listed)")
            lines += [f"         - {m}" for m in msgs]
        for k, v in self.info.items():
            lines.append(f"[INFO] {k}: {v}")
        lines.append("[SKIP] covered by existing tests:")
        for k, v in COVERED_BY_EXISTING_TESTS.items():
            lines.append(f"         * {k}: {', '.join(v)}")
        lines.append("RESULT: " + ("ALL CHECKS PASSED" if self.success else "INTEGRITY CHECK FAILED"))
        return "\n".join(lines)


def _load(dir_: Path):
    man = pd.read_csv(dir_ / "manifest.csv")
    lab = pd.read_csv(dir_ / "window_labels.csv", low_memory=False)
    fea = pd.read_csv(dir_ / "window_features.csv", low_memory=False)
    meta = json.loads((dir_ / "runs_metadata.json").read_text(encoding="utf-8"))
    return man, lab, fea, meta


def check_class_counts(man, lab, rep: Report):
    msgs = []
    counts = lab.groupby(["split", "label"]).size().unstack(fill_value=0)
    rep.info["label counts per split"] = "\n" + counts.to_string()
    per_run = lab.groupby("simulation_id")
    n_win = per_run.size()
    labset = per_run["label"].agg(lambda s: set(s))
    # stress runs: a secondary condition is recorded in `label_set`, not in the priority `label`
    setset = lab.groupby("simulation_id")["label_set"].agg(
        lambda s: {t for v in s.dropna() for t in str(v).split("|") if t})
    for _, r in man.iterrows():
        sid = r["simulation_id"]
        if sid not in n_win.index:
            msgs.append(f"{sid}: no labelled windows")
            continue
        if int(n_win[sid]) != int(r["n_windows"]):
            msgs.append(f"{sid}: {n_win[sid]} labelled windows != manifest n_windows {r['n_windows']}")
        conds = str(r["conditions"]).split("|")
        got = labset[sid]
        if r["split"] != "stress":
            if r["run_class"] == "normal" and (got - {"normal"}):
                msgs.append(f"{sid}: normal run has non-normal labels {sorted(got - {'normal'})}")
            if r["run_class"] != "normal" and r["run_class"] not in got:
                msgs.append(f"{sid}: run_class {r['run_class']} never appears in its labels {sorted(got)}")
        else:
            for c in conds:
                if c not in got and c not in setset[sid]:
                    msgs.append(f"{sid}: stress condition {c} never labelled (labels {sorted(got)}, "
                                f"label_set members {sorted(setset[sid])})")
        bad = got - set(C.CLASSES) - {L.TRANSITION}
        if bad:
            msgs.append(f"{sid}: unknown label(s) {sorted(bad)}")
    if msgs:
        rep.fail("class_counts_after_labelling", msgs)
    else:
        rep.ok("class_counts_after_labelling")


def check_missing(man, lab, fea, rep: Report):
    msgs = []
    for name, df, allowed in (("labels", lab, ALLOWED_NAN["labels"]), ("manifest", man, ALLOWED_NAN["manifest"]),
                              ("features", fea, ALLOWED_NAN["features"])):
        num = df.select_dtypes(include=[np.number])
        inf_cols = [c for c in num.columns if np.isinf(num[c].to_numpy(dtype=float)).any()]
        for c in inf_cols:
            msgs.append(f"{name}.{c}: contains inf")
        for c in df.columns:
            if c in allowed:
                continue
            n = int(df[c].isna().sum())
            if n:
                msgs.append(f"{name}.{c}: {n} undocumented missing value(s)")
    # documented NaN: t_rel_event_start_s must be NaN only for runs without any event interval
    if "t_rel_event_start_s" in lab.columns:
        ev_runs = set(man.loc[man["run_class"] != "normal", "simulation_id"]) | \
            set(man.loc[man["conditions"].astype(str).str.contains(r"\|"), "simulation_id"])
        nan_runs = set(lab.loc[lab["t_rel_event_start_s"].isna(), "simulation_id"])
        for sid in sorted(nan_runs & ev_runs):
            msgs.append(f"labels.t_rel_event_start_s: NaN in event/fault run {sid} (allowed only for normal runs)")
    (rep.fail("missing_values", msgs) if msgs else rep.ok("missing_values"))


def check_duplicates(man, lab, fea, rep: Report):
    msgs = []
    for name, df in (("labels", lab), ("features", fea)):
        d = df.duplicated(["simulation_id", "window_idx"]).sum()
        if d:
            msgs.append(f"{name}: {int(d)} duplicate (simulation_id, window_idx) rows")
    if man["simulation_id"].duplicated().any():
        msgs.append("manifest: duplicate simulation_id")
    if man["seed"].duplicated().any():
        msgs.append("manifest: duplicate seed")
    # identical feature rows under different window ids within a run would also be suspicious
    key = ["simulation_id", "t_start"]
    if fea.duplicated(key).any():
        msgs.append(f"features: {int(fea.duplicated(key).sum())} duplicate (simulation_id, t_start) rows")
    (rep.fail("duplicate_windows", msgs) if msgs else rep.ok("duplicate_windows"))


def check_impossible(man, fea, rep: Report):
    msgs = []
    N = F.NODES
    frac = [f"{n}_missing_frac" for n in N] + ["comm_missing_frac_net"]
    for c in frac:
        bad = fea[(fea[c] < 0) | (fea[c] > 1)]
        if len(bad):
            msgs.append(f"{c}: {len(bad)} values outside [0,1]")
    nonneg = ["comm_max_burst_len", "comm_nodes_unavailable", "comm_latency_mean_s", "comm_latency_max_s",
              "piezo_impulse_total", "piezo_nodes_with_impulse", "hub_staleness_max_s", "hub_residual_abs_max_m",
              *[f"{n}_piezo_impulse_count" for n in N], *[f"{n}_lidar_staleness_s" for n in N],
              *[f"{n}_piezo_peak_max" for n in N], *[f"{n}_piezo_rms_mean" for n in N], *[f"{n}_std" for n in N],
              *[f"{n}_mean" for n in N]]
    for c in nonneg:
        if (fea[c] < 0).any():
            msgs.append(f"{c}: {int((fea[c] < 0).sum())} negative value(s)")
    if (fea["comm_max_burst_len"] > WINDOW_SAMPLES).any():
        msgs.append(f"comm_max_burst_len exceeds samples per window ({WINDOW_SAMPLES})")
    if (fea["comm_nodes_unavailable"] > 4).any() or (fea["piezo_nodes_with_impulse"] > 4).any():
        msgs.append("node count column > 4")
    cnt_cols = [f"{n}_piezo_impulse_count" for n in N]
    if (fea[cnt_cols] > WINDOW_SAMPLES).any().any():
        msgs.append(f"piezo impulse count > samples per window ({WINDOW_SAMPLES})")
    if ((fea["t_end"] - fea["t_start"]).round(6) != 5.0).any():
        msgs.append("window length != 5.0 s")
    if (fea["t_start"] < 0).any():
        msgs.append("negative t_start")
    # tilt limits: only runs with an implausible_reading fault may exceed the normal physical limit
    fl = man.set_index("simulation_id")["fault_types"].fillna("")
    mean_cols = [f"{n}_mean" for n in N]
    mx = fea.groupby("simulation_id")[mean_cols].max().max(axis=1)
    for sid, v in mx.items():
        implausible = "implausible_reading" in str(fl.get(sid, ""))
        if v > TILT_LIMIT_ABSOLUTE_DEG:
            msgs.append(f"{sid}: tilt magnitude {v:.1f} deg beyond absolute limit")
        elif v > TILT_LIMIT_NORMAL_DEG and not implausible:
            msgs.append(f"{sid}: tilt magnitude {v:.1f} deg > {TILT_LIMIT_NORMAL_DEG} deg without an implausible_reading fault")
    basin = man[man["conditions"].astype(str).str.contains("gradual|rapid")]
    if ((basin["peak_tilt_deg"] < 0.05 - 1e-9) | (basin["peak_tilt_deg"] > 3.0 + 1e-9)).any():
        msgs.append("basin peak_tilt_deg outside configured [0.05, 3.0] deg")
    if (man["snr_vs_imu_noise"] < 0).any() or (man["compression_factor"] <= 0).any():
        msgs.append("manifest: negative snr or non-positive compression factor")
    (rep.fail("impossible_values", msgs) if msgs else rep.ok("impossible_values"))


def check_label_consistency(man, lab, meta, rep: Report, win_s=5.0, overlap=0.5, min_overlap=0.5):
    msgs = []
    lab_by = {sid: g.sort_values("window_idx") for sid, g in lab.groupby("simulation_id")}
    for _, r in man.iterrows():
        sid = r["simulation_id"]
        P = json.loads(r["parameters_json"])
        conds = str(r["conditions"]).split("|")
        T = float(P["duration_s"])
        rate = float(P["sample_rate_hz"])
        ivs = []
        if "gradual" in conds or "rapid" in conds:
            t = np.arange(int(round(T * rate))) / rate
            on = t[t >= P["t_on"]]
            start = float(on[0]) if len(on) else T
            ivs.append({"class": "rapid" if "rapid" in conds else "gradual", "start_s": start,
                        "end_s": float(T), "source": "state_timeline"})
        ivs += L.intervals_from_meta({"faults": json.loads(r["faults_json"]), "events": json.loads(r["events_json"])})
        exp = L.label_windows(W.make_windows(T, win_s, overlap), ivs, min_overlap=min_overlap)
        got = lab_by.get(sid)
        if got is None or len(got) != len(exp):
            msgs.append(f"{sid}: window count {None if got is None else len(got)} != expected {len(exp)}")
            continue
        diff = np.nonzero(got["label"].to_numpy() != exp["label"].to_numpy())[0]
        if len(diff):
            i = int(diff[0])
            msgs.append(f"{sid}: {len(diff)} label mismatch(es); first at window {i}: "
                        f"file={got['label'].iloc[i]} expected={exp['label'].iloc[i]}")
        gs = got["label_set"].fillna("").to_numpy()
        es = exp["label_set"].fillna("").to_numpy()
        if (gs != es).any():
            msgs.append(f"{sid}: label_set mismatch in {int((gs != es).sum())} window(s)")
        if not np.allclose(got["t_rel_event_start_s"].to_numpy(dtype=float),
                           exp["t_rel_event_start_s"].to_numpy(dtype=float), equal_nan=True):
            msgs.append(f"{sid}: t_rel_event_start_s mismatch")
    (rep.fail("label_consistency", msgs) if msgs else rep.ok("label_consistency"))


def run_all(dataset_dir: str | Path, raise_on_fail: bool = True) -> Report:
    d = Path(dataset_dir)
    man, lab, fea, meta = _load(d)
    rep = Report()
    check_class_counts(man, lab, rep)
    check_missing(man, lab, fea, rep)
    check_duplicates(man, lab, fea, rep)
    check_impossible(man, fea, rep)
    check_label_consistency(man, lab, meta, rep)
    if raise_on_fail and not rep.success:
        raise IntegrityError("\n" + rep.render())
    return rep
