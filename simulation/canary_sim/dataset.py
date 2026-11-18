"""Dataset generation (SYNTHETIC / SIMULATION): 6 classes, run-level splits, labels & windowed features.

Guarantees:
- Splits (`train`, `val`, `test`, `shifted_test`, `stress`, `broad_basin`) are strictly BY RUN;
  no run ID or seed ever appears in more than one split, and all windows of a run stay in its split.
- State timeline schema_v1 is untouched; class labels live in separate labels files / tables derived
  from the state timeline + fault/event metadata.
- Gradual and rapid runs stratify `peak_tilt_deg` log-uniformly across strata within each split so
  weak events (low SNR) are guaranteed to be represented.
- Manifest includes `peak_tilt_deg` and `snr_vs_imu_noise` (= peak_tilt_deg / effective IMU white-noise sd).
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import (
    classes as C,
    confounders as cf,
    engine,
    features as F,
     labels as L,
    sensors as S,
    windows as W,
)


def load_dataset_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def stratified_peaks(n: int, lo: float, hi: float, seed: int) -> list[float]:
    """Stratified log-uniform draws across [lo, hi] in n equal log-strata, shuffled deterministically."""
    if n <= 0:
        return []
    r = S.rng_for(seed, 500)
    edges = np.linspace(math.log(lo), math.log(hi), n + 1)
    u = edges[:-1] + r.random(n) * np.diff(edges)
    vals = np.exp(u)
    r.shuffle(vals)
    return [float(x) for x in vals]


def imu_noise_sd_deg(profile: dict, noise_multiplier: float) -> float:
    """Pooled 2-axis IMU white-noise standard deviation (deg) for a run."""
    sd = profile["imu_white_sd_deg"]
    base = math.sqrt(0.5 * (sd["tilt_x"] ** 2 + sd["tilt_y"] ** 2))
    return float(base * noise_multiplier)


def generate_one_run(
    spec: dict,
    scen_cfg: dict,
    sens_cfg: dict,
    conf_cfg: dict,
    ds_cfg: dict,
    profile: dict,
    save_run_dir: Path | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Generate one run end-to-end: state timeline -> telemetry -> labels -> window features.

    Returns (manifest_row, labels_df, features_df).
    """
    seed = int(spec["seed"])
    split = str(spec["split"])
    run_class = str(spec["run_class"])
    conds = list(spec.get("conditions", [run_class]))
    broad = bool(spec.get("broad_basin", False))
    shifted = bool(spec.get("shifted_noise", False))
    peak_override = spec.get("peak_tilt_deg")

    # 1. Scenario config & state timeline (frozen schema v1)
    sc = copy.deepcopy(scen_cfg)
    if broad:
        sc["ranges"]["broad_basin_subset"] = True
    has_basin = ("gradual" in conds) or ("rapid" in conds)
    basin_mode = "rapid" if "rapid" in conds else "gradual"

    state_df, state_meta = engine.run_scenario(
        sc,
        seed=seed,
        mode=basin_mode,
        peak_tilt_deg=peak_override if has_basin else None,
        null_field=not has_basin,
    )
    sim_id = f"sim_{split}_{run_class}_{seed}"
    state_df["simulation_id"] = sim_id
    state_meta["simulation_id"] = sim_id

    # 2. Sensor & confounder config (handle shifted-noise split)
    s_cfg = copy.deepcopy(sens_cfg)
    c_cfg = copy.deepcopy(conf_cfg)
    n_bursts = None
    if shifted:
        sh = ds_cfg["shifted_test"]
        s_cfg["imu"]["noise_multiplier_range"] = list(sh["noise_multiplier_range"])
        s_cfg["imu"]["node_bias_sd_deg"] = float(sh["node_bias_sd_deg"])
        c_cfg = cf.scale_config(c_cfg, float(sh["confounder_scale"]))
        n_bursts = int(sh.get("confounder_n_bursts", 2))

    C.check_abnormal_vibration_margins(ds_cfg["abnormal_vibration"], c_cfg)

    # 3. Faults / events / confounders according to conditions
    T = float(state_meta["parameters"]["duration_s"])
    fault_plan = []
    hooks = []
    use_conf = None
    if "normal" in conds:
        use_conf = c_cfg
    if "abnormal_vibration" in conds:
        abn_cfg = ds_cfg["abnormal_vibration"]
        hooks.append(lambda buf, s=seed, ac=abn_cfg: C.inject_abnormal_vibration(buf, s, ac))
    if "sensor_fault" in conds:
        fk = spec.get("sensor_fault_kind")
        fault_plan.append(C.sample_single_fault_spec(seed + 17, T, "sensor_fault", fault_kind=fk))
    if "comm_fault" in conds:
        fk = spec.get("comm_fault_kind")
        fault_plan.append(C.sample_single_fault_spec(seed + 31, T, "comm_fault", fault_kind=fk))

    wcfg = ds_cfg["windowing"]
    lead_in_s = float(wcfg.get("lead_in_baseline_s", 30.0))

    tele_df, tele_meta = S.generate_telemetry(
        state_df,
        seed=seed,
        sensor_cfg=s_cfg,
        profile=profile,
        fault_plan=fault_plan if fault_plan else None,
        confounder_cfg=use_conf,
        lead_in_end_s=lead_in_s if not has_basin else None,
        hooks=hooks,
        n_bursts=n_bursts,
    )

    # 4. Separate labels derived from state timeline + fault/event metadata
    wins = W.make_windows(T, float(wcfg["window_s"]), float(wcfg["overlap"]))
    ivs = L.intervals_from_state(state_df) + L.intervals_from_meta(tele_meta)
    labels_df = L.label_windows(
        wins,
        ivs,
        min_overlap=float(wcfg.get("min_label_overlap", 0.5)),
    )
    labels_df.insert(0, "simulation_id", sim_id)
    labels_df["split"] = split
    labels_df["run_class"] = run_class

    # 5. Windowed features (including diag_ columns)
    P = state_meta["parameters"]
    centre_xy = (P["x0"], P["y0"]) if has_basin else None
    feats_df = F.extract_window_features(
        tele_df,
        win_s=float(wcfg["window_s"]),
        overlap=float(wcfg["overlap"]),
        long_horizon_windows=int(wcfg.get("long_horizon_windows", 12)),
        lead_in_s=lead_in_s,
        impulse_peak_thresh=float(wcfg.get("impulse_peak_thresh", 0.035)),
        basin_centre_xy=centre_xy,
        mountings=state_meta.get("mountings"),
    )
    feats_df.insert(0, "simulation_id", sim_id)
    feats_df["split"] = split
    feats_df["run_class"] = run_class
    feats_df["label"] = labels_df["label"]
    feats_df["label_set"] = labels_df["label_set"]
    feats_df["t_rel_event_start_s"] = labels_df["t_rel_event_start_s"]

    # 6. SNR vs IMU noise and manifest entry
    mult = float(tele_meta["noise_multiplier"])
    imu_sd = imu_noise_sd_deg(profile, mult)
    peak_deg = float(P["peak_tilt_deg"]) if has_basin else 0.0
    snr = float(peak_deg / imu_sd) if has_basin else 0.0

    if save_run_dir is not None:
        rdir = Path(save_run_dir) / sim_id
        rdir.mkdir(parents=True, exist_ok=True)
        state_df.to_csv(rdir / "state_timeline.csv", index=False)
        tele_df.to_csv(rdir / "telemetry.csv", index=False)
        labels_df.to_csv(rdir / "window_labels.csv", index=False)
        feats_df.to_csv(rdir / "window_features.csv", index=False)
        with open(rdir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {"state": state_meta, "telemetry": tele_meta, "spec": spec},
                f,
                indent=2,
                default=float,
            )

    manifest_row = {
        "simulation_id": sim_id,
        "seed": seed,
        "split": split,
        "run_class": run_class,
        "conditions": "|".join(conds),
        "broad_basin_subset": broad,
        "shifted_noise": shifted,
        "peak_tilt_deg": peak_deg,
        "imu_noise_sd_deg": imu_sd,
        "noise_multiplier": mult,
        "snr_vs_imu_noise": snr,
        "compression_factor": float(P["compression_factor"]),
        "duration_s": T,
        "n_windows": len(labels_df),
        "n_trainable_windows": int(L.trainable(labels_df).sum()),
        "n_transition_windows": int((labels_df.label == L.TRANSITION).sum()),
        "n_faults": len(tele_meta["faults"]),
        "n_events": len(tele_meta["events"]),
        "n_confounders": len(tele_meta["confounders"]),
        "fault_types": "|".join(sorted({f["type"] for f in tele_meta["faults"]})),
        "parameters_json": json.dumps(P, default=float),
        "faults_json": json.dumps(tele_meta["faults"], default=float),
        "events_json": json.dumps(tele_meta["events"], default=float),
        "confounders_json": json.dumps(tele_meta["confounders"], default=float),
    }
    return manifest_row, labels_df, feats_df


def build_run_specs(ds_cfg: dict, scen_cfg: dict) -> list[dict]:
    """Deterministically build all run specifications across all splits with non-overlapping seeds."""
    mseed = int(ds_cfg["master_seed"])
    lo_p, hi_p = scen_cfg["ranges"]["peak_tilt_deg"]
    strat = bool(ds_cfg.get("stratification", {}).get("enabled", True))
    specs = []
    seed_cursor = mseed * 100 + 1

    def next_seeds(k: int) -> list[int]:
        nonlocal seed_cursor
        out = list(range(seed_cursor, seed_cursor + k))
        seed_cursor += k
        return out

    # 1. Headline splits: train, val, test
    for sp_idx, (split, count_per_cls) in enumerate(ds_cfg["splits"].items()):
        n = int(count_per_cls)
        for cls_idx, cls in enumerate(C.CLASSES):
            seeds = next_seeds(n)
            peaks = (
                stratified_peaks(n, lo_p, hi_p, mseed + sp_idx * 10 + cls_idx)
                if (strat and cls in ("gradual", "rapid"))
                else [None] * n
            )
            for i, s in enumerate(seeds):
                sp = {
                    "seed": s,
                    "split": split,
                    "run_class": cls,
                    "conditions": [cls],
                    "broad_basin": False,
                    "shifted_noise": False,
                    "peak_tilt_deg": peaks[i],
                }
                if cls == "sensor_fault":
                    sp["sensor_fault_kind"] = C.SENSOR_FAULT_TYPES[i % len(C.SENSOR_FAULT_TYPES)]
                elif cls == "comm_fault":
                    sp["comm_fault_kind"] = C.COMM_FAULT_TYPES[i % len(C.COMM_FAULT_TYPES)]
                specs.append(sp)

    # 2. Shifted-noise test set
    n_sh = int(ds_cfg["shifted_test"]["runs_per_class"])
    for cls_idx, cls in enumerate(C.CLASSES):
        seeds = next_seeds(n_sh)
        peaks = (
            stratified_peaks(n_sh, lo_p, hi_p, mseed + 700 + cls_idx)
            if (strat and cls in ("gradual", "rapid"))
            else [None] * n_sh
        )
        for i, s in enumerate(seeds):
            sp = {
                "seed": s,
                "split": "shifted_test",
                "run_class": cls,
                "conditions": [cls],
                "broad_basin": False,
                "shifted_noise": True,
                "peak_tilt_deg": peaks[i],
            }
            if cls == "sensor_fault":
                sp["sensor_fault_kind"] = C.SENSOR_FAULT_TYPES[i % len(C.SENSOR_FAULT_TYPES)]
            elif cls == "comm_fault":
                sp["comm_fault_kind"] = C.COMM_FAULT_TYPES[i % len(C.COMM_FAULT_TYPES)]
            specs.append(sp)

    # 3. Broad-basin subset (gradual and rapid with broad_basin_subset=True)
    n_bb = int(ds_cfg["broad_basin_subset"]["runs_per_class"])
    for cls_idx, cls in enumerate(("gradual", "rapid")):
        seeds = next_seeds(n_bb)
        peaks = (
            stratified_peaks(n_bb, lo_p, hi_p, mseed + 800 + cls_idx)
            if strat
            else [None] * n_bb
        )
        for i, s in enumerate(seeds):
            specs.append({
                "seed": s,
                "split": "broad_basin",
                "run_class": cls,
                "conditions": [cls],
                "broad_basin": True,
                "shifted_noise": False,
                "peak_tilt_deg": peaks[i],
            })

    # 4. Stress set (combined conditions)
    n_st = int(ds_cfg["stress_set"]["n_runs"])
    combos = ds_cfg["stress_set"]["combos"]
    seeds = next_seeds(n_st)
    peaks = stratified_peaks(n_st, lo_p, hi_p, mseed + 900)
    for i, s in enumerate(seeds):
        conds = list(combos[i % len(combos)])
        primary = [c for c in C.PRIORITY if c in conds][0]
        has_b = ("gradual" in conds) or ("rapid" in conds)
        specs.append({
            "seed": s,
            "split": "stress",
            "run_class": primary,
            "conditions": conds,
            "broad_basin": False,
            "shifted_noise": False,
            "peak_tilt_deg": peaks[i] if has_b else None,
            "sensor_fault_kind": C.SENSOR_FAULT_TYPES[i % len(C.SENSOR_FAULT_TYPES)],
            "comm_fault_kind": C.COMM_FAULT_TYPES[i % len(C.COMM_FAULT_TYPES)],
        })

    return specs


def build_datasets(
    ds_cfg: dict,
    scen_cfg: dict,
    sens_cfg: dict,
    conf_cfg: dict,
    profile: dict,
    out_dir: str | Path | None = None,
    save_per_run_files: bool = False,
    config_files: dict | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate the full dataset suite and optionally write manifest, labels, features and metadata.

    Whenever `out_dir` is given a `metadata.json` (content hashes, config contents + hashes, git
    commit/dirty flag, library versions, creation time) is also written (see `versioning.py`).
    """
    specs = build_run_specs(ds_cfg, scen_cfg)
    manifest_rows = []
    label_frames = []
    feat_frames = []
    run_store = (Path(out_dir) / "runs") if (out_dir and save_per_run_files) else None

    for sp in specs:
        mrow, ldf, fdf = generate_one_run(
            sp, scen_cfg, sens_cfg, conf_cfg, ds_cfg, profile, save_run_dir=run_store
        )
        manifest_rows.append(mrow)
        label_frames.append(ldf)
        feat_frames.append(fdf)

    manifest = pd.DataFrame(manifest_rows)
    labels_all = pd.concat(label_frames, ignore_index=True)
    features_all = pd.concat(feat_frames, ignore_index=True)

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out / "manifest.csv", index=False)
        labels_all.to_csv(out / "window_labels.csv", index=False)
        features_all.to_csv(out / "window_features.csv", index=False)
        for sp_name, g in features_all.groupby("split", sort=False):
            g.to_csv(out / f"features_{sp_name}.csv", index=False)
        for sp_name, g in labels_all.groupby("split", sort=False):
            g.to_csv(out / f"labels_{sp_name}.csv", index=False)
        meta_records = {
            r["simulation_id"]: {
                "split": r["split"],
                "run_class": r["run_class"],
                "conditions": r["conditions"].split("|"),
                "peak_tilt_deg": r["peak_tilt_deg"],
                "snr_vs_imu_noise": r["snr_vs_imu_noise"],
                "noise_multiplier": r["noise_multiplier"],
                "parameters": json.loads(r["parameters_json"]),
                "faults": json.loads(r["faults_json"]),
                "events": json.loads(r["events_json"]),
                "confounders": json.loads(r["confounders_json"]),
            }
            for r in manifest_rows
        }
        with open(out / "runs_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_records, f, indent=2)
        from .versioning import write_dataset_metadata
        write_dataset_metadata(out, config_files)

    return {"manifest": manifest, "labels": labels_all, "features": features_all}
