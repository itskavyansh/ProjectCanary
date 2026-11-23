"""Installation-yaw sensitivity evaluation script (SYNTHETIC / SIMULATION).

Measures how much the frozen v3 models (`outputs/eval_v3/trained_models_v3.pkl`) and frozen v3
operating points (`outputs/eval_v3/operating_points_v3.pkl`) degrade when the nominal installation
yaw (`yaw_deg`) used ONLY by the world-frame feature rotation is perturbed by:
  - `sd_0deg` (0 deg SD — reproduces `outputs/eval_v3/` 100% identically)
  - `sd_5deg` (zero-mean Gaussian, sd = 5 deg per node)
  - `sd_10deg` (zero-mean Gaussian, sd = 10 deg per node)
  - `sd_20deg` (zero-mean Gaussian, sd = 20 deg per node)
  - `single_node_45deg` (one node per run has a fixed +/- 45 deg error)

Also compares against the yaw-free `v2_baseline` (old features from `outputs/datasets/` scored with
frozen `outputs/eval/trained_models.pkl` and `outputs/eval_v2/operating_points_v2.pkl`) on the
exact same runs (`test`, `shifted_test`, `stress`, `broad_basin`), for all SNR and separately for
`SNR > 2`.

Writes all results to `outputs/eval_v3_yaw/`. Never modifies any frozen file or model.
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from canary_sim import (
    alarm as A,
    confounders as cf,
    dataset as ds,
    engine,
    evaluate as ev,
    features as F,
    labels as L,
    sensors as S,
    yaw_sensitivity as ys,
)

BANNER = "SYNTHETIC / SIMULATION"
EVAL_SPLITS = ("test", "shifted_test", "stress", "broad_basin")
MODEL_NAMES = ("threshold_rule", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "threshold_rule": "Threshold Rule (v3)",
    "logistic_regression": "Logistic Regression (v3)",
    "random_forest": "Random Forest (v3)",
}


def stamp_banner(fig, comp_grad: float = 360.0, comp_rap: float = 1.0, subtitle: str = "") -> None:
    fig.suptitle(
        f"{BANNER}  |  time compression: gradual x{comp_grad:g} (240 s = 24 h), rapid x{comp_rap:g} (real-time)\n{subtitle}",
        color="red",
        fontsize=11,
        fontweight="bold",
    )


def _df_to_md(df_in: pd.DataFrame) -> str:
    cols_h = [str(c) for c in df_in.columns]
    lines = [
        "| " + " | ".join(cols_h) + " |",
        "| " + " | ".join(["---"] * len(cols_h)) + " |",
    ]
    for _, r in df_in.iterrows():
        vals = []
        for v in r:
            if isinstance(v, float):
                vals.append("NaN" if np.isnan(v) else f"{v:g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def save_table(df_tab: pd.DataFrame, out_dir: Path, name: str) -> None:
    df_tab.to_csv(out_dir / f"{name}.csv", index=False)
    with open(out_dir / f"{name}.md", "w", encoding="utf-8") as f:
        f.write(f"# {name} (SYNTHETIC / SIMULATION)\n\n")
        f.write(_df_to_md(df_tab))
        f.write("\n")


def score_condition_df(
    df_sp: pd.DataFrame,
    w_preds: np.ndarray,
    k: int = 3,
    m: int = 3,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    """Compute window macro-F1, alarm macro-F1, FA episodes/normal run, run exact detect rate,
    and gradual <-> sensor_fault confusion counts (with run-bootstrap 95% CIs).
    """
    w_res = ev.evaluate_window_level_with_run_bootstrap(df_sp, w_preds, n_boot=n_boot, seed=seed)
    al_res = A.evaluate_alarm_level(df_sp, w_preds, k=k, m=m, n_boot=n_boot, seed=seed)
    al_w_res = ev.evaluate_window_level_with_run_bootstrap(
        df_sp, al_res["alarm_preds"], n_boot=n_boot, seed=seed
    )

    tr_mask = L.trainable(df_sp).to_numpy()
    sub = df_sp.loc[tr_mask, ["simulation_id", "label"]].copy()
    p_w = np.asarray(w_preds)[tr_mask]
    g_to_sf_arr = ((sub["label"] == "gradual") & (p_w == "sensor_fault")).to_numpy(dtype=int)
    sf_to_g_arr = ((sub["label"] == "sensor_fault") & (p_w == "gradual")).to_numpy(dtype=int)
    tot_conf_arr = g_to_sf_arr + sf_to_g_arr

    # Bootstrap CI over runs for gradual <-> sensor_fault confusion count
    run_ids = sub["simulation_id"].unique()
    n_runs = len(run_ids)
    r_map = {rid: i for i, rid in enumerate(run_ids)}
    r_codes = np.array([r_map[rid] for rid in sub["simulation_id"]], dtype=int)
    conf_by_run = np.zeros(n_runs, dtype=int)
    np.add.at(conf_by_run, r_codes, tot_conf_arr)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 955]))
    boot_conf = [int(conf_by_run[rng.integers(0, n_runs, size=n_runs)].sum()) for _ in range(n_boot)]

    return {
        "n_runs": w_res["n_runs"],
        "n_windows": w_res["n_windows"],
        "window_macro_f1": round(w_res["macro_f1"], 4),
        "window_macro_f1_ci_lo": round(w_res["macro_f1_ci95"][0], 4),
        "window_macro_f1_ci_hi": round(w_res["macro_f1_ci95"][1], 4),
        "alarm_macro_f1": round(al_w_res["macro_f1"], 4),
        "alarm_macro_f1_ci_lo": round(al_w_res["macro_f1_ci95"][0], 4),
        "alarm_macro_f1_ci_hi": round(al_w_res["macro_f1_ci95"][1], 4),
        "fa_episodes_per_normal_run": round(al_res["fa_episodes_per_normal_run"], 3),
        "fa_episodes_ci_lo": round(al_res["fa_episodes_ci95"][0], 3),
        "fa_episodes_ci_hi": round(al_res["fa_episodes_ci95"][1], 3),
        "fa_windows_per_normal_run": round(al_res["fa_windows_per_normal_run"], 3),
        "run_exact_detect_rate": round(al_res["exact_detect_rate"], 4),
        "run_exact_detect_ci_lo": round(al_res["exact_detect_ci95"][0], 4),
        "run_exact_detect_ci_hi": round(al_res["exact_detect_ci95"][1], 4),
        "gradual_to_sf_confusions": int(g_to_sf_arr.sum()),
        "sf_to_gradual_confusions": int(sf_to_g_arr.sum()),
        "gradual_sf_confusions": int(tot_conf_arr.sum()),
        "gradual_sf_confusions_ci_lo": float(np.percentile(boot_conf, 2.5)),
        "gradual_sf_confusions_ci_hi": float(np.percentile(boot_conf, 97.5)),
    }


def plot_yaw_sensitivity(
    all_snr_df: pd.DataFrame,
    snr2_df: pd.DataFrame,
    out_dir: Path,
    comp_grad: float,
    comp_rap: float,
) -> Path:
    """Plot metric vs yaw error (with 95% run-bootstrap CIs and yaw-free v2 baseline reference)."""
    cond_order = ["sd_0deg", "sd_5deg", "sd_10deg", "sd_20deg", "single_node_45deg"]
    x_labels = ["0° (nominal)", "5° SD", "10° SD", "20° SD", "1-node 45°"]
    x = np.arange(len(cond_order))
    colors = {"threshold_rule": "tab:orange", "logistic_regression": "tab:green", "random_forest": "tab:blue"}
    markers = {"threshold_rule": "s", "logistic_regression": "^", "random_forest": "o"}

    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10))

    # Top row: Test (SNR > 2) Alarm Macro-F1, Gradual<->Sensor_Fault Confusions (Raw RF/TH/LR), and Broad-Basin Run Detect Rate
    panels = [
        (axes[0, 0], snr2_df, "test", "episode_fa<=1 (k=3,m=3)", "alarm_macro_f1", "alarm_macro_f1_ci_lo", "alarm_macro_f1_ci_hi", "Test (SNR > 2): Persisted Alarm Macro-F1", "Alarm Macro-F1 (95% CI)"),
        (axes[0, 1], snr2_df, "test", "raw (k=1)", "gradual_sf_confusions", "gradual_sf_confusions_ci_lo", "gradual_sf_confusions_ci_hi", "Test (SNR > 2): Gradual <-> Sensor_Fault Confusions", "Confusion window count (95% CI)"),
        (axes[0, 2], all_snr_df, "broad_basin", "episode_fa<=1 (k=3,m=3)", "run_exact_detect_rate", "run_exact_detect_ci_lo", "run_exact_detect_ci_hi", "Broad-Basin: Run Exact Alarm Detect Rate", "Run exact detect rate (95% CI)"),
        (axes[1, 0], all_snr_df, "shifted_test", "episode_fa<=1 (k=3,m=3)", "alarm_macro_f1", "alarm_macro_f1_ci_lo", "alarm_macro_f1_ci_hi", "Shifted-Test (All SNR): Alarm Macro-F1", "Alarm Macro-F1 (95% CI)"),
        (axes[1, 1], all_snr_df, "stress", "episode_fa<=1 (k=3,m=3)", "alarm_macro_f1", "alarm_macro_f1_ci_lo", "alarm_macro_f1_ci_hi", "Stress Set (Combined): Alarm Macro-F1", "Alarm Macro-F1 (95% CI)"),
        (axes[1, 2], all_snr_df, "test", "episode_fa<=1 (k=3,m=3)", "fa_episodes_per_normal_run", "fa_episodes_ci_lo", "fa_episodes_ci_hi", "Test: False-Alarm Episodes / Normal Run", "FA episodes / normal run (95% CI)"),
    ]

    for ax, df_src, sp_name, reg_name, col_y, col_lo, col_hi, ttl, y_lbl in panels:
        sub = df_src[(df_src["split"] == sp_name) & (df_src["regime"] == reg_name)]
        for off_i, mname in enumerate(MODEL_NAMES):
            m_v3 = sub[(sub["model"] == mname) & (sub["condition"].isin(cond_order))].set_index("condition").reindex(cond_order)
            y = m_v3[col_y].to_numpy(dtype=float)
            lo = m_v3[col_lo].to_numpy(dtype=float)
            hi = m_v3[col_hi].to_numpy(dtype=float)
            dx = (off_i - 1) * 0.08
            ax.errorbar(
                x + dx,
                y,
                yerr=np.vstack([np.maximum(0, y - lo), np.maximum(0, hi - y)]),
                marker=markers[mname],
                color=colors[mname],
                lw=1.8,
                capsize=3,
                label=MODEL_LABELS[mname],
            )
            # Plot horizontal dashed reference for yaw-free v2 RF baseline
            if mname == "random_forest":
                v2_row = sub[(sub["model"] == "random_forest") & (sub["condition"] == "v2_yaw_free_baseline")]
                if len(v2_row) > 0:
                    v2_val = float(v2_row[col_y].iloc[0])
                    ax.axhline(v2_val, color="navy", ls="--", lw=1.3, alpha=0.75, label=f"v2 Yaw-Free RF ({v2_val:.3g})")
                    # Shade 95% CI of the 0-deg v3 RF result
                    ax.axhspan(lo[0], hi[0], color="tab:blue", alpha=0.08)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=18, ha="right", fontsize=8.5)
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.set_title(ttl, fontsize=10, fontweight="bold")
        ax.text(0.5, 1.08, BANNER, transform=ax.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="best")

    stamp_banner(
        fig,
        comp_grad,
        comp_rap,
        "Installation-Yaw Error Sensitivity of Frozen v3 Models vs Yaw-Free v2 Baseline (with 95% Run-Bootstrap CIs)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    p = out_dir / "plot_yaw_error_sensitivity.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-v1", default=str(ROOT / "outputs" / "datasets"))
    ap.add_argument("--data-v3", default=str(ROOT / "outputs" / "datasets_v3"))
    ap.add_argument("--eval-v1", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--eval-v2", default=str(ROOT / "outputs" / "eval_v2"))
    ap.add_argument("--eval-v3", default=str(ROOT / "outputs" / "eval_v3"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_v3_yaw"))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--n-boot", type=int, default=300)
    a = ap.parse_args()

    v1_data_dir = Path(a.data_v1)
    v3_data_dir = Path(a.data_v3)
    v1_eval_dir = Path(a.eval_v1)
    v2_eval_dir = Path(a.eval_v2)
    v3_eval_dir = Path(a.eval_v3)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record SHA-256 of frozen v3 models & operating points before and after to prove zero refitting
    v3_models_path = v3_eval_dir / "trained_models_v3.pkl"
    v3_ops_path = v3_eval_dir / "operating_points_v3.pkl"
    sha_models_before = hashlib.sha256(v3_models_path.read_bytes()).hexdigest()
    sha_ops_before = hashlib.sha256(v3_ops_path.read_bytes()).hexdigest()

    with open(v3_models_path, "rb") as f:
        models_v3 = pickle.load(f)
    with open(v3_ops_path, "rb") as f:
        ops_v3 = pickle.load(f)
    with open(v1_eval_dir / "trained_models.pkl", "rb") as f:
        models_v2_base = pickle.load(f)
    with open(v2_eval_dir / "operating_points_v2.pkl", "rb") as f:
        ops_v2 = pickle.load(f)

    manifest = pd.read_csv(v3_data_dir / "manifest.csv")
    comp_grad = float(manifest[manifest.run_class == "gradual"]["compression_factor"].iloc[0])
    comp_rap = float(manifest[manifest.run_class == "rapid"]["compression_factor"].iloc[0])

    snr_map = dict(zip(manifest["simulation_id"], manifest["snr_vs_imu_noise"]))
    rc_map = dict(zip(manifest["simulation_id"], manifest["run_class"]))
    snr2_keep_runs = {
        sid
        for sid, rc in rc_map.items()
        if (rc not in ("gradual", "rapid")) or (snr_map.get(sid, 0.0) > 2.0)
    }

    # 1. Extract unperturbed sensor-frame window vectors (vx_win, vy_win) and mountings for all 350 eval runs
    ds_cfg = ds.load_dataset_config(ROOT / "configs" / "dataset_default.yaml")
    scen_cfg = engine.load_config(ROOT / "configs" / "scenario_default.yaml")
    sens_cfg = S.load_sensor_config(ROOT / "configs" / "sensor_default.yaml")
    conf_cfg = cf.load_confounder_config(ROOT / "configs" / "confounder_default.yaml")
    profile = S.load_noise_profile(ROOT / "configs" / "noise_profile.json")

    eval_specs = [s for s in ds.build_run_specs(ds_cfg, scen_cfg) if s["split"] in EVAL_SPLITS]
    splits_v3_orig = {sp: pd.read_csv(v3_data_dir / f"features_{sp}.csv") for sp in EVAL_SPLITS}
    splits_v1_orig = {sp: pd.read_csv(v1_data_dir / f"features_{sp}.csv") for sp in EVAL_SPLITS}

    print(f"Precomputing sensor-frame window vectors for {len(eval_specs)} evaluation runs...")
    run_cache: dict[str, tuple[dict, pd.DataFrame, np.ndarray, np.ndarray, dict]] = {}
    for sp_spec in eval_specs:
        sim_id = f"sim_{sp_spec['split']}_{sp_spec['run_class']}_{sp_spec['seed']}"
        df_sp_orig = splits_v3_orig[sp_spec["split"]]
        base_f = df_sp_orig[df_sp_orig["simulation_id"] == sim_id].reset_index(drop=True)
        vx, vy, mounts, _ = ys.extract_sensor_frame_window_vectors_for_spec(
            sp_spec, scen_cfg, sens_cfg, conf_cfg, ds_cfg, profile
        )
        run_cache[sim_id] = (sp_spec, base_f, vx, vy, mounts)

    # 2. Build perturbed feature tables for each yaw condition & record every node's yaw error in manifest
    yaw_manifest_rows = []
    all_snr_rows = []
    snr2_rows = []

    for cond in ys.YAW_CONDITIONS:
        cname = cond["name"]
        print(f"Evaluating yaw condition: {cname} ({cond['label']})...")
        perturbed_by_split: dict[str, list[pd.DataFrame]] = {sp: [] for sp in EVAL_SPLITS}

        for sim_id, (sp_spec, base_f, vx, vy, mounts) in run_cache.items():
            errs = ys.sample_nominal_yaw_perturbation(sp_spec["seed"], cond, master_seed=a.seed)
            yaw_manifest_rows.append({
                "condition": cname,
                "simulation_id": sim_id,
                "split": sp_spec["split"],
                "run_class": sp_spec["run_class"],
                "run_seed": int(sp_spec["seed"]),
                "yaw_err_NODE_01_deg": round(errs["NODE_01"], 4),
                "yaw_err_NODE_02_deg": round(errs["NODE_02"], 4),
                "yaw_err_NODE_03_deg": round(errs["NODE_03"], 4),
                "yaw_err_NODE_04_deg": round(errs["NODE_04"], 4),
                "rms_yaw_err_deg": round(float(np.sqrt(np.mean([v**2 for v in errs.values()]))), 4),
            })
            f_pert = ys.recompute_yaw_dependent_features_for_run(base_f, vx, vy, mounts, errs)
            perturbed_by_split[sp_spec["split"]].append(f_pert)

        for sp in EVAL_SPLITS:
            df_pert = pd.concat(perturbed_by_split[sp], ignore_index=True)
            if cname == "sd_0deg":
                # Verify 0 deg reproduces outputs/datasets_v3 exactly
                mcols = F.feature_columns(df_pert)
                pd.testing.assert_frame_equal(
                    df_pert[mcols], splits_v3_orig[sp][mcols], check_exact=False, rtol=1e-12, atol=1e-12
                )

            df_pert_s2 = df_pert[df_pert["simulation_id"].isin(snr2_keep_runs)].reset_index(drop=True)

            for mname in MODEL_NAMES:
                raw_p = models_v3[mname].predict(df_pert)
                ep_p = raw_p if mname == "threshold_rule" else ops_v3["episode_fa_le_1"][mname].predict_window(df_pert)
                win_p = raw_p if mname == "threshold_rule" else ops_v3["strict_window_fa_le_1"][mname].predict_window(df_pert)

                for reg_lbl, w_p, kv, mv in [
                    ("raw (k=1)", raw_p, 1, 1),
                    ("episode_fa<=1 (k=3,m=3)", ep_p, 3, 3),
                    ("strict_window_fa<=1", win_p, 5 if mname == "threshold_rule" else 3, 2),
                ]:
                    res_all = score_condition_df(df_pert, w_p, k=kv, m=mv, n_boot=a.n_boot, seed=a.seed)
                    all_snr_rows.append({
                        "split": sp,
                        "condition": cname,
                        "yaw_sd_deg": cond["sd_deg"],
                        "regime": reg_lbl,
                        "model": mname,
                        "model_version": "v3_frozen",
                        **res_all,
                    })

                    if sp in ("test", "shifted_test", "broad_basin"):
                        s2_mask = df_pert["simulation_id"].isin(snr2_keep_runs).to_numpy()
                        res_s2 = score_condition_df(
                            df_pert_s2, w_p[s2_mask], k=kv, m=mv, n_boot=a.n_boot, seed=a.seed
                        )
                        snr2_rows.append({
                            "split": sp,
                            "condition": cname,
                            "yaw_sd_deg": cond["sd_deg"],
                            "regime": reg_lbl,
                            "model": mname,
                            "model_version": "v3_frozen",
                            **res_s2,
                        })

    # 3. Also score the yaw-free `v2_yaw_free_baseline` (old features + frozen v1/v2 models) on the exact same splits
    for sp in EVAL_SPLITS:
        df1 = splits_v1_orig[sp]
        df1_s2 = df1[df1["simulation_id"].isin(snr2_keep_runs)].reset_index(drop=True)
        for mname in MODEL_NAMES:
            raw_p = models_v2_base[mname].predict(df1)
            ep_p = raw_p if mname == "threshold_rule" else ops_v2["episode_fa_le_1"][mname].predict_window(df1)
            win_p = raw_p if mname == "threshold_rule" else ops_v2["strict_window_fa_le_1"][mname].predict_window(df1)

            for reg_lbl, w_p, kv, mv in [
                ("raw (k=1)", raw_p, 1, 1),
                ("episode_fa<=1 (k=3,m=3)", ep_p, 3, 3),
                ("strict_window_fa<=1", win_p, 5 if mname == "threshold_rule" else 3, 2),
            ]:
                res_all = score_condition_df(df1, w_p, k=kv, m=mv, n_boot=a.n_boot, seed=a.seed)
                all_snr_rows.append({
                    "split": sp,
                    "condition": "v2_yaw_free_baseline",
                    "yaw_sd_deg": np.nan,
                    "regime": reg_lbl,
                    "model": mname,
                    "model_version": "v2_frozen_yaw_free",
                    **res_all,
                })
                if sp in ("test", "shifted_test", "broad_basin"):
                    s2_mask = df1["simulation_id"].isin(snr2_keep_runs).to_numpy()
                    res_s2 = score_condition_df(
                        df1_s2, w_p[s2_mask], k=kv, m=mv, n_boot=a.n_boot, seed=a.seed
                    )
                    snr2_rows.append({
                        "split": sp,
                        "condition": "v2_yaw_free_baseline",
                        "yaw_sd_deg": np.nan,
                        "regime": reg_lbl,
                        "model": mname,
                        "model_version": "v2_frozen_yaw_free",
                        **res_s2,
                    })

    assert hashlib.sha256(v3_models_path.read_bytes()).hexdigest() == sha_models_before
    assert hashlib.sha256(v3_ops_path.read_bytes()).hexdigest() == sha_ops_before

    yaw_manifest_df = pd.DataFrame(yaw_manifest_rows)
    all_snr_df = pd.DataFrame(all_snr_rows)
    snr2_df = pd.DataFrame(snr2_rows)

    yaw_manifest_df.to_csv(out_dir / "yaw_perturbation_manifest.csv", index=False)
    save_table(all_snr_df, out_dir, "yaw_sensitivity_all_snr")
    save_table(snr2_df, out_dir, "yaw_sensitivity_snr_gt_2")

    # Save one separate table per condition as requested
    for cond in [*[c["name"] for c in ys.YAW_CONDITIONS], "v2_yaw_free_baseline"]:
        sub_c = all_snr_df[all_snr_df["condition"] == cond]
        save_table(sub_c, out_dir, f"table_{cond}_all_snr")
        sub_c_s2 = snr2_df[snr2_df["condition"] == cond]
        save_table(sub_c_s2, out_dir, f"table_{cond}_snr_gt_2")

    p_plot = plot_yaw_sensitivity(all_snr_df, snr2_df, out_dir, comp_grad, comp_rap)

    print("=== YAW SENSITIVITY ON SNR > 2 (EPISODE_FA<=1 & RAW) ===")
    print(snr2_df[snr2_df["regime"].isin(["episode_fa<=1 (k=3,m=3)", "raw (k=1)"])].to_string(index=False))
    print("\n=== YAW SENSITIVITY ON ALL SNR (EPISODE_FA<=1) ===")
    print(all_snr_df[all_snr_df["regime"] == "episode_fa<=1 (k=3,m=3)"].to_string(index=False))
    print("\nSaved plot:", p_plot)


if __name__ == "__main__":
    main()
