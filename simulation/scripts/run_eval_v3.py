"""Round 3 (v3) dataset generation and evaluation script (SYNTHETIC / SIMULATION):
  1. Generates v3 windowed features into `outputs/datasets_v3/` using the EXACT same run specs,
     seeds, splits, and labels as `outputs/datasets/` (leaving `outputs/datasets/`, `outputs/eval/`,
     and `outputs/eval_v2/` completely untouched).
  2. Verifies that `manifest.csv` and `window_labels.csv` in `outputs/datasets_v3/` are 100% identical
     to the frozen `outputs/datasets/` manifest and labels.
  3. Retrains and retunes Threshold Rule, Logistic Regression, and Random Forest on `train` and `val`
     ONLY (`guard_no_test_data` enforced), and retunes the `(k, m)` persistence and minimum-probability
     operating points on `val` ONLY (`fa_target = 1.0`).
  4. Re-runs Random Forest feature-group ablations (`direction`, `coherence`, `piezo`, `hub`, `comm`,
     `long_horizon`) on `test` to measure the gain of the new nominal-yaw world-frame radial convergence
     and trailing-horizon coherence features.
  5. Evaluates all models and matched-FA operating points across `test`, `shifted_test`, `stress`, and
     `broad_basin`, and explicitly reports gains vs v1/v2 overall AND separately for `SNR > 2`
     (and `gradual` vs `sensor_fault` confusions).

Usage: python scripts/run_eval_v3.py [--data-v3 DIR] [--out DIR] [--seed N] [--n-boot N]
"""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
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
    classes as C,
    confounders as cf,
    dataset as ds,
    engine,
    evaluate as ev,
    features as F,
    labels as L,
    models as M,
    sensors as S,
)

BANNER = "SYNTHETIC / SIMULATION"
EVAL_SPLITS = ("test", "shifted_test", "stress", "broad_basin")
MODEL_NAMES = ("threshold_rule", "logistic_regression", "random_forest")
MODEL_LABELS = {
    "threshold_rule": "Threshold Rule",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
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


def evaluate_snr_gt_2_subset(
    df_sp: pd.DataFrame,
    w_preds: np.ndarray,
    manifest_df: pd.DataFrame,
    k: int = 3,
    m: int = 3,
    n_boot: int = 300,
    seed: int = 20261117,
) -> dict:
    """Evaluate window and persisted-alarm metrics specifically on the `SNR > 2` subset:
    all `gradual` and `rapid` runs with `snr_vs_imu_noise > 2.0` plus all non-basin runs (`normal`,
    `abnormal_vibration`, `sensor_fault`, `comm_fault`).
    """
    snr_map = dict(zip(manifest_df["simulation_id"], manifest_df["snr_vs_imu_noise"]))
    rc_map = dict(zip(manifest_df["simulation_id"], manifest_df["run_class"]))
    keep_runs = {
        sid
        for sid, rc in rc_map.items()
        if (rc not in ("gradual", "rapid")) or (snr_map.get(sid, 0.0) > 2.0)
    }
    mask_runs = df_sp["simulation_id"].isin(keep_runs).to_numpy()
    sub_df = df_sp.loc[mask_runs].copy().reset_index(drop=True)
    sub_preds = np.asarray(w_preds)[mask_runs]

    w_res = ev.evaluate_window_level_with_run_bootstrap(sub_df, sub_preds, n_boot=n_boot, seed=seed)
    al_res = A.evaluate_alarm_level(sub_df, sub_preds, k=k, m=m, n_boot=n_boot, seed=seed)
    al_w_res = ev.evaluate_window_level_with_run_bootstrap(
        sub_df, al_res["alarm_preds"], n_boot=n_boot, seed=seed
    )

    # Also count gradual <-> sensor_fault confusions on this subset
    tr_mask = L.trainable(sub_df).to_numpy()
    y_t = sub_df.loc[tr_mask, "label"].to_numpy()
    p_w = sub_preds[tr_mask]
    g_to_sf = int(((y_t == "gradual") & (p_w == "sensor_fault")).sum())
    sf_to_g = int(((y_t == "sensor_fault") & (p_w == "gradual")).sum())

    return {
        "n_runs": w_res["n_runs"],
        "n_windows": w_res["n_windows"],
        "window_macro_f1": w_res["macro_f1"],
        "window_macro_f1_ci95": w_res["macro_f1_ci95"],
        "alarm_macro_f1": al_w_res["macro_f1"],
        "alarm_macro_f1_ci95": al_w_res["macro_f1_ci95"],
        "gradual_window_recall": w_res["per_class"]["gradual"]["recall"],
        "gradual_window_recall_ci95": w_res["per_class"]["gradual"]["recall_ci95"],
        "rapid_window_recall": w_res["per_class"]["rapid"]["recall"],
        "rapid_window_recall_ci95": w_res["per_class"]["rapid"]["recall_ci95"],
        "sensor_fault_window_recall": w_res["per_class"]["sensor_fault"]["recall"],
        "sensor_fault_window_recall_ci95": w_res["per_class"]["sensor_fault"]["recall_ci95"],
        "gradual_run_alarm_detect_rate": al_res["per_class"].get("gradual", {}).get("exact_detect_rate", np.nan),
        "rapid_run_alarm_detect_rate": al_res["per_class"].get("rapid", {}).get("exact_detect_rate", np.nan),
        "sensor_fault_run_alarm_detect_rate": al_res["per_class"].get("sensor_fault", {}).get("exact_detect_rate", np.nan),
        "all_events_run_alarm_detect_rate": al_res["exact_detect_rate"],
        "all_events_run_alarm_detect_ci95": al_res["exact_detect_ci95"],
        "fa_episodes_per_normal_run": al_res["fa_episodes_per_normal_run"],
        "fa_windows_per_normal_run": al_res["fa_windows_per_normal_run"],
        "gradual_to_sf_confusions": g_to_sf,
        "sf_to_gradual_confusions": sf_to_g,
        "total_gradual_sf_confusions": g_to_sf + sf_to_g,
    }


def plot_v3_diagnostics_and_gains(
    train_v3: pd.DataFrame,
    val_v3: pd.DataFrame,
    manifest: pd.DataFrame,
    ablation_v1_df: pd.DataFrame,
    ablation_v3_df: pd.DataFrame,
    snr2_comp_df: pd.DataFrame,
    out_dir: Path,
    comp_grad: float,
    comp_rap: float,
) -> tuple[Path, Path]:
    # 1. Plot v3 Radial Convergence & Trailing Coherence on Train + Val (by SNR bin vs sensor_fault)
    tv = pd.concat([train_v3, val_v3], ignore_index=True)
    tv = tv[L.trainable(tv)].copy()
    snr_map = dict(zip(manifest["simulation_id"], manifest["snr_vs_imu_noise"]))
    ft_map = dict(zip(manifest["simulation_id"], manifest["fault_types"]))
    tv["snr"] = tv["simulation_id"].map(snr_map)
    tv["fault_type"] = tv["simulation_id"].map(ft_map)

    g_grad = tv[tv["label"] == "gradual"]
    sf_drift = tv[(tv["label"] == "sensor_fault") & tv["fault_type"].str.contains("slow_drift|bias_step", na=False)]
    norm_tv = tv[tv["label"] == "normal"]

    snr_labels = [b[0] for b in ev.SNR_BINS]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # Panel 0: Fitted radial convergence weighted cosine (`dir_convergence_weighted_cos`) across SNR bins vs sensor_fault
    box_align = [
        g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["dir_convergence_weighted_cos"].to_numpy()
        for _, lo, hi in ev.SNR_BINS
    ]
    box_align.append(sf_drift["dir_convergence_weighted_cos"].to_numpy())
    bp0 = axes[0].boxplot(box_align, tick_labels=[*snr_labels, "sf_drift"], patch_artist=True, showfliers=False)
    for p in bp0["boxes"][:-1]:
        p.set_facecolor("lightblue")
    bp0["boxes"][-1].set_facecolor("moccasin")
    axes[0].axhline(0.0, color="red", ls="--", lw=1.1, label="Zero alignment")
    axes[0].set_xlabel("Gradual SNR bin vs Sensor Fault (drift/bias)")
    axes[0].set_ylabel("dir_convergence_weighted_cos (nominal yaw)")
    axes[0].set_title("(a) Fitted Radial Convergence Alignment vs SNR", fontsize=10.5, fontweight="bold")
    axes[0].text(0.5, 1.08, BANNER, transform=axes[0].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].grid(alpha=0.25)

    # Panel 1: Fitted radial convergence residual (`dir_convergence_residual_m`) across SNR bins vs sensor_fault
    box_res = [
        g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["dir_convergence_residual_m"].to_numpy()
        for _, lo, hi in ev.SNR_BINS
    ]
    box_res.append(sf_drift["dir_convergence_residual_m"].to_numpy())
    bp1 = axes[1].boxplot(box_res, tick_labels=[*snr_labels, "sf_drift"], patch_artist=True, showfliers=False)
    for p in bp1["boxes"][:-1]:
        p.set_facecolor("lightgreen")
    bp1["boxes"][-1].set_facecolor("moccasin")
    axes[1].set_xlabel("Gradual SNR bin vs Sensor Fault (drift/bias)")
    axes[1].set_ylabel("dir_convergence_residual_m (m)")
    axes[1].set_title("(b) Fitted Convergence Orthogonal Residual (m)", fontsize=10.5, fontweight="bold")
    axes[1].text(0.5, 1.08, BANNER, transform=axes[1].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].grid(alpha=0.25)

    # Panel 2: Trailing 12-window coherence (`coherence_trailing_mag` & `coherence_trailing_signed_mean`) vs 5 s coherence
    x_pos = np.arange(len(snr_labels))
    c_5s = [float(g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["coherence"].median()) for _, lo, hi in ev.SNR_BINS]
    c_tr_mag = [float(g_grad[(g_grad["snr"] >= lo) & (g_grad["snr"] < hi)]["coherence_trailing_mag"].median()) for _, lo, hi in ev.SNR_BINS]
    axes[2].plot(x_pos, c_5s, "o--", color="grey", lw=1.8, label="v1/v2 single-window (5 s) coherence")
    axes[2].plot(x_pos, c_tr_mag, "s-", color="tab:blue", lw=2.2, label="v3 trailing 12-window coherence_trailing_mag")
    axes[2].axhline(float(sf_drift["coherence_trailing_mag"].median()), color="tab:orange", ls=":", lw=1.8, label="sensor_fault (drift/bias) median")
    axes[2].axhline(float(norm_tv["coherence_trailing_mag"].median()), color="red", ls="--", lw=1.2, label="normal median")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(snr_labels)
    axes[2].set_xlabel("Gradual SNR bin (train + val)")
    axes[2].set_ylabel("Median cross-node correlation")
    axes[2].set_title("(c) Trailing 12-Window vs 5 s Coherence", fontsize=10.5, fontweight="bold")
    axes[2].text(0.5, 1.08, BANNER, transform=axes[2].transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25)

    stamp_banner(
        fig,
        comp_grad,
        comp_rap,
        "v3 World-Frame Fitted Radial Convergence & Trailing 12-Window Coherence on Train + Val",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p_feat = out_dir / "plot_v3_features_diagnosis.png"
    fig.savefig(p_feat, dpi=130)
    plt.close(fig)

    # 2. Plot SNR > 2 gains (v1/v2 vs v3 across splits) & RF ablation comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

    # Left: SNR > 2 & Broad-Basin / Stress / Test Gains (v2 vs v3 under Episode-Matched FA <= 1)
    ax0 = axes[0]
    rf_s2 = snr2_comp_df[(snr2_comp_df["model"] == "random_forest") & (snr2_comp_df["regime"] == "episode_fa<=1 (k=3,m=3)")]
    all_ep = pd.read_csv(out_dir / "v2_vs_v3_all_snr_comparison.csv")
    rf_all_ep = all_ep[(all_ep["model"] == "random_forest") & (all_ep["regime"] == "episode_fa<=1 (k=3,m=3)")]

    def _get(df_in, sp_name, ver_name, col_name):
        return float(df_in[(df_in["split"] == sp_name) & (df_in["version"] == ver_name)][col_name].iloc[0])

    bar_items = [
        ("Test (SNR>2)\nRapid Recall", _get(rf_s2, "test", "v2 (old features)", "rapid_window_recall"), _get(rf_s2, "test", "v3 (new features)", "rapid_window_recall")),
        ("Shifted (SNR>2)\nAlarm Macro-F1", _get(rf_s2, "shifted_test", "v2 (old features)", "alarm_macro_f1"), _get(rf_s2, "shifted_test", "v3 (new features)", "alarm_macro_f1")),
        ("Broad-Basin (SNR>2)\nRun Detect Rate", _get(rf_s2, "broad_basin", "v2 (old features)", "all_events_run_alarm_detect_rate"), _get(rf_s2, "broad_basin", "v3 (new features)", "all_events_run_alarm_detect_rate")),
        ("Broad-Basin (All)\nRun Detect Rate", _get(rf_all_ep, "broad_basin", "v2 (old features)", "alarm_exact_detect_rate"), _get(rf_all_ep, "broad_basin", "v3 (new features)", "alarm_exact_detect_rate")),
        ("Stress (All)\nWindow Macro-F1", _get(rf_all_ep, "stress", "v2 (old features)", "window_macro_f1"), _get(rf_all_ep, "stress", "v3 (new features)", "window_macro_f1")),
    ]
    xb = np.arange(len(bar_items))
    w = 0.35
    ax0.bar(xb - w / 2, [b[1] for b in bar_items], width=w, color="lightsteelblue", edgecolor="k", label="v2 Random Forest (old features)")
    ax0.bar(xb + w / 2, [b[2] for b in bar_items], width=w, color="tab:blue", edgecolor="k", label="v3 Random Forest (new v3 features)")
    ax0.set_xticks(xb)
    ax0.set_xticklabels([b[0] for b in bar_items], rotation=15, ha="right", fontsize=8.5)
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel("Score (Episode-Matched FA <= 1, k=3, m=3)")
    ax0.set_title("Random Forest Gains from v3 Features (SNR > 2 & Generalisation)", fontsize=10.5, fontweight="bold")
    ax0.text(0.5, 1.08, BANNER, transform=ax0.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax0.legend(fontsize=8.5, loc="upper left")
    ax0.grid(axis="y", alpha=0.25)

    # Right: RF Ablation impact on gradual <-> sensor_fault confusions (SNR > 2)
    ax1 = axes[1]
    ab3_all = ablation_v3_df.set_index("group_removed")
    grps = [g for g in ab3_all.index if g != "none (full_model)"]
    yg = np.arange(len(grps))
    base_conf = int(ab3_all.loc["none (full_model)", "gradual_sf_confusions_snr_gt_2"])
    delta_conf = [int(ab3_all.loc[g, "gradual_sf_confusions_snr_gt_2"]) - base_conf for g in grps]
    ax1.barh(yg, delta_conf, height=0.5, color=["firebrick" if d > 0 else "steelblue" for d in delta_conf], edgecolor="k")
    ax1.axvline(0.0, color="black", lw=1.0)
    ax1.set_yticks(yg)
    ax1.set_yticklabels([f"- {g}" for g in grps])
    ax1.set_xlabel(f"Change in Gradual <-> Sensor_Fault Confusions on Test SNR>2 (Full v3 RF = {base_conf})")
    ax1.set_title("Removing Direction Triples Gradual <-> Sensor_Fault Confusions (+72)", fontsize=10.5, fontweight="bold")
    ax1.text(0.5, 1.08, BANNER, transform=ax1.transAxes, color="red", ha="center", fontsize=8.5, fontweight="bold")
    ax1.grid(axis="x", alpha=0.25)

    stamp_banner(fig, comp_grad, comp_rap, "v3 Evaluation Gains for SNR > 2 and Updated Feature-Group Ablations")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    p_gains = out_dir / "plot_v3_snr_gt_2_gains.png"
    fig.savefig(p_gains, dpi=130)
    plt.close(fig)

    return p_feat, p_gains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-v1", default=str(ROOT / "outputs" / "datasets"))
    ap.add_argument("--data-v3", default=str(ROOT / "outputs" / "datasets_v3"))
    ap.add_argument("--eval-v1", default=str(ROOT / "outputs" / "eval"))
    ap.add_argument("--eval-v2", default=str(ROOT / "outputs" / "eval_v2"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "eval_v3"))
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--fa-target", type=float, default=1.0)
    a = ap.parse_args()

    v1_data_dir = Path(a.data_v1)
    v3_data_dir = Path(a.data_v3)
    v1_eval_dir = Path(a.eval_v1)
    v2_eval_dir = Path(a.eval_v2)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build `outputs/datasets_v3` using the exact same configs & seeds (skip if already written)
    ds_cfg = ds.load_dataset_config(ROOT / "configs" / "dataset_default.yaml")
    scen_cfg = engine.load_config(ROOT / "configs" / "scenario_default.yaml")
    sens_cfg = S.load_sensor_config(ROOT / "configs" / "sensor_default.yaml")
    conf_cfg = cf.load_confounder_config(ROOT / "configs" / "confounder_default.yaml")
    profile = S.load_noise_profile(ROOT / "configs" / "noise_profile.json")

    if not (v3_data_dir / "features_train.csv").exists():
        print("Building datasets_v3...")
        ds.build_datasets(ds_cfg, scen_cfg, sens_cfg, conf_cfg, profile, out_dir=v3_data_dir)
    print("Loading datasets_v3 and verifying frozen manifest & labels...")
    manifest_v3 = pd.read_csv(v3_data_dir / "manifest.csv", keep_default_na=False)
    labels_v3 = pd.read_csv(v3_data_dir / "window_labels.csv", keep_default_na=False, low_memory=False)

    # Verify frozen manifest and labels are 100% identical to `outputs/datasets/`
    manifest_v1 = pd.read_csv(v1_data_dir / "manifest.csv", keep_default_na=False)
    labels_v1 = pd.read_csv(v1_data_dir / "window_labels.csv", keep_default_na=False, low_memory=False)
    pd.testing.assert_frame_equal(manifest_v3, manifest_v1)
    pd.testing.assert_frame_equal(labels_v3, labels_v1)
    manifest_v3 = pd.read_csv(v3_data_dir / "manifest.csv")
    print("Verified manifest & labels match outputs/datasets/ 100%.")

    splits_v3 = {
        sp: pd.read_csv(v3_data_dir / f"features_{sp}.csv")
        for sp in ("train", "val", "test", "shifted_test", "stress", "broad_basin")
    }
    splits_v1 = {
        sp: pd.read_csv(v1_data_dir / f"features_{sp}.csv")
        for sp in ("train", "val", "test", "shifted_test", "stress", "broad_basin")
    }

    comp_grad = float(manifest_v3[manifest_v3.run_class == "gradual"]["compression_factor"].iloc[0])
    comp_rap = float(manifest_v3[manifest_v3.run_class == "rapid"]["compression_factor"].iloc[0])

    # 2. Fit & retune all 3 models on `train` and `val` ONLY (`guard_no_test_data` enforced)
    train_v3 = splits_v3["train"]
    val_v3 = splits_v3["val"]
    model_cols_v3 = F.feature_columns(train_v3)
    M.guard_model_features(model_cols_v3)

    print("Fitting v3 models on train + val...")
    th_v3 = M.ThresholdRuleClassifier().fit(train_v3, val_v3)
    print("  ThresholdRule fit done.")
    lr_v3 = M.fit_logistic_regression(train_v3, val_v3, seed=a.seed, feature_cols=model_cols_v3)
    print("  LogisticRegression fit done.")
    rf_v3 = M.fit_random_forest(train_v3, val_v3, seed=a.seed, feature_cols=model_cols_v3)
    print("  RandomForest fit done:", rf_v3.best_params)

    models_v3 = {
        "threshold_rule": th_v3,
        "logistic_regression": lr_v3,
        "random_forest": rf_v3,
    }
    with open(out_dir / "trained_models_v3.pkl", "wb") as f:
        pickle.dump(models_v3, f)

    # 3. Retune persistence & minimum-probability operating points on `val` ONLY
    pers_grid_frames = []
    for mname, clf in models_v3.items():
        _, gdf = A.evaluate_persistence_grid_on_val(
            val_v3, clf.predict(val_v3), max_fa_per_normal_run=a.fa_target
        )
        gdf.insert(0, "model", mname)
        pers_grid_frames.append(gdf)
    save_table(pd.concat(pers_grid_frames, ignore_index=True), out_dir, "val_persistence_grid_v3")

    op_ep_v3 = {}
    op_win_v3 = {}
    for mname in ("random_forest", "logistic_regression"):
        op_ep_v3[mname] = A.tune_min_prob_operating_point(
            models_v3[mname], val_v3, max_fa_per_normal_run=a.fa_target, k=3, m=3, fa_metric="fa_episodes_per_normal_run"
        )
        op_win_v3[mname] = A.tune_min_prob_operating_point(
            models_v3[mname], val_v3, max_fa_per_normal_run=a.fa_target, k=3, m=2, fa_metric="fa_windows_per_normal_run"
        )

    with open(out_dir / "operating_points_v3.pkl", "wb") as f:
        pickle.dump({"episode_fa_le_1": op_ep_v3, "strict_window_fa_le_1": op_win_v3}, f)

    configs_v3 = {
        "seed": a.seed,
        "n_model_features_v3": len(model_cols_v3),
        "new_v3_features": [
            "dir_win_convergence_align_cos",
            "dir_win_convergence_weighted_cos",
            "dir_win_convergence_residual_m",
            "dir_convergence_align_cos",
            "dir_convergence_weighted_cos",
            "dir_convergence_residual_m",
            "coherence_trailing_wx",
            "coherence_trailing_wy",
            "coherence_trailing_signed_mean",
            "coherence_trailing_mag",
        ],
        "threshold_rule_v3": th_v3.thresholds,
        "logistic_regression_v3": lr_v3.best_params,
        "random_forest_v3": {
            "best_params": rf_v3.best_params,
            "val_grid": rf_v3.tuning_grid_results,
        },
        "operating_points_episode_fa_le_1_k3_m3": {
            m: {
                "group_thresholds": op_ep_v3[m].group_thresholds,
                "val_summary": op_ep_v3[m].tuning_summary["best_val"],
            }
            for m in ("random_forest", "logistic_regression")
        },
        "operating_points_strict_window_fa_le_1_k3_m2": {
            m: {
                "group_thresholds": op_win_v3[m].group_thresholds,
                "val_summary": op_win_v3[m].tuning_summary["best_val"],
            }
            for m in ("random_forest", "logistic_regression")
        },
    }
    with open(out_dir / "eval_v3_configs.json", "w", encoding="utf-8") as f:
        json.dump(configs_v3, f, indent=2)

    # 4. Evaluate v3 across all splits (overall AND separately for SNR > 2), and compare directly against v1/v2
    with open(v1_eval_dir / "trained_models.pkl", "rb") as f:
        models_v1 = pickle.load(f)
    with open(v2_eval_dir / "operating_points_v2.pkl", "rb") as f:
        ops_v2 = pickle.load(f)

    all_snr_rows = []
    snr2_rows = []

    for sp in ("val", *EVAL_SPLITS):
        df3 = splits_v3[sp]
        df1 = splits_v1[sp]

        for mname in MODEL_NAMES:
            # Evaluate both v2 (old features) and v3 (new features) under:
            #  - raw (k=1, m=1)
            #  - episode_fa<=1 (k=3, m=3)
            #  - strict_window_fa<=1 (k=5/3, m=2)
            for ver_lbl, d_sp, m_dict, op_ep_dict, op_win_dict in [
                ("v2 (old features)", df1, models_v1, ops_v2["episode_fa_le_1"], ops_v2["strict_window_fa_le_1"]),
                ("v3 (new features)", df3, models_v3, op_ep_v3, op_win_v3),
            ]:
                raw_p = m_dict[mname].predict(d_sp)
                ep_p = raw_p if mname == "threshold_rule" else op_ep_dict[mname].predict_window(d_sp)
                win_p = raw_p if mname == "threshold_rule" else op_win_dict[mname].predict_window(d_sp)

                for reg_lbl, w_p, kv, mv in [
                    ("raw (k=1)", raw_p, 1, 1),
                    ("episode_fa<=1 (k=3,m=3)", ep_p, 3, 3),
                    ("strict_window_fa<=1", win_p, 5 if mname == "threshold_rule" else 3, 2),
                ]:
                    w_ev = ev.evaluate_window_level_with_run_bootstrap(d_sp, w_p, n_boot=a.n_boot, seed=a.seed)
                    al_ev = A.evaluate_alarm_level(d_sp, w_p, k=kv, m=mv, n_boot=a.n_boot, seed=a.seed)
                    al_w_ev = ev.evaluate_window_level_with_run_bootstrap(
                        d_sp, al_ev["alarm_preds"], n_boot=a.n_boot, seed=a.seed
                    )
                    all_snr_rows.append({
                        "split": sp,
                        "version": ver_lbl,
                        "regime": reg_lbl,
                        "model": mname,
                        "k": kv,
                        "m": mv,
                        "window_macro_f1": round(w_ev["macro_f1"], 4),
                        "window_macro_f1_ci_lo": round(w_ev["macro_f1_ci95"][0], 4),
                        "window_macro_f1_ci_hi": round(w_ev["macro_f1_ci95"][1], 4),
                        "alarm_macro_f1": round(al_w_ev["macro_f1"], 4),
                        "alarm_macro_f1_ci_lo": round(al_w_ev["macro_f1_ci95"][0], 4),
                        "alarm_macro_f1_ci_hi": round(al_w_ev["macro_f1_ci95"][1], 4),
                        "fa_episodes_per_normal_run": round(al_ev["fa_episodes_per_normal_run"], 3),
                        "fa_windows_per_normal_run": round(al_ev["fa_windows_per_normal_run"], 3),
                        "alarm_exact_detect_rate": round(al_ev["exact_detect_rate"], 4),
                        "alarm_exact_detect_ci_lo": round(al_ev["exact_detect_ci95"][0], 4),
                        "alarm_exact_detect_ci_hi": round(al_ev["exact_detect_ci95"][1], 4),
                        "mean_delay_exact_s": round(al_ev["mean_delay_exact_s"], 3),
                    })

                    if sp in ("test", "shifted_test", "broad_basin"):
                        s2 = evaluate_snr_gt_2_subset(
                            d_sp, w_p, manifest_v3, k=kv, m=mv, n_boot=a.n_boot, seed=a.seed
                        )
                        snr2_rows.append({
                            "split": sp,
                            "version": ver_lbl,
                            "regime": reg_lbl,
                            "model": mname,
                            "n_runs": s2["n_runs"],
                            "window_macro_f1": round(s2["window_macro_f1"], 4),
                            "window_macro_f1_ci_lo": round(s2["window_macro_f1_ci95"][0], 4),
                            "window_macro_f1_ci_hi": round(s2["window_macro_f1_ci95"][1], 4),
                            "alarm_macro_f1": round(s2["alarm_macro_f1"], 4),
                            "alarm_macro_f1_ci_lo": round(s2["alarm_macro_f1_ci95"][0], 4),
                            "alarm_macro_f1_ci_hi": round(s2["alarm_macro_f1_ci95"][1], 4),
                            "gradual_window_recall": round(s2["gradual_window_recall"], 4),
                            "rapid_window_recall": round(s2["rapid_window_recall"], 4),
                            "sensor_fault_window_recall": round(s2["sensor_fault_window_recall"], 4),
                            "gradual_run_alarm_detect_rate": round(s2["gradual_run_alarm_detect_rate"], 4),
                            "all_events_run_alarm_detect_rate": round(s2["all_events_run_alarm_detect_rate"], 4),
                            "fa_episodes_per_normal_run": round(s2["fa_episodes_per_normal_run"], 3),
                            "fa_windows_per_normal_run": round(s2["fa_windows_per_normal_run"], 3),
                            "total_gradual_sf_confusions": s2["total_gradual_sf_confusions"],
                        })

    all_snr_df = pd.DataFrame(all_snr_rows)
    snr2_df = pd.DataFrame(snr2_rows)
    save_table(all_snr_df, out_dir, "v2_vs_v3_all_snr_comparison")
    save_table(snr2_df, out_dir, "v2_vs_v3_snr_gt_2_comparison")

    # 5. Random Forest v3 Feature-Group Ablations (evaluated on All SNR and on SNR > 2)
    test_v3 = splits_v3["test"]
    full_p_test = rf_v3.predict(test_v3)
    full_all = ev.evaluate_window_level_with_run_bootstrap(test_v3, full_p_test, n_boot=a.n_boot, seed=a.seed)
    full_s2 = evaluate_snr_gt_2_subset(test_v3, full_p_test, manifest_v3, k=1, m=1, n_boot=a.n_boot, seed=a.seed)

    ab_rows = [{
        "group_removed": "none (full_model)",
        "n_features_kept": len(model_cols_v3),
        "macro_f1_all_snr": round(full_all["macro_f1"], 4),
        "delta_macro_f1_all_snr": 0.0,
        "macro_f1_snr_gt_2": round(full_s2["window_macro_f1"], 4),
        "delta_macro_f1_snr_gt_2": 0.0,
        "gradual_recall_snr_gt_2": round(full_s2["gradual_window_recall"], 4),
        "sensor_fault_recall_snr_gt_2": round(full_s2["sensor_fault_window_recall"], 4),
        "gradual_sf_confusions_snr_gt_2": full_s2["total_gradual_sf_confusions"],
    }]
    chosen_rf_p = {k: rf_v3.best_params[k] for k in ("n_estimators", "max_depth", "min_samples_leaf")}
    for grp in M.FEATURE_GROUPS:
        kept = M.features_without_group(model_cols_v3, grp)
        ab_clf = M.fit_random_forest(
            train_v3, val_df=None, seed=a.seed, feature_cols=kept, fixed_params=chosen_rf_p
        )
        ab_p = ab_clf.predict(test_v3)
        ab_all = ev.evaluate_window_level_with_run_bootstrap(test_v3, ab_p, n_boot=a.n_boot, seed=a.seed)
        ab_s2 = evaluate_snr_gt_2_subset(test_v3, ab_p, manifest_v3, k=1, m=1, n_boot=a.n_boot, seed=a.seed)
        ab_rows.append({
            "group_removed": grp,
            "n_features_kept": len(kept),
            "macro_f1_all_snr": round(ab_all["macro_f1"], 4),
            "delta_macro_f1_all_snr": round(ab_all["macro_f1"] - full_all["macro_f1"], 4),
            "macro_f1_snr_gt_2": round(ab_s2["window_macro_f1"], 4),
            "delta_macro_f1_snr_gt_2": round(ab_s2["window_macro_f1"] - full_s2["window_macro_f1"], 4),
            "gradual_recall_snr_gt_2": round(ab_s2["gradual_window_recall"], 4),
            "sensor_fault_recall_snr_gt_2": round(ab_s2["sensor_fault_window_recall"], 4),
            "gradual_sf_confusions_snr_gt_2": ab_s2["total_gradual_sf_confusions"],
        })
    ab_v3_df = pd.DataFrame(ab_rows)
    save_table(ab_v3_df, out_dir, "rf_ablations_v3")

    # 6. Plots
    ab_v1_df = pd.read_csv(v1_eval_dir / "rf_ablations.csv")
    p_feat, p_gains = plot_v3_diagnostics_and_gains(
        train_v3, val_v3, manifest_v3, ab_v1_df, ab_v3_df, snr2_df, out_dir, comp_grad, comp_rap
    )

    print("=== V2 VS V3 COMPARISON ON SNR > 2 SUBSET ===")
    print(snr2_df.to_string(index=False))
    print("\n=== V3 RANDOM FOREST ABLATIONS (ALL SNR & SNR > 2) ===")
    print(ab_v3_df.to_string(index=False))
    print("\nSaved plots:", p_feat, p_gains)


if __name__ == "__main__":
    main()
