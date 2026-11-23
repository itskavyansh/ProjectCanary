# Pre-Registered Expectations: Unsupervised "Unknown Anomaly" Detector (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — PRE-REGISTERED HYPOTHESES (WRITTEN & COMMITTED BEFORE SCORING).**
> This document records our predictions for the unsupervised `unknown_anomaly` detector (`IsolationForest` primary, `RobustMahalanobis` secondary) **before** writing or running the scoring pipeline in `outputs/eval_unknown/`. Both detectors are fit strictly on `train` windows labelled `normal` (using only the 81 model features from `feature_columns(df)`, never `diag_` columns) and thresholded on `val` normal windows under $(k=3, m=3)$ persistence to $\le 1.0$ false-alarm episode per normal run. The 6-class models and operating points remain **100% frozen** (`v1.0-clean`). Nothing here is validated on real field data.

---

## 1. Pre-Registered Predictions & Mechanistic Reasons

### (a) Performance by Known Class on `test` (Which Classes Flag Well vs. Poorly)
- **Predicted to flag WELL (`abnormal_vibration`, `comm_fault`, high-SNR `rapid`, and gross `sensor_fault`):**
  - *Reason:* In `train` normal windows (including benign confounders), multi-node piezo impulse counts (`piezo_nodes_with_impulse`), network packet-loss bursts (`comm_max_burst_len`, `comm_missing_frac_net`, `comm_latency_max_s`), short-window tilt rates (`{node}_slope`), and LiDAR range changes (`hub_common_mode_m`, `hub_residual_abs_max_m`) are tightly concentrated near zero.
  - When `abnormal_vibration`, `comm_fault`, high-SNR `rapid` subsidence, or gross `sensor_fault` (`implausible_reading`, `disconnect`, `isolated_spikes`, `hub_self_motion`) occurs, multiple features jump many IQRs outside the normal support. Both `IsolationForest` (short isolation tree depth) and `RobustMahalanobis` (large standardized squared deviation) will produce high `anomaly_score` values, yielding **high AUROC ($\ge 0.85\text{--}0.98$) and high run-level detection rates on `abnormal_vibration` and `comm_fault`**, and strong detection on $\text{SNR} > 5$ `rapid` and `gradual` runs.
- **Predicted to flag POORLY (weak `gradual` below $\text{SNR} < 2$, and subtle `sensor_fault`):**
  - *Reason:* An unsupervised detector across 81 features has no label supervision to isolate the few radial-convergence and 12-window slope features that distinguish a $0.05^\circ\text{--}0.3^\circ$ `gradual` basin from background IMU white noise ($\sigma \approx 0.19^\circ$/sample) and benign temperature/wind confounders. Across 81 features, a $\le 1\,\sigma$ shift in 4–6 tilt features is diluted by normal variance in the other 75 features.
  - Consequently, **`gradual` window recall and run-level detection below $\text{SNR} < 2$ will be low ($\le 0.15\text{--}0.30$)** for both unsupervised detectors.

### (b) Behavior on `shifted_test` (Distribution Shift in Noise and Confounders)
- **Prediction: Both detectors WILL flag `shifted_test` background noise as anomalous, raising false alarms on normal runs above the $\le 1.0$ target.**
  - *Reason:* `shifted_test` increases the IMU white-noise multiplier from $[0.5, 2.0]\times$ to $[1.5, 3.0]\times$, increases per-node mounting bias SD, and scales benign confounder amplitudes by $1.5\times$.
  - Because `IsolationForest` and `RobustMahalanobis` learn the normal feature support strictly from `train` normal windows and set their threshold on `val` normal windows, elevated `{node}_std`, `{node}_mean`, and confounder excursions in `shifted_test` shift normal windows into the tail of the training distribution.
  - We expect **false-alarm episodes per normal run on `shifted_test` to exceed $1.0$** (likely $\sim 2\text{--}10+$ FA episodes/run, with `RobustMahalanobis` exhibiting a larger jump than `IsolationForest` because quadratic distance sums squared $z$-scores across all 81 features).

### (c) Behavior on Shape-Mismatch, High-Tilt (`B1`), and Realistic-Depth (`B2`) Runs
1. **Shape-mismatch (`gaussian_control`, `trough_panel`, `asymmetric`, `two_basin`, `planar_slope`):**
   - *Prediction:* Unlike the supervised 6-class classifiers — where broken radial convergence on `planar_slope` (`dir_convergence_residual_m` $\approx 10\text{--}50\text{ m}$) and single-node rim concentration on `trough_panel` caused $3\times\text{--}7.5\times$ surges in `gradual -> sensor_fault` confusions and reduced `gradual` detection — **an unsupervised anomaly detector is class-agnostic**.
   - On `planar_slope`, all 4 corner nodes tilt at 100% of `peak_tilt_deg` and `dir_convergence_residual_m` is huge, making the feature vector *more* anomalous than on an off-centre Gaussian control! Thus **unsupervised run detection on `planar_slope` will be at least as high as (or higher than) on `gaussian_control`**, and `asymmetric`, `two_basin`, and `trough_panel` will be flagged whenever at least one node sees above-noise tilt or LiDAR displacement.
2. **High-tilt arms (`high_tilt_3_6` $[3^\circ, 6^\circ]$ and `high_tilt_6_12` $[6^\circ, 12^\circ]$):**
   - *Prediction:* In Part B1, `ThresholdRuleClassifier` collapsed on `rapid` runs in `6-12 deg` (`0.067` detection) by misclassifying $>3.5^\circ$ tilts as `sensor_fault`, and `RandomForest` misclassified $55.5\%$ of `gradual` windows as `rapid`. Because unsupervised `anomaly_score` increases monotonically with feature excursion from `normal`, **both `IsolationForest` and `RobustMahalanobis` will achieve near-100% ($\ge 0.90\text{--}1.00$) run-level detection on both `high_tilt_3_6` and `high_tilt_6_12`**, with zero saturation collapse!
3. **Realistic-depth Knothe spread (`B2`, $\sigma > 1.5\,R_{\text{net}}$):**
   - *Prediction:* Because wide-$\sigma$ basins ($26.6\text{--}79.8\text{ m}$) produce simultaneous 4-node tilt and large vertical settlement $s_{\max} \propto \sigma$ (driving large `hub_common_mode_m` and `{node}_lidar_range_change_m`), **both unsupervised detectors will flag realistic-depth runs at a rate equal to or higher than `gaussian_control`** (especially at $\text{SNR} > 2$ where LiDAR common-mode shift is large).

### (d) Complement Analysis: Does the Detector Help Where the Classifier Misses Events, and at What Cost?
- **On nominal `test`:**
  - Most events missed by the frozen v3 Random Forest at `episode_fa<=1` on `test` are **low-SNR ($\text{SNR} < 2$) `gradual` and `rapid` runs**, or runs where the classifier predicted a different non-normal fault class (`sensor_fault` instead of `gradual`).
  - For runs missed because the classifier predicted the *wrong fault class* (e.g., `sensor_fault` or `rapid` on a `gradual` run) or because class probability was split below the group threshold $\tau_g$, the unsupervised detector **can** flag the run as anomalous. However, for low-SNR runs ($\text{SNR} < 1$) buried in IMU noise, the unsupervised detector will also miss most of them. Thus on `test`, we expect the detector to flag a **modest subset ($\sim 15\%\text{--}45\%$) of the event runs where the classifier failed to raise an exact-class alarm**.
- **On out-of-distribution / model-mismatch regimes (`planar_slope`, `trough_panel`, `high_tilt_6_12`, `depth_all_gt_1p5`):**
  - Here, many runs missed by the supervised classifier have **strong physical signals ($\text{SNR} > 2$)** that were rejected or misclassified (`gradual -> sensor_fault` or `rapid -> sensor_fault`) because their geometry or amplitude violated the Gaussian $\le 3^\circ$ training assumptions.
  - **Prediction:** On `planar_slope`, `high_tilt_6_12`, and `depth_all_gt_1p5`, the unsupervised detector will flag a **high fraction ($\ge 50\%\text{--}90\%$) of the runs where the classifier missed the exact class**!
- **False-alarm cost:**
  - Adding the detector via `"classifier_alarm OR detector_flag"` (or raising `"unknown_anomaly"` when the detector flags and the classifier predicts `"normal"`) **is not free**: because the detector's false alarms on normal runs do not completely coincide with the classifier's false alarms, the combined rule will **add extra false-alarm episodes per normal run** (roughly $+0.2\text{ to }+0.8$ extra FA episodes/normal run on `test`, i.e., $+3\text{ to }+12$ extra FA/h in Definition B sensor time, and a much larger increase on `shifted_test`).

---

## 2. Summary Checklist of Pre-Registered Predictions

| ID | Pre-Registered Prediction | Target Comparison |
|---|---|---|
| **U1** | High AUROC and detection on `abnormal_vibration` and `comm_fault` ($\text{AUROC} \ge 0.85$), and strong detection on $\text{SNR}>5$ `rapid`/`gradual`, but poor detection ($\le 0.30$) on weak `gradual` below $\text{SNR} < 2$. | Per-class AUROC on `test` & SNR-bin recall table |
| **U2** | On `shifted_test`, normal-run false-alarm episodes per run rise well above the $\le 1.0$ `val` target for both unsupervised detectors (`RobustMahalanobis` higher than `IsolationForest`). | `shifted_test` normal-run FA episodes/run |
| **U3** | On `planar_slope`, `high_tilt_3_6`, `high_tilt_6_12`, and `depth_all_gt_1p5`, unsupervised run detection does **not** suffer the class-confusion collapse seen in the supervised models (`planar_slope` $\ge$ `gaussian_control`; `high_tilt` $\ge 0.85\text{--}1.00$). | Run-level flag rate across shape-mismatch, B1, and B2 |
| **U4** | Complement trade-off: the detector recovers only a modest fraction of classifier-missed runs on low-SNR `test`, but recovers a large fraction ($\ge 50\%$) of classifier-missed runs on high-tilt / shape-mismatch / realistic-depth arms, at the cost of extra false alarms per normal run (and extra Definition B / Definition A false alarms per hour). | Complement analysis table (`classifier OR detector` and `unknown when classifier==normal`) |
