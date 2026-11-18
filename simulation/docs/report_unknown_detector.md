# Unsupervised "Unknown Anomaly" Detector Report (`SYNTHETIC / SIMULATION`)

> [!CAUTION]
> **SYNTHETIC / SIMULATION — NOT A VALIDATED HAZARD DETECTOR.**
> Everything in this report is evaluated on synthetic simulation data (`ProjectCanary-Sim`). The unsupervised detector outputs an `anomaly_score` (never a probability or confidence) and raises an `unknown_anomaly` flag when a window sequence is **statistically unusual relative to the synthetic `train` normal distribution**. An `unknown_anomaly` flag does **not** identify a physical hazard, does **not** assign a known class, never modifies the frozen 6-class classifier's output, and is **not validated on real field data**.

---

## 1. Executive Summary & Design Rules

The frozen v3 6-class classifiers (`RandomForest`, `LogisticRegression`, and `ThresholdRuleClassifier`) were trained on a closed world of 5 non-normal classes (`gradual`, `rapid`, `abnormal_vibration`, `sensor_fault`, `comm_fault`) under specific synthetic assumptions (Gaussian/elliptical basins, peak tilt $\le 3^\circ$, Knothe spread $\sigma \le 1.5\,R_{\text{net}}$). Earlier stress tests (`outputs/eval_shape_mismatch/` and `outputs/eval_literature/`) showed that when physical events violate those assumptions — such as a planar slope (`planar_slope`), high-tilt longwall subsidence (`3-6 deg` and `6-12 deg`), or very wide deep troughs (`H=300 m, tanB=1.5`) — the supervised classifiers either miss the event or misclassify `gradual`/`rapid` ground movement as `sensor_fault`.

To test whether a class-agnostic sidecar can flag out-of-distribution or missed events without touching the frozen 6-class models, we implemented and evaluated an unsupervised `"something unusual"` detector (`canary_sim/unknown_detector.py`, `configs/unknown_detector.yaml`, `scripts/run_unknown_detector.py`) under strict hygiene rules:
1. **Frozen Baseline Untouched:** All files tagged `v1.0-clean`, the 6-class models (`outputs/eval_v3/trained_models_v3.pkl`), operating points (`outputs/eval_v3/operating_points_v3.pkl`), datasets (`outputs/datasets_v3/`), and earlier output folders remain 100% unchanged (verified by SHA-256 and `git diff v1.0-clean`).
2. **81 Model Features Only:** Uses strictly the 81 observable features returned by `feature_columns(df)`; `diag_` and metadata columns are forbidden (`guard_model_features` enforced).
3. **Fit Strictly on `train` Normal Windows:** Both the primary detector (`IsolationForestAnomalyDetector`, `n_estimators=200`, `seed=20261117`) and secondary comparison detector (`RobustMahalanobisDetector`, `trim_quantile=0.98`, `shrinkage=0.10`), along with their median imputers and robust scalers, are fit **strictly on the `21,305` windows with `label == "normal"` in `features_train.csv`** (which includes the 60 normal-only runs with benign temperature/wind/vehicle/spike confounders plus the normal lead-in/aftermath windows of event runs). Passing any non-`train` split or non-`normal` window to `.fit()` raises `ValueError`.
4. **Threshold Tuned Strictly on `val` Normal Windows:** The flag threshold on `anomaly_score` is chosen strictly on the **`20` normal-only runs (`1,900` windows) of `features_val.csv`** under the same persistence rule `(k=3, m=3)` used by the frozen operating points, targeting **at most `1.0` false-alarm episode per normal run**.
5. **Pre-Registered Expectations (`76ea014`):** All hypotheses (`U1`–`U4` in [`docs/unknown_detector_expectations.md`](file:///d:/Github%20Projects/ProjectCanary-Sim/docs/unknown_detector_expectations.md)) were written and committed in git **before** any scoring output in `outputs/eval_unknown/` existed.

---

## 2. Threshold Selection on `val` Normal Runs

Scanning `500` candidate quantiles ($q \in [0.50, 0.999]$) of `anomaly_score` on the `1,900` `val` normal windows under `(k=3, m=3)` persistence selects the smallest threshold achieving $\le 1.000$ false-alarm episodes per normal run (`outputs/eval_unknown/detector_thresholds.md`):

| Detector | Role | Fitted `train` Normal Windows | `val` Normal Runs (`windows`) | Persistence `(k, m)` | Target FA / Normal Run | Chosen `anomaly_score` Threshold | `val` Normal Window Quantile | Achieved `val` FA Episodes / Run | Achieved `val` FA Windows / Run |
| :--- | :--- | ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| `isolation_forest` | **Primary** | 21,305 | 20 (1,900) | `(3, 3)` | $\le 1.00$ | **0.467299** | `0.8700` | **1.000** | 4.40 |
| `robust_mahalanobis` | **Secondary** | 21,305 | 20 (1,900) | `(3, 3)` | $\le 1.00$ | **9.759898** | `0.8250` | **1.000** | 10.70 |

---

## 3. Per-Class vs `normal` on `test` (AUROC & Window Flag Recall)

Evaluated on the trainable windows of the held-out `test` split (`7,154` normal windows across all `test` runs vs each known event class, with 300-draw run-bootstrap 95% CIs; `outputs/eval_unknown/test_per_class_auroc_and_recall.md`):

| Detector | Class vs `normal` | Class Windows | Normal Windows | Window AUROC [95% CI] | Raw Window Flag Recall [95% CI] | Persisted `(k=3,m=3)` Window Flag Recall [95% CI] |
| :--- | :--- | ---: | ---: | :---: | :---: | :---: |
| `isolation_forest` | `gradual` | 1,333 | 7,154 | **0.594 [0.536, 0.662]** | 0.206 [0.135, 0.310] | 0.145 [0.064, 0.271] |
| `isolation_forest` | `rapid` | 1,167 | 7,154 | **0.617 [0.545, 0.693]** | 0.211 [0.145, 0.283] | 0.156 [0.078, 0.233] |
| `isolation_forest` | `abnormal_vibration` | 502 | 7,154 | **1.000 [1.000, 1.000]** | 1.000 [1.000, 1.000] | 0.966 [0.954, 0.978] |
| `isolation_forest` | `sensor_fault` | 555 | 7,154 | **0.664 [0.595, 0.733]** | 0.261 [0.191, 0.351] | 0.182 [0.095, 0.279] |
| `isolation_forest` | `comm_fault` | 530 | 7,154 | **0.909 [0.862, 0.946]** | 0.719 [0.597, 0.827] | 0.713 [0.567, 0.841] |
| `isolation_forest` | `all_known_anomalies` | 4,087 | 7,154 | **0.701 [0.655, 0.740]** | 0.379 [0.318, 0.438] | 0.328 [0.264, 0.391] |
| `robust_mahalanobis` | `gradual` | 1,333 | 7,154 | **0.621 [0.549, 0.698]** | 0.260 [0.154, 0.393] | 0.185 [0.081, 0.317] |
| `robust_mahalanobis` | `rapid` | 1,167 | 7,154 | **0.686 [0.571, 0.801]** | 0.397 [0.222, 0.570] | 0.347 [0.163, 0.530] |
| `robust_mahalanobis` | `abnormal_vibration` | 502 | 7,154 | **0.997 [0.992, 1.000]** | 1.000 [1.000, 1.000] | 0.962 [0.951, 0.972] |
| `robust_mahalanobis` | `sensor_fault` | 555 | 7,154 | **0.752 [0.666, 0.852]** | 0.510 [0.347, 0.674] | 0.454 [0.271, 0.628] |
| `robust_mahalanobis` | `comm_fault` | 530 | 7,154 | **0.983 [0.968, 0.993]** | 0.959 [0.920, 0.987] | 0.921 [0.885, 0.947] |
| `robust_mahalanobis` | `all_known_anomalies` | 4,087 | 7,154 | **0.750 [0.695, 0.798]** | 0.514 [0.422, 0.596] | 0.459 [0.365, 0.541] |

![Test AUROC and SNR Recall](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_unknown/unknown_test_auroc_and_snr_recall.png)

### Key Mechanism
- **`abnormal_vibration` and `comm_fault` flag strongly:** Both classes perturb features (`piezo_nodes_with_impulse`, `comm_max_burst_len`, `comm_missing_frac_net`, `comm_latency_max_s`) that sit near zero in normal windows, producing window AUROC of `1.000` / `0.909` for `IsolationForest` and `0.997` / `0.983` for `RobustMahalanobis`.
- **Whole-class `gradual` and `rapid` window AUROCs are modest (`0.594–0.686`) because half of the test runs have $\text{SNR} \le 2$:** Across 81 features without class supervision, a sub-$0.3^\circ$ tilt shift in 4–6 features is masked by normal variance in the remaining 75 features.

---

## 4. `gradual` and `rapid` Recall by `snr_vs_imu_noise` Bin

Breaking down `gradual` and `rapid` recall by `snr_vs_imu_noise` bin (`<1`, `1-2`, `2-5`, `>5`; full table in `outputs/eval_unknown/snr_bin_recall_gradual_rapid.md`) shows sharp SNR stratification:

| Split / Arm | Detector | Class | SNR `<1` Run Flag Rate (`Win Recall`) | SNR `1-2` Run Flag Rate (`Win Recall`) | SNR `2-5` Run Flag Rate (`Win Recall`) | SNR `>5` Run Flag Rate (`Win Recall`) | ALL Runs Flag Rate (`Win Recall`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `test` | `isolation_forest` | `gradual` | 0.400 (`0.104`) | 0.200 (`0.069`) | **1.000** (`0.112`) | **1.000** (`0.385`) | **0.650** (`0.206`) |
| `test` | `isolation_forest` | `rapid` | 0.429 (`0.128`) | 0.500 (`0.118`) | 0.500 (`0.244`) | **1.000** (`0.375`) | **0.600** (`0.211`) |
| `test` | `robust_mahalanobis` | `gradual` | 0.000 (`0.056`) | 0.000 (`0.042`) | 0.000 (`0.119`) | **1.000** (`0.569`) | **0.400** (`0.260`) |
| `test` | `robust_mahalanobis` | `rapid` | 0.143 (`0.086`) | 0.250 (`0.118`) | 0.750 (`0.513`) | **1.000** (`0.962`) | **0.500** (`0.397`) |
| `shape:gaussian_control` | `isolation_forest` | `gradual` | 0.600 (`0.130`) | 0.400 (`0.100`) | 0.667 (`0.230`) | **0.889** (`0.311`) | **0.667** (`0.200`) |
| `shape:trough_panel` | `isolation_forest` | `gradual` | 0.600 (`0.114`) | 0.400 (`0.094`) | **0.833** (`0.183`) | 0.556 (`0.132`) | **0.600** (`0.130`) |
| `shape:planar_slope` | `isolation_forest` | `gradual` | 0.600 (`0.154`) | **1.000** (`0.295`) | **1.000** (`0.600`) | **1.000** (`0.801`) | **0.867** (`0.462`) |
| `shape:planar_slope` | `isolation_forest` | `rapid` | 0.778 (`0.187`) | **1.000** (`0.504`) | **1.000** (`0.821`) | **1.000** (`0.838`) | **0.933** (`0.556`) |
| `lit:high_tilt_3_6` | `isolation_forest` | `gradual` | N/A | N/A | N/A | **0.900** (`0.485`) | **0.900** (`0.485`) |
| `lit:high_tilt_3_6` | `isolation_forest` | `rapid` | N/A | N/A | N/A | **0.967** (`0.508`) | **0.967** (`0.508`) |
| `lit:high_tilt_6_12` | `isolation_forest` | `gradual` | N/A | N/A | N/A | **1.000** (`0.715`) | **1.000** (`0.715`) |
| `lit:high_tilt_6_12` | `isolation_forest` | `rapid` | N/A | N/A | N/A | **0.967** (`0.690`) | **0.967** (`0.690`) |
| `lit:depth_all_gt_1p5` | `isolation_forest` | `gradual` | 0.304 (`0.098`) | 0.538 (`0.144`) | 0.692 (`0.256`) | **0.809** (`0.452`) | **0.571** (`0.246`) |
| `lit:depth_all_gt_1p5` | `isolation_forest` | `rapid` | 0.391 (`0.097`) | 0.714 (`0.191`) | **0.833** (`0.388`) | **0.952** (`0.591`) | **0.700** (`0.318`) |

- On `test`, at `SNR > 5`, both detectors flag **100% (`1.000`) of `gradual` and `rapid` runs**.
- Below `SNR 2` (`SNR 1-2`), `gradual` window recall drops to `0.069` (`IsolationForest`) and `0.042` (`RobustMahalanobis`), confirming that an 81-feature unsupervised detector cannot reliably pull sub-noise tilts out of background IMU white noise.

---

## 5. Run-Level Detection, Delay, and False Alarms Across Regimes

Table summarizing run-level flagging (`unknown_anomaly` persisted at `k=3, m=3`), mean detection delay (seconds from event onset), and normal-run false-alarm rates under Definition B (sensor time, `1x`, `0.0667 h/run`) and Definition A (slow-process time, `360x`, `24 h/run`; full table including all 7 `B2` cells in `outputs/eval_unknown/run_level_detection_and_fa.md`):

| Split / Arm | Detector | Event Runs | Normal Runs | Event Flagged Rate [95% CI] | Missed Event Rate | Mean Delay (s) [95% CI] | Normal FA Episodes / Run [95% CI] | Def B Sensor FA / Hour (`1x`) | Def A Slow FA / Hour (`360x`) |
| :--- | :--- | ---: | ---: | :---: | ---: | :---: | :---: | ---: | ---: |
| `val` | `isolation_forest` | 100 | 20 | **0.730 [0.640, 0.810]** | 0.270 | 29.86 [21.44, 38.96] | 1.000 [0.550, 1.550] | 15.000 | 0.0417 |
| `val` | `robust_mahalanobis` | 100 | 20 | **0.750 [0.665, 0.820]** | 0.250 | 23.17 [15.65, 30.83] | 1.000 [0.500, 1.600] | 15.000 | 0.0417 |
| `test` | `isolation_forest` | 100 | 20 | **0.790 [0.720, 0.860]** | 0.210 | 32.73 [23.50, 41.95] | 0.900 [0.400, 1.450] | 13.500 | 0.0375 |
| `test` | `robust_mahalanobis` | 100 | 20 | **0.720 [0.630, 0.795]** | 0.280 | 23.03 [15.85, 31.82] | 0.400 [0.100, 0.776] | 6.000 | 0.0167 |
| `shifted_test` | `isolation_forest` | 75 | 15 | **0.960 [0.920, 0.994]** | 0.040 | 29.48 [22.79, 37.42] | **2.067 [1.400, 2.768]** | 31.000 | 0.0861 |
| `shifted_test` | `robust_mahalanobis` | 75 | 15 | **0.893 [0.827, 0.960]** | 0.107 | 27.45 [19.68, 36.53] | **1.267 [0.800, 1.800]** | 19.000 | 0.0528 |
| `shared_normal_40` | `isolation_forest` | 0 | 40 | N/A | N/A | N/A | 0.750 [0.400, 1.075] | 11.250 | 0.0312 |
| `shared_normal_40` | `robust_mahalanobis` | 0 | 40 | N/A | N/A | N/A | 0.675 [0.375, 1.000] | 10.125 | 0.0281 |
| `shape:gaussian_control` | `isolation_forest` | 60 | 40 | **0.650 [0.525, 0.767]** | 0.350 | 59.29 [46.85, 70.65] | 0.750 [0.450, 1.175] | 11.250 | 0.0312 |
| `shape:trough_panel` | `isolation_forest` | 60 | 40 | **0.533 [0.400, 0.650]** | 0.467 | 67.72 [51.78, 85.13] | 0.750 [0.437, 1.113] | 11.250 | 0.0312 |
| `shape:asymmetric` | `isolation_forest` | 60 | 40 | **0.650 [0.525, 0.775]** | 0.350 | 61.22 [48.71, 76.34] | 0.750 [0.450, 1.175] | 11.250 | 0.0312 |
| `shape:two_basin` | `isolation_forest` | 60 | 40 | **0.633 [0.500, 0.750]** | 0.367 | 59.74 [48.38, 72.66] | 0.750 [0.437, 1.113] | 11.250 | 0.0312 |
| `shape:planar_slope` | `isolation_forest` | 60 | 40 | **0.900 [0.825, 0.967]** | 0.100 | 42.46 [33.96, 52.55] | 0.750 [0.400, 1.113] | 11.250 | 0.0312 |
| `lit:high_tilt_3_6` | `isolation_forest` | 60 | 40 | **0.933 [0.867, 0.983]** | 0.067 | 33.61 [25.03, 41.70] | 0.750 [0.412, 1.113] | 11.250 | 0.0312 |
| `lit:high_tilt_3_6` | `robust_mahalanobis` | 60 | 40 | **1.000 [1.000, 1.000]** | 0.000 | 27.77 [22.94, 33.44] | 0.675 [0.375, 1.025] | 10.125 | 0.0281 |
| `lit:high_tilt_6_12` | `isolation_forest` | 60 | 40 | **0.983 [0.950, 1.000]** | 0.017 | 24.27 [20.45, 28.38] | 0.750 [0.425, 1.163] | 11.250 | 0.0312 |
| `lit:high_tilt_6_12` | `robust_mahalanobis` | 60 | 40 | **1.000 [1.000, 1.000]** | 0.000 | 19.43 [16.65, 22.68] | 0.675 [0.375, 1.025] | 10.125 | 0.0281 |
| `lit:depth_all_gt_1p5` | `isolation_forest` | 140 | 40 | **0.636 [0.553, 0.718]** | 0.364 | 46.33 [37.46, 57.39] | 0.750 [0.362, 1.213] | 11.250 | 0.0312 |
| `lit:depth_all_gt_1p5` | `robust_mahalanobis` | 140 | 40 | **0.514 [0.429, 0.597]** | 0.486 | 47.89 [37.97, 59.73] | 0.675 [0.387, 0.988] | 10.125 | 0.0281 |

![Run Detection Across Regimes](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_unknown/unknown_run_detection_across_regimes.png)

---

## 6. Complement Analysis at the Frozen v3 Episode Operating Point (`k=3, m=3`)

We evaluate two ways of pairing the unsupervised detector with each frozen classifier at its `episode_fa<=1` `(k=3, m=3)` operating point (`outputs/eval_unknown/complement_analysis.md`):
- **Combined Rule 1 (`classifier_alarm OR detector_flag`):** A run is flagged if either the frozen classifier raises an exact-class alarm OR the unsupervised detector raises a persisted `unknown_anomaly` episode.
- **Combined Rule 2 / Third Rule (`unknown_only_when_detector_flags_and_classifier_says_normal`):** A window is labelled `unknown_anomaly` **only** when `anomaly_score >= threshold` **and** the classifier's window prediction is `normal`, then persisted at `(k=3, m=3)`. This suppresses redundant `unknown_anomaly` flags during windows where the classifier is already predicting a known class (even if that class has not yet accumulated 3 consecutive windows or is the wrong non-normal class).

| Split / Arm | Classifier | Detector | Event Runs | Clf Exact Detect | Det Flag Rate | Clf Missed Runs (`Flagged by Det`, `Frac`) | Rule 1 (`Clf OR Det`) Detect [95% CI] | Rule 2 (`Clf OR Gated Unknown`) Detect [95% CI] | Clf Alone FA / Run | Extra FA / Run (`OR` / `Gated`) | Total FA / Run (`OR` / `Gated`) | Def B Extra Sensor FA / h (`OR` / `Gated`) |
| :--- | :--- | :--- | ---: | ---: | ---: | :---: | :---: | :---: | ---: | :---: | :---: | :---: |
| `test` | `random_forest` | `isolation_forest` | 100 | 0.690 | 0.790 | 31 (19, **0.613**) | **0.880 [0.800, 0.930]** | 0.810 [0.730, 0.885] | 0.650 | +0.800 / +0.750 | 1.450 / 1.400 | +12.00 / +11.25 |
| `test` | `random_forest` | `robust_mahalanobis` | 100 | 0.690 | 0.720 | 31 (10, **0.323**) | **0.790 [0.700, 0.870]** | 0.740 [0.655, 0.820] | 0.650 | +0.400 / +0.400 | 1.050 / 1.050 | +6.00 / +6.00 |
| `test` | `threshold_rule` | `isolation_forest` | 100 | 0.700 | 0.790 | 30 (14, **0.467**) | **0.840 [0.760, 0.900]** | 0.800 [0.730, 0.865] | 0.600 | +0.700 / +0.500 | 1.300 / 1.100 | +10.50 / +7.50 |
| `test` | `logistic_regression` | `isolation_forest` | 100 | 0.610 | 0.790 | 39 (22, **0.564**) | **0.830 [0.735, 0.905]** | 0.790 [0.715, 0.870] | 0.100 | +0.850 / +0.750 | 0.950 / 0.850 | +12.75 / +11.25 |
| `shifted_test` | `random_forest` | `isolation_forest` | 75 | 0.773 | 0.960 | 17 (16, **0.941**) | **0.987 [0.947, 1.000]** | 0.907 [0.840, 0.960] | 1.400 | +1.533 / +1.600 | 2.933 / 3.000 | +23.00 / +24.00 |
| `shape:gaussian_control` | `random_forest` | `isolation_forest` | 60 | 0.617 | 0.650 | 23 (11, **0.478**) | **0.800 [0.708, 0.900]** | 0.733 [0.608, 0.850] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `shape:trough_panel` | `random_forest` | `isolation_forest` | 60 | 0.533 | 0.533 | 28 (12, **0.429**) | **0.733 [0.617, 0.833]** | 0.683 [0.541, 0.792] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `shape:asymmetric` | `random_forest` | `isolation_forest` | 60 | 0.600 | 0.650 | 24 (12, **0.500**) | **0.800 [0.700, 0.883]** | 0.700 [0.567, 0.809] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `shape:two_basin` | `random_forest` | `isolation_forest` | 60 | 0.583 | 0.633 | 25 (12, **0.480**) | **0.783 [0.683, 0.892]** | 0.700 [0.583, 0.817] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `shape:planar_slope` | `random_forest` | `isolation_forest` | 60 | 0.550 | 0.900 | 27 (23, **0.852**) | **0.933 [0.883, 1.000]** | 0.867 [0.783, 0.933] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `lit:high_tilt_3_6` | `random_forest` | `isolation_forest` | 60 | 0.767 | 0.933 | 14 (12, **0.857**) | **0.967 [0.917, 1.000]** | 0.833 [0.733, 0.933] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `lit:high_tilt_6_12` | `random_forest` | `isolation_forest` | 60 | 0.767 | 0.983 | 14 (14, **1.000**) | **1.000 [1.000, 1.000]** | 0.800 [0.683, 0.900] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |
| `lit:high_tilt_6_12` | `threshold_rule` | `isolation_forest` | 60 | 0.367 | 0.983 | 38 (37, **0.974**) | **0.983 [0.950, 1.000]** | 0.367 [0.250, 0.483] | 0.450 | +0.600 / +0.375 | 1.050 / 0.825 | +9.00 / +5.62 |
| `lit:depth_all_gt_1p5` | `random_forest` | `isolation_forest` | 140 | 0.629 | 0.636 | 52 (22, **0.423**) | **0.786 [0.721, 0.850]** | 0.700 [0.629, 0.771] | 0.450 | +0.700 / +0.650 | 1.150 / 1.100 | +10.50 / +9.75 |

![Complement Tradeoff](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_unknown/unknown_complement_tradeoff.png)

### Important Insight on Rule 1 (`Clf OR Det`) vs Rule 2 (`unknown_only_when_detector_flags_and_classifier_says_normal`)
Look closely at `lit:high_tilt_6_12` (`6-12 deg` peak tilt):
- Under **Rule 1 (`Clf OR Det`)**, `RandomForest + IsolationForest` detects **`1.000` (60/60)** of the high-tilt runs (`14/14` of the RF-missed runs are flagged by `IsolationForest`), and `ThresholdRule + IsolationForest` detects **`0.983` (59/60)** (up from `0.367`).
- Under **Rule 2 (`unknown_only_when_detector_flags_and_classifier_says_normal`)**, `RandomForest + Gated Unknown` only reaches `0.800` and `ThresholdRule + Gated Unknown` stays at `0.367`!
- **Why?** Because on `high_tilt_6_12`, when the classifier misses the true `gradual` or `rapid` class, it does **not** predict `"normal"` — it predicts the **wrong non-normal class** (`sensor_fault` for `ThresholdRule`, or `rapid`/`sensor_fault` for `RandomForest`)! Gating the `unknown_anomaly` flag on `classifier_prediction == "normal"` silences the detector during windows where the classifier is confidently wrong about which non-normal class is occurring.
- Conversely, on normal runs, **Rule 2 (`Gated Unknown`) has a lower false-alarm cost than Rule 1 (`OR`)** (`+0.375` vs `+0.600` extra FA/run for `ThresholdRule + IsolationForest` on `shared_normal_40`, and `+0.750` vs `+0.800` for `RandomForest + IsolationForest` on `test`) because any window where the classifier transiently predicts a non-normal class breaks the 3-window persistence chain of the gated `unknown_anomaly` stream.

---

## 7. Comparison Against Pre-Registered Predictions (`docs/unknown_detector_expectations.md`)

| ID | Pre-Registered Prediction (Committed in `76ea014` Before Scoring) | Observed Result | Verdict |
| :--- | :--- | :--- | :--- |
| **U1** | High AUROC ($\ge 0.85$) on `abnormal_vibration` and `comm_fault`, strong detection on $\text{SNR}>5$ `rapid`/`gradual`, and poor detection ($\le 0.30$) on weak `gradual` below $\text{SNR} < 2$. | On `test`, `IsolationForest` AUROC is **`1.000`** (`abnormal_vibration`) and **`0.909`** (`comm_fault`); `RobustMahalanobis` is **`0.997`** and **`0.983`**. On $\text{SNR}>5$, both detectors achieve **`1.000` run detection** on `gradual` and `rapid`. On $\text{SNR } 1\text{--}2$ `gradual`, window recall is **`0.069`** (`0.200` run flag) for `IsolationForest` and **`0.042`** (`0.000` run flag) for `RobustMahalanobis`. | **Confirmed.** |
| **U2** | On `shifted_test`, normal-run false-alarm episodes per run rise well above the $\le 1.0$ `val` target for both detectors (`RobustMahalanobis` predicted higher than `IsolationForest`). | On `shifted_test`, normal-run FA episodes/run jump to **`2.067 [1.400, 2.768]`** (`31.0` FA/h Def B) for `IsolationForest` and **`1.267 [0.800, 1.800]`** (`19.0` FA/h Def B) for `RobustMahalanobis`. Both exceed the `1.0` budget, though `IsolationForest` jumped more than `RobustMahalanobis` because axis-aligned splits isolate single-node noise/confounder spikes more sharply than trimmed shrinkage covariance. | **Main effect confirmed; detector ordering reversed.** |
| **U3** | On `planar_slope`, `high_tilt_3_6`, `high_tilt_6_12`, and `depth_all_gt_1p5`, unsupervised run detection avoids the class-confusion collapse of the supervised models (`planar_slope` $\ge$ `gaussian_control`; `high_tilt` $\ge 0.85\text{--}1.00$). | `IsolationForest` flags **`0.900`** of `planar_slope` runs (vs `0.650` on `gaussian_control` and `0.550` for supervised RF), **`0.933`** on `high_tilt_3_6`, **`0.983`** on `high_tilt_6_12` (`1.000` for `RobustMahalanobis`), and **`0.636`** on `depth_all_gt_1p5` (`0.809`/`0.952` at $\text{SNR}>5$). | **Confirmed.** |
| **U4** | Detector recovers a modest fraction ($\sim 15\%\text{--}45\%$) of classifier-missed runs on `test` and a large fraction ($\ge 50\%$) on high-tilt / shape-mismatch arms, at the cost of $+0.2\text{ to }+0.8$ extra FA episodes per normal run on `test`. | Paired with `RandomForest`, `RobustMahalanobis` flags **`32.3% (10/31)`** of RF-missed runs on `test` (`+0.400` extra FA/run) and `IsolationForest` flags **`61.3% (19/31)`** (`+0.800` extra FA/run, `+12.0` FA/h Def B). On `planar_slope`, `high_tilt_3_6`, and `high_tilt_6_12`, `IsolationForest` flags **`85.2% (23/27)`**, **`85.7% (12/14)`**, and **`100.0% (14/14)`** of RF-missed runs (`+0.700` extra FA/run on `shared_normal_40`). | **Confirmed (with higher-than-expected `61.3%` `IsolationForest` recovery on `test`).** |

---

## 8. Plain Summary: What the Detector Adds, What It Costs, and Limitations

1. **What it adds:**
   - **Catches out-of-range and shape-mismatched events where the 6-class classifier fails or misclassifies:** On `planar_slope` (where broken radial convergence drops `RandomForest` exact detection to `0.550`), `IsolationForest` flags `85.2% (23/27)` of the missed runs, lifting combined `RF OR IF` detection to **`0.933`**. On `high_tilt_6_12` (`6-12 deg`), it flags **`100% (14/14)`** of RF-missed runs (`0.767 -> 1.000`) and **`97.4% (37/38)`** of `ThresholdRule`-missed runs (`0.367 -> 0.983`).
   - **Catches near-threshold and misclassified runs on held-out `test`:** Paired with `RandomForest` at `episode_fa<=1`, `IsolationForest` flags `19/31 (61.3%)` of missed event runs (raising run detection from `0.690` to `0.880` under `RF OR IF`, or `0.810` under the gated rule).
2. **What it costs:**
   - **Substantial extra false alarms on normal runs:** Adding `IsolationForest` to `RandomForest` (`0.650` FA/run on `test`, `0.450` on `shared_normal_40`) adds **`+0.800` extra false-alarm episodes per normal run on `test`** (`1.450` total FA/run, or `+12.00` extra FA/h in Definition B sensor time / `+0.0333` extra FA/h in Definition A 360x slow-process time) and **`+0.700` extra FA/run on `shared_normal_40`** (`1.150` total FA/run). `RobustMahalanobis` costs half as many extra false alarms on `test` (`+0.400` extra FA/run, `1.050` total), but recovers fewer missed runs (`32.3%` vs `61.3%`).
   - **Vulnerability to background noise shifts (`shifted_test`):** When background IMU noise and benign confounders increase in `shifted_test`, `IsolationForest` normal-run false alarms more than double to **`2.067` FA episodes/run** (`31.0` FA/h in Definition B sensor time), pushing combined `RF OR IF` false alarms to **`2.933` episodes/run**.
3. **What it cannot do:**
   - **Cannot rescue sub-noise ($\text{SNR} < 2$) gradual subsidence:** When a physical tilt signal is buried below IMU white noise, an 81-feature unsupervised detector is even less sensitive to the 4-node radial pattern than the supervised `RandomForest`.
   - **Cannot distinguish a real ground-movement hazard from a benign distribution shift or sensor anomaly:** An `unknown_anomaly` flag means *only* that the 81-feature window vector lies outside the synthetic `train` normal support. It is **not** a validated hazard detector, its `anomaly_score` is **not** a probability, and nothing in this study is validated on real field data.
