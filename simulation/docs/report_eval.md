# Evaluation Report: Baselines, Models, Detection Limits & Failure Analysis (SYNTHETIC / SIMULATION)

> [!IMPORTANT]
> **SYNTHETIC / SIMULATION ONLY**: Every dataset, sensor reading, fault, confounder, model, and metric in this report is generated from synthetic simulation models. **None of these results are validated on real field or site-deployment data.**

Frozen items respected without modification:
- State-timeline schema v1 (`docs/schema_v1.md`, `canary_sim/schema.py`).
- Label policy (`canary_sim/labels.py`).
- Dataset splits and manifest (`outputs/datasets/manifest.csv` and `features_*.csv` — read-only, never regenerated).
- `Canary-Analyst` was neither touched nor imported.

---

## 1. Feature Rules, Leakage Guards & Missing-Value Handling
- **Feature Selection**: Models use only the **71 observable model features** returned by `feature_columns(df)`. All 5 diagnostic columns (`diag_NODE_01..04_angle_to_true_centre_deg` and `diag_mean_angle_to_true_centre_deg`, which use the ground-truth basin centre) are strictly excluded and blocked by `models.guard_model_features`.
- **Train-Only Preprocessing**: `SimpleImputer(strategy="median")` and `RobustScaler()` are fit on `train` windows only (`label != "transition"`). Extreme implausible readings (e.g. `120–175°` tilt spikes) are clipped to $\pm 10^4$ before linear evaluation.
- **No-Test-Data Guard (`models.guard_no_test_data`)**: Any call to `.fit()` or hyperparameter tuning that receives rows with `split` in `{"test", "shifted_test", "stress", "broad_basin"}` immediately raises a `ValueError`. Tested in `tests/test_models_eval.py::test_no_test_data_in_fit_guard`.

---

## 2. Models, Thresholds & Hyperparameters (`canary_sim/models.py`)

### 2.1 Hand-Built Threshold Rule (`ThresholdRuleClassifier`)
Calibrated strictly on `train` quantiles at the current MPU6050/ADS1115 noise profile and tuned over a 36-point quantile grid on `val` macro-F1 (`best val macro-F1 = 0.6308` at `q_tilt = 0.94`, `sf_ratio = 2.8`, `q_rap = 0.60`):

| Threshold Parameter | Chosen Value | How It Was Chosen (`train` + `val` only) |
|---|---:|---|
| `comm_missing_frac_net_th` | `0.12` | `max(0.12, p99.5 of train normal comm_missing_frac_net)`; triggers when $\ge 3$ of 20 packets in a 5 s window are lost. |
| `comm_max_burst_len_th` | `2.0` | Triggers when $\ge 2$ consecutive samples are dropped on a node (`burst_loss` / `node_unavailable`). |
| `comm_latency_max_s_th` | `3.0` s | `max(3.0, p99.9 of train normal comm_latency_max_s)` (normal p99.9 is $\approx 2.8\text{ s}$; `added_latency` adds $+5\text{ s}$). |
| `piezo_rms_network_mean_th` | `0.02803` | Midpoint between `train` `normal` p99.9 (`0.0195`) and `train` `abnormal_vibration` p1 (`0.0366`). |
| `piezo_nodes_with_impulse_th` | `3.0` | Minimum simultaneous nodes with `piezo_peak > 0.035` required by the confounder margin (`confounder max = 2`). |
| `hub_common_mode_abs_m_th` | `0.21148` m | `p99.9` of `|hub_common_mode_m|` on `train` `normal` windows (detects `hub_self_motion` of $\pm 0.30\text{ m}$). |
| `hub_residual_abs_max_m_th` | `0.24060` m | `p99.9` of `hub_residual_abs_max_m` on `train` `normal` windows (detects `motor_stall` and `wrong_dwell_position`). |
| `imu_implausible_tilt_deg_th` | `3.5` deg | Above maximum physical basin peak tilt (`3.0°`), catching `implausible_reading` (`120–175°`). |
| `imu_stuck_std_deg_th` | `0.0001` deg | Catches `stuck_at` and `intermittent_disconnect` where within-window tilt std collapses to `0` while packets arrive (`missing_frac < 0.8`). |
| `imu_spike_std_deg_th` | `0.82318` deg | `1.5 × p99.8` of max single-node tilt std on `train` `normal` windows (detects `isolated_spikes` of $\pm 3^\circ$). |
| `imu_single_node_ratio_th` | `2.8` | Tuned on `val` (`{2.0, 2.4, 2.8}`): ratio of highest node tilt to second-highest node tilt separating single-node `bias_step`/`slow_drift` from multi-node basins. |
| `basin_tilt_max_deg_th` | `0.60727` deg | Tuned on `val` (`q_tilt = 0.94` of `train` `normal` max node tilt across `{0.94, 0.96, 0.98, 0.99}`). |
| `basin_tilt_median_deg_th` | `0.44616` deg | `q_tilt - 0.02 = 0.92` quantile of `train` `normal` 4-node median tilt. |
| `rapid_win_slope_deg_s_th` | `0.11409` deg/s | Tuned on `val` (`q_rap = 0.60` of `train` `rapid` max within-window slope across `{0.60, 0.75, 0.85}`). |
| `gradual_long_slope_deg_s_th` | `0.00575` deg/s | `p55` of `train` `gradual` 12-window slope `long_slope_max_deg_s`. |

### 2.2 Logistic Regression (Balanced Class Weights)
- Pipeline: `SimpleImputer(strategy="median") -> RobustScaler() -> LogisticRegression(class_weight="balanced", max_iter=1000, random_state=20261117)`.
- Grid tuned on `val`: `C in {0.1, 1.0, 5.0}` $\to$ `val macro-F1 = {0.1: 0.5488, 1.0: 0.5467, 5.0: 0.5457}`. Chosen: **`C = 0.1`**.

### 2.3 Random Forest (Balanced Class Weights)
- Pipeline: `SimpleImputer(strategy="median") -> RandomForestClassifier(class_weight="balanced", random_state=20261117)`.
- Grid tuned on `val` only (`n_estimators in {120, 200}`, `max_depth in {12, 20}`, `min_samples_leaf in {1, 3}`):
  - `(120, 12, 1): 0.6108` | `(120, 12, 3): 0.6054` | `(120, 20, 1): 0.6211` | `(120, 20, 3): 0.6208`
  - `(200, 12, 1): 0.6157` | `(200, 12, 3): 0.6140` | `(200, 20, 1): 0.6173` | **`(200, 20, 3): 0.6224` (chosen)**.

---

## 3. Window-Level & Run-Level Results Across Splits (`outputs/eval/`)
All 95% confidence intervals are computed by **bootstrapping over whole runs (`B = 300` replicates)**, never isolated windows.

### 3.1 Summary Table Across All Evaluation Splits (`summary_by_split.csv`)
| Split | Model | Runs | Trainable Windows | Window Macro-F1 (95% Run-Boot CI) | Run Exact Detect Rate (95% CI) | Run Any-Alarm Rate | Mean Detection Delay (s, 95% CI) | False-Alarm Windows / Normal Run (95% CI) |
|---|---|---:|---:|---|---|---|---|---|
| **`test`** | **Threshold Rule** | 120 | 11,241 | **0.6551** `[0.6096, 0.6900]` | 0.790 `[0.700, 0.860]` | 0.840 | 19.84 s `[13.80, 27.33]` | **10.05** `[3.27, 19.33]` |
| **`test`** | **Logistic Regression** | 120 | 11,241 | 0.5331 `[0.5014, 0.5594]` | **0.960** `[0.920, 0.995]` | 1.000 | 16.65 s `[10.34, 22.66]` | 42.45 `[31.97, 52.90]` |
| **`test`** | **Random Forest** | 120 | 11,241 | 0.6398 `[0.5925, 0.6823]` | 0.830 `[0.760, 0.900]` | 0.960 | **14.18 s** `[9.23, 20.02]` | 30.05 `[20.15, 42.18]` |
| **`shifted_test`** | **Threshold Rule** | 90 | 8,431 | **0.6069** `[0.5580, 0.6440]` | 0.920 `[0.853, 0.973]` | 1.000 | **7.41 s** `[5.02, 10.14]` | **43.67** `[34.27, 53.48]` |
| **`shifted_test`** | **Logistic Regression** | 90 | 8,431 | 0.4592 `[0.4356, 0.4802]` | **0.960** `[0.920, 1.000]` | 1.000 | 11.34 s `[6.67, 16.59]` | 77.47 `[73.99, 81.20]` |
| **`shifted_test`** | **Random Forest** | 90 | 8,431 | 0.5372 `[0.4876, 0.5761]` | 0.853 `[0.773, 0.933]` | 1.000 | 8.95 s `[4.78, 13.79]` | 68.73 `[63.06, 74.08]` |
| **`stress`** | **Threshold Rule** | 100 | 9,359 | 0.4676 `[0.4286, 0.5077]` | 0.830 `[0.760, 0.900]` | 1.000 | 20.10 s `[13.84, 27.19]` | — *(no pure normal runs)* |
| **`stress`** | **Logistic Regression** | 100 | 9,359 | 0.4593 `[0.4223, 0.4975]` | **0.950** `[0.900, 0.990]` | 1.000 | 17.50 s `[12.01, 23.70]` | — |
| **`stress`** | **Random Forest** | 100 | 9,359 | **0.4970** `[0.4583, 0.5379]` | 0.860 `[0.790, 0.920]` | 1.000 | **13.08 s** `[8.38, 18.88]` | — |
| **`broad_basin`** | **Threshold Rule** | 40 | 3,762 | 0.4160 `[0.3474, 0.4834]` | 0.625 `[0.475, 0.775]` | 0.775 | 45.52 s `[34.67, 57.46]` | — |
| **`broad_basin`** | **Logistic Regression** | 40 | 3,762 | 0.4283 `[0.3554, 0.5005]` | **0.950** `[0.875, 1.000]` | 1.000 | **27.18 s** `[17.20, 38.64]` | — |
| **`broad_basin`** | **Random Forest** | 40 | 3,762 | **0.4967** `[0.3966, 0.6008]` | 0.825 `[0.700, 0.925]` | 0.925 | 28.11 s `[19.23, 41.29]` | — |

### 3.2 Per-Class Precision, Recall & F1 on `test` (`per_class_metrics.csv`)
| Class (`support`) | Model | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) |
|---|---|---|---|---|
| **`normal`** (`7,154`) | Threshold Rule | 0.782 `[0.743, 0.819]` | **0.928** `[0.892, 0.957]` | **0.849** `[0.819, 0.876]` |
| | Logistic Regression | **0.855** `[0.816, 0.889]` | 0.612 `[0.570, 0.654]` | 0.713 `[0.679, 0.745]` |
| | Random Forest | 0.809 `[0.768, 0.848]` | 0.740 `[0.697, 0.784]` | 0.773 `[0.737, 0.806]` |
| **`gradual`** (`1,333`) | Threshold Rule | **0.320** `[0.165, 0.467]` | 0.159 `[0.088, 0.236]` | 0.213 `[0.119, 0.305]` |
| | Logistic Regression | 0.205 `[0.114, 0.296]` | 0.284 `[0.181, 0.388]` | 0.238 `[0.145, 0.327]` |
| | Random Forest | 0.187 `[0.115, 0.258]` | **0.384** `[0.254, 0.511]` | **0.252** `[0.162, 0.335]` |
| **`rapid`** (`1,167`) | Threshold Rule | 0.402 `[0.236, 0.548]` | **0.279** `[0.167, 0.395]` | 0.329 `[0.201, 0.443]` |
| | Logistic Regression | 0.215 `[0.129, 0.306]` | 0.321 `[0.207, 0.436]` | 0.258 `[0.161, 0.354]` |
| | Random Forest | **0.455** `[0.281, 0.625]` | 0.253 `[0.132, 0.391]` | **0.325** `[0.181, 0.464]` |
| **`abnormal_vibration`** (`502`) | Threshold Rule | 0.990 `[0.976, 1.000]` | 0.998 `[0.994, 1.000]` | 0.994 `[0.987, 1.000]` |
| | Logistic Regression | **1.000** `[1.000, 1.000]` | 0.998 `[0.994, 1.000]` | 0.999 `[0.997, 1.000]` |
| | Random Forest | **1.000** `[1.000, 1.000]` | **1.000** `[1.000, 1.000]` | **1.000** `[1.000, 1.000]` |
| **`sensor_fault`** (`555`) | Threshold Rule | **0.962** `[0.903, 1.000]` | **0.454** `[0.281, 0.629]` | **0.617** `[0.425, 0.771]` |
| | Logistic Regression | 0.130 `[0.068, 0.205]` | 0.261 `[0.141, 0.395]` | 0.174 `[0.092, 0.266]` |
| | Random Forest | 0.810 `[0.659, 0.931]` | 0.422 `[0.258, 0.590]` | 0.555 `[0.371, 0.708]` |
| **`comm_fault`** (`530`) | Threshold Rule | 0.951 `[0.916, 0.980]` | 0.909 `[0.821, 0.980]` | 0.930 `[0.876, 0.974]` |
| | Logistic Regression | 0.772 `[0.692, 0.844]` | 0.906 `[0.816, 0.978]` | 0.833 `[0.765, 0.896]` |
| | Random Forest | **0.953** `[0.918, 0.982]` | **0.919** `[0.831, 0.987]` | **0.936** `[0.883, 0.980]` |

---

## 4. Plain Statement: Where Random Forest Does NOT Beat the Threshold Rule (`rf_vs_threshold_comparison.csv`)
Using paired run-bootstrap differences ($\Delta = \text{RF} - \text{TH}$), **Random Forest does NOT beat the hand-built Threshold Rule** in the following places:

1. **Overall Window Macro-F1 on `test`, `val`, and `shifted_test`**:
   - On **`test`**: RF macro-F1 is `0.6398` vs Threshold Rule `0.6551` ($\Delta = -0.0153$, 95% CI `[-0.0524, +0.0194]`).
   - On **`val`**: RF macro-F1 is `0.6224` vs Threshold Rule `0.6308` ($\Delta = -0.0085$, 95% CI `[-0.0376, +0.0267]`).
   - On **`shifted_test`** (1.5–2× IMU noise + 0.12° node bias): RF macro-F1 drops to `0.5372` vs Threshold Rule `0.6069` — **RF loses significantly by `-0.0697` (`95% CI [-0.1031, -0.0315]`)** because balanced tree splits over-predict `gradual` whenever static node bias is elevated.
   - Even on **`broad_basin`**, where RF leads in point estimate (`0.4967` vs `0.4160`, $\Delta = +0.0807$), the 95% CI `[-0.0013, +0.1619]` slightly crosses zero.
2. **`normal` Window Recall (False Alarms) on Every Split**:
   - **`test`**: RF `0.7404` vs TH `0.9280` ($\Delta = -0.1876$, 95% CI `[-0.2264, -0.1527]`). RF produces `30.05` false-alarm windows per normal run vs `10.05` for TH.
   - **`shifted_test`**: RF `0.2421` vs TH `0.5568` ($\Delta = -0.3147$, 95% CI `[-0.3564, -0.2653]`).
   - **`stress`**: RF `0.7105` vs TH `0.8951` ($\Delta = -0.1845$, 95% CI `[-0.2302, -0.1450]`).
   - **`broad_basin`**: RF `0.7611` vs TH `0.9574` ($\Delta = -0.1963$, 95% CI `[-0.2631, -0.1183]`).
3. **`rapid` Window Recall**:
   - **`test`**: RF `0.2528` vs TH `0.2785` ($\Delta = -0.0257$, 95% CI `[-0.1508, +0.0796]`).
   - **`shifted_test`**: RF `0.1714` vs TH `0.4094` ($\Delta = -0.2380$, 95% CI `[-0.3859, -0.0718]`).
   - **`stress`**: RF `0.1766` vs TH `0.2024` ($\Delta = -0.0258$, 95% CI `[-0.1064, +0.0570]`).
4. **`sensor_fault` Window Recall**:
   - **`test`**: RF `0.4216` vs TH `0.4541` ($\Delta = -0.0324$, 95% CI `[-0.1262, +0.0564]`).
   - **`shifted_test`**: RF `0.3431` vs TH `0.4335` ($\Delta = -0.0904$, 95% CI `[-0.2117, +0.0139]`).
   - **`stress`**: RF `0.2966` vs TH `0.3056` ($\Delta = -0.0090$, 95% CI `[-0.0457, +0.0258]`).

---

## 5. Detection Limit: Recall vs `snr_vs_imu_noise` (`recall_vs_snr.csv`)
Recall across `snr_vs_imu_noise` bins on `test` (`gradual` and `rapid`), showing both exact-class recall and any-basin recall (`pred in {"gradual", "rapid"}`):

| Class | SNR Bin | Runs ($n$) | Windows | Threshold Rule Exact (Basin) Recall | Logistic Reg Exact (Basin) Recall | Random Forest Exact (Basin) Recall |
|---|---|---:|---:|---|---|---|
| **`gradual`** | `<0.5` | 2 | 137 | 0.212 `[0.19, 0.23]` (0.569) | 0.555 `[0.48, 0.63]` (0.672) | **0.854** `[0.75, 0.94]` (0.883) |
| | `0.5-1` | 3 | 201 | 0.000 `[0.00, 0.00]` (0.000) | **0.204** `[0.01, 0.51]` (0.299) | 0.154 `[0.00, 0.40]` (0.169) |
| | `1-2` | 5 | 332 | 0.000 `[0.00, 0.00]` (0.000) | 0.054 `[0.01, 0.10]` (0.346) | **0.175** `[0.00, 0.35]` (0.226) |
| | `2-5` | 2 | 143 | 0.056 `[0.01, 0.10]` (0.070) | **0.231** `[0.18, 0.28]` (0.357) | 0.182 `[0.18, 0.18]` (0.210) |
| | `>5` | 8 | 520 | 0.337 `[0.31, 0.36]` (0.608) | 0.404 `[0.26, 0.53]` (0.783) | **0.538** `[0.36, 0.74]` (**0.792**) |
| **`rapid`** | `<0.5` | 5 | 300 | 0.023 `[0.00, 0.06]` (0.050) | **0.120** `[0.06, 0.18]` (0.333) | 0.017 `[0.00, 0.03]` (0.357) |
| | `0.5-1` | 2 | 105 | 0.048 `[0.02, 0.08]` (0.133) | **0.067** `[0.04, 0.09]` (0.581) | 0.019 `[0.00, 0.04]` (0.752) |
| | `1-2` | 4 | 237 | 0.177 `[0.00, 0.37]` (0.363) | **0.215** `[0.12, 0.31]` (0.468) | 0.051 `[0.00, 0.12]` (0.489) |
| | `2-5` | 4 | 234 | 0.333 `[0.00, 0.65]` (0.509) | **0.350** `[0.21, 0.59]` (0.658) | 0.325 `[0.10, 0.55]` (**0.769**) |
| | `>5` | 5 | 291 | 0.663 `[0.59, 0.74]` (**0.966**) | 0.684 `[0.53, 0.83]` (0.921) | **0.687** `[0.36, 0.96]` (0.945) |

**Physical Detection Limit**: Reliable exact window classification ($\ge 50\%$ recall with low false-alarm rate) is only achieved in the **`SNR > 5` bin** (`peak_tilt_deg > ~0.95°`). Between `SNR = 2` and `5`, models detect that *a basin event* is occurring (`basin_recall` `51%–77%` on `rapid`), but frequently confuse `gradual` and `rapid` during the static aftermath hold. Below `SNR = 2`, single-window tilt is buried in MPU6050 white noise (`~0.19°`) plus random mounting offsets, so any apparent recall on `<0.5` comes from balanced-weight classifiers guessing `gradual` on background noise.

---

## 6. Random Forest Feature-Group Ablations (`rf_ablations.csv`)
Evaluated on `test` using the chosen RF hyperparameters (`n_estimators=200, max_depth=20, min_samples_leaf=3`):

| Group Removed | Kept Features | Test Macro-F1 (95% CI) | $\Delta$ Macro-F1 vs Full RF (95% CI) | Largest Per-Class Recall Drop (95% CI) |
|---|---:|---|---|---|
| **None (Full RF)** | 71 | **0.6398** `[0.5925, 0.6823]` | `0.0000` | — |
| **`- piezo`** (19 cols) | 52 | 0.4891 `[0.4437, 0.5256]` | **-0.1506** `[-0.1663, -0.1378]` | `abnormal_vibration`: **-0.5139** `[-0.5944, -0.4272]` |
| **`- comm`** (9 cols) | 62 | 0.5856 `[0.5364, 0.6317]` | **-0.0542** `[-0.0859, -0.0281]` | `comm_fault`: **-0.3415** `[-0.5170, -0.1869]` |
| **`- hub`** (15 cols) | 56 | 0.6235 `[0.5811, 0.6657]` | **-0.0162** `[-0.0492, +0.0092]` | `sensor_fault`: **-0.1586** `[-0.3125, -0.0054]` |
| **`- long_horizon`** (5 cols) | 66 | 0.6294 `[0.5809, 0.6733]` | **-0.0103** `[-0.0199, -0.0024]` | `sensor_fault`: **-0.0396** `[-0.0725, -0.0090]` |
| **`- coherence`** (4 cols) | 67 | 0.6354 `[0.5859, 0.6790]` | -0.0044 `[-0.0111, +0.0028]` | `rapid`: -0.0146 `[-0.0341, +0.0065]` |
| **`- direction`** (7 cols) | 64 | 0.6373 `[0.5927, 0.6788]` | -0.0025 `[-0.0174, +0.0126]` | `rapid`: -0.0257 `[-0.0837, +0.0265]` |

---

## 7. Hard-Pair Analysis & Top 3 Failure Patterns (`hard_pairs_analysis.json`)

### Pair 1: `gradual` vs `sensor_fault`
- **Test Confusions (`gradual -> sensor_fault` / `sensor_fault -> gradual`)**: Threshold Rule `3 / 51` (total `54`), Random Forest `6 / 91` (total `97`), Logistic Regression `129 / 71` (total `200`).
- **Three Most Common Failure Patterns**:
  1. *Single/two-node IMU `slow_drift` or `bias_step` mimicking a basin tilt ramp* (`82` windows) — e.g. `sim_test_sensor_fault_2026112261` window `36` `[90.0, 95.0)s` (`bias_step`, `NODE_01_mean = 0.608°`, `long_slope_max = 0.0087°/s`, RF predicts `rapid`, TH predicts `gradual`).
  2. *Asymmetric off-centre gradual basin loading 1 corner node strongly* (`38` windows) — e.g. `sim_test_gradual_2026112201` window `48` `[120.0, 125.0)s` (`NODE_01_mean = 0.395°`, mistaken for single-node sensor fault).
  3. *Hub `wrong_dwell_position` / `stale_readings` window between LiDAR passes where IMU noise mimics minor tilt* (`4` windows) — e.g. `sim_test_sensor_fault_2026112268` window `32` `[80.0, 85.0)s` (`wrong_dwell_position`, RF predicts `gradual`).

### Pair 2: `normal` (with confounders) vs `abnormal_vibration`
- **Test Confusions (`normal -> abnormal_vibration` / `abnormal_vibration -> normal`)**: Threshold Rule `4 / 0` (total `4`), Logistic Regression `0 / 0`, Random Forest `0 / 0`.
- **Three Boundary / Failure Patterns**:
  1. *Two-node benign wind burst + coinciding background piezo impulse in a normal window* (`4` windows) — e.g. `sim_test_normal_2026112193` window `11` `[27.5, 32.5)s` (`piezo_rms_net = 0.0186` with 3 nodes crossing `0.035` peak, triggering TH `abnormal_vibration`).
  2. *Onset/tail boundary windows of `abnormal_vibration` where the ramp envelope is near minimum* (`62` windows) — e.g. `sim_test_abnormal_vibration_2026112254` window `27` `[67.5, 72.5)s`.
  3. *Abnormal vibration windows with the lowest network piezo RMS (`~0.04–0.06`)* (`199` windows) — separated from `normal` only because `piezo_rms_network_mean` stays above the `3×` confounder margin.

### Pair 3: `comm_fault` vs `sensor_fault`
- **Test Confusions**: Random Forest `0 / 2` (total `2`), Threshold Rule `1 / 2` (total `3`), Logistic Regression `18 / 7` (total `25`).
- **Three Most Common Failure Patterns**:
  1. *Sensor fault window coinciding with background 1% LoRa packet loss (`comm_missing_frac_net >= 0.05`)* (`3` windows) — e.g. `sim_test_sensor_fault_2026112261` window `65` `[162.5, 167.5)s` (`comm_miss_net = 0.10`, RF predicts `comm_fault`).
  2. *Sparse `packet_drops` `comm_fault` window (`missing_frac <= 0.20`) where remaining samples have an IMU noise excursion* (`18` windows) — e.g. `sim_test_comm_fault_2026112281` window `40` `[100.0, 105.0)s` (LR predicts `sensor_fault`).
  3. *`added_latency` `comm_fault` window with zero dropped packets where random tilt excursion triggers `sensor_fault`* (`26` windows).

### Pair 4: `gradual` vs `rapid`
- **Test Confusions (`gradual -> rapid` / `rapid -> gradual`)**: Threshold Rule `192 / 190` (total `382`), Random Forest `160 / 462` (total `622`), Logistic Regression `347 / 319` (total `666`).
- **Three Most Common Failure Patterns**:
  1. *`rapid` basin in post-ramp aftermath hold (`t_rel_event_start_s > 25 s`)* (`493` windows) — once a rapid event finishes ramping (`tau = 5–19 s`), the ground stays tilted at a constant angle; both 5 s slope and 30 s `long_slope` drop to `~0°/s`, making post-ramp `rapid` aftermath windows physically indistinguishable from held `gradual` windows (e.g. `sim_test_rapid_2026112221` window `55` `[137.5, 142.5)s`).
  2. *High-SNR (`SNR > 5`) `gradual` basin during its steepest mid-ramp phase* (`311` windows) — a large amplitude gradual bowl produces a steep deg/s slope that exceeds the `rapid` slope threshold (e.g. `sim_test_gradual_2026112201` window `43` `[107.5, 112.5)s`, `SNR = 6.49`).
  3. *Low-SNR (`SNR < 1.2`) basin windows where IMU white noise dominates both 5 s and 12-window slopes* (`316` windows) — e.g. `sim_test_gradual_2026112208` window `52` `[130.0, 135.0)s`.

---

## 8. Stress Set Policy & Separate Scoring (`stress_set_scoring.csv`)
- **Label Rule for Combined Windows**: In `stress` runs, each window's `label_set` lists all conditions with $\ge 50\%$ overlap (e.g. `"gradual|sensor_fault"`), while primary `label` picks the highest-priority active condition under `PRIORITY = (rapid, gradual, abnormal_vibration, sensor_fault, comm_fault, normal)`.
- **Separate Stress Scores (`9,359` trainable windows, including `2,043` simultaneous two-condition windows)**:
  - **Random Forest**: Primary Macro-F1 = **`0.4970`** `[0.4583, 0.5379]`; primary accuracy on simultaneous 2-condition windows = `27.9%`, rising to **`83.7%` under set-membership scoring** (credited when predicting *either* active condition in `label_set`) because single-label classifiers detect the dominant fault (`comm_fault`/`sensor_fault`) when it co-occurs with a gradual basin.
  - **Threshold Rule**: Primary Macro-F1 = `0.4676` `[0.4286, 0.5077]`; primary accuracy on 2-condition windows = `18.6%`, rising to **`82.9%` under set-membership scoring**.
  - **Logistic Regression**: Primary Macro-F1 = `0.4593` `[0.4223, 0.4975]`; primary accuracy on 2-condition windows = `28.8%`, rising to **`79.2%` under set-membership scoring**.

---

## 9. Honest Summary of Where Each Model Fails
1. **Threshold Rule**: Fails on low- and medium-SNR basins (`gradual` test recall is only `15.9%` because its tilt threshold `0.607°` is set high to keep `normal` recall at `92.8%`), and misses subtle IMU `slow_drift`/`wrong_dwell_position` faults before they cross the `0.607°` or `0.24 m` cutoffs.
2. **Logistic Regression**: Fails badly on specificity (`normal` test recall `61.2%`, `42.5` false-alarm windows per normal run; `sensor_fault` precision `13.0%`) because balanced linear hyperplanes cannot represent non-monotonic fault rules (such as `std == 0` for `stuck_at` vs `std > 0.82°` for `isolated_spikes`).
3. **Random Forest**: Fails in three specific ways:
   - **Vulnerability to noise/bias shift (`shifted_test`)**: Because `class_weight="balanced"` pushes low-tilt windows near the decision boundary into `gradual`, elevating per-node static bias (`0.05° -> 0.12°`) and IMU noise (`1.5–2×`) causes RF `normal` recall to collapse to `24.2%` (`macro-F1 = 0.5372`, losing to Threshold Rule by `-0.0697`).
   - **Aftermath ambiguity (`rapid` vs `gradual`)**: Once a `rapid` ramp finishes (`t > t_on + tau`), its 12-window slope returns to zero while tilt remains elevated, so RF misclassifies `462 / 1,167` (`39.6%`) of `rapid` windows as `gradual`.
   - **Single-node drift vs off-centre basin**: Without dense spatial coverage, a `slow_drift` or `bias_step` on one or two nodes looks identical in IMU space to an off-centre basin until the next 20 s LiDAR sweep updates `lidar_residual_m`.
