# Operational Evaluation Extras Report (SYNTHETIC / SIMULATION)

> [!CAUTION]
> **CRITICAL SYNTHETIC SIMULATION DISCLAIMER:**
> All telemetry, ground settlement fields, sensor faults, communication dropouts, and evaluation splits in **ProjectCanary-Sim** are **100% SYNTHETIC SIMULATIONS**. The physical settlement engine is an analytic kinematic bell curve, **NOT** a geotechnical or geomechanical continuum solver (e.g. FLAC, PLAXIS, or FEA). It has **NOT** been validated against real field ground truth (such as active mine slopes, tailings dams, or railway embankments). No numbers in this report should be construed as performance benchmarks on real physical assets.

---

## 1. Executive Summary & Objectives

This evaluation extras round addresses four critical operationalization questions using the **FROZEN v3 models**, **FROZEN v3 operating points**, and **FROZEN synthetic datasets** (`outputs/datasets_v3`):
1. **False Alarms per Simulated Hour**: Translating raw window/episode false-alarm rates into real-world operational time using the recorded metadata time-compression factor ($360\times$).
2. **Window Accuracy Alongside Macro-F1**: Providing plain window classification accuracy, Macro-F1, and per-class precision/recall with run-bootstrap 95% confidence intervals across all test and stress splits.
3. **Grouped-by-Run Permutation Feature Importance**: Evaluating the frozen v3 Random Forest strictly on the validation set (`val`), permuting feature blocks across whole runs to preserve temporal autocorrelation within runs.
4. **Model Score Labelling & Reliability**: Generating calibration diagrams on `val` only, mandating the field name `model_score` (never "probability" or "confidence"), and documenting why uncalibrated tree ensemble scores must not be interpreted as Bayesian probabilities.

---

## 2. False Alarms per Simulated Hour & Missed-Event Trade-offs

### 2.1 Time-Compression Factor and Simulated Hour Definition
In `outputs/datasets_v3/manifest.csv`, every simulation run records its physical time-compression factor:
$$\text{compression\_factor} = \frac{\Delta t_{\text{physical}}}{\Delta t_{\text{simulated}}}$$
- **Normal runs:** $T_{\text{run}} = 240.0\text{ s}$ of simulation time with a uniform compression factor of **$360.0$**.
  $$\text{Simulated physical duration per normal run} = 240.0\text{ s} \times 360.0 = 86,400\text{ s} = \mathbf{24.0\text{ simulated hours}}\text{ (1 full day)}$$
- **Rapid basin runs:** $T_{\text{run}} = 240.0\text{ s}$ with compression factor **$1.0$** (uncompressed physical time, $240\text{ s} = 0.0667\text{ hours}$).
- **Definition of False Alarms per Simulated Hour:**
  $$\text{FA episodes per simulated hour} = \frac{\text{FA episodes per normal run}}{24.0\text{ hours}}$$
  Because all normal runs across all splits (`train`, `val`, `test`, `shifted_test`) use the uniform compression factor of $360.0$, the conversion factor is constant across normal evaluation partitions.

### 2.2 Operational Headline Results
The table below presents false alarm rates per run and per simulated physical hour alongside the **missed-event rate** (fraction of event/fault runs where no correct alarm was confirmed):

| Split | Model | Operating Regime | FA Episodes / Run (95% CI) | FA Episodes / Sim Hour (95% CI) | Missed-Event Rate (95% CI) | Exact Event Detect Rate |
|---|---|---|---|---|---|---|
| **test** | **Random Forest** | `episode_fa<=1 (k=3,m=3)` | 0.650 [0.200, 1.176] | **0.0271** [0.0083, 0.0479] | **0.310** [0.230, 0.410] | 0.690 |
| **test** | **Random Forest** | `strict_window_fa<=1` | 0.250 [0.050, 0.500] | **0.0104** [0.0021, 0.0187] | **0.360** [0.280, 0.460] | 0.640 |
| **test** | **Logistic Reg.** | `episode_fa<=1 (k=3,m=3)` | 0.100 [0.000, 0.200] | **0.0042** [0.0000, 0.0104] | **0.390** [0.290, 0.480] | 0.610 |
| **test** | **Threshold Rule**| `episode_fa<=1 (k=3,m=3)` | 0.600 [0.050, 1.326] | **0.0250** [0.0021, 0.0521] | **0.300** [0.210, 0.390] | 0.700 |
| **shifted_test** | **Random Forest** | `episode_fa<=1 (k=3,m=3)` | 1.400 [0.698, 2.333] | **0.0583** [0.0306, 0.0889] | **0.227** [0.147, 0.320] | 0.773 |
| **shifted_test** | **Random Forest** | `strict_window_fa<=1` | 0.200 [0.000, 0.400] | **0.0083** [0.0000, 0.0167] | **0.280** [0.187, 0.400] | 0.720 |
| **shifted_test** | **Threshold Rule**| `strict_window_fa<=1` | 0.200 [0.000, 0.400] | **0.0083** [0.0000, 0.0167] | **0.333** [0.240, 0.427] | 0.667 |

### 2.3 Operational Findings
1. **Meaning of the Numbers**: At the primary operating point (`episode_fa<=1`, $k=3, m=3$), Random Forest produces **0.0271 false alarm episodes per simulated hour** on the `test` split. Operationally, this corresponds to approximately **one false alarm every 36.9 simulated hours** (or ~0.65 false alarms per 24-hour day).
2. Under the strict operating regime (`strict_window_fa<=1`), Random Forest achieves **0.0104 false alarms per simulated hour** (one false alarm every 96.2 simulated hours, or ~1 false alarm every 4 simulated days), at the cost of increasing the missed-event rate from 31.0% to 36.0%.
3. Under sensor-noise distribution shift (`shifted_test`), the unconstrained raw model degrades severely (14.3 false alarms per run), but the persistence filter and tuned operating point successfully constrain false alarms to **0.0583 FA/sim-hour** ($k=3, m=3$) and **0.0083 FA/sim-hour** (strict regime).

---

## 3. Window Accuracy Alongside Macro-F1

> [!NOTE]
> **Table Caption Notice:**
> *Class balance in these synthetic evaluation splits is balanced/controlled for benchmark purposes and does NOT represent the sparse event incidence rate of a real-world field deployment.*

### 3.1 Headline Accuracy and Macro-F1 Table

| Split | Model | Operating Regime | Window Accuracy (95% CI) | Window Macro-F1 (95% CI) | Prec / Rec Normal | Prec / Rec Gradual | Prec / Rec Rapid | Prec / Rec Vibration | Prec / Rec Sensor | Prec / Rec Comm |
|---|---|---|---|---|---|---|---|---|---|---|
| `test` | `random_forest` | `raw (k=1)` | **0.7316** [0.6821, 0.7794] | **0.6385** [0.5865, 0.6796] | 0.81 / 0.90 | 0.27 / 0.20 | 0.33 / 0.29 | 1.00 / 1.00 | 0.74 / 0.39 | 0.94 / 0.92 |
| `test` | `random_forest` | `episode_fa<=1` | **0.7392** [0.6900, 0.7901] | **0.6304** [0.5779, 0.6717] | 0.80 / 0.91 | 0.33 / 0.16 | 0.32 / 0.28 | 1.00 / 1.00 | 0.63 / 0.42 | 0.90 / 0.93 |
| `test` | `random_forest` | `strict_window_fa<=1` | **0.7585** [0.7035, 0.8128] | **0.6234** [0.5695, 0.6683] | 0.78 / 0.96 | 0.50 / 0.11 | 0.39 / 0.22 | 1.00 / 1.00 | 0.60 / 0.43 | 0.90 / 0.93 |
| `test` | `threshold_rule`| `episode_fa<=1` | **0.7484** [0.7001, 0.7985] | **0.6547** [0.6074, 0.6919] | 0.78 / 0.93 | 0.32 / 0.16 | 0.40 / 0.27 | 0.99 / 1.00 | 0.91 / 0.47 | 0.95 / 0.91 |
| `test` | `logistic_reg.` | `episode_fa<=1` | **0.7303** [0.6802, 0.7823] | **0.5711** [0.5205, 0.6191] | 0.77 / 0.94 | 0.29 / 0.08 | 0.33 / 0.21 | 1.00 / 1.00 | 0.40 / 0.23 | 0.90 / 0.90 |
| `stress` | `random_forest` | `episode_fa<=1` | **0.6091** [0.5624, 0.6571] | **0.5300** [0.4866, 0.5746] | 0.71 / 0.90 | 0.37 / 0.14 | 0.57 / 0.38 | 0.98 / 0.92 | 0.37 / 0.32 | 0.28 / 0.93 |
| `broad_basin` | `random_forest`| `episode_fa<=1` | **0.4955** [0.4341, 0.5568] | **0.4647** [0.3924, 0.5281] | 0.51 / 0.92 | 0.42 / 0.23 | 0.54 / 0.38 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |

### 3.2 What Accuracy Shows vs What It Obscures
- **Why Macro-F1 is Essential**: Plain window accuracy on `test` is **73.9% to 75.8%**, which looks deceptively high. However, `normal` windows constitute over 65% of all trainable windows in the test set. A model that over-predicts `normal` achieves high plain accuracy while completely missing slow-onset `gradual` events (where recall is only 16%).
- Macro-F1 weights all 6 classes equally, accurately reflecting the severe difficulty of identifying early-stage gradual subsidence amidst IMU noise.

---

## 4. Grouped-by-Run Permutation Feature Importance (VAL Only)

### 4.1 Methodology
Permutation importance was evaluated exclusively on the validation split (`val_df`, 120 runs, 11,244 trainable windows) using the frozen v3 Random Forest.
To respect the temporal correlation between consecutive windows of the same run, **whole runs were permuted**: all windows of run $A$ were replaced with the corresponding windows of run $B$. $N_{\text{repeats}} = 5$ random run permutations were drawn with fixed seeds.

### 4.2 Feature Group Importance Ranking

| Rank | Feature Group | Feature Count | Mean Macro-F1 Drop | Std (Spread) | Min | Max |
|---|---|---|---|---|---|---|
| **1** | **`piezo`** | 19 | **0.17731** | 0.00834 | 0.16799 | 0.19043 |
| **2** | **`comm`** | 9 | **0.17507** | 0.01184 | 0.16386 | 0.19137 |
| **3** | **`direction`** | 13 | **0.07202** | 0.00965 | 0.05607 | 0.08115 |
| **4** | **`magnitude`** | 12 | **0.05225** | 0.01210 | 0.03828 | 0.06728 |
| **5** | **`hub`** | 15 | **0.03375** | 0.00555 | 0.02891 | 0.04110 |
| **6** | **`long_horizon`** | 5 | **0.02815** | 0.00533 | 0.02025 | 0.03488 |
| **7** | **`coherence`** | 8 | **0.00317** | 0.00137 | 0.00138 | 0.00471 |

### 4.3 Top 10 Individual Features

| Rank | Feature Name | Group | Mean Macro-F1 Drop | Std (Spread) | Min | Max |
|---|---|---|---|---|---|---|
| 1 | `comm_missing_frac_net` | `comm` | **0.03560** | 0.00550 | 0.03010 | 0.04347 |
| 2 | `comm_max_burst_len` | `comm` | **0.03500** | 0.00680 | 0.02922 | 0.04611 |
| 3 | `comm_latency_mean_s` | `comm` | **0.02354** | 0.00471 | 0.01661 | 0.02968 |
| 4 | `NODE_03_mean` | `magnitude` | **0.02299** | 0.00202 | 0.02025 | 0.02524 |
| 5 | `comm_latency_max_s` | `comm` | **0.02205** | 0.00297 | 0.01921 | 0.02635 |
| 6 | `long_slope_max_deg_s` | `long_horizon` | **0.01850** | 0.00427 | 0.01194 | 0.02296 |
| 7 | `NODE_01_std` | `magnitude` | **0.01056** | 0.00763 | -0.00250 | 0.01592 |
| 8 | `NODE_04_mean` | `magnitude` | **0.01010** | 0.00242 | 0.00741 | 0.01391 |
| 9 | `NODE_03_std` | `magnitude` | **0.00829** | 0.00370 | 0.00377 | 0.01244 |
| 10| `dir_convergence_residual_m` | `direction` | **0.00780** | 0.00347 | 0.00212 | 0.01120 |

### 4.4 Synthesis with Ablation Results (Agreement & Divergence)
- **High Agreement on Orthogonal Modalities**:
  - `piezo` and `comm` are ranked #1 and #2 in permutation importance (each causing a ~0.175 Macro-F1 drop when permuted). This strongly agrees with ablation experiments: piezoelectric vibration and packet loss characteristics are distinct physical phenomena with zero correlation to IMU tilt. When permuted or removed, the classifier cannot identify `abnormal_vibration` or `comm_fault`.
- **Direction vs Magnitude**:
  - In earlier v2 ablations, local sensor-frame direction features added almost zero value because random mounting yaws scrambled local angles.
  - In v3, with nominal installation yaw rotation and radial convergence point fitting (`dir_convergence_align_cos`, `dir_convergence_residual_m`), `direction` is ranked **#3** (Macro-F1 drop of **0.07202**), surpassing raw `magnitude` (0.05225). This confirms that multi-node radial convergence is the primary discriminator separating true ground subsidence from single-node sensor drift.
- **Why Coherence is Low**:
  - `coherence` ranks last (Macro-F1 drop of **0.00317**). As diagnosed in `docs/report_eval_v2.md`, within a 5-second window, sample-to-sample IMU white noise completely overwhelms the slow subsidence ramp, leaving pairwise correlation near zero across all classes.

---

## 5. Model Score Labelling & Calibration Guidelines

### 5.1 Mandated Score Terminology
In any dashboard, alert feed, or telemetry table:
- Field name must strictly be **`model_score`**.
- The terms `"probability"`, `"confidence"`, and `"likelihood"` are **strictly prohibited**.
- The models were trained with standard log-loss / Gini impurity and balanced class weighting, but have **never undergone Platt scaling or isotonic calibration**.

### 5.2 Reliability Calibration Table on VAL (`score_calibration_top1_val.csv`)

| Model | Bin Range | Mean `model_score` | Empirical Accuracy | Window Count |
|---|---|---|---|---|
| `random_forest` | $[0.2, 0.3)$ | 0.2879 | 0.3878 | 49 |
| `random_forest` | $[0.3, 0.4)$ | 0.3549 | 0.6206 | 4,433 |
| `random_forest` | $[0.4, 0.5)$ | 0.4438 | 0.7092 | 3,704 |
| `random_forest` | $[0.5, 0.6)$ | 0.5398 | 0.7688 | 995 |
| `random_forest` | $[0.6, 0.7)$ | 0.6464 | 0.7576 | 396 |
| `random_forest` | $[0.7, 0.8)$ | 0.7508 | 0.8860 | 228 |
| `random_forest` | $[0.8, 0.9)$ | 0.8549 | 0.9563 | 252 |
| `random_forest` | $[0.9, 1.0]$ | 0.9756 | 1.0000 | 1,187 |
| `logistic_regression` | $[0.2, 0.3)$ | 0.2840 | 0.2672 | 348 |
| `logistic_regression` | $[0.3, 0.4)$ | 0.3563 | 0.3835 | 3,364 |
| `logistic_regression` | $[0.4, 0.5)$ | 0.4460 | 0.4713 | 3,246 |
| `logistic_regression` | $[0.5, 0.6)$ | 0.5432 | 0.5718 | 1,796 |
| `logistic_regression` | $[0.6, 0.7)$ | 0.6418 | 0.6645 | 918 |
| `logistic_regression` | $[0.7, 0.8)$ | 0.7392 | 0.7370 | 308 |
| `logistic_regression` | $[0.8, 0.9)$ | 0.8439 | 0.6842 | 152 |
| `logistic_regression` | $[0.9, 1.0]$ | 0.9943 | 0.9532 | 1,112 |

### 5.3 Architectural Explanation of Calibration Behavior
1. **Random Forest Score Compression**:
   - Random Forest probabilities are computed by averaging predictions across 200 trees. Because each individual tree rarely predicts 100% certainty on noisy windows, the ensemble average clusters tightly in the range $[0.30, 0.50]$.
   - As shown in the table, a Random Forest `model_score` of **0.355** corresponds to an empirical accuracy of **62.1%**, and a score of **0.444** corresponds to **70.9%** accuracy. Operators must **not** interpret 0.35 as a "low 35% chance" — in this multi-class setting, 0.35 is often a strong positive indication.
2. **Logistic Regression Linearity**:
   - Logistic Regression scores follow the diagonal calibration line much more closely (e.g. score 0.446 $\rightarrow$ accuracy 47.1%; score 0.739 $\rightarrow$ accuracy 73.7%). However, Logistic Regression is more susceptible to extreme false alarms in raw mode (21.1 FA / run vs 5.65 for RF).
3. **No "Uncertain" Classification**:
   - The edge pipeline operates without an "uncertain" output state. Each window outputs a single deterministic prediction based on whether the class group threshold is met; otherwise, it reverts to `normal`.

---

## 6. Generated Publication Figures

All figures have been generated in `outputs/eval_extras/` with the required red banner:

1. [outputs/eval_extras/false_alarm_per_hour_tradeoff.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_extras/false_alarm_per_hour_tradeoff.png):
   - Shows the trade-off curve between False Alarm Episodes per Simulated Hour and Missed-Event Rate across `test` and `shifted_test`.
2. [outputs/eval_extras/window_accuracy_and_f1.csv](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_extras/window_accuracy_and_f1.csv):
   - Side-by-side accuracy, Macro-F1, and per-class precision/recall across all headline splits.
3. [outputs/eval_extras/permutation_importance_val.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_extras/permutation_importance_val.png):
   - Horizontal bar charts displaying Top 20 individual features and all 7 feature groups with 5-repeat standard deviation error bars on `val`.
4. [outputs/eval_extras/reliability_diagram_val.png](file:///d:/Github%20Projects/ProjectCanary-Sim/outputs/eval_extras/reliability_diagram_val.png):
   - Reliability diagrams for Random Forest and Logistic Regression on `val` with twin-axis sample count histograms.
